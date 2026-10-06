"""Train or evaluate the FastViT detector on BDD100K."""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from bdd100k_dataset import (
    BDD100K_CLASSES,
    BDD100KDetectionDataset,
    bdd100k_collate,
)
from detection.fastvit_detector import FastViTDetector
from detection.losses import DetectionLoss
from detection.metrics import print_eval_results
from engine import (
    WarmupCosineScheduler,
    WarmupStepDecayScheduler,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)

logger = logging.getLogger("bdd100k_detection")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FastViT bounding-box detection on BDD100K"
    )
    parser.add_argument("--data-dir", default="./data/bdd100k")
    parser.add_argument("--train-img", default=None)
    parser.add_argument("--train-ann", default=None)
    parser.add_argument("--val-img", default=None)
    parser.add_argument("--val-ann", default=None)
    parser.add_argument("--img-size", type=int, default=800)
    parser.add_argument("--cache-ram", action="store_true")

    parser.add_argument(
        "--model",
        default="fastvit_sa12",
        choices=[
            "fastvit_t8",
            "fastvit_t12",
            "fastvit_s12",
            "fastvit_sa12",
            "fastvit_sa24",
            "fastvit_sa36",
            "fastvit_ma36",
        ],
    )
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--pretrained-backbone", default=None)
    parser.add_argument(
        "--anchor-sizes",
        nargs=5,
        type=int,
        default=[11, 19, 28, 51, 153],
        metavar=("P2", "P3", "P4", "P5", "P6"),
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--workers", "-j", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--scheduler", choices=["step", "cosine"], default="step")
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-steps", nargs="+", type=int, default=[8, 11])
    parser.add_argument("--lr-gamma", type=float, default=0.1)

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--json-output-dir", default=None)
    parser.add_argument("--save-visualizations", action="store_true")

    parser.add_argument("--output", default="./output/bdd100k")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--wandb-project", default="fastvit-bdd100k")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.eval_only and args.no_eval:
        parser.error("--eval-only and --no-eval are mutually exclusive")
    if args.eval_only and not args.resume:
        parser.error("--eval-only requires --resume")
    if args.accum_steps < 1:
        parser.error("--accum-steps must be at least 1")
    if any(size <= 0 for size in args.anchor_sizes):
        parser.error("all five --anchor-sizes must be positive")
    return args


def _dataset_paths(args):
    root = Path(args.data_dir)
    return {
        "train_img": Path(args.train_img) if args.train_img else root / "images/100k/train",
        "train_ann": Path(args.train_ann) if args.train_ann else root / "annotations/bdd100k_det_train_coco.json",
        "val_img": Path(args.val_img) if args.val_img else root / "images/100k/val",
        "val_ann": Path(args.val_ann) if args.val_ann else root / "annotations/bdd100k_det_val_coco.json",
    }


def _make_loader(dataset, batch_size, workers, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=bdd100k_collate,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
        persistent_workers=workers > 0,
    )


def _parameter_groups(model, lr, weight_decay):
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        prefix = "backbone" if "backbone" in name else "head"
        suffix = "no_decay" if any(key in name for key in ("bias", "bn", "norm", "layer_scale")) else "decay"
        groups[f"{prefix}_{suffix}"].append(parameter)
    return [
        {"params": groups["backbone_decay"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["backbone_no_decay"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["head_no_decay"], "lr": lr, "weight_decay": 0.0},
    ]


def _run_evaluation(model, loader, device, args, output_dir):
    return evaluate(
        model,
        loader,
        device,
        class_names=BDD100K_CLASSES,
        amp=args.amp,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        max_detections=args.max_detections,
        max_samples=args.max_eval_samples,
        output_dir=output_dir,
        save_visualizations=args.save_visualizations,
        save_json=args.save_json,
        json_output_dir=args.json_output_dir,
        model_name=args.model,
    )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    logger.info("Using device: %s", device)

    paths = _dataset_paths(args)
    eval_batch_size = args.eval_batch_size or min(args.batch_size, 4)
    train_loader = None
    val_loader = None
    if not args.eval_only:
        train_dataset = BDD100KDetectionDataset(
            paths["train_img"],
            paths["train_ann"],
            img_size=args.img_size,
            augment=True,
            cache_ram=args.cache_ram,
        )
        train_loader = _make_loader(
            train_dataset, args.batch_size, args.workers, True, device
        )
    if not args.no_eval:
        val_dataset = BDD100KDetectionDataset(
            paths["val_img"],
            paths["val_ann"],
            img_size=args.img_size,
            augment=False,
            cache_ram=args.cache_ram,
        )
        val_loader = _make_loader(
            val_dataset, eval_batch_size, min(args.workers, 2), False, device
        )

    model = FastViTDetector(
        model_name=args.model,
        num_classes=len(BDD100K_CLASSES),
        fpn_channels=args.fpn_channels,
        pretrained_backbone=args.pretrained_backbone,
        anchor_sizes=tuple(args.anchor_sizes),
    ).to(device)

    if args.eval_only:
        load_checkpoint(args.resume, model)
        results = _run_evaluation(model, val_loader, device, args, output_dir)
        print_eval_results(results, logger_fn=logger.info)
        return

    criterion = DetectionLoss(
        num_classes=len(BDD100K_CLASSES),
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, args.lr, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    if args.scheduler == "step":
        scheduler = WarmupStepDecayScheduler(
            optimizer,
            warmup_iters=args.warmup_iters,
            milestones=args.lr_steps,
            gamma=args.lr_gamma,
        )
    else:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
        )
    scaler = GradScaler("cuda") if args.amp and device.type == "cuda" else None

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("--wandb requires the optional wandb package") from error
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"{args.model}_{timestamp}",
            config=vars(args),
            dir=output_dir,
        )

    start_epoch = 0
    best_map = -1.0
    if args.resume:
        start_epoch, best_map = load_checkpoint(
            args.resume, model, optimizer, scaler, scheduler
        )

    for epoch in range(start_epoch, args.epochs):
        scheduler.step_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            amp=args.amp,
            accum_steps=args.accum_steps,
            clip_grad=args.clip_grad,
            log_interval=args.log_interval,
            scheduler=scheduler,
            wandb_run=wandb_run,
        )
        logger.info("Epoch %d metrics: %s", epoch, train_metrics)

        current_map = None
        should_evaluate = val_loader is not None and (
            (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs
        )
        if should_evaluate:
            results = _run_evaluation(model, val_loader, device, args, output_dir)
            print_eval_results(results, logger_fn=logger.info)
            current_map = float(results["mAP"])
            is_best = current_map > best_map
            best_map = max(best_map, current_map)
        else:
            is_best = False

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": best_map,
            "args": vars(args),
        }
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        save_checkpoint(state, output_dir, "last.pth")
        if is_best:
            save_checkpoint(state, output_dir, "best.pth")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(state, output_dir, f"epoch_{epoch + 1}.pth")

        if wandb_run is not None:
            payload = {f"epoch/{key}": value for key, value in train_metrics.items()}
            if current_map is not None:
                payload.update({"eval/mAP": current_map, "eval/best_mAP": best_map})
            wandb_run.log(payload, step=(epoch + 1) * len(train_loader))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
