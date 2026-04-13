#
# FastViT Object Detection Training on PASCAL VOC
#
# Usage:
#   python object_detection.py --data-dir ./data --model fastvit_sa12 --epochs 150
#   python object_detection.py --data-dir ./data --model fastvit_t8 --batch-size 32
#   python object_detection.py --wandb-project fastvit-det --wandb-name run1
#

import argparse
import os
import sys
import time
import math
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Project imports
from detection.fastvit_detector import FastViTDetector
from detection.losses import DetectionLoss
from detection.eval_voc import evaluate_voc, print_eval_results
from detection.visualize import save_detection_results, VOC_CLASSES
from voc_dataset import build_voc_datasets, detection_collate


# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("detection")


# ============================================================================
# Argument parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="FastViT Object Detection Training on PASCAL VOC"
    )

    # Data
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Root directory for VOC dataset (default: ./data)"
    )
    parser.add_argument(
        "--img-size", type=int, default=512,
        help="Input image size (default: 512)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Don't auto-download VOC dataset"
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="fastvit_sa12",
        choices=[
            "fastvit_t8", "fastvit_t12", "fastvit_s12",
            "fastvit_sa12", "fastvit_sa24", "fastvit_sa36", "fastvit_ma36",
        ],
        help="FastViT backbone variant (default: fastvit_sa12)"
    )
    parser.add_argument(
        "--fpn-channels", type=int, default=256,
        help="FPN output channels (default: 256)"
    )
    parser.add_argument(
        "--pretrained-backbone", type=str, default=None,
        help="Path to pretrained backbone checkpoint"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05,
        help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5,
        help="Warmup epochs (default: 5)"
    )
    parser.add_argument(
        "--clip-grad", type=float, default=5.0,
        help="Gradient clipping norm (default: 5.0)"
    )

    # Loss
    parser.add_argument(
        "--focal-alpha", type=float, default=0.25,
        help="Focal loss alpha (default: 0.25)"
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss gamma (default: 2.0)"
    )

    # AMP
    parser.add_argument(
        "--amp", action="store_true", default=True,
        help="Use automatic mixed precision (default: True)"
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable AMP"
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval", type=int, default=5,
        help="Evaluate every N epochs (default: 5)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation on the validation set"
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="./output/detection",
        help="Output directory (default: ./output/detection)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--save-visualizations", action="store_true",
        help="Save detection visualizations during evaluation"
    )

    # Wandb
    parser.add_argument(
        "--wandb-project", type=str, default="fastvit-detection",
        help="Wandb project name (default: fastvit-detection)"
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None,
        help="Wandb run name (default: auto-generated)"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="Wandb entity/team name (optional)"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging"
    )

    # Misc
    parser.add_argument(
        "--workers", "-j", type=int, default=4,
        help="Number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=20,
        help="Log every N batches (default: 20)"
    )

    args = parser.parse_args()

    if args.no_amp:
        args.amp = False

    # Resolve wandb availability
    args.use_wandb = HAS_WANDB and not args.no_wandb
    if not args.no_wandb and not HAS_WANDB:
        logging.getLogger("detection").warning(
            "wandb not installed — logging disabled. Install with: pip install wandb"
        )

    return args


# ============================================================================
# Learning rate scheduler with warmup + cosine annealing
# ============================================================================
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.min_lrs = [lr * min_lr_ratio for lr in self.base_lrs]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (epoch + 1) / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            for pg, base_lr, min_lr in zip(self.optimizer.param_groups, self.base_lrs, self.min_lrs):
                pg["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (
                    1 + math.cos(math.pi * progress)
                )

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ============================================================================
# Training
# ============================================================================
def train_one_epoch(model, criterion, dataloader, optimizer, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0
    num_batches = len(dataloader)

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad()

        if args.amp and device.type == "cuda":
            with autocast('cuda'):
                cls_preds, reg_preds, anchors = model(images)
                loss_dict = criterion(cls_preds, reg_preds, anchors, targets)
                loss = loss_dict["cls_loss"] + loss_dict["reg_loss"]

            scaler.scale(loss).backward()
            grad_norm = None
            if args.clip_grad:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_preds, reg_preds, anchors = model(images)
            loss_dict = criterion(cls_preds, reg_preds, anchors, targets)
            loss = loss_dict["cls_loss"] + loss_dict["reg_loss"]

            loss.backward()
            grad_norm = None
            if args.clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item()
            optimizer.step()

        batch_size = images.shape[0]
        cls_loss_val = loss_dict["cls_loss"].item()
        reg_loss_val = loss_dict["reg_loss"].item()
        total_loss_val = loss.item()
        total_cls_loss += cls_loss_val * batch_size
        total_reg_loss += reg_loss_val * batch_size
        total_samples += batch_size

        # Wandb per-step logging
        global_step = epoch * num_batches + batch_idx
        if args.use_wandb:
            log_dict = {
                "train/cls_loss": cls_loss_val,
                "train/reg_loss": reg_loss_val,
                "train/total_loss": total_loss_val,
                "train/num_pos": loss_dict["num_pos"],
                "train/lr_backbone": optimizer.param_groups[0]["lr"],
                "train/lr_head": optimizer.param_groups[2]["lr"],
            }
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm
            wandb.log(log_dict, step=global_step)

        # Console logging
        if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[2]["lr"]
            logger.info(
                f"Epoch [{epoch}][{batch_idx+1}/{num_batches}] "
                f"cls_loss: {cls_loss_val:.4f} "
                f"reg_loss: {reg_loss_val:.4f} "
                f"total: {total_loss_val:.4f} "
                f"pos: {loss_dict['num_pos']} "
                f"lr_head: {lr_head:.6f} "
                f"grad_norm: {(grad_norm or 0.0):.2f} "
                f"ETA: {eta:.0f}s"
            )

    avg_cls = total_cls_loss / max(total_samples, 1)
    avg_reg = total_reg_loss / max(total_samples, 1)

    return {"cls_loss": avg_cls, "reg_loss": avg_reg, "total_loss": avg_cls + avg_reg}


# ============================================================================
# Evaluation
# ============================================================================
@torch.no_grad()
def evaluate(model, dataloader, device, args, save_vis=False, output_dir=None):
    """Evaluate model on validation set."""
    model.eval()

    all_predictions = []
    all_ground_truths = []

    logger.info("Running evaluation...")
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)

        # Get predictions
        predictions = model.predict(
            images, score_thresh=0.01, nms_thresh=0.5, max_detections=200
        )

        all_predictions.extend(predictions)
        all_ground_truths.extend(targets)

        # Save visualizations for first few batches
        if save_vis and output_dir and batch_idx < 5:
            vis_dir = os.path.join(output_dir, "visualizations")
            save_detection_results(
                images,
                predictions,
                vis_dir,
                class_names=VOC_CLASSES,
                score_thresh=0.3,
            )

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Eval batch {batch_idx+1}/{len(dataloader)}")

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s")

    # Compute VOC mAP
    results = evaluate_voc(
        all_predictions,
        all_ground_truths,
        num_classes=20,
        iou_threshold=0.5,
        class_names=VOC_CLASSES,
    )

    return results


# ============================================================================
# Checkpoint management
# ============================================================================
def save_checkpoint(state, output_dir, filename="checkpoint.pth"):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scaler=None):
    """Load checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_map = checkpoint.get("best_map", 0.0)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(
        f"Resumed from epoch {start_epoch - 1}, best mAP: {best_map * 100:.2f}%"
    )
    return start_epoch, best_map


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"{args.model}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # ========================================================================
    # Wandb init
    # ========================================================================
    if args.use_wandb:
        wandb_run_name = args.wandb_name or f"{args.model}_{timestamp}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            dir=output_dir,
            resume="allow",
        )
        logger.info(f"Wandb initialized: {wandb.run.url}")
        # Define x-axis for epoch-level and eval metrics
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")

    # ========================================================================
    # Datasets
    # ========================================================================
    logger.info("Building datasets...")
    train_dataset, val_dataset = build_voc_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        download=not args.no_download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ========================================================================
    # Model
    # ========================================================================
    logger.info(f"Building model: {args.model}")
    model = FastViTDetector(
        model_name=args.model,
        num_classes=20,
        fpn_channels=args.fpn_channels,
        pretrained_backbone=args.pretrained_backbone,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Watch model gradients in wandb
    if args.use_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    # ========================================================================
    # Loss, Optimizer, Scheduler
    # ========================================================================
    criterion = DetectionLoss(
        num_classes=20,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )

    # Separate weight decay for conv/linear vs bias/norm, and backbone vs head
    decay_params = []
    no_decay_params = []
    backbone_decay_params = []
    backbone_no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        is_backbone = "backbone" in name
        
        if "bias" in name or "bn" in name or "norm" in name or "layer_scale" in name:
            if is_backbone:
                backbone_no_decay_params.append(param)
            else:
                no_decay_params.append(param)
        else:
            if is_backbone:
                backbone_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_decay_params, "weight_decay": args.weight_decay, "lr": args.lr * 0.1},
            {"params": backbone_no_decay_params, "weight_decay": 0.0, "lr": args.lr * 0.1},
            {"params": decay_params, "weight_decay": args.weight_decay, "lr": args.lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": args.lr},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    scaler = GradScaler('cuda') if args.amp and device.type == "cuda" else None

    # ========================================================================
    # Resume
    # ========================================================================
    start_epoch = 0
    best_map = 0.0

    if args.resume:
        start_epoch, best_map = load_checkpoint(
            args.resume, model, optimizer, scaler
        )

    # ========================================================================
    # Eval only mode
    # ========================================================================
    if args.eval_only:
        if not args.resume:
            logger.error("--eval-only requires --resume to specify checkpoint")
            sys.exit(1)
        results = evaluate(
            model, val_loader, device, args,
            save_vis=args.save_visualizations, output_dir=output_dir,
        )
        print_eval_results(results, logger_fn=logger.info)
        return

    # ========================================================================
    # Training loop
    # ========================================================================
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  LR:         {args.lr}")
    logger.info(f"  Image size: {args.img_size}")
    logger.info(f"  AMP:        {args.amp}")
    logger.info(f"  Output:     {output_dir}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[2]["lr"]
        logger.info(f"\nEpoch {epoch}/{args.epochs - 1} | LR Head: {lr_head:.6f} | LR Backbone: {lr_backbone:.6f}")

        # Train
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, epoch, args
        )

        logger.info(
            f"Epoch {epoch} summary - "
            f"cls_loss: {train_metrics['cls_loss']:.4f}, "
            f"reg_loss: {train_metrics['reg_loss']:.4f}, "
            f"total: {train_metrics['total_loss']:.4f}"
        )

        # Evaluate
        is_eval_epoch = (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1
        current_map = None
        is_best = False
        if is_eval_epoch:
            results = evaluate(
                model, val_loader, device, args,
                save_vis=args.save_visualizations and epoch == args.epochs - 1,
                output_dir=output_dir,
            )
            print_eval_results(results, logger_fn=logger.info)

            current_map = results["mAP"]
            is_best = current_map > best_map
            best_map = max(best_map, current_map)

        # Wandb epoch-level logging (train + eval in one call)
        if args.use_wandb:
            global_step = (epoch + 1) * len(train_loader) - 1
            epoch_log = {
                "epoch": epoch,
                "epoch/cls_loss": train_metrics["cls_loss"],
                "epoch/reg_loss": train_metrics["reg_loss"],
                "epoch/total_loss": train_metrics["total_loss"],
                "epoch/lr_backbone": optimizer.param_groups[0]["lr"],
                "epoch/lr_head": optimizer.param_groups[2]["lr"],
            }
            if current_map is not None:
                epoch_log["eval/mAP"] = current_map
                epoch_log["eval/best_mAP"] = best_map
                # Per-class AP
                for cls_name, ap in results["ap_per_class"].items():
                    if ap is not None:
                        epoch_log[f"eval/AP/{cls_name}"] = ap
            wandb.log(epoch_log, step=global_step)

        if is_eval_epoch:
            # Save checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_map": best_map,
                "args": vars(args),
            }
            if scaler:
                checkpoint_state["scaler_state_dict"] = scaler.state_dict()

            save_checkpoint(checkpoint_state, output_dir, "last.pth")
            if is_best:
                save_checkpoint(checkpoint_state, output_dir, "best.pth")
                logger.info(f"*** New best mAP: {best_map * 100:.2f}% ***")
        else:
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_map": best_map,
                    "args": vars(args),
                }
                if scaler:
                    checkpoint_state["scaler_state_dict"] = scaler.state_dict()
                save_checkpoint(checkpoint_state, output_dir, f"epoch_{epoch}.pth")

    logger.info(f"\nTraining completed! Best mAP: {best_map * 100:.2f}%")
    logger.info(f"Checkpoints saved to: {output_dir}")

    # Wandb finish
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
