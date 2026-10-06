"""One-epoch training loop for the FastViT BDD100K detector."""

import logging
import time

import torch
from torch.amp import autocast

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

logger = logging.getLogger(__name__)


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    *,
    amp=True,
    accum_steps=1,
    clip_grad=5.0,
    log_interval=100,
    scheduler=None,
    wandb_run=None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    totals = {"cls_loss": 0.0, "reg_loss": 0.0, "samples": 0}
    num_batches = len(dataloader)
    start_time = time.time()
    progress = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc=f"Epoch {epoch}",
        leave=False,
    )

    for batch_idx, (images, targets) in progress:
        images = images.to(device, non_blocking=True)
        targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]

        window_start = (batch_idx // accum_steps) * accum_steps
        window_size = min(accum_steps, num_batches - window_start)
        use_amp = amp and device.type == "cuda"
        with autocast("cuda", enabled=use_amp):
            cls_preds, reg_preds, anchors = model(images)
            loss_dict = criterion(cls_preds, reg_preds, anchors, targets)
            unscaled_loss = loss_dict["cls_loss"] + loss_dict["reg_loss"]
            loss = unscaled_loss / window_size

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (batch_idx + 1) % accum_steps == 0 or batch_idx + 1 == num_batches
        grad_norm = None
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad
                ).item()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step_iter()

        batch_size = images.shape[0]
        cls_value = float(loss_dict["cls_loss"].detach())
        reg_value = float(loss_dict["reg_loss"].detach())
        totals["cls_loss"] += cls_value * batch_size
        totals["reg_loss"] += reg_value * batch_size
        totals["samples"] += batch_size

        if wandb_run is not None:
            payload = {
                "train/cls_loss": cls_value,
                "train/reg_loss": reg_value,
                "train/total_loss": cls_value + reg_value,
                "train/num_pos": loss_dict["num_pos"],
                "train/lr_backbone": optimizer.param_groups[0]["lr"],
                "train/lr_head": optimizer.param_groups[2]["lr"],
            }
            if grad_norm is not None:
                payload["train/grad_norm"] = grad_norm
            wandb_run.log(payload, step=epoch * num_batches + batch_idx)

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                cls=f"{cls_value:.4f}",
                reg=f"{reg_value:.4f}",
                pos=loss_dict["num_pos"],
            )
        elif (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == num_batches:
            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d [%d/%d] loss=%.4f pos=%s elapsed=%.1fs",
                epoch,
                batch_idx + 1,
                num_batches,
                cls_value + reg_value,
                loss_dict["num_pos"],
                elapsed,
            )

    samples = max(totals["samples"], 1)
    cls_avg = totals["cls_loss"] / samples
    reg_avg = totals["reg_loss"] / samples
    return {
        "cls_loss": cls_avg,
        "reg_loss": reg_avg,
        "total_loss": cls_avg + reg_avg,
    }
