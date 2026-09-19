"""Checkpoint persistence for detector training."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(state, output_dir, filename="checkpoint.pth"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(state, path)
    logger.info("Checkpoint saved to %s", path)
    return path


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None):
    """Restore a trusted training checkpoint and return `(start_epoch, best_map)`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_map = float(checkpoint.get("best_map", 0.0))
    logger.info(
        "Resumed from epoch %d with best mAP %.2f%%",
        start_epoch - 1,
        best_map * 100,
    )
    return start_epoch, best_map
