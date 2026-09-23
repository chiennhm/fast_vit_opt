"""Checkpoint persistence for detector training."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _validate_metadata_subset(expected, actual, path="metadata"):
    """Fail when checkpoint-critical metadata differs from the active model."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"Checkpoint {path} must be an object")
        for key, expected_value in expected.items():
            if key not in actual:
                raise ValueError(f"Checkpoint is missing {path}.{key}")
            _validate_metadata_subset(
                expected_value, actual[key], f"{path}.{key}"
            )
        return
    if isinstance(expected, (list, tuple)):
        if list(expected) != list(actual):
            raise ValueError(
                f"Checkpoint {path} mismatch: expected {expected!r}, got {actual!r}"
            )
        return
    if expected != actual:
        raise ValueError(
            f"Checkpoint {path} mismatch: expected {expected!r}, got {actual!r}"
        )


def save_checkpoint(state, output_dir, filename="checkpoint.pth"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(state, path)
    logger.info("Checkpoint saved to %s", path)
    return path


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    *,
    expected_architecture=None,
    expected_training=None,
    allow_missing_metadata=False,
    weights_only=False,
):
    """Restore a trusted checkpoint after validating its architecture metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    architecture = metadata.get("architecture") if metadata else None
    if architecture is None:
        if not allow_missing_metadata:
            raise ValueError(
                "Checkpoint has no architecture metadata. To inspect an old "
                "checkpoint, select --architecture-version legacy and pass "
                "--allow-missing-checkpoint-metadata explicitly."
            )
        if not weights_only and optimizer is not None:
            raise ValueError(
                "Optimizer state cannot be resumed from a checkpoint without "
                "architecture metadata; use --resume-weights-only."
            )
        logger.warning("Loading checkpoint without architecture metadata")
    elif expected_architecture is not None:
        _validate_metadata_subset(expected_architecture, architecture, "architecture")
    if metadata is not None and expected_training is not None:
        training = metadata.get("training")
        if training is None:
            raise ValueError("Checkpoint is missing metadata.training")
        _validate_metadata_subset(expected_training, training, "training")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict or state_dict")
    model.load_state_dict(state_dict)

    if weights_only:
        logger.info("Loaded model weights from %s without training state", path)
        return 0, 0.0

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
