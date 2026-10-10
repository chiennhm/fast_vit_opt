"""Export intermediate FastViT detector activations for one image.

Example:
    python inspect_feature_maps.py \
        --image /mnt/bdd100k/images/100k/val/example.jpg \
        --checkpoint /checkpoints/run/best.pth \
        --output-dir /checkpoints/run/feature_debug/example \
        --scope key

Use ``--scope all`` to record every leaf module. This is intentionally slower
and can create hundreds of images; raw tensors are only written when
``--save-tensors`` is explicitly selected.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.amp import autocast

from bdd100k_dataset import BDD100K_CLASSES
from detection.feature_debug import (
    FeatureMapRecorder,
    draw_predictions,
    prepare_single_image,
    select_debug_modules,
    serialize_predictions,
)


LOGGER = logging.getLogger("feature_debug")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one image through a FastViT detector and export layer feature "
            "maps, activation diagnostics, per-class maps, and predictions."
        )
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--checkpoint", required=True, help="Trusted detector checkpoint")
    parser.add_argument("--output-dir", required=True, help="New or empty output directory")
    parser.add_argument(
        "--scope",
        choices=["key", "all"],
        default="key",
        help="key: stages/FPN/head; all: every leaf module",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="REGEX",
        help="Only hook layer names matching a regex; may be repeated and overrides --scope",
    )
    parser.add_argument("--top-channels", type=int, default=16)
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Write numerical summaries without activation PNG files",
    )
    parser.add_argument(
        "--save-tensors",
        action="store_true",
        help="Also save full layer outputs as .pt files; can require substantial disk space",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match evaluator mixed precision on CUDA (default: enabled)",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override checkpoint preprocessing short side (normally 800)",
    )
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--padding-multiple", type=int, default=32)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--pre-nms-topk", type=int, default=2000)
    parser.add_argument("--max-detections", type=int, default=100)

    # Fallbacks for a legacy checkpoint without architecture metadata.
    parser.add_argument("--model", default="fastvit_sa12")
    parser.add_argument(
        "--architecture-version",
        choices=["legacy", "fixed_c5"],
        default="fixed_c5",
    )
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument(
        "--anchor-sizes",
        nargs=5,
        type=int,
        default=[11, 19, 28, 51, 153],
    )
    parser.add_argument("--reg-max", type=int, default=None)
    parser.add_argument(
        "--allow-missing-checkpoint-metadata",
        action="store_true",
        help="Allow a legacy checkpoint and use explicit architecture CLI fallbacks",
    )
    return parser.parse_args()


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(value)


def _load_checkpoint_payload(path: Path):
    # Checkpoints must be trusted: weights_only=False may deserialize Python objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict or state_dict")
    return checkpoint, state_dict


def _normalise_metadata_value(value):
    if isinstance(value, (tuple, list)):
        return [_normalise_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise_metadata_value(item) for key, item in value.items()}
    return value


def _assert_architecture_matches(expected, actual):
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"Checkpoint architecture metadata is missing {key!r}")
        if _normalise_metadata_value(expected_value) != _normalise_metadata_value(actual[key]):
            raise ValueError(
                f"Checkpoint architecture mismatch for {key}: "
                f"model={expected_value!r}, checkpoint={actual[key]!r}"
            )


def _build_model(args, checkpoint):
    # Import lazily so --help and helper tests still work before all model
    # dependencies (notably timm) are installed.
    from detection.fastvit_detector import FastViTDetector

    metadata = checkpoint.get("metadata") or {}
    architecture = metadata.get("architecture")
    if architecture is None and not args.allow_missing_checkpoint_metadata:
        raise ValueError(
            "Checkpoint has no metadata.architecture. Pass "
            "--allow-missing-checkpoint-metadata with matching legacy "
            "--model/--architecture-version/--anchor-sizes values."
        )
    architecture = architecture or {}

    model = FastViTDetector(
        model_name=architecture.get("model_name", args.model),
        num_classes=int(architecture.get("num_classes", len(BDD100K_CLASSES))),
        fpn_channels=int(architecture.get("fpn_channels", args.fpn_channels)),
        anchor_sizes=tuple(architecture.get("anchor_sizes", args.anchor_sizes)),
        anchor_ratios=tuple(architecture.get("anchor_ratios", (0.5, 1.0, 2.0))),
        anchor_scales=tuple(
            architecture.get("anchor_scales", (1.0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)))
        ),
        reg_max=architecture.get("reg_max", args.reg_max),
        architecture_version=architecture.get(
            "architecture_version", args.architecture_version
        ),
    )
    if architecture:
        _assert_architecture_matches(model.architecture_metadata(), architecture)
    return model, metadata


def _prepare_output_dir(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new directory so "
            "results from different checkpoints cannot be mixed."
        )
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.top_channels < 1:
        raise ValueError("--top-channels must be positive")
    if not 0 <= args.score_thresh <= 1 or not 0 <= args.nms_thresh <= 1:
        raise ValueError("Score and NMS thresholds must be in [0, 1]")
    _prepare_output_dir(output_dir)

    device = _device_from_arg(args.device)
    LOGGER.info("Loading checkpoint metadata from %s", checkpoint_path)
    checkpoint, state_dict = _load_checkpoint_payload(checkpoint_path)
    model, checkpoint_metadata = _build_model(args, checkpoint)
    model.load_state_dict(state_dict)
    del state_dict
    model.to(device).eval()

    preprocessing = checkpoint_metadata.get("preprocessing") or {}
    short_side = args.img_size or int(preprocessing.get("short_side", 800))
    max_size = args.max_size or int(preprocessing.get("long_side_max", 1333))
    image_tensor, original_image, padded_image, image_metadata = prepare_single_image(
        image_path,
        short_side=short_side,
        max_size=max_size,
        padding_multiple=args.padding_multiple,
    )
    original_image.save(output_dir / "input_original.png")
    padded_image.save(output_dir / "input_preprocessed.png")

    modules = select_debug_modules(model, scope=args.scope, include_patterns=args.include)
    LOGGER.info("Registering hooks on %d modules (%s scope)", len(modules), args.scope)
    recorder = FeatureMapRecorder(
        output_dir,
        padded_image,
        BDD100K_CLASSES,
        model.num_anchors,
        top_channels=args.top_channels,
        save_visualizations=not args.stats_only,
        save_tensors=args.save_tensors,
    )
    recorder.register(modules)

    try:
        image_tensor = image_tensor.to(device)
        with torch.inference_mode(), autocast(
            "cuda", enabled=args.amp and device.type == "cuda"
        ):
            predictions = model.predict(
                image_tensor,
                score_thresh=args.score_thresh,
                nms_thresh=args.nms_thresh,
                pre_nms_topk=args.pre_nms_topk,
                max_detections=args.max_detections,
                image_sizes=[tuple(image_metadata["resized_size"])],
            )
    finally:
        recorder.remove()

    prediction_rows = serialize_predictions(
        predictions[0], image_metadata, BDD100K_CLASSES
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(prediction_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    draw_predictions(original_image, prediction_rows).save(output_dir / "predictions.png")

    run_metadata = {
        "image": image_metadata,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "checkpoint_best_map": checkpoint.get("best_map"),
        "architecture": model.architecture_metadata(),
        "checkpoint_training": checkpoint_metadata.get("training"),
        "device": str(device),
        "amp": args.amp and device.type == "cuda",
        "scope": args.scope,
        "include_patterns": args.include,
        "hooked_layers": list(modules),
        "score_threshold": args.score_thresh,
        "nms_threshold": args.nms_thresh,
        "pre_nms_topk": args.pre_nms_topk,
        "max_detections": args.max_detections,
        "prediction_count": len(prediction_rows),
    }
    recorder.write_reports(run_metadata)

    anomalies = [record for record in recorder.records if record["flags"]]
    LOGGER.info(
        "Done: %d tensors, %d class maps, %d flagged tensors, %d predictions",
        len(recorder.records),
        len(recorder.class_records),
        len(anomalies),
        len(prediction_rows),
    )
    LOGGER.info("Open %s", (output_dir / "index.html").resolve())


if __name__ == "__main__":
    main()

