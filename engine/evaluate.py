"""BDD100K detector evaluation and canonical COCO JSON export."""

import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from detection.metrics import (
    COCO_MAX_DETECTIONS,
    build_coco_ground_truth,
    build_coco_predictions,
    evaluate_coco,
)
from detection.visualize import save_detection_results

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


logger = logging.getLogger(__name__)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _clean_record(record):
    return {key: _as_numpy(value) for key, value in record.items()}


def _size_tuple(target, key):
    values = target[key]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if len(values) != 2:
        raise ValueError(f"{key} must contain [height, width]")
    return int(values[0]), int(values[1])


def _restore_boxes_to_original(boxes, resized_size, original_size):
    """Map xyxy boxes from resized coordinates back to the source image."""
    resized_h, resized_w = resized_size
    original_h, original_w = original_size
    restored = boxes.detach().clone().float()
    if restored.numel() == 0:
        return restored.reshape(-1, 4)
    restored[:, [0, 2]] = restored[:, [0, 2]] * (original_w / resized_w)
    restored[:, [1, 3]] = restored[:, [1, 3]] * (original_h / resized_h)
    restored[:, [0, 2]] = restored[:, [0, 2]].clamp(0, original_w)
    restored[:, [1, 3]] = restored[:, [1, 3]].clamp(0, original_h)
    return restored


def _prediction_in_original_space(prediction, target):
    original_size = _size_tuple(target, "orig_size")
    resized_size = _size_tuple(target, "size")
    return {
        "boxes": _restore_boxes_to_original(
            prediction["boxes"], resized_size, original_size
        ),
        "scores": prediction["scores"],
        "labels": prediction["labels"],
        "image_id": target["image_id"],
        "orig_size": target["orig_size"],
    }


def _target_in_original_space(target):
    original_size = _size_tuple(target, "orig_size")
    resized_size = _size_tuple(target, "size")
    boxes = _restore_boxes_to_original(target["boxes"], resized_size, original_size)
    restored = {
        "boxes": boxes,
        "labels": target["labels"],
        "iscrowd": target.get(
            "iscrowd", torch.zeros(len(boxes), dtype=torch.int64)
        ),
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        "image_id": target["image_id"],
        "orig_size": target["orig_size"],
    }
    if "difficults" in target:
        restored["difficults"] = target["difficults"]
    return restored


def _write_json_results(
    predictions,
    ground_truths,
    output_dir,
    class_names,
    model_name,
    protocol,
    architecture,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions_BDD100K.json"
    gt_path = output_dir / "ground_truths_BDD100K.json"

    prediction_rows = build_coco_predictions(predictions)
    ground_truth = build_coco_ground_truth(
        ground_truths, len(class_names), class_names
    )
    pred_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
    gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "bdd100k",
                "task": "bbox_detection",
                "model": model_name,
                "architecture": architecture,
                "protocol": protocol,
                "num_images": len(ground_truth["images"]),
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pred_path, gt_path


@torch.inference_mode()
def evaluate(
    model,
    dataloader,
    device,
    *,
    class_names,
    amp=True,
    score_thresh=0.05,
    nms_thresh=0.5,
    max_detections=100,
    pre_nms_topk=2000,
    max_samples=None,
    output_dir=None,
    save_visualizations=False,
    save_json=False,
    json_output_dir=None,
    model_name="fastvit_sa12",
):
    model.eval()
    predictions_all = []
    ground_truths_all = []
    images_seen = 0

    progress = tqdm(dataloader, total=len(dataloader), desc="Evaluating", leave=False)
    for batch_index, (images, targets) in enumerate(progress):
        if max_samples is not None:
            remaining = max_samples - images_seen
            if remaining <= 0:
                break
            if len(images) > remaining:
                images = images[:remaining]
                targets = targets[:remaining]

        valid_image_sizes = [_size_tuple(target, "size") for target in targets]
        images = images.to(device, non_blocking=True)
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            predictions = model.predict(
                images,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                max_detections=max_detections,
                pre_nms_topk=pre_nms_topk,
                image_sizes=valid_image_sizes,
            )

        restored_predictions = [
            _prediction_in_original_space(prediction, target)
            for prediction, target in zip(predictions, targets)
        ]
        restored_targets = [
            _target_in_original_space(target) for target in targets
        ]
        predictions_all.extend(_clean_record(item) for item in restored_predictions)
        ground_truths_all.extend(_clean_record(item) for item in restored_targets)
        images_seen += len(images)

        if save_visualizations and output_dir and batch_index < 5:
            save_detection_results(
                images,
                predictions,
                Path(output_dir) / "visualizations" / f"batch_{batch_index:03d}",
                class_names=class_names,
                score_thresh=0.3,
            )

        del images, targets, predictions
        if device.type == "cuda" and (batch_index + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    results = evaluate_coco(
        predictions_all,
        ground_truths_all,
        num_classes=len(class_names),
        class_names=class_names,
        max_detections=COCO_MAX_DETECTIONS,
    )
    results["num_images"] = images_seen
    results["inference_protocol"] = {
        "score_threshold": score_thresh,
        "nms_threshold": nms_thresh,
        "pre_nms_topk_anchor_locations": pre_nms_topk,
        "max_detections_per_image": max_detections,
        "clip_size": "resized_unpadded_image",
        "evaluation_coordinates": "original_image",
    }

    result_paths = None
    if save_json:
        destination = json_output_dir or Path(output_dir or ".") / "json_results"
        architecture = (
            model.architecture_metadata()
            if hasattr(model, "architecture_metadata")
            else None
        )
        result_paths = _write_json_results(
            predictions_all,
            ground_truths_all,
            destination,
            class_names,
            model_name,
            {
                "coco": results["coco_protocol"],
                "inference": results["inference_protocol"],
            },
            architecture,
        )
        results["pred_json"] = str(result_paths[0])
        results["gt_json"] = str(result_paths[1])

    if output_dir:
        output_path = Path(output_dir) / "eval_results.json"
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results
