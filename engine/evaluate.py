"""BDD100K detector evaluation and JSON export."""

import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from detection.metrics import evaluate_coco
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


def _clean_prediction(prediction):
    return {key: _as_numpy(value) for key, value in prediction.items()}


def _clean_target(target):
    return {key: _as_numpy(value) for key, value in target.items()}


def _write_json_results(predictions, ground_truths, output_dir, class_names, model_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions_BDD100K.json"
    gt_path = output_dir / "ground_truths_BDD100K.json"

    prediction_rows = []
    for image_id, prediction in enumerate(predictions):
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            x1, y1, x2, y2 = box
            prediction_rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    images = []
    annotations = []
    annotation_id = 1
    for image_id, target in enumerate(ground_truths):
        images.append({"id": image_id, "width": 0, "height": 0})
        crowds = target.get("iscrowd", np.zeros(len(target["boxes"]), dtype=np.int64))
        areas = target.get("area")
        for index, (box, label) in enumerate(zip(target["boxes"], target["labels"])):
            x1, y1, x2, y2 = box
            width = float(x2 - x1)
            height = float(y2 - y1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), width, height],
                    "area": float(areas[index]) if areas is not None else width * height,
                    "iscrowd": int(crowds[index]),
                }
            )
            annotation_id += 1

    categories = [
        {"id": index + 1, "name": name}
        for index, name in enumerate(class_names)
    ]
    pred_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
    gt_path.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories},
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "bdd100k",
                "arch": "fastvit_detector",
                "model": model_name,
                "num_images": len(images),
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

        images = images.to(device, non_blocking=True)
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            predictions = model.predict(
                images,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                max_detections=max_detections,
            )

        predictions_all.extend(_clean_prediction(pred) for pred in predictions)
        ground_truths_all.extend(_clean_target(target) for target in targets)
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

    result_paths = None
    if save_json:
        destination = json_output_dir or Path(output_dir or ".") / "json_results"
        result_paths = _write_json_results(
            predictions_all,
            ground_truths_all,
            destination,
            class_names,
            model_name,
        )

    thresholds = np.linspace(0.5, 0.95, 10).tolist()
    results = evaluate_coco(
        predictions_all,
        ground_truths_all,
        num_classes=len(class_names),
        iou_threshold=thresholds,
        class_names=class_names,
    )
    results["num_images"] = images_seen

    if result_paths is not None:
        results["pred_json"] = str(result_paths[0])
        results["gt_json"] = str(result_paths[1])

    if output_dir:
        output_path = Path(output_dir) / "eval_results.json"
        serializable = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in results.items()
        }
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    return results
