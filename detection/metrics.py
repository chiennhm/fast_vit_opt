"""Official COCO bbox evaluation for the BDD100K detector."""

import contextlib
import io
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


COCO_IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
COCO_MAX_DETECTIONS = (1, 10, 100)
COCO_AREA_RANGES = (
    (0.0, 1e10),
    (0.0, 32.0**2),
    (32.0**2, 96.0**2),
    (96.0**2, 1e10),
)
COCO_AREA_LABELS = ("all", "small", "medium", "large")


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_int(value):
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar value, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def _image_size(record):
    if "orig_size" not in record:
        raise KeyError("Evaluation records must include orig_size=[height, width]")
    size = _as_numpy(record["orig_size"]).reshape(-1)
    if size.size != 2:
        raise ValueError(f"orig_size must contain [height, width], got {size}")
    height, width = (int(size[0]), int(size[1]))
    if height < 1 or width < 1:
        raise ValueError(f"Invalid original image size {(height, width)}")
    return height, width


def build_coco_ground_truth(ground_truths, num_classes, class_names=None):
    """Convert targets in original-image coordinates to a COCO dataset dict."""
    images = []
    annotations = []
    annotation_id = 1
    seen_image_ids = set()

    for fallback_id, target in enumerate(ground_truths):
        image_id = _as_int(target.get("image_id", fallback_id))
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate image_id in evaluation data: {image_id}")
        seen_image_ids.add(image_id)
        height, width = _image_size(target)
        images.append({"id": image_id, "width": width, "height": height})

        boxes = _as_numpy(target["boxes"])
        labels = _as_numpy(target["labels"])
        crowds = _as_numpy(
            target.get("iscrowd", np.zeros(len(boxes), dtype=np.int64))
        )
        areas = target.get("area")
        areas = _as_numpy(areas) if areas is not None else None

        for index, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = (float(value) for value in box)
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, box_width, box_height],
                    "area": (
                        float(areas[index])
                        if areas is not None
                        else box_width * box_height
                    ),
                    "iscrowd": int(crowds[index]),
                }
            )
            annotation_id += 1

    categories = [
        {
            "id": category_id,
            "name": (
                class_names[category_id - 1]
                if class_names
                else f"class_{category_id}"
            ),
        }
        for category_id in range(1, num_classes + 1)
    ]
    return {
        "info": {"description": "BDD100K bounding-box evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def build_coco_predictions(predictions):
    """Convert predictions in original-image coordinates to COCO rows."""
    results = []
    for fallback_id, prediction in enumerate(predictions):
        image_id = _as_int(prediction.get("image_id", fallback_id))
        boxes = _as_numpy(prediction["boxes"])
        scores = _as_numpy(prediction["scores"])
        labels = _as_numpy(prediction["labels"])
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(value) for value in box)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    return results


def _mean_valid(values):
    values = np.asarray(values)
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else None


def summarize_detection_counts(tp, fp, fn):
    """Micro operating-point metrics; no true negatives exist for detection."""
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1}


def _add_operating_points(result, ground_truth, detections, class_names, evaluator=None):
    """Reuse COCO's IoU=.50 matches/ignore masks, including crowd handling."""
    by_image = {image["id"]: [0, 0, 0] for image in ground_truth["images"]}
    by_class = {index: [0, 0, 0] for index in range(1, len(class_names) + 1)}
    if evaluator is None:
        # No detections or no GT: COCOeval has no evalImgs to inspect.
        for ann in ground_truth["annotations"]:
            if not ann.get("iscrowd", 0) and not ann.get("ignore", 0):
                by_image[ann["image_id"]][2] += 1
                by_class[ann["category_id"]][2] += 1
        for row in detections:
            by_image[row["image_id"]][1] += 1
            by_class[row["category_id"]][1] += 1
    else:
        iou_indices = np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))
        if not len(iou_indices):
            raise ValueError("Operating-point counts require IoU 0.50")
        iou_index = int(iou_indices[0])
        for item in evaluator.evalImgs:
            if item is None or list(item["aRng"]) != list(COCO_AREA_RANGES[0]):
                continue
            ignored = np.asarray(item["dtIgnore"][iou_index], dtype=bool)
            matched = np.asarray(item["dtMatches"][iou_index]) > 0
            tp = int(np.count_nonzero(matched & ~ignored))
            fp = int(np.count_nonzero(~matched & ~ignored))
            fn = int(np.count_nonzero(~np.asarray(item["gtIgnore"], dtype=bool))) - tp
            for bucket in (by_image[item["image_id"]], by_class[item["category_id"]]):
                bucket[0] += tp
                bucket[1] += fp
                bucket[2] += fn
    totals = np.asarray(list(by_image.values()), dtype=np.int64).reshape(-1, 3).sum(axis=0)
    result["operating_point"] = {
        **summarize_detection_counts(*totals),
        "iou_threshold": 0.5, "averaging": "micro",
        "ignored_detections_excluded": True,
    }
    result["operating_points_by_image"] = {
        str(image_id): summarize_detection_counts(*counts)
        for image_id, counts in by_image.items()
    }
    for category_id, counts in by_class.items():
        result["per_class"][class_names[category_id - 1]]["operating_point"] = (
            summarize_detection_counts(*counts)
        )
    return result


def _empty_results(class_names, ground_truth, thresholds, max_detections):
    gt_categories = {row["category_id"] for row in ground_truth["annotations"]}
    per_class = {
        name: {
            "AP": 0.0 if index + 1 in gt_categories else None,
            "AP50": 0.0 if index + 1 in gt_categories else None,
            "AP75": 0.0 if index + 1 in gt_categories else None,
            "AR100": 0.0 if index + 1 in gt_categories else None,
        }
        for index, name in enumerate(class_names)
    }
    return {
        "mAP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "APS": 0.0,
        "APM": 0.0,
        "APL": 0.0,
        "AR100": 0.0,
        "ARS": 0.0,
        "ARM": 0.0,
        "ARL": 0.0,
        "ap_per_class": {name: values["AP"] for name, values in per_class.items()},
        "per_class": per_class,
        "mAP_per_threshold": {float(value): 0.0 for value in thresholds},
        "coco_protocol": {
            "iou_thresholds": [float(value) for value in thresholds],
            "recall_thresholds": 101,
            "area_ranges": [list(value) for value in COCO_AREA_RANGES],
            "area_labels": list(COCO_AREA_LABELS),
            "max_detections": list(max_detections),
        },
    }


def evaluate_coco(
    predictions,
    ground_truths,
    num_classes=10,
    iou_threshold=None,
    class_names=None,
    max_detections=COCO_MAX_DETECTIONS,
    include_operating_points=False,
):
    """Evaluate bbox predictions with an explicit, canonical COCO protocol."""
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError(
            "BDD100K evaluation requires pycocotools. "
            "Install the dependencies from requirements.txt."
        )
    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Predictions and ground truths must contain the same number of images"
        )
    if class_names is None:
        class_names = [f"class_{index}" for index in range(1, num_classes + 1)]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must equal num_classes")

    if iou_threshold is None:
        thresholds = COCO_IOU_THRESHOLDS.copy()
    elif isinstance(iou_threshold, (list, tuple, np.ndarray)):
        thresholds = np.asarray(iou_threshold, dtype=np.float64)
    else:
        thresholds = np.asarray([iou_threshold], dtype=np.float64)
    max_detections = tuple(int(value) for value in max_detections)
    if len(max_detections) != 3 or sorted(max_detections) != list(max_detections):
        raise ValueError("max_detections must be three increasing integers")

    ground_truth = build_coco_ground_truth(
        ground_truths, num_classes, class_names
    )
    detections = build_coco_predictions(predictions)
    valid_image_ids = {image["id"] for image in ground_truth["images"]}
    invalid_detection_ids = sorted({row["image_id"] for row in detections} - valid_image_ids)
    if invalid_detection_ids:
        raise ValueError(f"Predictions reference image IDs absent from ground truth: {invalid_detection_ids[:10]}")
    if not ground_truth["annotations"]:
        logger.warning("No ground-truth annotations; returning undefined class AP.")
        result = _empty_results(
            class_names, ground_truth, thresholds, max_detections
        )
        return (_add_operating_points(result, ground_truth, detections, class_names)
                if include_operating_points else result)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()

    if not detections:
        logger.warning("No detections; returning zero AP for classes with GT.")
        result = _empty_results(
            class_names, ground_truth, thresholds, max_detections
        )
        return (_add_operating_points(result, ground_truth, detections, class_names)
                if include_operating_points else result)

    ground_truth_image_ids = [image["id"] for image in ground_truth["images"]]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = ground_truth_image_ids
        coco_eval.params.catIds = list(range(1, num_classes + 1))
        coco_eval.params.iouThrs = thresholds
        coco_eval.params.recThrs = COCO_RECALL_THRESHOLDS.copy()
        coco_eval.params.maxDets = list(max_detections)
        coco_eval.params.areaRng = [list(value) for value in COCO_AREA_RANGES]
        coco_eval.params.areaRngLbl = list(COCO_AREA_LABELS)
        coco_eval.evaluate()
        coco_eval.accumulate()

    precision = coco_eval.eval["precision"]  # T,R,K,A,M
    recall = coco_eval.eval["recall"]  # T,K,A,M

    def threshold_index(value):
        matches = np.where(np.isclose(thresholds, value))[0]
        return int(matches[0]) if matches.size else None

    def ap(area_index=0, threshold=None):
        values = precision[:, :, :, area_index, -1]
        if threshold is not None:
            index = threshold_index(threshold)
            return None if index is None else _mean_valid(values[index])
        return _mean_valid(values)

    def ar(area_index=0):
        return _mean_valid(recall[:, :, area_index, -1])

    per_threshold = {
        float(threshold): (_mean_valid(precision[index, :, :, 0, -1]) or 0.0)
        for index, threshold in enumerate(thresholds)
    }
    per_class = {}
    for category_index, name in enumerate(class_names):
        class_precision = precision[:, :, category_index, 0, -1]
        class_recall = recall[:, category_index, 0, -1]
        index_50 = threshold_index(0.50)
        index_75 = threshold_index(0.75)
        per_class[name] = {
            "AP": _mean_valid(class_precision),
            "AP50": (
                _mean_valid(class_precision[index_50])
                if index_50 is not None
                else None
            ),
            "AP75": (
                _mean_valid(class_precision[index_75])
                if index_75 is not None
                else None
            ),
            "AR100": _mean_valid(class_recall),
        }

    result = {
        "mAP": ap() or 0.0,
        "AP50": ap(threshold=0.50),
        "AP75": ap(threshold=0.75),
        "APS": ap(area_index=1),
        "APM": ap(area_index=2),
        "APL": ap(area_index=3),
        "AR100": ar(),
        "ARS": ar(area_index=1),
        "ARM": ar(area_index=2),
        "ARL": ar(area_index=3),
        "ap_per_class": {
            name: values["AP"] for name, values in per_class.items()
        },
        "per_class": per_class,
        "mAP_per_threshold": per_threshold,
        "coco_protocol": {
            "iou_thresholds": [float(value) for value in thresholds],
            "recall_thresholds": len(COCO_RECALL_THRESHOLDS),
            "area_ranges": [list(value) for value in COCO_AREA_RANGES],
            "area_labels": list(COCO_AREA_LABELS),
            "max_detections": list(max_detections),
            "image_ids": ground_truth_image_ids,
            "category_ids": list(range(1, num_classes + 1)),
        },
    }
    return (_add_operating_points(result, ground_truth, detections, class_names, coco_eval)
            if include_operating_points else result)


def print_eval_results(results, logger_fn=None):
    """Print aggregate and per-class COCO metrics."""
    if logger_fn is None:
        logger_fn = logger.info

    def display(value):
        return "N/A" if value is None else f"{value * 100:.2f}%"

    logger_fn("=" * 72)
    logger_fn(
        "AP50:95={} AP50={} AP75={} APS={} APM={} APL={} AR100={}".format(
            display(results.get("mAP")),
            display(results.get("AP50")),
            display(results.get("AP75")),
            display(results.get("APS")),
            display(results.get("APM")),
            display(results.get("APL")),
            display(results.get("AR100")),
        )
    )
    logger_fn(f"{'Class':<20} {'AP':>10} {'AP50':>10} {'AP75':>10} {'AR100':>10}")
    logger_fn("-" * 64)
    for class_name, values in results.get("per_class", {}).items():
        logger_fn(
            f"{class_name:<20} {display(values['AP']):>10} "
            f"{display(values['AP50']):>10} {display(values['AP75']):>10} "
            f"{display(values['AR100']):>10}"
        )
    logger_fn("=" * 72)
