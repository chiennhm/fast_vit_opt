#
# VOC-style mAP evaluation
#

import torch
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_ap(recall, precision):
    """Compute Average Precision using the VOC 2010+ method (all-point interpolation).

    Args:
        recall: (N,) array of recall values
        precision: (N,) array of precision values

    Returns:
        ap: average precision value
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _compute_iou_batch(pred_box, gt_boxes_array):
    """Vectorized IoU between one predicted box and multiple GT boxes.

    Args:
        pred_box: (4,) array [x1, y1, x2, y2]
        gt_boxes_array: (G, 4) array [x1, y1, x2, y2]

    Returns:
        ious: (G,) array of IoU values
    """
    x1 = np.maximum(pred_box[0], gt_boxes_array[:, 0])
    y1 = np.maximum(pred_box[1], gt_boxes_array[:, 1])
    x2 = np.minimum(pred_box[2], gt_boxes_array[:, 2])
    y2 = np.minimum(pred_box[3], gt_boxes_array[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area2 = (gt_boxes_array[:, 2] - gt_boxes_array[:, 0]) * \
            (gt_boxes_array[:, 3] - gt_boxes_array[:, 1])

    return inter / (area1 + area2 - inter + 1e-7)


def evaluate_voc(
    predictions,
    ground_truths,
    num_classes=20,
    iou_threshold=0.5,
    class_names=None,
):
    """Evaluate detection results using VOC mAP metric.

    Args:
        predictions: list of dicts, each with:
            - 'boxes': (N, 4) tensor [x1, y1, x2, y2]
            - 'scores': (N,) tensor
            - 'labels': (N,) tensor (1-indexed)
        ground_truths: list of dicts, each with:
            - 'boxes': (N, 4) tensor [x1, y1, x2, y2]
            - 'labels': (N,) tensor (1-indexed)
            - 'difficults': (N,) tensor of bools (optional)
        num_classes: number of classes
        iou_threshold: IoU threshold for positive match (float or list of floats).
            If a list, computes mAP at each threshold and returns the average
            (COCO-style mAP@[.5:.95]).
        class_names: list of class names for reporting

    Returns:
        results: dict with 'mAP', 'ap_per_class', 'class_names'.
            When iou_threshold is a list, also includes 'mAP_per_threshold'.
    """
    # Support list of thresholds (e.g. COCO-style mAP@[.5:.95])
    if isinstance(iou_threshold, (list, tuple)):
        thresholds = list(iou_threshold)
    else:
        thresholds = [iou_threshold]

    assert len(predictions) == len(ground_truths)

    # Collect all predictions and ground truths per class
    all_preds = defaultdict(list)  # class_id -> list of (image_idx, score, box)
    all_gts = defaultdict(list)  # class_id -> list of (image_idx, box, difficult)
    n_gt_per_class = defaultdict(int)  # class_id -> total non-difficult GT count

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Ground truths
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        gt_difficults = gt.get("difficults", None)
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()
        if gt_difficults is not None and isinstance(gt_difficults, torch.Tensor):
            gt_difficults = gt_difficults.cpu().numpy()

        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            cls_id = int(label)
            diff = bool(gt_difficults[i]) if gt_difficults is not None else False
            all_gts[cls_id].append((img_idx, box, diff))
            if not diff:
                n_gt_per_class[cls_id] += 1

        # Predictions
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            cls_id = int(label)
            all_preds[cls_id].append((img_idx, float(score), box))

    # Evaluate at each threshold
    ap_per_threshold = {}
    for thresh in thresholds:
        aps = _evaluate_at_threshold(
            all_preds, all_gts, n_gt_per_class, num_classes, thresh,
        )
        ap_per_threshold[thresh] = aps

    # Use first (or only) threshold for primary results
    primary_thresh = thresholds[0]
    primary_aps = ap_per_threshold[primary_thresh]

    # Compute mAP: only over classes that have GT
    valid_aps = [
        ap for cls_id, ap in primary_aps.items()
        if ap is not None  # None = no GT for this class
    ]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    # Format results
    if class_names is None:
        class_names = [f"class_{i}" for i in range(1, num_classes + 1)]

    ap_per_class = {}
    for cls_id in range(1, num_classes + 1):
        name = class_names[cls_id - 1] if cls_id - 1 < len(class_names) else f"class_{cls_id}"
        ap_per_class[name] = primary_aps.get(cls_id)  # None if no GT

    result = {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
    }

    # Multi-threshold: compute mean across thresholds (COCO-style)
    if len(thresholds) > 1:
        per_thresh_maps = {}
        for thresh, aps in ap_per_threshold.items():
            valid = [ap for ap in aps.values() if ap is not None]
            per_thresh_maps[thresh] = float(np.mean(valid)) if valid else 0.0
        result["mAP_per_threshold"] = per_thresh_maps
        result["mAP"] = float(np.mean(list(per_thresh_maps.values())))

    return result


def _evaluate_at_threshold(all_preds, all_gts, n_gt_per_class, num_classes, iou_threshold):
    """Compute per-class AP at a single IoU threshold.

    Returns:
        aps: dict mapping cls_id -> AP (float), or cls_id -> None if class has no GT.
    """
    aps = {}
    for cls_id in range(1, num_classes + 1):
        preds = all_preds.get(cls_id, [])
        gts = all_gts.get(cls_id, [])
        n_gt = n_gt_per_class.get(cls_id, 0)

        if n_gt == 0:
            # No (non-difficult) ground truths for this class
            aps[cls_id] = None
            continue

        if len(preds) == 0:
            aps[cls_id] = 0.0
            continue

        # Sort predictions by score (descending)
        preds = sorted(preds, key=lambda x: -x[1])

        # Build GT lookup: image_idx -> (boxes_array, difficults_array, matched_array)
        gt_by_image = defaultdict(lambda: {"boxes": [], "difficults": [], "matched": []})
        for img_idx, box, diff in gts:
            gt_by_image[img_idx]["boxes"].append(box)
            gt_by_image[img_idx]["difficults"].append(diff)
            gt_by_image[img_idx]["matched"].append(False)

        # Convert lists to numpy arrays for vectorized IoU
        gt_lookup = {}
        for img_idx, data in gt_by_image.items():
            gt_lookup[img_idx] = {
                "boxes": np.array(data["boxes"]),           # (G, 4)
                "difficults": np.array(data["difficults"]),  # (G,)
                "matched": np.array(data["matched"]),        # (G,)
            }

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for pred_idx, (img_idx, score, pred_box) in enumerate(preds):
            img_gt = gt_lookup.get(img_idx)
            if img_gt is None or len(img_gt["boxes"]) == 0:
                fp[pred_idx] = 1
                continue

            # Vectorized IoU against all GT boxes in this image
            ious = _compute_iou_batch(pred_box, img_gt["boxes"])
            best_gt_idx = int(np.argmax(ious))
            best_iou = ious[best_gt_idx]

            if best_iou >= iou_threshold:
                if img_gt["difficults"][best_gt_idx]:
                    # Match with difficult GT: neither TP nor FP (VOC protocol)
                    continue
                if not img_gt["matched"][best_gt_idx]:
                    tp[pred_idx] = 1
                    img_gt["matched"][best_gt_idx] = True
                else:
                    fp[pred_idx] = 1  # Duplicate detection
            else:
                fp[pred_idx] = 1

        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / n_gt
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-7)

        ap = compute_ap(recall, precision)
        aps[cls_id] = ap

    return aps


def print_eval_results(results, logger_fn=None):
    """Pretty-print evaluation results.

    Args:
        results: dict from evaluate_voc()
        logger_fn: optional callable for logging (e.g. logger.info).
            If None, uses the module logger.
    """
    if logger_fn is None:
        logger_fn = logger.info

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  VOC mAP@0.5 = {results['mAP']*100:.2f}%")
    lines.append(f"{'='*60}")
    lines.append(f"  {'Class':<20} {'AP':>10}")
    lines.append(f"  {'-'*30}")
    for cls_name, ap in results["ap_per_class"].items():
        if ap is None:
            lines.append(f"  {cls_name:<20} {'N/A':>10}")
        else:
            lines.append(f"  {cls_name:<20} {ap*100:>10.2f}%")

    # Multi-threshold summary
    if "mAP_per_threshold" in results:
        lines.append(f"  {'-'*30}")
        for thresh, m in sorted(results["mAP_per_threshold"].items()):
            lines.append(f"  mAP@{thresh:.2f} = {m*100:.2f}%")

    lines.append(f"{'='*60}\n")

    for line in lines:
        logger_fn(line)
