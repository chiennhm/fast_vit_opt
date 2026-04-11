#
# VOC-style mAP evaluation
#

import torch
import numpy as np
from collections import defaultdict


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
        num_classes: number of classes
        iou_threshold: IoU threshold for positive match
        class_names: list of class names for reporting

    Returns:
        results: dict with 'mAP', 'ap_per_class', 'class_names'
    """
    assert len(predictions) == len(ground_truths)

    # Collect all predictions and ground truths per class
    all_preds = defaultdict(list)  # class_id -> list of (image_idx, score, box)
    all_gts = defaultdict(list)  # class_id -> list of (image_idx, box)
    n_gt_per_class = defaultdict(int)  # class_id -> total GT count

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Ground truths
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()

        for box, label in zip(gt_boxes, gt_labels):
            cls_id = int(label)
            all_gts[cls_id].append((img_idx, box))
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

    # Compute AP per class
    aps = {}
    for cls_id in range(1, num_classes + 1):
        preds = all_preds.get(cls_id, [])
        gts = all_gts.get(cls_id, [])
        n_gt = n_gt_per_class.get(cls_id, 0)

        if n_gt == 0:
            # No ground truths for this class
            aps[cls_id] = 0.0
            continue

        if len(preds) == 0:
            aps[cls_id] = 0.0
            continue

        # Sort predictions by score (descending)
        preds.sort(key=lambda x: -x[1])

        # Build GT lookup: image_idx -> list of (box, matched)
        gt_lookup = defaultdict(list)
        for img_idx, box in gts:
            gt_lookup[img_idx].append({"box": box, "matched": False})

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for pred_idx, (img_idx, score, pred_box) in enumerate(preds):
            img_gts = gt_lookup.get(img_idx, [])
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_info in enumerate(img_gts):
                iou = _compute_iou_numpy(pred_box, gt_info["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                if not img_gts[best_gt_idx]["matched"]:
                    tp[pred_idx] = 1
                    img_gts[best_gt_idx]["matched"] = True
                else:
                    fp[pred_idx] = 1  # Duplicate detection
            else:
                fp[pred_idx] = 1

        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recall, precision)
        aps[cls_id] = ap

    # Compute mAP
    valid_aps = [ap for cls_id, ap in aps.items() if n_gt_per_class.get(cls_id, 0) > 0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0

    # Format results
    if class_names is None:
        class_names = [f"class_{i}" for i in range(1, num_classes + 1)]

    ap_per_class = {}
    for cls_id in range(1, num_classes + 1):
        name = class_names[cls_id - 1] if cls_id - 1 < len(class_names) else f"class_{cls_id}"
        ap_per_class[name] = aps.get(cls_id, 0.0)

    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
    }


def _compute_iou_numpy(box1, box2):
    """Compute IoU between two boxes (numpy arrays)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-7)


def print_eval_results(results):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  VOC mAP@0.5 = {results['mAP']*100:.2f}%")
    print(f"{'='*60}")
    print(f"  {'Class':<20} {'AP':>10}")
    print(f"  {'-'*30}")
    for cls_name, ap in results["ap_per_class"].items():
        print(f"  {cls_name:<20} {ap*100:>10.2f}%")
    print(f"{'='*60}\n")
