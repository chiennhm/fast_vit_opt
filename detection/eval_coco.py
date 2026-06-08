#
# COCO-style mAP evaluation using pycocotools
#
# Drop-in replacement for eval_voc.evaluate_voc().  Uses the official
# pycocotools COCOeval engine which is C-accelerated, handles
# COCO-style mAP@[.5:.95] natively, and is far less likely to have
# edge-case bugs than a pure-Python reimplementation.
#
# Falls back to eval_voc.evaluate_voc() if pycocotools is unavailable.
#

import logging
import io
import contextlib

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Try importing pycocotools — graceful fallback
# ============================================================================
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


# ============================================================================
# Convert internal prediction / GT lists → COCO JSON dicts
# ============================================================================
def _build_coco_gt(ground_truths, num_classes, class_names=None):
    """Build a COCO-format ground-truth dict from a list of target dicts.

    Args:
        ground_truths: list of dicts with keys
            'boxes' (N,4), 'labels' (N,), and optionally 'difficults' (N,)
            or 'iscrowd' (N,) or 'area' (N,).
        num_classes: total number of foreground classes.
        class_names: optional list of class name strings.

    Returns:
        coco_gt_dict: dict ready to load via COCO().
    """
    images = []
    annotations = []
    ann_id = 1

    for img_idx, gt in enumerate(ground_truths):
        # Register each image (width/height not used by COCOeval for bbox)
        images.append({
            "id": img_idx,
            "width": 0,
            "height": 0,
        })

        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()

        # difficults / iscrowd
        difficults = gt.get("difficults", None)
        iscrowd = gt.get("iscrowd", None)
        if difficults is not None and isinstance(difficults, torch.Tensor):
            difficults = difficults.cpu().numpy()
        if iscrowd is not None and isinstance(iscrowd, torch.Tensor):
            iscrowd = iscrowd.cpu().numpy()

        # area
        areas = gt.get("area", None)
        if areas is not None and isinstance(areas, torch.Tensor):
            areas = areas.cpu().numpy()

        for i in range(len(gt_boxes)):
            x1, y1, x2, y2 = gt_boxes[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            area = float(areas[i]) if areas is not None else w * h

            # iscrowd: use explicit iscrowd, else treat difficults as crowd
            if iscrowd is not None:
                is_crowd = int(iscrowd[i])
            elif difficults is not None:
                is_crowd = int(difficults[i])
            else:
                is_crowd = 0

            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": int(gt_labels[i]),
                "bbox": [float(x1), float(y1), w, h],  # COCO format: [x, y, w, h]
                "area": area,
                "iscrowd": is_crowd,
            })
            ann_id += 1

    # Categories
    categories = []
    for cls_id in range(1, num_classes + 1):
        name = class_names[cls_id - 1] if class_names else f"class_{cls_id}"
        categories.append({"id": cls_id, "name": name})

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _build_coco_dt(predictions):
    """Build a COCO-format results list from a list of prediction dicts.

    Args:
        predictions: list of dicts with keys
            'boxes' (N,4), 'labels' (N,), 'scores' (N,).

    Returns:
        coco_dt_list: list of result dicts for COCOeval.
    """
    results = []
    for img_idx, pred in enumerate(predictions):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()

        for i in range(len(pred_boxes)):
            x1, y1, x2, y2 = pred_boxes[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            results.append({
                "image_id": img_idx,
                "category_id": int(pred_labels[i]),
                "bbox": [float(x1), float(y1), w, h],
                "score": float(pred_scores[i]),
            })

    return results


# ============================================================================
# Main evaluation entry point
# ============================================================================
def evaluate_coco(
    predictions,
    ground_truths,
    num_classes=20,
    iou_threshold=0.5,
    class_names=None,
):
    """Evaluate detection results using pycocotools COCOeval.

    API-compatible with evaluate_voc() — accepts the same arguments and
    returns the same result dict structure so that callers (object_detection.py)
    need no changes beyond switching the import.

    Args:
        predictions: list of dicts, each with:
            - 'boxes': (N, 4) tensor [x1, y1, x2, y2]
            - 'scores': (N,) tensor
            - 'labels': (N,) tensor (1-indexed)
        ground_truths: list of dicts, each with:
            - 'boxes': (N, 4) tensor [x1, y1, x2, y2]
            - 'labels': (N,) tensor (1-indexed)
            - 'difficults': (N,) tensor of bools (optional, mapped to iscrowd)
        num_classes: number of foreground classes
        iou_threshold: IoU threshold (float or list of floats).
            If a list, computes mAP at each threshold and averages
            (COCO-style mAP@[.5:.95]).
        class_names: list of class names for reporting

    Returns:
        results: dict with 'mAP', 'ap_per_class', and optionally
            'mAP_per_threshold'.
    """
    if not HAS_PYCOCOTOOLS:
        logger.warning(
            "pycocotools not installed — falling back to eval_voc. "
            "Install with: pip install pycocotools"
        )
        from .eval_voc import evaluate_voc
        return evaluate_voc(
            predictions, ground_truths, num_classes=num_classes,
            iou_threshold=iou_threshold, class_names=class_names,
        )

    assert len(predictions) == len(ground_truths)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(1, num_classes + 1)]

    # Normalise threshold to list
    if isinstance(iou_threshold, (list, tuple)):
        thresholds = list(iou_threshold)
    else:
        thresholds = [iou_threshold]

    # ── Build COCO objects ────────────────────────────────────────────────
    gt_dict = _build_coco_gt(ground_truths, num_classes, class_names)

    # Guard: if there are no GT annotations, return zeros
    if len(gt_dict["annotations"]) == 0:
        logger.warning("No ground truth annotations — returning zero mAP.")
        ap_per_class = {
            class_names[i]: None for i in range(num_classes)
        }
        return {"mAP": 0.0, "ap_per_class": ap_per_class}

    # Suppress pycocotools print spam
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()

    dt_list = _build_coco_dt(predictions)

    # Guard: if there are no detections, return zeros
    if len(dt_list) == 0:
        logger.warning("No detections — returning zero mAP.")
        ap_per_class = {
            class_names[i]: 0.0
            for i in range(num_classes)
            if any(
                a["category_id"] == i + 1
                for a in gt_dict["annotations"]
            )
        }
        # Fill classes with no GT as None
        for i in range(num_classes):
            name = class_names[i]
            if name not in ap_per_class:
                ap_per_class[name] = None
        return {"mAP": 0.0, "ap_per_class": ap_per_class}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt_list)

    # ── Run COCOeval ──────────────────────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.iouThrs = np.array(thresholds)
        coco_eval.evaluate()
        coco_eval.accumulate()

    # ── Extract per-class AP at primary (first) threshold ─────────────────
    # coco_eval.eval['precision'] shape:
    #   (T, R, K, A, M) = (nThresholds, nRecallThresholds, nCategories, nAreas, nMaxDets)
    # We want per-class AP at primary threshold, area=all, maxDet=last
    precision = coco_eval.eval["precision"]  # (T, R, K, A, M)
    cat_ids = coco_eval.params.catIds        # list of category ids evaluated

    ap_per_class = {}
    primary_aps = []  # for computing primary mAP
    for k_idx, cat_id in enumerate(cat_ids):
        # precision at primary threshold (idx=0), all recall thresholds,
        # this category, area=all (idx=0), maxDet=last (idx=-1)
        pr_curve = precision[0, :, k_idx, 0, -1]
        if pr_curve.size == 0 or (pr_curve == -1).all():
            # No GT for this category at this threshold
            ap_val = None
        else:
            # Mean over recall thresholds where precision > -1
            valid = pr_curve[pr_curve > -1]
            ap_val = float(np.mean(valid)) if len(valid) > 0 else 0.0
            primary_aps.append(ap_val)

        # Map back to class name
        name_idx = cat_id - 1  # category_ids are 1-indexed
        if 0 <= name_idx < len(class_names):
            ap_per_class[class_names[name_idx]] = ap_val
        else:
            ap_per_class[f"class_{cat_id}"] = ap_val

    # Fill missing classes (no GT) as None
    for i in range(num_classes):
        name = class_names[i]
        if name not in ap_per_class:
            ap_per_class[name] = None

    primary_mAP = float(np.mean(primary_aps)) if primary_aps else 0.0

    result = {
        "mAP": primary_mAP,
        "ap_per_class": ap_per_class,
    }

    # ── Multi-threshold: also report per-threshold mAP ────────────────────
    if len(thresholds) > 1:
        per_thresh_maps = {}
        for t_idx, thresh in enumerate(thresholds):
            t_aps = []
            for k_idx in range(len(cat_ids)):
                pr_curve = precision[t_idx, :, k_idx, 0, -1]
                if pr_curve.size > 0 and not (pr_curve == -1).all():
                    valid = pr_curve[pr_curve > -1]
                    if len(valid) > 0:
                        t_aps.append(float(np.mean(valid)))
            per_thresh_maps[thresh] = float(np.mean(t_aps)) if t_aps else 0.0

        result["mAP_per_threshold"] = per_thresh_maps
        # Overall mAP is mean across thresholds (COCO convention)
        result["mAP"] = float(np.mean(list(per_thresh_maps.values())))

    return result


# Re-export print_eval_results from eval_voc (unchanged formatting)
from .eval_voc import print_eval_results  # noqa: F401, E402
