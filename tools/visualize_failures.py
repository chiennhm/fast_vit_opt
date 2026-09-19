#!/usr/bin/env python3
"""
Per-Class Failed Case Visualization and Error Analysis for Object Detection.

This script identifies, categorizes, and visualizes failed detection cases
for each BDD100K class.

Failure Modes Categorized per Class:
1. False Positives (FP):
   - FP Background: Prediction of class C on background (IoU < 0.1)
   - FP Misclass: Prediction of class C where GT is class B != C
   - FP Localization: Prediction of class C matching GT C with 0.1 <= IoU < 0.5
   - FP Duplicate: Second prediction of class C matching already-matched GT C
2. False Negatives (FN):
   - FN Missed: Ground truth of class C not detected by any prediction of class C
   - FN Misclass: Ground truth of class C predicted as class B != C

Outputs per class directory (output/failed_cases_per_class/<class_name>/):
- 3-panel images (GT | Prediction | Error Overlay)
- summary_grid_<class_name>.jpg (Collage of top failed cases)
- failed_cases_summary.csv & failed_cases_summary.json
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Category Definitions
BDD100K_CLASSES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", "train", "truck"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("visualize_failed_cases")


def box_iou_xyxy(boxes1, boxes2):
    """Compute IoU between two sets of boxes [x1, y1, x2, y2]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-7, a_max=None)


def load_predictions_and_gt(pred_json_path, gt_json_path, coco_ann_path=None, num_classes=10):
    """Load predictions and ground truth from JSON files and build image mapping."""
    with open(pred_json_path, "r", encoding="utf-8") as f:
        raw_preds = json.load(f)

    if isinstance(raw_preds, dict) and "predictions" in raw_preds:
        raw_preds = raw_preds["predictions"]

    with open(gt_json_path, "r", encoding="utf-8") as f:
        raw_gts = json.load(f)

    # Build image_id -> file_name mapping
    img_id_to_name = {}

    # Check coco_ann_path first if provided
    if coco_ann_path and os.path.isfile(coco_ann_path):
        with open(coco_ann_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        coco_images = coco_data.get("images", [])
        for idx, img in enumerate(coco_images):
            fn = img.get("file_name", img.get("name"))
            img_id_to_name[img["id"]] = fn
            # Also map 0-indexed dataset index
            img_id_to_name[idx] = fn

    if isinstance(raw_gts, dict):
        if "images" in raw_gts:
            for idx, img in enumerate(raw_gts["images"]):
                fn = img.get("file_name", img.get("name"))
                if fn and img["id"] not in img_id_to_name:
                    img_id_to_name[img["id"]] = fn
                if fn and idx not in img_id_to_name:
                    img_id_to_name[idx] = fn

    # Group GTs by image_id
    gt_by_img = {}
    if isinstance(raw_gts, dict) and "annotations" in raw_gts:
        for ann in raw_gts["annotations"]:
            img_id = ann.get("image_id", ann.get("name"))
            if img_id not in gt_by_img:
                gt_by_img[img_id] = {"boxes": [], "labels": []}
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                xyxy = [x, y, x + w, y + h]
            elif "box2d" in ann:
                b = ann["box2d"]
                xyxy = [b["x1"], b["y1"], b["x2"], b["y2"]]
            else:
                continue

            lbl = ann.get("category_id", ann.get("label", 0))
            if 1 <= lbl <= num_classes:
                lbl = lbl - 1

            gt_by_img[img_id]["boxes"].append(xyxy)
            gt_by_img[img_id]["labels"].append(lbl)

    # Group Predictions by image_id
    pred_by_img = {}
    for pred in raw_preds:
        img_id = pred.get("image_id", pred.get("name"))
        if img_id not in pred_by_img:
            pred_by_img[img_id] = {"boxes": [], "labels": [], "scores": []}

        if "bbox" in pred:
            x, y, w, h = pred["bbox"]
            xyxy = [x, y, x + w, y + h]
        elif "box2d" in pred:
            b = pred["box2d"]
            xyxy = [b["x1"], b["y1"], b["x2"], b["y2"]]
        else:
            continue

        lbl = pred.get("category_id", pred.get("label", 0))
        if 1 <= lbl <= num_classes:
            lbl = lbl - 1

        score = pred.get("score", 1.0)
        pred_by_img[img_id]["boxes"].append(xyxy)
        pred_by_img[img_id]["labels"].append(lbl)
        pred_by_img[img_id]["scores"].append(score)

    all_img_ids = sorted(list(set(list(gt_by_img.keys()) + list(pred_by_img.keys()))))

    parsed_gt = {}
    parsed_pred = {}

    for img_id in all_img_ids:
        g = gt_by_img.get(img_id, {"boxes": [], "labels": []})
        p = pred_by_img.get(img_id, {"boxes": [], "labels": [], "scores": []})

        file_name = img_id_to_name.get(img_id, str(img_id))
        if not file_name.endswith((".jpg", ".png", ".jpeg")):
            file_name = f"{file_name}.jpg"

        parsed_gt[img_id] = {
            "file_name": file_name,
            "boxes": np.array(g["boxes"], dtype=np.float32) if g["boxes"] else np.zeros((0, 4), dtype=np.float32),
            "labels": np.array(g["labels"], dtype=np.int64) if g["labels"] else np.zeros((0,), dtype=np.int64),
        }
        parsed_pred[img_id] = {
            "file_name": file_name,
            "boxes": np.array(p["boxes"], dtype=np.float32) if p["boxes"] else np.zeros((0, 4), dtype=np.float32),
            "labels": np.array(p["labels"], dtype=np.int64) if p["labels"] else np.zeros((0,), dtype=np.int64),
            "scores": np.array(p["scores"], dtype=np.float32) if p["scores"] else np.zeros((0,), dtype=np.float32),
        }

    return parsed_pred, parsed_gt, img_id_to_name


def analyze_failed_cases(predictions, ground_truths, class_names, score_thresh=0.3, iou_thresh=0.5):
    """Categorize detection errors for each class.

    Returns:
        per_class_failures: dict mapping class_idx -> list of error dicts
        metrics_summary: per-class counts of TP, FP types, FN types
    """
    num_classes = len(class_names)
    per_class_failures = {c: [] for c in range(num_classes)}
    per_class_counts = {
        c: {
            "total_gt": 0, "total_pred": 0, "TP": 0,
            "FP_background": 0, "FP_misclass": 0, "FP_localization": 0, "FP_duplicate": 0,
            "FN_missed": 0, "FN_misclass": 0
        }
        for c in range(num_classes)
    }

    for img_id, g_data in ground_truths.items():
        p_data = predictions.get(img_id, {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "scores": np.zeros((0,), dtype=np.float32),
        })

        file_name = g_data["file_name"]
        g_boxes, g_labels = g_data["boxes"], g_data["labels"]
        p_boxes, p_labels, p_scores = p_data["boxes"], p_data["labels"], p_data["scores"]

        # Filter predictions by score threshold
        if len(p_scores) > 0:
            mask = p_scores >= score_thresh
            p_boxes = p_boxes[mask]
            p_labels = p_labels[mask]
            p_scores = p_scores[mask]

        # Record GT counts
        for lbl in g_labels:
            if 0 <= lbl < num_classes:
                per_class_counts[lbl]["total_gt"] += 1

        # Record Pred counts
        for lbl in p_labels:
            if 0 <= lbl < num_classes:
                per_class_counts[lbl]["total_pred"] += 1

        # Overall IoU matrix between all predictions and all GTs in image
        all_ious = box_iou_xyxy(p_boxes, g_boxes) if (len(p_boxes) > 0 and len(g_boxes) > 0) else np.zeros((len(p_boxes), len(g_boxes)))

        # Track GT matching status
        matched_gt = np.zeros(len(g_boxes), dtype=bool)

        # Process each class
        for c in range(num_classes):
            c_p_indices = np.where(p_labels == c)[0]
            c_g_indices = np.where(g_labels == c)[0]

            if len(c_p_indices) == 0 and len(c_g_indices) == 0:
                continue

            # Sort predictions of class c by score descending
            if len(c_p_indices) > 0:
                c_scores = p_scores[c_p_indices]
                sort_order = np.argsort(-c_scores)
                c_p_indices = c_p_indices[sort_order]

            # 1. Match Predictions of Class C
            for p_idx in c_p_indices:
                box_p = p_boxes[p_idx]
                score_p = float(p_scores[p_idx])

                if len(g_boxes) == 0:
                    # No GT in image -> FP Background
                    per_class_counts[c]["FP_background"] += 1
                    per_class_failures[c].append({
                        "img_id": img_id,
                        "file_name": file_name,
                        "error_type": "FP_background",
                        "target_class": c,
                        "pred_box": box_p.tolist(),
                        "pred_score": score_p,
                        "gt_box": None,
                        "gt_class": None,
                        "iou": 0.0,
                        "all_pred_boxes": p_boxes.tolist(),
                        "all_pred_labels": p_labels.tolist(),
                        "all_pred_scores": p_scores.tolist(),
                        "all_gt_boxes": g_boxes.tolist(),
                        "all_gt_labels": g_labels.tolist(),
                    })
                    continue

                # Find best GT match across all GTs
                gt_ious = all_ious[p_idx]
                best_gt_idx = np.argmax(gt_ious) if len(gt_ious) > 0 else -1
                max_iou = float(gt_ious[best_gt_idx]) if best_gt_idx >= 0 else 0.0

                if max_iou >= iou_thresh:
                    gt_cls = g_labels[best_gt_idx]
                    if gt_cls == c:
                        if not matched_gt[best_gt_idx]:
                            # True Positive
                            matched_gt[best_gt_idx] = True
                            per_class_counts[c]["TP"] += 1
                        else:
                            # Duplicate FP on same GT
                            per_class_counts[c]["FP_duplicate"] += 1
                            per_class_failures[c].append({
                                "img_id": img_id,
                                "file_name": file_name,
                                "error_type": "FP_duplicate",
                                "target_class": c,
                                "pred_box": box_p.tolist(),
                                "pred_score": score_p,
                                "gt_box": g_boxes[best_gt_idx].tolist(),
                                "gt_class": int(gt_cls),
                                "iou": max_iou,
                                "all_pred_boxes": p_boxes.tolist(),
                                "all_pred_labels": p_labels.tolist(),
                                "all_pred_scores": p_scores.tolist(),
                                "all_gt_boxes": g_boxes.tolist(),
                                "all_gt_labels": g_labels.tolist(),
                            })
                    else:
                        # FP Misclass: Model predicted class C, but GT was class B
                        per_class_counts[c]["FP_misclass"] += 1
                        per_class_failures[c].append({
                            "img_id": img_id,
                            "file_name": file_name,
                            "error_type": "FP_misclass",
                            "target_class": c,
                            "pred_box": box_p.tolist(),
                            "pred_score": score_p,
                            "gt_box": g_boxes[best_gt_idx].tolist(),
                            "gt_class": int(gt_cls),
                            "iou": max_iou,
                            "all_pred_boxes": p_boxes.tolist(),
                            "all_pred_labels": p_labels.tolist(),
                            "all_pred_scores": p_scores.tolist(),
                            "all_gt_boxes": g_boxes.tolist(),
                            "all_gt_labels": g_labels.tolist(),
                        })
                elif max_iou >= 0.1:
                    gt_cls = g_labels[best_gt_idx]
                    if gt_cls == c:
                        # FP Localization
                        per_class_counts[c]["FP_localization"] += 1
                        per_class_failures[c].append({
                            "img_id": img_id,
                            "file_name": file_name,
                            "error_type": "FP_localization",
                            "target_class": c,
                            "pred_box": box_p.tolist(),
                            "pred_score": score_p,
                            "gt_box": g_boxes[best_gt_idx].tolist(),
                            "gt_class": int(gt_cls),
                            "iou": max_iou,
                            "all_pred_boxes": p_boxes.tolist(),
                            "all_pred_labels": p_labels.tolist(),
                            "all_pred_scores": p_scores.tolist(),
                            "all_gt_boxes": g_boxes.tolist(),
                            "all_gt_labels": g_labels.tolist(),
                        })
                    else:
                        # FP Misclass with low IoU
                        per_class_counts[c]["FP_misclass"] += 1
                        per_class_failures[c].append({
                            "img_id": img_id,
                            "file_name": file_name,
                            "error_type": "FP_misclass",
                            "target_class": c,
                            "pred_box": box_p.tolist(),
                            "pred_score": score_p,
                            "gt_box": g_boxes[best_gt_idx].tolist(),
                            "gt_class": int(gt_cls),
                            "iou": max_iou,
                            "all_pred_boxes": p_boxes.tolist(),
                            "all_pred_labels": p_labels.tolist(),
                            "all_pred_scores": p_scores.tolist(),
                            "all_gt_boxes": g_boxes.tolist(),
                            "all_gt_labels": g_labels.tolist(),
                        })
                else:
                    # FP Background
                    per_class_counts[c]["FP_background"] += 1
                    per_class_failures[c].append({
                        "img_id": img_id,
                        "file_name": file_name,
                        "error_type": "FP_background",
                        "target_class": c,
                        "pred_box": box_p.tolist(),
                        "pred_score": score_p,
                        "gt_box": None,
                        "gt_class": None,
                        "iou": 0.0,
                        "all_pred_boxes": p_boxes.tolist(),
                        "all_pred_labels": p_labels.tolist(),
                        "all_pred_scores": p_scores.tolist(),
                        "all_gt_boxes": g_boxes.tolist(),
                        "all_gt_labels": g_labels.tolist(),
                    })

            # 2. Check False Negatives for GTs of Class C
            for g_idx in c_g_indices:
                if matched_gt[g_idx]:
                    continue

                box_g = g_boxes[g_idx]
                if len(p_boxes) > 0:
                    p_ious = all_ious[:, g_idx]
                    best_p_idx = np.argmax(p_ious) if len(p_ious) > 0 else -1
                    max_p_iou = float(p_ious[best_p_idx]) if best_p_idx >= 0 else 0.0
                else:
                    best_p_idx = -1
                    max_p_iou = 0.0

                if max_p_iou >= 0.3 and best_p_idx >= 0:
                    pred_cls = p_labels[best_p_idx]
                    pred_score = float(p_scores[best_p_idx])
                    # FN Misclass: GT was class C, but model predicted class B
                    per_class_counts[c]["FN_misclass"] += 1
                    per_class_failures[c].append({
                        "img_id": img_id,
                        "file_name": file_name,
                        "error_type": "FN_misclass",
                        "target_class": c,
                        "pred_box": p_boxes[best_p_idx].tolist(),
                        "pred_score": pred_score,
                        "gt_box": box_g.tolist(),
                        "gt_class": c,
                        "other_pred_class": int(pred_cls),
                        "iou": max_p_iou,
                        "all_pred_boxes": p_boxes.tolist(),
                        "all_pred_labels": p_labels.tolist(),
                        "all_pred_scores": p_scores.tolist(),
                        "all_gt_boxes": g_boxes.tolist(),
                        "all_gt_labels": g_labels.tolist(),
                    })
                else:
                    # FN Missed GT
                    per_class_counts[c]["FN_missed"] += 1
                    per_class_failures[c].append({
                        "img_id": img_id,
                        "file_name": file_name,
                        "error_type": "FN_missed",
                        "target_class": c,
                        "pred_box": None,
                        "pred_score": 0.0,
                        "gt_box": box_g.tolist(),
                        "gt_class": c,
                        "iou": 0.0,
                        "all_pred_boxes": p_boxes.tolist(),
                        "all_pred_labels": p_labels.tolist(),
                        "all_pred_scores": p_scores.tolist(),
                        "all_gt_boxes": g_boxes.tolist(),
                        "all_gt_labels": g_labels.tolist(),
                    })

    return per_class_failures, per_class_counts


def draw_3panel_failed_case(
    raw_img,
    failure_info,
    class_names,
    score_thresh=0.3
):
    """Draw a 3-panel visualization for a single failed case.

    Panels:
    1. GROUND TRUTH (Green boxes)
    2. PREDICTION (Blue/Yellow boxes)
    3. ERROR ANALYSIS OVERLAY (Color-coded: Red FP, Orange Misclass, Blue FN, Green TP)
    """
    w, h = raw_img.size

    # Detect if boxes are in model evaluation space and calculate scale factors
    eval_scale = min(800.0 / max(min(w, h), 1), 1333.0 / max(max(w, h), 1))
    w_eval = int(round(w * eval_scale))
    h_eval = int(round(h * eval_scale))

    max_x = 0
    for b in failure_info["all_gt_boxes"] + failure_info["all_pred_boxes"]:
        if len(b) == 4:
            max_x = max(max_x, b[2])

    if max_x > w * 1.01:
        scale_x = w / float(w_eval)
        scale_y = h / float(h_eval)
    else:
        scale_x = 1.0
        scale_y = 1.0

    def _scale(box):
        if box is None:
            return None
        return [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]

    header_h = 45
    combined_w = w * 3
    combined_h = h + header_h

    canvas = Image.new("RGB", (combined_w, combined_h), (25, 25, 25))

    # Panel 1: Ground Truth
    img_gt = raw_img.copy()
    draw_gt = ImageDraw.Draw(img_gt)
    all_gt_boxes = failure_info["all_gt_boxes"]
    all_gt_labels = failure_info["all_gt_labels"]

    try:
        font_sm = ImageFont.truetype("arial.ttf", 13)
        font_hdr = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font_sm = ImageFont.load_default()
        font_hdr = ImageFont.load_default()

    for box, lbl in zip(all_gt_boxes, all_gt_labels):
        x1, y1, x2, y2 = _scale(box)
        c_name = class_names[lbl] if 0 <= lbl < len(class_names) else f"cls_{lbl}"
        for dw in range(2):
            draw_gt.rectangle([x1 - dw, y1 - dw, x2 + dw, y2 + dw], outline=(0, 220, 100))
        draw_gt.rectangle([x1, max(0, y1 - 18), x1 + len(c_name) * 8 + 6, y1], fill=(0, 180, 80))
        draw_gt.text((x1 + 3, max(0, y1 - 16)), c_name, fill=(255, 255, 255), font=font_sm)

    # Panel 2: Predictions
    img_pred = raw_img.copy()
    draw_pred = ImageDraw.Draw(img_pred)
    all_pred_boxes = failure_info["all_pred_boxes"]
    all_pred_labels = failure_info["all_pred_labels"]
    all_pred_scores = failure_info["all_pred_scores"]

    for box, lbl, sc in zip(all_pred_boxes, all_pred_labels, all_pred_scores):
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = _scale(box)
        c_name = class_names[lbl] if 0 <= lbl < len(class_names) else f"cls_{lbl}"
        tag = f"{c_name}: {sc:.2f}"
        for dw in range(2):
            draw_pred.rectangle([x1 - dw, y1 - dw, x2 + dw, y2 + dw], outline=(255, 200, 0))
        draw_pred.rectangle([x1, max(0, y1 - 18), x1 + len(tag) * 7.5 + 6, y1], fill=(200, 150, 0))
        draw_pred.text((x1 + 3, max(0, y1 - 16)), tag, fill=(255, 255, 255), font=font_sm)

    # Panel 3: Error Overlay
    img_err = raw_img.copy()
    draw_err = ImageDraw.Draw(img_err)

    err_type = failure_info["error_type"]
    target_cls = failure_info["target_class"]
    target_name = class_names[target_cls]

    # Draw all GT as thin gray lines for context
    for box, lbl in zip(all_gt_boxes, all_gt_labels):
        x1, y1, x2, y2 = _scale(box)
        draw_err.rectangle([x1, y1, x2, y2], outline=(120, 120, 120))

    # Draw target failure highlight
    if err_type.startswith("FP"):
        p_box = _scale(failure_info["pred_box"])
        sc = failure_info["pred_score"]
        x1, y1, x2, y2 = p_box

        if err_type == "FP_misclass":
            color = (255, 120, 0) # Orange
            gt_cls = failure_info.get("gt_class")
            gt_name = class_names[gt_cls] if (gt_cls is not None and 0 <= gt_cls < len(class_names)) else "Unknown"
            tag = f"FP Misclass: Pred {target_name} ({sc:.2f}) vs GT {gt_name}"
        elif err_type == "FP_localization":
            color = (255, 50, 50) # Red
            tag = f"FP Loc: {target_name} ({sc:.2f}) IoU={failure_info['iou']:.2f}"
        elif err_type == "FP_duplicate":
            color = (255, 50, 150) # Pink-Red
            tag = f"FP Dup: {target_name} ({sc:.2f})"
        else:
            color = (255, 0, 0) # Bright Red
            tag = f"FP Background: {target_name} ({sc:.2f})"

        for dw in range(3):
            draw_err.rectangle([x1 - dw, y1 - dw, x2 + dw, y2 + dw], outline=color)
        t_w = len(tag) * 7.5 + 8
        draw_err.rectangle([x1, max(0, y1 - 22), x1 + t_w, y1], fill=color)
        draw_err.text((x1 + 4, max(0, y1 - 19)), tag, fill=(255, 255, 255), font=font_sm)

        # Draw corresponding GT box if exists
        if failure_info.get("gt_box"):
            gx1, gy1, gx2, gy2 = _scale(failure_info["gt_box"])
            for dw in range(2):
                draw_err.rectangle([gx1 - dw, gy1 - dw, gx2 + dw, gy2 + dw], outline=(0, 200, 255))

    elif err_type.startswith("FN"):
        g_box = _scale(failure_info["gt_box"])
        x1, y1, x2, y2 = g_box

        if err_type == "FN_misclass":
            color = (220, 100, 255) # Purple
            other_cls = failure_info.get("other_pred_class")
            other_name = class_names[other_cls] if (other_cls is not None and 0 <= other_cls < len(class_names)) else "Other"
            tag = f"FN Misclass: GT {target_name} predicted as {other_name}"
        else:
            color = (0, 150, 255) # Blue
            tag = f"FN Missed GT: {target_name}"

        for dw in range(3):
            draw_err.rectangle([x1 - dw, y1 - dw, x2 + dw, y2 + dw], outline=color)
        t_w = len(tag) * 7.5 + 8
        draw_err.rectangle([x1, max(0, y1 - 22), x1 + t_w, y1], fill=color)
        draw_err.text((x1 + 4, max(0, y1 - 19)), tag, fill=(255, 255, 255), font=font_sm)

    # Paste panels
    canvas.paste(img_gt, (0, header_h))
    canvas.paste(img_pred, (w, header_h))
    canvas.paste(img_err, (w * 2, header_h))

    # Header Titles
    draw_canvas = ImageDraw.Draw(canvas)
    titles = [
        f"GROUND TRUTH (Class: {target_name})",
        f"PREDICTIONS (Score >= {score_thresh})",
        f"ERROR ANALYSIS OVERLAY ({err_type})"
    ]

    for idx, title in enumerate(titles):
        px = idx * w
        bbox = draw_canvas.textbbox((0, 0), title, font=font_hdr)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pos_x = px + (w - tw) // 2
        pos_y = (header_h - th) // 2
        draw_canvas.text((pos_x, pos_y), title, fill=(255, 255, 255), font=font_hdr)

    return canvas


def create_summary_grid(image_paths, class_name, save_path, cols=3, thumb_size=(400, 300)):
    """Create a multi-image collage grid summarizing top failed cases for a class."""
    if not image_paths:
        return

    n_imgs = len(image_paths)
    rows = (n_imgs + cols - 1) // cols

    tw, th = thumb_size
    header_h = 50
    grid_w = cols * tw
    grid_h = rows * th + header_h

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (IOError, OSError):
        font = ImageFont.load_default()

    title = f"FAILED CASES SUMMARY GRID FOR CLASS: {class_name.upper()} ({n_imgs} Examples)"
    bbox = draw.textbbox((0, 0), title, font=font)
    tw_title = bbox[2] - bbox[0]
    draw.text(((grid_w - tw_title) // 2, 12), title, fill=(255, 215, 0), font=font)

    for i, p in enumerate(image_paths):
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        im.thumbnail(thumb_size)

        r = i // cols
        c = i % cols
        x = c * tw + (tw - im.width) // 2
        y = r * th + header_h + (th - im.height) // 2
        canvas.paste(im, (x, y))

    canvas.save(save_path)
    logger.info(f"Saved class summary grid collage to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-class detection failed cases (FP background, FP misclass, FN missed, etc.)"
    )
    parser.add_argument(
        "--json-dir", type=str, default=None,
        help="Directory containing predictions_*.json and ground_truths_*.json"
    )
    parser.add_argument("--pred-json", type=str, default=None, help="Path to predictions JSON")
    parser.add_argument("--gt-json", type=str, default=None, help="Path to ground truths JSON")
    parser.add_argument(
        "--coco-ann", type=str, default=None,
        help="Path to COCO instance annotations file (e.g. ./data/bdd100k/val/annotations/coco_instances_val.json)"
    )
    parser.add_argument(
        "--img-dir", type=str, default=None,
        help="Path to image directory (e.g. ./data/bdd100k/val/images)"
    )
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Score threshold for predictions")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument(
        "--max-samples-per-class", type=int, default=10,
        help="Maximum failed case images to generate per class"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./output/failed_cases_per_class",
        help="Directory to save per-class failed case visualizations"
    )
    args = parser.parse_args()

    # Auto-detect JSON files if json-dir given
    if args.json_dir and os.path.isdir(args.json_dir):
        jdir = Path(args.json_dir)
        if not args.pred_json:
            p_cand = list(jdir.glob("predictions_*.json")) + list(jdir.glob("predictions.json"))
            if p_cand:
                args.pred_json = str(p_cand[0])
        if not args.gt_json:
            g_cand = list(jdir.glob("ground_truths_*.json")) + list(jdir.glob("ground_truth.json"))
            if g_cand:
                args.gt_json = str(g_cand[0])

    if not args.pred_json or not os.path.isfile(args.pred_json):
        logger.error(f"Predictions JSON file not found: {args.pred_json}")
        sys.exit(1)

    if not args.gt_json or not os.path.isfile(args.gt_json):
        logger.error(f"Ground truth JSON file not found: {args.gt_json}")
        sys.exit(1)

    class_names = BDD100K_CLASSES
    if not args.img_dir:
        args.img_dir = "./data/bdd100k/images/100k/val"
    if not args.coco_ann and os.path.isfile("./data/bdd100k/annotations/bdd100k_det_val_coco.json"):
        args.coco_ann = "./data/bdd100k/annotations/bdd100k_det_val_coco.json"

    logger.info("Loading BDD100K predictions and ground truth...")
    parsed_pred, parsed_gt, img_id_map = load_predictions_and_gt(
        args.pred_json, args.gt_json, args.coco_ann, num_classes=len(class_names)
    )

    logger.info(f"Categorizing per-class detection errors across {len(parsed_gt)} images...")
    failures, counts = analyze_failed_cases(
        parsed_pred, parsed_gt, class_names,
        score_thresh=args.score_thresh, iou_thresh=args.iou_thresh
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Print Per-Class Summary Table
    print("\n" + "=" * 110)
    print(f"{'Class Name':<16} | {'Total GT':<9} | {'Preds':<7} | {'TP':<6} | {'FP (BG)':<8} | {'FP (Mis)':<8} | {'FP (Loc)':<8} | {'FN (Mis)':<8} | {'FN (Misclass)':<12}")
    print("-" * 110)
    for c, name in enumerate(class_names):
        cnt = counts[c]
        print(f"{name:<16} | {cnt['total_gt']:<9} | {cnt['total_pred']:<7} | {cnt['TP']:<6} | {cnt['FP_background']:<8} | {cnt['FP_misclass']:<8} | {cnt['FP_localization']:<8} | {cnt['FN_missed']:<8} | {cnt['FN_misclass']:<12}")
    print("=" * 110)

    # Save summary CSV & JSON
    csv_path = os.path.join(args.save_dir, "failed_cases_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Class,Total_GT,Total_Pred,TP,FP_background,FP_misclass,FP_localization,FP_duplicate,FN_missed,FN_misclass,Precision,Recall\n")
        for c, name in enumerate(class_names):
            cnt = counts[c]
            tp = cnt['TP']
            fp = cnt['FP_background'] + cnt['FP_misclass'] + cnt['FP_localization'] + cnt['FP_duplicate']
            fn = cnt['FN_missed'] + cnt['FN_misclass']
            prec = tp / max(tp + fp, 1e-7)
            rec = tp / max(cnt['total_gt'], 1e-7)
            f.write(f"{name},{cnt['total_gt']},{cnt['total_pred']},{tp},"
                    f"{cnt['FP_background']},{cnt['FP_misclass']},{cnt['FP_localization']},{cnt['FP_duplicate']},"
                    f"{cnt['FN_missed']},{cnt['FN_misclass']},{prec:.4f},{rec:.4f}\n")
    logger.info(f"Saved failed case summary CSV to: {csv_path}")

    json_path = os.path.join(args.save_dir, "failed_cases_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "class_names": class_names}, f, indent=2)

    # Generate Image Visualizations per Class
    logger.info(f"Generating 3-panel visualization images per class (max {args.max_samples_per_class} per class)...")

    for c, name in enumerate(class_names):
        c_failures = failures[c]
        if not c_failures:
            continue

        class_out_dir = os.path.join(args.save_dir, name.replace(" ", "_"))
        os.makedirs(class_out_dir, exist_ok=True)

        # Sort failures to get diverse/important ones (e.g. high-score FP, FN)
        # Prioritize FP_misclass, FP_background, FN_missed
        c_failures.sort(key=lambda x: (0 if "misclass" in x["error_type"] else 1, -x.get("pred_score", 0.0)))

        rendered_paths = []
        for rank, item in enumerate(c_failures[:args.max_samples_per_class]):
            fn = item["file_name"]
            img_path = os.path.join(args.img_dir, fn)
            if not os.path.isfile(img_path):
                # Try finding in img_dir directly or subfolders
                cand = list(Path(args.img_dir).rglob(fn))
                if cand:
                    img_path = str(cand[0])
                else:
                    continue

            try:
                raw_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not open image {img_path}: {e}")
                continue

            panel_img = draw_3panel_failed_case(
                raw_img, item, class_names, score_thresh=args.score_thresh
            )
            save_name = f"{item['error_type']}_rank{rank+1}_{Path(fn).stem}.jpg"
            save_path = os.path.join(class_out_dir, save_name)
            panel_img.save(save_path)
            rendered_paths.append(save_path)

        # Create summary grid collage for class
        if rendered_paths:
            grid_path = os.path.join(class_out_dir, f"summary_grid_{name.replace(' ', '_')}.jpg")
            create_summary_grid(rendered_paths[:9], name, grid_path)

    logger.info(f"Per-class failed case visualization complete! Outputs saved in: {os.path.abspath(args.save_dir)}\n")


if __name__ == "__main__":
    main()
