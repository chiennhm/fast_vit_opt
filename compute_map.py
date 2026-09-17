#!/usr/bin/env python3
"""
Compute mAP, Precision-Recall Curves, and ROC Curves from saved JSON prediction files.

This script loads offline prediction and ground truth files saved during evaluation
(e.g., by object_detection.py with --save-json flag), computes detection metrics
(mAP@0.5, mAP@[0.5:0.95], per-class AP, ROC-AUC, Best F1), and generates
high-resolution Precision-Recall and ROC curve plots.

Usage:
    # Auto-detect predictions and ground truths from json directory
    python compute_map.py --json-dir ./output/bdd100k_retina/fastvit_sa12_20260826_183041/json_results

    # Explicit file paths and custom save directory
    python compute_map.py \
        --pred-json ./json_results/predictions_BDD100K.json \
        --gt-json ./json_results/ground_truths_BDD100K.json \
        --dataset bdd100k \
        --save-dir ./eval_plots
"""

import os
import sys
import glob
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Category Definitions (1-indexed category_id 1..10 mapped to index 0..9)
# ============================================================================
BDD100K_CLASSES = [
    "bike",          # cat_id 1
    "bus",           # cat_id 2
    "car",           # cat_id 3
    "motor",         # cat_id 4
    "person",        # cat_id 5
    "rider",         # cat_id 6
    "traffic light", # cat_id 7
    "traffic sign",  # cat_id 8
    "train",         # cat_id 9
    "truck",         # cat_id 10
]

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("compute_map")


# ============================================================================
# Geometry and Metric Utilities
# ============================================================================
def box_iou_xyxy(boxes1, boxes2):
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


def compute_coco_101_point_ap(recalls, precisions):
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_thresholds, side='left')
    return float(np.mean(mpre[inds]))


# ============================================================================
# JSON Loading Functions
# ============================================================================
def load_predictions_from_json(pred_json_path, num_classes=10):
    with open(pred_json_path, "r", encoding="utf-8") as f:
        raw_preds = json.load(f)

    if isinstance(raw_preds, dict) and "predictions" in raw_preds:
        raw_preds = raw_preds["predictions"]

    grouped_predictions = {}
    for pred in raw_preds:
        img_id = pred.get("image_id", pred.get("name"))
        if img_id not in grouped_predictions:
            grouped_predictions[img_id] = {
                "boxes": [],
                "labels": [],
                "scores": [],
            }

        if "bbox" in pred:
            bbox = pred["bbox"]
            if len(bbox) == 4:
                x, y, w, h = bbox
                grouped_predictions[img_id]["boxes"].append([x, y, x + w, y + h])
        elif "box2d" in pred:
            b = pred["box2d"]
            grouped_predictions[img_id]["boxes"].append([b["x1"], b["y1"], b["x2"], b["y2"]])

        label = pred.get("category_id", pred.get("label", 0))
        # If 1-indexed (1..10), convert to 0-indexed (0..9)
        if 1 <= label <= num_classes:
            label = label - 1

        score = pred.get("score", 1.0)
        grouped_predictions[img_id]["labels"].append(label)
        grouped_predictions[img_id]["scores"].append(score)

    predictions = {}
    for img_id, pred in grouped_predictions.items():
        predictions[img_id] = {
            "boxes": np.array(pred["boxes"], dtype=np.float32) if pred["boxes"] else np.zeros((0, 4), dtype=np.float32),
            "labels": np.array(pred["labels"], dtype=np.int64) if pred["labels"] else np.zeros((0,), dtype=np.int64),
            "scores": np.array(pred["scores"], dtype=np.float32) if pred["scores"] else np.zeros((0,), dtype=np.float32),
        }
    return predictions


def load_ground_truths_from_json(gt_json_path, num_classes=10):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        coco_gt = json.load(f)

    category_names = []
    annotations_by_image = {}
    if isinstance(coco_gt, dict) and "annotations" in coco_gt:
        for ann in coco_gt["annotations"]:
            img_id = ann.get("image_id", ann.get("name"))
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []

            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                xyxy = [x, y, x + w, y + h]
            elif "box2d" in ann:
                b = ann["box2d"]
                xyxy = [b["x1"], b["y1"], b["x2"], b["y2"]]
            else:
                xyxy = [0, 0, 0, 0]

            label = ann.get("category_id", ann.get("label", 0))
            if 1 <= label <= num_classes:
                label = label - 1

            annotations_by_image[img_id].append({
                "boxes": xyxy,
                "labels": label,
                "iscrowd": ann.get("iscrowd", 0),
            })
    elif isinstance(coco_gt, list):
        for item in coco_gt:
            img_id = item.get("image_id", item.get("name"))
            labels = item.get("labels", [])
            for lbl in labels:
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                if "box2d" in lbl:
                    b = lbl["box2d"]
                    xyxy = [b["x1"], b["y1"], b["x2"], b["y2"]]
                elif "bbox" in lbl:
                    x, y, w, h = lbl["bbox"]
                    xyxy = [x, y, x + w, y + h]
                else:
                    xyxy = [0, 0, 0, 0]

                label = lbl.get("category_id", lbl.get("label", 0))
                if 1 <= label <= num_classes:
                    label = label - 1

                annotations_by_image[img_id].append({
                    "boxes": xyxy,
                    "labels": label,
                    "iscrowd": lbl.get("iscrowd", 0),
                })

    img_ids = set()
    if isinstance(coco_gt, dict) and "images" in coco_gt:
        img_ids.update(img.get("id", img.get("name")) for img in coco_gt["images"])
    img_ids.update(annotations_by_image.keys())

    ground_truths = {}
    for img_id in img_ids:
        if img_id in annotations_by_image:
            anns = annotations_by_image[img_id]
            ground_truths[img_id] = {
                "boxes": np.array([a["boxes"] for a in anns], dtype=np.float32),
                "labels": np.array([a["labels"] for a in anns], dtype=np.int64),
                "iscrowds": np.array([a["iscrowd"] for a in anns], dtype=np.int64),
            }
        else:
            ground_truths[img_id] = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "iscrowds": np.zeros((0,), dtype=np.int64),
            }

    return ground_truths, category_names


# ============================================================================
# Metric Evaluation & Curve Construction
# ============================================================================
def evaluate_detection_curves(predictions, ground_truths, class_names, iou_thresh=0.5):
    num_classes = len(class_names)
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    all_img_ids = sorted(list(set(list(predictions.keys()) + list(ground_truths.keys()))))

    processed_images = []
    num_gt_by_class = {c: 0 for c in range(num_classes)}

    for img_id in all_img_ids:
        p_data = predictions.get(img_id, {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "scores": np.zeros((0,), dtype=np.float32),
        })
        g_data = ground_truths.get(img_id, {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "iscrowds": np.zeros((0,), dtype=np.int64),
        })

        p_boxes, p_labels, p_scores = p_data["boxes"], p_data["labels"], p_data["scores"]
        g_boxes, g_labels = g_data["boxes"], g_data["labels"]
        g_iscrowds = g_data.get("iscrowds", np.zeros(len(g_boxes), dtype=np.int64))

        img_class_data = {}
        for c in range(num_classes):
            c_gt_mask = (g_labels == c)
            c_g_boxes = g_boxes[c_gt_mask] if len(g_boxes) > 0 else np.zeros((0, 4))
            c_g_crowds = g_iscrowds[c_gt_mask] if len(g_iscrowds) > 0 else np.zeros(0)

            valid_gts = np.sum(c_g_crowds == 0)
            num_gt_by_class[c] += int(valid_gts)

            c_p_mask = (p_labels == c)
            c_p_boxes = p_boxes[c_p_mask] if len(p_boxes) > 0 else np.zeros((0, 4))
            c_p_scores = p_scores[c_p_mask] if len(p_scores) > 0 else np.zeros(0)

            if len(c_p_boxes) > 0:
                sort_idx = np.argsort(-c_p_scores)
                c_p_boxes = c_p_boxes[sort_idx]
                c_p_scores = c_p_scores[sort_idx]
                ious = box_iou_xyxy(c_p_boxes, c_g_boxes) if len(c_g_boxes) > 0 else np.zeros((len(c_p_boxes), 0))
            else:
                ious = np.zeros((0, len(c_g_boxes)), dtype=np.float32)

            img_class_data[c] = {
                "scores": c_p_scores,
                "g_crowds": c_g_crowds,
                "num_gt_boxes": len(c_g_boxes),
                "ious": ious
            }

        processed_images.append(img_class_data)

    def eval_at_thresh(t_iou):
        preds_by_class = {c: [] for c in range(num_classes)}

        for img_class_data in processed_images:
            for c in range(num_classes):
                c_data = img_class_data[c]
                scores = c_data["scores"]
                if len(scores) == 0:
                    continue

                g_crowds = c_data["g_crowds"]
                num_gt_boxes = c_data["num_gt_boxes"]
                ious = c_data["ious"]

                matched_gt = np.zeros(num_gt_boxes, dtype=bool)

                for i in range(len(scores)):
                    score = float(scores[i])
                    if num_gt_boxes > 0:
                        best_match = np.argmax(ious[i])
                        max_iou = ious[i, best_match]

                        if max_iou >= t_iou:
                            if not matched_gt[best_match]:
                                if g_crowds[best_match] == 0:
                                    matched_gt[best_match] = True
                                    preds_by_class[c].append({"score": score, "is_tp": True})
                                else:
                                    continue
                            else:
                                preds_by_class[c].append({"score": score, "is_tp": False})
                        else:
                            preds_by_class[c].append({"score": score, "is_tp": False})
                    else:
                        preds_by_class[c].append({"score": score, "is_tp": False})

        ap_per_class = {}
        full_details = {}
        for c in range(num_classes):
            num_gt = num_gt_by_class[c]
            preds = preds_by_class[c]
            if num_gt == 0 or len(preds) == 0:
                ap_per_class[c] = 0.0
                full_details[c] = {
                    "ap": 0.0, "roc_auc": 0.0, "precision": [], "recall": [],
                    "fpr": [], "tpr": [], "scores": [], "num_gt": num_gt,
                    "num_pred": len(preds), "best_f1": 0.0, "best_threshold": 0.0
                }
                continue

            preds.sort(key=lambda x: x["score"], reverse=True)
            scores = np.array([p["score"] for p in preds])
            tp_flags = np.array([1 if p["is_tp"] else 0 for p in preds])
            fp_flags = 1 - tp_flags

            cum_tp = np.cumsum(tp_flags)
            cum_fp = np.cumsum(fp_flags)

            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)
            recall = cum_tp / num_gt
            tpr = recall

            total_fp = np.maximum(cum_fp[-1], 1)
            fpr = cum_fp / total_fp

            ap = compute_coco_101_point_ap(recall, precision)
            ap_per_class[c] = ap

            fpr_roc = np.concatenate(([0.0], fpr))
            tpr_roc = np.concatenate(([0.0], tpr))
            sort_order = np.argsort(fpr_roc)
            roc_auc = float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(tpr_roc[sort_order], fpr_roc[sort_order]))

            f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-7)
            best_idx = np.argmax(f1_scores)

            full_details[c] = {
                "ap": float(ap),
                "roc_auc": float(roc_auc),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "fpr": fpr_roc.tolist(),
                "tpr": tpr_roc.tolist(),
                "scores": scores.tolist(),
                "num_gt": int(num_gt),
                "num_pred": int(len(preds)),
                "best_f1": float(f1_scores[best_idx]),
                "best_threshold": float(scores[best_idx])
            }

        return ap_per_class, full_details

    # Evaluate at primary IoU threshold (default 0.5)
    main_aps, main_details = eval_at_thresh(iou_thresh)

    # Evaluate across IoU thresholds [0.50, 0.55, ..., 0.95] for mAP@0.5:0.95
    all_threshold_aps = {c: [] for c in range(num_classes)}
    for t_iou in iou_thresholds:
        if abs(t_iou - iou_thresh) < 1e-5:
            thresh_aps = main_aps
        else:
            thresh_aps, _ = eval_at_thresh(t_iou)
        for c in range(num_classes):
            all_threshold_aps[c].append(thresh_aps[c])

    class_results = {}
    all_aps_50 = []
    all_aps_50_95 = []
    all_aucs = []

    for c in range(num_classes):
        c_name = class_names[c]
        d = main_details[c]
        ap_50 = d["ap"]
        ap_50_95 = float(np.mean(all_threshold_aps[c]))

        all_aps_50.append(ap_50)
        all_aps_50_95.append(ap_50_95)
        all_aucs.append(d["roc_auc"])

        class_results[c_name] = {
            "ap": ap_50,
            "ap_50": ap_50,
            "ap_50_95": ap_50_95,
            "roc_auc": d["roc_auc"],
            "precision": d["precision"],
            "recall": d["recall"],
            "fpr": d["fpr"],
            "tpr": d["tpr"],
            "scores": d["scores"],
            "num_gt": d["num_gt"],
            "num_pred": d["num_pred"],
            "best_f1": d["best_f1"],
            "best_threshold": d["best_threshold"],
        }

    mAP_50 = float(np.mean(all_aps_50)) if all_aps_50 else 0.0
    mAP_50_95 = float(np.mean(all_aps_50_95)) if all_aps_50_95 else 0.0
    macro_auc = float(np.mean(all_aucs)) if all_aucs else 0.0

    return {
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "macro_ROC_AUC": macro_auc,
        "iou_thresh": iou_thresh,
        "classes": class_results
    }


# ============================================================================
# Artifact Visualization and Storage
# ============================================================================
def plot_and_save_artifacts(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Plot Precision-Recall Curves
    plt.figure(figsize=(10, 8), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.5)

    for c_name, data in results["classes"].items():
        if len(data["recall"]) > 0:
            rec = np.array(data["recall"])
            prec = np.array(data["precision"])
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            plt.plot(mrec, mpre, label=f"{c_name} (AP50={data['ap_50']*100:.1f}%, AP50-95={data['ap_50_95']*100:.1f}%)", linewidth=1.5)

    plt.xlabel("Recall", fontsize=13, fontweight="bold")
    plt.ylabel("Precision", fontsize=13, fontweight="bold")
    plt.title(f"Precision - Recall Curve (mAP@0.5: {results['mAP_50']*100:.2f}%, mAP@[0.5:0.95]: {results['mAP_50_95']*100:.2f}%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    pr_path = os.path.join(save_dir, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    logger.info(f"Saved Precision-Recall curve to: {pr_path}")

    # 2. Plot ROC Curves
    plt.figure(figsize=(10, 8), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (AUC=0.50)")

    for c_name, data in results["classes"].items():
        if len(data["fpr"]) > 0:
            plt.plot(data["fpr"], data["tpr"], label=f"{c_name} (AUC={data['roc_auc']*100:.1f}%)", linewidth=1.5)

    plt.xlabel("False Positive Rate (FPR)", fontsize=13, fontweight="bold")
    plt.ylabel("True Positive Rate / Recall (TPR)", fontsize=13, fontweight="bold")
    plt.title(f"ROC Curve (Macro AUC: {results['macro_ROC_AUC']*100:.2f}%)", fontsize=15, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"Saved ROC curve to: {roc_path}")

    # 3. Save JSON evaluation summary
    json_path = os.path.join(save_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation JSON to: {json_path}")

    # 4. Save CSV summary table
    csv_path = os.path.join(save_dir, "per_class_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Class,AP@0.5 (%),AP@0.5:0.95 (%),ROC-AUC (%),Best F1,Best Threshold,Num GT,Num Predictions\n")
        for c_name, data in results["classes"].items():
            f.write(f"{c_name},{data['ap_50']*100:.2f},{data['ap_50_95']*100:.2f},{data['roc_auc']*100:.2f},"
                    f"{data['best_f1']:.4f},{data['best_threshold']:.4f},"
                    f"{data['num_gt']},{data['num_pred']}\n")
        f.write(f"mAP@0.5,{results['mAP_50']*100:.2f},-,{results['macro_ROC_AUC']*100:.2f},-, -, -, -\n")
        f.write(f"mAP@0.5:0.95,-,{results['mAP_50_95']*100:.2f},-,-, -, -, -\n")
    logger.info(f"Saved per-class metrics CSV to: {csv_path}")


# ============================================================================
# Argument Parser & Entry Point
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mAP, PR curves, and ROC curves from saved JSON prediction files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json-dir", "--json_dir", type=str, default=None,
        help="Directory containing prediction and ground truth JSON files"
    )
    parser.add_argument(
        "--pred-json", "--pred_json", type=str, default=None,
        help="Path to predictions JSON file (overrides auto-detection from --json-dir)"
    )
    parser.add_argument(
        "--gt-json", "--gt_json", type=str, default=None,
        help="Path to ground truths JSON file (overrides auto-detection from --json-dir)"
    )
    parser.add_argument(
        "--save-dir", "--save_dir", type=str, default=None,
        help="Path to save evaluation plots and metrics (defaults to json-dir)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["voc", "coco", "bdd100k"],
        help="Dataset type to determine class names (auto-detected if not specified)"
    )
    parser.add_argument(
        "--iou-thresh", "--iou-threshold", "--iou_thresh", type=float, default=0.5,
        help="IoU threshold for box matching (default: 0.5)"
    )
    parser.add_argument(
        "--eval-weather", "--eval_weather", action="store_true", default=True,
        help="Evaluate mAP per weather subset for BDD100K (default: True)"
    )
    parser.add_argument(
        "--no-eval-weather", action="store_false", dest="eval_weather",
        help="Disable evaluation per weather subset"
    )
    parser.add_argument(
        "--visualize-failed-cases", "--visualize_failed_cases", action="store_true", default=False,
        help="Visualize per-class failed cases (FP background, FP misclass, FN missed, etc.)"
    )
    return parser.parse_args()



def evaluate_weather_subsets(predictions, ground_truths, class_names, save_dir, iou_thresh=0.5,
                              raw_label_path="./data/bdd100k/val/annotations/bdd100k_labels_images_val.json",
                              coco_val_path="./data/bdd100k/val/annotations/coco_instances_val.json"):
    """
    Evaluates mAP@0.5 and mAP@0.5:0.95 across 7 weather conditions in BDD100K.
    """
    if not os.path.exists(raw_label_path):
        raw_label_path = os.path.join(os.path.dirname(__file__), "data/bdd100k/val/annotations/bdd100k_labels_images_val.json")
    if not os.path.exists(coco_val_path):
        coco_val_path = os.path.join(os.path.dirname(__file__), "data/bdd100k/val/annotations/coco_instances_val.json")

    if not os.path.exists(raw_label_path) or not os.path.exists(coco_val_path):
        logger.warning("Could not find BDD100K weather labels/annotations to run weather subset evaluation.")
        return

    logger.info(f"Loading weather attributes for subset evaluation from: {raw_label_path}")
    with open(raw_label_path, "r", encoding="utf-8") as f:
        raw_val = json.load(f)

    weather_map = {}
    for item in raw_val:
        name = item.get("name")
        w = item.get("attributes", {}).get("weather", "undefined").strip().lower().replace(" ", "_")
        if name:
            weather_map[name] = w
            weather_map[os.path.basename(name)] = w

    with open(coco_val_path, "r", encoding="utf-8") as f:
        coco_val = json.load(f)

    weather_to_indices = {}
    for idx, img in enumerate(coco_val.get("images", [])):
        fn = os.path.basename(img.get("file_name", img.get("name", "")))
        w = weather_map.get(fn, "undefined")
        if w not in weather_to_indices:
            weather_to_indices[w] = set()
        weather_to_indices[w].add(idx)

    logger.info(f"Evaluating predictions across {len(weather_to_indices)} weather subsets...")
    weather_summary = []
    weather_dir = os.path.join(save_dir, "weather")

    for weather, idx_set in sorted(weather_to_indices.items()):
        w_preds = {idx: predictions[idx] for idx in idx_set if idx in predictions}
        w_gts = {idx: ground_truths[idx] for idx in idx_set if idx in ground_truths}

        if len(w_gts) == 0:
            continue

        res = evaluate_detection_curves(
            predictions=w_preds,
            ground_truths=w_gts,
            class_names=class_names,
            iou_thresh=iou_thresh
        )

        w_save_dir = os.path.join(weather_dir, weather)
        plot_and_save_artifacts(res, w_save_dir)

        weather_summary.append({
            "weather": weather,
            "images": len(w_preds),
            "mAP_50": res["mAP_50"],
            "mAP_50_95": res["mAP_50_95"],
            "macro_auc": res["macro_ROC_AUC"],
            "save_dir": w_save_dir
        })

    print("\n" + "=" * 82)
    print(f"{'Weather Subset':<18} | {'Images':<8} | {'mAP@0.5 (%)':<12} | {'mAP@0.5:0.95 (%)':<16} | {'ROC-AUC (%)':<12}")
    print("-" * 82)
    for w in weather_summary:
        print(f"{w['weather']:<18} | {w['images']:<8} | {w['mAP_50']*100:<12.2f} | {w['mAP_50_95']*100:<16.2f} | {w['macro_auc']*100:<12.2f}")
    print("=" * 82)

    summary_csv_path = os.path.join(weather_dir, "weather_subsets_mAP_summary.csv")
    os.makedirs(weather_dir, exist_ok=True)
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        f.write("Weather,Images,mAP@0.5 (%),mAP@0.5:0.95 (%),ROC-AUC (%)\n")
        for w in weather_summary:
            f.write(f"{w['weather']},{w['images']},{w['mAP_50']*100:.2f},{w['mAP_50_95']*100:.2f},{w['macro_auc']*100:.2f}\n")

    logger.info(f"Saved weather subset evaluation summary to: {summary_csv_path}")


def main():
    args = parse_args()

    json_dir = Path(args.json_dir) if args.json_dir else None
    pred_json = args.pred_json
    gt_json = args.gt_json

    if json_dir is not None:
        if not json_dir.exists():
            logger.error(f"JSON directory not found: {json_dir}")
            sys.exit(1)

        if pred_json is None:
            pred_candidates = list(json_dir.glob("predictions_*.json")) + \
                              list(json_dir.glob("predictions.json")) + \
                              list(json_dir.glob("coco_results.json")) + \
                              list(json_dir.glob("bbox.json"))
            if pred_candidates:
                pred_json = str(pred_candidates[0])
                logger.info(f"Auto-detected predictions: {pred_json}")
            else:
                logger.error(f"No predictions JSON file found in {json_dir}")
                sys.exit(1)

        if gt_json is None:
            gt_candidates = list(json_dir.glob("ground_truths_*.json")) + \
                            list(json_dir.glob("ground_truth.json")) + \
                            list(json_dir.glob("targets.json")) + \
                            list(json_dir.glob("annotations.json"))
            if gt_candidates:
                gt_json = str(gt_candidates[0])
                logger.info(f"Auto-detected ground truths: {gt_json}")
            else:
                default_gts = [
                    "./data/bdd100k/labels/bdd100k_labels_images_val_coco.json",
                    "../data/bdd100k/labels/bdd100k_labels_images_val_coco.json",
                ]
                for dg in default_gts:
                    if os.path.isfile(dg):
                        gt_json = dg
                        logger.info(f"Auto-detected dataset ground truth: {gt_json}")
                        break

    if not pred_json or not os.path.isfile(pred_json):
        logger.error(f"Predictions file not found: {pred_json}")
        sys.exit(1)

    if not gt_json or not os.path.isfile(gt_json):
        logger.error(f"Ground truth file not found: {gt_json}. Please pass --gt-json path/to/ground_truth.json")
        sys.exit(1)

    save_dir = args.save_dir if args.save_dir else (str(json_dir) if json_dir else str(Path(pred_json).parent))

    dataset = args.dataset
    if dataset is None:
        combined_name = (os.path.basename(pred_json) + " " + os.path.basename(gt_json)).lower()
        if "bdd100k" in combined_name or "bdd" in combined_name:
            dataset = "bdd100k"
        elif "coco" in combined_name:
            dataset = "coco"
        elif "voc" in combined_name:
            dataset = "voc"
        else:
            dataset = "bdd100k"
        logger.info(f"Auto-inferred dataset type: {dataset}")

    if dataset == "bdd100k":
        class_names = BDD100K_CLASSES
    elif dataset == "coco":
        class_names = COCO_CLASSES
    else:
        class_names = VOC_CLASSES

    logger.info(f"Loading predictions from: {pred_json}")
    predictions = load_predictions_from_json(pred_json, num_classes=len(class_names))
    logger.info(f"  Loaded predictions for {len(predictions)} images")

    logger.info(f"Loading ground truths from: {gt_json}")
    ground_truths, _ = load_ground_truths_from_json(gt_json, num_classes=len(class_names))
    logger.info(f"  Loaded ground truths for {len(ground_truths)} images")

    logger.info(f"Evaluating predictions (mAP@0.5 and mAP@0.5:0.95) across {len(class_names)} classes...")
    results = evaluate_detection_curves(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
        iou_thresh=args.iou_thresh
    )

    print("\n" + "=" * 82)
    print(f"{'Class':<16} | {'AP@0.5 (%)':<12} | {'AP@0.5:0.95 (%)':<16} | {'ROC-AUC (%)':<12} | {'Best F1':<8}")
    print("-" * 82)
    for c_name, data in results["classes"].items():
        print(f"{c_name:<16} | {data['ap_50']*100:<12.2f} | {data['ap_50_95']*100:<16.2f} | {data['roc_auc']*100:<12.2f} | {data['best_f1']:<8.3f}")
    print("=" * 82)
    print(f"{'mAP@0.5':<16} | {results['mAP_50']*100:<12.2f} | {'-':<16} | {results['macro_ROC_AUC']*100:<12.2f} | -")
    print(f"{'mAP@0.5:0.95':<16} | {'-':<12} | {results['mAP_50_95']*100:<16.2f} | {'-':<12} | -")
    print("=" * 82)

    plot_and_save_artifacts(results, save_dir)
    logger.info(f"Evaluation complete. Output artifacts saved to: {os.path.abspath(save_dir)}\n")

    if dataset == "bdd100k" and getattr(args, "eval_weather", True):
        evaluate_weather_subsets(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            save_dir=save_dir,
            iou_thresh=args.iou_thresh
        )

    if getattr(args, "visualize_failed_cases", False):
        logger.info("Running per-class failed case visualization...")
        from visualize_class_failed_cases import main as viz_main
        failed_save_dir = os.path.join(save_dir, "failed_cases_per_class")
        sys.argv = [
            "visualize_class_failed_cases.py",
            "--pred-json", pred_json,
            "--gt-json", gt_json,
            "--dataset", dataset,
            "--save-dir", failed_save_dir,
            "--score-thresh", "0.3",
            "--iou-thresh", str(args.iou_thresh)
        ]
        viz_main()


if __name__ == "__main__":
    main()
