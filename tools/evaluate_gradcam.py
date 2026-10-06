#!/usr/bin/env python3
"""
Quantitative evaluation of Grad-CAM against BDD100K bounding boxes.

Computes standard explainability & localization metrics across the dataset:
  1. Pointing Game Accuracy (%): Percentage of images where the peak CAM point lands inside a GT box.
  2. Energy Ratio (%): Ratio of CAM attention mass inside GT boxes vs total image energy.
  3. CAM-IoU (@0.3, @0.5): Intersection over Union between thresholded CAM heatmap and GT box regions.
  4. Attention Precision & Recall: Precision and recall of attention coverage over GT boxes.
  5. Per-class Breakdown: Pointing accuracy and Energy Ratio per object category.

Usage:
  python -m tools.evaluate_gradcam --data-dir ./data/bdd100k --checkpoint /path/to/last.pth
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Project imports
from detection.gradcam import GradCAM
from detection.fastvit_detector import FastViTDetector
from bdd100k_dataset import BDD100KDetectionDataset, bdd100k_collate, BDD100K_CLASSES


def compute_gt_binary_mask(boxes, height, width):
    """Create a binary mask (H, W) where 1 indicates inside at least one GT box."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(boxes) == 0:
        return mask

    for box in boxes:
        x1 = int(np.clip(box[0], 0, width - 1))
        y1 = int(np.clip(box[1], 0, height - 1))
        x2 = int(np.clip(box[2], 0, width - 1))
        y2 = int(np.clip(box[3], 0, height - 1))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    return mask


def eval_single_cam(cam_map, gt_boxes, gt_labels, class_names):
    """Evaluate quantitative metrics for a single image CAM map.

    Args:
        cam_map: (H, W) float numpy array in [0, 1]
        gt_boxes: (N, 4) float numpy array [x1, y1, x2, y2]
        gt_labels: (N,) int numpy array of 1-indexed class IDs
        class_names: list of class name strings

    Returns:
        dict of metrics for this single image
    """
    H, W = cam_map.shape
    gt_mask = compute_gt_binary_mask(gt_boxes, H, W)

    total_cam_energy = cam_map.sum()
    if total_cam_energy < 1e-7 or len(gt_boxes) == 0:
        return None  # Skip empty GT or blank CAM

    # 1. Pointing Game (Peak Hit Rate)
    max_pos = np.unravel_index(np.argmax(cam_map), cam_map.shape)
    peak_y, peak_x = max_pos
    pointing_hit = int(gt_mask[peak_y, peak_x] > 0)

    # 2. Energy Ratio (Inside GT Box Energy / Total Energy)
    energy_inside = (cam_map * gt_mask).sum()
    energy_ratio = energy_inside / (total_cam_energy + 1e-7)

    # 3. CAM-IoU at Thresholds 0.3 and 0.5
    cam_bin_03 = (cam_map >= 0.3).astype(np.uint8)
    cam_bin_05 = (cam_map >= 0.5).astype(np.uint8)

    intersection_03 = np.logical_and(cam_bin_03, gt_mask).sum()
    union_03 = np.logical_or(cam_bin_03, gt_mask).sum()
    iou_03 = intersection_03 / (union_03 + 1e-7)

    intersection_05 = np.logical_and(cam_bin_05, gt_mask).sum()
    union_05 = np.logical_or(cam_bin_05, gt_mask).sum()
    iou_05 = intersection_05 / (union_05 + 1e-7)

    # 4. Attention Precision & Recall (@0.3)
    precision_03 = intersection_03 / (cam_bin_03.sum() + 1e-7)
    recall_03 = intersection_03 / (gt_mask.sum() + 1e-7)

    # 5. Per-class tracking: compute class-level GT masks per image
    class_stats = {}
    unique_labels = np.unique(gt_labels)

    for label in unique_labels:
        cls_idx = int(label) - 1
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"cls_{label}"

        # Create binary mask for all GT boxes belonging to this specific class
        cls_mask = np.zeros((H, W), dtype=np.uint8)
        for box, l in zip(gt_boxes, gt_labels):
            if l == label:
                x1 = int(np.clip(box[0], 0, W - 1))
                y1 = int(np.clip(box[1], 0, H - 1))
                x2 = int(np.clip(box[2], 0, W - 1))
                y2 = int(np.clip(box[3], 0, H - 1))
                if x2 > x1 and y2 > y1:
                    cls_mask[y1:y2, x1:x2] = 1

        # Class Pointing Hit: Does peak land inside ANY GT box of this class?
        cls_hit = int(cls_mask[peak_y, peak_x] > 0)

        # Class Energy Ratio: Energy inside ALL GT boxes of this class / Total CAM energy
        cls_energy = float((cam_map * cls_mask).sum() / (total_cam_energy + 1e-7))

        class_stats[cls_name] = {
            "hit": cls_hit,
            "energy": cls_energy,
        }

    return {
        "pointing_hit": pointing_hit,
        "energy_ratio": float(energy_ratio),
        "iou_03": float(iou_03),
        "iou_05": float(iou_05),
        "precision_03": float(precision_03),
        "recall_03": float(recall_03),
        "class_stats": class_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantify Grad-CAM alignment with Ground-Truth bounding boxes across dataset"
    )
    parser.add_argument("--data-dir", type=str, default="./data/bdd100k")
    parser.add_argument("--val-img", type=str, default=None)
    parser.add_argument("--val-ann", type=str, default=None)
    parser.add_argument("--model", type=str, default="fastvit_sa12")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max validation samples to evaluate (default: full set)"
    )
    parser.add_argument(
        "--output", type=str, default="./output/gradcam_eval_metrics.json", help="Path to save output JSON report"
    )
    args = parser.parse_args()

    print("============================================================")
    print("  Grad-CAM Quantitative Evaluation")
    print("============================================================")
    print("  Dataset:      bdd100k")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Model:        {args.model}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Max samples:  {args.max_samples or 'Full Dataset'}")
    print("============================================================")

    val_img = args.val_img or os.path.join(args.data_dir, "images", "100k", "val")
    val_ann = args.val_ann or os.path.join(
        args.data_dir, "annotations", "bdd100k_det_val_coco.json"
    )
    val_dataset = BDD100KDetectionDataset(
        val_img, val_ann, img_size=800, augment=False
    )
    num_classes = len(BDD100K_CLASSES)
    class_names = BDD100K_CLASSES

    if args.max_samples and args.max_samples < len(val_dataset):
        indices = list(range(args.max_samples))
        val_dataset = Subset(val_dataset, indices)

    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=bdd100k_collate)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastViTDetector(model_name=args.model, num_classes=num_classes)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint ({len(filtered)}/{len(model_dict)} keys matched)")
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        sys.exit(1)

    model.to(device)
    model.eval()

    gradcam_generator = GradCAM(model)

    pointing_hits = []
    energy_ratios = []
    ious_03 = []
    ious_05 = []
    precisions_03 = []
    recalls_03 = []
    per_class_hits = defaultdict(list)
    per_class_energies = defaultdict(list)

    print("\nRunning Grad-CAM evaluation...")
    iterator = tqdm(dataloader, desc="Evaluating Grad-CAM") if HAS_TQDM else dataloader

    for i, (images, targets) in enumerate(iterator):
        images = images.to(device)
        gt_boxes = targets[0]["boxes"].cpu().numpy()
        gt_labels = targets[0]["labels"].cpu().numpy()

        cam_map = gradcam_generator.generate_cam_map(images)

        res = eval_single_cam(cam_map, gt_boxes, gt_labels, class_names)
        if res is None:
            continue

        pointing_hits.append(res["pointing_hit"])
        energy_ratios.append(res["energy_ratio"])
        ious_03.append(res["iou_03"])
        ious_05.append(res["iou_05"])
        precisions_03.append(res["precision_03"])
        recalls_03.append(res["recall_03"])

        for cls_name, stats in res["class_stats"].items():
            per_class_hits[cls_name].append(stats["hit"])
            per_class_energies[cls_name].append(stats["energy"])

    gradcam_generator.remove_hooks()

    num_eval_images = len(pointing_hits)
    mean_pointing = float(np.mean(pointing_hits)) * 100.0 if pointing_hits else 0.0
    mean_energy = float(np.mean(energy_ratios)) * 100.0 if energy_ratios else 0.0
    mean_iou_03 = float(np.mean(ious_03)) * 100.0 if ious_03 else 0.0
    mean_iou_05 = float(np.mean(ious_05)) * 100.0 if ious_05 else 0.0
    mean_prec_03 = float(np.mean(precisions_03)) * 100.0 if precisions_03 else 0.0
    mean_rec_03 = float(np.mean(recalls_03)) * 100.0 if recalls_03 else 0.0

    print("\n" + "=" * 62)
    print("  Grad-CAM Quantitative Metrics Summary")
    print("=" * 62)
    print(f"  Evaluated Images (with GT): {num_eval_images}")
    print(f"  Pointing Game Accuracy:     {mean_pointing:.2f}%  (Hits in GT box)")
    print(f"  Energy in GT Boxes:         {mean_energy:.2f}%  (CAM mass in GT box)")
    print(f"  Mean CAM-IoU @ 0.3:         {mean_iou_03:.2f}%")
    print(f"  Mean CAM-IoU @ 0.5:         {mean_iou_05:.2f}%")
    print(f"  Attention Precision @ 0.3:  {mean_prec_03:.2f}%")
    print(f"  Attention Recall @ 0.3:     {mean_rec_03:.2f}%")
    print("=" * 62)

    # Per-class table
    print("\n" + "-" * 62)
    print(f"  {'Class Name':<20} | {'Pointing Acc (%)':<18} | {'Energy Ratio (%)':<18}")
    print("-" * 62)

    class_report = {}
    for cls_name in class_names:
        hits = per_class_hits.get(cls_name, [])
        energies = per_class_energies.get(cls_name, [])
        p_acc = float(np.mean(hits)) * 100.0 if hits else 0.0
        e_rat = float(np.mean(energies)) * 100.0 if energies else 0.0
        class_report[cls_name] = {
            "pointing_accuracy": p_acc,
            "energy_ratio": e_rat,
            "sample_count": len(hits),
        }
        print(f"  {cls_name:<20} | {p_acc:<18.2f} | {e_rat:<18.2f}")
    print("-" * 62)

    # Save to JSON
    report = {
        "num_eval_images": num_eval_images,
        "pointing_game_accuracy": mean_pointing,
        "energy_in_boxes_ratio": mean_energy,
        "mean_cam_iou_03": mean_iou_03,
        "mean_cam_iou_05": mean_iou_05,
        "attention_precision_03": mean_prec_03,
        "attention_recall_03": mean_rec_03,
        "per_class_report": class_report,
        "args": vars(args),
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to {args.output}")


if __name__ == "__main__":
    main()
