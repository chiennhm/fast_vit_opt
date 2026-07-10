#
# Exploratory Data Analysis (EDA) for BDD100K Detection Dataset
#
# This script performs detailed analysis of BDD100K annotations (COCO-style JSON).
# It generates comprehensive statistics and visualizations covering:
#   1. Dataset overview (images, annotations, categories)
#   2. Class distribution and imbalance analysis
#   3. Bounding box size analysis (area, width, height)
#   4. Aspect ratio distribution
#   5. Object count per image
#   6. Spatial distribution heatmaps (object center locations)
#   7. Box size vs class analysis
#   8. Co-occurrence matrix between classes
#   9. Small/Medium/Large object breakdown (COCO-style)
#   10. Sample image visualizations with bounding boxes
#
# Usage:
#   python eda_bdd100k.py [--data-dir ./data/bdd100k] [--output-dir ./eda_output]
#

import os
import sys
import json
import argparse
import math
from collections import defaultdict, Counter

import numpy as np

# --- Lazy import for matplotlib (handle headless environments) ---
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    return plt, patches, gridspec, LinearSegmentedColormap


# ============================================================================
# Constants
# ============================================================================
BDD100K_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]

# COCO-style area thresholds (in pixels^2)
AREA_SMALL = 32 ** 2     # < 1024
AREA_MEDIUM = 96 ** 2    # < 9216
# AREA_LARGE = everything above AREA_MEDIUM

# Color palette for classes (tab10-inspired)
CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ============================================================================
# Data Loading
# ============================================================================
def find_annotation_files(data_dir):
    """Auto-discover COCO-style annotation files for BDD100K."""
    ann_dir = os.path.join(data_dir, "annotations")
    train_ann = os.path.join(ann_dir, "bdd100k_det_train_coco.json")
    val_ann = os.path.join(ann_dir, "bdd100k_det_val_coco.json")

    found = {}
    if os.path.exists(train_ann):
        found["train"] = train_ann
    if os.path.exists(val_ann):
        found["val"] = val_ann

    # Fallback: search for any bdd100k COCO JSON
    if not found:
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".json") and "bdd100k" in f.lower() and "coco" in f.lower():
                    split = "train" if "train" in f.lower() else "val"
                    found[split] = os.path.join(root, f)

    return found


def load_coco_annotations(json_path):
    """Load a COCO-style JSON annotation file and return structured data."""
    print(f"  Loading: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    annotations = data["annotations"]

    print(f"    → {len(images):,} images, {len(annotations):,} annotations, {len(categories)} categories")
    return images, categories, annotations


# ============================================================================
# Statistics Computation
# ============================================================================
def compute_statistics(images, categories, annotations):
    """Compute comprehensive statistics from COCO-format annotations."""
    stats = {}

    # --- Basic counts ---
    stats["num_images"] = len(images)
    stats["num_annotations"] = len(annotations)
    stats["num_categories"] = len(categories)

    # --- Image sizes ---
    img_widths = [img["width"] for img in images.values()]
    img_heights = [img["height"] for img in images.values()]
    stats["img_widths"] = img_widths
    stats["img_heights"] = img_heights

    # --- Per-class annotation counts ---
    class_counts = Counter()
    for ann in annotations:
        cat_name = categories.get(ann["category_id"], f"unknown_{ann['category_id']}")
        class_counts[cat_name] += 1
    stats["class_counts"] = class_counts

    # --- Bounding box analysis ---
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    bbox_aspect_ratios = []
    bbox_centers_x = []  # Normalized [0, 1]
    bbox_centers_y = []  # Normalized [0, 1]
    bbox_class = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        img_info = images.get(ann["image_id"])
        if img_info is None:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        bbox_widths.append(w)
        bbox_heights.append(h)
        bbox_areas.append(w * h)
        if h > 0:
            bbox_aspect_ratios.append(w / h)
        bbox_centers_x.append((x + w / 2) / img_w)
        bbox_centers_y.append((y + h / 2) / img_h)
        bbox_class.append(categories.get(ann["category_id"], "unknown"))

    stats["bbox_widths"] = np.array(bbox_widths)
    stats["bbox_heights"] = np.array(bbox_heights)
    stats["bbox_areas"] = np.array(bbox_areas)
    stats["bbox_aspect_ratios"] = np.array(bbox_aspect_ratios)
    stats["bbox_centers_x"] = np.array(bbox_centers_x)
    stats["bbox_centers_y"] = np.array(bbox_centers_y)
    stats["bbox_class"] = bbox_class

    # --- Objects per image ---
    objs_per_image = Counter()
    for ann in annotations:
        objs_per_image[ann["image_id"]] += 1

    # Include images with zero annotations
    for img_id in images:
        if img_id not in objs_per_image:
            objs_per_image[img_id] = 0

    stats["objs_per_image"] = list(objs_per_image.values())

    # --- Per-class objects per image ---
    class_per_image = defaultdict(lambda: Counter())
    for ann in annotations:
        cat_name = categories.get(ann["category_id"], "unknown")
        class_per_image[cat_name][ann["image_id"]] += 1
    stats["class_per_image"] = dict(class_per_image)

    # --- Small / Medium / Large breakdown ---
    areas = stats["bbox_areas"]
    stats["num_small"] = int(np.sum(areas < AREA_SMALL))
    stats["num_medium"] = int(np.sum((areas >= AREA_SMALL) & (areas < AREA_MEDIUM)))
    stats["num_large"] = int(np.sum(areas >= AREA_MEDIUM))

    # --- Per-class area stats ---
    class_areas = defaultdict(list)
    for area, cls in zip(bbox_areas, bbox_class):
        class_areas[cls].append(area)
    stats["class_areas"] = {k: np.array(v) for k, v in class_areas.items()}

    # --- Per-class bbox widths and heights ---
    class_bbox_w = defaultdict(list)
    class_bbox_h = defaultdict(list)
    for w, h, cls in zip(bbox_widths, bbox_heights, bbox_class):
        class_bbox_w[cls].append(w)
        class_bbox_h[cls].append(h)
    stats["class_bbox_w"] = {k: np.array(v) for k, v in class_bbox_w.items()}
    stats["class_bbox_h"] = {k: np.array(v) for k, v in class_bbox_h.items()}

    # --- Co-occurrence matrix ---
    img_to_classes = defaultdict(set)
    for ann in annotations:
        cat_name = categories.get(ann["category_id"], "unknown")
        img_to_classes[ann["image_id"]].add(cat_name)

    class_names = sorted(categories.values())
    n_cls = len(class_names)
    cls_idx = {name: i for i, name in enumerate(class_names)}
    co_matrix = np.zeros((n_cls, n_cls), dtype=np.int64)
    for img_id, cls_set in img_to_classes.items():
        for c1 in cls_set:
            for c2 in cls_set:
                co_matrix[cls_idx[c1], cls_idx[c2]] += 1
    stats["co_occurrence_matrix"] = co_matrix
    stats["co_occurrence_labels"] = class_names

    # --- Images with no annotations ---
    annotated_imgs = set(ann["image_id"] for ann in annotations)
    stats["num_empty_images"] = len(images) - len(annotated_imgs)

    # --- iscrowd stats ---
    num_crowd = sum(1 for ann in annotations if ann.get("iscrowd", 0) == 1)
    stats["num_crowd"] = num_crowd

    return stats


# ============================================================================
# Text Report
# ============================================================================
def print_text_report(split_name, stats):
    """Print a detailed text summary of dataset statistics."""
    print(f"\n{'='*70}")
    print(f"  BDD100K EDA Report — {split_name.upper()} split")
    print(f"{'='*70}")

    print(f"\n📊 Overview:")
    print(f"  Total images:       {stats['num_images']:>10,}")
    print(f"  Total annotations:  {stats['num_annotations']:>10,}")
    print(f"  Categories:         {stats['num_categories']:>10}")
    print(f"  Empty images (0 obj): {stats['num_empty_images']:>8,}")
    print(f"  Crowd annotations:  {stats['num_crowd']:>10,}")

    print(f"\n📐 Image Dimensions:")
    ws = stats["img_widths"]
    hs = stats["img_heights"]
    unique_sizes = set(zip(ws, hs))
    print(f"  Unique sizes:       {len(unique_sizes)}")
    for w, h in sorted(unique_sizes):
        count = sum(1 for ww, hh in zip(ws, hs) if ww == w and hh == h)
        print(f"    {w}×{h}: {count:,} images")

    print(f"\n📦 Class Distribution:")
    total = stats["num_annotations"]
    for cls_name in BDD100K_CLASSES:
        cnt = stats["class_counts"].get(cls_name, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {cls_name:<15s} {cnt:>8,} ({pct:5.1f}%) {bar}")

    # Imbalance ratio
    counts = [stats["class_counts"].get(c, 0) for c in BDD100K_CLASSES]
    if min(counts) > 0:
        imbalance = max(counts) / min(counts)
        print(f"\n  Imbalance ratio (max/min): {imbalance:.1f}x")
    print(f"  Most common:  {max(stats['class_counts'], key=stats['class_counts'].get)}")
    print(f"  Least common: {min(stats['class_counts'], key=stats['class_counts'].get)}")

    print(f"\n📏 Bounding Box Statistics:")
    areas = stats["bbox_areas"]
    widths = stats["bbox_widths"]
    heights = stats["bbox_heights"]
    ars = stats["bbox_aspect_ratios"]
    for name, arr in [("Width (px)", widths), ("Height (px)", heights), ("Area (px²)", areas), ("Aspect Ratio (w/h)", ars)]:
        print(f"  {name}:")
        print(f"    Mean: {np.mean(arr):>10.1f}  |  Median: {np.median(arr):>10.1f}")
        print(f"    Std:  {np.std(arr):>10.1f}  |  Min: {np.min(arr):>10.1f}  |  Max: {np.max(arr):>10.1f}")
        print(f"    Q25:  {np.percentile(arr, 25):>10.1f}  |  Q75: {np.percentile(arr, 75):>10.1f}")

    print(f"\n📊 Object Size Breakdown (COCO-style):")
    print(f"  Small  (area < {AREA_SMALL:,}):     {stats['num_small']:>8,} ({stats['num_small']/total*100:.1f}%)")
    print(f"  Medium ({AREA_SMALL:,} ≤ area < {AREA_MEDIUM:,}): {stats['num_medium']:>8,} ({stats['num_medium']/total*100:.1f}%)")
    print(f"  Large  (area ≥ {AREA_MEDIUM:,}):    {stats['num_large']:>8,} ({stats['num_large']/total*100:.1f}%)")

    print(f"\n🔢 Objects Per Image:")
    opi = np.array(stats["objs_per_image"])
    print(f"  Mean:   {np.mean(opi):.1f}")
    print(f"  Median: {np.median(opi):.0f}")
    print(f"  Std:    {np.std(opi):.1f}")
    print(f"  Min:    {np.min(opi)}")
    print(f"  Max:    {np.max(opi)}")
    print(f"  Images with 0 objects: {np.sum(opi == 0)}")

    print(f"\n📊 Per-Class Avg Objects/Image (when present):")
    for cls_name in BDD100K_CLASSES:
        if cls_name in stats["class_per_image"]:
            counts_per_img = list(stats["class_per_image"][cls_name].values())
            avg = np.mean(counts_per_img) if counts_per_img else 0
            num_imgs_with = len(counts_per_img)
            print(f"  {cls_name:<15s}  avg={avg:.1f}  in {num_imgs_with:,} images ({num_imgs_with/stats['num_images']*100:.1f}%)")

    print(f"\n📐 Per-Class Bbox Size (mean area):")
    for cls_name in BDD100K_CLASSES:
        if cls_name in stats["class_areas"]:
            a = stats["class_areas"][cls_name]
            print(f"  {cls_name:<15s}  mean_area={np.mean(a):>10.0f}  median={np.median(a):>10.0f}  std={np.std(a):>10.0f}")

    print(f"\n{'='*70}\n")


# ============================================================================
# Visualization
# ============================================================================
def plot_all(split_name, stats, output_dir):
    """Generate and save all EDA plots."""
    plt, patches, gridspec, LinearSegmentedColormap = setup_matplotlib()

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Style configuration
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560",
        "axes.labelcolor": "#eee",
        "text.color": "#eee",
        "xtick.color": "#ccc",
        "ytick.color": "#ccc",
        "grid.color": "#333",
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    # ---- 1. Class Distribution Bar Chart ----
    fig, ax = plt.subplots(figsize=(14, 7))
    class_names = BDD100K_CLASSES
    counts = [stats["class_counts"].get(c, 0) for c in class_names]
    bars = ax.barh(class_names, counts, color=CLASS_COLORS, edgecolor="#333", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", fontsize=10, color="#eee")
    ax.set_xlabel("Number of Annotations")
    ax.set_title(f"BDD100K Class Distribution — {split_name.upper()}", fontweight="bold", fontsize=16)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "01_class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 01_class_distribution.png")

    # ---- 2. Bbox Area Distribution (log scale) ----
    fig, ax = plt.subplots(figsize=(12, 6))
    areas = stats["bbox_areas"]
    log_areas = np.log10(areas[areas > 0])
    ax.hist(log_areas, bins=80, color="#e94560", alpha=0.85, edgecolor="#333", linewidth=0.3)
    # Vertical lines for COCO thresholds
    ax.axvline(np.log10(AREA_SMALL), color="#00d2ff", linestyle="--", linewidth=2, label=f"Small threshold ({AREA_SMALL:,} px²)")
    ax.axvline(np.log10(AREA_MEDIUM), color="#ff6b6b", linestyle="--", linewidth=2, label=f"Medium threshold ({AREA_MEDIUM:,} px²)")
    ax.set_xlabel("log₁₀(Bbox Area in px²)")
    ax.set_ylabel("Count")
    ax.set_title(f"Bbox Area Distribution (log scale) — {split_name.upper()}", fontweight="bold")
    ax.legend(facecolor="#16213e", edgecolor="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "02_bbox_area_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 02_bbox_area_distribution.png")

    # ---- 3. Bbox Width vs Height Scatter ----
    fig, ax = plt.subplots(figsize=(10, 10))
    widths = stats["bbox_widths"]
    heights = stats["bbox_heights"]
    # Subsample for performance
    n = len(widths)
    idx = np.random.choice(n, min(n, 50000), replace=False) if n > 50000 else np.arange(n)
    ax.scatter(widths[idx], heights[idx], s=1, alpha=0.15, c="#00d2ff")
    # Aspect ratio reference lines
    for ratio, label in [(0.5, "1:2"), (1.0, "1:1"), (2.0, "2:1")]:
        max_dim = max(widths.max(), heights.max())
        xx = np.linspace(0, max_dim, 100)
        ax.plot(xx, xx / ratio, "--", alpha=0.5, linewidth=1.5, label=f"AR={label}")
    ax.set_xlabel("Bbox Width (px)")
    ax.set_ylabel("Bbox Height (px)")
    ax.set_title(f"Bbox Width vs Height — {split_name.upper()}", fontweight="bold")
    ax.legend(facecolor="#16213e", edgecolor="#555", loc="upper left")
    ax.set_xlim(0, widths.max() * 1.05)
    ax.set_ylim(0, heights.max() * 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "03_bbox_width_vs_height.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 03_bbox_width_vs_height.png")

    # ---- 4. Aspect Ratio Distribution ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ars = stats["bbox_aspect_ratios"]
    ars_clipped = np.clip(ars, 0, 6)
    ax.hist(ars_clipped, bins=100, color="#7f5af0", alpha=0.85, edgecolor="#333", linewidth=0.3)
    for r, label in [(0.5, "0.5"), (1.0, "1.0"), (2.0, "2.0")]:
        ax.axvline(r, color="#ff6b6b", linestyle="--", linewidth=1.5, alpha=0.7, label=f"AR={label}")
    ax.set_xlabel("Aspect Ratio (width / height)")
    ax.set_ylabel("Count")
    ax.set_title(f"Aspect Ratio Distribution — {split_name.upper()}", fontweight="bold")
    ax.legend(facecolor="#16213e", edgecolor="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "04_aspect_ratio_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 04_aspect_ratio_distribution.png")

    # ---- 5. Objects Per Image Distribution ----
    fig, ax = plt.subplots(figsize=(12, 6))
    opi = np.array(stats["objs_per_image"])
    max_opi = min(int(np.percentile(opi, 99.5)), opi.max())
    bins = np.arange(0, max_opi + 2) - 0.5
    ax.hist(opi, bins=bins, color="#2ee59d", alpha=0.85, edgecolor="#333", linewidth=0.3)
    ax.axvline(np.mean(opi), color="#ff6b6b", linestyle="--", linewidth=2, label=f"Mean = {np.mean(opi):.1f}")
    ax.axvline(np.median(opi), color="#00d2ff", linestyle="--", linewidth=2, label=f"Median = {np.median(opi):.0f}")
    ax.set_xlabel("Number of Objects per Image")
    ax.set_ylabel("Number of Images")
    ax.set_title(f"Objects Per Image — {split_name.upper()}", fontweight="bold")
    ax.legend(facecolor="#16213e", edgecolor="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "05_objects_per_image.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 05_objects_per_image.png")

    # ---- 6. Spatial Heatmap (Object Centers) ----
    fig, ax = plt.subplots(figsize=(14, 8))
    cx = stats["bbox_centers_x"]
    cy = stats["bbox_centers_y"]
    heatmap, xedges, yedges = np.histogram2d(cx, cy, bins=[64, 36], range=[[0, 1], [0, 1]])
    # Custom colormap
    colors_cm = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ff6b6b", "#ffd93d"]
    cm = LinearSegmentedColormap.from_list("custom", colors_cm, N=256)
    im = ax.imshow(heatmap.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cm, interpolation="gaussian")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Annotation Count", color="#eee")
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#ccc")
    ax.set_xlabel("Normalized X (left → right)")
    ax.set_ylabel("Normalized Y (top → bottom)")
    ax.set_title(f"Object Center Spatial Heatmap — {split_name.upper()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "06_spatial_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 06_spatial_heatmap.png")

    # ---- 7. Per-Class Spatial Heatmaps (grid) ----
    n_classes = len(BDD100K_CLASSES)
    n_cols = 5
    n_rows = math.ceil(n_classes / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 9))
    axes = axes.flatten()
    for i, cls_name in enumerate(BDD100K_CLASSES):
        ax = axes[i]
        cls_mask = np.array([c == cls_name for c in stats["bbox_class"]])
        if cls_mask.any():
            hm, _, _ = np.histogram2d(cx[cls_mask], cy[cls_mask], bins=[32, 18], range=[[0, 1], [0, 1]])
            ax.imshow(hm.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cm, interpolation="gaussian")
        ax.set_title(cls_name, fontsize=10, fontweight="bold", color=CLASS_COLORS[i])
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Per-Class Object Center Heatmaps — {split_name.upper()}", fontweight="bold", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "07_per_class_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 07_per_class_heatmaps.png")

    # ---- 8. Per-Class Bbox Area Box Plot ----
    fig, ax = plt.subplots(figsize=(14, 7))
    data_for_box = []
    labels_for_box = []
    for cls_name in BDD100K_CLASSES:
        if cls_name in stats["class_areas"]:
            a = stats["class_areas"][cls_name]
            data_for_box.append(np.log10(a[a > 0]))
            labels_for_box.append(cls_name)
    bp = ax.boxplot(data_for_box, vert=True, patch_artist=True, labels=labels_for_box,
                    medianprops=dict(color="#ffd93d", linewidth=2),
                    whiskerprops=dict(color="#aaa"),
                    capprops=dict(color="#aaa"),
                    flierprops=dict(marker=".", markersize=2, markerfacecolor="#666", alpha=0.3))
    for patch, color in zip(bp["boxes"], CLASS_COLORS[:len(data_for_box)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("log₁₀(Area in px²)")
    ax.set_title(f"Per-Class Bbox Area Distribution — {split_name.upper()}", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "08_per_class_area_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 08_per_class_area_boxplot.png")

    # ---- 9. Small / Medium / Large Breakdown (stacked bar per class) ----
    fig, ax = plt.subplots(figsize=(14, 7))
    small_counts = []
    medium_counts = []
    large_counts = []
    for cls_name in BDD100K_CLASSES:
        if cls_name in stats["class_areas"]:
            a = stats["class_areas"][cls_name]
            small_counts.append(int(np.sum(a < AREA_SMALL)))
            medium_counts.append(int(np.sum((a >= AREA_SMALL) & (a < AREA_MEDIUM))))
            large_counts.append(int(np.sum(a >= AREA_MEDIUM)))
        else:
            small_counts.append(0)
            medium_counts.append(0)
            large_counts.append(0)

    x = np.arange(len(BDD100K_CLASSES))
    w = 0.6
    b1 = ax.bar(x, small_counts, w, label=f"Small (< {AREA_SMALL:,} px²)", color="#e94560", alpha=0.85)
    b2 = ax.bar(x, medium_counts, w, bottom=small_counts, label=f"Medium ({AREA_SMALL:,}–{AREA_MEDIUM:,} px²)", color="#0f3460", alpha=0.85)
    bottoms_2 = [s + m for s, m in zip(small_counts, medium_counts)]
    b3 = ax.bar(x, large_counts, w, bottom=bottoms_2, label=f"Large (≥ {AREA_MEDIUM:,} px²)", color="#2ee59d", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(BDD100K_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Number of Annotations")
    ax.set_title(f"Object Size Breakdown by Class — {split_name.upper()}", fontweight="bold")
    ax.legend(facecolor="#16213e", edgecolor="#555", loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "09_size_breakdown_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 09_size_breakdown_by_class.png")

    # ---- 10. Co-occurrence Matrix ----
    fig, ax = plt.subplots(figsize=(10, 9))
    co = stats["co_occurrence_matrix"].astype(float)
    # Normalize by diagonal (fraction of class A images also containing class B)
    diag = np.diag(co).copy()
    diag[diag == 0] = 1
    co_norm = co / diag[:, None]
    im = ax.imshow(co_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    labels = stats["co_occurrence_labels"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = co_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("P(col class | row class present)", color="#eee")
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#ccc")
    ax.set_title(f"Class Co-occurrence Matrix — {split_name.upper()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "10_co_occurrence_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 10_co_occurrence_matrix.png")

    # ---- 11. Per-Class Bbox Width & Height Violin Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax_idx, (metric_name, metric_data) in enumerate([("Width", stats["class_bbox_w"]), ("Height", stats["class_bbox_h"])]):
        ax = axes[ax_idx]
        data = []
        labels_v = []
        for cls_name in BDD100K_CLASSES:
            if cls_name in metric_data:
                data.append(metric_data[cls_name])
                labels_v.append(cls_name)
        vp = ax.violinplot(data, showmeans=True, showmedians=True)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(CLASS_COLORS[i % len(CLASS_COLORS)])
            body.set_alpha(0.7)
        vp["cmeans"].set_color("#ffd93d")
        vp["cmedians"].set_color("#00d2ff")
        ax.set_xticks(range(1, len(labels_v) + 1))
        ax.set_xticklabels(labels_v, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"Bbox {metric_name} (px)")
        ax.set_title(f"Per-Class Bbox {metric_name}", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Per-Class Bbox Dimensions — {split_name.upper()}", fontweight="bold", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(split_dir, "11_per_class_bbox_dimensions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 11_per_class_bbox_dimensions.png")

    # ---- 12. Summary Dashboard ----
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 12a. Class distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    counts_pie = [stats["class_counts"].get(c, 0) for c in BDD100K_CLASSES]
    wedges, texts, autotexts = ax1.pie(
        counts_pie, labels=BDD100K_CLASSES, colors=CLASS_COLORS,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        textprops={"fontsize": 8, "color": "#eee"},
        startangle=90, pctdistance=0.8
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax1.set_title("Class Distribution", fontweight="bold", fontsize=12)

    # 12b. Size breakdown pie
    ax2 = fig.add_subplot(gs[0, 1])
    size_labels = ["Small", "Medium", "Large"]
    size_vals = [stats["num_small"], stats["num_medium"], stats["num_large"]]
    size_colors = ["#e94560", "#0f3460", "#2ee59d"]
    ax2.pie(size_vals, labels=size_labels, colors=size_colors,
            autopct="%1.1f%%", textprops={"fontsize": 10, "color": "#eee"}, startangle=90)
    ax2.set_title("Object Size Breakdown", fontweight="bold", fontsize=12)

    # 12c. Objects per image histogram
    ax3 = fig.add_subplot(gs[0, 2])
    opi = np.array(stats["objs_per_image"])
    max_opi = min(int(np.percentile(opi, 99)), opi.max())
    ax3.hist(opi, bins=np.arange(0, max_opi + 2) - 0.5, color="#7f5af0", alpha=0.85, edgecolor="#333")
    ax3.set_xlabel("Objects per Image")
    ax3.set_ylabel("Count")
    ax3.set_title("Objects Per Image", fontweight="bold", fontsize=12)

    # 12d. Area distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(log_areas, bins=60, color="#e94560", alpha=0.85, edgecolor="#333", linewidth=0.3)
    ax4.set_xlabel("log₁₀(Area)")
    ax4.set_ylabel("Count")
    ax4.set_title("Bbox Area (log)", fontweight="bold", fontsize=12)

    # 12e. Aspect ratio
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(ars_clipped, bins=60, color="#00d2ff", alpha=0.85, edgecolor="#333", linewidth=0.3)
    ax5.set_xlabel("Aspect Ratio (w/h)")
    ax5.set_ylabel("Count")
    ax5.set_title("Aspect Ratio", fontweight="bold", fontsize=12)

    # 12f. Spatial heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    hm_sum, _, _ = np.histogram2d(cx, cy, bins=[32, 18], range=[[0, 1], [0, 1]])
    ax6.imshow(hm_sum.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cm, interpolation="gaussian")
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.set_title("Spatial Heatmap", fontweight="bold", fontsize=12)

    fig.suptitle(f"BDD100K EDA Summary Dashboard — {split_name.upper()}", fontweight="bold", fontsize=18, y=1.01)
    fig.savefig(os.path.join(split_dir, "00_summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 00_summary_dashboard.png")

    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)


# ============================================================================
# Sample Image Visualization (if images are available)
# ============================================================================
def visualize_samples(split_name, images, categories, annotations, img_dir, output_dir, num_samples=12):
    """Draw bounding boxes on sample images and save."""
    plt, patches, gridspec, _ = setup_matplotlib()

    if not os.path.isdir(img_dir):
        print(f"  ⚠️ Image directory not found: {img_dir} — skipping sample visualization")
        return

    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # Pick images with the most annotations for interesting samples
    img_ids_sorted = sorted(img_to_anns.keys(), key=lambda x: len(img_to_anns[x]), reverse=True)
    sample_ids = img_ids_sorted[:num_samples]

    n_cols = 4
    n_rows = math.ceil(num_samples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 5))

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "#eee",
    })

    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    cat_color_map = {cat_name: CLASS_COLORS[i % len(CLASS_COLORS)] for i, cat_name in enumerate(BDD100K_CLASSES)}

    for ax_idx, img_id in enumerate(sample_ids):
        ax = axes[ax_idx]
        img_info = images[img_id]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_dir, file_name)

        if not os.path.exists(img_path):
            ax.set_title(f"Not found: {file_name}", fontsize=8)
            ax.axis("off")
            continue

        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            ax.imshow(np.array(img))
        except Exception as e:
            ax.set_title(f"Error: {e}", fontsize=8)
            ax.axis("off")
            continue

        anns = img_to_anns.get(img_id, [])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_name = categories.get(ann["category_id"], "unknown")
            color = cat_color_map.get(cat_name, "#ffffff")
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x, y - 3, cat_name, fontsize=6, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#000", alpha=0.6, edgecolor="none"))

        ax.set_title(f"{file_name} ({len(anns)} objects)", fontsize=8, color="#eee")
        ax.axis("off")

    for j in range(ax_idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Sample Images with Annotations — {split_name.upper()}", fontweight="bold", fontsize=16, color="#eee")
    fig.tight_layout()
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    fig.savefig(os.path.join(split_dir, "12_sample_images.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  ✅ 12_sample_images.png")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BDD100K Detection Dataset — Exploratory Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eda_bdd100k.py --data-dir ./data/bdd100k
  python eda_bdd100k.py --data-dir ./data/bdd100k --output-dir ./eda_output
  python eda_bdd100k.py --train-ann ./annotations/train.json --val-ann ./annotations/val.json
        """,
    )
    parser.add_argument("--data-dir", type=str, default="./data/bdd100k",
                        help="Root directory of BDD100K data (default: ./data/bdd100k)")
    parser.add_argument("--train-ann", type=str, default=None,
                        help="Path to train annotation JSON (overrides auto-discovery)")
    parser.add_argument("--val-ann", type=str, default=None,
                        help="Path to val annotation JSON (overrides auto-discovery)")
    parser.add_argument("--train-img-dir", type=str, default=None,
                        help="Path to train images directory (for sample visualization)")
    parser.add_argument("--val-img-dir", type=str, default=None,
                        help="Path to val images directory (for sample visualization)")
    parser.add_argument("--output-dir", type=str, default="./eda_output",
                        help="Output directory for plots and reports (default: ./eda_output)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (text report only)")
    parser.add_argument("--no-samples", action="store_true",
                        help="Skip sample image visualization")

    args = parser.parse_args()

    print("=" * 70)
    print("  BDD100K Detection Dataset — Exploratory Data Analysis")
    print("=" * 70)

    # --- Discover annotation files ---
    ann_files = {}
    if args.train_ann:
        ann_files["train"] = args.train_ann
    if args.val_ann:
        ann_files["val"] = args.val_ann

    if not ann_files:
        ann_files = find_annotation_files(args.data_dir)

    if not ann_files:
        print(f"\n❌ No BDD100K annotation files found in: {os.path.abspath(args.data_dir)}")
        print("  Expected: <data_dir>/annotations/bdd100k_det_train_coco.json")
        print("  Expected: <data_dir>/annotations/bdd100k_det_val_coco.json")
        print("\n  Run `python download_bdd100k.py` first to download and convert annotations.")
        sys.exit(1)

    print(f"\n📁 Found annotation files:")
    for split, path in ann_files.items():
        print(f"  {split}: {path}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Image directories ---
    img_dirs = {}
    if args.train_img_dir:
        img_dirs["train"] = args.train_img_dir
    else:
        img_dirs["train"] = os.path.join(args.data_dir, "images", "100k", "train")
    if args.val_img_dir:
        img_dirs["val"] = args.val_img_dir
    else:
        img_dirs["val"] = os.path.join(args.data_dir, "images", "100k", "val")

    # --- Process each split ---
    for split_name, ann_path in sorted(ann_files.items()):
        print(f"\n{'─'*70}")
        print(f"  Processing: {split_name.upper()} split")
        print(f"{'─'*70}")

        images, categories, annotations = load_coco_annotations(ann_path)
        stats = compute_statistics(images, categories, annotations)
        print_text_report(split_name, stats)

        if not args.no_plots:
            print(f"\n🎨 Generating plots for {split_name}...")
            plot_all(split_name, stats, args.output_dir)

        if not args.no_samples and split_name in img_dirs:
            print(f"\n🖼️ Visualizing sample images for {split_name}...")
            visualize_samples(split_name, images, categories, annotations,
                              img_dirs[split_name], args.output_dir)

    # --- Save consolidated text report ---
    report_path = os.path.join(args.output_dir, "eda_report.txt")
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    for split_name, ann_path in sorted(ann_files.items()):
        images, categories, annotations = load_coco_annotations(ann_path)
        stats = compute_statistics(images, categories, annotations)
        print_text_report(split_name, stats)

    report_text = buffer.getvalue()
    sys.stdout = old_stdout

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📄 Text report saved: {report_path}")

    print(f"\n{'='*70}")
    print(f"  ✅ EDA Complete! All outputs saved to: {os.path.abspath(args.output_dir)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
