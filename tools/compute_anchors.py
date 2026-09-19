#
# Compute optimal anchor boxes from BDD100K COCO-format annotations.
#
# This script:
#   1. Scans training annotations to extract bounding box dimensions
#   2. Normalizes (width, height) relative to image size
#   3. Runs K-Means++ to find optimal anchor (width, height) clusters
#   4. Assigns anchors to 5 FPN levels (P2, P3, P4, P5, P6) based on area
#   5. Prints ready-to-use anchor config for FastViTDetector
#
# Usage:
#   python -m tools.compute_anchors --ann-file ./data/bdd100k/annotations/bdd100k_det_train_coco.json
#

import argparse
import os
import math
import numpy as np
 


def collect_coco_boxes(ann_file):
    """Collect bounding box (width, height) from COCO/BDD100K JSON annotations.

    Returns:
        wh_absolute: np.ndarray of shape (N, 2)
        wh_relative: np.ndarray of shape (N, 2)
    """
    import json
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Build image sizes lookup
    images = {img["id"]: img for img in coco_data["images"]}

    wh_absolute = []
    wh_relative = []

    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in images:
            continue
        
        img_w = float(images[img_id]["width"])
        img_h = float(images[img_id]["height"])
        
        if img_w <= 0 or img_h <= 0:
            continue

        bbox = ann["bbox"]  # [x, y, w, h]
        w = float(bbox[2])
        h = float(bbox[3])

        if w <= 0 or h <= 0:
            continue

        wh_absolute.append([w, h])
        wh_relative.append([w / img_w, h / img_h])

    print(f"\nTotal: {len(images)} images, {len(wh_absolute)} boxes collected.")
    return np.array(wh_absolute, dtype=np.float64), np.array(wh_relative, dtype=np.float64)


# ============================================================================
# K-Means++ clustering
# ============================================================================
def kmeans_pp_init(data, k, rng):
    """K-Means++ initialization.

    Selects initial centroids with probability proportional to squared distance
    from the nearest existing centroid — ensures well-spread initial centers.

    Args:
        data: (N, D) array
        k: number of clusters
        rng: numpy random generator

    Returns:
        centroids: (k, D) initial centroids
    """
    n = data.shape[0]

    # 1. Choose the first centroid uniformly at random
    idx = rng.integers(n)
    centroids = [data[idx]]

    for _ in range(1, k):
        # 2. Compute squared distance from each point to nearest centroid
        dists = np.min(
            np.stack([np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0),
            axis=0,
        )

        # 3. Sample next centroid with probability ∝ dist²
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids.append(data[idx])

    return np.array(centroids)


def iou_distance(wh, centroids):
    """Compute IoU-based distance between box dimensions and centroids.

    Uses the approach from YOLO: compute IoU assuming both boxes are centered at origin.

    Args:
        wh: (N, 2) box widths and heights
        centroids: (k, 2) centroid widths and heights

    Returns:
        distances: (N, k) where distance = 1 - IoU
    """
    N = wh.shape[0]
    k = centroids.shape[0]

    wh_exp = wh[:, np.newaxis, :]       # (N, 1, 2)
    c_exp = centroids[np.newaxis, :, :]  # (1, k, 2)

    inter = np.prod(np.minimum(wh_exp, c_exp), axis=2)  # (N, k)

    area_wh = wh[:, 0] * wh[:, 1]               # (N,)
    area_c = centroids[:, 0] * centroids[:, 1]   # (k,)

    union = area_wh[:, np.newaxis] + area_c[np.newaxis, :] - inter  # (N, k)

    iou = inter / (union + 1e-9)  # (N, k)

    return 1.0 - iou


def kmeans_iou(wh, k, max_iter=300, num_trials=10, seed=42):
    """K-Means++ with IoU distance metric.

    Runs multiple trials and returns the best (lowest total distance) result.

    Args:
        wh: (N, 2) box widths and heights
        k: number of clusters
        max_iter: maximum iterations per trial
        num_trials: number of independent trials
        seed: random seed

    Returns:
        best_centroids: (k, 2) optimal cluster centers
        best_assignments: (N,) cluster assignment for each box
        best_avg_iou: average IoU across all boxes
    """
    rng = np.random.default_rng(seed)
    best_centroids = None
    best_assignments = None
    best_avg_iou = 0.0

    for trial in range(num_trials):
        # K-Means++ init
        centroids = kmeans_pp_init(wh, k, rng)

        prev_assignments = None
        for iteration in range(max_iter):
            # Assign each box to nearest centroid (IoU distance)
            dists = iou_distance(wh, centroids)  # (N, k)
            assignments = np.argmin(dists, axis=1)  # (N,)

            # Check convergence
            if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
                break
            prev_assignments = assignments

            # Update centroids: median of assigned boxes (more robust than mean)
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    new_centroids[j] = np.median(wh[mask], axis=0)
                else:
                    # Re-initialize empty cluster
                    new_centroids[j] = wh[rng.integers(len(wh))]
            centroids = new_centroids

        # Compute average IoU
        dists = iou_distance(wh, centroids)
        min_dists = np.min(dists, axis=1)
        avg_iou = 1.0 - np.mean(min_dists)

        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_centroids = centroids.copy()
            best_assignments = assignments.copy()

        print(f"  Trial {trial + 1}/{num_trials}: avg IoU = {avg_iou:.4f} ({iteration + 1} iters)")

    return best_centroids, best_assignments, best_avg_iou


def kmeans_euclidean(wh, k, max_iter=300, num_trials=10, seed=42):
    """K-Means++ with Euclidean distance (standard).

    Args:
        wh: (N, 2) box widths and heights
        k: number of clusters
        max_iter: maximum iterations per trial
        num_trials: number of independent trials
        seed: random seed

    Returns:
        best_centroids: (k, 2)
        best_assignments: (N,)
        best_inertia: total sum of squared distances
    """
    rng = np.random.default_rng(seed)
    best_centroids = None
    best_assignments = None
    best_inertia = float("inf")

    for trial in range(num_trials):
        centroids = kmeans_pp_init(wh, k, rng)

        prev_assignments = None
        for iteration in range(max_iter):
            # Euclidean distance
            dists = np.sum((wh[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            assignments = np.argmin(dists, axis=1)

            if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
                break
            prev_assignments = assignments

            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    new_centroids[j] = np.mean(wh[mask], axis=0)
                else:
                    new_centroids[j] = wh[rng.integers(len(wh))]
            centroids = new_centroids

        dists = np.sum((wh[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        inertia = np.sum(np.min(dists, axis=1))

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()
            best_assignments = assignments.copy()

        print(f"  Trial {trial + 1}/{num_trials}: inertia = {inertia:.2f} ({iteration + 1} iters)")

    return best_centroids, best_assignments, best_inertia


# ============================================================================
# Assign anchors to FPN levels
# ============================================================================
def assign_to_fpn_levels(centroids, num_levels=5):
    """Assign anchors to FPN levels based on anchor area.

    Smaller anchors → lower FPN levels (P2, P3),
    larger anchors → higher FPN levels (P4, P5, P6).

    Args:
        centroids: (k, 2) anchor widths/heights
        num_levels: number of FPN levels (default: 5 for P2-P6)

    Returns:
        level_anchors: dict mapping level_idx → list of (w, h) tuples
    """
    # Sort by area
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_centroids = centroids[sorted_indices]

    # Distribute anchors across levels as evenly as possible
    k = len(sorted_centroids)
    per_level = k // num_levels
    remainder = k % num_levels

    level_anchors = {}
    idx = 0
    for level in range(num_levels):
        count = per_level + (1 if level < remainder else 0)
        level_anchors[level] = sorted_centroids[idx:idx + count]
        idx += count

    return level_anchors


def compute_base_sizes_and_ratios(level_anchors):
    """Convert per-level anchor (w,h) pairs into base_size + aspect_ratios + scales.

    Args:
        level_anchors: dict mapping level_idx → (num_anchors_per_level, 2) array

    Returns:
        anchor_sizes: tuple of base sizes per FPN level
    """
    anchor_sizes = []

    for level in sorted(level_anchors.keys()):
        anchors = level_anchors[level]
        if len(anchors) == 0:
            anchor_sizes.append(32)
            continue

        # Base size = geometric mean of sqrt(area) for this level
        areas = anchors[:, 0] * anchors[:, 1]
        base_size = np.exp(np.mean(np.log(np.sqrt(areas) + 1e-9)))
        anchor_sizes.append(float(base_size))

    return tuple(anchor_sizes)


# ============================================================================
# Visualization & Results
# ============================================================================
def print_box_statistics(wh_abs, wh_rel, img_size):
    """Print bounding box size statistics."""
    print("\n" + "=" * 60)
    print("Bounding Box Statistics")
    print("=" * 60)

    print(f"\nAbsolute (pixels):")
    print(f"  Width  — min: {wh_abs[:, 0].min():.1f}, max: {wh_abs[:, 0].max():.1f}, "
          f"mean: {wh_abs[:, 0].mean():.1f}, median: {np.median(wh_abs[:, 0]):.1f}")
    print(f"  Height — min: {wh_abs[:, 1].min():.1f}, max: {wh_abs[:, 1].max():.1f}, "
          f"mean: {wh_abs[:, 1].mean():.1f}, median: {np.median(wh_abs[:, 1]):.1f}")

    areas = wh_abs[:, 0] * wh_abs[:, 1]
    print(f"  Area   — min: {areas.min():.0f}, max: {areas.max():.0f}, "
          f"mean: {areas.mean():.0f}, median: {np.median(areas):.0f}")

    ratios = wh_abs[:, 0] / (wh_abs[:, 1] + 1e-9)
    print(f"  Ratio (w/h) — min: {ratios.min():.2f}, max: {ratios.max():.2f}, "
          f"mean: {ratios.mean():.2f}, median: {np.median(ratios):.2f}")

    # Size distribution in percentiles
    print(f"\n  Size percentiles (sqrt(area) in pixels):")
    sqrt_areas = np.sqrt(areas)
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    {p}th: {np.percentile(sqrt_areas, p):.1f}")

    # Scale to img_size
    print(f"\nRelative to image (img_size={img_size}):")
    scaled_w = wh_rel[:, 0] * img_size
    scaled_h = wh_rel[:, 1] * img_size
    print(f"  Width  — mean: {scaled_w.mean():.1f}, median: {np.median(scaled_w):.1f}")
    print(f"  Height — mean: {scaled_h.mean():.1f}, median: {np.median(scaled_h):.1f}")


def print_anchor_results(centroids_rel, avg_iou, img_size, level_anchors):
    """Print the computed anchor boxes."""
    print("\n" + "=" * 60)
    print(f"K-Means++ Anchor Boxes (avg IoU = {avg_iou:.4f})")
    print("=" * 60)

    # Sort by area
    areas = centroids_rel[:, 0] * centroids_rel[:, 1]
    sorted_idx = np.argsort(areas)

    print(f"\nAll anchors (sorted by area, scaled to img_size={img_size}):")
    for i, idx in enumerate(sorted_idx):
        w = centroids_rel[idx, 0] * img_size
        h = centroids_rel[idx, 1] * img_size
        ratio = w / (h + 1e-9)
        print(f"  Anchor {i:2d}: {w:7.1f} x {h:7.1f}  (ratio={ratio:.2f}, area={w * h:.0f})")

    print(f"\nAnchors assigned to FPN levels (P2, P3, P4, P5, P6):")
    anchor_sizes = []
    level_names = ["P2", "P3", "P4", "P5", "P6"]
    for level in sorted(level_anchors.keys()):
        anchors = level_anchors[level]
        name = level_names[level] if level < len(level_names) else f"P{level+2}"
        if len(anchors) == 0:
            print(f"  Level {level} ({name}): (empty)")
            continue

        scaled = anchors * img_size
        areas_level = scaled[:, 0] * scaled[:, 1]
        base_size = np.exp(np.mean(np.log(np.sqrt(areas_level) + 1e-9)))
        anchor_sizes.append(base_size)

        print(f"  Level {level} ({name}, base_size≈{base_size:.1f}):")
        for j, (w, h) in enumerate(scaled):
            ratio = w / (h + 1e-9)
            scale = math.sqrt(w * h) / base_size
            print(f"    {w:7.1f} x {h:7.1f}  (ratio={ratio:.2f}, scale={scale:.2f})")

    # Print ready-to-use config
    print("\n" + "=" * 60)
    print("Ready-to-use configuration")
    print("=" * 60)

    # anchor_sizes as integer tuple
    anchor_sizes_int = tuple(int(round(s)) for s in anchor_sizes)
    print(f"\nanchor_sizes = {anchor_sizes_int}")

    # Compute unique ratios and scales across all levels
    all_ratios = []
    all_scales = []
    for level in sorted(level_anchors.keys()):
        anchors = level_anchors[level]
        scaled = anchors * img_size
        areas_level = scaled[:, 0] * scaled[:, 1]
        base_size = np.exp(np.mean(np.log(np.sqrt(areas_level) + 1e-9)))

        for w, h in scaled:
            ratio = w / (h + 1e-9)
            scale = math.sqrt(w * h) / base_size
            all_ratios.append(round(ratio, 2))
            all_scales.append(round(scale, 2))

    unique_ratios = sorted(set(all_ratios))
    unique_scales = sorted(set(all_scales))
    print(f"aspect_ratios = {tuple(unique_ratios)}")
    print(f"scales = {tuple(unique_scales)}")

    # Print for FastViTDetector constructor
    print(f"\n# For FastViTDetector constructor:")
    print(f"model = FastViTDetector(")
    print(f"    model_name='fastvit_sa12',")
    print(f"    anchor_sizes={anchor_sizes_int},")
    print(f")")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute optimal anchor boxes from BDD100K annotations"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data/bdd100k",
        help="BDD100K root directory (default: ./data/bdd100k)"
    )
    parser.add_argument(
        "--ann-file", type=str, default=None,
        help="BDD100K COCO-format annotation file"
    )
    parser.add_argument(
        "--num-anchors", "-k", type=int, default=15,
        help="Total number of anchor clusters (default: 15 for 5 FPN levels x 3 anchors)"
    )
    parser.add_argument(
        "--num-levels", type=int, default=5,
        help="Number of FPN levels (default: 5, matching P2, P3, P4, P5, P6 stages)"
    )
    parser.add_argument(
        "--img-size", type=int, default=800,
        help="Target image size for scaling (default: 800, matching paper standard)"
    )
    parser.add_argument(
        "--distance", type=str, default="iou", choices=["iou", "euclidean"],
        help="Distance metric for K-Means (default: iou)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=300,
        help="Max iterations per K-Means trial (default: 300)"
    )
    parser.add_argument(
        "--num-trials", type=int, default=10,
        help="Number of K-Means trials (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("K-Means++ Anchor Box Computation for BDD100K")
    print("=" * 60)
    print(f"  Ann file:     {args.ann_file}")
    print(f"  Num anchors:  {args.num_anchors}")
    print(f"  Num levels:   {args.num_levels}")
    print(f"  Image size:   {args.img_size}")
    print(f"  Distance:     {args.distance}")
    print(f"  Trials:       {args.num_trials}")
    print(f"  Seed:         {args.seed}")
    print()

    if args.ann_file is None:
        args.ann_file = os.path.join(
            args.data_dir, "annotations", "bdd100k_det_train_coco.json"
        )
    if not os.path.exists(args.ann_file):
        parser.error(f"annotation file not found: {args.ann_file}")

    print("Step 1: Collecting bounding boxes from BDD100K annotations...")
    wh_abs, wh_rel = collect_coco_boxes(args.ann_file)

    if len(wh_abs) == 0:
        print("ERROR: No bounding boxes found! Check paths.")
        return

    # Print statistics
    print_box_statistics(wh_abs, wh_rel, args.img_size)

    # Step 2: K-Means++
    print(f"\nStep 2: Running K-Means++ (k={args.num_anchors}, metric={args.distance})...")
    if args.distance == "iou":
        centroids, assignments, avg_iou = kmeans_iou(
            wh_rel, k=args.num_anchors,
            max_iter=args.max_iter, num_trials=args.num_trials,
            seed=args.seed,
        )
    else:
        centroids, assignments, inertia = kmeans_euclidean(
            wh_rel, k=args.num_anchors,
            max_iter=args.max_iter, num_trials=args.num_trials,
            seed=args.seed,
        )
        # Compute avg IoU for comparison
        dists = iou_distance(wh_rel, centroids)
        avg_iou = 1.0 - np.mean(np.min(dists, axis=1))
        print(f"  Euclidean K-Means avg IoU: {avg_iou:.4f}")

    # Step 3: Assign to FPN levels
    print(f"\nStep 3: Assigning {args.num_anchors} anchors to {args.num_levels} FPN levels...")
    level_anchors = assign_to_fpn_levels(centroids, num_levels=args.num_levels)

    # Step 4: Print results
    print_anchor_results(centroids, avg_iou, args.img_size, level_anchors)

    # Step 5: Cluster size distribution
    print(f"\nCluster sizes:")
    for j in range(args.num_anchors):
        count = np.sum(assignments == j)
        print(f"  Cluster {j:2d}: {count:>6d} boxes ({count / len(assignments) * 100:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"Average IoU with anchors: {avg_iou:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
