"""Final-checkpoint evaluation by weather and confidence using one inference cache."""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from detection.metrics import (
    build_coco_ground_truth, build_coco_predictions, evaluate_coco,
    print_eval_results, summarize_detection_counts,
)
from engine.evaluate import evaluate
from tools.download_bdd100k import find_label_file
from tools.split_by_weather import load_weather_attribute_map

logger = logging.getLogger(__name__)
DEFAULT_CONFIDENCES = (0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
METRIC_KEYS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR100")


def prepare_weather_groups(images, data_dir, weather_labels=None):
    """Use filenames, never raw-label enumeration positions, to join weather."""
    raw_path = Path(weather_labels) if weather_labels else find_label_file(data_dir, "val")
    weather_map = load_weather_attribute_map(raw_path, strict=True)
    groups, missing = {}, []
    for image in images:
        name = str(image["file_name"]).replace(chr(92), "/")
        weather = weather_map.get(name, weather_map.get(name.rsplit("/", 1)[-1]))
        if weather is None:
            missing.append(name)
        else:
            groups.setdefault(weather, []).append(int(image["id"]))
    if missing:
        raise ValueError(
            f"{len(missing)} validation images have no weather metadata; examples: "
            f"{missing[:5]}. Check --weather-labels and the validation split.")
    if not groups:
        raise ValueError("No validation images available for weather evaluation")
    return groups, str(raw_path.resolve())


def select_final_checkpoint(output_dir, resume=None):
    """Prefer a newly saved best, then the best in the resume ancestry."""
    output_dir = Path(output_dir)
    best = output_dir / "best.pth"
    if best.is_file():
        return best, "best_in_current_run"
    candidate, seen = Path(resume) if resume else None, set()
    while candidate is not None and candidate.is_file():
        resolved = candidate.resolve()
        if resolved in seen:
            raise ValueError("Cycle in resume checkpoint ancestry")
        seen.add(resolved)
        best = candidate.parent / "best.pth"
        if best.is_file():
            return best, "best_in_resume_ancestry"
        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        parent = checkpoint.get("args", {}).get("resume")
        del checkpoint
        candidate = Path(parent) if parent else None
    last = output_dir / "last.pth"
    if last.is_file():
        logger.warning("No best.pth artifact found; final evaluation uses current last.pth")
        return last, "last_fallback_no_best_artifact"
    if resume and Path(resume).is_file():
        return Path(resume), "input_checkpoint_fallback"
    raise FileNotFoundError("No saved checkpoint available for final evaluation")


def filter_predictions(predictions, confidence):
    """Filter post-NMS predictions upward, retaining empty images and ordering."""
    filtered = []
    for prediction in predictions:
        mask = np.asarray(prediction["scores"], dtype=np.float64) > confidence
        item = dict(prediction)
        for key in ("boxes", "scores", "labels"):
            item[key] = np.asarray(prediction[key])[mask]
        filtered.append(item)
    return filtered


def _json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(value):
    return None if value is None else 100 * float(value)


def _counts_row(counts, images):
    return {
        "Precision50": _percent(counts["precision"]),
        "Recall50": _percent(counts["recall"]),
        "F1_50": _percent(counts["f1"]),
        "TP50": counts["TP"], "FP50": counts["FP"], "FN50": counts["FN"],
        "FP_per_image": counts["FP"] / images if images else None,
    }


def _metric_row(result, confidence, images, annotations, weather="all"):
    has_ground_truth = result["operating_point"]["TP"] + result["operating_point"]["FN"] > 0
    return {
        "Weather": weather, "Confidence": confidence, "Images": images,
        "Annotations": annotations,
        **{key: (_percent(result.get("mAP" if key == "AP" else key))
                 if has_ground_truth else None)
           for key in METRIC_KEYS},
        **_counts_row(result["operating_point"], images),
    }


def _compact(result):
    return {key: value for key, value in result.items()
            if key != "operating_points_by_image"}


def _table(rows, columns):
    def display(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.3f}" if 0 < abs(value) < 0.01 else f"{value:.2f}"
        return str(value)
    header = "| " + " | ".join(columns) + " |\n"
    header += "|" + "|".join("---" for _ in columns) + "|\n"
    return header + "\n".join(
        "| " + " | ".join(display(row.get(key)) for key in columns) + " |"
        for row in rows)


def _plots(output_dir, confidence_rows, weather_confidence_rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    x = [row["Confidence"] for row in confidence_rows]
    for key in ("Precision50", "Recall50", "F1_50"):
        axes[0, 0].plot(x, [row[key] for row in confidence_rows], marker="o", label=key)
    axes[0, 0].set_ylabel("% at IoU 0.50 (micro)")
    axes[0, 0].legend()
    axes[0, 1].plot(x, [row["FP_per_image"] for row in confidence_rows], marker="o")
    axes[0, 1].set_ylabel("False positives / image at IoU 0.50")
    for key in ("AP", "AP50", "AP75"):
        axes[1, 0].plot(x, [row[key] for row in confidence_rows], marker="o", label=key)
    axes[1, 0].set_ylabel("COCO AP (%) after confidence filtering")
    axes[1, 0].legend()
    for weather in sorted({row["Weather"] for row in weather_confidence_rows}):
        rows = [row for row in weather_confidence_rows if row["Weather"] == weather]
        axes[1, 1].plot(
            [row["Confidence"] for row in rows],
            [row["FP_per_image"] for row in rows], marker=".", label=weather)
    axes[1, 1].set_ylabel("False positives / image by weather")
    axes[1, 1].legend(fontsize=7)
    for axis in axes.flat:
        axis.set_xlabel("Confidence cutoff (score > cutoff)")
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "confidence_analysis.png", dpi=180)


def run_evaluation_suite(
    model, loader, device, *, class_names, output_dir, weather_groups,
    source_images, checkpoint_info, reference_confidence=0.05,
    confidence_thresholds=DEFAULT_CONFIDENCES, amp=True, nms_thresh=0.5,
    max_detections=100, pre_nms_topk=2000, max_samples=None,
    weather_labels_path=None, model_name="fastvit_sa12",
):
    """Reference COCO eval, weather eval, and confidence/FP analysis after training."""
    thresholds = sorted(set([float(reference_confidence),
                             *(float(value) for value in confidence_thresholds)]))
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Confidence thresholds must be finite and between 0 and 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "inference_cache"
    cache_dir.mkdir(exist_ok=True)
    floor = min(thresholds)
    logger.info("Final evaluation checkpoint: %s", checkpoint_info["path"])
    logger.info("One validation inference pass at confidence %.4f", floor)
    _, predictions, targets = evaluate(
        model, loader, device, class_names=class_names, amp=amp,
        score_thresh=floor, nms_thresh=nms_thresh, max_detections=max_detections,
        pre_nms_topk=pre_nms_topk, max_samples=max_samples,
        output_dir=cache_dir, show_progress=False, return_records=True,
    )
    if not targets:
        raise ValueError("Final evaluation received an empty validation split")
    image_ids = [int(np.asarray(item["image_id"]).reshape(-1)[0]) for item in targets]
    id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    if len(id_to_index) != len(image_ids):
        raise ValueError("Duplicate validation image IDs")
    groups = {weather: [id_to_index[int(image_id)] for image_id in group_ids
                        if int(image_id) in id_to_index]
              for weather, group_ids in weather_groups.items()}
    groups = {weather: indices for weather, indices in groups.items() if indices}
    grouped_ids = [image_ids[index] for indices in groups.values() for index in indices]
    if sorted(grouped_ids) != sorted(image_ids):
        raise ValueError("Weather groups must partition every evaluated image exactly once")

    ground_truth = build_coco_ground_truth(targets, len(class_names), class_names)
    source_images = {int(row["id"]): row for row in source_images}
    weather_by_id = {image_ids[index]: weather for weather, indices in groups.items()
                     for index in indices}
    for image in ground_truth["images"]:
        image["file_name"] = source_images[image["id"]]["file_name"]
        image["weather"] = weather_by_id[image["id"]]
    _json(cache_dir / "ground_truths_BDD100K.json", ground_truth)
    _json(cache_dir / "predictions_min_confidence.json", build_coco_predictions(predictions))

    confidence_rows, class_rows, weather_confidence_rows = [], [], []
    confidence_results, reference = {}, None
    for confidence in thresholds:
        filtered = filter_predictions(predictions, confidence)
        result = evaluate_coco(filtered, targets, num_classes=len(class_names),
                               class_names=class_names, include_operating_points=True)
        confidence_results[str(confidence)] = _compact(result)
        confidence_rows.append(_metric_row(
            result, confidence, len(targets), len(ground_truth["annotations"])))
        for name, metric in result["per_class"].items():
            class_rows.append({
                "Class": name, "Confidence": confidence,
                **{key: _percent(metric[key]) for key in ("AP", "AP50", "AP75", "AR100")},
                **_counts_row(metric["operating_point"], len(targets)),
            })
        for weather, indices in sorted(groups.items()):
            points = [result["operating_points_by_image"][str(image_ids[index])]
                      for index in indices]
            counts = summarize_detection_counts(
                *(sum(point[key] for point in points) for key in ("TP", "FP", "FN")))
            weather_confidence_rows.append({
                "Weather": weather, "Confidence": confidence, "Images": len(indices),
                **_counts_row(counts, len(indices)),
            })
        logger.info("Final confidence %.4f: AP=%.2f%% FP/image=%.3f", confidence,
                    result["mAP"] * 100, confidence_rows[-1]["FP_per_image"])
        if confidence == reference_confidence:
            reference = result
            _json(output_dir / "predictions_BDD100K.json", build_coco_predictions(filtered))
    if reference is None:
        raise RuntimeError("Reference confidence was not evaluated")

    weather_rows, weather_results = [], {}
    filtered = filter_predictions(predictions, reference_confidence)
    for weather, indices in sorted(groups.items()):
        subset_gt = [targets[index] for index in indices]
        result = evaluate_coco(
            [filtered[index] for index in indices], subset_gt,
            num_classes=len(class_names), class_names=class_names,
            include_operating_points=True)
        annotations = sum(len(item["labels"]) for item in subset_gt)
        weather_rows.append(_metric_row(
            result, reference_confidence, len(indices), annotations, weather))
        weather_results[weather] = _compact(result)
        weather_ap = weather_rows[-1]["AP"]
        logger.info("Final weather %s: %d images, AP=%s", weather, len(indices),
                    "N/A" if weather_ap is None else f"{weather_ap:.2f}%")

    reference_row = next(row for row in confidence_rows
                         if row["Confidence"] == reference_confidence)
    for filename, rows in (
        ("metrics_overall.csv", [reference_row]),
        ("metrics_by_class.csv", [row for row in class_rows
                                  if row["Confidence"] == reference_confidence]),
        ("metrics_by_weather.csv", weather_rows),
        ("metrics_by_confidence.csv", confidence_rows),
        ("metrics_by_class_confidence.csv", class_rows),
        ("metrics_by_weather_confidence.csv", weather_confidence_rows),
    ):
        _csv(output_dir / filename, [{"Model": model_name, **row} for row in rows])
    _json(output_dir / "eval_results.json", _compact(reference))
    _json(output_dir / "confidence_results.json", confidence_results)
    _json(output_dir / "weather_results.json", weather_results)
    valid_f1 = [row for row in confidence_rows if row["F1_50"] is not None]
    best_f1 = max(valid_f1, key=lambda row: row["F1_50"]) if valid_f1 else None
    manifest = {
        "model": model_name,
        "checkpoint": checkpoint_info, "reference_confidence": reference_confidence,
        "confidence_thresholds": thresholds, "inference_confidence_floor": floor,
        "weather_labels": weather_labels_path,
        "weather_image_counts": {weather: len(indices) for weather, indices in groups.items()},
        "evaluated_images": len(targets), "max_eval_samples": max_samples,
        "subset_evaluation": max_samples is not None,
        "inference": {"amp": amp, "nms_iou": nms_thresh,
                      "max_detections_per_image": max_detections,
                      "pre_nms_topk_anchor_locations": pre_nms_topk},
        "coco_protocol": reference["coco_protocol"],
        "confidence_sweep_method": "strict_score_filter_on_one_post_nms_cache",
        "operating_point": {"iou": 0.5, "averaging": "micro",
                            "crowd_matches": "ignored", "true_negatives": "not_defined",
                            "zero_predictions_precision": 0.0, "no_gt_recall_f1": None},
        "best_validation_f1_diagnostic": best_f1,
        "small_weather_groups": [weather for weather, indices in groups.items() if len(indices) < 100],
    }
    _json(output_dir / "evaluation_manifest.json", manifest)
    _plots(output_dir, confidence_rows, weather_confidence_rows)

    report = [
        "# Final checkpoint evaluation", "",
        f"Checkpoint: §{checkpoint_info['path']}§. Selection: §{checkpoint_info['selection']}§.",
        f"Completed epochs: {checkpoint_info.get('completed_epochs')}.",
        f"Images: {len(targets)}. Reference confidence: {reference_confidence}.",
        f"Confidence floor: {floor}; one inference pass, fixed NMS/top-k/max detections.",
        "", "## Reference COCO metrics", "",
        _table([reference_row], ["Confidence", *METRIC_KEYS]), "",
        "## Weather at reference confidence", "",
        _table(weather_rows, ["Weather", "Images", "AP", "AP50", "AR100",
                              "Precision50", "Recall50", "F1_50", "FP_per_image"]), "",
        "## Confidence operating points", "",
        _table(confidence_rows, ["Confidence", "AP", "AP50", "AP75",
                                 "Precision50", "Recall50", "F1_50", "FP_per_image"]), "",
        "CSV metrics are percentages; confidence is 0..1 and FP/image is a count.",
        "Precision/recall/F1 are micro-averaged at IoU 0.50 using COCO matches and ignore masks.",
        "Weather/confidence CSV contains operating-point counts; weather COCO AP is computed at reference confidence.",
        "No true-negative count or detection ROC is reported.",
        "AP at higher cutoffs is computed after removing low-score predictions.",
        "AP/FP counts are bounded by the fixed pre-NMS top-k and max detections.",
        "Sparse weather groups and classes without GT must be read with their sample counts/N/A values.",
        "The best F1 threshold on this validation set is diagnostic, not an independent test result.",
        "For YOLO comparisons, keep the same split, categories, confidence, NMS and detection limits.",
        "Also compare models at matched recall or FP/image; equal confidence values can have different calibration.",
        "", "![Confidence analysis](confidence_analysis.png)", "",
        "[Overall](metrics_overall.csv) | [Classes](metrics_by_class.csv) | "
        "[Weather](metrics_by_weather.csv) | [Confidence](metrics_by_confidence.csv) | "
        "[Weather/confidence](metrics_by_weather_confidence.csv)", "",
    ]
    if max_samples is not None:
        report.insert(2, "**SUBSET evaluation: not a full-validation benchmark.**")
    (output_dir / "REPORT.md").write_text("\n".join(report).replace("§", chr(96)), encoding="utf-8")
    print_eval_results(reference, logger_fn=logger.info)
    logger.info("Final evaluation reports saved to %s", output_dir)
    return manifest
