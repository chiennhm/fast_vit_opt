#!/usr/bin/env python3
"""
Split BDD100K Dataset into Weather-Based Subsets (COCO Format).

This script reads BDD100K raw image metadata attributes (weather: clear, rainy, snowy,
overcast, partly cloudy, foggy, undefined) and partitions COCO instance annotation files
into individual COCO-format JSON files per weather condition under `./data/bdd100k/annotations/weather/`.

Usage:
    # Split validation set
    python split_bdd100k_by_weather.py --split val

    # Split training set
    python split_bdd100k_by_weather.py --split train

    # Split both train and val splits
    python split_bdd100k_by_weather.py --split both
"""

import os
import sys
import json
import argparse
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("split_bdd100k_by_weather")

BDD100K_CATEGORIES = [
    {"id": 1, "name": "bike"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motor"},
    {"id": 5, "name": "person"},
    {"id": 6, "name": "rider"},
    {"id": 7, "name": "traffic light"},
    {"id": 8, "name": "traffic sign"},
    {"id": 9, "name": "train"},
    {"id": 10, "name": "truck"},
]


def sanitize_weather_name(name):
    """Normalize weather attribute name to lowercase string without spaces."""
    if not name or name == "unknown":
        return "undefined"
    name = str(name).strip().lower().replace(" ", "_")
    return name


def load_weather_attribute_map(raw_label_path):
    """
    Map image file_name/name -> weather attribute from raw BDD100K JSON file.
    """
    if not os.path.exists(raw_label_path):
        logger.error(f"Raw BDD100K label file not found: {raw_label_path}")
        return {}

    logger.info(f"Loading raw BDD100K labels from: {raw_label_path}")
    with open(raw_label_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    weather_map = {}
    weather_counts = Counter()

    for item in raw_items:
        img_name = item.get("name")
        attrs = item.get("attributes", {})
        weather_raw = attrs.get("weather", "undefined")
        weather = sanitize_weather_name(weather_raw)

        if img_name:
            weather_map[img_name] = weather
            base_name = os.path.basename(img_name)
            weather_map[base_name] = weather
            weather_counts[weather] += 1

    logger.info(f"Loaded weather attributes for {len(raw_items)} images.")
    logger.info("Weather Distribution:")
    for w, c in weather_counts.most_common():
        logger.info(f"  - {w:<20}: {c:6d} images")

    return weather_map


def split_coco_json_by_weather(coco_path, weather_map, output_dir, split_name="val"):
    """
    Split a COCO-format instances JSON file into separate JSON files per weather condition.
    """
    if not os.path.exists(coco_path):
        logger.error(f"COCO instances file not found: {coco_path}")
        return

    logger.info(f"Loading COCO instances from: {coco_path}")
    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", BDD100K_CATEGORIES)

    logger.info(f"  Loaded {len(images)} images and {len(annotations)} annotations.")

    images_by_weather = {}
    img_id_to_weather = {}

    for img in images:
        img_id = img["id"]
        file_name = os.path.basename(img.get("file_name", img.get("name", "")))
        weather = weather_map.get(file_name, "undefined")

        if weather not in images_by_weather:
            images_by_weather[weather] = []
        images_by_weather[weather].append(img)
        img_id_to_weather[img_id] = weather

    annotations_by_weather = {w: [] for w in images_by_weather.keys()}
    for ann in annotations:
        img_id = ann["image_id"]
        weather = img_id_to_weather.get(img_id, "undefined")
        if weather in annotations_by_weather:
            annotations_by_weather[weather].append(ann)

    os.makedirs(output_dir, exist_ok=True)
    summary_table = []

    for weather, w_images in sorted(images_by_weather.items()):
        w_anns = annotations_by_weather.get(weather, [])
        out_filename = f"coco_{split_name}_{weather}.json"
        out_path = os.path.join(output_dir, out_filename)

        w_coco_data = {
            "info": coco_data.get("info", {"description": f"BDD100K {split_name} subset - {weather}"}),
            "licenses": coco_data.get("licenses", []),
            "images": w_images,
            "annotations": w_anns,
            "categories": categories,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(w_coco_data, f)

        summary_table.append({
            "weather": weather,
            "file": out_filename,
            "images": len(w_images),
            "annotations": len(w_anns),
            "path": out_path,
        })
        logger.info(f"Saved {weather:<15} subset ({len(w_images):5d} images, {len(w_anns):6d} boxes) -> {out_path}")

    return summary_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split BDD100K dataset into weather-based COCO subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", "--data_dir", type=str, default="./data/bdd100k",
        help="Root directory of BDD100K dataset (default: ./data/bdd100k)"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "train", "both"],
        help="Dataset split to partition (default: val)"
    )
    parser.add_argument(
        "--output-dir", "--output_dir", type=str, default=None,
        help="Output directory for weather JSON files (default: <data-dir>/annotations/weather)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = args.output_dir if args.output_dir else os.path.join(data_dir, "annotations", "weather")

    splits_to_process = ["val", "train"] if args.split == "both" else [args.split]

    for split in splits_to_process:
        logger.info(f"\n=======================================================")
        logger.info(f"Processing BDD100K '{split}' split...")
        logger.info(f"=======================================================")

        raw_label_path = os.path.join(data_dir, split, "annotations", f"bdd100k_labels_images_{split}.json")
        if not os.path.exists(raw_label_path):
            raw_label_path = os.path.join(data_dir, "annotations", f"bdd100k_labels_images_{split}.json")

        coco_path = os.path.join(data_dir, split, "annotations", f"coco_instances_{split}.json")
        if not os.path.exists(coco_path):
            coco_path = os.path.join(data_dir, "annotations", f"bdd100k_det_{split}_coco.json")

        weather_map = load_weather_attribute_map(raw_label_path)
        if weather_map:
            split_coco_json_by_weather(coco_path, weather_map, output_dir, split_name=split)

    logger.info(f"\nAll weather subsets generated under: {output_dir}")


if __name__ == "__main__":
    main()
