#
# BDD100K Kaggle Downloader and COCO-Style Converter
#

import argparse
import os
import sys
import time
import zipfile
import urllib.request
import json
import shutil
from pathlib import Path, PurePosixPath

BDD100K_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/solesensei/solesensei_bdd100k"
)


def format_size(bytes_size):
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def download_with_progress(url, dest_path, descr):
    """Download a file showing progress bar and download speed."""
    print(f"\nDownloading {descr} from:\n  {url}")
    start_time = time.time()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    partial_path = str(dest_path) + ".part"

    try:
        downloaded = 0
        block_size = 1024 * 1024

        with opener.open(url, timeout=120) as response, open(partial_path, "wb") as f:
            file_size = int(response.info().get("Content-Length", 0))
            print(f"Total size: {format_size(file_size) if file_size else 'unknown'}")
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break

                downloaded += len(buffer)
                f.write(buffer)

                elapsed = time.time() - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                eta = (file_size - downloaded) / speed if speed > 0 else 0

                percent = (downloaded / file_size) * 100 if file_size > 0 else 0
                bar_len = 30
                filled_len = (
                    int(round(bar_len * downloaded / float(file_size)))
                    if file_size > 0
                    else 0
                )
                bar = "█" * filled_len + "-" * (bar_len - filled_len)

                status = f"\rProgress: |{bar}| {percent:5.1f}% ({format_size(downloaded)}/{format_size(file_size)}) | {format_size(speed)}/s | ETA: {eta:.0f}s"
                sys.stdout.write(status)
                sys.stdout.flush()

        if file_size and downloaded != file_size:
            raise IOError(f"Incomplete download: {downloaded}/{file_size} bytes")
        if not zipfile.is_zipfile(partial_path):
            raise ValueError("Kaggle did not return a ZIP archive; check dataset access.")
        os.replace(partial_path, dest_path)
        print(f"\nDownload complete in {time.time() - start_time:.1f}s.")
        return True
    except Exception as e:
        print(f"\nError downloading {descr}: {e}")
        if os.path.exists(partial_path):
            os.remove(partial_path)
        return False


def dataset_member_path(name):
    """Map nested Kaggle paths to the detector's image and label layout."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if not parts or ".." in parts or any(":" in part for part in parts):
        return None
    for index in range(len(parts) - 3):
        if parts[index:index + 2] == ("images", "100k"):
            if parts[index + 2] in ("train", "val", "test"):
                return Path(*parts[index:])
    for split in ("train", "val"):
        if parts[-1] in (f"bdd100k_labels_images_{split}.json", f"det_{split}.json"):
            return Path("labels", "det_20", f"det_{split}.json")
    return None


def extract_zip(zip_path, extract_to):
    """Extract detection data, removing any enclosing Kaggle directories."""
    print(f"Extracting {os.path.basename(zip_path)} to {extract_to}...")
    start_time = time.time()
    root = Path(extract_to).resolve()
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            total_files = len(members)

            for i, member in enumerate(members):
                relative_path = dataset_member_path(member.filename)
                if member.is_dir() or relative_path is None:
                    continue
                destination = Path(extract_to) / relative_path
                if not destination.resolve().is_relative_to(root):
                    raise ValueError(f"Archive path escapes destination: {member.filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if i % max(1, total_files // 100) == 0 or i == total_files - 1:
                    percent = (i + 1) / total_files * 100
                    sys.stdout.write(
                        f"\rExtraction progress: {percent:.1f}% ({i + 1}/{total_files} files)"
                    )
                    sys.stdout.flush()

        print(f"\nExtraction complete in {time.time() - start_time:.1f}s.")
        return True
    except Exception as e:
        print(f"\nError extracting {zip_path}: {e}")
        return False


def find_label_file(data_dir, split):
    """Locate normalized labels or the original Kaggle labels release."""
    root = Path(data_dir)
    filename = f"bdd100k_labels_images_{split}.json"
    candidates = [
        root / "labels" / "det_20" / f"det_{split}.json",
        root / "bdd100k_labels_release" / "bdd100k" / "labels" / filename,
        root / "bdd100k" / "labels" / filename,
        root / "labels" / filename,
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def find_images(image_dir):
    """Index local images once, including every nested shard directory."""
    images = {}
    for path in sorted(Path(image_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            images.setdefault(path.name, []).append(path)
    return images


def convert_bdd_to_coco(bdd_json_path, coco_json_path, image_dir=None):
    """Convert BDD100K format annotations to COCO-style format."""
    print(f"\nConverting {bdd_json_path} to COCO-style format...")
    if not os.path.exists(bdd_json_path):
        print(f"Error: {bdd_json_path} not found!")
        return False

    with open(bdd_json_path, "r") as f:
        bdd_data = json.load(f)

    image_root = Path(image_dir) if image_dir is not None else None
    image_index = find_images(image_root) if image_root is not None else None
    if image_index is not None:
        print(f"Found {sum(map(len, image_index.values()))} images under {image_root}")

    categories = [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "rider"},
        {"id": 3, "name": "car"},
        {"id": 4, "name": "truck"},
        {"id": 5, "name": "bus"},
        {"id": 6, "name": "train"},
        {"id": 7, "name": "motorcycle"},
        {"id": 8, "name": "bicycle"},
        {"id": 9, "name": "traffic light"},
        {"id": 10, "name": "traffic sign"},
    ]

    cat_to_id = {cat["name"]: cat["id"] for cat in categories}
    alt_name_map = {
        "person": "pedestrian",
        "bike": "bicycle",
        "motor": "motorcycle",
    }

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_id, frame in enumerate(bdd_data, start=1):
        file_name = frame["name"]
        if image_root is not None:
            relative_name = PurePosixPath(file_name.replace("\\", "/"))
            image_path = image_root / relative_name
            if not image_path.is_file():
                candidates = image_index.get(relative_name.name, [])
                if not candidates:
                    raise FileNotFoundError(f"Annotation image not found under {image_root}: {file_name}")
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous image name {file_name}: {candidates}")
                image_path = candidates[0]
            file_name = image_path.relative_to(image_root).as_posix()
        width = 1280
        height = 720

        coco_images.append(
            {
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": file_name,
            }
        )

        labels = frame.get("labels", [])
        if labels is None:
            labels = []

        for label in labels:
            category = label.get("category")
            if category in alt_name_map:
                category = alt_name_map[category]

            if category not in cat_to_id:
                continue

            cat_id = cat_to_id[category]

            box2d = label.get("box2d")
            if box2d is None:
                continue

            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])

            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))

            w = x2 - x1
            h = y2 - y1

            if w > 0 and h > 0:
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    with open(coco_json_path, "w") as f:
        json.dump(coco_data, f)

    print(f"Successfully created: {coco_json_path}")
    print(f"  - Images: {len(coco_images)}")
    print(f"  - Annotations: {len(coco_annotations)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download BDD100K images and labels from Kaggle and convert labels to COCO JSON"
    )
    parser.add_argument("--dest-dir", default="./data/bdd100k")
    parser.add_argument("--archive", help="Use an existing Kaggle ZIP instead of downloading")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded ZIP after successful conversion")
    parser.add_argument("--convert-only", action="store_true", help="Scan existing local images and convert labels without downloading or extracting")
    args = parser.parse_args()
    if args.convert_only and args.archive:
        parser.error("--convert-only and --archive are mutually exclusive")
    dest_dir = os.path.expanduser(args.dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    zip_path = (
        os.path.expanduser(args.archive) if args.archive
        else os.path.join(dest_dir, "bdd100k.zip")
    )
    if args.archive and not os.path.isfile(zip_path):
        parser.error(f"Archive not found: {zip_path}")

    # Download
    print("============================================================")
    print("  BDD100K Kaggle Downloader and Converter")
    print("============================================================")
    print(f"Destination directory: {os.path.abspath(dest_dir)}")
    print("------------------------------------------------------------")

    # A marker is written only after extraction and conversion both succeed.
    expected_train = find_label_file(dest_dir, "train")
    expected_val = find_label_file(dest_dir, "val")
    marker = Path(dest_dir) / ".kaggle_bdd100k_complete"
    images_ready = all(
        bool(find_images(Path(dest_dir) / "images" / "100k" / split))
        for split in ("train", "val")
    )
    ready = marker.is_file() and images_ready and all(
        os.path.isfile(path) for path in (expected_train, expected_val)
    )

    if not args.convert_only and (args.archive or not ready):
        marker.unlink(missing_ok=True)
        if not os.path.exists(zip_path):
            success = download_with_progress(
                BDD100K_DATASET_URL,
                zip_path,
                "BDD100K images and labels (Kaggle ZIP)",
            )
            if not success:
                print("Download failed. Exiting.")
                sys.exit(1)

        # Extract
        success = extract_zip(zip_path, dest_dir)
        if not success:
            print("Extraction failed. Exiting.")
            sys.exit(1)

    label_paths = {split: find_label_file(dest_dir, split) for split in ("train", "val")}
    for path in label_paths.values():
        if not os.path.isfile(path):
            sys.exit(f"Missing detection labels after extraction: {path}")
    for split in ("train", "val"):
        image_dir = Path(dest_dir) / "images" / "100k" / split
        if not find_images(image_dir):
            sys.exit(f"Missing {split} images after extraction: {image_dir}")

    # Convert
    annotations_dir = os.path.join(dest_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    for split in ("train", "val"):
        if not convert_bdd_to_coco(
            bdd_json_path=label_paths[split],
            coco_json_path=os.path.join(annotations_dir, f"bdd100k_det_{split}_coco.json"),
            image_dir=Path(dest_dir) / "images" / "100k" / split,
        ):
            sys.exit(f"Failed to convert {split} annotations")
    marker.write_text(BDD100K_DATASET_URL + "\n", encoding="utf-8")
    if not args.convert_only and not args.archive and not args.keep_zip and os.path.isfile(zip_path):
        os.remove(zip_path)

    print("\n============================================================")
    print("  All done! BDD100K images and COCO annotations are ready.")
    print("============================================================")


if __name__ == "__main__":
    main()
