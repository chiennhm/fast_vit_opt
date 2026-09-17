#
# BDD100K Annotations Downloader and COCO-Style Converter
#

import os
import sys
import time
import zipfile
import urllib.request
import json

BDD100K_LABELS_URL = (
    "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip"
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
    urllib.request.install_opener(opener)

    try:
        response = urllib.request.urlopen(url)
        meta = response.info()
        file_size = int(meta.get("Content-Length", 0))
        print(f"Total size: {format_size(file_size)}")

        downloaded = 0
        block_size = 1024 * 1024

        with open(dest_path, "wb") as f:
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

        print(f"\nDownload complete in {time.time() - start_time:.1f}s.")
        return True
    except Exception as e:
        print(f"\nError downloading {descr}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file to the target directory."""
    print(f"Extracting {os.path.basename(zip_path)} to {extract_to}...")
    start_time = time.time()
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            total_files = len(members)

            for i, member in enumerate(members):
                zip_ref.extract(member, extract_to)
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


def convert_bdd_to_coco(bdd_json_path, coco_json_path):
    """Convert BDD100K format annotations to COCO-style format."""
    print(f"\nConverting {bdd_json_path} to COCO-style format...")
    if not os.path.exists(bdd_json_path):
        print(f"Error: {bdd_json_path} not found!")
        return False

    with open(bdd_json_path, "r") as f:
        bdd_data = json.load(f)

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
    dest_dir = "./data/bdd100k"
    os.makedirs(dest_dir, exist_ok=True)

    zip_path = os.path.join(dest_dir, "bdd100k_det_20_labels_trainval.zip")

    # Download
    print("============================================================")
    print("  BDD100K Labels Downloader and Converter")
    print("============================================================")
    print(f"Destination directory: {os.path.abspath(dest_dir)}")
    print("------------------------------------------------------------")

    # If extracted files already exist, we can skip download
    expected_train = os.path.join(dest_dir, "labels", "det_20", "det_train.json")
    expected_val = os.path.join(dest_dir, "labels", "det_20", "det_val.json")

    if not (os.path.exists(expected_train) and os.path.exists(expected_val)):
        if not os.path.exists(zip_path):
            success = download_with_progress(
                BDD100K_LABELS_URL,
                zip_path,
                "BDD100K Detection Labels ZIP (~43 MB)",
            )
            if not success:
                print("Download failed. Exiting.")
                sys.exit(1)

        # Extract
        success = extract_zip(zip_path, dest_dir)
        if not success:
            print("Extraction failed. Exiting.")
            sys.exit(1)

        # Clean up zip
        print(f"Deleting temporary zip file: {os.path.basename(zip_path)}...")
        try:
            os.remove(zip_path)
            print("Deleted successfully.")
        except Exception as e:
            print(f"Could not delete zip file: {e}")

    # Convert
    annotations_dir = os.path.join(dest_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    convert_bdd_to_coco(
        bdd_json_path=os.path.join(dest_dir, "labels", "det_20", "det_train.json"),
        coco_json_path=os.path.join(annotations_dir, "bdd100k_det_train_coco.json"),
    )
    convert_bdd_to_coco(
        bdd_json_path=os.path.join(dest_dir, "labels", "det_20", "det_val.json"),
        coco_json_path=os.path.join(annotations_dir, "bdd100k_det_val_coco.json"),
    )

    print("\n============================================================")
    print("  All done! BDD100K annotations converted to COCO-style format.")
    print("============================================================")


if __name__ == "__main__":
    main()
