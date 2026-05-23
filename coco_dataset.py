#
# MS-COCO Dataset for Object Detection (pure Python parser)
#

import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import random
from collections import defaultdict


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

COCO_CAT_TO_IDX = {cat_id: i + 1 for i, cat_id in enumerate(COCO_CAT_IDS)}


class COCODetectionDataset(Dataset):
    """MS-COCO Detection Dataset using pure Python JSON parser.

    Avoids dependency on pycocotools for easy Windows compilation.
    Annotations are converted from COCO JSON to [x1, y1, x2, y2] format.
    """

    def __init__(
        self,
        img_dir,
        ann_file,
        img_size=512,
        augment=True,
    ):
        """
        Args:
            img_dir: Directory containing COCO images.
            ann_file: Path to COCO JSON annotations.
            img_size: Target image size (square).
            augment: Apply data augmentation.
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        print(f"Loading COCO annotations from {ann_file}...")
        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Build lookup tables
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            # Only keep annotation if the category is valid in COCO's 80 classes
            if ann["category_id"] in COCO_CAT_TO_IDX:
                self.img_to_anns[ann["image_id"]].append(ann)

        self.img_ids = list(self.images.keys())
        print(f"COCO Dataset: {len(self.img_ids)} images, {len(coco_data['annotations'])} annotations loaded")

    def __len__(self):
        return len(self.img_ids)

    def _get_annotations(self, img_id):
        """Retrieve annotations for a given image ID.

        Returns:
            boxes: list of [x1, y1, x2, y2]
            labels: list of class indices (1-indexed)
            difficults: list of bools (mapped from iscrowd)
        """
        anns = self.img_to_anns[img_id]
        boxes = []
        labels = []
        difficults = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in COCO_CAT_TO_IDX:
                continue

            # Convert bbox [x, y, w, h] to [x1, y1, x2, y2]
            bbox = ann["bbox"]
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = x1 + float(bbox[2])
            y2 = y1 + float(bbox[3])

            # Treat iscrowd=1 as difficult
            is_difficult = bool(ann.get("iscrowd", 0) == 1)

            boxes.append([x1, y1, x2, y2])
            labels.append(COCO_CAT_TO_IDX[cat_id])
            difficults.append(is_difficult)

        return boxes, labels, difficults

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, file_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get annotations
        boxes, labels, difficults = self._get_annotations(img_id)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
            difficults = np.array([], dtype=bool)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            difficults = np.array(difficults, dtype=bool)

        if self.augment:
            # For training: exclude difficult (iscrowd) objects from augmentation & targets
            easy_mask = ~difficults
            train_boxes = boxes[easy_mask] if easy_mask.any() else np.zeros((0, 4), dtype=np.float32)
            train_labels = labels[easy_mask] if easy_mask.any() else np.array([], dtype=np.int64)

            if len(train_boxes) > 0:
                image, train_boxes = self._augment(image, train_boxes)

            image, train_boxes = self._resize(image, train_boxes, self.img_size)

            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)

            targets = {
                "boxes": torch.tensor(train_boxes, dtype=torch.float32),
                "labels": torch.tensor(train_labels, dtype=torch.int64),
            }
        else:
            # For eval: keep all objects, pass difficult flag
            image, boxes = self._resize(image, boxes, self.img_size)

            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)

            targets = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "difficults": torch.tensor(difficults, dtype=torch.bool),
            }

        return image, targets

    def _augment(self, image, boxes):
        """Apply detection-safe augmentations."""
        w, h = image.size

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            new_boxes = boxes.copy()
            new_boxes[:, 0] = w - boxes[:, 2]
            new_boxes[:, 2] = w - boxes[:, 0]
            boxes = new_boxes

        # Random color jitter
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))

        # Random expand (zoom out)
        if random.random() > 0.5:
            ratio = random.uniform(1.0, 2.0)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            left = random.randint(0, new_w - w)
            top = random.randint(0, new_h - h)

            expanded = Image.new("RGB", (new_w, new_h),
                                  (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)))
            expanded.paste(image, (left, top))
            image = expanded

            boxes = boxes.copy()
            boxes[:, 0] += left
            boxes[:, 1] += top
            boxes[:, 2] += left
            boxes[:, 3] += top

        # Random crop (IoU-aware)
        if random.random() > 0.5:
            image, boxes = self._random_crop(image, boxes)

        return image, boxes

    def _random_crop(self, image, boxes):
        """Random crop ensuring at least one box center remains."""
        w, h = image.size
        if len(boxes) == 0:
            return image, boxes

        for _ in range(50):  # Max attempts
            min_scale = 0.5
            scale = random.uniform(min_scale, 1.0)
            crop_h = int(h * scale)
            crop_w = int(w * scale)

            left = random.randint(0, max(w - crop_w, 0))
            top = random.randint(0, max(h - crop_h, 0))
            right = left + crop_w
            bottom = top + crop_h

            # Check if any box center is inside crop
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            mask = (cx >= left) & (cx <= right) & (cy >= top) & (cy <= bottom)

            if not mask.any():
                continue

            image = image.crop((left, top, right, bottom))

            # Adjust boxes
            new_boxes = boxes[mask].copy()
            new_boxes[:, 0] = np.clip(new_boxes[:, 0] - left, 0, crop_w)
            new_boxes[:, 1] = np.clip(new_boxes[:, 1] - top, 0, crop_h)
            new_boxes[:, 2] = np.clip(new_boxes[:, 2] - left, 0, crop_w)
            new_boxes[:, 3] = np.clip(new_boxes[:, 3] - top, 0, crop_h)

            # Filter out boxes that are too small
            valid = (new_boxes[:, 2] - new_boxes[:, 0] > 5) & \
                    (new_boxes[:, 3] - new_boxes[:, 1] > 5)
            if valid.any():
                return image, new_boxes[valid]

        return image, boxes

    def _resize(self, image, boxes, target_size):
        """Resize image and scale boxes accordingly."""
        orig_w, orig_h = image.size

        image = image.resize((target_size, target_size), Image.BILINEAR)

        if len(boxes) > 0:
            scale_x = target_size / orig_w
            scale_y = target_size / orig_h
            boxes = boxes.copy()
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y

        return image, boxes


def coco_collate(batch):
    """Custom collate function for COCO detection.

    Handles variable number of boxes per image.
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


def build_coco_datasets(
    data_dir="./data/coco",
    img_size=512,
    train_img_dir=None,
    train_ann_file=None,
    val_img_dir=None,
    val_ann_file=None,
):
    """Build train and validation COCO datasets.

    Args:
        data_dir: Root directory for COCO dataset.
        img_size: Input image size.
        train_img_dir: Custom path to train images.
        train_ann_file: Custom path to train annotations.
        val_img_dir: Custom path to val images.
        val_ann_file: Custom path to val annotations.

    Returns:
        train_dataset, val_dataset
    """
    if train_img_dir is None:
        train_img_dir = os.path.join(data_dir, "train2017")
    if train_ann_file is None:
        train_ann_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
    if val_img_dir is None:
        val_img_dir = os.path.join(data_dir, "val2017")
    if val_ann_file is None:
        val_ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")

    train_dataset = COCODetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        img_size=img_size,
        augment=True,
    )

    val_dataset = COCODetectionDataset(
        img_dir=val_img_dir,
        ann_file=val_ann_file,
        img_size=img_size,
        augment=False,
    )

    return train_dataset, val_dataset
