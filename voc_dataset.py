#
# PASCAL VOC Dataset for Object Detection
#

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import random


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(VOC_CLASSES)}  # 1-indexed


class VOCDetectionDataset(Dataset):
    """PASCAL VOC Detection Dataset.

    Supports VOC2007 and VOC2012 with automatic download.
    Annotations are converted from XML to [x1, y1, x2, y2] format.
    """

    def __init__(
        self,
        root="./data",
        years=("2007", "2012"),
        image_sets=("trainval",),
        img_size=512,
        augment=True,
        download=True,
    ):
        """
        Args:
            root: root directory for VOC data
            years: tuple of VOC years to use
            image_sets: tuple of image set names (trainval, test, etc.)
            img_size: target image size (square)
            augment: apply data augmentation
            download: auto-download if not present
        """
        self.root = root
        self.img_size = img_size
        self.augment = augment

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Collect all image entries
        self.entries = []

        for year in years:
            for image_set in image_sets:
                try:
                    dataset = torchvision.datasets.VOCDetection(
                        root=root,
                        year=year,
                        image_set=image_set,
                        download=download,
                    )
                    for i in range(len(dataset)):
                        img_path = dataset.images[i]
                        ann_path = dataset.annotations[i]
                        self.entries.append((img_path, ann_path))
                except Exception as e:
                    print(f"Warning: Could not load VOC{year} {image_set}: {e}")

        print(f"VOC Dataset: {len(self.entries)} images loaded")

    def __len__(self):
        return len(self.entries)

    def _parse_annotation(self, ann_path):
        """Parse VOC XML annotation file.

        Returns:
            boxes: list of [x1, y1, x2, y2]
            labels: list of class indices (1-indexed)
            difficults: list of bools
        """
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []
        difficults = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_TO_IDX:
                continue

            diff_elem = obj.find("difficult")
            is_difficult = diff_elem is not None and int(diff_elem.text) == 1

            bndbox = obj.find("bndbox")
            x1 = float(bndbox.find("xmin").text) - 1  # VOC coords are 1-indexed
            y1 = float(bndbox.find("ymin").text) - 1
            x2 = float(bndbox.find("xmax").text) - 1
            y2 = float(bndbox.find("ymax").text) - 1

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[name])
            difficults.append(is_difficult)

        return boxes, labels, difficults

    def __getitem__(self, idx):
        img_path, ann_path = self.entries[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Parse annotation (includes difficult objects)
        boxes, labels, difficults = self._parse_annotation(ann_path)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
            difficults = np.array([], dtype=bool)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            difficults = np.array(difficults, dtype=bool)

        if self.augment:
            # For training: exclude difficult objects from augmentation & targets
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


def detection_collate(batch):
    """Custom collate function for detection.

    Handles variable number of boxes per image.
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


def build_voc_datasets(data_dir="./data", img_size=512, download=True):
    """Build train and validation VOC datasets.

    Train: VOC2007 trainval + VOC2012 trainval
    Val:   VOC2007 val

    Args:
        data_dir: root data directory
        img_size: input image size
        download: auto-download datasets

    Returns:
        train_dataset, val_dataset
    """
    train_dataset = VOCDetectionDataset(
        root=data_dir,
        years=("2007", "2012"),
        image_sets=("trainval",),
        img_size=img_size,
        augment=True,
        download=download,
    )

    # VOC2007 val is used for evaluation
    # (VOC2007 test annotations are not always auto-downloadable)
    val_dataset = VOCDetectionDataset(
        root=data_dir,
        years=("2007",),
        image_sets=("val",),
        img_size=img_size,
        augment=False,
        download=download,
    )

    return train_dataset, val_dataset
