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


def filter_and_clip_boxes(boxes, img_w, img_h, labels, difficults):
    """Clip boxes to image size, and filter out invalid boxes (width <= 0 or height <= 0).

    Also filters corresponding labels and difficults arrays.
    """
    if len(boxes) == 0:
        return boxes, labels, difficults

    # Clip coordinates to [0, img_w] and [0, img_h]
    boxes = boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

    # Filter out boxes with width <= 0 or height <= 0
    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    boxes = boxes[valid_mask]
    labels = labels[valid_mask]
    difficults = difficults[valid_mask]

    return boxes, labels, difficults


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
        cache_ram=False,
    ):
        """
        Args:
            root: root directory for VOC data
            years: tuple of VOC years to use
            image_sets: tuple of image set names (trainval, test, etc.)
            img_size: target image size (square)
            augment: apply data augmentation
            download: auto-download if not present
            cache_ram: preload dataset to RAM
        """
        self.root = root
        self.img_size = img_size
        self.augment = augment

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

        self.cache_ram = cache_ram
        if self.cache_ram:
            print(f"Caching VOC dataset (augment={augment}) to RAM...")
            self.cached_images = []
            self.cached_annotations = []
            try:
                from tqdm import tqdm
                pbar = tqdm(self.entries, desc=f"Caching VOC {'trainval' if augment else 'val'} to RAM")
            except ImportError:
                pbar = self.entries

            for img_path, ann_path in pbar:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                self.cached_images.append(img_bytes)

                boxes, labels, difficults = self._parse_annotation(ann_path)
                self.cached_annotations.append((boxes, labels, difficults))

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Collect all image entries already populated above in __init__ for caching
        pass

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

            # Ensure coordinates are sorted
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[name])
            difficults.append(is_difficult)

        return boxes, labels, difficults

    def __getitem__(self, idx):
        if self.cache_ram:
            import io
            img_bytes = self.cached_images[idx]
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            boxes, labels, difficults = self.cached_annotations[idx]
        else:
            img_path, ann_path = self.entries[idx]
            # Load image
            image = Image.open(img_path).convert("RGB")
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

            # Clip bbox to image size initially
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        if self.augment:
            # For training: exclude difficult objects from augmentation & targets
            easy_mask = ~difficults
            train_boxes = boxes[easy_mask] if easy_mask.any() else np.zeros((0, 4), dtype=np.float32)
            train_labels = labels[easy_mask] if easy_mask.any() else np.array([], dtype=np.int64)
            train_difficults = np.zeros(len(train_boxes), dtype=bool)

            if len(train_boxes) > 0:
                image, train_boxes, train_labels, train_difficults = self._augment(
                    image, train_boxes, train_labels, train_difficults
                )
                # Filter invalid boxes after _augment
                train_boxes, train_labels, train_difficults = filter_and_clip_boxes(
                    train_boxes, image.size[0], image.size[1], train_labels, train_difficults
                )

            image, train_boxes = self._resize(image, train_boxes, self.img_size)
            # Filter invalid boxes after _resize
            new_w, new_h = image.size
            train_boxes, train_labels, train_difficults = filter_and_clip_boxes(
                train_boxes, new_w, new_h, train_labels, train_difficults
            )

            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)

            targets = {
                "boxes": torch.tensor(train_boxes, dtype=torch.float32),
                "labels": torch.tensor(train_labels, dtype=torch.int64),
            }
        else:
            # For eval: keep all objects, pass difficult flag
            image, boxes = self._resize(image, boxes, self.img_size)
            # Filter invalid boxes after _resize
            new_w, new_h = image.size
            boxes, labels, difficults = filter_and_clip_boxes(
                boxes, new_w, new_h, labels, difficults
            )

            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)

            targets = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "difficults": torch.tensor(difficults, dtype=torch.bool),
            }

        return image, targets

    def _augment(self, image, boxes, labels, difficults):
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
            image, boxes, labels, difficults = self._random_crop(image, boxes, labels, difficults)

        return image, boxes, labels, difficults

    def _random_crop(self, image, boxes, labels, difficults):
        """Random crop ensuring at least one box center remains."""
        w, h = image.size
        if len(boxes) == 0:
            return image, boxes, labels, difficults

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

            # Adjust boxes
            new_boxes = boxes[mask].copy()
            new_boxes[:, 0] = np.clip(new_boxes[:, 0] - left, 0, crop_w)
            new_boxes[:, 1] = np.clip(new_boxes[:, 1] - top, 0, crop_h)
            new_boxes[:, 2] = np.clip(new_boxes[:, 2] - left, 0, crop_w)
            new_boxes[:, 3] = np.clip(new_boxes[:, 3] - top, 0, crop_h)

            new_labels = labels[mask]
            new_difficults = difficults[mask]

            # Filter out boxes that are too small
            valid = (new_boxes[:, 2] - new_boxes[:, 0] > 5) & \
                    (new_boxes[:, 3] - new_boxes[:, 1] > 5)
            if valid.any():
                cropped_image = image.crop((left, top, right, bottom))
                return cropped_image, new_boxes[valid], new_labels[valid], new_difficults[valid]

        return image, boxes, labels, difficults

    def _resize(self, image, boxes, target_size, max_size=1333):
        """Resize image preserving aspect ratio.

        Shortest side is scaled to *target_size* and the longest side is
        capped at *max_size* (default 1333), following the standard
        detection convention (800 / 1333).
        """
        orig_w, orig_h = image.size

        # Compute scale so that the shortest side == target_size
        min_side = min(orig_w, orig_h)
        max_side = max(orig_w, orig_h)
        scale = target_size / min_side

        # Cap so that the longest side does not exceed max_size
        if scale * max_side > max_size:
            scale = max_size / max_side

        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        image = image.resize((new_w, new_h), Image.BILINEAR)

        if len(boxes) > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = boxes.copy()
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y

        return image, boxes


def detection_collate(batch):
    """Custom collate function for detection.

    Handles variable number of boxes per image and variable image sizes
    (aspect-ratio preserving resize produces different-sized tensors).
    Images are zero-padded to the nearest multiple of 32.
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    # Pad images to same size (multiple of 32) within the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32

    padded = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
        padded.append(img)

    images = torch.stack(padded, dim=0)
    return images, targets


def build_voc_datasets(data_dir="./data", img_size=512, download=True, cache_ram=False):
    """Build train and validation VOC datasets.

    Train: VOC2007 trainval + VOC2012 trainval
    Val:   VOC2007 val

    Args:
        data_dir: root data directory
        img_size: input image size
        download: auto-download datasets
        cache_ram: preload dataset to RAM

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
        cache_ram=cache_ram,
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
        cache_ram=cache_ram,
    )

    return train_dataset, val_dataset
