#
# BDD100K Dataset for Object Detection
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

BDD100K_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

BDD100K_CAT_IDS = list(range(1, 11))
BDD100K_CAT_TO_IDX = {cat_id: cat_id for cat_id in BDD100K_CAT_IDS}


def filter_and_clip_boxes(boxes, img_w, img_h, labels, masks, iscrowd):
    """Clip boxes to image size, and filter out invalid boxes (width <= 0 or height <= 0).

    Also filters corresponding labels, masks, and iscrowd arrays.
    """
    if len(boxes) == 0:
        return boxes, labels, masks, iscrowd

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
    if len(masks) > 0:
        masks = masks[valid_mask]
    iscrowd = iscrowd[valid_mask]

    return boxes, labels, masks, iscrowd


class BDD100KDetectionDataset(Dataset):
    """BDD100K Detection Dataset using pure Python JSON parser.

    Annotations are converted from COCO JSON to [x1, y1, x2, y2] format.
    """

    def __init__(
        self,
        img_dir,
        ann_file,
        img_size=512,
        augment=True,
        cache_ram=False,
    ):
        """
        Args:
            img_dir: Directory containing BDD100K images.
            ann_file: Path to BDD100K COCO-formatted JSON annotations.
            img_size: Target image size (square).
            augment: Apply data augmentation.
            cache_ram: Preload dataset to RAM.
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment
        self.load_masks = False

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        print(f"Loading BDD100K annotations from {ann_file}...")
        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Build lookup tables
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            if ann["category_id"] in BDD100K_CAT_TO_IDX:
                self.img_to_anns[ann["image_id"]].append(ann)

        self.img_ids = list(self.images.keys())
        print(
            f"BDD100K Dataset: {len(self.img_ids)} images, {len(coco_data['annotations'])} annotations loaded"
        )

        self.cache_ram = cache_ram
        if self.cache_ram:
            print(f"Caching BDD100K dataset (augment={augment}) to RAM...")
            self.cached_images = []
            self.cached_annotations = []
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    self.img_ids,
                    desc=f"Caching BDD100K {'train' if augment else 'val'} to RAM",
                )
            except ImportError:
                pbar = self.img_ids

            for img_id in pbar:
                img_info = self.images[img_id]
                file_name = img_info["file_name"]
                img_path = os.path.join(self.img_dir, file_name)
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                self.cached_images.append(img_bytes)

                boxes, labels, iscrowd, masks = self._get_annotations(
                    img_id, img_h=img_info["height"], img_w=img_info["width"]
                )
                self.cached_annotations.append((boxes, labels, iscrowd, masks))

    def __len__(self):
        return len(self.img_ids)

    def _get_annotations(self, img_id, img_h, img_w):
        """Retrieve annotations for a given image ID.

        Returns:
            boxes:     list of [x1, y1, x2, y2]
            labels:    list of class indices (1-indexed)
            iscrowd:   list of ints (0 or 1)
            masks:     list of empty binary np.ndarray (H, W) uint8 placeholder
        """
        anns = self.img_to_anns[img_id]
        boxes = []
        labels = []
        iscrowd = []
        masks = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in BDD100K_CAT_TO_IDX:
                continue

            # Convert bbox [x, y, w, h] → [x1, y1, x2, y2]
            bbox = ann["bbox"]
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = x1 + float(bbox[2])
            y2 = y1 + float(bbox[3])

            # Clip bbox to image size
            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))

            # Ensure coordinates are sorted
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Check if valid (width > 0 and height > 0)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(BDD100K_CAT_TO_IDX[cat_id])
                iscrowd.append(int(ann.get("iscrowd", 0)))
                # Bdd100k detection only uses bbox, so use a 1x1 placeholder
                masks.append(np.zeros((1, 1), dtype=np.uint8))

        return boxes, labels, iscrowd, masks

    def __getitem__(self, idx):
        if self.cache_ram:
            import io

            img_bytes = self.cached_images[idx]
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            boxes, labels, iscrowd, masks = self.cached_annotations[idx]
            orig_w, orig_h = image.size
        else:
            img_id = self.img_ids[idx]
            img_info = self.images[img_id]
            file_name = img_info["file_name"]
            img_path = os.path.join(self.img_dir, file_name)
            if not os.path.exists(img_path):
                # Fallback search
                base_name = os.path.basename(file_name)
                for folder in ["train", "val"]:
                    test_path = os.path.join(self.img_dir, folder, base_name)
                    if os.path.exists(test_path):
                        img_path = test_path
                        break

            # Load image
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            # Get annotations
            boxes, labels, iscrowd, masks = self._get_annotations(
                img_id, img_h=orig_h, img_w=orig_w
            )

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
            iscrowd = np.array([], dtype=np.int64)
            masks = np.zeros((0, orig_h, orig_w), dtype=np.uint8)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            iscrowd = np.array(iscrowd, dtype=np.int64)
            masks = np.stack(masks, axis=0)  # (N, H, W)

        if self.augment:
            easy_mask = iscrowd == 0
            train_boxes = (
                boxes[easy_mask]
                if easy_mask.any()
                else np.zeros((0, 4), dtype=np.float32)
            )
            train_labels = (
                labels[easy_mask] if easy_mask.any() else np.array([], dtype=np.int64)
            )
            train_masks = (
                masks[easy_mask]
                if easy_mask.any()
                else np.zeros((0, orig_h, orig_w), dtype=np.uint8)
            )
            train_iscrowd = (
                iscrowd[easy_mask] if easy_mask.any() else np.array([], dtype=np.int64)
            )

            if len(train_boxes) > 0:
                image, train_boxes, train_masks, train_labels, train_iscrowd = (
                    self._augment(
                        image, train_boxes, train_masks, train_labels, train_iscrowd
                    )
                )
                train_boxes, train_labels, train_masks, train_iscrowd = (
                    filter_and_clip_boxes(
                        train_boxes,
                        image.size[0],
                        image.size[1],
                        train_labels,
                        train_masks,
                        train_iscrowd,
                    )
                )

            image, train_boxes, train_masks = self._resize(
                image, train_boxes, self.img_size, train_masks
            )
            new_w, new_h = image.size
            train_boxes, train_labels, train_masks, train_iscrowd = (
                filter_and_clip_boxes(
                    train_boxes, new_w, new_h, train_labels, train_masks, train_iscrowd
                )
            )

            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)

            areas = (train_boxes[:, 2] - train_boxes[:, 0]) * (
                train_boxes[:, 3] - train_boxes[:, 1]
            )

            targets = {
                "boxes": torch.tensor(train_boxes, dtype=torch.float32),
                "labels": torch.tensor(train_labels, dtype=torch.int64),
                "masks": torch.tensor(train_masks, dtype=torch.bool),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(train_iscrowd, dtype=torch.int64),
            }
        else:
            image, boxes = self._resize(image, boxes, self.img_size)
            masks = np.zeros((len(boxes), 1, 1), dtype=np.uint8)
            new_w, new_h = image.size
            boxes, labels, masks, iscrowd = filter_and_clip_boxes(
                boxes, new_w, new_h, labels, masks, iscrowd
            )

            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            targets = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(masks, dtype=torch.bool),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
                "difficults": torch.tensor(iscrowd == 1, dtype=torch.bool),
            }

        return image, targets

    def _augment(self, image, boxes, masks, labels, iscrowd):
        w, h = image.size

        if random.random() > 0.5:
            image = TF.hflip(image)
            new_boxes = boxes.copy()
            new_boxes[:, 0] = w - boxes[:, 2]
            new_boxes[:, 2] = w - boxes[:, 0]
            boxes = new_boxes
            if len(masks) > 0:
                masks = masks[:, :, ::-1].copy()

        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))

        if random.random() > 0.5:
            ratio = random.uniform(1.0, 2.0)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            left = random.randint(0, new_w - w)
            top = random.randint(0, new_h - h)

            expanded = Image.new(
                "RGB",
                (new_w, new_h),
                (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)),
            )
            expanded.paste(image, (left, top))
            image = expanded

            boxes = boxes.copy()
            boxes[:, 0] += left
            boxes[:, 1] += top
            boxes[:, 2] += left
            boxes[:, 3] += top

            if len(masks) > 0:
                new_masks = np.zeros((masks.shape[0], new_h, new_w), dtype=masks.dtype)
                new_masks[:, top : top + h, left : left + w] = masks
                masks = new_masks

        if random.random() > 0.5:
            image, boxes, masks, labels, iscrowd = self._random_crop(
                image, boxes, masks, labels, iscrowd
            )

        return image, boxes, masks, labels, iscrowd

    def _random_crop(self, image, boxes, masks, labels, iscrowd):
        w, h = image.size
        if len(boxes) == 0:
            return image, boxes, masks, labels, iscrowd

        for _ in range(50):
            scale = random.uniform(0.5, 1.0)
            crop_h = int(h * scale)
            crop_w = int(w * scale)
            left = random.randint(0, max(w - crop_w, 0))
            top = random.randint(0, max(h - crop_h, 0))
            right = left + crop_w
            bottom = top + crop_h

            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            keep = (cx >= left) & (cx <= right) & (cy >= top) & (cy <= bottom)

            if not keep.any():
                continue

            new_boxes = boxes[keep].copy()
            new_boxes[:, 0] = np.clip(new_boxes[:, 0] - left, 0, crop_w)
            new_boxes[:, 1] = np.clip(new_boxes[:, 1] - top, 0, crop_h)
            new_boxes[:, 2] = np.clip(new_boxes[:, 2] - left, 0, crop_w)
            new_boxes[:, 3] = np.clip(new_boxes[:, 3] - top, 0, crop_h)

            new_masks = (
                masks[keep, top:bottom, left:right].copy()
                if len(masks) > 0
                else masks[keep]
            )

            new_labels = labels[keep]
            new_iscrowd = iscrowd[keep]

            valid = ((new_boxes[:, 2] - new_boxes[:, 0]) > 5) & (
                (new_boxes[:, 3] - new_boxes[:, 1]) > 5
            )
            if valid.any():
                cropped_image = image.crop((left, top, right, bottom))
                return (
                    cropped_image,
                    new_boxes[valid],
                    new_masks[valid],
                    new_labels[valid],
                    new_iscrowd[valid],
                )

        return image, boxes, masks, labels, iscrowd

    def _resize(self, image, boxes, target_size, masks=None, max_size=1333):
        orig_w, orig_h = image.size

        min_side = min(orig_w, orig_h)
        max_side = max(orig_w, orig_h)
        scale = target_size / min_side

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

        if masks is not None:
            if len(masks) > 0:
                resized = np.zeros((masks.shape[0], new_h, new_w), dtype=masks.dtype)
                for i, m in enumerate(masks):
                    pil_m = Image.fromarray(m).resize((new_w, new_h), Image.NEAREST)
                    resized[i] = np.array(pil_m, dtype=masks.dtype)
                masks = resized
            else:
                masks = np.zeros((0, new_h, new_w), dtype=masks.dtype)

        if masks is not None:
            return image, boxes, masks
        return image, boxes


def bdd100k_collate(batch):
    import torch.nn.functional as F

    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

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


def build_bdd100k_datasets(
    data_dir="./data/bdd100k",
    img_size=512,
    train_img_dir=None,
    train_ann_file=None,
    val_img_dir=None,
    val_ann_file=None,
    cache_ram=False,
):
    """Build train and validation BDD100K datasets."""
    if train_img_dir is None:
        train_img_dir = os.path.join(data_dir, "images", "100k", "train")
    if train_ann_file is None:
        train_ann_file = os.path.join(
            data_dir, "annotations", "bdd100k_det_train_coco.json"
        )
    if val_img_dir is None:
        val_img_dir = os.path.join(data_dir, "images", "100k", "val")
    if val_ann_file is None:
        val_ann_file = os.path.join(
            data_dir, "annotations", "bdd100k_det_val_coco.json"
        )

    train_dataset = BDD100KDetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        img_size=img_size,
        augment=True,
        cache_ram=cache_ram,
    )

    val_dataset = BDD100KDetectionDataset(
        img_dir=val_img_dir,
        ann_file=val_ann_file,
        img_size=img_size,
        augment=False,
        cache_ram=cache_ram,
    )

    return train_dataset, val_dataset
