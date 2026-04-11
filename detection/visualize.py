#
# Detection visualization utilities
#

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


# VOC class names
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Color palette for each class (distinct colors)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
]


def draw_detections(
    image,
    boxes,
    labels,
    scores=None,
    class_names=None,
    score_thresh=0.3,
    line_width=2,
):
    """Draw bounding boxes on an image.

    Args:
        image: PIL Image or numpy array (H, W, 3) or torch tensor (3, H, W)
        boxes: (N, 4) bounding boxes [x1, y1, x2, y2]
        labels: (N,) class labels (1-indexed)
        scores: (N,) confidence scores (optional)
        class_names: list of class name strings
        score_thresh: minimum score to draw
        line_width: bbox line width

    Returns:
        PIL Image with drawn detections
    """
    if class_names is None:
        class_names = VOC_CLASSES

    if isinstance(image, torch.Tensor):
        image = image.cpu()
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
        # Denormalize if values are in [0, 1] or normalized
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    draw = ImageDraw.Draw(image)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i in range(len(boxes)):
        if scores is not None and scores[i] < score_thresh:
            continue

        box = boxes[i]
        label = int(labels[i])
        color = CLASS_COLORS[(label - 1) % len(CLASS_COLORS)]

        # Draw bbox
        x1, y1, x2, y2 = box
        for j in range(line_width):
            draw.rectangle(
                [x1 - j, y1 - j, x2 + j, y2 + j],
                outline=color,
            )

        # Label text
        cls_name = class_names[label - 1] if label - 1 < len(class_names) else f"cls_{label}"
        if scores is not None:
            text = f"{cls_name}: {scores[i]:.2f}"
        else:
            text = cls_name

        # Draw text background
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
            fill=color,
        )
        draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=font)

    return image


def save_detection_results(
    images,
    predictions,
    output_dir,
    class_names=None,
    score_thresh=0.3,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Save detection visualization results to disk.

    Args:
        images: (B, 3, H, W) tensor of normalized images
        predictions: list of dicts with 'boxes', 'labels', 'scores'
        output_dir: directory to save results
        class_names: class name list
        score_thresh: minimum score threshold
        mean: normalization mean
        std: normalization std
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (img, pred) in enumerate(zip(images, predictions)):
        # Denormalize image
        img = img.cpu().clone()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = (img * 255).clamp(0, 255).byte()

        result_img = draw_detections(
            img,
            pred["boxes"],
            pred["labels"],
            pred.get("scores", None),
            class_names=class_names,
            score_thresh=score_thresh,
        )
        result_img.save(os.path.join(output_dir, f"det_{i:04d}.jpg"))
