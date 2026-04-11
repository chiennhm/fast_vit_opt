#
# Detection losses: Focal Loss + Smooth L1 + Anchor matching
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) tensor in [x1, y1, x2, y2] format
        boxes2: (M, 4) tensor in [x1, y1, x2, y2] format

    Returns:
        iou: (N, M) tensor of IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(
        min=0
    )
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in dense detection.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="sum"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, num_classes) raw logits
            targets: (N,) class indices, -1 for ignore
        """
        num_classes = inputs.shape[1]

        # Create one-hot targets
        # targets: 0 = background (all-zeros), 1..C = object classes
        # cls_preds has shape (N, num_classes) where index 0..C-1 maps to class 1..C
        target_one_hot = torch.zeros_like(inputs)
        # Only positive samples (target > 0) get a one-hot entry
        pos_mask = targets > 0
        pos_targets = (targets[pos_mask] - 1).long()  # Convert 1-indexed → 0-indexed
        target_one_hot[pos_mask] = F.one_hot(pos_targets, num_classes).to(target_one_hot.dtype)

        # valid_mask: include both background (0) and positive (>0), exclude ignored (-1)
        valid_mask = targets >= 0

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, target_one_hot, reduction="none"
        )

        p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)
        alpha_t = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        # Ignore samples with target == -1
        loss = loss[valid_mask]

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else loss.sum()
        return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for bounding box regression."""

    def __init__(self, beta=1.0 / 9.0, reduction="sum"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets, weights=None):
        """
        Args:
            inputs: (N, 4) predicted box deltas
            targets: (N, 4) target box deltas
            weights: (N,) optional per-sample weights
        """
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )
        if weights is not None:
            loss = loss * weights.unsqueeze(1)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else loss.sum()
        return loss


class AnchorGenerator:
    """Generate anchors for each FPN level."""

    def __init__(
        self,
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        scales=(1.0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)),
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.num_anchors = len(aspect_ratios) * len(scales)
        self._cache = {}

    def _generate_base_anchors(self, size):
        """Generate base anchors for a given size."""
        anchors = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                h = size * scale * math.sqrt(ratio)
                w = size * scale / math.sqrt(ratio)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def generate(self, feature_maps, image_size, device):
        """Generate anchors for all FPN levels.

        Args:
            feature_maps: list of (H, W) feature map sizes
            image_size: (H, W) of input image
            device: torch device

        Returns:
            anchors: (total_anchors, 4) in [x1, y1, x2, y2] format
        """
        cache_key = (tuple(feature_maps), tuple(image_size), str(device))
        if cache_key in self._cache:
            return self._cache[cache_key]

        all_anchors = []
        for idx, (fh, fw) in enumerate(feature_maps):
            stride_h = image_size[0] / fh
            stride_w = image_size[1] / fw

            base_anchors = self._generate_base_anchors(self.sizes[idx]).to(device)

            # Create grid
            shift_y = (torch.arange(fh, device=device, dtype=torch.float32) + 0.5) * stride_h
            shift_x = (torch.arange(fw, device=device, dtype=torch.float32) + 0.5) * stride_w
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack(
                [shift_x.reshape(-1), shift_y.reshape(-1),
                 shift_x.reshape(-1), shift_y.reshape(-1)],
                dim=1,
            )

            # Combine shifts with base anchors
            anchors = (
                shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
            ).reshape(-1, 4)
            all_anchors.append(anchors)

        result = torch.cat(all_anchors, dim=0)
        self._cache[cache_key] = result
        return result


def encode_boxes(gt_boxes, anchors):
    """Encode ground truth boxes relative to anchors.

    Args:
        gt_boxes: (N, 4) in [x1, y1, x2, y2]
        anchors: (N, 4) in [x1, y1, x2, y2]

    Returns:
        deltas: (N, 4) encoded as [tx, ty, tw, th]
    """
    # Convert to center form
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]

    g_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    g_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    g_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    g_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    tx = (g_cx - a_cx) / (a_w + 1e-7)
    ty = (g_cy - a_cy) / (a_h + 1e-7)
    tw = torch.log(g_w / (a_w + 1e-7) + 1e-7)
    th = torch.log(g_h / (a_h + 1e-7) + 1e-7)

    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(deltas, anchors):
    """Decode predicted box deltas relative to anchors.

    Args:
        deltas: (N, 4) encoded as [tx, ty, tw, th]
        anchors: (N, 4) in [x1, y1, x2, y2]

    Returns:
        boxes: (N, 4) in [x1, y1, x2, y2]
    """
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]

    pred_cx = deltas[:, 0] * a_w + a_cx
    pred_cy = deltas[:, 1] * a_h + a_cy
    pred_w = torch.exp(deltas[:, 2].clamp(max=math.log(1000.0))) * a_w
    pred_h = torch.exp(deltas[:, 3].clamp(max=math.log(1000.0))) * a_h

    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


class DetectionLoss(nn.Module):
    """Combined detection loss: Focal Loss + Smooth L1.

    Handles anchor-target matching internally.
    """

    def __init__(
        self,
        num_classes=20,
        pos_iou_thresh=0.5,
        neg_iou_thresh=0.4,
        alpha=0.25,
        gamma=2.0,
        box_loss_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.box_loss_weight = box_loss_weight

        # Background = class 0, object classes = 1..num_classes
        # But for focal loss we use num_classes (no explicit background)
        self.cls_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="sum")
        self.reg_loss = SmoothL1Loss(reduction="sum")

    def forward(self, cls_preds, reg_preds, anchors, targets):
        """
        Args:
            cls_preds: (B, total_anchors, num_classes)
            reg_preds: (B, total_anchors, 4)
            anchors: (total_anchors, 4)
            targets: list of dicts, each with 'boxes' (N, 4) and 'labels' (N,)

        Returns:
            loss_dict: dict with 'cls_loss' and 'reg_loss'
        """
        batch_size = cls_preds.shape[0]
        device = cls_preds.device

        all_cls_preds = []
        all_cls_targets = []
        all_reg_preds = []
        all_reg_targets = []
        total_pos = 0

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"]  # (N, 4)
            gt_labels = targets[b]["labels"]  # (N,)

            if len(gt_boxes) == 0:
                # No GT boxes: all anchors are negative
                cls_targets = torch.zeros(
                    anchors.shape[0], dtype=torch.long, device=device
                )
                reg_targets = torch.zeros(
                    anchors.shape[0], 4, dtype=torch.float32, device=device
                )
                all_cls_preds.append(cls_preds[b])
                all_cls_targets.append(cls_targets)
                continue

            # Compute IoU between anchors and GT boxes
            iou = box_iou(anchors, gt_boxes)  # (A, N)
            max_iou, max_idx = iou.max(dim=1)  # (A,)

            # Assign labels
            # -1 = ignore, 0 = background, 1..C = object classes
            cls_targets = torch.zeros(
                anchors.shape[0], dtype=torch.long, device=device
            )

            # Negative: IoU < neg_thresh
            cls_targets[max_iou < self.neg_iou_thresh] = 0

            # Ignore: between neg_thresh and pos_thresh
            ignore_mask = (max_iou >= self.neg_iou_thresh) & (
                max_iou < self.pos_iou_thresh
            )
            cls_targets[ignore_mask] = -1

            # Positive: IoU >= pos_thresh
            pos_mask = max_iou >= self.pos_iou_thresh
            cls_targets[pos_mask] = gt_labels[max_idx[pos_mask]]

            # Ensure every GT has at least one anchor
            gt_max_iou, gt_max_idx = iou.max(dim=0)  # (N,)
            for gt_i in range(len(gt_boxes)):
                anchor_i = gt_max_idx[gt_i]
                cls_targets[anchor_i] = gt_labels[gt_i]
                pos_mask[anchor_i] = True

            # Encode regression targets
            matched_gt_boxes = gt_boxes[max_idx]
            reg_targets = encode_boxes(matched_gt_boxes, anchors)

            num_pos = pos_mask.sum().item()
            total_pos += num_pos

            all_cls_preds.append(cls_preds[b])
            all_cls_targets.append(cls_targets)
            all_reg_preds.append(reg_preds[b][pos_mask])
            all_reg_targets.append(reg_targets[pos_mask])

        # Flatten and compute losses
        all_cls_preds = torch.cat(all_cls_preds, dim=0)
        all_cls_targets = torch.cat(all_cls_targets, dim=0)

        # For focal loss, background = class 0, objects = 1..C
        # Remap: 0 (bg) stays 0, classes 1..C stay
        cls_loss = self.cls_loss(all_cls_preds, all_cls_targets)

        if total_pos > 0:
            all_reg_preds = torch.cat(all_reg_preds, dim=0)
            all_reg_targets = torch.cat(all_reg_targets, dim=0)
            reg_loss = self.reg_loss(all_reg_preds, all_reg_targets)
        else:
            reg_loss = torch.tensor(0.0, device=device)

        # Normalize by number of positive samples
        normalizer = max(total_pos, 1)
        cls_loss = cls_loss / normalizer
        reg_loss = reg_loss / normalizer

        return {
            "cls_loss": cls_loss,
            "reg_loss": self.box_loss_weight * reg_loss,
            "num_pos": total_pos,
        }
