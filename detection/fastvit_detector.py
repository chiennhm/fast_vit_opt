#
# FastViT Detector: FastViT backbone + FPN + RetinaNet Head
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models import create_model
import models  # noqa: F401, registers FastViT variants

from .losses import AnchorGenerator, decode_boxes


class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes multi-scale features from backbone and produces
    feature maps with unified channel dimensions.
    """

    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: list of input channel counts from backbone stages
            out_channels: number of output channels for all FPN levels
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            lateral = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral)
            self.fpn_convs.append(fpn_conv)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: list of tensors from backbone stages (low to high level)

        Returns:
            fpn_features: list of tensors with unified channels
        """
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )

        # FPN output convolutions
        fpn_out = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        return fpn_out


class RetinaNetHead(nn.Module):
    """RetinaNet classification and regression heads.

    Shared convolutional heads applied to each FPN level.
    """

    def __init__(self, in_channels=256, num_classes=20, num_anchors=9, num_convs=4):
        """
        Args:
            in_channels: FPN output channels
            num_classes: number of object classes (excluding background)
            num_anchors: number of anchors per spatial location
            num_convs: number of conv layers in each head
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification subnet
        cls_layers = []
        for _ in range(num_convs):
            cls_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            cls_layers.append(nn.GroupNorm(32, in_channels))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        # Regression subnet
        reg_layers = []
        for _ in range(num_convs):
            reg_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            reg_layers.append(nn.GroupNorm(32, in_channels))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_subnet = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, padding=1
        )

        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_subnet, self.reg_subnet]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Initialize cls_score bias for focal loss
        # Prior probability of 0.01 for positive class
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)
        nn.init.normal_(self.cls_score.weight, std=0.01)

        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)

    def forward(self, fpn_features):
        """
        Args:
            fpn_features: list of FPN feature maps

        Returns:
            cls_preds: (B, total_anchors, num_classes)
            reg_preds: (B, total_anchors, 4)
        """
        cls_preds = []
        reg_preds = []

        for feature in fpn_features:
            B, C, H, W = feature.shape

            cls_out = self.cls_subnet(feature)
            cls_out = self.cls_score(cls_out)
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(
                B, -1, self.num_classes
            )
            cls_preds.append(cls_out)

            reg_out = self.reg_subnet(feature)
            reg_out = self.reg_pred(reg_out)
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, -1, 4)
            reg_preds.append(reg_out)

        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)

        return cls_preds, reg_preds


class FastViTDetector(nn.Module):
    """Object detector using FastViT backbone + FPN + RetinaNet head.

    Args:
        model_name: FastViT variant name (e.g., 'fastvit_sa12')
        num_classes: number of object classes (default: 20 for VOC)
        fpn_channels: FPN output channels (default: 256)
        pretrained_backbone: path to pretrained backbone weights (optional)
        anchor_sizes: anchor sizes per FPN level
        anchor_ratios: anchor aspect ratios
        anchor_scales: anchor sub-scales within each level
    """

    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    # Embed dims for each FastViT variant
    EMBED_DIMS = {
        "fastvit_t8": [48, 96, 192, 384],
        "fastvit_t12": [64, 128, 256, 512],
        "fastvit_s12": [64, 128, 256, 512],
        "fastvit_sa12": [64, 128, 256, 512],
        "fastvit_sa24": [64, 128, 256, 512],
        "fastvit_sa36": [64, 128, 256, 512],
        "fastvit_ma36": [76, 152, 304, 608],
    }

    def __init__(
        self,
        model_name="fastvit_sa12",
        num_classes=20,
        fpn_channels=256,
        pretrained_backbone=None,
        anchor_sizes=(32, 64, 128, 256),
        anchor_ratios=(0.5, 1.0, 2.0),
        anchor_scales=(1.0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        # Get embed dims for this variant
        embed_dims = self.EMBED_DIMS[model_name]

        # Create backbone with fork_feat=True for multi-scale features
        self.backbone = create_model(
            model_name,
            fork_feat=True,
            num_classes=num_classes,
        )

        # Load pretrained backbone if provided
        if pretrained_backbone is not None:
            checkpoint = torch.load(pretrained_backbone, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            # Scrub incompatible keys
            model_dict = self.backbone.state_dict()
            filtered = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            self.backbone.load_state_dict(filtered, strict=False)
            print(
                f"Loaded {len(filtered)}/{len(model_dict)} "
                f"keys from pretrained backbone"
            )

        # FPN
        self.fpn = FPN(in_channels_list=embed_dims, out_channels=fpn_channels)

        # Detection Head
        num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.head = RetinaNetHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

        # Anchor Generator
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_ratios,
            scales=anchor_scales,
        )
        self.num_anchors = num_anchors

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) input images

        Returns:
            cls_preds: (B, total_anchors, num_classes)
            reg_preds: (B, total_anchors, 4)
            anchors: (total_anchors, 4)
        """
        # Extract multi-scale features
        features = self.backbone(images)  # list of 4 feature tensors

        # FPN
        fpn_features = self.fpn(features)

        # Detection head
        cls_preds, reg_preds = self.head(fpn_features)

        # Generate anchors
        feature_sizes = [(f.shape[2], f.shape[3]) for f in fpn_features]
        image_size = (images.shape[2], images.shape[3])
        anchors = self.anchor_generator.generate(
            feature_sizes, image_size, images.device
        )

        return cls_preds, reg_preds, anchors

    @torch.no_grad()
    def predict(
        self,
        images,
        score_thresh=0.05,
        nms_thresh=0.5,
        max_detections=200,
    ):
        """Run inference and return post-processed detections.

        Args:
            images: (B, 3, H, W) normalized input images
            score_thresh: minimum score threshold
            nms_thresh: NMS IoU threshold
            max_detections: maximum detections per image

        Returns:
            results: list of dicts with 'boxes', 'labels', 'scores'
        """
        self.eval()
        cls_preds, reg_preds, anchors = self.forward(images)

        batch_size = cls_preds.shape[0]
        results = []

        for b in range(batch_size):
            scores = torch.sigmoid(cls_preds[b])  # (A, C)
            box_deltas = reg_preds[b]  # (A, 4)

            # Decode boxes
            boxes = decode_boxes(box_deltas, anchors)

            # Clip to image boundaries
            boxes[:, 0].clamp_(min=0)
            boxes[:, 1].clamp_(min=0)
            boxes[:, 2].clamp_(max=images.shape[3])
            boxes[:, 3].clamp_(max=images.shape[2])

            all_boxes = []
            all_scores = []
            all_labels = []

            for cls_idx in range(self.num_classes):
                cls_scores = scores[:, cls_idx]
                keep = cls_scores > score_thresh
                if not keep.any():
                    continue

                cls_scores = cls_scores[keep]
                cls_boxes = boxes[keep]

                # NMS
                from torchvision.ops import nms

                keep_idx = nms(cls_boxes, cls_scores, nms_thresh)
                all_boxes.append(cls_boxes[keep_idx])
                all_scores.append(cls_scores[keep_idx])
                all_labels.append(
                    torch.full(
                        (len(keep_idx),),
                        cls_idx + 1,  # 1-indexed labels
                        dtype=torch.long,
                        device=images.device,
                    )
                )

            if len(all_boxes) > 0:
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Keep top-k detections
                if len(all_scores) > max_detections:
                    topk = all_scores.topk(max_detections).indices
                    all_boxes = all_boxes[topk]
                    all_scores = all_scores[topk]
                    all_labels = all_labels[topk]
            else:
                all_boxes = torch.zeros((0, 4), device=images.device)
                all_scores = torch.zeros((0,), device=images.device)
                all_labels = torch.zeros((0,), dtype=torch.long, device=images.device)

            results.append(
                {
                    "boxes": all_boxes,
                    "scores": all_scores,
                    "labels": all_labels,
                }
            )

        return results
