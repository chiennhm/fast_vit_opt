#
# Mask R-CNN Detector: FastViT-SA12 backbone + FPN + torchvision Mask R-CNN heads
#
# Architecture:
#   FastViT-SA12 (ImageNet pretrained, fork_feat=True)
#     → 4 multi-scale feature maps [C1..C4]
#     → torchvision FeaturePyramidNetwork  [P2..P6]
#     → torchvision RPN
#     → torchvision RoI Align
#     → Box Head (2-layer FC) + Mask Head (4×Conv + upsample)
#
# Usage:
#   model = FastViTMaskRCNN(num_classes=21, pretrained_backbone="best.pth")
#   model.train()
#   loss_dict = model(images, targets)   # training
#   model.eval()
#   preds = model(images)                # inference → List[Dict]
#

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)

from timm.models import create_model
import models  # noqa: F401 — registers FastViT variants in timm


# ============================================================================
# FastViT backbone wrapper for torchvision FPN
# ============================================================================

class FastViTBackbone(nn.Module):
    """Wraps FastViT (fork_feat=True) to output an OrderedDict of feature maps.

    torchvision's FeaturePyramidNetwork expects a backbone that returns
    ``OrderedDict[str, Tensor]``.  FastViT with fork_feat=True returns a
    plain list of 4 tensors — this class bridges that gap.

    Feature map spatial resolutions (for 512×512 input, SA12/S12):
        '0'  → (B,  64, 128, 128)   stride 4
        '1'  → (B, 128,  64,  64)   stride 8
        '2'  → (B, 256,  32,  32)   stride 16
        '3'  → (B, 512,  16,  16)   stride 32

    Args:
        model_name: timm model name, e.g. ``'fastvit_sa12'``.
        inference_mode: If True, build the backbone with reparameterised
            (fused) convolutions so that a ``*_reparam.pth`` checkpoint can
            be loaded directly.  Default: ``False``.
    """

    # Channel dims per stage for SA12 / S12 / T12 variants
    OUT_CHANNELS = [64, 128, 256, 512]

    def __init__(
        self,
        model_name: str = "fastvit_sa12",
        inference_mode: bool = False,
    ):
        super().__init__()
        self.body = create_model(
            model_name,
            fork_feat=True,
            inference_mode=inference_mode,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: List[torch.Tensor] = self.body(x)
        return OrderedDict([
            ("0", features[0]),
            ("1", features[1]),
            ("2", features[2]),
            ("3", features[3]),
        ])


# ============================================================================
# Backbone + FPN combined (as expected by torchvision MaskRCNN)
# ============================================================================

class FastViTWithFPN(nn.Module):
    """FastViT backbone fused with a torchvision FeaturePyramidNetwork.

    The combined module satisfies the torchvision MaskRCNN backbone contract:
        - ``out_channels`` attribute (int)
        - ``forward(images) → OrderedDict[str, Tensor]``

    FPN outputs 5 levels: '0', '1', '2', '3' from backbone + '4' from MaxPool.
    """

    def __init__(
        self,
        model_name: str = "fastvit_sa12",
        fpn_out_channels: int = 256,
        inference_mode: bool = False,
    ):
        super().__init__()

        self.backbone = FastViTBackbone(model_name, inference_mode=inference_mode)

        in_channels_list = FastViTBackbone.OUT_CHANNELS  # [64, 128, 256, 512]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),  # adds P6 level
        )
        self.out_channels = fpn_out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        return fpn_features


# ============================================================================
# Main Mask R-CNN detector
# ============================================================================

class FastViTMaskRCNN(nn.Module):
    """Mask R-CNN with FastViT-SA12 (ImageNet pretrained) backbone.

    This is the public-facing class used by object_detection.py.

    It wraps ``torchvision.models.detection.MaskRCNN`` and exposes:
      - ``forward(images, targets)``  → loss_dict during training
      - ``forward(images)``           → List[Dict] during inference
      - ``predict(images, ...)``      → post-processed List[Dict] (eval mode)

    The ``predict()`` output format matches ``FastViTDetector.predict()``:
        [{'boxes': Tensor[N,4], 'labels': Tensor[N], 'scores': Tensor[N]}]

    Args:
        num_classes: Number of object classes **including background** (e.g. 21 for VOC).
        fpn_out_channels: FPN output channel count. Default: 256.
        pretrained_backbone: Path to ImageNet pretrained FastViT checkpoint. Default: None.
        min_size: Minimum image side for internal resizing. Default: 800.
        max_size: Maximum image side for internal resizing. Default: 1333.
        box_score_thresh: Minimum score for keeping a detection. Default: 0.05.
        box_nms_thresh: NMS IoU threshold. Default: 0.5.
        box_detections_per_img: Max detections per image. Default: 100.
    """

    def __init__(
        self,
        num_classes: int = 21,
        fpn_out_channels: int = 256,
        pretrained_backbone: Optional[str] = None,
        min_size: int = 800,
        max_size: int = 1333,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        rpn_pre_nms_top_n_train: int = 1000,
        rpn_post_nms_top_n_train: int = 1000,
        rpn_batch_size_per_image: int = 128,
        box_batch_size_per_image: int = 128,
    ):
        super().__init__()

        # ── Detect checkpoint format BEFORE building backbone ─────────────
        # A reparameterized checkpoint has fused conv keys (``reparam_conv``).
        # An inference-mode backbone must be built to match those key names.
        inference_mode = False
        if pretrained_backbone is not None:
            inference_mode = FastViTMaskRCNN._is_reparam_checkpoint(pretrained_backbone)
            if inference_mode:
                print(
                    "[MaskRCNN] Reparameterized checkpoint detected → "
                    "building backbone in inference_mode=True"
                )

        # ── Backbone + FPN ────────────────────────────────────────────────
        backbone_with_fpn = FastViTWithFPN(
            model_name="fastvit_sa12",
            fpn_out_channels=fpn_out_channels,
            inference_mode=inference_mode,
        )

        # ── Load ImageNet pretrained weights ──────────────────────────────
        if pretrained_backbone is not None:
            self._load_pretrained(backbone_with_fpn.backbone.body, pretrained_backbone)

        # ── RPN Anchor Generator ──────────────────────────────────────────
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        # ── RoI Align pool output sizes ───────────────────────────────────
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2,
        )

        # ── Assemble Mask R-CNN ───────────────────────────────────────────
        self.model = MaskRCNN(
            backbone=backbone_with_fpn,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler,
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_batch_size_per_image=rpn_batch_size_per_image,
            box_batch_size_per_image=box_batch_size_per_image,
        )

        # ── Xavier init for added layers (FPN, RPN, heads) ────────────────
        self._xavier_init_added_layers()

    def _xavier_init_added_layers(self) -> None:
        """Apply Xavier uniform init to all added layers (FPN, RPN, box/mask heads).

        The backbone is excluded — it uses pretrained ImageNet weights.
        Paper specifies Xavier init for all "added layers" on top of the backbone.
        """
        # Modules that belong to added layers (everything except backbone.backbone)
        added_modules = [
            self.model.backbone.fpn,     # FPN lateral + output convs
            self.model.rpn,              # RPN head
            self.model.roi_heads,        # Box head + Mask head
        ]
        for parent in added_modules:
            for m in parent.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────────
    # Weight loading helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_reparam_checkpoint(ckpt_path: str) -> bool:
        """Return True if the checkpoint was saved in reparameterized (inference) mode.

        Reparameterized checkpoints have fused convolution keys that contain
        the substring ``reparam_conv``, which is absent in training-mode ones.
        """
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = (
                ckpt.get("state_dict")
                or ckpt.get("model")
                or ckpt
            )
            return any("reparam_conv" in k for k in sd.keys())
        except Exception:
            return False

    @staticmethod
    def _load_pretrained(backbone_model: nn.Module, ckpt_path: str) -> None:
        """Load ImageNet pretrained weights into the FastViT body.

        Handles checkpoint formats:
          - Plain state_dict
          - {'state_dict': ...}
          - {'model': ...}

        Works with both training-mode and reparameterized checkpoints.
        Keys that don't exist in the backbone or have mismatched shapes are
        silently skipped (e.g. classifier head, different model variant).
        """
        print(f"[MaskRCNN] Loading pretrained backbone from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model_state = backbone_model.state_dict()
        total_keys  = len(model_state)

        # Exact match: key name + shape both agree
        exact = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        missing, unexpected = backbone_model.load_state_dict(exact, strict=False)

        # Diagnostic
        reparam_keys_ckpt  = sum(1 for k in state_dict  if "reparam_conv" in k)
        reparam_keys_model = sum(1 for k in model_state if "reparam_conv" in k)
        print(
            f"[MaskRCNN] Checkpoint: {len(state_dict)} keys "
            f"({'reparam' if reparam_keys_ckpt else 'training'} mode)"
        )
        print(
            f"[MaskRCNN] Backbone  : {total_keys} keys "
            f"({'reparam' if reparam_keys_model else 'training'} mode)"
        )
        print(
            f"[MaskRCNN] Loaded {len(exact)}/{total_keys} backbone keys "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
        if len(exact) < total_keys * 0.5:
            print(
                "[MaskRCNN] WARNING: Less than 50% of backbone keys loaded. "
                "Check that the checkpoint variant matches fastvit_sa12 "
                "and that reparam mode is consistent."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ):
        """Delegate to the internal torchvision MaskRCNN.

        Training:  ``forward(images, targets)`` → Dict[str, Tensor] (losses)
        Inference: ``forward(images)``          → List[Dict[str, Tensor]]

        Note: torchvision Mask R-CNN expects ``List[Tensor]`` for images,
        not a batched (B, C, H, W) tensor.  Conversion is handled here if
        a batched tensor is passed (for compatibility with existing code).
        """
        if isinstance(images, torch.Tensor):
            images = list(images.unbind(0))

        return self.model(images, targets)

    # ──────────────────────────────────────────────────────────────────────
    # Predict (eval mode, compatible with FastViTDetector.predict API)
    # ──────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict(
        self,
        images,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_detections: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run inference and return detections in the same format as FastViTDetector.

        Args:
            images: (B, C, H, W) batched tensor OR List[Tensor].
            score_thresh: Minimum confidence to keep. Default: 0.05.
            nms_thresh: NMS IoU threshold. Default: 0.5.
            max_detections: Max detections per image. Default: 100.

        Returns:
            List of dicts, one per image:
                {'boxes': (N, 4), 'labels': (N,), 'scores': (N,)}
        """
        # Note: torchvision MaskRCNN uses its own score/nms thresholds set
        # at construction time. We re-apply score filtering here for
        # consistency with the FastViTDetector API (caller can override).
        if isinstance(images, torch.Tensor):
            images = list(images.unbind(0))

        raw = self.model(images)  # List[Dict] with boxes, labels, scores, masks

        results = []
        for pred in raw:
            keep = pred["scores"] >= score_thresh
            results.append({
                "boxes":  pred["boxes"][keep],
                "labels": pred["labels"][keep],
                "scores": pred["scores"][keep],
            })

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Parameter groups for optimizer (backbone vs head, with/without decay)
    # ──────────────────────────────────────────────────────────────────────

    def get_param_groups(
        self,
        base_lr: float = 1e-4,
        backbone_lr_scale: float = 0.1,
        weight_decay: float = 0.05,
    ) -> List[Dict]:
        """Return optimizer parameter groups with separate LR for backbone vs head.

        Mirrors the grouping strategy in object_detection.py for FastViTDetector.
        """
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_backbone = "backbone" in name
            no_decay_key = any(
                k in name for k in ("bias", ".bn.", "norm", "layer_scale")
            )

            if is_backbone:
                (backbone_no_decay if no_decay_key else backbone_decay).append(param)
            else:
                (head_no_decay if no_decay_key else head_decay).append(param)

        return [
            {"params": backbone_decay,    "lr": base_lr * backbone_lr_scale, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": base_lr * backbone_lr_scale, "weight_decay": 0.0},
            {"params": head_decay,        "lr": base_lr,                     "weight_decay": weight_decay},
            {"params": head_no_decay,     "lr": base_lr,                     "weight_decay": 0.0},
        ]
