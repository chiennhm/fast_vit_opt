#
# Grad-CAM visualization for Object Detection
#

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Union


class GradCAM:
    """Grad-CAM for the FastViT BDD100K detector."""

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        if self.target_layer is None:
            self.target_layer = self._find_target_layer()

        self.hook_handles = []
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Find a suitable target layer (high-level feature conv in backbone or FPN)."""
        target = None
        target_name = ""

        # Priority 1: Look for backbone network blocks (e.g. stage 3/4 convs)
        if hasattr(self.model, "backbone"):
            for name, module in self.model.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Prefer convs in stage 3 or 4 of backbone ('network.4' or 'network.5')
                    if "network.4" in name or "network.5" in name or "stage" in name:
                        target = module
                        target_name = name
            if target is None:
                # Fall back to any Conv2d in backbone
                for name, module in self.model.backbone.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target = module
                        target_name = name

        # Priority 2: Look for FPN lateral/block convs
        if target is None and hasattr(self.model, "fpn"):
            for name, module in self.model.fpn.named_modules():
                if isinstance(module, nn.Conv2d) and "lateral" in name:
                    target = module
                    target_name = name

        # Priority 3: Fall back to Conv2d in model
        if target is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target = module
                    target_name = name

        if target is not None:
            print(f"[GradCAM] Using target layer: {target_name} ({target})")

        return target

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if not torch.is_grad_enabled():
                return

            if isinstance(output, (list, tuple)):
                act = output[-1]
            else:
                act = output
            self.activations = act

            def save_grad(grad):
                self.gradients = grad

            if hasattr(act, "requires_grad") and act.requires_grad:
                act.register_hook(save_grad)

        if self.target_layer is not None:
            self.hook_handles.append(
                self.target_layer.register_forward_hook(forward_hook)
            )

    def generate_cam_map(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate normalized 2D float CAM map (H, W) in [0, 1].

        Returns:
            cam: (H, W) float32 numpy array in [0, 1] range
        """
        self.activations = None
        self.gradients = None

        device = next(self.model.parameters()).device
        img_input = image_tensor.to(device).clone().detach()
        img_input.requires_grad_(True)

        self.model.zero_grad()

        # Explicitly enable grad tracking for Grad-CAM
        with torch.enable_grad():
            if hasattr(self.model, "forward"):
                outputs = self.model(img_input)

            # Compute target score for backward pass
            if isinstance(outputs, (list, tuple)):
                if len(outputs) == 3:
                    # FastViTDetector: (cls_preds, reg_preds, anchors)
                    cls_preds, reg_preds, anchors = outputs
                    scores = torch.sigmoid(cls_preds)  # (1, num_anchors, num_classes)
                    max_scores, _ = scores[0].max(dim=-1)
                    top_k = min(50, max_scores.numel())
                    target_score = max_scores.topk(top_k).values.sum()
                else:
                    target_score = sum(
                        o.sum() for o in outputs if isinstance(o, torch.Tensor)
                    )
            elif isinstance(outputs, dict):
                target_score = sum(
                    v.sum() for v in outputs.values() if isinstance(v, torch.Tensor)
                )
            else:
                target_score = outputs.sum()

            # Backward pass
            target_score.backward()

        H, W = image_tensor.shape[2], image_tensor.shape[3]

        if self.activations is None or self.gradients is None:
            return np.zeros((H, W), dtype=np.float32)

        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Global average pooling over spatial dimensions
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolate CAM map to input image dimensions
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_heatmap(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single image tensor (1, 3, H, W).

        Returns:
            heatmap_rgb: (H, W, 3) uint8 numpy array in RGB format
        """
        cam = self.generate_cam_map(image_tensor)
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        return heatmap_rgb

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


def overlay_gradcam(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    heatmap_rgb: np.ndarray,
    alpha: float = 0.5,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """Overlay Grad-CAM heatmap on input image.

    Args:
        image: PIL Image, numpy array (H,W,3) uint8, or normalized torch Tensor (3,H,W)
        heatmap_rgb: (H,W,3) uint8 numpy array
        alpha: heatmap blending weight (0=only image, 1=only heatmap)
        mean: normalization mean for denormalizing torch Tensor
        std: normalization std for denormalizing torch Tensor

    Returns:
        blended: PIL Image
    """
    if isinstance(image, torch.Tensor):
        img_tensor = image.cpu().clone()
        for c in range(3):
            img_tensor[c] = img_tensor[c] * std[c] + mean[c]
        img_np = (img_tensor * 255).clamp(0, 255).permute(1, 2, 0).byte().numpy()
    elif isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    h, w = img_np.shape[:2]
    if heatmap_rgb.shape[:2] != (h, w):
        heatmap_rgb = cv2.resize(heatmap_rgb, (w, h))

    blended = (
        img_np.astype(np.float32) * (1.0 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    ).astype(np.uint8)

    return Image.fromarray(blended)
