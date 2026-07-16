#
# FastViT Detector: FastViT backbone + FPN + RetinaNet Head
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from typing import Tuple, List, Union
from torchvision.ops import batched_nms

from timm.models import create_model
import models  # noqa: F401, registers FastViT variants

from .losses import AnchorGenerator, decode_boxes, BOX_WEIGHTS


class DWRep(nn.Module):
    """Depthwise Reparameterizable Block.
    During training:
        - 3x3 depthwise conv + BN
        - 1x1 depthwise conv + BN
        - BN (identity branch)
    During inference:
        - A single 3x3 depthwise conv with bias.
    """
    def __init__(self, channels: int, inference_mode: bool = False):
        super().__init__()
        self.channels = channels
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True
            )
        else:
            # 3x3 depthwise conv + BN
            self.rbr_conv = nn.Sequential()
            self.rbr_conv.add_module(
                "conv",
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
            )
            self.rbr_conv.add_module("bn", nn.BatchNorm2d(num_features=channels))

            # 1x1 depthwise conv + BN
            self.rbr_scale = nn.Sequential()
            self.rbr_scale.add_module(
                "conv",
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=channels, bias=False)
            )
            self.rbr_scale.add_module("bn", nn.BatchNorm2d(num_features=channels))

            # BN (identity branch)
            self.rbr_skip = nn.BatchNorm2d(num_features=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.reparam_conv(x)
        
        return self.rbr_conv(x) + self.rbr_scale(x) + self.rbr_skip(x)

    def reparameterize(self) -> None:
        if self.inference_mode:
            return
        
        kernel, bias = self._get_kernel_bias()
        reparam_conv = nn.Conv2d(
            self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True
        )
        reparam_conv = reparam_conv.to(device=kernel.device, dtype=kernel.dtype)
        
        with torch.no_grad():
            reparam_conv.weight.copy_(kernel)
            reparam_conv.bias.copy_(bias)
            
        self.reparam_conv = reparam_conv
        
        # Delete training branches
        for para in self.parameters():
            para.detach_()
        del self.rbr_conv
        del self.rbr_scale
        del self.rbr_skip
        
        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fuse skip (BatchNorm2d)
        kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
        
        # Fuse conv branch (3x3 depthwise)
        kernel_conv, bias_conv = self._fuse_bn_tensor(self.rbr_conv)
        
        # Fuse scale branch (1x1 depthwise)
        kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
        # Pad scale branch kernel from 1x1 to 3x3
        kernel_scale = F.pad(kernel_scale, [1, 1, 1, 1])
        
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch: Union[nn.Sequential, nn.BatchNorm2d]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            # Create identity kernel tensor for depthwise conv (groups = channels)
            kernel_value = torch.zeros(
                (self.channels, 1, 3, 3),
                dtype=branch.weight.dtype,
                device=branch.weight.device,
            )
            for i in range(self.channels):
                kernel_value[i, 0, 1, 1] = 1.0
            kernel = kernel_value
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class DWRepBlock(nn.Module):
    """
    C2 (or C3) -> Projection 1x1 -> DWRep -> PW 1x1 -> Residual -> L2 (or L3)
    
    Structure:
        x = input
        proj = Projection(x)
        dw = DWRep(proj)
        act1 = ReLU(dw)
        pw = PW(act1)
        out = proj + pw
        act2 = ReLU(out)
    """
    def __init__(self, in_channels: int, out_channels: int, inference_mode: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dw_rep = DWRep(out_channels, inference_mode=inference_mode)
        self.act1 = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.proj(x)
        dw = self.act1(self.dw_rep(proj))
        if hasattr(self, "pw_bn"):
            pw = self.pw_bn(self.pw_conv(dw))
        else:
            pw = self.pw_conv(dw)
        out = self.act2(proj + pw)
        return out

    def reparameterize(self) -> None:
        self.dw_rep.reparameterize()
        if hasattr(self, "pw_bn"):
            fused_conv = nn.Conv2d(
                self.pw_conv.in_channels,
                self.pw_conv.out_channels,
                kernel_size=self.pw_conv.kernel_size,
                stride=self.pw_conv.stride,
                padding=self.pw_conv.padding,
                dilation=self.pw_conv.dilation,
                groups=self.pw_conv.groups,
                bias=True,
            )
            # Ensure correct device and dtype
            fused_conv = fused_conv.to(device=self.pw_conv.weight.device, dtype=self.pw_conv.weight.dtype)
            
            w = self.pw_conv.weight
            b_conv = self.pw_conv.bias if self.pw_conv.bias is not None else torch.zeros(self.pw_conv.out_channels, device=w.device, dtype=w.dtype)
            mean = self.pw_bn.running_mean
            var = self.pw_bn.running_var
            gamma = self.pw_bn.weight
            beta = self.pw_bn.bias
            eps = self.pw_bn.eps
            
            std = (var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            
            w_fused = w * t
            b_fused = beta + (b_conv - mean) * (gamma / std)
            
            with torch.no_grad():
                fused_conv.weight.copy_(w_fused)
                fused_conv.bias.copy_(b_fused)
            
            self.pw_conv = fused_conv
            del self.pw_bn


class FPN(nn.Module):
    """Feature Pyramid Network.
    Takes multi-scale features from backbone and produces
    feature maps with unified channel dimensions.
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 256, inference_mode: bool = False):
        super().__init__()
        assert len(in_channels_list) == 4, "Expected 4 stages from backbone (C2, C3, C4, C5)"
        
        in_c2, in_c3, in_c4, in_c5 = in_channels_list
        
        # P2 / L2 block
        self.l2_block = DWRepBlock(in_c2, out_channels, inference_mode=inference_mode)
        
        # P3 / L3 block
        self.l3_block = DWRepBlock(in_c3, out_channels, inference_mode=inference_mode)
        
        # P4 / L4 lateral
        self.l4_lateral = nn.Conv2d(in_c4, out_channels, kernel_size=1)
        
        # P5 / L5 lateral
        self.l5_lateral = nn.Conv2d(in_c5, out_channels, kernel_size=1)
        
        # P6 conv (takes C5)
        self.p6_conv = nn.Conv2d(in_c5, out_channels, kernel_size=3, stride=2, padding=1)
        
        # Output 3x3 convs for P2, P3, P4, P5
        self.p2_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p3_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p4_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p5_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reparameterize(self) -> None:
        """Reparameterize FPN blocks."""
        self.l2_block.reparameterize()
        self.l3_block.reparameterize()

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"Expected 4 stage features from backbone, got {len(features)}")
        c2, c3, c4, c5 = features
        
        # Lateral / Block features
        l2 = self.l2_block(c2)
        l3 = self.l3_block(c3)
        l4 = self.l4_lateral(c4)
        l5 = self.l5_lateral(c5)
        
        # Top-down pathway
        lat5 = l5
        lat4 = l4 + F.interpolate(lat5, size=l4.shape[2:], mode="nearest")
        lat3 = l3 + F.interpolate(lat4, size=l3.shape[2:], mode="nearest")
        lat2 = l2 + F.interpolate(lat3, size=l2.shape[2:], mode="nearest")
        
        # FPN output convolutions
        p2 = self.p2_conv(lat2)
        p3 = self.p3_conv(lat3)
        p4 = self.p4_conv(lat4)
        p5 = self.p5_conv(lat5)
        p6 = self.p6_conv(c5)
        
        return [p2, p3, p4, p5, p6]


class RetinaNetHead(nn.Module):
    """RetinaNet classification and regression heads.
    Shared convolutional heads applied to each FPN level.
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 20, num_anchors: int = 9, num_convs: int = 4, reg_max: int = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reg_max = reg_max

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
        reg_channels = num_anchors * 4 * (reg_max + 1) if reg_max is not None else num_anchors * 4
        self.reg_pred = nn.Conv2d(
            in_channels, reg_channels, kernel_size=3, padding=1
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for modules in [self.cls_subnet, self.reg_subnet]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Initialize cls_score bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)
        nn.init.normal_(self.cls_score.weight, std=0.01)

        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)

    def forward(self, fpn_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_preds = []
        reg_preds = []

        for feature in fpn_features:
            B, C, H, W = feature.shape

            cls_out = self.cls_subnet(feature)
            cls_out = self.cls_score(cls_out)
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            cls_preds.append(cls_out)

            reg_out = self.reg_subnet(feature)
            reg_out = self.reg_pred(reg_out)
            reg_channels = 4 * (self.reg_max + 1) if self.reg_max is not None else 4
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, -1, reg_channels)
            reg_preds.append(reg_out)

        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)

        return cls_preds, reg_preds


class FastViTDetector(nn.Module):
    """Object detector using FastViT backbone + FPN + RetinaNet head.
    """
    VOC_CLASSES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

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
        model_name: str = "fastvit_sa12",
        num_classes: int = 20,
        fpn_channels: int = 256,
        pretrained_backbone: str = None,
        anchor_sizes: Tuple[int, ...] = (16, 32, 64, 128, 256),
        anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        anchor_scales: Tuple[float, ...] = (1.0, 2 ** (1.0 / 3), 2 ** (2.0 / 3)),
        inference_mode: bool = False,
        reg_max: int = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.reg_max = reg_max

        embed_dims = self.EMBED_DIMS[model_name]

        # FPN now has 5 levels (P2, P3, P4, P5, P6)
        assert len(anchor_sizes) == len(embed_dims) + 1, (
            f"anchor_sizes ({len(anchor_sizes)}) must match FPN levels ({len(embed_dims) + 1})"
        )

        self.backbone = create_model(
            model_name,
            fork_feat=True,
        )

        if pretrained_backbone is not None:
            checkpoint = torch.load(pretrained_backbone, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model_dict = self.backbone.state_dict()
            filtered = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            self.backbone.load_state_dict(filtered, strict=False)
            print(
                f"Loaded {len(filtered)}/{len(model_dict)} "
                f"keys from pretrained backbone"
            )

        # FPN
        self.fpn = FPN(
            in_channels_list=embed_dims,
            out_channels=fpn_channels,
            inference_mode=inference_mode,
        )

        # Detection Head
        num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.head = RetinaNetHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            reg_max=reg_max,
        )

        # Anchor Generator
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=anchor_ratios,
            scales=anchor_scales,
        )
        self.num_anchors = num_anchors

    def reparameterize(self) -> None:
        """Reparameterize backbone and FPN blocks."""
        from models.modules.mobileone import reparameterize_model
        
        # Reparameterize backbone
        self.backbone = reparameterize_model(self.backbone)
        
        # Reparameterize FPN
        if hasattr(self.fpn, "reparameterize"):
            self.fpn.reparameterize()

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @torch.inference_mode()
    def predict(
        self,
        images: torch.Tensor,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_detections: int = 200,
    ) -> List[dict]:
        cls_preds, reg_preds, anchors = self.forward(images)

        batch_size = cls_preds.shape[0]
        img_h, img_w = images.shape[2], images.shape[3]
        results = []

        # Check if regression head outputs distribution bins
        num_reg_channels = reg_preds.shape[-1]
        use_dfl = num_reg_channels > 4
        
        if use_dfl:
            reg_max = num_reg_channels // 4 - 1
            B, A, _ = reg_preds.shape
            reg_dist = reg_preds.reshape(B, A, 4, reg_max + 1)
            prob = F.softmax(reg_dist, dim=-1)
            weights = torch.arange(reg_max + 1, dtype=prob.dtype, device=prob.device)
            pred_offsets = torch.sum(prob * weights, dim=-1)  # (B, A, 4)
        else:
            pred_offsets = reg_preds  # (B, A, 4)

        for b in range(batch_size):
            scores = torch.sigmoid(cls_preds[b])  # (A, C)
            box_deltas = pred_offsets[b]  # (A, 4)

            max_scores, _ = scores.max(dim=1)  # (A,)

            pre_nms_top_n = min(2000, max_scores.numel())
            _, topk_anchors_idx = max_scores.topk(pre_nms_top_n)

            candidate_mask = max_scores > score_thresh
            topk_mask = torch.zeros_like(candidate_mask)
            topk_mask[topk_anchors_idx] = True
            candidate_mask = candidate_mask & topk_mask

            if not candidate_mask.any():
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=images.device),
                        "scores": torch.zeros((0,), device=images.device),
                        "labels": torch.zeros(
                            (0,), dtype=torch.long, device=images.device
                        ),
                    }
                )
                continue

            scores = scores[candidate_mask]  # (K, C)
            if use_dfl:
                a_cx = (cand_anchors[:, 0] + cand_anchors[:, 2]) / 2
                a_cy = (cand_anchors[:, 1] + cand_anchors[:, 3]) / 2
                a_w = (cand_anchors[:, 2] - cand_anchors[:, 0]).clamp(min=1e-7)
                a_h = (cand_anchors[:, 3] - cand_anchors[:, 1]).clamp(min=1e-7)
                l, t, r, b_ = box_deltas.unbind(-1)
                boxes = torch.stack([
                    a_cx - l * a_w,
                    a_cy - t * a_h,
                    a_cx + r * a_w,
                    a_cy + b_ * a_h
                ], dim=1)
            else:
                boxes = decode_boxes(box_deltas, cand_anchors, weights=BOX_WEIGHTS)

            boxes[:, 0].clamp_(min=0)
            boxes[:, 1].clamp_(min=0)
            boxes[:, 2].clamp_(max=img_w)
            boxes[:, 3].clamp_(max=img_h)

            K, C = scores.shape
            flat_boxes = boxes.unsqueeze(1).expand(K, C, 4).reshape(-1, 4)  # (K*C, 4)
            flat_scores = scores.reshape(-1)  # (K*C,)
            flat_labels = (
                torch.arange(C, device=scores.device)
                .unsqueeze(0)
                .expand(K, C)
                .reshape(-1)
                + 1
            )  # 1-indexed

            keep = flat_scores > score_thresh
            flat_boxes = flat_boxes[keep]
            flat_scores = flat_scores[keep]
            flat_labels = flat_labels[keep]

            if len(flat_scores) == 0:
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=images.device),
                        "scores": torch.zeros((0,), device=images.device),
                        "labels": torch.zeros(
                            (0,), dtype=torch.long, device=images.device
                        ),
                    }
                )
                continue

            keep_idx = batched_nms(flat_boxes, flat_scores, flat_labels, nms_thresh)

            if len(keep_idx) > max_detections:
                keep_idx = keep_idx[:max_detections]

            results.append(
                {
                    "boxes": flat_boxes[keep_idx],
                    "scores": flat_scores[keep_idx],
                    "labels": flat_labels[keep_idx],
                }
            )

        return results
