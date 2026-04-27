#
# Centralized configuration for FastViT Object Detection
#
# All hyperparameters in one place for easy tuning.
# Import and use throughout the project:
#
#   from config import cfg
#   model = FastViTDetector(model_name=cfg.MODEL_NAME, ...)
#

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# ============================================================================
# Dataset
# ============================================================================
@dataclass
class DataConfig:
    """PASCAL VOC dataset settings."""

    root: str = "./data"
    """Root directory for VOC data."""

    years: Tuple[str, ...] = ("2007", "2012")
    """VOC years to include in training."""

    train_sets: Tuple[str, ...] = ("trainval",)
    """Image sets used for training."""

    val_years: Tuple[str, ...] = ("2007",)
    """VOC years for validation."""

    val_sets: Tuple[str, ...] = ("val",)
    """Image sets used for validation."""

    download: bool = True
    """Auto-download VOC if not present."""

    img_size: int = 512
    """Input image size (square resize)."""

    # ImageNet normalization
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    num_classes: int = 20
    """Number of object classes (VOC = 20)."""


# ============================================================================
# Augmentation
# ============================================================================
@dataclass
class AugConfig:
    """Data augmentation settings (training only)."""

    horizontal_flip: float = 0.5
    """Probability of horizontal flip."""

    brightness_range: Tuple[float, float] = (0.8, 1.2)
    """Random brightness adjustment range."""

    contrast_range: Tuple[float, float] = (0.8, 1.2)
    """Random contrast adjustment range."""

    saturation_range: Tuple[float, float] = (0.8, 1.2)
    """Random saturation adjustment range."""

    hue_range: Tuple[float, float] = (-0.05, 0.05)
    """Random hue adjustment range."""

    expand_ratio: Tuple[float, float] = (1.0, 2.0)
    """Random expand (zoom-out) ratio range."""

    expand_prob: float = 0.5
    """Probability of applying expand."""

    crop_prob: float = 0.5
    """Probability of applying random crop."""

    crop_min_scale: float = 0.5
    """Minimum crop scale (relative to image)."""

    crop_max_attempts: int = 50
    """Maximum attempts for IoU-aware random crop."""

    min_box_size: int = 5
    """Minimum box side length (pixels) after crop — smaller boxes are discarded."""


# ============================================================================
# Model / Architecture
# ============================================================================
@dataclass
class ModelConfig:
    """FastViT detector architecture settings."""

    backbone: str = "fastvit_sa12"
    """FastViT variant: fastvit_t8, fastvit_t12, fastvit_s12,
    fastvit_sa12, fastvit_sa24, fastvit_sa36, fastvit_ma36."""

    pretrained_backbone: Optional[str] = None
    """Path to pretrained backbone weights (None = train from scratch)."""

    fpn_channels: int = 256
    """Feature Pyramid Network output channels."""

    num_head_convs: int = 4
    """Number of conv layers in cls/reg heads."""

    # --- Anchor configuration ---
    anchor_sizes: Tuple[int, ...] = (32, 64, 128, 256)
    """Base anchor size per FPN level (one per level)."""

    anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    """Aspect ratios (w/h) for each anchor."""

    anchor_scales: Tuple[float, ...] = (1.0, 1.2599, 1.5874)
    """Sub-scales within each level (default: 2^(0/3), 2^(1/3), 2^(2/3))."""


# ============================================================================
# Loss
# ============================================================================
@dataclass
class LossConfig:
    """Detection loss settings."""

    # Focal loss
    focal_alpha: float = 0.25
    """Focal loss alpha (foreground weight)."""

    focal_gamma: float = 2.0
    """Focal loss gamma (hard-example mining exponent)."""

    # Box regression
    smooth_l1_beta: float = 1.0 / 9.0
    """Smooth L1 beta threshold."""

    box_loss_weight: float = 1.0
    """Weight for bounding box regression loss relative to classification loss."""

    box_weights: Tuple[float, ...] = (10.0, 10.0, 5.0, 5.0)
    """Normalization weights for box deltas [tx, ty, tw, th]."""

    # Anchor matching
    pos_iou_thresh: float = 0.5
    """IoU threshold for positive anchor matching."""

    neg_iou_thresh: float = 0.4
    """IoU threshold for negative anchor matching.
    Anchors with IoU between neg and pos thresholds are ignored."""


# ============================================================================
# Optimizer
# ============================================================================
@dataclass
class OptimizerConfig:
    """Optimizer and learning rate settings."""

    optimizer: str = "adamw"
    """Optimizer type: adamw, sgd."""

    lr: float = 1e-4
    """Base learning rate for detection head."""

    backbone_lr_scale: float = 0.1
    """Backbone LR = lr * backbone_lr_scale."""

    weight_decay: float = 0.05
    """Weight decay (applied to conv/linear weights only)."""

    betas: Tuple[float, float] = (0.9, 0.999)
    """Adam betas."""

    momentum: float = 0.9
    """SGD momentum (only used if optimizer='sgd')."""


# ============================================================================
# Scheduler
# ============================================================================
@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""

    warmup_epochs: int = 5
    """Linear warmup duration."""

    min_lr_ratio: float = 0.01
    """Minimum LR as a fraction of base LR (for cosine annealing)."""


# ============================================================================
# Training
# ============================================================================
@dataclass
class TrainConfig:
    """Training loop settings."""

    epochs: int = 50
    """Total training epochs."""

    batch_size: int = 16
    """Training batch size."""

    val_batch_size: Optional[int] = None
    """Validation batch size (None = same as batch_size)."""

    workers: int = 4
    """DataLoader num_workers."""

    seed: int = 42
    """Random seed for reproducibility."""

    # AMP
    amp: bool = True
    """Use automatic mixed-precision (FP16) training."""

    # Gradient clipping
    clip_grad: float = 5.0
    """Max gradient norm (0 = disabled)."""

    # Logging
    log_interval: int = 20
    """Console log every N batches."""

    # Evaluation
    eval_interval: int = 5
    """Evaluate every N epochs."""

    # Checkpoints
    output_dir: str = "./output/detection"
    """Root output directory."""

    save_visualizations: bool = False
    """Save detection visualizations during eval."""

    # Resume
    resume: Optional[str] = None
    """Path to checkpoint to resume from."""


# ============================================================================
# Inference / Post-processing
# ============================================================================
@dataclass
class InferenceConfig:
    """Inference / post-processing settings."""

    score_thresh: float = 0.05
    """Minimum confidence score to keep a detection."""

    nms_thresh: float = 0.5
    """NMS IoU threshold."""

    max_detections: int = 200
    """Maximum detections per image."""


# ============================================================================
# K-Means++ Anchor Computation
# ============================================================================
@dataclass
class AnchorKMeansConfig:
    """Settings for compute_anchors.py."""

    num_anchors: int = 12
    """Total number of anchor clusters."""

    num_levels: int = 4
    """Number of FPN levels to distribute anchors across."""

    distance: str = "iou"
    """Distance metric: 'iou' or 'euclidean'."""

    max_iter: int = 300
    """Max K-Means iterations per trial."""

    num_trials: int = 10
    """Number of independent K-Means trials (best is kept)."""


# ============================================================================
# Wandb
# ============================================================================
@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    enabled: bool = True
    """Enable wandb logging (auto-disabled if wandb not installed)."""

    project: str = "fastvit-detection"
    """Wandb project name."""

    entity: Optional[str] = None
    """Wandb team/entity (None = personal)."""

    name: Optional[str] = None
    """Run name (None = auto-generated)."""


# ============================================================================
# Master Config
# ============================================================================
@dataclass
class DetectionConfig:
    """Master configuration — contains all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    anchor_kmeans: AnchorKMeansConfig = field(default_factory=AnchorKMeansConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def summary(self) -> str:
        """Return a human-readable summary of key settings."""
        lines = [
            "=" * 60,
            "Detection Config Summary",
            "=" * 60,
            f"  Backbone:       {self.model.backbone}",
            f"  Image size:     {self.data.img_size}",
            f"  Num classes:    {self.data.num_classes}",
            f"  FPN channels:   {self.model.fpn_channels}",
            f"  Anchor sizes:   {self.model.anchor_sizes}",
            f"  Anchor ratios:  {self.model.anchor_ratios}",
            f"  Anchor scales:  {self.model.anchor_scales}",
            f"  Epochs:         {self.train.epochs}",
            f"  Batch size:     {self.train.batch_size}",
            f"  LR (head):      {self.optim.lr}",
            f"  LR (backbone):  {self.optim.lr * self.optim.backbone_lr_scale}",
            f"  Weight decay:   {self.optim.weight_decay}",
            f"  AMP:            {self.train.amp}",
            f"  Focal alpha:    {self.loss.focal_alpha}",
            f"  Focal gamma:    {self.loss.focal_gamma}",
            f"  Grad clip:      {self.train.clip_grad}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# Default global instance
# ============================================================================
cfg = DetectionConfig()
