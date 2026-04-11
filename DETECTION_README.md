# FastViT Object Detection trên PASCAL VOC

## Tổng quan

Module này implement pipeline **object detection** sử dụng **FastViT** làm backbone, train from scratch trên dataset **PASCAL VOC 2007+2012**.

### Kiến trúc tổng thể

```
Input Image (3 × 512 × 512)
        │
        ▼
┌─────────────────────────┐
│   FastViT Backbone      │  (fork_feat=True)
│   (Structural Reparam)  │
└─────────────────────────┘
        │
        ├── Stage 0: (B, 64, 128, 128)   stride=4
        ├── Stage 1: (B, 128, 64, 64)    stride=8
        ├── Stage 2: (B, 256, 32, 32)    stride=16
        └── Stage 3: (B, 512, 16, 16)    stride=32
        │
        ▼
┌─────────────────────────┐
│   FPN (Feature Pyramid  │  Lateral + Top-down
│   Network)              │  All levels → 256 channels
└─────────────────────────┘
        │
        ├── P0: (B, 256, 128, 128)
        ├── P1: (B, 256, 64, 64)
        ├── P2: (B, 256, 32, 32)
        └── P3: (B, 256, 16, 16)
        │
        ▼
┌─────────────────────────┐
│   RetinaNet Head        │  Shared conv head
│   (4 conv + GroupNorm)  │
└─────────────────────────┘
        │
        ├── Classification: (B, total_anchors, 20)
        └── Regression:     (B, total_anchors, 4)
        │
        ▼
┌─────────────────────────┐
│   Post-processing       │  Score threshold + NMS
└─────────────────────────┘
        │
        ▼
  Final Detections: boxes, labels, scores
```

---

## Cấu trúc thư mục

```
ml-fastvit/
├── models/                          # FastViT backbone (có sẵn)
│   ├── fastvit.py                   #   Các variant: T8, T12, S12, SA12, SA24, SA36, MA36
│   └── modules/
│       ├── mobileone.py             #   MobileOne blocks
│       └── replknet.py              #   ReparamLargeKernelConv
│
├── detection/                       # Module detection (MỚI)
│   ├── __init__.py
│   ├── fastvit_detector.py          #   FastViTDetector = Backbone + FPN + Head
│   ├── losses.py                    #   Focal Loss + Smooth L1 + Anchor matching
│   ├── eval_voc.py                  #   VOC mAP@0.5 evaluation
│   └── visualize.py                 #   Vẽ bounding boxes lên ảnh
│
├── voc_dataset.py                   # PASCAL VOC dataset + augmentation (MỚI)
├── object_detection.py              # Main training script (MỚI)
├── train_detection.sh               # Shell script chạy train (MỚI)
└── train_detection.bat              # Batch script chạy train (MỚI)
```

---

## Dataset: PASCAL VOC

### Thông tin chung

| Thuộc tính | Giá trị |
|-----------|---------|
| Số lớp | 20 |
| Train set | VOC 2007 trainval + VOC 2012 trainval (~16,551 ảnh) |
| Val set | VOC 2007 val (~2,510 ảnh) |
| Metric | mAP@0.5 (VOC-style) |
| Download | Tự động qua `torchvision.datasets.VOCDetection` |

### 20 Classes

```
aeroplane   bicycle     bird        boat        bottle
bus         car         cat         chair       cow
diningtable dog         horse       motorbike   person
pottedplant sheep       sofa        train       tvmonitor
```

### Annotation format

VOC sử dụng XML annotations, mỗi object gồm:
- `name`: tên lớp (ví dụ: "dog")
- `bndbox`: bounding box với `xmin, ymin, xmax, ymax` (1-indexed)
- `difficult`: flag đánh dấu object khó (bỏ qua khi train)

Code parse tại `voc_dataset.py → _parse_annotation()`:
```python
# Chuyển từ VOC XML → [x1, y1, x2, y2] (0-indexed)
x1 = float(xmin) - 1
y1 = float(ymin) - 1
x2 = float(xmax) - 1
y2 = float(ymax) - 1
```

### Data Augmentation

Các augmentation áp dụng khi training (detection-safe):

| Augmentation | Mô tả | Probability |
|-------------|--------|-------------|
| Horizontal Flip | Lật ngang ảnh + điều chỉnh bbox | 50% |
| Color Jitter | Brightness, contrast, saturation, hue | 50% mỗi loại |
| Random Expand | Zoom out (tỷ lệ 1.0–2.0), padding với mean color | 50% |
| Random Crop | Crop ngẫu nhiên, đảm bảo center của ≥1 box nằm trong crop | 50% |
| Resize | Resize về 512×512, scale bbox tương ứng | 100% |

> **Quan trọng**: Mọi augmentation đều điều chỉnh bounding boxes tương ứng để đảm bảo label chính xác.

---

## Kiến trúc Detection

### 1. FastViT Backbone

FastViT là hybrid vision transformer sử dụng **structural reparameterization**. Trong chế độ `fork_feat=True`, backbone xuất multi-scale features từ 4 stages:

| Variant | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Tổng params |
|---------|---------|---------|---------|---------|-------------|
| fastvit_t8 | 48 | 96 | 192 | 384 | ~4M |
| fastvit_s12 | 64 | 128 | 256 | 512 | ~9M |
| fastvit_sa12 | 64 | 128 | 256 | 512 | ~11M |
| fastvit_sa24 | 64 | 128 | 256 | 512 | ~21M |

**SA variants** (SA12, SA24, SA36) sử dụng **self-attention** ở stage cuối → tốt hơn cho detection nhờ global receptive field.

### 2. FPN (Feature Pyramid Network)

FPN kết hợp features từ các scales khác nhau:

```
Stage 3 (16×16) ──→ Lateral Conv ──→ P3 ──────────────────→ Output P3
                                       │
                                       ▼ Upsample 2×
Stage 2 (32×32) ──→ Lateral Conv ──→ (+) ──→ FPN Conv ──→ Output P2
                                       │
                                       ▼ Upsample 2×
Stage 1 (64×64) ──→ Lateral Conv ──→ (+) ──→ FPN Conv ──→ Output P1
                                       │
                                       ▼ Upsample 2×
Stage 0 (128×128) → Lateral Conv ──→ (+) ──→ FPN Conv ──→ Output P0
```

- Lateral conv: 1×1 conv, giảm channels về 256
- FPN conv: 3×3 conv, refine features sau merge
- Output: 4 feature maps, tất cả đều có 256 channels

### 3. RetinaNet Detection Head

Hai subnet chia sẻ kiến trúc song song:

```
                FPN Feature (256 channels)
                    │              │
            ┌───────┘              └───────┐
            ▼                              ▼
    Classification Subnet          Regression Subnet
    ┌─────────────────┐           ┌─────────────────┐
    │ 4× (Conv3×3 +   │           │ 4× (Conv3×3 +   │
    │  GroupNorm + ReLU)│           │  GroupNorm + ReLU)│
    └────────┬────────┘           └────────┬────────┘
             ▼                              ▼
    Conv3×3 → (A×20)              Conv3×3 → (A×4)
    Class scores                   Box deltas [tx,ty,tw,th]
```

- **A = 9 anchors** per location (3 ratios × 3 scales)
- Classification output: `num_anchors × 20` (20 VOC classes)
- Regression output: `num_anchors × 4` (box deltas)

### 4. Anchors

Mỗi vị trí trên feature map có **9 anchors**:

| Tham số | Giá trị |
|---------|---------|
| Base sizes | 32, 64, 128, 256 (per FPN level) |
| Aspect ratios | 0.5, 1.0, 2.0 |
| Scales | 1.0, 2^(1/3), 2^(2/3) |
| Tổng anchors (512×512 input) | ~190,000 |

---

## Loss Functions

### Focal Loss (Classification)

Giải quyết **extreme class imbalance** (background >> foreground):

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| α (alpha) | 0.25 | Cân bằng positive/negative |
| γ (gamma) | 2.0 | Giảm loss cho easy examples |

### Smooth L1 Loss (Box Regression)

Cho bounding box regression, robust hơn L2:

```
SmoothL1(x) = 0.5x²/β   nếu |x| < β
              |x| - 0.5β  nếu |x| ≥ β
```

### Anchor-Target Matching

| IoU Range | Assignment |
|-----------|-----------|
| IoU ≥ 0.5 | Positive (assigned to GT) |
| 0.4 ≤ IoU < 0.5 | Ignored (-1) |
| IoU < 0.4 | Negative (background) |
| Highest IoU per GT | Forced positive |

### Total Loss

```
Loss = Focal_cls_loss/N_pos + λ × SmoothL1_reg_loss/N_pos
```
- `N_pos`: số positive anchors (normalization)
- `λ = 1.0`: box loss weight

---

## Training Pipeline

### Hyperparameters mặc định

| Tham số | Giá trị |
|---------|---------|
| Backbone | FastViT-SA12 |
| Input size | 512 × 512 |
| Batch size | 8 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1e-3 |
| Weight decay | 0.05 (không áp dụng cho bias/norm) |
| LR Schedule | Linear warmup (5 epochs) + Cosine annealing |
| Min LR | 1e-6 |
| Epochs | 150 |
| AMP | Enabled (mixed precision) |
| Gradient clipping | Max norm = 1.0 |

### LR Schedule

```
LR
 ↑
1e-3│         ╭──────╮
    │        ╱        ╲
    │       ╱          ╲
    │      ╱            ╲          Cosine Annealing
    │     ╱              ╲
    │    ╱                ╲
    │   ╱                  ╲
    │  ╱ Warmup              ╲___
1e-6│ ╱                           min_lr
    └──────────────────────────────→ epoch
    0    5                      150
```

### Weight Decay Groups

```python
# Có weight decay (0.05)
decay_params = [conv.weight, linear.weight, ...]

# Không weight decay
no_decay_params = [bias, bn.weight, bn.bias, norm, layer_scale, ...]
```

---

## Evaluation

### VOC mAP@0.5

Metric chính: **mean Average Precision** tại IoU threshold = 0.5

Quy trình:
1. Predict trên toàn bộ val set
2. Per-class: sort predictions theo score giảm dần
3. Match predictions với ground truths (IoU ≥ 0.5)
4. Tính Precision-Recall curve
5. AP = diện tích dưới PR curve (VOC 2010+ all-point interpolation)
6. mAP = trung bình AP của 20 classes

### Post-processing (Inference)

```
Raw predictions
    │
    ▼ Score threshold (0.01 khi eval, 0.3 khi visualize)
    │
    ▼ Decode box deltas → absolute coordinates
    │
    ▼ Clip boxes to image boundary
    │
    ▼ Per-class NMS (IoU threshold = 0.5)
    │
    ▼ Top-K filtering (max 200 detections/image)
    │
    ▼ Final detections
```

---

## Cách chạy

### Train

```bash
# Shell (Linux/Kaggle)
bash train_detection.sh                      # Default: SA12, batch=8
bash train_detection.sh fastvit_t8 16        # T8, batch=16

# Python trực tiếp
python object_detection.py \
    --data-dir ./data \
    --model fastvit_sa12 \
    --batch-size 8 \
    --epochs 150 \
    --lr 0.001 \
    --eval-interval 5 \
    --save-visualizations
```

### Eval only

```bash
python object_detection.py \
    --data-dir ./data \
    --model fastvit_sa12 \
    --eval-only \
    --resume ./output/detection/best.pth
```

### Resume training

```bash
python object_detection.py \
    --data-dir ./data \
    --model fastvit_sa12 \
    --resume ./output/detection/last.pth
```

### Checkpoint format

```python
{
    "epoch": 50,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scaler_state_dict": ...,     # AMP scaler
    "best_map": 0.65,
    "args": {...},
}
```

---

## Output

```
output/detection/fastvit_sa12_20260411_180000/
├── args.txt               # Training arguments
├── best.pth               # Best mAP checkpoint
├── last.pth               # Latest checkpoint
├── epoch_10.pth           # Periodic checkpoints
├── epoch_20.pth
└── visualizations/        # Detection results (nếu --save-visualizations)
    ├── det_0000.jpg
    ├── det_0001.jpg
    └── ...
```

---

## Tham khảo

- **FastViT**: [Vasu et al., ICCV 2023](https://arxiv.org/abs/2303.14189) — Backbone architecture
- **RetinaNet**: [Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002) — Focal Loss + detection head
- **FPN**: [Lin et al., CVPR 2017](https://arxiv.org/abs/1612.03144) — Feature Pyramid Network
- **PASCAL VOC**: [Everingham et al., IJCV 2010](http://host.robots.ox.ac.uk/pascal/VOC/) — Dataset & evaluation protocol
