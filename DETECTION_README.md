# FastViT Object Detection & Instance Segmentation

Tài liệu hướng dẫn toàn diện về module **Object Detection & Instance Segmentation** sử dụng **FastViT** làm backbone, hỗ trợ 2 kiến trúc chính (**RetinaNet** và **Mask R-CNN**) trên các tập dữ liệu **PASCAL VOC**, **MS-COCO**, và **BDD100K**.

---

## 1. Kiến trúc Hệ thống

Codebase cung cấp 2 pipeline detector linh hoạt được xây dựng trên backbone FastViT:

### 1.1 FastViT + RetinaNet (1-Stage Object Detector)
Phù hợp cho bài toán phát hiện vật thể thời gian thực (real-time detection) với tốc độ cao và số lượng tham số tối ưu.

```
Input Image (3 × H × W)
        │
        ▼
┌─────────────────────────┐
│   FastViT Backbone      │  (fork_feat=True, multi-scale stages)
│   (Structural Reparam)  │
└─────────────────────────┘
        │
        ├── Stage 0 / C1: (B, C0, H/4,  W/4)   stride=4
        ├── Stage 1 / C2: (B, C1, H/8,  W/8)   stride=8
        ├── Stage 2 / C3: (B, C2, H/16, W/16)  stride=16
        └── Stage 3 / C4: (B, C3, H/32, W/32)  stride=32
        │
        ▼
┌─────────────────────────┐
│   FPN (Feature Pyramid  │  Lateral Conv (1×1) + Top-down merge + FPN Conv (3×3)
│   Network)              │  Đưa tất cả feature levels về 256 channels
└─────────────────────────┘
        │
        ├── P0: (B, 256, H/4,  W/4)
        ├── P1: (B, 256, H/8,  W/8)
        ├── P2: (B, 256, H/16, W/16)
        └── P3: (B, 256, H/32, W/32)
        │
        ▼
┌─────────────────────────┐
│   RetinaNet Head        │  Shared 4×(Conv3×3 + GroupNorm + ReLU)
└─────────────────────────┘
        │
        ├── Classification Subnet ──→ (B, Total_Anchors, Num_Classes)
        └── Regression Subnet     ──→ (B, Total_Anchors, 4)  [tx, ty, tw, th]
        │
        ▼
┌─────────────────────────┐
│   Post-processing       │  Score Threshold + Decode deltas + Per-class NMS
└─────────────────────────┘
        │
        ▼
  Final Detections: Bounding Boxes, Class Labels, Confidence Scores
```

### 1.2 FastViT + Mask R-CNN (2-Stage Detection & Instance Segmentation)
Sử dụng backbone FastViT kết hợp FPN và torchvision Mask R-CNN head để đồng thời dự đoán Bounding Box và Pixel-level Instance Segmentation Mask.

```
Input Image (3 × H × W)
        │
        ▼
┌─────────────────────────┐
│   FastViT Backbone      │  Load pretrained ImageNet (`best.pth`)
└─────────────────────────┘
        │
        ├── C1, C2, C3, C4
        │
        ▼
┌─────────────────────────┐
│   FeaturePyramidNetwork │  Xuất 5 levels: P2, P3, P4, P5, P6 (256 channels)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Region Proposal (RPN) │  Tạo RoIs (Region of Interests) tiềm năng
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Multi-scale RoIAlign  │  Trích xuất features cố định 7×7 (Box) và 14×14 (Mask)
└─────────────────────────┘
        │
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
┌───────────────────────┐   ┌──────────────────────────────────┐
│  Fast R-CNN Box Head  │   │  Mask Head (4×Conv + ConvTrans)  │
│  (2×FC 1024)          │   │  Dự đoán binary mask 28×28       │
├───────────────────────┤   └──────────────────────────────────┘
│ - Cls logits (C + 1)  │
│ - Box deltas (4)      │
└───────────────────────┘
```

---

## 2. FastViT Backbone Variants

FastViT sử dụng cơ chế **Structural Reparameterization** (khi train là multi-branch, khi inference gom về 1 branch conv đơn giản) giúp tốc độ cực nhanh:

| Variant | Stage 0 (stride 4) | Stage 1 (stride 8) | Stage 2 (stride 16) | Stage 3 (stride 32) | Self-Attention | Tổng Params |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `fastvit_t8` | 48 | 96 | 192 | 384 | Không | ~4M |
| `fastvit_t12` | 64 | 128 | 256 | 512 | Không | ~7M |
| `fastvit_s12` | 64 | 128 | 256 | 512 | Không | ~9M |
| `fastvit_sa12` | 64 | 128 | 256 | 512 | Có (Stage 3) | ~11M |
| `fastvit_sa24` | 64 | 128 | 256 | 512 | Có (Stage 3) | ~21M |
| `fastvit_sa36` | 64 | 128 | 256 | 512 | Có (Stage 3) | ~36M |
| `fastvit_ma36` | 76 | 152 | 304 | 608 | Có (Stage 3) | ~44M |

> **Khuyên dùng**: Các phiên bản **SA (Self-Attention)** như `fastvit_sa12` và `fastvit_sa24` đem lại độ chính xác phát hiện vật thể vượt trội nhờ Receptive Field toàn cục ở feature level sâu nhất.

---

## 3. Cấu trúc thư mục

```
fast_vit_opt/
├── models/                          # FastViT Backbone implementation
│   ├── fastvit.py                   #   Định nghĩa backbone và các variants (T8..MA36)
│   └── modules/
│       ├── mobileone.py             #   MobileOne block reparameterization
│       └── replknet.py              #   Large Kernel Convolution
│
├── detection/                       # Module Detection & Segmentation
│   ├── __init__.py
│   ├── fastvit_detector.py          #   FastViTDetector: 1-stage RetinaNet
│   ├── maskrcnn_detector.py         #   FastViTMaskRCNN: 2-stage Mask R-CNN
│   ├── losses.py                    #   Focal Loss, Smooth L1, DFL, Target Matcher
│   ├── eval_voc.py                  #   VOC-style mAP@0.5 Evaluator
│   ├── eval_coco.py                 #   COCO-standard mAP@[0.5:0.95] (Box & Mask)
│   └── visualize.py                 #   Vẽ Bounding Box & Segmentation Mask
│
├── voc_dataset.py                   # Dataset loader & augmentations cho PASCAL VOC
├── coco_dataset.py                  # Dataset loader (BBox + Mask) cho MS-COCO
├── bdd100k_dataset.py               # Dataset loader cho Berkeley DeepDrive 100K
├── download_coco.py                 # Script tải tự động MS-COCO
├── download_bdd100k.py              # Script tải & convert annotation BDD100K
├── compute_anchors.py               # Thuật toán K-Means++ tính anchors tối ưu
├── object_detection.py              # Entrypoint chính cho Training & Evaluation
│
├── train_detection.sh               # Bash script train RetinaNet (VOC / COCO)
├── train_maskrcnn.sh                # Bash script train Mask R-CNN (COCO)
├── run_pipeline.sh                  # Pipeline hoàn chỉnh: K-Means -> Train -> Eval -> Benchmark
├── resume_training.sh               # Script tiếp tục train từ checkpoint
├── eval_samples.sh                  # Quick inference & visualization trên vài mẫu val
├── compute_anchors.sh               # Script chạy K-Means++ anchors
└── benchmark.py                     # Đánh giá Latency, Throughput, Memory, FLOPs
```

---

## 4. Các Dataset Hỗ trợ

### 4.1 PASCAL VOC (`--dataset voc`)
- **Số lớp**: 20 classes (`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`).
- **Tập dữ liệu**: VOC 2007 trainval + VOC 2012 trainval (~16,551 ảnh), VOC 2007 val (~2,510 ảnh).
- **Download**: Tự động tải khi truyền flag `--download` hoặc chạy lần đầu.

### 4.2 MS-COCO (`--dataset coco`)
- **Số lớp**: 80 object classes.
- **Tập dữ liệu**: COCO `train2017` (~118k ảnh), `val2017` (~5k ảnh).
- **Hỗ trợ**: Bounding box detection + Instance Segmentation polygon/RLE masks.
- **Download**: Tải nhanh qua:
  ```bash
  python download_coco.py --only-val    # Tải val2017 + annotations (~1GB)
  python download_coco.py               # Tải đầy đủ train + val (~20GB)
  ```

### 4.3 BDD100K (`--dataset bdd100k`)
- **Số lớp**: 10 driving classes (`pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`).
- **Định dạng**: Chuyển đổi về chuẩn COCO format JSON annotations qua `download_bdd100k.py`.

---

## 5. Augmentation Pipeline

Mọi phép biến đổi hình ảnh khi training đều tự động căn chỉnh tọa độ Bounding Box và Masks tương ứng:

| Phép biến đổi | Mô tả | Xác suất |
|:---|:---|:---:|
| **Horizontal Flip** | Lật ngang ảnh, đảo trục tọa độ $x$ của bbox và lật mask | 50% |
| **Color Jitter** | Điều chỉnh ngẫu nhiên Brightness, Contrast, Saturation, Hue | 50% mỗi loại |
| **Random Expand** | Zoom out (1.0x - 2.0x), chèn mean color padding | 50% |
| **Random Crop** | Cắt ngẫu nhiên đảm bảo giữ lại center của $\ge 1$ ground-truth box | 50% |
| **Resize & Normalize** | Scale về kích thước `--img-size` (e.g. 512×512 hoặc 800×800) và chuẩn hóa ImageNet | 100% |

---

## 6. Training & Optimization Features

Codebase tích hợp đầy đủ các kỹ thuật huấn luyện hiện đại cho Vision Transformer:

1. **Pretrained Backbone Loading (`--pretrained-backbone`)**:
   Tự động load weights trích xuất đặc trưng từ mô hình ImageNet Classification (`best.pth`) vào backbone detector, bỏ qua classification head cũ.
2. **Hệ thống LR Schedulers (`--scheduler`)**:
   - `step`: Warmup ($N$ epochs hoặc $K$ iterations) + Multi-step decay tại các mốc `--lr-steps 8 11` với hệ số `--lr-gamma 0.1`.
   - `cosine`: Warmup + Cosine Annealing giảm dần về `--min-lr 1e-6`.
3. **Gradient Accumulation (`--accum-steps`)**:
   Tích lũy gradient qua nhiều micro-batches, cho phép train effective batch size lớn (ví dụ batch 4 × accum 4 = effective batch 16) mà không bị GPU OOM.
4. **Quản lý bộ nhớ VRAM cho Mask R-CNN**:
   - `--rpn-pre-nms-train` & `--rpn-post-nms-train`: Giới hạn số lượng proposals trước và sau NMS.
   - `--rpn-batch-size` & `--box-batch-size`: Điều chỉnh số anchors/proposals sample per-image khi tính RPN & RoI head loss.
5. **Bộ nhớ RAM Caching (`--cache-ram`)**:
   Preload toàn bộ ảnh và annotations vào bộ nhớ RAM giúp giảm tải bottleneck đọc đĩa I/O khi train trên ổ cứng mạng/NAS.
6. **Automatic Mixed Precision (`--amp`)**:
   Sử dụng `torch.cuda.amp.autocast()` và `GradScaler` tăng tốc training 2x và tiết kiệm 50% VRAM.
7. **Weights & Biases (WandB) Logging**:
   Tự động log losses, learning rate, gradient norm, mAP metrics, và visual detection results lên Wandb (`--wandb-project`, `--wandb-name`).

---

## 7. Hướng dẫn Chạy (Quick Start & Usage)

### 7.1 Huấn luyện Mask R-CNN trên COCO (Instance Segmentation)

Chạy nhanh qua bash script:
```bash
# Mặc định: batch_size=4, accum_steps=4, data_dir=./data/coco, pretrained=./best.pth
bash train_maskrcnn.sh

# Tùy chỉnh batch size
bash train_maskrcnn.sh 8
```

Hoặc chạy trực tiếp bằng Python:
```bash
python object_detection.py \
    --arch maskrcnn \
    --model fastvit_sa12 \
    --pretrained-backbone ./best.pth \
    --dataset coco \
    --data-dir ./data/coco \
    --batch-size 4 \
    --accum-steps 4 \
    --rpn-pre-nms-train 1000 \
    --rpn-post-nms-train 1000 \
    --rpn-batch-size 128 \
    --box-batch-size 128 \
    --img-size 512 \
    --epochs 50 \
    --lr 0.0002 \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    --clip-grad 5.0 \
    --output ./output/maskrcnn_coco \
    --eval-interval 1 \
    --amp \
    --save-visualizations
```

---

### 7.2 Huấn luyện RetinaNet trên PASCAL VOC hoặc COCO

Chạy qua bash script:
```bash
# Train FastViT-SA12 trên VOC (mặc định batch=8)
bash train_detection.sh

# Train FastViT-T8 trên VOC với batch=16
bash train_detection.sh fastvit_t8 16

# Train FastViT-SA12 trên COCO
bash train_detection.sh fastvit_sa12 8 coco
```

Hoặc chạy trực tiếp bằng Python:
```bash
python object_detection.py \
    --arch fastvit \
    --model fastvit_sa12 \
    --dataset voc \
    --data-dir ./data \
    --batch-size 8 \
    --img-size 512 \
    --epochs 50 \
    --lr 0.0001 \
    --warmup-epochs 5 \
    --scheduler step \
    --lr-steps 30 42 \
    --output ./output/detection \
    --eval-interval 1 \
    --save-visualizations
```

---

### 7.3 Tính toán Anchors Tối ưu với K-Means++

Tính bộ anchor boxes phù hợp nhất với phân phối bounding boxes của dataset:
```bash
# Tính 12 anchors cho VOC với input size 512x512
bash compute_anchors.sh voc 12 iou 512

# Tính 12 anchors cho BDD100K
bash compute_anchors.sh bdd100k 12 iou 512

# Tính 9 anchors cho COCO
bash compute_anchors.sh coco 9 iou 640
```

---

### 7.4 Tiếp tục Huấn luyện (Resume Training)

Tiếp tục train từ checkpoint gần nhất:
```bash
bash resume_training.sh ./output/maskrcnn_coco/last.pth 4 coco
```

---

### 7.5 Đánh giá (Evaluation & Visualization)

**Chỉ chạy đánh giá trên toàn bộ Validation Set:**
```bash
python object_detection.py \
    --eval-only \
    --arch maskrcnn \
    --model fastvit_sa12 \
    --dataset coco \
    --data-dir ./data/coco \
    --resume ./output/maskrcnn_coco/best.pth \
    --save-visualizations
```

**Chạy thử nghiệm nhanh trên vài mẫu (Sanity Check):**
```bash
bash eval_samples.sh
```
*(Kết quả trực quan hóa BBox và Mask sẽ được lưu tại `./output/detection/visualizations/`)*

---

### 7.6 Pipeline Khép kín (End-to-End Pipeline)

Chạy từ A-Z bao gồm: Tính Anchors -> Training -> Evaluation -> Đo Benchmark hiệu năng:
```bash
bash run_pipeline.sh fastvit_sa12 8 512 50 12
```

---

## 8. Bảng Tổng hợp Tham số Dòng lệnh (`object_detection.py`)

| Tham số | Mặc định | Mô tả |
|:---|:---:|:---|
| `--arch` | `fastvit` | Kiến trúc detector: `fastvit` (RetinaNet) hoặc `maskrcnn` (Mask R-CNN) |
| `--model` | `fastvit_sa12` | Variant backbone FastViT: `fastvit_t8`, `t12`, `s12`, `sa12`, `sa24`, `sa36`, `ma36` |
| `--dataset` | `voc` | Dataset sử dụng: `voc`, `coco`, hoặc `bdd100k` |
| `--data-dir` | `./data` | Đường dẫn thư mục dữ liệu gốc |
| `--pretrained-backbone` | `None` | Đường dẫn checkpoint pretrained backbone (e.g. `best.pth`) |
| `--img-size` | `800` (hoặc `512`) | Kích thước ảnh đầu vào |
| `--batch-size` | `16` | Batch size mỗi GPU |
| `--accum-steps` | `1` | Số bước tích lũy gradient trước khi update weights |
| `--lr` | `1e-4` | Learning rate khởi tạo |
| `--scheduler` | `step` | Loại scheduler: `step` hoặc `cosine` |
| `--warmup-epochs` | `5` | Số epochs linear warmup |
| `--weight-decay` | `0.05` | Weight decay cho AdamW |
| `--clip-grad` | `5.0` | Giới hạn norm gradient clipping |
| `--rpn-batch-size` | `128` | Số proposals sample cho RPN loss (Mask R-CNN) |
| `--box-batch-size` | `512` | Số proposals sample cho Fast R-CNN box/mask loss |
| `--amp` / `--no-amp` | `True` | Bật/tắt Automatic Mixed Precision |
| `--cache-ram` | `False` | Preload toàn bộ dataset lên RAM |
| `--eval-interval` | `5` | Chu kỳ đánh giá sau mỗi N epochs |
| `--eval-only` | `False` | Chỉ chạy evaluation |
| `--max-eval-samples`| `None` | Giới hạn số lượng mẫu eval (dùng cho debug/quick test) |
| `--resume` | `None` | Đường dẫn checkpoint để khôi phục trạng thái train |
| `--save-visualizations`| `False` | Lưu ảnh kết quả dự đoán có vẽ boxes/masks |
| `--wandb-project` | `fastvit-detection`| Tên project trên WandB |

---

## 9. Tham khảo

- **FastViT**: [Vasu et al., ICCV 2023 - FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/abs/2303.14189)
- **Mask R-CNN**: [He et al., ICCV 2017 - Mask R-CNN](https://arxiv.org/abs/1703.06870)
- **RetinaNet & Focal Loss**: [Lin et al., ICCV 2017 - Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **FPN**: [Lin et al., CVPR 2017 - Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- **MS COCO**: [Lin et al., ECCV 2014 - Microsoft COCO: Common Objects in Context](https://cocodataset.org/)
- **BDD100K**: [Yu et al., CVPR 2020 - BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://bdd-data.berkeley.edu/)
