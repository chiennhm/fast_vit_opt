#!/bin/bash
# ============================================================
#  Compute optimal anchor boxes using K-Means++ on PASCAL VOC
# ============================================================
#
#  This script:
#    1. Downloads VOC dataset if not present (via voc_dataset.py)
#    2. Runs K-Means++ clustering on all GT bounding boxes
#    3. Outputs anchor_sizes, aspect_ratios, scales for FastViTDetector
#
#  Usage:
#    bash compute_anchors.sh                    # defaults
#    bash compute_anchors.sh 12                 # 12 anchor clusters
#    bash compute_anchors.sh 12 iou             # 12 clusters, IoU metric
#    bash compute_anchors.sh 9 euclidean 256    # 9 clusters, Euclidean, img=256
#
# ============================================================

set -e

# --- Configuration (overridable via positional args) ---
NUM_ANCHORS=${1:-12}
DISTANCE=${2:-iou}
IMG_SIZE=${3:-512}
NUM_LEVELS=4
NUM_TRIALS=10
MAX_ITER=300
SEED=42
DATA_DIR=./data

# --- Print config ---
echo "============================================================"
echo "  K-Means++ Anchor Box Computation"
echo "============================================================"
echo "  Num anchors:  ${NUM_ANCHORS}"
echo "  FPN levels:   ${NUM_LEVELS}"
echo "  Distance:     ${DISTANCE}"
echo "  Image size:   ${IMG_SIZE}"
echo "  Trials:       ${NUM_TRIALS}"
echo "  Data dir:     ${DATA_DIR}"
echo "============================================================"
echo ""

# --- Step 1: Ensure VOC dataset is downloaded ---
VOC2007_DIR="${DATA_DIR}/VOCdevkit/VOC2007"
VOC2012_DIR="${DATA_DIR}/VOCdevkit/VOC2012"

if [ ! -d "${VOC2007_DIR}/Annotations" ] || [ ! -d "${VOC2012_DIR}/Annotations" ]; then
    echo "[INFO] VOC dataset not found — downloading via voc_dataset.py ..."
    echo ""
    python -c "
from voc_dataset import build_voc_datasets
print('Downloading PASCAL VOC 2007 + 2012 ...')
train_ds, val_ds = build_voc_datasets(data_dir='${DATA_DIR}', img_size=512, download=True)
print(f'Download complete: {len(train_ds)} train + {len(val_ds)} val images')
"
    echo ""
    echo "[INFO] Download finished."
    echo ""
else
    echo "[INFO] VOC dataset found at ${DATA_DIR}/VOCdevkit"
fi

# --- Step 2: Run K-Means++ anchor computation ---
echo ""
echo "[INFO] Running K-Means++ ..."
echo ""

python compute_anchors.py \
    --data-dir ${DATA_DIR} \
    --num-anchors ${NUM_ANCHORS} \
    --num-levels ${NUM_LEVELS} \
    --img-size ${IMG_SIZE} \
    --distance ${DISTANCE} \
    --max-iter ${MAX_ITER} \
    --num-trials ${NUM_TRIALS} \
    --seed ${SEED} \
    --years 2007 2012

echo ""
echo "============================================================"
echo "  Done! Copy the printed anchor config into"
echo "  config.py → ModelConfig or FastViTDetector constructor."
echo "============================================================"
