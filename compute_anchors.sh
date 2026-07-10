#!/bin/bash
# ============================================================
#  Compute optimal anchor boxes using K-Means++ on PASCAL VOC/COCO/BDD100K
# ============================================================
#
#  This script:
#    1. Downloads dataset annotations if not present
#    2. Runs K-Means++ clustering on all GT bounding boxes
#    3. Outputs anchor_sizes, aspect_ratios, scales for FastViTDetector
#
#  Usage:
#    bash compute_anchors.sh [dataset] [num_anchors] [distance] [img_size]
#
#  Examples:
#    bash compute_anchors.sh voc 12 iou 512
#    bash compute_anchors.sh bdd100k 12 iou 512
#    bash compute_anchors.sh coco 9 iou 640
#
# ============================================================

set -e

# --- Configuration (overridable via positional args) ---
DATASET=${1:-voc}
NUM_ANCHORS=${2:-12}
DISTANCE=${3:-iou}
IMG_SIZE=${4:-512}
NUM_LEVELS=4
NUM_TRIALS=10
MAX_ITER=300
SEED=42
DATA_DIR=./data

# --- Print config ---
echo "============================================================"
echo "  K-Means++ Anchor Box Computation"
echo "============================================================"
echo "  Dataset:      ${DATASET}"
echo "  Num anchors:  ${NUM_ANCHORS}"
echo "  FPN levels:   ${NUM_LEVELS}"
echo "  Distance:     ${DISTANCE}"
echo "  Image size:   ${IMG_SIZE}"
echo "  Trials:       ${NUM_TRIALS}"
echo "  Data dir:     ${DATA_DIR}"
echo "============================================================"
echo ""

# --- Step 1: Ensure dataset annotations are present ---
if [ "${DATASET}" = "voc" ]; then
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
elif [ "${DATASET}" = "bdd100k" ]; then
    BDD_ANN="${DATA_DIR}/bdd100k/annotations/bdd100k_det_train_coco.json"
    if [ ! -f "${BDD_ANN}" ]; then
        echo "[INFO] BDD100K COCO annotations not found — running download_bdd100k.py ..."
        echo ""
        python download_bdd100k.py
        echo ""
        echo "[INFO] Download and conversion finished."
        echo ""
    else
        echo "[INFO] BDD100K annotations found at ${BDD_ANN}"
    fi
elif [ "${DATASET}" = "coco" ]; then
    COCO_ANN="${DATA_DIR}/coco/annotations/instances_train2017.json"
    if [ ! -f "${COCO_ANN}" ]; then
        echo "[INFO] COCO annotations not found — downloading val & annotations via download_coco.py ..."
        echo ""
        python download_coco.py --only-val
        echo ""
        echo "[INFO] Download finished."
        echo ""
    else
        echo "[INFO] COCO annotations found at ${COCO_ANN}"
    fi
fi

# --- Step 2: Run K-Means++ anchor computation ---
echo ""
echo "[INFO] Running K-Means++ for ${DATASET} ..."
echo ""

python compute_anchors.py \
    --data-dir ${DATA_DIR} \
    --dataset ${DATASET} \
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
