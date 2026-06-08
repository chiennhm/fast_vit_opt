#!/bin/bash
# ============================================================
#  FastViT Object Detection - Resume Training Script
# ============================================================
#
#  Usage:
#    bash resume_training.sh <path_to_checkpoint> [batch_size] [dataset]
#    e.g., bash resume_training.sh ./output/detection/fastvit_sa12_timestamp/last.pth 8 voc
# ============================================================

CKPT=$1
if [ -z "$CKPT" ]; then
    echo "[ERROR] Please provide the path to the checkpoint file to resume from."
    echo "Example: bash resume_training.sh ./output/detection/fastvit_sa12_timestamp/last.pth"
    exit 1
fi

BATCH_SIZE=${2:-8}
DATASET=${3:-voc}

# Set data directory based on dataset selection
if [ "$DATASET" = "coco" ]; then
    DATA_DIR=./data/coco
else
    DATA_DIR=./data
fi

# --- Paper-aligned Hyperparameters (1x Schedule) ---
IMG_SIZE=800
EPOCHS=12
LR=0.0001
WEIGHT_DECAY=0.05
WARMUP_ITERS=500
LR_STEPS="8 11"
LR_GAMMA=0.1
WORKERS=4
OUTPUT_DIR=./output/detection_resumed

echo "============================================================"
echo "  Resuming FastViT Training from Checkpoint"
echo "============================================================"
echo "  Checkpoint:        ${CKPT}"
echo "  Batch size:        ${BATCH_SIZE}"
echo "  Image size:        ${IMG_SIZE}"
echo "  Dataset:           ${DATASET}"
echo "  Data dir:          ${DATA_DIR}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "============================================================"
echo ""

python object_detection.py \
    --resume "${CKPT}" \
    --dataset ${DATASET} \
    --data-dir ${DATA_DIR} \
    --batch-size ${BATCH_SIZE} \
    --img-size ${IMG_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-iters ${WARMUP_ITERS} \
    --lr-steps ${LR_STEPS} \
    --lr-gamma ${LR_GAMMA} \
    --workers ${WORKERS} \
    --output ${OUTPUT_DIR} \
    --eval-interval 1 \
    --save-visualizations

echo ""
echo "Training completed!"
