#!/bin/bash
# ============================================================
#  FastViT Object Detection - Training Script (PASCAL VOC)
# ============================================================
#
#  Usage:
#    bash train_detection.sh                        (default: fastvit_sa12)
#    bash train_detection.sh fastvit_t8             (lighter model)
#    bash train_detection.sh fastvit_sa12 16        (custom batch size)
#
#  VOC dataset will be auto-downloaded on first run (~2GB)
# ============================================================

# --- Configuration ---
MODEL=${1:-fastvit_sa12}
BATCH_SIZE=${2:-8}
IMG_SIZE=512
EPOCHS=50
LR=0.001
WORKERS=4
DATA_DIR=./data
OUTPUT_DIR=./output/detection
EVAL_INTERVAL=1

# Wandb
WANDB_PROJECT="fastvit-detection"
WANDB_NAME="${MODEL}_bs${BATCH_SIZE}_ep${EPOCHS}"

# --- Print config ---
echo "============================================================"
echo "  FastViT Object Detection Training"
echo "============================================================"
echo "  Model:        ${MODEL}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Image size:   ${IMG_SIZE}"
echo "  Epochs:       ${EPOCHS}"
echo "  LR:           ${LR}"
echo "  Data dir:     ${DATA_DIR}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Wandb:        ${WANDB_PROJECT} / ${WANDB_NAME}"
echo "============================================================"
echo ""


# --- Run training ---
python object_detection.py \
    --data-dir ${DATA_DIR} \
    --model ${MODEL} \
    --batch-size ${BATCH_SIZE} \
    --img-size ${IMG_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --workers ${WORKERS} \
    --output ${OUTPUT_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${WANDB_NAME} \
    --save-visualizations

echo ""
echo "Training completed!"
