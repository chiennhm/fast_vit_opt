#!/bin/bash
# ============================================================
#  FastViT Object Detection - Training Script (PASCAL VOC / COCO)
# ============================================================
#
#  Usage:
#    bash train_detection.sh                        (default: fastvit_sa12, voc)
#    bash train_detection.sh fastvit_t8             (use lighter model)
#    bash train_detection.sh fastvit_sa12 16        (custom batch size)
#    bash train_detection.sh fastvit_sa12 8 coco    (train on COCO dataset)
#
#  VOC dataset will be auto-downloaded on first run (~2GB)
# ============================================================

# --- Configuration ---
MODEL=${1:-fastvit_sa12}
BATCH_SIZE=${2:-8}
DATASET=${3:-voc}
IMG_SIZE=512
EPOCHS=50
LR=0.0001
WORKERS=4
OUTPUT_DIR=./output/detection
EVAL_INTERVAL=1

# Set data directory based on dataset selection
if [ "$DATASET" = "coco" ]; then
    DATA_DIR=./data/coco
else
    DATA_DIR=./data
fi

# Wandb
WANDB_PROJECT="fastvit-detection"
WANDB_NAME="${MODEL}_${DATASET}_bs${BATCH_SIZE}_ep${EPOCHS}"

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
    --dataset ${DATASET} \
    --data-dir ${DATA_DIR} \
    --model ${MODEL} \
    --batch-size ${BATCH_SIZE} \
    --img-size ${IMG_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --workers ${WORKERS} \
    --output ${OUTPUT_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${WANDB_NAME} \
    --save-visualizations

echo ""
echo "Training completed!"
