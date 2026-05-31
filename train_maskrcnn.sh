#!/bin/bash
# ============================================================
#  Mask R-CNN (FastViT-SA12) - Training Script (COCO)
# ============================================================
#
#  Usage:
#    bash train_maskrcnn.sh                          (default: batch=4, coco)
#    bash train_maskrcnn.sh 8                        (custom batch size)
#    bash train_maskrcnn.sh 8 ./data ./best.pth      (custom data + pretrained)
#
#  COCO layout under DATA_DIR:
#    train2017/  val2017/  annotations/instances_{train,val}2017.json
# ============================================================

# --- Configuration ---
BATCH_SIZE=${1:-4}
DATA_DIR=${2:-./data/coco}
PRETRAINED=${3:-./best.pth}
ACCUM_STEPS=${4:-4}
DATASET=coco
IMG_SIZE=512
EPOCHS=50
LR=0.0002
WORKERS=4
OUTPUT_DIR=./output/maskrcnn_coco
EVAL_INTERVAL=1

# Memory stability defaults for Mask R-CNN
RPN_PRE_NMS=1000
RPN_POST_NMS=1000
RPN_BATCH_SIZE=128
BOX_BATCH_SIZE=128

# Wandb
WANDB_PROJECT="fastvit-maskrcnn-coco"
WANDB_NAME="maskrcnn_fastvit_sa12_bs${BATCH_SIZE}_acc${ACCUM_STEPS}_ep${EPOCHS}"

# --- Print config ---
echo "============================================================"
echo "  Mask R-CNN (FastViT-SA12) Training on COCO"
echo "============================================================"
echo "  Batch size:   ${BATCH_SIZE} (Effective batch size: $((BATCH_SIZE * ACCUM_STEPS)))"
echo "  Accum steps:  ${ACCUM_STEPS}"
echo "  Image size:   ${IMG_SIZE}"
echo "  Epochs:       ${EPOCHS}"
echo "  LR:           ${LR}"
echo "  Data dir:     ${DATA_DIR}"
echo "  Pretrained:   ${PRETRAINED}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Wandb:        ${WANDB_PROJECT} / ${WANDB_NAME}"
echo "============================================================"
echo ""


# --- Run training ---
python object_detection.py \
    --arch maskrcnn \
    --model fastvit_sa12 \
    --pretrained-backbone ${PRETRAINED} \
    --dataset ${DATASET} \
    --data-dir ${DATA_DIR} \
    --coco-train-img ${DATA_DIR}/train2017 \
    --coco-train-ann ${DATA_DIR}/annotations/instances_train2017.json \
    --coco-val-img   ${DATA_DIR}/val2017 \
    --coco-val-ann   ${DATA_DIR}/annotations/instances_val2017.json \
    --batch-size ${BATCH_SIZE} \
    --accum-steps ${ACCUM_STEPS} \
    --rpn-pre-nms-train ${RPN_PRE_NMS} \
    --rpn-post-nms-train ${RPN_POST_NMS} \
    --rpn-batch-size ${RPN_BATCH_SIZE} \
    --box-batch-size ${BOX_BATCH_SIZE} \
    --img-size ${IMG_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    --clip-grad 5.0 \
    --fpn-channels 256 \
    --workers ${WORKERS} \
    --output ${OUTPUT_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${WANDB_NAME} \
    --amp \
    --save-visualizations

echo ""
echo "Training completed!"


