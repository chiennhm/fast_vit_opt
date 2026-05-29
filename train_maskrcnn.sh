#!/usr/bin/env bash
# ============================================================
#  Train Mask R-CNN (FastViT-SA12 + ImageNet pretrained)
#  on PASCAL VOC 2007+2012
#
#  Usage:
#    bash train_maskrcnn.sh                         (default settings)
#    bash train_maskrcnn.sh 8 150                   (batch=8, epochs=150)
#    bash train_maskrcnn.sh 4 150 ./data best.pth   (custom data dir + ckpt)
#
#  Args: [batch_size] [epochs] [data_dir] [pretrained_backbone]
# ============================================================

BATCH_SIZE=${1:-4}
EPOCHS=${2:-150}
DATA_DIR=${3:-./data}
PRETRAINED=${4:-./best.pth}

echo "============================================================"
echo " Mask R-CNN Training: FastViT-SA12 + ImageNet pretrained"
echo " Batch size  : $BATCH_SIZE"
echo " Epochs      : $EPOCHS"
echo " Data dir    : $DATA_DIR"
echo " Pretrained  : $PRETRAINED"
echo "============================================================"

python object_detection.py \
    --arch maskrcnn \
    --model fastvit_sa12 \
    --pretrained-backbone "$PRETRAINED" \
    --dataset voc \
    --data-dir "$DATA_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr 2e-4 \
    --weight-decay 0.05 \
    --warmup-epochs 5 \
    --clip-grad 5.0 \
    --fpn-channels 256 \
    --img-size 512 \
    --eval-interval 5 \
    --workers 4 \
    --amp \
    --output ./output/maskrcnn \
    --no-wandb

echo ""
echo "Training complete. Checkpoints in ./output/maskrcnn/"
