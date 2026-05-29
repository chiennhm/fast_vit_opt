@echo off
REM ============================================================
REM  Train Mask R-CNN (FastViT-SA12 + ImageNet pretrained)
REM  on PASCAL VOC 2007+2012
REM
REM  Usage:
REM    train_maskrcnn.bat                         (default settings)
REM    train_maskrcnn.bat 8 150                   (batch=8, epochs=150)
REM    train_maskrcnn.bat 4 150 ./data best.pth   (custom data dir + ckpt)
REM
REM  Args: [batch_size] [epochs] [data_dir] [pretrained_backbone]
REM ============================================================

SET BATCH_SIZE=%1
IF "%BATCH_SIZE%"=="" SET BATCH_SIZE=4

SET EPOCHS=%2
IF "%EPOCHS%"=="" SET EPOCHS=150

SET DATA_DIR=%3
IF "%DATA_DIR%"=="" SET DATA_DIR=./data

SET PRETRAINED=%4
IF "%PRETRAINED%"=="" SET PRETRAINED=./best.pth

echo ============================================================
echo  Mask R-CNN Training: FastViT-SA12 + ImageNet pretrained
echo  Batch size  : %BATCH_SIZE%
echo  Epochs      : %EPOCHS%
echo  Data dir    : %DATA_DIR%
echo  Pretrained  : %PRETRAINED%
echo ============================================================

python object_detection.py ^
    --arch maskrcnn ^
    --model fastvit_sa12 ^
    --pretrained-backbone %PRETRAINED% ^
    --dataset voc ^
    --data-dir %DATA_DIR% ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr 2e-4 ^
    --weight-decay 0.05 ^
    --warmup-epochs 5 ^
    --clip-grad 5.0 ^
    --fpn-channels 256 ^
    --img-size 512 ^
    --eval-interval 5 ^
    --workers 4 ^
    --amp ^
    --output ./output/maskrcnn ^
    --no-wandb

echo.
echo Training complete. Checkpoints in ./output/maskrcnn/
