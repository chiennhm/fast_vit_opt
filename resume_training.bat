@echo off
REM ============================================================
REM  FastViT Object Detection - Resume Training Script
REM ============================================================
REM
REM  Usage:
REM    resume_training.bat <path_to_checkpoint>                    (e.g., ./output/detection/fastvit_sa12_timestamp/last.pth)
REM    resume_training.bat <path_to_checkpoint> <batch_size>       (e.g., ./output/detection/fastvit_sa12_timestamp/last.pth 8)
REM    resume_training.bat <path_to_checkpoint> <batch_size> coco  (COCO dataset)
REM ============================================================

setlocal

set CKPT=%1
if "%CKPT%"=="" (
    echo [ERROR] Please provide the path to the checkpoint file to resume from.
    echo Example: resume_training.bat ./output/detection/fastvit_sa12_timestamp/last.pth
    pause
    exit /b 1
)

set BATCH_SIZE=%2
if "%BATCH_SIZE%"=="" set BATCH_SIZE=8

set DATASET=%3
if "%DATASET%"=="" set DATASET=voc

if "%DATASET%"=="coco" (
    set DATA_DIR=./data/coco
) else (
    set DATA_DIR=./data
)

REM --- Paper-aligned Hyperparameters (1x Schedule) ---
set IMG_SIZE=800
set EPOCHS=12
set LR=1e-4
set WEIGHT_DECAY=0.05
set WARMUP_ITERS=500
set LR_STEPS=8 11
set LR_GAMMA=0.1
set WORKERS=4
set OUTPUT_DIR=./output/detection_resumed

echo ============================================================
echo   Resuming FastViT Training from Checkpoint
echo ============================================================
echo   Checkpoint:        %CKPT%
echo   Batch size:        %BATCH_SIZE%
echo   Image size:        %IMG_SIZE%
echo   Dataset:           %DATASET%
echo   Data dir:          %DATA_DIR%
echo   Output dir:        %OUTPUT_DIR%
echo ============================================================
echo.

python object_detection.py ^
    --resume "%CKPT%" ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --batch-size %BATCH_SIZE% ^
    --img-size %IMG_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --weight-decay %WEIGHT_DECAY% ^
    --warmup-iters %WARMUP_ITERS% ^
    --lr-steps %LR_STEPS% ^
    --lr-gamma %LR_GAMMA% ^
    --workers %WORKERS% ^
    --output %OUTPUT_DIR% ^
    --eval-interval 1 ^
    --save-visualizations

echo.
echo Training resumed and completed!
pause
