@echo off
REM ============================================================
REM  FastViT Object Detection - Training Script (PASCAL VOC)
REM ============================================================
REM
REM  Usage:
REM    train_detection.bat                    (default: fastvit_sa12)
REM    train_detection.bat fastvit_t8         (use lighter model)
REM    train_detection.bat fastvit_sa12 8     (custom batch size)
REM
REM  VOC dataset will be auto-downloaded on first run (~2GB)
REM ============================================================

setlocal

REM --- Configuration ---
set MODEL=%1
if "%MODEL%"=="" set MODEL=fastvit_sa12

set BATCH_SIZE=%2
if "%BATCH_SIZE%"=="" set BATCH_SIZE=8

set IMG_SIZE=512
set EPOCHS=150
set LR=0.001
set WORKERS=4
set DATA_DIR=./data
set OUTPUT_DIR=./output/detection
set EVAL_INTERVAL=5

REM --- Print config ---
echo ============================================================
echo   FastViT Object Detection Training
echo ============================================================
echo   Model:        %MODEL%
echo   Batch size:   %BATCH_SIZE%
echo   Image size:   %IMG_SIZE%
echo   Epochs:       %EPOCHS%
echo   LR:           %LR%
echo   Data dir:     %DATA_DIR%
echo   Output dir:   %OUTPUT_DIR%
echo ============================================================
echo.

REM --- Run training ---
python object_detection.py ^
    --data-dir %DATA_DIR% ^
    --model %MODEL% ^
    --batch-size %BATCH_SIZE% ^
    --img-size %IMG_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --workers %WORKERS% ^
    --output %OUTPUT_DIR% ^
    --eval-interval %EVAL_INTERVAL% ^
    --save-visualizations

echo.
echo Training completed!
pause
