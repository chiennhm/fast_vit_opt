#!/bin/bash

echo "========================================================"
echo "Chay danh gia (evaluation) tren 1 vai mau (5 mau) COCO"
echo "Ket qua hinh anh se duoc luu tai: ./output/detection/visualizations"
echo "========================================================"

python object_detection.py \
    --eval-only \
    --arch maskrcnn \
    --model fastvit_sa12 \
    --dataset coco \
    --data-dir ./data/coco \
    --resume best.pth \
    --max-eval-samples 5 \
    --save-visualizations \
    --eval-batch-size 1
