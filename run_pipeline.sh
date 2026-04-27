#!/bin/bash
# ============================================================
#  End-to-End Detection Pipeline
#  Step 1: K-Means++ anchor computation
#  Step 2: Training with optimized anchors
#  Step 3: Final evaluation on VOC2007 val
#  Step 4: Benchmark (throughput, latency, FLOPs, memory, energy)
# ============================================================
#
#  Usage:
#    bash run_pipeline.sh                                # defaults
#    bash run_pipeline.sh fastvit_t8 8 256               # lighter setup
#    bash run_pipeline.sh fastvit_sa12 16 512 50 12      # full config
#
# ============================================================

set -e

# --- Configuration ---
MODEL=${1:-fastvit_sa12}
BATCH_SIZE=${2:-8}
IMG_SIZE=${3:-512}
EPOCHS=${4:-50}
NUM_ANCHORS=${5:-12}
LR=0.0001
WORKERS=4
DATA_DIR=./data
OUTPUT_DIR=./output/detection
EVAL_INTERVAL=1
SEED=42

# Derived
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${MODEL}_pipeline_${TIMESTAMP}"
ANCHOR_FILE="${RUN_DIR}/anchors.txt"

mkdir -p "${RUN_DIR}"

# --- Print config ---
echo "============================================================"
echo "  FastViT Detection Pipeline"
echo "============================================================"
echo "  Model:        ${MODEL}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Image size:   ${IMG_SIZE}"
echo "  Epochs:       ${EPOCHS}"
echo "  Num anchors:  ${NUM_ANCHORS}"
echo "  LR:           ${LR}"
echo "  Data dir:     ${DATA_DIR}"
echo "  Output dir:   ${RUN_DIR}"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Download VOC + Compute Anchors via K-Means++
# ============================================================
echo "============================================================"
echo "  STEP 1/4: K-Means++ Anchor Computation"
echo "============================================================"
echo ""

# Ensure VOC is downloaded
VOC2007_DIR="${DATA_DIR}/VOCdevkit/VOC2007"
VOC2012_DIR="${DATA_DIR}/VOCdevkit/VOC2012"

if [ ! -d "${VOC2007_DIR}/Annotations" ] || [ ! -d "${VOC2012_DIR}/Annotations" ]; then
    echo "[INFO] VOC dataset not found — downloading ..."
    python -c "
from voc_dataset import build_voc_datasets
train_ds, val_ds = build_voc_datasets(data_dir='${DATA_DIR}', img_size=${IMG_SIZE}, download=True)
print(f'Download complete: {len(train_ds)} train + {len(val_ds)} val images')
"
    echo ""
fi

# Run K-Means++ and capture anchor_sizes
python compute_anchors.py \
    --data-dir ${DATA_DIR} \
    --num-anchors ${NUM_ANCHORS} \
    --num-levels 4 \
    --img-size ${IMG_SIZE} \
    --distance iou \
    --num-trials 10 \
    --seed ${SEED} \
    --years 2007 2012 \
    2>&1 | tee "${RUN_DIR}/kmeans_log.txt"

# Extract anchor_sizes from output (line: anchor_sizes = (x, y, z, w))
ANCHOR_SIZES=$(grep "^anchor_sizes" "${RUN_DIR}/kmeans_log.txt" | head -1 | sed 's/anchor_sizes = //')
echo ""
echo "[INFO] Computed anchor_sizes: ${ANCHOR_SIZES}"
echo "${ANCHOR_SIZES}" > "${ANCHOR_FILE}"
echo ""

echo "============================================================"
echo "  STEP 1 COMPLETE"
echo "============================================================"
echo ""

# ============================================================
# Step 2: Train Detection Model
# ============================================================
echo "============================================================"
echo "  STEP 2/4: Training (${EPOCHS} epochs)"
echo "============================================================"
echo ""

python object_detection.py \
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
    --output ${RUN_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --no-wandb \
    --save-visualizations \
    2>&1 | tee "${RUN_DIR}/train_log.txt"

echo ""
echo "============================================================"
echo "  STEP 2 COMPLETE"
echo "============================================================"
echo ""

# ============================================================
# Step 3: Final Evaluation on best checkpoint
# ============================================================
echo "============================================================"
echo "  STEP 3/4: Final Evaluation (best.pth)"
echo "============================================================"
echo ""

# Find the best.pth — it's inside a timestamped sub-folder created by object_detection.py
BEST_CKPT=$(find "${RUN_DIR}" -name "best.pth" -type f 2>/dev/null | head -1)

if [ -z "${BEST_CKPT}" ]; then
    # Fall back to last.pth
    BEST_CKPT=$(find "${RUN_DIR}" -name "last.pth" -type f 2>/dev/null | head -1)
fi

if [ -n "${BEST_CKPT}" ]; then
    echo "[INFO] Evaluating checkpoint: ${BEST_CKPT}"
    python object_detection.py \
        --data-dir ${DATA_DIR} \
        --model ${MODEL} \
        --img-size ${IMG_SIZE} \
        --batch-size ${BATCH_SIZE} \
        --workers ${WORKERS} \
        --output ${RUN_DIR} \
        --resume "${BEST_CKPT}" \
        --eval-only \
        --save-visualizations \
        --no-wandb \
        2>&1 | tee "${RUN_DIR}/eval_log.txt"
else
    echo "[WARN] No checkpoint found — skipping evaluation."
fi

echo ""
echo "============================================================"
echo "  STEP 3 COMPLETE"
echo "============================================================"
echo ""

# ============================================================
# Step 4: Benchmark (Throughput, Latency, FLOPs, Memory, Energy)
# ============================================================
echo "============================================================"
echo "  STEP 4/4: Benchmark"
echo "============================================================"
echo ""

python benchmark.py \
    --model ${MODEL} \
    --mode detection \
    --img-size ${IMG_SIZE} \
    --batch-size 1 ${BATCH_SIZE} \
    --amp \
    --energy \
    --iterations 200 \
    --warmup 50 \
    --output "${RUN_DIR}/benchmark" \
    2>&1 | tee "${RUN_DIR}/benchmark_log.txt"

echo ""
echo "============================================================"
echo "  STEP 4 COMPLETE"
echo "============================================================"
echo ""

# ============================================================
# Pipeline Summary
# ============================================================
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo "  Outputs:        ${RUN_DIR}"
echo "  K-Means log:    ${RUN_DIR}/kmeans_log.txt"
echo "  Train log:      ${RUN_DIR}/train_log.txt"
echo "  Eval log:       ${RUN_DIR}/eval_log.txt"
echo "  Benchmark log:  ${RUN_DIR}/benchmark_log.txt"
echo "  Benchmark data: ${RUN_DIR}/benchmark/"
echo "  Anchor sizes:   $(cat ${ANCHOR_FILE} 2>/dev/null || echo 'N/A')"
echo "============================================================"
