#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${1:-fastvit_sa12}"
BATCH_SIZE="${2:-8}"
IMG_SIZE="${3:-800}"
EPOCHS="${4:-50}"
DATA_DIR="${5:-./data/bdd100k}"
RUN_ROOT="./output/bdd100k_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_ROOT}"

ANN_FILE="${DATA_DIR}/annotations/bdd100k_det_train_coco.json"
python -m tools.compute_anchors \
  --ann-file "${ANN_FILE}" \
  --num-anchors 15 \
  --num-levels 5 \
  --img-size "${IMG_SIZE}" | tee "${RUN_ROOT}/anchors.log"

ANCHOR_SIZES="$(grep '^anchor_sizes = ' "${RUN_ROOT}/anchors.log" | head -1 | tr -d '(),' | cut -d= -f2-)"
if [[ "$(wc -w <<< "${ANCHOR_SIZES}")" -ne 5 ]]; then
  echo "Failed to extract five anchor sizes" >&2
  exit 1
fi

python train_bdd100k.py \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --img-size "${IMG_SIZE}" \
  --epochs "${EPOCHS}" \
  --anchor-sizes ${ANCHOR_SIZES} \
  --output "${RUN_ROOT}/training" \
  --save-json \
  --save-visualizations | tee "${RUN_ROOT}/train.log"

CHECKPOINT="$(find "${RUN_ROOT}/training" -name best.pth -type f -print -quit)"
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find "${RUN_ROOT}/training" -name last.pth -type f -print -quit)"
fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "No training checkpoint found" >&2
  exit 1
fi

python -m tools.benchmark \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --img-size "${IMG_SIZE}" \
  --batch-size 1 \
  --output "${RUN_ROOT}/benchmark"
