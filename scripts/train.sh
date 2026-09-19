#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${1:-fastvit_sa12}"
BATCH_SIZE="${2:-8}"
DATA_DIR="${3:-./data/bdd100k}"

python train_bdd100k.py \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --img-size 800 \
  --epochs 50 \
  --workers 4 \
  --output ./output/bdd100k \
  --save-visualizations
