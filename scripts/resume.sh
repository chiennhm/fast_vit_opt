#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/resume.sh CHECKPOINT [BATCH_SIZE] [DATA_DIR] [MODEL]" >&2
  exit 2
fi

CHECKPOINT="$1"
BATCH_SIZE="${2:-8}"
DATA_DIR="${3:-./data/bdd100k}"
MODEL="${4:-fastvit_sa12}"

python train_bdd100k.py \
  --resume "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --img-size 800 \
  --epochs 50 \
  --workers 4 \
  --output ./output/bdd100k_resumed \
  --save-visualizations
