#!/usr/bin/env bash
# FASTVIT_TRAINER_ARGS_V1
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  printf '%s\n' \
    'Usage: bash scripts/train.sh [MODEL] [BATCH_SIZE] [DATA_DIR]' \
    '       FASTVIT_PYTHON=/path/to/venv/bin/python bash scripts/train.sh -- TRAINER_ARGS...' \
    'After --, arguments go unchanged to train_bdd100k.py, without wrapper defaults.'
  exit 0
fi

# Launchers such as run_h200.sh own their recipe; do not add legacy defaults.
if [[ "${1:-}" == -- ]]; then
  shift
  exec "${FASTVIT_PYTHON:-python}" -u "${ROOT_DIR}/train_bdd100k.py" "$@"
fi

MODEL="${1:-fastvit_sa12}"
BATCH_SIZE="${2:-8}"
DATA_DIR="${3:-./data/bdd100k}"

exec "${FASTVIT_PYTHON:-python}" -u "${ROOT_DIR}/train_bdd100k.py" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --img-size 800 \
  --epochs 50 \
  --workers 4 \
  --output ./output/bdd100k \
  --save-visualizations
