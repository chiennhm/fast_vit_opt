#!/usr/bin/env bash
# FASTVIT_TRAINER_ARGS_V1
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/resume.sh CHECKPOINT [BATCH_SIZE] [DATA_DIR] [MODEL]' \
    '       FASTVIT_PYTHON=/path/to/venv/bin/python bash scripts/resume.sh CHECKPOINT -- TRAINER_ARGS...' >&2
  exit 2
fi
if [[ "$1" == -h || "$1" == --help ]]; then
  printf '%s\n' \
    'Usage: bash scripts/resume.sh CHECKPOINT [BATCH_SIZE] [DATA_DIR] [MODEL]' \
    '       FASTVIT_PYTHON=/path/to/venv/bin/python bash scripts/resume.sh CHECKPOINT -- TRAINER_ARGS...' \
    'After --, forward trainer arguments without wrapper defaults; add --eval-only for evaluation.'
  exit 0
fi

CHECKPOINT="$1"
shift
# Keep the positional checkpoint authoritative; all other flags are forwarded.
if [[ "${1:-}" == -- ]]; then
  shift
  exec "${FASTVIT_PYTHON:-python}" -u "${ROOT_DIR}/train_bdd100k.py" "$@" --resume "$CHECKPOINT"
fi
BATCH_SIZE="${1:-8}"
DATA_DIR="${2:-./data/bdd100k}"
MODEL="${3:-fastvit_sa12}"

exec "${FASTVIT_PYTHON:-python}" -u "${ROOT_DIR}/train_bdd100k.py" \
  --resume "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --img-size 800 \
  --epochs 50 \
  --workers 4 \
  --output ./output/bdd100k_resumed \
  --save-visualizations
