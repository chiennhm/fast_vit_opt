#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

ANN_FILE="${1:-./data/bdd100k/annotations/bdd100k_det_train_coco.json}"
IMG_SIZE="${2:-800}"

python -m tools.compute_anchors \
  --ann-file "${ANN_FILE}" \
  --num-anchors 15 \
  --num-levels 5 \
  --img-size "${IMG_SIZE}"
