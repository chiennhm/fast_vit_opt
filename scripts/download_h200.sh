#!/usr/bin/env bash
# Stage 2: prepare BDD100K and ImageNet backbone with the existing venv (no GPU).
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash download_h200.sh [options]
First run setup_h200.sh. Ready data and cached weights are reused.
  --work-dir DIR                 Default: /datastore/locth/ChienAnh
  --venv DIR                     Default: WORK_DIR/.venv
  --log-file FILE                Default: WORK_DIR/logs/download_h200.log
  --data-dir DIR                 Default: WORK_DIR/data/bdd100k
  --train-img DIR --train-ann FILE --val-img DIR --val-ann FILE
  --mode train|eval              Required data splits; default train (train + val)
  --download-data                Auto-prepare missing data (also the default)
  --no-download-data             Validate prepared data without downloading/converting
  --archive ZIP                  Use an existing ZIP; reuse already prepared data
  --convert-only                 Regenerate COCO annotations from extracted labels
                                Choose at most one data preparation option.
  --pretrained-backbone PATH      Normalize trusted local unfused ImageNet weights
  --model NAME                   Default fastvit_sa12; automatic download: SA12 only
  --redownload-pretrained         Refresh official Apple cache after validation
  --data-only                    Skip pretrained preparation (e.g. resume/from scratch)
  --pretrained-only              Skip all dataset work
  --dry-run                      Show plan without writes/network/Python
  -h, --help
Default weights: official unfused FastViT-SA12 ImageNet, cached in WORK_DIR/checkpoints.
Trainer-ready weights: WORK_DIR/checkpoints/MODEL_backbone.pth.
No Git fetch, pip install, training or GPU check. Logs append to a file, not the console.
Run where cluster policy allows network access and dataset extraction.
USAGE
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
show_command() { printf '+ '; printf '%q ' "$@"; printf '\n'; }

WORK_DIR=/datastore/locth/ChienAnh
VENV_DIR=
LOG_FILE=
DATA_DIR=
TRAIN_IMG= TRAIN_ANN= VAL_IMG= VAL_ANN=
MODE=train
DATA_ACTION=auto
CUSTOM_DATA_PATHS=0
ARCHIVE=
MODEL=fastvit_sa12
PRETRAINED=
AUTO_PRETRAINED=0
REDOWNLOAD_PRETRAINED=0
DATA_ONLY=0
PRETRAINED_ONLY=0
DRY_RUN=0
# Preparation checks that data is nonempty; run_h200.sh checks the actual batch.
BATCH_SIZE=1
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --redownload-pretrained) REDOWNLOAD_PRETRAINED=1; shift ;;
    --data-only) DATA_ONLY=1; shift ;;
    --pretrained-only) PRETRAINED_ONLY=1; shift ;;
    --download-data|--no-download-data|--convert-only)
      [[ "$DATA_ACTION" == auto ]] || die "Choose only one data preparation option"
      DATA_ACTION="$1"; shift ;;
    --work-dir|--venv|--log-file|--data-dir|--train-img|--train-ann|--val-img|--val-ann|\
    --mode|--archive|--pretrained-backbone|--model)
      need_value "$@"
      case "$1" in
        --work-dir) WORK_DIR="$2" ;; --venv) VENV_DIR="$2" ;; --log-file) LOG_FILE="$2" ;;
        --data-dir) DATA_DIR="$2" ;; --mode) MODE="$2" ;; --model) MODEL="$2" ;;
        --pretrained-backbone) PRETRAINED="$2" ;;
        --train-img) TRAIN_IMG="$2"; CUSTOM_DATA_PATHS=1 ;;
        --train-ann) TRAIN_ANN="$2"; CUSTOM_DATA_PATHS=1 ;;
        --val-img) VAL_IMG="$2"; CUSTOM_DATA_PATHS=1 ;;
        --val-ann) VAL_ANN="$2"; CUSTOM_DATA_PATHS=1 ;;
        --archive)
          [[ "$DATA_ACTION" == auto ]] || die "Choose only one data preparation option"
          DATA_ACTION=--archive; ARCHIVE="$2" ;;
      esac
      shift 2 ;;
    *) die "Unknown argument: $1 (see --help)" ;;
  esac
done
[[ "$MODE" == train || "$MODE" == eval ]] || die "--mode must be train or eval"
[[ "$MODEL" =~ ^[a-zA-Z0-9_]+$ ]] || die "Invalid model name"
(( ! DATA_ONLY || ! PRETRAINED_ONLY )) || die "Choose --data-only or --pretrained-only"
if (( DATA_ONLY )); then
  [[ -z "$PRETRAINED" && "$REDOWNLOAD_PRETRAINED" == 0 ]] || die "--data-only excludes pretrained options"
elif [[ -z "$PRETRAINED" ]]; then
  [[ "$MODEL" == fastvit_sa12 ]] || die "Automatic ImageNet weights support SA12; pass --pretrained-backbone"
  AUTO_PRETRAINED=1
fi
if (( REDOWNLOAD_PRETRAINED && ! AUTO_PRETRAINED )); then
  die "--redownload-pretrained requires the automatic ImageNet cache"
fi
WORK_DIR="$(absolute_path "$WORK_DIR")"
VENV_DIR="$(absolute_path "${VENV_DIR:-$WORK_DIR/.venv}")"
LOG_FILE="$(absolute_path "${LOG_FILE:-$WORK_DIR/logs/download_h200.log}")"
DATA_DIR="$(absolute_path "${DATA_DIR:-$WORK_DIR/data/bdd100k}")"
TRAIN_IMG="$(absolute_path "${TRAIN_IMG:-$DATA_DIR/images/100k/train}")"
TRAIN_ANN="$(absolute_path "${TRAIN_ANN:-$DATA_DIR/annotations/bdd100k_det_train_coco.json}")"
VAL_IMG="$(absolute_path "${VAL_IMG:-$DATA_DIR/images/100k/val}")"
VAL_ANN="$(absolute_path "${VAL_ANN:-$DATA_DIR/annotations/bdd100k_det_val_coco.json}")"
[[ -z "$ARCHIVE" ]] || ARCHIVE="$(absolute_path "$ARCHIVE")"
if (( AUTO_PRETRAINED )); then
  PRETRAINED="$WORK_DIR/checkpoints/fastvit_sa12_imagenet_unfused.pth.tar"
fi
[[ -z "$PRETRAINED" ]] || PRETRAINED="$(absolute_path "$PRETRAINED")"
SOURCE_DIR="$WORK_DIR/source"
ENV_PYTHON="$VENV_DIR/bin/python"
if (( DRY_RUN )); then
  printf 'DRY RUN: download/preparation only; no writes/network/Python/GPU operations.\n'
  printf 'Source: %s; venv: %s\n' "$SOURCE_DIR" "$VENV_DIR"
  printf 'Data directory: %s; action: %s; splits: %s\n' "$DATA_DIR" "$DATA_ACTION" "$MODE"
  printf 'Data only: %s; pretrained only: %s\n' "$DATA_ONLY" "$PRETRAINED_ONLY"
  printf 'Pretrained input: %s; forced redownload: %s\n' "${PRETRAINED:-skipped}" "$REDOWNLOAD_PRETRAINED"
  printf 'Prepared backbone: %s/checkpoints/%s_backbone.pth\n' "$WORK_DIR" "$MODEL"
  printf 'Runtime log (file only): %s\n' "$LOG_FILE"
  exit 0
fi
mkdir -p "$(dirname -- "$LOG_FILE")"
exec >> "$LOG_FILE" 2>&1
printf '\nH200 data/pretrained preparation (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ "$OSTYPE" == linux* ]] || die "Run on Linux; --dry-run can be used on other hosts"
(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) )) || die "Bash >=4.4 required"
[[ -x "$ENV_PYTHON" && -f "$SOURCE_DIR/train_bdd100k.py" ]] || die "Run setup_h200.sh first (same --work-dir and --venv)"
command -v git >/dev/null || die "Missing program: git"
command -v find >/dev/null || die "Missing program: find"
if (( ! PRETRAINED_ONLY )) && [[ -n "$ARCHIVE" ]]; then
  [[ -f "$ARCHIVE" ]] || die "Archive does not exist: $ARCHIVE"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LAUNCH_DIR="$WORK_DIR/preparations/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$LAUNCH_DIR"
git -C "$SOURCE_DIR" rev-parse HEAD > "$LAUNCH_DIR/source_commit.txt"
printf '%s\n' "$LOG_FILE" > "$LAUNCH_DIR/log_file.txt"

# Prepare only backbone weights on CPU; no detector/optimizer state is loaded.
if (( ! DATA_ONLY )); then
  (cd "$SOURCE_DIR" && "$ENV_PYTHON" - "$PRETRAINED" "$WORK_DIR/checkpoints/${MODEL}_backbone.pth" \
    "$LAUNCH_DIR/pretrained.json" "$AUTO_PRETRAINED" "$MODEL" "$REDOWNLOAD_PRETRAINED" <<'PY_PRETRAINED'
import json, os, shutil, sys, tempfile, urllib.request
from pathlib import Path
import torch
from timm.models import create_model
import models  # noqa: F401, registers the cloned FastViT variants

raw_path, output_path, report_path, auto_download, model_name, redownload = sys.argv[1:]
raw_path, output_path, report_path = map(Path, (raw_path, output_path, report_path))
url = "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_models/fastvit_sa12.pth.tar"
if auto_download == "1" and model_name != "fastvit_sa12":
    raise ValueError("Automatic ImageNet weights are configured for fastvit_sa12 only")
if redownload == "1" and auto_download != "1":
    raise ValueError("Redownload applies only to the automatic ImageNet cache")
temp_path = None
prepared_temp = None
try:
    candidate = raw_path
    if not raw_path.is_file() or redownload == "1":
        if auto_download != "1":
            raise FileNotFoundError(f"Missing trusted ImageNet backbone: {raw_path}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=raw_path.parent, suffix=".part", delete=False) as handle:
            temp_path = Path(handle.name)
            print(f"Downloading official unfused ImageNet weights: {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, handle)
        if temp_path.stat().st_size == 0:
            raise ValueError("Downloaded ImageNet checkpoint is empty")
        candidate = temp_path
    else:
        print(f"Reusing local ImageNet backbone: {raw_path}", flush=True)

    # Official Apple weights or a trusted user-provided ImageNet checkpoint.
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" in checkpoint:
        raise ValueError("Expected ImageNet backbone weights, not a trained detector checkpoint")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("ImageNet checkpoint state_dict must be a mapping")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    backbone = create_model(model_name, fork_feat=True, feature_version="fixed_c5")
    expected = backbone.state_dict()
    matched = {key: value for key, value in state.items()
               if key in expected and torch.is_tensor(value) and value.shape == expected[key].shape}
    # ImageNet has no detector output norms; those may initialize fresh.
    # All patch embedding/stage parameters must load, preventing silent partial/zero loads.
    required = {key for key, _ in backbone.named_parameters()
                if key.startswith(("patch_embed.", "network."))}
    missing = sorted(required - matched.keys())
    if not required or missing:
        raise ValueError(f"Incompatible/unfused backbone required: {len(missing)} core parameters missing; "
                         f"examples: {missing[:8]}")
    backbone.load_state_dict(matched, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Separate tensor-only artifact for the trainer; never carry detection-head/optimizer state.
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".part", delete=False) as handle:
        prepared_temp = Path(handle.name)
    torch.save({"state_dict": matched}, prepared_temp)
    os.replace(prepared_temp, output_path)
    # A forced refresh replaces the cache only after download and compatibility checks pass.
    if temp_path is not None and (redownload == "1" or not raw_path.exists()):
        os.replace(temp_path, raw_path)
    report = {"source_path": str(raw_path), "source_url": url if auto_download == "1" else None,
              "downloaded": temp_path is not None, "forced_redownload": redownload == "1",
              "trainer_path": str(output_path), "model": model_name,
              "initialization": "imagenet_backbone_only", "start_epoch": 0,
              "matched_state_keys": len(matched), "total_backbone_state_keys": len(expected),
              "matched_core_parameters": len(required),
              "fresh_backbone_keys": sorted(expected.keys() - matched.keys()),
              "fresh_detection_head": True, "fresh_optimizer_scheduler_scaler": True}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
finally:
    if prepared_temp is not None:
        prepared_temp.unlink(missing_ok=True)
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
PY_PRETRAINED
  )
fi

if (( ! PRETRAINED_ONLY )); then
# BEGIN_DATA_PREPARATION
has_images() {
  [[ -d "$1" ]] || return 1
  local sample
  sample="$(find "$1" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)" || return 1
  [[ -n "$sample" ]]
}

dataset_present() {
  [[ -s "$VAL_ANN" ]] && has_images "$VAL_IMG" || return 1
  if [[ "$MODE" != eval ]]; then
    [[ -s "$TRAIN_ANN" ]] && has_images "$TRAIN_IMG" || return 1
  fi
}

has_raw_labels() {
  local split="$1" candidate
  for candidate in \
    "$DATA_DIR/labels/det_20/det_${split}.json" \
    "$DATA_DIR/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_${split}.json" \
    "$DATA_DIR/bdd100k/labels/bdd100k_labels_images_${split}.json" \
    "$DATA_DIR/labels/bdd100k_labels_images_${split}.json"; do
    [[ ! -s "$candidate" ]] || return 0
  done
  return 1
}

prepare_dataset() {
  local prep_args=(--dest-dir "$DATA_DIR" --keep-zip)
  if [[ "$DATA_ACTION" == --no-download-data ]]; then
    printf 'Automatic data preparation disabled; validating existing data only.\n'
    return 0
  fi
  # Do not re-download merely because the converter's completion marker is absent.
  # Keep explicit --convert-only as a request to regenerate annotations.
  if [[ "$DATA_ACTION" != --convert-only ]] && dataset_present; then
    printf 'Dataset already present: reusing images/COCO annotations; skipping download/extraction/conversion.\n'
    return 0
  fi
  case "$DATA_ACTION" in
    --archive) prep_args+=(--archive "$ARCHIVE") ;;
    --convert-only) prep_args+=(--convert-only) ;;
    auto|--download-data)
      if (( CUSTOM_DATA_PATHS )); then
        die "Custom image/annotation paths are incomplete. Prepare them manually, or use --data-dir with the default layout."
      fi
      if has_images "$DATA_DIR/images/100k/train" && has_images "$DATA_DIR/images/100k/val" \
          && has_raw_labels train && has_raw_labels val; then
        printf 'Local images/raw labels found: converting to COCO only; no download.\n'
        prep_args+=(--convert-only)
      elif [[ -f "$DATA_DIR/bdd100k.zip" ]]; then
        printf 'Local BDD100K ZIP found: reusing %s; no download.\n' "$DATA_DIR/bdd100k.zip"
        prep_args+=(--archive "$DATA_DIR/bdd100k.zip")
      else
        printf 'Dataset missing/incomplete: downloading BDD100K from Kaggle into %s.\n' "$DATA_DIR"
      fi ;;
    *) die "Invalid data preparation action: $DATA_ACTION" ;;
  esac
  (cd "$SOURCE_DIR" && "$ENV_PYTHON" -m tools.download_bdd100k "${prep_args[@]}")
}

prepare_dataset
# END_DATA_PREPARATION
"$ENV_PYTHON" - "$MODE" "$TRAIN_IMG" "$TRAIN_ANN" "$VAL_IMG" "$VAL_ANN" "$BATCH_SIZE" "$LAUNCH_DIR/dataset.json" <<'PY_DATA'
import json, sys
from pathlib import Path
mode, train_img, train_ann, val_img, val_ann, batch, output = sys.argv[1:]
classes = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike",
           "traffic light", "traffic sign"]
aliases = {"pedestrian": "person", "motorcycle": "motor", "bicycle": "bike"}
report = {}
splits = [("val", val_img, val_ann)]
if mode != "eval":
    splits.insert(0, ("train", train_img, train_ann))
for split, root, ann in splits:
    root, ann = Path(root).resolve(), Path(ann)
    if not root.is_dir() or not ann.is_file():
        raise ValueError(f"Missing {split} data: {root}, {ann}; prepare data explicitly")
    with ann.open(encoding="utf-8") as handle:
        data = json.load(handle)
    categories = data.get("categories", [])
    mapping = {c["id"]: aliases.get(c["name"], c["name"]) for c in categories}
    if len(categories) != 10 or mapping != dict(enumerate(classes, 1)):
        raise ValueError(f"{ann}: wrong BDD100K category mapping")
    images = data.get("images", [])
    if not images or (split == "train" and len(images) < int(batch)):
        raise ValueError(f"{split} is empty or smaller than the training batch")
    for item in images[:32]:
        image = (root / item["file_name"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Missing or unsafe image path: {image}")
    report[split] = {"images": len(images), "annotations": len(data.get("annotations", [])),
                     "annotation_path": str(ann), "sampled_paths": min(32, len(images))}
    if split == "val" and len(images) != 10000:
        print(f"WARNING: val contains {len(images)} images, not 10,000")
Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY_DATA
fi
printf 'Preparation complete. Next: sbatch run_h200.sh\n'
