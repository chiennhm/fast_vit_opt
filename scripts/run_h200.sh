#!/usr/bin/env bash
#SBATCH --job-name=fastvit_sa12
#SBATCH --output=/datastore/locth/ChienAnh/logs/fastvit_%j.log
#SBATCH --error=/datastore/locth/ChienAnh/logs/fastvit_%j.log
#SBATCH --open-mode=append
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --chdir=/datastore/locth/ChienAnh
# Stage 3: offline preflight + training/evaluation. Run setup_h200.sh and
# download_h200.sh first. This file also works from Slurm's spool directory.
# Create the logs directory BEFORE sbatch; Slurm opens its log before Bash runs.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  mkdir -p /datastore/locth/ChienAnh/logs
  sbatch run_h200.sh --epochs 50

  # Or reuse your own trusted, unfused ImageNet weights:
  sbatch run_h200.sh --pretrained-backbone /path/to/fastvit_sa12_pretrained.pth \
    --epochs 50

  # Direct Bash is for an already allocated GPU/server, not a login-node job:
  bash run_h200.sh \
    --data-dir /datastore/locth/ChienAnh/data/bdd100k --epochs 50

Modes:
  --mode train|resume|eval        Default: train (new run from epoch 0)
  --checkpoint PATH              Requires --mode resume or --mode eval
  --pretrained-backbone PATH      Local unfused ImageNet weights (new run only)
                                Default: WORK_DIR/checkpoints/fastvit_sa12_backbone.pth
  --from-scratch                 Explicit random initialization for a new run
  --model NAME                   New-run model only; default fastvit_sa12

Prepared source/environment (no clone, pip install or download here):
                                Calls source/scripts/train.sh or resume.sh;
                                these must support the FASTVIT_TRAINER_ARGS_V1 API.
  --work-dir DIR                 Default: /datastore/locth/ChienAnh
  --venv DIR                     Default: WORK_DIR/.venv
  --log-file FILE                Append all runtime stdout/stderr to this file
                                Default: WORK_DIR/launches/TIMESTAMP/run.log
                                SLURM default: ROOT/logs/fastvit_JOBID.log
  --gpu DEVICE                   Direct Bash only: one GPU index/UUID; default 0
                                Under SLURM, keep the scheduler's assigned GPU
  --allow-other-gpu              Intentionally permit a non-H200 GPU

Data:
  --data-dir DIR                 Default: WORK_DIR/data/bdd100k
  --train-img DIR --train-ann FILE --val-img DIR --val-ann FILE
                                All required data must already exist.
                                Use download_h200.sh to prepare missing data/weights.

Training/evaluation:
  --epochs N                     Total target epochs, NOT extra; default 50
  --batch-size N --accum-steps N  Defaults: 4 and 4 (effective batch 16)
  --eval-batch-size N --workers N Defaults: 2 and 2 (previous run)
                                Workers are reduced for smaller CPU allocations.
  --score-thresh FLOAT            Default: 0.05
  --nms-thresh FLOAT              Default: 0.5
  --max-detections N              Default: 100
  --pre-nms-topk N                Default: 2000
  --max-eval-samples N            Smoke test only, not full-val metrics
  --epoch-log-only               Default: epoch summaries; no batch logs/progress
  --no-epoch-log-only            Restore per-batch logging/progress for diagnostics
  --post-train-eval              Default: final overall/weather/confidence evaluation
  --no-post-train-eval           Skip the final analysis suite
  --weather-labels FILE          Raw BDD100K val labels with attributes.weather;
                                default: locate automatically under DATA_DIR
  --confidence-thresholds CSV     Default: 0.001,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9
  --prepare-only                 Offline preflight without training/evaluation
  --dry-run                      Print plan, no writes/network/Python/GPU
  -h, --help

Requires Linux, Bash >=4.4, Git, a prepared venv and a CUDA driver.
Prepare refactor-bdd source/venv with setup_h200.sh, then run download_h200.sh.
New training creates fresh optimizer/scheduler/scaler state and runs warmup.
New runs default to ImageNet backbone initialization; --from-scratch opts out.
Resume preserves the checkpoint loss/architecture/LR/scheduler recipe.
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
show_command() { printf '+ '; printf '%q ' "$@"; printf '\n'; }

MODE=train
WORK_DIR=/datastore/locth/ChienAnh
VENV_DIR=
CHECKPOINT=
PRETRAINED=
AUTO_PRETRAINED=0
FROM_SCRATCH=0
MODEL=fastvit_sa12
DATA_DIR=
TRAIN_IMG= TRAIN_ANN= VAL_IMG= VAL_ANN=
EPOCHS=50
# User-selected batch; other new-run recipe settings follow the previous run.
BATCH_SIZE=4
ACCUM_STEPS=4
EVAL_BATCH_SIZE=2
WORKERS=2
GPU=0
SCORE_THRESH=0.05
NMS_THRESH=0.5
MAX_DETECTIONS=100
PRE_NMS_TOPK=2000
MAX_EVAL_SAMPLES=
ALLOW_OTHER_GPU=0
PREPARE_ONLY=0
EPOCH_LOG_ONLY=1
POST_TRAIN_EVAL=1
WEATHER_LABELS=
CONFIDENCE_THRESHOLDS=0.001,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9
DRY_RUN=0
LOG_FILE=
GPU_EXPLICIT=0
SLURM_MODE=0
SLURM_MAX_WORKERS=0

# Keep the configured micro-batch/accumulation on the H200 MIG as well.
# Only reduce DataLoader workers when the CPU allocation requires it.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SLURM_MODE=1
  slurm_cpu_budget="${SLURM_CPUS_PER_TASK:-8}"
  [[ "$slurm_cpu_budget" =~ ^[0-9]+$ ]] && (( 10#$slurm_cpu_budget > 0 )) || die "Invalid SLURM_CPUS_PER_TASK"
  # Leave capacity for the trainer and up to two persistent validation workers.
  SLURM_MAX_WORKERS=$(( 10#$slurm_cpu_budget > 3 ? 10#$slurm_cpu_budget - 3 : 0 ))
  if (( WORKERS > SLURM_MAX_WORKERS )); then WORKERS="$SLURM_MAX_WORKERS"; fi
fi

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --from-scratch) FROM_SCRATCH=1; shift ;;
    --repo-url|--branch|--commit|--skip-install)
      die "$1 belongs to setup_h200.sh; run it before submitting this job" ;;
    --redownload-pretrained|--download-data|--no-download-data|--convert-only|--archive)
      die "$1 belongs to download_h200.sh; run it before submitting this job" ;;
    --allow-other-gpu) ALLOW_OTHER_GPU=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --epoch-log-only) EPOCH_LOG_ONLY=1; shift ;;
    --no-epoch-log-only) EPOCH_LOG_ONLY=0; shift ;;
    --post-train-eval) POST_TRAIN_EVAL=1; shift ;;
    --no-post-train-eval) POST_TRAIN_EVAL=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --mode|--work-dir|--venv|--log-file|--checkpoint|\
    --pretrained-backbone|--model|--data-dir|--train-img|--train-ann|--val-img|\
    --val-ann|--epochs|--batch-size|--accum-steps|--eval-batch-size|\
    --workers|--gpu|--score-thresh|--nms-thresh|--max-detections|--pre-nms-topk|\
    --max-eval-samples|--weather-labels|--confidence-thresholds)
      need_value "$@"
      case "$1" in
        --mode) MODE="$2" ;; --work-dir) WORK_DIR="$2" ;; --venv) VENV_DIR="$2" ;;
        --log-file) LOG_FILE="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;; --pretrained-backbone) PRETRAINED="$2" ;;
        --model) MODEL="$2" ;; --data-dir) DATA_DIR="$2" ;;
        --train-img) TRAIN_IMG="$2" ;;
        --train-ann) TRAIN_ANN="$2" ;;
        --val-img) VAL_IMG="$2" ;;
        --val-ann) VAL_ANN="$2" ;;
        --epochs) EPOCHS="$2" ;; --batch-size) BATCH_SIZE="$2" ;;
        --accum-steps) ACCUM_STEPS="$2" ;; --eval-batch-size) EVAL_BATCH_SIZE="$2" ;;
        --workers) WORKERS="$2" ;; --gpu) GPU="$2"; GPU_EXPLICIT=1 ;;
        --score-thresh) SCORE_THRESH="$2" ;; --nms-thresh) NMS_THRESH="$2" ;;
        --max-detections) MAX_DETECTIONS="$2" ;; --pre-nms-topk) PRE_NMS_TOPK="$2" ;;
        --max-eval-samples) MAX_EVAL_SAMPLES="$2" ;;
        --weather-labels) WEATHER_LABELS="$2" ;;
        --confidence-thresholds) CONFIDENCE_THRESHOLDS="$2" ;;
      esac
      shift 2 ;;
    *) die "Unknown argument: $1 (see --help)" ;;
  esac
done

case "$MODE" in
  resume|eval)
    [[ -n "$CHECKPOINT" ]] || die "$MODE requires --checkpoint"
    [[ -z "$PRETRAINED" && "$FROM_SCRATCH" == 0 ]] || die "Initialization options apply to train only" ;;
  train)
    [[ -z "$CHECKPOINT" ]] || die "Use --mode resume for --checkpoint"
    [[ -z "$PRETRAINED" || "$FROM_SCRATCH" == 0 ]] || die "Choose one initialization option"
    if [[ -z "$PRETRAINED" && "$FROM_SCRATCH" == 0 ]]; then
      [[ "$MODEL" == fastvit_sa12 ]] || die "Default prepared ImageNet weights support SA12; pass --pretrained-backbone or --from-scratch"
      AUTO_PRETRAINED=1
    fi ;;
  *) die "Invalid --mode: $MODE" ;;
esac
for option in EPOCHS BATCH_SIZE ACCUM_STEPS EVAL_BATCH_SIZE MAX_DETECTIONS PRE_NMS_TOPK; do
  value="${!option}"
  [[ "$value" =~ ^[0-9]+$ ]] && (( 10#$value > 0 )) || die "$option must be a positive integer"
done
[[ "$WORKERS" =~ ^[0-9]+$ ]] || die "Workers must be a nonnegative integer"
if (( SLURM_MODE && 10#$WORKERS > SLURM_MAX_WORKERS )); then
  die "--workers exceeds the CPU budget (max $SLURM_MAX_WORKERS for this allocation)"
fi
if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  [[ "$MAX_EVAL_SAMPLES" =~ ^[0-9]+$ ]] && (( 10#$MAX_EVAL_SAMPLES > 0 )) || die "Invalid max-eval-samples"
fi
# BEGIN_GPU_ENVIRONMENT
configure_gpu_visibility() {
  if (( SLURM_MODE )); then
    (( GPU_EXPLICIT == 0 )) || die "Do not pass --gpu under SLURM; select resources with sbatch --gres"
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != -1 \
        && "$CUDA_VISIBLE_DEVICES" != NoDevFiles && "$CUDA_VISIBLE_DEVICES" != none \
        && "$CUDA_VISIBLE_DEVICES" != *,* ]] || die "SLURM must expose exactly one assigned CUDA/MIG device"
    GPU="$CUDA_VISIBLE_DEVICES"
    # Do not replace Slurm's GPU/MIG UUID or its cgroup-relative index with 0.
  else
    [[ -n "$GPU" && "$GPU" != *,* ]] || die "One GPU per job; this trainer does not support DDP"
    export CUDA_VISIBLE_DEVICES="$GPU"
  fi
}
configure_gpu_visibility
# END_GPU_ENVIRONMENT

WORK_DIR="$(absolute_path "$WORK_DIR")"
if (( AUTO_PRETRAINED )); then
  PRETRAINED="$WORK_DIR/checkpoints/fastvit_sa12_backbone.pth"
fi
VENV_DIR="$(absolute_path "${VENV_DIR:-$WORK_DIR/.venv}")"
DATA_DIR="$(absolute_path "${DATA_DIR:-$WORK_DIR/data/bdd100k}")"
TRAIN_IMG="$(absolute_path "${TRAIN_IMG:-$DATA_DIR/images/100k/train}")"
TRAIN_ANN="$(absolute_path "${TRAIN_ANN:-$DATA_DIR/annotations/bdd100k_det_train_coco.json}")"
VAL_IMG="$(absolute_path "${VAL_IMG:-$DATA_DIR/images/100k/val}")"
VAL_ANN="$(absolute_path "${VAL_ANN:-$DATA_DIR/annotations/bdd100k_det_val_coco.json}")"
[[ -z "$CHECKPOINT" ]] || CHECKPOINT="$(absolute_path "$CHECKPOINT")"
[[ -z "$PRETRAINED" ]] || PRETRAINED="$(absolute_path "$PRETRAINED")"
[[ -z "$WEATHER_LABELS" ]] || WEATHER_LABELS="$(absolute_path "$WEATHER_LABELS")"
if (( SLURM_MODE )) && [[ -z "$LOG_FILE" ]]; then
  # Match the literal SBATCH stdout/stderr path so there is one runtime log.
  LOG_FILE="/datastore/locth/ChienAnh/logs/fastvit_${SLURM_JOB_ID}.log"
fi
[[ -z "$LOG_FILE" ]] || LOG_FILE="$(absolute_path "$LOG_FILE")"
SOURCE_DIR="$WORK_DIR/source"
ENV_PYTHON="$VENV_DIR/bin/python"
TRAIN_ARGS=(
  --data-dir "$DATA_DIR" --train-img "$TRAIN_IMG" --train-ann "$TRAIN_ANN"
  --val-img "$VAL_IMG" --val-ann "$VAL_ANN" --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE" --accum-steps "$ACCUM_STEPS"
  --eval-batch-size "$EVAL_BATCH_SIZE" --workers "$WORKERS"
  --score-thresh "$SCORE_THRESH" --nms-thresh "$NMS_THRESH"
  --max-detections "$MAX_DETECTIONS" --pre-nms-topk "$PRE_NMS_TOPK"
  --eval-interval 1 --log-interval 100 --output "$WORK_DIR/runs" --save-json --no-wandb
)
INIT_ARGS=()
[[ -z "$PRETRAINED" ]] || INIT_ARGS=(--pretrained-backbone "$PRETRAINED")
[[ -z "$MAX_EVAL_SAMPLES" ]] || TRAIN_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
[[ "$MODE" != eval ]] || TRAIN_ARGS+=(--eval-only)
(( ! EPOCH_LOG_ONLY )) || TRAIN_ARGS+=(--epoch-log-only)
if (( POST_TRAIN_EVAL )); then
  IFS=, read -r -a CONFIDENCE_VALUES <<< "$CONFIDENCE_THRESHOLDS"
  (( ${#CONFIDENCE_VALUES[@]} )) || die "--confidence-thresholds cannot be empty"
  TRAIN_ARGS+=(--post-train-eval --confidence-thresholds "${CONFIDENCE_VALUES[@]}")
  [[ -z "$WEATHER_LABELS" ]] || TRAIN_ARGS+=(--weather-labels "$WEATHER_LABELS")
fi

# Use the entrypoints inside the prepared clone, not sibling files beside this
# launcher (Slurm may copy only run_h200.sh into its spool directory).
SOURCE_WRAPPER="$SOURCE_DIR/scripts/train.sh"
WRAPPER_ARGS=(--)
if [[ "$MODE" != train ]]; then
  SOURCE_WRAPPER="$SOURCE_DIR/scripts/resume.sh"
  WRAPPER_ARGS=("$CHECKPOINT" --)
fi
JOB_PREFIX=(env "FASTVIT_PYTHON=$ENV_PYTHON" bash "$SOURCE_WRAPPER" "${WRAPPER_ARGS[@]}")

if (( DRY_RUN )); then
  printf 'DRY RUN: no files, network, Python or GPU operations.\n'
  printf 'Mode: %s; initialization: %s\n' "$MODE" "${PRETRAINED:-${CHECKPOINT:-random weights}}"
  printf 'Offline run: prepared source, venv, dataset and weights only.\n'
  printf 'Source directory: %s (prepared by setup_h200.sh)\n' "$SOURCE_DIR"
  printf 'Work directory: %s; venv: %s; GPU: %s\n' "$WORK_DIR" "$VENV_DIR" "$GPU"
  printf 'SLURM mode: %s; effective batch: %s\n' "$SLURM_MODE" "$(( 10#$BATCH_SIZE * 10#$ACCUM_STEPS ))"
  printf 'Runtime log (file only): %s\n' "${LOG_FILE:-$WORK_DIR/launches/TIMESTAMP/run.log}"
  printf 'Recipe is read from checkpoint at launch (new-run defaults otherwise).\n'
  show_command "${JOB_PREFIX[@]}" "${TRAIN_ARGS[@]}" "${INIT_ARGS[@]}"
  exit 0
fi

# BEGIN_FILE_LOGGING
# Redirect once, before offline preflight. Child processes
# inherit both descriptors; nothing is mirrored to the terminal during a job.
LAUNCH_DIR="$WORK_DIR/launches/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$LAUNCH_DIR"
LOG_FILE="${LOG_FILE:-$LAUNCH_DIR/run.log}"
mkdir -p "$(dirname -- "$LOG_FILE")"
exec >> "$LOG_FILE" 2>&1
printf '\nFastViT H200 launch (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Mode: %s; GPU: %s\nLog file: %s\n' "$MODE" "$GPU" "$LOG_FILE"
if [[ "$MODE" == train ]]; then
  printf 'New training from epoch 0; fresh optimizer/scheduler/scaler; initialization: %s\n' \
    "${PRETRAINED:-random weights (no pretrained backbone)}"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'SLURM job: %s; CPUs/task: %s; requested RAM: %s MB; visible CUDA: %s\n' \
    "$SLURM_JOB_ID" "${SLURM_CPUS_PER_TASK:-unknown}" \
    "${SLURM_MEM_PER_NODE:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}"
fi
# END_FILE_LOGGING

[[ "$OSTYPE" == linux* ]] || die "Run on Linux; --dry-run can be used on other hosts"
(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) )) || die "Bash >=4.4 required"
# Read-only source inspection; this stage never fetches or modifies the clone.
command -v git >/dev/null || die "Missing program: git"
[[ -x "$ENV_PYTHON" && -f "$SOURCE_DIR/train_bdd100k.py" ]] || die "Run setup_h200.sh first (same --work-dir and --venv)"
# Read the version marker without executing an old wrapper: old scripts silently
# ignore extra arguments and could start a different training recipe.
if [[ ! -f "$SOURCE_WRAPPER" ]] || [[ "$(< "$SOURCE_WRAPPER")" != *"# FASTVIT_TRAINER_ARGS_V1"* ]]; then
  die "Prepared source needs updated scripts/train.sh and scripts/resume.sh (FASTVIT_TRAINER_ARGS_V1). Push them to refactor-bdd, then rerun setup_h200.sh before submitting."
fi
if (( EPOCH_LOG_ONLY )) && [[ "$(< "$SOURCE_DIR/train_bdd100k.py")" != *'"--epoch-log-only"'* ]]; then
  die "Prepared source lacks --epoch-log-only. Update train_bdd100k.py, engine/train.py and engine/evaluate.py from refactor-bdd before submitting."
fi
if (( POST_TRAIN_EVAL )); then
  [[ -f "$SOURCE_DIR/engine/evaluation_suite.py" && "$(< "$SOURCE_DIR/train_bdd100k.py")" == *'"--post-train-eval"'* ]] \
    || die "Prepared source needs the final evaluation suite; update source from refactor-bdd before submitting"
fi
[[ -z "$CHECKPOINT" || -f "$CHECKPOINT" ]] || die "Checkpoint does not exist on server: $CHECKPOINT"
if [[ -n "$PRETRAINED" ]]; then
  [[ -s "$PRETRAINED" ]] || die "Missing pretrained backbone: $PRETRAINED; run download_h200.sh first"
fi
[[ -d "$VAL_IMG" && -s "$VAL_ANN" ]] || die "Missing validation data; run download_h200.sh first"
if [[ "$MODE" != eval ]]; then
  [[ -d "$TRAIN_IMG" && -s "$TRAIN_ANN" ]] || die "Missing training data; run download_h200.sh first"
fi
SOURCE_SHA="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
[[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=all)" ]] || die "Prepared source has local changes; use a clean clone"
printf '%s\n' "$SOURCE_SHA" > "$LAUNCH_DIR/source_commit.txt"
printf '%s\n' "$LOG_FILE" > "$LAUNCH_DIR/log_file.txt"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$ENV_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required"'
"$ENV_PYTHON" -m pip check
"$ENV_PYTHON" - "$ALLOW_OTHER_GPU" "$LAUNCH_DIR/gpu.json" <<'PY_GPU'
import json, sys, torch, torchvision
from pathlib import Path
from torchvision.ops import nms
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing CPU fallback")
p = torch.cuda.get_device_properties(0)
if "H200" not in p.name.upper() and sys.argv[1] != "1":
    raise RuntimeError("Selected GPU is not H200 (override: --allow-other-gpu)")
x = torch.randn(32, 32, device="cuda")
assert torch.isfinite(x @ x).all().item()
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
assert nms(boxes, torch.tensor([.9, .8], device="cuda"), .5).numel() == 1
torch.cuda.synchronize()
report = {"name": p.name, "memory_gib": p.total_memory / 2**30,
          "torch": torch.__version__, "torchvision": torchvision.__version__,
          "cuda": torch.version.cuda}
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY_GPU
(cd "$SOURCE_DIR" && "$ENV_PYTHON" train_bdd100k.py --help >/dev/null)

# Validate local weights against the prepared source. No URL or download fallback.
if [[ -n "$PRETRAINED" ]]; then
  (cd "$SOURCE_DIR" && "$ENV_PYTHON" - "$PRETRAINED" "$LAUNCH_DIR/pretrained_backbone.pth" \
    "$LAUNCH_DIR/pretrained.json" "$MODEL" <<'PY_BACKBONE'
import json, sys
from pathlib import Path
import torch
from timm.models import create_model
import models  # noqa: F401

raw_path, output_path, report_path, model_name = sys.argv[1:]
candidate, output_path, report_path = map(Path, (raw_path, output_path, report_path))
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
torch.save({"state_dict": matched}, output_path)
report = {"source_path": str(candidate), "trainer_path": str(output_path),
          "model": model_name, "initialization": "imagenet_backbone_only",
          "start_epoch": 0, "matched_core_parameters": len(required),
          "matched_state_keys": len(matched), "total_backbone_state_keys": len(expected),
          "fresh_backbone_keys": sorted(expected.keys() - matched.keys()),
          "fresh_detection_head": True, "fresh_optimizer_scheduler_scaler": True}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY_BACKBONE
  )
  INIT_ARGS=(--pretrained-backbone "$LAUNCH_DIR/pretrained_backbone.pth")
fi

# Arguments are serialized as NUL-separated data, never evaluated as shell code.
"$ENV_PYTHON" - "$MODE" "$CHECKPOINT" "$EPOCHS" "$BATCH_SIZE" "$ACCUM_STEPS" "$MODEL" "$LAUNCH_DIR" <<'PY_RECIPE'
import json, sys
from pathlib import Path
mode, path, target, batch, accum, model, output = sys.argv[1:]
recipe = {
    "model": model, "architecture_version": "fixed_c5", "fpn_channels": 256,
    "anchor_sizes": [11, 19, 28, 51, 153], "img_size": 800,
    "lr": 1e-4, "weight_decay": .05, "clip_grad": 5.,
    "classification_loss": "qfl", "focal_alpha": .25, "focal_gamma": 2.,
    "scheduler": "step", "warmup_iters": 500, "warmup_epochs": 5,
    "lr_steps": [8, 11], "lr_gamma": .1, "seed": 42, "amp": True,
}
checkpoint_report = None
if path:
    import torch
    print("Loading trusted checkpoint; do not use an untrusted pickle/.pth.")
    c = torch.load(path, map_location="cpu", weights_only=False)
    saved, metadata = c.get("args"), c.get("metadata")
    if not isinstance(saved, dict) or not isinstance(metadata, dict):
        raise ValueError("Checkpoint requires saved args and metadata")
    missing = set(recipe) - set(saved)
    if missing:
        raise ValueError(f"Checkpoint is missing recipe args: {sorted(missing)}")
    recipe.update({key: saved[key] for key in recipe})
    if mode == "resume":
        required = ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict"]
        if recipe["amp"]:
            required.append("scaler_state_dict")
        if any(c.get(key) is None for key in required):
            raise ValueError("Checkpoint lacks full model/optimizer/scheduler/scaler state")
        if not isinstance(c.get("epoch"), int) or c["epoch"] < 0 or int(target) <= c["epoch"] + 1:
            raise ValueError("--epochs must exceed completed checkpoint epochs")
        if recipe["scheduler"] == "cosine" and int(target) != saved.get("epochs"):
            raise ValueError("Keep original --epochs to preserve the cosine schedule")
        old_batch = saved.get("batch_size", 8) * saved.get("accum_steps", 1)
        if old_batch != int(batch) * int(accum):
            print(f"WARNING: effective batch changes {old_batch} -> {int(batch) * int(accum)}; LR is not scaled")
        if (saved.get("batch_size"), saved.get("accum_steps", 1)) != (int(batch), int(accum)):
            print("WARNING: micro-batch/accumulation differ from the checkpoint; "
                  "equal effective batch does not guarantee identical training")
        if (recipe["scheduler"] == "step" and recipe["lr_steps"]
                and int(target) > max(recipe["lr_steps"]) + 1):
            print(f"NOTE: keeping checkpoint LR milestones {recipe['lr_steps']} "
                  f"unchanged for target {target}; extending epochs does not stretch the schedule")
        print(f"Completed epochs: {c['epoch'] + 1}; total target: {target}")
    expected_arch = {
        "model_name": recipe["model"], "architecture_version": recipe["architecture_version"],
        "fpn_channels": recipe["fpn_channels"], "anchor_sizes": recipe["anchor_sizes"],
        "num_classes": 10, "regression_mode": "box_delta", "reg_max": None,
    }
    expected_training = {
        "classification_loss": recipe["classification_loss"],
        "classification_beta": recipe["focal_gamma"],
        "classification_alpha": recipe["focal_alpha"]
            if recipe["classification_loss"] == "binary_quality_focal" else None,
        "regression_loss": "ciou", "matcher": "fixed_iou_0.4_0.5_force_positive",
    }
    for section, expected in (("architecture", expected_arch), ("training", expected_training)):
        actual = metadata.get(section, {})
        for key, value in expected.items():
            if key not in actual or actual[key] != value:
                raise ValueError(f"Unsupported/mismatching checkpoint {section}.{key}")
    checkpoint_report = {"path": path, "epoch": c.get("epoch"),
                         "best_map": c.get("best_map"), "args": saved, "metadata": metadata}
    del c
arguments = []
for key, value in recipe.items():
    if key == "amp":
        arguments.append("--amp" if value else "--no-amp")
    else:
        arguments.append("--" + key.replace("_", "-"))
        arguments.extend(str(v) for v in (value if isinstance(value, list) else [value]))
out = Path(output)
(out / "recipe.args").write_bytes(b"".join(a.encode("utf-8") + b"\0" for a in arguments))
(out / "recipe.json").write_text(json.dumps({
    "recipe": recipe, "checkpoint": checkpoint_report,
    "effective_batch": int(batch) * int(accum),
}, indent=2) + "\n", encoding="utf-8")
PY_RECIPE
mapfile -d '' -t RECIPE_ARGS < "$LAUNCH_DIR/recipe.args"

# Audit existing annotations and sample image paths; never download or convert here.
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
if (( POST_TRAIN_EVAL )); then
  (cd "$SOURCE_DIR" && "$ENV_PYTHON" - "$DATA_DIR" "$VAL_ANN" "$WEATHER_LABELS" \
    "$CONFIDENCE_THRESHOLDS" "$LAUNCH_DIR/weather.json" <<'PY_WEATHER'
import json, sys
from pathlib import Path
from engine.evaluation_suite import prepare_weather_groups
data_dir, val_ann, raw_labels, confidence_csv, report_path = sys.argv[1:]
thresholds = [float(value) for value in confidence_csv.split(",")]
if any(not 0 <= value <= 1 for value in thresholds):
    raise ValueError("Confidence thresholds must be finite and between 0 and 1")
with open(val_ann, encoding="utf-8") as handle:
    images = json.load(handle)["images"]
groups, raw_path = prepare_weather_groups(images, data_dir, raw_labels or None)
report = {"raw_labels": raw_path, "confidence_thresholds": thresholds,
          "weather_images": {key: len(value) for key, value in groups.items()}}
Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY_WEATHER
  )
fi
[[ -z "$MAX_EVAL_SAMPLES" ]] || printf 'WARNING: subset evaluation is not a full-val benchmark.\n'
"$ENV_PYTHON" -m pip freeze > "$LAUNCH_DIR/packages.txt"
JOB=("${JOB_PREFIX[@]}" "${RECIPE_ARGS[@]}" "${TRAIN_ARGS[@]}" "${INIT_ARGS[@]}")
printf '%q ' "${JOB[@]}" > "$LAUNCH_DIR/command.txt"
printf '\n' >> "$LAUNCH_DIR/command.txt"
printf 'Source commit: %s\nLogs/provenance: %s\n' "$SOURCE_SHA" "$LAUNCH_DIR"
show_command "${JOB[@]}"
if (( PREPARE_ONLY )); then
  printf 'Offline preflight complete; no training/evaluation started.\n'
  exit 0
fi

set +e
(cd "$SOURCE_DIR" && "${JOB[@]}")
JOB_STATUS=$?
set -e
printf '%s\n' "$JOB_STATUS" > "$LAUNCH_DIR/exit_code.txt"
printf 'Training/evaluation finished with exit code: %s\n' "$JOB_STATUS"
exit "$JOB_STATUS"
