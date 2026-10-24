#!/usr/bin/env bash
#SBATCH --job-name=fastvit_resume32
#SBATCH --output=/datastore/locth/ChienAnh/logs/fastvit_resume_%j.log
#SBATCH --error=/datastore/locth/ChienAnh/logs/fastvit_resume_%j.log
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
# Create ROOT/logs before sbatch; Slurm opens its log before running Bash.
# Uses the prepared launcher/source/venv; safe from Slurm's spool directory.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  cd /datastore/locth/ChienAnh
  mkdir -p logs
  sbatch ./resume_h200.sh

Resume only: WORK_DIR/runs/fastvit_sa12_20261017_165236/last.pth
Target: 50 TOTAL epochs. If 32 epochs were saved, 18 remain.
The actual start epoch is read from the checkpoint, never hard-coded.

  --work-dir DIR       Default: /datastore/locth/ChienAnh
  --launcher FILE      Default: WORK_DIR/run_h200.sh (already copied to server)
  --checkpoint FILE    Override checkpoint; e.g. a later resumed run's last.pth
  --epochs N           Total target, not extra epochs; default 50
  --log-file FILE      Append stdout/stderr to this file
                       SLURM: ROOT/logs/fastvit_resume_JOBID.log
                       Direct Bash: WORK_DIR/logs/fastvit_resume_TIMESTAMP_PID.log
  --prepare-only       Validate checkpoint/data/GPU without training
  --epoch-log-only     Default: epoch summaries without batch logs/progress
  --no-epoch-log-only  Enable detailed batch logs/progress for diagnostics
  --post-train-eval    Default: final best-checkpoint analysis suite
  --no-post-train-eval Skip final overall/weather/confidence analysis
  --weather-labels FILE --confidence-thresholds CSV
  --dry-run            Show command; no writes, Python, network or GPU checks
  -h, --help

Forwarded to run_h200.sh:
  --venv DIR --data-dir DIR --train-img DIR --train-ann FILE
  --val-img DIR --val-ann FILE
  --batch-size N --accum-steps N --eval-batch-size N --workers N
  --score-thresh FLOAT --nms-thresh FLOAT --max-detections N --pre-nms-topk N
  --max-eval-samples N --allow-other-gpu
  --gpu DEVICE (direct Bash in an allocated session only, never under SLURM)

Defaults: batch 4, accumulation 4, eval batch 2, workers 2.
The existing launcher validates full model/optimizer/scheduler/AMP state and
inherits architecture/loss/LR recipe from the checkpoint.
Uses WORK_DIR/source/scripts/resume.sh and source/train_bdd100k.py.
New checkpoints go to a new run directory under WORK_DIR/runs.
USAGE
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
show_command() { printf '+ '; printf '%q ' "$@"; printf '\n'; }

WORK_DIR=/datastore/locth/ChienAnh
LAUNCHER=
CHECKPOINT=
EPOCHS=50
LOG_FILE=
DRY_RUN=0
EXTRA_ARGS=()
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --prepare-only|--allow-other-gpu) EXTRA_ARGS+=("$1"); shift ;;
    --epoch-log-only|--no-epoch-log-only) EXTRA_ARGS+=("$1"); shift ;;
    --post-train-eval|--no-post-train-eval) EXTRA_ARGS+=("$1"); shift ;;
    --work-dir|--launcher|--checkpoint|--epochs|--log-file)
      need_value "$@"
      case "$1" in
        --work-dir) WORK_DIR="$2" ;; --launcher) LAUNCHER="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;; --epochs) EPOCHS="$2" ;;
        --log-file) LOG_FILE="$2" ;;
      esac
      shift 2 ;;
    --venv|--data-dir|--train-img|--train-ann|--val-img|--val-ann|\
    --batch-size|--accum-steps|--eval-batch-size|--workers|--gpu|\
    --score-thresh|--nms-thresh|--max-detections|--pre-nms-topk|--max-eval-samples|\
    --weather-labels|--confidence-thresholds)
      need_value "$@"
      EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    *) die "Unknown argument: $1; this script only resumes training (see --help)" ;;
  esac
done
[[ "$EPOCHS" =~ ^[0-9]+$ ]] && (( 10#$EPOCHS > 0 )) || die "--epochs must be a positive integer"
WORK_DIR="$(absolute_path "$WORK_DIR")"
LAUNCHER="$(absolute_path "${LAUNCHER:-$WORK_DIR/run_h200.sh}")"
CHECKPOINT="$(absolute_path "${CHECKPOINT:-$WORK_DIR/runs/fastvit_sa12_20261017_165236/last.pth}")"
if [[ -n "${SLURM_JOB_ID:-}" && -z "$LOG_FILE" ]]; then
  # Match the literal SBATCH paths even when --work-dir is overridden.
  LOG_FILE="/datastore/locth/ChienAnh/logs/fastvit_resume_${SLURM_JOB_ID}.log"
elif [[ -z "$LOG_FILE" ]]; then
  if (( DRY_RUN )); then
    LOG_FILE="$WORK_DIR/logs/fastvit_resume_TIMESTAMP_PID.log"
  else
    LOG_FILE="$WORK_DIR/logs/fastvit_resume_$(date -u +%Y%m%dT%H%M%S)_$$.log"
  fi
fi
LOG_FILE="$(absolute_path "$LOG_FILE")"
JOB=(bash "$LAUNCHER" --mode resume --checkpoint "$CHECKPOINT"
  --epochs "$EPOCHS" --work-dir "$WORK_DIR" --log-file "$LOG_FILE"
  --batch-size 4 --accum-steps 4 --eval-batch-size 2 --workers 2
  --epoch-log-only
  --post-train-eval
  "${EXTRA_ARGS[@]}")
if (( DRY_RUN )); then
  printf 'DRY RUN: resume only; no writes, Python, network or GPU operations.\n'
  printf 'Checkpoint: %s\nTotal target epochs: %s\n' "$CHECKPOINT" "$EPOCHS"
  printf 'Actual start epoch is read from the checkpoint by run_h200.sh.\n'
  printf 'Runtime log (file only): %s\n' "$LOG_FILE"
  show_command "${JOB[@]}" --dry-run
  exit 0
fi

mkdir -p "$(dirname -- "$LOG_FILE")"
exec >> "$LOG_FILE" 2>&1
printf '\nH200 resume launch (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Checkpoint: %s\nTotal target epochs: %s\n' "$CHECKPOINT" "$EPOCHS"
[[ -f "$LAUNCHER" ]] || die "Missing launcher: $LAUNCHER; copy run_h200.sh to WORK_DIR or pass --launcher"
[[ -s "$CHECKPOINT" ]] || die "Checkpoint missing or empty: $CHECKPOINT"
show_command "${JOB[@]}"
# Keep Slurm's GPU binding; return the launcher's/trainer's exit code unchanged.
exec "${JOB[@]}"
