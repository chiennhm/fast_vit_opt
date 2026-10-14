#!/usr/bin/env bash
# Stage 1: clone/update refactor-bdd and prepare the Python environment (no GPU).
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash setup_h200.sh [options]
Run this before download_h200.sh and run_h200.sh.
  --work-dir DIR       Default: /datastore/locth/ChienAnh
  --venv DIR           Default: WORK_DIR/.venv
  --repo-url URL       Default: https://github.com/chiennhm/fast_vit_opt.git
  --branch NAME        Default: refactor-bdd
  --commit FULL_SHA    Optional full 40-character source commit
  --skip-install       Reuse venv; still fetch source and check dependencies
  --log-file FILE      Default: WORK_DIR/logs/setup_h200.log (append, file only)
  --dry-run            Show plan without writes/network/Python
  -h, --help
Requires Linux, Bash >=4.4, Git and Python >=3.10 with venv.
No GPU allocation or dataset is needed. Run where cluster policy allows setup.
Do not update this shared source/venv while another job is using it.
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
REPO_URL=https://github.com/chiennhm/fast_vit_opt.git
BRANCH=refactor-bdd
COMMIT=
LOG_FILE=
SKIP_INSTALL=0
DRY_RUN=0
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --work-dir|--venv|--repo-url|--branch|--commit|--log-file)
      need_value "$@"
      case "$1" in
        --work-dir) WORK_DIR="$2" ;; --venv) VENV_DIR="$2" ;;
        --repo-url) REPO_URL="$2" ;; --branch) BRANCH="$2" ;;
        --commit) COMMIT="$2" ;; --log-file) LOG_FILE="$2" ;;
      esac
      shift 2 ;;
    *) die "Unknown argument: $1 (see --help)" ;;
  esac
done
[[ "$BRANCH" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "$BRANCH" != *..* ]] || die "Invalid branch"
[[ -z "$COMMIT" || "$COMMIT" =~ ^[a-fA-F0-9]{40}$ ]] || die "Commit must be a full 40-character SHA"
if [[ "$REPO_URL" == https://* ]]; then
  [[ "$REPO_URL" != *"@"* && "$REPO_URL" != *"?"* && "$REPO_URL" != *"#"* ]] || die "Do not put credentials in the repo URL"
else
  [[ "$REPO_URL" == git@*:* ]] || die "Use HTTPS or git@host:path SSH URL"
fi

WORK_DIR="$(absolute_path "$WORK_DIR")"
VENV_DIR="$(absolute_path "${VENV_DIR:-$WORK_DIR/.venv}")"
LOG_FILE="$(absolute_path "${LOG_FILE:-$WORK_DIR/logs/setup_h200.log}")"
SOURCE_DIR="$WORK_DIR/source"
ENV_PYTHON="$VENV_DIR/bin/python"
if (( DRY_RUN )); then
  printf 'DRY RUN: setup only; no writes/network/Python/GPU operations.\n'
  printf 'Source: %s; branch: %s; commit: %s\n' "$REPO_URL" "$BRANCH" "${COMMIT:-branch HEAD}"
  printf 'Clone: %s; venv: %s; skip install: %s\n' "$SOURCE_DIR" "$VENV_DIR" "$SKIP_INSTALL"
  printf 'Runtime log (file only): %s\n' "$LOG_FILE"
  exit 0
fi

mkdir -p "$(dirname -- "$LOG_FILE")"
exec >> "$LOG_FILE" 2>&1
printf '\nH200 setup (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ "$OSTYPE" == linux* ]] || die "Run on Linux; --dry-run can be used on other hosts"
(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) )) || die "Bash >=4.4 required"
for program in git python3; do command -v "$program" >/dev/null || die "Missing program: $program"; done
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required"'
export PYTHONUNBUFFERED=1 GIT_TERMINAL_PROMPT=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LAUNCH_DIR="$WORK_DIR/setups/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$LAUNCH_DIR"
if [[ -e "$SOURCE_DIR" ]]; then
  repo_root="$(git -C "$SOURCE_DIR" rev-parse --show-toplevel)"
  [[ "$repo_root" == "$(cd "$SOURCE_DIR" && pwd -P)" ]] || die "Source is not a standalone clone"
  [[ "$(git -C "$SOURCE_DIR" remote get-url origin)" == "$REPO_URL" ]] || die "Existing clone has a different origin"
  [[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=all)" ]] || die "Existing clone has local changes; refusing overwrite"
else
  git clone --single-branch --branch "$BRANCH" -- "$REPO_URL" "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch origin "${COMMIT:-refs/heads/$BRANCH}"
git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
SOURCE_SHA="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
[[ -z "$COMMIT" || "${SOURCE_SHA,,}" == "${COMMIT,,}" ]] || die "Fetched SHA mismatch"
[[ -f "$SOURCE_DIR/train_bdd100k.py" && -f "$SOURCE_DIR/requirements.txt" ]] || die "Missing trainer/requirements in clone"
for entrypoint in train.sh resume.sh; do
  wrapper="$SOURCE_DIR/scripts/$entrypoint"
  if [[ ! -f "$wrapper" ]] || [[ "$(< "$wrapper")" != *"# FASTVIT_TRAINER_ARGS_V1"* ]]; then
    die "Source scripts/$entrypoint needs FASTVIT_TRAINER_ARGS_V1 support. Push the updated training wrappers to refactor-bdd, then rerun setup."
  fi
done

printf '%s\n' "$SOURCE_SHA" > "$LAUNCH_DIR/source_commit.txt"
printf '%s\n' "$LOG_FILE" > "$LAUNCH_DIR/log_file.txt"
if [[ ! -x "$ENV_PYTHON" ]]; then
  (( SKIP_INSTALL == 0 )) || die "--skip-install requires an existing venv/bin/python"
  python3 -m venv "$VENV_DIR"
fi
if (( SKIP_INSTALL == 0 )); then
  printf 'torch==2.9.1\ntorchvision==0.24.1\n' > "$WORK_DIR/torch-constraints.txt"
  "$ENV_PYTHON" -m pip install --upgrade pip
  "$ENV_PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128
  "$ENV_PYTHON" -m pip install -r "$SOURCE_DIR/requirements.txt" \
    -c "$WORK_DIR/torch-constraints.txt"
fi
"$ENV_PYTHON" -m pip check
# Import check only; CUDA is checked later inside the allocated training job.
(cd "$SOURCE_DIR" && "$ENV_PYTHON" train_bdd100k.py --help >/dev/null)
"$ENV_PYTHON" -m pip freeze > "$LAUNCH_DIR/packages.txt"
printf 'Setup complete. Source commit: %s\nVenv: %s\nNext: bash download_h200.sh\n' "$SOURCE_SHA" "$VENV_DIR"
