#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${RAYTREK4090_HOST:-raytrek4090}"
REMOTE_DIR="${RAYTREK4090_MAGENTA_DIR:-workspaces/magenta-realtime}"
REMOTE_EXEC_MODE="${RAYTREK4090_EXEC_MODE:-auto}"
WSL_DISTRO="${RAYTREK4090_WSL_DISTRO:-}"
WSL_USER="${RAYTREK4090_WSL_USER:-}"
JAX_CUDA_EXTRA="${RAYTREK4090_JAX_CUDA_EXTRA:-cuda12}"
SOURCE_DIR_REL="assets/dark_ambient_harmonies_128bars"
OUTPUT_DIR_REL="outputs/polyrhythm_prompt_modulation/dark_ambient_harmonies_128bars_prompt2_jax_cuda_16"
MODEL="mrt2_small"
DURATION="256.0"
CHUNK_SECONDS="1.0"
LIMIT="0"
SKIP_SYNC="0"
SKIP_INSTALL="0"
SKIP_MODELS="0"
DRY_RUN="0"
NO_RUN="0"
NO_SYNC_BACK="0"
AUTO_INSTALL_UV="1"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
SSH_BATCH_OPTS=(-o BatchMode=yes "${SSH_OPTS[@]}")
WSL_PREFIX=(wsl.exe)

usage() {
  cat <<'EOF'
Usage:
  scripts/raytrek4090_magenta_dark_ambient_batch.sh [options]

Options:
  --remote-host HOST       SSH host/alias (default: raytrek4090)
  --remote-dir PATH        Remote checkout path; relative paths are resolved under remote $HOME
  --remote-exec MODE       auto, linux, or windows-wsl (default: auto)
  --wsl-distro NAME        WSL distro for windows-wsl mode
  --wsl-user USER          WSL Linux user for windows-wsl mode
  --jax-cuda-extra NAME    JAX CUDA extra to install, e.g. cuda12 or cuda13 (default: cuda12)
  --source-dir RELPATH     MIDI source dir relative to repo root
  --output-dir RELPATH     Output dir relative to repo root
  --model NAME             JAX checkpoint model name (default: mrt2_small)
  --duration SECONDS       Render duration (default: 256.0)
  --chunk-seconds SECONDS  JAX control-update chunk size (default: 1.0)
  --limit N                Smoke-test first N takes; 0 means all
  --skip-sync              Do not rsync local repo to raytrek before running
  --skip-install           Do not create/update the remote venv or JAX deps
  --skip-uv-bootstrap      Do not install uv automatically when it is missing
  --skip-model-download    Do not run mrt models/checkpoints downloads
  --dry-run                Write prompts/weight JSON/manifest only; no model load
  --no-run                 Only preflight and sync code; do not execute batch
  --no-sync-back           Do not rsync generated output back to local machine
  -h, --help               Show this help

Environment:
  RAYTREK4090_EXEC_MODE    Same as --remote-exec
  RAYTREK4090_WSL_DISTRO   Same as --wsl-distro
  RAYTREK4090_WSL_USER     Same as --wsl-user
  RAYTREK4090_JAX_CUDA_EXTRA
                            Same as --jax-cuda-extra

Examples:
  # Windows SSH host, run everything inside WSL2 Linux + CUDA + JAX.
  scripts/raytrek4090_magenta_dark_ambient_batch.sh \
    --remote-exec windows-wsl --wsl-distro Ubuntu --dry-run --limit 1

  # Direct Linux SSH host, full Raytrek JAX/CUDA render and sync back.
  scripts/raytrek4090_magenta_dark_ambient_batch.sh --remote-exec linux
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

warn() {
  echo "warning: $*" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="$2"; shift 2 ;;
    --remote-dir)
      REMOTE_DIR="$2"; shift 2 ;;
    --remote-exec)
      REMOTE_EXEC_MODE="$2"; shift 2 ;;
    --wsl-distro)
      WSL_DISTRO="$2"; shift 2 ;;
    --wsl-user)
      WSL_USER="$2"; shift 2 ;;
    --jax-cuda-extra)
      JAX_CUDA_EXTRA="$2"; shift 2 ;;
    --source-dir)
      SOURCE_DIR_REL="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR_REL="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --duration)
      DURATION="$2"; shift 2 ;;
    --chunk-seconds)
      CHUNK_SECONDS="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --skip-sync)
      SKIP_SYNC="1"; shift ;;
    --skip-install)
      SKIP_INSTALL="1"; shift ;;
    --skip-uv-bootstrap)
      AUTO_INSTALL_UV="0"; shift ;;
    --skip-model-download)
      SKIP_MODELS="1"; shift ;;
    --dry-run)
      DRY_RUN="1"; shift ;;
    --no-run)
      NO_RUN="1"; shift ;;
    --no-sync-back)
      NO_SYNC_BACK="1"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

case "$REMOTE_EXEC_MODE" in
  auto|linux|windows-wsl) ;;
  *) die "--remote-exec must be auto, linux, or windows-wsl" ;;
esac

if [[ -z "$JAX_CUDA_EXTRA" ]]; then
  die "--jax-cuda-extra must not be empty"
fi

if [[ ! -d "$ROOT/$SOURCE_DIR_REL" ]]; then
  die "Missing source MIDI dir: $ROOT/$SOURCE_DIR_REL"
fi

build_wsl_prefix() {
  WSL_PREFIX=(wsl.exe)
  if [[ -n "$WSL_DISTRO" ]]; then
    WSL_PREFIX+=(-d "$WSL_DISTRO")
  fi
  if [[ -n "$WSL_USER" ]]; then
    WSL_PREFIX+=(-u "$WSL_USER")
  fi
  WSL_PREFIX+=(-e)
}

remote_exec() {
  if [[ "$REMOTE_EXEC_MODE" == "windows-wsl" ]]; then
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "${WSL_PREFIX[@]}" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
  fi
}

remote_bash() {
  if [[ "$REMOTE_EXEC_MODE" == "windows-wsl" ]]; then
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "${WSL_PREFIX[@]}" bash -s -- "$@"
  else
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- "$@"
  fi
}

tailscale_ping_check() {
  if [[ ! -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
    return
  fi

  set +e
  local output
  output="$(/Applications/Tailscale.app/Contents/MacOS/Tailscale ping -c 1 100.87.26.1 2>&1)"
  local status=$?
  set -e

  printf '%s\n' "$output"
  if ! grep -q "pong from" <<<"$output"; then
    die "Tailscale ping did not reach raytrek4090. Start Tailscale and retry."
  fi
  if [[ "$status" -ne 0 ]]; then
    warn "Tailscale reached raytrek4090, but not by direct path; continuing with SSH."
  fi
}

detect_remote_exec_mode() {
  if [[ "$REMOTE_EXEC_MODE" != "auto" ]]; then
    return
  fi

  set +e
  local linux_uname
  linux_uname="$(ssh "${SSH_BATCH_OPTS[@]}" "$REMOTE_HOST" uname -s 2>/dev/null)"
  local linux_status=$?
  set -e
  if [[ "$linux_status" -eq 0 && "$linux_uname" == *Linux* ]]; then
    REMOTE_EXEC_MODE="linux"
    return
  fi

  set +e
  local wsl_uname
  wsl_uname="$(ssh "${SSH_BATCH_OPTS[@]}" "$REMOTE_HOST" "${WSL_PREFIX[@]}" uname -s 2>/dev/null)"
  local wsl_status=$?
  set -e
  if [[ "$wsl_status" -eq 0 && "$wsl_uname" == *Linux* ]]; then
    REMOTE_EXEC_MODE="windows-wsl"
    return
  fi

  die "Could not detect remote execution mode. Try --remote-exec linux or --remote-exec windows-wsl."
}

resolve_remote_dir() {
  local remote_home
  remote_home="$(remote_bash <<'REMOTE'
printf '%s\n' "$HOME"
REMOTE
)"

  case "$REMOTE_DIR" in
    /*)
      ;;
    ~/*)
      REMOTE_DIR="${remote_home%/}/${REMOTE_DIR#~/}"
      ;;
    *)
      REMOTE_DIR="${remote_home%/}/$REMOTE_DIR"
      ;;
  esac
}

rsync_remote_args() {
  if [[ "$REMOTE_EXEC_MODE" == "windows-wsl" ]]; then
    printf '%s\n' "--rsync-path=${WSL_PREFIX[*]} rsync"
  fi
}

build_wsl_prefix

echo "== Raytrek preflight =="
tailscale_ping_check
detect_remote_exec_mode
echo "remote_exec_mode=$REMOTE_EXEC_MODE"

remote_bash <<'REMOTE'
set -euo pipefail
echo "hostname=$(hostname)"
echo "user=$(whoami)"
echo "home=$HOME"
echo "kernel=$(uname -srm)"
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  echo "wsl_distro=$WSL_DISTRO_NAME"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi=missing"
fi
if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  echo "python3=missing"
fi
if command -v uv >/dev/null 2>&1; then
  uv --version
else
  echo "uv=missing"
fi
if command -v rsync >/dev/null 2>&1; then
  rsync --version | head -1
else
  echo "rsync=missing"
fi
REMOTE

resolve_remote_dir
echo "remote_dir=$REMOTE_DIR"

if [[ "$SKIP_SYNC" != "1" ]]; then
  echo "== Sync repo to $REMOTE_HOST:$REMOTE_DIR =="
  remote_bash <<'REMOTE'
command -v rsync >/dev/null 2>&1 || {
  echo "remote rsync is required for sync. Install it inside the Linux/WSL environment or rerun with --skip-sync." >&2
  exit 1
}
REMOTE
  remote_exec mkdir -p "$REMOTE_DIR"
  RSYNC_ARGS=()
  if [[ "$REMOTE_EXEC_MODE" == "windows-wsl" ]]; then
    while IFS= read -r arg; do
      RSYNC_ARGS+=("$arg")
    done < <(rsync_remote_args)
  fi
  rsync -az --progress \
    "${RSYNC_ARGS[@]}" \
    --exclude='.git/' \
    --exclude='.venv*/' \
    --exclude='node_modules/' \
    --exclude='build/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='*.mov' \
    --exclude='*.mp4' \
    --exclude='*.wav' \
    "$ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
fi

if [[ "$NO_RUN" == "1" ]]; then
  echo "NO_RUN=1: finished preflight/sync only."
  exit 0
fi

echo "== Remote JAX batch =="
remote_bash \
  "$REMOTE_DIR" "$SOURCE_DIR_REL" "$OUTPUT_DIR_REL" "$MODEL" "$DURATION" "$CHUNK_SECONDS" \
  "$LIMIT" "$SKIP_INSTALL" "$SKIP_MODELS" "$DRY_RUN" "$JAX_CUDA_EXTRA" "$AUTO_INSTALL_UV" <<'REMOTE'
set -euo pipefail

REMOTE_DIR="$1"
SOURCE_DIR_REL="$2"
OUTPUT_DIR_REL="$3"
MODEL="$4"
DURATION="$5"
CHUNK_SECONDS="$6"
LIMIT="$7"
SKIP_INSTALL="$8"
SKIP_MODELS="$9"
DRY_RUN="${10}"
JAX_CUDA_EXTRA="${11}"
AUTO_INSTALL_UV="${12}"

cd "$REMOTE_DIR"
export PATH="$HOME/.local/bin:$PATH"
export MAGENTA_HOME="${MAGENTA_HOME:-$HOME/Documents/Magenta/magenta-rt-v2}"
mkdir -p "$MAGENTA_HOME"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  UV="$(command -v uv || true)"
  if [[ -z "$UV" && "$AUTO_INSTALL_UV" == "1" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
      UV="$(command -v uv || true)"
    fi
  fi
  if [[ -z "$UV" ]]; then
    echo "uv was not found on remote PATH. Install uv or rerun without --skip-uv-bootstrap." >&2
    exit 1
  fi
  "$UV" venv --python 3.11 .venv-raytrek
  # shellcheck disable=SC1091
  source .venv-raytrek/bin/activate
  "$UV" pip install -e ".[jax]"
  "$UV" pip install -U "jax[$JAX_CUDA_EXTRA]"
  PYTHON="python"
else
  if [[ -x .venv-raytrek/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv-raytrek/bin/activate
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" - <<'PY'
import jax
devices = [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
print("JAX devices:", devices)
if not any(d.platform == "gpu" for d in jax.devices()):
    raise SystemExit("No JAX GPU device is visible")
PY
fi

if [[ "$DRY_RUN" != "1" && "$SKIP_MODELS" != "1" ]]; then
  command -v mrt >/dev/null 2>&1 || {
    echo "mrt CLI was not found after environment setup." >&2
    exit 1
  }
  mrt models init --download-path "$MAGENTA_HOME"
  mrt checkpoints download "${MODEL}.safetensors" --download-path "$MAGENTA_HOME"
fi

args=(
  "$PYTHON" experiments/polyrhythm_prompt_modulation/run_dark_ambient_harmony_jax_batch.py
  --source-dir "$SOURCE_DIR_REL"
  --output-dir "$OUTPUT_DIR_REL"
  --model "$MODEL"
  --duration "$DURATION"
  --chunk-seconds "$CHUNK_SECONDS"
)
if [[ "$LIMIT" != "0" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

echo "+ ${args[*]}"
"${args[@]}"
REMOTE

if [[ "$NO_SYNC_BACK" != "1" ]]; then
  echo "== Sync output back to local $ROOT/$OUTPUT_DIR_REL =="
  mkdir -p "$ROOT/$OUTPUT_DIR_REL"
  RSYNC_ARGS=()
  if [[ "$REMOTE_EXEC_MODE" == "windows-wsl" ]]; then
    while IFS= read -r arg; do
      RSYNC_ARGS+=("$arg")
    done < <(rsync_remote_args)
  fi
  rsync -az --progress "${RSYNC_ARGS[@]}" "$REMOTE_HOST:$REMOTE_DIR/$OUTPUT_DIR_REL/" "$ROOT/$OUTPUT_DIR_REL/"
fi

echo "Done."
