#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${RAYTREK4090_HOST:-raytrek4090}"
REMOTE_DIR="${RAYTREK4090_MAGENTA_DIR:-workspaces/magenta-realtime}"
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

usage() {
  cat <<'EOF'
Usage:
  scripts/raytrek4090_magenta_dark_ambient_batch.sh [options]

Options:
  --remote-host HOST       SSH host/alias (default: raytrek4090)
  --remote-dir PATH        Remote checkout path (default: workspaces/magenta-realtime under remote home)
  --source-dir RELPATH     MIDI source dir relative to repo root
  --output-dir RELPATH     Output dir relative to repo root
  --model NAME             JAX checkpoint model name (default: mrt2_small)
  --duration SECONDS       Render duration (default: 256.0)
  --chunk-seconds SECONDS  JAX control-update chunk size (default: 1.0)
  --limit N                Smoke-test first N takes; 0 means all
  --skip-sync              Do not rsync local repo to raytrek before running
  --skip-install           Do not create/update the remote venv or JAX deps
  --skip-model-download    Do not run mrt models/checkpoints downloads
  --dry-run                Write prompts/weight JSON/manifest only; no model load
  --no-run                 Only preflight and sync code; do not execute batch
  --no-sync-back           Do not rsync generated output back to local machine
  -h, --help               Show this help

Examples:
  # Verify the remote path and copy code without generating audio.
  scripts/raytrek4090_magenta_dark_ambient_batch.sh --dry-run --limit 1 --skip-install --skip-model-download

  # Full Raytrek JAX/CUDA render and sync back.
  scripts/raytrek4090_magenta_dark_ambient_batch.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="$2"; shift 2 ;;
    --remote-dir)
      REMOTE_DIR="$2"; shift 2 ;;
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

if [[ ! -d "$ROOT/$SOURCE_DIR_REL" ]]; then
  echo "Missing source MIDI dir: $ROOT/$SOURCE_DIR_REL" >&2
  exit 1
fi

echo "== Raytrek preflight =="
if [[ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
  /Applications/Tailscale.app/Contents/MacOS/Tailscale ping -c 1 100.87.26.1 || {
    echo "Tailscale ping failed. Start Tailscale and retry." >&2
    exit 1
  }
fi
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$REMOTE_HOST" \
  'hostname && whoami && pwd && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true'

if [[ "$SKIP_SYNC" != "1" ]]; then
  echo "== Sync repo to $REMOTE_HOST:$REMOTE_DIR =="
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
  rsync -az --progress \
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
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$REMOTE_HOST" 'bash -s' -- \
  "$REMOTE_DIR" "$SOURCE_DIR_REL" "$OUTPUT_DIR_REL" "$MODEL" "$DURATION" "$CHUNK_SECONDS" \
  "$LIMIT" "$SKIP_INSTALL" "$SKIP_MODELS" "$DRY_RUN" <<'REMOTE'
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

cd "$REMOTE_DIR"
export PATH="$HOME/.local/bin:$PATH"
export MAGENTA_HOME="${MAGENTA_HOME:-$HOME/Documents/Magenta/magenta-rt-v2}"
mkdir -p "$MAGENTA_HOME"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  UV="$(command -v uv || true)"
  if [[ -z "$UV" ]]; then
    echo "uv was not found on remote PATH. Expected \$HOME/.local/bin/uv." >&2
    exit 1
  fi
  "$UV" venv --python 3.11 .venv-raytrek
  # shellcheck disable=SC1091
  source .venv-raytrek/bin/activate
  "$UV" pip install -e ".[jax]"
  "$UV" pip install -U "jax[cuda12]"
  PYTHON="python"
else
  if [[ -x .venv-raytrek/bin/python ]]; then
    PYTHON=".venv-raytrek/bin/python"
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
  rsync -az --progress "$REMOTE_HOST:$REMOTE_DIR/$OUTPUT_DIR_REL/" "$ROOT/$OUTPUT_DIR_REL/"
fi

echo "Done."
