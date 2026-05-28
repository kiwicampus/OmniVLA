#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-config_nav/train_omnivla.yaml}"

usage() {
  cat <<'EOF'
OmniVLA training helper.

Usage:
  bash scripts/omnivla_training.sh <command> [args...]

Commands:
  setup      Create/update a conda training environment
  check      Run static repo checks that do not require GPU
  doctor     Validate config, imports, torchrun, CUDA, and GPU readiness
  inspect    Inspect the dataset described by CONFIG
  smoke      Run doctor and one optimizer step
  train      Run full training

Common environment variables:
  CONFIG=config_nav/train_omnivla.yaml
  PYTHON_BIN=python
  TORCHRUN_BIN=torchrun
  NPROC_PER_NODE=1
  SKIP_DOCTOR=0

Setup environment variables:
  ENV_NAME=omnivla
  PYTHON_VERSION=3.10
  TORCH_VERSION=2.7.0
  TORCHVISION_VERSION=0.22.0
  TORCHAUDIO_VERSION=2.7.0
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
  LEROBOT_VERSION=0.4.3
  RUN_STATIC_CHECK=1
  RUN_FULL_DOCTOR=0

Examples:
  bash scripts/omnivla_training.sh setup
  bash scripts/omnivla_training.sh check
  bash scripts/omnivla_training.sh inspect --sample-idx 100
  bash scripts/omnivla_training.sh smoke
  NPROC_PER_NODE=2 bash scripts/omnivla_training.sh train
EOF
}

cd "$ROOT_DIR"

detect_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python executable found on PATH. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
}

detect_torchrun() {
  if [[ -n "${TORCHRUN_BIN:-}" ]]; then
    return
  fi
  detect_python
  local python_exe sibling_torchrun
  python_exe="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
  sibling_torchrun="$(dirname "$python_exe")/torchrun"
  if [[ -x "$sibling_torchrun" ]]; then
    TORCHRUN_BIN="$sibling_torchrun"
  elif command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_BIN="torchrun"
  else
    echo "No torchrun executable found. Activate the training env or set TORCHRUN_BIN=/path/to/torchrun." >&2
    exit 1
  fi
}

cmd_setup() {
  local env_name python_version torch_version torchvision_version torchaudio_version torch_index_url lerobot_version
  env_name="${ENV_NAME:-omnivla}"
  python_version="${PYTHON_VERSION:-3.10}"
  torch_version="${TORCH_VERSION:-2.7.0}"
  torchvision_version="${TORCHVISION_VERSION:-0.22.0}"
  torchaudio_version="${TORCHAUDIO_VERSION:-2.7.0}"
  torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  lerobot_version="${LEROBOT_VERSION:-0.4.3}"

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required. Install Miniconda/Mambaforge first, then rerun setup." >&2
    exit 1
  fi

  if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "[setup] Creating conda env: $env_name (python=$python_version)"
    conda create -n "$env_name" "python=$python_version" -y
  else
    echo "[setup] Reusing existing conda env: $env_name"
  fi

  run_in_env() {
    conda run -n "$env_name" "$@"
  }

  echo "[setup] Upgrading packaging tools"
  run_in_env python -m pip install --upgrade pip setuptools wheel packaging

  echo "[setup] Installing PyTorch CUDA wheels from $torch_index_url"
  run_in_env python -m pip install \
    --index-url "$torch_index_url" \
    "torch==$torch_version" \
    "torchvision==$torchvision_version" \
    "torchaudio==$torchaudio_version"

  echo "[setup] Installing OmniVLA editable package"
  run_in_env python -m pip install -e .

  echo "[setup] Installing LeRobot without dependency rewrites"
  run_in_env python -m pip install --no-deps "lerobot==$lerobot_version"

  if [[ "${RUN_STATIC_CHECK:-1}" == "1" ]]; then
    echo "[setup] Running static repo checks"
    run_in_env bash scripts/omnivla_training.sh check
  fi

  if [[ "${RUN_FULL_DOCTOR:-0}" == "1" ]]; then
    echo "[setup] Running full training doctor"
    run_in_env python vla-scripts/doctor_training_env.py
  else
    cat <<EOF

[setup] Environment created.

Activate it with:
  conda activate $env_name

Then run:
  bash scripts/omnivla_training.sh doctor
  bash scripts/omnivla_training.sh smoke
EOF
  fi
}

cmd_check() {
  detect_python
  echo "[check] Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
  echo "[check] Unit tests"
  "$PYTHON_BIN" -m unittest discover -s tests -v

  echo "[check] Static config/manifest doctor"
  "$PYTHON_BIN" vla-scripts/doctor_training_env.py --config "$CONFIG" --static-only

  echo "[check] Compile changed training scripts"
  "$PYTHON_BIN" -m py_compile \
    vla-scripts/train_omnivla.py \
    vla-scripts/doctor_training_env.py \
    vla-scripts/inspect_training_dataset.py \
    omnivla_training/episode_manifest.py \
    omnivla_training/training_scripts/curate_episodes.py \
    omnivla_training/training_scripts/vertex_train_omnivla_wrapper.py \
    omnivla_training/vertex/submit_training_job.py

  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[check] Whitespace check"
    git diff --check
  fi

  echo "[check] Static checks passed"
}

cmd_doctor() {
  detect_python
  "$PYTHON_BIN" vla-scripts/doctor_training_env.py --config "$CONFIG" "$@"
}

cmd_inspect() {
  detect_python
  "$PYTHON_BIN" vla-scripts/inspect_training_dataset.py --config "$CONFIG" "$@"
}

run_torch_training() {
  local smoke_flag=("$@")
  detect_python
  detect_torchrun

  if [[ "${SKIP_DOCTOR:-0}" != "1" ]]; then
    "$PYTHON_BIN" vla-scripts/doctor_training_env.py --config "$CONFIG"
  fi

  "$TORCHRUN_BIN" --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE:-1}" \
    vla-scripts/train_omnivla.py \
    --config "$CONFIG" \
    "${smoke_flag[@]}"
}

cmd="${1:-}"
case "$cmd" in
  ""|-h|--help|help)
    usage
    ;;
  setup)
    shift
    cmd_setup "$@"
    ;;
  check)
    shift
    cmd_check "$@"
    ;;
  doctor)
    shift
    cmd_doctor "$@"
    ;;
  inspect)
    shift
    cmd_inspect "$@"
    ;;
  smoke)
    shift
    run_torch_training --smoke-test
    ;;
  train)
    shift
    run_torch_training
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
