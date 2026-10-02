#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version; then
  echo "This setup targets Ubuntu on WSL2." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is unavailable in WSL. Check the Windows NVIDIA driver and WSL2 first." >&2
  exit 1
fi
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export PIP_CACHE_DIR="$PWD/.cache/pip"
if command -v uv >/dev/null; then
  uv venv --python 3.12 --seed .venv
  uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python .venv/bin/python -e '.[dev,train,browser]'
elif command -v python3.12 >/dev/null; then
  python3.12 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  .venv/bin/python -m pip install -e '.[dev,train,browser]'
else
  echo "Install Python 3.12 with venv support, or install uv, then rerun this script." >&2
  exit 1
fi
echo "Activate with: source .venv/bin/activate"
echo "Then run: chess-ai doctor && chess-ai smoke-test"
