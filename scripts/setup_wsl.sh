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
if ! command -v python3.12 >/dev/null; then
  echo "Python 3.12 is required; install it in Ubuntu before continuing." >&2
  exit 1
fi
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -e '.[dev,train,browser]'
echo "Activate with: source .venv/bin/activate"
echo "Then run: chess-ai doctor && chess-ai smoke-test"
