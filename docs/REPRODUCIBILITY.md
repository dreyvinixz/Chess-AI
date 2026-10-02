# Reproducibility

Use a recorded Git commit, `configs/*.yaml`, source PGN hash and dataset manifest, checkpoint, seed, and `runs/NAME/environment.json`. The latter records Python, PyTorch, CUDA runtime, GPU, timestamp, and dataset fingerprint. `chess-ai doctor` writes `reports/system.json` with runtime hardware information. `reports/` contains only measured local outputs.

Start with `bash scripts/setup_wsl.sh`, `source .venv/bin/activate`, `chess-ai doctor`, and `chess-ai smoke-test`. Run `pytest -q` and `ruff check src tests` for code verification. Hardware behavior can differ between Windows drivers; record `nvidia-smi` output in experiment notes.
