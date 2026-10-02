# Reproducibility

Use a recorded Git commit, `configs/*.yaml`, source PGN hash and dataset manifest, checkpoint, seed, and `runs/NAME/environment.json`. The latter records Python, PyTorch, CUDA runtime, GPU, timestamp, and dataset fingerprint. `chess-ai doctor` writes `reports/system.json` with runtime hardware information. `reports/` contains only measured local outputs.

Start with `bash scripts/setup_wsl.sh`, `source .venv/bin/activate`, `chess-ai doctor`, and `chess-ai smoke-test`. Run `pytest -q` and `ruff check src tests scripts` for code verification. Hardware behavior can differ between Windows drivers; record `nvidia-smi` output in experiment notes.

The public 100-game experiment is reproducible from `python scripts/download_lichess.py --month 2013-01` and the preparation/training commands in the README. `reports/datasets/` records the official archive checksum, transformed split hashes, and Stockfish label hashes. `reports/training/` records exact configs, source commits, fingerprints, GPU, and measured throughput. Local binaries, downloaded games, labels, and model weights remain ignored; rebuild them with the documented commands.
