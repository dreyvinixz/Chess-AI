# Chess-AI execution tracker

Updated: 2026-10-02. This file records implementation evidence; check boxes mean code, tests, documentation, and a commit exist.

## Sprint 0 — Audit and decisions

- [x] Inspect destination Git state, FrameBridge presentation, WSL2, Python, and GPU.
- [x] Research official WSL CUDA, PyTorch, Lichess data, and GPL dependency sources.
- [ ] Record measured architecture proof of concept in `docs/MODEL_DESIGN.md`.
- [x] Create GitHub issues #1 through #6 for the sprint deliverables.

## Sprint 1 — Local vertical slice

- [x] Package and configure the CLI.
- [x] Encode positions and every legal move, including special moves.
- [x] Run the compact network, mask legal moves, apply a move.
- [x] Run CUDA doctor and smoke test on WSL2 GTX 1650.
- [ ] Verify CI on CPU.

## Sprint 2 — Data and training

- [ ] Stream licensed PGN with game-level splits and source fingerprints.
- [ ] Generate offline Stockfish multi-PV labels.
- [ ] Train, resume, and evaluate checkpoints.
- [ ] Measure VRAM, latency, and throughput on the target GPU.

## Sprint 3 — Search and local strength

- [ ] Compare policy-only and PUCT against random and greedy baselines.
- [ ] Add local terminal play and optional self-play.
- [ ] Publish measured benchmark results without invented strength claims.

## Sprint 4 — Human experience

- [ ] Store observed games in SQLite.
- [ ] Annotate human moves after the game and weight fine-tuning examples.
- [ ] Implement assist mode with local suggestions and move capture.

## Sprint 5 — Chess.com bot integration

- [ ] Inspect current computer-game DOM and save minimal fixtures.
- [ ] Parse board, orientation, turn, bot identity, and result.
- [ ] Implement multi-signal fail-closed BotModeGuard.
- [ ] Execute only verified legal bot-game moves.
- [ ] Implement autoplay and bot ladder reporting.

## Sprint 6 — Release quality

- [ ] Complete public documentation, examples, troubleshooting, and license audit.
- [ ] Add dashboard only after the core and browser features work.
- [ ] Push branches, open PRs, resolve CI, and merge reviewed changes where authorized.

## Current evidence

- Destination repository began empty on `main` (`origin/main` absent).
- WSL2 Ubuntu 26.04 sees NVIDIA GeForce GTX 1650, 4096 MiB via `nvidia-smi`.
- Python 3.12.13 and PyTorch 2.6.0+cu124 run in the repository `.venv`; `chess-ai doctor` confirms CUDA on GTX 1650, 4096 MiB, compute capability 7.5, and an FP16 matrix operation.
- `pytest -q`: 12 passed. `ruff check src tests`: passed.
- `CUDA_VISIBLE_DEVICES="" pytest -q`: 12 passed. CUDA smoke test passed. The final debug checkpoint recorded Git commit `2b89cb6`, `git_dirty=false`, and the 14-position dataset SHA-256.
- `python -m build`: source distribution and wheel built. The policy projection was reduced from 21,016,149 to 2,665,831 parameters after profiling; see `reports/`.
- No Chess.com bot results have been measured.
