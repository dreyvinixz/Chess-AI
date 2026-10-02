# Chess-AI execution tracker

Updated: 2026-10-02. This file records implementation evidence; check boxes mean code, tests, documentation, and a commit exist.

## Sprint 0 — Audit and decisions

- [x] Inspect destination Git state, FrameBridge presentation, WSL2, Python, and GPU.
- [x] Research official WSL CUDA, PyTorch, Lichess data, and GPL dependency sources.
- [ ] Record measured architecture proof of concept in `docs/MODEL_DESIGN.md`.
- [ ] Create GitHub issues for the remaining sprint deliverables.

## Sprint 1 — Local vertical slice

- [ ] Package and configure the CLI.
- [ ] Encode positions and every legal move, including special moves.
- [ ] Run the compact network, mask legal moves, apply a move.
- [ ] Run CUDA doctor and smoke test on WSL2 GTX 1650.
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
- Python 3.12 and PyTorch 2.6.0 CUDA 12.4 wheel were installed in the repository `.venv`; CUDA execution still needs verification.
- No Chess.com bot results have been measured.
