# Training

`chess-ai train --dataset PATH --config configs/gtx1650.yaml --run-dir runs/NAME` trains on prepared or teacher-labeled JSONL. PGN records use a one-hot move target and game outcome from the current player's perspective. Teacher records use MultiPV probabilities and a bounded engine value. Training masks illegal moves, uses AdamW, gradient accumulation, gradient clipping, optional CUDA FP16 autocast and GradScaler, and records a metrics CSV.

`--resume runs/NAME/checkpoint.pt` restores model and optimizer state. Checkpoints include the config, epoch, global step, dataset SHA-256, Git commit, metrics, and timestamp. A resumed run should use the same dataset and compatible config; strict fingerprint enforcement is planned. Current CUDA OOM handling stops with an actionable batch-size message; automatic fallback is planned.

Use `configs/debug.yaml` for a short functional run. The sample PGN is insufficient for chess strength. For a useful experiment, use separate validation/test sets and record their source manifests. Never describe Stockfish teacher labels as student-selected moves.
