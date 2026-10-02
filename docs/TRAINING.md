# Training

`chess-ai train --dataset PATH --config configs/gtx1650.yaml --run-dir runs/NAME` trains on prepared or teacher-labeled JSONL. PGN records use a one-hot move target and game outcome from the current player's perspective. Teacher records use MultiPV probabilities and UCI WDL when available, otherwise a bounded centipawn value. Training masks illegal moves, uses AdamW, a cosine learning-rate scheduler, gradient accumulation, gradient clipping, optional CUDA FP16 autocast and GradScaler, and records CSV, JSONL epoch summaries, and optional TensorBoard events.

`--resume runs/NAME/checkpoint.pt` restores model, optimizer, scheduler, and scaler state. Checkpoints include the config, epoch, global step, dataset SHA-256, Git commit and dirty-source flag, metrics, and timestamp. Resume rejects a changed dataset fingerprint or model shape and uses a deterministic per-epoch shuffle. `inference.pt` stores only model and config for smaller evaluation artifacts. The dataset indexes JSONL byte offsets and reads records on demand, so row contents do not all reside in RAM. CUDA batch-size probing respects `hardware.max_vram_mb`; a forward/backward OOM halves the microbatch and retries. A retry resets partial accumulated gradients, which is recorded by the microbatch size in metrics.

Use `configs/debug.yaml` for a short functional run. The sample PGN is insufficient for chess strength. For a useful experiment, use separate validation/test sets and record their source manifests. Metrics include policy/value loss, entropy, learning rate, examples per second, allocated VRAM, and periodic `nvidia-smi` utilization when available. Never describe Stockfish teacher labels as student-selected moves.

For a reproducible 100-step supervised experiment on a small public dataset, follow the download and preparation commands in the README, then run:

```bash
chess-ai train --dataset data/processed/lichess-2013-01-100/train.jsonl --config configs/gtx1650.yaml --max-steps 100 --run-dir runs/lichess-supervised-100
```

The CLI writes a compact JSON summary under `reports/training/`; detailed CSV, TensorBoard events, and checkpoints remain in ignored `runs/`. The public-data split must be kept separate when evaluating generalization.

For offline teacher distillation from the supervised weights:

```bash
python scripts/download_stockfish.py
chess-ai label-data data/processed/lichess-2013-01-100/train.jsonl --output data/processed/lichess-2013-01-100/teacher-256.jsonl --depth 8 --multipv 3 --limit 256 --sample-seed 42
chess-ai train --dataset data/processed/lichess-2013-01-100/teacher-256.jsonl --config configs/gtx1650.yaml --max-steps 20 --init-checkpoint runs/lichess-supervised-100/inference.pt --run-dir runs/lichess-distill-256
```

`--init-checkpoint` copies model weights and starts a fresh optimizer and dataset fingerprint. `--resume` restores optimizer, scheduler, scaler, epoch, and global step for the same dataset; it rejects a changed dataset hash or model shape. The teacher target uses a centipawn temperature softmax over legal MultiPV moves, with mate distance mapped through `mate_score=10000`. Value uses Stockfish UCI WDL from the side to move when available, otherwise `tanh(cp/600)`. The original PGN game outcome is retained separately in teacher-labeled rows.

After supervised training, optional student-only self-play produces a dataset with PUCT root visit targets and final outcomes from the side to move:

```bash
chess-ai self-play --checkpoint runs/lichess-distill-256/inference.pt --games 2 --search puct --simulations 16 --output data/processed/self-play.jsonl
chess-ai train --dataset data/processed/self-play.jsonl --config configs/gtx1650.yaml --init-checkpoint runs/lichess-distill-256/inference.pt --run-dir runs/self-play-1
```

Self-play stores only completed games. Games still running at `--max-plies` are marked `U` and omitted from training rather than labeled draws. The adjacent manifest includes checkpoint and output hashes, search settings, seed, game results, node/inference totals, and elapsed time. PUCT root noise is used only for self-play. Set `--noise-alpha 0` to disable it. Self-play on a weak checkpoint can reinforce errors, so compare the new checkpoint on held-out data and local matches before adopting it.
