# Benchmarks

Only preliminary local match results have been published. Run `chess-ai benchmark --checkpoint PATH` to measure model parameter count, checkpoint size, and inference latency on the current machine. It writes `reports/benchmark.json` and `reports/model_summary.md`. Run `chess-ai evaluate` separately for W/D/L. Never use inference latency as a strength metric.

The measured debug-profile sizing run on a GTX 1650 Max-Q, PyTorch 2.6.0+cu124, used the 14-position example and two training steps. Its checkpoint is not a chess-strength model. Before the policy projection change: 21,016,149 parameters, 252,221,346-byte checkpoint, 1.48 ms per inference over 10 iterations. After: 2,665,831 parameters, 32,017,506-byte checkpoint, 2.19 ms over 100 iterations. The runs used different iteration counts and show no demonstrated speed gain. JSON evidence is in `reports/benchmark_before.json` and `reports/benchmark.json`.

| Metric | Result |
|---|---|
| GTX 1650 inference, distillation checkpoint | 2.24 ms/position over 100 calls; 20.6 MiB peak allocated VRAM; 2,965,255 parameters |
| GTX 1650 training, 100 supervised steps | 112 MiB peak allocated VRAM; 1,261 examples/s over first 5,495-position epoch |
| GTX 1650 training, 20 distillation steps | 135 MiB peak allocated VRAM; 256 positions, five epochs |
| Student vs random, policy-only, 10 attempted games | 2W / 7D / 0L / 1 unfinished; 61.1% over 9 completed |
| Student vs greedy, policy-only, 10 attempted games | 0W / 8D / 1L / 1 unfinished; 44.4% over 9 completed |
| Student score vs Stockfish | Not measured |
| Chess.com bot ladder | Not measured |

The game results use the **supervised** checkpoint after 100 steps, before teacher distillation. The public sample contains 100 rated Lichess games from January 2013, filtered to both players at least 2000 and deduplicated by board state. Ten attempted baseline games per opponent are too few for an Elo estimate. Unfinished games at 300 plies are excluded from match score. Training throughput on this small dataset is not a sustained large-dataset benchmark. The JSON reports contain exact run metadata and per-game outcomes.

## Held-out validation and local search proof

The same 728-position validation split was used for each row. Lower cross entropy and value MSE are better; higher top-one agreement is better. PGN move agreement is not a strength rating.

| Checkpoint | Policy cross entropy | Value MSE | Top-one agreement |
|---|---:|---:|---:|
| 100-step supervised | 3.170 | 1.201 | 12.91% |
| 20-step teacher distillation | 3.189 | 1.106 | 12.64% |
| Five-step self-play proof | 3.237 | 2.236 | 12.64% |

The 20-step teacher run reduced value error on this split but slightly worsened imitation of PGN moves. Fine-tuning on one self-play game markedly worsened value error, so this proof checkpoint should **not** replace the teacher checkpoint. Reports contain exact SHA-256 hashes under `reports/evaluation/`.

Using the distillation checkpoint against the greedy material baseline, policy-only and 16-simulation PUCT each scored four draws in four games, with zero wins. This tiny sample shows no strength advantage from PUCT. The two modes played different game trajectories. `reports/evaluation/distill_policy_vs_greedy.json` and `distill_puct16_vs_greedy.json` record each termination.

A single policy-only self-play game ended in threefold repetition after 46 plies. A single 16-simulation PUCT game ended in checkmate after 160 plies and took 5.68 seconds for 2,560 simulations and 2,695 model inferences on the GTX 1650 Max-Q. The saved PUCT game supplied 160 training positions. These are pipeline and speed observations, not strength evidence. The 5-step training run used 134 MiB peak allocated VRAM and recorded `git_dirty=true` because the implementation was still being edited during the proof. See `reports/self_play/` and `reports/training/self-play-puct-proof.json`.
