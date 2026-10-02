# Benchmarks

No chess strength result has been published yet. Run `chess-ai benchmark --checkpoint PATH` to measure model parameter count, checkpoint size, and inference latency on the current machine. It writes `reports/benchmark.json` and `reports/model_summary.md`. Run `chess-ai evaluate` separately for W/D/L. Never use inference latency as a strength metric.

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
