# Benchmarks

No chess strength result has been published yet. Run `chess-ai benchmark --checkpoint PATH` to measure model parameter count, checkpoint size, and inference latency on the current machine. It writes `reports/benchmark.json` and `reports/model_summary.md`. Run `chess-ai evaluate` separately for W/D/L. Never use inference latency as a strength metric.

The measured debug-profile sizing run on a GTX 1650 Max-Q, PyTorch 2.6.0+cu124, used the 14-position example and two training steps. Its checkpoint is not a chess-strength model. Before the policy projection change: 21,016,149 parameters, 252,221,346-byte checkpoint, 1.48 ms per inference over 10 iterations. After: 2,665,831 parameters, 32,017,506-byte checkpoint, 2.19 ms over 100 iterations. The runs used different iteration counts and show no demonstrated speed gain. JSON evidence is in `reports/benchmark_before.json` and `reports/benchmark.json`.

| Metric | Result |
|---|---|
| GTX 1650 inference peak allocated VRAM, debug profile | 19.4 MiB after projection change; excludes CUDA context and training |
| Training positions/second | Not measured |
| Student score vs random | Not measured |
| Student score vs greedy | Not measured |
| Student score vs Stockfish | Not measured |
| Chess.com bot ladder | Not measured |
