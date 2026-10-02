# Evaluation

`chess-ai evaluate --checkpoint PATH --opponent random --games 10` alternates colors and stores full game W/D/L/U details as JSON. `U` means unfinished at the configured ply cap and is excluded from the match score. `--opponent greedy` uses a simple immediate material-capture heuristic. `--search puct` enables student search. These are development baselines, not calibrated Elo or proof of bot strength. The result JSON includes checkpoint, search settings, seed, color, plies, termination, and score over completed games.

Future bot ladder reports will distinguish **first win** from **match score** across repeated games. No Chess.com opponent or rating is hardcoded. Stockfish baseline matches, when added, must be explicitly labeled as an external-engine baseline.

Use `chess-ai validate --checkpoint PATH --dataset data/processed/lichess-2013-01-100/validation.jsonl` to measure held-out policy cross entropy, top-one move agreement, and value mean squared error. The JSON report records the checkpoint and dataset SHA-256 hashes. These metrics test fit to PGN moves and game outcomes; they do not establish playing strength. Keep the test split untouched while selecting models on validation results.
