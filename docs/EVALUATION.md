# Evaluation

`chess-ai evaluate --checkpoint PATH --opponent random --games 10` alternates colors and stores full game W/D/L details as JSON. `--opponent greedy` uses a simple immediate material-capture heuristic. `--search puct` enables student search. These are development baselines, not calibrated Elo or proof of bot strength. The result JSON includes checkpoint, search settings, seed, color, plies, and score. Termination at the configured maximum plies is recorded as a draw; reports should mention that cap.

Future bot ladder reports will distinguish **first win** from **match score** across repeated games. No Chess.com opponent or rating is hardcoded. Stockfish baseline matches, when added, must be explicitly labeled as an external-engine baseline.
