# Assist mode and human experience

`chess-ai assist` currently plays a **local** game against the random or greedy baseline. It displays the board, FEN, side to move, five student suggestions, probabilities or PUCT visit shares, value, node count, and search time. The user enters SAN or UCI; the program records the actual legal move and whether it matched the student's top choice. This mode does not yet read or move a Chess.com board.

```bash
chess-ai assist --checkpoint runs/lichess-distill-256/inference.pt --opponent greedy
```

The default SQLite database is `data/processed/experience.sqlite`, excluded from Git. `games` records source, opponent, optional rating, checkpoint, color, time, and PGN result. `positions` records FEN, legal moves, the human move, complete model policy, model value, search nodes/time, and later teacher fields. SQLite transactions keep a game in one local file. Typing `quit` leaves result `*` and retains recorded positions; unfinished games cannot be annotated as completed games.

After a finished game, the default workflow reviews every recorded human position with Stockfish and exports `data/processed/human-experience.jsonl`. If Stockfish is missing, the game remains in SQLite and can be reviewed later:

```bash
python scripts/download_stockfish.py
chess-ai annotate-experience 1 --database data/processed/experience.sqlite --depth 8
chess-ai export-experience --database data/processed/experience.sqlite --output data/processed/human-experience.jsonl
chess-ai train --dataset data/processed/human-experience.jsonl --config configs/gtx1650.yaml --init-checkpoint runs/lichess-distill-256/inference.pt --run-dir runs/human-001
```

`--train-after --fine-tune-steps 10` on `assist` performs the final training step automatically after annotation. It warm-starts from the supplied checkpoint at a learning rate no higher than `1e-4`. Keep the previous checkpoint and compare held-out validation and local matches before adopting the new one; a small experience set can degrade the model.

## Quality weighting

Post-game Stockfish evaluates the best move and forces the human move at the **same search depth**, both from the original side's perspective. The approximate centipawn loss is `best_cp - chosen_cp`. The human move's policy weight is `max(0.02, exp(-max(0, cp_loss) / 120))`. A good move receives weight 1; a severe error approaches 0.02. The teacher's pre-move evaluation becomes the value target and is trained independently of the policy weight. Mate scores use python-chess's bounded `mate_score=10000`; a move permitting mate gets the minimum imitation weight. These are approximate annotations, not calibrated move-quality labels.

Exported JSONL contains `fen`, human `move`, `value`, `policy_weight`, teacher best and chosen centipawn scores, game ID, ply, result, opponent, and checkpoint. Training multiplies **only policy cross entropy** by `policy_weight`; value loss still uses the teacher position evaluation. The export manifest records SHA-256 and sample count. `--min-weight` on `export-experience` can filter low-quality moves entirely.

## Reproducible scripted proof

With a downloaded Stockfish and a trained checkpoint:

```bash
python scripts/demo_experience.py --checkpoint runs/lichess-distill-256/inference.pt --depth 4
chess-ai train --dataset data/processed/demo-experience.jsonl --config configs/gtx1650.yaml --init-checkpoint runs/lichess-distill-256/inference.pt --max-steps 1 --run-dir runs/assist-demo
```

The script executes Fool's Mate with fixed moves. Its `source=scripted_demo` and opponent name explicitly distinguish it from a human or Chess.com game. On the recorded GTX 1650 run it exported two positions and the training command completed one step. The `g4` blunder received policy weight 0.02; this is a pipeline check, not evidence of model improvement.
