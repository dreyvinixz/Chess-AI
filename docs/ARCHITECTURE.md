# Architecture

The local chess stack depends only on python-chess, PyTorch, NumPy, and configuration. `core.py` encodes a board and maps legal moves to policy indices. `model.py` predicts policy logits and a side-to-move value. `search.py` masks to legal actions and runs policy-only or PUCT search. `data.py` streams PGN into game-level splits. `teacher.py` runs Stockfish offline to label existing positions. `training.py` consumes JSONL examples and writes checkpoints and metrics. `evaluation.py` plays local baseline games.

The teacher module is never imported by student search. A future browser adapter will consume the same `chess.Board` and student search API after verifying a bot-only game. See [Chess.com boundary](CHESSCOM.md).
