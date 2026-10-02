import random

import chess
import torch

from chess_ai.evaluation import baseline_move, play_match
from chess_ai.model import PolicyValueNet


def test_ply_cap_is_unfinished_not_draw():
    model = PolicyValueNet(channels=16, blocks=1)
    report = play_match(
        model,
        torch.device("cpu"),
        "random",
        games=1,
        search="policy",
        simulations=1,
        max_plies=2,
    )
    assert report["results"] == {"W": 0, "D": 0, "L": 0, "U": 1}
    assert report["completed_games"] == 0
    assert report["score"] is None
    assert report["details"][0]["termination"] == "PLY_CAP"


def test_baselines_return_legal_moves():
    board = chess.Board()
    for name in ("random", "greedy"):
        assert baseline_move(board, name, random.Random(42)) in board.legal_moves
