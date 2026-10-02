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


def test_external_engine_is_explicit_opponent_only(monkeypatch):
    class FakeEngine:
        moves = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def move(self, board):
            self.moves += 1
            return next(iter(board.legal_moves))

        def provenance(self):
            return {"engine": {"name": "fake"}, "depth": 1}

    monkeypatch.setattr("chess_ai.evaluation.StockfishAdapter", FakeEngine)
    model = PolicyValueNet(channels=16, blocks=1)
    report = play_match(
        model, torch.device("cpu"), "stockfish", 1, "policy", 1, max_plies=2
    )
    assert report["opponent"] == "stockfish"
    assert report["external_engine_opponent"]["engine"]["name"] == "fake"
    assert report["stockfish_runtime_assistance_for_student"] is False
    assert report["details"][0]["plies"] == 2
