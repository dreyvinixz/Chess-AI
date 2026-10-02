import chess
import chess.engine
import pytest

from chess_ai.teacher import score_cp, teacher_targets


def test_score_perspective_and_mate():
    score = chess.engine.PovScore(chess.engine.Cp(120), chess.WHITE)
    assert score_cp(score, chess.WHITE) == 120
    assert score_cp(score, chess.BLACK) == -120
    mate = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
    assert score_cp(mate, chess.WHITE) == 9998
    assert score_cp(mate, chess.BLACK) == -9998


def test_teacher_policy_and_value():
    board = chess.Board()
    infos = [
        {
            "score": chess.engine.PovScore(chess.engine.Cp(100), chess.WHITE),
            "pv": [chess.Move.from_uci("e2e4")],
            "depth": 8,
        },
        {
            "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE),
            "pv": [chess.Move.from_uci("d2d4")],
            "depth": 8,
        },
    ]
    target = teacher_targets(infos, board, 100)
    assert target["policy"]["e2e4"] > target["policy"]["d2d4"]
    assert sum(target["policy"].values()) == pytest.approx(1)
    assert target["value"] > 0
