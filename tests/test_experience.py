import json
import math

import chess
import chess.engine
import pytest

from chess_ai.experience import ExperienceStore, quality_weight
from chess_ai.training import PositionDataset


def test_quality_weight_discount_and_bounds():
    assert quality_weight(-50) == 1.0
    assert quality_weight(0) == 1.0
    assert 0.08 < quality_weight(300) < 0.09
    assert quality_weight(10000) == 0.02
    with pytest.raises(ValueError):
        quality_weight(50, scale_cp=0)


def test_experience_annotation_export_and_training_weight(tmp_path, monkeypatch):
    class FakeEngine:
        def analyse(self, _board, _limit, root_moves=None):
            cp = -200 if root_moves else 100
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
                "pv": root_moves or [chess.Move.from_uci("e2e4")],
            }

    class FakeAdapter:
        def __init__(self, *_args, **_kwargs):
            self.engine = FakeEngine()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def provenance(self):
            return {"engine": {"name": "fake"}, "depth": 8}

    monkeypatch.setattr("chess_ai.experience.StockfishAdapter", FakeAdapter)
    database = tmp_path / "experience.sqlite"
    output = tmp_path / "human.jsonl"
    board = chess.Board()
    move = chess.Move.from_uci("d2d4")
    with ExperienceStore(database) as store:
        game_id = store.create_game("local_assist", "local-greedy", "model.pt", "white")
        with pytest.raises(ValueError, match="Finish the game"):
            store.annotate_game(game_id)
        with pytest.raises(ValueError, match="illegal human move"):
            store.record_human_move(
                game_id, board, chess.Move.from_uci("e2e5"), {"e2e4": 1.0}, 0, 1, 0.1
            )
        store.record_human_move(game_id, board, move, {"e2e4": 0.7, "d2d4": 0.3}, 0.2, 4, 0.01)
        store.finish_game(game_id, "1-0")
        annotation = store.annotate_game(game_id)
        assert annotation["positions"] == 1
        manifest = store.export_training(output)
        assert manifest["positions"] == 1
        stored = store.connection.execute("SELECT * FROM positions").fetchone()
        assert stored["teacher_move"] == "e2e4"
        assert stored["teacher_best_cp"] == 100
        assert stored["teacher_chosen_cp"] == -200
        assert stored["policy_weight"] == pytest.approx(math.exp(-300 / 120))
    row = json.loads(output.read_text())
    assert row["move"] == "d2d4"
    assert row["value"] == pytest.approx(math.tanh(100 / 600))
    assert row["policy_weight"] < 0.1
    features, policy, value, legal, weight = PositionDataset(output)[0]
    assert features.shape == (18, 8, 8)
    assert policy.sum() == 1
    assert legal.sum() == 20
    assert value.item() == pytest.approx(row["value"])
    assert weight.item() == pytest.approx(row["policy_weight"])


def test_experience_rejects_illegal_model_policy(tmp_path):
    with ExperienceStore(tmp_path / "experience.sqlite") as store:
        game_id = store.create_game("local_assist", "local-random", "model.pt", "white")
        with pytest.raises(ValueError, match="Model policy"):
            store.record_human_move(
                game_id, chess.Board(), chess.Move.from_uci("e2e4"),
                {"e2e5": 1.0}, 0.0, 1, 0.1,
            )
