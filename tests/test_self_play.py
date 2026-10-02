import json

import chess
import pytest
import torch

from chess_ai.model import PolicyValueNet
from chess_ai.search import SearchResult, puct
from chess_ai.self_play import generate_self_play, sample_move
from chess_ai.training import PositionDataset
from chess_ai.validation import validate_dataset


def test_self_play_marks_ply_cap_unfinished(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    destination = tmp_path / "self-play.jsonl"
    model = PolicyValueNet(channels=16, blocks=1)
    report = generate_self_play(
        model, torch.device("cpu"), checkpoint, destination, 1,
        search="policy", max_plies=1, noise_alpha=None,
    )
    assert report["results_white_perspective"]["U"] == 1
    assert report["completed_positions"] == 0
    assert destination.read_text() == ""
    assert report["output_sha256"] == json.loads(
        destination.with_suffix(".manifest.json").read_text()
    )["output_sha256"]


def test_completed_self_play_targets_and_dataset(tmp_path, monkeypatch):
    sequence = ["f2f3", "e7e5", "g2g4", "d8h4"]

    def forced_policy(_model, board, _device):
        move = chess.Move.from_uci(sequence[board.ply()])
        assert move in board.legal_moves
        return SearchResult(move, 0.0, 1, 1, 0.001, [(move.uci(), 1.0)], {move.uci(): 1.0})

    monkeypatch.setattr("chess_ai.self_play.policy_only", forced_policy)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    destination = tmp_path / "self-play.jsonl"
    model = PolicyValueNet(channels=16, blocks=1)
    report = generate_self_play(
        model, torch.device("cpu"), checkpoint, destination, 1,
        search="policy", max_plies=10, noise_alpha=None,
    )
    assert report["results_white_perspective"]["L"] == 1
    assert report["completed_positions"] == 4
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["value"] for row in rows] == [-1.0, 1.0, -1.0, 1.0]
    assert len(PositionDataset(destination)) == 4
    metrics = validate_dataset(model, torch.device("cpu"), destination, checkpoint, 2)
    assert metrics["positions"] == 4
    assert 0 <= metrics["top1_target_agreement"] <= 1


def test_self_play_rejects_invalid_sampling():
    with pytest.raises(ValueError):
        sample_move({}, 1, __import__("random").Random(42))
    with pytest.raises(ValueError):
        sample_move({"e2e4": 1.0}, -1, __import__("random").Random(42))


def test_puct_policy_contains_only_legal_moves_and_is_deterministic():
    import random

    torch.manual_seed(42)
    model = PolicyValueNet(channels=16, blocks=1)
    board = chess.Board()
    first = puct(model, board, torch.device("cpu"), simulations=4)
    second = puct(model, board, torch.device("cpu"), simulations=4)
    assert first.move == second.move
    assert first.policy == second.policy
    assert abs(sum(first.policy.values()) - 1) < 1e-6
    assert all(chess.Move.from_uci(move) in board.legal_moves for move in first.policy)
    noisy = puct(
        model, board, torch.device("cpu"), simulations=4,
        root_noise_alpha=0.3, rng=random.Random(42),
    )
    assert noisy.move in board.legal_moves
    assert abs(sum(noisy.policy.values()) - 1) < 1e-6


def test_dataset_rejects_policy_without_legal_weight(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        json.dumps({"fen": chess.STARTING_FEN, "policy": {"e2e5": 1.0}, "value": 0.0})
        + "\n"
    )
    with pytest.raises(ValueError, match="no legal positive weight"):
        PositionDataset(dataset)[0]
