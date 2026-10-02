from pathlib import Path

import chess
import pytest
import torch
import zstandard

from chess_ai.config import load_config
from chess_ai.data import prepare_pgn, records, reservoir_sample
from chess_ai.search import policy_only, puct
from chess_ai.training import load_model, train


def test_pgn_split_and_checkpoint(tmp_path):
    pgn = tmp_path / "games.pgn"
    game = (
        '[Event "Example"]\n[Site "local"]\n'
        '[WhiteElo "2200"]\n[BlackElo "2200"]\n'
        '[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n'
    )
    pgn.write_text(game + "\n" + game, encoding="utf-8")
    manifest = prepare_pgn(pgn, tmp_path / "processed", min_elo=1800, max_games=2)
    assert manifest["games"] == 2
    assert sum(manifest["positions"].values()) == 4
    assert manifest["duplicates_skipped"] == 4
    assert len(manifest["files"]["train"]) == 64
    compressed = tmp_path / "games.pgn.zst"
    compressed.write_bytes(zstandard.ZstdCompressor().compress(pgn.read_bytes()))
    compressed_manifest = prepare_pgn(
        compressed, tmp_path / "compressed", min_elo=1800, max_games=2
    )
    assert compressed_manifest["positions"] == manifest["positions"]
    assert compressed_manifest["files"] == manifest["files"]
    split = next(name for name, count in manifest["positions"].items() if count)
    dataset = tmp_path / "processed" / f"{split}.jsonl"
    assert len(list(records(dataset))) == 4
    config = load_config(Path(__file__).parents[1] / "configs/debug.yaml")
    config["training"]["epochs"] = 2
    config["training"]["max_steps"] = 1
    checkpoint = train(config, dataset, tmp_path / "run")
    model, saved = load_model(checkpoint, torch.device("cpu"))
    assert saved["global_step"] == 1
    assert saved["scheduler"] is not None
    config["training"]["max_steps"] = 4
    checkpoint = train(config, dataset, tmp_path / "run", resume=checkpoint)
    _, resumed = load_model(checkpoint, torch.device("cpu"))
    assert resumed["epoch"] == 2
    assert resumed["global_step"] == 2
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(dataset.read_bytes() + dataset.read_bytes())
    with pytest.raises(ValueError, match="fingerprint"):
        train(config, changed, tmp_path / "invalid", resume=checkpoint)
    inference_model, _ = load_model(tmp_path / "run" / "inference.pt", torch.device("cpu"))
    board = chess.Board()
    assert policy_only(model, board, torch.device("cpu")).move in board.legal_moves
    assert policy_only(inference_model, board, torch.device("cpu")).move in board.legal_moves
    assert puct(model, board, torch.device("cpu"), simulations=2).move in board.legal_moves
    warm_config = load_config(Path(__file__).parents[1] / "configs/debug.yaml")
    warm_path = train(
        warm_config, dataset, tmp_path / "warm", init_checkpoint=tmp_path / "run" / "inference.pt"
    )
    _, warm = load_model(warm_path, torch.device("cpu"))
    assert warm["config"]["initialization"]["checkpoint"].endswith("inference.pt")
    assert len(warm["config"]["initialization"]["sha256"]) == 64


def test_reservoir_sample_is_deterministic():
    first = reservoir_sample(range(100), 10, 42)
    assert first == reservoir_sample(range(100), 10, 42)
    assert len(first) == len(set(first)) == 10
    assert any(value >= 50 for value in first)
