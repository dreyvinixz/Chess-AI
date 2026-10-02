
import chess
import torch

from chess_ai.config import load_config
from chess_ai.data import prepare_pgn, records
from chess_ai.search import policy_only, puct
from chess_ai.training import load_model, train


def test_pgn_split_and_checkpoint(tmp_path):
    pgn = tmp_path / "games.pgn"
    pgn.write_text(
        '[Event "Example"]\n[Site "local"]\n'
        '[WhiteElo "2200"]\n[BlackElo "2200"]\n'
        '[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n',
        encoding="utf-8",
    )
    manifest = prepare_pgn(pgn, tmp_path / "processed", min_elo=1800, max_games=1)
    assert manifest["games"] == 1
    assert sum(manifest["positions"].values()) == 4
    split = next(name for name, count in manifest["positions"].items() if count)
    dataset = tmp_path / "processed" / f"{split}.jsonl"
    assert len(list(records(dataset))) == 4
    config = load_config(__import__("pathlib").Path(__file__).parents[1] / "configs/debug.yaml")
    checkpoint = train(config, dataset, tmp_path / "run")
    model, saved = load_model(checkpoint, torch.device("cpu"))
    assert saved["global_step"] > 0
    board = chess.Board()
    assert policy_only(model, board, torch.device("cpu")).move in board.legal_moves
    assert puct(model, board, torch.device("cpu"), simulations=2).move in board.legal_moves
