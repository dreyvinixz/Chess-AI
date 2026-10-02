"""Reproducible scripted checkmate to exercise the human-experience pipeline."""

import argparse
import json
from pathlib import Path

import chess
import torch

from chess_ai.experience import ExperienceStore
from chess_ai.search import policy_only
from chess_ai.training import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--database", type=Path, default=Path("data/processed/demo-experience.sqlite")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/demo-experience.jsonl"))
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device)
    board = chess.Board()
    with ExperienceStore(args.database) as store:
        game_id = store.create_game(
            "scripted_demo", "scripted-fools-mate", str(args.checkpoint), "white"
        )
        for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
            move = board.parse_uci(uci)
            if board.turn == chess.WHITE:
                suggestion = policy_only(model, board, device)
                store.record_human_move(
                    game_id, board, move, suggestion.policy,
                    suggestion.value, suggestion.nodes, suggestion.elapsed,
                )
            board.push(move)
        assert board.is_checkmate()
        store.finish_game(game_id, board.result())
        annotation = store.annotate_game(game_id, depth=args.depth)
        manifest = store.export_training(args.output)
    print(
        json.dumps(
            {"game_id": game_id, "result": board.result(), "annotation": annotation,
             "export": manifest},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
