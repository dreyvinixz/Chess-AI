"""Local strength baselines and match reports."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

from chess_ai.model import PolicyValueNet
from chess_ai.search import policy_only, puct

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def baseline_move(board: chess.Board, name: str, rng: random.Random) -> chess.Move:
    moves = list(board.legal_moves)
    if name == "random":
        return rng.choice(moves)
    if name == "greedy":
        return max(
            moves,
            key=lambda move: (
                PIECE_VALUE.get(board.piece_type_at(move.to_square), 0)
                + PIECE_VALUE.get(move.promotion, 0),
                rng.random(),
            ),
        )
    raise ValueError(f"Unknown baseline: {name}")


def play_match(
    model: PolicyValueNet,
    device: torch.device,
    opponent: str,
    games: int,
    search: str,
    simulations: int,
    seed: int = 42,
    max_plies: int = 300,
) -> dict:
    rng = random.Random(seed)
    results = {"W": 0, "D": 0, "L": 0}
    details = []
    for game_number in range(games):
        board = chess.Board()
        student_color = chess.WHITE if game_number % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if board.turn == student_color:
                choice = (
                    puct(model, board, device, simulations)
                    if search == "puct"
                    else policy_only(model, board, device)
                )
                move = choice.move
            else:
                move = baseline_move(board, opponent, rng)
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        result = (
            "D"
            if outcome is None or outcome.winner is None
            else "W"
            if outcome.winner == student_color
            else "L"
        )
        results[result] += 1
        details.append(
            {
                "game": game_number + 1,
                "color": "white" if student_color else "black",
                "result": result,
                "plies": board.ply(),
                "pgn_result": board.result(claim_draw=True),
            }
        )
    return {
        "opponent": opponent,
        "games": games,
        "search": search,
        "simulations": simulations,
        "seed": seed,
        "results": results,
        "score": (results["W"] + 0.5 * results["D"]) / games,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
