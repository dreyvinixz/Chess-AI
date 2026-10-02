"""Local strength baselines and match reports."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

from chess_ai.engines import StockfishAdapter
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
    engine_path: str = "stockfish",
    engine_depth: int = 4,
    engine_nodes: int = 0,
    engine_time_sec: float = 0.0,
    engine_elo: int | None = None,
) -> dict:
    rng = random.Random(seed)
    results = {"W": 0, "D": 0, "L": 0, "U": 0}
    details = []
    engine = (
        StockfishAdapter(engine_path, engine_depth, engine_nodes, engine_time_sec, engine_elo)
        if opponent == "stockfish"
        else None
    )
    if engine is not None:
        engine.__enter__()
    try:
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
                    move = engine.move(board) if engine else baseline_move(board, opponent, rng)
                board.push(move)
            outcome = board.outcome(claim_draw=True)
            result = (
                "U"
                if outcome is None
                else "D"
                if outcome.winner is None
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
                    "termination": outcome.termination.name if outcome else "PLY_CAP",
                }
            )
        engine_provenance = engine.provenance() if engine else None
    finally:
        if engine is not None:
            engine.__exit__(None, None, None)
    completed = results["W"] + results["D"] + results["L"]
    return {
        "opponent": opponent,
        "games": games,
        "search": search,
        "simulations": simulations if search == "puct" else 0,
        "seed": seed,
        "results": results,
        "completed_games": completed,
        "score": (results["W"] + 0.5 * results["D"]) / completed if completed else None,
        "external_engine_opponent": engine_provenance,
        "stockfish_runtime_assistance_for_student": False,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
