"""Generate local student self-play data without an external engine."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import chess
import torch

from chess_ai.data import sha256_file
from chess_ai.model import PolicyValueNet
from chess_ai.search import policy_only, puct


def sample_move(policy: dict[str, float], temperature: float, rng: random.Random) -> chess.Move:
    """Sample a legal root move from visit shares or choose its argmax."""
    if not policy or temperature < 0:
        raise ValueError("Expected a nonempty policy and nonnegative temperature")
    if temperature == 0:
        return chess.Move.from_uci(max(policy, key=policy.get))
    moves = list(policy)
    weights = [max(policy[move], 0.0) ** (1.0 / temperature) for move in moves]
    if not any(weights):
        raise ValueError("Search returned no visited move")
    return chess.Move.from_uci(rng.choices(moves, weights=weights, k=1)[0])


def generate_self_play(
    model: PolicyValueNet,
    device: torch.device,
    checkpoint: Path,
    destination: Path,
    games: int,
    search: str = "puct",
    simulations: int = 16,
    max_plies: int = 300,
    temperature: float = 1.0,
    temperature_plies: int = 20,
    noise_alpha: float | None = 0.3,
    seed: int = 42,
) -> dict:
    if games < 1 or simulations < 1 or max_plies < 1 or temperature_plies < 0:
        raise ValueError("Games, simulations and max plies must be positive")
    if search not in {"puct", "policy"}:
        raise ValueError("Self-play search must be puct or policy")
    if search == "policy" and noise_alpha is not None:
        raise ValueError("Root noise requires PUCT search")
    rng = random.Random(seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    results = {"W": 0, "D": 0, "L": 0, "U": 0}
    completed_positions = 0
    total_nodes = 0
    total_inferences = 0
    details = []
    with destination.open("w", encoding="utf-8") as stream:
        for game_index in range(games):
            board = chess.Board()
            game_rows = []
            while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
                result = (
                    puct(
                        model,
                        board,
                        device,
                        simulations=simulations,
                        root_noise_alpha=noise_alpha,
                        rng=rng,
                    )
                    if search == "puct"
                    else policy_only(model, board, device)
                )
                move = sample_move(
                    result.policy,
                    temperature if board.ply() < temperature_plies else 0,
                    rng,
                )
                if move not in board.legal_moves:
                    raise RuntimeError("Student search selected an illegal self-play move")
                game_rows.append(
                    {
                        "fen": board.fen(),
                        "move": move.uci(),
                        "policy": result.policy,
                        "game_id": game_index + 1,
                        "ply": board.ply(),
                        "source": "student_self_play",
                    }
                )
                total_nodes += result.nodes
                total_inferences += result.inferences
                board.push(move)
            outcome = board.outcome(claim_draw=True)
            result_code = (
                "U"
                if outcome is None
                else "D"
                if outcome.winner is None
                else "W"
                if outcome.winner == chess.WHITE
                else "L"
            )
            results[result_code] += 1
            details.append(
                {
                    "game": game_index + 1,
                    "result": result_code,
                    "plies": board.ply(),
                    "termination": outcome.termination.name if outcome else "PLY_CAP",
                }
            )
            if outcome is None:
                continue
            for row in game_rows:
                if outcome.winner is None:
                    row["value"] = 0.0
                else:
                    turn = chess.Board(row["fen"]).turn
                    row["value"] = 1.0 if outcome.winner == turn else -1.0
                stream.write(json.dumps(row) + "\n")
                completed_positions += 1
    report = {
        "schema_version": 1,
        "source": "student_self_play",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "games": games,
        "results_white_perspective": results,
        "completed_positions": completed_positions,
        "search": search,
        "simulations": simulations if search == "puct" else 0,
        "temperature": temperature,
        "temperature_plies": temperature_plies,
        "noise_alpha": noise_alpha,
        "seed": seed,
        "max_plies": max_plies,
        "nodes": total_nodes,
        "inferences": total_inferences,
        "elapsed_sec": perf_counter() - started,
        "details": details,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
