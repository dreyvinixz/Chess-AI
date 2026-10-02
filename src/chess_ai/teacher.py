"""Offline UCI Stockfish labels. Never imported by student search."""

import json
import math
from pathlib import Path

import chess
import chess.engine

from chess_ai.data import records


def score_cp(score: chess.engine.PovScore, turn: chess.Color, mate_cp: int = 10000) -> int:
    return score.pov(turn).score(mate_score=mate_cp) or 0


def teacher_targets(infos: list[dict], board: chess.Board, temperature_cp: float) -> dict:
    if temperature_cp <= 0:
        raise ValueError("temperature_cp must be positive")
    candidates = []
    for info in infos:
        pv = info.get("pv", [])
        if not pv or pv[0] not in board.legal_moves:
            continue
        cp = score_cp(info["score"], board.turn)
        candidates.append(
            {
                "move": pv[0].uci(),
                "cp": cp,
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
                "pv": [move.uci() for move in pv],
            }
        )
    if not candidates:
        raise ValueError("Teacher supplied no legal candidate")
    maximum = max(item["cp"] for item in candidates)
    weights = [math.exp((item["cp"] - maximum) / temperature_cp) for item in candidates]
    total = sum(weights)
    policy = {
        item["move"]: weight / total for item, weight in zip(candidates, weights, strict=True)
    }
    value = math.tanh(candidates[0]["cp"] / 600)
    return {"policy": policy, "value": value, "teacher": candidates}


def label_positions(
    source: Path,
    destination: Path,
    engine_path: str,
    depth: int = 8,
    multipv: int = 3,
    temperature_cp: float = 100,
    limit: int = 0,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        chess.engine.SimpleEngine.popen_uci(engine_path) as engine,
        destination.open("w", encoding="utf-8") as target,
    ):
        for row in records(source):
            board = chess.Board(row["fen"])
            if board.is_game_over():
                continue
            info = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=min(multipv, board.legal_moves.count()),
            )
            row.update(teacher_targets(info, board, temperature_cp))
            row["source"] = "stockfish"
            target.write(json.dumps(row) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count
