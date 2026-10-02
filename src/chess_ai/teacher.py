"""Offline UCI Stockfish labels. Never imported by student search."""

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine

from chess_ai.data import records, reservoir_sample, sha256_file


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
        wdl = info.get("wdl")
        wdl_counts = None
        if wdl is not None:
            expected = wdl.pov(board.turn)
            wdl_counts = [expected.wins, expected.draws, expected.losses]
        candidates.append(
            {
                "move": pv[0].uci(),
                "cp": cp,
                "mate": info["score"].pov(board.turn).mate(),
                "wdl": wdl_counts,
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
    best = max(candidates, key=lambda candidate: candidate["cp"])
    if best["wdl"] and sum(best["wdl"]):
        wins, draws, losses = best["wdl"]
        value = (wins - losses) / (wins + draws + losses)
        value_source = "uci_wdl"
    else:
        value = math.tanh(best["cp"] / 600)
        value_source = "centipawn_tanh"
    return {"policy": policy, "value": value, "value_source": value_source, "teacher": candidates}


def label_positions(
    source: Path,
    destination: Path,
    engine_path: str,
    depth: int = 8,
    multipv: int = 3,
    temperature_cp: float = 100,
    limit: int = 0,
    sample_seed: int | None = None,
) -> int:
    if sample_seed is not None and limit <= 0:
        raise ValueError("A positive --limit is required with --sample-seed")
    local_engine = Path.cwd() / "tools" / "stockfish-local" / "stockfish"
    if engine_path == "stockfish" and local_engine.is_file():
        engine_path = str(local_engine)
    executable = shutil.which(engine_path) or (
        str(Path(engine_path)) if Path(engine_path).is_file() else None
    )
    if executable is None:
        raise FileNotFoundError(
            "Stockfish not found. Run python scripts/download_stockfish.py or pass --engine PATH"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        chess.engine.SimpleEngine.popen_uci(executable) as engine,
        destination.open("w", encoding="utf-8") as target,
    ):
        engine_id = engine.id.copy()
        if "UCI_ShowWDL" in engine.options:
            engine.configure({"UCI_ShowWDL": True})
        positions = (
            reservoir_sample(records(source), limit, sample_seed)
            if sample_seed is not None
            else records(source)
        )
        for row in positions:
            board = chess.Board(row["fen"])
            if not board.is_valid():
                raise ValueError(f"Invalid source FEN: {row['fen']}")
            if board.is_game_over():
                continue
            info = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=min(multipv, board.legal_moves.count()),
            )
            row["game_outcome"] = row.get("value")
            row.update(teacher_targets(info, board, temperature_cp))
            row["source"] = "stockfish"
            target.write(json.dumps(row) + "\n")
            count += 1
            if limit and count >= limit:
                break
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "engine": engine_id,
        "engine_sha256": sha256_file(Path(executable)),
        "depth": depth,
        "multipv": multipv,
        "temperature_cp": temperature_cp,
        "positions": count,
        "sample_seed": sample_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return count
