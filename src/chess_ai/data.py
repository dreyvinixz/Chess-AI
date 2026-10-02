"""Streaming PGN preparation and line oriented training records."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn


def records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pgn(
    source: Path,
    destination: Path,
    min_elo: int = 1800,
    max_games: int = 0,
    seed: int = 42,
    deduplicate: bool = True,
) -> dict:
    """Split by game before sampling positions to keep adjacent plies together."""
    destination.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "test": 0}
    duplicates_skipped = 0
    seen = sqlite3.connect(destination / "seen_positions.sqlite3") if deduplicate else None
    if seen:
        seen.execute("CREATE TABLE IF NOT EXISTS seen (key BLOB PRIMARY KEY) WITHOUT ROWID")
        seen.execute("DELETE FROM seen")
    handles = {
        split: (destination / f"{split}.jsonl").open("w", encoding="utf-8") for split in counts
    }
    accepted = 0
    try:
        with source.open(encoding="utf-8", errors="replace") as stream:
            while game := chess.pgn.read_game(stream):
                if game.errors:
                    continue
                try:
                    white_elo = int(game.headers.get("WhiteElo", "0"))
                    black_elo = int(game.headers.get("BlackElo", "0"))
                except ValueError:
                    continue
                if min(white_elo, black_elo) < min_elo or game.headers.get("Result") not in {
                    "1-0",
                    "0-1",
                    "1/2-1/2",
                }:
                    continue
                if game.headers.get("Variant", "Standard") != "Standard":
                    continue
                split_value = (
                    int.from_bytes(
                        hashlib.sha256(
                            f"{seed}:{game.headers.get('Site', '')}:{accepted}".encode()
                        ).digest()[:4],
                        "big",
                    )
                    % 10
                )
                split = "train" if split_value < 8 else "validation" if split_value == 8 else "test"
                board = game.board()
                result = game.headers["Result"]
                white_outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
                for move in game.mainline_moves():
                    if not board.is_legal(move):
                        break
                    if seen:
                        key = hashlib.sha256(board.epd().encode()).digest()
                        inserted = seen.execute(
                            "INSERT OR IGNORE INTO seen(key) VALUES (?)", (key,)
                        )
                        if inserted.rowcount == 0:
                            duplicates_skipped += 1
                            board.push(move)
                            continue
                    row = {
                        "fen": board.fen(),
                        "move": move.uci(),
                        "value": white_outcome if board.turn else -white_outcome,
                        "source": "pgn",
                    }
                    handles[split].write(json.dumps(row) + "\n")
                    counts[split] += 1
                    board.push(move)
                accepted += 1
                if seen and accepted % 100 == 0:
                    seen.commit()
                if max_games and accepted >= max_games:
                    break
    finally:
        for handle in handles.values():
            handle.close()
        if seen:
            seen.commit()
            seen.close()
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "sha256": sha256_file(source),
        "files": {split: sha256_file(destination / f"{split}.jsonl") for split in counts},
        "games": accepted,
        "positions": counts,
        "duplicates_skipped": duplicates_skipped,
        "deduplicate": deduplicate,
        "min_elo": min_elo,
        "split_seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
