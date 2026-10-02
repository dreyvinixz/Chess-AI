"""SQLite human experience with offline teacher quality annotation."""

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine

from chess_ai.data import sha256_file
from chess_ai.engines import StockfishAdapter
from chess_ai.teacher import score_cp


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def quality_weight(cp_loss: int, scale_cp: float = 120.0) -> float:
    """Give a blunder little imitation weight while retaining teacher value learning."""
    if scale_cp <= 0:
        raise ValueError("scale_cp must be positive")
    return max(0.02, math.exp(-max(0, cp_loss) / scale_cp))


class ExperienceStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                opponent TEXT NOT NULL,
                opponent_rating INTEGER,
                checkpoint TEXT NOT NULL,
                human_color TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                result TEXT NOT NULL DEFAULT '*'
            );
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                game_id INTEGER NOT NULL REFERENCES games(id),
                ply INTEGER NOT NULL,
                fen TEXT NOT NULL,
                legal_moves TEXT NOT NULL,
                human_move TEXT NOT NULL,
                model_policy TEXT NOT NULL,
                model_value REAL NOT NULL,
                search_nodes INTEGER NOT NULL,
                search_elapsed_sec REAL NOT NULL,
                teacher_move TEXT,
                teacher_value REAL,
                teacher_best_cp INTEGER,
                teacher_chosen_cp INTEGER,
                policy_weight REAL,
                timestamp TEXT NOT NULL,
                UNIQUE(game_id, ply)
            );
            """
        )
        self.connection.commit()

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.connection.close()

    def create_game(
        self,
        source: str,
        opponent: str,
        checkpoint: str,
        human_color: str,
        opponent_rating: int | None = None,
    ) -> int:
        if human_color not in {"white", "black"}:
            raise ValueError("Human color must be white or black")
        cursor = self.connection.execute(
            """INSERT INTO games
            (source, opponent, opponent_rating, checkpoint, human_color, started_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (source, opponent, opponent_rating, checkpoint, human_color, now_utc()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_human_move(
        self,
        game_id: int,
        board: chess.Board,
        move: chess.Move,
        model_policy: dict[str, float],
        model_value: float,
        search_nodes: int,
        search_elapsed_sec: float,
    ) -> None:
        if not board.is_valid() or move not in board.legal_moves:
            raise ValueError("Cannot record an invalid board or illegal human move")
        if not model_policy or any(
            chess.Move.from_uci(uci) not in board.legal_moves
            or not math.isfinite(probability)
            or probability < 0
            for uci, probability in model_policy.items()
        ) or sum(model_policy.values()) <= 0:
            raise ValueError("Model policy must contain only legal nonnegative moves")
        if not math.isfinite(model_value) or not -1 <= model_value <= 1:
            raise ValueError("Model value must lie within [-1, 1]")
        self.connection.execute(
            """INSERT INTO positions
            (game_id, ply, fen, legal_moves, human_move, model_policy, model_value,
             search_nodes, search_elapsed_sec, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id,
                board.ply(),
                board.fen(),
                json.dumps([legal.uci() for legal in board.legal_moves]),
                move.uci(),
                json.dumps(model_policy),
                model_value,
                search_nodes,
                search_elapsed_sec,
                now_utc(),
            ),
        )
        self.connection.commit()

    def finish_game(self, game_id: int, result: str) -> None:
        if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
            raise ValueError("Invalid PGN game result")
        cursor = self.connection.execute(
            "UPDATE games SET result=?, finished_at=? WHERE id=?",
            (result, now_utc(), game_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Unknown game id: {game_id}")
        self.connection.commit()

    def game(self, game_id: int) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM games WHERE id=?", (game_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown game id: {game_id}")
        return row

    def annotate_game(
        self,
        game_id: int,
        engine_path: str = "stockfish",
        depth: int = 8,
        scale_cp: float = 120.0,
    ) -> dict:
        game = self.game(game_id)
        if game["result"] == "*":
            raise ValueError("Finish the game before offline annotation")
        positions = self.connection.execute(
            "SELECT * FROM positions WHERE game_id=? ORDER BY ply", (game_id,)
        ).fetchall()
        if not positions:
            raise ValueError("Game has no recorded human positions")
        with StockfishAdapter(engine_path, depth=depth) as adapter:
            assert adapter.engine is not None
            for row in positions:
                board = chess.Board(row["fen"])
                turn = board.turn
                best_info = adapter.engine.analyse(board, chess.engine.Limit(depth=depth))
                best_cp = score_cp(best_info["score"], turn)
                best_move = best_info["pv"][0]
                chosen_info = adapter.engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    root_moves=[chess.Move.from_uci(row["human_move"])],
                )
                chosen_cp = score_cp(chosen_info["score"], turn)
                teacher_value = math.tanh(best_cp / 600)
                weight = quality_weight(best_cp - chosen_cp, scale_cp)
                self.connection.execute(
                    """UPDATE positions SET teacher_move=?, teacher_value=?,
                    teacher_best_cp=?, teacher_chosen_cp=?, policy_weight=? WHERE id=?""",
                    (best_move.uci(), teacher_value, best_cp, chosen_cp, weight, row["id"]),
                )
            provenance = adapter.provenance()
        self.connection.commit()
        return {"game_id": game_id, "positions": len(positions), "teacher": provenance}

    def export_training(self, destination: Path, min_weight: float = 0.02) -> dict:
        if not 0 <= min_weight <= 1:
            raise ValueError("min_weight must lie within [0, 1]")
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with destination.open("w", encoding="utf-8") as target:
            rows = self.connection.execute(
                """SELECT p.*, g.result, g.opponent, g.checkpoint FROM positions p
                JOIN games g ON p.game_id=g.id
                WHERE p.teacher_value IS NOT NULL AND p.policy_weight >= ?
                ORDER BY p.game_id, p.ply""",
                (min_weight,),
            )
            for row in rows:
                target.write(
                    json.dumps(
                        {
                            "fen": row["fen"],
                            "move": row["human_move"],
                            "value": row["teacher_value"],
                            "policy_weight": row["policy_weight"],
                            "source": "annotated_human_experience",
                            "game_id": row["game_id"],
                            "ply": row["ply"],
                            "teacher_move": row["teacher_move"],
                            "teacher_best_cp": row["teacher_best_cp"],
                            "teacher_chosen_cp": row["teacher_chosen_cp"],
                            "result": row["result"],
                            "opponent": row["opponent"],
                            "checkpoint": row["checkpoint"],
                        }
                    )
                    + "\n"
                )
                count += 1
        manifest = {
            "schema_version": 1,
            "source": str(self.path),
            "output": str(destination),
            "output_sha256": sha256_file(destination),
            "positions": count,
            "min_weight": min_weight,
            "created_at": now_utc(),
        }
        destination.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest
