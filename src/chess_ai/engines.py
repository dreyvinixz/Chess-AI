"""External UCI engine adapter for offline teaching and explicit baselines."""

import shutil
from pathlib import Path

import chess
import chess.engine

from chess_ai.data import sha256_file


def resolve_stockfish(path: str = "stockfish") -> str:
    local = Path.cwd() / "tools" / "stockfish-local" / "stockfish"
    if path == "stockfish" and local.is_file():
        path = str(local)
    resolved = shutil.which(path) or (str(Path(path)) if Path(path).is_file() else None)
    if resolved is None:
        raise FileNotFoundError(
            "Stockfish not found. Run python scripts/download_stockfish.py or pass --engine PATH"
        )
    return resolved


class StockfishAdapter:
    """A clearly identified opponent; never called inside student search."""

    def __init__(
        self,
        path: str = "stockfish",
        depth: int = 4,
        nodes: int = 0,
        time_sec: float = 0.0,
        elo: int | None = None,
    ) -> None:
        if depth < 0 or nodes < 0 or time_sec < 0:
            raise ValueError("Engine limits cannot be negative")
        if not (depth or nodes or time_sec):
            raise ValueError("At least one Stockfish search limit is required")
        self.executable = resolve_stockfish(path)
        self.depth = depth
        self.nodes = nodes
        self.time_sec = time_sec
        self.elo = elo
        self.engine: chess.engine.SimpleEngine | None = None
        self.engine_id: dict = {}

    def __enter__(self) -> "StockfishAdapter":
        self.engine = chess.engine.SimpleEngine.popen_uci(self.executable)
        self.engine_id = self.engine.id.copy()
        if self.elo is not None:
            if (
                "UCI_LimitStrength" not in self.engine.options
                or "UCI_Elo" not in self.engine.options
            ):
                self.engine.quit()
                self.engine = None
                raise ValueError("This UCI engine does not support Elo limiting")
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def move(self, board: chess.Board) -> chess.Move:
        if self.engine is None:
            raise RuntimeError("StockfishAdapter must be opened before play")
        limit = chess.engine.Limit(
            depth=self.depth or None,
            nodes=self.nodes or None,
            time=self.time_sec or None,
        )
        move = self.engine.play(board, limit).move
        if move not in board.legal_moves:
            raise RuntimeError("External engine returned an illegal move")
        return move

    def provenance(self) -> dict:
        return {
            "engine": self.engine_id,
            "engine_sha256": sha256_file(Path(self.executable)),
            "depth": self.depth or None,
            "nodes": self.nodes or None,
            "time_sec": self.time_sec or None,
            "elo": self.elo,
        }
