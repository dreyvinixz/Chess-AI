"""Reconstruct complete chess state from visible DOM pieces and SAN history."""

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import chess


class BrowserStateError(RuntimeError):
    """The visible board cannot be proved to match one legal game state."""


class HumanGameProtectionError(BrowserStateError):
    """The page does not positively identify a bot-only game."""


@dataclass(frozen=True)
class BrowserSnapshot:
    url: str
    board_id: str
    piece_classes: list[str]
    coordinates: list[str]
    san_moves: list[str]
    bot_name: str | None
    bot_rating_text: str | None
    bot_speech: bool
    modal_open: bool
    human_name: str | None = None
    selection_open: bool = False


@dataclass(frozen=True)
class ObservedGame:
    board: chess.Board
    orientation: str
    human_color: chess.Color
    bot_name: str
    bot_rating: int | None
    last_move: chess.Move | None


PIECE_RE = re.compile(r"^[wb][pnbrqk]$")
SQUARE_RE = re.compile(r"^square-([1-8])([1-8])$")


def piece_placement(piece_classes: list[str]) -> dict[int, chess.Piece]:
    pieces: dict[int, chess.Piece] = {}
    for class_name in piece_classes:
        tokens = class_name.split()
        if "piece" not in tokens:
            raise BrowserStateError(f"Unexpected board element: {class_name}")
        symbols = [token for token in tokens if PIECE_RE.fullmatch(token)]
        squares = [SQUARE_RE.fullmatch(token) for token in tokens]
        squares = [match for match in squares if match is not None]
        if len(symbols) != 1 or len(squares) != 1:
            raise BrowserStateError(f"Ambiguous piece element: {class_name}")
        symbol = symbols[0]
        square = chess.square(int(squares[0][1]) - 1, int(squares[0][2]) - 1)
        if square in pieces:
            raise BrowserStateError("Two DOM pieces occupy the same square")
        piece_symbol = symbol[1].upper() if symbol[0] == "w" else symbol[1]
        pieces[square] = chess.Piece.from_symbol(piece_symbol)
    if len(pieces) < 2 or len(pieces) > 32:
        raise BrowserStateError("Unexpected piece count")
    return pieces


def board_orientation(coordinates: list[str]) -> str:
    labels = [label.strip().lower() for label in coordinates]
    ranks = [label for label in labels if label in {str(number) for number in range(1, 9)}]
    files = [label for label in labels if label in "abcdefgh" and len(label) == 1]
    if len(ranks) != 8 or len(files) != 8:
        raise BrowserStateError("Missing or ambiguous board coordinate labels")
    if ranks == list("87654321") and files == list("abcdefgh"):
        return "white"
    if ranks == list("12345678") and files == list("hgfedcba"):
        return "black"
    raise BrowserStateError("Unknown board orientation")


class BotModeGuard:
    """Require independent current-page, board, and bot identity signals."""

    @staticmethod
    def verify(snapshot: BrowserSnapshot) -> None:
        parsed = urlparse(snapshot.url)
        valid_url = (
            parsed.scheme == "https"
            and parsed.hostname in {"chess.com", "www.chess.com"}
            and parsed.path.rstrip("/") == "/play/computer"
        )
        if not (
            valid_url
            and snapshot.board_id == "board-play-computer"
            and snapshot.bot_name
            and snapshot.bot_name.strip()
            and snapshot.human_name
            and snapshot.human_name.strip()
            and snapshot.bot_name.strip() != snapshot.human_name.strip()
            and snapshot.bot_speech
            and not snapshot.modal_open
        ):
            raise HumanGameProtectionError(
                "Autoplay is restricted to Chess.com computer/bot games. "
                "URL, board, bot metadata, and bot marker must agree."
            )


def parse_snapshot(snapshot: BrowserSnapshot) -> ObservedGame:
    BotModeGuard.verify(snapshot)
    orientation = board_orientation(snapshot.coordinates)
    placement = piece_placement(snapshot.piece_classes)
    board = chess.Board()
    last_move = None
    for san in snapshot.san_moves:
        try:
            last_move = board.parse_san(san.strip())
        except ValueError as error:
            raise BrowserStateError(f"Cannot parse mainline SAN move {san!r}") from error
        board.push(last_move)
    if not board.is_valid():
        raise BrowserStateError("Reconstructed board is invalid")
    if board.piece_map() != placement:
        raise BrowserStateError("DOM pieces disagree with the reconstructed move history")
    rating_match = re.search(r"\d+", snapshot.bot_rating_text or "")
    rating = int(rating_match.group()) if rating_match else None
    return ObservedGame(
        board=board,
        orientation=orientation,
        human_color=chess.WHITE if orientation == "white" else chess.BLACK,
        bot_name=snapshot.bot_name.strip(),
        bot_rating=rating,
        last_move=last_move,
    )
