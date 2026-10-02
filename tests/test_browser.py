from dataclasses import replace

import chess
import pytest

from chess_ai.browser.board_reader import (
    BrowserSnapshot,
    BrowserStateError,
    HumanGameProtectionError,
    parse_snapshot,
)


def snapshot_for(board: chess.Board, moves: list[str] | None = None) -> BrowserSnapshot:
    pieces = [
        f"piece {'w' if piece.color else 'b'}{piece.symbol().lower()} "
        f"square-{chess.square_file(square) + 1}{chess.square_rank(square) + 1}"
        for square, piece in board.piece_map().items()
    ]
    return BrowserSnapshot(
        url="https://www.chess.com/play/computer",
        board_id="board-play-computer",
        piece_classes=pieces,
        coordinates=list("87654321abcdefgh"),
        san_moves=moves or [],
        bot_name="Observed bot",
        bot_rating_text="(250)",
        bot_speech=True,
        modal_open=False,
        human_name="Guest",
    )


def test_browser_reconstructs_fen_turn_bot_and_last_move():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("c5")
    observation = parse_snapshot(snapshot_for(board, ["e4", "c5"]))
    assert observation.board.fen() == board.fen()
    assert observation.board.turn == chess.WHITE
    assert observation.last_move == chess.Move.from_uci("c7c5")
    assert observation.bot_name == "Observed bot"
    assert observation.bot_rating == 250
    assert observation.orientation == "white"


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://www.chess.com/play/online"},
        {"url": "https://evil.example/play/computer"},
        {"board_id": "board-live"},
        {"bot_name": None},
        {"human_name": None},
        {"bot_speech": False},
        {"modal_open": True},
    ],
)
def test_bot_guard_fails_closed(change):
    with pytest.raises(HumanGameProtectionError, match="restricted"):
        parse_snapshot(replace(snapshot_for(chess.Board()), **change))


def test_browser_rejects_invalid_or_stale_piece_placement():
    snapshot = snapshot_for(chess.Board())
    with pytest.raises(BrowserStateError, match="disagree"):
        parse_snapshot(replace(snapshot, san_moves=["e4"]))
    with pytest.raises(BrowserStateError, match="same square"):
        duplicated = snapshot.piece_classes + snapshot.piece_classes[:1]
        parse_snapshot(replace(snapshot, piece_classes=duplicated))
    with pytest.raises(BrowserStateError, match="orientation"):
        parse_snapshot(replace(snapshot, coordinates=list("12345678abcdefgh")))


def test_browser_accepts_flipped_orientation():
    snapshot = snapshot_for(chess.Board())
    observed = parse_snapshot(replace(snapshot, coordinates=list("12345678hgfedcba")))
    assert observed.orientation == "black"
    assert observed.human_color == chess.BLACK
