import chess
import numpy as np
import pytest

from chess_ai.core import (
    ACTION_SIZE,
    action_index,
    action_move,
    encode_board,
    legal_indices,
    select_legal,
)


@pytest.mark.parametrize(
    "fen,uci",
    [
        (chess.STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
    ],
)
def test_special_actions(fen, uci):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    assert action_move(action_index(move)) == move
    assert action_index(move) in legal_indices(board)


def test_legal_mask_never_selects_illegal_move():
    board = chess.Board()
    scores = np.zeros(ACTION_SIZE)
    scores[action_index(chess.Move.from_uci("e2e5"))] = 100
    selected = select_legal(board, scores)
    assert selected in board.legal_moves


def test_turn_and_castling_planes():
    board = chess.Board()
    white = encode_board(board)
    assert white.shape == (18, 8, 8)
    assert white[12].all()
    assert white[13].all() and white[14].all()
    board.push_uci("e2e4")
    black = encode_board(board)
    assert not black[12].any()
    assert black[17].sum() == 1
