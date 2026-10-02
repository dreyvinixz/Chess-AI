"""Chess position planes and a lossless legal action index."""

import chess
import numpy as np

PLANES = 18
ACTION_SIZE = 64 * 64 * 5
PROMOTIONS = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
        planes[plane, chess.square_rank(square), chess.square_file(square)] = 1
    planes[12] = float(board.turn == chess.WHITE)
    for offset, color in enumerate((chess.WHITE, chess.BLACK)):
        planes[13 + 2 * offset] = float(board.has_kingside_castling_rights(color))
        planes[14 + 2 * offset] = float(board.has_queenside_castling_rights(color))
    if board.ep_square is not None:
        planes[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    return planes


def action_index(move: chess.Move) -> int:
    try:
        promotion = PROMOTIONS.index(move.promotion)
    except ValueError as exc:
        raise ValueError(f"Unsupported promotion: {move}") from exc
    return (move.from_square * 64 + move.to_square) * 5 + promotion


def action_move(index: int) -> chess.Move:
    if not 0 <= index < ACTION_SIZE:
        raise ValueError(f"Action out of range: {index}")
    pair, promotion = divmod(index, 5)
    source, target = divmod(pair, 64)
    return chess.Move(source, target, promotion=PROMOTIONS[promotion])


def legal_indices(board: chess.Board) -> list[int]:
    return [action_index(move) for move in board.legal_moves]


def select_legal(board: chess.Board, logits: np.ndarray) -> chess.Move:
    indices = legal_indices(board)
    if not indices:
        raise ValueError("No legal move in terminal position")
    return action_move(max(indices, key=lambda index: float(logits[index])))
