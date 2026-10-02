"""Student-only move selection and compact PUCT search."""

from dataclasses import dataclass
from time import perf_counter

import chess
import numpy as np
import torch

from chess_ai.core import action_index, encode_board
from chess_ai.model import PolicyValueNet


@dataclass
class SearchResult:
    move: chess.Move
    value: float
    nodes: int
    inferences: int
    elapsed: float
    candidates: list[tuple[str, float]]


def infer(
    model: PolicyValueNet, board: chess.Board, device: torch.device
) -> tuple[dict[chess.Move, float], float]:
    model.eval()
    with torch.inference_mode():
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
        logits, value = model(x)
        moves = list(board.legal_moves)
        indices = [action_index(move) for move in moves]
        probabilities = torch.softmax(logits[0, indices], dim=0).cpu().numpy()
    return dict(zip(moves, (float(p) for p in probabilities), strict=True)), float(value.item())


def policy_only(model: PolicyValueNet, board: chess.Board, device: torch.device) -> SearchResult:
    start = perf_counter()
    priors, value = infer(model, board, device)
    if not priors:
        raise ValueError("Cannot search terminal position")
    move = max(priors, key=priors.get)
    return SearchResult(
        move,
        value,
        1,
        1,
        perf_counter() - start,
        [(m.uci(), p) for m, p in sorted(priors.items(), key=lambda item: -item[1])[:5]],
    )


class Node:
    def __init__(self, prior: float = 1.0) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, Node] = {}

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def puct(
    model: PolicyValueNet,
    board: chess.Board,
    device: torch.device,
    simulations: int = 32,
    c_puct: float = 1.5,
) -> SearchResult:
    if board.is_game_over():
        raise ValueError("Cannot search terminal position")
    start = perf_counter()
    root = Node()
    priors, root_value = infer(model, board, device)
    root.children = {move: Node(prior) for move, prior in priors.items()}
    inferences = 1
    for _ in range(simulations):
        position = board.copy(stack=False)
        node = root
        path = [node]
        while node.children:
            move, child = max(
                node.children.items(),
                key=lambda item: (
                    -item[1].value
                    + c_puct * item[1].prior * np.sqrt(max(1, node.visits)) / (1 + item[1].visits)
                ),
            )
            position.push(move)
            node = child
            path.append(node)
        if position.is_game_over():
            outcome = position.outcome(claim_draw=True)
            value = (
                0.0
                if outcome is None or outcome.winner is None
                else (1.0 if outcome.winner == position.turn else -1.0)
            )
        else:
            leaf_priors, value = infer(model, position, device)
            node.children = {move: Node(prior) for move, prior in leaf_priors.items()}
            inferences += 1
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value
    move = max(root.children, key=lambda candidate: root.children[candidate].visits)
    total = sum(child.visits for child in root.children.values()) or 1
    candidates = sorted(
        ((m.uci(), child.visits / total) for m, child in root.children.items()),
        key=lambda item: -item[1],
    )[:5]
    return SearchResult(
        move, root_value, simulations, inferences, perf_counter() - start, candidates
    )
