# Model design decision (ADR-001)

**Status:** provisional until profiling and strength measurements on GTX 1650 are complete.

The first implementation uses 18 board planes, a configurable residual CNN, a 20,480-action policy head, and a side-to-move value head. `configs/gtx1650.yaml` chooses 64 channels and four residual blocks; `debug.yaml` makes CPU and smoke runs cheap. All legal moves, including castling, en passant, and four promotion pieces, have distinct indices. Illegal logits are masked.

| Candidate | Advantage | Cost or risk | Decision |
|---|---|---|---|
| AlphaZero-scale self-play from scratch | Clean learning signal | Prohibitive games and search budget on 4 GB | Deferred |
| Strong-game PGN imitation | Cheap warm start | Human move quality varies | Implemented |
| Offline Stockfish distillation | Strong labels and values | CPU labeling time; teacher bias | Implemented |
| Compact policy/value with PUCT | Student participates directly | Python search overhead | Implemented for comparison |
| Alpha-beta hybrid | Tactical potential | Requires ordering/evaluation engineering | Future comparison |
| Pretrained Lc0/Maia weights | Faster strength | Architecture, license, size, and fine-tuning mismatch | No weights selected |
| Opening book | Fast openings | Must be separately identified in results | Future optional feature |
| Incremental experience and self-play | Adaptation to user and bots | Can regress without quality controls | Planned |

Books of prose are not structured policy labels. The data pipeline uses PGN and FEN. The policy head is intentionally simple and larger than a 73-plane convolutional head; profiling will determine whether a spatial head is worth the complexity. No speed or strength superiority is claimed without a benchmark.
