# Model design decision (ADR-001)

**Status:** provisional until profiling and strength measurements on GTX 1650 are complete.

The first implementation uses 18 board planes, a configurable residual CNN, a 20,480-action policy head, and a side-to-move value head. `configs/gtx1650.yaml` chooses 64 channels and four residual blocks; `debug.yaml` makes CPU and smoke runs cheap. All legal moves, including castling, en passant, and four promotion pieces, have distinct indices. Illegal logits are masked. The policy projection uses two channels before flattening.

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

Books of prose are not structured policy labels. The data pipeline uses PGN and FEN. An initial 16-channel policy projection produced 21,016,149 parameters and a 252 MB training checkpoint on the debug profile. Reducing that projection to two channels produced 2,665,831 parameters and a 32 MB checkpoint, with the same action mapping. The local CUDA inference timings were 1.48 ms over 10 iterations before and 2.19 ms over 100 iterations after. These runs establish the size reduction; they do not establish a speed gain. See the [measured reports](../reports/). A spatial 73-plane head remains a future comparison. No strength improvement is implied by this change.
