# Search

`policy_only` masks all illegal actions and picks the highest probability legal move. `puct` runs configurable simulations with the student policy as prior and student value at leaves. Search returns a move, root value, nodes, inference count, elapsed time, and top candidate visit shares. Evaluation is deterministic; no Dirichlet noise is added. The implementation uses one network inference per expanded leaf and runs on Python; batching and transposition caching remain profiling candidates.

`chess-ai evaluate --search policy` and `--search puct --simulations N` compare the two approaches. Stockfish is never called by these functions.
