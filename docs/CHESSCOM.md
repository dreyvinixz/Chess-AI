# Chess.com integration and safety

Browser integration is planned and is **not** exposed by the current CLI. The adapter must inspect the current computer-game UI instead of relying on old selector lists. It must positively establish a computer/bot game from multiple independent signals such as URL, opponent metadata, and DOM indicators. Human, live, rated, or ambiguous games must fail closed before any move. Parsed board state must be validated by python-chess, with orientation and side to move unambiguous. A mismatch after a move requires resynchronization or safe abort.

Assist mode will observe and record the human's actual move. Autoplay will use only student model/search moves. Stockfish may annotate a completed game offline. No Chess.com account credentials or cookies belong in this repository.
