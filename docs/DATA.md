# Data

Use permitted standard-chess PGN. [Lichess database exports](https://database.lichess.org/) state that standard game exports are CC0; Lichess broadcast exports have a separate CC BY-SA 4.0 license. Check the exact source before reuse. Do not commit commercial books or large downloaded files.

`chess-ai prepare-data INPUT.pgn --output data/processed/NAME --min-elo 1800` streams games, filters for standard chess, known result, and both player ratings above the threshold, then writes `train.jsonl`, `validation.jsonl`, `test.jsonl`. The split is 80/10/10 by game, preserving all positions of a game in one split. The manifest stores source path and SHA-256, counts, filter, and seed. Exact duplicate positions across distinct games can still leak; deduplication across splits is required before publishing any measured validation accuracy.

The sample PGN in `examples/` is authored for this repository solely to exercise the pipeline. It is not a meaningful strength dataset. Large Lichess monthly files are tens of GB compressed; use a small legal subset or your own PGN first. Teacher labeling reads prepared JSONL and writes additional policy/value fields. The teacher executable is obtained separately; no binary is bundled.
