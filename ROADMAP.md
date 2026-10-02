# Roadmap

Chess-AI aims to measure how far a locally trained, compact chess student can progress on a GTX 1650 with 4 GB VRAM. Beating every Chess.com bot is an experimental goal, not a release promise.

| Sprint | Deliverable | Exit evidence | Status |
|---|---|---|---|
| 0 | Audit, ADR, reproducible work plan | `task.md`, source links, architecture note | Complete |
| 1 | Local CUDA vertical slice | Special-move tests, smoke test, CI | Complete: PR #7, CI passed |
| 2 | PGN, Stockfish teacher, trainer | Dataset manifest, training checkpoint, resume | Complete: PR #8, CI passed |
| 3 | Search, self-play, evaluation | Local match JSON, speed and VRAM report | In progress: issue #3 |
| 4 | Assist and incremental experience | SQLite records, post-game quality weights | Planned |
| 5 | Bot-only browser adapter and ladder | Guard tests, current DOM fixtures, real games | Planned |
| 6 | Release quality and dashboard | Docs, reproducibility replay, CI, PR | Planned |

## Acceptance rules

Each deliverable needs executable code, meaningful tests, English documentation, reproducible commands, and a conventional commit. Results must identify the checkpoint, configuration, data fingerprint, opponent, and whether an opening book or external engine was used. Stockfish is excluded from student move selection.

## Branch and issue plan

The first commit establishes the empty repository on `main`. Subsequent sprints use `feat/sprint-N-topic` branches and focused GitHub issues. PRs will describe scope, evidence, measured limits, and remaining work. GitHub operations require a working authenticated connection; local work proceeds independently.
