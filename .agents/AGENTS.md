# Repository agent instructions

Scope: `C:/Users/User/Chess-AI` only. Do not delete existing files or change files outside this repository. Preserve the Git history; never force push. Use Conventional Commits for real, tested units of work.

Read `task.md` and `ROADMAP.md` before a new sprint. Keep them synchronized with evidence and mark a task complete only after code, tests, docs, and commit. Public docs and code identifiers are in English.

The primary runtime is native Ubuntu on WSL2 with a GTX 1650 4 GB. Run CUDA checks there. CPU CI must not require CUDA. Never use Stockfish during student move selection. Browser moves are allowed only after positive, multi-signal verification of a Chess.com computer/bot game. On uncertain board state or opponent identity, stop automation.

Project-local skills are in `.agents/skills/`. They were downloaded from the curated `openai/skills` repository with the skill installer and are reference material for CLI, Playwright, and security work. Keep skill use subordinate to the project's requirements.

Do not commit credentials, cookies, browser profiles, large datasets, or weights. Record all reported measurements from actual runs.
