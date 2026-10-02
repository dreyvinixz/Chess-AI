# Troubleshooting

- **`torch.cuda.is_available()` is false:** verify `wsl -l -v` reports WSL2, run `nvidia-smi` inside Ubuntu, and use the setup script's CUDA wheel. See [WSL2 guide](WSL2_CUDA.md).
- **GPU visible in Windows but not WSL:** update WSL and the Windows NVIDIA driver; do not install a Linux display driver inside WSL.
- **CUDA OOM:** lower `training.batch_size`, use `configs/tiny.yaml`, or close other GPU processes. The GTX 1650 also drives the desktop, reducing available VRAM.
- **Stockfish absent:** local play and PGN imitation still work. Teacher labeling requires a separately installed UCI Stockfish executable.
- **Empty training split:** use a larger PGN or inspect the Elo/result filters and dataset manifest. The sample file has one game and can land in any one split.
- **Browser command missing:** browser integration is still a roadmap item; no Chess.com automation is claimed.
