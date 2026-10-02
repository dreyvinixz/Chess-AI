# WSL2 + CUDA setup

The supported path is Windows → WSL2 Ubuntu → Python virtual environment → PyTorch CUDA → GTX 1650. The host must have a compatible [Windows NVIDIA driver](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). NVIDIA explicitly says **do not install a Linux NVIDIA display driver inside WSL**. The Windows driver supplies the CUDA stub to WSL.

1. In Windows PowerShell, run `wsl --install` if needed, then `wsl --update` and `wsl -l -v`. Confirm Ubuntu is version 2.
2. Enter Ubuntu with `wsl`. Run `nvidia-smi` and confirm the GTX 1650 is listed. The CUDA version shown by `nvidia-smi` is the driver's supported version, not necessarily PyTorch's bundled runtime.
3. Install Python 3.12, its `venv` support, and Git using your Ubuntu distribution's supported packages or a trusted Python installer. The current test environment is Ubuntu 26.04 with Python 3.12.13.
4. Clone this repository and run `bash scripts/setup_wsl.sh`. Activate with `source .venv/bin/activate`.
5. Run `chess-ai doctor` and `chess-ai smoke-test`.

The script installs PyTorch 2.6.0 with a CUDA 12.4 wheel from the [official PyTorch wheel index](https://download.pytorch.org/whl/cu124) and then installs this package. A standalone CUDA Toolkit is not needed to run the wheel. Verify manually with:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(torch.cuda.get_device_name(0), p.total_memory, (p.major, p.minor))
PY
```

If CUDA is unavailable, inspect `nvidia-smi` inside WSL first. Update the Windows driver and WSL kernel before changing Python packages. A driver/runtime mismatch may require a supported PyTorch wheel; consult [PyTorch's current installation selector](https://pytorch.org/get-started/locally/). For OOM, reduce batch size and account for memory used by Windows graphics processes. Do not install `nvidia-driver-*` or `cuda-drivers` inside WSL.
