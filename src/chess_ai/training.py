"""Supervised and distilled training with resumable checkpoints."""

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from chess_ai.core import ACTION_SIZE, PLANES, action_index, encode_board, legal_indices
from chess_ai.model import PolicyValueNet


class PositionDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: list[int] = []
        self._stream: BinaryIO | None = None
        self._stream_pid: int | None = None
        with path.open("rb") as stream:
            while line := stream.readline():
                if line.strip():
                    self.offsets.append(stream.tell() - len(line))
        if not self.offsets:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_stream"] = None
        state["_stream_pid"] = None
        return state

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._stream is None or self._stream_pid != os.getpid():
            if self._stream is not None:
                self._stream.close()
            self._stream = self.path.open("rb")
            self._stream_pid = os.getpid()
        self._stream.seek(self.offsets[index])
        row = json.loads(self._stream.readline())
        board = chess.Board(row["fen"])
        if not board.is_valid():
            raise ValueError(f"Invalid dataset FEN at row {index}: {row['fen']}")
        target = torch.zeros(ACTION_SIZE)
        if "policy" in row:
            for uci, probability in row["policy"].items():
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    target[action_index(move)] = float(probability)
        else:
            move = chess.Move.from_uci(row["move"])
            if move not in board.legal_moves:
                raise ValueError(f"Illegal dataset move: {move}")
            target[action_index(move)] = 1
        legal = torch.zeros(ACTION_SIZE, dtype=torch.bool)
        legal[legal_indices(board)] = True
        return (
            torch.from_numpy(encode_board(board)),
            target,
            torch.tensor(float(row["value"])),
            legal,
        )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "configs", "pyproject.toml"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    step: int,
    fingerprint: str,
    metrics: dict,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "config": config,
            "epoch": epoch,
            "global_step": step,
            "dataset_fingerprint": fingerprint,
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> tuple[PolicyValueNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    dimensions = checkpoint["config"]["model"]
    model = PolicyValueNet(**dimensions).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def probe_batch_size(
    model: PolicyValueNet,
    requested: int,
    device: torch.device,
    amp: bool,
    max_vram_mb: int,
) -> int:
    """Probe model forward/backward memory while leaving batch norm statistics untouched."""
    if device.type != "cuda":
        return requested
    free_bytes, _ = torch.cuda.mem_get_info(device)
    budget_mb = min(max_vram_mb, int(free_bytes / 1048576) - 256)
    if budget_mb <= 0:
        raise RuntimeError("Insufficient free GPU memory for training")
    model.eval()
    batch_size = requested
    while batch_size:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            features = torch.zeros((batch_size, PLANES, 8, 8), device=device)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits, value = model(features)
                loss = logits.square().mean() + value.square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / 1048576
            if peak_mb <= budget_mb:
                return batch_size
        except torch.OutOfMemoryError:
            pass
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        batch_size //= 2
    raise RuntimeError("Even a single training position exceeds the GPU memory budget")


def gpu_utilization() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    try:
        return int(result.stdout.splitlines()[0].strip()) if result.returncode == 0 else None
    except (IndexError, ValueError):
        return None


def train(config: dict, dataset_path: Path, run_dir: Path, resume: Path | None = None) -> Path:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config["training"]["epochs"]),
        eta_min=config["training"]["learning_rate"] * 0.1,
    )
    use_amp = device.type == "cuda" and config["training"]["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fingerprint = file_sha256(dataset_path)
    start_epoch = 0
    step = 0
    if resume:
        _, saved = load_model(resume, device)
        if saved["dataset_fingerprint"] != fingerprint:
            raise ValueError("Resume dataset fingerprint differs from the checkpoint")
        if saved["config"]["model"] != config["model"]:
            raise ValueError("Resume model configuration differs from the checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if saved.get("scheduler"):
            scheduler.load_state_dict(saved["scheduler"])
        if saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"]
        step = saved["global_step"]
    max_steps = config["training"]["max_steps"]
    if start_epoch >= config["training"]["epochs"] or (max_steps and step >= max_steps):
        raise ValueError("Resume target already reached; increase epochs and max_steps")
    dataset = PositionDataset(dataset_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "git_dirty": git_dirty(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "seed": config["seed"],
                "dataset_fingerprint": fingerprint,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    batch_size = probe_batch_size(
        model,
        config["training"]["batch_size"],
        device,
        use_amp,
        config["hardware"]["max_vram_mb"],
    )
    workers = config["training"]["workers"]
    shuffle_generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset, generator=shuffle_generator),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    path = run_dir / "checkpoint.pt"
    try:
        from torch.utils.tensorboard import SummaryWriter

        summary = SummaryWriter(log_dir=run_dir / "tensorboard")
    except ImportError:
        summary = None
    fieldnames = [
        "epoch",
        "step",
        "policy_loss",
        "value_loss",
        "loss",
        "entropy",
        "learning_rate",
        "vram_mb",
        "gpu_util_pct",
        "examples_per_sec",
        "batch_size",
        "microbatch_size",
        "epoch_elapsed_sec",
    ]
    with (run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if stream.tell() == 0:
            writer.writeheader()
        for epoch in range(start_epoch, config["training"]["epochs"]):
            shuffle_generator.manual_seed(config["seed"] + epoch)
            epoch_started = perf_counter()
            epoch_examples = 0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            for features, target, value, legal in loader:
                batch_started = perf_counter()
                features, target, value, legal = (
                    x.to(device, non_blocking=True) for x in (features, target, value, legal)
                )
                microbatch_size = len(features)
                while True:
                    try:
                        policy_total = value_total = loss_total = entropy_total = 0.0
                        for start in range(0, len(features), microbatch_size):
                            end = min(start + microbatch_size, len(features))
                            weight = (end - start) / len(features)
                            with torch.autocast(device_type=device.type, enabled=use_amp):
                                logits, estimate = model(features[start:end])
                                log_probs = F.log_softmax(
                                    logits.masked_fill(~legal[start:end], -1e4), dim=-1
                                )
                                policy_loss = -(target[start:end] * log_probs).sum(dim=-1).mean()
                                value_loss = F.mse_loss(estimate, value[start:end])
                                loss = policy_loss + config["training"]["value_weight"] * value_loss
                                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                            scaler.scale(
                                loss * weight / config["training"]["accumulation_steps"]
                            ).backward()
                            policy_total += float(policy_loss.detach()) * weight
                            value_total += float(value_loss.detach()) * weight
                            loss_total += float(loss.detach()) * weight
                            entropy_total += float(entropy.detach()) * weight
                        break
                    except torch.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True)
                        pending = 0
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        if microbatch_size == 1:
                            raise RuntimeError(
                                "CUDA OOM at one position; close GPU processes"
                            ) from None
                        microbatch_size = max(1, microbatch_size // 2)
                pending += 1
                if pending >= config["training"]["accumulation_steps"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step += 1
                epoch_examples += len(features)
                metrics = {
                    "epoch": epoch + 1,
                    "step": step,
                    "policy_loss": policy_total,
                    "value_loss": value_total,
                    "loss": loss_total,
                    "entropy": entropy_total,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "vram_mb": round(torch.cuda.max_memory_allocated() / 1048576)
                    if device.type == "cuda"
                    else 0,
                    "gpu_util_pct": gpu_utilization()
                    if device.type == "cuda" and step % 10 == 0
                    else None,
                    "examples_per_sec": len(features) / max(perf_counter() - batch_started, 1e-6),
                    "batch_size": len(features),
                    "microbatch_size": microbatch_size,
                    "epoch_elapsed_sec": perf_counter() - epoch_started,
                }
                writer.writerow(metrics)
                stream.flush()
                if summary:
                    for key in (
                        "policy_loss",
                        "value_loss",
                        "loss",
                        "entropy",
                        "learning_rate",
                        "vram_mb",
                        "examples_per_sec",
                    ):
                        summary.add_scalar(key, metrics[key], step)
                if max_steps and step >= max_steps:
                    break
            if pending:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            with (run_dir / "epochs.jsonl").open("a", encoding="utf-8") as epochs_stream:
                epochs_stream.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "duration_sec": perf_counter() - epoch_started,
                            "examples": epoch_examples,
                            "examples_per_sec": epoch_examples
                            / max(perf_counter() - epoch_started, 1e-6),
                            "batch_size": batch_size,
                        }
                    )
                    + "\n"
                )
            save_checkpoint(
                path,
                model,
                optimizer,
                config,
                epoch + 1,
                step,
                fingerprint,
                metrics,
                scheduler,
                scaler,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "git_commit": git_commit(),
                    "dataset_fingerprint": fingerprint,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                run_dir / "inference.pt",
            )
            if max_steps and step >= max_steps:
                break
    if summary:
        summary.close()
    return path
