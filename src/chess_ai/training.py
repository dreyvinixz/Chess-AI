"""Supervised and distilled training with resumable checkpoints."""

import csv
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chess_ai.core import ACTION_SIZE, action_index, encode_board, legal_indices
from chess_ai.data import records
from chess_ai.model import PolicyValueNet


class PositionDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = list(records(path))
        if not self.rows:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        board = chess.Board(row["fen"])
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


def save_checkpoint(
    path: Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    step: int,
    fingerprint: str,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
    model = PolicyValueNet(dimensions["channels"], dimensions["blocks"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def train(config: dict, dataset_path: Path, run_dir: Path, resume: Path | None = None) -> Path:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    start_epoch = 0
    step = 0
    if resume:
        _, saved = load_model(resume, device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"]
        step = saved["global_step"]
    dataset = PositionDataset(dataset_path)
    fingerprint = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
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
    batch_size = config["training"]["batch_size"]
    workers = config["training"]["workers"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    use_amp = device.type == "cuda" and config["training"]["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    path = run_dir / "checkpoint.pt"
    with (run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "step", "policy_loss", "value_loss", "loss", "vram_mb"]
        )
        if stream.tell() == 0:
            writer.writeheader()
        for epoch in range(start_epoch, config["training"]["epochs"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for features, target, value, legal in loader:
                features, target, value, legal = (
                    x.to(device) for x in (features, target, value, legal)
                )
                try:
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        logits, estimate = model(features)
                        log_probs = F.log_softmax(logits.masked_fill(~legal, -1e4), dim=-1)
                        policy_loss = -(target * log_probs).sum(dim=-1).mean()
                        value_loss = F.mse_loss(estimate, value)
                        loss = policy_loss + config["training"]["value_weight"] * value_loss
                    scaler.scale(loss / config["training"]["accumulation_steps"]).backward()
                    if (step + 1) % config["training"]["accumulation_steps"] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                except torch.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM: lower training.batch_size in the config"
                    ) from None
                step += 1
                metrics = {
                    "epoch": epoch + 1,
                    "step": step,
                    "policy_loss": float(policy_loss),
                    "value_loss": float(value_loss),
                    "loss": float(loss),
                    "vram_mb": round(torch.cuda.max_memory_allocated() / 1048576)
                    if device.type == "cuda"
                    else 0,
                }
                writer.writerow(metrics)
                stream.flush()
                if config["training"]["max_steps"] and step >= config["training"]["max_steps"]:
                    break
            save_checkpoint(path, model, optimizer, config, epoch + 1, step, fingerprint, metrics)
            if config["training"]["max_steps"] and step >= config["training"]["max_steps"]:
                break
    return path
