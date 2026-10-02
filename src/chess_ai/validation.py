"""Held-out policy and value metrics for a saved student checkpoint."""

from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chess_ai.data import sha256_file
from chess_ai.model import PolicyValueNet
from chess_ai.training import PositionDataset


def validate_dataset(
    model: PolicyValueNet,
    device: torch.device,
    dataset_path: Path,
    checkpoint: Path,
    batch_size: int = 32,
) -> dict:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    dataset = PositionDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=device.type == "cuda")
    model.eval()
    policy_sum = value_sum = top1_sum = 0.0
    positions = 0
    with torch.inference_mode():
        for features, target, value, legal in loader:
            features, target, value, legal = (
                tensor.to(device, non_blocking=True)
                for tensor in (features, target, value, legal)
            )
            logits, estimate = model(features)
            masked = logits.masked_fill(~legal, -1e4)
            log_probs = F.log_softmax(masked, dim=-1)
            policy_sum += float(-(target * log_probs).sum())
            value_sum += float(F.mse_loss(estimate, value, reduction="sum"))
            top1_sum += int((masked.argmax(dim=-1) == target.argmax(dim=-1)).sum())
            positions += len(features)
    return {
        "positions": positions,
        "policy_cross_entropy": policy_sum / positions,
        "value_mse": value_sum / positions,
        "top1_target_agreement": top1_sum / positions,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
