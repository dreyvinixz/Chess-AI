"""Small residual policy and value network."""

import torch
from torch import nn

from chess_ai.core import ACTION_SIZE, PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.layers(x))


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4, policy_channels: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *(ResidualBlock(channels) for _ in range(blocks)),
        )
        self.policy = nn.Sequential(
            nn.Conv2d(channels, policy_channels, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_channels * 64, ACTION_SIZE),
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(x)
        return self.policy(features), self.value(features).squeeze(-1)
