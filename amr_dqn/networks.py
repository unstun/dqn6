from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 128, hidden_layers: int = 2):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Backwards-compatible name (historically this repo only had an MLP Q-network).
QNetwork = MLPQNetwork


@dataclass(frozen=True)
class FlatObsCnnLayout:
    scalar_dim: int
    map_channels: int
    map_size: int


def infer_flat_obs_cnn_layout(obs_dim: int) -> FlatObsCnnLayout:
    """Infer (scalar_dim, map_channels, map_size) for this repo's flat observations.

    Supported layouts:
    - AMRGridEnv:   obs = [5 scalars] + [1 * (N*N) map]
    - AMRBicycleEnv:obs = [11 scalars] + [3 * (N*N) maps]  (occ + cost + edt)
    """

    d = int(obs_dim)
    if d <= 0:
        raise ValueError("obs_dim must be > 0")

    candidates: list[FlatObsCnnLayout] = []
    for scalar_dim, channels in ((5, 1), (11, 3)):
        rem = d - int(scalar_dim)
        if rem <= 0:
            continue
        if rem % int(channels) != 0:
            continue
        per = rem // int(channels)
        n = int(round(math.sqrt(per)))
        if n > 0 and n * n == per:
            candidates.append(FlatObsCnnLayout(scalar_dim=int(scalar_dim), map_channels=int(channels), map_size=int(n)))

    if not candidates:
        raise ValueError(
            f"Cannot infer CNN layout from obs_dim={d}. Expected 5+N^2 (grid) or 11+3*N^2 (bicycle)."
        )
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous CNN layout for obs_dim={d}: {candidates}")
    return candidates[0]


class CNNQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        scalar_dim: int,
        map_channels: int,
        map_size: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        self.scalar_dim = int(scalar_dim)
        self.map_channels = int(map_channels)
        self.map_size = int(map_size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        if self.scalar_dim < 0:
            raise ValueError("scalar_dim must be >= 0")
        if self.map_channels < 1:
            raise ValueError("map_channels must be >= 1")
        if self.map_size < 1:
            raise ValueError("map_size must be >= 1")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        expected = int(self.scalar_dim) + int(self.map_channels) * int(self.map_size) * int(self.map_size)
        if int(input_dim) != expected:
            raise ValueError(
                f"CNNQNetwork expected input_dim={expected} (scalar_dim={self.scalar_dim}, "
                f"map_channels={self.map_channels}, map_size={self.map_size}), got {int(input_dim)}"
            )

        # A real 2D CNN over the downsampled global maps. Designed for small maps (e.g. 12x12).
        self.conv = nn.Sequential(
            nn.Conv2d(self.map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, self.map_channels, self.map_size, self.map_size), dtype=torch.float32)
            conv_out = self.conv(dummy)
            conv_out_dim = int(conv_out.flatten(start_dim=1).shape[1])
        fc_in_dim = int(self.scalar_dim) + int(conv_out_dim)

        layers: list[nn.Module] = []
        layers.append(nn.Linear(fc_in_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        for _ in range(int(hidden_layers) - 1):
            layers.append(nn.Linear(int(hidden_dim), int(hidden_dim)))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(int(hidden_dim), int(output_dim)))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError("CNNQNetwork expects (batch, obs_dim) input")
        if int(x.shape[1]) != int(self.input_dim):
            raise ValueError(f"CNNQNetwork expected input_dim={self.input_dim}, got {int(x.shape[1])}")

        scalars = x[:, : self.scalar_dim]
        maps_flat = x[:, self.scalar_dim :]
        maps = maps_flat.reshape(int(x.shape[0]), self.map_channels, self.map_size, self.map_size)
        conv = self.conv(maps)
        conv_flat = conv.flatten(start_dim=1)
        feats = torch.cat([scalars, conv_flat], dim=1)
        return self.head(feats)
