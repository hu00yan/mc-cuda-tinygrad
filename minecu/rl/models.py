"""Pure-TinyGrad CNN policy and actor-critic models."""
from __future__ import annotations

from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters


class CNNBackbone:
    """3-layer convolutional encoder + MLP projection for [B, C, H, W] observations."""

    def __init__(self, obs_shape: tuple[int, int, int], hidden_dim: int = 128):
        h, w, c = obs_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        dummy = self._forward_conv(Tensor.zeros(1, c, h, w))
        flat_dim = int(dummy.reshape(1, -1).shape[1])
        self.fc = nn.Linear(flat_dim, hidden_dim)

    def _forward_conv(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        return x

    def __call__(self, x: Tensor) -> Tensor:
        x = self._forward_conv(x).flatten(1)
        return self.fc(x).relu()


class CNNPolicy:
    """CNN policy returning action logits [B, action_dim]."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int = 10,
        hidden_dim: int = 128,
    ):
        self.backbone = CNNBackbone(obs_shape, hidden_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.policy_head(self.backbone(x))

    def parameters(self) -> list[Tensor]:
        return get_parameters(self)


class CNNActorCritic:
    """Shared CNN backbone with policy logits [B, action_dim] and value [B] heads."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int = 10,
        hidden_dim: int = 128,
    ):
        self.backbone = CNNBackbone(obs_shape, hidden_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.backbone(x)
        logits = self.policy_head(feat)
        values = self.value_head(feat).squeeze(-1)
        return logits, values

    def parameters(self) -> list[Tensor]:
        return get_parameters(self)
