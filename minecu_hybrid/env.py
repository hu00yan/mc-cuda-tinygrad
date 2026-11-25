"""Stable upper API — backend is cuda (perf) or tinygrad (rides codegen)."""
from __future__ import annotations

import numpy as np

from .backend import make_backend
from .config import BUTTON_DIM, BUTTON_NAMES, LOOK_DIM, BlockType, EnvConfig


class MineCUEnv:
    """Batched MineCU RL env.

    Parameters mirror Infatoshi ``MineEnv`` where practical. Pass
    ``backend="cuda"`` (default on NVIDIA) or ``backend="tinygrad"``.
    """

    def __init__(self, cfg: EnvConfig | None = None, **kwargs):
        if cfg is None:
            cfg = EnvConfig.from_dict(kwargs)
        elif kwargs:
            d = {**cfg.__dict__, **kwargs}
            cfg = EnvConfig.from_dict(d)
        self.cfg = cfg
        self._backend = make_backend(cfg)

    # --- spaces / metadata (Gym-ish, without depending on gym) ---
    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return self.cfg.observation_shape

    @property
    def button_dim(self) -> int:
        return BUTTON_DIM

    @property
    def look_dim(self) -> int:
        return LOOK_DIM

    @property
    def button_names(self) -> tuple[str, ...]:
        return BUTTON_NAMES

    @property
    def batch_size(self) -> int:
        return self.cfg.batch

    # --- lifecycle ---
    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        return self._backend.reset(mask)

    def step(
        self,
        buttons: np.ndarray | None = None,
        look: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._backend.step(buttons, look)

    def get_obs(self) -> np.ndarray:
        """Re-render without advancing physics."""
        return self._backend.get_obs()

    def close(self) -> None:
        self._backend.close()

    # --- convenience ---
    def zero_actions(self) -> tuple[np.ndarray, np.ndarray]:
        B = self.cfg.batch
        return (
            np.zeros((B, BUTTON_DIM), dtype=np.int8),
            np.zeros((B, LOOK_DIM), dtype=np.float32),
        )

    @classmethod
    def from_config(cls, cfg: EnvConfig) -> "MineCUEnv":
        return cls(cfg=cfg)


__all__ = [
    "MineCUEnv",
    "EnvConfig",
    "BlockType",
    "BUTTON_DIM",
    "LOOK_DIM",
    "BUTTON_NAMES",
]
