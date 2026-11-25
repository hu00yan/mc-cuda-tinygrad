"""Lightweight spaces — Gymnasium-compatible shapes without a hard dependency."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    low: float | int | np.ndarray
    high: float | int | np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype

    def sample(self, batch: int | None = None) -> np.ndarray:
        shape = (batch, *self.shape) if batch is not None else self.shape
        if np.issubdtype(self.dtype, np.integer):
            hi = int(np.asarray(self.high).flat[0])
            lo = int(np.asarray(self.low).flat[0])
            return np.random.randint(lo, hi + 1, size=shape, dtype=self.dtype)
        return np.random.uniform(self.low, self.high, size=shape).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        a = np.asarray(x)
        return a.shape[-len(self.shape) :] == self.shape


@dataclass(frozen=True)
class Dict:
    spaces: dict

    def __getitem__(self, k):
        return self.spaces[k]

    def sample(self, batch: int | None = None) -> dict:
        return {k: v.sample(batch) for k, v in self.spaces.items()}


def try_gymnasium():
    try:
        import gymnasium as gym
        from gymnasium import spaces as gs

        return gym, gs
    except ImportError:
        return None, None
