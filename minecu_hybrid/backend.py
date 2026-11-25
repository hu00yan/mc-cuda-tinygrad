"""Backend protocol — upper env stays stable; swap CUDA ↔ TinyGrad underneath."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import EnvConfig


class Backend(Protocol):
    cfg: EnvConfig

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray: ...
    def step(
        self,
        buttons: np.ndarray | None = None,
        look: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def get_obs(self) -> np.ndarray: ...
    def close(self) -> None: ...


def make_backend(cfg: EnvConfig) -> Backend:
    name = (cfg.backend or "cuda").lower()
    if name in ("cuda", "hybrid", "ref"):
        from .backend_cuda import CudaBackend

        return CudaBackend(cfg)
    if name in ("tinygrad", "tg"):
        from .backend_tinygrad import TinygradBackend

        return TinygradBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}; use 'cuda' or 'tinygrad'")
