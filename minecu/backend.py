"""Deprecated dual-backend factory — prefer MineCUEnv(accel=...).

Kept so old imports keep working; both paths now share the same MineCUEnv.
"""
from __future__ import annotations

from .config import EnvConfig
from .env import MineCUEnv


def make_backend(cfg: EnvConfig):
    """Return a MineCUEnv configured like the old cuda/tinygrad split."""
    if cfg.backend in ("tinygrad", "tg"):
        return MineCUEnv(cfg=cfg, accel="off")
    if cfg.backend in ("cuda", "hybrid", "ref"):
        return MineCUEnv(cfg=cfg, accel="on")
    return MineCUEnv(cfg=cfg, accel="auto")
