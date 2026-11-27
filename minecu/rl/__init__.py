"""Pure-TinyGrad RL building blocks and algorithms for MineCUEnv."""
from __future__ import annotations

from .common import ACTION_DIM, buttons_from_actions
from .models import CNNActorCritic, CNNPolicy

__all__ = [
    "ACTION_DIM",
    "buttons_from_actions",
    "CNNPolicy",
    "CNNActorCritic",
    "PPOConfig",
    "train_ppo",
    "GRPOConfig",
    "train_grpo",
    "AsyncConfig",
    "train_async",
]


def __getattr__(name: str):
    if name in ("PPOConfig", "train_ppo"):
        from .ppo import PPOConfig, train_ppo

        return PPOConfig if name == "PPOConfig" else train_ppo
    if name in ("GRPOConfig", "train_grpo"):
        from .grpo import GRPOConfig, train_grpo

        return GRPOConfig if name == "GRPOConfig" else train_grpo
    if name in ("AsyncConfig", "train_async"):
        from .async_rl import AsyncConfig, train_async

        return AsyncConfig if name == "AsyncConfig" else train_async
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
