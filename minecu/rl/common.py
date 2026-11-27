"""Shared RL utilities for pure-TinyGrad agents on MineCUEnv."""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

ACTION_DIM = 10


def buttons_from_actions(actions_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map discrete action index [B] -> (buttons [B,8] int8, look [B,2] float32).

    0: noop, 1: forward, 2: back, 3: left, 4: right, 5: jump,
    6: break, 7: place, 8: forward+sprint, 9: forward+break
    """
    a = np.asarray(actions_np, dtype=np.int64).reshape(-1)
    B = a.shape[0]
    buttons = np.zeros((B, 8), dtype=np.int8)
    look = np.zeros((B, 2), dtype=np.float32)

    buttons[:, 0] = ((a == 1) | (a == 8) | (a == 9)).astype(np.int8)
    buttons[:, 1] = (a == 2).astype(np.int8)
    buttons[:, 2] = (a == 3).astype(np.int8)
    buttons[:, 3] = (a == 4).astype(np.int8)
    buttons[:, 4] = (a == 5).astype(np.int8)
    buttons[:, 5] = ((a == 6) | (a == 9)).astype(np.int8)
    buttons[:, 6] = (a == 7).astype(np.int8)
    buttons[:, 7] = (a == 8).astype(np.int8)
    return buttons, look


def obs_to_tensor(obs_np: np.ndarray) -> Tensor:
    """Convert [B, H, W, 3] uint8/float32 observation to [B, 3, H, W] float32 in [0, 1]."""
    arr = np.asarray(obs_np)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32, copy=False)
    nchw = np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))
    return Tensor(nchw, dtype=dtypes.float32)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation over [T, B] rollouts."""
    rewards_f = np.asarray(rewards, dtype=np.float32)
    values_f = np.asarray(values, dtype=np.float32)
    dones_f = np.asarray(dones, dtype=np.float32)
    last_val_f = np.asarray(last_values, dtype=np.float32)

    T, B = rewards_f.shape
    advantages = np.zeros((T, B), dtype=np.float32)
    last_gae = np.zeros((B,), dtype=np.float32)

    for t in range(T - 1, -1, -1):
        non_terminal = 1.0 - dones_f[t]
        next_values = last_val_f if t == T - 1 else values_f[t + 1]
        delta = rewards_f[t] + gamma * next_values * non_terminal - values_f[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values_f
    return advantages, returns


def categorical_sample_and_logp(
    logits_np: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample discrete actions from [B, A] logits and return (actions, log_probs)."""
    logits = np.asarray(logits_np, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    probs = np.exp(log_probs)

    cdf = np.cumsum(probs, axis=-1)
    cdf[:, -1] = 1.0
    u = rng.random((logits.shape[0], 1), dtype=np.float32)
    actions = np.argmax(cdf >= u, axis=-1).astype(np.int64)
    action_logp = log_probs[np.arange(logits.shape[0]), actions].astype(np.float32)
    return actions, action_logp
