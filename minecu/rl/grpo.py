"""Pure-TinyGrad Group Relative Policy Optimization (GRPO) on MineCUEnv.

Critic-free policy optimization: runs ``B = batch_groups * group_size``
parallel environments, computes group-normalized trajectory advantages
``A = (R - mean(R_group)) / (std(R_group) + 1e-8)``, and optimizes a clipped
surrogate objective with an optional KL penalty using TinyGrad Adam.

Usage:
    PYTHONPATH=. DEV=CPU python -m minecu.rl.grpo --total-steps 256
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

from minecu import MineCUEnv
from minecu.rl.common import ACTION_DIM, buttons_from_actions, obs_to_tensor
from minecu.rl.models import CNNPolicy


@dataclass
class GRPOConfig:
    """Configuration for pure-TinyGrad GRPO training."""

    batch_groups: int = 4
    group_size: int = 4
    rollout_steps: int = 16
    total_steps: int = 4096
    lr: float = 3e-4
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    epochs: int = 2
    gamma: float = 1.0
    hidden_dim: int = 128
    world: int = 16
    render: int = 16
    max_steps: int = 16
    episode_length: int = 16
    accel: str = "off"
    seed: int = 42
    log_interval: int = 1


def _sample_actions_and_logp(
    logits_np: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample discrete actions from logits [B, A] and return (actions, log_probs)."""
    shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random((logits_np.shape[0], 1), dtype=np.float32)
    actions = (cdf < u).sum(axis=-1).astype(np.int32)
    actions = np.clip(actions, 0, logits_np.shape[1] - 1)
    chosen_logp = log_probs[np.arange(logits_np.shape[0]), actions].astype(np.float32)
    return actions, chosen_logp


def _compute_group_advantages(
    rewards: np.ndarray, batch_groups: int, group_size: int, gamma: float
) -> np.ndarray:
    """Compute critic-free group-normalized advantages of shape [T, B]."""
    T, B = rewards.shape
    if gamma == 1.0:
        traj_returns = rewards.sum(axis=0)
    else:
        discounts = (gamma ** np.arange(T, dtype=np.float32))[:, None]
        traj_returns = (rewards * discounts).sum(axis=0)

    grouped = traj_returns.reshape(batch_groups, group_size)
    mean = grouped.mean(axis=1, keepdims=True)
    std = grouped.std(axis=1, keepdims=True)
    adv_grouped = (grouped - mean) / (std + 1e-8)
    adv_b = adv_grouped.reshape(1, B).astype(np.float32)
    return np.broadcast_to(adv_b, (T, B)).copy()


def train_grpo(cfg: GRPOConfig) -> dict[str, float]:
    """Run GRPO training loop on MineCUEnv and return final metrics."""
    B = cfg.batch_groups * cfg.group_size
    if B <= 0 or cfg.group_size <= 0:
        raise ValueError("batch_groups and group_size must be positive")

    env = MineCUEnv(
        batch=B,
        world=cfg.world,
        width=cfg.render,
        height=cfg.render,
        max_steps=cfg.max_steps,
        episode_length=cfg.episode_length,
        accel=cfg.accel,
        seed=cfg.seed,
    )
    policy = CNNPolicy(env.observation_shape, action_dim=ACTION_DIM, hidden_dim=cfg.hidden_dim)
    opt = Adam(get_parameters(policy), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)

    obs = env.reset()
    global_steps = 0
    update_idx = 0
    last_loss = 0.0
    last_kl = 0.0
    last_mean_ret = 0.0
    t_start = time.perf_counter()

    try:
        while global_steps < cfg.total_steps:
            T = cfg.rollout_steps
            obs_buf = np.empty((T, B, *env.observation_shape), dtype=obs.dtype)
            act_buf = np.empty((T, B), dtype=np.int32)
            old_logp_buf = np.empty((T, B), dtype=np.float32)
            rew_buf = np.empty((T, B), dtype=np.float32)

            with Context(TRAINING=0):
                for t in range(T):
                    obs_buf[t] = obs
                    logits_np = policy(obs_to_tensor(obs)).numpy()
                    actions, old_logp = _sample_actions_and_logp(logits_np, rng)
                    buttons, look = buttons_from_actions(actions)
                    obs, rew, _done = env.step(buttons, look)
                    act_buf[t] = actions
                    old_logp_buf[t] = old_logp
                    rew_buf[t] = np.asarray(rew, dtype=np.float32)

            adv_buf = _compute_group_advantages(
                rew_buf, cfg.batch_groups, cfg.group_size, cfg.gamma
            )
            last_mean_ret = float(rew_buf.sum(axis=0).mean())

            obs_flat = obs_buf.reshape(T * B, *env.observation_shape)
            act_flat = act_buf.reshape(T * B, 1)
            old_logp_flat = old_logp_buf.reshape(T * B)
            adv_flat = adv_buf.reshape(T * B)

            with Context(TRAINING=1):
                obs_t = obs_to_tensor(obs_flat)
                act_t = Tensor(act_flat, dtype=dtypes.int32)
                old_logp_t = Tensor(old_logp_flat, dtype=dtypes.float32)
                adv_t = Tensor(adv_flat, dtype=dtypes.float32)

                for _ in range(cfg.epochs):
                    logits = policy(obs_t)
                    log_probs = logits.log_softmax(axis=-1)
                    new_logp = log_probs.gather(1, act_t).squeeze(1)
                    log_ratio = new_logp - old_logp_t
                    ratio = log_ratio.exp()

                    surr1 = ratio * adv_t
                    surr2 = ratio.clip(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
                    pg_loss = -surr1.minimum(surr2).mean()

                    if cfg.kl_coef > 0.0:
                        kl_t = ((-log_ratio).exp() + log_ratio - 1.0).mean()
                        loss = pg_loss + cfg.kl_coef * kl_t
                    else:
                        kl_t = Tensor(0.0, dtype=dtypes.float32)
                        loss = pg_loss

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.numpy())
                    last_kl = float(kl_t.numpy())

            global_steps += T * B
            update_idx += 1
            if update_idx % cfg.log_interval == 0 or global_steps >= cfg.total_steps:
                elapsed = max(time.perf_counter() - t_start, 1e-6)
                sps = global_steps / elapsed
                print(
                    f"[grpo] update={update_idx} steps={global_steps}/{cfg.total_steps} "
                    f"loss={last_loss:.4f} kl={last_kl:.5f} "
                    f"mean_ret={last_mean_ret:.4f} sps={sps:.1f}"
                )
    finally:
        env.close()

    total_elapsed = max(time.perf_counter() - t_start, 1e-6)
    return {
        "steps": float(global_steps),
        "updates": float(update_idx),
        "loss": last_loss,
        "kl": last_kl,
        "mean_return": last_mean_ret,
        "sps": global_steps / total_elapsed,
    }


def _parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser(description="Pure-TinyGrad GRPO on MineCUEnv")
    p.add_argument("--batch-groups", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=16)
    p.add_argument("--total-steps", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--world", type=int, default=16)
    p.add_argument("--render", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=16)
    p.add_argument("--accel", type=str, default="off")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=1)
    args = p.parse_args()
    return GRPOConfig(
        batch_groups=args.batch_groups,
        group_size=args.group_size,
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
        lr=args.lr,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        epochs=args.epochs,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        world=args.world,
        render=args.render,
        max_steps=args.max_steps,
        episode_length=args.episode_length,
        accel=args.accel,
        seed=args.seed,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    train_grpo(_parse_args())
