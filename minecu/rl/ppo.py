"""Pure-TinyGrad clipped PPO trainer on MineCUEnv."""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
from tinygrad import Context, Tensor, dtypes
from tinygrad.nn.optim import Adam

from minecu import BlockType, MineCUEnv

from .common import (
    ACTION_DIM,
    buttons_from_actions,
    categorical_sample_and_logp,
    compute_gae,
    obs_to_tensor,
)
from .models import CNNActorCritic


@dataclass
class PPOConfig:
    batch: int = 16
    rollout_steps: int = 8
    total_steps: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 2
    minibatches: int = 2
    hidden_dim: int = 128
    world: int = 16
    render: int = 16
    max_steps: int = 16
    episode_length: int = 60
    accel: str = "off"
    seed: int = 0
    log_interval: int = 1


def _one_hot(actions: np.ndarray, num_classes: int) -> Tensor:
    oh = np.eye(num_classes, dtype=np.float32)[actions]
    return Tensor(oh, dtype=dtypes.float32)


def train_ppo(cfg: PPOConfig) -> dict[str, float]:
    if "DEV" in os.environ and "MINECU_TG_DEV" not in os.environ:
        os.environ["MINECU_TG_DEV"] = os.environ["DEV"]
    Tensor.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = MineCUEnv(
        batch=cfg.batch,
        world=cfg.world,
        width=cfg.render,
        height=cfg.render,
        max_steps=cfg.max_steps,
        episode_length=cfg.episode_length,
        target_block=int(BlockType.OAKLOG),
        reward_value=1.0,
        accel=cfg.accel,
        seed=cfg.seed,
    )
    model = CNNActorCritic(env.observation_shape, ACTION_DIM, hidden_dim=cfg.hidden_dim)
    params = model.parameters()
    opt = Adam(params, lr=cfg.lr)

    init_param_snap = params[0].numpy().copy()

    T = cfg.rollout_steps
    B = cfg.batch
    obs_shape = env.observation_shape
    obs = np.asarray(env.reset())

    obs_buf = np.zeros((T, B, *obs_shape), dtype=obs.dtype)
    act_buf = np.zeros((T, B), dtype=np.int64)
    logp_buf = np.zeros((T, B), dtype=np.float32)
    rew_buf = np.zeros((T, B), dtype=np.float32)
    done_buf = np.zeros((T, B), dtype=np.bool_)
    val_buf = np.zeros((T, B), dtype=np.float32)

    global_steps = 0
    update_idx = 0
    t_start = time.perf_counter()
    last_metrics: dict[str, float] = {}

    while global_steps < cfg.total_steps:
        Tensor.training = False
        for t in range(T):
            obs_buf[t] = obs
            logits_t, values_t = model(obs_to_tensor(obs))
            logits_np = logits_t.numpy()
            values_np = values_t.numpy()

            actions_np, logp_np = categorical_sample_and_logp(logits_np, rng)
            buttons_np, look_np = buttons_from_actions(actions_np)
            next_obs, rewards_np, dones_np = env.step(buttons_np, look_np)

            act_buf[t] = actions_np
            logp_buf[t] = logp_np
            rew_buf[t] = np.asarray(rewards_np, dtype=np.float32)
            done_buf[t] = np.asarray(dones_np, dtype=np.bool_)
            val_buf[t] = values_np

            obs = np.asarray(next_obs)
            global_steps += B

        _, last_values_t = model(obs_to_tensor(obs))
        last_values_np = last_values_t.numpy()

        adv_np, ret_np = compute_gae(
            rew_buf,
            val_buf,
            done_buf,
            last_values_np,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        N = T * B
        b_obs = obs_buf.reshape(N, *obs_shape)
        b_act = act_buf.reshape(N)
        b_logp = logp_buf.reshape(N)
        b_adv = adv_np.reshape(N)
        b_ret = ret_np.reshape(N)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        mb_size = max(1, N // max(1, cfg.minibatches))
        losses, pi_losses, vf_losses, ents = [], [], [], []
        with Context(TRAINING=1):
            Tensor.training = True
            for _ in range(cfg.epochs):
                perm = rng.permutation(N)
                for start in range(0, N, mb_size):
                    idx = perm[start : start + mb_size]
                    obs_mb = obs_to_tensor(b_obs[idx])
                    act_oh_mb = _one_hot(b_act[idx], ACTION_DIM)
                    old_logp_mb = Tensor(b_logp[idx], dtype=dtypes.float32)
                    adv_mb = Tensor(b_adv[idx], dtype=dtypes.float32)
                    ret_mb = Tensor(b_ret[idx], dtype=dtypes.float32)

                    opt.zero_grad()
                    logits, values = model(obs_mb)
                    log_probs = logits.log_softmax(axis=-1)
                    probs = log_probs.exp()
                    new_logp = (log_probs * act_oh_mb).sum(axis=-1)

                    ratio = (new_logp - old_logp_mb).exp()
                    surr1 = ratio * adv_mb
                    surr2 = ratio.clip(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
                    policy_loss = -surr1.minimum(surr2).mean()
                    value_loss = ((values - ret_mb) ** 2).mean()
                    entropy = -(probs * log_probs).sum(axis=-1).mean()
                    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                    loss.backward()
                    opt.step()

                    losses.append(float(loss.numpy()))
                    pi_losses.append(float(policy_loss.numpy()))
                    vf_losses.append(float(value_loss.numpy()))
                    ents.append(float(entropy.numpy()))

        update_idx += 1
        elapsed = max(time.perf_counter() - t_start, 1e-9)
        sps = global_steps / elapsed
        last_metrics = {
            "steps": float(global_steps),
            "sps": float(sps),
            "loss": float(np.mean(losses)),
            "pi_loss": float(np.mean(pi_losses)),
            "vf_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(ents)),
            "mean_reward": float(rew_buf.mean()),
        }
        if update_idx % cfg.log_interval == 0 or global_steps >= cfg.total_steps:
            print(
                f"update={update_idx} steps={global_steps}/{cfg.total_steps} "
                f"sps={sps:.1f} loss={last_metrics['loss']:.4f} "
                f"pi={last_metrics['pi_loss']:.4f} vf={last_metrics['vf_loss']:.4f} "
                f"ent={last_metrics['entropy']:.4f} rew={last_metrics['mean_reward']:.4f}"
            )

    weight_delta = float(np.max(np.abs(params[0].numpy() - init_param_snap)))
    last_metrics["weight_delta"] = weight_delta
    print(f"done steps={global_steps} weight_delta={weight_delta:.6f}")
    return last_metrics


def _parse_args() -> PPOConfig:
    p = argparse.ArgumentParser(description="Pure-TinyGrad PPO on MineCUEnv")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=8)
    p.add_argument("--total-steps", "--steps", dest="total_steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatches", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--world", type=int, default=16)
    p.add_argument("--render", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=60)
    p.add_argument("--accel", type=str, default="off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=1)
    args = p.parse_args()
    return PPOConfig(**vars(args))


if __name__ == "__main__":
    train_ppo(_parse_args())
