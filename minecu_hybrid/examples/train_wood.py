#!/usr/bin/env python3
"""Train a tiny CNN to break wood on MineCUEnv (REINFORCE).

Runs on CUDA accel when available (Kaggle/Colab T4). Uses cupy buffers when
``device_buffers=True`` so the policy and env stay on device.

Usage:
    PYTHONPATH=. python -m minecu_hybrid.examples.train_wood --steps 200000
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class SimpleCNN:
    def __init__(self, torch, nn, obs_shape, action_dim, device):
        h, w, c = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
        ).to(device)
        with torch.no_grad():
            n = self.net(torch.zeros(1, c, h, w, device=device)).numel()
        self.fc = nn.Sequential(
            nn.Linear(n, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        ).to(device)
        self.torch = torch

    def parameters(self):
        return list(self.net.parameters()) + list(self.fc.parameters())

    def logits(self, obs_bchw):
        x = self.net(obs_bchw).flatten(1)
        return self.fc(x)


def buttons_from_actions(torch, actions, batch, device):
    """Map discrete action index → multi-hot buttons [B,8].

    0: noop, 1: forward, 2: back, 3: left, 4: right, 5: jump,
    6: break, 7: place, 8: forward+sprint, 9: forward+break
    """
    B = batch
    buttons = torch.zeros(B, 8, dtype=torch.int8, device=device)
    # look zeros
    look = torch.zeros(B, 2, dtype=torch.float32, device=device)
    a = actions
    buttons[:, 0] = ((a == 1) | (a == 8) | (a == 9)).to(torch.int8)
    buttons[:, 1] = (a == 2).to(torch.int8)
    buttons[:, 2] = (a == 3).to(torch.int8)
    buttons[:, 3] = (a == 4).to(torch.int8)
    buttons[:, 4] = (a == 5).to(torch.int8)
    buttons[:, 5] = ((a == 6) | (a == 9)).to(torch.int8)
    buttons[:, 6] = (a == 7).to(torch.int8)
    buttons[:, 7] = (a == 8).to(torch.int8)
    return buttons, look


ACTION_DIM = 10


def train(
    total_steps: int = 500_000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    episode_length: int = 60,
    render_size: int = 32,
    log_interval: int = 20,
    seed: int = 0,
) -> None:
    torch, nn, F = _torch()
    from minecu_hybrid import BlockType, MineCUEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda_env = device == "cuda"
    env = MineCUEnv(
        batch_size=batch_size,
        world_size=32,
        render_width=render_size,
        render_height=render_size,
        episode_length=episode_length,
        target_block=int(BlockType.OAKLOG),
        reward_value=1.0,
        obs_dtype="uint8_prebasis_full",
        device_buffers=use_cuda_env,
        sync_every_step=False,
        accel="on" if use_cuda_env else "off",
        seed=seed,
        max_steps=32,
        view_distance=32.0,
    )
    print(
        f"device={device} cuda_accel={env.using_cuda_accel} "
        f"batch={batch_size} total_steps={total_steps:,}"
    )
    policy = SimpleCNN(torch, nn, env.observation_shape, ACTION_DIM, device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    obs = env.reset()
    step = 0
    reward_sum = 0.0
    t0 = time.time()
    log_every = log_interval * batch_size

    while step < total_steps:
        if env.cfg.device_buffers:
            import cupy as cp

            obs_np = cp.asnumpy(obs)
        else:
            obs_np = np.asarray(obs)
        obs_t = torch.from_numpy(obs_np).to(device)
        if obs_t.dtype == torch.uint8:
            obs_t = obs_t.float() / 255.0
        obs_t = obs_t.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            logits = policy.logits(obs_t)
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)

        buttons_t, look_t = buttons_from_actions(torch, actions, batch_size, device)
        if env.cfg.device_buffers:
            import cupy as cp

            buttons = cp.asarray(buttons_t.detach().cpu().numpy())
            look = cp.asarray(look_t.detach().cpu().numpy())
        else:
            buttons = buttons_t.detach().cpu().numpy()
            look = look_t.detach().cpu().numpy()

        obs, rewards, dones = env.step(buttons, look)
        if env.cfg.device_buffers:
            import cupy as cp

            rew_np = cp.asnumpy(rewards)
        else:
            rew_np = np.asarray(rewards)
        rew_t = torch.from_numpy(rew_np).to(device=device, dtype=torch.float32)
        reward_sum += float(rew_np.sum())

        logits = policy.logits(obs_t)
        logp = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = -(logp * rew_t).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        step += batch_size
        if step % log_every == 0 or step >= total_steps:
            elapsed = time.time() - t0
            sps = step / max(elapsed, 1e-9)
            print(
                f"step {step:>10,} | sps {sps:>10,.0f} | "
                f"reward/batch {reward_sum / max(step / batch_size, 1):.3f} | "
                f"loss {float(loss):.4f}"
            )

    elapsed = time.time() - t0
    print(
        f"done {total_steps:,} steps in {elapsed:.1f}s "
        f"({total_steps / elapsed:,.0f} sps) total_reward={reward_sum:,.0f}"
    )
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--episode", type=int, default=60)
    ap.add_argument("--render", type=int, default=32)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        total_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        episode_length=args.episode,
        render_size=args.render,
        log_interval=args.log_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
