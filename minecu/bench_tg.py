"""Benchmark MineCUEnv(accel="off") reset + steady-state step throughput.

Run:
    PYTHONPATH=. DEV=CPU .venv/bin/python -m minecu.bench_tg
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("MINECU_TG_DEV", os.environ.get("MINECU_TG_DEV", "CPU"))


def run_bench(
    batch: int = 64,
    world: int = 16,
    width: int = 16,
    height: int = 16,
    max_steps: int = 16,
    episode_length: int = 8,
    warmup: int = 3,
    steps: int = 24,
) -> dict[str, float]:
    from minecu import MineCUEnv

    t0 = time.perf_counter()
    env = MineCUEnv(
        accel="off",
        batch=batch,
        world=world,
        width=width,
        height=height,
        max_steps=max_steps,
        episode_length=episode_length,
        seed=0,
    )
    obs0 = env.reset()
    t_init_reset = time.perf_counter() - t0

    buttons, look = env.zero_actions()
    buttons = buttons.copy()
    buttons[:, 0] = 1  # walk forward

    t_w0 = time.perf_counter()
    for _ in range(warmup):
        obs, rew, done = env.step(buttons, look)
    t_warmup = time.perf_counter() - t_w0

    t_s0 = time.perf_counter()
    for _ in range(steps):
        obs, rew, done = env.step(buttons, look)
    t_steady = time.perf_counter() - t_s0

    ms_per_step = (t_steady / steps) * 1000.0
    batch_sps = steps / t_steady
    env_sps = (steps * batch) / t_steady

    print(
        f"MineCUEnv(accel='off', batch={batch}, world={world}, "
        f"width={width}, height={height}, max_steps={max_steps}, episode_length={episode_length})"
    )
    print(f"  init+reset : {t_init_reset * 1000.0:.2f} ms (obs={obs0.shape} {obs0.dtype})")
    print(f"  warmup ({warmup}): {t_warmup * 1000.0:.2f} ms ({(t_warmup / max(warmup, 1)) * 1000.0:.2f} ms/step)")
    print(
        f"  steady ({steps}): {t_steady * 1000.0:.2f} ms | "
        f"{ms_per_step:.2f} ms/step | {batch_sps:.2f} batch_steps/s | {env_sps:.1f} env_steps/s"
    )
    env.close()
    return {
        "init_reset_ms": t_init_reset * 1000.0,
        "warmup_ms": t_warmup * 1000.0,
        "steady_ms": t_steady * 1000.0,
        "ms_per_step": ms_per_step,
        "batch_sps": batch_sps,
        "env_sps": env_sps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark MineCUEnv(accel='off')")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--world", type=int, default=16)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--height", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=24)
    args = p.parse_args()
    run_bench(
        batch=args.batch,
        world=args.world,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        episode_length=args.episode_length,
        warmup=args.warmup,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
