"""Lockstep correctness vs Infatoshi MineEnv — two-process dump compare.

Loading ``minecu._C`` and our ``libminecu.so`` in one process can SIGSEGV
(duplicate CUDA graph globals / extension clash). So:

  1. ``--dump-upstream PATH``  — only Infatoshi MineEnv, write trajectory npz
  2. ``--check-hybrid PATH``   — only MineCUEnv, compare to that npz

Same seed + same RNG-driven actions ⇒ obs/reward/done must match exactly.

  python -m minecu_hybrid.lockstep_mineenv --dump-upstream /tmp/up.npz
  python -m minecu_hybrid.lockstep_mineenv --check-hybrid /tmp/up.npz
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _to_np(x) -> np.ndarray:
    if type(x).__module__.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _actions(steps: int, batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    buttons = rng.integers(0, 2, size=(steps, batch, 8), dtype=np.int8)
    look = (rng.standard_normal((steps, batch, 2)).astype(np.float32) * 0.05)
    return buttons, look


def dump_upstream(
    path: Path,
    *,
    steps: int,
    batch: int,
    world: int,
    wh: int,
    obs_dtype: str,
    episode_length: int,
    seed: int,
) -> None:
    from minecu import BlockType, MineEnv

    env = MineEnv(
        batch_size=batch,
        world_size=world,
        render_width=wh,
        render_height=wh,
        max_steps=wh,
        view_distance=float(wh),
        fov_degrees=70.0,
        ground_height=world // 4,
        target_block=int(BlockType.OAKLOG),
        reward_value=1.0,
        episode_length=episode_length if episode_length > 0 else None,
        seed=seed,
        obs_dtype=obs_dtype,
        device="cuda",
    )
    import torch

    buttons, look = _actions(steps, batch)
    obs0 = _to_np(env.reset())
    obs_l, rew_l, done_l = [], [], []
    for t in range(steps):
        bu = torch.tensor(buttons[t], device="cuda", dtype=torch.int8)
        lu = torch.tensor(look[t], device="cuda", dtype=torch.float32)
        o, r, d = env.step(bu, lu)
        obs_l.append(_to_np(o))
        rew_l.append(_to_np(r))
        done_l.append(_to_np(d).astype(np.bool_))
    np.savez_compressed(
        path,
        obs0=obs0,
        obs=np.stack(obs_l),
        rew=np.stack(rew_l),
        done=np.stack(done_l),
        buttons=buttons,
        look=look,
        meta=np.array(
            [steps, batch, world, wh, episode_length, seed], dtype=np.int64
        ),
        obs_dtype=np.array(obs_dtype),
    )
    print(f"upstream dump {path} reset_sha={_sha(obs0)} steps={steps}")


def check_hybrid(path: Path) -> None:
    from minecu_hybrid import BlockType, MineCUEnv, accel

    if not accel.available():
        raise SystemExit(f"FAIL: hybrid CUDA unavailable: {accel.why_unavailable()}")

    z = np.load(path, allow_pickle=False)
    steps, batch, world, wh, episode_length, seed = [int(x) for x in z["meta"]]
    obs_dtype = str(z["obs_dtype"])
    buttons, look = z["buttons"], z["look"]

    env = MineCUEnv(
        batch_size=batch,
        world_size=world,
        render_width=wh,
        render_height=wh,
        max_steps=wh,
        view_distance=float(wh),
        fov_degrees=70.0,
        ground_height=world // 4,
        target_block=int(BlockType.OAKLOG),
        reward_value=1.0,
        episode_length=episode_length,
        seed=seed,
        obs_dtype=obs_dtype,
        accel="on",
        device_buffers=False,
        sync_every_step=True,
    )
    obs0 = _to_np(env.reset())
    if not np.array_equal(obs0, z["obs0"]):
        raise SystemExit(
            f"FAIL reset maxabs="
            f"{np.max(np.abs(obs0.astype(np.float64)-z['obs0'].astype(np.float64)))} "
            f"sha_h={_sha(obs0)} sha_u={_sha(z['obs0'])}"
        )
    print(f"reset OK sha={_sha(obs0)}")

    for t in range(steps):
        o, r, d = env.step(buttons[t], look[t])
        o, r, d = _to_np(o), _to_np(r), _to_np(d).astype(np.bool_)
        if not np.array_equal(o, z["obs"][t]):
            raise SystemExit(
                f"FAIL step {t} obs maxabs="
                f"{np.max(np.abs(o.astype(np.float64)-z['obs'][t].astype(np.float64)))}"
            )
        if not np.array_equal(r, z["rew"][t]):
            raise SystemExit(f"FAIL step {t} reward")
        if not np.array_equal(d, z["done"][t]):
            raise SystemExit(f"FAIL step {t} done {d} vs {z['done'][t]}")
        if (t + 1) % 8 == 0 or t == steps - 1:
            print(f"  step {t+1}/{steps} ok sha={_sha(o)}")

    print(f"LOCKSTEP PASS steps={steps} batch={batch} obs_dtype={obs_dtype}")
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-upstream", type=Path, default=None)
    ap.add_argument("--check-hybrid", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--wh", type=int, default=32)
    ap.add_argument("--obs-dtype", default="uint8_prebasis_full")
    ap.add_argument("--episode", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.dump_upstream:
        dump_upstream(
            args.dump_upstream,
            steps=args.steps,
            batch=args.batch,
            world=args.world,
            wh=args.wh,
            obs_dtype=args.obs_dtype,
            episode_length=args.episode,
            seed=args.seed,
        )
    elif args.check_hybrid:
        check_hybrid(args.check_hybrid)
    else:
        raise SystemExit("need --dump-upstream PATH or --check-hybrid PATH")


if __name__ == "__main__":
    main()
