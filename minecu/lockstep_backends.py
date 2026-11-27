"""Lockstep MineCUEnv (accel) vs independent CudaBackend — same ``libminecu.so``.

Infatoshi's pybind ``minecu._C`` SIGSEGVs on some Kaggle images when loaded;
``kernels.cu`` is already byte-identical to Infatoshi (see ``pin_kernels``).
This gate proves the high-level env matches a second, MineEnv-shaped ctypes
driver on those kernels — obs/reward/done exact for N steps.

  PYTHONPATH=. python -m minecu.lockstep_backends
"""
from __future__ import annotations

import hashlib

import numpy as np

from minecu import MineCUEnv, accel
from minecu.backend_cuda import CudaBackend
from minecu.config import EnvConfig


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> None:
    if not accel.available():
        raise SystemExit(f"skip/fail: {accel.why_unavailable()}")

    cfg = EnvConfig(
        batch=16,
        world=16,
        width=32,
        height=32,
        max_steps=32,
        view_distance=32.0,
        episode_length=12,
        seed=42,
        obs_dtype="uint8_prebasis_full",
        device_buffers=False,
        sync_every_step=True,
        accel="on",
    )
    # MineCUEnv path
    env = MineCUEnv(cfg=cfg)
    # Independent ctypes driver (MineEnv-shaped)
    ref = CudaBackend(cfg)

    steps = 48
    rng = np.random.default_rng(0)
    o1 = env.reset()
    o2 = ref.reset()
    if not np.array_equal(o1, o2):
        raise SystemExit(
            f"FAIL reset maxabs={np.max(np.abs(o1.astype(float)-o2.astype(float)))} "
            f"{_sha(o1)} vs {_sha(o2)}"
        )
    print(f"reset OK sha={_sha(o1)}")

    for t in range(steps):
        buttons = rng.integers(0, 2, size=(cfg.batch, 8), dtype=np.int8)
        look = (rng.standard_normal((cfg.batch, 2)).astype(np.float32) * 0.05)
        a1, r1, d1 = env.step(buttons, look)
        a2, r2, d2 = ref.step(buttons, look)
        if not np.array_equal(a1, a2):
            raise SystemExit(f"FAIL step {t} obs")
        if not np.array_equal(r1, r2):
            raise SystemExit(f"FAIL step {t} reward {r1} vs {r2}")
        if not np.array_equal(d1.astype(bool), d2.astype(bool)):
            raise SystemExit(f"FAIL step {t} done")
        if (t + 1) % 8 == 0 or t == steps - 1:
            print(f"  step {t+1}/{steps} ok sha={_sha(a1)} rew={float(r1.sum()):.1f}")

    print("LOCKSTEP PASS MineCUEnv == CudaBackend (same kernels.cu as Infatoshi)")
    env.close()
    ref.close()


if __name__ == "__main__":
    main()
