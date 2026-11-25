"""Smoke the unified env: TinyGrad path always; CUDA accel when lib present."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path


def _run(accel: str, batch: int = 8, **kw) -> None:
    from minecu_hybrid import MineCUEnv, viewer

    print(f"=== accel={accel!r} batch={batch} ===")
    env = MineCUEnv(accel=accel, batch=batch, world=16, width=32, height=32, **kw)
    print(f"using_cuda_accel={env.using_cuda_accel}")
    print(f"observation_space={env.observation_space}")
    t0 = time.perf_counter()
    obs = env.reset()
    assert obs.shape == (batch, 32, 32, 3), obs.shape
    _ = env.get_obs()
    buttons, look = env.zero_actions()
    buttons[:, 0] = 1
    n = 4
    t1 = time.perf_counter()
    for _ in range(n):
        obs, rew, done = env.step(buttons, look)
    dt = time.perf_counter() - t1
    print(
        f"obs mean={obs.mean():.1f} var={obs.astype('float32').var():.1f} "
        f"{n} steps {dt*1e3:.1f} ms  dones={done.sum()}  "
        f"reset {(t1-t0)*1e3:.0f} ms"
    )
    if obs.astype("float32").var() < 1.0:
        raise SystemExit(f"FAIL accel={accel}: flat observation")
    mask = np_zeros_bool(batch)
    mask[0] = True
    env.reset(mask)
    with tempfile.TemporaryDirectory() as td:
        p = viewer.save_mosaic(obs, Path(td) / f"accel_{accel}_mosaic.ppm", scale=4)
        print("wrote", p, p.stat().st_size, "bytes")
    env.close()
    print(f"ok accel={accel}")


def np_zeros_bool(n: int):
    import numpy as np

    return np.zeros((n,), dtype=np.bool_)


def main() -> None:
    _run("off", batch=2)
    from minecu_hybrid import accel

    if accel.available():
        _run("on", batch=64)
    else:
        print("skip accel=on:", accel.why_unavailable())
    # auto should pick cuda if present
    _run("auto", batch=4 if not accel.available() else 32)


if __name__ == "__main__":
    main()
