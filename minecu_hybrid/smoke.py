"""Smoke unified env: TinyGrad path + CUDA accel (host and device-resident)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path


def _run(accel: str, batch: int = 8, **kw) -> None:
    from minecu_hybrid import MineCUEnv, viewer

    print(f"=== accel={accel!r} batch={batch} kw={kw} ===")
    env = MineCUEnv(accel=accel, batch=batch, world=16, width=32, height=32, **kw)
    print(f"using_cuda_accel={env.using_cuda_accel} obs_dtype={env.cfg.obs_dtype}")
    t0 = time.perf_counter()
    obs = env.reset()
    buttons, look = env.zero_actions()
    # host path uses numpy; device path may return cupy
    try:
        import cupy as cp

        is_dev = isinstance(obs, cp.ndarray)
    except Exception:
        is_dev = False
    if is_dev:
        import cupy as cp

        assert obs.shape == (batch, 32, 32, 3)
        if hasattr(buttons, "fill"):
            buttons.fill(0)
            buttons[:, 0] = 1
    else:
        assert obs.shape == (batch, 32, 32, 3), getattr(obs, "shape", None)
        buttons = buttons.copy()
        buttons[:, 0] = 1
    n = 8
    t1 = time.perf_counter()
    for _ in range(n):
        obs, rew, done = env.step(buttons, look)
    if is_dev:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
        obs_np = cp.asnumpy(obs)
        var = float(obs_np.astype("float32").var())
        mean = float(obs_np.mean())
    else:
        var = float(obs.astype("float32").var())
        mean = float(obs.mean())
    dt = time.perf_counter() - t1
    print(
        f"obs mean={mean:.1f} var={var:.1f} {n} steps {dt*1e3:.1f} ms "
        f"({batch * n / dt:.0f} env-steps/s) reset {(t1-t0)*1e3:.0f} ms"
    )
    if var < 1.0:
        raise SystemExit(f"FAIL accel={accel}: flat observation")
    if not is_dev:
        with tempfile.TemporaryDirectory() as td:
            viewer.save_mosaic(obs, Path(td) / "m.ppm", scale=2)
    env.close()
    print(f"ok accel={accel}")


def main() -> None:
    _run("off", batch=2)
    from minecu_hybrid import accel

    if accel.available():
        _run("on", batch=64)
        _run(
            "on",
            batch=256,
            device_buffers=True,
            sync_every_step=False,
            obs_dtype="uint8_prebasis_full",
        )
        _run("on", batch=64, obs_dtype="uint8")
    else:
        print("skip cuda:", accel.why_unavailable())


if __name__ == "__main__":
    main()
