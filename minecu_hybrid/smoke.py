"""GPU smoke: build check + reset/step + crude obs sanity."""
from __future__ import annotations

import time


def main() -> None:
    import cupy as cp

    from minecu_hybrid import MineCUEnv
    from minecu_hybrid.lib import lib_path, load

    print("gpu:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print("so:", lib_path(), "exists", lib_path().is_file())
    load()

    env = MineCUEnv(batch=64, world=16, width=32, height=32)
    t0 = time.perf_counter()
    obs = env.reset()
    print(f"reset obs shape={obs.shape} dtype={obs.dtype} mean={obs.mean():.2f}")

    # walk forward a bit
    buttons = cp.zeros((env.cfg.batch, 8), dtype=cp.int8)
    buttons[:, 0] = 1  # forward
    look = cp.zeros((env.cfg.batch, 2), dtype=cp.float32)

    n = 32
    t1 = time.perf_counter()
    for _ in range(n):
        obs, rew, done = env.step(buttons, look)
    cp.cuda.runtime.deviceSynchronize()
    dt = time.perf_counter() - t1
    print(
        f"{n} steps in {dt*1e3:.1f} ms "
        f"({env.cfg.batch * n / dt:.0f} env-steps/s) "
        f"obs_mean={obs.mean():.2f} rew_sum={rew.sum():.1f} done={done.sum()}"
    )
    # not an all-sky frame (sky LUT is bright blue-ish; empty would still have colour,
    # so just require some spatial variance)
    var = float(obs.astype("float32").var())
    print(f"obs variance={var:.1f}")
    if var < 1.0:
        raise SystemExit("FAIL: observation looks flat")
    print(f"ok (reset { (t1-t0)*1e3:.1f} ms)")


if __name__ == "__main__":
    main()
