"""Smoke both backends when available; always exercise the API surface."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path


def _run(backend: str, batch: int = 8) -> None:
    from minecu_hybrid import MineCUEnv, viewer

    print(f"=== backend={backend} batch={batch} ===")
    env = MineCUEnv(backend=backend, batch=batch, world=16, width=32, height=32)
    t0 = time.perf_counter()
    obs = env.reset()
    assert obs.shape == (batch, 32, 32, 3), obs.shape
    obs2 = env.get_obs()
    assert obs2.shape == obs.shape
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
        raise SystemExit(f"FAIL {backend}: flat observation")

    # masked reset of one env
    mask = (buttons[:, 0] * 0).astype(bool)
    mask[0] = True
    env.reset(mask)

    with tempfile.TemporaryDirectory() as td:
        p = viewer.save_mosaic(obs, Path(td) / f"{backend}_mosaic.ppm", scale=4)
        print("wrote", p, p.stat().st_size, "bytes")
    env.close()
    print(f"ok {backend}")


def main() -> None:
    # TinyGrad path runs on Metal/CPU too
    _run("tinygrad", batch=4)
    try:
        import cupy  # noqa: F401
        from minecu_hybrid.lib import lib_path

        if lib_path().is_file():
            _run("cuda", batch=64)
        else:
            print("skip cuda: no libminecu.so (build_lib.sh on NVIDIA host)")
    except Exception as e:
        print("skip cuda:", e)


if __name__ == "__main__":
    main()
