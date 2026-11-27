"""Correctness: hybrid TG vs ref; optional fla vs ref."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kda"))

import ref  # noqa: E402
import test_ref as T  # noqa: E402
from kda.hybrid import available, kda_chunk, why_unavailable  # noqa: E402

TOL = 1e-5


def relerr(a, b):
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-300))


def main() -> None:
    print("=== kda.hybrid parity ===")
    print("accel available:", available(), why_unavailable())
    worst_tg = 0.0
    worst_fla = 0.0
    cases = [
        (2, 16, 2, 2, 4, 8, 8),
        (2, 32, 2, 2, 8, 8, 8),
        (1, 64, 2, 2, 8, 8, 16),
        (2, 16, 1, 3, 4, 4, 8),
    ]
    for i, (B, Tt, H, HV, K, V, C) in enumerate(cases):
        q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=i)
        o_ref, S_ref = ref.kda_recurrent(q, k, v, g, beta)
        o_tg, S_tg = kda_chunk(q, k, v, g, beta, chunk_size=C, accel_mode="off")
        e = max(relerr(o_tg, o_ref), relerr(S_tg, S_ref))
        worst_tg = max(worst_tg, e)
        print(f"  [TG]  case{i} C={C} relerr={e:.3e}")
        if e >= TOL:
            raise SystemExit(f"TG FAIL case{i} {e}")

        if available():
            o_a, S_a = kda_chunk(q, k, v, g, beta, chunk_size=C, accel_mode="on")
            e2 = max(relerr(o_a, o_ref), relerr(S_a, S_ref))
            worst_fla = max(worst_fla, e2)
            print(f"  [FLA] case{i} C={C} relerr={e2:.3e}")
            if e2 >= TOL:
                raise SystemExit(f"FLA FAIL case{i} {e2}")

    print(f"worst TG={worst_tg:.3e} FLA={worst_fla:.3e}")

    # microbench
    B, Tt, H, HV, K, V, C = 2, 512, 4, 4, 64, 64, 64
    q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=0)
    for mode in ("off", "on") if available() else ("off",):
        # warmup
        for _ in range(2):
            kda_chunk(q, k, v, g, beta, chunk_size=C, accel_mode=mode)
        if mode == "on":
            import torch

            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            kda_chunk(q, k, v, g, beta, chunk_size=C, accel_mode=mode)
        if mode == "on":
            import torch

            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n * 1e3
        print(f"  bench accel={mode!r} B={B} T={Tt} C={C}: {dt:.1f} ms/call")
    print("parity ok")


if __name__ == "__main__":
    main()
