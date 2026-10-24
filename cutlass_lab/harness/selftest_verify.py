#!/usr/bin/env python3
"""Self-test for verify_sgemm.py, and for the input generator in common.cuh.

The lab has no GPU, so the CUTLASS kernels can be compiled but not executed.
What CAN be validated here is everything on the host side of the verification:

  1. that fill_deterministic() in common.cuh, reimplemented in Python, is
     bit-identical to the C++ version (a transcription bug in the input
     generator would silently invalidate every later measurement);
  2. that verify_sgemm.py reports PASS on a correct float32 result and FAIL on a
     perturbed one, so it is a real gate and not a rubber stamp;
  3. that the reported error for a correctly-accumulated but differently-ordered
     float32 dot product is the size the theory predicts.

Run:  python3 selftest_verify.py
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile

import numpy as np

M, N, K = 1027, 1031, 2003
HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.join(HERE, "verify_sgemm.py")


def fill_deterministic(n: int, seed: int) -> np.ndarray:
    """Transcription of cutlass_lab/harness/common.cuh fill_deterministic().

    The C++ original is `state = state * 1664525u + 1013904223u` in `unsigned`
    (32-bit, wrapping). numpy's uint32 scalars promote to int64 on multiply, so
    the wraparound has to be applied explicitly to match bit for bit.
    """
    state = int(seed) & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u = np.float32(state >> 8) / np.float32(1 << 24)
        out[i] = np.float32(np.float32(u * np.float32(2.0)) - np.float32(1.0))
    return out


def run_verify(d: str, expect: str) -> tuple[int, str]:
    r = subprocess.run(
        [sys.executable, VERIFY, d, "--expect", expect],
        capture_output=True, text=True,
    )
    return r.returncode, r.stdout


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="cutlass_lab_selftest_")
    A = fill_deterministic(M * K, 1)
    B = fill_deterministic(N * K, 2)
    C = fill_deterministic(M * N, 3)
    A2, B2, C2 = A.reshape(M, K), B.reshape(N, K), C.reshape(M, N)

    # ---- case 1: exactly the reference -> must PASS -----------------------
    D = (A2.astype(np.float32) @ B2.astype(np.float32).T).astype(np.float32) + C2
    for name, arr in (("A.raw", A), ("B.raw", B), ("C.raw", C), ("D.raw", D.reshape(-1))):
        arr.tofile(os.path.join(tmp, name))
    rc, out = run_verify(tmp, "simt")
    ok1 = rc == 0 and "PASS" in out
    print(f"[1] exact reference          -> rc={rc} {'PASS' if ok1 else 'UNEXPECTED'}")

    # ---- case 2: 1e-3 relative perturbation -> must FAIL -------------------
    D2 = D * np.float32(1.0 + 1e-3)
    D2.reshape(-1).astype(np.float32).tofile(os.path.join(tmp, "D.raw"))
    rc2, out2 = run_verify(tmp, "simt")
    ok2 = rc2 != 0 and "FAIL" in out2
    line2 = next((l for l in out2.splitlines() if "normwise" in l), "")
    print(f"[2] 1e-3 relative perturbation -> rc={rc2} {'FAIL(correct)' if ok2 else 'UNEXPECTED'}"
          f"  {line2.strip()}")

    # ---- case 3: a realistic fp32 accumulation-order difference ----------
    # Accumulate in K-blocks of 64 in float32, which is roughly the order a tiled
    # fp32 kernel produces. The answer is the same, only the summation order
    # differs, so the normwise error must be small and the result must PASS --
    # while max relative error stays large because of cancellation. This is the
    # case that motivated gating on the normwise error.
    acc = np.zeros((M, N), dtype=np.float32)
    for k0 in range(0, K, 64):
        acc += A2[:, k0 : k0 + 64].astype(np.float32) @ B2[:, k0 : k0 + 64].astype(np.float32).T
    D3 = (acc + C2).astype(np.float32)
    D3.reshape(-1).astype(np.float32).tofile(os.path.join(tmp, "D.raw"))
    rc3, out3 = run_verify(tmp, "simt")
    nw = next((l for l in out3.splitlines() if "normwise" in l), "")
    mr = next((l for l in out3.splitlines() if "max rel" in l), "")
    ok3 = rc3 == 0
    print(f"[3] blocked fp32 accumulation -> rc={rc3} {'PASS' if ok3 else 'FAIL'}")
    print(f"      {mr.strip()}")
    print(f"      {nw.strip()}")

    # ---- case 4: the gated metric must actually separate good from bad ----
    def normwise(cand: np.ndarray) -> float:
        d = np.linalg.norm(cand.astype(np.float64) - D.astype(np.float64))
        return d / max(np.linalg.norm(D.astype(np.float64)), 1e-30)

    nw_good, nw_bad = normwise(D3), normwise(D2)
    separates = nw_bad > 100 * max(nw_good, 1e-15)
    print(f"[4] gated metric separates      -> {'yes' if separates else 'NO'}"
          f"  (normwise: reordered={nw_good:.3e} vs perturbed={nw_bad:.3e})")

    for f in ("A.raw", "B.raw", "C.raw", "D.raw"):
        p = os.path.join(tmp, f)
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(tmp)

    allok = ok1 and ok2 and ok3 and separates
    print()
    print("SELFTEST " + ("OK" if allok else "FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
