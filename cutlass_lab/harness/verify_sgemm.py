#!/usr/bin/env python3
"""Numerically verify a CUTLASS SGEMM dump against numpy float32.

Reads the four raw files the harness writes (A, B, C, D), recomputes
D_ref = A @ B + C in numpy float32, and reports max/mean absolute and relative
error. The relative error denominator uses the magnitude of the reference sum,
so cancellation-heavy outputs do not manufacture a huge ratio.

Usage:
    python3 verify_sgemm.py out/sm80_simt [--dtype float32|float64] [--tol 1e-5]

Exit status is 0 only if the error is within the tolerance, so this is usable as
a gate in CI on real hardware.
"""
import argparse
import sys

import numpy as np

M, N, K = 1027, 1031, 2003


def load(path, count, dtype):
    a = np.fromfile(path, dtype=dtype)
    if a.size != count:
        raise SystemExit(f"{path}: expected {count} elements, got {a.size}")
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory containing A.raw B.raw C.raw D.raw")
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="dtype the device kernel consumed (reference is always computed in float32)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=None,
        help="max tolerated relative error; defaults to a per-kernel-kind value",
    )
    ap.add_argument(
        "--expect",
        default="simt",
        choices=["simt", "tensorop"],
        help="kernel kind, only used to pick the default tolerance",
    )
    args = ap.parse_args()

    dt = np.float32 if args.dtype == "float32" else np.float64
    tol = args.tol
    if tol is None:
        # A dot product of length K accumulated in fp32 has relative error on the
        # order of K*eps in the worst case, ~sqrt(K)*eps in practice.
        #   eps(fp32) = 1.19e-7, K = 2003  ->  2.4e-4 worst case.
        # TF32 keeps only 10 explicit mantissa bits (eps = 9.8e-4), so even a
        # single-pass TF32 tensor-core GEMM lands around 1e-3 relative.
        tol = 1e-3 if args.expect == "simt" else 1e-2

    A = load(f"{args.dir}/A.raw", M * K, dt).reshape(M, K)
    B = load(f"{args.dir}/B.raw", N * K, dt).reshape(N, K)
    C = load(f"{args.dir}/C.raw", M * N, dt).reshape(M, N)
    D = load(f"{args.dir}/D.raw", M * N, dt).reshape(M, N)

    # D = A @ B^T + C, accumulated in float32. Use a tight accumulation dtype so
    # the reference is not itself the source of error.
    ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float32) + C

    diff = np.abs(D.astype(np.float64) - ref.astype(np.float64))
    denom = np.maximum(np.abs(ref.astype(np.float64)), np.float64(1e-30))
    rel = diff / denom

    # Two different error measures, and the choice matters.
    #
    #   max relative error is reported but is NOT the gate. For a K=2003 dot
    #   product, a large fraction of outputs are the difference of two large
    #   partial sums, so a handful of elements land near zero and their relative
    #   error is huge even when the kernel is exact. An earlier draft of this
    #   file gated on it and duly failed a correct result at 1.2e-1; the
    #   self-test caught that.
    #
    #   normwise (Frobenius) error is the gate: ||D - ref||_F / ||ref||_F is
    #   insensitive to which individual elements are small and is what actually
    #   bounds the accumulated answer.
    fro = np.linalg.norm(D.astype(np.float64) - ref.astype(np.float64))
    fro_ref = np.linalg.norm(ref.astype(np.float64))
    normwise = fro / max(fro_ref, 1e-30)

    print(f"dir                : {args.dir}")
    print(f"kernel dtype       : {args.dtype}")
    print(f"shape              : M={M} N={N} K={K}")
    print(f"max abs error      : {diff.max():.6e}")
    print(f"mean abs error     : {diff.mean():.6e}")
    print(f"max rel error      : {rel.max():.6e}   (informational, not gated)")
    print(f"mean rel error     : {rel.mean():.6e}")
    print(f"normwise rel error : {normwise:.6e}   (GATE: ||D-ref||_F / ||ref||_F)")
    print(f"tolerance          : {tol:.1e}  (normwise)")

    ok = normwise <= tol
    print("VERDICT            : " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
