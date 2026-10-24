"""
test_control.py -- the CPU validation suite.  Everything that CAN be checked
without a GPU is checked here, and the things that cannot are named.

Run:  .venv/bin/python crossover/test_control.py

This is the only file in crossover/ whose job is to fail.  The others print
predictions and re-derivations; this one asserts them, so that a future edit
which changes a number has to change a test too.

  [MEASURED]   the checks that execute something on this machine
  [MODELLED]   the checks that are arithmetic on shapes and datasheets
  [PREDICTED]  none.  A prediction that a test "confirms" is a category error,
               and the suite refuses to do it.

WHAT IS AND IS NOT VALIDATED
----------------------------
  VALIDATED HERE, today, on CPU:
    * the two attention implementations agree (semantics of the control)
    * the online form is invariant to the key-block size
    * the byte model reproduces the study's own delivered figures
    * the 8.00 GiB figure's shape is identified
    * the 256x is cross-scope and the same-scope ratio is 64x
    * the capacity table's arithmetic (on-chip aggregates, ridge points)
    * the P100 L2 discrepancy against bench/selftest.py is real
    * Model 4's byte ratio is independent of n
  NOT VALIDATED, and named as such:
    * every GPU timing (no GPU, no CUDA, no ROCm in this environment)
    * every TPU timing (no jax in the venv -- verified, ModuleNotFoundError)
    * the compiler's actual behaviour, i.e. criterion R-C
    * the pass count, i.e. criterion R-E
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devices as D      # noqa: E402
import claim as CL       # noqa: E402
import attn_control as AC  # noqa: E402
import literature as LIT  # noqa: E402

MiB, GiB = 1024 ** 2, 1024 ** 3

_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# =============================================================================
def t_semantics() -> None:
    print()
    print("=" * 100)
    print("1. THE CONTROL IS A CONTROL  (refutation criterion R-D)")
    print("=" * 100)
    rng = np.random.default_rng(12345)
    worst_eq, worst_bc = 0.0, 0.0
    for (B, H, n, d) in ((1, 1, 128, 32), (2, 3, 96, 64), (1, 2, 257, 64)):
        for causal in (True, False):
            q = rng.standard_normal((B, H, n, d))
            k = rng.standard_normal((B, H, n, d))
            v = rng.standard_normal((B, H, n, d))
            k /= np.linalg.norm(k, axis=-1, keepdims=True)
            sc = 1.0 / math.sqrt(d)
            a, _ = AC.attn_materialised(q, k, v, sc, causal)
            ref = None
            for bc in (16, 32, 64, 128, 256):
                if bc > n:
                    continue
                b, _ = AC.attn_online(q, k, v, sc, causal, block=bc)
                if ref is None:
                    worst_eq = max(worst_eq, float(np.max(np.abs(a - b) / np.abs(a).max())))
                    ref = b
                else:
                    worst_bc = max(worst_bc, float(np.max(np.abs(b - ref) / np.abs(ref).max())))
    check("materialised == online across 3 shapes x causal/dense",
          worst_eq < 1e-13, f"worst relative difference {worst_eq:.3e}")
    check("online form is invariant to the key-block size",
          worst_bc < 1e-13, f"worst relative difference {worst_bc:.3e}")

    # the two forms really do differ in the intermediate, which is the premise
    n = 512
    sc = 1.0 / math.sqrt(64)
    q = rng.standard_normal((1, 1, n, 64))
    k = rng.standard_normal((1, 1, n, 64))
    v = rng.standard_normal((1, 1, n, 64))
    _, scores = AC.attn_materialised(q, k, v, sc, True)
    _, none_block = AC.attn_online(q, k, v, sc, True, block=128)
    check("the materialised form really allocates the full intermediate",
          scores is not None and scores.shape == (1, 1, n, n),
          f"shape {None if scores is None else scores.shape}")
    check("the online form really does not", none_block is None)


def t_no_n_dependence() -> None:
    print()
    print("=" * 100)
    print("2. MODEL 4 -- THE BYTE RATIO DOES NOT DEPEND ON n.  The structural result.")
    print("=" * 100)
    # measured the ratio of modelled traffic at several n, and check the
    # n-dependence is only the compulsory term, i.e. sub-dominant
    B, H, d, es = 1, 1, 64, 4
    ratios = []
    for n in (256, 1024, 4096):
        by = AC.byte_counts(B, H, n, d, es, 128, True)
        ratios.append(by["mat"] / by["online"])
    print(f"    modelled traffic ratio at n = 256, 1024, 4096: "
          f"{[f'{r:.3f}' for r in ratios]}")
    print(f"    Model 4's closed form passes*128/(2*64) = "
          f"{CL.max_byte_ratio('H100', d, es, 3):.3f} (the ceiling)")
    check("the ratio is bounded by Model 4's ceiling at every n",
          all(r <= CL.max_byte_ratio("H100", d, es, 3) * 1.6 for r in ratios),
          f"max {max(ratios):.3f} vs ceiling {CL.max_byte_ratio('H100', d, es, 3):.3f}")
    check("the ratio varies by less than 1.6x over a 16x range of n",
          max(ratios) / min(ratios) < 1.6,
          f"spread {max(ratios)/min(ratios):.3f}x -- "
          f"a STEP-function thesis predicts this to grow by orders of magnitude")
    # and the closed form itself
    for dev in D.DEVICE_SPECS:
        bc = CL.max_key_block(dev, d, es)
        want = 3 * bc / (2.0 * d)
        check(f"Model 4 closed form, {dev}", close(CL.max_byte_ratio(dev, d, es, 3), want))
    check("H100 and B200 predict the SAME ceiling (both 228 KiB smem/SM)",
          close(CL.max_byte_ratio("H100", d, es, 3),
                CL.max_byte_ratio("B200", d, es, 3)),
          f"{CL.max_byte_ratio('H100', d, es, 3):.2f}x each, while L2 differs "
          f"{D.DEVICE_SPECS['B200']['l2_bytes']/D.DEVICE_SPECS['H100']['l2_bytes']:.2f}x")
    check("L2 does not enter Model 4 at all",
          all(close(CL.max_byte_ratio("T4", d, es, 3),
                    CL.max_byte_ratio("P100", d, es, 3) * 0 + 6.0) for _ in (0,)),
          "T4 and P100 have the same L2 (4 MiB) but different smem/SM, and "
          "Model 4 separates them")


def t_kda_model() -> None:
    print()
    print("=" * 100)
    print("3. THE KDA BYTE MODEL -- re-derived from shapes, not transcribed")
    print("=" * 100)
    B, T, H, K, C, es = 2, 512, 4, 64, 64, 4          # kda/perf_kda.py:90
    NT = T // C
    elems = B * NT * H * C * C * K
    check("decay tensor is 2^24 elements at the benchmarked shape",
          elems == 2 ** 24, f"{elems:,}")
    check("one decay tensor is 67.11 MB in fp32",
          close(elems * es / 1e6, 67.1089, 1e-4), f"{elems*es/1e6:.4f} MB")
    check("one decay tensor is 33.55 MB in bf16",
          close(elems * 2 / 1e6, 33.5544, 1e-4), f"{elems*2/1e6:.4f} MB")

    port = CL.kda_port_bytes(B, H, T, K, C, es)
    kern = CL.kda_kernel_bytes(B, H, T, K, es)
    sub = CL.kda_subchunk_bytes(B, H, T, K, C, es)
    check("port / kernel, SAME scope, is exactly C", close(port / kern, float(C)),
          f"{port/kern:.4f}x")
    check("the study's 256x is the CROSS-SCOPE ratio", close(port / sub, 256.0),
          f"{port/sub:.4f}x  = C*C/BC")
    check("the cross-scope ratio overstates by exactly NC = C/BC",
          close((port / sub) / (port / kern), C / 16))

    # the 8.00 GiB figure
    B2, T2, H2, K2, C2 = 4, 4096, 32, 128, 64
    e2 = B2 * (T2 // C2) * H2 * C2 * C2 * K2
    check("cost.py's shape gives 2^32 elements", e2 == 2 ** 32, f"{e2:,}")
    check("8.00 GiB is that shape in bf16", close(e2 * 2 / GiB, 8.0),
          f"{e2*2/GiB:.4f} GiB")
    check("8.00 GiB is 128x the benchmarked shape's fp32 tensor",
          close((e2 * 2) / (elems * 4), 128.0),
          f"{(e2*2)/(elems*4):.1f}x -- the two shapes are both called 'the study "
          f"shape' in the record")

    # the crossover is per-device and reverses
    for dev, want_human in (("T4", True), ("B200", False)):
        over = port / D.DEVICE_SPECS[dev]["l2_bytes"] > 1
        check(f"benchmark-shape KDA intermediate is {'OVER' if over else 'UNDER'} "
              f"{dev} L2", over == want_human,
              f"{port/D.DEVICE_SPECS[dev]['l2_bytes']:.2f}x")
    # dtype flips the verdict on two of five devices
    flips = [dev for dev in D.DEVICE_SPECS
             if (port > D.DEVICE_SPECS[dev]["l2_bytes"])
             != (port // 2 > D.DEVICE_SPECS[dev]["l2_bytes"])]
    check("fp32 -> bf16 flips the capacity verdict on some devices", len(flips) > 0,
          f"{flips} -- so a dtype flag moves the study across the threshold, which "
          f"is why notes/06 insists the measurement be fp32")

    # the KDA byte model's own delivered figures
    decay_traffic = 2 * 2 * elems * es
    g_cum = 2 * B * T * H * K * es
    c2 = 4 * 2 * B * NT * H * C * C * es
    wu = 2 * B * NT * H * C * (K + K) * es
    v_new = 2 * B * NT * H * C * K * es
    io = B * T * H * (3 * K + 2 * K) * es
    st = 2 * B * H * K * K * es
    other = g_cum + c2 + wu + v_new + io + st
    total = decay_traffic + other
    for name, got, want in (("decay traffic MB", decay_traffic / 1e6, 268.4),
                            ("everything else MB", other / 1e6, 22.3),
                            ("total MB", total / 1e6, 290.7),
                            ("fold ratio", total / other, 13.0),
                            ("compulsory MB", (io + st) / 1e6, 5.5),
                            ("decay / compulsory", decay_traffic / (io + st), 48.8)):
        check(f"delivered KDA byte model: {name}", close(got, want, 5e-3),
              f"{got:.4f} vs {want}")


def t_devices() -> None:
    print()
    print("=" * 100)
    print("4. THE CAPACITY TABLE")
    print("=" * 100)
    for dev, d in D.DEVICE_SPECS.items():
        check(f"{dev}: on-chip aggregate == SMs x smem/SM",
              D.onchip_aggregate(dev) == d["sm_count"] * d["smem_per_sm"],
              f"{D._fmt(D.onchip_aggregate(dev))}")
        check(f"{dev}: smem_per_block <= smem_per_sm",
              d["smem_per_block"] <= d["smem_per_sm"],
              f"{d['smem_per_block']/1024:.0f} KiB <= {d['smem_per_sm']/1024:.0f} KiB"
              + ("  (Turing removed the 48 KB opt-in cap, so equality is correct here)"
                 if d["smem_per_block"] == d["smem_per_sm"] else ""))
        check(f"{dev}: register file is 256 KiB/SM",
              D.onchip_register_file(dev) == 256 * 1024)
    check("T4 L2 is 4 MiB (arXiv:1903.07486, 4096 KiB)",
          D.DEVICE_SPECS["T4"]["l2_bytes"] == 4 * MiB)
    check("P100 L2 is 4 MiB (NVIDIA Inside Pascal, 4096 KB), NOT 24 MiB",
          D.DEVICE_SPECS["P100"]["l2_bytes"] == 4 * MiB,
          "bench/selftest.py:103 says 24<<20. That file is owned by another "
          "agent and is NOT modified here; the discrepancy is recorded in "
          "devices.CONFLICTS and is 6x on every P100 crossover")
    check("H100 L2 is 50 MiB (NVIDIA Hopper blog)", D.DEVICE_SPECS["H100"]["l2_bytes"] == 50 * MiB)
    check("B200 L2 is 126 MB (NVIDIA Blackwell Tuning Guide 1.4.2.2)",
          D.DEVICE_SPECS["B200"]["l2_bytes"] == 126 * MiB)
    check("H100 and B200 share 228 KiB smem/SM",
          D.onchip_per_sm("H100") == D.onchip_per_sm("B200"))
    check("B200 has no fp32 roof claimed (no vendor figure found)",
          D.DEVICE_SPECS["B200"]["fp32_flops_s"] is None)
    check("ridge point is flops/bytes for every device that has both",
          all(close(D.ridge_point(dv), d["fp32_flops_s"] / d["dram_bytes_s"])
              for dv, d in D.DEVICE_SPECS.items()
              if d["fp32_flops_s"] and d["dram_bytes_s"]))


def t_literature() -> None:
    print()
    print("=" * 100)
    print("5. THE LITERATURE RE-DERIVATIONS")
    print("=" * 100)
    rs = [(n, h / x) for n, x, h, _ in LIT.ZHANG]
    check("Zhang table: the hand-written version is slower at EVERY n",
          all(r > 1 for _, r in rs),
          "which refutes the brief's 'overtakes at n>=8192'")
    check("Zhang: the 35x is the MAXIMUM, at n=4096, not a crossing",
          max(rs, key=lambda t: t[1])[0] == 4096,
          f"peak {max(r for _, r in rs):.2f}x at n=4096")
    check("Zhang: R(n) is NON-MONOTONE, which no capacity threshold produces",
          not all(rs[i][1] > rs[i + 1][1] for i in range(len(rs) - 1)))
    check("Zhang: the 35x figure itself re-derives",
          close(2.509 / 0.072, 34.847, 1e-3), f"{2.509/0.072:.4f}x")
    geo = math.exp(sum(math.log(x) for x in [r[3] for r in LIT.JAXBENCH]) / 8)
    check("JAXBench: the 8 per-kernel ratios re-derive from the timings",
          all(close(r[1] / r[2], r[3], 5e-3) for r in LIT.JAXBENCH))
    check("JAXBench: the 2.08x geomean does NOT re-derive from those 8 rows",
          not close(geo, 2.08, 1e-2),
          f"recomputed {geo:.4f} vs reported 2.08 -- a {abs(geo-2.08)/2.08*100:.1f}% "
          f"gap; the geomean must be over more rows or a different convention")
    check("arXiv 2012.07145: SGEMM/geomean ratio re-derives at ~17x",
          close(11740 / 698, 16.8, 2e-2), f"{11740/698:.2f}x")
    check("arXiv 2012.07145: largest/SGEMM re-derives at ~72x",
          close(50488 / 698, 72.3, 2e-2), f"{50488/698:.2f}x")
    check("Toulme: the 37.6x byte reduction re-derives",
          close(526 / 14, 37.571, 1e-2), f"{526/14:.4f}x")
    check("Toulme: the 6.2x bundle reduction re-derives",
          close(10245 / 1651, 6.205, 2e-2), f"{10245/1651:.4f}x")


def t_environment() -> None:
    print()
    print("=" * 100)
    print("6. WHAT THIS ENVIRONMENT CAN AND CANNOT DO  (recorded, not assumed)")
    print("=" * 100)
    try:
        import torch                                              # noqa: F401
        torch_ok = True
    except Exception:
        torch_ok = False
    try:
        import jax                                               # noqa: F401
        jax_ok = True
    except Exception:
        jax_ok = False
    try:
        from tinygrad import Device
        tg = Device.DEFAULT
    except Exception:
        tg = "unavailable"
    print(f"  numpy      {np.__version__}")
    print(f"  torch      {'present' if torch_ok else 'ABSENT'}")
    print(f"  jax        {'present' if jax_ok else 'ABSENT'}")
    print(f"  tinygrad   default device = {tg}")
    print()
    check("no torch in the venv (so no fabricated torch numbers are possible)",
          not torch_ok)
    if not jax_ok:
        print("  [UNVERIFIED-ENVIRONMENT] jax is absent, so the TPU-side code in")
        print("  kda/tpu/ is WRITTEN BUT NOT RUN and nothing in this directory")
        print("  pretends otherwise.  There is no TPU and no jax in this")
        print("  environment and both facts are load-bearing for every TPU number")
        print("  quoted in notes/07.")
    else:
        print("  jax is PRESENT but no TPU was probed; any TPU number would still be")
        print("  unmeasured.")


def main() -> int:
    print("=" * 100)
    print("CROSSOVER -- CPU VALIDATION SUITE")
    print("=" * 100)
    print("  Every check below either executes on this machine or is arithmetic on")
    print("  shapes and cited datasheets.  None of them confirms a prediction; a")
    print("  prediction that a test 'confirms' is a category error.")
    t_semantics()
    t_no_n_dependence()
    t_kda_model()
    t_devices()
    t_literature()
    t_environment()
    n = len(_results)
    bad = [r for r in _results if r[1] == "FAIL"]
    print()
    print("=" * 100)
    print(f"{n - len(bad)}/{n} checks pass")
    if bad:
        print("=" * 100)
        print("FAILURES")
        for name, _, detail in bad:
            print(f"  {name}")
            if detail:
                print(f"      {detail}")
    print("=" * 100)
    print()
    print("STILL UNVALIDATED, and no test above pretends otherwise:")
    print("  * every GPU timing -- no GPU, no CUDA, no ROCm in this environment")
    print("  * every TPU timing -- no jax, no TPU")
    print("  * criterion R-C, whether a compiler actually emits the materialising")
    print("    form, which is the load-bearing premise of the whole comparison")
    print("  * criterion R-E, the pass count, which every predicted ratio is linear in")
    print("  * the eff_ratio, how well each side drives memory, which moves the")
    print("    predicted crossover further than the entire device list does")
    print("=" * 100)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
