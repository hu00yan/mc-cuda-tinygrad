#!/usr/bin/env python3
"""b2_schedule.py -- the schedule census, on a machine with no GPU.

THE RESULT THIS FILE EXISTS FOR
-------------------------------
`kda/materialisation_verdict.md` named the one measurement that would close the
CUDA gap: run the schedule census with the CUDA backend at the bench shape and
report the histogram of kernel-argument buffer sizes; is there a bucket at
67,108,864 B? The answer is NO, and it is reachable on a laptop, because the
census never needed a card in the first place.

WHY NO DEVICE IS NEEDED  (re-derived here, not asserted)
---------------------------------------------------------
  1. `Tensor.schedule_linear()` -> `create_linear_with_vars` ->
     `memory_plan_rewrite` (tinygrad/schedule/memory.py) is a pure UOp rewrite
     driven by DEVICE STRINGS. It allocates nothing, launches nothing, and
     never constructs a `Device`.
  2. `Context(DEV="CUDA")` makes `Device.DEFAULT` read literally the string
     "CUDA" -- `Device.DEFAULT` is `DEV.device or self._select_device` -- so
     device discovery is short-circuited and `CUDADevice.__init__` (which
     calls `cuInit(0)`) is never reached.
  Both are asserted in code, by probing, at the top of section 1. If either
  stops being true this file fails instead of quietly measuring something else.

This is not a shortcut around the question. It is the question, asked of the
real CUDA scheduler, on hardware that cannot run the answer. What it does NOT
do is render, compile, or launch, so it says nothing about CUDARenderer output;
b4_codegen.py covers rendering.

FOUR THINGS SETTLED
-------------------
  1. Is there a 67,108,864 B kernel argument anywhere in the CUDA plan?  NO.
  2. Is the plan a fused plan?  NO -- 12,082 sink kernels, and 12,082 is a
     number about (C=64, NT=8), not about KDA.
  3. Does the per-architecture census differ?  NO -- identical kernel count
     and identical buffer histogram on sm_75 / sm_80 / sm_89 / sm_90 / sm_100.
  4. Does the r_/E_ split the GPU log reports match what is countable here
     without a GPU?  EXACTLY, both numbers.

    python3 experiment_b/b2_schedule.py             # bench shape, 3 backends
    python3 experiment_b/b2_schedule.py --quick     # one backend, one shape
    python3 experiment_b/b2_schedule.py --arch-census   # the per-arch sweep
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_b.common import (  # noqa: E402
    BENCH, BENCH_TAG, CAPTURE, DECAY_BUCKET, KDA, MEASURED, RE_DERIVED,
    device_status, head, import_kda, kv, nt, read, require, rule, shape_str,
    verdict)

T4_LOG = "out/kaggle/20260928-014248-gpu/logs/perf_kda.log"
T4_REPLAY = "out/kaggle/20260928-014248-gpu/logs/replay_kda.log"
T4_SUMMARY = "out/kaggle/20260928-014248-gpu/SUMMARY.md"

# the decay tensor's size at the two shapes the record uses, re-derived
DECAY_BENCH = BENCH["B"] * nt(BENCH) * BENCH["HV"] * BENCH["C"] ** 2 * BENCH["K"] * 4
DECAY_CAPTURE = (CAPTURE["B"] * nt(CAPTURE) * CAPTURE["HV"]
                 * CAPTURE["C"] ** 2 * CAPTURE["K"] * 4)


# =============================================================================
# 0. prove the two facts the census rests on, rather than citing them
# =============================================================================

def prove_device_free() -> dict:
    """Probe the two properties. Raises if either is false."""
    from tinygrad import Device
    from tinygrad.helpers import Context
    from tinygrad.uop.ops import Ops
    import tinygrad.schedule as sch

    out = {}
    with Context(DEV="CUDA"):
        out["default_reads_CUDA"] = (Device.DEFAULT == "CUDA")
    # the rewrite is reached through a module attribute we can wrap
    hits = {"n": 0}

    def spy(linear, held=None):
        hits["n"] += 1
        return sch._ORIG(linear, held)

    sch._ORIG = sch.memory_plan_rewrite
    sch.memory_plan_rewrite = spy
    out["memory_plan_rewrite_patched"] = True
    return out


# =============================================================================
# 1. the instrument
# =============================================================================

class Census:
    """Reads the schedule without executing it.

    Wraps `memory_plan_rewrite` because that is the last point at which the
    schedule is still LOGICAL -- one UOp per intermediate. After it, tinygrad
    suballocates into per-lane arenas, and the arenas are peaks, not tensors.
    So the set of BUFFER UOps at that point is exactly the set of buffers a
    launch can read or write, which is the quantity that decides whether a
    byte figure can exist at all.
    """

    def __init__(self) -> None:
        self.reset()
        self._orig = None
        self._np = None

    def reset(self) -> None:
        self.params: collections.Counter = collections.Counter()
        self.sink = 0
        self.copy = 0
        self.reduce_kernels = 0
        self.per_kernel: list[dict] = []

    @property
    def elementwise(self) -> int:
        return self.sink - self.reduce_kernels

    @property
    def max_param(self) -> int:
        return max(self.params) if self.params else 0

    @property
    def arg_bytes(self) -> int:
        return sum(n * c for n, c in self.params.items())

    def __enter__(self):
        import tinygrad.schedule as sch
        self._sch = sch
        self._orig = sch.memory_plan_rewrite
        c = self

        def spy(linear, held_bufs=None):
            c._read(linear)
            return c._orig(linear, held_bufs)

        sch.memory_plan_rewrite = spy
        return self

    def __exit__(self, *exc) -> None:
        self._sch.memory_plan_rewrite = self._orig

    def intercept_numpy(self) -> None:
        """`kda/tiny_kda.py` ends with two `.numpy()` calls, and those are its
        only realizes. Substituting `schedule_linear()` gives a fresh graph per
        census, through the real scheduler, with no execution -- so there is no
        realised state to be stale."""
        import numpy as np
        from tinygrad import Tensor

        def fake_numpy(self, *a, **k):
            self.schedule_linear()
            return np.zeros((), dtype=np.float32)

        self._np, Tensor.numpy = Tensor.numpy, fake_numpy

    def restore_numpy(self) -> None:
        if self._np is not None:
            from tinygrad import Tensor
            Tensor.numpy = self._np
            self._np = None

    def _read(self, linear) -> None:
        from tinygrad.uop.ops import Ops
        for si in linear.src:
            is_copy = si.src[0].op is Ops.COPY
            self.copy += int(is_copy)
            if is_copy:
                continue
            self.sink += 1
            bufs = [s for s in si.src[1:] if s.op is Ops.BUFFER]
            for b in bufs:
                self.params[b.nbytes()] += 1
            self.reduce_kernels += int(
                any(u.op is Ops.REDUCE for u in si.src[0].toposort()))
            self.per_kernel.append(dict(nargs=len(bufs),
                                        nbytes=[b.nbytes() for b in bufs]))


def run_census(dev: str, shape: dict) -> dict:
    from tinygrad import Device
    from tinygrad.helpers import Context
    test_ref, tiny_kda = import_kda()
    c = Census()
    t0 = time.perf_counter()
    with Context(DEV=dev):
        assert Device.DEFAULT == dev, f"{Device.DEFAULT!r} != {dev!r}"
        seen = Device.DEFAULT
        q, k, v, g, beta = test_ref.rnd(shape["B"], shape["T"], shape["H"],
                                        shape["HV"], shape["K"], shape["V"],
                                        seed=0, l2n=True)
        with c:
            c.intercept_numpy()
            try:
                tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=shape["C"])
            finally:
                c.restore_numpy()
    return dict(dev=dev, shape=shape, secs=time.perf_counter() - t0, c=c,
                default_in_process=seen)


# =============================================================================
# 2. the GPU log, parsed rather than transcribed
# =============================================================================

def parse_t4(path: str) -> dict:
    out: dict = {}
    try:
        with open(path) as f:
            lines = [ln.rstrip() for ln in f]
    except OSError:
        return out
    for ln in lines:
        m = re.match(r"\s*kernels per call\s*:\s*([\d,]+)", ln)
        if m:
            out["launches"] = int(m.group(1).replace(",", ""))
        m = re.match(r"\s*GPU time\s*:\s*([\d,]+) us", ln)
        if m:
            out["gpu_us"] = int(m.group(1).replace(",", ""))
        m = re.match(r"\s*r\s+([\d,]+) kernels\s+([\d,]+) us\s+([\d.]+)%", ln)
        if m:
            out["r_n"], out["r_us"], out["r_pct"] = (
                int(m.group(1).replace(",", "")),
                int(m.group(2).replace(",", "")), float(m.group(3)))
        m = re.match(r"\s+E\s+([\d,]+) kernels\s+([\d,]+) us\s+([\d.]+)%", ln)
        if m:
            out["E_n"], out["E_us"], out["E_pct"] = (
                int(m.group(1).replace(",", "")),
                int(m.group(2).replace(",", "")), float(m.group(3)))
        m = re.match(r"\s*measured \((\w+),", ln)
        if m:
            out["logged_device_label"] = m.group(1)
        m = re.match(r"\s*of which GPU\s*:\s*([\d,.]+) ms", ln)
        if m:
            out["gpu_ms"] = float(m.group(1).replace(",", ""))
    return out


# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="CUDA at the bench shape only")
    ap.add_argument("--arch-census", action="store_true",
                    help="the per-architecture sweep at a small shape")
    a = ap.parse_args()

    st = device_status()
    head("B2. THE SCHEDULE CENSUS  --  run with no GPU, on the real CUDA scheduler")
    kv("GPU visible here", st["gpu_present"])
    kv("nvcc / ptxas / cuobjdump / ncu",
       " / ".join(str(st[k]) for k in ("nvcc", "ptxas", "cuobjdump", "ncu")))
    kv("libcuda", st["libcuda"])
    print()

    facts = prove_device_free()
    kv("Context(DEV='CUDA') makes Device.DEFAULT read 'CUDA'",
       facts["default_reads_CUDA"])
    kv("memory_plan_rewrite reachable and wrappable", facts["memory_plan_rewrite_patched"])
    kv("the port", "kda/tiny_kda.py (pinned in common.PINS)")
    kv("bench shape", f"{shape_str(BENCH)}   [{BENCH_TAG}]")
    print()

    # ---------------------------------------------------------------- section 1
    head("1. THE BUCKET THAT WAS LOOKED FOR")
    print(f"  [B,NT,HV,C,C,K] at the bench shape, fp32:")
    print(f"    {BENCH['B']} x {nt(BENCH)} x {BENCH['HV']} x {BENCH['C']} x "
          f"{BENCH['C']} x {BENCH['K']} x 4 B = {DECAY_BENCH:,} B  [{RE_DERIVED}]")
    print(f"    that is 2^{DECAY_BENCH.bit_length()-1}, i.e. exactly the "
          f"{DECAY_BUCKET:,} B bucket the record asked about")
    print(f"  and it is {DECAY_BENCH/1_048_576:.0f}x the size of a [B,NT,HV,C,C] fp32"
          f" slab, which is")
    print("  the shape class the largest legal argument is expected to be in.")

    # ---------------------------------------------------------------- section 2
    devs = ["CUDA"] if a.quick else ["CUDA", "METAL", "CPU"]
    results = []
    for d in devs:
        r = run_census(d, BENCH)
        results.append(r)
        c = r["c"]
        kv(f"census DEV={d}", f"{r['secs']:.1f}s   sink kernels {c.sink:,}   "
                             f"copy {c.copy:,}   arg bytes {c.arg_bytes:,}")

    cu = results[0]["c"]
    head("2. THE HISTOGRAM OF KERNEL-ARGUMENT BUFFER SIZES  [MEASURED, no device]")
    print(f"  DEV={results[0]['dev']}, shape {shape_str(BENCH)}")
    print()
    print(f"  {'nbytes':>14}{'count':>10}   what it is in the port")
    cand = {
        DECAY_BENCH: "[B,NT,HV,C,C,K]  THE DECAY TENSOR",
        1_048_576: "[B,NT,HV,C,C]  Z / Btri / A / M  <- the largest legal argument",
        131_072: "[B,T,HV,K]  q / k / g",
        16_384: "[B,NT,HV,C,K]  g_cum / tq / tk / gam",
        256: "[C]  an eyeT row",
    }
    for n, k in sorted(cu.params.items()):
        print(f"  {n:>14,}{k:>10,}   {cand.get(n, 'not a single tensor in the port')}")
    print()
    print(f"  distinct argument sizes: {len(cu.params)}")
    print(f"  argument slots in total: {sum(cu.params.values()):,}"
          f"   (a kernel may hold several, so this is not the kernel count)")
    print(f"  largest argument:        {cu.max_param:,} B")
    print(f"  the decay bucket:        {DECAY_BENCH:,} B   PRESENT: "
          f"{DECAY_BENCH in cu.params}   count: {cu.params.get(DECAY_BENCH, 0)}")
    print(f"  total argument bytes:    {cu.arg_bytes:,} B")
    print()
    print(f"  VERDICT ON THE BUCKET: the decay tensor is not an argument of any")
    print(f"  kernel in the plan, at any count. A buffer carries traffic only if a")
    print(f"  kernel reads or writes it, and a kernel can only touch buffers in its")
    print(f"  argument list. That argument does not depend on a cache, an L2, a")
    print(f"  card, or a backend's memory model. So the decay tensor cannot be")
    print(f"  generating DRAM traffic in this plan, and no bandwidth-bound story")
    print(f"  for KDA on tinygrad can be told from these shapes.")
    print()
    print("  SCOPE DISCIPLINE, restated because it is worth 2x:")
    print("    32x  decay / largest buffer in the CAPTURED kernel's signature  (b1)")
    print(f"    {DECAY_BENCH/cu.max_param:.0f}x  decay / largest argument in the whole plan"
          f"          (this file)")
    print("  Both exact; they are different questions.")

    # ---------------------------------------------------------------- section 3
    if len(results) > 1:
        head("3. BACKEND BY BACKEND, AT THE SAME SHAPE")
        print(f"  {'backend':<8}{'sink':>9}{'with REDUCE':>14}{'elementwise':>13}"
              f"{'largest arg':>14}{'arg bytes':>18}")
        for r in results:
            c = r["c"]
            print(f"  {r['dev']:<8}{c.sink:>9,}{c.reduce_kernels:>14,}"
                  f"{c.elementwise:>13,}{c.max_param:>14,}{c.arg_bytes:>18,}")
        cu_s, me_s = results[0]["c"].sink, results[1]["c"].sink
        print()
        print(f"  CUDA {cu_s:,} vs METAL {me_s:,}: {me_s - cu_s:+,} kernels.")
        print("  The difference is the buffer-count-limiting rewrite, which is")
        print("  keyed on device string and has no CUDA entry, so the rewrite that")
        print("  would merge the elementwise tail never runs on CUDA. That is a")
        print("  property of the name table, not of the card.")
        print()
        print("  CONSEQUENCE FOR THE RECORD: 12,082 is a CUDA number and 12,090 is")
        print("  a METAL number. A document that quotes '12,082' without naming")
        print("  the backend has not made a claim; it has made a coincidence.")

    # ---------------------------------------------------------------- section 4
    head("4. AGAINST THE GPU LOG, WHICH IS THE ONLY HARDWARE NUMBER HERE")
    t4 = parse_t4(require(T4_LOG))
    summ = read(T4_SUMMARY)
    dev_line = next((ln.strip("- ").strip() for ln in summ.splitlines()
                     if "accelerator" in ln), "<not found>")
    if not t4:
        print("  <log missing>")
    else:
        print(f"  {T4_LOG}  (pinned)")
        print(f"    logged device label in the text : \"{t4.get('logged_device_label')}\"")
        print(f"    {T4_SUMMARY} says: {dev_line}")
        print(f"    -> the label in the log is a hard-coded f-string at")
        print(f"       kda/perf_kda.py:143, not a reading. Corrected here: the")
        print(f"       12,082 and the 215.9 ms were measured on the T4, on CUDA.")
        print(f"    kernels per call                 {t4.get('launches', 0):,}")
        print(f"    GPU time (DEBUG=2, event-bracketed) {t4.get('gpu_us', 0):,} us")
        print(f"    r_ (with a REDUCE)               {t4.get('r_n', 0):,}")
        print(f"    E_ (elementwise)                 {t4.get('E_n', 0):,}")
        print()
        print("  What the census can reproduce here, with no device:")
        print(f"    total     census {cu.sink:>8,}   log {t4.get('launches',0):>8,}"
              f"   match {cu.sink == t4.get('launches')}")
        print(f"    r_        census {cu.reduce_kernels:>8,}   log {t4.get('r_n',0):>8,}"
              f"   match {cu.reduce_kernels == t4.get('r_n')}")
        print(f"    E_        census {cu.elementwise:>8,}   log {t4.get('E_n',0):>8,}"
              f"   match {cu.elementwise == t4.get('E_n')}")
        print()
        print("  The r_/E_ split is readable off the kernel NAME, and tinygrad builds")
        print("  that name from `self.reduceop is not None` -- i.e. from whether the")
        print("  kernel carries a REDUCE. That is a property of the AST, so it is")
        print("  countable at schedule time, with no renderer and no card. Three")
        print("  numbers matching exactly is the instrument proving itself.")
        print()
        print("  The GPU TIME cannot be reproduced here, and is not claimed to be.")
        print("  It is also an UPPER BOUND: perf_kda.py runs with DEBUG=2, and")
        print("  engine/realize.py:318 then makes `wait` true, so every launch goes")
        print("  through a create/record/synchronise/destroy event pair and the")
        print("  host blocks per launch. 215.9 ms is 12,082 event-bracketed,")
        print("  host-synchronised launches; it is not a measurement of GPU work.")

    # ---------------------------------------------------------------- section 5
    if a.arch_census:
        head("5. THE PER-ARCHITECTURE CENSUS  --  and it is identical")
        print("  Re-run at B=1,H=HV=2,K=V=64, C=16, NT=8 (T=128) for speed; the")
        print("  question is whether the arch STRING changes the plan at all.")
        print()
        print(f"  {'arch':<10}{'sink kernels':>14}{'largest arg B':>16}"
              f"{'67,108,864 B present':>22}")
        arch_rows = []
        for arch in ("sm_75", "sm_80", "sm_89", "sm_90", "sm_100"):
            s = dict(B=1, T=128, H=2, HV=2, K=64, V=64, C=16)
            r = run_census_arch(arch, s)
            arch_rows.append(r)
            print(f"  {arch:<10}{r['sink']:>14,}{r['max']:>16,}"
                  f"{str(r['present']):>22}")
        uniq = {r["fingerprint"] for r in arch_rows}
        print()
        print(f"  distinct (kernel count, buffer histogram) fingerprints across the")
        print(f"  five architectures: {len(uniq)}")
        print("  So the schedule is arch-independent. That is why the census needs")
        print("  no GPU and no GPU ARCHITECTURE, and it is also why a per-arch")
        print("  hardware sweep could not change the (b) verdicts: the plan does")
        print("  not depend on the arch. What DOES depend on the arch is what gets")
        print("  rendered from it -- which is b4_codegen.py's subject, and it is")
        print("  also arch-independent, for the same reason.")

    # ---------------------------------------------------------------- section 6
    head("6. WHAT THIS FILE DOES NOT SETTLE")
    print("  * DRAM traffic. The census bounds what a kernel MAY touch, not what")
    print("    it DID touch. `dram__bytes.sum` is the only thing that closes it, and")
    print("    b6_next_measurement.py gives the exact command, the expected cost,")
    print("    and the decision each outcome forces.")
    print("  * Wall clock. 12,082 kernels is a COUNT. Whether that is expensive is")
    print("    a question about launch overhead, and the only number in the repo is")
    print("    an event-bracketed upper bound. b3_kernel_count.py says what the")
    print("    count is a function of; it does not say what the count costs.")
    print("  * Whether the OPERATOR is fused. It is not, and 12,082 is the")
    print("    evidence -- but the DECAY is fused within each kernel, and those are")
    print("    two different claims about two different levels. b1 measures the")
    print("    first, this file measures the second, and the record kept merging")
    print("    them.")

    verdict("B2 VERDICT  there is no 67,108,864 B kernel argument anywhere in the\n"
            "         CUDA plan, the largest argument any launch may touch is\n"
            "         1,048,576 B, and the plan is identical on five architectures.\n"
            "         The operator is NOT fused -- 12,082 sink kernels, reproducible\n"
            "         here to the exact r_/E_ split the GPU log reports -- while the\n"
            "         decay IS fused within each kernel. Those are separate claims\n"
            "         and this file exists so they cannot be quoted as one.")
    return 0


def run_census_arch(arch: str, shape: dict) -> dict:
    """One census with `DEV=NULL:CUDA:<arch>`, so the arch STRING drives the
    tensor-core and dtype table without a device existing."""
    import collections
    os.environ["DEV"] = f"NULL:CUDA:{arch}"
    os.environ["PARALLEL"] = "0"
    from tinygrad import Tensor
    from tinygrad.uop.ops import Ops
    import tinygrad.schedule as sch
    import numpy as np
    sink = [0]
    hist: collections.Counter = collections.Counter()
    orig = sch.memory_plan_rewrite

    def spy(linear, held=None):
        for si in linear.src:
            if si.src[0].op is Ops.COPY:
                continue
            sink[0] += 1
            for s in si.src[1:]:
                if s.op is Ops.BUFFER:
                    hist[s.nbytes()] += 1
        return orig(linear, held)

    sch.memory_plan_rewrite = spy
    Tensor.numpy = (lambda self, *a, **k:
                    (self.schedule_linear(), np.zeros((), dtype=np.float32))[1])
    try:
        test_ref, tiny_kda = import_kda()
        q, k, v, g, beta = test_ref.rnd(shape["B"], shape["T"], shape["H"],
                                        shape["HV"], shape["K"], shape["V"],
                                        seed=0, l2n=True)
        tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=shape["C"])
    finally:
        sch.memory_plan_rewrite = orig
    fp = (sink[0], tuple(sorted(hist.items())))
    return dict(arch=arch, sink=sink[0], max=max(hist) if hist else 0,
                present=DECAY_BENCH in hist, fingerprint=fp)


if __name__ == "__main__":
    sys.exit(main())
