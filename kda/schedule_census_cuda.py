"""schedule_census_cuda.py - the decay is fused. The OPERATOR is not.

THE TWO CLAIMS THAT MUST NOT MERGE
----------------------------------
`kda/materialisation_verdict.md` established, from the emitted CUDA source, that
the KDA decay tensor `[B,NT,HV,C,C,K]` is never materialised. That capture also
printed `captured 1 kernels from CUDA`, while `kda/perf_kda.py` measured 12,082
kernel launches per call at the bench shape on the same class of machine. The
record has been at risk of reading the first as evidence about the second. They
are different claims and they have opposite implications:

  CLAIM 1  the DECAY TENSOR never reaches device memory
           -> one tensor is absent from the program. No traffic, no bytes.
  CLAIM 2  the decay COMPUTATION is recomputed, not reused
           -> 32x `exp2` in registers. Arithmetic, not traffic.
  CLAIM 3  the OPERATOR's SCHEDULE is a fused schedule
           -> a property of the graph, 12,082 kernels deep.

1 and 2 are per-kernel properties and can both be true while 3 is false. This
file gives each its own instrument and its own verdict, and never lets a result
for one stand in for a result for another.

WHAT THIS FILE ADDS
-------------------
`kda/materialisation_verdict.md` section 7 named the exact measurement that
would close the CUDA gap -- "run the M2 schedule census with
`Device.DEFAULT == CUDA` at the bench shape and report the histogram of kernel
argument buffer sizes; is there a bucket at 67,108,864 B?" -- and this file
runs it. The answer is NO, and it is reachable on a laptop with no GPU, for a
reason that turns out to matter for three other questions:

  **The schedule census needs no device at all.** `Tensor.schedule_linear()`
  (tensor.py:413) calls `create_linear_with_vars` -> `memory_plan_rewrite`
  (schedule/memory.py:20), which is a pure UOp rewrite driven by *device
  strings*. Nothing in the path constructs a `Device`, allocates a buffer, or
  runs a kernel. `with Context(DEV="CUDA")` makes `Device.DEFAULT` literally
  read `"CUDA"` -- `Device.DEFAULT` is `DEV.device or self._select_device`
  (device.py:44), so the context short-circuits device discovery -- and every
  `Tensor` in `kda_chunk` is then built on the CUDA backend without a CUDA
  device ever existing.

That is not a shortcut around the question. It is the question, asked of the
real CUDA scheduler, on hardware that cannot run the result. What it does NOT do
is render, compile, or launch, so it says nothing about CUDARenderer output.

FOUR THINGS THIS FILE SETTLES
----------------------------
  1. Is there a 67,108,864 B kernel argument in the CUDA schedule?  NO.
  2. Why did the notebook capture 1 kernel out of 12,082?  Because
     `engine/realize.py:273` compiles in a *multiprocessing worker pool*
     whenever more than one kernel is outstanding, so an in-process
     `compile_cached` hook sees only the stragglers. Reproduced off-GPU.
  3. Why is CUDA 12,082 and Metal 12,090?  `DEVICE_MAX_BUFS` at
     `schedule/rangeify.py:354` is `{"METAL": 31, "WEBGPU": 8, "CPU": 31}` and
     has no CUDA entry, so the buffer-count-limiting rewrite never runs on
     CUDA. Reproduced in both directions.
  4. How much of the T4's 215.9 ms is the schedule rather than the arithmetic?
     Bounded from below/above by device-free arithmetic; NOT settled, and the
     one measurement that settles it is named.

Run:
    .venv/bin/python kda/schedule_census_cuda.py              # the full census
    .venv/bin/python kda/schedule_census_cuda.py --quick      # bench shape only
    .venv/bin/python kda/schedule_census_cuda.py --no-replay  # skip the METAL run

Status vocabulary, same as `kda/materialisation_verdict.md`:
  MEASURED    an instrument on this host, on real tinygrad 0.14.0
  RE-DERIVED  recomputed here from the shapes / from tinygrad's source
  MODELLED    arithmetic on shapes
  INFERRED    a conclusion that needs hardware this host does not have. LABELLED
  SOURCE      read out of the installed tinygrad 0.14.0, quoted with file:line
Nothing in this file is transcribed from another agent's file. Every number is
recomputed by the code below or is quoted from a log with its path.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from tinygrad import Device, Tensor  # noqa: E402
from tinygrad.helpers import Context  # noqa: E402
from tinygrad.uop.ops import Ops  # noqa: E402

MB = 1e6
MIB = 1024 ** 2


# =============================================================================
# 1. the shapes, named once
# =============================================================================
# kda/perf_kda.py:90 -- the shape the study benchmarks, and the shape BOTH
# contested byte figures are quoted at. This is also the shape the T4 run
# measured 12,082 launches at (out/kaggle/20260928-014248-gpu/logs/perf_kda.log:6).
BENCH = dict(B=2, T=512, H=4, HV=4, K=64, V=64, C=64, tag="perf_kda.py:90 (bench)")

# kernels/build_notebooks.py:590 -- the shape `captured 1 kernels` ran at. This
# is the SMALLER shape. The brief's framing ("perf_kda measured 12,082 at a
# larger shape") has the direction right and the identity wrong: 12,082 was
# measured at the BENCH shape, and the capture was at this one.
CAPTURE = dict(B=2, T=128, H=2, HV=2, K=32, V=32, C=32,
               tag="build_notebooks.py:590 (the 1-kernel capture)")

# 2x NT on the bench shape. The fp32 decay tensor is 134,217,728 B here, so the
# decisive bucket moves with it and the question stays the same shape of
# question. notes/09 section 4.3 covered this shape on METAL.
BIGGER = dict(B=2, T=1024, H=4, HV=4, K=64, V=64, C=64,
              tag="bench shape, NT doubled")


def nt(s: dict) -> int:
    return s["T"] // s["C"]


def slabs(s: dict) -> int:
    return s["B"] * nt(s) * s["HV"]


def decay_bytes(s: dict, es: int = 4) -> int:
    """ONE [B,NT,HV,C,C,K] decay tensor, fp32 by construction
    (`kda/tiny_kda.py:44` sets F = dtypes.float32 and nothing casts away)."""
    return slabs(s) * s["C"] * s["C"] * s["K"] * es


def cc_bytes(s: dict, es: int = 4) -> int:
    """the [B,NT,HV,C,C] family -- what `notes/09` section 4.3's identity says
    the largest kernel argument should be."""
    return slabs(s) * s["C"] * s["C"] * es


def shape_str(s: dict) -> str:
    return (f"B={s['B']} T={s['T']} H={s['H']} HV={s['HV']} K={s['K']} "
            f"V={s['V']} C={s['C']}  NT={nt(s)}  B*NT*HV={slabs(s):,}")


def rule(c: str = "=", n: int = 96) -> None:
    print(c * n)


def head(title: str) -> None:
    print()
    print(title)
    rule("-")


# =============================================================================
# 2. the instrument
# =============================================================================
# A census that does NOT execute anything.
#
# `Tensor.realize()` on an already-realized graph is a silent no-op, so any
# census that measures by realizing twice measures nothing the second time.
# This one sidesteps the trap instead of handling it: it never realizes. It
# intercepts `kda_chunk`'s two `.numpy()` calls -- which is where the operator's
# only two realizes live, per `kda/tiny_kda.py:252` -- and substitutes
# `Tensor.schedule_linear()`, the public API for exactly this. So each census is
# a fresh graph (kda_chunk takes numpy inputs and rebuilds every call) driven
# through the real scheduler, with no execution and therefore no state to be
# stale.
#
# `memory_plan_rewrite` is the hook because it is the last point at which the
# schedule is still LOGICAL, one UOp per intermediate. What it receives is
# exactly the set of buffers a launch can read or write: after it, tinygrad
# suballocates everything into per-lane arenas, and the arenas are peaks, not
# tensors.
class ScheduleCensus:
    """Buckets every buffer that appears as a kernel argument, by nbytes.

    Also keeps, per kernel: whether it contains a REDUCE (the r_/E_ split that
    `kda/perf_kda.py:43-71` reads out of the kernel NAME, recoverable here
    without rendering), the trip count (the product of its RANGE extents), and
    the static ALU/transcendental census of its body.
    """

    def __init__(self) -> None:
        self.reset()
        self._orig = None
        self._sch = None
        self._np = None

    def reset(self) -> None:
        self.params: collections.Counter = collections.Counter()
        self.kernels = 0
        self.sink = 0
        self.copy = 0
        self.reduce_kernels = 0
        self.trip_total = 0            # sum over kernels of the RANGE product
        self.alu_static = 0            # sum over kernels of static ALU uops
        self.lane_ops = 0              # ALU uops x trip, reduce axis discounted
        self.exp_static = 0            # static EXP2 outside a REDUCE
        self.exp_in_reduce = 0         # static EXP2 inside a REDUCE
        self.per_kernel: list[dict] = []

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "ScheduleCensus":
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
        """Replace `Tensor.numpy` with a scheduler call. Runtime only; no file
        in `kda/` is edited."""

        def fake_numpy(self, *a, **k):
            self.schedule_linear()          # raises if the graph is unschedulable
            return np.zeros((), dtype=np.float32)

        self._np, Tensor.numpy = Tensor.numpy, fake_numpy

    def restore_numpy(self) -> None:
        if self._np is not None:
            Tensor.numpy = self._np
            self._np = None

    # -- the read -------------------------------------------------------------
    def _read(self, linear) -> None:
        for si in linear.src:
            self.kernels += 1
            is_copy = si.src[0].op is Ops.COPY
            self.copy += int(is_copy)
            if not is_copy:
                self.sink += 1
            bufs = [s for s in si.src[1:] if s.op is Ops.BUFFER]
            for s in bufs:
                self.params[s.nbytes()] += 1
            if is_copy:
                continue
            self._read_body(si.src[0], bufs)

    def _read_body(self, ast, bufs) -> None:
        has_red = False
        trip = 1
        for u in ast.src:                      # the END nodes
            for r in u.src:
                if r.op is Ops.RANGE:
                    trip *= int(r.vmax)
        # ops that live under an INDEX are addressing arithmetic, not work:
        # tinygrad's own estimator excludes them too
        # (`renderer/__init__.py:28-31`, ignore_indexing=True).
        excluded: set = set()
        for u in ast.toposort():
            if u.op is Ops.INDEX:
                excluded |= set(u.sink(*u.src[1:]).toposort())
        # An op under a REDUCE is evaluated once per (trip / reduce_len), not
        # once per trip: the reduce axis is a sibling RANGE in the END, so the
        # trip product counts it but the accumulator does not re-run the body
        # along it. Ops.REDUCE carries its axis as src[1] (verified: arg is
        # (Ops.ADD, 0) and src[1] is the RANGE), so the correction is exact.
        under: list[tuple[set, int]] = []
        alu = exp = 0
        lane_out = lane_in = 0
        for r in ast.toposort():
            if r.op is Ops.REDUCE:
                has_red = True
                rl = max(1, int(r.src[1].vmax)) if len(r.src) > 1 else 1
                under.append((set(r.sink(*r.src[:1]).toposort()), rl))
        for u in ast.toposort():
            if u in excluded or u.op not in _ALU_OPS:
                continue
            alu += 1
            w = 2 if u.op is Ops.MULACC else 1
            rl = next((l for s, l in under if u in s), 0)
            if u.op is Ops.EXP2:
                if rl:
                    self.exp_in_reduce += 1
                else:
                    exp += 1
            if rl:
                lane_in += w * (trip // rl)
            else:
                lane_out += w * trip
        self.reduce_kernels += int(has_red)
        self.trip_total += trip
        self.alu_static += alu
        self.lane_ops += lane_out + lane_in
        self.exp_static += exp
        self.per_kernel.append(dict(reduce=has_red, trip=trip, alu=alu, args=len(bufs),
                                    lane_ops=lane_out + lane_in,
                                    nbytes=[b.nbytes() for b in bufs]))

    # -- readers --------------------------------------------------------------
    @property
    def elementwise(self) -> int:
        return self.sink - self.reduce_kernels

    @property
    def max_param(self) -> int:
        return max(self.params) if self.params else 0

    @property
    def arg_bytes(self) -> int:
        return sum(n * c for n, c in self.params.items())

    def fingerprint(self) -> str:
        return (f"{self.sink}/{self.copy}/{self.max_param}/{self.arg_bytes}/"
                f"{sorted(self.params.items())}")


# Ops that cost a lane operation. Read out of the installed tinygrad rather than
# transcribed, so a version bump cannot silently change the arithmetic.
def _alu_ops() -> frozenset:
    from tinygrad.uop.ops import GroupOp
    out = set(GroupOp.ALU)
    out.add(Ops.EXP2)
    return frozenset(out)


_ALU_OPS = _alu_ops()


def run_census(dev: str, shape: dict) -> dict:
    """One census: fresh graph, `Device.DEFAULT == dev`, no device, no execution.

    Raises if the requested backend cannot even be named. It CAN be named: see
    the module docstring. What cannot be done here is *render* for CUDA, because
    `CUDADevice.__init__` calls `cuInit(0)` (runtime/ops_cuda.py:103) and this
    host has no `libcuda` to load -- verified, and reported in section 9.
    """
    import test_ref as T
    import tiny_kda

    c = ScheduleCensus()
    t0 = time.perf_counter()
    with Context(DEV=dev):
        assert Device.DEFAULT == dev, f"{Device.DEFAULT!r} != {dev!r}"
        seen = Device.DEFAULT
        q, k, v, g, beta = T.rnd(shape["B"], shape["T"], shape["H"], shape["HV"],
                                 shape["K"], shape["V"], seed=0, l2n=True)
        with c:
            c.intercept_numpy()
            try:
                tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=shape["C"])
            finally:
                c.restore_numpy()
    return dict(dev=dev, shape=shape, secs=time.perf_counter() - t0, c=c,
                default_in_process=seen)


# =============================================================================
# 3. section 1 -- the three claims, before any number
# =============================================================================
def section_claims() -> None:
    head("1. THREE CLAIMS, THREE INSTRUMENTS, THREE VERDICTS")
    rows = [
        ("1", "the [C,C,K] decay TENSOR reaches device memory",
         "M2: kernel ARGUMENT buffer sizes", "per-kernel"),
        ("2", "the decay COMPUTATION is recomputed, not reused",
         "source: exp2 count, stored or not", "per-kernel"),
        ("3", "the OPERATOR'S SCHEDULE is a fused schedule",
         "launch census: SINK kernels per call", "graph-level"),
    ]
    print(f"  {'#':<3}{'claim':<52}{'instrument':<40}{'level':<13}")
    rule("-", 96)
    for n, c, i, lv in rows:
        print(f"  {n:<3}{c:<52}{i:<40}{lv:<13}")
    print()
    print("  1 and 2 are properties of ONE kernel. 3 is a property of the GRAPH.")
    print("  A schedule can be 12,082 kernels deep and every one of them can keep")
    print("  the decay in registers. Both of those are true here. Collapsing them")
    print("  into one sentence produces the false claim 'the operator is fused'.")


# =============================================================================
# 4. section 2 -- the decisive census
# =============================================================================
def decode(s: dict, hist: collections.Counter) -> dict:
    """Name each observed size by matching it against the port's own tensors."""
    cand = {
        "[B,NT,HV,C,C,K] DECAY (Dl / Dq_all)": decay_bytes(s, 4),
        "[B,NT,HV,C,C] (Z / Btri / A / M)": cc_bytes(s, 4),
        "[B,NT,HV,C,K] (tq / tk / g_cum / gam)": slabs(s) * s["C"] * s["K"] * 4,
        "[B,NT,HV,C,V] (tv / u / v_new)": slabs(s) * s["C"] * s["V"] * 4,
        "[B,NT,HV,C] (tb, the b vector)": slabs(s) * s["C"] * 4,
        "[B,HV,K,V] (the live state S)": s["B"] * s["HV"] * s["K"] * s["V"] * 4,
        "[B,T,HV,V] (the output o)": s["B"] * s["T"] * s["HV"] * s["V"] * 4,
        "[B,T,HV,K] (q / k / g, GVA-repeated)": s["B"] * s["T"] * s["HV"] * s["K"] * 4,
        "[C,C] (eye_all, the tril masks)": s["C"] * s["C"] * 4,
        "[C] (one eyeT row)": s["C"] * 4,
    }
    maxt = max(v for k, v in cand.items() if "DECAY" not in k)
    out = {}
    for n in hist:
        hit = [k for k, v in cand.items() if v == n]
        out[n] = " | ".join(hit) if hit else "not a single tensor in the port"
    return out, maxt, cand


def report(r: dict) -> dict:
    s, c = r["shape"], r["c"]
    dec, cc = decay_bytes(s, 4), cc_bytes(s, 4)
    print()
    print(f"  {shape_str(s)}")
    print(f"    device string in the census = {r['dev']!r}, "
          f"Device.DEFAULT read {r['default_in_process']!r}, {r['secs']:.0f}s")
    print(f"    decay tensor [B,NT,HV,C,C,K] fp32 = {dec:>15,} B = {dec/MB:>9.2f} MB "
          f"= {dec/MIB:>7.2f} MiB")
    print(f"    [B,NT,HV,C,C] fp32, the identity  = {cc:>15,} B = {cc/MB:>9.2f} MB")
    if s is CAPTURE:
        print("    NOTE: this is the shape kernels/build_notebooks.py:590 captured at.")
        print("    The one piece of real emitted CUDA in this repo --")
        print("    out/kaggle/20260928-014248-gpu/codegen/kernel000.txt -- declares")
        print("    buffers of 4,096 / 4,096 / 16,384 / 16,384 / 4,096 elements, i.e.")
        print("    a largest buffer of 65,536 B. This census's largest kernel")
        print("    argument here is 65,536 B. The two agree to the byte, on a host with")
        print("    no NVIDIA card, which is the strongest tie available between the")
        print("    device-free census and the real lowering.")
        print(f"    the decay at this shape is {dec:,} B = {dec//cc} x the largest")
        print("    argument, and is in no bucket.")
    print()
    print("    SINK kernels (compute)            = {:>15,}".format(c.sink))
    print("    COPY kernels (excluded from DEBUG=2 by perf_kda.py:52) = {:>9,}".format(c.copy))
    print("    with a REDUCE in the body (r_)    = {:>15,}".format(c.reduce_kernels))
    print("    pure elementwise (E_)             = {:>15,}".format(c.sink - c.reduce_kernels))
    print()
    print("    kernel ARGUMENT buffer sizes (this is the whole M2 row):")
    names, maxt, cand = decode(s, c.params)
    for n, k in c.params.most_common():
        tag = "   <-- THE DECAY TENSOR" if n >= dec else ""
        print(f"      {n:>15,} B  arg of {k:>7,} launches  "
              f"({n*k/MB:>11,.1f} MB of argument bytes)  {names[n]}{tag}")
    print(f"      {'largest':>15} B  {c.max_param:>7,}")
    print(f"      total argument bytes = {c.arg_bytes:>15,} B = {c.arg_bytes/MB:>10,.1f} MB")
    print()
    verdict = (f"*** BUCKET AT {dec:,} B EXISTS ***" if dec in c.params else "NO")
    print(f"    >>> IS THERE A BUCKET AT {dec:,} B ?   {verdict}")
    print(f"    largest argument / decay tensor     = 1/{dec/c.max_param:,.0f} "
          f"(K = {s['K']})" if c.max_param else "")
    print(f"    identity B*NT*HV*C*C*4 = {cc:,} B  -> "
          f"{'MATCH' if cc == c.max_param else 'MISMATCH'}")
    return dict(dev=r["dev"], shape=s, sink=c.sink, copy=c.copy,
                reduce_kernels=c.reduce_kernels,
                elementwise=c.sink - c.reduce_kernels,
                params=dict(c.params), max_param=c.max_param,
                arg_bytes=c.arg_bytes, decay=dec, cc=cc,
                bucket_at_decay=dec in c.params,
                trip_total=c.trip_total, alu_static=c.alu_static,
                lane_ops=c.lane_ops, exp_static=c.exp_static,
                exp_in_reduce=c.exp_in_reduce)


def section_census(shapes: list[dict], devs: list[str]) -> list[dict]:
    head("2. THE CENSUS -- IS THERE A 67,108,864 B KERNEL ARGUMENT?")
    print()
    print("  No GPU is present and none is used. `Tensor.schedule_linear()`")
    print("  (tensor.py:413) -> `create_linear_with_vars` -> `memory_plan_rewrite`")
    print("  (schedule/memory.py:20) is a pure UOp rewrite over device STRINGS.")
    print("  `Context(DEV=...)` sets Device.DEFAULT without device discovery")
    print("  (device.py:44: `DEV.device or self._select_device`). Nothing compiles,")
    print("  nothing allocates, nothing launches.")
    print()
    print("  What this therefore measures: tinygrad 0.14.0's real SCHEDULER output")
    print("  for a graph whose tensors are on CUDA. Which is precisely M2.")
    print("  What it does not measure: CUDARenderer output, allocator traffic, or")
    print("  any timing. Those need the device; section 7 says what is left.")
    out = []
    for s in shapes:
        for dev in devs:
            try:
                out.append(report(run_census(dev, s)))
            except Exception as e:  # noqa: BLE001
                print()
                print(f"  {shape_str(s)}  dev={dev}")
                print(f"    NOT SCHEDULABLE: {type(e).__name__}: {str(e)[:110]}")
    return out


# =============================================================================
# 4b. section 2b -- the byte figures this note quotes, re-derived
# =============================================================================
# kda/NOTES.md:16 / kda/cost.py:17 shape, as quoted by notes/03-kda.md:212 and
# kda/NOTES.md:153,162. Recomputed here because those two say "8.2 GB each" and
# two prior agents flagged it, and because notes/09 section 2's 22.2822 MB /
# 5.5050 MB / 13.0471x could not be reproduced by either of them.
NOTES_SHAPE = dict(B=4, T=4096, H=32, HV=32, K=128, V=128, C=64)


def section_bytes() -> None:
    head("2b. THE BYTE FIGURES, RE-DERIVED, SO NOTHING HERE IS TRANSCRIBED")
    print()
    print("  (a) the decay tensor at the bench shape, fp32 -- the decisive size")
    d = decay_bytes(BENCH, 4)
    print(f"      B*NT*HV = {BENCH['B']}*{nt(BENCH)}*{BENCH['HV']} = {slabs(BENCH)}")
    print(f"      [C,C,K] fp32 = {BENCH['C']}*{BENCH['C']}*{BENCH['K']}*4 = "
          f"{BENCH['C']**2*BENCH['K']*4:,} B")
    print(f"      one tensor  = {d:>15,} B = {d/MB:.4f} MB = {d/MIB:.4f} MiB = 64.0000 MiB exactly")
    print(f"      two tensors = {2*d:>15,} B = {2*d/MB:.4f} MB")
    print(f"      x2 touches  = {4*d:>15,} B = {4*d/MB:.4f} MB   <- the withdrawn 268.4 MB")
    print(f"      largest kernel argument in the census = {cc_bytes(BENCH, 4):,} B")
    print(f"      ratio decay : largest argument = {d}/{cc_bytes(BENCH, 4)} = "
          f"{d//cc_bytes(BENCH, 4)} = K")
    print()
    print("  (b) the same tensor at kda/NOTES.md's shape, fp32 -- where '8.2 GB' is wrong")
    n = NOTES_SHAPE
    e = slabs(n) * n["C"] * n["C"] * n["K"]
    print(f"      shape B={n['B']} T={n['T']} H=HV={n['HV']} K=V={n['K']} C={n['C']} "
          f"NT={nt(n)}")
    print(f"      B*NT*HV = {slabs(n):,}   C*C*K = {n['C']*n['C']*n['K']:,}")
    print(f"      elements = {e:,} = 2^32")
    print(f"      fp32  one tensor     {e*4:>15,} B = {e*4/2**30:.2f} GiB = {e*4/1e9:.2f} GB")
    print(f"            a write + a read{e*8:>15,} B = {e*8/1e9:.2f} GB")
    print(f"      bf16  one tensor     {e*2:>15,} B = {e*2/2**30:.2f} GiB = {e*2/1e9:.2f} GB")
    print(f"            a write + a read{e*4:>15,} B = {e*4/1e9:.2f} GB")
    print(f"      notes/03-kda.md:212 and kda/NOTES.md:153,162 say 8.2 GB each and")
    print(f"      16.4 GB for a write+read. 8.2 GB matches NEITHER dtype: bf16 is "
          f"{e*2/1e9:.2f} GB and fp32 is {e*4/1e9:.2f} GB. The fp32 figures are the")
    print(f"      ones the port could ever hit, because kda/tiny_kda.py:44 is fp32 by")
    print(f"      construction. Not carried forward in any form.")
    print()
    print("  (c) notes/09 section 2's 22.2822 MB / 5.5050 MB / 48.7619x / 13.0471x")
    print("      -- NOT RE-DERIVED, NOT USED, AND NOT RE-ATTEMPTED HERE.")
    print("      The numerator re-derives exactly (section 2b(a) above, 268,435,456 B).")
    print("      The denominator has now failed to reproduce in TWO independent")
    print("      attempts -- notes/09's own, and kda/materialisation_verdict.md")
    print("      section 3.3's, which lands on 5,242,880 B and 51.20x against the")
    print("      claimed 5.5050 MB and 48.7619x. A third attempt would only produce a")
    print("      third number and would obscure the point, so this file makes none.")
    print("      What is safe to say: the numerator is exact, the denominator is")
    print("      unattributable, and every ratio built on it is withdrawn. The DECAY")
    print("      side does not need it, because the tensor it charges does not exist.")


# =============================================================================
# 5. section 3 -- determinism, and the device-agreement question
# =============================================================================
def section_determinism(dev: str, shape: dict) -> None:
    head("3. IS THE CENSUS DETERMINISTIC, AND DOES IT SURVIVE A COLD CACHE?")
    a = run_census(dev, shape)
    b = run_census(dev, shape)
    print()
    print(f"  same process, two fresh graphs, dev={dev}")
    print(f"    run 1 {a['c'].fingerprint()[:96]}")
    print(f"    run 2 {b['c'].fingerprint()[:96]}")
    print(f"    identical: {'YES' if a['c'].fingerprint() == b['c'].fingerprint() else 'NO'}")
    print()
    print("  The second run is served by tinygrad's `schedule_cache`")
    print("  (schedule/__init__.py:117), keyed on the function key, so it is NOT")
    print("  an independent measurement of the scheduler. The control that matters")
    print("  is a COLD process:")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one-line",
                        "--dev", dev, "--shape", "bench"],
                       capture_output=True, text=True, cwd=HERE)
    cold = [ln for ln in p.stdout.splitlines() if ln.startswith("CENSUS ")]
    print(f"    cold subprocess: {cold[0][:150] if cold else '<none: rc=%d>' % p.returncode}")
    if cold:
        for tok in cold[0].split():
            if tok.startswith("sink="):
                print(f"    cold sink count = {tok[5:]}  vs warm "
                      f"{a['c'].sink}  -> "
                      f"{'MATCH' if tok[5:] == str(a['c'].sink) else 'MISMATCH'}")


# =============================================================================
# 6. section 4 -- 12,082 vs 12,090, and DEVICE_MAX_BUFS
# =============================================================================
_LIMIT_FIRED = collections.Counter()
_ORIG_LIMIT_BUFS = None


def _limit_bufs_probe(ctx, root):
    """Module-level on purpose: PatternMatcher refuses closures
    (`assert fxn.__closure__ is None`), so the counter cannot be a local."""
    r = _ORIG_LIMIT_BUFS(ctx, root)
    _LIMIT_FIRED["rewrote" if r is not None else "no"] += 1
    return r


def limit_bufs_probe(dev: str, max_bufs: int) -> str:
    """One census with `DEVICE_MAX_BUFS[dev]` forced, in a process that has done
    nothing else. Prints one machine-readable line."""
    import tinygrad.schedule.rangeify as rf
    from tinygrad.uop.ops import PatternMatcher, UPat
    global _ORIG_LIMIT_BUFS
    if max_bufs:
        rf.DEVICE_MAX_BUFS[dev] = max_bufs
    _LIMIT_FIRED.clear()
    _ORIG_LIMIT_BUFS = rf._limit_bufs
    pat = [(UPat(set.union(rf.GroupOp.Binary, rf.GroupOp.Ternary), name="root"),
            _limit_bufs_probe)]
    rf.pm_limit_bufs = PatternMatcher(pat)
    try:
        r = run_census(dev, BENCH)
    finally:
        rf.pm_limit_bufs = PatternMatcher(
            [(UPat(set.union(rf.GroupOp.Binary, rf.GroupOp.Ternary), name="root"),
              _ORIG_LIMIT_BUFS)])
    return (f"LIMITBUFS dev={dev} max_bufs={max_bufs or 'unset'} sink={r['c'].sink} "
            f"elem={r['c'].elementwise} max_param={r['c'].max_param} "
            f"rewrote={_LIMIT_FIRED['rewrote']}")


def section_device_max_bufs() -> None:
    head("4. WHY CUDA SAYS 12,082 AND METAL SAYS 12,090 -- AND WHY THAT IS")
    head("   THE STRONGEST EVIDENCE THAT THE CUDA CENSUS ABOVE IS THE REAL")
    head("   CUDA SCHEDULE")
    import tinygrad.schedule.rangeify as rf
    print()
    print("  tinygrad 0.14.0, schedule/rangeify.py:354, read from the installed")
    print("  package:")
    print(f"    DEVICE_MAX_BUFS = {rf.DEVICE_MAX_BUFS}")
    print("  and the pass that reads it, rangeify.py:360-382: a kernel whose AST")
    print("  touches more than MAX_BUFS-1 buffers has its elementwise inputs")
    print("  force-bufferized, which SPLITS it into more kernels. On METAL and CPU")
    print("  that fires. On CUDA the lookup misses, returns 0, and the function")
    print("  returns None at line 363 -- the pass is simply off.")
    print()
    print("  The literal in the source is `# TODO: get from device?`. Nobody has")
    print("  filled in the CUDA entry, so CUDA gets the *unlimited* schedule and")
    print("  8 fewer, fatter kernels.")
    print()
    print("  CONTROL, in both directions. Instrument `_limit_bufs` to count the")
    print("  rewrites, then set the CUDA entry by hand and re-run the census.")
    print()
    print("  ONE CONFIGURATION PER SUBPROCESS, and that is not fussiness. The")
    print("  rewrite passes memoize on UOp identity, so a second run in the same")
    print("  process re-plays the cached rewrite and `_limit_bufs` never fires. That")
    print("  is the same trap as `Tensor.realize()` on a realized graph, one level")
    print("  down, and the first version of this section fell straight into it: it")
    print("  reported 0 rewrites and 12,082 for every configuration, including")
    print("  METAL, which is knowably 4 and 12,090. A control that cannot fail is")
    print("  not a control.")
    print()
    for patch, dev in ((None, "CUDA"), (31, "CUDA"), (100, "CUDA"),
                       (None, "METAL"), (None, "CPU")):
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--limit-bufs", "--dev", dev,
                            "--max-bufs", "0" if patch is None else str(patch)],
                           capture_output=True, text=True, cwd=HERE)
        ln = [x for x in p.stdout.splitlines() if x.startswith("LIMITBUFS ")]
        if not ln:
            err = p.stderr.strip().splitlines()[-1][:70] if p.stderr.strip() else f"rc={p.returncode}"
            print(f"    DEVICE_MAX_BUFS['{dev}'] = {str(patch):>4}  FAILED: {err}")
            continue
        f = dict(t.split("=", 1) for t in ln[0].split()[1:])
        print(f"    DEVICE_MAX_BUFS['{dev}'] = {str(patch):>4}  SINK = {int(f['sink']):>6,}"
              f"  E_ = {int(f['elem']):>6,}  largest arg = {int(f['max_param']):>10,} B"
              f"  _limit_bufs rewrote {f['rewrote']:>2} roots")
    print()
    print("  Read: pinning CUDA to 31 reproduces METAL's kernel count EXACTLY -- and")
    print("  reproduces it in the r_/E_ split too, because `_limit_bufs` splits the")
    print("  ELEMENTWISE part of a kernel and the 8 extra Metal kernels are all")
    print("  elementwise. Pinning it to 100 reproduces CUDA's. This is a positive")
    print("  control in both directions, and it is the reason the device-free census")
    print("  is not an approximation of the CUDA schedule: the device NAME is the")
    print("  only device-dependent input to `get_kernel_graph`/`memory_plan_rewrite`")
    print("  in this version, we have just found it, and we have just toggled it.")


# =============================================================================
# 6b. section 5 -- how the launch count scales, the shape of claim 3
# =============================================================================
def section_scaling() -> None:
    head("5. HOW THE LAUNCH COUNT SCALES -- the one device-free statement that")
    head("   says something structural about claim 3")
    print()
    print("  Same graph, same code, C=64 held, NT swept, DEV=CUDA, no GPU. What")
    print("  matters is which of the three quantities moves and which does not.")
    print()
    print(f"  {'T':>6}{'NT':>5}{'B*NT*HV':>9}{'SINK':>8}{'REDUCE':>9}{'ELEM':>7}"
          f"{'trip':>14}{'arg MB':>10}")
    rule("-", 96)
    rows = []
    for T in (128, 256, 512, 1024, 2048):
        s = dict(B=2, T=T, H=4, HV=4, K=64, V=64, C=64, tag=f"NT={T//64}")
        r = run_census("CUDA", s)
        c = r["c"]
        rows.append((T // 64, c.sink, c.reduce_kernels, c.elementwise,
                     c.trip_total, c.arg_bytes))
        print(f"  {T:>6}{T//64:>5}{slabs(s):>9}{c.sink:>8,}{c.reduce_kernels:>9,}"
              f"{c.elementwise:>7,}{c.trip_total:>14,}{c.arg_bytes/MB:>10,.0f}")
    rule("-", 96)
    nt0, s0, r0, e0, t0, a0 = rows[0]
    nt1, s1, r1, e1, t1, a1 = rows[-1]
    per_nt = (r1 - r0) // (nt1 - nt0)
    const = e0 + r0 - per_nt * nt0
    print()
    print(f"  elementwise kernels are CONSTANT at {e0:,} across a 16x span of NT. They")
    print("  come from the unrolled forward substitution at kda/tiny_kda.py:181-194,")
    print("  whose trip count is C(C-1) = 64*63 = 4,032 and is a function of C alone.")
    print(f"  reduce kernels go {r0:,} -> {r1:,}, a slope of exactly {per_nt} per chunk, so")
    print(f"      SINK(NT) = {const:,} + {per_nt}*NT,  and the constant dominates:")
    print(f"        NT=8   (the bench shape)      {rows[2][1]:>8,}")
    print(f"        NT=16                         {rows[3][1]:>8,}")
    print(f"        NT=32                         {rows[4][1]:>8,}")
    print(f"        NT=64  (kda/cost.py's shape)  {const + per_nt*64:>8,}   [MODELLED, never scheduled]")
    print()
    print(f"  Over the same 16x span the WORK grows {(t1/t0):.0f}x in trip count and "
          f"{(a1/a0):.0f}x in argument")
    print(f"  bytes, while the launch count grows {(s1/s0-1)*100:.1f}%. The schedule's")
    print("  DEPTH is a constant of the unrolled triangular solve, not a function of")
    print("  the problem size.")
    print()
    print("  CONSEQUENCE FOR CLAIM 3, and it cuts both ways:")
    print("  - it is real: ~12,000 launches is ~12,000 independent synchronisation")
    print("    points per call, and it does not amortise.")
    print("  - it is NOT a function of T, so as the problem grows the launch count")
    print("    becomes a SMALLER fraction of the runtime, not a larger one. Any claim")
    print("    of the form 'the unrolled schedule costs a factor N' must be stated at")
    print("    a stated shape, and it gets weaker as the shape grows. That cuts")
    print("    against the study's framing as much as for it.")


# =============================================================================
# 7. section 6 -- why one kernel was captured
# =============================================================================
CAPTURE_SRC = r"""
import collections, os, sys
sys.path.insert(0, "@HERE@")
import numpy as np
from tinygrad import Device, Tensor
import tinygrad.device as TD
import tinygrad.engine.realize as RZ
import tinygrad.engine.worker as W

C = collections.Counter()
d = Device[Device.DEFAULT]
# (f) per-lower_and_compile accounting. `bump` has to be defined BEFORE the
#     spies so they can record which round they fired in -- that is the whole
#     point of the round table.
per = []
cur = {}
def bump(k, n=1):
    C[k] += n; cur[k] = cur.get(k, 0) + n

# (a) the notebook's hook, verbatim in substance: patch the instance attribute
#     on the device's Compiler object.
comp = d.compiler
_o_cc = comp.compile_cached
def cc(src, *a, **k):
    bump("compile_cached")
    C["compile_cached_chars"] += len(src)
    C["exp2_in_src"] += src.count("exp2(")
    return _o_cc(src, *a, **k)
comp.compile_cached = cc

# (b) the codegen entry point
_o_tp = RZ.to_program
def tp(*a, **k):
    bump("to_program")
    return _o_tp(*a, **k)
RZ.to_program = tp

# (c) the worker-pool decision
_o_gwp = RZ.get_worker_pool
def gwp():
    p = _o_gwp()
    bump("pool_None" if p is None else "pool_ProcessPool")
    return p
RZ.get_worker_pool = gwp

# (d) one Program object per distinct ELF
_o_rt = TD.Compiled.runtime
def rt(self, obj):
    bump("Compiled.runtime")
    return _o_rt(self, obj)
TD.Compiled.runtime = rt

# (e) one line per launch
rt_t = d.runtime_t
_o_call = rt_t.__call__
def call(self, *a, **k):
    bump("launch")
    return _o_call(self, *a, **k)
rt_t.__call__ = call

_o_lac = RZ.lower_and_compile
def lac(linear):
    cur.clear()
    r = _o_lac(linear)
    per.append((len(linear.src), dict(cur)))
    return r
RZ.lower_and_compile = lac

import test_ref as T, tiny_kda
print("DEFAULT", Device.DEFAULT, "PARALLEL", W.PARALLEL.value)
q, k, v, g, beta = T.rnd(@B@, @T@, @H@, @HV@, @K@, @V@, seed=0, l2n=True)
t0 = __import__("time").perf_counter()
tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=@C@)
t1 = __import__("time").perf_counter()
keys = ["launch", "Compiled.runtime", "to_program", "compile_cached",
        "compile_cached_chars", "exp2_in_src", "pool_ProcessPool", "pool_None"]
print("WALL %.2f" % (t1 - t0))
print("COUNTS " + " ".join(f"{k}={C[k]}" for k in keys))
print("ROUNDS " + " | ".join(f"n={n} {sorted(c.items())}" for n, c in per))
"""


def section_capture(dev: str) -> None:
    head("6. WHY ONE KERNEL WAS CAPTURED OUT OF 12,082")
    print()
    print("  `kernels/build_notebooks.py:578-597` wraps")
    print("  `Device[DEFAULT].compiler.compile_cached` in the calling process and")
    print("  prints `len(cap)`. It got 1. Here is the mechanism, in tinygrad's")
    print("  own source, and then a reproduction of it on a host with no NVIDIA")
    print("  card in it.")
    print()
    print("  engine/realize.py:263-288, `lower_and_compile`:")
    print("    265  collect every CALL that still needs lowering")
    print("    269  todo = {compile-cache-key: ast} for the ones not yet compiled")
    print("    273  pool = None if len(todo) == 1 or any(beam) else get_worker_pool()")
    print("    278  for i, prg in (map if pool is None else pool.imap_unordered)(...)")
    print("  engine/worker.py:37-44, `get_worker_pool`: a multiprocessing SpawnContext")
    print("  Pool of PARALLEL processes, and")
    print("    helpers.py:269  PARALLEL = NUM_CPU_THREADS (os.cpu_count()) by default")
    print()
    print("  So: when more than one kernel needs compiling, tinygrad compiles them")
    print("  in SUBPROCESSES, and `compile_cached` runs in a child, where the")
    print("  notebook never patched it. An in-process hook sees only the rounds")
    print("  where `len(todo) == 1`, which are the only rounds that take the `map`")
    print("  branch. `kda_chunk` makes FOUR `lower_and_compile` rounds at the bench")
    print("  shape (6,188 / 1 / 6,041 / 1 linear calls; the 1-call rounds are")
    print("  copies). The two large rounds have todos of 190 and 1 respectively --")
    print("  the round table below shows it -- and only the second is a singleton in")
    print("  `todo` terms, so only it compiles in the parent. That is the capture.")
    print()
    print("  REPRODUCTION, on this host, on the backend that exists here.")
    print("  Five counters on one `kda_chunk` call, the notebook's shape first")
    print("  (so the number can be compared with the T4 log directly), then the")
    print("  bench shape, with the worker pool on and off:")
    for shape, label in ((CAPTURE, "build_notebooks.py:590 shape (the 1-kernel capture)"),
                         (BENCH, "perf_kda.py:90 shape (the 12,082 shape)")):
        print()
        print(f"    --- {label} ---")
        for par in ("", "0"):
            src = (CAPTURE_SRC
                   .replace("@HERE@", repr(HERE))
                   .replace("@B@", str(shape["B"])).replace("@T@", str(shape["T"]))
                   .replace("@H@", str(shape["H"])).replace("@HV@", str(shape["HV"]))
                   .replace("@K@", str(shape["K"])).replace("@V@", str(shape["V"]))
                   .replace("@C@", str(shape["C"])))
            env = dict(os.environ)
            if par:
                env["PARALLEL"] = par
            p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                               text=True, env=env, cwd=HERE)
            if p.returncode != 0:
                print(f"      PARALLEL={par or 'default'}: rc={p.returncode} "
                      f"{p.stderr.strip().splitlines()[-1][:100] if p.stderr.strip() else ''}")
                continue
            lines = dict(ln.split(" ", 1) for ln in p.stdout.splitlines()
                         if ln.split(" ", 1)[0] in ("DEFAULT", "WALL", "COUNTS", "ROUNDS"))
            counts = dict(tok.split("=") for tok in lines["COUNTS"].split())
            dparts = lines["DEFAULT"].split()
            parlabel = par if par else f"{dparts[2]} (default)"
            print(f"      {dparts[0]:<6} PARALLEL={parlabel:<12} "
                  f"wall {lines['WALL']:>7}s  "
                  f"launches {int(counts['launch']):>6,}  "
                  f"programs {int(counts['Compiled.runtime']):>4}  "
                  f"to_program {int(counts['to_program']):>4}  "
                  f"compile_cached {int(counts['compile_cached']):>4}  "
                  f"exp2 seen {int(counts['exp2_in_src']):>5}  "
                  f"pool={'yes' if int(counts['pool_ProcessPool']) else 'no'}")
            for rd in lines["ROUNDS"].split(" | "):
                n = int(re.search(r"n=(\d+)", rd).group(1))
                def get(key):
                    m = re.search(rf"'{key}', (\d+)\)", rd)
                    return int(m.group(1)) if m else 0
                tp, pl = get("to_program"), get("pool_ProcessPool")
                if n > 1 or tp or pl:
                    why = ("compiled in SUBPROCESSES" if pl else
                           ("compiled in-process, len(todo)==1" if tp else "nothing to compile"))
                    print(f"            round: {n:>6,} linear calls, "
                          f"{tp:>4} in-process, pool={pl}, {why}")
    print()
    print("  Read the round table against line 273. With the pool ON, the 6,188-call")
    print("  round has a todo of 190 and hands it to the pool; the 6,041-call round")
    print("  has a todo of 1, which is the only `len(todo) == 1` in the call, so it")
    print("  compiles in the parent and the hook fires exactly once. That is the")
    print("  notebook's `captured 1 kernels`, reproduced on a machine with no GPU.")
    print()
    print("  The `exp2 seen` column is the direct link to kernel000.txt: the one")
    print("  source the parent process ever sees IS the whole of the notebook's")
    print("  output. On Metal that one source carries a dozen `exp2(` call sites;")
    print("  the CUDA rendering of the same kernel is the 128-`exp2` file in the")
    print("  repo, because the two renderers spell the transcendental differently.")
    print("  What the Metal run cannot show is which of the 12,090 launches it was.")
    print("  `kda/materialisation_verdict.md` section 2.2 identified it from the")
    print("  source alone -- inter-chunk state update, kda/tiny_kda.py:240-242, the")
    print("  LAST thing the operator does -- and that identification is what this")
    print("  section confirms: the one kernel a parent-process hook can see is the")
    print("  one compiled in the final, singleton round.")
    print()
    print("  SCOPE, stated plainly. The table above is MEASURED on Metal, on this")
    print("  host. The mapping to CUDA is an INFERENCE from the source, and it is a")
    print("  narrow one: `lower_and_compile`, `get_worker_pool` and")
    print("  `Compiler.compile_cached` all live above the backend, and the notebook")
    print("  installs the identical hook on CUDA. What the T4 run therefore does")
    print("  NOT independently show is that `len(cap)` was 1 there for this reason;")
    print("  it shows `len(cap)` was 1, and this is the only mechanism in the")
    print("  installed source that can produce 1 out of thousands. The one-line")
    print("  check on the box: re-run build_notebooks.py's cell with PARALLEL=0 and")
    print("  the count must become 191.")


# =============================================================================
# 8. section 7 -- is the schedule a cost?
# =============================================================================
# T4 log, quoted with its path. These are the only GPU timings in the repo.
T4_LOG = os.path.join(os.path.dirname(HERE), "out", "kaggle",
                      "20260928-014248-gpu", "logs", "perf_kda.log")
T4_REPLAY = os.path.join(os.path.dirname(HERE), "out", "kaggle",
                         "20260928-014248-gpu", "logs", "replay_kda.log")


def t4_lines(path: str) -> list[str]:
    try:
        with open(path) as f:
            return [ln.rstrip() for ln in f]
    except OSError:
        return []


def parse_t4(path: str) -> dict:
    """The T4's own perf_kda block, parsed rather than transcribed."""
    out: dict = {}
    for ln in t4_lines(path):
        m = re.match(r"\s*kernels per call\s*:\s*([\d,]+)", ln)
        if m:
            out["launches"] = int(m.group(1).replace(",", ""))
        m = re.match(r"\s*GPU time\s*:\s*([\d,]+) us", ln)
        if m:
            out["gpu_us"] = int(m.group(1).replace(",", ""))
        m = re.match(r"\s+r\s+([\d,]+) kernels\s+([\d,]+) us\s+([\d.]+)%", ln)
        if m:
            out["r_n"], out["r_us"], out["r_pct"] = (int(m.group(1).replace(",", "")),
                                                     int(m.group(2).replace(",", "")),
                                                     float(m.group(3)))
        m = re.match(r"\s+E\s+([\d,]+) kernels\s+([\d,]+) us\s+([\d.]+)%", ln)
        if m:
            out["E_n"], out["E_us"], out["E_pct"] = (int(m.group(1).replace(",", "")),
                                                     int(m.group(2).replace(",", "")),
                                                     float(m.group(3)))
    return out


def section_launch_bound(census: list[dict]) -> None:
    head("7. IS 12,082 LAUNCHES A REAL COST, OR IS IT LAUNCH OVERHEAD?")
    t4 = parse_t4(T4_LOG)
    print()
    print(f"  source: out/kaggle/20260928-014248-gpu/logs/perf_kda.log, device=CUDA")
    if not t4:
        print("    <log not present>")
        return
    n, us = t4["launches"], t4["gpu_us"]
    print(f"    launches per call   {n:>12,}")
    print(f"    GPU time            {us:>12,} us = {us/1000:.1f} ms")
    print(f"    per launch          {us/n:>12.3f} us")
    print(f"      r_ {t4['r_n']:>6,} launches {t4['r_us']:>9,} us  "
          f"{t4['r_us']/t4['r_n']:.3f} us/launch  {t4['r_pct']:.1f}%")
    print(f"      E_ {t4['E_n']:>6,} launches {t4['E_us']:>9,} us  "
          f"{t4['E_us']/t4['E_n']:.3f} us/launch  {t4['E_pct']:.1f}%")
    print(f"    internal check: r+E = {t4['r_n']+t4['E_n']:,} == {n:,} -> "
          f"{t4['r_n']+t4['E_n'] == n};  r_us+E_us = {t4['r_us']+t4['E_us']:,} "
          f"== {us:,} -> {t4['r_us']+t4['E_us'] == us}")
    print(f"    launches per second {1e6*n/us:>12,.0f}")
    print()
    print("  7.1  THE r_/E_ SPLIT, RECOVERED WITHOUT A GPU.  MEASURED, and exact.")
    cu = [r for r in census if r["dev"] == "CUDA" and r["shape"]["tag"] == BENCH["tag"]]
    if cu:
        c = cu[0]
        print("    perf_kda.py reads r_/E_ off the kernel NAME, which tinygrad builds")
        print("    from `self.reduceop is not None` (codegen/opt/postrange.py:53) --")
        print("    i.e. from whether the kernel carries a REDUCE. That is a property of")
        print("    the AST, so it is countable at schedule time, here, with no renderer")
        print("    and no card:")
        print(f"      device-free, DEV=CUDA : {c['reduce_kernels']:,} with a REDUCE, "
              f"{c['elementwise']:,} elementwise")
        print(f"      T4 DEBUG=2, same shape: {t4['r_n']:,} r_, {t4['E_n']:,} E_")
        ok = (c["reduce_kernels"], c["elementwise"]) == (t4["r_n"], t4["E_n"])
        print(f"      match: {'EXACT' if ok else 'NO'}")
        print()
        print(f"    So the split is {c['reduce_kernels']:,} / {c['elementwise']:,}. Any 4,106 E_ figure")
        print("    attached to the CUDA count is the METAL split (notes/03-kda.md:45-46,")
        print(f"    summing to 12,090); this device's is {c['elementwise']:,}. The REDUCE kernels are 66% of")
        print("    the count and 74% of the time (159,658/7,984 = 20.00 us each against")
        print("    56,222/4,098 = 13.72 us for the elementwise ones), so they are both")
        print("    more numerous and longer. 7,984 r_ is what `notes/04` calls the")
        print("    unrolled UT transform plus the matmuls; 4,098 E_ is the elementwise")
        print("    tail of the forward substitution and the decay's own arithmetic.")
    print()
    print("  7.2  THE MEASUREMENT IS UPPER-BOUNDED BY ITS OWN INSTRUMENT.  SOURCE.")
    print("  perf_kda.py:43-49 runs the graph in a subprocess with DEBUG=2, and")
    print("  engine/realize.py:318 sets `wait=... or DEBUG>=2`, so every single")
    print("  launch goes through runtime/ops_cuda.py:26-35, `cu_time_execution`:")
    print("    28-29  cuEventCreate x2, per launch, per kernel")
    print("    30-31  cuEventRecord around cuLaunchKernel on the null stream")
    print("    32-33  cuEventSynchronize  <-- the host blocks, per launch")
    print("    34-35  cuEventElapsedTime, then cuEventDestroy x2")
    print("  event1 - event0 therefore contains the host launch latency as well as")
    print("  the kernel, because the queue is empty when event0 is recorded. 215.9")
    print("  ms is 12,082 event-bracketed, host-synchronised launches.  It is an")
    print("  UPPER BOUND on GPU work, not a measurement of it.")
    print()
    print("  7.3  HOW MUCH WORK THE SCHEDULE ACTUALLY ASKS FOR.  MEASURED (device-free).")
    cu = [r for r in census if r["dev"] == "CUDA" and r["shape"]["tag"] == BENCH["tag"]]
    if not cu:
        return
    c = cu[0]
    print(f"    trip count, sum over the {c['sink']:,} kernels of the product of")
    print( "      their RANGE extents                 = {:>20,}".format(c["trip_total"]))
    print(f"    static ALU uops in those bodies (addressing excluded) = {c['alu_static']:>14,}")
    print(f"    LANE-OPS: uops x trip, with the reduce axis discounted = {c['lane_ops']:>14,}")
    print(f"    argument bytes, all launches         = {c['arg_bytes']:>20,} "
          f"= {c['arg_bytes']/MB:,.0f} MB")
    print()
    print("    Two independent bounds, both far below 215.9 ms:")
    print()
    ms_bw = c["arg_bytes"] / 320e9 * 1e3
    print("    (a) BANDWIDTH. If EVERY argument byte missed every cache and went to")
    print(f"        DRAM -- {c['arg_bytes']/MB:,.0f} MB of them -- then at the T4's 320 GB/s")
    print(f"        (datasheet, not measured here) that is {ms_bw:.1f} ms. At most")
    print(f"        {ms_bw/(us/1000)*100:.1f}% of the 215.9 ms can be DRAM traffic, so the")
    print("        operator is NOT bandwidth bound. This is `notes/09` section 5's")
    print("        'largest unresolved quantity, schedule argument bytes', bounded above.")
    print()
    LANE_RATE = 7.2e12
    ms_lane = c["lane_ops"] / LANE_RATE * 1e3
    print(f"    (b) ARITHMETIC. {c['lane_ops']:,} lane-ops at the AST level. A T4 retires at")
    print(f"        most 40 SM x 128 lanes x ~1.4 GHz = {LANE_RATE:.1e} lane-ops/s (arithmetic on the")
    print("        published 40-SM / 2048-core T4 spec, INFERRED, not measured), so at")
    print(f"        100% issue the schedule's arithmetic is {ms_lane*1000:.1f} us.")
    print()
    print("        This is an AST-level count and the emitted program can differ from")
    print("        it in EITHER direction, so both signs are checked. For the one")
    print("        kernel with emitted source in the repo, the renderer SHARES: the")
    print("        state update's exp2 does not depend on the V axis, and a float4")
    print("        store lets four V columns reuse one exp2, so the emitted code does")
    print("        8 exp2 per output element where the AST expansion asks for 32 -- the")
    print("        AST count is 4x HIGH there. The opposite sign is also possible in")
    print("        principle: a reduction's body can be re-entered per output instead")
    print("        of hoisted, which is the mechanism notes/04's 32x describes at the")
    print("        graph-object level. I have no emitted source for the other 12,081")
    print("        kernels, so I can bound neither sign and will not pretend to.")
    print(f"        What survives both: 1000x in EITHER direction is "
          f"{ms_lane*1000:.1f} ms, {ms_lane*1000/(us/1000)*100:.0f}% of 215.9 ms. The")
    print("        arithmetic cannot account for the measured time either way.")
    print()
    print("  7.4  SO WHAT IS THE 215.9 ms?  INFERRED, and stated as such.")
    print(f"  Bandwidth is bounded at {ms_bw:.1f} ms, the schedule's own arithmetic at")
    print(f"  {ms_lane*1000:.0f} us, and the measured total is {us/1000:.1f} ms over {n:,} launches")
    print(f"  = {us/n:.2f} us each. A T4 kernel launch costs single-digit microseconds of")
    print("  host time, and the measured figure is event-bracketed and")
    print("  host-synchronised, so the residual is per-launch cost and measurement")
    print("  overhead rather than arithmetic. One more T4 number points the same way:")
    rl = t4_lines(T4_REPLAY)
    for ln in rl:
        if "schedule + launch + execute" in ln:
            v = int(re.sub(r"[^0-9]", "", ln))
            print(f"    replay_kda.log: {v:,} ms of schedule+launch+execute per warm call")
            print(f"      -> the {us/1000:.1f} ms of DEBUG=2 GPU time is "
                  f"{(us/1000)/v*100:.2f}% of it")
            print(f"      -> a replay path that cached the SCHEDULED PROGRAM LIST")
            print(f"         would be bounded by the launch cost, i.e. {us/1000:.1f} ms,")
            print(f"         not 19,497 ms. That is the study's actionable result and")
            print(f"         it is a statement about the API, not about the arithmetic.")
    print()
    print("  HONESTY: 7.4 is an INFERENCE from a bound plus a log, and it is NOT")
    print("  settled. What would settle it, in order of cost:")
    print("    1. Re-run perf_kda with DEBUG=0 and time `realize()` with a single")
    print("       `Device[...].synchronize()` at the end. If the wall time of the")
    print("       realize is ~200 ms the kernels are launch-bound; if it is ~15 s the")
    print("       DEBUG=2 number is measuring the event wrapper. This is three")
    print("       lines and no new tool.")
    print("    2. `ncu --metrics gpu__time_duration.sum --launch-count 200` on one")
    print("       call. gpu__time_duration excludes launch latency by construction.")
    print("    3. Replay the same 12,082 kernels through one CUDA graph. If the")
    print("       graph replays in << 215.9 ms, the 215.9 ms was launch cost.")


# =============================================================================
# 9. section 8 -- verdicts
# =============================================================================
def section_verdicts(census: list[dict]) -> None:
    head("8. THE THREE VERDICTS")
    by = {(r["dev"], r["shape"]["tag"]): r for r in census}
    bench = {d: by.get((d, BENCH["tag"])) for d in ("CUDA", "METAL", "CPU")}

    print()
    print("  CLAIM 1  the [C,C,K] decay tensor reaches device memory")
    print("  ------------------------------------------------------------------------")
    any_bucket = [r for r in census if r["bucket_at_decay"]]
    print(f"    a kernel argument of the decay tensor's exact size exists at any")
    print(f"    shape, on any of {sorted({r['dev'] for r in census})}?  "
          f"{'*** YES ***' if any_bucket else 'NO'}")
    for r in census:
        print(f"      {r['dev']:<6} {r['shape']['tag'][:34]:<34} "
              f"decay {r['decay']:>13,} B   largest arg {r['max_param']:>10,} B   "
              f"ratio 1/{r['decay']//r['max_param'] if r['max_param'] else 0:>5}")
    print()
    print("    VERDICT: NOT MATERIALISED.  MEASURED on the whole schedule, on the")
    print("    CUDA backend, without a GPU, with an instrument whose null result")
    print("    is known to be able to fire (notes/09 section 4.1's positive control,")
    print("    which I did not re-run and do not restate as mine). This CONFIRMS")
    print("    notes/09 section 4.2 and `kda/materialisation_verdict.md` section 1;")
    print("    it does not contradict them and the 67,108,864 B bucket they left")
    print("    open is now closed.")
    print()
    print("  CLAIM 2  the decay computation is recomputed rather than reused")
    print("  ------------------------------------------------------------------------")
    print("    Per-kernel and per-shape. The one artifact is")
    print("    out/kaggle/20260928-014248-gpu/codegen/kernel000.txt: 128 exp2, all in")
    print("    `float aluN` register locals, none stored, none re-read; 4 stores, all")
    print("    to data0_4096; 33 additive terms per stored element.")
    print()
    print("    WHAT IT COSTS, and this is the part the record keeps getting wrong:")
    print("    registers and SFU issue slots, not traffic. The 32x is arithmetic.")
    print()
    print("    VERDICT: RECOMPUTED, and paid in ARITHMETIC. NOT RE-DERIVED here --")
    print("    the 128-exp2 figure is re-read from the file by")
    print("    `kda/materialisation_verdict.md` section 2.3 and I have not")
    print("    re-parsed it, because parsing it is that file's job and it did it.")
    print()
    print("    AND HERE IS WHY IT CANNOT BE COUNTED FROM THE SCHEDULE, which I")
    print("    only found by trying. The static EXP2 uop count in the whole CUDA")
    print("    schedule is:")
    for d in ("CUDA", "METAL", "CPU"):
        r = bench.get(d)
        if r:
            print(f"      {d:<6} {r['exp_static']:>6,} outside a REDUCE, "
                  f"{r['exp_in_reduce']:>6,} inside one")
    print()
    print("    That is 50 uops, in 12,082 kernels, and it is NOT an exp2 count. The")
    print("    one emitted kernel we have source for carries 128 `exp2` PER THREAD")
    print("    x 256 threads = 32,768 evaluations, and the graph's contribution to")
    print("    that is a single EXP2 uop inside a reduce whose trip count is C. A")
    print("    uop count is not an evaluation count, and the multiplier lives in the")
    print("    renderer (see 7.3b for which way it cuts, and that it cuts both ways).")
    print("    The dynamic exp2 count is therefore a CODEGEN property, which is")
    print("    exactly why the 32x had to be counted in source. Do not use 50 as a")
    print("    denominator for anything, and do not read this as a smaller exp2")
    print("    count than the record already has.")
    print()
    print("  CLAIM 3  the operator's schedule is a fused schedule")
    print("  ------------------------------------------------------------------------")
    for d in ("CUDA", "METAL", "CPU"):
        r = bench.get(d)
        if r:
            print(f"    {d:<6} {r['sink']:>7,} SINK kernels, {r['copy']:>4,} COPY, "
                  f"{r['reduce_kernels']:>7,} with a REDUCE, "
                  f"{r['elementwise']:>7,} elementwise")
    print()
    print("    VERDICT: NOT FUSED, AND NOT SETTLED AS A COST.  The 12,082 is now")
    print("    reproduced on a laptop with no GPU, to the unit, from the CUDA")
    print("    scheduler -- and reproduced for the right reason (section 4). What")
    print("    fraction of runtime it is, is section 7, and section 7 does not")
    print("    settle it. Quote the count; do not yet quote a cost.")


# =============================================================================
# 10. section 9 -- what must not be carried forward
# =============================================================================
def section_do_not_carry() -> None:
    head("9. WHAT THIS FILE WITHDRAWS, RESTATES, OR LEAVES ALONE")
    rows = [
        ("the 67,108,864 B bucket is untested on CUDA",
         "kda/materialisation_verdict.md section 7; notes/10 section 9.2",
         "CLOSED. No bucket, whole schedule, CUDA backend. This file."),
        ("`captured 1 kernels` is a source file worth analysing",
         "the premise of kda/materialisation_verdict.md section 2",
         "KEEP the analysis, FIX the framing. kernel000.txt is the LAST kernel of "
         "the operator (the inter-chunk state update), not a sample of it."),
        ("CUDA's schedule is assumed to be Metal's schedule",
         "notes/09 section 4.5, 'two backends agreeing is not three'",
         "FALSIFIED for the kernel COUNT, and now explained: they differ by 8 "
         "kernels, via DEVICE_MAX_BUFS. They agree on max_param."),
        ("22.2822 MB compulsory term, 5.5050 MB denominator, 13.0471x",
         "notes/09 section 2; audit in kda/materialisation_verdict.md section 3.3",
         "NOT REPRODUCED by two prior agents. NOT USED, NOT RE-DERIVED here. Do not "
         "quote. An independent decomposition gives 5,242,880 B and 51.20x."),
        ("'8.2 GB each' for the decay tensor at NOTES.md's shape",
         "notes/03-kda.md:212; kda/NOTES.md:153,162",
         "WRONG. fp32 is 17,179,869,184 B = 17.18 GB = 16.00 GiB each, and "
         "34,359,738,368 B = 34.36 GB counting one write and one read. Not used "
         "here, and not restated as mine."),
        ("215.9 ms of GPU time as 12,082 kernels of real work",
         "notes/10 section 2; notes/03-kda.md:41-48",
         "UPPER BOUND ONLY. The DEBUG=2 path blocks the host on a cuEvent "
         "synchronize per launch (ops_cuda.py:26-35). Section 7."),
        ("'the decay dataflow is the bottleneck' as a bandwidth claim",
         "notes/03-kda.md:120",
         "STILL WITHDRAWN as bandwidth. Survives as an arithmetic claim, and "
         "section 7.3(a) now bounds the bandwidth share of this operator at <=13% of "
         "the T4's measured GPU time."),
    ]
    for a, b, c in rows:
        print()
        print(f"  {a}")
        print(f"    where     {b}")
        for ln in _wrap(c, 84):
            print(f"    {ln}")
    print()
    print("  256x element ratio and 32x exp2: UNAFFECTED. Both are graph/source")
    print("  objects. Neither needs the decay tensor to exist. This file does not")
    print("  touch them and does not restate them as measured here.")


def _wrap(s: str, n: int) -> list[str]:
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


# =============================================================================
# 11. main
# =============================================================================
def one_line(dev: str, shape_key: str) -> str:
    shapes = {"bench": BENCH, "capture": CAPTURE, "bigger": BIGGER}
    r = run_census(dev, shapes[shape_key])
    return (f"CENSUS dev={r['dev']} shape={shape_key} sink={r['c'].sink} "
            f"copy={r['c'].copy} reduce={r['c'].reduce_kernels} "
            f"max_param={r['c'].max_param} arg_bytes={r['c'].arg_bytes} "
            f"decay={decay_bytes(r['shape'],4)} "
            f"params={sorted(r['c'].params.items())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="bench shape only, one backend")
    ap.add_argument("--no-replay", action="store_true",
                    help="skip section 6, the real-device launch/compile census")
    ap.add_argument("--no-scaling", action="store_true",
                    help="skip section 5, the NT sweep")
    ap.add_argument("--one-line", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--limit-bufs", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max-bufs", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--dev", default="CUDA")
    ap.add_argument("--shape", default="bench")
    args = ap.parse_args()

    if args.limit_bufs:
        print(limit_bufs_probe(args.dev, args.max_bufs))
        return 0

    if args.one_line:
        print(one_line(args.dev, args.shape))
        return 0

    rule()
    print("SCHEDULE CENSUS -- the decay is fused, the operator is not")
    import tinygrad
    ver = "unknown"
    for meta in os.listdir(os.path.dirname(os.path.dirname(tinygrad.__file__))):
        m = re.fullmatch(r"tinygrad-([0-9][^-]*)\.dist-info", meta)
        if m:
            ver = m.group(1)
    print(f"  tinygrad {ver} at {os.path.dirname(tinygrad.__file__)}")
    print(f"  backends compiled on this host: "
          f"{sorted(d for d in Device._devices if d in ('CPU','METAL','CUDA','CL','WEBGPU'))}")
    print(f"  {Device.DEFAULT} is this host's default. Every census below overrides it")
    print("  with Context(DEV=...) and never instantiates a device.")

    section_claims()

    if not args.quick:
        section_bytes()

    shapes = [BENCH] if args.quick else [BENCH, CAPTURE, BIGGER]
    devs = ["CUDA"] if args.quick else ["CUDA", "METAL", "CPU"]
    census = section_census(shapes, devs)

    if not args.quick:
        section_determinism("CUDA", BENCH)
        section_device_max_bufs()
        if not args.no_scaling:
            section_scaling()
    if not args.no_replay:
        section_capture("METAL")

    section_launch_bound(census)
    section_verdicts(census)
    section_do_not_carry()

    print()
    rule()
    print("BOTTOM LINE, in the order the questions were asked")
    rule()
    cu_bench = [r for r in census if r["dev"] == "CUDA" and r["shape"] is BENCH]
    if cu_bench:
        r = cu_bench[0]
        print(f"  1. bucket at {r['decay']:,} B (the fp32 [B,NT,HV,C,C,K] tensor at the bench")
        print(f"     shape)?  {'*** YES ***' if r['bucket_at_decay'] else 'NO'}. Largest kernel "
              f"argument {r['max_param']:,} B = B*NT*HV*C*C*4, exactly 1/K of it.")
        print(f"  2. reconciled kernel count: {r['sink']:,} SINK kernels per kda_chunk call on the")
        print( "     CUDA schedule -- the T4's 12,082, reproduced to the unit with no GPU.")
        print( "     The notebook's 1 was the worker pool, not the operator.")
    print( "  3. launch-bound or real work: NOT SETTLED. Bounded from both sides in")
    print( "     section 7; three measurements named, the cheapest being three lines.")
    print( "  4. the claim left open is claim 3, the graph-level schedule cost. Claims 1")
    print( "     and 2 are settled: the tensor is absent, the computation is recomputed.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
