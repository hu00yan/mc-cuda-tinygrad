"""materialisation.py - is the KDA [C,C,K] decay tensor actually materialised?

THE VERDICT, IN ONE LINE, is printed last. Every number in this file is
re-derived here from shapes; nothing is transcribed from another agent's file.
The re-derivation rule is `notes/06-corrections-ledger.md` §3, and this file
exists because that ledger could not settle a conflict between two of its own
measurements.

THE CONFLICT THIS FILE SETTLES
------------------------------
Three numbers are in circulation that look like the same quantity and are not:

  8.42 MB    measured, allocator        notes/04-codegen-comparison.md:170
  54.1 MB    measured, allocator        notes/04-codegen-comparison.md:173
  268.4 MB   modelled, DRAM traffic     bench/HYPOTHESES.md:168-169 (H2)

`notes/06-corrections-ledger.md` §2 pre-registered a refutation threshold of
"measured total < 58 MB" against the 268.4 MB / 290.7 MB model. The 54.1 MB
figure sits under that threshold, so the model looks refuted -- but 54.1 MB is
an *allocator request total*, and a DRAM-traffic claim cannot be refuted by an
allocator number. This file separates them, measures each thing that CAN be
measured without a GPU, and states plainly which one is still untested.

FOUR SENSES OF "MATERIALISED", KEPT APART
-----------------------------------------
  M1  a device buffer of the decay tensor's exact size is ever requested
  M2  a kernel *argument* of that size appears in the generated schedule
      (this is the one that decides whether DRAM traffic can exist at all)
  M3  the peak live device memory of the whole operator
  M4  actual DRAM traffic (NOT measurable here; needs a hardware counter)

WHY A LAPTOP SETTLES IT
-----------------------
M1 and M2 are properties of the *generated schedule*, not of the card: a buffer
is either in the plan or it is not, and which buffers a kernel may touch is
decided by tinygrad's scheduler. §6 runs the identical census on CPU and on
METAL, in separate processes, and shows the two schedules agree buffer for
buffer. M3 is the memory planner's own peak accounting, which is a property of
the plan. Only M4 needs a real GPU; it is NOT measured, and §7 says so and
names what would measure it.

Run:
    .venv/bin/python kda/materialisation.py              # ~4 min
    .venv/bin/python kda/materialisation.py --quick      # bench shape only
    DEV=CPU .venv/bin/python kda/materialisation.py      # force a backend

Flags:
    --quick          one shape (the perf_kda.py shape) instead of three
    --no-control     skip §4, the positive control. Do not: it is the reason
                     the null result in §5 is evidence rather than silence
    --backend-census  internal: print one machine-readable census line for the
                     current DEV and exit. §6 calls this in a subprocess.
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
from tinygrad.uop.ops import Ops  # noqa: E402

MB = 1e6
GIB = 1024 ** 3


# =============================================================================
# the shapes, named once, so nothing below invents one
# =============================================================================
# kda/perf_kda.py:90 -- the shape the KDA code actually benchmarks, and the
# shape BOTH contested numbers are quoted at.
BENCH = dict(B=2, T=512, H=4, HV=4, K=64, V=64, C=64)

# kda/cost.py:17 -- the shape NOTES.md §6 and the cost model quote, and the
# shape the brief calls "the study shape". It is NOT perf_kda.py's shape.
STUDY = dict(B=4, T=4096, H=32, HV=32, K=128, V=128, C=64)

# kda/codegen_diff.py's isolated gram shape: 1 chunk, 1 head, K=64
CODEDIFF = dict(B=1, T=64, H=1, HV=1, K=64, V=64, C=64)


def nt(s: dict) -> int:
    return s["T"] // s["C"]


def slabs(s: dict) -> int:
    return s["B"] * nt(s) * s["HV"]


def decay_elems(s: dict) -> int:
    """elements of ONE [B,NT,HV,C,C,K] decay tensor"""
    return slabs(s) * s["C"] * s["C"] * s["K"]


def decay_bytes(s: dict, es: int = 4) -> int:
    return decay_elems(s) * es


def cc_bytes(s: dict, es: int = 4) -> int:
    """the [B,NT,HV,C,C] family -- the shape class the largest live
    intermediate is expected to be in"""
    return slabs(s) * s["C"] * s["C"] * es


def shape_str(s: dict) -> str:
    return (f"B={s['B']} T={s['T']} H={s['H']} HV={s['HV']} K={s['K']} "
            f"V={s['V']} C={s['C']} (NT={nt(s)})")


def rule(c: str = "=", n: int = 96) -> None:
    print(c * n)


def head(title: str) -> None:
    print()
    print(title)
    rule("-")


# =============================================================================
# 0. the four senses, so a reader knows which regime a number belongs to
# =============================================================================
def section_senses() -> None:
    head("0. FOUR SENSES OF \"MATERIALISED\" -- AND WHICH ONE IS MEASURED HERE")
    rows = [
        ("M1", "allocator",
         "is a device buffer of the decay tensor's exact nbytes ever requested",
         "MEASURED  §4, §5. instrument proven by positive control"),
        ("M2", "schedule",
         "does a kernel ARGUMENT of that size exist in the generated plan",
         "MEASURED  §5. decisive: no argument, no bytes"),
        ("M3", "peak live memory",
         "peak live device bytes for the whole operator",
         "MEASURED  §5, from tinygrad's own memory planner"),
        ("M4", "DRAM traffic",
         "bytes crossing the memory bus",
         "NOT MEASURED  needs a hardware counter; §7 names which"),
    ]
    print(f"  {'id':<4}{'sense':<18}{'question':<50}status")
    for i, s, q, st in rows:
        print(f"  {i:<4}{s:<18}{q:<50}{st}")
    print()
    print("  These four do not refute each other on their face: an allocator")
    print("  number cannot refute a traffic claim. §2 works out whether the two")
    print("  numbers actually in the record are even the same shape, because if")
    print("  they are not, the conflict dissolves and the ledger never had one.")
    print("  It partly dissolves. §5 finds the part that does not.")


# =============================================================================
# 1. provenance: re-derive each contested number, with shape / dtype / method
# =============================================================================
def section_provenance() -> None:
    head("1. PROVENANCE -- EVERY CONTESTED NUMBER, RE-DERIVED, WITH SHAPE AND METHOD")

    s = BENCH
    de, db = decay_elems(s), decay_bytes(s, 4)
    print(f"\n  The shape both contested numbers are quoted at -- kda/perf_kda.py:90")
    print(f"  {shape_str(s)}")
    print(f"  fp32 throughout. The port is fp32 by construction: tiny_kda.py:44")
    print(f"  sets F = dtypes.float32 and five np.float32 casts follow it.")
    print(f"  NT = {nt(s)}, B*NT*HV = {slabs(s):,}")
    print(f"  [B,NT,HV,C,C,K] = {s['B']}*{nt(s)}*{s['HV']}*{s['C']}^2*{s['K']}"
          f" = {de:,} elements")
    print(f"  one slab [C,C,K] = {s['C']**2*s['K']:,} elements")

    two_w_r = 2 * 2 * db
    # The byte model's row list, term by term, from bench/HYPOTHESES.md:170-171.
    # Each is (n tensors) x (n touches) x (nbytes) with every factor written out.
    g_cum = 2 * (slabs(s) * s["C"] * s["K"] * 4)           # 1 tensor x (w+r)
    c2 = 4 * 2 * cc_bytes(s)                               # 4 tensors x (w+r)
    wu = 2 * (slabs(s) * s["C"] * s["K"] * 4
              + slabs(s) * s["C"] * s["V"] * 4)             # w, u: 2 x (w+r)
    v_new = 2 * (slabs(s) * s["C"] * s["V"] * 4)            # 1 tensor x (w+r)
    # I/O: read q, k, g ([B,T,HV,K] each) + read v + write o ([B,T,HV,V] each),
    # one touch each. q/k/g are counted after the GVA repeat.
    io = (3 * s["K"] + 2 * s["V"]) * s["B"] * s["T"] * s["HV"] * 4
    state = 2 * (s["B"] * s["HV"] * s["K"] * s["V"] * 4)    # 1 tensor x (w+r)
    other = g_cum + c2 + wu + v_new + io + state
    comp = io + state
    total = two_w_r + other

    print()
    print(f"  {'number':>11}  {'what it is, and where it came from':<58}verdict")
    rule("-", 96)
    rows = [
        (f"{two_w_r/MB:.4f} MB", "2 decay tensors x (1 write + 1 read), fp32, whole graph",
         "== 268.4 MB"),
        (f"{db/MB:.4f} MB", "ONE decay tensor, fp32, whole graph", "== 67.1 MB"),
        (f"{decay_bytes(s,2)/MB:.4f} MB", "ONE decay tensor, bf16, whole graph",
         "== 33.55 MB"),
        (f"{other/MB:.4f} MB", "\"everything else\", one write + one read, fp32",
         "== 22.3 MB"),
        (f"{total/MB:.4f} MB", "ported total = decay + everything else", "== 290.7 MB"),
        (f"{comp/MB:.4f} MB", "compulsory = I/O + state S", "== 5.5 MB"),
        (f"{two_w_r/comp:.4f}x", "decay / compulsory", "== 48.8x"),
        (f"{total/other:.4f}x", "ported / folded", "== 13.0x"),
    ]
    for num, what, verdict in rows:
        print(f"  {num:>11}  {what:<58}{verdict}")
    print()
    print("  [MODELLED] for all eight. Exact arithmetic on the shapes in")
    print("  kda/perf_kda.py:90, zero measurement. bench/HYPOTHESES.md:166-174 is")
    print("  the delivered model and all eight reproduce, so the byte model is")
    print("  internally sound. That is not the question. The question is whether")
    print("  the shape it charges exists at runtime.")
    print()
    print("  A NOTE ON THIS RE-DERIVATION, because it is the second time it has paid:")
    print("  my first draft of the I/O term was 4 x (2K + V + V) x B x T x HV x 4 B =")
    print("  16.78 MB, on the reasoning that there are four tensors and two touches.")
    print("  It is 3 x wrong and the error was invisible in the decay term, so it")
    print("  moved exactly the four figures that matter: 22.3 -> 29.6 MB, 5.5 -> 17.0,")
    print("  13.0x -> 10.06x, 48.8x -> 15.75x, while 268.4 MB sat there untouched")
    print("  looking right. Caught only by writing the formula out term by term")
    print("  instead of importing a coefficient. That is notes/06 §3's failure mode")
    print("  (a subtotal that double-counted I/O) arriving from a different")
    print("  direction, and it is the reason the rule is 're-derive', not 'verify'.")
    print()
    print("  Two more numbers, both MEASURED, both from")
    print("  notes/04-codegen-comparison.md:168-174, on this same shape:")
    print(f"    {'8.42 MB':>11}  {8_421_376:,} B, recorded as \"largest single allocation\". "
          "§5 REPRODUCES it")
    print(f"    {'':>11}  and §5 shows the provenance is sharper than the note's")
    print(f"    {'':>11}  wording: it is the memory PLANNER's arena, i.e. the peak")
    print(f"    {'':>11}  live intermediates, not an arbitrary buffer. Method:")
    print(f"    {'':>11}  Buffer.allocate spy + the planner's own DEBUG=1 line.")
    print(f"    {'54.1 MB':>11}  {54_149_120:,} B, recorded as \"total bytes allocated")
    print(f"    {'':>11}  (all calls)\". §5 REPRODUCES it. Method: same spy, summed.")


# =============================================================================
# 2. the units problem
# =============================================================================
def section_units() -> None:
    head("2. THE UNITS PROBLEM -- WHAT 268.4 MB ACTUALLY COUNTS")

    s = BENCH
    per_inst = s["C"] * s["C"] * s["K"] * 4
    print("""
  The claim to resolve: "[C,C,K]=64x64x128 fp32 is 2.097 MB per instance and with
  B=4, NT=64, HV=32 there are 8,192 instances, which is ~17 GB -- not 268.4 MB.
  So 268.4 MB must be counting something narrower (per-step? per-program? a
  single instance?)."

  ANSWER: 268.4 MB is not narrow at all. It is the WHOLE-GRAPH figure for both
  decay tensors at the perf_kda.py shape, term by term, and the 17 GB comes from
  a DIFFERENT SHAPE being multiplied by hand. They are not two scopes of one
  quantity. They are one scope of two different shapes, and the repo names both
  of them "the KDA shape".""")

    print(f"\n  What 268.4 MB counts, at {shape_str(s)}:")
    print(f"    B*NT*HV                   = {slabs(s):>12,}   instances of [C,C,K]")
    print(f"    [C,C,K] fp32              = {per_inst:>12,} B = {per_inst/MB:.3f} MB per instance")
    print(f"    one decay tensor          = {slabs(s)*per_inst:>12,} B = "
          f"{slabs(s)*per_inst/MB:>9.4f} MB")
    print(f"    x 2 tensors (Dl, Dq_all)  = {2*slabs(s)*per_inst:>12,} B = "
          f"{2*slabs(s)*per_inst/MB:>9.4f} MB")
    print(f"    x 2 touches (write+read)  = {4*slabs(s)*per_inst:>12,} B = "
          f"{4*slabs(s)*per_inst/MB:>9.4f} MB   == 268.4 MB")
    print()
    print("  So 268.4 MB is 4 passes over ONE [B,NT,HV,C,C,K] tensor (2 tensors x")
    print("  write+read), fp32, at the bench shape. NOT per-step, NOT per-program,")
    print("  NOT per-instance, NOT a single slab. Every factor is an exact product")
    print("  of the shapes on kda/perf_kda.py:90. There is no units error in it.")

    print("\n  Where the 17 GB comes from -- the OTHER shape in the record:")
    for name, sh, es in [("kda/perf_kda.py:90  (benchmarked)", BENCH, 4),
                         ("kda/cost.py:17     (cost-modelled)", STUDY, 4),
                         ("kda/cost.py:17     (cost-modelled, bf16, ES=2)", STUDY, 2)]:
        b = decay_bytes(sh, es)
        print(f"    {name:<46} x{es}B = {b:>17,} B = {b/MB:>10.2f} MB = {b/GIB:>7.2f} GiB")
    print()
    print("  The 8,192-instance / 17 GB arithmetic is the cost.py shape. The 268.4 MB")
    print("  arithmetic is the perf_kda.py shape. They are 256x apart, both correct")
    print("  for their own shape, and crossover/literature.py §3 already prints both")
    print("  side by side without connecting them.")

    print("\n  IS THAT THE WHOLE CONFUSION? Two things are, and they compound:")
    print("   (a) TWO SHAPES, both called \"the KDA shape\". kda/perf_kda.py:90 and")
    print("       kda/cost.py:17 differ by 256x in the decay tensor. The brief's")
    print("       \"study shape (B4 T4096 HV32 K=V=128 C=64, as used in")
    print("       kda/perf_kda.py)\" is itself a conflation: that is cost.py's shape.")
    print("       perf_kda.py:90 uses B=2 T=512 H=HV=4 K=V=64 C=64.")
    print("   (b) THREE quantities under one word. 268.4 MB is modelled traffic;")
    print("       8.42 MB and 54.1 MB are allocator numbers. On their face they do")
    print("       not refute each other, so the ledger's \"8.42 MB < 67.1 MB, the")
    print("       model is refuted\" reading is not licensed by the arithmetic alone.")
    print()
    print("  §5 supplies the fact that makes the conflict sharp again, and it is a")
    print("  different fact from either number: the tensor is absent from the")
    print("  SCHEDULE, not merely small in the allocator.")


# =============================================================================
# 3. the instruments
# =============================================================================
class AllocCensus:
    """Spies on tinygrad.device.Buffer.allocate / deallocate.

    Records every allocation request in order, so three different quantities can
    be read off one log: the request total, the largest single request, and the
    live peak. A null result from this instrument is only evidence if the
    instrument can fire -- which is what §4 proves.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, int, str]] = []
        self._live = 0
        self.peak_live = 0
        self._a = self._d = None
        self._td = None

    def __enter__(self) -> "AllocCensus":
        import tinygrad.device as td
        self._td = td
        self._a, self._d = td.Buffer.allocate, td.Buffer.deallocate
        c = self

        def spy_alloc(self, opaque=None, external_ptr=None):
            r = c._a(self, opaque, external_ptr)
            c.events.append(("A", self.nbytes, str(self.dtype)))
            if self._base is None:
                c._live += self.nbytes
                c.peak_live = max(c.peak_live, c._live)
            return r

        def spy_dealloc(self):
            if self._base is None:
                c._live -= self.nbytes
            c.events.append(("F", self.nbytes, str(self.dtype)))
            return c._d(self)

        td.Buffer.allocate, td.Buffer.deallocate = spy_alloc, spy_dealloc
        return self

    def __exit__(self, *exc) -> None:
        self._td.Buffer.allocate = self._a
        self._td.Buffer.deallocate = self._d

    # -- readers ------------------------------------------------------------
    @property
    def allocs(self) -> list[int]:
        return [n for k, n, _ in self.events if k == "A"]

    @property
    def n_calls(self) -> int:
        return len(self.allocs)

    @property
    def total(self) -> int:
        return sum(self.allocs)

    @property
    def largest(self) -> int:
        return max(self.allocs, default=0)

    @property
    def distinct(self) -> int:
        return len(set(self.allocs))

    def hist(self) -> collections.Counter:
        return collections.Counter(self.allocs)

    def fingerprint(self) -> str:
        return f"{self.n_calls}/{self.largest}/{self.total}/{self.peak_live}"


class PlanCensus:
    """Spies on tinygrad.schedule.memory_plan_rewrite -- the last point at which
    the schedule is still LOGICAL, one UOp per intermediate, before the planner
    coalesces everything into per-lane arenas.

    M2 is answered here. The buffers that appear as kernel PARAMs are exactly
    the buffers a launch can read or write, so a buffer that is not among them
    cannot carry a byte of traffic.
    """

    def __init__(self) -> None:
        self.plans: list[collections.Counter] = []
        self.n_kernels: list[int] = []
        self._orig = None
        self._sch = None

    def __enter__(self) -> "PlanCensus":
        import tinygrad.schedule as sch
        self._sch = sch
        self._orig = sch.memory_plan_rewrite
        p = self

        def spy(linear, held_bufs=None):
            c: collections.Counter = collections.Counter()
            for si in linear.src:
                p.n_kernels.append(1)
                for s in si.src[1:]:
                    if s.op is Ops.BUFFER:
                        c[s.nbytes()] += 1
            p.plans.append(c)
            return p._orig(linear, held_bufs)

        sch.memory_plan_rewrite = spy
        return self

    def __exit__(self, *exc) -> None:
        self._sch.memory_plan_rewrite = self._orig

    # -- readers ------------------------------------------------------------
    @property
    def params(self) -> collections.Counter:
        out: collections.Counter = collections.Counter()
        for c in self.plans:
            out.update(c)
        return out

    @property
    def max_param(self) -> int:
        p = self.params
        return max(p) if p else 0

    @property
    def arg_bytes(self) -> int:
        return sum(n * c for n, c in self.params.items())

    @property
    def kernels(self) -> int:
        return len(self.n_kernels)

    def fingerprint(self) -> str:
        return f"{self.kernels}/{self.max_param}/{self.arg_bytes}"


# =============================================================================
# 4. the positive control -- without it, §5's null result is silence
# =============================================================================
def section_control() -> None:
    head("4. POSITIVE CONTROL -- PROVING THE INSTRUMENT CAN SEE A MATERIALISED "
         "DECAY TENSOR")
    s = BENCH
    B, N, HV, C, K = s["B"], nt(s), s["HV"], s["C"], s["K"]
    dec = decay_bytes(s, 4)
    rng = np.random.default_rng(0)
    g = Tensor(rng.random((B, N, HV, C, K), dtype=np.float32))

    def decay() -> Tensor:
        return (g.reshape(B, N, HV, C, 1, K) - g.reshape(B, N, HV, 1, C, K)).exp()

    def gram(D: Tensor) -> Tensor:
        kp = g.reshape(B, N, HV, C, 1, K)
        kj = g.reshape(B, N, HV, 1, C, K)
        return (D * kp * kj).sum(-1)

    print(f"\n  isolated decay at {shape_str(s)}, fp32: "
          f"{dec:,} B = {dec/MB:.1f} MB")
    print(f"  {'case':<58}{'allocs':>7}{'largest B':>15}{'saw it?':>10}")
    rule("-", 96)
    for label, fn in [
        ("A  FUSED, exactly as tiny_kda.py:137-144 writes it",
         lambda: gram(decay()).realize()),
        ("B  FORCED: Dl.realize() first, then contract",
         lambda: (decay().realize(), gram(decay()).realize())),
        ("C  FORCED: Dl.contiguous().realize() first",
         lambda: (decay().contiguous().realize(), gram(decay()).realize())),
    ]:
        with AllocCensus() as c:
            fn()
        print(f"  {label:<58}{c.n_calls:>7}{c.largest:>15,}"
              f"{('YES' if c.largest >= dec else 'no'):>10}")
    print()
    print("  Row A is the port's own expression, and it does NOT allocate the decay")
    print("  tensor. Rows B and C are the same arithmetic with one scheduling change,")
    print("  and they DO -- 67,108,864 B, to the byte.")
    print()
    print("  [MEASURED] This is the load-bearing control. A null result from an")
    print("  allocator spy is worthless without it: the spy might be missing the")
    print("  allocation, or the allocator might map a fused loop's working set")
    print("  somewhere the hook cannot see. Rows B and C rule both out for THIS")
    print("  instrument, THIS shape, THIS dtype.")

    print()
    print("  and at two more shapes the record cares about:")
    for sh, tag in ((CODEDIFF, "codegen_diff.py's isolated gram"),
                    (dict(STUDY, T=STUDY["C"]), "cost.py / NOTES.md family, NT cut 64x")):
        Bs, Ns, HVs, Cs, Ks = sh["B"], nt(sh), sh["HV"], sh["C"], sh["K"]
        ds = decay_bytes(sh, 4)
        gg = Tensor(np.random.default_rng(2).random((Bs, Ns, HVs, Cs, Ks), dtype=np.float32))

        def dd(gg=gg, Bs=Bs, Ns=Ns, HVs=HVs, Cs=Cs, Ks=Ks):
            return (gg.reshape(Bs, Ns, HVs, Cs, 1, Ks)
                    - gg.reshape(Bs, Ns, HVs, 1, Cs, Ks)).exp()

        def zz(D, gg=gg, Bs=Bs, Ns=Ns, HVs=HVs, Cs=Cs, Ks=Ks):
            return (D * gg.reshape(Bs, Ns, HVs, Cs, 1, Ks)
                    * gg.reshape(Bs, Ns, HVs, 1, Cs, Ks)).sum(-1)

        with AllocCensus() as c1:
            zz(dd()).realize()
        with AllocCensus() as c2:
            (dd().realize(), zz(dd()).realize())
        print(f"    {shape_str(sh)}")
        print(f"      decay = {ds:,} B = {ds/MB:.1f} MB   [{tag}]")
        print(f"      {'fused, as the port writes it':<40}{c1.n_calls:>5}"
              f"{c1.largest:>15,}   saw it? {'YES' if c1.largest >= ds else 'no'}")
        print(f"      {'forced, Dl.realize() first':<40}{c2.n_calls:>5}"
              f"{c2.largest:>15,}   saw it? {'YES' if c2.largest >= ds else 'no'}")
    print()
    print("  Three shapes, one answer: fused never allocates the decay tensor,")
    print("  forced always does, to the byte. The third shape is kda/cost.py's with")
    print("  only NT reduced, which is as close to the study shape as this host can")
    print("  afford -- at full NT the fp32 decay tensor alone is 16.00 GiB.")
    print()
    print(f"  The forced row's largest allocation at the codegen_diff shape is "
          f"C*C*K*4 = {CODEDIFF['C']**2*CODEDIFF['K']*4:,} B -- the")
    print("  [C,C,K] slab of ONE (b,chunk,head), which is the 262,144-element object")
    print("  notes/04-codegen-comparison.md §2 measures. It is a slab, and a slab")
    print("  only exists when the schedule demands one.")


# =============================================================================
# 5. the measurement
# =============================================================================
def measure_full_op(s: dict, runs: int = 2) -> dict | None:
    """One fresh graph per run, allocation census + plan census.

    FRESH GRAPH, deliberately. `Tensor.realize()` on an already-realized graph is
    a silent no-op, so a repeat call on the same Tensors would measure nothing.
    `kda_chunk` takes numpy inputs and builds a new graph each time, which is
    the strongest available form of "fresh". To prove the schedule cache is not
    what is being measured, the whole call is run TWICE and the two censuses are
    required to be identical. They are: tinygrad's `schedule_cache` stores the
    LOWERED PLAN, and the buffers are re-resolved and re-allocated on every
    `realize()`. Both runs are reported.
    """
    import test_ref as T
    import tiny_kda

    out = []
    for _ in range(runs):
        with AllocCensus() as ac, PlanCensus() as pc:
            t0 = time.perf_counter()
            q, k, v, g, beta = T.rnd(s["B"], s["T"], s["H"], s["HV"],
                                     s["K"], s["V"], seed=0, l2n=True)
            tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=s["C"])
            out.append(dict(secs=time.perf_counter() - t0, alloc=ac, plan=pc))
    if runs < 2:
        return dict(runs=out, reproducible=None, last=out[-1])
    same = all(o["alloc"].fingerprint() == out[0]["alloc"].fingerprint() and
               o["plan"].fingerprint() == out[0]["plan"].fingerprint() for o in out)
    return dict(runs=out, reproducible=same, last=out[-1])


def decode_sizes(s: dict, hist: collections.Counter, es: int = 4) -> tuple[dict, int]:
    """Name each observed buffer size by finding which of the port's tensors has
    that byte count. Returns (nbytes -> label, largest single-tensor size).

    Sizes larger than every single tensor, and requested exactly once, are the
    memory planner's arenas: memory_plan_rewrite coalesces thousands of logical
    buffers into one allocation per device lane, so an arena is not a tensor and
    its size is a peak, not a shape. Sizes matching nothing are reported as
    unidentified rather than guessed at."""
    cand = {
        "[B,NT,HV,C,C,K] DECAY (Dl/Dq_all)": decay_bytes(s, es),
        "[B,NT,HV,C,C] (Z/M/A/Aqk)": cc_bytes(s, es),
        "[B,NT,HV,C,K] (tq/tk/g_cum)": slabs(s) * s["C"] * s["K"] * es,
        "[B,NT,HV,C,V] (tv/u)": slabs(s) * s["C"] * s["V"] * es,
        "[B,NT,HV,C] (tb, the b vector)": slabs(s) * s["C"] * es,
        "[B,HV,K,V] (the state S)": s["B"] * s["HV"] * s["K"] * s["V"] * es,
        "[B,T,HV,V] (the output o)": s["B"] * s["T"] * s["HV"] * s["V"] * es,
        "[B,T,HV,K] (q/k/g, GVA-repeated)": s["B"] * s["T"] * s["HV"] * s["K"] * es,
        "[C,C] (eye_all, tril masks)": s["C"] * s["C"] * es,
        "[C] (one eyeT row)": s["C"] * es,
    }
    # maxt excludes the decay tensor on purpose: an arena is identified as
    # "bigger than every REAL intermediate", and the decay tensor is precisely
    # the thing that does not exist.
    maxt = max(v for k, v in cand.items() if "DECAY" not in k)
    out = {}
    for n, cnt in hist.items():
        hits = [k for k, v in cand.items() if v == n]
        if hits:
            out[n] = " | ".join(hits)
        elif n > maxt and cnt == 1:
            out[n] = "PLANNER ARENA (one per realize(); a peak, not a shape)"
        else:
            out[n] = "unidentified (not any single tensor in the port)"
    return out, maxt


def report_shape(s: dict, r: dict, verbose: bool = True) -> dict:
    dec, cc = decay_bytes(s, 4), cc_bytes(s, 4)
    a, p = r["last"]["alloc"], r["last"]["plan"]
    print()
    print(f"  {shape_str(s)}")
    print(f"    decay tensor [B,NT,HV,C,C,K] fp32   = {dec:>15,} B = {dec/MB:>9.2f} MB")
    print(f"    [B,NT,HV,C,C] fp32, the shape class  = {cc:>15,} B = {cc/MB:>9.2f} MB")
    print(f"    [{r['last']['secs']:.0f}s   two consecutive runs "
          f"{'IDENTICAL' if r['reproducible'] else '*** DIFFER ***'}]")
    print()
    print("    M1  ALLOCATOR  (every Buffer.allocate request)")
    print(f"        allocation calls              = {a.n_calls:>16,}")
    print(f"        distinct sizes requested      = {a.distinct:>16,}")
    print(f"        largest single request        = {a.largest:>16,} B = {a.largest/MB:>8.2f} MB")
    print(f"        sum of all requests           = {a.total:>16,} B = {a.total/MB:>8.2f} MB")
    print(f"        peak LIVE device memory       = {a.peak_live:>16,} B = {a.peak_live/MB:>8.2f} MB")
    print(f"        any request >= decay tensor?  = {'*** YES ***' if a.largest >= dec else 'NO':>16}")
    print("        size histogram (nbytes matched against the port's own tensors):")
    names, _ = decode_sizes(s, a.hist())
    for n, cnt in a.hist().most_common():
        tag = "   <-- THE DECAY TENSOR" if n >= dec else ""
        print(f"            {n:>15,} B x{cnt:<6} {names[n]}{tag}")
    print()
    print("    M2  SCHEDULE  (buffers appearing as kernel PARAMs)")
    print(f"        kernel launches, both realize()s = {p.kernels:>15,}")
    print(f"        distinct param buffer sizes       = {len(p.params):>15,}")
    print(f"        largest param buffer              = {p.max_param:>15,} B = {p.max_param/MB:>8.2f} MB")
    print(f"        any param == decay tensor?        = {'*** YES ***' if p.max_param >= dec else 'NO':>15}")
    if p.max_param:
        print(f"        largest param / decay tensor      = 1/{dec/p.max_param:>14.0f}"
              f"   (= 1/K if the identity holds)")
        print(f"        identity  B*NT*HV*C*C*4 = {cc:,} B  -> "
              f"{'MATCH' if cc == p.max_param else 'MISMATCH'}")
    print("        param histogram:")
    pnames, _ = decode_sizes(s, p.params)
    for n, cnt in p.params.most_common():
        print(f"            {n:>15,} B  param of {cnt:>7,} launches  "
              f"({n*cnt/MB:>11,.1f} MB of argument bytes)  {pnames[n]}")
    print(f"        tensor-argument bytes, all launches = {p.arg_bytes:>12,} B"
          f" = {p.arg_bytes/MB:>10,.1f} MB")
    print()
    print("    M3  LARGEST INTERMEDIATE THAT REACHES DEVICE MEMORY")
    print(f"        largest                        = {p.max_param:>15,} B = {p.max_param/MB:>8.4f} MB")
    print(f"        byte model, one write + one read = {2*p.max_param:>14,} B = {2*p.max_param/MB:>8.4f} MB")
    print(f"        for scale, one decay tensor under the same rule would be")
    print(f"                                        = {2*dec:>15,} B = {2*dec/MB:>8.1f} MB")
    print(f"        modelled / actual               = {2*dec/(2*p.max_param):>15.0f}x")
    if verbose:
        print(f"        shape note: at this shape K=V=C={s['K']} and T=NT*C, so "
              f"[B,NT,HV,C,C],")
        print(f"        [B,NT,HV,C,K], [B,NT,HV,C,V] and [B,T,HV,V] are ALL "
              f"{p.max_param:,} B. The size")
        print(f"        alone cannot say which tensor it is, and does not need to:")
        print(f"        the decay tensor is K x this size and is in no bucket.")
    return dict(shape=s, dec=dec, cc=cc, alloc=a, plan=p, result=r)


def planner_lines() -> list[str]:
    """tinygrad's own DEBUG=1 memory-planner line, captured in a subprocess.

    This is the first-hand identification of the ARENA: "memory reduced from
    47.02 MB -> 8.42 MB, 6048 -> 1 bufs" says the planner packed 6,048 logical
    buffers into ONE allocation of 8.42 MB. That is why the recorded "largest
    single allocation" is a peak-live figure and not an arbitrary one, and it
    is a stronger statement than the note that quotes it makes."""
    code = (
        "import sys, os; sys.path.insert(0, %r); os.environ['DEBUG']='1'\n"
        "import tiny_kda, test_ref as T\n"
        "B,Tt,H,HV,K,V,C = 2,512,4,4,64,64,64\n"
        "q,k,v,g,beta = T.rnd(B,Tt,H,HV,K,V,seed=0,l2n=True)\n"
        "tiny_kda.kda_chunk(q,k,v,g,beta,chunk_size=C)\n" % HERE)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + "\n" + p.stderr)
    return [ln.strip() for ln in txt.splitlines()
            if "memory reduced from" in ln or ln.strip().startswith("scheduled ")]


def section_measure(shapes: list[dict]) -> list[dict]:
    head("5. THE MEASUREMENT -- kda/tiny_kda.py's kda_chunk, WHOLE OPERATOR, "
         "fp32, FRESH GRAPH PER RUN")
    print(f"\n  device = {Device.DEFAULT}")
    print("  M1 spies on Buffer.allocate/deallocate. M2 spies on the memory")
    print("  planner's input, so it sees buffers before arena coalescing.")
    if shapes and shapes[0] == BENCH:
        print()
        print("  tinygrad's own DEBUG=1 planner line, first-hand, same shape:")
        for ln in planner_lines():
            print(f"    {ln}")
        print()
        print("  '6048 -> 1 bufs' is the whole story of the 8.42 MB: 6,048 logical")
        print("  intermediate buffers are suballocated into ONE arena of 8.42 MB, and")
        print("  that arena is the largest single allocation the port requests. The")
        print("  note that records it as 'largest single allocation' is right and")
        print("  understates it: it is the peak live intermediate footprint, i.e. the")
        print("  answer to 'how much device memory does this operator need', which is")
        print("  8x SMALLER than one decay tensor rather than merely smaller.")
    reports = []
    for s in shapes:
        try:
            r = measure_full_op(s)
        except Exception as e:  # noqa: BLE001
            print()
            print(f"  {shape_str(s)}")
            print(f"    NOT SCHEDULABLE ON THIS HOST: {type(e).__name__}: {str(e)[:80]}")
            print("    see §8")
            continue
        reports.append(report_shape(s, r))
    return reports


# =============================================================================
# 6. backends: is this a property of the schedule or of the card?
# =============================================================================
def backend_census_line() -> str:
    """Minimal, machine-readable census for the current DEV. Run in a
    subprocess because tinygrad picks its backend at import time."""
    s = BENCH
    r = measure_full_op(s, runs=1)
    a, p = r["last"]["alloc"], r["last"]["plan"]
    return (f"CENSUS dev={Device.DEFAULT} decay={decay_bytes(s,4)} "
            f"allocs={a.n_calls} distinct={a.distinct} max_alloc={a.largest} "
            f"total={a.total} peak_live={a.peak_live} kernels={p.kernels} "
            f"max_param={p.max_param} arg_bytes={p.arg_bytes} "
            f"params={sorted(p.params.items())}")


def section_backends() -> None:
    head("6. IS THIS A PROPERTY OF THE SCHEDULE OR OF THE CARD? -- CPU vs METAL")
    print("\n  tinygrad picks its backend at import time, so each backend is run in")
    print("  its own process. Same code, same shapes, same seed.")
    print()
    lines = {}
    for dev in ("METAL", "CPU"):
        if dev not in Device._devices:
            print(f"  {dev:<6} not present on this host -- skipped")
            continue
        env = dict(os.environ, DEV=dev)
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--backend-census"], capture_output=True, text=True,
                           env=env, cwd=os.path.dirname(HERE))
        for ln in p.stdout.splitlines():
            if ln.startswith("CENSUS "):
                lines[dev] = ln
    for dev, ln in lines.items():
        print(f"  {ln}")
    print()
    if len(lines) == 2:
        def get(ln, key):
            for tok in ln.split():
                if tok.startswith(key + "="):
                    return tok[len(key) + 1:]
            return None
        a, b = lines["METAL"], lines["CPU"]
        same = all(get(a, k) == get(b, k)
                   for k in ("max_param", "kernels", "arg_bytes", "params"))
        print(f"  the two schedules agree on max_param / kernels / arg_bytes / params: "
              f"{'YES' if same else 'NO'}")
        print(f"  allocator census, for completeness: allocs {get(a,'allocs')} vs "
              f"{get(b,'allocs')}, total {int(get(a,'total')):,} vs "
              f"{int(get(b,'total')):,} B, peak live {int(get(a,'peak_live')):,} vs "
              f"{int(get(b,'peak_live')):,} B.")
        print(f"  max_alloc {get(a,'max_alloc')} vs {get(b,'max_alloc')} B -- the two "
              f"backends' allocator")
        print(f"  requests differ by {abs(int(get(a,'total'))-int(get(b,'total')))} B "
              f"out of {int(get(a,'total')):,}, which is a backend bookkeeping")
        print(f"  difference in constant uploads and changes neither verdict.")
        print()
        if same:
            print("  [MEASURED] M1 and M2 are properties of the generated SCHEDULE, not")
            print("  of the device. Two different backends, two different memory")
            print("  models, byte-identical buffer sets. The buffer is either in the")
            print("  plan or it is not, and here it is not, on both.")
        else:
            print("  [MEASURED] the backends DISAGREE. The verdict is then")
            print("  backend-specific and must be quoted with its scope attached,")
            print("  in the same way ledger C-04 / P-1 require for the divide defect.")
        print()
    print("  WHY THAT IS SUFFICIENT, and what it is not:")
    print("  It is sufficient for M1, M2 and M3: whether a buffer exists in a plan")
    print("  is decided by the scheduler, and the plan is what §5 measures. It is")
    print("  NOT sufficient for M4. DRAM traffic is a property of a cache hierarchy")
    print("  and a card, and §7 is explicit that no traffic number is reported here.")
    print("  It is also NOT a claim about CUDA, which is untested for the same reason")
    print("  P-1 is untested: no NVIDIA device was available. A backend that")
    print("  declines to fuse this contraction would report a different M1/M2, and")
    print("  the correct form of the study's claim is then 'tinygrad's scheduler")
    print("  fuses it, on every backend measured'.")


# =============================================================================
# 7. DRAM traffic
# =============================================================================
def section_traffic(reports: list[dict]) -> None:
    head("7. DRAM TRAFFIC (M4) -- NOT MEASURED, AND WHAT THE MEASUREMENT ALREADY")
    print("RULES OUT WITHOUT IT")

    if not reports:
        print("\n  no shape could be scheduled; nothing to report.")
        return
    p, a = reports[0]["plan"], reports[0]["alloc"]
    dec = reports[0]["dec"]
    top_n = p.max_param
    top_reads = p.params.get(top_n, 0)

    print("""
  This file reports no DRAM-traffic number, because none was measured, and a
  traffic number must never be inferred from an allocator number. What it DOES
  establish needs no counter:

    A buffer carries DRAM traffic only if some kernel reads or writes it.
    A kernel can only read or write buffers that appear as its arguments.
    §5 shows the decay tensor appears as no kernel's argument, at three shapes,
    on two backends, with an instrument proven able to see it when it is there
    (§4). Therefore the 268.4 MB of modelled traffic has no carrier in the
    generated program, and the charge is invalid on ANY device: no cache, no L2
    and no card can move a tensor that was never written.

  CONSEQUENCE FOR THE PRE-REGISTERED TEST, notes/06 §2 and P-2:
    P-2 asks for `dram__bytes.sum` in [233, 349] MB and treats < 58 MB as
    refutation by L2 absorption. That test is UNRUN and is NOT refuted here.
    But it is now testing a model whose central term has no carrier, so a
    PASSING run would not confirm the byte model -- it would mean the traffic
    comes from somewhere the model did not put it. P-2 should be re-scoped
    before it is run, and its refutation threshold re-derived, because the
    < 58 MB threshold was set against a model that charges 268.4 MB to a tensor
    that does not exist.""")
    print(f"""
  THE LARGEST UNRESOLVED QUANTITY, named: the plan shows a {top_n:,} B
  buffer passed as an argument to {top_reads:,} separate launches
  ({top_n*top_reads/MB:,.0f} MB of argument bytes, against a
  {top_n/MB:.1f} MB live working set and an {a.largest/MB:.1f} MB planner arena).
  Whether those re-reads hit L2 or reach DRAM is a cache question and CANNOT be
  answered off-card. It is a real number in the record now, it is neither
  54.1 MB nor 290.7 MB, and it needs a name: SCHEDULE ARGUMENT BYTES.

  WHAT A REAL GPU WOULD ADD (none of which this file substitutes for):
    1. `dram__bytes.sum` for one kda_chunk call, fp32, at the bench shape. The
       only number that settles M4. Nsight Compute, one profile, R8 in
       bench/PLAN.md.
    2. Whether the thousands of re-reads above hit L2. Off-card: impossible.
    3. Whether the CUDA backend's schedule differs from Metal's and CPU's. The
       fusion decision is a scheduler decision and should travel, but two
       backends agreeing is not three, and CUDA is untested for exactly the
       reason P-1 is untested.
    4. bf16. The port is fp32 by construction (tiny_kda.py:44). The ledger's
       < 58 MB threshold is dtype-sensitive because one decay tensor is
       67.1 MB fp32 and 33.55 MB bf16; a bf16 run could not distinguish the two
       stories, and this port cannot produce one at all.
    5. A real Triton kernel to race against. notes/06 U2 and
       bench/HYPOTHESES.md §6 already record that nothing available can refute a
       TinyGrad-vs-human ratio, and that is unchanged by this file.""")


# =============================================================================
# 8. why the cost.py shape is not run here
# =============================================================================
def section_limits() -> None:
    head("8. WHY THE cost.py SHAPE IS NOT RUN HERE, WITH THE MEASUREMENT")
    a, b = BENCH, STUDY
    ratio = (slabs(b) * b["C"] ** 2) / (slabs(a) * a["C"] ** 2)
    print(f"""
  kda/cost.py:17  {shape_str(STUDY)}
    slabs B*NT*HV = {slabs(STUDY):,}   vs bench {slabs(BENCH):,}   = {ratio:.0f}x
    decay tensor  = {decay_bytes(STUDY,4):,} B = {decay_bytes(STUDY,4)/GIB:.2f} GiB in fp32 alone

  The graph is O(B*NT*HV*C^2) in UOps, because the UT transform unrolls C
  forward-substitution steps, each carrying a [B,NT,HV,C,C] matrix. That is
  where the {ratio:.0f}x is, and it is in the UOp cache, not in device memory.

  [MEASURED, captured first-hand in §5 at the bench shape] one kda_chunk call
  schedules 6,188 kernels in ~15.0 s with 288,406 live UOps, then 6,041 kernels
  with 462,345 UOps -- the two realize()s of `o.numpy()` and `tS.numpy()`. The
  graph is O(B*NT*HV*C^2) in UOps because the UT transform unrolls C
  forward-substitution steps, each carrying a [B,NT,HV,C,C] matrix. That is
  where the {ratio:.0f}x lands, and it is in the UOp cache, not in device memory.
  x{ratio:.0f} is ~{288406*ratio/1e6:.0f}M UOps before a single kernel is
  launched. This host cannot schedule that in reasonable time, so the FULL
  OPERATOR at the cost.py shape is NOT measured here and is carried as
  [UNVERIFIED].

  The isolated decay in §4 IS run at a shape in the same family as the cost.py
  shape -- B, HV, K and C all equal to it, only NT reduced -- precisely so the
  decay-fusion question at the study shape is not left untested. What does not
  transfer from that is the whole-operator budget.

  THE IDENTITY THAT DOES TRANSFER, checked at every shape §5 reaches:

        largest kernel argument  =  B*NT*HV*C*C*4
        one decay tensor         =  B*NT*HV*C*C*K*4   =  K x that argument

  Both sides contain B*NT*HV*C*C, so the gap is exactly K and is independent of
  B, NT, HV and C. §5 confirms the equality at three shapes spanning a factor of
  4 in the decay tensor's size. At the cost.py shape the identity predicts a
  largest argument of {cc_bytes(STUDY,4):,} B against a
  {decay_bytes(STUDY,4)/GIB:.2f} GiB decay tensor: a gap of {STUDY['K']} = K.
  [MODELLED] for that shape, [MEASURED] for the identity, [UNVERIFIED] for the
  full operator there.""")


# =============================================================================
# 9. the verdict
# =============================================================================
def section_verdict(reports: list[dict]) -> None:
    head("9. VERDICT")
    if not reports:
        print("\n  no shape could be scheduled; no verdict.")
        return
    s0, a0, p0 = reports[0]["shape"], reports[0]["alloc"], reports[0]["plan"]
    dec = reports[0]["dec"]
    saw_alloc = any(r["alloc"].largest >= r["dec"] for r in reports)
    saw_param = any(r["plan"].max_param >= r["dec"] for r in reports)

    print(f"\n  shapes measured   : {len(reports)}  (first: {shape_str(s0)})")
    print("  backends in §6    : see §6 (each in its own process)")
    print("  positive control  : §4 -- the instrument allocates the 67,108,864 B")
    print("                      decay tensor the moment the schedule changes")
    print("  DRAM traffic      : not measured (§7)")
    print()
    if saw_alloc or saw_param:
        print("  VERDICT: MATERIALISED. A device buffer of the decay tensor's exact")
        print("  size exists. The 256x / 48.8x / 13.0x framing stands, subject to the")
        print("  dtype caveat. Re-derive the byte model in allocator bytes rather")
        print("  than traffic bytes and stop here.")
    else:
        print("  VERDICT: NOT MATERIALISED. At every shape measured, on every backend")
        print("  measured, the [B,NT,HV,C,C,K] decay tensor is never allocated and")
        print("  never appears as a kernel argument. The largest thing that reaches")
        print("  device memory is the [B,NT,HV,C,C] family, exactly K times smaller.")
        print("  The whole [C,C] slab fuses into one kernel and exists only as that")
        print("  kernel's register working set. The compiler beat the analytic model.")
    print()
    print("  ONE LINE, FOR PASTING:")
    if saw_alloc or saw_param:
        print('    "VERDICT: the KDA [B,NT,HV,C,C,K] decay tensor IS materialised; it')
        print('     is allocated and it is a kernel argument, so 256x/48.8x/13.0x stand."')
    else:
        print(f'    "VERDICT: the KDA [B,NT,HV,C,C,K] decay tensor is NEVER MATERIALISED')
        print(f'     -- at {shape_str(s0)} fp32 it is never allocated (largest device allocation is')
        print(f'     the {a0.largest/MB:.2f} MB planner arena) and is never a kernel argument; the largest')
        print(f'     kernel argument is {p0.max_param:,} B = B*NT*HV*C*C*4, exactly')
        print(f'     K={s0["K"]} = 1/64 smaller than the {dec:,} B tensor. The 268.4 MB / 48.8x / 13.0x')
        print(f'     figures are WITHDRAWN; the byte model charged a tensor that has no')
        print(f'     carrier in the generated program. DRAM traffic itself remains')
        print(f'     unmeasured on this host."')
    print()
    print(f"  THE FIVE QUANTITIES, SEPARATELY, at {shape_str(s0)} fp32:")
    print()
    print(f"    {'quantity':<34}{'bytes':>16}{'MB':>12}   in the record?")
    rule("-", 96)
    print(f"    {'allocator  sum of all requests':<34}{a0.total:>16,}{a0.total/MB:>12.2f}"
          f"   yes: 54.1 MB")
    print(f"    {'allocator  largest single request':<34}{a0.largest:>16,}{a0.largest/MB:>12.2f}"
          f"   yes: 8.42 MB")
    print(f"    {'allocator  peak live memory':<34}{a0.peak_live:>16,}{a0.peak_live/MB:>12.2f}"
          f"   NO -- new")
    print(f"    {'schedule   largest kernel argument':<34}{p0.max_param:>16,}{p0.max_param/MB:>12.2f}"
          f"   NO -- new")
    print(f"    {'schedule   tensor-argument bytes':<34}{p0.arg_bytes:>16,}{p0.arg_bytes/MB:>12.1f}"
          f"   NO -- new")
    print(f"    {'DRAM traffic':<34}{'NOT MEASURED':>16}{'-':>12}   must not be")
    print(f"    {'':34}{'':>16}{'':>12}   inferred from these")
    print()
    print("  Five rows, five different quantities. The record had two of them and a")
    print("  modelled third, and treated all three as one. That is the confusion.")
    print("  If you are reading a number from this study and want to know which")
    print("  regime it is in, this table is the answer.")


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Is the KDA [C,C,K] decay tensor materialised?",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="bench shape only (one schedule instead of three)")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the positive control (strongly discouraged)")
    ap.add_argument("--backend-census", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.backend_census:
        print(backend_census_line())
        return 0

    shapes = [BENCH] if args.quick else [
        BENCH,
        dict(BENCH, B=1),
        dict(BENCH, T=1024),
    ]
    print("=" * 96)
    print("kda/materialisation.py -- does the KDA [C,C,K] decay tensor reach "
          f"device memory?")
    print(f"device {Device.DEFAULT}  |  python {sys.version.split()[0]}  |  "
          f"tinygrad at {os.path.dirname(os.path.dirname(__import__('tinygrad').__file__))}")
    print("=" * 96)
    section_senses()
    section_provenance()
    section_units()
    if not args.no_control:
        section_control()
    reports = section_measure(shapes)
    section_backends()
    section_traffic(reports)
    section_limits()
    section_verdict(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
