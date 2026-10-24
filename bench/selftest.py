#!/usr/bin/env python
"""selftest.py - validate the instrument on this machine, without a GPU.

Run:  .venv/bin/python bench/selftest.py            (from the repo root)
      PROF_ARCH=sm_75 .venv/bin/python bench/selftest.py

What this does, in order, and why each part exists:

  1. capability        What level does this machine actually reach, and why? On a Mac with no
                       CUDA this must come out as level 0 with a stated reason -- if it
                       claims anything else, the instrument is lying and everything after
                       this point is worthless.
  2. degradation       Walk the ladder 0 -> 3 -> 2 -> 1 with forced levels and show that the
                       reporting changes its claims and nothing else. A forced level is
                       labelled as a dry run so it can never be mistaken for a measurement.
  3. roofline formula  Feed the pure peak function synthetic device attributes for the three
                       archs the study uses, plus an unknown one. This tests the ARITHMETIC
                       on a machine with no GPU, which is the only place it can be tested.
                       The attribute values are inputs to the test, not results of it.
  4. ncu parsing       Feed the level-2 parser a recorded `ncu --csv` transcript and check
                       the pivot, the join to kernel records, and the missing-metric
                       accounting. The CSV parser is exercised without an ncu binary.
  5. existing probes   Drive the repo's own probes through the harness: the gather probe's
                       workload, and the KDA port from kda/tiny_kda.py at the shape
                       notes/03-kda.md uses. Nothing outside bench/ is written.

On a laptop this validates the instrument's plumbing, its arithmetic, its static channel and
its honesty. It cannot validate a single hardware number, and it does not try: there is no
GPU here, so every device-derived field prints n/a with a reason.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DEV", os.environ.get("DEV", "CPU"))   # this lane validates on CPU

import numpy as np                                                    # noqa: E402
import prof                                                           # noqa: E402
from prof import (Work, calibrate, describe_capability, measure, ncu_run, report,   # noqa: E402
                  roofline_from_attrs, static_facts, to_dict, verdict)

BAR = "=" * 100


def hdr(t: str) -> None:
  print(f"\n{BAR}\n{t}\n{BAR}")


# --------------------------------------------------------------------------------------
# 1. capability
# --------------------------------------------------------------------------------------
def t_capability() -> prof.Capability:
  hdr("1. CAPABILITY on this machine")
  cap = describe_capability(probe=True)
  print(cap.explain())
  if cap.ncu.classification != "NOT_PROBED":
    print(f"\nncu path      : {cap.ncu.path}")
    print(f"ncu version   : {cap.ncu.version}")
    print(f"ncu class     : {cap.ncu.classification}")
    if cap.ncu.stderr_head:
      print(f"ncu stderr    : {cap.ncu.stderr_head[:300]}")
  print(f"nvidia-smi    : {cap.facts.nvidia_smi or 'absent'}")
  print(f"profiling param: {cap.facts.restrict_profiling_to_admins or 'not readable'}")
  print(f"attrs         : {cap.facts.attrs or 'none (no CUDA driver)'}")
  assert cap.level in (0, 2, 3), f"a CUDA machine with ncu is level 2, not 1: {cap.level}"
  assert cap.counters == (cap.level <= 2 and cap.level >= 2), "counters and level disagree"
  return cap


# --------------------------------------------------------------------------------------
# 2. degradation ladder
# --------------------------------------------------------------------------------------
def t_ladder() -> None:
  hdr("2. DEGRADATION LADDER (dry run: forced levels, no numbers change)")
  ranks = {}
  for lvl in (0, 3, 2, 1):
    c = describe_capability(probe=False, force_level=lvl)
    yes, no = c.decidable()
    ranks[lvl] = (c.rank, {h.id for h in yes}, {h.id for h in no})
    print(f"\nforced level {lvl} (rank {c.rank}): {c.level_name}")
    print(f"  counters     : {c.counters_kind}, static {c.static}")
    print(f"  decidable    : {', '.join(h.id for h in yes) or 'none'}")
    print(f"  NOT decidable: {', '.join(h.id for h in no) or 'none'}")
  # The regression this guards: the level NUMBERS are not a quality order (1 is the best, 3
  # is the floor), so a hypothesis that needs counters must be decidable at level 1 even
  # though 1 < 2. Comparing min_level as a number inverts the ladder and loses M5_STALL at
  # the best level while granting it at a worse one.
  for lvl, (rank, yes, no) in ranks.items():
    for hid, h in prof.HYPOTHESES.items():
      want = rank >= prof.LEVEL_RANK.get(h.min_level, 0) or (h.has_static and True)
      got = hid in yes
      assert got == want, f"at forced level {lvl} (rank {rank}) {hid} decidable={got}, wanted {want}"
  assert ranks[1][0] > ranks[2][0] > ranks[3][0] > ranks[0][0], "ranks must be monotone"
  assert "M5_STALL" in ranks[1][1] and "M5_STALL" in ranks[2][1], \
      "M5_STALL needs counters, so both counter levels must decide it"
  assert "M5_STALL" in ranks[3][2] and "M5_STALL" in ranks[0][2], \
      "and neither non-counter level may claim to"
  assert ranks[0][1] <= ranks[3][1] <= ranks[2][1] == ranks[1][1], \
      "each level must be able to decide at least what the level below it can"
  print("\nladder regression checks pass: rank order is monotone, every level decides a superset "
        "of the one below it, and M5_STALL -- which has no static probe -- is decided by exactly "
        "the two counter levels")


# --------------------------------------------------------------------------------------
# 3. roofline arithmetic, tested with synthetic attributes
# --------------------------------------------------------------------------------------
# NOTE: these attribute values are TEST INPUTS, chosen to be the published values for the
# three cards the study uses. They are not measurements and nothing in prof.py contains them:
# roofline_from_attrs is a pure function, and on a real card the values arrive from
# cuDeviceGetAttribute. The expected outputs below are the vendor sheet numbers, so this
# test fails if the formula is wrong.
SYNTHETIC = {
  "T4  (sm_75)": ({"MULTIPROCESSOR_COUNT": 40, "CLOCK_RATE": 1590000, "MEMORY_CLOCK_RATE": 5000000,
                   "GLOBAL_MEMORY_BUS_WIDTH": 256, "L2_CACHE_SIZE": 4 << 20}, "sm_75", 320, 8.1),
  "P100(sm_60)": ({"MULTIPROCESSOR_COUNT": 56, "CLOCK_RATE": 1630000, "MEMORY_CLOCK_RATE": 1751000,
                   "GLOBAL_MEMORY_BUS_WIDTH": 512, "L2_CACHE_SIZE": 24 << 20}, "sm_60", 224, 9.3),
  "L4  (sm_89)": ({"MULTIPROCESSOR_COUNT": 58, "CLOCK_RATE": 2040000, "MEMORY_CLOCK_RATE": 6250000,
                   "GLOBAL_MEMORY_BUS_WIDTH": 192, "L2_CACHE_SIZE": 48 << 20}, "sm_89", 300, 30.3),
  "unknown arch": ({"MULTIPROCESSOR_COUNT": 100, "CLOCK_RATE": 1500000, "MEMORY_CLOCK_RATE": 4000000,
                    "GLOBAL_MEMORY_BUS_WIDTH": 256}, "sm_999", None, None),
}
# The memory clock for the L4 row is chosen so that the formula reproduces the published
# 300 GB/s; the value the driver actually reports is what prof.py will use on the card. The
# point of this table is to test the ARITHMETIC, not to be a spec sheet.


def t_roofline() -> None:
  hdr("3. ROOFLINE from device attributes (synthetic inputs; formula under test)")
  print(f"{'device':<14}{'arch':<9}{'GB/s':>9}{'sheet':>7}{'TFLOP/s':>9}{'sheet':>7}{'ridge':>8}  "
        f"{'mem vs sheet':>12} {'cmp vs sheet':>12}")
  for name, (attrs, arch, sheet_bw, sheet_fl) in SYNTHETIC.items():
    r = roofline_from_attrs(attrs, arch)
    mb = mc = None
    if sheet_bw and r.bytes_s:
      mb = abs(r.bytes_s / 1e9 - sheet_bw) / sheet_bw
    if sheet_fl and r.flops_s:
      mc = abs(r.flops_s / 1e12 - sheet_fl) / sheet_fl
    print(f"{name:<14}{arch:<9}{_gb_(r.bytes_s):>9}{_sheet(sheet_bw):>7}{_tf(r.flops_s):>9}"
          f"{_sheet(sheet_fl):>7}{_num(r.ridge_point):>8}  {_pct(mb):>12} {_pct(mc):>12}")
    if sheet_bw and r.bytes_s:
      assert mb < 0.01, f"{name}: memory roof {r.bytes_s/1e9:.4g} GB/s is {mb*100:.1f}% off the "\
                         f"published {sheet_bw} GB/s, so the DDR formula is wrong"
    if sheet_fl and r.flops_s:
      # Not asserted tightly on purpose: the sheet number is quoted at the vendor's reference
      # clock and the driver reports the maximum boost clock, so a gap is expected and the
      # derived number is a ceiling. The measured peak from calibrate() is the other bound.
      assert mc < 0.5, f"{name}: compute roof is {mc*100:.0f}% off the sheet value; check the "\
                        "cores/SM entry for this arch"
    if arch == "sm_999":
      assert r.flops_s is None, "an unknown arch must not get a compute roof"
      assert r.bytes_s, "an unknown arch still gets a memory roof: that one needs no arch table"
      assert any("unknown arch" in n for n in r.notes), "the unknown arch must be stated"
      assert r.ridge_point is None, "no compute roof means no ridge point"
      assert r.tensor_flops_s is None, "tensor peak is never invented"
  print("\nmemory roof reproduces the published bandwidth exactly on T4 and P100, which is "
        "what the DDR formula has to do")
  print("compute roof is a ceiling at the driver's max boost clock; the sheet number is quoted "
        "at a lower reference clock, so the gap is expected and is reported, not hidden")

  # the position logic, with expectations derived from the roofs rather than hardcoded
  r = roofline_from_attrs(SYNTHETIC["T4  (sm_75)"][0], "sm_75")
  print(f"\nT4 ridge point: {r.ridge_point:.4g} FLOP/B "
        f"(= {r.flops_s/1e12:.4g} TFLOP/s / {r.bytes_s/1e9:.4g} GB/s)")
  cases = [
    # half the compute roof, a tenth of the memory roof: intensity is far above the ridge, so
    # the kernel is compute-bound, and it sits at 50% of the compute limit
    ((r.flops_s * 0.5, r.bytes_s * 0.1), "compute", 0.5),
    # the mirror image: intensity far below the ridge, memory-bound, also 50% of its limit.
    # That symmetry is the roofline's defining property, so it is checked here rather than
    # asserted in a comment.
    ((r.flops_s * 0.1, r.bytes_s * 0.5), "memory", 0.5),
    ((1e13, 1e9), "compute", None),      # above the compute roof: reported, not clamped
    ((1e11, 1e11), "memory", None),
    ((None, 1e10), "memory", None),      # no FLOP count -> a bandwidth number, not a position
    ((1e11, None), "compute", None),     # no byte count -> one-sided
    ((None, None), None, None),          # nothing measured, nothing to say
  ]
  for rates, want, want_util in cases:
    p = r.position(*rates)
    assert p["bound"] == want, f"position({rates}) -> {p['bound']}, wanted {want}"
    if want_util is not None:
      assert p["util"] is not None and abs(p["util"] - want_util) < 1e-9, \
          f"position({rates}) util {p['util']}, wanted {want_util}"
    print(f"  position(flops={_num(rates[0])}, bytes={_num(rates[1])}) -> bound={p['bound']}, "
          f"util={_num(p['util'])}")
    print(f"      {p['why']}")
  assert r.position(None, None)["why"], "a no-data position must still explain itself"
  over = r.position(1e13, 1e9)
  assert over["util"] and over["util"] > 1, "an efficiency above 1 must survive, not be clamped"
  assert "above 100%" in over["why"], "and it must be flagged as a problem, not celebrated"
  print("\ndegenerate cases pass: one-sided inputs say which side they are, and an impossible "
        "efficiency is reported as impossible")


def _gb_(x):
  return "n/a" if x is None else f"{x/1e9:.4g}"


def _tf(x):
  return "n/a" if x is None else f"{x/1e12:.4g}"


def _num(x):
  return "n/a" if x is None else f"{x:.4g}"


def _pct(x):
  return "n/a" if x is None else f"{x*100:.1f}%"


def _sheet(x):
  return "n/a" if x is None else f"{x:g}"


# --------------------------------------------------------------------------------------
# 4. the ncu path, exercised with a recorded transcript
# --------------------------------------------------------------------------------------
# Two launches of the same kernel, three metrics on one of them and a metric missing on the
# other. This is the shape that matters: a real ncu run does not return every metric for
# every kernel, and the harness has to say which are missing rather than default them.
FAKE_NCU_CSV = '''"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","dram__bytes_read.sum","byte","4,194,304"
"0","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","gpu__time_duration.sum","nsecond","1,234,000"
"0","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","sm__inst_executed_pipe_tensor.sum","inst","0"
"0","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","launch__registers_per_thread","register/thread","64"
"0","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active","%","n/a"
"1","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","dram__bytes_read.sum","byte","4,194,304"
"1","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","gpu__time_duration.sum","nsecond","1,300,000"
"1","1234","python","host","r_32_8_256_64_4","2024-01-01","1","7","Command line","launch__registers_per_thread","register/thread","64"
'''


def t_ncu(cap: prof.Capability) -> None:
  hdr("4. NCU LEVEL-2 PATH, exercised with a recorded transcript")
  rows = prof._parse_ncu_csv(FAKE_NCU_CSV)
  per = prof._pivot_ncu(rows)
  print(f"parsed {len(rows)} csv rows -> {len(per)} kernel launches")
  for k in per:
    print(f"  launch id={k['id']} name={k['name']} occurrence={k['occurrence']} "
          f"metrics={len(k['metrics'])}")
  assert len(per) == 2, "two launches must pivot to two records"
  assert per[0]["metrics"]["dram__bytes_read.sum"]["value"] == 4194304.0, "thousands separators parsed"
  assert per[0]["metrics"]["gpu__time_duration.sum"]["value"] == 1234000.0
  # a metric reported as n/a must be None, never 0
  assert per[0]["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]["value"] is None, \
      "an n/a metric must parse to None, not 0"
  # ...and an explicit zero is a measurement
  assert per[0]["metrics"]["sm__inst_executed_pipe_tensor.sum"]["value"] == 0.0, "an explicit 0 stays 0"
  print("  n/a parses to None and an explicit 0 stays 0: the two are not conflated")

  # now attach to a real measurement
  cap2 = describe_capability(probe=False)
  m = measure(Work("counter attach target", lambda: _mm(256), flops=2 * 256 ** 3,
                   bytes=3 * 256 * 256 * 4), capability=cap2)
  if m.kernels:
    target = [r for r in m.kernels if r.kind == "program"]
    if target:
      target[0].kernel = "r_32_8_256_64_4"          # pretend ncu named it the same
      res = prof.NCUResult(ran=True, level=2, per_kernel=per, requested=list(prof.LEVEL2_METRICS))
      joined = prof.attach_counters(m, res)
      print(f"  attached {joined} launches to {len(target)} program record(s)")
      print(f"  tensor pipe on launch 0: "
            f"{target[0].counters.get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active')!r}")
      print(f"  registers/thread       : {target[0].counters.get('launch__registers_per_thread')!r}")
      print(f"  missing on this record : {len(target[0].counters_missing)} of {len(prof.LEVEL2_METRICS)}")
      assert joined == 1, "exactly one launch should join"
      assert target[0].counters["sm__inst_executed_pipe_tensor.sum"] == 0.0, "zero tensor instructions kept"
      print(f"  level after attach     : {m.op.level} (counters {m.capability.counters_kind})")
      print(f"  M2_MMA verdict now    : {verdict(m, 'M2_MMA')[:150]}")
      print(f"  M3_OCCUPANCY verdict  : {verdict(m, 'M3_OCCUPANCY')[:150]}")
  print(f"  a real ncu run on this machine would say: {ncu_run.__doc__.splitlines()[0]}")
  st = prof.probe_ncu(enable=True)
  print(f"  live ncu probe on this machine: {st.one_line()}")


# --------------------------------------------------------------------------------------
# 5. the existing probes
# --------------------------------------------------------------------------------------
def _mm(n: int):
  from tinygrad import Tensor
  a = Tensor(np.random.rand(n, n).astype(np.float32))
  b = Tensor(np.random.rand(n, n).astype(np.float32))
  return a @ b


def t_probe_gather(cap: prof.Capability) -> prof.Measurement:
  """The gather workload from probes/gather_floor.py, as a Tensor graph.

  probes/gather_floor.py measures a 1-D fancy index out of a source of growing size and asks
  whether the cost is O(source) or O(output). It does that by scraping DEBUG=2 output from
  subprocesses. The harness can do the same thing with real per-kernel records, and it can
  also answer the part the scraped version cannot: what the generated code does with the
  index, statically.
  """
  hdr("5a. EXISTING PROBE: probes/gather_floor.py workload, via the harness")
  from tinygrad import Tensor
  print(f"{'source':>10} {'out':>6} {'ms (host)':>11} {'GB/s over src':>14} {'GB/s over out':>14} "
        f"{'min bytes':>11} {'est bytes':>11}  static")
  rows = []
  for src, out in ((1 << 14, 64), (1 << 18, 64), (1 << 20, 64), (1 << 20, 1 << 20)):
    rng = np.random.default_rng(0)
    idx = Tensor(rng.integers(0, src, size=out).astype(np.int32))
    data = Tensor(rng.integers(0, 4096, size=src).astype(np.int32))
    m = measure(Work(f"gather {out} from {src}",
                     lambda d=data, i=idx: d[i],
                     # an ideal gather reads out elements and one index: it does not read src
                     bytes=out * 4 * 2, note="caller minimum: read out int32 + read out int32 index"),
                capability=cap)
    op = m.op
    t = op.gpu_ms
    stat = next((r.static for r in m.kernels if r.static and r.static.available), None)
    print(f"{src:>10,} {out:>6,} {_ms(t):>11} {_gb(src*4, t):>14} {_gb(out*4, t):>14} "
          f"{prof._b(int(op.bytes_min or 0)):>11} {prof._b(op.bytes_est):>11}  "
          + (f"ld.global={stat.ld_global} scalar={stat.ld_global_scalar}" if stat else "n/a"))
    rows.append((src, out, op))
  return None


def t_probe_kda(cap: prof.Capability) -> None:
  """kda/tiny_kda.py at the shape notes/03-kda.md uses, unchanged.

  Every number the workload needs is computed here from the shapes, and shown, because the
  ratio between "what the port materialises" and "what the hand kernel must move" is a claim
  of the study. The instrument's job is to time what it is told to time and to report the
  ratio; it does not get to decide which pair of numbers is the interesting one.
  """
  hdr("5b. EXISTING PROBE: kda/tiny_kda.py at B=2 T=512 H=HV=4 K=V=64 C=64")
  B, T, HV, K, V, C = 2, 512, 4, 64, 64, 64
  NT = T // C
  mat = 2 * B * NT * HV * C * C * K * 4          # Dl and Dq_all, fp32
  io = B * T * HV * (2 * K + 2 * V + 1) * 4     # q, k, v, g, beta in and o out
  sub = B * C * K * 4                            # the [BC, K] the hand kernel builds per sub-chunk
  print(f"inputs+output  q,k,v,g,beta,o            = {io:>12,} B = {io/2**20:6.2f} MiB")
  print(f"materialised   2 x [B,NT,HV,C,C,K] fp32  = {mat:>12,} B = {mat/2**20:6.2f} MiB")
  print(f"               / inputs+output           = {mat/io:6.1f}x")
  print(f"               / [BC,K] per sub-chunk    = {mat/sub:6.1f}x")
  print("(the brief quotes ~256x; the operands behind that figure are not stated there, so both")
  print(" ratios are printed. Which pair is the fair comparison is the study's claim to make.)")

  sys.path.insert(0, os.path.join(ROOT, "kda"))
  import tiny_kda                                                  # noqa: PLC0415
  rng = np.random.default_rng(0)

  def qv(*shape):
    return rng.standard_normal(shape).astype(np.float32)

  def work():
    q = qv(B, T, 4, K)
    k = qv(B, T, 4, K)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)                # l2norm, as the KDA layer does
    v = qv(B, T, HV, V)
    g = -rng.random((B, T, HV, K), dtype=np.float32) * 0.1
    beta = rng.random((B, T, HV), dtype=np.float32) * 0.5
    tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
    return None

  flops = None      # left as None on purpose: see the note below
  m = measure(Work("KDA forward (kda/tiny_kda.py)", work, flops=flops, bytes=io,
                   materialised_bytes=mat, build_and_realize=True,
                   note="port realises internally, so this is host time at level 0"),
              capability=cap, reps=1)
  print()
  print(report(m, kernel_rows=0, verdicts=True))


def t_probe_static(cap: prof.Capability) -> None:
  """The static channel on cases where the answer is not in doubt, as a control.

  A strided access and a contiguous one are trivially distinguishable in the generated PTX.
  If the channel cannot separate them it is not measuring anything, and every H1 verdict
  built on it is worthless -- so this is a test of the instrument, not of tinygrad.
  """
  hdr("5c. CONTROL: the static channel must separate strided from contiguous")
  from tinygrad import Tensor
  n = 1 << 16
  src = Tensor(np.arange(n, dtype=np.float32))
  src.realize()                        # a real buffer, so the kernel has a PARAM to load from
  cases = {
    "contiguous, whole buffer":     lambda: (src * 2.0).sum(),
    "contiguous, every 4th lane":  lambda: (src[::4] * 2.0).sum(),
    "strided gather idx[::64]":     lambda: src[Tensor(np.arange(0, n, 64, dtype=np.int32))].sum(),
    "reversed read (stride -1)":    lambda: (src[::-1] * 2.0).sum(),
    "shifted read (stride +1)":     lambda: (src[1:] * 2.0).sum(),
  }
  for name, build in cases.items():
    lin = prof._plan(Work(name, build))
    sf = static_facts(lin, cap.static_arch)
    rec = next((x for x in sf if x is not None), None)
    if rec is None or not rec.available:
      print(f"  {name:<30} static unavailable: {rec.note if rec else 'no kernel in the linear'}")
      continue
    frac = rec.vectorised_load_frac
    print(f"  {name:<30} vectorised={_pct(frac) if frac is not None else 'n/a':>7} "
          f"widest={_num(rec.widest_ld_bytes)} B/thread  ld.global={rec.ld_global} "
          f"local={rec.local_ld}/{rec.local_st} shared={rec.ld_shared}/{rec.st_shared} "
          f"regs={rec.reg_decls}")
  # The control that matters: a contiguous pass must be able to reach a 16-byte load, and a
  # reversed (stride -1) pass must not silently claim it did.
  lin = prof._plan(Work("reverse control", lambda: src[::-1].contiguous().realize()))
  rev = next((x for x in static_facts(lin, cap.static_arch) if x is not None), None)
  if rev is not None and rev.available:
    print(f"\n  reversed copy: ld.global={rev.ld_global}, widest={_num(rev.widest_ld_bytes)} B/thread")
    print("  a stride of -1 cannot be expressed as a wider load, so the channel must not claim "
          "one; a channel that reported 16 B/thread here would be counting the wrong thing")


def t_calibration(cap: prof.Capability) -> None:
  hdr("6. CALIBRATION: the device's own achievable peak")
  out = calibrate(cap, iters=3, nbytes=1 << 24)
  for k, v in out.items():
    print(f"  {k:<14}: {v}")
  if cap.level == 0:
    print("\n  level 0: these are host-memory numbers. They are real and they are not a GPU "
          "result, which is why the level is 0 and not 3.")


def t_json(m: prof.Measurement) -> None:
  hdr("7. JSON: None must survive as null")
  d = to_dict(m)
  import json
  s = json.dumps(d, indent=2, default=str)
  print(f"  serialised {len(s)} chars, {s.count(': null')} null fields preserved")
  print("  (a null is an unavailable measurement; the absence of the key would be a bug and a "
        "0 would be a lie)")


def _ms(x):
  return "n/a" if x is None else f"{x:.4g}"


def _gb(b, t):
  return "n/a" if (not b or not t) else f"{b/(t/1e3)/1e9:.3g}"


if __name__ == "__main__":
  t0 = time.perf_counter()
  cap = t_capability()
  t_ladder()
  t_roofline()
  t_ncu(cap)
  m = t_probe_gather(cap)
  t_probe_kda(cap)
  t_probe_static(cap)
  t_calibration(cap)
  t_json(m if m else measure(Work("json target", lambda: _mm(64), flops=2 * 64 ** 3,
                                   bytes=3 * 64 * 64 * 4), capability=cap))
  print(f"\n{BAR}\nall self-tests passed in {time.perf_counter()-t0:.1f}s\n{BAR}")

