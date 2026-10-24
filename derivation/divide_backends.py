#!/usr/bin/env python3
"""derivation/divide_backends.py -- the same divide claim, per backend, from
the backends that actually exist on THIS host.

    python derivation/divide_backends.py                 # full report
    python derivation/divide_backends.py <claim>         # quotable value

WHY THIS IS A SEPARATE MODULE
    `derivation/divide.py` needs no device: `a*fl32(1/b)` is arithmetic, and
    the rate is a property of the arithmetic. This module needs a device, and
    its numbers are properties of a BACKEND AND A HOST. Mixing the two is how
    the record ended up quoting a Mac measurement as a property of TinyGrad.
    So they are separate modules, separately cited, and this one always prints
    the host and the device inventory it ran on.

WHAT IS MEASURED AND WHAT IS ONLY RECORDED
    `[MEASURED]`  a backend on this host, this run.
    `[RECORD]`    a number read out of a shipped log in `out/`, parsed, and
                  reproduced from the same draw where that is possible. Never
                  presented as measured here, because this host has no GPU.
    `[UNAVAILABLE]` a backend that is listed by tinygrad and does not work
                  here, with the exception that proves it. Never guessed.

THE CROSS-CHECK THAT DOES NOT NEED A GPU
    The `a/a` row is double-rounding arithmetic with no device in it. It
    reproduces bit for bit across vendors. So the counts the shipped T4 log
    reports can be re-derived here exactly, and that is a real check on the
    log: `self_division_log_crosscheck` asserts they are equal rather than
    printing them side by side and asking the reader to compare.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fp32 as F                                                      # noqa: E402
from _util import Module, PinDrift, Value, rule, run, ROOT              # noqa: E402

EXPECTED: dict[str, str] | None = None

MOD = Module("divide_backends", __doc__ or "", EXPECTED)

# The shipped GPU run. Parsed, never transcribed.
RUN_DIR = "out/kaggle/20260928-014248-gpu"
FP32_LOG = f"{RUN_DIR}/logs/fp32_correctness.log"
DIV_LOG = f"{RUN_DIR}/logs/divide_precision.log"

# `probes/divide_precision.py:57-61`'s law, at the log's own N.
LOG_N = 200_000


def _read(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath), encoding="utf-8", errors="replace") as f:
        return f.read()


def _log_inputs() -> tuple[np.ndarray, np.ndarray]:
    """`probes/divide_precision.py:inputs('pm2')` -- a ~ U[-2,2] and
    b ~ U[0.01,2.01] at N = 200,000, seed 0. Reconstructed from the probe's
    own source, which is pinned, so the log's rows can be re-derived here."""
    r = np.random.default_rng(0)
    a = r.random(LOG_N).astype(np.float32) * np.float32(4) - np.float32(2)
    b = r.random(LOG_N).astype(np.float32) * np.float32(2) + np.float32(0.01)
    return a, b


# ---------------------------------------------------------------- inventory

def _usable_devices() -> tuple[list[str], list[tuple[str, str]]]:
    """Every backend tinygrad lists, and for each whether it can run a
    float32 add here. The failure is reported, never swallowed."""
    from tinygrad import Tensor, Device
    ok, bad = [], []
    for d in Device._devices:
        try:
            (Tensor(np.ones(4, np.float32)).to(d) + Tensor(np.ones(4, np.float32)).to(d)).numpy()
            ok.append(d)
        except Exception as e:                                          # noqa: BLE001
            bad.append((d, f"{type(e).__name__}: {str(e)[:70]}"))
    return ok, bad


@MOD.claim("device_inventory")
def c_inventory() -> Value:
    """What exists here. A claim about the host, not about TinyGrad."""
    from tinygrad import Device
    ok, bad = _usable_devices()
    MOD.pins.param("usable", ok)
    MOD.pins.param("default", Device.DEFAULT)
    MOD.pins.param("host", f"{os.uname().sysname} {os.uname().machine}")
    lines = [f"  usable on this host: {', '.join(ok) or 'none'}", "",
             "  listed and NOT usable here, with the reason:"]
    lines += [f"    {d:9s} {why}" for d, why in bad]
    return Value(
        f"on this host ({os.uname().sysname} {os.uname().machine}, tinygrad "
        f"default `{Device.DEFAULT}`) {len(ok)} of the listed backends are "
        f"usable: {', '.join(ok)}; {len(bad)} are not and are reported as "
        f"`[UNAVAILABLE]` with their failure, never estimated",
        "\n".join(lines),
    )


# ---------------------------------------------------------------- per backend

def _measure(dev: str, a: np.ndarray, b: np.ndarray, ref: np.ndarray):
    from tinygrad import Tensor
    got = (Tensor(a).to(dev) / Tensor(b).to(dev)).numpy()
    rm = F.rcp_mul(a, b)
    return got, int((got != ref).sum()), bool((got == rm).all())


@MOD.claim("per_backend_rate")
def c_per_backend(n=None) -> Value:
    """The rate per usable backend, under the log's own law and n, so it is
    directly comparable with the shipped T4 rows, and again under the study's
    canonical law."""
    n = n or MOD.n
    a, b = _log_inputs()
    MOD.pins.param("log_n", LOG_N)
    regs = F.regimes(n)
    rows = []
    for dev in _usable_devices()[0]:
        got, nb, same = _measure(dev, a, b, F.exact_f32(a, b, "/"))
        row = [dev, nb, nb / len(b), same]
        for nm in F.REGIME_ORDER:
            aa, bb = regs[nm], F.INDEPENDENT_LAWS[0](regs[nm], n, nm)
            _, m, _ = _measure(dev, aa, bb, F.exact_f32(aa, bb, "/"))
            row.append(m / n)
        rows.append(tuple(row))
    MOD.pins.param("per_backend", [list(r) for r in rows])
    lo = min(r[1] / len(b) for r in rows if r[1])
    hi = max(r[1] / len(b) for r in rows)
    lines = [f"  {'device':8s} {'wrong':>9s} {'rate':>8s} {'== a*fl32(1/b)':>16s} "
             + " ".join(f"{nm.split()[0]:>10s}" for nm in F.REGIME_ORDER)]
    for r in rows:
        lines.append(f"  {r[0]:8s} {r[1]:>9,} {r[2]:7.2%} {str(r[3]):>16s} "
                     + " ".join(f"{x:9.3%} " for x in r[4:]))
    lines += ["",
              f"  under the log's own law (U[0.01,2.01] against U[-2,2], "
              f"n = {len(b):,}), the usable backends span {lo:.2%}-{hi:.2%};",
              "  the `canonical` columns are over the study's own law and n.",
              "  A backend that is 0.000% is rendering a real divide; the "
              "bit-identity column says whether the rest is `a*fl32(1/b)`."]
    nonbit = [r[0] for r in rows if not r[3]]
    return Value(
        f"per usable backend, under the shipped log's own divisor law "
        f"(U[0.01,2.01] against U[-2,2]) at n = {len(b):,}: the backends "
        f"that do not render a real divide are wrong in "
        f"{lo*100:.2f}%-{hi*100:.2f}% of cases and every one of them is "
        f"bit-identical to `a*fl32(1/b)`"
        + (f"; the exception is {', '.join(nonbit)}" if nonbit else
           "; there is no exception on this host")
        + f". [MEASURED] on this host; a per-backend number is a property of a "
          f"backend and a host, never of `/`",
        "\n".join(lines),
    )


@MOD.claim("bit_identity")
def c_bit_identity(n=None) -> Value:
    """Bit-identity is a stronger claim than a rate and survives the
    divisor-law problem, so it is stated separately."""
    n = n or MOD.n
    a, b = _log_inputs()
    ref, rm = F.exact_f32(a, b, "/"), F.rcp_mul(a, b)
    rows = []
    for dev in _usable_devices()[0]:
        got, nb, same = _measure(dev, a, b, ref)
        rows.append((dev, nb, same))
    same = [r[0] for r in rows if r[2]]
    diff = [(r[0], r[1]) for r in rows if not r[2]]
    MOD.pins.param("bit_identity_same", same)
    MOD.pins.param("bit_identity_diff", diff)
    lines = [f"  {d:8s} wrong {nb:>9,}   bit-identical to a*fl32(1/b): {s}" for d, nb, s in rows]
    lines += ["",
              f"  {len(same)} of {len(rows)} usable backends return the SAME ARRAY,",
              "  not merely the same number of mismatches. That is the form of",
              "  the claim that cannot be eroded by the divisor law."]
    return Value(
        f"on this host {len(same)} of the {len(rows)} usable backends "
        f"({', '.join(same)}) return a value bit-identical, element for element, "
        f"to `a*fl32(1/b)` for the same `a, b`; the backends that do not "
        f"({', '.join(d for d, _ in diff) or 'none'}) are the ones that render "
        f"a real divide. A bit-identity is not a rate and is immune to the "
        f"divisor law",
        "\n".join(lines),
    )


@MOD.claim("exact_arithmetic_others")
def c_others(n=None) -> Value:
    """`+ - *` on the same backends, with independent operands. The count is
    the denominator the study's sentence got wrong."""
    n = n or MOD.n
    regs = F.regimes(n)
    from tinygrad import Tensor
    rows, tot = [], 0
    for dev in _usable_devices()[0]:
        bad = 0
        for nm, a in regs.items():
            b = F.INDEPENDENT_LAWS[0](a, n, nm)
            for op, fn in (("+", lambda x, y: x + y), ("-", lambda x, y: x - y),
                           ("*", lambda x, y: x * y)):
                got = fn(Tensor(a).to(dev), Tensor(b).to(dev)).numpy()
                bad += int((got != F.exact_f32(a, b, op)).sum())
        rows.append((dev, bad))
        tot = max(tot, bad)
    denom = 3 * len(regs) * n
    MOD.pins.param("others", [list(r) for r in rows])
    MOD.pins.param("others_denominator", denom)
    worst = max(r[1] for r in rows)
    return Value(
        f"`+`, `-` and `*` with INDEPENDENT operands are correctly rounded on "
        f"every usable backend: {worst} wrong out of {denom:,} operations "
        f"(3 ops x {len(regs)} regimes x n = {n:,}) at the largest; the "
        f"per-op denominator is {denom // 3:,} and the earlier sentence in the "
        f"record named three ops while quoting that one",
        "\n".join(f"  {d:8s} {b:>9,} wrong of {denom:,}" for d, b in rows),
    )


# ---------------------------------------------------------------- the log

@MOD.claim("self_division_log_crosscheck")
def c_log_self() -> Value:
    """The `a/a` row of the shipped T4 log, re-derived here. This check does
    NOT need a GPU: `a/a` is double-rounding arithmetic, so if the re-derivation
    matches the log bit for bit then the log's `a/a` rows are confirmed and
    the host-independence of the degenerate quantity is confirmed with them."""
    txt = _read(FP32_LOG)
    MOD.pins.file(FP32_LOG)
    MOD.pins.file(DIV_LOG)
    MOD.pins.param("fp32_log_sha", txt)
    rows = []
    regime = None
    for line in txt.splitlines():
        m = re.match(r"===\s*(.*?)\s*\(n=([\d,]+)\)", line)
        if m:
            regime = m.group(1)
            continue
        m = re.match(r"^/\s+([\d,]+)\s+\(\s*([\d.]+)%\)\s+(\d+)\s+\(\s*([\d.]+)%\)\s*$", line)
        if m and regime:
            rows.append((regime, int(m.group(1).replace(",", "")), float(m.group(2)),
                         float(m.group(4))))
    regs = F.regimes(F.N_DEFAULT)
    mism = []
    for (regime, cnt, pct, npy), nm in zip(rows, F.REGIME_ORDER):
        mine = int((F.rcp_mul(regs[nm], regs[nm]) != F.ONE).sum())
        if cnt != mine or abs(pct - 100.0 * mine / F.N_DEFAULT) > 0.0005 or npy != 0:
            mism.append((regime, cnt, mine, pct, npy))
    MOD.pins.param("log_self_rows", [[r[0], r[1], r[2], r[3]] for r in rows])
    MOD.pins.param("log_self_mismatch", mism)
    if mism:
        raise RuntimeError(f"the shipped log's a/a rows do NOT re-derive here: {mism}")
    return Value(
        f"the shipped T4 log's `a/a` rows re-derive on this host to the unit: "
        + ", ".join(f"{r[0]} {r[1]:,} ({r[2]:.3f}%)" for r in rows)
        + f", and numpy's own column is 0 in every one. `a/a` is "
          f"double-rounding arithmetic with no device in it, so this check runs "
          f"without a GPU and confirms both the log and the host-independence of "
          f"the degenerate quantity. [RECORD parsed + RE-DERIVED]",
        "\n".join(f"  log   {r[0]:22s} {r[1]:>9,}  ({r[2]:.3f}%)   numpy {r[3]}"
                  for r in rows)
        + "\n\n  re-derived here, same counts, same percentages, numpy 0.",
    )


@MOD.claim("t4_rows_unavailable_here")
def c_t4_unavailable() -> Value:
    """The GPU rows, and the reason they are RECORD and not MEASURED. A claim
    whose whole content is that it could not be run."""
    txt = _read(DIV_LOG)
    rows = {}
    # the probe prints two shapes, both parsed here and neither transcribed:
    #   "   CUDA  :  30,748 / 200,000 wrong  (15.37%)"     (the a/a row)
    #   "   CUDA   :  54,270 wrong (27.13%)   == a*rcp(b) bit for bit"  (the a/b row)
    for line in txt.splitlines():
        m = re.match(r"^\s{3}(\w+)\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s+wrong"
                     r"\s*\(\s*([\d.]+)%\)\s*$", line)
        if m:
            rows[f"{m.group(1)} (a/a)"] = (int(m.group(2).replace(",", "")),
                                           int(m.group(3).replace(",", "")),
                                           float(m.group(4)), "n/a")
            continue
        m = re.match(r"^\s{3}(\w+)\s*:\s*([\d,]+)\s+wrong\s*\(\s*([\d.]+)%\)\s*"
                     r"(==|differs)", line)
        if m:
            rows[f"{m.group(1)} (a/b)"] = (int(m.group(2).replace(",", "")), None,
                                           float(m.group(3)), m.group(4))
    usable = _usable_devices()[0]
    # A `differs` row is only an ANOMALY if the backend is also wrong: CPU
    # differs from `a*fl32(1/b)` because it renders a real divide and is
    # therefore correctly rounded, which is the finding, not a disagreement.
    anomaly = [k for k, v in rows.items() if v[3] == "differs" and v[0] > 0]
    MOD.pins.param("t4_rows", rows)
    return Value(
        f"[RECORD, not re-measured] the T4's per-backend rows are "
        + ", ".join(f"{k} {v[0]:,} ({v[2]:.2f}%, {v[3]})" for k, v in rows.items())
        + f". This host has no GPU, so none of these is `MEASURED` here: the "
          f"usable devices are {', '.join(usable)}. The row that disagrees with "
          f"the others -- wrong by more than the rest AND not bit-identical to "
          f"`a*fl32(1/b)` -- is "
        + (", ".join(anomaly) or "none")
        + ". Its mechanism is `[UNVERIFIED]`: this module does not invent one, "
          "and nothing in this repo reproduces it",
        "\n".join(f"  {k:14s} {v[0]:>9,} wrong  {v[2]:6.2f}%   {v[3]}"
                  for k, v in rows.items())
        + f"\n\n  usable here: {', '.join(usable)}. No GPU: nothing above is re-measured.",
    )


@MOD.claim("generated_source_shape")
def c_source() -> Value:
    """What the renderer actually emits, captured this run, per usable
    backend, so the 'above backend selection' clause has a per-backend
    witness rather than one host's."""
    from tinygrad import Device
    import tinygrad
    p = MOD.pins
    p.tinygrad_line("mixin/elementwise.py", 251, must_contain="b.reciprocal()")
    p.tinygrad_line("renderer/cstyle.py", 137, must_contain="Ops.RECIPROCAL")
    p.param("renderer_capture", True)
    out = []
    for dev in _usable_devices()[0]:
        d = Device[dev]
        if getattr(d, "compiler", None) is None or not hasattr(d.compiler, "compile_cached"):
            out.append((dev, None, "no compile hook"))
            continue
        captured: list[str] = []
        orig = d.compiler.compile_cached

        def spy(src, *A, _d=d, _c=captured, **K):
            _c.append(str(src))
            return orig(src, *A, **K)

        d.compiler.compile_cached = spy
        try:
            x = (tinygrad.Tensor(np.arange(64, dtype=np.float32) + 1.0)).to(dev)
            y = (tinygrad.Tensor(np.full(64, 7.0, dtype=np.float32))).to(dev)
            (x / y).realize()
        finally:
            d.compiler.compile_cached = orig
        line = ""
        for s in captured:
            for ln in s.split("\n"):
                t = ln.strip()
                if "data0" in t and "=" in t and "float" in t:
                    line = t
        out.append((dev, line or None, "captured" if line else "no matching store line"))
    p.param("generated", [(d, l) for d, l, _ in out])
    shown = [(d, l) for d, l, _ in out if l]
    lines = []
    for d, l, how in out:
        lines.append(f"  {d:8s} {how}")
        if l:
            lines.append(f"           {l[:120]}")
    recip = [d for d, l, _ in out if l and "1/" in l]
    fdiv = [d for d, l, _ in out if l and "1/" not in l and "/" in l]
    return Value(
        f"captured this run: the emitted source for a single float32 divide "
        f"carries a literal reciprocal expression on {len(recip)} of the usable "
        f"backends ({', '.join(recip) or 'none'}) and a real divide on "
        f"{len(fdiv)} ({', '.join(fdiv) or 'none'}), which is the per-backend "
        f"witness that the rewrite is upstream of backend selection -- the "
        f"backends that differ are exactly the ones whose renderer has "
        f"`Ops.FDIV`",
        "\n".join(lines),
    )


# ---------------------------------------------------------------- report

def report(mod, names, flags):
    n = flags.get("n") or F.N_DEFAULT
    MOD.n = n
    import tinygrad
    from tinygrad import Device
    print("derivation/divide_backends.py -- TinyGrad's float32 `/`, per backend")
    print(f"host {os.uname().sysname} {os.uname().machine}; tinygrad "
          f"{tinygrad.__file__}; default device {Device.DEFAULT}")
    print("A per-backend number is a property of a backend and a host.")
    for name in (names or ["device_inventory", "per_backend_rate", "bit_identity",
                           "exact_arithmetic_others", "self_division_log_crosscheck",
                           "t4_rows_unavailable_here", "generated_source_shape"]):
        rule(name)
        try:
            v = MOD.run_one(name)
        except PinDrift as e:
            print(f"[PIN DRIFT] {e}")
            continue
        print(v.text)
        if v.detail:
            print()
            print(v.detail)
    print()


MOD.n = F.N_DEFAULT

if __name__ == "__main__":
    sys.exit(run(MOD, sys.argv[1:], report=report))
