#!/usr/bin/env python3
"""lowering.py -- what the framework actually EMITS for each rung, on this host.

WHY THIS FILE EXISTS
====================
A ratio says how much slower one arm is. It does not say *why*, and the four-way
classification in `classify.py` cannot be closed without knowing which of the two
surviving classes (ii) "a cost model could find it" and (iii) "not implemented" a
rung is in -- and that question is about the generated program, not the API.

So this file captures the program. `classify.py` reads the framework's source for
*what exists*; this reads what comes out for *this shape*.

HOW IT CAPTURES, AND WHY IT IS HONEST ABOUT IT
----------------------------------------------
It wraps `Device[...].compiler.compile_cached` and records the `src` handed to the
compiler, then realizes the rung's TinyGrad arm. `compile_cached` is the single
point every device's kernels pass through, so the capture is complete, and it is
observation -- nothing in the framework is modified.

THE LANGUAGE IS NOT THE SAME, AND THE SCHEDULE IS. The run is on **CPU**, so the
captured text is C, not CUDA. That is a real limitation and it is stated at the
point of use: the *renderer* is a backend of tinygrad, but the **schedule** --
how many kernels, what each one's index space is, and the trip counts of its
loops -- is decided before rendering and is the same one the CUDA backend gets.
So the kernel COUNT and the per-kernel WORK are the framework's, and are
quotable; the exact CUDA text is not, and is not claimed. Anything that depends
on the language (register allocation, `#pragma unroll`, shared-memory layout) is
out of scope here by construction.

This costs no GPU quota, which is the point: on this study the two most
informative facts about the largest gap on the ladder both turned out to be
readable from source, and neither was readable from a timing.

USAGE
-----
    DEV=CPU .venv/bin/python axis/lowering.py [rung ...]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("PARALLEL", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from tinygrad import Device, Tensor  # noqa: E402

import rungs as R  # noqa: E402
import verify as V  # noqa: E402

KEYS = list(V.BY_KEY)

# A smaller extent for the batched/matrix rungs, used to get the KERNEL COUNT on
# this host. A batched elementwise or matmul does not gain kernels as the batch
# grows -- the same [B,BC,BC] expression is realized as the same schedule at
# B=8 and B=16384 -- so the count is invariant and a small B answers the
# question. The TIME is emphatically not invariant, and nothing here is used for
# time. Saying so is the point: without it, "measured on the host" would quietly
# license a smaller shape than the note's table quotes.
COUNT_PROBE: dict[str, dict] = {
    "trisolve": {"B": 8, "C": 64, "BC": 16},
    "gemm32":   {"M": 128, "N": 128, "K": 128},
    "gemm16":   {"M": 64, "N": 64, "K": 64},
    "red":      {"R": 64, "K": 256},
    "gather":   {"n": 1 << 16, "table": 1 << 12},
    "ew":       {"n": 1 << 16},
}


# ===========================================================================
# the capture
# ===========================================================================

def capture(key: str, shapes: dict | None = None) -> list[str]:
    """Realize the rung's TinyGrad arm, returning every kernel's source."""
    r = V.BY_KEY[key]
    s = shapes or V.SMALL_SHAPES[key]
    d = r.inputs(s)

    dev = Device[Device.DEFAULT]
    comp = getattr(dev, "compiler", None)
    if comp is None or not hasattr(comp, "compile_cached"):
        raise RuntimeError(f"{Device.DEFAULT} has no compile_cached to wrap; "
                           f"the capture point is not available on this backend")

    got: list[str] = []
    orig = comp.compile_cached

    def wrapped(src, *a, **k):
        got.append(src)
        return orig(src, *a, **k)

    comp.compile_cached = wrapped
    try:
        out = r.tg(s, d)
        if isinstance(out, Tensor):
            out.realize()
    finally:
        comp.compile_cached = orig
    return got


# ===========================================================================
# structural facts, counted from the captured text
# ===========================================================================

def _loops(src: str) -> list[tuple[str, str]]:
    """(name, trip count expression) for every counted loop in the kernel."""
    out = []
    for m in re.finditer(r"for\s*\(\s*int\s+(\w+)\s*=\s*[^;]+;\s*\1\s*<\s*([^;]+);", src):
        out.append((m.group(1), m.group(2).strip()))
    return out


def describe(src: str) -> dict:
    """Structural facts, counted exactly.

    Deliberately no load/store split. Distinguishing a read from a write in
    generated source needs a parser, and a regex that half-works produces a
    number that looks like evidence and is not. `n_buffer_refs` counts every
    dereference of a global buffer, which is countable and is the traffic proxy
    that matters here; whether each one reads or writes is not claimed.
    """
    loops = _loops(src)
    return {
        "lines": len(src.splitlines()),
        "n_loops": len(loops),
        "loops": loops,
        "n_buffer_refs": len(re.findall(r"\*\((?:float\d\*\)?\()?data", src)),
        "has_float4": "float4" in src,
        "has_unroll_pragma": "unroll" in src.lower(),
        "ranges": [t for _, t in loops],
    }


def print_kernel(i: int, src: str) -> None:
    d = describe(src)
    print(f"  --- kernel {i}: {d['lines']} lines, {d['n_loops']} loops, "
          f"{d['n_buffer_refs']} buffer refs"
          f"{', float4' if d['has_float4'] else ''} ---")
    for name, trip in d["loops"]:
        print(f"        for {name} < {trip}")
    for line in src.splitlines():
        print(f"      {line}")


# ===========================================================================
# the specific claim this file exists to check
# ===========================================================================

def cumsum_lowering_facts() -> list[dict]:
    """Read `_split_cumalu` out of the framework's source and report its constants.

    This is the part of the scan rung that decides class (ii) versus class (iii).
    Whether the scan's decomposition is CHOSEN or HAND-PLACED is visible in the
    source, and it is the whole question: a hand-placed constant with the
    framework's own TODO attached is (ii) -- a cost model could find this -- and
    not (iii), which would be an op that does not exist.
    """
    import tinygrad.mixin.op as mop
    p = Path(mop.__file__)
    lines = p.read_text().splitlines()
    rows = []
    for i, ln in enumerate(lines, 1):
        if "def cumsum" in ln or "def _split_cumalu" in ln or "def _cumalu" in ln:
            rows.append({"line": i, "text": ln.strip(), "what": "definition"})
        elif re.match(r"\s*SPLIT\s*=", ln):
            rows.append({"line": i, "text": ln.strip(), "what": "the constant"})
        elif "someday the optimizer" in ln or "two stage" in ln:
            rows.append({"line": i, "text": ln.strip(), "what": "the TODO"})
    return [{"file": f"{p.relative_to(Path(mop.__file__).parents[2])}:{r['line']}",
             "what": r["what"], "text": r["text"]} for r in rows]


# ===========================================================================

def main(argv: list[str]) -> int:
    keys = argv[1:] or KEYS
    print("=" * 78)
    print("axis/lowering.py -- what the framework EMITS, captured on this host.")
    print("  device:", Device.DEFAULT, "| the TEXT is this backend's language;")
    print("  the SCHEDULE (kernel count, index spaces, trip counts) is the")
    print("  framework's and is what the note quotes.")
    print("=" * 78)

    print("\n== the scan lowering, read from source: is the decomposition chosen "
          "or hand-placed? ==")
    for row in cumsum_lowering_facts():
        print(f"  {row['file']:<28} [{row['what']}]")
        print(f"      {row['text']}")

    print("\n== per rung: the kernel COUNT the framework emits ==")
    print("   gate shape      the shape verify.py's CPU gate runs")
    print("   count probe     a smaller extent; the COUNT is batch-invariant,")
    print("                   the TIME is not, and no time is taken here")
    print("   bench shape     the shape the card measured, with the bodies")
    for key in keys:
        r = V.BY_KEY[key]
        tiers = [(V.SMALL_SHAPES[key], "gate shape"),
                 (COUNT_PROBE.get(key, V.SMALL_SHAPES[key]), "count probe")]
        if key in ("scan", "gather", "ew", "red"):
            tiers.append((V.SHAPES[key], "bench shape"))
        # dedupe tiers that resolve to the same shape, so a rung whose probe
        # equals its gate shape is reported once instead of twice
        seen, uniq = set(), []
        for shapes, tag in tiers:
            key_s = tuple(sorted(shapes.items()))
            if key_s in seen:
                continue
            seen.add(key_s)
            uniq.append((shapes, tag))
        tiers = uniq
        print(f"\n== {key} -- {r.title} ==")
        for shapes, tag in tiers:
            try:
                srcs = capture(key, shapes)
            except Exception as e:
                print(f"  [{tag:<12}] {shapes} capture FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            print(f"  [{tag:<12}] {shapes} -> {len(srcs)} kernel(s)")
            if tag == "bench shape":
                for i, src in enumerate(srcs):
                    print_kernel(i, src)
            else:
                for i, src in enumerate(srcs):
                    d = describe(src)
                    print(f"                 kernel {i}: {d['lines']} lines, "
                          f"{d['n_loops']} loops, {d['n_buffer_refs']} buffer refs"
                          f"{', float4' if d['has_float4'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
