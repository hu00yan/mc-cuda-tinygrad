#!/usr/bin/env python3
"""why38.py -- why is -maxrregcount inert on this workload?  (OFFLINE)

`design38.py` found that on `flat_gather`'s captured PTX, nineteen different
`-maxrregcount` settings produced ONE cubin: 255 registers, a 1 312-byte frame,
481 960 bytes of code, every time. If that is true then the register budget is
not a lever on this workload at all, and an experiment built on it measures
nothing.

The suspicion is `.maxntid`. The PTX that the framework's own path produces
carries `.maxntid 128, 1, 1` and no `.maxnreg`, and ptxas may be resolving the
register ceiling from the block size and ignoring the command line. This script
separates the two possibilities with a controlled 2x4:

    {kernel needing ~200 registers} x {no .maxntid, .maxntid 128}

and reports ptxas's own -v line for each cell, plus the cubin's sha256 so
"identical" is a fact and not an impression.

Every number here is OFFLINE and from CUDA 12.9. It decides which lever the
on-card experiment can use; it is not quoted as a measurement in notes/38.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PTXAS = "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas"
NVCC = None  # not in this image; PTX is hand-written below

FRAME_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
REG_RE = re.compile(r"Used (\d+) registers")

say = lambda *x: print(*x, flush=True)

#: ~200 live values, all of which are used after the point they are defined, so
#: an honest allocator needs well over 128 registers to keep them all. This is
#: the same shape as the workload's fused region: one straight-line body, a wide
#: live set, and no loop for the allocator to lean on.
def make_ptx(directive: str, nlive: int = 200) -> str:
    """A straight-line body with `nlive` simultaneously-live floats.

    This is the shape the experiment is actually about: ONE region, no loop for
    the register allocator to lean on, and a live set wide enough that keeping
    everything in registers is impossible at a small budget. Every value is
    defined early and consumed late, so the live ranges all overlap -- which is
    what forces a spill, as opposed to merely having a lot of code.

    Generated rather than hand-written because a hand-written 200-register PTX
    body is exactly the kind of long literal that has an off-by-one in its
    declared `.reg` size, and that failure mode is quiet: ptxas prints "Unknown
    symbol" and every cell comes back empty, which reads as "the lever does
    nothing" if you do not print the error.

    The parameter is declared as `.param .u64 p_` with the trailing underscore,
    which is what nvcc emits and what ptxas's ABI requires: a bare `%p` name
    collides with the ABI's own register state space. Getting that wrong makes
    every cell fail identically for a reason that has nothing to do with the
    register budget.
    """
    L = [".version 8.8", ".target sm_75",
         ".visible .entry k(.param .u64 p_)", "{"]
    # `.maxntid` and the `.reg` declarations both go INSIDE the entry body.
    # At module scope `.reg` is an ABI error and `.maxntid` is a parse error,
    # and both fail identically at every budget -- which looks exactly like
    # "the lever does nothing" if the error is not printed.
    if directive:
        L.append(f"  {directive}")
    L += [f"  .reg .f32  %f<{nlive}>;",
          "  .reg .b32  %r<8>;",
          "  .reg .b64  %rd<8>;",
          "  mov.u64 %rd1, p_;"]
    # Every value is loaded, and then every value is consumed only at the very
    # end, by a single sum. That is what makes the live ranges OVERLAP: a value
    # loaded at the top is still needed at the bottom, so the allocator cannot
    # reuse its register and must either keep all `nlive` of them or spill.
    #
    # The first version of this generator consumed the values pairwise with
    # `fma` as it went. ptxas turned the whole thing into a 7-register kernel:
    # a pairwise fold has a live set of TWO, no matter how long it is. A wide
    # live set needs a wide *join*, not a long chain, and that distinction is
    # the whole reason the register budget is worth sweeping.
    for i in range(nlive):
        L.append(f"  ld.global.f32 %f{i}, [%rd1+{4 * i}];")
    for i in range(2, nlive):
        L.append(f"  add.f32 %f1, %f1, %f{i};")
    L += [f"  st.global.f32 [%rd1+{4*nlive}], %f1;", "  ret;", "}"]
    return "\n".join(L) + "\n"


CASES = {
    "no_maxntid": "",
    "maxntid_128": ".maxntid 128, 1, 1",
}
BUDGETS = [None, 200, 160, 128, 96, 64, 40, 32]


def one(ptx_text: str, mrr, td: Path) -> dict:
    p = Path(td) / f"k_{mrr}.ptx"
    c = Path(td) / f"k_{mrr}.cubin"
    p.write_text(ptx_text)
    cmd = [PTXAS, "-arch=sm_75", "-v", "-O3"]
    if mrr is not None:
        cmd.append(f"-maxrregcount={mrr}")
    cmd += ["-o", str(c), str(p)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    txt = (r.stderr or "") + (r.stdout or "")
    fr, rg = FRAME_RE.search(txt), REG_RE.search(txt)
    return {
        "budget": mrr,
        "rc": r.returncode,
        "stack_frame": int(fr.group(1)) if fr else None,
        "spill_stores": int(fr.group(2)) if fr else None,
        "spill_loads": int(fr.group(3)) if fr else None,
        "registers": int(rg.group(1)) if rg else None,
        "cubin_sha256": hashlib.sha256(c.read_bytes()).hexdigest()[:16] if c.exists() else None,
        "err": txt[-600:] if (r.returncode or fr is None) else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / ".agent-scratch" / "lm38b")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    v = subprocess.run([PTXAS, "--version"], capture_output=True, text=True)
    say("=" * 78)
    say("  why38 -- is -maxrregcount inert, and is .maxntid the reason?  (OFFLINE)")
    say("=" * 78)
    say(f"  ptxas {(v.stdout or v.stderr).strip().splitlines()[0]}")
    say("")
    rep = {"ptxas": (v.stdout or v.stderr).strip(), "cases": {}}
    for name, directive in CASES.items():
        text = make_ptx(directive)
        say(f"--- directive: {name or '(none)'} ---")
        say(f"    {'budget':>7} {'frame B':>9} {'spill st':>9} {'spill ld':>9} "
            f"{'regs':>5}  cubin sha")
        rows = []
        with tempfile.TemporaryDirectory(dir=str(a.out)) as td:
            for mrr in BUDGETS:
                r = one(text, mrr, Path(td))
                rows.append(r)
                say(f"    {str(mrr):>7} {str(r['stack_frame']):>9} "
                    f"{str(r['spill_stores']):>9} {str(r['spill_loads']):>9} "
                    f"{str(r['registers']):>5}  {r['cubin_sha256']}")
                if r["err"]:
                    say(f"        !! {r['err'][-300:]}")
        rep["cases"][name] = rows
        regset = sorted({r["registers"] for r in rows if r["registers"]})
        frmset = sorted({r["stack_frame"] for r in rows if r["stack_frame"] is not None})
        shas = {r["cubin_sha256"] for r in rows}
        say(f"    -> {len(regset)} distinct register counts {regset}; "
            f"{len(frmset)} distinct frames {frmset}; "
            f"{len(shas)} distinct cubins")
        say("")
    (a.out / "why38.json").write_text(json.dumps(rep, indent=1, default=str))
    say(f"wrote {a.out / 'why38.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
