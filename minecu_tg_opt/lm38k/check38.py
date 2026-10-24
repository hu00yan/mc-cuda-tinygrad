#!/usr/bin/env python3
"""check38.py -- everything about card38.py that can be checked without a GPU,
checked here so a T4 session is not spent finding out.

    1. the assembler and the script agree, and the script compiles
    2. the pinned reference hash is the one notes/35/36 were written against
    3. `lm38_falsify.py` imports, and its pure functions do what they claim
       -- ptxas -v parsing, `.maxntid` rewriting, source facts
    4. the codegen path runs end to end on `NULL:CUDA:sm_75`: the framework's
       real scheduler, real fusion decision and real C renderer, with nothing
       executed. This is the string the card compiles, so it is where a typo in
       an intervention would show up.
    5. the generated CUDA for the discriminator's rung really does carry
       `__launch_bounds__` and its sha256 is stable across rebuilds
    6. `to_program_key` really does exclude the compiler -- the claim
       `clear_program_cache` is built on

    .venv/bin/python minecu_tg_opt/lm38k/check38.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
#: `ops_null.py:28` reads NULL_ALLOW_COPYOUT at realization time, but the device
#: is selected at import time, so DEV has to be set before anything imports
#: tinygrad. The reference's world is built on the CPU and copied in, which is
#: a copyout, hence the flag.
os.environ.setdefault("DEV", "NULL:CUDA:sm_75")
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

say = lambda *x: print(*x, flush=True)
fails: list[str] = []


def ck(name, cond, detail=""):
    say(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        fails.append(name)
    return cond


say("=" * 78)
say("  check38 -- the no-GPU half of the notes/38 experiment")
say("=" * 78)

# ---------------------------------------------------------------- 1, 2
say("")
say("--- 1/2. the assembler and the pinned reference ---")
p = subprocess.run([sys.executable, str(HERE / "build_card38.py"), "--check"],
                   capture_output=True, text=True)
ck("build_card38 --check", p.returncode == 0, (p.stdout or p.stderr).splitlines()[0]
   if (p.stdout or p.stderr) else "")
from minecu_tg_opt.lm38_falsify import (  # noqa: E402
    MAXNTID_RE, Nolocals, PtxBudgetCompiler, clear_program_cache, parse_ptxas_v,
    source_facts)
import hashlib  # noqa: E402
k = REPO / "minecu_ref" / "src" / "kernels.cu"
h = hashlib.sha256(k.read_bytes()).hexdigest()
ck("kernels.cu pinned", h == "c6ed8f06ee64d40a8a8ab7ebce3b3ceea1e3ac93f9ee50cfc1e63d3f090d878f",
   h[:16])

# ---------------------------------------------------------------- 3
say("")
say("--- 3. the instrument's pure functions ---")
V = ("ptxas info    : Compiling entry function 'E_512' for 'sm_75'\n"
     "ptxas info    : Function properties for E_512\n"
     "    1312 bytes stack frame, 1772 bytes spill stores, 1296 bytes spill loads\n"
     "ptxas info    : Used 255 registers, used 0 barriers, 1312 bytes cumulative "
     "stack size, 416 bytes cmem[0]\n")
f = parse_ptxas_v(V)
ck("parse_ptxas_v finds the function", "_hot" in f, str(f.get("_hot")))
ck("parse_ptxas_v stack_frame = 1312", f.get("stack_frame") == 1312, str(f.get("stack_frame")))
ck("parse_ptxas_v spill_total = 3068", f.get("spill_total") == 3068, str(f.get("spill_total")))
ck("spill_total is NOT the stack frame", f.get("spill_total") != f.get("stack_frame"),
   f"{f.get('spill_total')} vs {f.get('stack_frame')}")
ck("parse_ptxas_v registers = 255", f.get("registers") == 255, str(f.get("registers")))

ptx = (".version 8.8\n.target sm_75\n\n.visible .entry E(\n\t.param .u64 p_,\n"
       "\t.param .u64 q_\n)\n.maxntid 128, 1, 1\n{\n\tret;\n}\n")
m = MAXNTID_RE.search(ptx)
ck("MAXNTID_RE matches the real directive", m is not None and m.group(1) == "128",
   m.group(0) if m else "no match")
ck("dropping .maxntid leaves the entry intact",
   ".maxntid" not in MAXNTID_RE.sub("", ptx, count=1) and ".entry E(" in MAXNTID_RE.sub("", ptx, count=1))
sf = source_facts('extern "C" __global__ void __launch_bounds__(128) E_1_2_3('
                  'int* a, float* b, int n) {\n  a[0]=1;\n}\n')
ck("source_facts reads __launch_bounds__", sf["launch_bounds"] == [128], str(sf["launch_bounds"]))
ck("source_facts counts the kernel", sf["n_kernels"] == 1, str(sf))
ck("source_facts sha is stable",
   source_facts("x;\n")["source_sha256"] == source_facts("x;\n")["source_sha256"])

# ---------------------------------------------------------------- 6
say("")
say("--- 6. to_program_key excludes the compiler ---")
src = (REPO / ".venv/lib/python3.12/site-packages/tinygrad/codegen/__init__.py").read_text()
i = src.index("def to_program_key")
ck("to_program_key does not name the compiler",
   "compiler" not in src[i:i + 300].split("return")[1],
   src[i:i + 220].splitlines()[1].strip())
j = src.index("to_program_config = (")
ck("to_program_config does include NOLOCALS", "NOLOCALS" in src[j:j + 300])

# ---------------------------------------------------------------- 4, 5
say("")
say("--- 4/5. the codegen path, on the NULL device ---")
os.environ["DEV"] = "NULL:CUDA:sm_75"
os.environ["NULL_ALLOW_COPYOUT"] = "1"
try:
    from tinygrad.renderer import cstyle
    from tinygrad import Device
    say(f"  Device.DEFAULT = {Device.DEFAULT}")
    _orig = cstyle.CUDARenderer.__init__

    def _keep_nvrtc(self, target, use_nvcc=False):
        return _orig(self, target, use_nvcc=False)   # the stock path: nvrtc
    cstyle.CUDARenderer.__init__ = _keep_nvrtc

    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.codegen import to_program_cache
    from minecu_tg import generate_world
    from minecu_tg.render import camera_basis
    from minecu_tg_opt.render import render_variant, VARIANTS

    world, ground, res, steps, batch = 16, 4, 32, 32, 1024
    one = generate_world(world, ground).numpy()
    vox = Tensor(np.ascontiguousarray(np.stack([one] * batch)), dtype=dtypes.int8)
    cams = np.array([[world / 2.0, float(ground), world / 2.0, 0.001 * (i % 97 - 48), 0.0]
                     for i in range(batch)], dtype=np.float32)
    basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, res, res).contiguous()
    basis.realize()

    from minecu_tg.codegen import install_capture, uninstall
    sink, _ = install_capture()
    try:
        sink.clear()
        g = render_variant(vox, basis, res, res, steps, "flat_gather")
        g.realize()
        srcs = list(sink)
        # sha stability across a rebuild -- the basis of 'same graph size'.
        # The capture hook has to stay installed across BOTH builds: a second
        # install replaces the first, and uninstalling between them leaves the
        # second build with nothing to capture, which looks like "the rebuild
        # compiled nothing" rather than like a mistake in the check.
        to_program_cache.clear()
        sink.clear()
        g2 = render_variant(vox, basis, res, res, steps, "flat_gather")
        g2.realize()
        srcs2 = list(sink)
    finally:
        uninstall(sink)
    ck("the graph renders on the NULL device", len(srcs) > 0, f"{len(srcs)} sources")
    ck("a rebuild recompiles (the cache clear works)", len(srcs2) > 0,
       f"{len(srcs2)} sources")
    big = max(srcs, key=len)
    ck("the fused kernel is the big one", len(big) > 100_000, f"{len(big):,} B")
    ck("the stock CUDA carries __launch_bounds__",
       "__launch_bounds__" in big,
       str(sorted({int(b) for b in re.findall(r'__launch_bounds__\((\d+)\)', big)})))
    lbs = sorted({int(b) for b in re.findall(r'__launch_bounds__\((\d+)\)', big)})
    ck("the launch bound is 128 (so nvrtc emits .maxntid 128, 1, 1)", lbs == [128], str(lbs))
    f1 = source_facts(big)
    say(f"       source: {f1['source_bytes']:,} B, {f1['source_statements']:,} stmts, "
        f"{f1['n_kernels']} kernels, sha {f1['source_sha256'][:16]}")
    say(f"       NOTE: nvrtc is what turns __launch_bounds__ into .maxntid; the "
        f"string above proves the source carries it.")
    big2 = max(srcs2, key=len)
    ck("the same rung rebuilt gives a byte-identical source",
       hashlib.sha256(big2.encode()).hexdigest() == f1["source_sha256"],
       "this is what makes 'graph size is equal' a fact and not an assumption")
    ck("all sources of the graph are identical across rebuilds",
       sorted(hashlib.sha256(s.encode()).hexdigest() for s in srcs) ==
       sorted(hashlib.sha256(s.encode()).hexdigest() for s in srcs2),
       f"{len(srcs)} kernels")
    ck("the whole graph is more than one kernel", len(srcs) > 1,
       f"{len(srcs)} kernels, so 'same graph size' is checked over the whole set")
except Exception as e:
    import traceback
    ck("the codegen path runs", False, f"{type(e).__name__}: {e}")
    say(traceback.format_exc()[-1500:])

say("")
say("=" * 78)
say(f"  {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
say("=" * 78)
sys.exit(0 if not fails else 1)
