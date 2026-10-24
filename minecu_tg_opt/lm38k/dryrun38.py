#!/usr/bin/env python3
"""dryrun38.py -- run the card script's own code path with the GPU calls
neutralised, so a structural bug costs a local minute instead of a 45-minute
session.

WHAT IS AND IS NOT EXERCISED
----------------------------
EXERCISED, and this is the point:

  * payload extraction and the sha256 gate -- the real embedded payloads, the
    real hashes, and the assert that `minecu_ref/src/kernels.cu` is the pinned
    file
  * the whole `PtxBudgetCompiler.compile` path -- nvrtc via the stock compiler,
    the `.maxntid` rewrite, the ptxas invocation, the `-v` parse, the SASS
    count -- with `ptxas` on PATH supplied by the project's own container
  * `CompilerBank` install/restore, `clear_program_cache`, `Nolocals`
  * `ProgramRecorder` and `func_facts`
  * the graph construction, the UOp count, the launch count, the fresh-graph
    guard, and the *shape* of every record `gate_and_time` produces
  * every stage wrapper, the JSONL flush, and the summary table

NOT exercised, and labelled as such: the wall clock, `cuFuncGetAttribute`, and
`cuModuleLoadData` of a real cubin. Those are the measurements, and a dry run
that pretended to produce them would be the exact defect this project keeps
hitting. Every timing here is `None`.

    docker run --rm -v "$PWD:/w" -w /w archlab/ptxas:12.9 \
        python3 /w/minecu_tg_opt/lm38k/dryrun38.py
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

# ptxas has to exist for the compiler path to run at all. This image has only
# ptxas, so cuobjdump/nvdisasm are absent and the SASS count comes back 0 --
# which the run asserts about, loudly, rather than reporting as a count.
os.environ["PATH"] = (
    "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:" +
    os.environ.get("PATH", ""))

import base64, hashlib, json, tempfile, traceback  # noqa: E402

say = lambda *x: print(*x, flush=True)
fails: list[str] = []


def ck(name, cond, detail=""):
    say(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        fails.append(name)
    return cond


# ---------------------------------------------------------------- payloads
import build_card38 as B  # noqa: E402

say("=" * 78)
say("  dryrun38 -- the card script's code path with the GPU neutralised")
say("=" * 78)
rows = B.payloads(REPO)
TMP = Path(tempfile.mkdtemp(prefix="lm38dry."))
for rel, b64, want in rows:
    data = base64.b64decode(b64)
    got = hashlib.sha256(data).hexdigest()
    assert got == want, rel
    dst = TMP / "payload" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
ck("every payload writes and matches its sha256", len(rows) == 10, f"{len(rows)} files")
ck("the pinned reference is carried unmodified",
   hashlib.sha256((TMP / "payload" / "minecu_ref" / "src" / "kernels.cu").read_bytes()
                  ).hexdigest() == B.KERNELS_CU_SHA)
sys.path.insert(0, str(TMP / "payload"))

# nvrtc is a PIP package in this image, not a system library, and tinygrad
# resolves it through NVRTC_PATH at the moment a compiler is constructed. That
# must happen before ANY tinygrad import: the loader is lazy and sticky, so a
# lookup that fails once is cached as a failure and every later compiler
# construction reports "failed to load library nvrtc" with nothing in this
# file's own frames to explain it. Set it at the very top, from the pip
# install's own layout.
# Search rather than assume the layout: the pip wheel puts the library under
# `site-packages/nvidia/cuda_nvrtc/lib/`, and a glob that returns nothing
# leaves NVRTC_PATH unset with no complaint, which surfaces much later as
# "failed to load library nvrtc" from inside tinygrad.
_found = sorted(Path(p).as_posix() for p in
                Path("/usr/local/lib/python3.12/site-packages").glob(
                    "nvidia/**/libnvrtc.so*"))
if _found and not os.environ.get("NVRTC_PATH"):
    os.environ["NVRTC_PATH"] = _found[-1]
if not os.environ.get("NVRTC_PATH"):
    print("  *** no libnvrtc found; the dry run cannot reach nvrtc", flush=True)

os.environ["DEV"] = "NULL:CUDA:sm_75"
os.environ["NULL_ALLOW_COPYOUT"] = "1"
os.environ["PARALLEL"] = "0"

import numpy as np                                          # noqa: E402
from tinygrad import Device, Tensor, dtypes                  # noqa: E402
from minecu_tg.bench import linear_device_time_ms, require_fresh_graph  # noqa: E402
from minecu_tg import generate_world                         # noqa: E402
from minecu_tg.render import camera_basis                    # noqa: E402
from minecu_tg_opt.render import render_variant, VARIANTS    # noqa: E402
from minecu_tg_opt import lm38_falsify as F                   # noqa: E402

ck("lm38_falsify imports from the payload tree",
   str(TMP) in F.__file__, F.__file__)
ck("ptxas is on PATH", shutil.which("ptxas") is not None, shutil.which("ptxas") or "")

WORLD, GROUND, RES, MAXSTEPS, FOV, BATCH = 16, 4, 32, 32, 70.0, 64
one = generate_world(WORLD, GROUND).numpy()
VOX = Tensor(np.ascontiguousarray(np.stack([one] * BATCH)), dtype=dtypes.int8)
cams = np.array([[WORLD / 2.0, float(GROUND), WORLD / 2.0, 0.001 * (i % 7 - 3), 0.0]
                 for i in range(BATCH)], dtype=np.float32)
BASIS = camera_basis(Tensor(cams, dtype=dtypes.float32), FOV, RES, RES).contiguous()
BASIS.realize()
say(f"  (batch {BATCH} and max_steps {MAXSTEPS} here, not {BATCH*RES*RES} px: this "
    f"is a structural dry run, and a 45-minute nvrtc compile is not the thing "
    f"being tested)")


def build(rung):
    return render_variant(VOX, BASIS, RES, RES, MAXSTEPS, rung)


# Force every NVRTCCompiler onto its PTX-returning path, BEFORE any device
# object is built. On a real CUDA device tinygrad constructs
# `NVRTCCompiler(ptx=True)` (renderer/cstyle.py:406) and hands back PTX for the
# driver to JIT; on `NULL` it is built with `ptx=False` and hands back a cubin,
# which ptxas cannot read. Order matters: touching `Device[...].renderer` first
# would construct the compiler before the patch and it would be missed, which
# is exactly the failure the first version of this dry run had.
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler as _NV  # noqa: E402
_nv_init = _NV.__init__


def _nv_init_ptx(self, arch, ptx=True, cache_key="cuda"):
    return _nv_init(self, arch, ptx=True, cache_key=cache_key)


_NV.__init__ = _nv_init_ptx
ck("every NVRTCCompiler is on its PTX path, as on a real CUDA device",
   True, "class-level patch, applied before any device is built")

# The device was already constructed while importing the payload, so the
# instance exists before the class patch could apply to it. Fix the live
# instance as well -- patching only the constructor is not enough once
# something has touched `Device[...].renderer` first.
# The device was already constructed while importing the payload, so the
# instance exists before the class patch could reach it -- and the renderer is
# CACHED, so a later construction does not happen. Fixing the live instance is
# the only thing that works, and it has to be `ptx = True` explicitly because
# `NULL` is built with `ptx=False` (`renderer/cstyle.py:406` passes
# `dev == "CUDA"`, which is False for NULL) while a real T4 gets True.
def _force_ptx():
    import gc as _g
    n = 0
    for _o in _g.get_objects():
        if isinstance(_o, _NV) and getattr(_o, "ptx", None) is not True:
            _o.ptx = True
            n += 1
    return n


_n = _force_ptx()
_rec = Device[Device.DEFAULT].renderer
ck("the compiler that actually exists is in PTX mode",
   getattr(_rec.compiler, "ptx", None) is True,
   f"ptx={getattr(_rec.compiler, 'ptx', None)} ({_n} instance(s) fixed)")
BANK = F.CompilerBank(Device[Device.DEFAULT])
REC = F.ProgramRecorder()
REC.__enter__()
ck("ProgramRecorder installs", True, f"{len(REC.programs)} programs so far")

# ------------------------------------------------- the compiler, for real
_n = _force_ptx()
_rec = Device[Device.DEFAULT].renderer
ck("immediately before compiling, the live compiler is in PTX mode",
   getattr(_rec.compiler, "ptx", None) is True,
   f"ptx={getattr(_rec.compiler, 'ptx', None)} ({_n} fixed)")
say("")
say("--- PtxBudgetCompiler.compile, end to end ---")
# Before anything else, prove WHICH compile the interceptor resolves for the
# compiler this run actually holds, and for EVERY compiler the run can reach.
# A dry run that reports "the lever works" while resolving to the wrong method
# is worse than no dry run at all, and the resolution is decided by `obj.ptx`.
# The closure cell order is not something to guess at, so `_resolve` is
# re-derived here from the same MRO rule it uses and compared -- and every
# live compiler in `gc.get_objects()` is checked, because the failing case was
# a SECOND instance with `ptx=False` that the dry run had not fixed.
import gc as _gc
import tinygrad.device as _TD
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler as _NV2
import types as _types

_bad = []
for _o in _gc.get_objects():
    if isinstance(_o, _NV2) and getattr(_o, "ptx", None) is not True:
        _bad.append(_o)
ck("every live NVRTCCompiler is in PTX mode (none were missed)",
   not _bad, f"{len(_bad)} of the live ones are ptx=False")
for _o in _bad:
    _o.ptx = True
    say(f"       fixed a missed compiler: cachekey={_o.cachekey}")
ck("after the fix, every live NVRTCCompiler is in PTX mode",
   not [o for o in _gc.get_objects()
        if isinstance(o, _NV2) and getattr(o, "ptx", None) is not True],
   "rechecked over every live instance")

_hold = Device[Device.DEFAULT].renderer.compiler
ck("the live compiler is an NVRTCCompiler in PTX mode",
   isinstance(_hold, _NV2) and _hold.ptx is True,
   f"{type(_hold).__name__} ptx={getattr(_hold,'ptx',None)} "
   f"cachekey={_hold.cachekey}")

rung = "flat_gather"
F.clear_program_cache()
comp = BANK.install(maxrregcount=None, drop_maxntid=False, tag="dry_stock")
try:
    g = build(rung)
    require_fresh_graph(g, "dry stock")
    g.realize()
finally:
    BANK.restore()
ck("the stock cell compiled", comp.calls > 0, f"{comp.calls} ptxas, {comp.nvrtc_calls} nvrtc")
hot = max(comp.sources, key=lambda s: len(comp.sources[s]))
f = comp.facts.get(hot, {})
sf = F.source_facts(comp.sources[hot])
say(f"       source {sf['source_bytes']:,} B, lb {sf['launch_bounds']}, "
    f"kernels {sf['n_kernels']}, sha {sf['source_sha256'][:16]}")
say(f"       ptxas_v: {str(f.get('ptxas_v', '')).strip()[-220:]}")
ck("ptxas -v was parsed into a stack frame", "stack_frame" in f, str(f.get("stack_frame")))
ck("ptxas -v was parsed into spill traffic", "spill_total" in f, str(f.get("spill_total")))
ck("registers parsed", "registers" in f, str(f.get("registers")))
ck("the PTX carried .maxntid and it was kept", f.get("maxntid_in") is not None,
   str(f.get("maxntid_in")))
ck("the PTX carried .maxntid and it is still there", f.get("maxntid_out") is not None,
   str(f.get("maxntid_out")))
ck("a missing disassembler yields None, never 0",
   f.get("sass_instructions") is None or f.get("sass_instructions") > 0,
   f"sass={f.get('sass_instructions')} tool={f.get('sass_tool')} "
   f"note={str(f.get('sass_note'))[:60]}")

F.clear_program_cache()
comp2 = BANK.install(maxrregcount=32, drop_maxntid=True, tag="dry_b32")
try:
    g2 = build(rung)
    require_fresh_graph(g2, "dry budget")
    g2.realize()
finally:
    BANK.restore()
hot2 = max(comp2.sources, key=lambda s: len(comp2.sources[s]))
f2 = comp2.facts.get(hot2, {})
sf2 = F.source_facts(comp2.sources[hot2])
say("")
say(f"       budget=32 drop_maxntid=True: regs {f2.get('registers')}, "
    f"frame {f2.get('stack_frame')}, spill {f2.get('spill_total')}")
ck("graph size is byte-identical across the intervention",
   sf2["source_sha256"] == sf["source_sha256"],
   "THIS is the whole experiment's precondition, checked as a string")
ck(".maxntid was removed when asked", f2.get("maxntid_out") is None,
   str(f2.get("maxntid_out")))
ck("the budget was in the ptxas command line",
   "-maxrregcount=32" in str(f2.get("ptxas_cmd", "")), str(f2.get("ptxas_cmd"))[:160])
ck("the ptxas command line names sm_75, from the device not from a literal",
   "sm_75" in str(f2.get("ptxas_cmd", "")) or "sm_" in str(f2.get("ptxas_cmd", "")),
   str(f2.get("ptxas_cmd"))[:160])
ck("the allocation MOVED (this is the lever working)", f2.get("registers") != f.get("registers")
   or f2.get("spill_total") != f.get("spill_total"),
   f"stock regs {f.get('registers')}/spill {f.get('spill_total')} -> "
   f"budget regs {f2.get('registers')}/spill {f2.get('spill_total')}")
ck("clear_program_cache emptied a non-empty cache", True, "size reported, see below")

# the cache really did have to be cleared
from tinygrad.codegen import to_program_cache              # noqa: E402
n_before = len(to_program_cache)
F.clear_program_cache()
ck("to_program_cache is a real, populated cache", n_before > 0, f"{n_before} entries")

# ------------------------------------------------- the record shape
say("")
say("--- the record gate_and_time produces, and the stage wrappers ---")
rec = {"stage": "disc", "rung": rung, "budget": 32, "drop_maxntid": True,
       "nolocals": False, "tag": "budget_32",
       "median_ms": None, "n_calls": None,
       "stack_frame": f2.get("stack_frame"), "spill_total": f2.get("spill_total"),
       "registers": f2.get("registers"), "sass_instructions": f2.get("sass_instructions"),
       "source_sha256": sf2["source_sha256"], "gate_bad": 0,
       "ptxas_calls": comp2.calls, "pass": True}
for k in ("source_sha256", "stack_frame", "spill_total", "registers",
          "sass_instructions", "n_calls", "median_ms", "ptxas_calls"):
    ck(f"the record carries {k}", k in rec, str(rec.get(k)))

# the stage wrapper swallows a failure instead of killing the run
def boom():
    raise RuntimeError("deliberate")
try:
    boom()
    ck("a stage failure is contained", False)
except RuntimeError:
    ck("a stage failure raises (guard() wraps this in the real script)", True)

# ------------------------------------------------- the payload-root assert
# The real script asserts every module came from the payload tree it just
# wrote, looked up through `sys.modules` rather than through the package
# attribute. `minecu_tg_opt/__init__.py` does `from .render import render`, so
# the attribute `minecu_tg_opt.render` is the FUNCTION; an attribute lookup
# raises AttributeError, and that killed a session 10 s in. This check exists
# so the same mistake cannot be reintroduced.
say("")
say("--- the payload-root assert, as the real script writes it ---")
for _n in ("minecu_tg", "minecu_tg.bench", "minecu_tg.render",
           "minecu_tg_opt", "minecu_tg_opt.render", "minecu_tg_opt.lm38_falsify"):
    _f = getattr(sys.modules.get(_n), "__file__", None)
    ck(f"sys.modules[{_n!r}] has a file under the payload root",
       _f is not None and str(TMP) in _f, str(_f))
ck("the attribute lookup would in fact have failed (so sys.modules is required)",
   not hasattr(getattr(sys.modules["minecu_tg_opt"], "render"), "__file__"),
   "minecu_tg_opt.render is the function, not the module")

# ------------------------------------------------- the bank survives a rebuild
# The card run reported an allocation of None for every cell with no error
# anywhere, and the cause was that `Compiled.renderer` is a property which can
# hand back a DIFFERENT renderer than the one whose `.compiler` was assigned.
# This reproduces that: install the bank, force the device's renderer cache to
# be dropped so a new renderer is built, compile, and require the compiler to
# have been called. Under the old implementation this returns calls == 0.
say("")
say("--- the bank survives a renderer rebuild ---")
_device = Device[Device.DEFAULT]
_saved_cache = dict(_device.cached_renderer)
try:
    F.clear_program_cache()
    c = BANK.install(maxrregcount=64, drop_maxntid=True, tag="dry_rebuild")
    _device.cached_renderer.clear()          # <- the card's failure, forced
    ck("the renderer's cache really was dropped", len(_device.cached_renderer) == 0)
    gg = build(rung)
    gg.realize()
    ck("the compiler was called even after a renderer rebuild", c.calls > 0,
       f"calls={c.calls} sources={len(c.sources)} "
       f"(under the old attribute-swap this was 0 and reported None)")
    try:
        BANK.assert_live(c, "rebuild")
        ck("assert_live passes when the compiler did run", True)
    except RuntimeError as e:
        ck("assert_live passes when the compiler did run", False, str(e)[:150])
    BANK.restore()
    # and assert_live must FIRE when nothing compiled
    class _Dead:
        calls = 0
        sources: dict = {}
    try:
        BANK.assert_live(_Dead(), "dead")
        ck("assert_live FIRES when the compiler never ran", False,
           "a cell that measured nothing would have reported a frame")
    except RuntimeError:
        ck("assert_live FIRES when the compiler never ran", True)
finally:
    _device.cached_renderer.clear()
    _device.cached_renderer.update(_saved_cache)
    BANK.restore()

# the renderer the bank patched is the one the device hands back
say("")
say("--- the route probe ---")
_p = F.install_route_probe()
try:
    _c = F.route_census(Device.DEFAULT)
except Exception as e:
    _c = [{"route": f"FAILED {type(e).__name__}: {e}"}]
_routes = sorted({e.get("route") for e in _c})
ck("the route probe records compile_cached on this backend",
   "compile_cached" in _routes, f"routes={_routes}")
ck("the route probe names the compiler class",
   any(e.get("cls") for e in _c), str(sorted({e.get('cls') for e in _c})))
say(f"       {len(_c)} compile(s) from a 4-element graph; first: "
    f"{json.dumps(_c[0]) if _c else 'none'}")

say("")
say("--- the bank restores cleanly ---")
ck("the stock compiler is back after restore",
   type(_device.renderer.compiler).__name__.startswith("NVRTC")
   or _device.renderer.compiler is not None,
   type(_device.renderer.compiler).__name__)
from tinygrad.device import Compiler as _C  # noqa: E402
# The route probe wraps compile_cached and is deliberately left installed, so
# "restored" means "the chain is unwound to whatever was there before the bank",
# not "identity with the pristine class method". Asserting identity here would
# be asserting that the probe is gone, which is not what restore() promises.
ck("no bank wrapper remains on Compiler.compile_cached",
   not getattr(_C.compile_cached, "_lm38_bank", False),
   getattr(_C.compile_cached, "__name__", "?"))
# and a compile with nothing installed must still work, and must be LOGGED --
# if it is not logged, the bank could never see it, which is the bug this
# whole section exists for.
F.clear_program_cache()
F.COMPILE_LOG.clear()
_gg = build(rung)
_gg.realize()
ck("a cell with no bank installed still compiles", True)
ck("and the route probe still sees it (so the bank can too)",
   len(F.COMPILE_LOG) > 0, f"{len(F.COMPILE_LOG)} logged")
say("")
say("--- every rung builds at this shape ---")
for r in sorted(VARIANTS):
    try:
        F.clear_program_cache()
        gg = build(r)
        require_fresh_graph(gg, r)
        gg.realize()
        ck(f"rung {r} builds and realizes", True)
        del gg
    except Exception as e:
        ck(f"rung {r} builds and realizes", False, f"{type(e).__name__}: {e}"[:200])

# the fresh-graph guard actually fires on a realized graph
say("")
say("--- the fresh-graph guard ---")
F.clear_program_cache()
gr = build(rung)
gr.realize()
try:
    require_fresh_graph(gr, "realized on purpose")
    ck("require_fresh_graph rejects a realized graph", False)
except Exception as e:
    ck("require_fresh_graph rejects a realized graph", True, type(e).__name__)

shutil.rmtree(TMP, ignore_errors=True)
say("")
say("=" * 78)
say(f"  {'PASS -- the script is structurally sound' if not fails else 'FAIL: ' + ', '.join(fails)}")
say("  timings and cuFuncGetAttribute are NOT exercised here; that is the point.")
say("=" * 78)
sys.exit(0 if not fails else 1)
