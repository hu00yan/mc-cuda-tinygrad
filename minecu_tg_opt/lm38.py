"""lm38.py -- the register-allocation falsifier for `notes/35` sections 5 and 6.

WHAT THIS MODULE IS FOR
-----------------------
`notes/35` section 5 says, on the strength of `notes/36`:

    Graph size is upstream of the cost, but the channel is register allocation of
    the fused straight-line region -- not the launch budget, and not the
    instruction count.

and section 6 says the same thing from the other side of the abstraction:
crossing the framework's boundary is free until it spills and costs 2.4x once
it does. Both rest on one quantity -- ptxas's local-memory frame for the fused
kernel -- ordering five points' wall clock at Pearson +0.9986.

**Five points is a correlation.** A single quantity that orders five points is
also what you get when the quantity is a symptom of something else that all
five rungs also move with. Graph size is the obvious such something: it is
upstream of the frame by construction. This module exists to build the case
that separates them --

  * **same graph size** (the emitted CUDA is byte-identical, so the UOp count,
    the fusion decision and the instruction stream are identical too), and
  * **a different frame**,

and then to time both. Under the register-allocation reading the two differ in
wall clock; under the "frame is a symptom of graph size" reading they are equal.

THE LEVER, AND WHY IT IS THE RIGHT ONE
---------------------------------------
The only knob that moves the register allocation of a fused region without
touching the graph is the one ptxas itself exposes: `-maxrregcount`. It is a
property of the *compilation*, not of the program. So for every setting:

  * the CUDA tinygrad's renderer emits is **the same string** (verified, sha256,
    recorded per cell), therefore
  * the UOp count, the fusion decision, the number of launches, the op mix and
    the SASS *instruction stream the compiler sees* are the same string's
    properties and are the same, and
  * only the register budget changes.

`__launch_bounds__` is the framework's own handle on the same quantity --
TinyGrad's `NOLOCALS` knob sets it from 128 to 1 -- so it is swept too, which
gives a two-factor grid over the two things the mechanism implicates
(registers, and the local frame they spill into).

WHAT IS AND IS NOT CLAIMED ABOUT A KNOB-CLOBBERED KERNEL
---------------------------------------------------------
Nothing here is a port. `minecu_tg_opt/render.py` is not modified, and no
figure produced under a knob is a candidate answer to "how fast can this be
ported". The knobs are used only as *instruments*: to move the claimed channel
and see whether time follows. That is the whole question `notes/35` section 5
leaves open, and it is not answerable from the five-rung ladder alone.

Every rung this module times is gated byte-for-byte against the hand-written
driver's own output on the card first. A wrong register allocation cannot pass
that gate, so the gate is also the correctness check on the intervention.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

# The frame/register numbers are read out of ptxas's own `-v` report by the
# project's parser rather than parsed a second time here. Importing it is the
# point: two parsers of the same report would be two chances to be wrong.
from minecu_tg_opt.opcount import (  # noqa: E402
    count_instructions, parse_ptxas_v, split_functions,
)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# the compiler: stock nvrtc to PTX, then ptxas with an explicit register budget
# ---------------------------------------------------------------------------
class PtxCappedCompiler:
    """nvrtc (stock options) -> PTX -> `ptxas` -> cubin, with `-v` captured.

    WHY NOT JUST PASS THE FLAG TO nvrtc. TinyGrad's CUDA backend hands PTX to
    the driver's JIT (`CUDARenderer.compiler` is an `NVRTCCompiler` with
    `ptx=True` for `dev == "CUDA"`, and `CUDAProgram` calls `cuModuleLoadData`
    on what comes back). A `-maxrregcount` handed to nvrtc is therefore a claim
    about a compilation this process never performs and never sees, so the
    frame it produced could not be reported for the binary that was timed.
    Doing the ptxas step here makes three things true at once:

      * the cubin that is loaded is the cubin `ptxas -v` just described, so the
        frame reported is the frame of the kernel being timed and not of a
        separate offline compilation of a possibly-different string;
      * the flag's effect is visible rather than assumed, and
      * the stock path stays available as a control -- run the same wrapper with
        no flag at all, and the numbers have to agree with the stock run or the
        wrapper is not measuring the same thing.

    The PTX itself is produced by the *stock* nvrtc invocation with the *stock*
    options, so the only difference from the stock path is that ptxas runs in
    this process instead of inside the driver.
    """

    def __init__(self, base, arch: str, maxrregcount: int | None = None,
                 extra: tuple[str, ...] = (), tag: str = ""):
        self.base, self.arch = base, arch
        self.maxrregcount = maxrregcount
        self.extra = [e for e in extra]
        self.tag = tag or f"mrr{maxrregcount if maxrregcount else 'off'}"
        # A cache key that cannot collide with the stock one, and cannot collide
        # between two settings of the same source. `Compiler.compile_cached`
        # keys on (cachekey, src), so without the setting in the key the second
        # setting measured would silently be served the first setting's binary.
        self.cachekey = f"lm38_{self.tag}_{hashlib.sha256(' '.join(
            [str(maxrregcount)] + self.extra).encode()).hexdigest()[:10]}"
        self.sources: dict[str, str] = {}          # sha256 -> generated CUDA
        self.compiled: dict[str, dict] = {}        # sha256 -> per-function facts
        self.errors: list[str] = []
        self.calls = 0

    # -- the source, recorded even when the binary comes from the cache -------
    def compile_cached(self, src: str) -> bytes:
        self.sources[hashlib.sha256(src.encode()).hexdigest()] = src
        return self.base.compile_cached(src)

    # -- the stock nvrtc call, unchanged ------------------------------------
    def _ptx(self, src: str) -> bytes:
        return self.base.compile(src)

    def compile(self, src: str) -> bytes:
        self.calls += 1
        sha = hashlib.sha256(src.encode()).hexdigest()
        ptx = self._ptx(src)
        cmd = ["ptxas", f"-arch={self.arch}", "-v", "-O3"]
        if self.maxrregcount is not None:
            cmd.append(f"-maxrregcount={int(self.maxrregcount)}")
        cmd += self.extra
        with tempfile.TemporaryDirectory() as td:
            ptxf, cubf = os.path.join(td, "k.ptx"), os.path.join(td, "k.cubin")
            open(ptxf, "wb").write(ptx.rstrip(b"\x00"))
            cmd += ["-o", cubf, ptxf]
            p = subprocess.run(cmd, capture_output=True, text=True)
            self.compiled.setdefault(sha, {"sha256": sha, "cmd": " ".join(cmd),
                                           "ptx_bytes": len(ptx), "functions": {}})
            if p.returncode != 0 or not os.path.exists(cubf):
                self.errors.append(f"{sha[:12]} ptxas rc={p.returncode} "
                                   f"{(p.stderr or '')[-400:]}")
                raise RuntimeError(f"ptxas failed: {p.stderr[-400:]}")
            # `parse_ptxas_v` returns {function name: {stack_frame, spill_stores,
            # spill_loads, registers}}. Keyed BY NAME on purpose: pairing two
            # per-kernel lists by index is how a register count ends up attached
            # to the wrong kernel, and this graph has six kernels in it.
            fns = parse_ptxas_v(p.stderr)
            self.compiled[sha]["functions"] = fns
            self.compiled[sha]["ptxas_stderr"] = p.stderr[-4000:]
            cub = open(cubf, "rb").read()
        # SASS of the EXACT cubin that is about to be loaded. cuobjdump is used
        # rather than nvdisasm because the project's census is built on
        # cuobjdump output, so the instruction counts here and the counts in
        # `notes/23`/`notes/36` are the same kind of number.
        d = subprocess.run(["cuobjdump", "-sass", "-"], input=cub,
                           capture_output=True)
        sass = d.stdout.decode("utf-8", "replace")
        if not sass.strip():
            sass = self._sass_via_nvdisasm(cub)
        for fname, body in split_functions(sass).items():
            fns.setdefault(fname, {})["sass_instructions"] = count_instructions(
                "\n".join(body))
        # Stash the SASS and the PTX so the run directory holds the evidence.
        if getattr(self, "keep_dir", None):
            k = str(len(list(Path(self.keep_dir).glob("*"))))
            (Path(self.keep_dir) / f"{k}_{sha[:12]}.sass").write_text(sass)
            (Path(self.keep_dir) / f"{k}_{sha[:12]}.ptx").write_bytes(ptx)
        return cub

    def _sass_via_nvdisasm(self, cub: bytes) -> str:
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "k.cubin")
            open(f, "wb").write(cub)
            d = subprocess.run(["nvdisasm", "-c", f], capture_output=True, text=True)
        return d.stdout


class CompilerBank:
    """Install a `PtxCappedCompiler` on a device, and take it back off again.

    `Compiled.renderer` returns a *cached* renderer instance, so assigning
    `renderer.compiler` changes what every subsequent compile in this process
    uses. The stock compiler is kept and restored, because a session that
    silently times its baseline through a patched path has no control.
    """

    def __init__(self, dev, keep_dir: str | os.PathLike | None = None):
        self.dev = dev
        self.renderer = dev.renderer
        self.stock = self.renderer.compiler
        self.keep_dir = str(keep_dir) if keep_dir else None
        self.saved: list = []

    def install(self, maxrregcount: int | None = None,
                extra: tuple[str, ...] = (), tag: str | None = None) -> PtxCappedCompiler:
        c = PtxCappedCompiler(self.stock, self.dev.arch, maxrregcount, extra,
                              tag or "")
        if self.keep_dir:
            c.keep_dir = self.keep_dir
        self.saved.append(self.renderer.compiler)
        self.renderer.compiler = c
        return c

    def restore(self) -> None:
        if self.saved:
            self.renderer.compiler = self.saved.pop()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.restore()
        return False


# ---------------------------------------------------------------------------
# one measured configuration, end to end
# ---------------------------------------------------------------------------
class Cell:
    """A rung x batch x compiler-setting, measured the same way every time.

    Deliberately one function, so that "every rung you time is gated on the
    card first" is a property of the code path rather than of anybody's memory
    at 1400 s into a Kaggle session. The order inside is fixed:

      1. install the compiler setting and empty the program cache;
      2. build a FRESH graph, refuse it if `realize()` would be a no-op
         (`minecu_tg.bench.require_fresh_graph` -- the framework's own
         predicate, not a re-derivation);
      3. realize it and compare its bytes against the hand-written driver's
         own bytes -- before any clock is read. A configuration that does not
         reproduce the reference is not a faster version of the same program;
      4. only then time it, with `minecu_tg.bench.linear_device_time_ms`, which
         re-executes the compiled CALLs and is immune to the realized-graph
         no-op;
      5. collect the frame, the register count and the SASS count of the binary
         that was just executed.
    """

    def __init__(self, render_fn, label: str):
        self.render_fn, self.label = render_fn, label
        self.rows: list[dict] = []

    def run(self, bank: "CompilerBank", recorder: "ProgramRecorder",
            batch: int, ref_obs, vox_t, basis_t, timing_inputs,
            maxrregcount: int | None = None, nolocals: bool = False,
            extra: tuple[str, ...] = (), repeats: int = 3, iters: int = 5,
            tag: str | None = None) -> dict:
        """One configuration. Returns the record; appends it to `self.rows`.

        `ref_obs`/`vox_t`/`basis_t` are the reference's own state, for the gate.
        `timing_inputs` is the `(vox, basis)` pair the timed graph runs on.
        """
        import gc
        import numpy as np
        from tinygrad import Device
        from minecu_tg import bench as TG_BENCH

        from minecu_tg_opt.lm38 import Nolocals

        rec = {"label": self.label, "batch": batch,
               "maxrregcount": maxrregcount, "nolocals": bool(nolocals),
               "ptxas_extra": list(extra),
               "cell_tag": tag or (f"mrr{maxrregcount}" if maxrregcount else "stock")}
        cleared = clear_program_cache()
        rec["programs_cache_cleared"] = cleared
        n_prog0 = len(recorder.programs)

        with Nolocals(nolocals):
            comp = bank.install(maxrregcount, extra, tag=rec["cell_tag"])
            try:
                # ---- 3. the gate, before any clock is read ------------------
                t0 = time.time()
                g = self.render_fn(vox_t, basis_t, *timing_inputs[2:])
                TG_BENCH.require_fresh_graph(g, f"{self.label} gate graph")
                img = g.numpy()
                rec["gate_pixels"] = int(img.reshape(len(img), -1).shape[0])
                rec["gate_pixels_differing"] = int(
                    (img.reshape(len(img), -1, 3).astype(np.int32)
                     != ref_obs.reshape(len(ref_obs), -1, 3).astype(np.int32)
                     ).any(axis=2).sum())
                rec["gate_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(img)).hexdigest()
                rec["gate_identical"] = rec["gate_pixels_differing"] == 0
                rec["gate_s"] = round(time.time() - t0, 1)
                del g, img
                gc.collect()

                # ---- 4. the timing, on a fresh graph -----------------------
                times, calls = [], []
                for _ in range(max(1, repeats)):
                    t = self.render_fn(timing_inputs[0], timing_inputs[1],
                                       *timing_inputs[2:])
                    TG_BENCH.require_fresh_graph(
                        t, f"{self.label} timed graph")
                    with TG_BENCH.MemProbe(Device.DEFAULT) as mp:
                        mp.reset()
                        r = TG_BENCH.linear_device_time_ms(t, iters=iters) or {}
                    r.pop("mem", None)
                    if r.get("error"):
                        rec.setdefault("errors", []).append(r["error"])
                    else:
                        times.append(r["median_ms"])
                        calls.append(r.get("n_kernels"))
                    del t
                    gc.collect()
                rec["times_ms"] = times
                rec["median_ms"] = statistics.median(times) if times else None
                rec["n_calls"] = max(calls) if calls else None
                rec["spread_pct"] = (100.0 * (max(times) - min(times))
                                     / min(times)) if len(times) > 1 else 0.0

                # ---- 5. what the binary that just ran actually was ---------
                rec["compiler_calls"] = comp.calls
                rec["n_sources_seen"] = len(comp.sources)
                rec["source_sha256"] = {s[:16]: n for s, n in comp.sources.items()}
                hot = max(comp.sources.values(), key=len) if comp.sources else ""
                sf = source_facts(hot) if hot else {}
                rec["hot_source"] = sf
                infos = comp.compiled.get(
                    hashlib.sha256(hot.encode()).hexdigest() if hot else "", {})
                names = list(sf.get("kernel_names", []))
                # The fused region is the kernel with the most instructions; the
                # other five are the prologue and epilogue kernels of the same
                # graph and every one of them is reported too, so the reader can
                # see the frame belongs to the fused one and not to a bystander.
                fns = infos.get("functions", {})
                rec["ptxas_all_functions"] = fns
                hotfn = max(fns, key=lambda k: fns[k].get("sass_instructions", 0)) \
                    if fns else None
                rec["ptxas_hot_function"] = hotfn
                rec["ptxas"] = fns.get(hotfn, {}) if hotfn else {}
                rec["ptxas_cmd"] = infos.get("cmd")
                rec["ptxas_errors"] = comp.errors
                progs = recorder.facts_for(set(names))
                rec["driver_facts"] = progs
                hotprog = next((p for p in progs
                                if p.get("kernel_name") == hotfn), None)
                if hotprog:
                    rec["registers"] = hotprog.get("driver_regs")
                    rec["driver_local_bytes"] = hotprog.get("driver_local_bytes")
                    rec["driver_attrs_present"] = True
                else:
                    rec["driver_attrs_present"] = False
            finally:
                bank.restore()
        rec["pass"] = bool(rec.get("gate_identical") and times)
        self.rows.append(rec)
        return rec



# ---------------------------------------------------------------------------
# the register budget as the framework itself exposes it
# ---------------------------------------------------------------------------
class Nolocals:
    """TinyGrad's own `NOLOCALS`, as a context manager.

    `helpers.py:239` defines `NOLOCALS` as a ContextVar and
    `codegen/opt/heuristic.py:163` turns it into `Opt(OptOps.NOLOCALS)`, which
    `postrange.py:122` turns into `dont_use_locals`, which
    `renderer/cstyle.py:156-157` turns into
    `__launch_bounds__(prod(local_dims.vmax))`. So the knob changes the
    `__launch_bounds__` written into the source -- and it may also change which
    local dims are chosen (`heuristic.py:81` uses it in a threshold). Both are
    recorded: the `__launch_bounds__` line and the source sha are printed for
    every cell, so a cell whose *source* changed is not silently described as a
    same-source cell.
    """

    def __init__(self, on: bool = True):
        self.on, self._old = on, None

    def __enter__(self):
        from tinygrad.helpers import NOLOCALS
        self._old = NOLOCALS.value
        NOLOCALS.value = 1 if self.on else 0
        return self

    def __exit__(self, *a):
        from tinygrad.helpers import NOLOCALS
        NOLOCALS.value = self._old
        return False


# ---------------------------------------------------------------------------
# reading the register budget back out of the binary that was actually timed
# ---------------------------------------------------------------------------
def func_facts(program) -> dict:
    """`NUM_REGS` and `LOCAL_SIZE_BYTES` read from a loaded `CUfunction`.

    These two are the driver talking about *this* module -- the one whose kernel
    is being timed -- rather than about a text file compiled somewhere else.
    `LOCAL_SIZE_BYTES` is the local frame in bytes, so this is the same quantity
    `ptxas -v` calls a stack frame, measured a second way and on the live
    binary. A disagreement between the two is a finding, not a nuisance, so both
    are reported.
    """
    from tinygrad.runtime.autogen import cuda
    out = {}
    for attr, key in ((4, "driver_regs"),      # CU_FUNC_ATTRIBUTE_NUM_REGS
                      (3, "driver_local_bytes")):  # ..._LOCAL_SIZE_BYTES
        v = ctypes.c_int(0)
        rc = cuda.cuFuncGetAttribute(ctypes.byref(v), attr, program.prg)
        out[key] = int(v.value) if rc == 0 else None
        out[key + "_rc"] = int(rc)
    out["kernel_name"] = getattr(program, "name", None)
    return out


LAUNCH_BOUNDS_RE = re.compile(
    r'__global__\s+void\s+__launch_bounds__\((\d+)\)\s+(\w+)')


def source_facts(src: str) -> dict:
    """What can be read off the generated CUDA without a compiler."""
    lb = LAUNCH_BOUNDS_RE.findall(src)
    return {
        "source_bytes": len(src),
        "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "source_statements": len(re.findall(r";\s*(?://.*)?$", src, re.M)),
        "launch_bounds": sorted({int(b) for b, _ in lb}),
        "kernel_names": sorted({n for _, n in lb}),
        "n_kernels": len(lb),
    }


# ---------------------------------------------------------------------------
# two caches that would silently defeat the whole experiment
# ---------------------------------------------------------------------------
def clear_program_cache() -> None:
    """Empty TinyGrad's in-memory program cache.

    **`to_program_key` is `(ast.key, type(renderer), renderer.target,
    *to_program_config)` -- `codegen/__init__.py:496-497`. The COMPILER IS NOT IN
    IT.** So once a source has been compiled by one compiler, every later
    request for that source returns the *first* compiler's binary and no
    compilation happens at all. Install a register budget, rebuild the identical
    graph, and you get the un-capped binary back with no error and no warning.

    That is the same class of defect as `Tensor.realize()` on a realized graph:
    a silent no-op that reads as a measurement. So the cache is emptied before
    every configuration, and `PtxCappedCompiler.calls` is asserted afterwards so
    a configuration whose compilation did not happen cannot report a frame.

    `to_program_config` does include `NOLOCALS` (`codegen/__init__.py:493`), so
    the framework's own knob does key the cache correctly; only a compiler swap
    needs this.
    """
    from tinygrad.codegen import to_program_cache
    n = len(to_program_cache)
    to_program_cache.clear()
    return n


class ProgramRecorder:
    """Every `CUDAProgram` the runtime actually builds, with its resources.

    The register count and the local frame are read from the `CUfunction` the
    driver loaded, i.e. from the binary the timed kernel is, by a different
    route than `ptxas -v`. Two independent readings of the same quantity, from
    two different components, on the same artifact.
    """

    def __init__(self):
        self.programs: list = []
        self._patched = False

    def __enter__(self):
        if self._patched:
            return self
        from tinygrad.runtime.ops_cuda import CUDADevice
        rec = self
        base = CUDADevice.runtime

        def _runtime(self, obj):
            p = base(self, obj)
            rec.programs.append(p)
            return p
        CUDADevice.runtime = _runtime
        self._base, self._patched = base, True
        return self

    def __exit__(self, *a):
        if self._patched:
            from tinygrad.runtime.ops_cuda import CUDADevice
            CUDADevice.runtime = self._base
            self._patched = False
        return False

    def since(self, n: int) -> list:
        return self.programs[n:]

    def facts_for(self, names) -> list:
        """Resource facts for the programs whose kernel name is in `names`."""
        out = []
        for p in self.programs:
            if getattr(p, "name", None) in names:
                out.append(func_facts(p))
        return out



# ===========================================================================
# ARM C -- live-range structure at an unchanged operation count
# ===========================================================================
# WHY THIS ARM IS HERE AT ALL, GIVEN THAT IT PROBABLY COMES BACK EMPTY
# ---------------------------------------------------------------------
# The discriminator needs a frame that moves while the graph size does not.
# The register-budget sweep moves the frame with the graph held fixed *by
# construction* (identical source string), but it moves it with a compiler
# flag. The interesting question a source-level lever would answer is whether
# the frame is a *free parameter* of the program at all: if a rewrite that
# changes only live ranges -- same operations, same semantics -- can move the
# frame, then "register allocation" and "graph size" are two things and the
# report's mechanism has a handle inside the framework's own vocabulary. If
# every such rewrite comes back with an identical source string, then they are
# one thing, the frame is pinned by the operation count, and the only way to
# move it is to leave the framework. Either answer is worth having and both are
# cheap, so both are measured rather than argued.
#
# Both bodies below are op-neutral BY CONSTRUCTION -- same operations, same
# order of effects, algebraically identical results -- and their operation
# counts are re-derived and compared by the UOp census rather than trusted.

def _body_sum_last(S, hit, found, steps, k, count_steps):
    """`flat_gather` with the three `t_max` increments materialised first.

    The original writes each `t_max` back immediately, so at any point only two
    of the three old values are live. This form computes all three increments
    and then does all three selects, so all three `t_max` values and all three
    increments are live simultaneously -- five more live values at the peak of
    the step body, for exactly the same arithmetic.
    """
    i32, wx, wy, wz = S["i32"], S["wx"], S["wy"], S["wz"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]
    vx, vy, vz = S["v"]
    from tinygrad import Tensor as _T

    if count_steps:
        steps = (~found).where(k + 1, steps)

    inside = ((vx >= 0) & (vx < wx) & (vy >= 0) & (vy < wy) & (vz >= 0) & (vz < wz))
    wxz = wx * wz
    addr = S["b_off"] + (vy * wxz + vx * wz + vz)
    block = S["flat"][inside.where(addr, 0)].cast(i32)

    solid = (block >= 0) & inside & ~found
    hit = solid.where(block, hit)
    found = found | solid

    take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
    take_y = (t_max_y < t_max_z) & ~take_x
    take_z = ~(take_x | take_y)
    nx, ny, nz = t_max_x + td_x, t_max_y + td_y, t_max_z + td_z
    t_max_x = take_x.where(nx, t_max_x)
    t_max_y = take_y.where(ny, t_max_y)
    t_max_z = take_z.where(nz, t_max_z)
    S["t_max"] = (t_max_x, t_max_y, t_max_z)
    sx, sy, sz = S["sign"]
    S["v"] = (take_x.where(vx + sx, vx), take_y.where(vy + sy, vy),
              take_z.where(vz + sz, vz))
    return hit, found, steps


def _body_late_addr(S, hit, found, steps, k, count_steps):
    """`flat_gather` with the gather index built where it is used.

    The original forms `addr` early and keeps ONE int live to the gather. This
    form puts the same expression inside the `where`, so `vx`, `vy`, `vz` and
    `b_off` are live from the top of the step body to the gather instead of one
    derived value. Same operations, same result.
    """
    i32, wx, wy, wz = S["i32"], S["wx"], S["wy"], S["wz"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]
    vx, vy, vz = S["v"]

    if count_steps:
        steps = (~found).where(k + 1, steps)

    inside = ((vx >= 0) & (vx < wx) & (vy >= 0) & (vy < wy) & (vz >= 0) & (vz < wz))
    wxz = wx * wz
    block = S["flat"][inside.where(S["b_off"] + (vy * wxz + vx * wz + vz), 0)].cast(i32)

    solid = (block >= 0) & inside & ~found
    hit = solid.where(block, hit)
    found = found | solid

    take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
    take_y = (t_max_y < t_max_z) & ~take_x
    take_z = ~(take_x | take_y)
    t_max_x = take_x.where(t_max_x + td_x, t_max_x)
    t_max_y = take_y.where(t_max_y + td_y, t_max_y)
    t_max_z = take_z.where(t_max_z + td_z, t_max_z)
    S["t_max"] = (t_max_x, t_max_y, t_max_z)
    sx, sy, sz = S["sign"]
    S["v"] = (take_x.where(vx + sx, vx), take_y.where(vy + sy, vy),
              take_z.where(vz + sz, vz))
    return hit, found, steps


#: The arm-C bodies, keyed by the name the census and the note use.
ARM_C = {"fg_sum_last": _body_sum_last, "fg_late_addr": _body_late_addr}


def render_arm_c(voxels, basis, width: int, height: int, max_steps: int,
                 variant: str, count_steps: bool = False):
    """`minecu_tg_opt.render.render_variant` with the step body replaced.

    The prologue is the *stock* one -- `minecu_tg_opt.render._setup` and
    `_ladder_init(S, "flat_gather")`, imported, not copied -- so the only thing
    that differs from rung `flat_gather` is the step body. If that prologue is
    ever edited by another lane, this arm follows the edit rather than pinning
    a stale copy of it, which is the only safe way to add a body to a module
    somebody else owns.
    """
    from tinygrad import Tensor, dtypes
    from minecu_tg.blocks import AIR, colour_lut
    from minecu_tg_opt.render import _ladder_init, _setup

    if variant not in ARM_C:
        raise SystemExit(f"unknown arm-C variant {variant!r}; have {sorted(ARM_C)}")
    body = ARM_C[variant]
    S = _setup(voxels, basis, width, height)
    _ladder_init(S, "flat_gather")
    B, H, W, i32 = S["B"], S["height"], S["width"], S["i32"]

    hit = Tensor.full((B, 1, H, W), AIR, dtype=i32)
    found = Tensor.full((B, 1, 1, 1), False, dtype=dtypes.bool)
    steps = Tensor.zeros((B, 1, H, W), dtype=i32)
    for k in range(max_steps):
        hit, found, steps = body(S, hit, found, steps, k, count_steps)

    colour_row = (hit >= 0).where(hit, 13)
    obs = colour_lut()[colour_row].reshape(B, H, W, 3)
    return (obs, steps) if count_steps else obs.contiguous()
