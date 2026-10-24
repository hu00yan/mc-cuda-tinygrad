#!/usr/bin/env python3
"""codegen.py -- what each knob does to the CODE, on this host, with no GPU.

    python -m minecu_tg_knobs.codegen --knobs --out .agent-scratch/knobs/codegen

This is the offline half of arm B. It produces, for every (port, knob
configuration), the **real sm_75 CUDA C** that tinygrad would compile, and a
fingerprint of the schedule in it. `sass.sh` then compiles those exact files to
real SASS, so the instruction counts in the note come from a cubin and not from
a source reading.

THE NULL DEVICE, AND WHY IT IS NOT A DIFFERENT EXPERIMENT
--------------------------------------------------------
This host is arm64 macOS with no CUDA, so the codegen census runs on
`DEV=NULL:CUDA:sm_75`: tinygrad's no-op device with the **real CUDA renderer**
attached to an **sm_75 target**. The renderer is a pure function of the UOp
graph (`renderer/cstyle.py`), so the schedule, the fusion, the register
allocation decisions and the emitted C are the ones the T4 run gets. Nothing
executes.

Two things follow, and both are why the *timings* are on the card and only the
timings:

  * no number from this file is a performance number. It is instruction counts
    and schedule shape, and a schedule can be identical on two architectures
    and cost different amounts.
  * `use_nvcc` is forced on (`capture_cuda.py:52-58` established this): the
    default NVRTC path dlopen()s libnvrtc, which does not exist here. And the
    compile itself is REPLACED by a capture, so no nvcc is invoked. The source
    is finished before `compile_cached` is called, so what is captured is the
    real generated C. The cubins come from `sass.sh` in the archlab container.

THE THREE GATES, AND WHAT EACH ONE PREVENTS
--------------------------------------------
  1. `registry.check_all()` -- every swept knob's Context key must exist and be
     in tinygrad's `to_program_config` cache tuple. A knob outside that tuple
     returns the DEFAULT's compiled program, so the sweep would report the
     default with a knob's name on it.
  2. fresh process per configuration -- `to_program_cache` and the `ContextVar`
     values are process-global. A configuration that ran after another one in
     the same interpreter could read the other's cache entry.
  3. the sha256 column -- a knob is only called inert if the emitted source is
     BYTE-IDENTICAL to the default's. That is the difference between "the knob
     was set and did nothing" and "the knob was not set".

`PARALLEL=0` is set before any tinygrad import and the device is checked, not
the env var: `engine/realize.py:273` compiles in a worker pool whenever more
than one kernel is outstanding, and an in-process hook then sees only the
`len(todo)==1` rounds. That is how `notes/13` captured 1 kernel out of 12,082.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# BEFORE any tinygrad import: the device is constructed at import time.
os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")
# the NULL allocator refuses copyout. This run never reads a result back --
# every input is built on the CPU first and handed over as bytes -- and the flag
# only stops a stray `.numpy()` from aborting the census.
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------
# the port under test. Both are ordinary TinyGrad; the difference between them
# is a property of the PORT, and the census exists so that difference is
# measured rather than assumed.
# --------------------------------------------------------------------------
PORTS = {
    # arm A: `minecu_tg/render.py`, imported unchanged. The naive port, whose
    # 26.2x-vs-7198.9us comparison is withdrawn as a compiler measurement by
    # ledger C-14.
    "naive": ("minecu_tg.render", "render", {}),
    # arm B's port: the same masked-array formulation, carrying the index
    # arithmetic Amanatides-Woo specifies. See `lean.py`.
    "lean": ("minecu_tg_knobs.lean", "render",
             dict(cursor_int=True, flat_advance=True, unsigned_bounds=True)),
    # the two intermediates, so the port's own contribution is separable from
    # the knob's. `lean-minus-flat` is "lean, but recompute the address each
    # step" -- i.e. keep the integer cursor and the folded bounds test, drop
    # only the advanced address, which is C-14's specific term.
    "lean-no-flat": ("minecu_tg_knobs.lean", "render",
                     dict(cursor_int=True, flat_advance=False, unsigned_bounds=True)),
    "lean-no-unsigned": ("minecu_tg_knobs.lean", "render",
                         dict(cursor_int=True, flat_advance=True, unsigned_bounds=False)),
}


def _install_null_cuda(arch: str) -> None:
    """Attach the real CUDA renderer to the NULL device, targeting `arch`.

    tinygrad's tensor-core table does `int(arch[3:])` and cannot parse an
    architecture-specific suffix, so the base arch is handed to the renderer
    and the full name goes to nvcc later. Same table on both sides.
    """
    dev_arch = re.sub(r"(sm_\d+)[a-z]$", r"\1", arch)
    os.environ["DEV"] = f"NULL:CUDA:{dev_arch}"

    from tinygrad.renderer import cstyle
    _orig_init = cstyle.CUDARenderer.__init__

    def _init(self, target, use_nvcc=False):
        return _orig_init(self, target, use_nvcc=True)

    cstyle.CUDARenderer.__init__ = _init


def _capture_compiler() -> list[str]:
    """Replace `Compiler.compile_cached` with a capture. Returns the sink.

    Installed BEFORE any tinygrad import that could build a device, because the
    renderer is constructed on first use and compiles then. Returns the
    generated source and an empty "binary": the program never runs, so the
    binary is never needed, and the source is complete before the call.
    """
    from tinygrad.device import Compiler
    sink: list[str] = []

    def _capture(self, src, *a, **k):
        sink.append(str(src))
        return b""

    Compiler.compile_cached = _capture
    return sink


def assert_in_process(dev: str) -> None:
    """PARALLEL=0 must have actually taken effect, checked on the device.

    The KDA lesson from `notes/13-schedule-vs-decay.md` §4: the env var looked
    set and the census still captured 1 kernel out of 12,082.
    """
    from tinygrad import Device
    d = Device[dev]
    for attr in ("worker_pool", "_worker_pool", "pool"):
        if getattr(d, attr, None) is not None:
            raise AssertionError(
                f"device {dev!r} has {attr}={getattr(d, attr)!r}: compilation is "
                f"in a worker process, so this in-process capture would miss "
                f"every kernel not compiled alone. PARALLEL=0 did not take "
                f"effect. Refusing to report a partial census.")


# --------------------------------------------------------------------------
# inputs, built on the CPU in a child process
# --------------------------------------------------------------------------
def build_inputs(world: int, ground: int, batch: int, res: int,
                 cache: Path) -> tuple["np.ndarray", "np.ndarray"]:
    """voxels [B,WY,WX,WZ] int8 and cameras [B,5] float32, as numpy.

    A child process, not this one, because the NULL device cannot copy results
    back and the inputs have to be REAL BYTES before they reach a device that
    cannot execute. A subprocess rather than an in-process `DEV` switch, because
    tinygrad resolves `Device.DEFAULT` once and caches it.

    The child runs with `CLANG=0` and the *CPU* device, and that combination
    is the third thing that had to be right:

      * this host has no `clang`, and a CPU realization died with
        `FileNotFoundError: 'clang'`;
      * Kaggle's image has a `clang` too, and the same child died there the
        same way -- so "is there a clang" was the wrong question;
      * and `CLANG=0` alone did not fix it, because in this TinyGrad
        `CLANG=0` selects a *different CPU compiler*, not none: it selects
        `X86Compiler`, which does **not** compile C and takes the graph as
        pre-built hex. Feeding it C source fails differently.

    The real answer is that this child should not be compiling anything. The
    world is a few nested `where`s over constants (`minecu_tg/world.py`) and
    this arm does not need a CPU backend at all -- it needs the BYTES. So the
    child is given `DEV=PYTHON`, tinygrad's interpreter backend, which
    evaluates the graph rather than compiling it, and the failure mode of
    getting that wrong is a number instead of an exception.

    Three wrong answers to one question ("how do I get these bytes?") is the
    reason the answer is now a device name rather than a compiler flag.
    """
    import subprocess

    import numpy as np

    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        # DEV=PYTHON: evaluate, do not compile. See the docstring.
        env = {**os.environ, "DEV": "PYTHON"}
        child = (
            "import sys; sys.path.insert(0, %r)\n"
            "import numpy as np\n"
            "from minecu_tg.world import generate_world\n"
            "w, g, B, R = %d, %d, %d, %d\n"
            "# one world, replicated: generate_world_kernel ignores its seed\n"
            "# argument, so every env gets the same volume (minecu_tg/world.py)\n"
            "one = np.ascontiguousarray(generate_world(w, g).numpy())\n"
            "vox = np.ascontiguousarray(np.stack([one]*B))\n"
            "# cameras spread in yaw, so the batch is not 1024 identical rays and\n"
            "# adjacent threads do not all march the same path\n"
            "yaw = np.linspace(-1.1, 1.1, B, dtype=np.float32)\n"
            "pit = np.linspace(0.45, -0.25, B, dtype=np.float32)\n"
            "cams = np.stack([\n"
            "  np.full(B, w/2.0 + 0.5, np.float32),\n"
            "  np.full(B, g + 0.6, np.float32),\n"
            "  np.full(B, w/2.0 + 0.5, np.float32), yaw, pit], axis=1)\n"
            "np.savez(%r, vox=vox, cams=cams)\n"
        ) % (str(REPO), world, ground, batch, res, str(cache))
        # capture the child's own output. `check=True` alone reports
        # "returned non-zero exit status 1" and discards the reason, which on a
        # remote is the only thing there is: the traceback goes to the child's
        # stderr, which nothing else in the chain reads.
        proc = subprocess.run([sys.executable, "-c", child], cwd=str(REPO),
                              env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"building the inputs on the CPU child failed (rc="
                f"{proc.returncode}). The child's own output follows, because "
                f"without it this is an unactionable 'exit status 1' on a "
                f"machine nobody is sitting at.\n"
                f"--- stdout ---\n{proc.stdout[-2000:]}\n"
                f"--- stderr ---\n{proc.stderr[-2000:]}")

    z = np.load(cache)
    return z["vox"], z["cams"]


# --------------------------------------------------------------------------
# the schedule fingerprint, read out of the emitted source
# --------------------------------------------------------------------------
def fingerprint(src: str) -> dict:
    """The per-thread shape, extracted from the generated C.

    tinygrad renders a schedule as C, so the schedule is legible in the text: a
    `__local__` array whose suffix names the local extent, `#pragma unroll N`,
    and the vector types the upcast produced. Reading those is a measurement of
    the schedule; reading a knob's `.value` is a measurement of the knob, and
    the two are what the note has to keep apart.
    """
    loc = re.search(r"__local__\s+([A-Za-z0-9_]+)\s*\[(\d+)\]", src)
    unroll = sorted({int(m) for m in re.findall(r"#pragma unroll (\d+)", src)})
    # the vector types the upcast produced, e.g. float4, uint32
    vec = sorted({a + b for a, b in re.findall(r"\b(float|uint|int)([248])\b", src)})
    gx = re.search(r"<<<[^,]*,\s*([^,\)]+)", src)
    return {
        "local_elems": int(loc.group(2)) if loc else 0,
        "local_name": loc.group(1) if loc else None,
        "unroll": unroll,
        "vec_types": vec,
        "global_dim": (gx.group(1).strip() if gx else None),
        # tinygrad's C renderer writes a global read as `*(dataN_<size>+idx)`.
        # Counting the SASS mnemonics here would be nonsense -- this is C, not
        # SASS -- so the read is counted from the dereference, and the BUFFERS
        # are listed so the gather can be identified by which buffer it reads.
        # A global read is `*(dataN_<size>+idx)`, possibly inside a predicated
        # ternary `cond ? *(dataN+idx) : 0`. The second form is the interesting
        # one for this kernel -- it is how tinygrad renders a clipped gather --
        # so both are counted, and a predicated read is counted separately.
        "g_reads": len(re.findall(r"\*\(\s*data\d+_\d+\s*\+", src)),
        "g_pred_reads": len(re.findall(r"\?[^:;]{0,200}?\*\(\s*data\d+_\d+\s*\+", src)),
        "g_writes": len(re.findall(r"\*\(\s*data\d+_\d+\s*\+[^)]*\)\s*=", src)),
        "g_buffers_read": sorted({int(n) for n in
                                 re.findall(r"\*\(\s*data(\d+)_\d+\s*\+", src)}),
        "launch_bounds": (re.search(r"__launch_bounds__\((\d+)\)", src).group(1)
                          if re.search(r"__launch_bounds__\((\d+)\)", src) else None),
        "shared": len(re.findall(r"__shared__", src)),
        "syncthreads": len(re.findall(r"__syncthreads", src)),
        "lines": src.count("\n"),
    }


def hot_kernel(kernels: list[dict]) -> dict | None:
    """The kernel that contains the DDA: the biggest one, by the evidence.

    Picking it by name or by index would be a guess, and the guess is load-
    bearing -- every SASS number in the note is read off this kernel. So it is
    identified by a property that has to be true of the DDA kernel and of no
    other: **it reads an int8 buffer and is the longest kernel in the graph.**

    The int8 read is the voxel gather: the voxel grid is int8, the camera basis
    is float32, the observation is uint8, and the ray state is int32. A kernel
    that reads no int8 buffer is not traversing the world. Among those, the DDA
    kernel is the one that is longest by source lines, because the 32 step
    bodies are inlined into it.
    """
    cands = [k for k in kernels if _INT8_BUFFERS.get(k["index"])]
    if not cands:
        raise AssertionError(
            "no kernel in this graph takes an int8 buffer, so the DDA kernel "
            "could not be identified. The identification is load-bearing -- "
            "every SASS count in the note is read off this kernel -- so this "
            "refuses rather than falling back to 'the biggest one'.")
    # tie-break on the SIZE OF THE LARGEST int8 ARGUMENT, not on source length.
    # At batch 1024 the voxel grid is [B,WY,WX,WZ] int8 = 32 MiB, and nothing
    # else in the graph is an int8 buffer of that size; the 14-float camera
    # basis and the int32 ray state are not int8 at all. So the traversal kernel
    # is the one holding the biggest int8 argument, and that is a fact about the
    # emitted signature rather than a guess about which kernel is "hot".
    return max(cands, key=lambda k: (_INT8_MAX_BYTES.get(k["index"], 0),
                                     k["fingerprint"]["lines"]))


# The int8 argument buffers, by kernel index. Filled in by `run_one`, which has
# the source and can read the parameter list. Kept as a module dict rather than
# a closure so `hot_kernel` stays a pure function of the census.
_INT8_BUFFERS: dict[int, set[int]] = {}

# The size in BYTES of the largest int8 argument, by kernel index. tinygrad
# encodes it in the parameter name -- `signed char* data4_33554432` is a
# 33 554 432-byte int8 buffer -- so this is read off the generated signature.
_INT8_MAX_BYTES: dict[int, int] = {}


def _int8_max_bytes(src: str) -> int:
    sizes = [int(s) for s in
             re.findall(r"(?:signed char|int8_t|char)\s*\*\s*data\d+_(\d+)",
                        _kernel_params(src))]
    return max(sizes, default=0)


def _kernel_params(src: str) -> str:
    """The parameter list of the generated kernel, as text.

    Not `src.split("{")[0]`: the C prelude contains
    `template <class T, class F> ... { union U { F f; T t; }; U u; ... }`,
    whose braces come first, so splitting on `{` truncates the source before
    the signature and reports an empty argument list. Likewise a regex like
    `__global__ void [^(]+\\(([^)]*)\\)` matches the `(` of
    `__launch_bounds__(128)` and captures "128".

    So: find the kernel definition, then walk the parentheses to the one that
    closes the parameter list.
    """
    # `__global__ void __launch_bounds__(16) r_8_16_...(args) {` -- the FIRST
    # paren after `void` belongs to `__launch_bounds__`, not to the parameter
    # list, so the search has to skip that whole call. The 4-arg case
    # `__launch_bounds__(a, b)` is handled by the same skip.
    m = re.search(r"__global__\s+void\s+", src)
    if not m:
        return ""
    i = m.end()
    lb = re.match(r"__launch_bounds__\s*\(", src[i:])
    if lb:
        depth, k = 0, i + lb.end() - 1
        while k < len(src):
            depth += src[k] == "("
            depth -= src[k] == ")"
            if depth == 0:
                i = k + 1
                break
            k += 1
    i = src.find("(", i)
    if i < 0:
        return ""
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    return ""


def _int8_buffer_indices(src: str) -> set[int]:
    """Which of this kernel's arguments are int8 (`signed char`) buffers.

    Read from the generated signature, not assumed from a tensor dtype. The
    DDA kernel is the one that takes the voxel grid, and the grid is int8; the
    camera basis is float32, the ray state int32, the observation uint8. So this
    is a property of the emitted code that identifies the traversal kernel
    without naming it.
    """
    params = _kernel_params(src)
    return {int(n) for n in re.findall(
        r"(?:signed char|int8_t|char)\s*\*\s*data(\d+)_", params)}


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------
def configurations() -> list[tuple[str, dict]]:
    """(label, {Context key: value}) for every configuration in the sweep.

    One knob at a time, plus the combinations that matter. The rule for which
    combinations are worth running: **only pairs drawn from the knobs whose read
    site is reachable for this kernel.** Every other knob's only consumer is
    behind a guard this kernel cannot enter, so a combination of two of them is
    inert and there are C(7,2)=21 of them to not run.
    """
    from minecu_tg_knobs.registry import KNOBS

    cfgs: list[tuple[str, dict]] = [("default", {})]
    for k in KNOBS:
        for v in k.sweep:
            if v == k.default:
                continue
            cfgs.append((f"{k.name}={v}", {k.context_key: v}))

    # The pairs drawn from the knobs with a REACHABLE read site. There are
    # exactly two of them -- NOLOCALS and TRANSCENDENTAL -- so one pair, and it
    # is run at BOTH off/on values rather than only at one, because the
    # interesting question is the interaction: TRANSCENDENTAL=2 makes the
    # prologue bigger, and NOLOCALS changes the register budget, so the pair can
    # be worse than either alone. Running only one corner of a 2x2 would miss
    # exactly the case worth looking for.
    reach = [k for k in KNOBS if k.reachable_here]
    if len(reach) == 2:
        (a, b) = reach
        for av in a.sweep:
            for bv in b.sweep:
                cfgs.append((f"{a.name}={av}+{b.name}={bv}",
                             {a.context_key: av, b.context_key: bv}))
    return cfgs


def run_one(port: str, cfg: dict, vox, cams, res: int, max_steps: int,
            sink: list[str]) -> dict:
    """Build one (port, config) render and return its kernel census."""
    from tinygrad import Context, Tensor, dtypes

    from minecu_tg_knobs import registry
    from minecu_tg_knobs.lean import camera_basis

    registry.check_all()          # gate 1
    assert_in_process("NULL")     # gate 3 (worker pool)

    mod, fname, kwargs = PORTS[port]
    render_fn = getattr(__import__(mod, fromlist=[fname]), fname)

    with Context(**cfg):
        vox_t = Tensor(vox, dtype=dtypes.int8)
        basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, res, res)
        basis = basis.contiguous()
        sink.clear()
        out = render_fn(vox_t, basis, res, res, max_steps, **kwargs)
        out = out[0] if isinstance(out, tuple) else out
        # REALIZE, never read back: a `.numpy()` would add a device-to-host copy
        # to the launch count, and that copy belongs to the harness.
        out.realize()

    kernels = [
        {"index": i, "sha256": hashlib.sha256(s.encode()).hexdigest(),
         "bytes": len(s), "fingerprint": fingerprint(s)}
        for i, s in enumerate(sink)
    ]
    # the int8-argument map is process-local state that `hot_kernel` reads, so
    # it is rebuilt for THIS call rather than accumulated across calls
    _INT8_BUFFERS.clear()
    _INT8_MAX_BYTES.clear()
    for i, s in enumerate(sink):
        _INT8_BUFFERS[i] = _int8_buffer_indices(s)
        _INT8_MAX_BYTES[i] = _int8_max_bytes(s)

    hot = hot_kernel(kernels)
    return {
        "port": port,
        "kernel_count": len(kernels),
        "kernels": kernels,
        "hot_index": hot["index"] if hot else None,
        "hot_sha256": hot["sha256"] if hot else None,
        "hot_fingerprint": hot["fingerprint"] if hot else None,
        "hot_criterion": "takes the largest int8 argument (the [B,WY,WX,WZ] "
                         "voxel grid) and is the longest such kernel; the "
                         "max_steps step bodies are inlined into it",
        # `hot_source` is filled in by `_write`, which knows where it put the
        # .cu files. It is not set here because a path invented here would point
        # somewhere the text does not exist.
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", default="naive,lean,lean-no-flat,lean-no-unsigned")
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--world", type=int, default=32)
    ap.add_argument("--ground", type=int, default=8)
    ap.add_argument("--out", default="", help="dir for the JSON and the .cu files")
    ap.add_argument("--configs", default="", help="comma-separated labels subset")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run every configuration in THIS process (debug only)")
    args = ap.parse_args(argv)

    _install_null_cuda(args.arch)
    sink = _capture_compiler()

    from minecu_tg_knobs.registry import KNOBS, check_all
    info = check_all()
    print("=" * 78)
    print("  minecu_tg_knobs -- the codegen census, offline, sm_75 target")
    print(f"  device NULL:CUDA:{args.arch}   PARALLEL={os.environ.get('PARALLEL')}")
    print(f"  registry gates PASS: {info['knobs']} knobs, {info['sweepable']} sweepable")
    print("=" * 78)

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "inputs.npz" if out_dir else \
        REPO / ".agent-scratch" / "knobs" / "inputs.npz"

    ports = [p for p in args.ports.split(",") if p]
    for p in ports:
        if p not in PORTS:
            print(f"unknown port {p!r}; known: {sorted(PORTS)}", file=sys.stderr)
            return 2

    vox, cams = build_inputs(args.world, args.ground, args.batch, args.res, cache)

    # `--configs` is an EXACT filter, not a "these plus the control" filter:
    # the isolate driver runs one configuration per child and parses one record
    # per port, so silently adding the control here would double the child's
    # output. The parent asks for `default` explicitly when it wants it.
    cfgs = configurations()
    if args.configs:
        want = set(args.configs.split(","))
        cfgs = [c for c in cfgs if c[0] in want]

    # The control has to be in the same run as everything it is compared
    # against, or "identical to default" is a claim about a different
    # compilation. So a SUBSET run (the top-level one) always includes it. A
    # child invoked by the isolate driver already has its one configuration
    # named exactly and must not gain a second.
    if (not args.no_isolate) and "default" not in {c[0] for c in cfgs}:
        cfgs = [("default", {})] + cfgs

    records: list[dict] = []
    if args.no_isolate:
        records = _sweep_here(args, cfgs, ports, vox, cams, sink, out_dir)
    else:
        records = _sweep_isolated(args, cfgs, ports, out_dir)

    if out_dir:
        # The aggregate is re-read from each record's own census.json rather
        # than from the in-memory `records`, because `_write` fills in
        # `hot_source` AFTER the record dict was built -- the aggregate built
        # from the dicts would carry `hot_source: null` for every row, and
        # `knobs_table.py` needs that path to compute the SASS-per-source-op
        # ratio from the same text the SASS was compiled from.
        on_disk = []
        for r in records:
            p = Path(r.get("source_dir", "")) / "census.json" if r.get("source_dir") else None
            if p and p.exists():
                on_disk.append(json.loads(p.read_text()))
            else:
                print(f"  [FAIL] no census.json on disk for {r['port']}/{r['label']}; "
                      f"the aggregate would carry a null hot_source", file=sys.stderr)
                return 1
        (out_dir / "census.json").write_text(json.dumps({
            "device": f"NULL:CUDA:{args.arch}",
            "note": "codegen census on the NULL device: the CUDA renderer and "
                    "the sm_75 target are real, nothing executes. No performance "
                    "number in this file.",
            "batch": args.batch, "res": args.res, "max_steps": args.max_steps,
            "world": args.world, "ground": args.ground,
            "cache_key": info["cache_key"],
            "reachable_knobs": [k.name for k in KNOBS if k.reachable_here],
            "records": on_disk,
        }, indent=1))

    _print_table(records)
    return 0


def _sweep_here(args, cfgs, ports, vox, cams, sink, out_dir) -> list[dict]:
    """Every configuration in THIS process. For debugging only.

    Not trustworthy for the sweep itself: `to_program_cache` and the ContextVar
    values are process-global, so a later configuration can read an earlier
    one's cache entry. `--isolate` is the default for that reason.

    When `--configs` names exactly one label -- which is how the isolate driver
    invokes a child -- the single record is printed on a marked line so the
    parent can parse it out. With several labels there is no such line, and the
    parent's parse fails loudly rather than silently averaging records.
    """
    records = []
    for label, cfg in cfgs:
        for port in ports:
            rec = {"label": label, "port_label": label, "config": cfg,
                   "arch": args.arch,
                   **run_one(port, cfg, vox, cams, args.res, args.max_steps, sink)}
            # `_write` fills in `hot_source` from the directory it wrote the .cu
            # files to, so the record must be written AFTER `_write` and
            # re-serialised by it -- a claim computed from a `hot_source` that
            # points into a staging directory would be computed from text that
            # is no longer there.
            _write(out_dir, rec, sink)
            records.append(rec)

    # The parent parses one `GPUSHJSON:` line per record, so a child that ran
    # several (config, port) pairs emits one line for each. The marker is
    # printed even in the multi-record case: a parent that gets no line fails
    # loudly, and one that gets the wrong number would have to notice, so the
    # count is asserted on the parent's side too.
    for rec in records:
        print("GPUSHJSON:" + json.dumps(rec))
    return records


def _sweep_isolated(args, cfgs, ports, out_dir) -> list[dict]:
    """One fresh interpreter per configuration. The trustworthy path."""
    import subprocess
    records = []
    for label, cfg in cfgs:
        child_out = (out_dir / label.replace("/", "_").replace("=", "-")) \
            if out_dir else None
        cmd = [sys.executable, "-m", "minecu_tg_knobs.codegen",
               "--arch", args.arch, "--batch", str(args.batch),
               "--res", str(args.res), "--max-steps", str(args.max_steps),
               "--world", str(args.world), "--ground", str(args.ground),
               "--ports", ",".join(ports), "--configs", label,
               "--no-isolate"]
        if child_out:
            cmd += ["--out", str(child_out)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO), env={**os.environ})
        marker = "GPUSHJSON:"
        lines = [ln[len(marker):] for ln in proc.stdout.splitlines()
                 if ln.startswith(marker)]
        if not lines:
            print(f"  [FAIL] {label}: the child produced no census", file=sys.stderr)
            print(proc.stdout[-1500:], file=sys.stderr)
            print(proc.stderr[-2500:], file=sys.stderr)
            return []
        if len(lines) != len(ports):
            print(f"  [FAIL] {label}: the child returned {len(lines)} records "
                  f"for {len(ports)} ports; the census is incomplete and every "
                  f"downstream count would be understated.", file=sys.stderr)
            return []
        for line in lines:
            records.append(json.loads(line))
        # The .cu files and each record's own census.json were written BY THE
        # CHILD, from the same call that produced the sha256. The parent only
        # aggregates. Writing them again from here would need a second render
        # and would leave open the possibility of the two disagreeing.
    return records


def _write(out_dir, rec, sink) -> None:
    """Write one record, and the .cu files when the sources are in hand.

    The .cu files are what `sass.sh` compiles. Writing them from the SAME call
    that produced the sha256 is the point: the cubin in the SASS table and the
    hash in the census table are then provably the same program, with no second
    render in between that could differ.
    """
    if not out_dir:
        return
    d = out_dir / rec["port"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "census.json").write_text(json.dumps(rec, indent=1))
    if sink is not None and len(sink) == rec["kernel_count"]:
        for i, s in enumerate(sink):
            (d / f"kernel{i:03d}.cu").write_text(s)
        # the record must point at the .cu it was written next to, so that a
        # later claim computed from `hot_source` is computed from the same text
        if rec.get("hot_index") is not None:
            rec["hot_source"] = str((d / f"kernel{rec['hot_index']:03d}.cu")
                                    .resolve())
            rec["source_dir"] = str(d.resolve())
        (d / "census.json").write_text(json.dumps(rec, indent=1))
    else:
        # A census record with no sources next to it would let the SASS stage
        # compile a different program from the one that was hashed. Loud.
        raise AssertionError(
            f"{rec['port']}/{rec['label']}: the census recorded "
            f"{rec['kernel_count']} kernels but {0 if sink is None else len(sink)} "
            f"sources are in hand. Refusing to write a census whose .cu files "
            f"do not correspond to it.")


def _print_table(records: list[dict]) -> None:
    print()
    print("  one row per (port, knob configuration). `same as default` is a "
          "BYTE comparison")
    print("  of the hot kernel's sha256, not an impression.")
    print()
    hdr = (f"  {'port':<17} {'knob configuration':<32} {'kern':>5} {'hot':>5} "
           f"{'src lines':>10} {'g_reads':>8} {'g_writes':>9} {'local':>6} "
           f"{'lb':>5} {'vs default':>18}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    default_hot: dict[str, str] = {}
    for r in records:
        if r["label"] == "default":
            default_hot[r["port"]] = r.get("hot_sha256") or ""

    for r in records:
        hot = r.get("hot_fingerprint") or {}
        if r["label"] == "default":
            same = "(control)"
        elif r.get("hot_sha256") == default_hot.get(r["port"]):
            same = "IDENTICAL source"
        else:
            same = "**SOURCE DIFFERS**"
        print(f"  {r['port']:<17} {r['label']:<32} {r['kernel_count']:>5} "
              f"{(r['hot_index'] if r.get('hot_index') is not None else -1):>5} "
              f"{hot.get('lines', 0):>10} {hot.get('g_reads', 0):>8} "
              f"{hot.get('g_writes', 0):>9} {hot.get('local_elems', 0):>6} "
              f"{(hot.get('launch_bounds') or '-'):>5} {same:>18}")


if __name__ == "__main__":
    sys.exit(main())
