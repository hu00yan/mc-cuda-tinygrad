#!/usr/bin/env python3
"""harness.py -- the measurement. Both arms, one process, one card, one run.

THE DESIGN, AND WHY IT IS THIS SHAPE
====================================
The task requires that each rung's hand-written point and its TinyGrad point be
"measured in the same run on the same card", and the previous two comparisons in
this study were invalid not because of what they measured but because of what
they compared. So this harness is built to make three things impossible rather
than merely unlikely:

1. **A DIFFERENT CARD, OR A DIFFERENT RUN.** There is one process. The
   hand-written kernels are loaded as modules into *TinyGrad's own CUDA context*
   (`CUDADevice.context`), launched on the same stream, and timed with the same
   cuEvent path TinyGrad's `CUDAProgram.__call__` uses. The reference does not
   run in a subprocess and the two arms never see different clocks, different
   driver state, or different thermal conditions.

2. **DIFFERENT DATA.** Both arms read the SAME device buffers. The inputs are
   generated once by `rungs.py`'s `inputs()` with a fixed seed, uploaded once,
   and both arms are handed pointers into the same allocation. A disagreement
   about the data cannot hide inside a performance comparison.

3. **A RATIO BETWEEN TWO DIFFERENT COMPUTATIONS.** Every rung is gated BEFORE it
   is timed: the reference and the TinyGrad arm are both compared against a
   float64 oracle on the host, and against each other. A rung whose two arms
   disagree by more than its tolerance is reported as INVALID and no ratio is
   printed for it. notes/06 §5b is the reason: two comparisons in this study
   passed every correctness check while being 72x off, because the thing that was
   ported was not the thing the GPU runs.

WHY A CUBIN AND NOT A SUBPROCESS
================================
nvcc is available on the Kaggle T4 image (minecu_ref/kaggle/make_notebook.py
checks for it and the study's own runs depend on that). The harness compiles
`refkernels.cu` to a device-only cubin for sm_75 ON THE CARD, loads it with
`cuModuleLoadData`, and launches with `cuLaunchKernel`. That is what makes both
arms share a context, a stream and a clock. It is also why the hand-written side
is a cubin rather than a linked binary: a cubin has no host code and no
allocator, so it can only touch memory this process already owns.

WHAT IS AND IS NOT MEASURED HERE
================================
  * WALL CLOCK, via cuEvent, on the device. Never instruction counts. Three
    separate census-based attempts in this project disagreed with a real device
    (notes/06 C-20, C-22), and one profiling approach produced a number that was
    never produced at all.
  * THE SAME cuEvent INSTRUMENT FOR BOTH ARMS. The ratio is therefore a ratio of
    two measurements from one clock, not of two measurement pipelines.
  * A MEASURED MEMORY FLOOR (axis_fill, axis_copy) in the same run, so every
    bandwidth figure is a percentage of what THIS card actually did, not of a
    datasheet number. This is minecu_ref's discipline and it is the only reason
    a "% of peak" claim in this repo is worth anything.
  * Hardware counters (ncu/nsys) are NOT collected. Kaggle does not expose them
    and a container cannot grant the privilege. Section 6 of the note states
    which conclusions survive without them; the short version is that every
    conclusion here is a wall-clock ratio plus a roofline position, and none of
    them depends on a counter.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("PARALLEL", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import rungs as R  # noqa: E402
import verify as V  # noqa: E402

DEV = os.environ.get("DEV", "CUDA")


def log(*a):
    print(*a, flush=True)


def section(t: str) -> None:
    log("")
    log("=" * 78)
    log(f"  {t}")
    log("=" * 78)


# ===========================================================================
# the driver-API shim
# ===========================================================================
# Everything here is reached through TinyGrad's own autogen bindings, so the
# context, the stream and the allocator are the framework's. Nothing here is on
# the TinyGrad side of the comparison; `verify.py` proves `rungs.py` cannot reach
# any of it.

class Cuda:
    def __init__(self):
        from tinygrad.runtime.autogen import cuda as _c
        self.c = _c
        from tinygrad.runtime.support.c import init_c_var
        self.init_c_var = init_c_var

    def check(self, status, what=""):
        if status != 0:
            raise RuntimeError(f"CUDA error {status} in {what}")

    def event_timer(self):
        c, mk = self.c, self.init_c_var
        return c, mk


def load_module(cuda_mod, image: bytes, ctx):
    """Load a cubin into TinyGrad's context. Returns the CUmodule."""
    c = cuda_mod
    check = lambda s, w="": (_ for _ in ()).throw(RuntimeError(f"CUDA error {s} in {w}")) if s else None
    from tinygrad.runtime.support.c import init_c_var
    check(c.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    mod = c.CUmodule()
    buf = ctypes.create_string_buffer(image, len(image))
    check(c.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)), "cuModuleLoadData")
    return mod


def get_function(cuda_mod, mod, name: str):
    c = cuda_mod
    from tinygrad.runtime.support.c import init_c_var
    fn = c.CUfunction()
    st = c.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode())
    if st != 0:
        raise RuntimeError(f"cuModuleGetFunction({name!r}) -> {st}; "
                           f"the kernel is not in this cubin under that name")
    return fn


def launch(cuda_mod, fn, grid, block, args, smem=0, stream=None):
    """cuLaunchKernel with a ctypes-built parameter array.

    Pointers are passed as CUdeviceptr_v2 (a 64-bit device address) and scalars
    as their natural width, matching the ABI nvcc emits. Every parameter is
    wrapped in its own ctypes object and the array of pointers to them is what
    the driver reads, which is the documented calling convention.
    """
    c = cuda_mod
    holders, ptrs = [], []
    for a in args:
        if isinstance(a, int):
            h = ctypes.c_int32(a) if -(2 ** 31) < a < 2 ** 31 else ctypes.c_int64(a)
        elif isinstance(a, float):
            h = ctypes.c_float(a)
        else:                                   # a device address
            h = c.CUdeviceptr_v2(int(a))
        holders.append(h)
        ptrs.append(ctypes.cast(ctypes.byref(h), ctypes.c_void_p))
    arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
    st = c.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                          smem, stream, arr, None)
    if st != 0:
        raise RuntimeError(f"cuLaunchKernel -> {st}")


def timed(cuda_mod, ctx, call, iters: int, warmup: int) -> dict:
    """Median / min / max device time in ms, via cuEvent, on TinyGrad's stream.

    The event pair brackets the WHOLE loop rather than each call, so per-call
    event overhead is not being timed as if it were work. `warmup` untimed calls
    absorb both the first-touch page faults and the autotuner's first run.
    """
    c = cuda_mod
    from tinygrad.runtime.support.c import init_c_var
    check = lambda s, w="": (_ for _ in ()).throw(RuntimeError(f"CUDA error {s} in {w}")) if s else None
    check(c.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    for _ in range(warmup):
        call()
    check(c.cuCtxSynchronize(), "warmup sync")
    ts = []
    for _ in range(iters):
        e0 = init_c_var(c.CUevent, lambda x: c.cuEventCreate(ctypes.byref(x), 0))
        e1 = init_c_var(c.CUevent, lambda x: c.cuEventCreate(ctypes.byref(x), 0))
        check(c.cuEventRecord(e0, None), "record")
        for _ in range(1):
            call()
        check(c.cuEventRecord(e1, None), "record")
        check(c.cuEventSynchronize(e1), "event sync")
        ms = ctypes.c_float()
        check(c.cuEventElapsedTime(ctypes.byref(ms), e0, e1), "elapsed")
        ts.append(ms.value)
        for e in (e0, e1):
            c.cuEventDestroy_v2(e)
    ts.sort()
    return {"median_ms": statistics.median(ts), "min_ms": ts[0], "max_ms": ts[-1],
            "iters": iters, "all_ms": ts}


# ===========================================================================
# device memory that BOTH arms can address
# ===========================================================================
# The buffers are TinyGrad Tensors, so they live in TinyGrad's allocator and the
# reference is handed the raw address. This is the mechanism that makes "the same
# data" true rather than aspirational: there is one allocation and both arms
# write into it.


def as_tensor(x):
    """Wrap a numpy array in a Tensor, and pass a Tensor through unchanged.

    This exists because of a measurement bug that invalidated every full-shape
    number in the first completed run. The rung builders were called with the
    NUMPY inputs, so `Tensor(d["a"])` inside each builder put a host-to-device
    COPY inside the graph -- three copies of 67 MB for `ew` at n=16.7M -- and
    the "device time" then included the PCIe traffic. The rung read 44.1 ms
    against a 1.08 ms reference, and essentially all of the difference was the
    upload, not the arithmetic.

    The builders now receive the already-uploaded device Tensors, so the graph
    contains the kernels and only the kernels. The oracle still gets the numpy
    arrays, and the reference still gets the raw addresses; all three views of
    the same bytes, which is the point of the design.
    """
    from tinygrad import Tensor
    return x if isinstance(x, Tensor) else Tensor(x)


def dev_ptr(t) -> int:
    """The device address of a realised TinyGrad Tensor's buffer."""
    b = t.uop.buffer
    b.ensure_allocated()
    raw = b.get_buf(t.device)
    val = getattr(raw, "value", raw)
    if val is None and hasattr(raw, "_buf"):
        val = getattr(raw._buf, "value", raw._buf)
    if val is None:
        raise RuntimeError(f"cannot get a device address from {type(raw)}")
    return int(val)


def upload(np_arr):
    """A numpy array as a realised TinyGrad Tensor on the device.

    `.contiguous().realize()` rather than `.realize()`: the first realise
    materialises a COPY node whose buffer is the staging allocation, and the
    address handed to a cubin has to be the address of a plain BUFFER. The
    double realise is what turns it into one. (notes/26 6f: realize on an
    already-realised graph is a silent no-op, so the pair is also the thing
    that makes a repeated call safe rather than a no-op that reads as ~3ns.)
    """
    from tinygrad import Tensor
    t = Tensor(np.ascontiguousarray(np_arr))
    t = t.contiguous()
    t.realize()
    return t


# ===========================================================================
# TinyGrad arm timing
# ===========================================================================
# `Tensor.realize()` on an already-realised graph is a silent no-op (notes/26
# 6f), so the graph is rebuilt on every timed call; otherwise this reads as
# ~3 ns, which is a number this study has already been fooled by. What is timed
# is `realize()` plus a device synchronize, which is device time plus the
# launch bookkeeping, and the SAME bracketing is applied to the reference.


def tg_time(build, cuda_mod, ctx, iters: int, warmup: int) -> dict:
    """END-TO-END time: build the graph, realize it, synchronize.

    Reported for completeness and NOT used for the ratio. It contains the Python
    cost of constructing the graph, which for a small rung dominates everything
    else: a first measurement of `ew` at n=4096 read 3.81 ms for TinyGrad
    against 0.025 ms for the reference, and essentially all of that 3.81 ms was
    Python, not the GPU. Dividing device time by wall time is the error
    notes/06 C-03 records, in a new place.

    The reference arm has no such term -- a cubin launch is one C call -- so this
    number is not comparable to it and the note says so at the point of use.
    """
    from tinygrad import Device
    dev = Device[DEV]

    def one():
        build().realize()
        dev.synchronize()

    return timed(cuda_mod, ctx, one, iters, warmup)


def tg_device_time(build, cuda_mod, ctx, iters: int, warmup: int) -> dict:
    """DEVICE time of the compiled program, with the Python build excluded.

    This is the number the ratio uses, and getting it right is the single
    largest correctness question in the harness.

    How it works: build the graph once, take its LINEAR program, compile and
    link it exactly as `Tensor.realize` would, and then replay the compiled
    calls inside the same cuEvent bracket used for the reference. The graph is
    NOT rebuilt per iteration -- the framework's own codegen has already happened,
    and what is left is the sequence of kernel launches, which is precisely what
    the cubin side also is.

    `run_linear(..., jit=True)` skips both `compile_linear` and `link_linear`
    (realize.py:317), so no compilation happens inside the timed region. The
    event brackets the same region the reference's brackets.

    The two numbers together are the honest statement: `tg_device_ms` is the
    GPU work, `tg` is what it costs to ask for it in Python, and the difference
    is this framework's other cost.
    """
    from tinygrad import Device, Tensor
    from tinygrad.engine.realize import (ExecContext, compile_linear, link_linear,
                                         pm_exec, run_linear)
    from tinygrad.uop.ops import UOp, Ops
    dev = Device[DEV]

    t = build()
    linear, var_vals = Tensor.linear_with_vars(t)
    # the input uops are the realized buffers the graph reads, discovered from
    # the graph itself rather than supplied by us
    inputs = [u for u in _collect_buffers(linear)]
    lin = link_linear(compile_linear(linear), cache=True)

    def one():
        ctx_obj = ExecContext(var_vals, tuple(inputs), update_stats=False, jit=True)
        for call in lin.src:
            pm_exec.rewrite(call, ctx_obj)
        dev.synchronize()

    return timed(cuda_mod, ctx, one, iters, warmup)


def _collect_buffers(uop, seen=None):
    """Every BUFFER uop reachable from a graph, deduplicated.

    `run_linear`'s `input_uops` is what `resolve_params` substitutes into the
    call's parameters; without it the compiled program has nothing to read from
    on a replay. They are found by walking the linear rather than passed in, so
    the set is whatever the graph actually references.
    """
    seen = set() if seen is None else seen
    from tinygrad.uop.ops import Ops
    if uop.op is Ops.BUFFER:
        key = (uop.arg.key if hasattr(uop.arg, "key") else id(uop))
        if key not in seen:
            seen.add(key)
            yield uop
    for s in uop.src:
        if s is not None and isinstance(getattr(s, "op", None), object):
            try:
                yield from _collect_buffers(s, seen)
            except RecursionError:
                return


def kernel_count(build) -> int:
    from tinygrad import GlobalCounters
    GlobalCounters.reset()
    build().realize()
    return GlobalCounters.kernel_count


# ===========================================================================
# the per-rung measurement
# ===========================================================================

def _ptx_param_patterns(ptx_text: str) -> dict:
    """Each kernel's parameter TYPE sequence, read out of the PTX.

    nvrtc/nvcc emit `.visible .entry name(.param .u64 ..., .param .u32 ...)`,
    and the leading type of each parameter is enough to distinguish a pointer
    from a 32-bit int. That gives a pattern like `pppiii`, which is checked
    against `Rung.ref_params` before anything is timed.

    This is the check that turns a lucky catch into a designed one. The two
    GEMM kernels originally declared the output pointer THIRD, so their pattern
    was `ppiii` where the harness expected `pppiii`; the launch itself was
    well-formed, the kernel ran, and the reference quietly overwrote an input
    buffer. Only the oracle comparison noticed, which is a coincidence, not a
    control.
    """
    pats: dict[str, str] = {}
    # The entry's parameter list is a parenthesised run of `.param .TYPE ...`
    # ending at the matching close paren. A non-greedy `\((.*?)\)` stops at the
    # FIRST `)`, which truncates or, for a kernel whose params span a line
    # break, runs on into the next entry -- the first version of this regex
    # reported axis_gemm_f32 as a 27-parameter kernel. So: find the entry, then
    # take everything up to the first `{` that follows the opening paren, which
    # is where PTX puts the function body.
    for m in re.finditer(r"\.visible\s+\.entry\s+(\w+)\s*\(", ptx_text):
        name = m.group(1)
        start = m.end()
        end = ptx_text.find("{", start)
        if end < 0:
            continue
        body = ptx_text[start:end]
        seq = ["p" if pm.group(1) in ("u64", "b64") else "i"
               for pm in re.finditer(r"\.param\s+\.(\w+)", body)]
        pats[name] = "".join(seq)
    return pats


def measure_rung(r, shapes: dict, iters: int, warmup: int, do_naive: bool, do_knob: bool) -> dict:
    from tinygrad import Device, GlobalCounters
    from tinygrad.helpers import Context

    out: dict = {"key": r.key, "title": r.title, "demands": r.demands,
                 "shapes": dict(shapes), "distinguishing_evidence": r.distinguishing_evidence}

    d = r.inputs(shapes)
    out["input_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
                           for k, v in d.items() if isinstance(v, np.ndarray)}
    out["seed_note"] = "fixed seeds in axis/rungs.py; both arms read the same buffers"

    dev = Device[DEV]
    t0 = time.time()
    tensors = {k: upload(v) for k, v in d.items() if isinstance(v, np.ndarray)}
    dev.synchronize()
    out["upload_s"] = round(time.time() - t0, 3)
    ptrs = {k: dev_ptr(v) for k, v in tensors.items()}

    want = V.ORACLES[r.key](d)          # float64, independent transcription

    # An input-integrity fingerprint, taken BEFORE anything runs. The reference
    # and the oracle must see the same bytes, and this is the cheapest way to
    # know that they do -- the alternative is discovering it as a mysterious
    # 0.068 in a scan that is provably correct in PTX, in SASS, and in a
    # Python transcription, which is where two long sessions went.
    def fingerprint():
        out = {}
        for k, t in tensors.items():
            raw = t.numpy()
            out[k] = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()[:16]
        return out

    fp_before = fingerprint()

    # The builders get the DEVICE tensors, the oracle got the numpy. Passing
    # numpy here would put a host-to-device copy inside every timed graph, which
    # is the bug described on `as_tensor` above and which inflated `ew` by 40x
    # in the first completed run.
    ddev = dict(d)
    ddev.update(tensors)

    # ---------------- the hand-written arm FIRST ----------------
    #
    # Order is a correctness decision, not a style one. The reference is timed
    # and gated BEFORE the framework touches anything, so its result cannot
    # depend on what the framework's replay did to the shared buffers. It was
    # run second, and the scan rung came back wrong by 0.068 with an error the
    # PTX, the SASS and a Python transcription all say is impossible -- which is
    # the signature of an input that changed underneath it.
    ref_block: dict = {}
    ref_np = None
    try:
        # The scan's symbol depends on N: the vectorised path is a macro
        # specialised on the float4s per thread, and the scalar fallback is a
        # separate entry point. `axis_scan` no longer exists, so the symbol is
        # RESOLVED here rather than declared on the rung.
        # The scan's symbol depends on N. Two different things depend on it, and
        # conflating them is a defect this study has already paid for once:
        #   * WHICH CORRECTNESS GATE the shape admits (float4 spans, so N must be
        #     a multiple of 1024 with a specialisation, or <= 256 for the scalar
        #     fallback);
        #   * WHICH KERNEL is the fastest CORRECT one, because PT >= 2 always
        #     spans 4096 and any larger N is also a multiple of 1024, so the
        #     specialisation with the FEWEST float4s per thread covers it.
        # Choosing by correctness alone picked `axis_scan_v4` for N=4096 and
        # `axis_scan_v1` for N=1024, and the width sweep then measured the
        # DENOMINATOR swinging 3.6x across the threshold -- which looked exactly
        # like the discontinuity the sweep existed to detect, and was an artefact
        # of the selector. The rule is: fewest PT that covers the span, falling
        # back to the scalar path only when no specialisation does.
        kernel_name = (_ref_scan_symbol_for_span(shapes) if r.key == "scan"
                       else r.ref_kernel)
        ref_block["kernel"] = kernel_name
        REF_SYMBOL[0] = kernel_name
        fn = get_function(CUR, MOD, kernel_name)
        grid = r.ref_grid(shapes)
        block = r.ref_block(shapes)
        ref_out = upload(np.zeros(want.shape, dtype=np.float32 if r.out_dtype == "float32"
                                  else np.float16))
        dev.synchronize()
        rptr = dev_ptr(ref_out)
        # The output buffer must not ALIAS an input. If it does, the reference
        # is computing in place, and the signature of that is a scan whose error
        # GROWS with the column. TinyGrad's LRU allocator recycles freed blocks,
        # so rather than reason about whether an output allocation can land on a
        # live input, CHECK.
        aliased = sorted(n for n, p in ptrs.items() if p == rptr)
        ref_block["out_ptr"] = int(rptr)
        ref_block["in_ptrs"] = {k: int(v) for k, v in ptrs.items()}
        if aliased:
            out["ref_error"] = f"output buffer aliases input(s) {aliased}"
            out["status"] = "INVALID_buffer_aliasing"
            dev.synchronize()
            return out
        args = _ref_args(r, shapes, d, ptrs, rptr)

        def one_ref():
            launch(CUR, fn, grid, block, args)

        t = timed(CUR, CTX, one_ref, iters, warmup)
        ref_block.update({k: t[k] for k in ("median_ms", "min_ms", "max_ms")})
        ref_block["grid"] = list(grid)
        ref_block["block"] = list(block)
        ref_block["smem"] = SASS_SMEM.get(kernel_name)
        ref_block["regs"] = SASS_REGS.get(kernel_name)
        ref_np = ref_out.numpy()
        ref_block["relerr_vs_oracle"] = V.relerr(np.asarray(ref_np, dtype=np.float64), want)
        derr = np.abs(np.asarray(ref_np, dtype=np.float64) - want)
        flat = derr.ravel()
        kk = int(flat.argmax())
        nz = int((flat > 0.05 * flat.max()).sum())
        diag = {
            "worst_abs": float(flat[kk]), "n_elements": int(flat.size),
            "n_bad": nz, "frac_bad": round(nz / max(1, flat.size), 4),
            "worst_index": list(np.unravel_index(kk, derr.shape)),
            "ref_at_worst": float(np.asarray(ref_np, dtype=np.float64).ravel()[kk]),
            "want_at_worst": float(want.ravel()[kk]),
        }
        try:
            if d.ndim >= 2:
                prof = derr[derr.shape[0] // 2]
                idx = np.linspace(0, prof.size - 1, 24).astype(int)
                diag["row_profile"] = {"cols": [int(i) for i in idx],
                                      "errs": [round(float(prof[int(i)]), 6) for i in idx]}
                diag["worst_cols"] = [int(i) for i in np.argsort(prof)[-8:][::-1]]
        except Exception:
            pass
        ref_block["diag"] = diag
        out["bytes"] = r.bytes_moved(shapes) if r.bytes_moved else None
        out["flops"] = r.flops(shapes) if r.flops else None
    except Exception as e:
        ref_block["error"] = f"{type(e).__name__}: {e}"
        ref_block["traceback"] = traceback.format_exc()[-1200:]
    out["ref"] = {k: v for k, v in ref_block.items()
                  if k in ("median_ms", "min_ms", "max_ms")}
    out["ref_kernel"] = ref_block.get("kernel")
    out["ref_grid"] = ref_block.get("grid")
    out["ref_block"] = ref_block.get("block")
    out["ref_smem"] = ref_block.get("smem")
    out["ref_regs"] = ref_block.get("regs")
    out["ref_relerr_vs_oracle"] = ref_block.get("relerr_vs_oracle")
    out["ref_diag"] = ref_block.get("diag")
    out["ref_error"] = ref_block.get("error")

    # ---------------- the TinyGrad arm ----------------
    def build_tg():
        return r.tg(shapes, ddev)

    t0 = time.time()
    tg_out = build_tg()
    tg_np = tg_out.numpy()
    out["tg_compile_s"] = round(time.time() - t0, 2)
    out["tg_shape"] = list(tg_np.shape)
    GlobalCounters.reset()
    build_tg().realize()
    out["tg_kernels"] = GlobalCounters.kernel_count
    out["tg_relerr_vs_oracle"] = V.relerr(np.asarray(tg_np, dtype=np.float64), want)
    # the two arms against EACH OTHER, which is a different question from either
    # against the oracle: a rung can be wrong in the same way on both sides
    if "ref_relerr_vs_oracle" in out and out["ref_relerr_vs_oracle"] is not None \
            and ref_np is not None:
        out["ref_vs_tg_relerr"] = V.relerr(np.asarray(ref_np, dtype=np.float64),
                                           np.asarray(tg_np, dtype=np.float64))

    t = tg_time(build_tg, CUR, CTX, iters, warmup)
    out["tg"] = {k: t[k] for k in ("median_ms", "min_ms", "max_ms")}
    out["tg_note"] = ("end-to-end: Python graph build + realize + sync. NOT the "
                      "ratio's numerator -- see tg_device.")
    try:
        td = tg_device_time(build_tg, CUR, CTX, iters, warmup)
        out["tg_device"] = {k: td[k] for k in ("median_ms", "min_ms", "max_ms")}
    except Exception as e:
        out["tg_device_error"] = f"{type(e).__name__}: {e}"

    # Did the framework's replay change the inputs underneath us? This is the
    # check that a Python transcription of a kernel, a correct PTX and a correct
    # SASS could not make: all three agree the scan is right, and it was still
    # wrong by 0.068 on the device, because the buffer it was reading had been
    # overwritten. The reference is now measured BEFORE this point, so its
    # number stands either way -- but if the fingerprint moved, the replay is
    # corrupting shared memory and every later rung inherits the risk, so it is
    # recorded loudly rather than merely noted.
    try:
        fp_after = fingerprint()
        moved = {k: (fp_before[k], fp_after[k]) for k in fp_before if fp_before[k] != fp_after[k]}
        out["input_fingerprint_before"] = fp_before
        out["input_fingerprint_after"] = fp_after
        out["inputs_mutated_by_tg_replay"] = moved
        if moved:
            out["input_mutation_warning"] = (
                f"the TinyGrad arm CHANGED the device contents of {sorted(moved)}; "
                f"the reference was measured before this and is unaffected, but a "
                f"replay that writes into its own inputs is a framework behaviour "
                f"worth knowing about")
    except Exception as e:
        out["input_fingerprint_error"] = f"{type(e).__name__}: {e}"

    # ---------------- the naive control ----------------
    # The report is flushed HERE, before the controls, because the controls are
    # the expensive and least essential part: the scan control builds N-1 graph
    # nodes and at the full N=4096 it was long enough to eat a whole 25-minute
    # session. A rung's two arms and its ratio are the result; a control is
    # context, and losing the context must not lose the result.
    out["status"] = _status_for(out)
    out["ratio_tg_over_ref"] = _ratio_for(out)
    _flush(REPORT_PATH[0], {"rungs_done": out})
    if do_naive and r.tg_naive is not None:
        # The naive spelling is a Python loop over the reduction axis, so its
        # cost is linear in the trip count and at scan's N=4096 it is 4 095
        # graph nodes to compile and launch -- minutes of session for a control.
        # It is therefore measured at a REDUCED trip count and the full-width
        # cost is stated as a scaling, not guessed: the naive form performs
        # N-1 dependent steps and materialises an [R,1] intermediate per step,
        # so its device cost is at least linear in N. What is printed is
        # measured; what is extrapolated is labelled.
        nshapes = dict(shapes)
        cap = 256
        if r.key == "scan" and nshapes.get("N", 0) > cap:
            nshapes["N"] = cap
        try:
            dnaive = r.inputs(nshapes)
            tn = {k: upload(v) for k, v in dnaive.items() if isinstance(v, np.ndarray)}
            Device[DEV].synchronize()
            dnaive = dict(dnaive)
            dnaive.update(tn)
            def build_naive():
                return r.tg_naive(nshapes, dnaive)
            t = tg_device_time(build_naive, CUR, CTX, max(3, iters // 4), 1)
            out["tg_naive"] = {k: t[k] for k in ("median_ms", "min_ms", "max_ms")}
            out["tg_naive_shapes"] = dict(nshapes)
            if nshapes.get("N", 0) < shapes.get("N", 0):
                scale = shapes["N"] / nshapes["N"]
                out["tg_naive_scaled_ms"] = t["median_ms"] * scale
                out["tg_naive_scale_note"] = (
                    f"[MODELLED] measured at N={nshapes['N']}, scaled by "
                    f"{scale:g}x to the rung's N={shapes['N']}. The naive form does "
                    f"N-1 dependent steps and writes an [R,1] intermediate per step, "
                    f"so its cost is at least linear in N; the scaling is a LOWER "
                    f"bound, not a prediction.")
        except Exception as e:
            out["tg_naive_error"] = f"{type(e).__name__}: {e}"

    # ---------------- the knob rung ----------------
    if do_knob and r.tg_knob is not None:
        label, kwargs, builder = r.tg_knob
        try:
            with Context(**kwargs):
                def build_knob():
                    return builder(shapes, ddev)
                t = tg_device_time(build_knob, CUR, CTX, iters, warmup)
                out["tg_knob"] = {k: t[k] for k in ("median_ms", "min_ms", "max_ms")}
                out["tg_knob_label"] = label
        except Exception as e:
            out["tg_knob_error"] = f"{type(e).__name__}: {e}"

    # ---------------- the verdict on comparability ----------------
    out["status"] = _status_for(out)
    out["ratio_tg_over_ref"] = _ratio_for(out)
    dev.synchronize()
    return out


def _status_for(out: dict) -> str:
    """OK only if both arms agree with the float64 oracle and with each other."""
    tol = {"gemm16": 5e-2, "trisolve": 1e-4}.get(out.get("key"), 1e-4)
    out["tolerance"] = tol
    ref_err = out.get("ref_relerr_vs_oracle")
    arms_agree = out.get("ref_vs_tg_relerr")
    # `.get(...) is not None`, not `in out`: the key is always present and is
    # None when there was no error, so an `in` test reported REF_UNAVAILABLE for
    # every rung in the run that introduced it. A status test that fires on a
    # None is the same shape as the stale-output bug in axis/kaggle/run.sh -- a
    # check that passes (or fails) for a reason unrelated to the thing it names.
    if out.get("ref_error") is not None:
        return "REF_UNAVAILABLE"
    if ref_err is None:
        return "REF_UNAVAILABLE"
    if ref_err > tol or out.get("tg_relerr_vs_oracle", 1.0) > tol:
        return "INVALID_reference_disagrees_with_oracle"
    if arms_agree is not None and arms_agree > max(tol, 1e-3):
        return "INVALID_arms_disagree"
    if "tg_device" not in out:
        return "INVALID_no_device_time"
    return "OK"


def _ratio_for(out: dict) -> float | None:
    """tg/ref on DEVICE time, or None if either side is missing or invalid.

    The denominator is the cubin and the numerator is the framework's compiled
    program replayed inside the same kind of cuEvent bracket. Not wall clock, and
    specifically not the end-to-end number, which carries a Python graph-build
    term the cubin side has no counterpart for.
    """
    if out.get("status") != "OK":
        return None
    try:
        return out["tg_device"]["median_ms"] / out["ref"]["median_ms"]
    except (KeyError, TypeError, ZeroDivisionError):
        return None


SCAN_THREADS, SCAN_VEC = 256, 4


def _ref_scan_symbol_for_span(s: dict) -> str:
    """The scan entry point to launch at this shape. THERE IS NO CHOICE.

    `SCAN_BODY` gives each thread a CONTIGUOUS run of `PT` float4s starting at
    `threadIdx.x * PT * SCAN_VEC`; the grid is one block per ROW; and there is
    **no grid-stride loop**. So a block of 256 threads covers exactly
    `256 * PT * 4 = 1024 * PT` elements of its row and nothing more, and a
    correct launch requires `1024 * PT == N` exactly. `PT` is forced by the
    shape, not chosen for speed.

    THIS FUNCTION WAS, BRIEFLY, A BUG, AND THE WIDTH SWEEP CAUGHT IT. An
    intermediate version returned `axis_scan_v1` for every N that was a
    multiple of 1024, on the reasoning that one float4 per thread is the
    shortest register run and therefore the fastest kernel. The sweep refuted
    that in a single run: at N=4096 with PT=1 a block covers 1024 of the row's
    4096 elements, leaves three quarters unwritten, and reports **INVALID
    against the oracle with a reference time of 0.047 ms -- FASTER than the
    correct kernel's 0.219 ms, because it is doing a quarter of the work.**

    That is the sharpest form of this study's recurring lesson and it is worth
    stating exactly: **a kernel that is correct for a smaller span is not a
    faster kernel for a larger one; it is a wrong one, and it is faster in
    proportion to how much of the answer it skipped.** Nothing that checks
    shapes, widths, types or compilation order can see it. Only the oracle can,
    because only the oracle knows what the answer was supposed to be.

    The separate function is kept anyway: "which kernels am I permitted to
    launch here" (a gate) and "which kernel did this launch use" (the record of
    a measurement) are different questions even when the answer is forced, and
    `verify.py` checks this one against the `.cu`'s signatures.
    """
    N = s["N"]
    span = SCAN_THREADS * SCAN_VEC
    if N % span == 0:
        per_thread = N // span
        if per_thread in (1, 2, 4, 8, 16):
            return f"axis_scan_v{per_thread}"
    if N <= SCAN_THREADS:
        return "axis_scan_scalar"
    raise KeyError(
        f"no scan kernel for N={N}: one block of 256 threads with PT float4s "
        f"each covers exactly 1024*PT elements and there is no grid-stride loop, "
        f"so N must be 1024*{1,2,4,8,16}; the scalar fallback only covers "
        f"N <= {SCAN_THREADS}. This rung is UNAVAILABLE at this shape rather "
        f"than measured with a kernel that would write part of the row.")


def _scan_symbol(s: dict) -> str:
    """Which of the two scan entry points a shape needs.

    The vectorised kernel is a macro specialisation on the number of float4s
    each thread owns, instantiated for 1/2/4/8/16, and requires N to be an exact
    multiple of THREADS*VEC.

    The scalar fallback is only correct for N <= THREADS: it gives each thread
    exactly one element, so a longer row is left partly unwritten. It is
    therefore offered ONLY in that range, and a shape outside both paths is
    reported as unavailable rather than silently given a kernel that will write
    part of the row and leave the rest stale. The first version selected it for
    any N it did not have a specialisation for, which is the "a kernel that only
    works at the shape it was measured at" failure in its purest form -- and it
    was RIGHT at the smoke shape (N=64) and wrong at any other.
    """
    N = s["N"]
    span = SCAN_THREADS * SCAN_VEC
    if N % span == 0:
        per_thread = N // span
        if per_thread in (1, 2, 4, 8, 16):
            return f"axis_scan_v{per_thread}"
    if N <= SCAN_THREADS:
        return "axis_scan_scalar"
    raise KeyError(
        f"no scan kernel for N={N}: the vectorised path needs N to be a multiple "
        f"of {span} with a specialisation in (1,2,4,8,16), and the scalar "
        f"fallback only covers N <= {SCAN_THREADS}. This rung is UNAVAILABLE at "
        f"this shape rather than measured with a kernel that would write part of "
        f"the row.")


# Set by `measure_rung` to the symbol actually launched, so `_ref_args` can
# dispatch on it. A module global because `_ref_args`'s signature predates
# multi-kernel rungs and threading it through would touch every call site for a
# value that is a property of the launch, not of the rung.
REF_SYMBOL: list[str] = [""]


def _ref_args(r, s, d, ptrs, out_ptr):
    """Each reference kernel's argument list, in signature order.

    Written out per rung rather than derived, because a silently mis-ordered
    argument list produces a kernel that runs and returns wrong numbers, and
    wrong numbers that are merely plausible are what this repository keeps
    having to take back.

    TWO bugs lived here and both are now structurally prevented rather than
    merely fixed:

    * `s` -- the SHAPES ACTUALLY ALLOCATED -- is a parameter, not a read of
      `r.shapes`. The first version read `r.shapes` while the buffers came from
      the smoke override, so every kernel was launched with the FULL shape's
      element count against a tiny allocation and every rung died with CUDA
      error 700.
    * `_assert_arg_counts_match_signatures` below checks the LENGTH of the list
      this function returns against the parameter count of the compiled kernel,
      read out of the cubin. The two GEMM kernels originally took the output
      pointer third while every other kernel -- and this function -- took it
      first, so the reference silently overwrote an INPUT buffer and read the
      output buffer as A. Length is checked automatically; order is the
      remaining thing a reader has to check, which is why every kernel here
      declares its output first and `_assert_arg_counts_match_signatures`
      refuses a mismatch.
    """
    if r.key == "ew":
        return (out_ptr, ptrs["a"], ptrs["b"], ptrs["z"], s["n"] // 4)
    if r.key == "red":
        return (out_ptr, ptrs["x"], ptrs["w"], s["R"], s["K"])
    if r.key == "gemm32":
        return (out_ptr, ptrs["A"], ptrs["B"], s["M"], s["N"], s["K"])
    if r.key == "gemm16":
        return (out_ptr, ptrs["A"], ptrs["B"], s["M"], s["N"], s["K"])
    if r.key == "gather":
        return (out_ptr, ptrs["tab"], ptrs["idx"], ptrs["s"], s["n"])
    if r.key == "scan":
        # ORDER MATTERS and the type-pattern check cannot see it, and there are
        # now TWO different orders in this one rung.
        #
        # `axis_scan_v{PT}` is `(const float* x, float* y, int R, int N)` --
        # input FIRST, the opposite of every other kernel in this file. Passing
        # (out, x) wrote the scan into the INPUT and read zeros from the output;
        # the result was a relative error of exactly 1.0, i.e. the output was
        # never written at all, and the signature check passed because
        # `(float*, float*, int, int)` and `(float*, float*, int, int)` are the
        # same pattern either way.
        #
        # `axis_scan_scalar` is `(float* y, const float* x, int R, int N)` --
        # output FIRST, the conventional order, and the OPPOSITE of its sibling.
        # The width sweep ran the scalar path (N <= 256) and every such point came
        # back INVALID against the oracle while the vectorised points were OK,
        # and the cause was here: one hardcoded order, two kernels, two
        # signatures. The fix is to dispatch on the SYMBOL, so the two entry
        # points cannot be confused again -- and `axis/simref.py` had already
        # verified both kernels correct against `np.cumsum` on the host, which is
        # why this was a launch bug and not a kernel bug.
        #
        # The generalisation, twice over: a type-pattern check verifies WIDTH,
        # never ORDER; and a per-rung argument list silently assumes the rung has
        # ONE kernel. A rung with a shape-dependent kernel needs the argument
        # list to depend on the kernel.
        if REF_SYMBOL[0] == "axis_scan_scalar":
            return (out_ptr, ptrs["x"], s["R"], s["N"])
        return (ptrs["x"], out_ptr, s["R"], s["N"])
    if r.key == "trisolve":
        return (out_ptr, ptrs["D"], s["B"])
    raise KeyError(r.key)


# ---- static properties of the reference kernels, read from the SASS the
# compile step already produced (free, and better than a guess) --------------

SASS_REGS: dict[str, int] = {}
SASS_SMEM: dict[str, int] = {}
PARAMS: dict[str, int] = {}


def _smem_of(_name):
    return SASS_SMEM.get(_name)


def _regs_of(name):
    return SASS_REGS.get(name)


def parse_ptxas(text: str) -> None:
    cur = None
    for line in text.splitlines():
        if "Compiling entry function" in line:
            cur = line.split("'")[1]
        elif cur and "Used " in line and "registers" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "registers" and i:
                    SASS_REGS[cur] = int(parts[i - 1])
                if p == "smem" and i:
                    SASS_SMEM[cur] = int(parts[i - 1])
            cur = None


# ===========================================================================
# the memory floor -- the denominator for every bandwidth claim
# ===========================================================================

def measure_floors(iters: int, warmup: int, n: int) -> dict:
    from tinygrad import Device
    dev = Device[DEV]
    out: dict = {}
    src = upload(np.zeros(n, dtype=np.float32))
    dst = upload(np.zeros(n, dtype=np.float32))
    dev.synchronize()
    ps, pd = dev_ptr(src), dev_ptr(dst)

    f_fill = get_function(CUR, MOD, "axis_fill")
    t = timed(CUR, CTX, lambda: launch(CUR, f_fill, ((n + 255) // 256, 1, 1), (256, 1, 1), (pd, n)),
             iters, warmup)
    out["fill_ms"] = t["median_ms"]
    out["fill_GBps"] = n * 4 / (t["median_ms"] * 1e-3) / 1e9

    f_copy = get_function(CUR, MOD, "axis_copy")
    t = timed(CUR, CTX, lambda: launch(CUR, f_copy, ((n + 255) // 256, 1, 1), (256, 1, 1), (pd, ps, n)),
             iters, warmup)
    out["copy_ms"] = t["median_ms"]
    # two directions: reads n and writes n
    out["copy_GBps"] = 2 * n * 4 / (t["median_ms"] * 1e-3) / 1e9
    out["n"] = n
    return out


# ===========================================================================
# main
# ===========================================================================

CUR = None
CTX = None
MOD = None
REPORT_PATH: list = [Path("axis_report.json")]


def environment() -> dict:
    section("1. environment")
    import tinygrad
    from tinygrad import Device
    dev = Device[DEV]
    # the version is read defensively: tinygrad 0.14.0 on the Kaggle image has no
    # `tinygrad.__version__`, and an ImportError on a reporting line cost the
    # first run of this harness the entire measurement. Nothing below depends on
    # it; the path is the more useful identifier anyway.
    ver = getattr(tinygrad, "__version__", None)
    if ver is None:
        try:
            import importlib.metadata as _md
            ver = _md.version("tinygrad")
        except Exception:
            ver = "unknown"
    env = {
        "Device.DEFAULT": DEV,
        "tinygrad": str(Path(tinygrad.__file__).resolve().parent),
        "tinygrad_version": ver,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        env["nvidia_smi"] = r.stdout.strip()
    except Exception as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    for k, v in env.items():
        log(f"  {k:<18} {v}")
    return env


def compile_reference(sass: Path) -> dict:
    """nvcc the reference for sm_75 ON THE CARD, and capture ptxas -v.

    On the card rather than on the host because (a) the card's nvcc is the one
    whose ptxas the kernels will actually meet, and (b) this is the step whose
    output the register/spill gate below reads.
    """
    section("2. compile the reference for sm_75 (on this card)")
    nvcc = None
    for cand in ("/usr/local/cuda/bin/nvcc", "nvcc"):
        r = subprocess.run(["bash", "-lc", f"command -v {cand}"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            nvcc = r.stdout.strip()
            break
    if nvcc is None:
        raise SystemExit("FATAL: no nvcc on this runner. minecu_ref's T4 runs depend on it being there.")
    cu = HERE / "refkernels.cu"
    cubin = HERE / "_ref_sm75.cubin"
    t0 = time.time()
    cmd = [nvcc, "-arch=sm_75", "-O3", "--use_fast_math", "-std=c++17",
           "-diag-suppress", "177", "-Xptxas", "-v",
           "-cubin", "-o", str(cubin), str(cu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"  nvcc rc={r.returncode} in {time.time()-t0:.1f}s   {nvcc}")
    if r.returncode != 0:
        log(r.stdout[-3000:])
        log(r.stderr[-3000:])
        raise SystemExit("FATAL: the reference did not compile")
    (HERE / "_ref_ptxas.txt").write_text(r.stdout + r.stderr)
    parse_ptxas(r.stdout + r.stderr)

    # The PTX, for the parameter-signature check. Compiled alongside the cubin
    # from the same source with the same flags, so it describes the same kernels.
    ptx = HERE / "_ref_sm75.ptx"
    rp = subprocess.run([nvcc, "-arch=sm_75", "-O3", "--use_fast_math", "-std=c++17",
                         "-diag-suppress", "177", "-ptx", "-o", str(ptx), str(cu)],
                        capture_output=True, text=True)
    if rp.returncode != 0:
        log(rp.stdout[-2000:])
        log(rp.stderr[-2000:])
        raise SystemExit("FATAL: the reference did not compile to PTX")
    ptx_text = ptx.read_text()

    # the SASS instruction census, from the same nvcc run. `cuobjdump -sass`
    # needs no device.
    sass = HERE / "ref.sass"
    r2 = subprocess.run(["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True)
    if r2.returncode != 0 or "Function :" not in r2.stdout:
        for cand in ("/usr/local/cuda/bin/cuobjdump", "cuobjdump"):
            r2 = subprocess.run([cand, "-sass", str(cubin)], capture_output=True, text=True)
            if r2.returncode == 0 and "Function :" in r2.stdout:
                break
    sass.write_text(r2.stdout)
    PARAMS.clear()
    PARAMS.update(_ptx_param_patterns(ptx_text))
    log(f"  SASS {len(r2.stdout)} B, {len(PARAMS)} kernels")
    log(f"  {'kernel':<18} {'pattern':>8}  {'expected':>8}  match")
    bad = []
    for r in V.RUNGS:
        # the scan's symbol is resolved the same way the launch resolves it, so
        # the check and the launch cannot disagree about which kernel is which
        name = _scan_symbol(r.shapes) if r.key == "scan" else r.ref_kernel
        got = PARAMS.get(name, "?")
        ok = (got == r.ref_params)
        log(f"  {name:<18} {got:>8}  {r.ref_params:>8}  {'yes' if ok else 'NO'}")
        if not ok:
            bad.append((name, got, r.ref_params))
    if bad:
        log("")
        log("  FATAL: a reference kernel's compiled parameter list does not match the")
        log("  list the harness builds for it. Launching anyway would overwrite the")
        log("  wrong buffer with no error, so this stops the run:")
        for name, got, want in bad:
            log(f"    {name}: compiled {got}, harness expects {want}")
        raise SystemExit("FATAL: reference signature mismatch")
    log("  all reference signatures match the launcher's argument lists")
    image = cubin.read_bytes()
    log(f"  cubin {len(image)} B  sha256 {hashlib.sha256(image).hexdigest()[:16]}")
    log(f"  {'kernel':<18} {'regs':>6} {'smem B':>9} {'params':>7}")
    for name in sorted(SASS_REGS):
        log(f"  {name:<18} {SASS_REGS[name]:>6} {SASS_SMEM.get(name, 0):>9}"
            f" {PARAMS.get(name, 0):>7}")
    return {"cubin_bytes": len(image), "cubin_sha256": hashlib.sha256(image).hexdigest(),
            "nvcc": nvcc, "regs": dict(SASS_REGS), "smem": dict(SASS_SMEM),
            "param_counts": dict(PARAMS)}


def _print_sweep(sweep: list) -> None:
    """Print the sweep and, more importantly, whether the threshold shows.

    The verdict is computed rather than narrated: a DISCONTINUITY is the ratio of
    the measured tg ns/element on each side of N=512. A smooth reference across
    the same points is the control that makes the comparison mean anything, and
    it is printed beside the framework's own line so the reader can see that the
    two disagree rather than being told that they do.
    """
    log(f"  {'N':>7} {'R':>8} {'tg ms':>10} {'ref ms':>10} {'ratio':>8} "
        f"{'tg ns/el':>10} {'ref ns/el':>10} {'k':>5}")
    log("  " + "-" * 74)
    for row in sweep:
        tg = row.get("tg_ms")
        rf = row.get("ref_ms")
        log(f"  {row['N']:>7} {row['R']:>8} "
            f"{('%.4f' % tg) if tg else '-':>10} "
            f"{('%.4f' % rf) if rf else '-':>10} "
            f"{('%.2fx' % row['ratio']) if row.get('ratio') else '-':>8} "
            f"{('%.4f' % row['ns_per_elem_tg']) if row.get('ns_per_elem_tg') else '-':>10} "
            f"{('%.4f' % row['ns_per_elem_ref']) if row.get('ns_per_elem_ref') else '-':>10} "
            f"{str(row.get('kernels', '-')):>5}")

    ok = [r for r in sweep if r.get("ns_per_elem_tg") and r.get("ns_per_elem_ref")]
    if len(ok) < 2:
        log("  too few successful points to say anything")
        return
    THRESH = 512
    below = [r for r in ok if r["N"] <= THRESH]
    above = [r for r in ok if r["N"] > THRESH]
    log("")
    if not below or not above:
        log(f"  the sweep does not straddle the threshold N={THRESH}; no verdict")
        return

    def median(rs, k, default=0.0):
        """Median over the points that HAVE the field.

        Points where the reference was UNAVAILABLE carry `None`, and sorting
        those raises -- which happened on the first run of this function. A
        missing measurement is skipped and the count is printed, so a gap in the
        data cannot silently become a zero and then a division by zero.
        """
        v = sorted(r[k] for r in rs if isinstance(r.get(k), (int, float)))
        if not v:
            return default
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    # THE NORMALISATION IS THE WHOLE ARGUMENT, SO IT IS STATED RATHER THAN
    # ASSUMED. ROWS x N is held CONSTANT across the sweep. Every point therefore
    # moves the same 2^22 elements, and for any kernel whose cost is
    # proportional to its data -- which a correct scan is, for both arms -- the
    # TOTAL TIME *is* the per-element cost up to one constant divisor. So the
    # per-element column is a fixed rescaling of `ms` and carries no extra
    # information, and an earlier version of this function reported a
    # "ns per element" that fell monotonically as 1/N purely because it divided
    # by N while the data stayed fixed. That is a normalisation artefact wearing
    # the costume of a measurement, and it is the same species of error as this
    # note's defect 10. The verdict below is computed on milliseconds and on the
    # ratio, both of which are invariant to that choice.
    log("")
    log(f"  threshold N={THRESH} = SPLIT*2, where mixin/op.py switches cumsum's")
    log(f"  decomposition. ROWS x N is held constant, so every point moves the same")
    log(f"  2^22 elements and total time IS per-element cost up to a constant.")
    log("")
    log(f"    {'side':<18} {'n':>3} {'tg ms (median)':>16} {'ref ms (median)':>16} "
        f"{'ratio (median)':>16}")
    for tag, rs in (("N <= 512", below), ("N >  512", above)):
        nrat = sum(1 for r in rs if isinstance(r.get("ratio"), (int, float)))
        nref = sum(1 for r in rs if isinstance(r.get("ref_ms"), (int, float)))
        extra = (f"   ({nrat}/{len(rs)} points have a ratio)" if nrat != len(rs)
                 else (f"   ({nref}/{len(rs)} have a reference)" if nref != len(rs) else ""))
        log(f"    {tag:<18} {len(rs):>3} {median(rs, 'tg_ms'):>16.4f} "
            f"{median(rs, 'ref_ms'):>16.4f} "
            f"{(median(rs, 'ratio') if nrat else float('nan')):>16.2f}{extra}")
    mtg = median(below, "tg_ms")
    atg = median(above, "tg_ms")
    tstep = atg / mtg if mtg else float("inf")
    log("")
    log(f"    framework: {mtg:.4f} -> {atg:.4f} ms   ({tstep:.2f}x step)")
    log("")

    # A CONFOUND, STATED BEFORE THE VERDICT RATHER THAN DISCOVERED BY IT.
    #
    # The reference DOES change entry point across 512: `axis_scan_scalar` is the
    # only correct kernel at N <= 256 (one element per thread, no float4), and
    # `axis_scan_v1` is the one above it. So the reference's step is not expected
    # to be zero, and "the reference does not step" would be a claim this harness
    # CANNOT support. What both arms share is the DATA and the card; what the
    # framework's side adds at 512 is a change of decomposition, and what the
    # reference's side adds is a change of kernel that does not change its
    # algorithm.
    #
    # So the ratio is the load-bearing quantity: it divides out the reference's
    # own entry-point switch, leaving how the framework's per-element cost
    # compares to a hand-written scan's. If cumsum's decomposition is the cause,
    # the RATIO steps at 512. If the ratio is flat across 512, the 26x is a
    # property of the whole N > 512 regime and the threshold story is wrong --
    # which is a real, and reported, outcome.
    br = median(below, "ratio", default=0.0)
    ar = median(above, "ratio", default=0.0)
    if br > 0 and ar > 0:
        rstep_ratio = ar / br
        log(f"    ratio (the load-bearing quantity): {br:.2f}x -> {ar:.2f}x   "
            f"({rstep_ratio:.2f}x step across 512)")
        log("")
        if rstep_ratio >= 1.5:
            verdict = (f"the RATIO steps {rstep_ratio:.2f}x across 512 -- the "
                       f"framework's decomposition changes there, as predicted")
        else:
            verdict = (f"the RATIO is flat across 512 ({rstep_ratio:.2f}x) -- the "
                       f"26x is a property of the whole N > 512 regime, NOT of the "
                       f"threshold, and the SPLIT account as stated is WRONG")
    else:
        verdict = "some points have no ratio; see the per-point table"
    log(f"  VERDICT: {verdict}")
    log("  PREDICTION (mixin/op.py SPLIT=256, threshold 512): the tg/ref ratio")
    log("  steps up across 512. A flat ratio refutes the threshold account.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--skip-controls", action="store_true",
                    help="skip the naive and knob arms; the ladder still runs in full")
    ap.add_argument("--rungs", default="", help="comma list; default all")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny shapes and 2 iterations: exercises the whole "
                         "driver-API path, the module load, the correctness "
                         "gates and the timing loop in about a minute, so a "
                         "mechanism bug is found before a full run is spent on it")
    ap.add_argument("--no-naive", action="store_true", help="skip the naive controls")
    ap.add_argument("--no-knob", action="store_true", help="skip the knob rungs")
    ap.add_argument("--floor-n", type=int, default=1 << 26, help="floats in the bandwidth floor")
    ap.add_argument("--sweep-n", default="",
                    help="comma list of N values; run the SCAN rung at each, "
                         "holding ROWS x N elements constant, and report the "
                         "ratio beside the framework's strategy threshold. "
                         "This is the falsifier for the note's central claim: "
                         "mixin/op.py switches cumsum's decomposition at "
                         "SPLIT*2 = 512, so the prediction is a DISCONTINUITY "
                         "in tg time per element at N=512 with a SMOOTH "
                         "reference, because the operator's difficulty does not "
                         "change at 512 and only the framework's strategy does")
    ap.add_argument("--sweep-rows", type=int, default=1 << 22,
                    help="elements held constant across the sweep (default 2^22)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    # The report is written after EVERY rung, not once at the end. The first run
    # of this harness died on a reporting line after the gate had passed and
    # produced no report at all; incremental writing means a late failure costs
    # the remaining rungs and not the completed ones.
    out_path = a.json or Path("/kaggle/working/axis_report.json")
    REPORT_PATH[0] = out_path

    global CUR, CTX, MOD
    env = environment()
    build = compile_reference(HERE / "ref.sass")

    from tinygrad import Device
    dev = Device[DEV]
    CUR = Cuda().c
    CTX = dev.context
    MOD = load_module(CUR, (HERE / "_ref_sm75.cubin").read_bytes(), CTX)
    log(f"  module loaded into TinyGrad's own context ({CTX})")

    report: dict = {"started": time.time(), "argv": sys.argv, "env": env, "build": build}

    # ---- the floor first, so every later percentage has a denominator
    try:
        report["floor"] = measure_floors(a.iters, a.warmup, a.floor_n)
        f = report["floor"]
        log("")
        log(f"  memory floor on THIS card: fill {f['fill_GBps']:.1f} GB/s (write-only), "
            f"copy {f['copy_GBps']:.1f} GB/s (read+write)")
    except Exception as e:
        report["floor"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"  floor FAILED: {e}")

    # ---- the device properties the evidence depends on
    try:
        c = CUR
        c.cuCtxSetCurrent(CTX)
        def attr(name_enum):
            v = ctypes.c_int()
            c.cuDeviceGetAttribute(ctypes.byref(v), name_enum, dev.cu_device)
            return v.value
        # The attribute ids are module-level constants in tinygrad's autogen
        # bindings, NOT members of the `CUdevice_attribute` type alias (which is
        # just c_uint32). Reading them off the type raised AttributeError on the
        # first run; the ids are the integers the driver API documents.
        props = {
            "SMs": attr(c.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            "l2_bytes": attr(c.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
            "max_smem_per_block": attr(c.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
            "clock_khz": attr(c.CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
            "compute_cap": (f"{attr(c.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)}."
                            f"{attr(c.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)}"),
        }
        report["device"] = props
        log(f"  SMs={props['SMs']}  L2={props['l2_bytes']/1e6:.1f} MB  "
            f"smem/block={props['max_smem_per_block']} B")
    except Exception as e:
        report["device"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"  device props unavailable: {e}")

    # ---- the rungs
    wanted = [k.strip() for k in a.rungs.split(",") if k.strip()] or [r.key for r in V.RUNGS]
    if a.smoke:
        # The same rungs, the same code path, the same gates -- only tiny. A
        # mechanism bug (a wrong cuLaunchKernel argument order, a module that
        # will not load, a buffer that cannot be addressed from a cubin) shows
        # up here in about a minute instead of after a queue.
        a.iters, a.warmup, a.floor_n = 2, 1, 1 << 16
        log("  SMOKE: tiny shapes, 2 iterations -- mechanism check, not a measurement")
    results = []
    for r in V.RUNGS:
        if r.key not in wanted:
            continue
        section(f"rung {r.key}: {r.title}")
        shapes = V.SMALL_SHAPES[r.key] if a.smoke else r.shapes
        log(f"  demands: {r.demands}")
        log(f"  shapes : {shapes}")
        try:
            res = measure_rung(r, shapes, a.iters, a.warmup,
                               not (a.no_naive or a.skip_controls),
                               not (a.no_knob or a.skip_controls))
        except Exception as e:
            res = {"key": r.key, "status": f"EXCEPTION {type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()[-1500:]}
        results.append(res)
        try:
            _print_rung(res)
        except Exception as e:
            # A display bug must never cost a rung. Two runs of this harness were
            # lost to a KeyError in this function while the measurements it was
            # about to print were already in the report.
            log(f"  (display failed: {type(e).__name__}: {e}; the report has the data)")
        report["rungs"] = list(results)
        _flush(out_path, report)
    report["rungs"] = results

    # ---- the width sweep: the scan rung across its strategy threshold
    #
    # WHY THIS IS IN THE HARNESS AND NOT ONLY IN THE NOTE. notes/06 C-19 is the
    # correction that a shape-gate claim read at one sweep point was wrong at the
    # shape that mattered, and the whole `SPLIT = 256` account of the 26x scan gap
    # rests on a threshold at N = 512 that has never been swept. A prediction
    # written down after seeing one shape is not a prediction; this makes it one
    # that a single run can refute.
    #
    # ROWS x N is held constant so the work per point is the same and the only
    # thing varying is the decomposition. The reference is swept identically: if
    # IT is smooth and the framework's is not, the discontinuity is the
    # framework's and the account stands. If both are smooth, the account is
    # wrong and that is the finding.
    if a.sweep_n:
        ns = [int(x) for x in a.sweep_n.split(",") if x.strip()]
        scan = V.BY_KEY["scan"]
        section(f"width sweep: scan across N (threshold SPLIT*2 = 512), "
                f"{a.sweep_rows} elements held constant")
        sweep = []
        for N in ns:
            R = max(1, a.sweep_rows // N)
            sh = {"R": R, "N": N}
            log(f"  N={N:<6} R={R:<7} ({R*N} elements)")
            try:
                # controls OFF: this is a width measurement, and a naive
                # spelling at large N is an N-1-step Python loop
                s = measure_rung(scan, sh, a.iters, a.warmup, False, False)
                tg = s.get("tg_device", {}).get("median_ms")
                rf = s.get("ref", {}).get("median_ms")
                row = {"N": N, "R": R, "status": s.get("status"),
                       "tg_ms": tg, "ref_ms": rf,
                       "ratio": s.get("ratio_tg_over_ref"),
                       "kernels": s.get("tg_kernels"),
                       "ns_per_elem_tg": (tg * 1e6 / N) if tg else None,
                       "ns_per_elem_ref": (rf * 1e6 / N) if rf else None}
                log(f"      tg {('%.4f' % tg) if tg else '-':>10} ms   "
                    f"ref {('%.4f' % rf) if rf else '-':>10} ms   "
                    f"ratio {(('%.2fx' % row['ratio']) if row['ratio'] else '-'):>7}"
                    f"   kernels {row['kernels']}")
            except Exception as e:
                row = {"N": N, "R": R, "status": f"EXCEPTION {type(e).__name__}: {e}"}
                log(f"      FAILED: {type(e).__name__}: {e}")
            sweep.append(row)
            report["sweep_scan_N"] = sweep
            _flush(out_path, report)
        report["sweep_scan_N"] = sweep
        report["sweep_threshold"] = 512
        section("the sweep, and whether the threshold shows")
        try:
            _print_sweep(sweep)
        except Exception as e:
            log(f"  (sweep table failed: {type(e).__name__}: {e}; the report has the data)")

    # ---- the summary table
    section("the axis")
    try:
        _print_axis(report)
    except Exception as e:
        log(f"  (table failed: {type(e).__name__}: {e})")

    _flush(out_path, report)
    log(f"\n  wrote {out_path}")

    bad = [r for r in results if not str(r.get("status", "")).startswith("OK")]
    log("")
    log(f"  {len(results)-len(bad)}/{len(results)} rungs OK; {len(bad)} not OK")
    return 0


def _flush(path: Path, report: dict) -> None:
    """Write the report so far. Never raises -- a write failure must not abort
    the run that is producing the numbers."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
    except Exception as e:
        log(f"  (report write failed: {e})")


def _print_rung(res: dict) -> None:
    st = res.get("status", "?")
    log(f"  status            {st}")
    for side in ("tg_device", "tg", "ref"):
        t = res.get(side)
        if not isinstance(t, dict) or "median_ms" not in t:
            continue
        note = "  <- the ratio's numerator" if side == "tg_device" else (
            "  (end-to-end, includes Python)" if side == "tg" else "")
        log(f"  {side:<18} {t['median_ms']:10.4f} ms   "
            f"(min {t.get('min_ms', float('nan')):.4f}, max {t.get('max_ms', float('nan')):.4f}){note}")
    if "tg_kernels" in res:
        log(f"  tg kernels        {res['tg_kernels']}")
    if "tg_relerr_vs_oracle" in res:
        log(f"  tg   vs oracle    {res['tg_relerr_vs_oracle']:.3e}  (tol {res.get('tolerance')})")
    if "ref_relerr_vs_oracle" in res:
        log(f"  ref  vs oracle    {res['ref_relerr_vs_oracle']:.3e}")
    if "ref_vs_tg_relerr" in res:
        log(f"  ref  vs tg        {res['ref_vs_tg_relerr']:.3e}")
    if res.get("ratio_tg_over_ref") is not None:
        log(f"  RATIO tg/ref      {res['ratio_tg_over_ref']:.3f}x")
    elif str(res.get("status", "")).startswith("OK"):
        log("  RATIO tg/ref      (not computed)")
    for k in ("tg_naive", "tg_knob"):
        if k in res:
            lab = res.get("tg_knob_label", "naive habit spelling")
            log(f"  {k:<18} {res[k]['median_ms']:10.4f} ms   ({lab})")
    if "ref_error" in res:
        log(f"  ref ERROR: {res['ref_error']}")


def _print_axis(report: dict) -> None:
    fl = report.get("floor", {})
    copy_gbps = fl.get("copy_GBps")
    log(f"  {'rung':<10} {'tg ms':>10} {'ref ms':>10} {'tg/ref':>8} {'tg GB/s':>9} {'%floor':>7}  status")
    log("  " + "-" * 74)
    for r in report.get("rungs", []):
        tg = r.get("tg", {}).get("median_ms")
        rf = r.get("ref", {}).get("median_ms")
        ratio = r.get("ratio_tg_over_ref")
        by = r.get("bytes")
        key = r.get("key", "?")
        gbps = pct = None
        if tg and by and r.get("status", "").startswith("OK"):
            gbps = by / (tg * 1e-3) / 1e9
            if copy_gbps and key in ("ew", "red", "scan", "gather"):
                pct = 100.0 * gbps / copy_gbps
        log(f"  {key:<10} "
            f"{tg if tg else float('nan'):10.4f} "
            f"{rf if rf else float('nan'):10.4f} "
            f"{(f'{ratio:.3f}x' if ratio else '-'):>8} "
            f"{(f'{gbps:.1f}' if gbps else '-'):>9} "
            f"{(f'{pct:.1f}%' if pct else '-'):>7}  {r.get('status','?')}")


if __name__ == "__main__":
    sys.exit(main())
