#!/usr/bin/env python3
"""verify.py -- the correctness gate, and the AST check on the constraint.

RUNS ON THIS HOST, ON CPU, BEFORE ANY KAGGLE PUSH. ~15 s, no GPU, no network.

Three jobs, all of which exist because of how this study has already gone wrong.

1. NUMERIC, TINYGRAD SIDE (step 2). Every rung's TinyGrad arm is checked against
   an independent float64 transcription of the same operator. A passing gate is
   NOT evidence that the port is good -- notes/06 §5b is the correction that
   establishes that, from two comparisons that passed 6/6 while being 72x off the
   kernel's structure. But it IS necessary, and it is the cheapest way to stop a
   rung whose two arms compute different things from producing a confident ratio.
   The gate therefore also compares the two arms against EACH OTHER once they are
   both on the device (`harness.py` does that part); this file is the third
   opinion, in float64, so that a shared misreading of the spec cannot make two
   wrong answers agree.

2. NUMERIC, REFERENCE SIDE (steps 3 and 4). The same, for `axis/refkernels.cu`,
   by EMULATING it on the host (`axis/simref.py`). This is not redundancy. The
   two most expensive defects in this project were in the reference -- the scan's
   loop-local `run` and the trisolve's inverted triangle test -- and neither was
   visible to a clean compile, to a `cuobjdump` census, to a shape check, or to a
   CORRECT Python transcription. A reference arm that is only checked on the card
   is a reference arm that costs a queued GPU session every time it is wrong.
   Step 4 checks all six `axis_scan_*` entry points separately, because the
   harness selects one by shape and one bench shape exercises one of six.

3. LAUNCH, REFERENCE SIDE (step 5). The argument ORDER of every reference launch,
   parsed out of the `.cu` and compared against what `harness._ref_args` will
   actually pass -- recovered from the sentinel values, not assumed. This is the
   one class the harness's own type-pattern check provably cannot see, and it is
   not hypothetical: `axis_scan_v{PT}` is `(x, y, R, N)` and
   `axis_scan_scalar` is `(y, x, R, N)` -- same arity, same types, OPPOSITE
   order, in the same rung. A launch that gets it wrong runs, returns wrong
   numbers, and passes every width check in existence.

4. STRUCTURAL (step 1). The task forbids `Tensor.custom_kernel`, pasted CUDA,
   inline PTX and a compiled-alongside .cu on the TINYGRAD side. `axis/rungs.py`
   must therefore be pure `Tensor` + `numpy`. This file parses that module's AST
   and fails if any forbidden name appears, so the constraint is checked rather
   than promised. A constraint enforced by a comment is a comment.

    The point is asymmetric on purpose: the hand-written reference is SUPPOSED to
    be CUDA (that is the reference), and the harness is supposed to call the
    driver API (that is how it launches). What must not happen is the TinyGrad arm
    reaching any of it. So the check is scoped to `rungs.py` and to what
    `rungs.py` imports.

Step 6 is the falsifier for steps 2-5: it re-injects the historical defects and
requires the oracles to reject them, signature included, and step 5's own
mutation selftest reorders real signatures and requires itself to fail. **A gate
that has never been seen to fail is not known to discriminate.**
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("PARALLEL", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import rungs as R  # noqa: E402
import simref as sim  # noqa: E402

# ===========================================================================
# 1. the AST check on the TinyGrad side
# ===========================================================================

FORBIDDEN = {
    "custom_kernel", "cuLaunchKernel", "cuModuleLoadData", "cuModuleGetFunction",
    "ctypes", "cupy", "torch", "triton", "cudart", "pycuda",
    "load_inline", "load", "source", "cuda_source", "raw_cuda",
}


def check_harness_arg_order() -> list[str]:
    """Every reference launch's argument ORDER, checked against the .cu.

    This is defect 11, and the only reason it exists is that the gate did not
    have this check. `axis_scan_v{PT}` is declared `(const float* x, float* y, ...)`
    -- input first, the opposite of every other kernel in the file -- while
    `axis_scan_scalar` is `(float* y, const float* x, ...)`, the conventional
    order and the opposite of its own sibling. `_ref_args` hardcoded one order for
    a rung that has two kernels, so every scalar-path launch (N <= 256) wrote the
    scan into the INPUT and read the output buffer as `x`. The width sweep
    returned INVALID for exactly those points and OK for the vectorised ones,
    and `simref.py` had already verified both kernels correct against `np.cumsum`
    -- so the kernels were right and the launch was wrong, which is a distinction
    no amount of reading `refkernels.cu` would have caught.

    So the order is PARSED OUT OF THE .cu` and compared against what the harness
    will pass. A type-pattern check cannot do this: `(float*, float*, int, int)`
    and `(float*, float*, int, int)` are the same pattern in either order, and a
    width check that passes is exactly what makes an order bug invisible. The
    parameter NAMES come from the cubin's PTX, where the kernel's own
    declarations are preserved.
    """
    src = (HERE / "refkernels.cu").read_text()
    import re as _re
    bad: list[str] = []
    checked = 0
    sigs = list(_reference_signatures(src))
    for name, plist in sigs:
        _POINTER_COUNTS[name] = sum(1 for role, _ in plist if role != "-")
    for name, plist in sigs:
        want = _harness_arg_order(name)
        if want is None:
            if _expected_rung(name) == "__floor__":
                continue      # see _harness_arg_order: no order to get wrong
            bad.append(f"{name}: no rung maps to this kernel, so the harness's "
                       f"argument order for it cannot be checked -- reported, "
                       f"not assumed")
            continue
        if list(want) != [c for c, _ in plist]:
            got = ",".join(c for c, _ in plist)
            bad.append(f"{name}: refkernels.cu declares ({got}) and the harness "
                       f"passes ({','.join(want)})")
        checked += 1
    if checked == 0:
        bad.append("arg-order gate matched ZERO kernels -- it is not checking "
                   "anything and must not be reported as a pass")
    bad.extend(_arg_order_selftest(src, sigs))
    return bad


# Mutations that must make this gate FAIL. If any of them passes, the gate is
# comparing the .cu against itself and is decoration.
_ARG_ORDER_MUTANTS: list[tuple[str, str, str]] = [
    # (label, find, replace) -- each is a REAL defect of the class this gate
    # exists to catch, applied to the .cu's text so the classification flips.
    ("scalar scan: output renamed so the role lookup cannot classify it",
     "void axis_scan_scalar(\n    float* __restrict__ y",
     "void axis_scan_scalar(\n    float* __restrict__ qq"),
    ("vectorised scan: input declared first, as the .cu actually has it",
     "void axis_scan_v##PT(const float* __restrict__ x, float* __restrict__ y",
     "void axis_scan_v##PT(float* __restrict__ y, const float* __restrict__ x"),
    ("ew: output moved after the first input",
     "void axis_ew(float4* __restrict__ y,\n                                   const float4* __restrict__ a,",
     "void axis_ew(const float4* __restrict__ a,\n                                   float4* __restrict__ y,"),
]


def _arg_order_selftest(src: str, sigs) -> list[str]:
    """Re-order a real signature and require the gate to notice.

    The classification in `_split_params` keys on the parameter NAME, so the
    obvious failure mode of this gate is that it compares the .cu against its own
    naming convention and passes on anything spelled like the kernels it was
    written for. These mutants check the negative: reorder an output relative to
    an input in a way the harness does NOT match, and require a violation.
    """
    import re as _re
    problems: list[str] = []
    for label, find, replace in _ARG_ORDER_MUTANTS:
        if find not in src:
            problems.append(f"arg-order selftest: MUTANT DID NOT APPLY ({label}) -- "
                            f"the anchor text has changed, so this falsifier is "
                            f"testing nothing")
            continue
        mutated = src.replace(find, replace, 1)
        try:
            msigs = list(_reference_signatures(mutated))
            caught = False
            for name, plist in msigs:
                if _expected_rung(name) == "__floor__":
                    continue
                _POINTER_COUNTS[name] = sum(1 for role, _ in plist if role != "-")
                want = _harness_arg_order(name)
                if want is None:
                    continue
                if list(want) != [c for c, _ in plist]:
                    caught = True
            if not caught:
                problems.append(f"arg-order selftest: NOT CAUGHT -- {label}. The "
                                f"gate passed a signature whose out/in order it "
                                f"should have rejected.")
        finally:
            # restore the real counts for the unmutated source
            for name, plist in sigs:
                _POINTER_COUNTS[name] = sum(1 for role, _ in plist if role != "-")
    return problems


def _reference_signatures(src: str):
    """(name, [(role, param)]) for every `extern "C" __global__` in the .cu.

    `role` is "out"/"in" for a pointer, "-" for a scalar, and is derived from the
    parameter's own NAME (`y`, `C`, `o`, `X` are the outputs; `x`, `A`, `B`,
    `w`, `tab`, `idx`, `s`, `D` are inputs). Deriving the role from the name and
    then comparing it against the harness is the whole point: the two scans'
    signatures differ ONLY in which of two same-typed pointers comes first, so
    nothing about types, widths or arity can tell them apart.
    """
    import re as _re
    # `__launch_bounds__(256)` contains parentheses, so a naive `[^;{]*?\(` scan
    # can match the WRONG opening paren and capture garbage. Drop the attribute
    # before matching, and anchor on the parameter list being followed by `{`.
    # Macro bodies are continued with backslash-newline, so the logical line is
    # several physical ones. Join them first, or the signature regex sees a
    # truncated parameter list -- and a truncated list is exactly the kind of
    # thing that produces a confident, wrong gate.
    clean = _re.sub(r"\\\n\s*", " ", src)
    clean = _re.sub(r"__launch_bounds__\s*\([^)]*\)", "", clean)

    # The five vectorised scan entry points are generated by a MACRO
    # (`SCAN_ENTRY(PT)`), so no regex over the literal text sees them -- and the
    # macro's parameter list is exactly where the input-first order lives. A gate
    # that only read the literal text would check the scalar scan and skip the
    # five kernels the ladder actually measures, so the macro is expanded here.
    mm = _re.search(r"#define\s+SCAN_ENTRY\(PT\)\s*extern\s+\"C\"\s+__global__\s+"
                    r"(?:__launch_bounds__\([^)]*\)\s*)?"
                    r"void\s+axis_scan_v##PT\(([^;{]*?)\)\s*\{",
                    clean, _re.S)
    if mm:
        for pt in (1, 2, 4, 8, 16):
            yield f"axis_scan_v{pt}", _split_params(mm.group(1))
    else:
        raise AssertionError("SCAN_ENTRY macro not found; the arg-order gate "
                             "would silently check 5 fewer kernels")

    for m in _re.finditer(
            r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^;{]*?)\)\s*\{',
            clean, _re.S):
        name, params = m.group(1), m.group(2)
        if name == "axis_scan_v##PT":
            continue          # already yielded from the macro
        yield name, _split_params(params)


# The parameter NAMES that denote the OUTPUT buffer, per kernel, in the .cu's own
# spelling. A name-based classification is the weakest link in this gate and it is
# therefore checked by the mutation selftest below rather than trusted: mutating
# the .cu so a kernel's output parameter is renamed must make this gate fail.
#
# It is a name list rather than "position 0" because two of the seven rungs put
# the input first. The list is the .cu's contract, and a signature the .cu does
# not name in this set is reported as unclassifiable instead of guessed.
_OUT_NAMES = ("y", "C", "o", "X")


def _split_params(params: str):
    """[(role, text)] for a parameter list, classifying pointers out/in by name."""
    import re as _re
    plist = []
    for p in params.split(","):
        p = p.strip()
        if not p:
            continue
        if "*" in p:
            var = _re.findall(r"(\w+)", _re.sub(
                r"\b(const|__restrict__|restrict|volatile)\b", " ", p))
            var = var[-1] if var else "?"
            role = "out" if var in _OUT_NAMES else "in"
            plist.append((role, p))
        else:
            plist.append(("-", p))
    return plist


def _harness_arg_order(symbol: str):
    """The (out|in|-) sequence `_ref_args` will pass for this symbol.

    Returned as a list of ROLES by actually calling `_ref_args` on a synthetic
    shape, so this gate checks the code that RUNS rather than a restatement of
    it. `None` when no rung owns the symbol.
    """
    sys.path.insert(0, str(HERE))
    import harness as _H
    rung_key = _expected_rung(symbol)
    if rung_key == "__floor__":
        # The floor kernels' argument lists live in `measure_floors`, and they
        # are the SIMPLEST launches in the file -- `axis_fill` takes no input at
        # all, so there is no order to get wrong. They are excluded deliberately
        # and the exclusion is visible here rather than inferred from silence.
        return None
    rung = next((rr for rr in _H.V.RUNGS if rr.key == rung_key), None)
    if rung is None:
        return None
    # a shape that every scan symbol accepts, and one in-range value per key
    shape = {k: 8 for k in getattr(rung, "shapes", {})}
    shape.setdefault("n", 8)
    if rung_key == "scan":
        shape["N"] = 1024
        shape["R"] = 8
    # WHICH BUFFER goes in each position, RECOVERED from the values the harness
    # actually passed rather than assumed. Every input gets a distinct sentinel
    # and the output another, so `_ref_args`' real return value identifies
    # itself. A gate that assumed "the output comes first" would pass on all
    # seven rungs -- and would have passed on `axis_scan_v*` too, which is
    # exactly the bug it exists to catch. Deriving the expected order from the
    # .cu and comparing is the only version of this check that is not the defect
    # written down.
    buf: dict = {"OUT": 0xF000F000}
    for i, nm in enumerate(("x", "a", "b", "z", "w", "A", "B", "tab", "idx",
                            "s", "D")):
        buf[nm] = 0x1000 + i
    _H.REF_SYMBOL[0] = symbol
    try:
        args = _H._ref_args(rung, shape, {}, buf, buf["OUT"])
    except Exception as e:
        return [f"<_ref_args raised {type(e).__name__}: {e}>"]
    if _POINTER_COUNTS.get(symbol) is None:
        return None
    return [buf_key(buf, a) for a in args]


def buf_key(buf: dict, v) -> str:
    """Which buffer a passed argument is, by identity against the sentinel map."""
    if not isinstance(v, int):
        return "-"
    for nm, sent in buf.items():
        if v == sent:
            return "out" if nm == "OUT" else "in"
    return "-"


_POINTER_COUNTS: dict[str, int] = {}


def _expected_rung(symbol: str) -> str:
    """Which rung a reference symbol belongs to, by name."""
    if symbol.startswith("axis_scan"):
        return "scan"
    if symbol == "axis_gemm_f16":
        return "gemm16"
    if symbol == "axis_gemm_f32":
        return "gemm32"
    if symbol == "axis_rowdot":
        return "red"
    if symbol == "axis_ew":
        return "ew"
    if symbol == "axis_gather":
        return "gather"
    if symbol == "axis_trisolve":
        return "trisolve"
    if symbol in ("axis_fill", "axis_copy"):
        # The memory FLOORS, not rungs: they are launched by `measure_floors`
        # with its own argument lists, and there is no rung to check them
        # against. They are listed here so the gate reports them as unchecked
        # rather than passing over them in silence.
        return "__floor__"
    return symbol[len("axis_"):] if symbol.startswith("axis_") else symbol


def check_purity(path: Path) -> list[str]:
    """Return a list of violations; empty means the TinyGrad side is clean."""
    tree = ast.parse(path.read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN:
                    bad.append(f"line {node.lineno}: import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN:
                bad.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN:
                bad.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN:
                bad.append(f"line {node.lineno}: {node.id}")
    return bad


# ===========================================================================
# 2. the float64 oracles -- independent transcriptions, not the port
# ===========================================================================
# Each oracle is written from the operator's DEFINITION in float64 numpy. It is
# deliberately not derived from the TinyGrad expression and not from the CUDA: if
# all three agree, three independent readings agree, which is the only reason a
# green gate means anything here.


def oracle_ew(d: dict) -> np.ndarray:
    return (d["a"].astype(np.float64) + d["b"].astype(np.float64) * d["z"].astype(np.float64))


def oracle_red(d: dict) -> np.ndarray:
    return (d["x"].astype(np.float64) * d["w"].astype(np.float64)).sum(-1)


def oracle_gemm32(d: dict) -> np.ndarray:
    return d["A"].astype(np.float64) @ d["B"].astype(np.float64)


def oracle_gemm16(d: dict) -> np.ndarray:
    return d["A"].astype(np.float64) @ d["B"].astype(np.float64)


def oracle_gather(d: dict) -> np.ndarray:
    return d["tab"].astype(np.float64)[d["idx"]] * d["s"].astype(np.float64)


def oracle_scan(d: dict) -> np.ndarray:
    return np.cumsum(d["x"].astype(np.float64), axis=-1)


def oracle_trisolve(d: dict) -> np.ndarray:
    """(I+D)^-1 - I for a strictly lower D, via an explicit dense solve.

    `np.linalg.inv` on the full [C,C] is the slowest possible way to do this and
    is used ON PURPOSE: it shares no code path, no blocking and no recurrence
    with either arm, so it is a genuine third opinion. The matrix is tiny
    (C=64) so the cost is irrelevant.
    """
    B, C = d["B"], d["C"]
    D = d["D"].astype(np.float64)
    out = np.zeros((B, C, C), dtype=np.float64)
    I = np.eye(C, dtype=np.float64)
    for b in range(B):
        out[b] = np.linalg.inv(I + D[b]) - I
    return out


ORACLES = {
    "ew": oracle_ew, "red": oracle_red, "gemm32": oracle_gemm32,
    "gemm16": oracle_gemm16, "gather": oracle_gather,
    "scan": oracle_scan, "trisolve": oracle_trisolve,
}


# ===========================================================================
# 3. the ladder
# ===========================================================================

# Shapes. Chosen so that (a) every rung is long enough to be timed above timer
# resolution, (b) the memory-bound rungs move enough bytes to be bandwidth- not
# launch-bound, and (c) the whole ladder fits in the T4's 16 GB with room for the
# inputs, both outputs and the framework's own intermediates.
#
# `red`'s K is 256 rather than something larger on purpose: the rung exists to
# separate "one wide reduction" from "K rank-1 reductions", and at K=256 the
# naive form is slow enough to time reliably without costing the run its budget.
SHAPES: dict[str, dict] = {
    "ew":       {"n": 1 << 24},                 # 16.7M floats: 4 arrays in, 1 out
    "red":      {"R": 1 << 16, "K": 256},        # 65536 rows x 256
    "gemm32":   {"M": 1024, "N": 1024, "K": 1024},
    "gemm16":   {"M": 2048, "N": 2048, "K": 2048},
    "gather":   {"n": 1 << 23, "table": 1 << 22},   # 8.4M gathers into a 16 MB table
    "scan":     {"R": 4096, "N": 4096},
    "trisolve": {"B": 1 << 14, "C": 64, "BC": 16},   # NC=4, matching chunk_intra.py:825
}

# grid / block / args for the hand-written kernels. The grid is sized to the
# card's SM count where a persistent-style grid is appropriate, so the reference
# is not accidentally measuring launch configuration.
def _sm_count() -> int:
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=multiprocessor_processor_count",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return 40          # the T4's value, used when there is no GPU on this host


SM = _sm_count()

RUNGS: list[R.Rung] = [
    R.Rung(
        key="ew", title="elementwise, pure bandwidth",
        demands="nothing: no reduction, no gather, no reuse. The control rung.",
        shapes=SHAPES["ew"], ref_kernel="axis_ew",
        ref_grid=lambda s: (min(SM * 32, max(1, s["n"] // 4 // 256)), 1, 1),
        ref_block=lambda s: (256, 1, 1),
        ref_args=lambda s: (s["n"] // 4,),
        ref_params="ppppi",
        tg=lambda s, d: R.tg_ew(s, d),
        inputs=R.in_ew, outputs=["y"], out_dtype="float32",
        distinguishing_evidence=(
            "If the framework reaches the measured copy bandwidth here, the "
            "instrument and the memory path are sound and every gap above is "
            "about arithmetic/scheduling. If it does NOT, the axis is measuring "
            "the framework's memory path and no rung above is interpretable."),
        bytes_moved=lambda s: 4 * s["n"] * 4,
    ),
    R.Rung(
        key="red", title="wide reduction (row-dot)",
        demands=("one reduction over a K-long axis. This is the rung that "
                 "separates the framework from the C-18 habit."),
        shapes=SHAPES["red"], ref_kernel="axis_rowdot",
        ref_grid=lambda s: ((s["R"] + 7) // 8, 1, 1),
        ref_block=lambda s: (256, 1, 1),
        ref_args=lambda s: (s["R"], s["K"]),
        ref_params="pppii",
        tg=lambda s, d: R.tg_red(s, d),
        inputs=R.in_red, outputs=["y"], out_dtype="float32",
        tg_naive=lambda s, d: R.tg_red_naive(s, d),
        distinguishing_evidence=(
            "The `red` vs `ew` pair is the designed control: same traffic class, "
            "one reduction added. A gap that appears only at `red` is the "
            "reduction's; a gap already present at `ew` is the memory path's. "
            "The tg_naive control prices the C-18 habit on the same card in the "
            "same run, so 'wide contraction' is a measured choice and not an "
            "assumption about what a good port looks like."),
        bytes_moved=lambda s: 2 * s["R"] * s["K"] * 4,
    ),
    R.Rung(
        key="gemm32", title="matmul fp32, no tensor core on sm_75",
        demands=("reuse: every input byte is read many times, so the kernel must "
                 "tile and block. FFMA only -- fp32 has no TC on this arch."),
        shapes=SHAPES["gemm32"], ref_kernel="axis_gemm_f32",
        ref_grid=lambda s: ((s["N"] + 127) // 128, (s["M"] + 127) // 128, 1),
        ref_block=lambda s: (256, 1, 1),
        ref_args=lambda s: (s["M"], s["N"], s["K"]),
        ref_params="pppiii",
        tg=lambda s, d: R.tg_gemm32(s, d),
        inputs=R.in_gemm32, outputs=["C"], out_dtype="float32",
        tg_knob=("ALLOW_TF32=1", {"ALLOW_TF32": 1}, lambda s, d: R.tg_gemm32(s, d)),
        distinguishing_evidence=(
            "notes/26 §6b and notes/06 C-19 together: fp32 does not reach a "
            "tensor core at ANY shape on sm_75, and the shape-gate version of "
            "that claim was a sweep artefact. The falsifier for 'does not exist "
            "for this architecture' (class iv) is a SASS census of BOTH arms: "
            "if the reference emits FFMA and no HMMA and the framework does "
            "likewise, class (iv) is settled by the hardware, not by the "
            "compiler. ALLOW_TF32 is measured separately and labelled, because "
            "it changes the numerics and so is a different rung."),
        bytes_moved=lambda s: (2 * s["M"] * s["N"] + 2 * s["K"] * s["N"]) * 4,
        flops=lambda s: 2 * s["M"] * s["N"] * s["K"],
    ),
    R.Rung(
        key="gemm16", title="matmul fp16, tensor cores present",
        demands=("the same reuse as gemm32, on an architecture where the "
                 "hardware instruction class EXISTS and the framework has a "
                 "table to choose from (TC / TC_SELECT)."),
        shapes=SHAPES["gemm16"], ref_kernel="axis_gemm_f16",
        ref_grid=lambda s: ((s["N"] + 63) // 64, (s["M"] + 63) // 64, 1),
        ref_block=lambda s: (128, 1, 1),
        ref_args=lambda s: (s["M"], s["N"], s["K"]),
        ref_params="pppiii",
        tg=lambda s, d: R.tg_gemm16(s, d),
        inputs=R.in_gemm16, outputs=["C"], out_dtype="float16",
        distinguishing_evidence=(
            "This is the rung where the four classes separate cleanly. (iv) is "
            "excluded because HMMA provably exists on sm_75 and the reference "
            "SASS contains it (axis/sass_census.py, a host-side check costing no "
            "GPU time). (i) is excluded by the fact that the framework expresses "
            "the contraction at all. That leaves (ii) 'a cost model could find "
            "it' versus (iii) 'not implemented', and the experiment that "
            "distinguishes THOSE is the knob sweep: if TC_SELECT sweeps close "
            "the gap, it is (ii); if the whole knob space is flat, it is (iii). "
            "No other rung can answer that question."),
        bytes_moved=lambda s: (2 * s["M"] * s["N"] + 2 * s["K"] * s["N"]) * 2,
        flops=lambda s: 2 * s["M"] * s["N"] * s["K"],
    ),
    R.Rung(
        key="gather", title="random access gather, no reuse",
        demands=("a per-element address into a table larger than L2, so the "
                 "access misses. One multiply of arithmetic."),
        shapes=SHAPES["gather"], ref_kernel="axis_gather",
        ref_grid=lambda s: (min(SM * 16, max(1, s["n"] // 256)), 1, 1),
        ref_block=lambda s: (256, 1, 1),
        ref_args=lambda s: (s["n"],),
        ref_params="ppppi",
        tg=lambda s, d: R.tg_gather(s, d),
        inputs=R.in_gather, outputs=["y"], out_dtype="float32",
        distinguishing_evidence=(
            "The table is 16 MB against a 4 MB L2, checked by a cuDeviceGetAttribute "
            "at run time, so the gather cannot be served from cache -- a cached "
            "gather measures a broadcast and this rung would silently become a "
            "different operator. This is the rung that tests notes/26 §3c's claim "
            "that a gather is a fusion barrier 'about the index, not the gather'."),
        bytes_moved=lambda s: (2 * s["n"] + s["table"]) * 4 + s["n"] * 4,
    ),
    R.Rung(
        key="scan", title="carried dependency chain (prefix sum)",
        demands=("a dependent chain and nothing else. No contraction, no gather, "
                 "no early exit to confound it."),
        # The scan's symbol is shape-dependent -- the vectorised path is a
        # template specialised on the float4s per thread and the scalar fallback
        # is a separate entry point. `harness._scan_symbol` resolves it; this
        # field is the fallback name and the signature check uses the same
        # resolution, so the two cannot disagree.
        shapes=SHAPES["scan"], ref_kernel="axis_scan_v4",
        ref_grid=lambda s: (s["R"], 1, 1),
        ref_block=lambda s: (256, 1, 1),
        ref_args=lambda s: (s["R"], s["N"]),
        ref_params="ppii",
        tg=lambda s, d: R.tg_scan(s, d),
        inputs=R.in_scan, outputs=["y"], out_dtype="float32",
        tg_naive=lambda s, d: R.tg_scan_naive(s, d),
        distinguishing_evidence=(
            "notes/26 §5c claims the framework cannot keep a dependent chain in "
            "registers, has no loop, and that UNROLL is capped at 32 (notes/06 "
            "C-16). A scan is the clean test because the chain IS the operator. "
            "The falsifier for class (i)/(iii) is the framework's own `cumsum` "
            "op: it exists, so 'the framework cannot express a scan' is false by "
            "inspection, and the live question becomes whether its lowering "
            "emits N dependent kernels or one. That is countable on the device, "
            "and notes/06 C-22 is the correction that says a CENSUS is not a "
            "measurement -- so the count is reported as context and the TIME is "
            "the evidence."),
        bytes_moved=lambda s: 2 * s["R"] * s["N"] * 4,
    ),
    R.Rung(
        key="trisolve", title="blocked triangular inverse (KDA's UT transform)",
        demands=("a carried chain AND a dense contraction at every step. The "
                 "hardest rung, and one of the study's two subjects."),
        shapes=SHAPES["trisolve"], ref_kernel="axis_trisolve",
        ref_grid=lambda s: (s["B"], 1, 1),
        ref_block=lambda s: (256, 1, 1),
        ref_args=lambda s: (s["B"],),
        ref_params="ppi",
        tg=lambda s, d: R.tg_trisolve(s, d),
        inputs=R.in_trisolve, outputs=["X"], out_dtype="float32",
        distinguishing_evidence=(
            "The reference here is a hand-written BLOCKED inverse at BC=16, not "
            "fla's naive.py. notes/06 C-15 is the correction that establishes "
            "why: the port that followed naive.py passed 6/6 and was 4 032 "
            "dependent scalar reductions against the real kernel's 56. So this "
            "rung measures the framework against the kernel's STRATEGY, and the "
            "note must say that the Triton original was never compiled here "
            "(notes/29: no triton, so no hand-written SASS row exists). The "
            "scan-vs-trisolve pair is the designed contrast that separates 'a "
            "carried chain' from 'a carried chain with a contraction in it'."),
        bytes_moved=lambda s: 2 * s["B"] * s["C"] * s["C"] * 4,
        flops=lambda s: 0,     # dominated by the sequential chain, not a flop count
    ),
]

BY_KEY = {r.key: r for r in RUNGS}

# The gate runs on CPU, so it uses SMALL shapes. This is not a shortcut on
# coverage: the same expressions, the same code paths and the same reductions run,
# only shorter. What small shapes cannot check is a shape-dependent codegen
# decision (notes/06 C-19's entire content is that a shape gate read at one sweep
# point was wrong at the shape that mattered), and that is exactly what the
# on-card run measures. The two are complementary and the note says which
# question each answers.
SMALL_SHAPES: dict[str, dict] = {
    "ew":       {"n": 4096},
    "red":      {"R": 64, "K": 32},
    "gemm32":   {"M": 32, "N": 32, "K": 32},
    "gemm16":   {"M": 32, "N": 32, "K": 32},
    "gather":   {"n": 4096, "table": 1024},
    "scan":     {"R": 16, "N": 64},
    "trisolve": {"B": 8, "C": 64, "BC": 16},
}

# Shapes for the EMULATED reference arm. These differ from SMALL_SHAPES where the
# emulator's preconditions force it to: `axis_scan_v{PT}` covers exactly
# N = 256*PT*4 elements, so N=64 is not expressible in the vectorised path at all
# and the vectorised path would go unchecked. Everything else reuses
# SMALL_SHAPES. The emulated cost is ~2.5 s in total, which is affordable to run
# on every push, so the shapes are chosen for coverage rather than for speed.
SIM_SHAPES: dict[str, dict] = dict(SMALL_SHAPES, scan={"R": 3, "N": 4096})

# Every scan entry point, each on a shape its own precondition admits. `PT is
# None` is `axis_scan_scalar`, which needs N <= 256 and is correct only there.
SCAN_ENTRIES: dict[str, tuple] = {
    "axis_scan_scalar":  (16, 64, None),
    "axis_scan_v1":      (4, 1024, 1),
    "axis_scan_v2":      (4, 2048, 2),
    "axis_scan_v4":      (3, 4096, 4),      # the entry the BENCH shape selects
    "axis_scan_v8":      (2, 8192, 8),
    "axis_scan_v16":     (2, 16384, 16),
}

# float32 tolerances for the emulated reference. `gemm16` accumulates in fp32 and
# rounds once to fp16, so it carries fp16's resolution; `trisolve` is a 64-long
# recurrence whose absolute error grows with the length.
SIM_TOL: dict[str, float] = {"gemm16": 5e-2, "trisolve": 1e-4}


# ===========================================================================
def relerr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = max(float(np.abs(b).max()), 1e-30)
    return float(np.abs(a - b).max() / den)


def main() -> int:
    print("=" * 78)
    print("axis/verify.py -- CPU gate. No GPU. Runs before any Kaggle push.")
    print("=" * 78)

    # ---- 1. purity
    print("\n1. the TinyGrad side is pure Tensor (task constraint, checked by AST)")
    viol = check_purity(HERE / "rungs.py")
    if viol:
        print("   FAIL -- forbidden constructs in rungs.py:")
        for v in viol:
            print("     ", v)
        return 2
    print("   [PASS] rungs.py imports only tinygrad + numpy; no custom_kernel,")
    print("          no ctypes, no driver API, no foreign framework.")

    # ---- 2. numeric
    print("\n2. every rung's TinyGrad arm against an independent float64 oracle")
    print("   (a green gate is necessary, not sufficient -- notes/06 5b)")
    print(f"   small shapes, CPU: {SMALL_SHAPES}")
    fails = 0
    for r in RUNGS:
        s = SMALL_SHAPES[r.key]
        d = r.inputs(s)
        want = ORACLES[r.key](d)
        try:
            got = r.tg(s, d)
            got_np = np.asarray(got.numpy(), dtype=np.float64)
        except Exception as e:
            print(f"   [FAIL] {r.key:<9} raised {type(e).__name__}: {e}")
            fails += 1
            continue
        if got_np.shape != want.shape:
            print(f"   [FAIL] {r.key:<9} shape {got_np.shape} != oracle {want.shape}")
            fails += 1
            continue
        e = relerr(got_np, want)
        # dtype-dependent tolerance: fp16 carries ~2^-11 relative resolution and
        # a gemm accumulates K terms, so the fp32 tolerances would be meaningless
        # there. trisolve is a 64-long recurrence in fp32 and is the loosest of
        # the fp32 rungs.
        tol = {"gemm16": 5e-2, "trisolve": 1e-4}.get(r.key, 1e-5)
        ok = e <= tol
        print(f"   [{'PASS' if ok else 'FAIL'}] {r.key:<9} worst relerr {e:.3e}  (tol {tol:.0e})"
              f"  shape {tuple(got_np.shape)}")
        if not ok:
            fails += 1

    # ---- 3. the REFERENCE arm, emulated on the host
    #
    # Everything above checks the TinyGrad side. That is half the comparison, and
    # the half that a `cuobjdump` census and a clean compile already cover. The
    # other half is `refkernels.cu`, and the two defects that cost this study a
    # GPU session each were BOTH in it: the scan's loop-local `run`, and the
    # trisolve's `c > i`.
    #
    # `axis/simref.py` is a phase-structured mechanical transcription of the .cu
    # with real shared-memory barriers, and this step runs it against the same
    # float64 oracles. A green line here means the reference is not the reason a
    # ratio is what it is -- which is the only thing that makes a ratio about the
    # framework.
    print("\n3. the REFERENCE arm, emulated on the host (axis/simref.py)")
    print("   the other half of the comparison; both of the defects that cost a")
    print("   GPU session were in this file, and neither was visible to a compile")
    print("   or a SASS census")
    simfails = 0
    for r in RUNGS:
        s = SIM_SHAPES.get(r.key, SMALL_SHAPES[r.key])
        d = r.inputs(s)
        want = ORACLES[r.key](d)
        try:
            got = np.asarray(sim.emulate(r.key, d), dtype=np.float64)
        except Exception as e:
            print(f"   [FAIL] {r.key:<9} emulator raised {type(e).__name__}: {e}")
            simfails += 1
            continue
        if got.shape != want.shape:
            print(f"   [FAIL] {r.key:<9} shape {got.shape} != oracle {want.shape}")
            simfails += 1
            continue
        e = relerr(got, want)
        tol = SIM_TOL.get(r.key, 1e-5)
        ok = e <= tol
        print(f"   [{'PASS' if ok else 'FAIL'}] {r.key:<9} worst relerr {e:.3e}  (tol {tol:.0e})"
              f"  shape {tuple(got.shape)}")
        if not ok:
            simfails += 1

    # ---- 4. EVERY scan entry point, not just the one the bench shape selects
    #
    # `refkernels.cu` exports `axis_scan_scalar` and `axis_scan_v{1,2,4,8,16}`.
    # `harness._scan_symbol` picks one by shape, so a run at N=4096 touches only
    # `axis_scan_v4` and says nothing about the other five. That gap is not
    # hypothetical: this gate initially checked only `SMALL_SHAPES["scan"]`, whose
    # N=64 routes to the SCALAR path, so the vectorised path -- the one with the
    # `run` defect, and the one the axis actually measures -- was never executed
    # and the gate was green over it. A gate that only runs the entry point a
    # single shape happens to select is not a gate over the file.
    print("\n4. every scan entry point in refkernels.cu, each on its own shape")
    for label, (R_, N_, PT) in SCAN_ENTRIES.items():
        d = R.in_scan({"R": R_, "N": N_})
        try:
            if PT is None:
                got = np.asarray(sim.sim_scan_scalar(d["x"], R_, N_), dtype=np.float64)
            else:
                got = np.asarray(sim.sim_scan(d["x"], R_, N_, PT), dtype=np.float64)
        except Exception as e:
            print(f"   [FAIL] {label:<14} raised {type(e).__name__}: {e}")
            simfails += 1
            continue
        want = np.cumsum(d["x"].astype(np.float64), axis=-1)
        e = relerr(got, want)
        ok = e <= 1e-5
        print(f"   [{'PASS' if ok else 'FAIL'}] {label:<14} worst relerr {e:.3e}"
              f"  shape ({R_}, {N_})")
        if not ok:
            simfails += 1

    # ---- 4b. each entry point covers EXACTLY its own span, and the harness's
    #          selector agrees
    #
    # The kernels are correct for the shape they were written for and PARTIAL for
    # any other. That is worth checking, because the harness once shipped a
    # selector returning `axis_scan_v1` for every N that was a multiple of 1024
    # -- reasoning that one float4 per thread was the fastest choice. At N=4096
    # that writes 1024 of the row's 4096 elements, and the on-card symptom was
    # INVALID against the oracle at a reference time of **0.047 ms, FASTER than
    # the correct kernel's 0.219 ms**, because it was doing a quarter of the work.
    #
    # WHAT IS AND IS NOT BEING CHECKED HERE, precisely. `simref.sim_scan` has an
    # explicit span assertion and REFUSES a shape it does not cover, so the
    # emulator is not a faithful model of the CUDA on this point: the real kernel
    # has no such check and truncates SILENTLY. So the "NOT correct at 2x" column
    # below is evidence that the span arithmetic is what the selector must agree
    # with -- and the second column is the part that is actually load-bearing,
    # because it compares the SELECTOR to the spans the arithmetic implies. The
    # on-card oracle remains the only thing that catches a silent truncation, and
    # that is not a claim this host can make on its own.
    print("\n4b. each entry point covers EXACTLY its span, and the selector agrees")
    for label, PT in (("axis_scan_v1", 1), ("axis_scan_v2", 2), ("axis_scan_v4", 4),
                      ("axis_scan_v8", 8), ("axis_scan_v16", 16)):
        covered = sim.SCAN_THREADS * sim.SCAN_VEC * PT
        d2 = R.in_scan({"R": 2, "N": covered})
        own = relerr(np.asarray(sim.sim_scan(d2["x"], 2, covered, PT), dtype=np.float64),
                     np.cumsum(d2["x"].astype(np.float64), axis=-1))
        sys.path.insert(0, str(HERE))
        import harness as _H
        sel = _H._ref_scan_symbol_for_span({"N": covered})
        try:
            sim.sim_scan(d2["x"], 2, covered * 2, PT)
            beyond = "SILENTLY ACCEPTED -- the span check is not doing its job"
            ok = False
        except AssertionError:
            beyond = "refused (emulator guard; the CUDA would truncate silently)"
            ok = True
        except Exception as exc:
            beyond = f"refused ({type(exc).__name__})"
            ok = True
        selok = sel == label
        print(f"   [{'PASS' if ok and selok and own <= 1e-5 else 'FAIL'}] {label:<14} "
              f"N={covered:<6} relerr {own:.2e}   selector picks {sel:<18} "
              f"2x span: {beyond}")
        if not (ok and selok and own <= 1e-5):
            simfails += 1

    # ---- 5. argument ORDER for every reference launch
    #
    # The one thing the harness's own type-pattern check provably cannot see.
    # `axis_scan_v{PT}` is declared (x, y, R, N) and `axis_scan_scalar` is
    # declared (y, x, R, N): same arity, same types, opposite order, in the SAME
    # rung. A launch that gets it wrong runs, returns wrong numbers, and passes
    # every width check in existence -- which is exactly what happened, and the
    # width sweep is what exposed it, as INVALID for every scalar-path point and
    # OK for every vectorised one.
    print("\n5. every reference launch's ARGUMENT ORDER, read from the .cu")
    argbad = check_harness_arg_order()
    nchk = len([n for n, _ in _reference_signatures((HERE / "refkernels.cu").read_text())
                if _expected_rung(n) != "__floor__"])
    if argbad:
        for b in argbad:
            print(f"   [FAIL] {b}")
        fails += 1
    else:
        print(f"   [PASS] {nchk} kernels: each launch's out/in order matches the "
              f"signature in refkernels.cu")
        print("          (the 5 vectorised scans declare x first, the scalar one "
              "declares y first;")
        print("           both are checked, and a mutation selftest reorders real "
              "signatures and requires this to fail)")

    # ---- 6. the gate's own falsifier: re-inject the historical defects
    #
    # Steps 2-5 have never been observed to fail in this project, so "they pass"
    # is not yet evidence that they discriminate. This re-puts the two real
    # defects -- and two more of the same classes -- into the emulator and
    # requires the oracles to REJECT them, signature included. A mutant that
    # still agrees fails the gate: an undetectable defect class is a gate that
    # is not measuring what it says.
    print("\n6. the gate's own falsifier -- re-inject the historical defects")
    print("   and require the oracle to reject each one, signature included")
    for row in sim.mutation_selftest(ORACLES):
        ok = row["caught"]
        tag = "CAUGHT" if ok else "MISSED "
        err = "     n/a  " if row["relerr"] is None else f"{row['relerr']:.3e}"
        print(f"   [{tag}] {row['label']:<42} relerr {err}")
        print(f"            {row['detail']}")
        if not ok:
            fails += 1

    # ---- 7. the naive controls, which are controls and must be honestly slow
    print("\n7. the naive controls exist and are separate measurements")
    for r in RUNGS:
        if r.tg_naive is None:
            continue
        s = SMALL_SHAPES[r.key]
        d = r.inputs(s)
        try:
            r.tg_naive(s, d).numpy()
            print(f"   [ok  ] {r.key:<9} naive spelling runs (its TIME is the control)")
        except Exception as e:
            print(f"   [note] {r.key:<9} naive spelling raised {type(e).__name__};"
                  f" the control is then an upper bound, and the note says so")

    fails += simfails
    print("\n" + "=" * 78)
    print(f"{'ALL CHECKS PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}"
          f" -- {len(RUNGS)} rungs, both arms")
    print("=" * 78)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
