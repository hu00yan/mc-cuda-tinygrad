"""
AOT-compile flash-linear-attention's KDA Triton kernels for every compute capability.
No GPU.  Only the target arch varies.

Why KDA is the right specimen: the whole fla KDA path is a `tl.dot` workload.
`grep -rn make_tensor_descriptor fla/ops/kda` returns nothing, so there is no TMA in
it either.  Whatever `tl.dot` lowers to on an old arch IS KDA's behaviour there.

How the kernel is obtained without torch: torch's wheel host is unreachable from this
network and is not needed -- the kernels are pure Triton.  This script parses the real
`fla/ops/kda/*.py` with `ast`, lifts every top-level `@triton.jit` function, strips
its autotune/heuristic decorators, and execs the result in a namespace containing
`triton`, `triton.language`, a permissive `torch` shim (the kernels only use
`torch.float32`-style dtype *names* as compile-time constants) and the fla math
aliases the device code expects.  The device code being compiled is unmodified
upstream source, not a paraphrase.

Signature and constexpr values are transcribed from the production launch site at
fla/ops/kda/chunk_intra.py:879 (K=128 head dim, BT=64 chunk, BC=16 sub-chunk).
"""
import ast
import importlib.metadata
import json
import pathlib
import re
import sys
import types

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as tcompile

TRITON_VERSION = importlib.metadata.version("triton")

FLA = pathlib.Path("/w/subj/fla/flash-linear-attention-main/fla/ops/kda")
OUT = pathlib.Path("/w/out/triton")
OUT.mkdir(parents=True, exist_ok=True)
CAPS = [70, 75, 80, 86, 89, 90, 100, 120]

MIX = [
    ("wgmma",    r"\bwgmma\b"),
    ("tcgen05",  r"\btcgen05\b"),
    ("mma.sync", r"\bmma\.sync\b"),
    ("FMA.f32",  r"\bfma\.rn\.f32\b"),
    ("cp.async", r"\bcp\.async\b"),
    ("tensormap", r"\btensormap\b"),
    ("mbarrier", r"\bmbarrier\b"),
    ("ldmatrix", r"\bldmatrix\b"),
    ("bar.sync", r"\bbar\.sync\b"),
]


def torch_shim():
    """The kernels reference torch.dtype *names* (torch.float32, torch.bfloat16)
    as compile-time constants. A permissive object satisfies that without torch."""
    m = types.ModuleType("torch")
    for n in ("float32", "float16", "bfloat16", "float64", "int32", "int64",
              "uint8", "int8", "bool", "float8_e4m3fn", "float8_e5m2"):
        setattr(m, n, n)

    class _Any:
        def __getattr__(self, k):
            return _Any()

        def __call__(self, *a, **kw):
            return _Any()

    m.Tensor = _Any
    m.dtype = _Any()
    return m


def make_ns():
    ns = {"triton": triton, "tl": tl, "torch": torch_shim(), "__name__": "fla_kda_lifted"}
    # fla device code imports these names from fla.utils / math directly
    for n in ("exp", "exp2", "log", "log2", "log1p", "sigmoid", "sqrt", "rsqrt",
              "floor", "ceil", "abs", "minimum", "maximum", "where", "sum", "cumsum",
              "softmax", "tanh", "erf", "clamp", "nextafter", "isnan", "full", "zeros",
              "arange", "dot_scaled", "make_block_ptr", "join", "split", "interleave",
              "permute", "trans", "reshape", "view", "expand", "broadcast_to", "cast",
              "dot", "fma", "mul", "add", "div", "sub"):
        if hasattr(tl, n):
            ns[n] = getattr(tl, n)
    return ns


NS = make_ns()


def lift_jit_functions(path: pathlib.Path):
    """Return [(name, source)] for every top-level @triton.jit function in `path`,
    each unparsed individually so blank lines inside a body cannot split it."""
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any("triton.jit" in ast.unparse(d) for d in node.decorator_list):
            continue
        # Keep ONLY the @triton.jit(...) decorator -- it is what turns the Python
        # function into a JITFunction, and fla always uses the calling form
        # `@triton.jit(do_not_specialize=[...])`.  Drop triton.heuristics and
        # fla's fla_cache_autotune: AOT compilation supplies the constexprs (incl.
        # IS_VARLEN) directly, and the autotune wrapper needs a live device.
        node.decorator_list = [d for d in node.decorator_list
                               if "triton.jit" in ast.unparse(d)]
        out.append((node.name, ast.unparse(node)))
    return out


LIFTED = pathlib.Path("/w/out/kda_lifted")
LIFTED.mkdir(parents=True, exist_ok=True)

# fla device code uses these as bare names (imported at module level from
# fla.utils / fla.ops.common).  They are all thin aliases of triton.language
# builtins, so the generated module binds them from _tl.
FLA_BARE_NAMES = (
    "exp2", "exp", "log", "log2", "log1p", "sigmoid", "sqrt", "rsqrt", "floor",
    "ceil", "abs", "minimum", "maximum", "where", "sum", "cumsum", "softmax",
    "tanh", "erf", "clamp", "nextafter", "isnan", "full", "zeros", "arange",
    "dot", "dot_scaled", "make_block_ptr", "join", "split", "interleave",
    "permute", "trans", "reshape", "view", "expand", "broadcast_to", "cast",
    "fma", "mul", "add", "div", "sub", "gather", "cdiv", "next_power_of_2",
    "use_gates_in_kernel", "is_amd", "is_intel", "is_cuda", "atom_add",
    "program_id", "num_programs", "load", "store", "static_assert", "device_assert",
    "make_block_ptr", "reduce", "associative_scan", "cumsum", "flip", "cat",
    "maximum", "minimum", "maximum", "sum", "inline_asm_elementwise",
)


def load_kernels():
    """Write each KDA module's lifted jit kernels to a real .py file (Triton's
    JITFunction calls inspect.getsourcelines, so exec-from-string fails with
    OSError), then import it."""
    import importlib.util
    shim = torch_shim()
    NS["torch"] = shim
    sys.modules["torch"] = shim
    for f in sorted(FLA.glob("*.py")):
        try:
            fns = lift_jit_functions(f)
        except SyntaxError as e:
            print(f"  (skip {f.name}: {e})")
            continue
        if not fns:
            continue
        blob = "\n\n".join(src for _, src in fns)
        mod_name = f"kda_lifted_{f.stem}"
        out_py = LIFTED / f"{mod_name}.py"
        out_py.write_text(
            "# GENERATED by src/kda_ladder.py -- do not edit.\n"
            "# Upstream source: fla/ops/kda/%s (github fla-org/flash-linear-attention)\n"
            "# Kernels lifted verbatim via ast; only @triton.heuristics and\n"
            "# @fla_cache_autotune are removed (AOT supplies the constexprs).\n"
            "import triton\n"
            "import triton.language as tl\n"
            "import torch\n"
            "import triton.language as _tl\n"
            "# fla device code imports these bare names from fla.utils; they are all\n"
            "# thin aliases of triton.language builtins, so bind them here.\n"
            "for _n in %r:\n"
            "    if hasattr(_tl, _n):\n"
            "        globals()[_n] = getattr(_tl, _n)\n"
            "# fla module-level constants the device code references.\n"
            "# SOLVE_TRIL_DOT_PRECISION: fla picks this at IMPORT time from\n"
            "# IS_TF32_SUPPORTED (a device query) -- see chunk_intra.py:21-24.\n"
            "# 'ieee' is the conservative branch and is what a no-GPU host gets.\n"
            "SOLVE_TRIL_DOT_PRECISION = _tl.constexpr('ieee')\n"
            "# fla/ops/utils/softplus.py generates this as inline PTX under\n"
            "# IS_NVIDIA; the expression below is the same function, written as a\n"
            "# @triton.jit helper, so the lifted kernel needs no vendor asm.\n"
            "# (It must be jit, not a plain global: Triton only lets a kernel read a\n"
            "# global that is itself constexpr or jit-compiled.)\n"
            "@triton.jit\n"
            "def softplus(x):\n"
            "    return _tl.log(1 + _tl.exp(x))\n"
            "\n%s\n" % (f.name, FLA_BARE_NAMES, blob))
        spec = importlib.util.spec_from_file_location(mod_name, out_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        got = [n for n, _ in fns if hasattr(getattr(mod, n, None), "arg_names")]
        print(f"  lifted {len(got)}/{len(fns)} jit kernels from {f.name}"
              f" -> {out_py.name}")
        for n in got:
            NS[n] = getattr(mod, n)
    return {k: v for k, v in NS.items()
            if hasattr(v, "arg_names") and hasattr(v, "src")}


def classify(ptx):
    return {n: len(re.findall(p, ptx, re.I)) for n, p in MIX
            if re.search(p, ptx, re.I)}


# dtypes per argument name, taken from the production launch sites:
#   chunk_intra.py:879 (inter_solve_fused / intra_sub_chunk) and
#   fused_recurrent.py (fused_recurrent_kda_fwd_kernel).
# KDA is bf16 activations with an fp32 gate/beta and an fp32 accumulator, so
# Aqk/Akk (the intra-chunk score matrices) are fp32.
ARG_TY = {
    "q": "*bf16", "k": "*bf16", "v": "*bf16", "u": "*bf16", "o": "*bf16", "w": "*bf16",
    "g": "*fp32", "beta": "*fp32", "A_log": "*fp32", "dt_bias": "*fp32",
    "Aqk": "*fp32", "Akkd": "*fp32", "Akk": "*fp32", "A": "*fp32",
    "h0": "*fp32", "ht": "*fp32", "lower_bound": "fp32", "scale": "fp32",
    "cu_seqlens": "*i32", "chunk_indices": "*i32", "ssm_state_indices": "*i32",
    "num_accepted_tokens": "*i32", "N": "i32", "T": "i32", "B": "i32",
}

# constexpr values.  K=128 (KDA head dim), BT=64 (chunk), BC=16 (sub-chunk),
# BK=64 -- the fla KDA defaults.  Feature flags are set to the small/normal path
# so the arch is the only variable.
CE_DEFAULTS = dict(
    H=4, HV=4, K=128, V=128, BK=64, BV=64, BT=64, BC=16, NC=16,
    IS_VARLEN=False, USE_SAFE_GATE=False, USE_GATHER=False, USE_GRAPH=False,
    stride_init_state_token=1, stride_final_state_token=1,
    stride_indices_seq=1, stride_indices_tok=1,
    USE_INITIAL_STATE=False, INPLACE_FINAL_STATE=False, IS_BETA_HEADWISE=False,
    USE_QK_L2NORM_IN_KERNEL=True, IS_CONTINUOUS_BATCHING=False,
    IS_SPEC_DECODING=False, STORE_FINAL_STATE=True, HAS_A=True, HAS_BIAS=True,
    USE_GATE_IN_KERNEL=True, USE_LOWER_BOUND=False, APPLY_BETA_SIGMOID=True,
    ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False, num_stages=1,
)

SPECS = [
    ("kda_inter_solve_fwd", "chunk_kda_fwd_kernel_inter_solve_fused"),
    ("kda_intra_sub_chunk", "chunk_kda_fwd_kernel_intra_sub_chunk"),
    ("kda_fused_recurrent", "fused_recurrent_kda_fwd_kernel"),
]


def build(fn):
    """Split the kernel's own parameter list into (signature, constexprs)."""
    sig, cex, missing = {}, {}, []
    for p in fn.params:
        if p.is_constexpr:
            if p.name in CE_DEFAULTS:
                cex[p.name] = CE_DEFAULTS[p.name]
            elif p.name == "scale":
                cex[p.name] = 0.088
            else:
                missing.append(f"constexpr:{p.name}")
        else:
            if p.name in ARG_TY:
                sig[p.name] = ARG_TY[p.name]
            else:
                missing.append(f"arg:{p.name}")
    return sig, cex, missing


def main():
    print(f"### fla KDA, triton {TRITON_VERSION}")
    kernels = load_kernels()
    print(f"  total jit kernels available: {len(kernels)}: {sorted(kernels)}")

    results = []
    for label, kname in SPECS:
        fn = kernels.get(kname)
        print(f"\n-- {label}  (fla/ops/kda/...::{kname})")
        if fn is None:
            print("   KERNEL-NOT-OBTAINED")
            results.append({"kernel": label, "cap": None,
                            "status": "KERNEL-NOT-OBTAINED", "error": "", "ptx_insts": {}})
            continue
        sig, cex, missing = build(fn)
        print(f"   {len(sig)} runtime args, {len(cex)} constexprs"
              + (f"   UNMAPPED: {missing}" if missing else ""))
        for cap in CAPS:
            rec = {"kernel": label, "cap": cap, "status": "", "error": "",
                   "ptx_insts": {}, "unmapped": missing}
            try:
                k = tcompile(ASTSource(fn=fn, signature=sig, constexprs=cex),
                             target=GPUTarget("cuda", cap, 32),
                             options={"num_warps": 4, "num_stages": 2})
                ptx = k.asm["ptx"]
                rec["status"] = "OK"
                rec["ptx_insts"] = classify(ptx)
                rec["ptx_lines"] = ptx.count("\n")
                rec["n_regs"] = getattr(k, "n_regs", None)
                rec["n_spills"] = getattr(k, "n_spills", None)
                rec["shared_bytes"] = getattr(getattr(k, "metadata", None), "shared", None)
                (OUT / f"{label}__sm{cap}.ptx").write_text(ptx)
            except Exception as e:
                rec["status"] = "REJECT"
                rec["error"] = re.sub(r"\s+", " ", f"{type(e).__name__}: {e}")[:300]
            results.append(rec)
            mix = ", ".join(f"{a}x{b}" for a, b in rec["ptx_insts"].items()) or "-"
            extra = ""
            if rec.get("ptx_lines"):
                extra = (f"  [{rec['ptx_lines']}ptx "
                         f"regs={rec.get('n_regs')} spill={rec.get('n_spills')} "
                         f"smem={rec.get('shared_bytes')}]")
            print(f"   sm_{cap:<4} {rec['status']:<7} {mix}{extra}"
                  f"{'   ' + rec['error'][:90] if rec['error'] else ''}")

    (OUT / "kda_ladder.json").write_text(json.dumps(
        {"triton_version": TRITON_VERSION,
         "fla": "github fla-org/flash-linear-attention main branch",
         "cells": results}, indent=1))
    got = [r for r in results if r["cap"] is not None]
    ok = sum(r["status"] == "OK" for r in got)
    print(f"\ncells: {len(got)}  compiled: {ok}  rejected: {len(got)-ok}")
    print(f"wrote {OUT/'kda_ladder.json'}")


if __name__ == "__main__":
    main()
