#!/usr/bin/env python3
"""
gate_observed.py -- turn the modelled config collapse into a log line.

WHAT IS BEING CORRECTED
-----------------------
`kda/baseline_pin.py:221-250` and the notes that quote it say, from source
reading, that a Tesla T4 fails fla's shared-memory gate.  The claim is right
about the direction and wrong about the consequence.  The gate is

    fla/utils/_device.py:196-201
        @cache
        def check_shared_mem(arch="none", tensor_idx=0):
            device_shared_mem_list = get_all_max_shared_mem()
            max_shared_memory = device_shared_mem_list[tensor_idx]
            return max_shared_memory >= Backend.get_shared_memory(arch)

and it is consumed as

    fla/ops/common/chunk_delta_h.py:40-48
        configs=[triton.Config({'BV': BV}, num_warps=nw, num_stages=ns)
                 for nw   in GATED_DELTA_RULE_FWD_H_NUM_WARPS
                 for ns   in ([2,3,4] if check_shared_mem('ampere') else [2,1])
                 for BV   in ([32,64] if check_shared_mem('ada')    else [32])]

so a False from the gate is a LIST COMPREHENSION CONDITION, not a refusal.  It
returns a shorter config list.  fla then autotunes over what is left and runs.
Two existing notes (`notes/10-t4-results.md:499-507`,
`notes/17-minecu-spec.md:846`) read the gate as a refusal, and the second states
"KDA does not compile for sm_75 at all".  This file is the evidence either way.

THE SCOPE LABEL IS BUILT HERE
-----------------------------
Every line of the label that will be printed next to every fla timing in this
study -- "num_stages=[2,1], BV=[32], process-global IEEE fp32" -- is produced
by this file from live objects on this card, not transcribed from the source
reading that motivated it.  Specifically it reads:

  1. `check_shared_mem()` and `check_shared_mem('ampere')` -- the call, with its
     return value
  2. `Backend.get_shared_memory` for every arch tag, and the enum itself
  3. `max_shared_mem` from four independent readers: fla, triton, torch, and
     cudaDeviceGetAttribute(97) through ctypes
  4. the LIVE autotune config list of the kernel that actually runs, read off
     the installed object -- the number of configs, and every (BV, num_warps,
     num_stages) triple in it
  5. the config autotune ACTUALLY PICKED, after a real launch, from
     `Autotuner.cache` -- the config space is what was offered, this is what
     won, and they are different facts
  6. `IS_TF32_SUPPORTED`, `os.environ['TRITON_F32_DEFAULT']`,
     `triton.knobs.language.fp32_default`, and the `allowed_dot_input_
     precisions` of a kernel that was really compiled
  7. a FUNCTIONAL dot-precision probe: a three-line `tl.dot` kernel with no
     explicit `input_precision`, compiled here, whose PTX is searched for
     `mma.sync` against `fma.rn.f32`.  This does not infer the precision from a
     variable; it reads it out of the artifact the compiler produced.
  8. what the T4's tensor cores are, checked by asking ptxas what it can emit
     for sm_75 rather than by quoting a table

Run:  python kda_t4/gate_observed.py
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
if HERE not in sys.path:
    sys.path.insert(0, HERE)          # so `import world` works from any cwd
os.makedirs(OUT, exist_ok=True)

LOG: list = []
R: dict = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def hr(t=""):
    say("")
    say("=" * 76)
    if t:
        say(t)
        say("=" * 76)


def note(tag, msg):
    say(f"  [{tag}] {msg}")


# --------------------------------------------------------------------- 1 + 2
def the_gate():
    hr("1. the gate itself, called")
    from fla.utils import _device as D

    say("  fla/utils/_device.py, read live")
    say(f"    IS_TF32_SUPPORTED          {getattr(D, 'IS_TF32_SUPPORTED', '<absent>')}")
    say(f"    IS_GATHER_SUPPORTED        {getattr(D, 'IS_GATHER_SUPPORTED', '<absent>')}")
    say(f"    IS_TMA_SUPPORTED           {getattr(D, 'IS_TMA_SUPPORTED', '<absent>')}")
    say(f"    IS_NVIDIA                  {getattr(D, 'IS_NVIDIA', '<absent>')}")
    say(f"    IS_NVIDIA_HOPPER           {getattr(D, 'IS_NVIDIA_HOPPER', '<absent>')}")
    say(f"    IS_NVIDIA_BLACKWELL        {getattr(D, 'IS_NVIDIA_BLACKWELL', '<absent>')}")
    R["fla_flags"] = {k: str(getattr(D, k, None)) for k in
                      ("IS_TF32_SUPPORTED", "IS_GATHER_SUPPORTED", "IS_TMA_SUPPORTED",
                       "IS_NVIDIA", "IS_NVIDIA_HOPPER", "IS_NVIDIA_BLACKWELL",
                       "IS_NVIDIA_SM100")}

    say("")
    say("  Backend enum, from the installed class")
    for m in D.Backend:
        say(f"    {m.name:<10} {m.value:>8,}")
    R["backend_enum"] = {m.name: m.value for m in D.Backend}

    say("")
    say("  get_shared_memory(tag) -> the threshold each tag actually asks for")
    thresholds = {}
    for tag in ("none", "hopper", "ampere", "ada", "DEFAULT", "nonsense"):
        try:
            v = D.Backend.get_shared_memory(tag)
        except BaseException as e:
            v = f"<{type(e).__name__}: {str(e)[:40]}>"
        thresholds[tag] = v
        say(f"    {tag:<10} {v}")
    R["thresholds"] = thresholds

    say("")
    say("  check_shared_mem(...), CALLED  (not inferred)")
    calls = {}
    for tag in ("none", "ada", "ampere", "hopper"):
        for idx in (0,):
            try:
                v = D.check_shared_mem(tag, idx)
            except BaseException as e:
                v = f"<{type(e).__name__}: {str(e)[:50]}>"
            calls[f"{tag}@{idx}"] = v
            say(f"    check_shared_mem({tag!r}, {idx}) = {v}")
    R["check_shared_mem"] = calls

    say("")
    say("  NOTE the difference between these two lines, because it is the whole")
    say("  point of this file:")
    say("      a False here is a list-comprehension condition inside an")
    say("      autotune config list (chunk_delta_h.py:40-48). It is not a raise,")
    say("      not an exception, not a skip. fla keeps going over a shorter list.")


# ------------------------------------------------------------------------- 3
def smem_readers():
    hr("2. max_shared_mem, read four ways")
    import torch
    import triton
    from fla.utils import _device as D

    got = {}

    # (a) fla
    try:
        v = D.get_all_max_shared_mem()
        got["fla.get_all_max_shared_mem"] = v
        say(f"  fla   get_all_max_shared_mem()        {v}")
    except BaseException as e:
        got["fla.get_all_max_shared_mem"] = f"<{type(e).__name__}>"
        say(f"  fla   get_all_max_shared_mem()        FAILED {type(e).__name__}")

    # (b) triton, directly -- this is what fla reads, one call lower
    for i in range(torch.cuda.device_count()):
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(i)
            got[f"triton[{i}]"] = int(props["max_shared_mem"])
            say(f"  triton get_device_properties({i})       "
                f"{int(props['max_shared_mem']):,}   keys={sorted(props.keys())}")
        except BaseException as e:
            got[f"triton[{i}]"] = f"<{type(e).__name__}: {str(e)[:50]}>"
            say(f"  triton get_device_properties({i})       FAILED {type(e).__name__}")

    # (c) torch
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        for f in ("shared_memory_per_block", "shared_memory_per_block_optin",
                  "shared_memory_per_multiprocessor"):
            v = getattr(p, f, None)
            if v is not None:
                got[f"torch.{f}[{i}]"] = int(v)
                say(f"  torch  {f}[{i}]{' ' * (28 - len(f))} {int(v):,}")

    # (d) the CUDA attribute itself, by its documented enum value
    lib = None
    for cand in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            lib = ctypes.CDLL(cand)
            break
        except BaseException:
            continue
    if lib is None:
        say("  ctypes cudaDeviceGetAttribute(97)       libcudart not loadable")
        got["ctypes.attr97"] = "<no libcudart>"
    else:
        for i in range(torch.cuda.device_count()):
            v = ctypes.c_int(0)
            try:
                rc = lib.cudaDeviceGetAttribute(ctypes.byref(v), 97, ctypes.c_int(i))
                got[f"ctypes.attr97[{i}]"] = int(v.value) if rc == 0 else f"rc={rc}"
                say(f"  ctypes cudaDeviceGetAttribute(97, {i})   "
                    f"{v.value if rc == 0 else 'rc=' + str(rc):,}")
            except BaseException as e:
                got[f"ctypes.attr97[{i}]"] = f"<{type(e).__name__}>"
    R["smem_readers"] = got

    # agreement, on the one number that matters
    vals = set()
    for k, v in got.items():
        if isinstance(v, int) and ("optin" in k or "max_shared_mem" in k
                                   or "attr97" in k or "per_block" in k):
            vals.add(v)
    say("")
    say(f"  distinct optin values seen: {sorted(vals)}")
    say(f"  the gate asks for >= {D.Backend.AMPERE.value:,} (ampere) and "
        f">= {D.Backend.ADA.value:,} (ada) and "
        f">= {D.Backend.DEFAULT.value:,} (default)")
    R["optin_distinct"] = sorted(vals)
    return got


# ------------------------------------------------------------------------- 8
def what_can_smx75_do():
    hr("3. what the T4's silicon can do with a tl.dot -- asked of the compiler")
    import torch
    cap = torch.cuda.get_device_capability(0)
    say(f"  compute capability: {cap[0]}.{cap[1]}  ->  sm_{cap[0]}{cap[1]}")
    R["cc"] = list(cap)

    import shutil
    ptxas = None
    for cand in (shutil.which("ptxas"), "/usr/local/cuda/bin/ptxas",
                 "/usr/local/cuda-12.8/bin/ptxas", "/usr/local/cuda-12.9/bin/ptxas"):
        if cand and os.path.exists(cand):
            ptxas = cand
            break
    if ptxas is None:
        say("  ptxas not found; the direct-assembly probe is skipped")
    else:
        v = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
        last = (v.stdout.strip().splitlines() or v.stderr.strip().splitlines() or [""])[-1]
        say(f"  ptxas: {ptxas}")
        say(f"  {last}")
        R["ptxas"] = ptxas
        R["ptxas_version"] = last

        # A hand-written PTX `mma.sync` probe was tried here first and is not
        # used: getting the operand register counts and layout qualifiers right
        # for five different shapes is a project in itself, and a malformed one
        # is REFUSED for a syntax reason that reads exactly like an arch
        # refusal.  An earlier run refused all five shapes INCLUDING at sm_80,
        # which is how the mistake became visible.  The Triton probe below
        # answers the same question from the same toolchain without that
        # failure mode, so the hand-written probe is not run at all.
        say("")
        say("  (a hand-written PTX mma.sync probe is deliberately NOT used here;")
        say("   a malformed operand list is refused for a syntax reason that reads")
        say("   like an arch refusal, and the Triton probe in section 6 answers")
        say("   the same question from the same ptxas with none of that risk.)")

    # Now the question that matters, in the only form that is unambiguous:
    # compile ONE tiny tl.dot kernel per arch and read what came out.
    say("")
    say("  the same 32x32x32 tl.dot, AOT-compiled per arch, PTX read for MMA:")
    say("  (this is a COMPILATION result per arch. Only the sm_75 row was")
    say("   executed; the others are what the compiler emits, nothing more.)")
    try:
        res = dot_arch_matrix()
        for prec in ("ieee", "tf32"):
            say("")
            say(f"    TRITON_F32_DEFAULT = {prec!r}"
                f"{'   <- what fla forces on this card' if prec == 'ieee' else '   <- what an sm_80+ card resolves to'}")
            say(f"    cc    mma.sync   mma(any)   wgmma  tcgen05   fma.rn.f32   cvt->tf32")
            for cc in (75, 80, 86, 89, 90, 100):
                r = res.get(f"cc{cc}_{prec}", {})
                if r.get("error"):
                    say(f"    {cc:<5} COMPILE FAILED: {r['error'][:80]}")
                    continue
                say(f"    {cc:<5} {r['mma_sync']:>8} {r['mma_any']:>10} "
                    f"{r['wgmma']:>7} {r['tcgen']:>8} {r['fma_f32']:>12} "
                    f"{r['cvt_tf32']:>12}")
        a = res.get("cc75_ieee", {})
        b = res.get("cc80_ieee", {})
        c = res.get("cc80_tf32", {})
        say("")
        if a and b and c and not a.get("error"):
            say(f"    sm_75 + ieee : {a['mma_sync']} mma.sync, {a['fma_f32']} fma.rn.f32")
            say(f"    sm_80 + ieee : {b['mma_sync']} mma.sync, {b['fma_f32']} fma.rn.f32")
            say(f"    sm_80 + tf32 : {c['mma_sync']} mma.sync, {c['fma_f32']} fma.rn.f32")
            say("")
            if c["mma_sync"] > 0 and a["mma_sync"] == 0:
                say("    Reading.  The silicon at sm_80 can take a tf32 MMA path; the")
                say("    silicon at sm_75 cannot, and fla additionally switches the")
                say("    whole PROCESS off it by writing TRITON_F32_DEFAULT='ieee' at")
                say("    import.  On this card the two gates fail independently and in")
                say("    the same direction, which is why every dot becomes fma.rn.f32")
                say("    and no MMA instruction is emitted anywhere -- section 6 shows")
                say("    that in the artifact of a real launch, and section A of")
                say("    sass_t4 shows it in 46 real cubins.")
            elif c["mma_sync"] == 0:
                say("    sm_80 with tf32 ALSO emits no MMA, so the switch is not the")
                say("    whole story and the report has to say so.  (Recorded rather")
                say("    than explained away.)")
    except BaseException as e:
        say(f"  the per-arch dot matrix failed: {type(e).__name__}: {str(e)[:160]}")
        R.setdefault("failures", []).append(f"dot_arch_matrix: {type(e).__name__}")


def dot_arch_matrix():
    """One tl.dot, no explicit input_precision, AOT-compiled for each arch.

    This is the arch question with an artifact as the witness.  A compiler
    target is enough to compile; no device of that arch is needed and none is
    used.
    """
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget

    src = (
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def dot_probe(a, b, c, M: tl.constexpr, N: tl.constexpr,"
        " K: tl.constexpr):\n"
        "    ia = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]\n"
        "    ib = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]\n"
        "    z = tl.dot(tl.load(a + ia), tl.load(b + ib))\n"
        "    tl.store(c + tl.arange(0, M)[:, None] * N +"
        " tl.arange(0, N)[None, :], z)\n"
    )
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    import importlib.util
    spec = importlib.util.spec_from_file_location("dot_arch_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.dot_probe

    sig = {"a": "*fp32", "b": "*fp32", "c": "*fp32",
           "M": "constexpr", "N": "constexpr", "K": "constexpr"}

    # The F32_DEFAULT CONTROL.  This function is called AFTER fla has been
    # imported by section 1, so os.environ['TRITON_F32_DEFAULT'] is already
    # 'ieee' and EVERY compile in it is an IEEE one.  A first version of this
    # probe therefore returned "0 mma.sync" for all six architectures, which
    # reads as "no architecture has a tensor-core dot path" and is a statement
    # about the ENVIRONMENT, not about the silicon.  So the knob is swept
    # explicitly: 'ieee' is what fla forces on this card, 'tf32' is what an
    # sm_80+ card would resolve to instead.
    #
    # The knob is read from os.environ at call time
    # (triton knobs.py env_opt_str.get -> os.environ), so setting and restoring
    # it here is enough and no Triton internals are touched.
    saved = os.environ.get("TRITON_F32_DEFAULT", "<unset>")
    out = {}
    for prec in ("ieee", "tf32"):
        os.environ["TRITON_F32_DEFAULT"] = prec
        for cc in (75, 80, 86, 89, 90, 100):
            key = f"cc{cc}_{prec}"
            try:
                import triton.knobs as _tk
                resolved = _tk.language.fp32_default
                s = triton.compiler.ASTSource(fn=fn, signature=dict(sig),
                                             constexprs={"M": 32, "N": 32, "K": 32})
                k = triton.compile(s, target=GPUTarget("cuda", cc, 32),
                                   options={"num_warps": 4, "num_stages": 2})
                ptx = k.asm["ptx"]
                hdr = " ".join(l.strip() for l in ptx.splitlines()[:8]
                               if l.startswith((".version", ".target")))
                out[key] = dict(
                    f32_default=prec, knob_resolved=repr(resolved), cc=cc,
                    header=hdr,
                    mma_sync=len(re.findall(r"\bmma\.sync\b", ptx)),
                    mma_any=len(re.findall(r"\bmma\.", ptx)),
                    wgmma=len(re.findall(r"\bwgmma\.", ptx)),
                    tcgen=len(re.findall(r"\btcgen05\.", ptx)),
                    fma_f32=len(re.findall(r"\bfma\.rn\.f32\b", ptx)),
                    cvt_tf32=len(re.findall(r"\bcvt\.[a-z0-9.]*tf32", ptx)),
                    n_regs=getattr(k, "n_regs", None),
                    cubin_sha256=hashlib.sha256(k.asm["cubin"]).hexdigest())
            except BaseException as e:
                out[key] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
    if saved == "<unset>":
        os.environ.pop("TRITON_F32_DEFAULT", None)
    else:
        os.environ["TRITON_F32_DEFAULT"] = saved
    try:
        os.unlink(path)
    except OSError:
        pass
    R["dot_arch_matrix"] = out
    return out


# --------------------------------------------------------------- 4 + 5 + 6 + 7
def unwrap_autotuner(obj, depth=0):
    """fla stacks @triton.heuristics outside @fla_cache_autotune outside
    @triton.jit, so the object bound to the module name is a `Heuristics` whose
    `.configs` do not exist.  The Autotuner is somewhere under `.fn`.  Walk it
    and say what was found, because a wrong walk would silently report an empty
    config space -- which is the exact shape of the error this study is about.
    """
    chain, cur = [], obj
    for _ in range(6):
        if cur is None:
            break
        chain.append(type(cur).__name__)
        if hasattr(cur, "configs"):
            return cur, chain
        cur = getattr(cur, "fn", None)
    return None, chain


def the_kernel_and_its_configs():
    hr("4. the kernel that actually runs, and the config list it was given")
    import fla.ops.common.chunk_delta_h as CDH

    raw = getattr(CDH, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", None)
    if raw is None:
        say("  kernel not found in fla.ops.common.chunk_delta_h")
        return
    fn, chain = unwrap_autotuner(raw)
    say(f"  decorator chain, outermost first: {' -> '.join(chain)}")
    say(f"  autotuner found: {fn is not None}")
    if fn is None:
        say("  no Autotuner in the chain; the config space cannot be read")
        R["autotune_configs"] = None
        return
    say(f"  module-level: NUM_WARPS = {getattr(CDH, 'NUM_WARPS', None)}")
    say(f"  module-level: GATED_DELTA_RULE_FWD_H_NUM_WARPS = "
        f"{getattr(CDH, 'GATED_DELTA_RULE_FWD_H_NUM_WARPS', None)}")
    R["num_warps_module"] = str(getattr(CDH, "GATED_DELTA_RULE_FWD_H_NUM_WARPS", None))

    cfgs = getattr(fn, "configs", None)
    if cfgs is None:
        say("  no .configs on the Autotuner")
        R["autotune_configs"] = None
        return
    say(f"  THE CONFIG SPACE, live: {len(cfgs)} configs")
    rows = []
    for c in cfgs:
        row = dict(kwargs={k: str(v) for k, v in c.kwargs.items()},
                   num_warps=c.num_warps, num_stages=c.num_stages,
                   num_ctas=getattr(c, "num_ctas", None))
        rows.append(row)
        say(f"    BV={c.kwargs.get('BV')!s:<4} num_warps={c.num_warps!s:<3} "
            f"num_stages={c.num_stages}")
    R["autotune_configs"] = rows
    R["autotune_n_configs"] = len(rows)
    bvs = sorted({c.kwargs.get("BV") for c in cfgs})
    nss = sorted({c.num_stages for c in cfgs})
    nws = sorted({c.num_warps for c in cfgs})
    say("")
    say(f"  distinct BV        {bvs}")
    say(f"  distinct num_stages{nss}")
    say(f"  distinct num_warps {nws}")
    R["distinct"] = dict(BV=bvs, num_stages=nss, num_warps=nws)

    # the same facts, computed from the gate functions, so the two can disagree
    from fla.utils import _device as D
    model_BV = [32, 64] if D.check_shared_mem("ada") else [32]
    model_ns = [2, 3, 4] if D.check_shared_mem("ampere") else [2, 1]
    say("")
    say("  the model says, from the gate return values:")
    say(f"    num_stages {model_ns}      (check_shared_mem('ampere') = {D.check_shared_mem('ampere')})")
    say(f"    BV         {model_BV}         (check_shared_mem('ada')    = {D.check_shared_mem('ada')})")
    say("")
    obs_BV, obs_ns, obs_nw = set(bvs), set(nss), set(nws)
    agree = (obs_BV == set(model_BV) and obs_ns == set(model_ns))
    say(f"  MODEL vs OBSERVED: {'AGREE' if agree else 'DISAGREE'}")
    if not agree:
        say(f"    observed num_stages {sorted(obs_ns)} vs modelled {sorted(model_ns)}")
        say(f"    observed BV         {sorted(obs_BV)} vs modelled {sorted(model_BV)}")
    if list(obs_ns) != sorted(model_ns):
        say(f"    *** the model's list and the live list are not even the same "
            f"ORDER: {model_ns} vs {sorted(obs_ns)}")
    R["model_vs_observed"] = dict(
        model_BV=model_BV, model_num_stages=model_ns,
        observed_BV=sorted(obs_BV), observed_num_stages=sorted(obs_ns),
        observed_num_warps=sorted(obs_nw), agree=agree)
    return fn


def fwd_and_selected():
    """Run the op once so autotune PICKED a config, then read Autotuner.cache."""
    hr("5. one real forward, then the config autotune actually picked")
    import torch
    from fla.ops.kda import chunk_kda
    import world

    B, T, H, HV, K, V = 1, 128, 2, 2, 128, 128
    dt = torch.bfloat16
    say(f"  calling fla.ops.kda.chunk_kda  B={B} T={T} H={H} HV={HV} K={K} V={V} "
        f"dtype={dt}  grad off")
    say("  inputs: the STUDY's own generator (kda/kda_vs_fla.py:296 `rnd`), with")
    say("  q,k L2-normalised. Raw Gaussians make the delta-rule matrix")
    say("  (I - beta k k^T) have eigenvalues near 1 - K*beta and the state")
    say("  diverges; that is a property of the generator, not of fla.")
    say(f"  dtype: {dt} is the only option. The FlashKDA verifier requires bf16")
    say("  (flash_kda.py:64-88) and the study's call shape is bf16")
    say("  (kda/baseline_pin.py:392). fp16 would be a different payload type")
    say("  with a different numerics story, not a free substitute.")
    R["call_shape"] = dict(B=B, T=T, H=H, HV=HV, K=K, V=V, dtype="bfloat16",
                           generator="kda/kda_vs_fla.py:296 rnd(seed=0)")

    q, k, v, g, beta = world.make(B, T, H, HV, K, V, seed=0)
    tq, tk, tv = (world.to_torch(x, "bfloat16") for x in (q, k, v))
    tg = world.to_torch(g, "float32")
    tb = world.to_torch(beta, "bfloat16")
    scale = world.scale_for(K)

    with torch.inference_mode():
        o, _ = chunk_kda(tq, tk, tv, tg, tb, scale=scale,
                         use_qk_l2norm_in_kernel=False)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(o).all())
    say(f"  returned o: shape={tuple(o.shape)} dtype={o.dtype} "
        f"finite={finite}  absmax={o.float().abs().max().item():.4e}")
    R["fwd_ok"] = True
    R["fwd_finite"] = finite
    R["fwd_absmax"] = float(o.float().abs().max().item())
    if not finite:
        say("  *** non-finite output. That is a result, not something to retry away.")

    import fla.ops.common.chunk_delta_h as CDH
    fn, chain = unwrap_autotuner(CDH.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
    cache = getattr(fn, "cache", None) if fn is not None else None
    say("")
    if not cache:
        say("  Autotuner.cache is empty -- the kernel may not have been reached, "
            "or autotune stores elsewhere")
        R["selected_configs"] = None
    else:
        say(f"  Autotuner.cache has {len(cache)} entries after ONE call:")
        sel = []
        for k, v in cache.items():
            d = dict(key=str(k),
                     kwargs={kk: str(vv) for kk, vv in v.kwargs.items()},
                     num_warps=v.num_warps, num_stages=v.num_stages)
            sel.append(d)
            say(f"    key={k}")
            say(f"      -> BV={v.kwargs.get('BV')} num_warps={v.num_warps} "
                f"num_stages={v.num_stages}  (from a space of "
                f"{len(getattr(fn, 'configs', []))})")
        R["selected_configs"] = sel
        say("")
        say("  the config space was the SHORTENED one, and autotune picked out of")
        say("  exactly that short list. There was no refusal anywhere in this run.")


# ------------------------------------------------------------------------- 7
DOT_PROBE = r'''
import triton, triton.language as tl, torch, os, json

@triton.jit
def _dot_probe(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """One tl.dot, NO explicit input_precision. Whatever the compiler resolves
    it to is the answer."""
    ia = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]
    ib = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(a + ia)
    y = tl.load(b + ib)
    z = tl.dot(x, y)
    tl.store(c + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], z)

M = N = K = 32
a = torch.randn(M, K, device="cuda")
b = torch.randn(K, N, device="cuda")
c = torch.empty(M, N, device="cuda")
k = _dot_probe[(1,)](a, b, c, M, N, K, num_warps=4, num_stages=2)
out = {}
ptx = k.asm["ptx"]
out["ptx_mma_sync"] = len(re.findall(r"\bmma\.sync\b", ptx))
out["ptx_mma"] = len(re.findall(r"\bmma\.", ptx))
out["ptx_fma_rn_f32"] = len(re.findall(r"\bfma\.rn\.f32\b", ptx))
out["ptx_cvt_tf32"] = len(re.findall(r"\bcvt\.(rn\.|tf32)", ptx))
out["ptx_lines_mma"] = [l.strip() for l in ptx.splitlines() if "mma." in l][:12]
out["ptx_target"] = [l.strip() for l in ptx.splitlines()[:8]
                     if l.startswith((".version", ".target"))]
out["ptx_sha256"] = __import__("hashlib").sha256(ptx.encode()).hexdigest()
out["cubin_sha256"] = __import__("hashlib").sha256(k.asm["cubin"]).hexdigest()
out["n_regs"] = k.n_regs
out["n_spills"] = k.n_spills
out["env_TRITON_F32_DEFAULT"] = __import__("os").environ.get("TRITON_F32_DEFAULT", "<unset>")
try:
    import triton.knobs as _tk
    out["triton_knob_fp32_default"] = repr(_tk.language.fp32_default)
except BaseException as _e:
    out["triton_knob_fp32_default"] = "<%s>" % type(_e).__name__
with open(os.environ["DOT_OUT"], "w") as f:
    json.dump(out, f, indent=2)
print("DOT_PROBE " + json.dumps({kk: vv for kk, vv in out.items()
                                 if kk != "ptx_lines_mma"}))
for l in out["ptx_lines_mma"]:
    print("    mma line:", l[:110])
'''


def dot_precision():
    hr("6. dot precision, read out of a compiled artifact")
    say("  A 32x32x32 tl.dot with no explicit input_precision, compiled here on")
    say("  this card, and the PTX searched for mma.sync vs fma.rn.f32.")
    say("  This is not a read of a variable; it is a read of the code ptxas was")
    say("  handed.  If fla had left TRITON_F32_DEFAULT alone, this would emit")
    say("  mma.sync with tf32 operands.  If it wrote 'ieee', it emits fma.rn.f32.")
    say("")
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".py", prefix="dotprobe_")
    with os.fdopen(fd, "w") as f:
        f.write("import re\n" + DOT_PROBE)
    env = dict(os.environ, DOT_OUT=os.path.join(OUT, "dot_probe.json"))
    p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                       env=env, cwd=OUT)
    for line in (p.stdout or "").splitlines():
        say("  " + line)
    if p.returncode != 0:
        say(f"  probe FAILED rc={p.returncode}")
        say("  --- stderr ---")
        say((p.stderr or "")[-2500:])
    R["dot_probe_rc"] = p.returncode
    try:
        os.unlink(path)
    except OSError:
        pass

    say("")
    say("  and the knob itself, after fla was imported by that subprocess:")
    import triton.knobs as tk
    say(f"    os.environ['TRITON_F32_DEFAULT']          "
        f"{os.environ.get('TRITON_F32_DEFAULT', '<unset>')}")
    say(f"    triton.knobs.language.fp32_default         {tk.language.fp32_default!r}")
    from fla.utils import _device as D
    say(f"    fla IS_TF32_SUPPORTED                     {D.IS_TF32_SUPPORTED}")
    say(f"    is 'ieee' set by fla, or was it already there when fla loaded?")
    # The order question has to be asked in a process that has NOT already
    # imported fla.  This process has (section 5 imported it), so fla already
    # wrote the variable into ITS os.environ and any child inherits it.  A
    # naive before/after here answers "before: ieee" and looks like someone
    # else set it.  So: scrub the variable out of the child's environment and
    # import fla in a pristine process.
    clean = {k: v for k, v in os.environ.items() if k != "TRITON_F32_DEFAULT"}
    chk = subprocess.run(
        [sys.executable, "-c",
         "import os;"
         "print('BEFORE fla import:', repr(os.environ.get('TRITON_F32_DEFAULT')));"
         "import fla;"
         "print('AFTER  fla import:', repr(os.environ.get('TRITON_F32_DEFAULT')));"
         "import triton.knobs as tk;"
         "print('triton knob        :', repr(tk.language.fp32_default))"],
        capture_output=True, text=True, cwd=OUT, env=clean)
    for line in (chk.stdout or "").splitlines():
        say("    " + line)
    if chk.returncode != 0:
        say("    (child failed) " + (chk.stderr or "").strip().splitlines()[-1][:110])
    R["f32_default_order"] = (chk.stdout or chk.stderr or "").strip()
    R["f32_default_order_scrubbed"] = ("BEFORE fla import: None" in (chk.stdout or ""))

    say("")
    say("  SOLVE_TRIL_DOT_PRECISION, the one dot fla pins explicitly:")
    try:
        from fla.ops.kda.chunk_intra import SOLVE_TRIL_DOT_PRECISION
        say(f"    {SOLVE_TRIL_DOT_PRECISION}")
        R["solve_tril_dot_precision"] = str(SOLVE_TRIL_DOT_PRECISION)
    except BaseException as e:
        say(f"    unavailable: {type(e).__name__}: {str(e)[:70]}")


# ------------------------------------------------------------------ the label
def build_label():
    hr("7. THE SCOPE LABEL, assembled from what was observed above")
    d = R.get("distinct", {})
    ns = d.get("num_stages")
    bv = d.get("BV")
    nw = d.get("num_warps")
    sol = R.get("f32_default_order", "")
    ieee = ("AFTER  fla import: 'ieee'" in sol) or (
        os.environ.get("TRITON_F32_DEFAULT") == "ieee")
    bits = [f"num_stages={ns}"]
    if bv is not None:
        bits.append(f"BV={bv}")
    if nw is not None:
        bits.append(f"num_warps={nw}")
    bits.append("process-global IEEE fp32 (TRITON_F32_DEFAULT='ieee', written by "
                "fla at import)" if ieee else "tf32")
    label = ("fla on T4, degraded configuration - " + ", ".join(bits)
             + " - this is what the reference does in its worst configuration, "
               "not what it does at its best")
    say("")
    say("  " + label)
    say("")
    say("  short form, for a table cell:")
    short = f"fla/T4 degraded [stages={'/'.join(str(x) for x in (ns or []))}, "
    short += f"BV={'/'.join(str(x) for x in (bv or []))}, "
    short += f"warps={'/'.join(str(x) for x in (nw or []))}, "
    short += "ieee-fp32 dots]" if ieee else "]"
    say("  " + short)
    R["scope_label"] = label
    R["scope_label_short"] = short
    R["scope_label_ieee"] = bool(ieee)
    say("")
    say("  NOT a claim about fla.  The same code on a card with >= 166,912 B")
    say("  optin shared memory and cc major >= 8 gets num_stages=[2,3,4],")
    say("  BV=[32,64] and tf32 dots.  Nothing here measures that.")


def main():
    hr("kda_t4 GATE -- the config collapse, observed on this card")
    for fn in (the_gate, smem_readers, what_can_smx75_do,
               the_kernel_and_its_configs, fwd_and_selected,
               dot_precision):
        try:
            fn()
        except BaseException as e:
            import traceback
            say(f"  !! {fn.__name__} raised {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc(file=sys.stdout)
            R.setdefault("failures", []).append(f"{fn.__name__}: {type(e).__name__}: {str(e)[:160]}")
    try:
        build_label()
    except BaseException as e:
        say(f"  label assembly failed: {type(e).__name__}: {str(e)[:120]}")

    with open(os.path.join(OUT, "gate_observed.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    with open(os.path.join(OUT, "gate_observed.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    say("")
    say("wrote kda_t4/out/gate_observed.{txt,json} and dot_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
