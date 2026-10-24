#!/usr/bin/env python3
"""
sass_t4.py -- the reference's real SASS, counted from a cubin, per arch.

WHY THIS FILE EXISTS
--------------------
`notes/16-generated-vs-handwritten.md` section 3.3 reports, for the hand-written
side, "128 HMMA (sm_80) / 16 HGMMA (sm_90a)" and "3 MUFU.EX2", and section 6
reports for the generated side "512 FFMA, 268 FMUL, 124 FADD, 128 MUFU.EX2.
0 tensor core".  Those are not source readings: section 3.3's own reproduction
block runs `tools/archlab/triton_aot.py`, which AOT-compiles a vendored extract
of `fla/ops/common/chunk_delta_h.py` at `GPUTarget("cuda", cc, 32)` and
disassembles the cubin with `cuobjdump -sass`.  So the method is right and the
numbers are measured.

Two things are still open, and this file closes both.

  1. THE ARCH.  The sweep in notes/16 covers cc 80 / 90 / 100.  It does not
     cover cc 75 -- the arch this entire study runs on.  A "reference SASS"
     table that stops at sm_80 says nothing about the card the timing is taken
     on.  This file adds sm_75, from the same toolchain.

  2. THE CONFIG.  The signature notes/16 compiled with is
     `tools/archlab/fla_kda_chunk_h_sig.json`, whose `__constexprs__` say
     `BV: 64, __num_stages__: 2, __num_warps__: 4`.  `BV=64` is only reachable
     when `check_shared_mem('ada')` is True, i.e. on a card with >= 101,376 B
     optin shared memory.  A T4 has 65,536 B.  So that cubin is the sm_80
     configuration, disassembled, and reading it as "what the reference
     compiles to on the card" would be a category error of exactly the kind
     this repository has retracted results for.

So this file produces TWO families of numbers, from two different artifacts,
and labels them as such:

  (A) THE REAL ARTIFACT.  The cubin fla itself compiled, on this card, for the
      config fla itself selected on this card.  Read out of Triton's on-disk
      kernel cache and out of the live `CompiledKernel` objects, then
      disassembled.  This is the only number that describes this T4.

  (B) THE AOT SWEEP.  The same kernel source, the same triton, the same ptxas,
      compiled for cc in {75, 80, 86, 89, 90, 100} at the config each arch's
      own gate would allow, plus the notes/16 config for replication.  This is
      the only way to say anything about the other cards, and it says it about
      a COMPILATION, never about a run.

Both go through the same categoriser, so the two are comparable, and both are
written to `kda_t4/out/sass/` as raw artifacts (ptx, cubin, sass text) so a
reader can re-count without running anything.

Run:  python kda_t4/sass_t4.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SASS = os.path.join(OUT, "sass")
if HERE not in sys.path:
    sys.path.insert(0, HERE)          # so `import world` works from any cwd
os.makedirs(SASS, exist_ok=True)

LOG: list = []
R: dict = {"artifacts": [], "note": ""}


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


# ------------------------------------------------------------------ counting
# One categoriser for both sides, so a "tensor core" means the same thing in
# every row.  The families are the ones the study already uses
# (notes/16 §3.1, notes/19 §4.1): the split is by SASS opcode prefix, and the
# tensor-core family is named explicitly rather than inferred from "no FMA".
TC = re.compile(r"^(HMMA|QMMA|IMMA|OMMA|BMMA|HGMMA|IMMA|TCGMMA|UMMA|OMMA|QMMA"
                r"|mma)", re.IGNORECASE)
# Matched as a PREFIX of the mnemonic, not as an exact key.  An earlier version
# used hist.get("HGMMA", 0) and reported 0 HGMMA for a cubin that plainly
# contains `HGMMA.64` sixteen times, because cuobjdump writes the operand shape
# into the mnemonic.  Two rows of notes/16's table were "wrong" against this
# instrument for exactly that reason.
TC_EXACT = ("HMMA", "QMMA", "IMMA", "OMMA", "BMMA", "HGMMA", "TCGMMA", "UMMA",
            "UTCHMMA", "TCGEN05", "QMMA")


def _pfx(hist, name):
    """count of every mnemonic that starts with `name`"""
    return sum(n for k, n in hist.items() if k.upper().startswith(name))


def parse_sass(text):
    """cuobjdump -sass output -> (total, mnemonic histogram, families).

    Line shape:
        code for sm_75
                /*0000*/                   MOV R1, c[0x0][0x28] ;
    so the mnemonic is the first token after the /*addr*/ comment.
    """
    hist = collections.Counter()
    for line in text.splitlines():
        m = re.search(r"/\*[0-9a-fA-F]{4,}\*/\s+(@!?\w+\s+)?([A-Z][A-Za-z0-9._]*)", line)
        if m:
            hist[m.group(2)] += 1
    total = sum(hist.values())
    fam = collections.Counter()
    for k, v in hist.items():
        head = k.split(".")[0].upper()
        if head in TC_EXACT or k.upper().startswith("HMMA"):
            fam["tensor_core"] += v
            if head not in ("HMMA",):
                fam.setdefault("tc_mnemonics", {})
                fam["tc_mnemonics"][head] = fam["tc_mnemonics"].get(head, 0) + v
        elif head == "MUFU":
            fam["MUFU"] += v
        elif head in ("FFMA", "FADD", "FMUL", "FSETP", "FFMA"):
            fam["float_alu"] += v
        elif head in ("IADD3", "IMAD", "LOP3", "SHF", "ISETP", "LEA", "POPC", "FLO"):
            fam["int_alu"] += v
        elif head.startswith("LDG") or head.startswith("LD") or head.startswith("STG") \
                or head.startswith("ST"):
            fam["memory"] += v
        elif head in ("BAR", "BSYNC", "NANOSLEEP", "BSSY", "BSYNC"):
            fam["sync"] += v
        else:
            fam["other"] += v
    return total, dict(hist), dict(fam)


def ptxas_v(ptx_path):
    """ptxas -v is the AUTHORITATIVE source for registers and spill bytes.

    tools/archlab/triton_aot.py:139-145 says why the other source is not used:
    `cuobjdump -res-usage` reports the local FRAME size, not spill traffic, and
    its SHARED field is wrong for the dynamic extern shared memory triton always
    emits.  ptxas writes its report to STDERR.
    """
    import shutil
    exe = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    if not os.path.exists(exe):
        return {}
    with open(ptx_path) as f:
        hdr = [l.strip() for l in f.read().splitlines()[:8] if l.startswith(".target")]
    if not hdr:
        return {}
    sm = re.search(r"(sm_\d+[a-z]?)", " ".join(hdr))
    if not sm:
        return {}
    out = {}
    p = subprocess.run([exe, f"-arch={sm.group(1)}", "-v", ptx_path, "-o", os.devnull],
                       capture_output=True, text=True)
    txt = p.stdout + p.stderr
    for pat, key in ((r"Used (\d+) registers", "registers"),
                     (r"(\d+) bytes stack frame", "stack_frame_bytes"),
                     (r"(\d+) bytes spill stores", "spill_store_bytes"),
                     (r"(\d+) bytes spill loads", "spill_load_bytes"),
                     (r"(\d+) bytes smem", "static_smem_bytes"),
                     (r"used (\d+) barriers", "barriers"),
                     (r"(\d+) bytes cmem", "const_bytes")):
        m = re.search(pat, txt)
        if m:
            out[key] = int(m.group(1))
    out["ptxas_rc"] = p.returncode
    if p.returncode != 0:
        out["ptxas_stderr_tail"] = txt.strip().splitlines()[-1][:200] if txt.strip() else ""
    return out


def disasm(cubin_path):
    """cuobjdump -sass, or nvdisasm, whichever exists. Returns (text, tool)."""
    for exe in ("cuobjdump", "nvdisasm"):
        p = shutil.which(exe)
        if not p:
            continue
        if exe == "cuobjdump":
            cmd = [p, "-sass", cubin_path]
        else:
            cmd = [p, "-c", "-g", cubin_path]
        v = subprocess.run(cmd, capture_output=True, text=True)
        if v.returncode == 0 and v.stdout.strip():
            return v.stdout, exe
    return None, None


def _cc_of(label):
    m = re.search(r"cc(\d+)", label)
    return int(m.group(1)) if m else None


def count(cubin_path, label, keep_text=True):
    if _cc_of(label) is not None and _cc_of(label) not in FULL_RECEIPT_CCS:
        keep_text = False
    text, tool = disasm(cubin_path)
    if text is None:
        say(f"    {label:<44} NO DISASSEMBLER")
        return None
    total, hist_d, fam = parse_sass(text)
    hist = collections.Counter(hist_d)
    keep_text = True
    pt = cubin_path[:-6] + ".ptx"
    out = dict(label=label, tool=tool,
               ptxas_v=ptxas_v(pt) if os.path.exists(pt) else {},
               cubin=os.path.relpath(cubin_path, HERE),
               cubin_sha256=hashlib.sha256(open(cubin_path, "rb").read()).hexdigest(),
               total=total, families=fam,
               HMMA=_pfx(hist, "HMMA"),
               HGMMA=_pfx(hist, "HGMMA"),
               TCGMMA=_pfx(hist, "TCGMMA"),
               UMMA=_pfx(hist, "UMMA"),
               UTCHMMA=_pfx(hist, "UTCHMMA"),
               TCGEN05=_pfx(hist, "TCGEN05"),
               tc_mnemonic_table={k: v for k, v in
                                   sorted(hist.items(), key=lambda kv: -kv[1])
                                   if k.split(".")[0].upper() in TC_EXACT},
               MUFU_EX2=hist.get("MUFU.EX2", 0),
               FFMA=hist.get("FFMA", 0), FMUL=hist.get("FMUL", 0),
               FADD=hist.get("FADD", 0),
               LDS=sum(v for k, v in hist.items() if k.startswith("LDS")),
               STS=sum(v for k, v in hist.items() if k.startswith("STS")),
               LDL=sum(v for k, v in hist.items() if k.startswith("LDL")),
               STL=sum(v for k, v in hist.items() if k.startswith("STL")),
               LDG=sum(v for k, v in hist.items() if k.startswith("LDG")),
               STG=sum(v for k, v in hist.items() if k.startswith("STG")),
               top=[(k, v) for k, v in hist.most_common(24)])
    if keep_text:
        with open(os.path.join(SASS, f"{label}.sass.txt"), "w") as f:
            f.write(text)
        R["artifacts"].append(f"sass/{label}.sass.txt")
    return out


# Volume control.  A previous run wrote ~540 files into out/sass/ and the
# Kaggle output download truncated and had to be retried.  The rule in
# notes/06 is that a number must be re-derivable BY A SCRIPT; the script is
# this file and the card, not the archive.  So the full receipt is kept for the
# three architectures this study actually discusses, and the cubin alone is
# kept for the rest.
FULL_RECEIPT_CCS = (75, 80, 90)


def ptx_census(ptx):
    return dict(
        ptx_mma_sync=len(re.findall(r"\bmma\.sync\b", ptx)),
        ptx_mma_any=len(re.findall(r"\bmma\.", ptx)),
        ptx_wgmma=len(re.findall(r"\bwgmma\.", ptx)),
        ptx_tcgen=len(re.findall(r"\btcgen05\.", ptx)),
        ptx_fma_rn_f32=len(re.findall(r"\bfma\.rn\.f32\b", ptx)),
        ptx_cvt_tf32=len(re.findall(r"\bcvt\.[a-z0-9.]*tf32[a-z0-9.]*", ptx)),
        ptx_ex2=len(re.findall(r"\bex2\.[a-z0-9.]*", ptx)),
        ptx_ldmatrix=len(re.findall(r"\bldmatrix\b", ptx)),
        ptx_cp_async=len(re.findall(r"\bcp\.async\b", ptx)),
        ptx_header=[l.strip() for l in ptx.splitlines()[:8]
                    if l.startswith((".version", ".target"))],
    )


def show(row, name):
    if not row:
        return
    say("")
    say(f"  {name}")
    say(f"    SASS total          {row['total']}")
    say(f"    tensor core         {row['families'].get('tensor_core', 0)}"
        f"   (HMMA {row['HMMA']}, HGMMA {row['HGMMA']}, TCGMMA {row['TCGMMA']},"
        f" UMMA {row['UMMA']})")
    say(f"    MUFU.EX2            {row['MUFU_EX2']}")
    say(f"    FFMA / FMUL / FADD  {row['FFMA']} / {row['FMUL']} / {row['FADD']}")
    say(f"    LDS / STS           {row['LDS']} / {row['STS']}")
    say(f"    LDL / STL (spill)   {row['LDL']} / {row['STL']}")
    say(f"    LDG / STG           {row['LDG']} / {row['STG']}")
    say(f"    top opcodes         {', '.join(f'{k}={v}' for k, v in row['top'][:12])}")


# =========================================================== (A) the real cubin
def real_artifacts():
    hr("A. THE REAL ARTIFACT -- cubins fla compiled on THIS card")
    import torch
    import triton

    say("  fusing nothing, mocking nothing: call the op, then take the objects")
    say("  triton actually produced.")

    from fla.ops.kda import chunk_kda
    import world
    B, T, H, HV, K, V = 1, 256, 2, 2, 128, 128
    dt = torch.bfloat16
    q, k, v, gg, beta = world.make(B, T, H, HV, K, V, seed=0)
    tq, tk, tv = (world.to_torch(x, "bfloat16") for x in (q, k, v))
    tg = world.to_torch(gg, "float32")
    tb = world.to_torch(beta, "bfloat16")
    with torch.inference_mode():
        o, _ = chunk_kda(tq, tk, tv, tg, tb, scale=world.scale_for(K),
                         use_qk_l2norm_in_kernel=False)
    torch.cuda.synchronize()
    say(f"  ran chunk_kda B={B} T={T} H={H} K={K} V={V} bf16; "
        f"finite={bool(torch.isfinite(o).all())}   "
        f"(inputs from the study's own generator, kda/kda_vs_fla.py:296)")
    R["real_call"] = dict(B=B, T=T, H=H, HV=HV, K=K, V=V, dtype="bfloat16",
                          generator="kda/kda_vs_fla.py:296 rnd(seed=0)")

    # Every kernel the run compiled, keyed by the triton cache directory triton
    # just wrote.  The cache dir is on the card and is real; this is the artifact.
    cache_dirs = []
    for attr in ("TRITON_CACHE_DIR",):
        if os.environ.get(attr):
            cache_dirs.append(os.environ[attr])
    cache_dirs.append(os.path.join(os.path.expanduser("~"), ".triton", "cache"))
    seen, rows, per_name = set(), [], {}
    say("")
    say("  walking triton's on-card kernel cache for the cubins just written:")
    for cd in cache_dirs:
        if not (cd and os.path.isdir(cd)):
            continue
        for dp, _dn, fn in os.walk(cd):
            for f in fn:
                if not f.endswith(".cubin"):
                    continue
                p = os.path.join(dp, f)
                mt = os.path.getmtime(p)
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                cache_dirs_hit = 0
                # triton's cache layout: <dir>/<kernel_name>/<hash>/<kernel>.cubin
                if mt < START_MT - 600:
                    continue      # not from this run
                # triton's cache layout is <cache>/<hash>/<kernel>.cubin plus a
                # <kernel>.json next to it holding {"name": "<kernel>"}.  The
                # directory two levels up is the HASH, not the kernel name, so
                # a name derived from the path prints "cache" for every kernel.
                # Read the name out of the metadata json instead; fall back to
                # the cubin stem, which triton also uses for the filename.
                name = os.path.splitext(f)[0]
                meta = p[:-6] + ".json"
                if os.path.exists(meta):
                    try:
                        j = json.load(open(meta))
                        for key in ("name", "kernel_name"):
                            if isinstance(j, dict) and j.get(key):
                                name = str(j[key])
                                break
                    except BaseException:
                        pass
                rows.append((name, p, mt, h))
    rows.sort(key=lambda r: r[2])
    say(f"  {len(rows)} cubins newer than this run started")
    per_kernel = {}
    for name, p, mt, h in rows:
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:90]
        tgt = os.path.join(SASS, f"real_{stem}.cubin")
        shutil.copyfile(p, tgt)
        R["artifacts"].append(f"sass/real_{stem}.cubin")
        # the ptx sits next to the cubin, in triton's cache
        ptxp = p[:-6] + ".ptx"
        pc = None
        if os.path.exists(ptxp):
            ptx = open(ptxp, errors="replace").read()
            shutil.copyfile(ptxp, os.path.join(SASS, f"real_{stem}.ptx"))
            R["artifacts"].append(f"sass/real_{stem}.ptx")
            pc = ptx_census(ptx)
        row = count(tgt, f"real_{stem}")
        if row is None:
            continue
        row["ptx"] = pc
        per_kernel[stem] = row
        per_name.setdefault(name, []).append(row)
        say(f"    {name[:64]:<64} {row['total']:>6} SASS  "
            f"TC={row['families'].get('tensor_core', 0):<4} "
            f"EX2={row['MUFU_EX2']:<4} "
            f"HMMA={row['HMMA']} HGMMA={row['HGMMA']} "
            f"spill={row['LDL']}/{row['STL']}")
    # Summarise by kernel NAME as well.  The per-cubin dict is keyed by the
    # artefact stem, which is unique; keying it by name would have every
    # variant overwrite the last one, which is what the first version did --
    # 45 cubins collapsed to 8 entries and the aggregate tensor-core count in
    # the JSON became meaningless.  The text log kept all 45 rows, which is
    # how the discrepancy was caught.
    import collections as _c
    agg = {}
    for name, lst in per_name.items():
        agg[name] = dict(
            n_cubins=len(lst),
            total_sass=sum(x["total"] for x in lst),
            max_sass=max(x["total"] for x in lst),
            tensor_core=sum(x["families"].get("tensor_core", 0) for x in lst),
            HMMA=sum(x["HMMA"] for x in lst), HGMMA=sum(x["HGMMA"] for x in lst),
            MUFU_EX2_max=max(x["MUFU_EX2"] for x in lst),
            FFMA=sum(x["FFMA"] for x in lst),
            max_spill_ld=max(x["LDL"] for x in lst),
            max_spill_st=max(x["STL"] for x in lst))
    fla_names = [n for n in agg if "dot" not in n]
    R["real_kernels"] = per_kernel
    R["real_kernels_by_name"] = agg
    R["n_cubins_total"] = len(per_kernel)
    R["n_cubins_fla"] = sum(agg[n]["n_cubins"] for n in fla_names)
    R["n_cubins_probe"] = sum(agg[n]["n_cubins"] for n in agg if n not in fla_names)
    R["tensor_core_fla_cubins_total"] = sum(agg[n]["tensor_core"] for n in fla_names)
    R["tensor_core_probe_cubins_total"] = sum(
        agg[n]["tensor_core"] for n in agg if n not in fla_names)
    say("")
    say("  per kernel name, summed over every cubin of that kernel:")
    for n in sorted(agg, key=lambda n: -agg[n]["total_sass"]):
        a = agg[n]
        tag = "control" if n not in fla_names else ""
        say(f"    {n[:56]:<56} n={a['n_cubins']:<3} total SASS {a['total_sass']:>7}"
            f"  TC={a['tensor_core']:<4} FFMA={a['FFMA']:<6}"
            f" maxEX2={a['MUFU_EX2_max']:<4}"
            f" max spill ld/st {a['max_spill_ld']}/{a['max_spill_st']}  {tag}")
    say("")
    say(f"  fla's own cubins: {R['n_cubins_fla']} across {len(fla_names)} kernels.")
    say(f"  TENSOR-CORE INSTRUCTIONS across all of them: "
        f"{R['tensor_core_fla_cubins_total']}")
    say(f"  (the {R['n_cubins_probe']} control cubins contribute "
        f"{R['tensor_core_probe_cubins_total']}, which is the control working:")
    say(f"   the same tl.dot DOES reach a tensor core when the knob says tf32.)")
    return per_kernel


START_MT = 0.0


# ============================================================== (B) the AOT sweep
def aot_sweep():
    hr("B. THE AOT SWEEP -- same source, same triton, five architectures")
    say("  A Triton compiler does not need the hardware it targets, so the")
    say("  other cards in this study's card table can be COMPILED here, on a T4,")
    say("  by passing an explicit GPUTarget.  What comes out is a statement about")
    say("  a compilation at that arch.  It is not a timing, and nothing in this")
    say("  section may be read as one.")

    import triton
    from triton.backends.compiler import GPUTarget

    say(f"  triton {triton.__version__}")

    # the kernel source: the INSTALLED fla kernel, with the autotune + heuristics
    # decorators stripped, because autotune benchmarks candidates and this is a
    # T4 with a different config space.  Everything between the decorators and
    # the def line is removed; the kernel body is untouched.
    import fla.ops.common.chunk_delta_h as CDH
    src_path = os.path.abspath(CDH.__file__)
    src = open(src_path).read()
    say(f"  source: {src_path}")
    say(f"  sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}...  "
        f"{len(src.splitlines())} lines")
    R["aot_source"] = dict(path=src_path, sha256=hashlib.sha256(src.encode()).hexdigest(),
                           lines=len(src.splitlines()))

    # Take EXACTLY this one function definition, by AST segment rather than by
    # slicing to end-of-file.  A previous version sliced from `def ...` to EOF
    # and that pulled in every LATER kernel in the same module together with its
    # own @fla_cache_autotune(...) decorator, so the stripped copy failed to
    # import with `NameError: name 'fla_cache_autotune' is not defined` -- an
    # error that reads like a source problem and is an extraction problem.
    # ast.get_source_segment returns the function's own bytes and nothing else.
    import ast as _ast
    tree = _ast.parse(src)
    fnode = None
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and \
                node.name == "chunk_gated_delta_rule_fwd_kernel_h_blockdim64":
            fnode = node
            break
    if fnode is None:
        say("  the kernel is not a top-level FunctionDef; cannot extract it")
        return []
    keep = _ast.get_source_segment(src, fnode)
    if keep is None:
        say("  ast.get_source_segment returned None; cannot extract the kernel")
        return []
    hdr = ("import triton\n"
           "import triton.language as tl\n"
           "\n\n"
           "@triton.jit\n"
           "def exp2(x): return tl.math.exp2(x.to(tl.float32))\n"
           "\n\n"
           "@triton.jit(do_not_specialize=['T'])\n")
    kernel_src = hdr + keep + "\n"
    scratch = os.path.join(SASS, "aot_kernel.py")
    with open(scratch, "w") as f:
        f.write(kernel_src)
    R["artifacts"].append("sass/aot_kernel.py")
    say("  stripped copy written to %s" % os.path.relpath(scratch, HERE))
    say("  extracted by ast.get_source_segment, %d lines, sha256 %s..."
        % (len(keep.splitlines()), hashlib.sha256(keep.encode()).hexdigest()[:16]))
    say("  the segment is byte-identical to the file's own text: %s" % (keep in src))
    R["aot_body_sha256"] = hashlib.sha256(keep.encode()).hexdigest()
    say("  removed: @triton.heuristics({USE_G: ..., USE_GK: ..., ...}) and")
    say("           @fla_cache_autotune(configs=[...], key=[...])")
    say("  the second is why the config list is passed as explicit constexprs")
    say("  below: an autotuner benchmarks its candidates, which needs a device,")
    say("  and it would benchmark them against THIS card's gate anyway.")

    # exp2 lives in fla.ops.utils.op; the header above provides the default
    # (FLA_USE_FAST_OPS=0) one inline so the stripped module has no fla imports.

    import importlib.util
    spec = importlib.util.spec_from_file_location("aot_kernel", scratch)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException as e:
        say(f"  could not import the stripped kernel: {type(e).__name__}: {str(e)[:150]}")
        R["aot_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return {}
    fn = getattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", None)
    if fn is None:
        say("  the stripped kernel has no entry point")
        return []

    sig = {
        "k": "*bf16", "v": "*bf16", "w": "*bf16", "v_new": "*bf16",
        "g": "*fp32", "gk": "*fp32", "h": "*fp32", "h0": "*fp32", "ht": "*fp32",
        "cu_seqlens": "*i32", "chunk_offsets": "*i32", "T": "i32",
        "H": "constexpr", "HV": "constexpr", "K": "constexpr", "V": "constexpr",
        "BT": "constexpr", "BV": "constexpr", "USE_G": "constexpr",
        "USE_GK": "constexpr", "USE_INITIAL_STATE": "constexpr",
        "STORE_FINAL_STATE": "constexpr", "SAVE_NEW_VALUE": "constexpr",
        "STATE_V_FIRST": "constexpr", "IS_VARLEN": "constexpr",
    }
    base_ce = dict(H=8, HV=8, K=128, V=128, BT=64, BV=64, USE_G=True, USE_GK=False,
                   USE_INITIAL_STATE=False, STORE_FINAL_STATE=False,
                   SAVE_NEW_VALUE=False, STATE_V_FIRST=False, IS_VARLEN=False)

    # (config, label) pairs.  `notes16` reproduces the config notes/16 section
    # 3.3 used, so that row can be checked against that table rather than
    # assumed; the rest are what each arch's own gate permits.
    smem_by_cc = {75: 65536, 80: 166912, 86: 166912, 89: 101376, 90: 232448,
                  100: 232448}
    AMPERE, ADA, DEFAULT = 166912, 101376, 102400

    def gate_config(cc):
        sm = smem_by_cc[cc]
        stages = [2, 3, 4] if sm >= AMPERE else [2, 1]
        bvs = [32, 64] if sm >= ADA else [32]
        warps = [2, 4]
        out = []
        for w in warps:
            for s in stages:
                for bv in bvs:
                    out.append((bv, w, s))
        return out

    say("")
    say("  cc   smem(optin)  gated config space        (BV, warps, stages)")
    for cc in (75, 80, 86, 89, 90, 100):
        g = gate_config(cc)
        say(f"  {cc:<4} {smem_by_cc[cc]:>10,}   {len(g):>3} configs        "
            f"BV={sorted({x[0] for x in g})} stages={sorted({x[2] for x in g})} "
            f"warps={sorted({x[1] for x in g})}")
    R["aot_gate_model"] = {str(cc): gate_config(cc) for cc in (75, 80, 86, 89, 90, 100)}

    rows = []
    for cc in (75, 80, 86, 89, 90, 100):
        for (bv, w, s) in gate_config(cc):
            label = f"aot_cc{cc}_BV{bv}_w{w}_s{s}"
            ce = dict(base_ce, BV=bv)
            try:
                src_obj = triton.compiler.ASTSource(fn=fn, signature=dict(sig),
                                                     constexprs=ce)
                tgt = GPUTarget("cuda", cc, 32)
                k = triton.compile(src_obj, target=tgt,
                                   options={"num_warps": w, "num_stages": s})
            except BaseException as e:
                say(f"  cc={cc} BV={bv} w={w} s={s}   COMPILE FAILED "
                    f"{type(e).__name__}: {str(e)[:110]}")
                R.setdefault("aot_failures", []).append(
                    dict(cc=cc, BV=bv, warps=w, stages=s,
                         err=f"{type(e).__name__}: {str(e)[:200]}"))
                continue
            cub = os.path.join(SASS, f"{label}.cubin")
            with open(cub, "wb") as f:
                f.write(k.asm["cubin"])
            ptx = k.asm.get("ptx", "")
            with open(os.path.join(SASS, f"{label}.ptx"), "w") as f:
                f.write(ptx)
            row = count(cub, label)
            if row is None:
                continue
            row.update(cc=cc, BV=bv, warps=w, stages=s,
                       n_regs=getattr(k, "n_regs", None),
                       n_spills=getattr(k, "n_spills", None),
                       shared=getattr(k, "metadata", None) and
                       getattr(k.metadata, "shared", None),
                       ptx=ptx_census(ptx))
            rows.append(row)
            r = row
            v = r.get("ptxas_v", {})
            say(f"  cc={cc:<4} BV={bv:<3} w={w} s={s}  SASS {r['total']:>5}  "
                f"TC={r['families'].get('tensor_core', 0):<4} "
                f"(HMMA {r['HMMA']}, HGMMA {r['HGMMA']}, UMMA {r['UMMA']})  "
                f"EX2={r['MUFU_EX2']:<3} FFMA={r['FFMA']:<5} "
                f"REG={v.get('registers')} "
                f"spill st/ld {v.get('spill_store_bytes')}/{v.get('spill_load_bytes')} B"
                f"  smem={v.get('static_smem_bytes')}")

    # the notes/16 configuration, for replication
    say("")
    say("  replication row: the config notes/16 section 3.3 used "
        "(BV=64, warps=4, stages=2), on every arch")
    for cc in (75, 80, 90, 100):
        ce = dict(base_ce, BV=64)
        label = f"aot16_cc{cc}_BV64_w4_s2"
        try:
            src_obj = triton.compiler.ASTSource(fn=fn, signature=dict(sig), constexprs=ce)
            k = triton.compile(src_obj, target=GPUTarget("cuda", cc, 32),
                               options={"num_warps": 4, "num_stages": 2})
        except BaseException as e:
            say(f"    cc={cc}: FAILED {type(e).__name__}: {str(e)[:110]}")
            R.setdefault("aot16_failures", []).append(
                dict(cc=cc, err=f"{type(e).__name__}: {str(e)[:200]}"))
            continue
        cub = os.path.join(SASS, f"{label}.cubin")
        with open(cub, "wb") as f:
            f.write(k.asm["cubin"])
        ptx = k.asm.get("ptx", "")
        with open(os.path.join(SASS, f"{label}.ptx"), "w") as f:
            f.write(ptx)
        row = count(cub, label)
        if row is None:
            continue
        row.update(cc=cc, BV=64, warps=4, stages=2,
                   n_regs=getattr(k, "n_regs", None),
                   n_spills=getattr(k, "n_spills", None),
                   ptx=ptx_census(ptx))
        rows.append(row)
        r = row
        say(f"    cc={cc:<4} SASS {r['total']:>5}  "
            f"TC={r['families'].get('tensor_core', 0):<4} "
            f"(HMMA {r['HMMA']}, HGMMA {r['HGMMA']}, UMMA {r['UMMA']})  "
            f"EX2={r['MUFU_EX2']:<3}  "
            f"ptx target {r['ptx']['ptx_header']}")
    R["aot_rows"] = rows
    return rows


def per_arch_table(rows):
    hr("C. per-arch summary, from the gated config space (B only)")
    if not rows:
        say("  no rows")
        return
    say("  cc | config (BV/warps/stages) | SASS | TC | HMMA | HGMMA | UMMA |"
        " FFMA | MUFU.EX2 | REG | spill st/ld B | smem B | other-TC mnemonics")
    for r in sorted(rows, key=lambda r: (r.get("cc", 0), r.get("BV", 0),
                                         r.get("warps", 0), r.get("stages", 0))):
        v = r.get("ptxas_v", {})
        oth = [k for k, n in r.items()
               if n and k.split(".")[0].upper() in
               ("IMMA", "OMMA", "BMMA", "QMMA", "TCGMMA", "UMMA")
               and k not in ("HMMA", "HGMMA", "TCGMMA", "UMMA")]
        say(f"  {r.get('cc'):<3}| {r.get('BV')}/{r.get('warps')}/{r.get('stages')}"
            f"{'':<9}| {r['total']:>5} | "
            f"{r['families'].get('tensor_core', 0):>2} | {r['HMMA']:>4} | "
            f"{r['HGMMA']:>5} | {r['UMMA']:>4} | {r['FFMA']:>5} | "
            f"{r['MUFU_EX2']:>9} | {str(v.get('registers')):>3} | "
            f"{v.get('spill_store_bytes')}/{v.get('spill_load_bytes')} | "
            f"{str(v.get('static_smem_bytes')):>5} | {','.join(oth) or '-'}")
    say("")
    say("  READ THIS ROW, NOT THE TABLE, AT sm_75:")
    at75 = [r for r in rows if r.get("cc") == 75]
    if at75:
        tc = sum(r["families"].get("tensor_core", 0) for r in at75)
        say(f"    every sm_75 configuration in the gated space has "
            f"{tc} tensor-core instructions in total across all "
            f"{len(at75)} cubins.")
        say("    Not 'few'.  None.")
    else:
        say("    no sm_75 rows were produced; that is itself the finding and it")
        say("    is in aot_failures above.")


def main():
    global START_MT
    import time
    START_MT = time.time()
    hr("kda_t4 SASS -- instruction mix of the reference, counted from cubins")
    try:
        real_artifacts()
    except BaseException as e:
        import traceback
        say(f"  !! (A) failed: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(file=sys.stdout)
        R.setdefault("failures", []).append(f"A: {type(e).__name__}: {str(e)[:200]}")
    try:
        rows = aot_sweep()
        per_arch_table(rows)
    except BaseException as e:
        import traceback
        say(f"  !! (B) failed: {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc(file=sys.stdout)
        R.setdefault("failures", []).append(f"B: {type(e).__name__}: {str(e)[:200]}")

    with open(os.path.join(OUT, "sass_t4.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    with open(os.path.join(OUT, "sass_t4.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    say("")
    say(f"wrote kda_t4/out/sass_t4.{{txt,json}} and {len(R['artifacts'])} "
        f"artifacts under kda_t4/out/sass/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
