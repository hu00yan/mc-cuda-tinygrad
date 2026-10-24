#!/usr/bin/env python3
"""
probe_env.py -- what box am I on, and does the fla source I am about to reason
about actually say what this repo says it says?

Two jobs, both cheap, both done before anything is timed.

1. THE BOX.  Every version, both devices, and the disassembly toolchain. The
   versions are not decoration: fla reads the triton version
   (`IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')`,
   fla/utils/_device.py:138) and warns below 3.3.0, so a triton change is a
   codegen change.

2. THE SOURCE CHECK.  `kda/baseline_pin.py` pins `kda/baseline_pin.py:28-30`
   against fla-core 0.5.2 and cites specific line numbers as the basis for the
   scope label on every fla number in this study. Those citations are only worth
   anything if they are true of the *installed* wheel, so this reads the
   installed file and checks that the cited line still contains the cited
   token. A line that has moved is reported as a FAIL, not silently accepted --
   a scope label built on a moved line is a scope label about nothing.

   This exists because the whole point of `kda_t4` is that an earlier note's
   claim ("fla needs sm_80+, comparison impossible") was an inference from a
   preflight print, and it was wrong. The cheapest defence against the next
   such error is to make the source say the thing out loud, on the card.

Run:  python kda_t4/probe_env.py            (writes kda_t4/out/probe_env.{txt,json})
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

R = {}          # the machine-readable record


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


LOG: list = []


def hr(t=""):
    say("")
    say("=" * 76)
    if t:
        say(t)
        say("=" * 76)


def try_(label, fn, default=None):
    try:
        return fn()
    except BaseException as e:                                   # noqa: BLE001
        say(f"  {label:<34} FAILED {type(e).__name__}: {str(e)[:90]}")
        R.setdefault("failures", []).append(f"{label}: {type(e).__name__}: {str(e)[:120]}")
        return default


# --------------------------------------------------------------------- box
def box():
    hr("1. the box")
    import importlib
    import importlib.metadata as md

    vers = {}
    for name, dist in (("torch", "torch"), ("triton", "triton"), ("numpy", "numpy"),
                       ("einops", "einops"), ("tinygrad", "tinygrad"),
                       ("fla-core", "fla-core"),
                       ("flash-linear-attention", "flash-linear-attention")):
        try:
            vers[dist] = md.version(dist)
        except BaseException:
            vers[dist] = None
    for dist, v in vers.items():
        say(f"  {dist:<28}{v}")
    R["versions"] = vers

    say("")
    for mod in ("fla", "flash_linear_attention", "tinygrad", "triton"):
        p = try_(f"import {mod}", lambda m=mod: importlib.import_module(m).__file__, "")
        say(f"  {mod:<28}{p}")
        R.setdefault("module_paths", {})[mod] = p

    # what fla actually thinks the triton requirement is, from the installed
    # metadata rather than from my memory of the wheel
    try:
        req = md.requires("fla-core") or []
        R["fla_core_requires"] = req
        say("")
        say("  fla-core 0.5.2 requires_dist, verbatim from the installed metadata:")
        for r in req:
            say(f"    {r}")
    except BaseException as e:
        say(f"  requires_dist unreadable: {type(e).__name__}")

    # the disassembly toolchain.  The SASS half of this study is worthless
    # without it, and finding that out now costs seconds instead of a session.
    say("")
    say("  disassembly toolchain (SASS counting needs these):")
    tc = {}
    for exe in ("nvcc", "ptxas", "cuobjdump", "nvdisasm"):
        p = try_(f"which {exe}", lambda e=exe: __import__("shutil").which(e), None)
        tc[exe] = p
        say(f"    {exe:<12}{p or 'NOT ON PATH'}")
    for d in ("/usr/local/cuda/bin", "/usr/local/cuda-12.8/bin",
              "/usr/local/cuda-12.9/bin", "/usr/local/cuda-13.0/bin"):
        if os.path.isdir(d):
            say(f"    (dir {d}: {sorted(x for x in os.listdir(d) if x in ('nvcc','ptxas','cuobjdump','nvdisasm'))})")
    if tc.get("cuobjdump") is None:
        # torch ships its own ptxas; find it, and say where
        for base in ("/usr/local/lib/python3.12/dist-packages/nvidia",
                     "/usr/local/lib/python3.11/dist-packages/nvidia"):
            if os.path.isdir(base):
                hits = []
                for dp, _dn, fn in os.walk(base):
                    for f in fn:
                        if f in ("ptxas", "nvdisasm", "cuobjdump"):
                            hits.append(os.path.join(dp, f))
                if hits:
                    say(f"    under {base}: {sorted(hits)[:8]}")
    R["toolchain"] = tc

    say("")
    ver = try_("nvcc --version",
               lambda: subprocess.run(["nvcc", "--version"], capture_output=True,
                                      text=True).stdout.strip().splitlines()[-1], "")
    say(f"  nvcc version line              {ver}")
    R["nvcc_version"] = ver
    return vers


# ----------------------------------------------------------------- devices
def devices():
    hr("2. devices (fla sizes its config space from DEVICE 0, always)")
    import torch
    import triton

    ds = []
    n = torch.cuda.device_count()
    say(f"  torch.cuda.device_count() = {n}")
    for i in range(n):
        d = {}
        cap = torch.cuda.get_device_capability(i)
        d["index"] = i
        d["name"] = torch.cuda.get_device_name(i)
        d["cc"] = list(cap)
        d["sm"] = f"sm_{cap[0]}{cap[1]}"
        p = torch.cuda.get_device_properties(i)
        d["total_memory"] = int(p.total_memory)
        d["multi_processor_count"] = int(p.multi_processor_count)
        d["shared_memory_per_block"] = int(getattr(p, "shared_memory_per_block", -1))
        d["shared_memory_per_block_optin"] = int(getattr(p, "shared_memory_per_block_optin", -1))
        d["shared_memory_per_multiprocessor"] = int(getattr(p, "shared_memory_per_multiprocessor", -1))
        # triton's own read -- the one fla actually compares against
        try:
            tp = triton.runtime.driver.active.utils.get_device_properties(i)
            d["triton_max_shared_mem"] = int(tp["max_shared_mem"])
            d["triton_props_keys"] = sorted(tp.keys())
        except BaseException as e:
            d["triton_max_shared_mem"] = None
            say(f"  triton props[{i}] failed: {type(e).__name__}: {str(e)[:70]}")
        ds.append(d)
        say("")
        say(f"  device {i}: {d['name']}  {d['sm']}  {d['total_memory']/2**30:.2f} GiB  "
            f"{d['multi_processor_count']} SM")
        say(f"    shared_memory_per_block            {d['shared_memory_per_block']}")
        say(f"    shared_memory_per_block_optin      {d['shared_memory_per_block_optin']}")
        say(f"    shared_memory_per_multiprocessor   {d['shared_memory_per_multiprocessor']}")
        say(f"    triton max_shared_mem              {d['triton_max_shared_mem']}")
    R["devices"] = ds
    R["heterogeneous"] = len({tuple(d["cc"]) for d in ds}) > 1 or \
        len({d["shared_memory_per_block_optin"] for d in ds}) > 1
    say("")
    say(f"  heterogeneous box: {R['heterogeneous']}")
    if len(ds) > 1:
        say("  NOTE fla reads device 0 for every architecture predicate, so the")
        say("       config space is device 0's even for a kernel on device 1.")
    return ds


# ------------------------------------------- the citations, re-read on the card
# (file, line, substring that must be on that line, what the citation is for)
CITATIONS = [
    ("fla/utils/_device.py", 137, "IS_TF32_SUPPORTED", "cc major >= 8 => no tf32 below sm_80"),
    ("fla/utils/_device.py", 138, "IS_GATHER_SUPPORTED", "triton-version switch"),
    ("fla/utils/_device.py", 152, "TRITON_F32_DEFAULT", "process-global ieee write"),
    ("fla/utils/_device.py", 170, "get_all_max_shared_mem", "triton max_shared_mem read"),
    ("fla/utils/_device.py", 181, "ADA", "Backend enum start"),
    ("fla/utils/_device.py", 182, "166912", "AMPERE threshold"),
    ("fla/utils/_device.py", 183, "232448", "HOPPER threshold"),
    ("fla/utils/_device.py", 184, "102400", "DEFAULT threshold"),
    ("fla/utils/_device.py", 196, "check_shared_mem", "the gate itself"),
    ("fla/ops/common/chunk_delta_h.py", 28, "GATED_DELTA_RULE_FWD_H_NUM_WARPS",
     "num_warps list (Blackwell pinned to [2])"),
    ("fla/ops/common/chunk_delta_h.py", 43, "[2, 3, 4]", "num_stages narrowing"),
    ("fla/ops/common/chunk_delta_h.py", 44, "[32, 64]", "BV narrowing"),
    ("fla/ops/common/chunk_delta_h.py", 50,
     "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "the KDA fwd state kernel"),
    ("fla/ops/kda/chunk_bwd.py", 25, "BK_LIST", "bwd BK narrowing"),
    ("fla/ops/kda/chunk_bwd.py", 26, "BV_LIST", "bwd BV narrowing"),
    ("fla/ops/kda/chunk.py", 176, "@dispatch('kda')", "the dispatch wrapper"),
    ("fla/ops/kda/backends/__init__.py", 16, "TritonAscendKDABackend", "registry order 0"),
    ("fla/ops/kda/backends/__init__.py", 17, "FlashKDABackend", "registry order 1"),
    ("fla/ops/kda/backends/__init__.py", 18, "KDATileLangBackend", "registry order 2"),
    ("fla/ops/kda/chunk_intra.py", 20, "SOLVE_TRIL_DOT_PRECISION", "the explicit dot precision"),
    ("fla/ops/kda/backends/flash_kda.py", 25, "Copyright",
     "the CUTLASS backend, if it is ever selected"),
]


def source_check():
    hr("3. the citations, re-read in the installed wheel")
    try:
        import fla
        root = os.path.dirname(os.path.abspath(fla.__file__))
    except BaseException as e:
        say(f"  fla not importable ({type(e).__name__}: {str(e)[:70]}) -- cannot check")
        R["citations"] = {"fla_importable": False}
        return
    say(f"  fla package root: {root}")
    rows = []
    for rel, line, needle, why in CITATIONS:
        p = os.path.join(root, rel[len("fla/"):] if rel.startswith("fla/") else rel)
        try:
            with open(p, "r", errors="replace") as f:
                lines = f.read().splitlines()
        except BaseException as e:
            rows.append(dict(file=rel, line=line, ok=False,
                             detail=f"unreadable: {type(e).__name__}"))
            say(f"  FAIL  {rel}:{line}  unreadable ({type(e).__name__})")
            continue
        # a citation is allowed to be off by a little; say how far
        hit = None
        for cand in (line, line + 1, line - 1, line + 2, line - 2):
            if 1 <= cand <= len(lines) and needle in lines[cand - 1]:
                hit = cand
                break
        ok = hit is not None
        rows.append(dict(file=rel, cited=line, found=hit, ok=ok, needle=needle, why=why))
        tag = "  ok " if ok else "  FAIL"
        drift = "" if hit == line else f"  <-- actually on line {hit}"
        say(f"  {tag} {rel}:{line}  {needle!r}{drift}")
        say(f"          {why}")
    bad = [r for r in rows if not r["ok"]]
    say("")
    say(f"  {len(rows) - len(bad)}/{len(rows)} citations hold on the installed wheel")
    R["citations"] = {"rows": rows, "n_ok": len(rows) - len(bad), "n": len(rows),
                      "failures": bad}


def main():
    hr("kda_t4 PROBE  (no measurement here; this is the box and the source)")
    try:
        box()
    except BaseException:
        traceback.print_exc()
        R["failures"] = R.get("failures", []) + ["box() raised"]
    try:
        devices()
    except BaseException:
        traceback.print_exc()
        R["failures"] = R.get("failures", []) + ["devices() raised"]
    try:
        source_check()
    except BaseException:
        traceback.print_exc()
        R["failures"] = R.get("failures", []) + ["source_check() raised"]

    with open(os.path.join(OUT, "probe_env.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    with open(os.path.join(OUT, "probe_env.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    say("")
    say(f"wrote {OUT}/probe_env.json and {OUT}/probe_env.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
