#!/usr/bin/env python3
"""lm38_preflight.py -- everything about this experiment that can be checked
without a GPU, checked before a T4 session is spent on it.

WHAT IS HERE
------------
1. **Correctness of the arm-C bodies on the CPU.** They must produce the same
   observation and the same per-pixel DDA step counts as the rung they perturb,
   on the CPU, over a real world with a wide camera spread. A body that is not
   the same computation is not a same-graph-size-different-frame case; it is a
   different program.
2. **The generated CUDA of every arm, captured on the NULL device**, with the
   sha256, the `__launch_bounds__`, the statement count and the UOp count. This
   is where "graph size is equal" is *checked* rather than asserted: the arm-C
   bodies are supposed to be operation-neutral and the register-budget sweep is
   supposed to leave the source untouched, and both claims are visible here as
   numbers before any GPU time is spent.
3. **The census instruments themselves**, invoked once, so a broken import is a
   pre-flight failure and not a failure 1400 s into a Kaggle session.

Runs on this host, CPU only, no CUDA, no network.

    .venv/bin/python minecu_tg_opt/lm38_preflight.py --out .agent-scratch/lm38/preflight
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "archlab"))


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
def stage1_correctness(a) -> dict:
    """Arm C vs `flat_gather` on the CPU: same bytes, same step counts."""
    os.environ["DEV"] = "CPU"
    import numpy as np
    from tinygrad import Tensor, dtypes
    from minecu_tg import generate_world
    from minecu_tg.render import camera_basis
    from minecu_tg_opt.render import render_variant
    from minecu_tg_opt.lm38 import ARM_C, render_arm_c

    out = {"rows": [], "checks": 0, "fails": []}
    # max_steps is REDUCED here on purpose. The CPU backend compiles the
    # fused C for every inlined step, so a 32-step graph is minutes of clang
    # per body; the property being checked -- that the body computes the same
    # observation and the same step counts as `flat_gather` -- does not depend
    # on the unroll count. The 32-step equality is established on the card, over
    # 1 048 576 pixels, against the hand-written driver's own bytes.
    cpu_steps = min(8, a.steps)
    for world, ground, batch, yaw in ((16, 4, 3, 1.2), (24, 6, 2, 0.05)):
        one = generate_world(world, ground).numpy()
        vox = Tensor(np.ascontiguousarray(np.stack([one] * batch)), dtype=dtypes.int8)
        cams = np.array([[world / 2.0, float(ground), world / 2.0,
                          float(yaw) * (1.0 if i % 2 == 0 else -1.0), 0.0]
                         for i in range(batch)], dtype=np.float32)
        basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, a.res, a.res)
        say(f"\n  world {world} ground {ground} batch {batch} yaw +-{yaw} "
            f"res {a.res}x{a.res} max_steps {cpu_steps} (CPU gate)")
        ref_o, ref_s = render_variant(vox, basis, a.res, a.res, cpu_steps,
                                      "flat_gather", count_steps=True)
        ref_o, ref_s = ref_o.numpy(), ref_s.numpy()
        say(f"    {'flat_gather (the rung perturbed)':<14} "
            f"obs sha {hashlib.sha256(np.ascontiguousarray(ref_o)).hexdigest()[:16]}"
            f"  DDA mean {float(ref_s.mean()):.4f}")
        for v in sorted(ARM_C):
            o, s = render_arm_c(vox, basis, a.res, a.res, cpu_steps, v,
                                count_steps=True)
            o, s = o.numpy(), s.numpy()
            same_o = bool(np.array_equal(o, ref_o))
            same_s = bool(np.array_equal(s, ref_s))
            out["checks"] += 2
            if not same_o:
                out["fails"].append(f"{v} obs differs from flat_gather "
                                    f"({int((o != ref_o).sum())} values)")
            if not same_s:
                out["fails"].append(f"{v} DDA step counts differ "
                                    f"(max |delta| {int(np.abs(s.astype(int) - ref_s.astype(int)).max())})")
            out["rows"].append({"variant": v, "world": world, "batch": batch,
                                "obs_identical": same_o, "steps_identical": same_s,
                                "max_steps": cpu_steps, "dda_mean": float(s.mean()),
                                "obs_sha256": hashlib.sha256(np.ascontiguousarray(o)).hexdigest()})
            say(f"    {v:<14} obs {'IDENTICAL' if same_o else '*** DIFFERS ***'}"
                f"   DDA {'IDENTICAL' if same_s else '*** DIFFERS ***'} "
                f"mean {float(s.mean()):.4f}")
            del o, s
            gc.collect()
    out["pass"] = not out["fails"]
    return out


# ---------------------------------------------------------------------------
def stage2_codegen(a) -> dict:
    """The generated CUDA of every arm, on the NULL device, with UOp counts.

    `DEV=NULL:CUDA:sm_75` runs the real scheduler, the real fusion decision and
    the real C renderer with nothing executed, so this is the string the CUDA
    backend compiles on the card -- not an approximation of it.

    **The counts come from `minecu_tg.codegen.census`, imported, not
    reimplemented.** `uops_total_graph` here is therefore the same integer
    `capture_opt.py` put in `notes/23`'s table, and not a second number that
    happens to share a name with it.
    """
    import re as _re
    os.environ["DEV"] = "NULL:CUDA:" + _re.sub(r"(sm_\d+)[a-z]$", r"\1", a.arch)
    os.environ["NULL_ALLOW_COPYOUT"] = "1"

    from tinygrad.renderer import cstyle
    _orig_init = cstyle.CUDARenderer.__init__

    def _no_nvcc(self, target, use_nvcc=False):
        return _orig_init(self, target, use_nvcc=True)
    cstyle.CUDARenderer.__init__ = _no_nvcc

    import numpy as np
    from tinygrad import Tensor, dtypes
    from minecu_tg import codegen, generate_world
    from minecu_tg.render import camera_basis
    from minecu_tg_opt.render import render_variant
    from minecu_tg_opt.lm38 import ARM_C, Nolocals, render_arm_c, source_facts

    world, ground = 16, 4
    one = generate_world(world, ground).numpy()
    vox = Tensor(np.ascontiguousarray(np.stack([one] * a.batch)), dtype=dtypes.int8)
    cams = np.array([[world / 2.0, float(ground), world / 2.0,
                      0.001 * (i % 97 - 48), 0.0] for i in range(a.batch)],
                    dtype=np.float32)
    basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, a.res, a.res)

    out = {"arch": a.arch, "batch": a.batch, "max_steps": a.steps, "arms": []}
    say(f"\n  generated CUDA per arm  (batch {a.batch}, max_steps {a.steps}, "
        f"{a.arch}, NULL device, UOps from minecu_tg.codegen.census)")
    say(f"    {'arm':<26} {'launch_bnd':>10} {'stmts':>7} {'src B':>8} "
        f"{'launches':>8} {'uops/px/step':>12}  sha256")

    def census_one(label, fn, nolocals=False):
        stage = a.out / "census" / label
        stage.mkdir(parents=True, exist_ok=True)
        if nolocals:
            with Nolocals(True):
                c = codegen.census(fn, [a.steps], stage, keep_sources=True)
        else:
            c = codegen.census(fn, [a.steps], stage, keep_sources=True)
        row = c["rows"][0]
        srcs = sorted((stage / f"max_steps_{a.steps:04d}").glob("kernel*.cu"),
                      key=lambda p: -p.stat().st_size)
        # The FUSED region is the largest source; the others are the small
        # prologue/epilogue kernels. Recording all of them, with their shas, is
        # what makes "the graph is the same" checkable rather than asserted --
        # and it is the whole set, not the biggest file, that has to match.
        files = []
        for p in srcs:
            txt = p.read_text()
            files.append({"name": p.name, **source_facts(txt)})
        big = files[0] if files else {"error": "no source captured"}
        rec = {"arm": label, "nolocals": nolocals,
               "hot_source": big,
               "all_sources": files,
               "source_sha256": big.get("source_sha256"),
               "n_sources": len(files),
               "kernel_launches": row["kernel_launches"],
               "distinct_programs_launched": row["distinct_programs_launched"],
               "uops_total_graph": row["uops_total_graph"],
               "uops_per_pixel_per_step": row["uops_total_graph"] / (a.batch * a.res * a.res) / a.steps}
        out["arms"].append(rec)
        say(f"    {label:<26} {str(big.get('launch_bounds')):>10} "
            f"{big.get('source_statements', 0):>7} {big.get('source_bytes', 0):>8} "
            f"{row['kernel_launches']:>8} {rec['uops_per_pixel_per_step']:>12.4f}  "
            f"{big.get('source_sha256', '')[:16]}")
        return rec

    for v in ("naive", "int_cursor", "flat_gather", "incremental", "lean"):
        census_one(v, lambda ms, vv=v: render_variant(vox, basis, a.res, a.res, ms, vv))
    for v in sorted(ARM_C):
        census_one(v, lambda ms, vv=v: render_arm_c(vox, basis, a.res, a.res, ms, vv))
    census_one("flat_gather+NOLOCALS",
               lambda ms: render_variant(vox, basis, a.res, a.res, ms, "flat_gather"),
               nolocals=True)
    census_one("naive+NOLOCALS",
               lambda ms: render_variant(vox, basis, a.res, a.res, ms, "naive"),
               nolocals=True)

    base = next((r for r in out["arms"] if r["arm"] == "flat_gather"), None)
    if base:
        out["graph_size_equality"] = {
            r["arm"]: {
                "hot_source_identical": r["source_sha256"] == base["source_sha256"],
                "all_sources_identical": [f["source_sha256"] for f in r["all_sources"]]
                                         == [f["source_sha256"] for f in base["all_sources"]],
                "launches_equal": r["kernel_launches"] == base["kernel_launches"],
                "uops_equal": r["uops_total_graph"] == base["uops_total_graph"],
                "uops_delta": r["uops_total_graph"] - base["uops_total_graph"],
                "uops_per_pixel_per_step": r["uops_per_pixel_per_step"],
            }
            for r in out["arms"]
            if r["arm"] in ARM_C or r["arm"] in ("flat_gather+NOLOCALS",)}
        say("")
        say("    graph-size equality against flat_gather (UOps are the census's):")
        for k, v in out["graph_size_equality"].items():
            say(f"      {k:<24} hot source identical {str(v['hot_source_identical']):<5} "
                f"all sources identical {str(v['all_sources_identical']):<5} "
                f"launches {v['launches_equal']!s:<5} "
                f"uops delta {v['uops_delta']:>12,}  "
                f"{v['uops_per_pixel_per_step']:.4f} uops/px/step")
    return out


# ---------------------------------------------------------------------------
def stage3_instruments(a) -> dict:
    """Every census instrument this note uses, invoked once, here."""
    out = {}
    src = REPO / "minecu_ref" / "src" / "kernels.cu"
    h = hashlib.sha256(src.read_bytes()).hexdigest() if src.exists() else None
    out["kernels_cu_sha256"] = h
    out["kernels_cu_pinned"] = (h == "c6ed8f06ee64d40a8a8ab7ebce3b3ceea1e3ac93f9ee50c"
                                      "fc1e63d3f090d878f")
    say(f"\n  hand-written reference minecu_ref/src/kernels.cu")
    say(f"    sha256 {h}")
    say(f"    matches the hash notes/36 used: {out['kernels_cu_pinned']}")
    from minecu_tg_opt import opcount
    out["opcount_parse_ptxas_v"] = callable(opcount.parse_ptxas_v)
    out["opcount_count_instructions"] = opcount.count_instructions('/*0000*/ MOV R1, c[0x0][0x28] ;')
    say(f"    opcount.count_instructions on one SASS line: "
        f"{out['opcount_count_instructions']} (must be 1)")
    from minecu_tg.bench import linear_device_time_ms, require_fresh_graph
    out["ruler"] = f"{linear_device_time_ms.__module__}.{linear_device_time_ms.__name__}"
    out["guard"] = f"{require_fresh_graph.__module__}.{require_fresh_graph.__name__}"
    say(f"    ruler {out['ruler']}, guard {out['guard']}")
    from minecu_tg_opt.lm38 import CompilerBank, PtxCappedCompiler, func_facts
    out["lm38_imports"] = [CompilerBank.__name__, PtxCappedCompiler.__name__,
                           func_facts.__name__]
    say(f"    lm38: {', '.join(out['lm38_imports'])}")
    out["pass"] = out["opcount_count_instructions"] == 1
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / ".agent-scratch" / "lm38" / "preflight")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--arch", default="sm_75")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "kernels.cu").write_bytes((REPO / "minecu_ref" / "src" / "kernels.cu").read_bytes())

    say("=" * 78)
    say("  lm38 pre-flight: correctness, codegen equality, instruments. CPU only.")
    say("=" * 78)
    t0 = time.time()
    rep = {"argv": sys.argv[1:], "python": sys.version.split()[0]}
    say("\n--- 1. the arm-C bodies compute the same thing (CPU) ---")
    rep["correctness"] = stage1_correctness(a)
    say("\n--- 2. the generated CUDA of every arm, and graph-size equality ---")
    rep["codegen"] = stage2_codegen(a)
    say("\n--- 3. the instruments ---")
    rep["instruments"] = stage3_instruments(a)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["pass"] = bool(rep["correctness"].get("pass") and rep["instruments"].get("pass"))
    (a.out / "preflight.json").write_text(json.dumps(rep, indent=1, default=str))
    say("\n" + "=" * 78)
    say(f"  pre-flight {'PASS' if rep['pass'] else 'FAIL'} in {rep['wall_s']}s  ->  {a.out}")
    if not rep["pass"]:
        for f in rep["correctness"].get("fails", []):
            say(f"    FAIL {f}")
    say("=" * 78)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
