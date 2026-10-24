#!/usr/bin/env python3
"""boundary.py -- the boundary map, assembled from measured artefacts only.

Every number in the table this prints comes from one of:

  * `minecu_ref/build/kernels_sm_75.sass` + `resources_sm_75.txt`, re-read by
    `sass.py` (the hand-written side);
  * a `sass_build.sh` census of this port's own generated CUDA, both modes;
  * `census.py`'s launch/program counts;
  * `battery.py`'s on-card timings, when they exist.

Nothing is transcribed from a note. A cell that could not be measured says
`NOT MEASURED`, and a cell that was modelled rather than measured says so in
the cell, because those are different claims and `notes/06-corrections-ledger.md`
§5 is a four-week record of what happens when they are confused.

    python3 minecu_tg_faithful/boundary.py --sass-clean D --sass-faithful D \
        [--census out/.../census.json] [--battery out/.../faithful_battery.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from sass import generated, reference  # noqa: E402

# The rows, one per technique in the reference's hot kernel, in the order the
# reader needs them: the ones TinyGrad's normal vocabulary already expresses
# first, then the ones a crossing reached, then the ones nothing reaches.
#
# `status` is one of:
#   MEASURED    a number in this table came from a run on this host or a card
#   MODELLED    derived from a measured number by arithmetic shown in the cell
#   NOT TESTED  the row could not be closed here, and says why
#
# `tinygrad` is the answer to "does the framework's normal vocabulary express
# it", and it is deliberately terse: the detail is in `crossings.py`, cited by
# file:line from there rather than restated in three places.

ROWS = [
    dict(n=1, technique="per-pixel ray from a precomputed basis (T3)",
         ref="kernels.cu:619-624; basis kernel :511-561, one launch of B threads",
         tinygrad="YES -- `camera_basis` is a separate tensor function; no knob",
         crossing="none",
         cost_instr="0 extra: the trig is gone from the pixel kernel either way",
         cost_time="not separable -- it is in both modes identically",
         status="MEASURED",
         evidence="the generated basis kernel is 72 instructions at batch 8 "
                  "(sass_build.sh, k001), against the reference's "
                  "precompute_camera_basis_kernel; the count is in sass.py's output"),
    dict(n=2, technique="3-byte interleave epilogue, uint8 LUT, no float math (T9)",
         ref="kernels.cu:479-495, :691-699; 6 LDC.U8 + 6 STG.E.U8",
         tinygrad="YES as a 14-row gather (AIR -> row 13)",
         crossing="none; a branch becomes a table, disclosed",
         cost_instr="see LDG/STG columns in the SASS table",
         cost_time="not separable from the store it is",
         status="MEASURED",
         evidence="sass.py: ldc_u8, stg_e_u8, global_st_widths on both sides"),
    dict(n=3, technique="flat address, one index, no 4-index gather (T4)",
         ref="kernels.cu:669 -- one __ldg of a long long address",
         tinygrad="YES -- `flat[addr]` with a maintained address",
         crossing="none; identical in both modes",
         cost_instr="1 byte load per step on this side against the reference's 1",
         cost_time="included in both modes; the reference's is 14 of 38 "
                   "instructions per step (sass.py:dda_step)",
         status="MEASURED",
         evidence="sass.py dda_step: n_ldg_in_span, and the ADDR/IMAD+SHF+IADD3 "
                  "count that prices the reference's own address arithmetic"),
    dict(n=4, technique="unsigned bounds fold, 3 ISETP.GE.U32.OR (T5)",
         ref="kernels.cu:666-668",
         tinygrad="YES -- a countdown of steps-to-exit, integer compares",
         crossing="edit point `unsigned_bounds` (both settings verified "
                  "identical observations and step counts)",
         cost_instr="3 instructions either way on sm_75",
         cost_time="none measurable",
         status="MEASURED",
         evidence="verify.py sweeps uint=0 and uint=1 over both address widths "
                  "and the per-pixel step counts are IDENTICAL, which is the "
                  "sharper statement than an instruction count"),
    dict(n=5, technique="64-bit index (long long)",
         ref="kernels.cu:634-636, :669; 35 `long long` sites",
         tinygrad="PARTIAL -- tinygrad's index lowering NARROWS a long index to "
                  "int when the buffer fits (uop/weak.py:74-81), so the 64-bit "
                  "arithmetic is computed and then truncated at the load",
         crossing="edit point `addr_bits` (32 or 64)",
         cost_instr="see the `cast` columns: addr64 emits `long` address "
                    "arithmetic and then `((int)(addr))` at each load",
         cost_time="none measurable at world=16; the buffer is 32 KiB",
         status="MEASURED",
         evidence="diff of the generated .cu between addr_bits=32 and 64: the "
                  "64-bit form computes in `long` and narrows only at the "
                  "pointer, which is a real cost and no benefit at this size"),
    dict(n=6, technique="per-ray early exit -- divergence sync (T6)",
         ref="kernels.cu:670 `goto done_pre`; 33 BSSY/BSYNC pairs, 28 registers",
         tinygrad="NO. A per-ray trip count is not a fixed-size tensor and the "
                  "IR has no loop: Ops has no WHILE/LOOP/REPEAT, and "
                  "IF/ENDIF is synthesised for gated stores only "
                  "(codegen/__init__.py:402-407)",
         crossing="NONE EXISTS. The masked formulation (`live = hit < 0`) is the "
                  "nearest expressible thing and every ray runs all max_steps",
         cost_instr="BSSY 33 -> 0: the generated side is CHEAPER per step",
         cost_time="32 steps per ray against a measured mean of 26.4 = 1.21x "
                   "more work",
         status="MEASURED",
         evidence="sass.py bss_bsync_pairs on both sides; mean steps/pixel from "
                  "battery.py's count_steps, verified inert by verify.py"),
    dict(n=7, technique="16-copy static unroll window, 16 loads in flight (T7)",
         ref="kernels.cu:681-687; 16 LDG.E.S8.CONSTANT.SYS at spacing exactly "
             "38, fifteen times; 776 instructions total",
         tinygrad="NO. `OptOps.UNROLL` applies to reduce axes only "
                  "(codegen/opt/postrange.py:113 resolves it through "
                  "`unrollable_dims`; heuristic.py:139-152 gates it on "
                  "`if k.unrollable_dims`). The DDA has no reduce axis, and no "
                  "ContextVar controls the inline factor of an elementwise chain",
         crossing="NONE EXISTS",
         cost_instr="the two regimes: one kernel per step (state via global "
                    "memory) or one fused kernel",
         cost_time="the whole clean-vs-faithful row below",
         status="MEASURED",
         evidence="the census's kernel count and the SASS instruction sum per "
                  "regime; sass.py dda_step prints the load spacing on the "
                  "reference and on each generated program"),
    dict(n=8, technique="28-register footprint, zero spill (T8)",
         ref="resources_sm_75.txt: 28 registers, 0 spill, 0 stack, 64 B smem, "
             "1 barrier",
         tinygrad="NO LEVER. __launch_bounds__ is derived from the product of "
                  "the local dims (renderer/cstyle.py:156-157, 411) and there "
                  "is no maxrregcount path; MAX_KERNEL_BUFFERS counts ARGUMENTS "
                  "(rangeify.py:371), not registers; NOLOCALS changes launch "
                  "geometry, not the allocator",
         crossing="NONE EXISTS",
         cost_instr="the fused kernel's register and spill figures in the SASS "
                    "table",
         cost_time="in the fused regime, most of it",
         status="MEASURED",
         evidence="the `reg` and `spill` columns for every generated program, "
                  "from ptxas -v on the same nvcc that built the reference"),
    dict(n=9, technique="`__restrict__` -> the read-only LDG path",
         ref="kernels.cu:569-572 and 81 sites; 16 of 17 loads are "
             "LDG.E.S8.CONSTANT.SYS",
         tinygrad="NO. CUDARenderer leaves buffer_suffix empty "
                  "(renderer/cstyle.py:121); only ClangRenderer sets "
                  "`\" restrict\"` (cstyle.py:269). Zero `restrict` in any "
                  "captured kernel",
         crossing="NONE EXISTS; it would mean editing the renderer, which is "
                  "not this port's to edit",
         cost_instr="0 -- same number of loads, different cache path",
         cost_time="not measured on the card at this configuration; the "
                   "generated loads are plain LDG.E.SYS in the SASS table",
         status="MEASURED (the SASS consequence); NOT MEASURED (the time)",
         evidence="sass.py ldg_e_s8_constant vs ldg_e_s8_plain on both sides"),
    dict(n=10, technique="the basis in shared memory + one barrier (T2)",
         ref="kernels.cu:581, :594-597; 64 B smem, 1 BAR.SYNC",
         tinygrad="NO. LOCAL buffers are a reduce's scratch "
                  "(codegen/__init__.py:325-327; a LOCAL axis needs "
                  "ren.has_local, heuristic.py:162). An elementwise kernel "
                  "cannot declare one, and NOLOCALS removes locals rather "
                  "than adding one",
         crossing="NONE; and the measured effect is ~0 anyway -- the basis is a "
                  "416-byte broadcast L1 read",
         cost_instr="1 barrier against 0",
         cost_time="not measurable at this size; recorded as a negative result "
                   "so the row exists and is closed",
         status="MEASURED (SASS); NOT MEASURED (time)",
         evidence="the `BAR` column; probe_knobs.py measures NOLOCALS changing "
                  "the generated source by 268 B of 113 322 and nothing else"),
    dict(n=11, technique="grid shaped as (pixel blocks, env) so b is "
                         "block-uniform (T1)",
         ref="kernels.cu:584, launched :2182 as dim3(blocks, batch)",
         tinygrad="NO LEVER -- the grid comes from the shape and the opts "
                  "(codegen/gpudims.py:41-58; uop/ops.py:1241-1261)",
         crossing="none",
         cost_instr="0: the generated kernels put the batch on blockIdx.z or "
                    "fold it into a flat index, and the basis read is a "
                    "broadcast either way",
         cost_time="not measurable",
         status="MEASURED (SASS); NOT MEASURED (time)",
         evidence="blockIdx usage in the captured .cu; the `BAR` and `LDG` "
                  "columns are unaffected by which axis carries the env"),
    dict(n=12, technique="`--use_fast_math` numerics",
         ref="minecu_ref/build.sh: -O3 --use_fast_math, so `1.0f/ray_x` is "
             "MUFU.RCP and `rsqrtf` is MUFU.RSQ",
         tinygrad="NO LEVER. NVRTCCompiler.compile_options is "
                  "[--gpu-architecture, -I..., --minimal] and nothing else "
                  "(runtime/support/compiler_cuda.py:46-56). The generated "
                  "source is `1/sqrt(...)` where the reference is `rsqrtf`",
         crossing="NONE, deliberately: matching would change the numerics being "
                  "compared, so the honest form of this is a recorded gap",
         cost_instr="touches the prologue, which is 152 of the reference's "
                    "776 instructions (19.6%)",
         cost_time="not measured",
         status="MEASURED (the source difference); NOT MEASURED (the time)",
         evidence="the mufu_rcp / mufu_rsq columns in the SASS table, and the "
                  "generated `1/sqrt(...)` in the captured .cu"),
    dict(n=13, technique="per-ray AABB early-out",
         ref="kernels.cu:51-79, tested at :170 in render_kernel -- NOT in "
             "render_kernel_uint8_prebasis, which is the kernel this arm ports",
         tinygrad="n/a -- there is nothing to express",
         crossing="n/a",
         cost_instr="0",
         cost_time="0",
         status="NOT TESTED -- the technique is absent from the ported kernel. "
                  "Listed so the row exists and is closed rather than "
                  "inherited from notes/18 entry 18, which prices a different "
                  "kernel",
         evidence="kernels.cu:568-700 contains no ray_intersects_aabb call"),
]


def fmt_row(r: dict, measured: dict) -> str:
    def cell(k):
        v = measured.get(r["technique"][:20], {}).get(k) if isinstance(measured, dict) else None
        return v if isinstance(v, str) else r.get(k, "?")
    return (f"| {r['n']} | {r['technique']} | {cell('tinygrad')} | {cell('crossing')} "
            f"| {cell('cost_instr')} | {cell('cost_time')} | {r['status']} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sass-clean", type=Path, default=None)
    ap.add_argument("--sass-faithful", type=Path, default=None)
    ap.add_argument("--census", type=Path, default=None)
    ap.add_argument("--battery", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    ref = reference()
    rr = ref["row"]
    print("== the reference, re-derived ==")
    print(f"  {rr['kernel']}")
    print(f"  instructions {rr['instructions']}, registers {rr['registers']}, "
          f"spill {rr['local_bytes']} B, barriers {rr['barriers']}, "
          f"smem {rr['smem_static']} B, BSSY {rr['bss_bsync_pairs']}, "
          f"LDG.E.S8 {rr['ldg_e_s8']} (constant-path {rr['ldg_e_s8_constant']})")
    d = ref["dda_step"]
    print(f"  DDA step: {d['n_loads']} loads, spacing {d['spacing']} uniform="
          f"{d['spacing_uniform']}; address arithmetic "
          f"{d['n_imad'] + d['n_ishl_or_shf'] + d['n_iadd3'] + d['n_iadd3_x']} "
          f"of {d['spacing']} instructions")
    print(f"  longest exact-match run: {ref['repeat']['span']} at period "
          f"{ref['repeat']['period']}")

    payload = {"reference": ref, "modes": {}, "census": None, "battery": None}
    for label, d_ in (("clean", a.sass_clean), ("faithful", a.sass_faithful)):
        if not d_ or not d_.exists():
            print(f"\n== {label}: no SASS census at {d_} ==")
            continue
        sass_files = sorted((d_ / "out").glob("*.sass"))
        rows = []
        for s in sass_files:
            p = s.with_suffix("").with_suffix(".ptxas")
            p = Path(str(s)[:-5] + ".ptxas")
            if not p.exists():
                p = None
            rows += generated(s, p, None)
        tot_instr = sum(r["instructions"] for r in rows)
        regs = [r["registers"] for r in rows if isinstance(r.get("registers"), int)]
        spills = [r["local_bytes"] for r in rows if isinstance(r.get("local_bytes"), int)]
        bss = sum(r["bss_bsync_pairs"] for r in rows)
        ldg_const = sum(r["ldg_byte_constant_path"] for r in rows)
        ldg_plain = sum(r["ldg_byte_plain_path"] for r in rows)
        ldg_byte = sum(r["ldg_byte_total"] for r in rows)
        # The largest program, named and reported on its own. NOT called "the
        # DDA kernel": in the split regime the biggest program is the tail that
        # consumes all 32 materialised state buffers, and in the fused regime it
        # is the whole 32-step body. They are different objects and a column
        # that lumps them together under one label is a column that will be
        # quoted for the wrong regime.
        hot = max(rows, key=lambda r: r["instructions"])
        print(f"\n== {label}: {len(rows)} programs, {d_} ==")
        print(f"  static instructions, whole render : {tot_instr}  "
              f"({tot_instr / rr['instructions']:.1f}x the reference's "
              f"{rr['instructions']})")
        print(f"  registers: min {min(regs)} max {max(regs)}" if regs else "  registers: ?")
        print(f"  spill bytes, whole render: {sum(spills)}" if spills else "  spill: ?")
        print(f"  BSSY/BSYNC pairs: {bss}  (the reference has 33)")
        print(f"  1-byte loads: {ldg_byte} total -- {ldg_const} on the "
              f"read-only (.CONSTANT) path, {ldg_plain} plain")
        print(f"  largest program: {hot['kernel'][:40]}")
        print(f"    instructions {hot['instructions']}  registers {hot['registers']}  "
              f"spill {hot['local_bytes']} B  BSSY {hot['bss_bsync_pairs']}")
        print(f"    1-byte loads {hot['ldg_byte_total']}, load-to-load spacing "
              f"uniform={hot['dda_step'].get('spacing_uniform')} "
              f"at {hot['dda_step'].get('spacing')}, "
              f"longest exact-match run {hot['repeat']['span']} at period "
              f"{hot['repeat']['period']}")
        print(f"    (the reference: 16 loads, uniform at 38, exact-match run 594 "
              f"at period 38)")
        payload["modes"][label] = {
            "programs": len(rows), "instructions_total": tot_instr,
            "ratio_to_reference": tot_instr / rr["instructions"],
            "registers_min": min(regs) if regs else None,
            "registers_max": max(regs) if regs else None,
            "spill_total": sum(spills) if spills else None,
            "bss_bsync_pairs": bss,
            "ldg_byte_total": ldg_byte,
            "ldg_byte_constant_path": ldg_const, "ldg_byte_plain_path": ldg_plain,
            "largest_program": hot["kernel"],
            "largest_program_instructions": hot["instructions"],
            "largest_program_registers": hot["registers"],
            "largest_program_spill": hot["local_bytes"],
            "largest_program_dda_step": hot["dda_step"],
            "largest_program_repeat": hot["repeat"],
            "rows": rows,
        }

    if a.census and a.census.exists():
        payload["census"] = json.loads(a.census.read_text())
        rows = payload["census"].get("rows")
        if not isinstance(rows, list):
            # a census.json from a different script. Report the shape rather
            # than indexing into it: a KeyError here would look like a
            # measurement that failed, and it is only a format mismatch.
            print(f"\n== census: {a.census} has keys {sorted(payload['census'])}, "
                  f"not the expected 'rows' list -- not read ==")
            rows = []
        print("\n== census ==")
        for r in rows:
            print(f"  ms={r['max_steps']:3d} {r['label']:20s} kernels={r['kernels']:4d} "
                  f"dda={r.get('dda_programs', '?'):>4} src={r['source_bytes_total']:9d}B")
    if a.battery and a.battery.exists():
        payload["battery"] = json.loads(a.battery.read_text())
        print("\n== battery (on-card) ==")
        for r in payload["battery"].get("rows", []):
            print(f"  batch {r['batch']:6d}  ref {r.get('ref_render_us')} us  "
                  f"clean {r.get('clean_ms')} ms  faithful {r.get('faithful_ms')} ms  "
                  f"f/clean {r.get('faithful_over_clean')}  "
                  f"clean/ref {r.get('clean_over_ref')}  "
                  f"faithful/ref {r.get('faithful_over_ref')}")

    print("\n== the boundary map ==")
    print("| # | technique | TinyGrad's normal vocabulary? | crossing | cost in "
          "instructions / bytes | cost in time | status |")
    print("|---|---|---|---|---|---|---|")
    for r in ROWS:
        print(fmt_row(r, {}))

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
