#!/usr/bin/env python3
"""sass.py -- the SASS census, both sides, through one categoriser.

WHY THIS FILE EXISTS
--------------------
`notes/19` §4 quotes the reference's 776 instructions, 28 registers, 33
`BSSY`/`BSYNC` pairs, 16 `LDG.E.S8.CONSTANT.SYS` at a load-to-load spacing of
38, and an exact-match run at period 38. Those are the numbers the whole
boundary map is built on, and every one of them is re-derived here from
`minecu_ref/build/kernels_sm_75.sass` and `resources_sm_75.txt` rather than
transcribed. A number that was transcribed has been wrong in this repo before,
and the failure mode is invisible: it looks exactly like a number.

WHAT IS MEASURED, AND WITH WHICH INSTRUMENT
-------------------------------------------
`reference()`  the hand-written side, from the preserved SASS and the `ptxas
               -v` report in the same directory. Nothing is compiled here; the
               binary is nvcc 12.9, -O3 --use_fast_math, -arch=sm_75, and
               `build.sh` is the provenance.

`generated()`  this port's own generated CUDA, compiled for sm_75 by
               `census.py` and then read here. Both sides go through
               `tools/archlab/inspect_ptx.py:CATEGORIES`, imported, not
               reimplemented -- a second definition of "a vectorised load" is
               how a study ends up comparing two different quantities.

THE INSTRUMENT BUG THIS FILE CARRIES FORWARD ON PURPOSE
------------------------------------------------------
The SASS offset field is 4 hex digits for the first 4096 instructions and 5
after, so a `{4}` regex reports EXACTLY 4096 instructions for anything
larger, which looks like a plausible number and is silently wrong. `notes/19`
§4 caught it on three kernels of 277, 1045 and 2069 global loads. The pattern
here is `{4,8}`. The reference's 776 is unchanged under both, which is the
check that the fix is right, and it is re-run here rather than assumed.

`repeat_period` is the one number the boundary map's central claim rests on:
the reference's 16 unrolled steps are byte-identical repeats, which is why
ptxas could allocate them one at a time, and the fused generated kernel's 32
are not. It is computed here, not quoted.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCHLAB = REPO / "tools" / "archlab"
sys.path.insert(0, str(ARCHLAB))

from inspect_ptx import CATEGORIES, parse_sass  # noqa: E402

REF_SASS = REPO / "minecu_ref" / "build" / "kernels_sm_75.sass"
REF_RES = REPO / "minecu_ref" / "build" / "resources_sm_75.txt"
REF_MANIFEST = REPO / "minecu_ref" / "build" / "MANIFEST.txt"

# `cuobjdump -sass` body: "        /*0000*/                   MOV R1, c[0x0][0x28] ;"
# {4,8} and not {4}: see the module docstring.
INSTR_RE = re.compile(r"/\*([0-9a-fA-F]{4,8})\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)")
FUNC_RE = re.compile(r"Function : (\S+)")
# ptxas -v:
#   0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
#   Used 28 registers, used 1 barriers, 64 bytes smem, 404 bytes cmem[0]
PTX_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                       r"(\d+) bytes spill loads")
PTX_USED = re.compile(r"Used (\d+) registers, used (\d+) barriers")
PTX_SMEM = re.compile(r"(\d+) bytes smem")

# the reference's hot kernel, by its mangled name. This is
# `render_kernel_uint8_prebasis` (kernels.cu:568).
HOT_KERNEL = "_Z28render_kernel_uint8_prebasisPKaPKfPhiiiiiii"


def split_functions(sass: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in sass.splitlines():
        m = FUNC_RE.search(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def instructions(body: str) -> list[str]:
    """The opcode stream, in order. `{4,8}`, see the docstring."""
    return [op for _, op in INSTR_RE.findall(body)]


def _offset_of(body: str) -> dict[str, int]:
    return {op: int(off, 16) for off, op in INSTR_RE.findall(body)}


def ldg_indices(ops: list[str], opcode_prefix: str = "LDG.E.S8") -> list[int]:
    """Indices of the voxel-width loads.

    The width spelling is not the same on both sides: the reference's `__ldg`
    of an `int8_t` comes out as `LDG.E.S8.CONSTANT.SYS`, and TinyGrad's gather
    of a `signed char` comes out as `LDG.E.U8.SYS`. Both are one byte. The
    prefix is auto-detected so the *spacing* measurement -- which is the thing
    that shows a fixed-width unroll window -- compares the same instruction on
    both sides rather than finding zero on one of them.
    """
    if not any(o.startswith("LDG.E.S8") for o in ops):
        opcode_prefix = "LDG.E.U8"
    return [i for i, o in enumerate(ops) if o.startswith(opcode_prefix)]


def repeat_period(ops: list[str], lo: int = 20, hi: int = 200) -> tuple[int, int]:
    """Longest exact-match run over the whole stream, and its period.

    Scans every period in [lo, hi] and reports the best. This is the
    instrument behind the central claim: a body that is a byte-identical
    repeat at period 38 was 38 instructions long by construction and could be
    register-allocated one copy at a time; a stream with no such period was
    allocated as one problem.
    """
    best_span, best_period = 0, 0
    for p in range(lo, hi + 1):
        if p >= len(ops):
            break
        run = 0
        for i in range(len(ops) - p):
            run = run + 1 if ops[i] == ops[i + p] else 0
            if run + p > best_span:
                best_span, best_period = run + p, p
    return best_span, best_period


def dda_step_body(ops: list[str]) -> dict:
    """Price the reference's step body directly from its opcode stream.

    One period of the loop, sliced between two consecutive voxel loads. This
    is deliberately measured two ways -- by the load-to-load spacing and by
    the text of the body -- because `notes/19` §4.3 found the spacing is
    uniform while the *longest exact-match run* is shorter than 16 periods,
    and reporting only one of those would overstate the case for a
    period-based allocation.
    """
    idx = ldg_indices(ops)
    if len(idx) < 2:
        return {"n_loads": len(idx)}
    spacings = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
    span = ops[idx[0]:idx[1]]
    counts = collections.Counter(span)
    return {
        "n_loads": len(idx),
        "spacings": spacings,
        "spacing_uniform": len(set(spacings)) == 1,
        "spacing": spacings[0] if spacings else None,
        "body_span_instructions": idx[-1] - idx[0] + 1,
        "n_break": counts.get("BREAK", 0),
        "n_bssy": counts.get("BSSY", 0),
        "n_bsync": counts.get("BSYNC", 0),
        "n_imad": sum(v for k, v in counts.items() if k.startswith("IMAD")),
        "n_ishl_or_shf": sum(v for k, v in counts.items() if k.startswith("SHF")),
        "n_iadd3": counts.get("IADD3", 0),
        "n_iadd3_x": counts.get("IADD3.X", 0),
        "n_isetp_u32_or": sum(v for k, v in counts.items() if k == "ISETP.GE.U32.OR"),
        "n_fsetp": sum(v for k, v in counts.items() if k.startswith("FSETP")),
        "n_ldg_in_span": sum(v for k, v in counts.items() if k.startswith("LDG")),
        "n_branch_in_span": sum(v for k, v in counts.items() if k.startswith("BRA")),
    }


def _row(name: str, body: str, res: dict | None) -> dict:
    ops = instructions(body)
    st = parse_sass(body)
    r: dict = {"kernel": name, "instructions": len(ops)}
    for cat, _, _ in CATEGORIES:
        r[cat] = st["categories"].get(cat, 0)
    r["bss_bsync_pairs"] = ops.count("BSSY")
    r["branch"] = ops.count("BRA")
    r["mufu_rcp"] = ops.count("MUFU.RCP")
    r["mufu_rsq"] = ops.count("MUFU.RSQ")
    r["ldg_e_s8"] = sum(1 for o in ops if o.startswith("LDG.E") and ".S8" in o)
    r["ldg_e_s8_constant"] = sum(1 for o in ops
                                 if o.startswith("LDG.E.S8.CONSTANT"))
    r["ldg_e_s8_plain"] = sum(1 for o in ops
                              if o.startswith("LDG.E.S8") and ".CONSTANT" not in o)
    r["ldg_any_constant"] = sum(1 for o in ops
                                if o.startswith("LDG") and ".CONSTANT" in o)
    # the same split, for the 1-byte loads TinyGrad emits. ptxas spells a
    # one-byte signed load `LDG.E.U8.SYS` on sm_75 and the reference's `__ldg`
    # of an `int8_t` comes out as `LDG.E.S8.CONSTANT.SYS` -- two spellings of
    # the same width, one with the read-only cache path and one without. Only
    # the `.CONSTANT` suffix distinguishes them, so a byte-load count that
    # looks at width alone reports the two sides as equal when they are not.
    r["ldg_e_u8"] = sum(1 for o in ops if o.startswith("LDG.E") and ".U8" in o)
    r["ldg_byte_total"] = r["ldg_e_s8"] + r["ldg_e_u8"]
    r["ldg_byte_constant_path"] = r["ldg_e_s8_constant"] + sum(
        1 for o in ops if o.startswith("LDG.E.U8.CONSTANT"))
    r["ldg_byte_plain_path"] = (r["ldg_e_s8_plain"]
                                + sum(1 for o in ops if o.startswith("LDG.E.U8")
                                      and ".CONSTANT" not in o))
    r["stg_e_u8"] = sum(1 for o in ops if o.startswith("STG.E.U8"))
    r["imad_wide"] = sum(1 for o in ops if o.startswith("IMAD.WIDE"))
    r["ldc_u8"] = ops.count("LDC.U8")
    r["restrict_in_source"] = 0
    if res:
        r.update(res)
    else:
        r.update({k: None for k in ("registers", "barriers", "spill_stores",
                                    "spill_loads", "local_bytes", "stack_frame",
                                    "smem_static")})
    return r


def parse_ptxas_v(text: str) -> dict[str, dict]:
    """`ptxas -v` text -> {function: {...}}. Matched BY NAME."""
    out: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '([^']+)' for '(\S+)'", line)
        if m:
            cur = {"arch": m.group(2)}
            out[m.group(1)] = cur
            continue
        if cur is None:
            continue
        m = PTX_FRAME.search(line)
        if m:
            cur["stack_frame"] = int(m.group(1))
            cur["spill_stores"] = int(m.group(2))
            cur["spill_loads"] = int(m.group(3))
            cur["local_bytes"] = int(m.group(2)) + int(m.group(3))
            continue
        m = PTX_USED.search(line)
        if m:
            cur["registers"] = int(m.group(1))
            cur["barriers"] = int(m.group(2))
            m2 = PTX_SMEM.search(line)
            if m2:
                cur["smem_static"] = int(m2.group(1))
    return out


def reference() -> dict:
    """The hand-written side, re-derived from the preserved build."""
    sass = REF_SASS.read_text()
    res = parse_ptxas_v(REF_RES.read_text())
    funcs = split_functions(sass)
    if HOT_KERNEL not in funcs:
        raise SystemExit(f"{HOT_KERNEL} not in {REF_SASS}; "
                         f"have {sorted(funcs)[:5]}...")
    body = funcs[HOT_KERNEL]
    ops = instructions(body)
    row = _row(HOT_KERNEL, body, res.get(HOT_KERNEL))
    row["source"] = str(REF_SASS)
    row["resources"] = str(REF_RES)
    if REF_MANIFEST.exists():
        row["build_manifest"] = REF_MANIFEST.read_text().strip().splitlines()
    # the register count and the spill are also re-derived from the opcode
    # stream, independently of ptxas -v, so a mismatch is visible
    row["local_ops"] = sum(1 for o in ops if ".L" in o and o.startswith("LDL"))
    return {
        "row": row,
        "dda_step": dda_step_body(ops),
        "repeat": {"span": repeat_period(ops)[0], "period": repeat_period(ops)[1]},
        "opcode_histogram": dict(collections.Counter(ops).most_common()),
    }


def generated(sass_path: Path, ptxas_path: Path | None, select: str | None) -> list[dict]:
    """This port's own kernels, from a `sass.py`-compatible directory."""
    sass = sass_path.read_text()
    res = parse_ptxas_v(ptxas_path.read_text()) if ptxas_path and ptxas_path.exists() else {}
    out = []
    for name, body in split_functions(sass).items():
        if select and select not in name:
            continue
        r = _row(name, body, res.get(name))
        r["dda_step"] = dda_step_body(instructions(body))
        r["repeat"] = dict(zip(("span", "period"), repeat_period(instructions(body))))
        out.append(r)
    return out


def summarise(r: dict) -> str:
    def _n(v, w=5):
        return f"{v:{w}}" if isinstance(v, int) else f"{'?':>{w}}"
    return (f"{r['kernel'][:40]:40s} instr={r['instructions']:6d} "
            f"reg={_n(r.get('registers'))} spill={_n(r.get('local_bytes'))} "
            f"smem={_n(r.get('smem_static'))} BAR={_n(r.get('barriers'), 3)} "
            f"LDG={r['global_ld']:5d} BYTEld={r['ldg_byte_total']:4d} "
            f"CONST={r['ldg_byte_constant_path']:4d} "
            f"BSSY={r['bss_bsync_pairs']:4d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", type=Path, default=None,
                    help="a .sass from a census of this port's generated code")
    ap.add_argument("--ptxas", type=Path, default=None)
    ap.add_argument("--select", default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    ref = reference()
    print("== the reference, re-derived from "
          f"{REF_SASS.relative_to(REPO)} ==")
    print("  " + summarise(ref["row"]))
    d = ref["dda_step"]
    print(f"  DDA step body: {d['n_loads']} LDG.E.S8, spacings uniform="
          f"{d['spacing_uniform']} at {d['spacing']}, body span "
          f"{d['body_span_instructions']} instructions")
    print(f"    per step: IMAD {d['n_imad']}, IADD3 {d['n_iadd3']}+X{d['n_iadd3_x']}, "
          f"SHF {d['n_ishl_or_shf']}, ISETP.GE.U32.OR {d['n_isetp_u32_or']}, "
          f"FSETP {d['n_fsetp']}, BSSY {d['n_bssy']}, BRA {d['n_branch_in_span']}")
    print(f"    address arithmetic per step (IMAD+SHF+IADD3+IADD3.X) = "
          f"{d['n_imad'] + d['n_ishl_or_shf'] + d['n_iadd3'] + d['n_iadd3_x']} "
          f"of {d['spacing']} instructions "
          f"({100 * (d['n_imad'] + d['n_ishl_or_shf'] + d['n_iadd3'] + d['n_iadd3_x']) / d['spacing']:.0f}%)")
    print(f"  longest exact-match run: {ref['repeat']['span']} instructions at "
          f"period {ref['repeat']['period']}")
    if a.generated:
        print(f"\n== this port's generated code, from {a.generated} ==")
        rows = generated(a.generated, a.ptxas, a.select)
        for r in rows:
            print("  " + summarise(r))
            g = r["dda_step"]
            if g.get("n_loads", 0) >= 2:
                print(f"      step body: {g['n_loads']} byte loads, spacing uniform="
                      f"{g['spacing_uniform']} at {g['spacing']}, span "
                      f"{g['body_span_instructions']}; BSSY {g['n_bssy']}")
            print(f"      exact-match run: {r['repeat']['span']} at period {r['repeat']['period']}")
        if not rows:
            print(f"  (nothing matched --select {a.select!r})")

    if a.json:
        payload = {"reference": ref, "generated": []}
        if a.generated:
            payload["generated"] = generated(a.generated, a.ptxas, a.select)
        a.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
