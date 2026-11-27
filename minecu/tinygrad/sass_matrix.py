#!/usr/bin/env python3
"""sass_matrix.py -- the instruction matrix for both sides of the comparison.

One table, one categoriser, both sides, so the columns mean the same thing in
every row. The categoriser is `tools/archlab/inspect_ptx.py:CATEGORIES`,
imported rather than reimplemented, for the reason that a second definition of
"a vectorised load" is how a study ends up comparing two different quantities.

INPUTS are `cuobjdump -sass` outputs (text) and `cuobjdump -res-usage`
outputs, from cubins built for one architecture. Both sides of this study are
sm_75, which `notes/17` section 6 established is reachable: the hand-written
`kernels.cu` compiles for sm_75 with zero spills and no sm_80+ instruction, so
a T4 is a valid card for BOTH sides and the comparison is not one-sided.

WHAT IS REPORTED PER KERNEL
    instructions        static count, the number a reader can check by counting
    per-kind            the category census (LDG, STG, FFMA, barriers, ...)
    registers           from cuobjdump -res-usage REG
    local bytes         the SPILL column. Any non-zero value here is the single
                        most important number in the table: it means the
                        register file could not hold the live set and the
                        kernel is paying memory traffic for its own arithmetic.
    shared bytes        static shared memory
    barriers            BAR.SYNC count; the reference has exactly one, in the
                        hot kernel, for the camera basis in shared memory
    launch bounds       __launch_bounds__ in the source, i.e. the register
                        budget the programmer asked for

The hot-kernel row is selected by a substring, not by index, so the row is
identified by what it is rather than by where it happens to sit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCHLAB = REPO / "tools" / "archlab"
sys.path.insert(0, str(ARCHLAB))

from inspect_ptx import CATEGORIES, parse_sass  # noqa: E402

FUNC_RE = re.compile(r"Function : (\S+)")
# cuobjdump -sass body: "        /*0000*/                   MOV R1, c[0x0][0x28] ;"
#
# The offset field is 4 hex digits for the first 4096 instructions of a kernel
# (0x0000..0xfff0, 4 bytes each) and 5 or more after that. A `{4}` pattern
# therefore reports EXACTLY 4096 instructions for any kernel bigger than that,
# which looks like a plausible number and is silently wrong: three kernels here
# with 277, 1045 and 2069 global loads all came out as "4096 instructions".
# `{4,8}` is what makes the count mean anything, and `tools/archlab`'s
# SASS_LINE already uses {4,6} for the same reason.
INSTR_RE = re.compile(r"/\*[0-9a-fA-F]{4,8}\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)")
# `ptxas -v` output, which is the authoritative source for registers and spills:
#   Compiling entry function 'NAME' for 'sm_75'
#       0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
#   Used 28 registers, used 1 barriers, 388 bytes cmem[0]
PTXV_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '(\S+)'")
PTXV_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                        r"(\d+) bytes spill loads")
PTXV_USED = re.compile(r"Used (\d+) registers, used (\d+) barriers")


def split_functions(sass: str) -> dict[str, str]:
    """cuobjdump -sass text -> {function name: that function's body}."""
    out: dict[str, str] = {}
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


def opcodes(body: str) -> list[str]:
    return INSTR_RE.findall(body)


def parse_ptxas_v(text: str) -> dict[str, dict]:
    """`ptxas -v` text -> {function name: {registers, spill_*, barriers, ...}}.

    Matched BY NAME, not by position. Both the mangled name and the plain name
    are indexed, because `-sass` prints the mangled name and the launcher
    tables and the notes use the plain one, and pairing two lists by index is
    how a register count ends up attached to the wrong kernel.
    """
    out: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        m = PTXV_ENTRY.search(line)
        if m:
            cur = {"arch": m.group(2)}
            out[m.group(1)] = cur
            continue
        if cur is None:
            continue
        m = PTXV_FRAME.search(line)
        if m:
            cur["stack_frame"] = int(m.group(1))
            cur["spill_stores"] = int(m.group(2))
            cur["spill_loads"] = int(m.group(3))
            # the spill column: any non-zero here means the register file could
            # not hold the live set and the kernel is paying memory traffic for
            # its own arithmetic
            cur["local_bytes"] = int(m.group(2)) + int(m.group(3))
            continue
        m = PTXV_USED.search(line)
        if m:
            cur["registers"] = int(m.group(1))
            cur["barriers"] = int(m.group(2))
    return out


def row(name: str, body: str, res: dict | None) -> dict:
    ops = opcodes(body)
    st = parse_sass(body)
    r: dict = {"kernel": name, "instructions": len(ops)}
    for cat, _, _ in CATEGORIES:
        r[cat] = st["categories"].get(cat, 0)
    r["bss_bsync_pairs"] = ops.count("BSSY")
    r["branch"] = ops.count("BRA")
    r["mufu_rcp"] = ops.count("MUFU.RCP")
    r["mufu_rsq"] = ops.count("MUFU.RSQ")
    r["isetp_u32"] = sum(1 for o in ops if o.startswith("ISETP") and "U32" in o)
    r["ldg_e_s8"] = sum(1 for o in ops if o.startswith("LDG.E") and ".S8" in o)
    r["stg_u8"] = sum(1 for o in ops if o.startswith("STG.E.U8"))
    r["global_ld_widths"] = st["global_ld_widths"]
    r["global_st_widths"] = st["global_st_widths"]
    r["any_128bit"] = st["any_128bit"]
    if res:
        for k in ("registers", "barriers", "spill_stores", "spill_loads",
                  "local_bytes", "stack_frame"):
            r[k] = res.get(k)
    else:
        for k in ("registers", "barriers", "spill_stores", "spill_loads",
                  "local_bytes", "stack_frame"):
            r[k] = None
    return r


def summarise(r: dict) -> str:
    def _n(v, width=4):
        return f"{v:{width}}" if isinstance(v, int) else f"{'?':>{width}}"

    return (f"{r['kernel'][:44]:44s} instr={r['instructions']:5d} "
            f"reg={_n(r.get('registers'))} "
            f"spill={_n(r.get('local_bytes'))} "
            f"smem={_n(r.get('shared_bytes'), 5)} "
            f"LDG={r['global_ld']:4d} STG={r['global_st']:4d} "
            f"FFMA={r['scalar_ffma']:4d} BAR={_n(r.get('barriers'), 3)} "
            f"BSSY={r['bss_bsync_pairs']:4d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sass", type=Path, required=True, help="cuobjdump -sass text")
    ap.add_argument("--res", type=Path, default=None,
                    help="ptxas -v text (registers, spills, barriers), matched by "
                         "function name")
    ap.add_argument("--select", default=None,
                    help="substring; only functions containing it are reported")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    sass = a.sass.read_text()
    res = parse_ptxas_v(a.res.read_text()) if a.res and a.res.exists() else {}
    funcs = split_functions(sass)
    names = list(funcs)
    rows = []
    for n, body in funcs.items():
        if a.select and a.select not in n:
            continue
        rows.append(row(n, body, res.get(n)))

    if a.json:
        print(json.dumps({"sass": str(a.sass), "functions": names, "rows": rows},
                         indent=2))
        return 0

    print(f"sass: {a.sass}")
    print(f"{'kernel':44s} {'instr':>10} {'reg':>5} {'spill':>5} {'smem':>5} "
          f"{'LDG':>4} {'STG':>4} {'FFMA':>4} {'BAR':>3} {'BSSY':>4}")
    for r in rows:
        print(summarise(r))
    if not rows:
        print(f"  (no function matched --select {a.select!r}; "
              f"{len(names)} functions in the file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
