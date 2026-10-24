#!/usr/bin/env python3
"""sass_census.py -- the SASS table for arm B, one row per (port, knob config).

Run through `minecu_tg_knobs/sass.sh`, which is what produces the `.sass` and
`ptxas -v` files this reads. It can also be pointed at a staged tree directly.

    python minecu_tg_knobs/sass_census.py --stage DIR [--arch sm_75] [--json]

WHAT IS COUNTED, AND FROM WHAT
------------------------------
Real SASS, produced by `nvcc -arch=sm_75 --use_fast_math` in the
`archlab/nvcc:12.9` container -- the same flags, the same compiler version and
the same categoriser that `notes/19`'s matrix used, so a kernel here is
comparable to a kernel there. Not source lines: a source line is a proxy that
this study has already been bitten by.

**One categoriser, imported not re-declared.** `CATEGORIES` is imported from
`tools/archlab/inspect_ptx.py` rather than defined again, so the column names
mean the same thing here as in every other table in the repo. A second
vocabulary would make the two tables look comparable without being comparable.

THE `{4}` OFFSET BUG, AND WHY THIS FILE PRINTS BOTH NUMBERS
----------------------------------------------------------
`notes/19` §4 records a real instrument bug: the SASS offset field is 4 hex
digits for the first 4096 instructions and 5 after, so a `{4}` pattern reports
**exactly 4096 instructions** for any larger kernel. Three kernels there with
277, 1045 and 2069 global loads all came out as "4096 instructions"; the fix
was `{4,8}`.

So this file counts with the corrected pattern and ALSO counts with the broken
`{4}` one, and prints both. The reason is not tidiness: **any kernel in this
table that is at or near 4096 instructions is in the region where the bug
bites**, and a reader who sees a 4096 either has hit the bug or has a
genuinely-4096-instruction kernel, with no way to tell which from the number
alone. Printing both makes the two cases distinguishable, and makes a
regression of the bug visible instead of silent.

WHAT IS *NOT* HERE
------------------
No time and no throughput. This is a static instruction census. A static
instruction count is not a cost: it does not know about occupancy, memory
latency, warp divergence or the clock. `battery.py` runs on the card for the
timings and the two tables are read against each other rather than merged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.archlab.inspect_ptx import CATEGORIES  # noqa: E402

# The CORRECT pattern: the offset field widens from 4 to 5+ hex digits after
# 4096 instructions, so the quantifier has to admit more than exactly 4.
SASS_OK = re.compile(r"/\*[0-9a-fA-F]{4,8}\*/\s+(@!?\w+\s+)?([A-Z][A-Z0-9._]*)")
# The BROKEN one, kept only to print the comparison that makes the bug visible.
SASS_BROKEN = re.compile(r"/\*[0-9a-fA-F]{4}\*/\s+(@!?\w+\s+)?([A-Z][A-Z0-9._]*)")

# The port names `codegen.py` lays the stage out with, imported rather than
# transcribed so a renamed port cannot silently become an unmatchable suffix.
_PORTS = tuple(__import__("minecu_tg_knobs.codegen", fromlist=["PORTS"]).PORTS)

REG_RE = re.compile(r"Used (\d+) registers")
SPILL_ST_RE = re.compile(r"(\d+) bytes spill stores")
SPILL_LD_RE = re.compile(r"(\d+) bytes spill loads")
STACK_RE = re.compile(r"(\d+) bytes stack frame")
SMEM_RE = re.compile(r"(\d+) bytes smem")


def parse_sass(text: str) -> dict:
    """Instruction histogram and the shared category table, from real SASS."""
    hist: dict[str, int] = {}
    for m in SASS_OK.finditer(text):
        mn = m.group(2)
        hist[mn] = hist.get(mn, 0) + 1

    cats = {name: 0 for name, _, _ in CATEGORIES}
    for name, pattern, _ in CATEGORIES:
        rx = re.compile(pattern)
        cats[name] = sum(c for mn, c in hist.items() if rx.match(mn))

    # the naive port's four-index gather lowers to four predicated 1-byte loads
    # of ONE address, so the WIDTHS matter as much as the count
    ldg = {mn: c for mn, c in hist.items() if mn.startswith("LDG")}

    def widths(pred):
        got: dict[str, int] = {}
        for mn, c in hist.items():
            if pred(mn):
                w = re.search(r"\.(\d+)$", mn)
                k = w.group(1) if w else "scalar"
                got[k] = got.get(k, 0) + c
        return dict(sorted(got.items(), key=lambda kv: -kv[1]))

    return {
        "total": sum(hist.values()),
        "total_broken_pattern": len(SASS_BROKEN.findall(text)),
        "categories": cats,
        "ldg_widths": widths(lambda mn: mn.startswith("LDG")),
        "stg_widths": widths(lambda mn: mn.startswith("STG")),
        "histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
    }


def parse_ptxas(text: str) -> dict:
    """Register / spill / stack-frame census, from `ptxas -v`."""
    def g(rx, cast=int):
        m = re.search(rx, text)
        return cast(m.group(1)) if m else None
    return {
        # EVERY field is a `re.search` over the WHOLE file, because `ptxas -v`
        # prints a report for every function it assembles and a `.ptx` here can
        # contain more than one. `re.search` takes the FIRST match, which is
        # the wrong one whenever the first function is not the one the census
        # identified as hot.
        #
        # This bit: the naive port's `default` row reported 70 registers and
        # ZERO spill against 36 384 SASS, which contradicts the same port's own
        # `NOLOCALS=1` row reporting 168 registers and 7 460 B of spill on a
        # kernel that differs by 40 instructions. Both cannot be that kernel.
        # The cause is that tinygrad's emitted `.ptx` carries the whole graph,
        # so the first function ptxas reports is whichever one the PTX lists
        # first, and it need not be the DDA kernel.
        #
        # So the report is parsed per function. The actual layout ptxas emits
        # puts the resources on TWO lines, not one:
        #
        #   ptxas info : Function properties for E_512_2_8_8_4_4
        #       5472 bytes stack frame, 7460 bytes spill stores, 13472 bytes spill loads
        #   ptxas info : Used 168 registers, used 0 barriers, ...
        #
        # and a whole-file `re.search` for "Used (\d+) registers" would find
        # whichever comes first. The broken behaviour was concrete: the naive
        # port's default row reported **70 registers and zero spill** against
        # 36 384 SASS, while the same port's own NOLOCALS=1 row reported 168 and
        # 7 460 B on a kernel differing by 40 instructions. Both cannot be that
        # kernel. The 70/0/0 came from a different entry function in the same
        # PTX, borrowed by a regex.
        "reports": _per_function(text),
        "registers": None, "spill_stores": None, "spill_loads": None,
        "stack_frame": None, "smem": None,
        "_note": "see per-function reports; the top-level fields are None by "
                 "design, because a whole-file regex attributes one function's "
                 "resources to another",
    }


def _per_function(text: str) -> list[dict]:
    """Every `ptxas -v` resource report in the file, paired with its function.

    ptxas spreads one function's resources over two lines, so both are read:

        ptxas info : Function properties for E_512_2_8_8_4_4
            5472 bytes stack frame, 7460 bytes spill stores, 13472 bytes spill loads
        ptxas info : Used 168 registers, used 0 barriers, ...

    A `.ptx` from one `.cu` normally has one entry function, but not always --
    and "normally" is exactly the kind of assumption that produces a confident
    wrong number, so the pairing is by name and a caller that wants a specific
    function's numbers looks them up rather than taking the first.
    """
    out: list[dict] = []
    cur: str | None = None
    props: dict = {}
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur, props = m.group(1), {}
            continue
        if cur is None:
            continue
        for rx, key in ((r"(\d+) bytes stack frame", "stack_frame"),
                        (r"(\d+) bytes spill stores", "spill_stores"),
                        (r"(\d+) bytes spill loads", "spill_loads")):
            m2 = re.search(rx, line)
            if m2:
                props[key] = int(m2.group(1))
        m = re.search(r"Used (\d+) registers", line)
        if m:
            out.append({"function": cur, "registers": int(m.group(1)), **props})
            cur, props = None, {}
    return out


def _resources_for(parsed: dict, sass_text: str) -> dict:
    """The register/spill report for THIS kernel, not for whichever one ptxas
    happened to report first.

    `ptxas -v` emits one block per entry function in the PTX, and a whole-file
    regex takes the first. That is how the naive port's default row came to
    claim 70 registers and zero spill against 36 384 SASS, while the same port's
    own `NOLOCALS=1` row claimed 168 and 7 460 B on a kernel 40 instructions
    away -- both cannot describe it. The reports are therefore matched by
    function name against the name in this kernel's own SASS header
    (`Function : E_512_2_8_8_4_4`).

    A kernel with no matching report returns all-`None` and says so. Returning
    a neighbour's numbers is the failure this exists to prevent; returning
    nothing is merely uninformative.
    """
    reports = parsed.get("reports") or []
    if not reports:
        return {"registers": None, "spill_stores": None, "spill_loads": None,
                "stack_frame": None, "why": "no ptxas -v report for this file"}
    if len(reports) == 1:
        return {k: reports[0].get(k) for k in
                ("registers", "spill_stores", "spill_loads", "stack_frame")} | \
               {"function": reports[0]["function"]}

    m = re.search(r"Function\s*:\s*(\S+)", sass_text)
    name = m.group(1) if m else None
    for r in reports:
        if r["function"] == name:
            return {k: r.get(k) for k in
                    ("registers", "spill_stores", "spill_loads", "stack_frame")} | \
                   {"function": name}
    return {"registers": None, "spill_stores": None, "spill_loads": None,
            "stack_frame": None,
            "why": f"ptxas reported {len(reports)} functions "
                   f"({[r['function'] for r in reports]}) and this kernel is "
                   f"{name!r}; no report is attributed to it rather than "
                   f"borrowing another function's numbers"}


def identify(rows: list[dict], label: str) -> dict | None:
    """The DDA kernel in one (port, knob) row: the longest SASS body.

    For SASS the criterion is direct rather than indirect: the DDA kernel is the
    one whose 32 inlined step bodies make it by far the longest, and the
    candidates here are all the kernels of one graph. The winner's identity and
    margin over the runner-up are reported, so a row where the margin is small
    is visible as "the identification is shaky here" rather than asserted.
    """
    cands = [r for r in rows if r["sass_total"]]
    if not cands:
        return None
    cands.sort(key=lambda r: -r["sass_total"])
    best = cands[0]
    best["margin_over_next"] = (best["sass_total"] - cands[1]["sass_total"]
                                if len(cands) > 1 else None)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, help="staged tree with out/*.sass")
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--json", default="", help="write the table here")
    a = ap.parse_args(argv)

    stage = Path(a.stage)
    outdir = stage / "out"
    if not outdir.is_dir():
        print(f"no out/ in {stage}", file=sys.stderr)
        return 3

    groups: dict[str, list[dict]] = {}
    for sass_file in sorted(outdir.glob("*.sass")):
        base = sass_file.stem
        text = sass_file.read_text(errors="replace")
        parsed = parse_sass(text)
        ptx = outdir / f"{base}.ptxas"
        res = parse_ptxas(ptx.read_text(errors="replace")) if ptx.exists() else {}

        # out/<label>_<port>_kernel<NNN>.sass. The port is stripped from the
        # RIGHT, longest-name-first, because `lean` is a suffix of
        # `lean-no-flat` and `lean-no-unsigned` -- matching `lean` first would
        # leave a stem of `default_lean-no-flat` and mislabel the whole group.
        # A port that does not match any known name is an error rather than a
        # `?`: a mislabelled row in a table whose whole purpose is attribution
        # is worse than a failed run.
        stem, port = base, None
        for p in sorted(_PORTS, key=len, reverse=True):
            suffix = f"_{p}_kernel"
            if suffix in stem:
                head, _, tail = stem.partition(suffix)
                stem, port = head, p
                kidx = int(tail) if tail.isdigit() else -1
                break
        if port is None:
            raise AssertionError(
                f"cannot tell which port {base!r} belongs to: none of "
                f"{sorted(_PORTS)} is a suffix. The stage was not laid out by "
                f"codegen.py, or a port was renamed without updating this.")
        groups.setdefault((stem, port), []).append(
            {"kernel": kidx, "sass_total": parsed["total"],
             "sass_total_broken_pattern": parsed["total_broken_pattern"],
             "categories": parsed["categories"], "ldg_widths": parsed["ldg_widths"],
             "stg_widths": parsed["stg_widths"],
             "res": _resources_for(res, text)})

    table = []
    for (label, port), rows in sorted(groups.items()):
        hot = identify(rows, label)
        table.append({"label": label, "port": port,
                      "kernel_count": len(rows),
                      "total_instructions": sum(r["sass_total"] for r in rows),
                      "hot": hot})

    _print(table, a.arch)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"arch": a.arch, "note": "real SASS from nvcc 12.9 in the archlab "
                                      "container, counted with the corrected "
                                      "offset pattern; the broken-pattern column "
                                      "is printed so a 4096 can be recognised as "
                                      "the instrument bug rather than a result.",
             "rows": table}, indent=1))
    return 0


def _print(table: list[dict], arch: str) -> None:
    print(f"  arch {arch}. `total` is the whole graph; `hot` is the DDA kernel,")
    print("  identified as the longest SASS body in its own graph, with the")
    print("  margin over the runner-up shown. `4096?` flags a kernel sitting")
    print("  exactly on the offset-field boundary where the `{4}` bug bites.")
    print()
    hdr = (f"  {'port':<17} {'knob configuration':<32} {'kern':>5} "
           f"{'total':>8} {'hot':>8} {'margin':>7} {'regs':>5} {'spill st':>9} "
           f"{'spill ld':>9} {'stack':>7} {'LDG':>6} {'LDL':>6} {'4096?':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in table:
        h = r["hot"] or {}
        res = h.get("res") or {}
        c = h.get("categories") or {}
        near = "YES" if abs(h.get("sass_total", 0) - 4096) <= 8 else ""
        print(f"  {r['port']:<17} {r['label']:<32} {r['kernel_count']:>5} "
              f"{r['total_instructions']:>8} {h.get('sass_total', 0):>8} "
              f"{(h.get('margin_over_next') if h.get('margin_over_next') is not None else '-'):>7} "
              f"{(res.get('registers') if res.get('registers') is not None else '-'):>5} "
              f"{(res.get('spill_stores') if res.get('spill_stores') is not None else '-'):>9} "
              f"{(res.get('spill_loads') if res.get('spill_loads') is not None else '-'):>9} "
              f"{(res.get('stack_frame') if res.get('stack_frame') is not None else '-'):>7} "
              f"{c.get('global_ld', 0):>6} {c.get('local_ldst', 0):>6} {near:>6}")


if __name__ == "__main__":
    sys.exit(main())
