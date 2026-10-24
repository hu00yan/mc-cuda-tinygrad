#!/usr/bin/env python3
"""
Render every measured cell as markdown, straight from the data files.

Nothing in MATRIX.md is hand-transcribed: this script reads the TSVs/JSON that the
runners produced and emits the tables.  Re-run it after any runner to refresh.

    python3 src/build_matrix.py            # writes out/tables.md
    python3 src/build_matrix.py --print    # also dump to stdout
"""
import csv
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

RUNG_ORDER = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_87", "sm_89",
              "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_101", "sm_100f",
              "sm_103", "sm_120", "sm_120a", "sm_121"]
SHORT = {a: a.replace("sm_", "") for a in RUNG_ORDER}

# marker name in MATRIX.src.md  ->  section title used as the key into `blocks`
MARKER_OF = {
    "LADDER": "PTX witness ladder (ptxas 12.9, 544 cells)",
    "CUTLASS": "CUTLASS 4.8.0 by arch",
    "TRITON": "Triton 3.4.0 by arch",
    "KDA": "flash-linear-attention KDA kernels",
    "FA": "FlashAttention by arch",
}


def rung_key(a):
    return int(a.split("_")[1].rstrip("af"))


def tick(s):
    return {"OK": "Y", "REJECT": "-", "": "?"}.get(s, s)


def code(s):
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"^error\s*:\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return f"`{s[:150]}`"


# ----------------------------------------------------------------- ladder
def table_ladder():
    rows = list(csv.DictReader((OUT / "ladder.tsv").open()))
    grid, notes = {}, {}
    for r in rows:
        w = r["witness"]
        grid.setdefault(w, {})[r["arch"]] = r["status"]
        if r["status"] == "REJECT" and r["detail"]:
            notes.setdefault(r["detail"], []).append(f"{w.split('@')[0]} @ {r['arch']}")
    arches = [a for a in RUNG_ORDER if any(a in v for v in grid.values())]

    # order witnesses by the lowest rung that accepts them
    def first_ok(w):
        oks = [rung_key(a) for a, s in grid[w].items() if s == "OK"]
        return min(oks) if oks else 999

    out = ["| mechanism (PTX witness) | " + " | ".join(SHORT[a] for a in arches) + " |",
           "|---|" + "---|" * len(arches)]
    for w in sorted(grid, key=lambda x: (first_ok(x), x)):
        out.append("| `" + w + "` | "
                   + " | ".join(tick(grid[w].get(a, "")) for a in arches) + " |")

    out.append("")
    out.append("Y = ptxas assembled it for that `-arch`. The witness file name "
               "records the arch it was written for; every other column is a "
               "genuine cross-target attempt with the *same* PTX.")
    out.append("")
    out.append("**Distinct rejection reasons** (verbatim, deduplicated):")
    out.append("")
    out.append("| cells | diagnostic |")
    out.append("|---|---|")
    for d, who in sorted(notes.items(), key=lambda kv: -len(kv[1])):
        d = re.sub(r"\s+", " ", d)
        out.append(f"| {len(set(who))} | `{d}` |")
    return "\n".join(out)


# ----------------------------------------------------------------- cutlass
def table_cutlass():
    comp = list(csv.DictReader((OUT / "cutlass" / "compile.tsv").open()))
    sass = list(csv.DictReader((OUT / "cutlass" / "sass.tsv").open()))
    mix = {}
    for s in sass:
        mix.setdefault(s["probe"], {}).setdefault(s["arch"], {})[s["cls"]] = int(s["count"])
    probes, seen = [], set()
    for c in comp:
        if c["probe"] not in seen:
            seen.add(c["probe"])
            probes.append(c["probe"])
    arches = []
    for c in comp:
        if c["arch"] not in arches:
            arches.append(c["arch"])
    arches.sort(key=rung_key)

    out = ["| CUTLASS 4.8.0 code path | " + " | ".join(SHORT[a] for a in arches)
           + " | cubin bytes (first / last) |", "|---|" + "---|" * len(arches) + "---|"]
    for p in probes:
        cells, sizes = [], []
        for a in arches:
            r = next((x for x in comp if x["probe"] == p and x["arch"] == a), None)
            if r is None:
                cells.append("?")
                continue
            m = mix.get(p, {}).get(a, {})
            cls = [k for k in ("WGMMMA", "TCGEN05", "HMMA", "LDGSTS", "TMA",
                               "LDG(staged)", "STS", "DEPBAR", "CALL/BRKPT")
                   if k in m]
            cells.append(f"{tick(r['status'])}<br>{'+'.join(cls)}" if cls else tick(r["status"]))
            sizes.append(int(r["cubin_bytes"] or 0))
        span = f"{min(sizes)} / {max(sizes)}" if sizes else "-"
        out.append(f"| `{p}` | " + " | ".join(cells) + f" | {span} |")
    out.append("")
    out.append("All 50 cells exit 0. `CALL/BRKPT` means the cubin contains a "
               "runtime trap (`CUTE_INVALID_CONTROL_PATH` = `assert(0); printf(x)`) "
               "and no MMA/TMA instruction at all.")
    return "\n".join(out)


# ----------------------------------------------------------------- triton
def _triton_block(js, title, kernels=None):
    d = json.load((OUT / "triton" / js).open())
    cells = d["cells"]
    caps = []
    for c in cells:
        if c["cap"] is not None and c["cap"] not in caps:
            caps.append(c["cap"])
    caps.sort()
    knames = kernels or []
    for c in cells:
        if c["kernel"] not in knames:
            knames.append(c["kernel"])

    lines = [f"### {title}", "",
             f"_triton {d['triton_version']}, no GPU; only `GPUTarget.arch` varies._", ""]
    for kn in knames:
        rows = [c for c in cells if c["kernel"] == kn]
        if not rows:
            continue
        lines += [f"**{kn}**", "",
                  "| cap | " + " | ".join(f"sm_{c}" for c in caps) + " |",
                  "|---|" + "---|" * len(caps)]
        # instruction mix per cell
        def mixof(c):
            m = c.get("ptx_insts") or {}
            if not m:
                return "compiles; no MMA/TMA/async-copy class present"
            return " ".join(f"{k}x{v}" for k, v in m.items())
        for c in caps:
            r = next((x for x in rows if x["cap"] == c), None)
            if r is None:
                lines.append(f"| sm_{c} | ? |")
            elif r["status"] != "OK":
                lines.append(f"| sm_{c} | **REJECT** |")
            else:
                lines.append(f"| sm_{c} | {mixof(r)} |")
        lines.append("")
        rej = [(cap, r["error"]) for cap in caps
               for r in rows if r["cap"] == cap and r["status"] != "OK"]
        if rej:
            lines.append("Refusals:")
            lines.append("")
            for cap, e in rej:
                lines.append(f"- `sm_{cap}`: {code(e)}")
            lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- fa
def table_fa():
    rows = list(csv.DictReader((OUT / "fa" / "fa_ladder.tsv").open()))
    out = ["| arch | status | cubin bytes | SASS |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['arch']} | {r['status']} | {r['cubin_bytes'] or '-'} | "
                   f"{r['sass_mix']} |")
    return "\n".join(out)


DUP_HEADINGS = (
    "### Triton mechanism ladder (synthetic kernels)\n\n"
    "_triton 3.4.0, no GPU; only `GPUTarget.arch` varies._\n\n",
    "### real upstream source\n\n"
    "_triton 3.4.0, no GPU; only `GPUTarget.arch` varies._\n\n",
)


def strip_dups(text):
    for d in DUP_HEADINGS:
        text = text.replace(d, "")
    return text


def main():
    parts = ["# Generated tables",
             "",
             "Produced by `src/build_matrix.py` from `out/*.tsv` and `out/triton/*.json`.",
             "Do not hand-edit: re-run the runners and this script.", ""]
    sections = [
        ("PTX witness ladder (ptxas 12.9, 544 cells)", table_ladder),
        ("CUTLASS 4.8.0 by arch", table_cutlass),
        ("Triton 3.4.0 by arch", lambda: _triton_block(
            "triton_ladder.json", "Triton mechanism ladder (synthetic kernels)")),
        ("flash-linear-attention KDA kernels", lambda: _triton_block(
            "kda_ladder.json", "real upstream source")),
        ("FlashAttention by arch", table_fa),
    ]
    blocks = {}
    for name, fn in sections:
        try:
            blocks[name] = fn()
        except Exception as e:
            blocks[name] = f"*(unavailable: {type(e).__name__}: {e})*"
        parts += [f"## {name}", "", blocks[name], ""]
    txt = "\n".join(parts) + "\n"
    (OUT / "tables.md").write_text(txt)
    print(f"wrote {OUT/'tables.md'} ({len(txt)} bytes)")

    # --extract: reverse the injection, producing the hand-written source template
    # with the <!--TABLES:X--> markers back in place.  MATRIX.src.md is the
    # editable artefact; MATRIX.md is generated from it.
    argv = __import__("sys").argv
    if "--extract" in argv:
        cur = (ROOT / "MATRIX.md").read_text()
        src_path = ROOT / "MATRIX.src.md"
        for name, fn in sections:
            try:
                blk = strip_dups(fn())
            except Exception:
                blk = ""
            start = cur.find(blk)
            if start < 0:
                print(f"  (extract: block for {name!r} not found verbatim; "
                      f"leaving it in place)")
                continue
            key = [k for k, v in MARKER_OF.items() if v == name][0]
            cur = cur[:start] + f"<!--TABLES:{key}-->" + cur[start + len(blk):]
        src_path.write_text(cur)
        print(f"wrote {src_path} (markers restored)")
        return

    # --inject: splice the tables into MATRIX.src.md -> MATRIX.md
    if "--inject" in argv:
        src_path = ROOT / "MATRIX.src.md"
        if not src_path.exists():
            raise SystemExit("MATRIX.src.md missing; run --extract first")
        text = src_path.read_text()
        n = 0
        for key, name in MARKER_OF.items():
            pat = f"<!--TABLES:{key}-->"
            if pat in text:
                text = text.replace(pat, blocks[name])
                n += 1
        # _triton_block emits its own heading; the section heading in the template
        # already says it, so drop the duplicate.
        text = strip_dups(text)
        (ROOT / "MATRIX.md").write_text(text)
        print(f"injected {n} blocks into {ROOT/'MATRIX.md'} from {src_path.name}")

    if "--print" in argv:
        print(txt)


if __name__ == "__main__":
    main()
