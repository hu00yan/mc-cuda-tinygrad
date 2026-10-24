"""
Structurally different THIRD implementation of the SASS counter.

WHY THIS EXISTS
---------------
`sass_lint.py` uses one regex to parse an instruction line. When @Orion wrote his
own independent count, he used *another* regex -- and it had the *same* offset-width
assumption, so it agreed with a wrong answer. Two implementations that share a bad
assumption are not two paths; they are one path run twice.

This file deliberately shares nothing with the regex approach. It tokenizes each
line with `str.split()` and classifies by the token following the offset marker,
skipping predicate guards. No `re` module, no quantifiers, no field widths.

    .venv/bin/python tools/sass_lint_independent.py aot16_cc75_BV64_w4_s2

Agreement between this and `sass_lint.py` is real triangulation. Disagreement
means at least one is wrong -- investigate rather than pick the convenient number.

VERIFIED 2026-10-24: agrees with sass_lint.py on every field for
aot16_cc75_BV64_w4_s2 (16952/1651/2684/33) and aot16_cc80_BV64_w4_s2 (2768/64/63/3).
"""
import glob, os, sys

def classify(path):
    counts = {"instr":0, "STL":0, "LDL":0, "EX2":0, "HMMA":0, "FFMA":0, "EXIT":0}
    for raw in open(path, errors="replace"):
        s = raw.strip()
        if not s.startswith("/*"):
            continue
        # split on whitespace; first token is /*offset*/
        parts = s.split()
        if not parts or not parts[0].startswith("/*") or not parts[0].endswith("*/"):
            continue
        rest = parts[1:]
        if not rest: continue
        # skip predicate guard tokens (@P0 / @!P0)
        i = 0
        while i < len(rest) and rest[i].startswith("@"):
            i += 1
        if i >= len(rest): continue
        op = rest[i].rstrip(";")
        counts["instr"] += 1
        base = op.split(".")[0]
        if base == "STL": counts["STL"] += 1
        elif base == "LDL": counts["LDL"] += 1
        elif op.startswith("MUFU.EX2"): counts["EX2"] += 1
        elif base in ("HMMA","HGMMA"): counts["HMMA"] += 1
        elif base == "FFMA": counts["FFMA"] += 1
        elif base == "EXIT": counts["EXIT"] += 1
    return counts

for name in sys.argv[1:]:
    f = glob.glob(f'out/**/{name}.sass.txt', recursive=True)
    if not f: print(f"{name}: not found"); continue
    c = classify(f[0])
    print(f"{name}: total={c['instr']} STL={c['STL']} LDL={c['LDL']} EX2={c['EX2']} HMMA={c['HMMA']} FFMA={c['FFMA']} EXIT={c['EXIT']}")
