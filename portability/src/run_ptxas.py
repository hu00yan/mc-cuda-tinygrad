#!/usr/bin/env python3
"""
Assemble every PTX witness against every target arch with ptxas.

Experimental design: the PTX file is FIXED and only `ptxas -arch=` varies.  That
isolates the single question "does this instruction class exist on arch X" from
every other variable (source language, front end, inlining, register allocation).
No GPU is involved; ptxas is a host binary.

One container invocation for the whole grid -- starting docker per cell costs
~1.5 s and the grid is 33 x 15.

    usage: run_ptxas.py [image]
    writes: out/ladder.tsv, out/ladder_<arch>.cubin, out/ladder.toolchain.txt
"""
import csv
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PTXDIR = ROOT / "ptx"
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "archlab/nvcc:12.9"
PTXAS = "/usr/local/cuda-12.9/bin/ptxas"

ARCHES = [
    "sm_70", "sm_75", "sm_80", "sm_86", "sm_87", "sm_89",
    "sm_90", "sm_90a",
    "sm_100", "sm_100a", "sm_101", "sm_100f", "sm_103",
    "sm_120", "sm_120a", "sm_121",
]

SCRIPT = r"""
export PATH=/usr/local/cuda-12.9/bin:$PATH
cd /w
rm -rf out/ladder_cubins && mkdir -p out/ladder_cubins
ver="$(ptxas --version | grep release)"
echo "TOOLCHAIN|$ver"
for f in ptx/*.ptx; do
  w=$(basename "$f" .ptx)
  neg=""
  case "$w" in *'!neg') neg=1; w=${w%!neg};; esac
  for a in __ARCHES__; do
    o="out/ladder_cubins/${w}__${a}.cubin"
    err="$(ptxas -arch=$a "$f" -o "$o" 2>&1)"
    if [ $? -eq 0 ]; then
      sz=$(stat -c %s "$o" 2>/dev/null || echo 0)
      echo "CELL|$w|$a|OK|$sz|"
    else
      rm -f "$o"
      msg=$(echo "$err" | grep -iE "error" | head -1 | sed 's|^.*error|error|' | tr -s ' ')
      echo "CELL|$w|$a|REJECT||${msg}|${neg}"
    fi
  done
done
echo "DONE"
""".replace("__ARCHES__", " ".join(ARCHES))

print(f"grid: {len(list(PTXDIR.glob('*.ptx')))} witnesses x {len(ARCHES)} arches "
      f"= {len(list(PTXDIR.glob('*.ptx'))) * len(ARCHES)} cells", flush=True)
r = subprocess.run(
    ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/w", IMAGE,
     "sh", "-lc", SCRIPT],
    capture_output=True, text=True, timeout=5400)
if "DONE" not in r.stdout:
    print(r.stdout[-3000:], r.stderr[-3000:])

rows, toolchain = [], ""
for line in r.stdout.splitlines():
    if line.startswith("TOOLCHAIN|"):
        toolchain = line.split("|", 1)[1].strip()
    elif line.startswith("CELL|"):
        parts = line.split("|")
        if len(parts) < 6:
            continue
        _, w, a, status, size, detail, neg = (parts + ["", ""])[:7]
        rows.append(dict(witness=w, arch=a, status=status,
                         cubin_bytes=size, detail=detail.strip(),
                         negative_control=bool(neg)))

with (OUT / "ladder.tsv").open("w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=["witness", "arch", "status",
                                         "cubin_bytes", "detail", "negative_control"])
    wcsv.writeheader()
    wcsv.writerows(rows)
(OUT / "ladder.toolchain.txt").write_text(toolchain + "\n" + IMAGE + "\n")

ok = sum(r["status"] == "OK" for r in rows)
print(f"toolchain: {toolchain}")
print(f"cells: {len(rows)}  assembled: {ok}  rejected: {len(rows) - ok}")
print(f"wrote {OUT / 'ladder.tsv'}")
