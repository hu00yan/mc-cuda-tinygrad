#!/usr/bin/env python3
"""
FA2 / FA3 by arch.

Two subjects, two very different situations, and the matrix must say so:

FA3 (Dao-AILab/flash-attention, main branch, hopper/) builds exclusively on
  sm_90a features: wgmma, TMA descriptors, mbarrier, warp specialisation.  Its own
  static_switch.h maps Arch<90 to a constexpr Arch=80 branch, but hopper/ contains
  no SM80 kernel to land in, and the features it does use are gated by
  cutlass/arch/config.h on `__CUDA_ARCH__ == 900 && __CUDA_ARCH_FEAT_SM90_ALL` --
  i.e. exactly `-arch=sm_90a`.  So the FA3 cells are REFUSED, and the refusal is
  attributable to specific PTX feature gates that THIS study measures directly
  (out/ladder.tsv) rather than by assertion.

FA2 (v2.8.3) degrades silently instead: csrc/flash_attn/src/kernel_traits.h
  chooses the MMA atom AND the pipeline depth from __CUDA_ARCH__ with no
  diagnostic.  Building FA2 for real needs libtorch headers, which this network
  cannot fetch, so the guard is transcribed into src/cu/fa2_traits_probe.cu and
  the *arch requirements* of the two instruction shapes are measured with the
  probe's own cubins.  That limitation is recorded in the matrix.

Outputs out/fa/fa_ladder.tsv.
"""
import csv
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
OUT = ROOT / "out" / "fa"
OUT.mkdir(parents=True, exist_ok=True)
IMAGE = "archlab/nvcc:12.9"
ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a",
         "sm_100", "sm_100a", "sm_120a"]

MIX = [
    ("WGMMMA",   r"\bWGMMMA\b"),
    ("TCGEN05",  r"\bTCGEN05[A-Z0-9]*\b"),
    ("HMMA",     r"\bHMMA\b"),
    ("MMA",      r"\bMMA\b"),
    ("LDGSTS",   r"\bLDGSTS\b"),
    ("UTMALDG",  r"\bUTMALDG[A-Z0-9.]*\b"),
    ("UBLKCP",   r"\bUBLKCP[A-Z0-9.]*\b"),
    ("LDSM",     r"\bLDSM[A-Z0-9.]*\b"),
    ("DEPBAR",   r"\bDEPBAR[A-Z0-9.]*\b"),
    ("BRKPT",    r"\b(BRKPT|CALL\.ABS[A-Z0-9.]*|BPT)\b"),
]

PROBES = {"fa2_traits": "src/cu/fa2_traits_probe.cu"}

SH = """export PATH=/usr/local/cuda-12.9/bin:$PATH
cd /w/portability
mkdir -p out/fa
echo "TOOL|$(nvcc --version | grep release)"
__CELLS__
echo "DONE"
"""
body = []
for probe, rel in PROBES.items():
    for arch in ARCHS:
        o = f"out/fa/{probe}__{arch}.cubin"
        body.append(f'''
o={o}; rm -f $o $o.sass
err=$(nvcc -std=c++17 -arch={arch} -O2 -cubin -o $o {rel} 2>&1)
if [ $? -eq 0 ] && [ -s $o ]; then
  echo "CELL|{probe}|{arch}|OK|$(stat -c %s $o)||"
  nvdisasm -c $o > $o.sass 2>/dev/null || :
else
  rm -f $o
  m=$(echo "$err" | grep -iE "error" | head -1 | sed 's|^.*error|error|' | tr -s ' ')
  echo "CELL|{probe}|{arch}|REJECT||$m|"
fi''')
SH = SH.replace("__CELLS__", "\n".join(body))

r = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64",
                    "-v", f"{REPO}:/w", IMAGE, "sh", "-lc", SH],
                   capture_output=True, text=True, timeout=7200, env={**os.environ})
if "DONE" not in r.stdout:
    print(r.stdout[-1500:], r.stderr[-1500:])
    raise SystemExit(1)

rows, tool = [], ""
for line in r.stdout.splitlines():
    if line.startswith("TOOL|"):
        tool = line.split("|", 1)[1]
    elif line.startswith("CELL|"):
        p = (line.split("|") + [""] * 7)[:6]
        rows.append(dict(probe=p[1], arch=p[2], status=p[3],
                         cubin_bytes=p[4], detail=p[5].strip()))

for row in rows:
    cf = OUT / f"{row['probe']}__{row['arch']}.cubin.sass"
    mix = {}
    if cf.exists():
        txt = cf.read_text(errors="replace")
        for label, pat in MIX:
            n = len(re.findall(pat, txt))
            if n:
                mix[label] = n
    row["sass_mix"] = ", ".join(f"{k}x{v}" for k, v in sorted(mix.items())) or "-"

with (OUT / "fa_ladder.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["probe", "arch", "status", "cubin_bytes",
                                     "sass_mix", "detail"])
    w.writeheader()
    w.writerows(rows)

print(f"# {tool}")
print(f"{'arch':<9} {'status':<8} {'bytes':>7}  SASS mix")
for row in rows:
    print(f"{row['arch']:<9} {row['status']:<8} {row['cubin_bytes']:>7}  {row['sass_mix']}"
          + (f"   {row['detail'][:70]}" if row['detail'] else ""))
print(f"\nwrote {OUT/'fa_ladder.tsv'}")
