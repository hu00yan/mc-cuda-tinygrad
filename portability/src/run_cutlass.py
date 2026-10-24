#!/usr/bin/env python3
"""
Compile the CUTLASS probes for every target arch and classify what lands in SASS.

This is the COMPILES-DEGRADED detector.  A refusal is obvious (nonzero exit).  The
case that matters is exit 0 with a *different instruction mix* than the
arch-native build -- the silent fallback: correct numbers, no diagnostic, and in
CUTLASS's case no synchronization either.

Reads cutlass_lab/cutlass read-only (another lane owns it).
Writes out/cutlass/{compile.tsv,sass.tsv,summary.txt}.
"""
import csv
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
CUTLASS = REPO / "cutlass_lab" / "cutlass"
OUT = ROOT / "out" / "cutlass"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "archlab/nvcc:12.9"
ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a",
         "sm_100", "sm_100a", "sm_120a"]

# probe name -> (extra -D flags, source file, what it is)
PROBES = {
    # cutlass/arch/memory_sm80.h cp_async_zfill / cp_async_fence / cp_async_wait.
    # Device-side gate on __CUDA_ARCH__ >= 800.  This is the 2.x multistage
    # mainloop's copy primitive, i.e. the thing every Ampere GEMM is built on.
    "cp_async": ([], "cp_async_probe.cu", "CUTLASS 2.x cp_async_zfill (multistage mainloop copy)"),
    # CuTe SM90 wgmma.  Host-side gate on CUTLASS_ARCH_MMA_SM90A_ENABLED.
    "wgmma": (["-DCL_ARG_WGMMA"], "cute_arch_probe.cu", "CuTe SM90 wgmma.mma_async"),
    # CuTe SM90 TMA descriptor load.
    "tma_load": (["-DCL_ARG_TMA"], "cute_arch_probe.cu", "CuTe SM90 TMA 1D load"),
    # CuTe SM90 1D bulk copy (the non-tensor half of TMA).
    "tma_bulk": (["-DCL_ARG_TMA_BULK"], "cute_arch_probe.cu", "CuTe SM90 bulk copy G2S"),
    # Control: wgmma with CUTLASS_ARCH_MMA_SM90A_ENABLED forced on from the command
    # line, at arches that have no wgmma.  cutlass/arch/config.h:45 additionally
    # requires __CUDA_ARCH__ == 900 && __CUDA_ARCH_FEAT_SM90_ALL, which only
    # `-arch=sm_90a` sets, so this is expected to change nothing -- and the point
    # is to measure that it really does change nothing.
    "wgmma_macro": (["-DCL_ARG_WGMMA", "-DCUTLASS_ARCH_MMA_SM90A_ENABLED"],
                    "cute_arch_probe.cu", "CuTe SM90 wgmma, arch macro forced on"),
}

INC = f"-I{CUTLASS}/include -I{CUTLASS}/tools/util/include".replace(str(REPO), "/w")

# SASS mnemonic -> class.  Ordered most-specific first; each match is counted.
SASS_CLASSES = [
    ("LDGSTS",     r"\bLDGSTS\b"),                     # cp.async, bypasses registers
    ("cp.async",   r"\bLDGSTS\b"),
    ("TMA",        r"\b(UTMALDG[A-Z0-9.]*|UBLKCP[A-Z0-9.]*|UTMALDGST[A-Z0-9.]*)\b"),
    ("WGMMMA",     r"\bWGMMMA\b"),
    ("HMMA",       r"\bHMMA\b"),
    ("MMA",        r"\bMMA\b"),
    ("TCGEN05",    r"\bTCGEN05[A-Z0-9]*\b"),
    ("LDG(staged)", r"\bLDG\.E[A-Z0-9.]*\b"),
    ("STS",        r"\bSTS(\.[A-Z0-9]+)?\b"),
    ("DEPBAR",     r"\bDEPBAR[A-Z0-9.]*\b"),
    ("CALL/BRKPT", r"\b(CALL\.ABS[A-Z0-9.]*|BRX|BPT|BRKPT|TRAP)\b"),
]

body = []
for probe, (defs, src, _desc) in PROBES.items():
    for arch in ARCHS:
        o = f"portability/out/cutlass/{probe}__{arch}.cubin"
        body.append(f'''
o={o}; rm -f $o {o}.sass
err=$(nvcc -std=c++17 -arch={arch} {" ".join(defs)} {INC} -O2 -cubin -o $o \\
      portability/src/cu/{src} 2>&1)
if [ $? -eq 0 ] && [ -s $o ]; then
  echo "CELL|{probe}|{arch}|OK|$(stat -c %s $o)||"
  nvdisasm -c $o > {o}.sass 2>/dev/null || :
else
  rm -f $o {o}.sass
  m=$(echo "$err" | grep -iE "error" | head -1 | sed 's|^.*error|error|' | tr -s ' ')
  echo "CELL|{probe}|{arch}|REJECT||"$m"|"
fi''')

SH = f"""export PATH=/usr/local/cuda-12.9/bin:$PATH
cd /w
mkdir -p portability/out/cutlass
rm -f portability/out/cutlass/sass_raw.txt
echo "TOOL|$(nvcc --version | grep release)"
{chr(10).join(body)}
echo "DONE"
"""

cells = [(p, a) for p in PROBES for a in ARCHS]
print(f"grid: {len(PROBES)} probes x {len(ARCHS)} arches = {len(cells)} cells", flush=True)
r = subprocess.run(
    ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{REPO}:/w", IMAGE,
     "sh", "-lc", SH], capture_output=True, text=True, timeout=10800, env={**os.environ})
if "DONE" not in r.stdout:
    print("STDOUT:", r.stdout[-2000:], "\nSTDERR:", r.stderr[-2000:])
    sys.exit(1)

rows, tool = [], ""
for line in r.stdout.splitlines():
    if line.startswith("TOOL|"):
        tool = line.split("|", 1)[1]
    elif line.startswith("CELL|"):
        p = (line.split("|") + [""] * 7)[:6]
        rows.append(dict(probe=p[1], arch=p[2], status=p[3],
                         cubin_bytes=p[4], detail=p[5].strip()))

with (OUT / "compile.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["probe", "arch", "status", "cubin_bytes", "detail"])
    w.writeheader()
    w.writerows(rows)

# ---- classify SASS (read the per-cell .sass files the container wrote)
raw = {}
for cf in sorted(OUT.glob("*.cubin.sass")):
    stem = cf.name[: -len(".cubin.sass")]
    probe, _, arch = stem.rpartition("__")
    raw[(probe, arch)] = cf.read_text(errors="replace")

srows = []
for (p, a), txt in raw.items():
    for label, pat in SASS_CLASSES:
        n = len(re.findall(pat, txt))
        if n:
            srows.append(dict(probe=p, arch=a, cls=label, count=n))
# the CLASSES list has LDGSTS under two labels on purpose-free accident; drop dup
seen = set()
uniq = []
for s in srows:
    k = (s["probe"], s["arch"], s["cls"])
    if k in seen:
        continue
    seen.add(k)
    uniq.append(s)
with (OUT / "sass.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["probe", "arch", "cls", "count"])
    w.writeheader()
    w.writerows(uniq)

mix = {}
for s in uniq:
    mix.setdefault(s["probe"], {}).setdefault(s["arch"], {})[s["cls"]] = s["count"]

# ---- summary table
lines = [f"# CUTLASS 4.8.0  |  {tool}  |  image {IMAGE}", ""]
lines.append(f"{'probe':<12} {'arch':<8} {'status':<8} {'bytes':>7}  SASS mix")
lines.append("-" * 100)
for p in PROBES:
    for a in ARCHS:
        r_ = next((x for x in rows if x["probe"] == p and x["arch"] == a), None)
        if not r_:
            continue
        m = mix.get(p, {}).get(a, {})
        s = ", ".join(f"{k}x{v}" for k, v in sorted(m.items())) or "-"
        lines.append(f"{p:<12} {a:<8} {r_['status']:<8} {r_['cubin_bytes']:>7}  {s}")
    lines.append("")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {OUT/'compile.tsv'} and {OUT/'sass.tsv'}")
