#!/usr/bin/env python3
"""
Disassemble the cubins produced by run_ptxas.py and classify which instruction
classes actually survive into SASS for each arch.

This is the measurement that distinguishes COMPILES-SAME from COMPILES-DEGRADED:
ptxas will happily assemble `wgmma`-shaped PTX down to `HMMA` on an older arch
*if and only if* the PTX did not actually contain a wgmma.  Where the *source* has
an arch-conditional (the CUTLASS / Triton case) the substitution happens in the
front end, so we read SASS to see what came out.
"""
import csv
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTDIR = pathlib.Path(sys.argv[1])
TSV = pathlib.Path(sys.argv[2])
IMAGE = sys.argv[3] if len(sys.argv) > 3 else "archlab/nvcc:12.9"
NVPRUNE = "/usr/local/cuda-12.9/bin/nvdisasm"

# SASS / PTX mnemonic -> (class label, regex)
CLASSES = {
    "wgmma":      r"\bWGMMMA\b",
    "tcgen05":    r"\bTCGEN05\w*",
    "HMMA-f16":   r"\bHMMA\.[0-9A-Z\.]*F16",
    "HMMA-other": r"\bHMMA\.(?!.*F16)",
    "IMMA":       r"\bIMMA\b",
    "QMMA":       r"\bQMMA\b",
    "OMMA":       r"\bOMMA\b",
    "cp.async":   r"\bLDGSTS\b",
    "TMA(bulk)":  r"\b(UTMA|UBLKCP|CP)\b.*|\bUTMALDG\b",
    "ldmatrix":   r"\bLDSM\b",
    "stmatrix":   r"\bSTS\.[MT]\b",
    "bar.sync":   r"\bBAR\.SYNC\b",
    "mbarrier":   r"\b(MBAR|ARRIVE|PROD|TRYWAIT|ELECT)\b",
    "cluster":    r"\b(MAPA|UMAPA|CLUSTERMAP)\b|\bR2UR\b",
}


def disasm(cubin):
    cmd = (f'docker run --rm --platform linux/amd64 -v "{ROOT}:/w" {IMAGE} '
           f'sh -lc "{NVPRUNE} -c /w/{cubin.relative_to(ROOT)} 2>/dev/null"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    return r.stdout


out = []
for row in csv.DictReader(TSV.open()):
    if row["status"] != "OK":
        continue
    cubin = OUTDIR / f"{row['witness']}@{row['arch']}.cubin"
    if not cubin.exists():
        continue
    txt = disasm(cubin)
    for label, pat in CLASSES.items():
        n = len(re.findall(pat, txt, re.I))
        if n:
            out.append(dict(witness=row["witness"], arch=row["arch"], cls=label, count=n))

with OUTDIR.joinpath("sass_classes.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["witness", "arch", "cls", "count"])
    w.writeheader()
    w.writerows(out)

# pivot for reading
piv = {}
for r in out:
    piv.setdefault(r["witness"], {}).setdefault(r["arch"], {})[r["cls"]] = r["count"]
for wit in sorted(piv):
    print(f"\n== {wit}")
    for arch in sorted(piv[wit], key=lambda a: int(a.split("_")[1].rstrip("af"))):
        cs = ", ".join(f"{k}x{v}" for k, v in sorted(piv[wit][arch].items()))
        print(f"   {arch:<8} {cs}")
