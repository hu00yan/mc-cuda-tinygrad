#!/usr/bin/env python3
"""Render analysis/matrix.jsonl into the two tables RESULTS.md needs."""
import json, pathlib, collections, re

HERE = pathlib.Path(__file__).parent
rows = [json.loads(l) for l in (HERE / "analysis/matrix.jsonl").read_text().splitlines()
        if l.strip() and not l.startswith('{"generated"')]

# gen/<tgarch>_<big|small>_<M>xKxN>_<indtype>_to_<acc>_<mode>__<i>.cu
TG_RE = re.compile(r"(big|small)_\d+x\d+x\d+_(float32|float16|bfloat16)_to_\w+")
# cutlass_<arch>_simt_fp32 / cutlass_<arch>_tensorop_fp16 / cutlass_sm90a / cutlass_sm100a
CL_RE = re.compile(r"cutlass_(sm[0-9a]+)_(simt_fp32|tensorop_fp16|sm90a|sm100a)?")


def dtype_of(label):
    m = TG_RE.search(label)
    if not m:
        return None
    return {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[m.group(2)], m.group(1)


def is_fp16_side(label):
    """True if this CUTLASS record is the fp16 tensor-op baseline."""
    return "tensorop" in label or label.endswith("sm90a") or label.endswith("sm100a")

CLASSES = ["fp32_fma", "tensor_core_mma_sync", "int_tensor_core", "cp_async", "tma",
           "wgmma", "tcgen05_umma", "ldmatrix", "stmatrix", "elect", "cluster",
           "fp8", "block_scaled"]
SHORT = {"fp32_fma": "FFMA", "tensor_core_mma_sync": "MMA.sync", "int_tensor_core": "IMMA",
         "cp_async": "cp.async", "tma": "TMA", "wgmma": "wgmma", "tcgen05_umma": "tcgen05",
         "ldmatrix": "ldmatrix", "stmatrix": "stmatrix", "elect": "elect",
         "cluster": "cluster", "fp8": "FP8", "block_scaled": "blk-scaled"}


def cell(rec, c):
    d = rec["classes"][c]
    if d["sass"] > 0:
        return f"**{d['sass']}**"
    if d["ptx"] > 0:
        return f"~{d['ptx']}"
    return "·"


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


ARCH_ORDER = ["sm_75", "sm_80", "sm_89", "sm_90a", "sm_100a"]

# ---- pick one representative per (arch, dtype-class) per side -------------
DT = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
tg = collections.defaultdict(list)
for r in sel(rows, side="tinygrad"):
    parts = r["label"].split("/")[-1].split("_")
    dt = DT.get(parts[-4], "?")
    shape = "big" if "_big_" in r["label"] else "small"
    tg[(r["arch"], dt, shape)].append(r)

print("=" * 100)
print("FEATURE-USAGE MATRIX  (cell = SASS instruction count; '~n' = present in PTX but folded away by ptxas; '·' = absent)")
print("=" * 100)
hdr = f"{'arch':<8} {'side':<9} {'dtype':<6} " + " ".join(f"{SHORT[c]:>11}" for c in CLASSES)
print(hdr)
print("-" * len(hdr))
for arch in ARCH_ORDER:
    for side in ["tinygrad", "cutlass"]:
        for dt in ["fp32", "fp16"]:
            rec = None
            for r in sel(rows, side=side, arch=arch):
                if side == "tinygrad":
                    d = dtype_of(r["label"])
                    if d and d[0] == dt and d[1] == "big":
                        rec = r
                        break
                else:
                    if (dt == "fp16") == is_fp16_side(r["label"]):
                        rec = r
                        break
            if rec is None:
                continue
            cells = " ".join(f"{cell(rec,c):>11}" for c in CLASSES)
            print(f"{arch:<8} {side:<9} {dt:<6} {cells}")
    print()

# ---- resource table --------------------------------------------------------
print()
print("=" * 100)
print("RESOURCES  (from `ptxas -arch=<arch> -v`)")
print("=" * 100)
h2 = f"{'arch':<8} {'side':<9} {'kernel':<44} {'regs':>5} {'spillSt':>8} {'spillLd':>8} {'stack':>7} {'stages':>7} {'maxLd':>6}"
print(h2)
print("-" * len(h2))
for arch in ARCH_ORDER:
    for side in ["tinygrad", "cutlass"]:
        for r in sel(rows, side=side, arch=arch):
            lbl = r["label"].split("/")[-1]
            print(f"{arch:<8} {side:<9} {lbl[:44]:<44} {str(r['registers']):>5} "
                  f"{str(r['spill_stores']):>8} {str(r['spill_loads']):>8} {str(r['stack_frame']):>7} "
                  f"{str(r['pipeline_stages']):>7} {str(r['max_load_bits']):>6}")
    print()
