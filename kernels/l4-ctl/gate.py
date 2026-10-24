#!/usr/bin/env python3
"""Which card did this kernel actually land on, and does the study's gate pass?

One question, answered from the device, with nothing taken on faith.

The `machine_shape` field is NOT evidence: `kaggle kernels push --accelerator`
accepts any string, so a wrong name is silently accepted and you get a different
card with no error. A push of a deliberately bogus name in this project was
accepted and ran. So the only admissible evidence of which card this is, is what
the card says about itself.

The gate being tested is the one in kda/baseline_pin.py CARD_GATE, whose L4 row
reads: cc major 8, 101 376 B, and -- for the study's own validity criterion --
166 912 B, which an L4 does NOT meet. What an L4 does meet is the ADA threshold of
101 376 B with `>=`, which is what stops `fla` degrading. Those are two different
questions and this prints both.
"""
import json, subprocess, sys, os

out = {}
def rec(k, v):
    out[k] = v
    print(f"{k:.<34} {v}", flush=True)

print("=" * 72, flush=True)
print("SECTION 0 -- card identity, four independent readers", flush=True)
print("=" * 72, flush=True)

# reader 1: the driver
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    rec("nvidia-smi", q.stdout.strip() or q.stderr.strip()[:120])
except Exception as e:
    rec("nvidia-smi", f"FAILED {type(e).__name__}: {e}")

# reader 2: CUDA runtime through torch
try:
    import torch
    rec("torch cuda arch list", torch.cuda.get_arch_list())
    rec("torch capability", torch.cuda.get_device_capability(0))
    p = torch.cuda.get_device_properties(0)
    rec("torch device name", p.name)
    rec("torch shared_memory_per_block", getattr(p, "shared_memory_per_block", "n/a"))
    rec("torch shared_memory_per_block_optin", getattr(p, "shared_memory_per_block_optin", "n/a"))
    rec("torch major.minor", f"{p.major}.{p.minor}")
except Exception as e:
    rec("torch", f"not importable/usable: {type(e).__name__}: {e}")

# reader 3: the driver API directly, via ctypes -- no torch in the loop
try:
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    val = ctypes.c_int(0)
    # The driver API's CUdevice_attribute enum. The first version of this probe
    # used 97, 75, 16 and LABELLED 75 as "MaxSharedMemoryPerBlock" and 16 as
    # "ComputeCapabilityMinor". Both labels were wrong, and 16 is not a compute
    # capability attribute at all -- it returned 40, a plausible-looking number
    # that means nothing. A mislabelled attribute is worse than a missing one,
    # because it survives a review. The mapping below is the enum's:
    #     8 = CUDA_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK          (non-optin)
    #    75 = CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    #    76 = CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    #    97 = CUDA_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    rc = lib.cuDeviceGetAttribute(ctypes.byref(val), 97, 0)
    rec("cuDeviceGetAttribute(97) rc", rc)
    rec("cuDeviceMaxSharedMemPerBlockOptin", val.value if rc == 0 else "unreadable")
    for attr, nm in ((8, "MaxSharedMemoryPerBlock (non-optin)"),
                     (75, "ComputeCapabilityMajor"),
                     (76, "ComputeCapabilityMinor")):
        v = ctypes.c_int(0)
        rc = lib.cuDeviceGetAttribute(ctypes.byref(v), attr, 0)
        rec(f"cuDeviceGetAttribute({attr}) {nm}", v.value if rc == 0 else f"rc={rc}")
except Exception as e:
    rec("libcuda ctypes", f"FAILED {type(e).__name__}: {e}")

# reader 4: triton, if present, which reads its own compiled-for arch
try:
    import triton
    rec("triton.__version__", getattr(triton, "__version__", "?"))
    import triton.backends
    try:
        tgt = triton.runtime.driver.active.get_current_target()
        rec("triton current target", tgt)
    except Exception as e:
        rec("triton target", f"{type(e).__name__}: {e}")
except Exception as e:
    rec("triton", f"not importable: {type(e).__name__}")

# ---- the gate, decided from what was just read, not from what was requested ----
print(flush=True)
print("=" * 72, flush=True)
print("SECTION 1 -- the gate, computed from the readings above", flush=True)
print("=" * 72, flush=True)

ccmaj = out.get("torch major.minor")
if isinstance(ccmaj, str) and "." in ccmaj:
    ccmaj = int(ccmaj.split(".")[0])
smem = out.get("cuDeviceMaxSharedMemPerBlockOptin")
if not isinstance(smem, int):
    smem = out.get("torch shared_memory_per_block_optin")
smem = smem if isinstance(smem, int) else None

rec("resolved cc major", ccmaj)
rec("resolved smem optin", smem)
rec("is L4 / cc 8.9", ccmaj == 89)
rec("fla ADA gate 101376 (>= holds)", (smem >= 101376) if smem else "unknown")
rec("study's own 166912 criterion", (smem >= 166912) if smem else "unknown")
rec("tf32 survives (cc major >= 8)", (ccmaj >= 8) if isinstance(ccmaj, int) else "unknown")

# the honest bottom line
if ccmaj == 89 and smem and smem >= 101376:
    verdict = "L4 CONFIRMED -- fla would not degrade; the KDA arm becomes measurable"
elif ccmaj == 75:
    verdict = "T4 -- fla degrades to 4 of 12 configs; the KDA arm stays unquantified"
else:
    verdict = f"UNEXPECTED CARD cc={ccmaj} smem={smem}; decide by hand, do not assume"
rec("VERDICT", verdict)

os.makedirs("out", exist_ok=True)
with open("out/l4_ctl.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nwrote out/l4_ctl.json", flush=True)
print(f"\nVERDICT: {verdict}", flush=True)
