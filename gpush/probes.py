"""probes.py -- the one script that runs on the far side, and its parser.

There is exactly one device probe in gpush and both backends use it, because
two probes would mean two device reports and therefore two chances to disagree
about what machine a number came from.

WHAT IT REPORTS
    GPU name, total memory, compute capability, driver version, CUDA runtime
    version as nvidia-smi states it, plus the kernel/OS and python version. All
    of it read from the machine. Nothing is assumed.

WHAT IT DOES NOT DO
    It does not decide whether the device is suitable. A T4 is sm_75 and has no
    bf16 tensor cores; that is a fact about the study, and gpush's job ends at
    reporting `75` in the provenance record. If a backend silently upgraded
    "no bf16 tensor cores" into "cannot run", every sm_75 result would vanish
    and the comparison the study exists to make would quietly disappear too.
    So: surface the number, interpret nothing.

    The one judgement gpush does make is binary and unavoidable: *is there a
    GPU at all*. A CPU-only Colab session is not a slower GPU run, it is a
    different machine, and a measurement taken there is not a GPU measurement.
    `--allow-cpu` records it honestly as CPU instead.
"""
from __future__ import annotations

import json

# Pure-python: no numpy, no torch, no shell. It has to run on a machine whose
# only guarantee is that python3 exists, because that is what a fresh Colab
# session is and what a Kaggle CPU fallback is.
DEVICE_PROBE = r'''
import json, os, platform, subprocess, sys

def sh(*a, t=20):
    try:
        return subprocess.run(list(a), capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception:
        return ""

def smi(q):
    # --query-gpu with csv is the only stable interface across driver versions.
    out = sh("nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits")
    return [l.strip() for l in out.splitlines() if l.strip()]

dev = {
    "gpu": False, "name": "", "memory_mb": None, "compute_cap": "unknown",
    "driver": "", "cuda_runtime": "", "count": 0, "reason": "",
}

rows = smi("name,memory.total,compute_cap,driver_version")
if rows:
    f = [x.strip() for x in rows[0].split(",")]
    dev["count"] = len(rows)
    dev["gpu"] = True
    dev["name"] = f[0]
    try:
        dev["memory_mb"] = int(float(f[1]))
    except Exception:
        pass
    dev["compute_cap"] = f[2]
    dev["driver"] = f[3]
    head = sh("nvidia-smi")
    for line in head.splitlines():
        if "CUDA Version" in line:
            dev["cuda_runtime"] = line.split("CUDA Version:")[1].split()[0].strip(" :")
            break
else:
    # Distinguish "no nvidia-smi" from "nvidia-smi found no device": the first
    # is a broken box and the second is a CPU-only session, and they mean
    # different things to whoever reads the provenance record.
    dev["reason"] = ("nvidia-smi not present on PATH"
                    if sh("which", "nvidia-smi") == ""
                    else "nvidia-smi ran and reported no device")

nvcc = sh("nvcc", "--version")
for line in nvcc.splitlines():
    if "release" in line:
        dev["nvcc"] = line.split("release")[1].split(",")[0].strip()
        break

dev["kernel"] = platform.release()
dev["machine"] = platform.machine()
dev["os"] = platform.platform()
dev["python"] = sys.version.split()[0]
dev["hostname"] = platform.node()
dev["cpu_count"] = os.cpu_count()
print("GPUSH_DEVICE_JSON:" + json.dumps(dev))
'''


def parse_device(text: str) -> dict:
    """Find the probe's JSON line in a stream of remote output.

    Robust on purpose: the remote's stdout also contains whatever the notebook
    or shell printed around it, and a device report that fails to parse is a
    device report that must be treated as *unknown*, never as CPU.
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("GPUSH_DEVICE_JSON:"):
            try:
                d = json.loads(line.split(":", 1)[1])
                if isinstance(d, dict):
                    return d
            except ValueError:
                continue
    return {"gpu": False, "name": "", "count": 0,
            "reason": "device probe produced no parseable output "
                      "(the remote may have died before it ran)"}


def describe(d: dict) -> str:
    if not d:
        return "device UNKNOWN (no probe ran)"
    if d.get("gpu"):
        return (f"{d.get('name', '?')}  {d.get('memory_mb', '?')} MB  "
                f"compute capability {d.get('compute_cap', 'unknown')}  "
                f"driver {d.get('driver', '?')}")
    return f"NO GPU ({d.get('reason', 'none reported')})"


def capability_note(d: dict) -> str:
    """State the capability, and only the capability.

    A number, and the plain fact of whether it is below 80. What that means for
    any particular kernel is not gpush's call and is not printed as a verdict.
    """
    if not d.get("gpu"):
        return ""
    cap = str(d.get("compute_cap", "")).strip()
    if not cap or cap == "unknown":
        return f"compute capability {cap or 'unknown'}"
    major = cap.split(".")[0]
    below80 = major.isdigit() and int(major) < 8
    return f"compute capability sm_{cap.replace('.', '')}" + (
        " (below sm_80: no bf16 tensor cores; recorded, not interpreted)"
        if below80 else "")
