#!/usr/bin/env bash
# preflight.py -- is this machine able to run our probes, and with what?
#
# Run this FIRST on any new box. It answers, in order:
#   1. what GPU is it
#   2. can TinyGrad see it
#   3. can TinyGrad actually EXECUTE on it (not just enumerate it)
#   4. is the CUDA driver discoverable, and if not, exactly where it is not
#   5. a 1-second smoke test of the real port
#
# Nothing here is specific to any one machine. It reports; it does not adapt.
import os
import subprocess
import sys


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def try_(label, fn):
    try:
        return fn()
    except Exception as e:
        print(f"  {label:<28} FAILED {type(e).__name__}: {str(e)[:100]}")
        return None


def main():
    ok = True
    hr("1. the machine")
    print(f"  hostname                 {os.uname().nodename}")
    print(f"  platform                 {sys.platform} {os.uname().machine}")
    print(f"  python                   {sys.version.split()[0]}")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    print(f"  ram                      {int(line.split()[1])/1e6:.1f} GB")
    except Exception:
        pass

    nvidia = try_("nvidia-smi", lambda: subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip())
    if nvidia:
        for line in nvidia.splitlines():
            print(f"  gpu                      {line.strip()}")
    else:
        print("  nvidia-smi               not found or failed")

    hr("2. torch (needed only for the fla/triton comparison)")
    try:
        import torch
        print(f"  torch                    {torch.__version__}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            print(f"  cuda available           yes, sm_{cap[0]}{cap[1]}")
            print(f"  bf16 tensor cores        {'yes' if cap[0] >= 8 else 'NO (sm<80)'}"
                  f"   <- fla KDA will use a non-MMA path")
            print(f"  vram                     {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        else:
            print("  cuda available           NO")
            print("  (the tinygrad probes still run; only the fla comparison is skipped)")
    except Exception as e:
        print(f"  torch                    not installed ({type(e).__name__})")
        print("  pip install torch        to enable the fla comparison")

    hr("3. can TinyGrad see a device, and which is default")
    try:
        from tinygrad import Device
        print(f"  available                {[d for d in Device._devices if d]}")
        print(f"  DEFAULT                  {Device.DEFAULT}")
    except Exception as e:
        print(f"  tinygrad                 FAILED {type(e).__name__}: {str(e)[:120]}")
        print("  pip install tinygrad")
        return 1

    hr("4. can TinyGrad EXECUTE on the default device")
    try:
        from tinygrad import Tensor
        import numpy as np
        a = Tensor(np.arange(1024, dtype=np.float32))
        b = (a * 2.0 + 1.0).sum().realize()
        want = float((np.arange(1024, dtype=np.float32) * 2 + 1).sum())
        # .item() forces a device->host copy and returns a python float. .data()
        # returns a memoryview, and float() on that raises.
        got = float(b.item())
        good = abs(got - want) <= abs(want) * 1e-4
        print(f"  elementwise + reduce     {'ok' if good else 'WRONG RESULT'} "
              f"(got {got:.1f}, want {want:.1f})")
        if not good:
            ok = False
    except Exception as e:
        print(f"  execute                  FAILED {type(e).__name__}: {str(e)[:160]}")
        print()
        print("  If this is the CUDA backend, the usual cause is that libcuda is")
        print("  not on the loader path. Find it and re-run with it exported:")
        print("      find / -name 'libcuda.so*' 2>/dev/null")
        print("      LD_LIBRARY_PATH=<that dir> python preflight.py")
        ok = False

    hr("5. the real port, 1 second of it")
    here = os.path.dirname(os.path.abspath(__file__))
    kda = os.path.join(os.path.dirname(here), "kda")
    if os.path.isdir(kda):
        sys.path.insert(0, kda)
        try:
            import numpy as np
            import tiny_kda
            import test_ref as T
            B, Tt, H, HV, K, V, C = 2, 128, 2, 2, 32, 32, 32
            q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=0, l2n=True)
            o, S = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
            import ref
            oref, _ = ref.kda_recurrent(q, k, v, g, beta)
            e = np.abs(o - oref).max() / np.abs(oref).max()
            print(f"  KDA port                 relerr {e:.3e}  "
                  f"{'ok' if e < 1e-4 else 'WRONG'}")
            if e >= 1e-4:
                ok = False
        except Exception as e:
            print(f"  KDA port                 FAILED {type(e).__name__}: {str(e)[:140]}")
            ok = False
    else:
        print(f"  (no kda/ next to this script)")

    hr("verdict")
    print(f"  {'READY' if ok else 'NOT READY -- fix the above first'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
