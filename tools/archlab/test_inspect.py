#!/usr/bin/env python3
"""
Self-test for inspect_ptx.py's parsers and occupancy model.

Run:  python3 tools/archlab/test_inspect.py
No GPU, no container, no network -- it only exercises the pure-Python parts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_ptx import occupancy, parse_ptxas, parse_sass, parse_ptx_source  # noqa: E402

FAILED = []


def eq(name, got, want):
    ok = got == want
    if not ok:
        FAILED.append(f"{name}: got {got!r}, want {want!r}")
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")


def approx(name, got, want, tol=0.01):
    ok = got is not None and abs(got - want) <= tol
    if not ok:
        FAILED.append(f"{name}: got {got!r}, want ~{want}")
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r} (want ~{want})")


print("\n-- ptxas -v parsing (the real sm_90a WGMMA report) --")
v = parse_ptxas("""ptxas info    : 11 bytes gmem
ptxas info    : Compiling entry function 'k' for 'sm_90a'
ptxas info    : Function properties for k
    376 bytes stack frame, 380 bytes spill stores, 372 bytes spill loads
ptxas info    : Used 255 registers, used 1 barriers, 376 bytes cumulative stack size
""")
eq("registers", v["registers"], 255)
eq("spill_stores", v["spill_stores"], 380)
eq("spill_loads", v["spill_loads"], 372)
eq("stack_frame", v["stack_frame"], 376)
eq("barriers", v["barriers"], 1)
eq("error", v["error"], None)

print("\n-- ptxas static smem is reported, dynamic is not --")
s = parse_ptxas("    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
                "ptxas info    : Used 30 registers, used 0 barriers, 2432 bytes smem\n")
eq("static smem", s["smem_bytes"], 2432)

print("\n-- a ptxas ERROR must be surfaced, not silently read as zeros --")
e = parse_ptxas("ptxas /x.ptx, line 5; error   : Unsupported .version 9.3; current version is '8.8'\n"
                "ptxas fatal   : Ptx assembly aborted due to errors\n")
assert e["error"] is not None, "error was swallowed"
eq("error captured", "Unsupported" in e["error"], True)

print("\n-- SASS categorising: LDGSTS must not be double-counted as a global load --")
sass = parse_sass("""
/*0010*/              LDGSTS.E.BYPASS.128 [R9], desc[UR14][R4.64] ;
/*0020*/                   HGMMA.64x8x16.F32.BF16 R24, gdesc[UR4].negA, R24, !UP0, gsb0 ;
/*0030*/                   WARPGROUP.ARRIVE ;
/*0040*/              @!P0 LDGSTS.E.BYPASS.128 [R7], desc[UR14][R2.64] ;
/*0050*/                       STG.E.128 [R2], R4 ;
/*0060*/                        LDG.E.U16 R8, [R4.64] ;
/*0070*/                      LDS.U16 R9, [R2] ;
/*0080*/                        FFMA R10, R8, R9, R10 ;
""")
c = sass["categories"]
eq("cp_async counts both LDGSTS", c["cp_async"], 2)
eq("global_ld excludes LDGSTS", c["global_ld"], 1)
eq("warpgroup_mma", c["warpgroup_mma"], 1)
eq("warpgroup_sync", c["warpgroup_sync"], 1)
eq("global_st", c["global_st"], 1)
eq("shared_ld", c["shared_ld"], 1)
eq("scalar_ffma", c["scalar_ffma"], 1)
eq("uses 128-bit", sass["any_128bit"] > 0, True)

print("\n-- PTX source facts --")
probe = Path(__file__).resolve().parents[2] / "out" / "archlab" / "probe.ptx"
if probe.exists():
    info = parse_ptx_source(probe)
    eq("probe has dynamic smem", info["has_dynamic_smem"], True)
    eq("probe block size not pinned", info["block_size_from_ptx"], None)
else:
    print("  skip  (out/archlab/probe.ptx not present)")

print("\n-- occupancy: H100 (sm_90a) = 65536 regs/SM, 2048 threads/SM, 64 warps --")
# reg=128 at 256-thread blocks: 128*256 = 32768, so only 2 blocks fit -> 16 of
# 64 warps. 25% is correct, and is the point: "few registers" is not a
# guarantee of full occupancy.
approx("255 regs / 128 thr (FLA KDA)", occupancy(255, 128, 1024, "sm_90a")[0], 12.5)
approx("30 regs / 128 thr (probe.ptx)", occupancy(30, 128, 2432, "sm_90a")[0], 100.0)
approx("31 regs / 128 thr (naive.ptx)", occupancy(31, 128, 2432, "sm_90a")[0], 100.0)
approx("128 regs / 256 thr", occupancy(128, 256, 4096, "sm_90a")[0], 25.0)
approx("64 regs / 256 thr", occupancy(64, 256, 4096, "sm_90a")[0], 50.0)
approx("32 regs / 1024 thr", occupancy(32, 1024, 4096, "sm_90a")[0], 100.0)
eq("255/128 binding constraint", occupancy(255, 128, 1024, "sm_90a")[1], "register-file")
eq("64/256 binding constraint", occupancy(64, 256, 4096, "sm_90a")[1], "register-file")
# shared memory can bind: 227KB/SM, 48KB blocks -> 4 blocks
approx("smem-limited 4096B blocks", occupancy(32, 128, 49152, "sm_90a")[0], 25.0)
eq("smem binding constraint", occupancy(32, 128, 49152, "sm_90a")[1], "shared-memory")

print("\n-- occupancy must REFUSE rather than guess --")
eq("no block size -> unknown", occupancy(30, None, 2432, "sm_90a")[0], None)
eq("no register count -> unknown", occupancy(None, 128, 2432, "sm_90a")[0], None)
eq("block not warp multiple -> unknown", occupancy(30, 100, 2432, "sm_90a")[0], None)
eq("arch outside table -> unknown", occupancy(30, 128, 2432, "sm_999a")[0], None)
# dynamic smem: must return a number but LABEL it an upper bound
pct, binding, notes = occupancy(30, 128, None, "sm_90a")
eq("dynamic smem still estimates", pct is not None, True)
eq("dynamic smem flagged as upper bound",
   any("upper bound" in n for n in notes), True)

print("\n-- a kernel that cannot launch at the given block size --")
pct, binding, notes = occupancy(255, 1024, 0, "sm_90a")
eq("255 regs x 1024 thr -> 0%", pct, 0.0)
eq("...reported as unlaunchable", "cannot be launched" in notes[-1], True)

print()
if FAILED:
    print(f"FAILED {len(FAILED)}:")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
