"""
Negative controls for sass_lint.py.

The rule: every automated check needs a deliberately-broken input that it MUST
reject, and a known-good input it MUST accept. Without both, a check that always
returns 0 is indistinguishable from a check that always passes.

Run:  .venv/bin/python tools/test_sass_lint.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_lint as S

FAILS = []

def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)

def write(text):
    fh = tempfile.NamedTemporaryFile("w", suffix=".sass.txt", delete=False)
    fh.write(text)
    fh.close()
    return fh.name

# --- control 1: a 5-hex-digit offset must be counted ------------------------
# This is bug 4. An instruction at /*10000*/ is past the first 64 KB of code and
# a {4}-quantifier regex silently drops it.
src = """
\t\tFunction : k
        /*fff0*/                   LDL.LU R13, [R1+0x43c] ;
        /*10000*/                  FFMA R46, R17, R28, R19 ;
        /*10010*/                  EXIT ;
"""
p = write(src)
st = S.analyze(p)
check("counts instructions at 5-digit offsets", st.total == 3, f"total={st.total}")
check("counts FFMA past 0xffff", st.get("FFMA") == 1, f"FFMA={st.get('FFMA')}")
os.unlink(p)

# --- control 2: spill opcodes carry width suffixes --------------------------
# STL.64 / LDL.LU / LDL.LU.64 are spills. Matching only bare STL/LDL undercounts.
src = """
\t\tFunction : k
        /*0000*/                   STL [R1], R2 ;
        /*0010*/                   STL.64 [R1], R2 ;
        /*0020*/                   LDL.LU R2, [R1] ;
        /*0030*/                   LDL.LU.64 R2, [R1] ;
        /*0040*/                   FFMA R1, R2, R3, R4 ;
        /*0050*/                   EXIT ;
"""
p = write(src)
st = S.analyze(p)
check("counts STL + STL.64 as 2 stores", st.get("STL") == 2, f"STL={st.get('STL')}")
check("counts LDL + LDL.LU + LDL.LU.64 as 2 loads", st.get("LDL") == 2, f"LDL={st.get('LDL')}")
os.unlink(p)

# --- control 3: predicate-guarded opcodes must be recognised ---------------
src = """
\t\tFunction : k
        /*0000*/              @!P0 EXIT ;
        /*0010*/              @P0 BRA 0x340 ;
        /*0020*/                   FFMA R1, R2, R3, R4 ;
"""
p = write(src)
st = S.analyze(p)
check("recognises @!P0 EXIT", st.exits == 1, f"exits={st.exits}")
os.unlink(p)

# --- control 4: a dump with NO floating-point instruction must be flagged ---
# This is bug 3. A matmul kernel cannot lack FFMA/HMMA; such a file is truncated.
src = """
\t\tFunction : k
        /*0000*/                   LOP3.LUT R2, R2, 0xfffeffff, RZ, 0xc0, !PT ;
        /*0010*/                   PRMT R3, R4, 0x6540, R5 ;
        /*0020*/                   EXIT ;
"""
p = write(src)
st = S.analyze(p)
check("flags dump with no FP instruction as INCOMPLETE", not st.complete, f"complete={st.complete}")
os.unlink(p)

# --- control 5: a healthy dump must NOT be flagged -------------------------
src = """
\t\tFunction : k
        /*0000*/                   HMMA.16816.F32 R4, R8, R12, R4 ;
        /*0010*/                   EXIT ;
"""
p = write(src)
st = S.analyze(p)
check("does not flag a healthy dump", st.complete, f"complete={st.complete}")
os.unlink(p)

print()
if FAILS:
    print(f"{len(FAILS)} control(s) FAILED: {FAILS}")
    sys.exit(1)
print("all negative controls passed")
