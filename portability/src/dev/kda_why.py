import pathlib, sys
src = pathlib.Path("/w/src/kda_ladder.py").read_text().replace(
    'if __name__ == "__main__":\n    main()\n', '')
m = type(sys)("kda_ladder"); m.__dict__["__file__"] = "/w/src/kda_ladder.py"
exec(compile(src, "/w/src/kda_ladder.py", "exec"), m.__dict__)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as tcompile
ks = m.load_kernels()
name = sys.argv[1]
fn = ks[name]
sig, cex, missing = m.build(fn)
print("SIG:", sig); print("CEX:", cex); print("MISSING:", missing)
try:
    tcompile(ASTSource(fn=fn, signature=sig, constexprs=cex),
             target=GPUTarget("cuda", 90, 32), options={"num_warps": 4, "num_stages": 2})
    print("OK")
except Exception as e:
    for ln in str(e).splitlines():
        if ln.strip():
            print("  ", ln[:220])
