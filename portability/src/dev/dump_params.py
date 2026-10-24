import importlib.util, pathlib, sys
src = pathlib.Path("/w/src/kda_ladder.py").read_text().replace(
    'if __name__ == "__main__":\n    main()\n', '')
m = type(sys)("kda_ladder")
m.__dict__["__file__"] = "/w/src/kda_ladder.py"
exec(compile(src, "/w/src/kda_ladder.py", "exec"), m.__dict__)
ks = m.load_kernels()
for name in ["chunk_kda_fwd_kernel_inter_solve_fused",
             "chunk_kda_fwd_kernel_intra_sub_chunk",
             "fused_recurrent_kda_fwd_kernel"]:
    fn = ks.get(name)
    if fn is None:
        print(name, "MISSING"); continue
    print(f"\n### {name}")
    for p in fn.params:
        print(f"   {'CE ' if p.is_constexpr else 'ARG'} {p.name}")
