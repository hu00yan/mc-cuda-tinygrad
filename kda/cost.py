"""
Cost model for the KDA chunked forward pass.

Run:  /Users/huyan00/mycode/mc-cuda-tinygrad/.venv/bin/python kda/cost.py

Every number quoted in NOTES.md comes from this file.  Nothing here is measured
on a GPU -- there is no KDA GPU kernel in this repo yet -- so the FLOP and byte
counts are ANALYTIC, derived from the shapes in ref.py.  Where a count is an
estimate rather than an exact formula it says so.

Default config is the one NOTES.md quotes:  B=4, T=4096, H=32, K=V=128, C=64.
"""

import sys

# the config NOTES.md quotes
B, T, H, K, V, C = 4, 4096, 32, 128, 128, 64
HV = H            # no grouped-value attention in the default config
NT = T // C
ES = 2            # bytes per element in the kernels' working dtype (bf16/fp16)
ES32 = 4          # bytes for anything that must stay fp32


def mb(n):
    return n / 1024 ** 2


def gib(n):
    return n / 1024 ** 3


# ---------------------------------------------------------------------------
# step table.  `per(b,n,h)` FLOPs are for ONE (batch, chunk, head); multiply by
# B*NT*HV for the whole batch.  `bytes` is DRAM traffic for the whole batch,
# counting one write + one read of the tensor (bf16), which is the optimistic
# case for a fused kernel and the realistic case for a naive one.
# ---------------------------------------------------------------------------
def build_table():
    BH = B * HV            # batch x value-heads
    BNH = B * NT * HV      # batch x chunk x value-heads
    rows = []

    def add(step, shape, flops_bn, byts, cls, note="", ref_only=False):
        rows.append(dict(step=step, shape=shape, flops=flops_bn,
                         bytes=byts, cls=cls, note=note, ref_only=ref_only))

    # --- 0. gate activation + 1. within-chunk cumsum ----------------------
    e_g = B * T * HV * K
    add("0  gate act.  -exp(A_log)*softplus(g+dt_bias)", "[B,T,HV,K]",
        0, 3 * 2 * e_g * ES, "elementwise", "exp, softplus, 2 fma", False)
    add("1  g_cum = cumsum(g) over chunk", "[B,T,HV,K]",
        0, 2 * 2 * e_g * ES, "scan", "C-long axis, not T", False)

    # --- 2. Z = (K . D) K^T  and its decay tensor --------------------------
    add("2a decay tensor D = exp(gc[i]-gc[j]), masked", "[B,NT,HV,C,C,K]",
        0, 2 * BNH * C * C * K * ES, "elementwise",
        "O(C^2 K) transient. ref.py only; kernels fold it into 2b.", True)
    add("2b Z = (K . D) K^T", "[B,NT,HV,C,C]",
        2 * C * C * K, 2 * BNH * C * C * ES, "matmul",
        "the k-vs-k gram. tensor-coreable", False)

    # --- 3. UT transform: forward substitution -----------------------------
    add("3  UT: forward substitution + I, * Diag(beta)", "[B,NT,HV,C,C]",
        C * C * (C - 1), 2 * BNH * C * C * ES, "SEQUENTIAL",
        "C-1 dependent row updates. NOT a matmul.", False)

    # --- 4/5. w and u ------------------------------------------------------
    add("4  w = M (Gamma . K)", "[B,NT,HV,C,K]",
        2 * C * C * K, 2 * BNH * C * K * ES, "matmul", "tensor-coreable", False)
    add("5  u = M V", "[B,NT,HV,C,V]",
        2 * C * C * V, 2 * BNH * C * V * ES, "matmul", "tensor-coreable", False)

    # --- 6. the inter-chunk scan, one pass per chunk -----------------------
    add("6a v_new = u - w S", "[B,NT,HV,C,V]",
        2 * C * K * V, 3 * BH * C * V * ES, "matmul", "S from accumulator", False)
    add("6b Aqk = (Q . D) K^T  causal", "[B,NT,HV,C,C]",
        2 * C * C * K, 2 * BNH * C * C * ES, "matmul", "tensor-coreable", False)
    add("6c o = (Gamma . Q) S", "[B,T,HV,V]",
        2 * C * K * V, 2 * B * T * HV * V * ES, "matmul", "tensor-coreable", False)
    add("6d o += Aqk v_new", "[B,T,HV,V]",
        2 * C * C * V, 0, "matmul", "fused into 6c, no extra traffic", False)
    add("6e S = Diag(gam^C) S + (K . D)^T v_new", "[B,HV,K,V]",
        2 * C * K * V, 2 * BH * K * V * ES32, "matmul+ew",
        "state is T-INDEPENDENT. lives in registers", False)
    return rows


def blocked_ut_flops():
    """What the 16x16-blocked forward substitution in the Triton kernel costs,
    versus the naive C-step one.  Diagonal blocks are forward-substituted row by
    row; the 6 off-diagonal inverse blocks are 3 matmuls each (batched)."""
    BC = 16
    NC = C // BC
    # within each diagonal block: sum over rows of 2*BC*i, i=0..BC-1
    within = NC * 2 * BC * (BC * (BC - 1) / 2)
    # off-diagonal: (NC*(NC-1)//2) blocks, 3 matmuls of BC x BC x BC each
    off = (NC * (NC - 1) // 2) * 3 * 2 * BC ** 3
    return within + off, C * C * (C - 1)


# published dense bf16 tensor-core throughput / HBM bandwidth
RIDGE = {"H100 SXM": (989.5e12, 3.35e12), "A100 80GB": (312e12, 1.55e12)}


def sweep_C():
    """FLOPs / traffic / intensity as a function of chunk size.

    The C^2 terms (the five intra-chunk matmuls and the forward substitution)
    cost O(T*C*K) in total, so they scale LINEARLY in C, while the C*K*V terms
    (the three inter-chunk matmuls) are C-independent.  So bigger C buys
    arithmetic intensity only by adding work."""
    global C, NT
    keep = C
    print(f"{'C':>5}{'NT':>6}{'GFLOP':>9}{'traffic MB':>12}{'FLOP/byte':>12}"
          f"{'  C^2 share':>12}")
    for c in (16, 32, 64, 128, 256):
        C, NT = c, T // c
        rows = build_table()
        f = sum(r["flops"] for r in rows) * B * NT * HV
        by = sum(r["bytes"] for r in rows if not r["ref_only"])
        # the C^2-proportional part: the 5 intra-chunk matmuls + the UT transform
        c2 = (2 * C * C * K * 3 + 2 * C * C * V * 2 + C * C * (C - 1))
        print(f"{c:>5}{NT:>6}{f/1e9:>9.2f}{mb(by):>12.1f}{f/by:>12.2f}"
              f"{c2*B*NT*HV/f*100:>11.1f}%")
    C, NT = keep, T // keep


def main():
    rows = build_table()
    krows = [r for r in rows if not r["ref_only"]]
    tot_flops = sum(r["flops"] for r in rows) * B * NT * HV
    kbytes = sum(r["bytes"] for r in krows)
    rbytes = sum(r["bytes"] for r in rows if r["ref_only"])

    print("=" * 100)
    print(f"KDA chunked forward cost model   B={B} T={T} H={HV} K={K} V={V} "
          f"C={C} NT={NT}  (bf16 working dtype)")
    print("=" * 100)
    print(f"{'step':<50}{'GFLOP':>8}{'MB':>8}{'FLOP/B':>9}  class")
    print("-" * 100)
    for r in rows:
        gf = r["flops"] * B * NT * HV / 1e9
        tag = " [ref-only]" if r["ref_only"] else ""
        ai = (gf * 1e9 / r["bytes"]) if r["bytes"] else float("inf")
        ais = "  inf" if ai == float("inf") else f"{ai:5.0f}"
        print(f"{r['step']:<50}{gf:>8.2f}{mb(r['bytes']):>8.1f}{ais:>9}  "
              f"{r['cls']}{tag}")
    print("-" * 100)
    print(f"{'TOTAL (all rows)':<50}{tot_flops/1e9:>8.2f}{mb(kbytes+rbytes):>8.1f}")
    print(f"{'TOTAL (kernel rows only)':<50}{tot_flops/1e9:>8.2f}{mb(kbytes):>8.1f}")
    print()
    print("  FLOP/B = 9 means the step is pure memory traffic.  A ridge point of")
    print("  ~295 FLOP/B (H100) / ~201 (A100) means anything under that is")
    print("  bandwidth-limited, so the whole game is not touching DRAM.")
    print("  'inf' = the step costs zero extra traffic only IF it is fused into")
    print("  its neighbour; materialised it would be [B,NT,HV,C,V] = 128 MB.")
    print()
    print(f"total forward FLOPs          {tot_flops/1e9:10.2f} GFLOP")
    print(f"traffic, kernel rows         {mb(kbytes):10.1f} MB  ({gib(kbytes):.2f} GiB)")
    print(f"traffic, ref-only rows       {mb(rbytes):10.1f} MB  (never built by a kernel)")
    ai = tot_flops / kbytes
    print(f"arithmetic intensity         {ai:10.2f} FLOP/byte  (kernel rows)")
    for name, (tf, bw) in RIDGE.items():
        rp = tf / bw
        print(f"  {name} bf16 ridge point {rp:7.0f} FLOP/byte  ->  this operator is "
              f"{rp/ai:5.1f}x MEMORY BOUND")
    print()
    sweep_C()
    print()

    # --- compare against the paper's own formula (Eq. 13) ------------------
    per_head_paper = 6 * T * K ** 2 + 3 * T * C * K + T * C ** 2
    mine = tot_flops / (B * H)
    print(f"paper Eq.13 FLOPs/head        {per_head_paper:10,}  "
          f"-> {per_head_paper*B*H/1e9:.2f} GFLOP")
    print(f"this count   FLOPs/head      {mine:10,.0f}  -> {tot_flops/1e9:.2f} GFLOP")
    print(f"ratio                          {mine/per_head_paper:10.3f}")
    print("  Counting every matmul at 2 FLOP/MAC, there are FIVE C x C x d_h")
    print("  intra-chunk matmuls (Z, w, u, Aqk, Aqk@v_new), not three.  Eq. 13's")
    print("  3TC*d_h also counts 1 FLOP/MAC while its 6Td_h^2 term counts 2.")
    print("  Sizing hardware off Eq. 13 undershoots by ~1.45x.")
    print()

    # --- the UT transform: naive vs the kernel's 16x16 blocking ------------
    blk, naive = blocked_ut_flops()
    print(f"UT fwd substitution, naive C-step:     {naive:>9,} FLOP/chunk/head")
    print(f"UT fwd substitution, 16x16 blocked:    {blk:>9,.0f}  "
          f"({naive/blk:.2f}x cheaper, and the blocked form is parallel)")
    print()

    # --- what a naive full materialisation would cost ----------------------
    h = B * NT * HV * K * V
    # --- what fusing across the scan is worth ------------------------------
    v_new = B * NT * HV * C * V
    unfused = kbytes + 2 * v_new * ES
    floor_b = 5 * B * T * HV * V * ES     # read q,k,v,g + write o (all bf16)
    rp = RIDGE["H100 SXM"][0] / RIDGE["H100 SXM"][1]
    print("--- what fusing across the scan boundary is worth ---")
    print(f"v_new [B,NT,HV,C,V] spilled (write+read)   {mb(2*v_new*ES):8.1f} MB")
    print(f"  intensity, fused across the scan (base)   {tot_flops/kbytes:8.1f} FLOP/B")
    print(f"  intensity, v_new spilled to DRAM           {tot_flops/unfused:8.1f} FLOP/B")
    print(f"  absolute floor: read q,k,v,g + write o     {mb(floor_b):8.1f} MB"
          f"  -> {tot_flops/floor_b:6.1f} FLOP/B")
    print(f"  so even a PERFECTLY fused KDA is still {rp/(tot_flops/floor_b):.1f}x "
          f"below the H100 ridge point ({rp:.0f}).")
    print(f"  The gate + cumsum alone (steps 0,1) are "
          f"{mb(sum(r['bytes'] for r in rows if r['step'].startswith(('0 ','1 ')))):.0f}"
          f" MB = {sum(r['bytes'] for r in rows if r['step'].startswith(('0 ','1 ')))/kbytes*100:.0f}%"
          f" of the traffic and 0 of the FLOPs.")
    print()

    print("--- the traps (what a naive tensorised port would materialise) ---")
    print(f"h, per-chunk states [B,NT,HV,K,V]  {mb(h*ES):9.1f} MB   "
          f"({h*ES/1e9:.2f} GB)  <- only avoidable by fusing the scan")
    tt = B * H * T * T
    print(f"full score matrix  [B,H,T,T]        {mb(tt*ES):9.1f} MB   "
          f"({tt*ES/1e9:.2f} GB)  <- softmax-attention shape; KDA must NOT do this")
    print(f"  per head: {mb(T*T*ES):.1f} MB, i.e. O(T^2).  A compiler that fuses the")
    print("  chunk loop into one graph never has to build it -- but it also")
    print("  cannot parallelise the loop, which is the actual difficulty.")
    print()
    decay = B * NT * HV * C * C * K
    print(f"reference-only decay tensor [B,NT,HV,C,C,K]  {mb(decay*ES):.1f} MB  "
          f"(ref.py builds this; kernels do not)")
    print()
    print("bytes for one [B,T,HV,V] bf16 tensor:      "
          f"{mb(B*T*HV*V*ES):.1f} MB")
    print("bytes for one [B,T,HV,K] bf16 tensor:      "
          f"{mb(B*T*HV*K*ES):.1f} MB")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
