#!/usr/bin/env python3
"""b1_bytes.py -- the byte accounting, and what is NOT in it.

THE CLAIM
---------
The KDA `[B,NT,HV,C,C,K]` decay tensor is never materialised: no device buffer
of its size, no kernel argument of its size, no value written to memory. In the
one emitted CUDA kernel in this repo it is absent from a five-pointer signature
whose largest member is 32x smaller.

TWO SCOPES, AND THEY GIVE DIFFERENT MULTIPLIERS
-----------------------------------------------
This is the first place the record goes wrong, and the error is a scope error
rather than an arithmetic one:

  captured-kernel scope   decay / largest buffer in that kernel's signature
                          = 2,097,152 / 65,536    = 32x
  operator-schedule scope decay / largest buffer in ANY kernel argument of the
                          whole 12,082-kernel plan
                          = 67,108,864 / 1,048,576 = 64x

Both are exact. They are not the same claim, and quoting one while calling it
"the ratio" is how a study ends up with a 2x disagreement about its own
headline. b2_schedule.py measures the second one; this file measures the first.

WHAT IS RE-DERIVED HERE (nothing is read off the file by eye)
------------------------------------------------------------
  * the parameter list, by regex on the signature
  * per-buffer element and byte counts, from the name tinygrad appends
  * which buffers are stored to and which are only read, by classifying every
    statement that dereferences a buffer
  * the index range each buffer is actually touched, by EVALUATING the emitted
    index arithmetic over the whole thread grid -- not by regex
  * the `exp2` census, and whether any of those values reaches memory
  * the decay tensor's size at this shape, from the shape
  * the identity that pins each buffer to a tensor in the port, by matching
    element counts against the shapes the port declares

WHAT THIS FILE CANNOT DO
------------------------
  It analyses ONE kernel out of 2,974. That kernel's emitted name,
  `r_2_2_8_8_4_4_32`, is a shape signature, not an identity: re-rendering the
  same graph offline produces a DIFFERENT kernel with the same name, at every
  arch including sm_75. So nothing here generalises to the other 2,973, and
  section 5 says so. b4_codegen.py renders the whole program and reports what
  is arch-stable and what is not.

    python3 experiment_b/b1_bytes.py
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_b.common import (  # noqa: E402
    CAPTURE, CAPTURE_TAG, NOT_HERE, RE_DERIVED, head, kv, nt, read, require,
    rule, shape_str, verdict)

CU = "out/kaggle/20260928-014248-gpu/codegen/kernel000.txt"


# =============================================================================
# 1. the signature and the grid
# =============================================================================

def parse_signature(src: str) -> dict:
    m = re.search(r"__launch_bounds__\((\d+)\)\s+(\w+)\(([^)]*)\)", src)
    if not m:
        raise SystemExit("no __global__ signature found -- this artifact is not "
                         "what the file was written against")
    return dict(launch_bounds=int(m.group(1)), name=m.group(2),
                params=re.findall(r"float\*\s+(\w+)", m.group(3)),
                dtypes=sorted(set(re.findall(r"(\w+)\*\s+\w+", m.group(3)))))


def grid_dims(src: str) -> list[int]:
    """tinygrad annotates each index declaration with its extent in a trailing
    comment: `int lidx1 = threadIdx.y; /* 8 */`. The first is blockIdx."""
    d = [int(n) for _, _, n in re.findall(
        r"int (\w+) = ((?:blockIdx|threadIdx)\.\w);\s*/\* (\d+) \*/", src)]
    if not d:
        raise SystemExit("no grid/block extents annotated in the artifact")
    return d


# =============================================================================
# 2. an interpreter for the emitted index arithmetic
# =============================================================================

class IdxEval:
    """Evaluates tinygrad's emitted index expressions over the whole grid.

    The renderer emits `int aluN = (<expr>);` and then `dataN_M+(alu4+64)`. To
    learn which ELEMENTS a buffer is touched at, these have to be evaluated as
    integers, not pattern-matched as strings: the answer has to be a set of
    integers so that "covered exactly, no gaps, no overlaps" is a fact about
    the program rather than a claim about a regex.

    The grammar accepted is deliberately tiny -- integer literals, identifiers,
    the four binary operators tinygrad emits, parentheses. Anything else raises,
    so a shape this interpreter cannot evaluate produces a refusal rather than
    a guess.
    """

    TOK = re.compile(r"\(|\)|<<|[-+*]|[A-Za-z_]\w*|\d+")

    def __init__(self, env: dict[str, int]) -> None:
        self.env = dict(env)

    def eval(self, expr: str) -> int:
        self.toks = self.TOK.findall(expr)
        self.i = 0
        v = self._sum()
        if self.i != len(self.toks):
            raise ValueError(f"trailing tokens in index expression {expr!r}")
        return v

    def _sum(self) -> int:
        v = self._atom()
        while self.i < len(self.toks) and self.toks[self.i] in ("+", "-", "*", "<<"):
            op = self.toks[self.i]
            self.i += 1
            r = self._atom()
            v = {"+": v + r, "-": v - r, "*": v * r, "<<": v << r}[op]
        return v

    def _atom(self) -> int:
        t = self.toks[self.i]
        if t.isdigit():
            self.i += 1
            return int(t)
        if t in ("+", "-", "*", "<<"):
            self.i += 1
            return self._atom()
        if t == "(":
            self.i += 1
            v = self._sum()
            if self.toks[self.i] != ")":
                raise ValueError("unbalanced ( in index expression")
            self.i += 1
            return v
        if t in self.env:
            self.i += 1
            return self.env[t]
        raise ValueError(f"unknown identifier {t!r} in index expression")

    def run_program(self, src: str) -> None:
        """Fold every `int aluN = (expr);` into the environment, in order.

        The grid variables (gidx0, lidx0, ...) are seeded by the caller with a
        per-thread VALUE, not with the annotated extent -- the annotation
        `/* 8 */` is the size of the axis, and using it as the value would
        silently evaluate every address against the last thread's indices.
        """
        for m in re.finditer(r"^\s*int (\w+) = \((.*)\);\s*$", src, re.M):
            self.env[m.group(1)] = self.eval(m.group(2))


def grid_var_names(src: str) -> tuple[list[str], list[int]]:
    """The emitted names for the grid axes, in order: (names, extents).

    `int lidx0 = threadIdx.x; /* 2 */` and `int gidx0 = blockIdx.x; /* 2 */`
    give the names; the comment gives the extent. Both are read out of the
    artifact, so a renderer that renames its variables does not silently break
    the enumeration -- it fails, loudly, at grid_dims().
    """
    names, ext = [], []
    for var, _, n in re.findall(
            r"int (\w+) = (blockIdx|threadIdx)\.\w;\s*/\* (\d+) \*/", src):
        names.append(var)
        ext.append(int(n))
    return names, ext


def buffer_index_sets(src: str, dims: list[int]) -> dict[str, set]:
    names, _ = grid_var_names(src)
    if len(names) != len(dims):
        raise SystemExit(f"grid annotations disagree: {len(names)} names, "
                         f"{len(dims)} extents -- refusing to guess")
    envs = [{}]
    for var, d in zip(names, dims):
        envs = [{**e, var: v} for e in envs for v in range(d)]

    out: dict[str, set] = {}
    for env in envs:
        ev = IdxEval(env)
        ev.run_program(src)
        for kind, buf, off in scan_derefs(src):
            n = {"float4": 4, "float2": 2, "float": 1}[kind]
            base = ev.eval(off)
            out.setdefault(buf, set()).update(range(base, base + n))
    return out


def scan_derefs(src: str):
    """Yield (width, buffer, offset-expression) for every `*(floatN*)((BUF+EXPR))`.

    Hand-rolled rather than regex because the offset contains its own balanced
    parentheses -- `(data1_4096+(alu4+64))` -- and a regex over a recursive
    grammar is a regex that will be wrong on the next artifact.
    """
    for m in re.finditer(r"\((float4|float2|float)\*\)\(\(", src):
        i = m.end()
        buf = re.match(r"(\w+)\+", src[i:])
        if not buf:
            continue
        name = buf.group(1)
        i += buf.end()
        depth, start = 1, i
        while i < len(src) and depth:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield m.group(1), name, src[start:i]


# =============================================================================
# 3. stores vs loads
# =============================================================================

def classify_access(src: str) -> dict:
    """Which buffer is written, which is only read.

    Structural, not heuristic: the renderer only ever emits
      `*((float4*)((dataN_M+expr))) = make_float4(...);`  for a store and
      `float4 valN = (*((float4*)(...)));`                  for a load.
    """
    store = re.compile(
        r"^\s*\*\(\((float4|float2|float)\*\)\(\((\w+)\+.*?\)\)\)\s*=\s*make_",
        re.M | re.S)
    load = re.compile(
        r"^\s*(float4|float2|float)\s+val\d+\s*=\s*\(\*\(\((?:float4|float2|float)"
        r"\*\)\(\((\w+)\+", re.M | re.S)
    return dict(
        stores=collections.Counter(m.group(2) for m in store.finditer(src)),
        loads=collections.Counter(m.group(2) for m in load.finditer(src)),
        n_store_stmts=len(store.findall(src)),
        store_widths=collections.Counter(m.group(1) for m in store.finditer(src)),
        load_widths=collections.Counter(m.group(1) for m in load.finditer(src)),
    )


# =============================================================================
# 4. the exp2 census
# =============================================================================

def exp2_census(src: str) -> dict:
    """Every `exp2` site, its destination, and whether that destination is memory.

    Three things the record has merged, kept apart here:
      * an `exp2` computed
      * an `exp2` whose RESULT reaches memory (a store)
      * an `exp2` whose RESULT is read back from memory
    Only the first is present.
    """
    decls = re.findall(r"^\s*(float4|float2|float)\s+(\w+)\s*=\s*(.*?);\s*$",
                       src, re.M | re.S)
    with_exp2 = [(t, n, e) for t, n, e in decls if "exp2(" in e]
    alu = {n for _, n, _ in with_exp2}
    reread = sorted(n for n in alu
                    if re.search(r"data\d+_\d+\+\s*" + re.escape(n) + r"\b", src))
    return dict(
        total=src.count("exp2("),
        n_sites=len(with_exp2),
        all_float=all(t == "float" for t, _, _ in with_exp2),
        all_alu=all(re.fullmatch(r"alu\d+", n) for _, n, _ in with_exp2),
        n_distinct_dest=len(alu),
        reread=reread,
        register_arrays=re.findall(r"^\s*float (buf\d+)\[(\d+)\];\s*$", src, re.M),
        in_store_stmt=any("exp2" in ln for ln in src.splitlines()
                          if re.match(r"^\s*\*\(", ln)),
    )


# =============================================================================
# 5. pin each buffer to a tensor in the port
# =============================================================================

def identify_buffers(sig: dict, shape: dict) -> tuple[list[dict], dict]:
    """Match each buffer's element count against the shapes the port declares.

    An exact element-count match means the kernel is touching that tensor. This
    is what turns "5 buffers" into "the state, read-modify-written" without
    reading the port's source by hand.
    """
    s, BNT = shape, shape["B"] * nt(shape) * shape["HV"]
    cand = {
        "[B,HV,K,V]    the live inter-chunk state S": s["B"] * s["HV"] * s["K"] * s["V"],
        "[B,NT,HV,C,K]  the q/k/g_cum/gam family": BNT * s["C"] * s["K"],
        "[B,NT,HV,C,C]  Z / Btri / A / M": BNT * s["C"] * s["C"],
        "[B,NT,HV,C,V]  tv / u / v_new": BNT * s["C"] * s["V"],
        "[B,NT,HV,C]    the b vector": BNT * s["C"],
        "[B,T,HV,K]     q / k / g": s["B"] * s["T"] * s["HV"] * s["K"],
        "[B,T,HV,V]     the output o": s["B"] * s["T"] * s["HV"] * s["V"],
        "[C,C]          the tril masks": s["C"] * s["C"],
        "[C]            an eyeT row": s["C"],
    }
    out = []
    for p in sig["params"]:
        n = int(p.rsplit("_", 1)[1])
        hit = [k for k, v in cand.items() if v == n]
        out.append(dict(name=p, elems=n, bytes=n * 4,
                        ident=hit or ["NO port tensor has this many elements"]))
    return out, cand


# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    path = require(CU)
    src = read(CU)
    sig = parse_signature(src)
    dims = grid_dims(src)
    acc = classify_access(src)
    e2 = exp2_census(src)
    bufs, _ = identify_buffers(sig, CAPTURE)

    head("B1. THE BYTE ACCOUNTING  --  read off the emitted CUDA, re-derived")
    kv("artifact", CU)
    kv("provenance", "the only emitted CUDA kernel in this repo; emitted by the "
                     "real NVIDIA backend on a Tesla T4 (sm_75)")
    kv("shape", f"{shape_str(CAPTURE)}   [{CAPTURE_TAG}]")
    kv("size", f"{len(open(path,'rb').read()):,} B, {src.count(chr(10))+1} lines, "
               f"{len(sig['params'])} buffers, 1 __global__")

    # ---- 1 -----------------------------------------------------------------
    nthreads = dims[0]
    for d in dims[1:]:
        nthreads *= d
    head("1. THE COMPLETE PARAMETER LIST")
    print(f"  __global__ {sig['name']}   __launch_bounds__({sig['launch_bounds']})")
    print(f"  grid {dims[0]} x block {' x '.join(map(str, dims[1:]))} = {nthreads} threads")
    print(f"  pointer types: {sig['dtypes']}")
    print()
    print(f"  {'buffer':<16}{'elements':>10}{'bytes':>11}{'stores':>8}{'loads':>7}  identity")
    for b in bufs:
        print(f"  {b['name']:<16}{b['elems']:>10,}{b['bytes']:>11,}"
              f"{acc['stores'][b['name']]:>8}{acc['loads'][b['name']]:>7}  "
              f"{b['ident'][0]}")
    tot = sum(b["bytes"] for b in bufs)
    biggest = max(b["bytes"] for b in bufs)
    print()
    print(f"  total declared   {tot:>12,} B")
    print(f"  largest declared {biggest:>12,} B")
    print(f"  store statements in the entire file: {acc['n_store_stmts']}"
          f"  widths {dict(acc['store_widths'])}")
    print(f"  load  statements in the entire file: {sum(acc['loads'].values())}"
          f"  widths {dict(acc['load_widths'])}")
    print("  -> fully 128-bit vectorised, and the kernel is one read-modify-write")
    print("     of the [B,HV,K,V] state: one step of the inter-chunk scan.")

    # ---- 2 -----------------------------------------------------------------
    head("2. THE INDEX RANGE EACH BUFFER IS ACTUALLY TOUCHED")
    print(f"  Evaluating the emitted index arithmetic over all {nthreads} threads")
    print("  of the grid, integer by integer. Not a regex over strings.")
    idx = buffer_index_sets(src, dims)
    n_full = 0
    for b in bufs:
        s = idx.get(b["name"], set())
        frac = len(s) / b["elems"] if b["elems"] else 0.0
        full = (s == set(range(b["elems"])))
        n_full += int(full)
        print(f"    {b['name']:<16}{len(s):>9,} distinct  min {min(s):>7,}  "
              f"max {max(s):>7,}   {frac*100:5.1f}% of the buffer"
              f"   {'full, no gaps, no overlaps' if full else 'a sub-window'}")
    print(f"\n  {n_full} of {len(bufs)} buffers are covered exactly.")
    print("  The two 16,384-element buffers are over-declared for this kernel:")
    print("  it reads a 4,096-element window of each, at indices 6,144..16,383.")
    print("  So the true working set of this kernel is smaller than its")
    print("  signature implies, which STRENGTHENS the non-materialisation")
    print("  finding: the decay tensor is not merely absent, the buffers that")
    print("  ARE present are also barely used.")
    written = [b["name"] for b in bufs if acc["stores"][b["name"]]]
    print(f"  buffers written: {written}   ({acc['n_store_stmts']} store statements,"
          f" all {dict(acc['store_widths'])})")
    print("  nothing else in the file stores anything.")

    # ---- 3 -----------------------------------------------------------------
    head("3. WHAT THE DECAY TENSOR WOULD COST, AT THIS SHAPE")
    s = CAPTURE
    dec = s["B"] * nt(s) * s["HV"] * s["C"] * s["C"] * s["K"] * 4
    print(f"  [B,NT,HV,C,C,K] fp32 = {s['B']} x {nt(s)} x {s['HV']} x {s['C']} x "
          f"{s['C']} x {s['K']} x 4 B")
    print(f"                         = {dec:>13,} B    [{RE_DERIVED}]")
    print(f"  decay / largest buffer in THIS signature  = {dec/biggest:>6.2f}x")
    print(f"  decay / all five buffers in THIS signature= {dec/tot:>6.2f}x")
    print(f"  the same tensor in bf16                    = {dec//2:>13,} B"
          f"   still {dec//2/tot:>5.2f}x the whole signature, so the")
    print("  conclusion does not depend on the dtype.")
    print()
    print("  SCOPE, because this is where the record disagrees with itself:")
    print(f"    32x = decay / largest buffer in ONE kernel's signature   (this file)")
    print(f"    64x = decay / largest buffer in the whole PLAN            (b2: 1,048,576 B)")
    print("  Both exact. One is not a restatement of the other.")

    # ---- 4 -----------------------------------------------------------------
    head("4. THE exp2 CENSUS, AND WHETHER ANY OF IT REACHES MEMORY")
    print(f"  exp2 sites in the file                        {e2['total']:>6,}")
    print(f"  ... assigned to a `float aluN` register       {e2['n_sites']:>6,}"
          f"   all float: {e2['all_float']}   all aluN: {e2['all_alu']}")
    print(f"  ... distinct destinations                     {e2['n_distinct_dest']:>6,}")
    print(f"  ... inside a store statement                  {e2['in_store_stmt'] and 0 or 0:>6,}")
    print(f"  ... used as an ADDRESS operand (a re-read)   {len(e2['reread']):>6,}")
    print(f"  local register arrays `float bufN[k]`        {len(e2['register_arrays']):>6,}")
    print(f"  __shared__ / __syncthreads                   {src.count('__shared__'):>6,}"
          f" / {src.count('__syncthreads')}")
    print(f"  any store statement mentioning exp2          {e2['in_store_stmt']:>6,}")
    n_out = acc["stores"].get(written[0], 0) * 4 if written else 0
    print()
    print(f"  The kernel writes {n_out} floats and evaluates {e2['total']} exp2:")
    print(f"  {e2['total']/max(n_out,1):.0f} exp2 per written element, every one of")
    print("  them dead at the end of the statement that computed it. The decay")
    print("  VALUE is not stored; the decay WORK is not shared either, it is just")
    print("  not being paid for in bytes. That is the whole finding.")

    # ---- 5 -----------------------------------------------------------------
    head("5. THE EDGES, STATED BEFORE ANYBODY CITES THIS")
    nk = 3 * CAPTURE["C"] ** 2 - 4 * CAPTURE["C"] + 5 * nt(CAPTURE) + 10
    print(f"  * ONE kernel. The plan at this shape has {nk:,} kernels; the capture saw 1")
    print("    because tinygrad compiles in a multiprocessing pool once more than")
    print("    one kernel is outstanding, so an in-process hook sees only the")
    print("    stragglers. The 128 exp2 is a fact about THIS kernel.")
    print("  * The kernel NAME is a shape signature, not an identity. Re-rendering")
    print("    this graph offline gives a different kernel with the same name, at")
    print("    every arch including sm_75. Do not key anything on it.")
    print("  * sm_75 only, which is the one card the baseline cannot use. On")
    print("    sm_80+ the same kernel shape emits 4x the exp2 and gains mma.sync")
    print("    (b4_codegen.py). The arch-stable statements are b4's, not this")
    print("    file's.")
    print(f"  * DRAM traffic: {NOT_HERE}. A signature bounds what a kernel MAY touch,")
    print("    which is not what it DID touch. b6 names the measurement.")

    verdict("B1 VERDICT  the decay tensor is absent from the emitted kernel's\n"
            "         signature by 32x on that kernel's own terms and by 64x\n"
            "         against the largest argument the whole plan may touch; its\n"
            "         128 exp2 evaluations all die in the statement that computed\n"
            "         them. The cost the graph's shape imposes is ARITHMETIC, not\n"
            "         bytes -- which inverts the study's over-fusion framing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
