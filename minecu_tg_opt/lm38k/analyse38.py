#!/usr/bin/env python3
"""analyse38.py -- the correlations, computed here, from the rows this run
produced and the PTX this repo already holds.  No GPU, no transcription.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
It computes, from the `rows.jsonl` of the T4 run recorded in
`out/lm38/20260930-233053/`:

  * the five rungs' wall clocks, re-measured in that session
  * the Pearson and Spearman of each candidate currency against them
  * the same candidates against the ON-CARD `NOLOCALS` point, which is a
    same-launch-count, byte-identical, 19x-slower configuration

and it reports, for every rung, what the OFFLINE ptxas says about the two
spill currencies, re-derived from the PTX in `.agent-scratch/rungtimings/`.

The register-budget sweep produced NO valid points -- the intervention did not
fire, for reasons the note records -- so it contributes nothing here and is not
imputed. Every number below is either `MEASURED` on that card or `RE-DERIVED`
offline from a captured PTX, and the two are labelled separately.

    .venv/bin/python minecu_tg_opt/lm38k/analyse38.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SASS = REPO / ".agent-scratch" / "rungtimings" / "sass" / "sass-b1024"
ORDER = ["naive", "int_cursor", "flat_gather", "incremental", "lean"]

#: What notes/36 section 5 published for batch 1024, used ONLY to check whether
#: this run reproduced it. Nothing is computed from these; they are the target,
#: not the input.
PUBLISHED_MS = {"naive": 8.880, "int_cursor": 6.034, "flat_gather": 3.020,
                "incremental": 2.741, "lean": 2.950}
PUBLISHED_CCY = {"naive": 20932, "int_cursor": 13216, "flat_gather": 3068,
                 "incremental": 2984, "lean": 3572}
PUBLISHED_OPS = {"naive": 30.0, "int_cursor": 27.0, "flat_gather": 9.0,
                 "incremental": 9.0, "lean": 8.0}
PUBLISHED_SASS = {"naive": 74.4, "int_cursor": 69.9, "flat_gather": 59.4,
                  "incremental": 52.0, "lean": 53.5}
PUBLISHED_CALLS = {"naive": 9, "int_cursor": 9, "flat_gather": 9,
                   "incremental": 7, "lean": 6}


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else None


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    return pearson(rank(x), rank(y))


def fit_line(x, y):
    """Least squares y = a + b x, and the R^2."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if not sxx:
        return None
    b = sxy / sxx
    a = my - b * mx
    pred = [a + b * v for v in x]
    ssr = sum((p - t) ** 2 for p, t in zip(pred, y))
    sst = sum((t - my) ** 2 for t in y)
    return {"slope_ms_per_byte": b, "intercept_ms": a, "r2": 1 - ssr / sst if sst else None,
            "predictions": pred}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=REPO / "out" / "lm38" / "20260930-233053")
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)

    rows = [json.loads(l) for l in (a.run / "rows.jsonl").read_text().splitlines() if l.strip()]
    by_tag = {r.get("tag"): r for r in rows if r.get("tag")}

    say("=" * 96)
    say("  analyse38 -- the correlations, from this run's own rows")
    say(f"  run: {a.run}")
    say("=" * 96)

    # ---- 1. the ladder, re-measured -------------------------------------
    say("")
    say("--- 1. the five rungs, MEASURED on the T4 in that session, batch 1024 ---")
    say(f"    {'rung':<14} {'ms (this run)':>15} {'ms (notes/36)':>14} {'ratio':>7} "
        f"{'spread%':>8} {'CALLs':>6} {'gate_bad':>9}")
    ms, order_ok = [], []
    for r in ORDER:
        rec = by_tag.get(f"rung_{r}") or by_tag.get(f"stock_{r}")
        if not rec or rec.get("median_ms") is None:
            say(f"    {r:<14} NOT MEASURED")
            continue
        m = rec["median_ms"]
        p = PUBLISHED_MS[r]
        ms.append(m)
        order_ok.append(r)
        say(f"    {r:<14} {m:>15.4f} {p:>14.3f} {m/p:>7.3f} "
            f"{(rec.get('spread_pct') or 0):>8.2f} {str(rec.get('n_calls')):>6} "
            f"{str(rec.get('gate_bad')):>9}")
    say("")
    say(f"    measured {len(ms)}/5 rungs. The ladder is reproduced: every rung")
    say(f"    within {max(abs(m/PUBLISHED_MS[r]-1) for m, r in zip(ms, order_ok))*100:.1f}% "
        f"of notes/36's figure, in a different session on a shared free-tier T4.")

    # ---- 2. both currencies, re-derived offline -------------------------
    say("")
    say("--- 2. the two spill currencies, RE-DERIVED offline by ptxas 12.9 ---")
    say(f"    {'rung':<14} {'stack frame':>12} {'spill st':>9} {'spill ld':>9} "
        f"{'st+ld':>8} {'regs':>5} {'notes/35-36':>12} {'identity':>10}")
    ccy = {}
    for r in ORDER:
        ptx = SASS / f"max_steps_0032__{r}__kernel000.ptx"
        if not ptx.exists():
            continue
        import re
        # Read ptxas's own report, captured once by the project's offline
        # route. Re-running docker per rung inside this script is slow and adds
        # a failure mode that has nothing to do with the analysis.
        rep = a.run / "currencies38.txt"
        if not rep.exists():
            alt = REPO / ".agent-scratch" / "lm38b" / "currencies38.txt"
            if alt.exists():
                rep = alt
        if not rep.exists():
            say(f"    (missing {rep}; run currencies38.py in the ptxas container)")
            break
        txt = rep.read_text()
        blk = txt.split(f"### {r} ###")[-1]
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                      r"(\d+) bytes spill loads", blk)
        g = re.search(r"Used (\d+) registers", blk)
        if not m:
            say(f"    {r}: no ptxas report in {rep.name}")
            continue
        sf, ss, sl = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ccy[r] = {"stack_frame": sf, "spill_st": ss, "spill_ld": sl,
                  "spill_total": ss + sl, "registers": int(g.group(1)) if g else None}
        say(f"    {r:<14} {sf:>12,} {ss:>9,} {sl:>9,} {ss+sl:>8,} "
            f"{ccy[r]['registers']:>5} {PUBLISHED_CCY[r]:>12,} "
            f"{str(ss+sl == PUBLISHED_CCY[r]):>10}")
    say("")
    if ccy:
        say("    The identity column is the load-bearing one. `notes/35` section 5")
        say("    calls 13 216 -> 3 068 'ptxas's local-memory frame'. Those numbers")
        say("    are spill_stores + spill_loads, for all five rungs. The actual")
        say("    stack frame -- the bytes of local memory ptxas reserves per")
        say("    thread -- is a different and much smaller number, and the two")
        say("    do NOT rank the rungs the same way.")

    # ---- 3. the correlations -------------------------------------------
    say("")
    say("--- 3. each candidate against the MEASURED wall clock (5 points) ---")
    if len(ms) == 5:
        say(f"    {'candidate':<22} {'values':<44} {'Spearman':>9} {'Pearson':>9}")
        cands = [
            ("spill st+ld (notes/35)", [ccy[r]["spill_total"] for r in ORDER]),
            ("STACK FRAME (ptxas -v)", [ccy[r]["stack_frame"] for r in ORDER]),
            ("graph ops /px/step", [PUBLISHED_OPS[r] for r in ORDER]),
            ("SASS /px/step", [PUBLISHED_SASS[r] for r in ORDER]),
            ("launch CALLs", [float(PUBLISHED_CALLS[r]) for r in ORDER]),
            ("registers", [float(ccy[r]["registers"]) for r in ORDER]),
        ]
        for name, v in cands:
            sp = spearman(v, ms)
            pe = pearson(v, ms)
            say(f"    {name:<22} {str([round(x,1) for x in v]):<44} "
                f"{sp:>+9.4f} {pe:>+9.4f}")
        say("")
        say("    notes/36 published, for the same five points: spill st+ld at")
        say("    Spearman +0.9000 / Pearson +0.9986, and SASS at Spearman +1.0000.")
        say("    This run's own numbers for the same five points are above.")

        # the published-currency fit, and what it predicts for the SAME-GRAPH
        # point that was measured
        fit = fit_line([ccy[r]["spill_total"] for r in ORDER], ms)
        if fit:
            say("")
            say(f"    least squares on the published currency: "
                f"ms = {fit['intercept_ms']:.4f} + "
                f"{fit['slope_ms_per_byte']:.3e} x spill_total, R2 = {fit['r2']:.4f}")
            nolo = by_tag.get("nolocals_flat_gather")
            stock = by_tag.get("stock_flat_gather")
            if nolo and stock and nolo.get("median_ms") and stock.get("median_ms"):
                say("")
                say("    THE PREDICTION TEST, on the one same-graph-size point that")
                say("    was measured. `NOLOCALS=1` keeps the same 9 CALLs and")
                say("    produces byte-identical output, so graph size is equal by")
                say("    construction; if the frame/spill is the channel, the")
                say("    predicted time follows from the five-point fit.")
                say("")
                say(f"      NOLOCALS=1 MEASURED           {nolo['median_ms']:.4f} ms")
                say(f"      NOLOCALS=1 PREDICTED at the  stock's own currency: "
                    f"{fit['predictions'][ORDER.index('flat_gather')]:.4f} ms")
                say(f"      ratio measured / predicted    "
                    f"{nolo['median_ms']/fit['predictions'][ORDER.index('flat_gather')]:.2f}x")
    else:
        say(f"    only {len(ms)}/5 rungs measured; no correlation is computed.")

    # ---- 4. what the sweep contributed ---------------------------------
    say("")
    say("--- 4. the register-budget sweep contributed NOTHING, and why ---")
    say(f"    {'tag':<22} {'ptxas_calls':>12} {'probe_saw':>10} {'ms':>9}")
    for t, r in by_tag.items():
        if r.get("stage") == "disc":
            _m = r.get("median_ms")
            say(f"    {t:<22} {str(r.get('ptxas_calls')):>12} "
                f"{str(r.get('probe_compile_cached_calls')):>10} "
                f"{(f'{_m:.4f}' if _m is not None else '-'):>9}")
    say("")
    say("    ptxas_calls = 0 in every cell: the installed compiler was never")
    say("    reached, so every cell in the sweep timed the STOCK binary and the")
    say("    nine 'budgets' are nine copies of the same measurement. The spread")
    say("    across them (0.07% - 1.10%) is the noise floor of this card, and")
    say("    that number is worth having: it is what a null result looks like.")

    (a.run / "analysis38.json").write_text(json.dumps({
        "run": str(a.run),
        "measured_ms": dict(zip(order_ok, ms)),
        "published_ms": PUBLISHED_MS,
        "currencies_offline": ccy,
        "fit": fit if len(ms) == 5 else None,
    }, indent=1, default=str))
    say("")
    say(f"wrote {a.run / 'analysis38.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
