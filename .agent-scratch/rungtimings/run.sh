#!/usr/bin/env bash
# run.sh -- push, poll, fetch. Bounded, self-verifying, and it does not stop
#          pushing when Kaggle refuses a push.
#
#   ./run.sh --build-only  assemble and validate, push nothing
#   ./run.sh --dry-run     everything except the network calls
#   ./run.sh               build, push, poll, download, print
#
# WHY THE PUSH LOOP IS THE WHOLE POINT OF THIS SCRIPT
# -----------------------------------------------------
# Kaggle's free tier allows 2 concurrent batch GPU sessions per account and this
# repository's other lanes are holding both. `kaggle kernels push` prints
#
#     Kernel push error: Maximum batch GPU session count of 2 reached
#
# and EXITS 0. A script that ignores the message goes on to poll for a status,
# sees the PREVIOUS run's COMPLETE, and downloads its output -- which is
# indistinguishable from a successful run of the code just pushed. That is how
# two rounds of a sibling notebook in this project were "measured" against a
# build that predated its own fixes. So the push message is checked, and a
# refused push is retried rather than treated as a push.
#
# WHY CANCEL IS NOT TREATED AS "OUT OF QUOTA"
# -------------------------------------------
# A CANCEL status from Kaggle is a `CellTimeoutError` in the card's log, not a
# kill. This script fetches the log on any non-COMPLETE status and prints the
# tail, so the reason is visible. The notebook is built to survive it anyway:
# one cell per stage, one cell per batch, SUMMARY.txt flushed after every cell.
#
# COST. One Kaggle GPU session. No GPU quota is spent by a refused push.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KDIR="$HERE/kernel"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="${OUTROOT:-$HERE/out}"
LOG="$OUTROOT/$STAMP-run.log"

BUILD_ONLY=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --build-only) BUILD_ONLY=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
mkdir -p "$OUTROOT"

say() { printf '%s\n' "$*" | tee -a "$LOG"; }

KID="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$KDIR/kernel-metadata.json" | head -1)"
REF="$KID"
SESSION_MIN="${SESSION_MIN:-70}"
QUEUE_WAIT_MIN="${QUEUE_WAIT_MIN:-60}"
MAX_PUSH_ATTEMPTS="${MAX_PUSH_ATTEMPTS:-240}"

# ---------------------------------------------------------------- 1. build
say "=== 1. assemble the notebook (payloads are hashed at build time) ==="
"$PY" "$HERE/make_notebook.py" --repo "$REPO" \
    --out "$KDIR/mc_minecu_rungs_t4.ipynb" 2>&1 | tee -a "$LOG" | sed 's/^/  /'

say ""
say "=== 2. validate: nbformat, every code cell compiles, no unresolved stub ==="
"$PY" - "$KDIR/mc_minecu_rungs_t4.ipynb" <<'PY' 2>&1 | tee -a "$LOG" | sed 's/^/  /'
import json, re, sys
nb = json.load(open(sys.argv[1]))
assert nb.get("nbformat") == 4, nb.get("nbformat")
assert nb["metadata"].get("kernelspec"), "no kernelspec"
n = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    compile("".join(c["source"]), f"cell{i}", "exec")
    n += 1
src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
assert src.count("PAYLOADS = ") == 1, "the driver must interpolate PAYLOADS once"
left = set(re.findall(r"__[A-Z][A-Z_]+__", src))
assert not left, f"unresolved placeholders: {left}"
# every batch gets its own cell, so one timeout cannot take the ladder with it
for b in (1024, 2048, 512, 256):
    assert f"time_batch({b})" in src, f"no cell for batch {b}"
print(f"  valid: {n} code cells, all compile, no stubs, one cell per batch")
PY

say ""
say "=== 3. pre-flight: the payload layout, exactly as the card will see it ==="
"$PY" - "$KDIR/payloads.json" <<'PY' 2>&1 | tee -a "$LOG" | sed 's/^/  /'
import base64, hashlib, json, os, shutil, sys
side = json.load(open(sys.argv[1]))
out = os.path.abspath(".pf_rungs")
shutil.rmtree(out, ignore_errors=True)
try:
    for name, b64, want in side["payloads"]:
        data = base64.b64decode(b64)
        assert hashlib.sha256(data).hexdigest() == want, f"hash mismatch {name}"
        dest = os.path.join(out, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(data)
    for p in ("minecu_tg/render.py", "minecu_tg/bench.py",
              "minecu_tg_opt/render.py", "minecu_tg_opt/battery_opt.py",
              "minecu_ref/src/kernels.cu", "minecu_tg/kaggle/refdrv.cu"):
        assert os.path.isfile(os.path.join(out, p)), f"missing {p}"
    print(f"  {len(side['payloads'])} payloads written, hashes match, "
          f"every file the card imports is present")
finally:
    shutil.rmtree(out, ignore_errors=True)
PY

if [ "$BUILD_ONLY" = 1 ] || [ "$DRY" = 1 ]; then
  say ""
  say "BUILD ONLY / DRY RUN: no network calls made."
  exit 0
fi

say ""
say "=== 4. auth ==="
if [ -z "${kaggle_API_TOKEN:-}" ] && [ ! -s "$HOME/.kaggle/access_token" ] \
   && [ ! -s "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]; then
  say "no Kaggle credentials."
  exit 2
fi
say "credentials present; kernel $REF"

# ----------------------------------------------------------------- push
say ""
say "=== 5. push, retrying while both T4 sessions are in use ==="
pushed=0
for i in $(seq 1 "$MAX_PUSH_ATTEMPTS"); do
  out=$("$KAGGLE" kernels push -p "$KDIR" -t $((SESSION_MIN * 60)) 2>&1)
  rc=$?
  printf '%s\n' "$out" | tee -a "$LOG"
  if printf '%s' "$out" | grep -q "successfully pushed"; then
    pushed=1
    say "  PUSHED on attempt $i at $(date +%H:%M:%S)"
    break
  fi
  say "  [$(date +%H:%M:%S)] attempt $i refused (rc=$rc): $(printf '%s' "$out" | head -1)"
  sleep 60
done
if [ "$pushed" != 1 ]; then
  say "gave up pushing after $MAX_PUSH_ATTEMPTS attempts; see $LOG"
  exit 3
fi

# ----------------------------------------------------------------- poll
say ""
say "=== 6. wait (poll 60s, up to ${QUEUE_WAIT_MIN}m) ==="
deadline=$(( $(date +%s) + QUEUE_WAIT_MIN * 60 ))
status=UNKNOWN; last=""
while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "  gave up waiting; stop it at https://www.kaggle.com/code/$REF"
    exit 4
  fi
  raw=$("$KAGGLE" kernels status "$REF" 2>&1) || true
  st="$(printf '%s' "$raw" | sed -n 's/.*status "\(.*\)".*/\1/p' | head -1)"
  st="${st##*.}"; [ -z "$st" ] && st=UNKNOWN
  if [ "$st" != "$last" ]; then say "  [$(date +%H:%M:%S)] $st"; last="$st"; fi
  status="$st"
  case "$st" in COMPLETE|ERROR|FAIL|CANCEL|CANCELLED|FAILED) break ;; esac
  sleep 60
done

if [ "$status" != COMPLETE ]; then
  say ""
  say "=== the card's log -- a CANCEL here is a CellTimeoutError, not a kill ==="
  "$KAGGLE" kernels logs "$REF" 2>&1 | tail -200 | tee -a "$LOG" | sed 's/^/  /'
  say "run did not complete: $status. Fetching output anyway -- SUMMARY.txt is"
  say "flushed after every cell, so an early stage's numbers are already in it."
fi

# --------------------------------------------------------------- download
DEST="$OUTROOT/$STAMP"
mkdir -p "$DEST"
say ""
say "=== 7. download -> $DEST ==="
got=0
for a in 1 2 3; do
  "$KAGGLE" kernels output "$REF" -p "$DEST" -o -q 2>&1 | tee -a "$LOG" | sed 's/^/  /'
  if [ -n "$(find "$DEST" -name '*.zip' -o -name 'SUMMARY.txt' 2>/dev/null | head -1)" ]; then
    got=1; break
  fi
  say "  attempt $a/3 produced nothing; sleeping 20s"; sleep 20
done
for z in "$DEST"/*.zip; do
  [ -e "$z" ] || continue
  unzip -qo "$z" -d "$DEST/$(basename "$z" .zip)" 2>&1 | sed 's/^/    /'
done
# the refstate dump is tens of MB and the comparison already happened in the
# notebook; keeping it would turn a small artefact set into a large one.
rm -rf "$DEST"/*/refstate "$DEST"/refstate 2>/dev/null || true

say ""
say "=== 8. what came back ==="
find "$DEST" -type f | sort | while read -r f; do
  printf '    %-50s %8s\n' "${f#$DEST/}" "$(du -h "$f" | cut -f1)"
done | tee -a "$LOG"

if [ "$got" = 0 ]; then
  say ""
  say "NOTHING CAME BACK. The kernel log above is where the reason is."
  exit 5
fi

say ""
say "=== 9. the headline table, re-derived from the run's own JSON ==="
"$PY" - "$DEST" <<'PY' 2>&1 | tee -a "$LOG" | sed 's/^/  /'
import glob, json, os, sys
dest = sys.argv[1]
def find(name):
    for p in [os.path.join(dest, name)] + glob.glob(os.path.join(dest, "*", name)):
        if os.path.exists(p):
            return json.load(open(p))
    return None
T, ref, eq = find("rung_timings.json"), find("ref_timing.json"), find("obs_equivalence.json")
if eq:
    ok = all(r["identical"] for r in eq.get("rows", []))
    print(f"GATE: {sum(r['identical'] for r in eq.get('rows',[]))}/"
          f"{len(eq.get('rows',[]))} rungs byte-identical to the hand-written "
          f"side's own bytes on {eq.get('pixels')} pixels  -> {'PASS' if ok else 'FAIL'}")
    d = eq.get("dda", {})
    if d:
        print("DDA  : mean steps/pixel " +
              "  ".join(f"{k}={v['mean']:.4f}" for k, v in d.items()))
if not T:
    print("no rung_timings.json")
    raise SystemExit(0)
runs = sorted(T.keys(), key=int)
variants = []
for b in runs:
    for r in T[b].get("rows", []):
        if r["rung"] not in variants:
            variants.append(r["rung"])
print()
print(f"{'batch':>6} {'ref us':>9} " + " ".join(f"{v[:12]:>14}" for v in variants))
for b in runs:
    ru = ((ref or {}).get(b) or {}).get("render_us", [None])[0]
    by = {r["rung"]: r for r in T[b].get("rows", []) if not r.get("error")}
    line = f"{b:>6} {round(ru,1) if ru else '-':>9} "
    for v in variants:
        r = by.get(v)
        line += (f"{r['median_ms']:8.3f}({r['median_ms']*1000/ru:5.1f}x) " if r and ru
                 else f"{'-':>14} ")
    print(line)
print()
for b in runs:
    print(f"batch {b}")
    for r in T[b].get("rows", []):
        if r.get("error"):
            print(f"  {r['rung']:<13} NOT MEASURED: {r['error'][:100]}")
            continue
        line = (f"  {r['rung']:<13} {r['median_ms']:9.3f} ms  "
                f"spread {r['spread_pct']:5.2f}%  "
                f"CALLs cold {r.get('n_kernels_cold')} "
                f"warm {r.get('n_kernels_warm')}  "
                f"peak {r['peak_live_bytes']/2**20:.1f} MiB  "
                f"guards {'OK' if r['guards_ok'] else 'FAILED'}")
        wr = r.get("walk_vs_ruler_pct")
        if wr is not None:
            line += f"  walk vs ruler {wr:+.3f}%"
        print(line)
    wl = T[b].get("timed_workload")
    if isinstance(wl, dict) and wl:
        print("    workload: " + "  ".join(
            f"{k}={v['mean']:.4f}" for k, v in wl.items()
            if isinstance(v, dict) and "mean" in v))
PY

say ""
say "done. artefacts in $DEST"
say "  the notebook as it ran: https://www.kaggle.com/code/$REF"
