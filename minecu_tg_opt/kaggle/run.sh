#!/usr/bin/env bash
# run.sh -- one command, bounded, self-verifying: measure both TinyGrad ports
#           against the hand-written CUDA baseline on Kaggle's free T4.
#
#   ./run.sh --dry-run     everything except the network calls
#   ./run.sh --build-only  assemble and validate the notebook, push nothing
#   ./run.sh --no-verify   skip the local ~17 min CPU gate (the card still runs it)
#   ./run.sh               build, push, poll, download, print
#
# WHY A KAGGLE KERNEL AND NOT A DATASET. `kaggle kernels push` uploads exactly
# one file: the notebook named by "code_file". Sidecars in the kernel directory
# are not uploaded, so `kernels.cu` (105 KB), `refdrv.cu`, `bench.cu` and both
# packages travel inside the notebook as base64 and are sha256-checked before
# anything is compiled. A mismatch is a hard failure, not a warning.
#
# WHY SASS IS NOT MEASURED HERE. An instruction count and a register count are
# properties of a compiler version as much as of a source. The T4 image carries
# CUDA 12.8; the offline route in `minecu_tg_opt/compile_sass.sh` is 12.9 on
# both sides. The note quotes the offline route, and this run quotes timings.
#
# COST. One Kaggle GPU run. The session cap is passed to `kernels push -t` so
# Kaggle enforces it server-side. One full run is a small fraction of the free
# tier's weekly hours.
#
# NO GPU ON THIS HOST. Nothing here needs one. The measurement happens on
# Kaggle; the SASS happens in the archlab container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # minecu_tg_opt/
REPO="$(cd "$HERE/.." && pwd)"                              # repo root
KDIR="$HERE/kaggle"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="${OUTROOT:-$REPO/out/minecu_tg_opt}"
LOG="$OUTROOT/$STAMP-run.log"
mkdir -p "$OUTROOT"

DRY=0
BUILD_ONLY=0
# The verify gate is ~17 min of CPU on this host (five shapes x five rungs x a
# float64 reference render each). Skipping it is only for iterating on the
# *notebook plumbing*; a real push runs it.
NO_VERIFY="${NO_VERIFY:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)    DRY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --no-verify)  NO_VERIFY=1; shift ;;
    -h|--help)    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

say() { printf '%s\n' "$*" | tee -a "$LOG"; }

# ------------------------------------------------------------- local gate
# Both correctness gates run on this host, with no GPU, before anything is
# pushed. The notebook runs them again on the card -- the CPU and the CUDA
# backend lower the same graph differently, and a gate that only ran on one
# would leave the other unverified.
say "=== minecu_tg_opt: verify both ports on this host (CPU, no GPU) ==="
if [ "$NO_VERIFY" = 1 ]; then
  say "  SKIPPED (--no-verify). The notebook runs minecu_tg_opt/verify_opt.py on"
  say "  the card as well, so a real push is still gated; this only skips the"
  say "  ~17 min of local CPU."
elif ! "$PY" "$HERE/verify_opt.py" 2>&1 | tee -a "$LOG" | tail -20 | sed 's/^/  /'; then
  say "  verification FAILED; refusing to push. A timing from an unverified"
  say "  implementation is a measurement of the wrong thing."
  exit 4
else
  say "  verification passed: every rung of the ladder equals the naive port's"
  say "  observation bit for bit, and that observation equals an independent"
  say "  float64 transcription of the same algorithm"
fi

say ""
say "=== assemble the notebook ==="
"$PY" "$KDIR/make_notebook.py" --repo "$REPO" \
    --out "$KDIR/minecu_tgopt_t4.ipynb" 2>&1 | tee -a "$LOG" | sed 's/^/  /'
"$PY" - "$KDIR/minecu_tgopt_t4.ipynb" <<'PY' 2>&1 | sed 's/^/  /' | tee -a "$LOG"
import json, sys
nb = json.load(open(sys.argv[1]))
assert nb.get("metadata", {}).get("kernelspec"), "no kernelspec"
assert nb.get("nbformat") == 4, f"nbformat {nb.get('nbformat')}"
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print(f"notebook valid: {len(nb['cells'])} cells, "
      f"{sum(len(''.join(c['source'])) for c in nb['cells'])} source chars, "
      f"all code cells compile")
PY

# --------------------------------------------------------------- pre-flight
# The payload layout, exactly as the notebook will materialise it. Three pushes
# of the sibling notebook died before they measured anything and each cost a
# run, so each failure is now a gate here.
say ""
say "=== pre-flight: the payload layout, as the notebook will materialise it ==="
if ! "$PY" - "$KDIR/minecu_tgopt_t4.ipynb" "$KDIR/payloads.json" <<'PY' 2>&1 | sed 's/^/  /'
import base64, hashlib, json, os, shutil, subprocess, sys, tempfile
nb_path, pl_path = sys.argv[1], sys.argv[2]
nb = json.load(open(nb_path))
side = json.load(open(pl_path))
payloads = side["payloads"]

# the sidecar and the notebook must agree, or the sidecar is checking a
# different run than the one that will be pushed
assert side["notebook_sha256"] == hashlib.sha256(open(nb_path, "rb").read()).hexdigest(), \
    "payloads.json is stale: it describes a different notebook. Regenerate."
src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
assert src.count("PAYLOADS = ") == 1, "the driver must interpolate PAYLOADS exactly once"
for name, _, want in payloads:
    assert f'"{name}"' in src, f"{name} is in the sidecar but not in the notebook"

OUT = tempfile.mkdtemp(prefix="tgopt_pf.")
try:
    n_pkg = 0
    for name, b64, want in payloads:
        data = base64.b64decode(b64)
        assert hashlib.sha256(data).hexdigest() == want, f"hash mismatch {name}"
        if name.startswith("minecu_tg_opt/") or name.startswith("minecu_tg/"):
            n_pkg += 1
        dest = os.path.join(OUT, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(data)
    assert n_pkg >= 15, f"only {n_pkg} package payloads: a package would be flattened"
    for p in ("minecu_tg/__init__.py", "minecu_tg_opt/__init__.py",
              "tools/archlab/inspect_ptx.py"):
        assert os.path.isfile(os.path.join(OUT, p)), f"missing {p}"
    print(f"  stage 1 write: {len(payloads)} payloads, hashes match, "
          f"{n_pkg} package files, both __init__.py present; sidecar agrees "
          f"with the notebook")

    # stage 2: the notebook's order -- sys.path, a real tinygrad, then the
    # packages. The sibling's import check sat BEFORE `pip install tinygrad`
    # and failed on a missing tinygrad.
    check = (
        "import sys, importlib\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import tinygrad\n"
        "m1 = importlib.import_module('minecu_tg.render')\n"
        "m2 = importlib.import_module('minecu_tg_opt.render')\n"
        "for s in ('blocks','world','cpu_reference','verify','codegen','bench',"
        "'battery','sass_matrix'):\n"
        "    importlib.import_module('minecu_tg.' + s)\n"
        "for s in ('verify_opt','capture_opt','opcount','census_opt','battery_opt'):\n"
        "    importlib.import_module('minecu_tg_opt.' + s)\n"
        "print('  stage 2 import:', m1.__file__)\n"
        "print('  stage 2 import:', m2.__file__)\n"
        "print('  stage 2 import: both packages import, all 12 submodules OK')\n"
    )
    p = subprocess.run([sys.executable, "-c", check, OUT],
                       capture_output=True, text=True,
                       env={**os.environ, "DEV": "CPU", "PARALLEL": "0"})
    if p.returncode != 0:
        print("  stage 2 import FAILED: " + p.stderr.strip().splitlines()[-1])
        sys.exit(1)
    print(p.stdout.rstrip())
    for s, mod in (("minecu_tg/verify.py", "minecu_tg/verify.py"),
                   ("minecu_tg_opt/verify_opt.py", "minecu_tg_opt/verify_opt.py"),
                   ("minecu_tg_opt/capture_opt.py", "minecu_tg_opt/capture_opt.py"),
                   ("minecu_tg_opt/opcount.py", "minecu_tg_opt/opcount.py"),
                   ("minecu_tg_opt/battery_opt.py", "minecu_tg_opt/battery_opt.py")):
        q = subprocess.run([sys.executable, os.path.join(OUT, s), "--help"],
                           capture_output=True, text=True,
                           env={**os.environ, "DEV": "NULL", "PARALLEL": "0",
                                "NULL_ALLOW_COPYOUT": "1"})
        assert q.returncode == 0, f"{s} --help rc={q.returncode}: {q.stderr[-400:]}"
    print("  stage 3 run: verify/verify_opt/capture_opt/opcount/battery_opt "
          "all --help cleanly")
finally:
    shutil.rmtree(OUT, ignore_errors=True)
PY
then
  say "  the payload layout or stage order the notebook will use is broken; refusing to push"
  exit 6
fi

say ""
say "=== pre-flight: the exact nvcc commands the notebook runs, for sm_75 ==="
# These MUST be the notebook's commands. refdrv.cu is compiled ALONE with
# -DMINECU_KERNELS (it #includes kernels.cu, and #include resolves relative to
# the including file); bench.cu is compiled WITH kernels.cu (it declares the
# extern "C" launchers and links against them). Getting this backwards is a link
# error, and a pre-flight that tests a different command than the notebook runs
# is worse than none, because it looks like a gate.
if command -v docker >/dev/null && docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1; then
  # Staged into a subdirectory of the repo, never a system temp dir, and the
  # `cp`s are separate: a failed `cp` inside a `&&` chain with `set -e` used to
  # abort the whole pre-flight and report nothing, which reads as "the gate
  # passed".
  mkdir -p "$REPO/.pf"
  docker run --rm -v "$REPO:/w" -w /w archlab/nvcc:12.9 bash -c '
       set -e
       export PATH=/usr/local/cuda-12.9/bin:$PATH
       F="-O3 --use_fast_math -std=c++17 -arch=sm_75 -diag-suppress 177"
       mkdir -p /w/.pf
       cp /w/minecu_tg/kaggle/refdrv.cu /w/.pf/refdrv.cu
       cp /w/minecu_ref/src/kernels.cu   /w/.pf/kernels.cu
       cp /w/minecu_ref/bench/bench.cu   /w/.pf/bench.cu
       echo "--- refdrv alone, kernels.cu named by -DMINECU_KERNELS ---"
       nvcc $F -DMINECU_KERNELS=\"/w/.pf/kernels.cu\" /w/.pf/refdrv.cu -o /w/.pf/refdrv
       echo "--- bench with kernels.cu on the command line ---"
       nvcc $F /w/.pf/kernels.cu /w/.pf/bench.cu -o /w/.pf/bench
       echo "--- sm_75 gate on the refdrv binary ---"
       cuobjdump -sass /w/.pf/refdrv > /w/.pf/refdrv.sass
       G="HMMA HGMMA LDGSTS REDUX ELECT BAR.RED F2FP"
       BAD=0
       for g in $G; do n=$(grep -c "$g" /w/.pf/refdrv.sass || true); printf "%s=%s " "$g" "$n"; [ "$n" != "0" ] && BAD=1; done
       echo
       [ $BAD = 0 ] || { echo "ILLEGAL sm_80+ INSTRUCTIONS"; exit 1; }
       echo "refdrv and bench both link for sm_75, no sm_80+ instructions"' \
     2>&1 | grep -v "does not match the detected host" | tail -8 | sed 's/^/  /'
  if docker run --rm -v "$REPO:/w" -w /w archlab/nvcc:12.9 bash -c \
      'export PATH=/usr/local/cuda-12.9/bin:$PATH
       cuobjdump -sass /w/.pf/refdrv 2>/dev/null | grep -cE "HMMA|HGMMA|LDGSTS|REDUX|ELECT|BAR\.RED|F2FP"' \
      2>/dev/null | tail -1 | grep -qv '^0$'; then
    say "  pre-flight FAILED: sm_80+ instructions in the reference binary"
    exit 5
  fi
  rm -rf "$REPO/.pf"
  say "  pre-flight passed: both binaries build and link for sm_75"
else
  say "  SKIPPED: no archlab/nvcc container. The push will find out instead."
fi

if [ "$DRY" = 1 ] || [ "$BUILD_ONLY" = 1 ]; then
  say ""
  say "DRY RUN / BUILD ONLY: no network calls made."
  exit 0
fi

# ------------------------------------------------------------------ auth
say ""
say "=== auth ==="
[ -x "$KAGGLE" ] || { say "kaggle CLI not found at $KAGGLE (set KAGGLE=...)"; exit 3; }
if [ -z "${kaggle_API_TOKEN:-}" ] && [ ! -s "$HOME/.kaggle/access_token" ] \
   && [ ! -s "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]; then
  say "no Kaggle credentials. Authenticate once, either:"
  say "    $KAGGLE auth login"
  say "  or write a token to   ~/.kaggle/access_token   (mode 600)"
  exit 2
fi
say "credentials present"

# ------------------------------------------------------------------ push
KID="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$KDIR/kernel-metadata.json" | head -1)"
[ -n "$KID" ] || { say 'no "id" in kernel-metadata.json'; exit 64; }
OWNER="${OWNER:-${KID%%/*}}"
SLUG="${KID#*/}"
REF="$OWNER/$SLUG"
SESSION_MIN="${SESSION_MIN:-30}"

say ""
say "=== push $REF (session cap ${SESSION_MIN}m) ==="
if ! "$KAGGLE" kernels push -p "$KDIR" -t $((SESSION_MIN * 60)) 2>&1 \
     | tee -a "$LOG" | sed 's/^/  /'; then
  say "push failed; nothing was started"
  exit 1
fi

# ------------------------------------------------------------------ poll
QUEUE_WAIT_MIN="${QUEUE_WAIT_MIN:-120}"
POLL_SECS="${POLL_SECS:-30}"
say ""
say "=== wait (queue up to ${QUEUE_WAIT_MIN}m, session cap ${SESSION_MIN}m) ==="
deadline=$(( $(date +%s) + QUEUE_WAIT_MIN * 60 ))
status=UNKNOWN
last=""
while :; do
  [ "$(date +%s)" -ge "$deadline" ] && { say "  gave up waiting; stop it at https://www.kaggle.com/code/$REF"; exit 4; }
  out="$("$KAGGLE" kernels status "$REF" 2>&1)" || true
  st="$(printf '%s' "$out" | sed -n 's/.*status "\(.*\)".*/\1/p' | head -1)"
  st="${st##*.}"; [ -z "$st" ] && st=UNKNOWN
  if [ "$st" != "$last" ]; then say "  [$(date +%H:%M:%S)] $st"; last="$st"; fi
  status="$st"
  case "$st" in
    COMPLETE) break ;;
    ERROR|CANCELLED|FAILED) break ;;
    UNKNOWN)
      if printf '%s' "$out" | grep -q "Cannot access kernel"; then
        say "  $out; the slug in kernel-metadata.json does not exist for $OWNER"
        exit 5
      fi ;;
  esac
  sleep "$POLL_SECS"
done

if [ "$status" != COMPLETE ]; then
  say ""
  say "=== kernel log (this is where the traceback is) ==="
  "$KAGGLE" kernels logs "$REF" 2>&1 | tail -150 | sed 's/^/  /' | tee -a "$LOG"
  say "run did not complete: $status"
  exit 6
fi

# ----------------------------------------------------------------- output
DEST="$OUTROOT/$STAMP"
mkdir -p "$DEST"
say ""
say "=== download ==="
got=0
for attempt in 1 2 3; do
  if "$KAGGLE" kernels output "$REF" -p "$DEST" --force 2>&1 \
       | tee -a "$LOG" | sed 's/^/  /'; then got=1; break; fi
  say "  attempt $attempt/3 failed; sleeping 20s"; sleep 20
done
[ "$got" = 1 ] || { say "all 3 download attempts failed; partial output left in $DEST"; exit 7; }

while IFS= read -r z; do
  [ -n "$z" ] || continue
  unzip -qo "$z" -d "$DEST/$(basename "$z" .zip)" 2>&1 | sed 's/^/    /'
done < <(find "$DEST" -name '*.zip' -type f | sort)

# the refstate dump is ~30 MB at the dump batch and must not come home; the
# comparison already happened in the notebook and its verdict is in
# obs_equivalence.json. Keeping the binaries would turn a 200 KB artefact set
# into a 60 MB one for no extra information.
rm -rf "$DEST"/*/refstate "$DEST"/refstate 2>/dev/null || true

say ""
say "=== what came back ==="
find "$DEST" -type f | sort | while read -r f; do
  printf '    %-56s %8s\n' "${f#$DEST/}" "$(du -h "$f" | cut -f1)"
done | tee -a "$LOG"

for s in "$DEST"/*/SUMMARY.txt "$DEST"/SUMMARY.txt; do
  [ -f "$s" ] || continue
  say ""
  say "--- SUMMARY ---"
  cat "$s" | sed 's/^/  /' | tee -a "$LOG"
  break
done

say ""
say "done. results in $DEST"
say "  notebook as it ran: https://www.kaggle.com/code/$REF"
