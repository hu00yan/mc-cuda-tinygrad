#!/usr/bin/env bash
# run.sh -- one command, bounded, self-verifying: measure the TinyGrad side
#           against the hand-written CUDA baseline on Kaggle's free T4.
#
#   ./run.sh --dry-run     everything except the network calls
#   ./run.sh --build-only  assemble and validate the notebook, push nothing
#   ./run.sh               build, push, poll, download, print
#
# WHY A KAGGLE KERNEL AND NOT A DATASET. `kaggle kernels push` uploads exactly
# one file: the notebook named by "code_file". Sidecars in the kernel directory
# are not uploaded, so `kernels.cu` (105 KB), `refdrv.cu` (14 KB) and the whole
# `minecu.tinygrad` package travel inside the notebook as base64 and are sha256-checked
# before anything is compiled. A mismatch is a hard failure, not a warning: a
# timing from a source that is not the one the write-up describes is worse than
# no timing.
#
# WHY INTERNET IS ON. `pip install tinygrad`. The alternative is vendoring a
# wheel into the payload, which pins a version silently; the installed version
# is printed in the log and hashed into the run.
#
# COST. One Kaggle GPU run. The notebook's own budget is enforced inside the
# notebook, and the session cap is passed to `kernels push -t` so Kaggle enforces
# it server-side. One full run is a small fraction of the ~30 GPU-hours/week
# free tier.
#
# NO GPU ON THIS HOST. Nothing here needs one. The measurement happens on Kaggle.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # minecu/tinygrad/
REPO="$(cd "$HERE/.." && pwd)"                              # repo root
KDIR="$HERE/kaggle/kernel"
PY="${PY:-$REPO/.venv/bin/python}"
KAGGLE="${KAGGLE:-$REPO/.venv/bin/kaggle}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTROOT="${OUTROOT:-$REPO/out/minecu.tinygrad}"

DRY=0
BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)    DRY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    -h|--help)    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

# ------------------------------------------------------------- local gate
# The verification must pass on this host, with no GPU, before anything is
# pushed. A run that measures an implementation which does not compute the
# right thing is worse than no run: it produces a ratio against a correct
# baseline, and the ratio is about the wrong thing.
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
mkdir -p "$OUTROOT"
LOG="$OUTROOT/$STAMP-run.log"

say "=== minecu.tinygrad: verify the tensor render on this host (CPU, no GPU) ==="
if ! "$PY" "$HERE/verify.py" 2>&1 | tee -a "$LOG" | sed 's/^/  /'; then
  say "  verification FAILED; refusing to push. A timing from an unverified"
  say "  implementation is a measurement of the wrong thing."
  exit 4
fi
say "  verification passed"

say ""
say "=== assemble the notebook ==="
mkdir -p "$KDIR"
"$PY" "$HERE/kaggle/make_notebook.py" \
    --repo "$REPO" \
    --out "$KDIR/minecu.tinygrad_t4.ipynb" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
"$PY" - "$KDIR/minecu.tinygrad_t4.ipynb" <<'PY' 2>&1 | sed 's/^/  /' | tee -a "$LOG"
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

# The refdrv must compile for sm_75 BEFORE the push. Compiling it here, in the
# archlab container, costs a minute and turns a 12-minute round trip that dies
# on a syntax error into a local failure. The reference lane's run.sh does the
# equivalent with bench_cuda/build.sh.
say ""
say "=== pre-flight: the payload layout, as the notebook will materialise it ==="
# Three pushes of this notebook died before it measured anything, and each cost
# a run. All three are now gates:
#   * `kernels.cu` passed to nvcc twice (refdrv.cu already includes it), so the
#     link failed with `multiple definition of launch_render_uint8_prebasis'.
#     Gate: the compile gate below, with the notebook's EXACT command.
#   * the payloads were written to /kaggle/working/<basename>, so `minecu.tinygrad`
#     was not a package. Gate: stage 1 below, which asserts the travel names.
#   * the import check sat BEFORE `pip install tinygrad`, so it failed on a
#     missing tinygrad. Gate: stage 2 below, which runs write-then-import with a
#     real tinygrad already importable, i.e. the notebook's order.
#
# A pre-flight that tests a different command than the notebook runs is worse
# than none, because it looks like a gate.
if ! "$PY" - "$KDIR/minecu.tinygrad_t4.ipynb" <<'PY' 2>&1 | sed 's/^/  /'
import base64, hashlib, json, os, shutil, subprocess, sys, tempfile
nb = json.load(open(sys.argv[1]))
src = "".join(c for cell in nb["cells"] if cell["cell_type"] == "code" for c in cell["source"])
i = src.index("PAYLOADS = ")
j = src.index("]\n", i)
payloads = json.loads(src[i + len("PAYLOADS = "):j + 1])

OUT = tempfile.mkdtemp(prefix="minecu.tinygrad_pf.")
try:
    n_pkg = 0
    for name, b64, want in payloads:
        data = base64.b64decode(b64)
        assert hashlib.sha256(data).hexdigest() == want, f"hash mismatch {name}"
        if name.startswith("minecu/tinygrad/"):
            n_pkg += 1
        dest = os.path.join(OUT, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)   # the package stays a package
        open(dest, "wb").write(data)
    assert n_pkg >= 9, f"only {n_pkg} payloads under minecu/tinygrad/: the package would be flattened"
    assert os.path.isfile(os.path.join(OUT, "minecu.tinygrad", "__init__.py")), "no __init__.py"
    # sass_matrix.py resolves tools/archlab relative to itself, so the card needs
    # it there too. Found this the hard way: the stage-2 import check below
    # failed on `No module named 'inspect_ptx'` and would have failed on the card
    # the same way, at the SASS stage, after the timings were already measured.
    assert os.path.isfile(os.path.join(OUT, "tools", "archlab", "inspect_ptx.py")), \
        "inspect_ptx.py not shipped: sass_matrix.py cannot import its categoriser"
    print(f"  stage 1 write: {len(payloads)} payloads, hashes match, "
          f"{n_pkg} under minecu/tinygrad/, tools/archlab/inspect_ptx.py present")

    # stage 2: the notebook's order -- sys.path, a real tinygrad, then the package
    check = (
        "import sys, importlib\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import tinygrad\n"
        "m = importlib.import_module('minecu.tinygrad.render')\n"
        "for s in ('blocks','world','cpu_reference','codegen','bench','battery',"
        "'capture_cuda','analyse_codegen','sass_matrix'):\n"
        "    importlib.import_module('minecu.tinygrad.' + s)\n"
        "print('  stage 2 import: minecu.tinygrad.render from', m.__file__)\n"
        "print('  stage 2 import: all 9 submodules OK')\n"
    )
    p = subprocess.run([sys.executable, "-c", check, OUT], capture_output=True, text=True)
    if p.returncode != 0:
        print("  stage 2 import FAILED: " + p.stderr.strip().splitlines()[-1])
        sys.exit(1)
    print(p.stdout.rstrip())
    for s in ("battery.py", "verify.py", "capture_cuda.py", "analyse_codegen.py"):
        q = subprocess.run([sys.executable, os.path.join(OUT, "minecu.tinygrad", s), "--help"],
                           capture_output=True, text=True)
        assert q.returncode == 0, f"{s} --help rc={q.returncode}: {q.stderr[-400:]}"
    print("  stage 3 run: battery/verify/capture_cuda/analyse_codegen all --help cleanly")

    # stage 4: STATIC check that every subprocess call in the driver is handed
    # strings. The previous gate stopped at `--help`, so a driver that passed a
    # PYTHON TUPLE where subprocess wanted a string -- `CODEGEN_STEPS = 1,4,16,32`
    # substituted unquoted, which is a tuple, not a string -- sailed through and
    # died on the card with "expected str, bytes or os.PathLike object, not
    # tuple".
    #
    # The subtlety, and why this is written the way it is: the tuple arrives at
    # the call site as a NAME, not as a literal, so a walk that only looks at the
    # arguments of `sh()` sees a `Name` node and passes. The first version of this
    # gate did exactly that and was decorative -- it was tested by injecting the
    # bug and it did not fire. So the check resolves each argument Name to its
    # module-level binding first, and only then asks what type that is.
    import ast, re
    tree = ast.parse(src)
    binds = {}                      # top-level `NAME = <literal>` -> type name
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
           and isinstance(node.targets[0], ast.Name):
            binds[node.targets[0].id] = type(node.value).__name__
    n_calls, bad = 0, []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "sh"):
            continue
        n_calls += 1
        if not node.args:
            continue
        a0 = node.args[0]
        if not isinstance(a0, (ast.List, ast.Tuple)):
            bad.append(f"line {node.lineno}: sh() first arg is {type(a0).__name__}, not a list")
            continue
        for el in a0.elts:
            t = type(el).__name__
            if isinstance(el, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
                bad.append(f"line {node.lineno}: sh() argument is a literal {t}")
            elif isinstance(el, ast.Name):
                b = binds.get(el.id)
                if b in ("Tuple", "List", "Dict", "Set"):
                    bad.append(f"line {node.lineno}: sh() argument is name {el.id!r}, "
                               f"which is bound to a {b} at module level "
                               f"(subprocess needs a string)")
    assert n_calls >= 3, f"only {n_calls} sh() calls found; the walk is not matching"
    assert not bad, "driver subprocess calls are malformed:\n    " + "\n    ".join(bad)
    print(f"  stage 4 args: all {n_calls} subprocess call(s) in the driver resolve to "
          f"strings; {len(binds)} module-level bindings checked for tuple-valued names")

    # and prove the gate is not decorative: inject the exact bug it exists to
    # catch and require the check to fire.
    inject = src.replace('CODEGEN_STEPS = "1,4,16,32"', "CODEGEN_STEPS = 1,4,16,32")
    if inject == src:
        inject = re.sub(r"CODEGEN_STEPS = [^\n]*", "CODEGEN_STEPS = 1,4,16,32", src)
    assert inject != src, "could not inject the tuple bug to self-test stage 4"
    t2 = ast.parse(inject)
    b2 = {}
    for node in t2.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
           and isinstance(node.targets[0], ast.Name):
            b2[node.targets[0].id] = type(node.value).__name__
    fired = False
    for node in ast.walk(t2):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
           and node.func.id == "sh" and node.args:
            a0 = node.args[0]
            if isinstance(a0, (ast.List, ast.Tuple)):
                for el in a0.elts:
                    if isinstance(el, ast.Name) and b2.get(el.id) == "Tuple":
                        fired = True
    assert fired, "stage 4 does NOT fire on the tuple bug it was written for; " \
                  "it is decorative and must not be trusted"
    print("  stage 4 self-test: fires on an injected unquoted-tuple substitution "
          "(so it is not decorative)")
finally:
    shutil.rmtree(OUT, ignore_errors=True)
PY
then
  say "  the payload layout or stage order the notebook will use is broken; refusing to push"
  exit 6
fi

say ""
say "=== pre-flight: the exact nvcc commands the notebook runs, for sm_75 ==="
# These MUST be the same commands, in the same shape, that the notebook driver
# issues. The first push of this notebook failed on `multiple definition of
# launch_render_uint8_prebasis' because the notebook passed BOTH kernels.cu and
# refdrv.cu to nvcc while refdrv.cu already #includes kernels.cu, and the
# pre-flight compiled refdrv.cu alone -- so the pre-flight passed and the
# notebook failed. A pre-flight that tests a different command is worse than no
# pre-flight, because it looks like a gate.
#
# refdrv.cu is compiled ALONE (it includes kernels.cu).
# bench.cu is compiled WITH kernels.cu (it declares and links the launchers).
if command -v docker >/dev/null && docker image inspect archlab/nvcc:12.9 >/dev/null 2>&1; then
  docker run --rm -v "$REPO:/w" -w /w archlab/nvcc:12.9 bash -c \
      'set -e
       export PATH=/usr/local/cuda-12.9/bin:$PATH
       FLAGS="-O3 --use_fast_math -std=c++17 -arch=sm_75 -diag-suppress 177"
       echo "--- refdrv alone (it includes kernels.cu) ---"
       nvcc $FLAGS -DMINECU_KERNELS='\''"/w/${UPSTREAM_KERNELS}"'\'' \
         /w/minecu/tinygrad/kaggle/refdrv.cu -o /w/.pf_refdrv
       echo "--- bench with kernels.cu (it declares and links the launchers) ---"
       nvcc $FLAGS /w/${UPSTREAM_KERNELS} /w/bench_cuda/bench.cu \
         -o /w/.pf_bench
       echo "--- link and sm_75 gate on the refdrv binary ---"
       /usr/local/cuda-12.9/bin/cuobjdump -sass /w/.pf_refdrv > /w/.pf_refdrv.sass
       G="HMMA HGMMA LDGSTS REDUX ELECT BAR.RED F2FP"
       BAD=0
       for g in $G; do
         n=$(grep -c "$g" /w/.pf_refdrv.sass || true)
         printf "  %s=%s" "$g" "$n"
         [ "$n" != "0" ] && BAD=1
       done
       echo
       [ $BAD = 0 ] || { echo "  ILLEGAL sm_80+ INSTRUCTIONS"; exit 1; }
       echo "  refdrv and bench both link for sm_75, no sm_80+ instructions"' \
    2>&1 | tail -14 | sed 's/^/  /'
  # the pipeline above ends with the container's exit status only through the
  # last command, so re-run a cheap explicit check for the gate
  if docker run --rm -v "$REPO:/w" -w /w archlab/nvcc:12.9 bash -c \
      'export PATH=/usr/local/cuda-12.9/bin:$PATH
       cuobjdump -sass /w/.pf_refdrv | grep -cE "HMMA|HGMMA|LDGSTS|REDUX|ELECT|BAR\.RED|F2FP"' \
      2>/dev/null | tail -1 | grep -qv '^0$'; then
    say "  pre-flight FAILED: sm_80+ instructions in the reference binary"
    exit 5
  fi
  rm -f "$REPO"/.pf_refdrv "$REPO"/.pf_bench "$REPO"/.pf_refdrv.sass
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
  say "  get one at https://www.kaggle.com/settings/api"
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

# ---------------------------------------------------------------- output
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

# The refstate dump is ~30 MB at the dump batch and must not come home; the
# comparison already happened in the notebook and its verdict is in
# obs_equivalence.json. Keeping the large binaries would turn a 200 KB
# artefact set into a 60 MB one for no extra information.
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
