#!/usr/bin/env bash
# gcp_run.sh -- provision a GPU box, run every probe, bring the results home,
#              and delete the box. No matter how this script exits.
#
#   ./remote/gcp_run.sh
#
# Cost control, because GPU instances are billed by the hour:
#   - a spot instance, not on-demand (typically ~60-70% cheaper, and if it is
#     preempted the trap below still cleans up and the logs are already home)
#   - --max-run-time, a hard cap the API enforces
#   - no persistent disk; results travel back over SSH, so the box holds
#     nothing that matters
#   - `trap cleanup EXIT` so the instance is deleted on success, on failure, and
#     on Ctrl-C. There is no path through this script that leaks an instance.
#
# GPU choice: L4 (sm_89, 24 GB). It is the cheapest card with bf16 tensor
# cores, which is the requirement -- flash-linear-attention's KDA uses a
# tensor-core MMA path on sm_80+ and falls back to a non-MMA path below that.
# T4 is sm_75 and would not exercise the real kernel.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/out/remote/$STAMP"
LOG="$OUT/run.log"
mkdir -p "$OUT"

ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-e2-standard-8}"
GPU="${GPU:-nvidia-l4}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu121-debian-11}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
PROVISIONING="${PROVISIONING:-SPOT}"     # SPOT or STANDARD
MAX_RUN_MIN="${MAX_RUN_MIN:-40}"
DISK="${DISK:-200GB}"
INSTANCE="probe-$STAMP"
SSH="gcloud compute ssh"

say() { printf '%s\n' "$*" | tee -a "$LOG"; }

cleanup() {
  say ""
  say "=== cleanup: deleting $INSTANCE ==="
  gcloud compute instances delete "$INSTANCE" --zone "$ZONE" --quiet \
    >>"$LOG" 2>&1 && say "deleted" || say "delete failed -- delete it manually:"
  say "    gcloud compute instances delete $INSTANCE --zone $ZONE"
}
trap cleanup EXIT INT TERM

say "=== gcloud run $STAMP ==="
say "zone=$ZONE machine=$MACHINE gpu=$GPU disk=$DISK"
say "provisioning=$PROVISIONING (max ${MAX_RUN_MIN} min)  out=$OUT"

# ---------------------------------------------------------------- auth check
say ""
say "=== auth ==="
if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>>"$LOG" | grep -q .; then
  say "not authenticated. run:"
  say "    gcloud auth login --no-launch-browser"
  say "then paste the verification code back here."
  exit 2
fi
say "account $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"

PROJECT="$(gcloud config get-value project 2>/dev/null)"
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  say "no project set. available:"
  gcloud projects list --format='table(projectId,name)' 2>&1 | tee -a "$LOG"
  say "set one with:  gcloud config set project PROJECT_ID"
  exit 2
fi
say "project $PROJECT"

# ---------------------------------------------------------------- quota
say ""
say "=== GPU quota in $ZONE (a zero here is the usual blocker) ==="
gcloud compute regions describe "$ZONE" --format="value(quotas.filter(
  metric:ACCT_GPU_NVIDIA_L4).limit)" 2>&1 | tee -a "$LOG" | sed 's/^/  nvidia-l4 limit: /' || true

# ---------------------------------------------------------------- create
say ""
say "=== create instance ==="
say "    machine $MACHINE, 1x $GPU, $DISK, $PROVISIONING, max ${MAX_RUN_MIN} min"
gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --maintenance-policy=TERMINATE \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK" \
  --provisioning-model="$PROVISIONING" \
  --max-run-time="${MAX_RUN_MIN}m" \
  --scopes=compute-rw,storage-ro \
  --metadata=enable-guest-attributes=true \
  --quiet 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" != 0 ]; then
  say ""
  say "create failed. The usual cause is GPU quota 0. To request it:"
  say "    gcloud compute regions describe $ZONE --format='value(quotas)'"
  say "  then raise nvidia-l4 in the Quotas page, or pick another zone:"
  say "    for z in us-central1-a us-central1-b us-east1-b us-west1-b; do \\"
  say "      echo -n \"\$z \"; gcloud compute regions describe \$z \\"
  say "        --format='value(quotas.filter(metric:ACCT_GPU_NVIDIA_L4).limit)'; done"
  exit 1
fi

# ---------------------------------------------------------------- wait
say ""
say "=== wait for ssh ==="
for i in $(seq 1 40); do
  if gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="true" --quiet >/dev/null 2>&1; then
    say "up after ${i}0s"; break
  fi
  [ "$i" = 40 ] && { say "did not come up in time"; exit 1; }
  sleep 10
done

run() {
  say ""
  say "--- $* ---"
  $SSH "$INSTANCE" --zone="$ZONE" --command="$*" 2>&1 | tee -a "$LOG"
}

# ---------------------------------------------------------------- report env
run "nvidia-smi; echo; nproc; free -g | head -2; df -h / | tail -1; python3 -V"

# ---------------------------------------------------------------- upload
say ""
say "=== upload ==="
$SSH "$INSTANCE" --zone="$ZONE" --command="mkdir -p ~/probe" 2>&1 | tee -a "$LOG"
gcloud compute scp --zone="$ZONE" --recurse \
  "$HERE/remote" "$HERE/kda" "$HERE/probes" "$HERE/README.md" \
  "$INSTANCE:~/probe/" 2>&1 | tail -3 | tee -a "$LOG"
# the C reference needs netherite's headers
$SSH "$INSTANCE" --zone="$ZONE" --command="\
  cd ~/probe && \
  (apt-get update -qq && apt-get install -y -qq git clang >/dev/null 2>&1); \
  mkdir -p netherite && cd netherite && \
  (git clone --filter=blob:none --sparse --depth 1 https://github.com/Infatoshi/netherite . >/dev/null 2>&1 \
   && git sparse-checkout set blaze docs >/dev/null 2>&1) && echo 'netherite headers ok' || echo 'netherite headers FAILED (camera C reference will be skipped)'" 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------- venv
run "cd ~/probe && \
  (command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)); \
  export PATH=\$HOME/.local/bin:\$PATH; \
  (uv venv --python 3.12 .venv >/dev/null 2>&1 && uv pip install --python .venv/bin/python -q tinygrad numpy tabulate >/dev/null 2>&1) \
    || (python3 -m venv .venv && .venv/bin/pip install -q tinygrad numpy tabulate); \
  echo '--- base deps ok ---'; \
  (uv pip install --python .venv/bin/python -q torch triton flash-linear-attention >/dev/null 2>&1 \
    || .venv/bin/pip install -q torch triton flash-linear-attention >/dev/null 2>&1) \
    && echo '--- torch/triton/fla ok ---' || echo '--- torch/triton/fla FAILED (fla comparison will be skipped) ---'; \
  .venv/bin/python -c 'import tinygrad,os; print(\"tinygrad\", tinygrad.__file__)'"

# ---------------------------------------------------------------- preflight
run "cd ~/probe && .venv/bin/python remote/preflight.py"

# ---------------------------------------------------------------- probes
run "cd ~/probe/probes/oc && make check 2>&1 | tail -20; echo '--- report ---'; make report 2>&1 | tail -45"
run "cd ~/probe/kda && for s in test_ref.py test_tiny.py; do echo \"### \$s\"; ../.venv/bin/python \$s 2>&1 | tail -12; done"
run "cd ~/probe/kda && ../.venv/bin/python perf_kda.py 2>&1 | tail -30"
run "cd ~/probe/kda && ../.venv/bin/python replay_kda.py 2>&1 | tail -26"
run "cd ~/probe/kda && ../.venv/bin/python codegen_diff.py 2>&1 | tail -45"
run "cd ~/probe/kda && ../.venv/bin/python kda_vs_fla.py 2>&1 | tail -90"
run "cd ~/probe && .venv/bin/python probes/fp32_correctness.py 2>&1 | tail -22"
run "cd ~/probe && .venv/bin/python probes/divide_precision.py 2>&1 | tail -26"
run "cd ~/probe && .venv/bin/python probes/gather_floor.py 2>&1 | tail -24"

# generated code from the NVIDIA backend
say ""
say "--- collect generated kernels ---"
$SSH "$INSTANCE" --zone="$ZONE" --command="cd ~/probe && .venv/bin/python - <<'PYEOF'
import os, sys
sys.path.insert(0,'kda')
from tinygrad import Tensor, Device
import numpy as np
d = Device[Device.DEFAULT]
cap=[]; orig=getattr(d.compiler,'compile_cached',None)
if orig is not None:
    def spy(src,*a,**k):
        cap.append(str(src)); return orig(src,*a,**k)
    d.compiler.compile_cached = spy
import test_ref as T, tiny_kda
q,k,v,g,beta = T.rnd(2,128,2,2,32,32,seed=0,l2n=True)
tiny_kda.kda_chunk(q,k,v,g,beta,chunk_size=32)
if orig is not None: d.compiler.compile_cached = orig
os.makedirs('out/remote-codegen', exist_ok=True)
for i,s in enumerate(cap): open(f'out/remote-codegen/kernel{i:03d}.txt','w').write(s)
print(f'wrote {len(cap)} kernels to out/remote-codegen/')
PYEOF" 2>&1 | tee -a "$LOG"

run "cd ~/probe && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# ---------------------------------------------------------------- collect
say ""
say "=== collect results ==="
gcloud compute scp --zone="$ZONE" --recurse "$INSTANCE:~/probe/out" "$OUT/" 2>&1 | tail -2 | tee -a "$LOG"
gcloud compute scp --zone="$ZONE" --recurse "$INSTANCE:~/probe/kda" "$OUT/kda-src" 2>&1 | tail -2 | tee -a "$LOG"
say "results in $OUT"
ls -R "$OUT" 2>/dev/null | head -40 | tee -a "$LOG"
