# gpush — syncing a working tree to a notebook GPU, and running work there

`gpush/` is a general-purpose command-line tool. It takes a local working
tree, works out what changed, transfers only what changed, runs one command on
a remote accelerator, and brings the results home with a record of exactly which
bytes produced them.

It is not specific to the mc-cuda-tinygrad study. The study is its first user,
and the tool was extracted because the study kept hitting the same two failure
modes for the third time.

Two backends, because they are genuinely different transports:

| backend | what it is | why it is not the other one |
|---|---|---|
| `kaggle` | REST only. A tree ships as a versioned **dataset**, a kernel attaches it with `dataset_sources` and walks `/kaggle/input`. | No SSH, no inbound route, ephemeral, session-capped. `kernels push` / `status` / `output` is the whole transport. |
| `colab` | `ssh` to a real Linux VM over a cloudflared reverse tunnel. tar in, argv, tar out. | A Colab GPU session is a real machine with a real POSIX userspace. The right client for it is a shell tool, not a notebook API. Wrapping it in a notebook schema would be inventing a transport that does not exist in order to look uniform. |

---

## 1. Install

There is nothing to install.

```bash
cd /Users/huyan00/mycode/mc-cuda-tinygrad
./gpush/gpush --list-backends
```

`gpush/` is a directory of **standard library only** Python 3.10+. No venv, no
`pip install`, no container. Verified: the only imports anywhere under `gpush/`
are `abc, argparse, ast, dataclasses, datetime, glob, gzip, hashlib, io, json,
os, platform, re, shlex, shutil, socket, stat, subprocess, sys, tarfile,
tempfile, textwrap, unittest, zipfile`.

That is a hard constraint here, not a preference. **No local containers.**
Anything needing `nvcc`, `ptxas` or SASS disassembly is planned to happen on the
remote side; this machine's disk was filled by container images once already
(`notes/DECISIONS.md`), and a tool that measures CUDA must not be the reason it
happens again.

The tools it *shells out to* are: `git` (for the manifest), `tar`, `ssh`,
`scp` (macOS ships all three), and the `kaggle` CLI for the Kaggle backend only.

### Optional: a convenient alias

```bash
alias gp='./gpush/gpush'      # or: python3 -m gpush
```

---

## 2. Configure

### Kaggle

`kaggle auth login` writes `~/.kaggle/access_token`; or place a token there
yourself. Username `tenghuzhang`, token 37 bytes at `~/.kaggle/access_token`,
verified working 2026-09-29.

The CLI is found on `PATH`, or named explicitly:

```bash
./gpush/gpush kaggle doctor \
  --dataset tenghuzhang/gpush-smoke-payload \
  --kernel  tenghuzhang/gpush-smoke \
  --kaggle-cli .venv/bin/kaggle
```

```
[ok  ] kaggle CLI             .venv/bin/kaggle
[ok  ] credentials            tenghuzhang (ACCESS_TOKEN at ~/.kaggle/access_token)
[ok  ] weekly quota           GPU 4.29h used, 25.71h remaining of 30.00h
[ok  ] dataset                tenghuzhang/gpush-smoke-payload (ready)
[ok  ] kernel                 tenghuzhang/gpush-smoke
[ok  ] payload transport      one dataset version + dataset_sources -> /kaggle/input/gpush-smoke-payload
```

### Colab

1. Paste `remote/colab_ssh.py` into a Colab notebook and run it. **Before you
   run it: Runtime → Change runtime type → GPU.**
2. It installs sshd, sets the root password, opens a cloudflared reverse
   tunnel, prints `nvidia-smi`, and prints the hostname.

```bash
./gpush/gpush colab doctor \
  --host xxxx-yyyy-zzzz.trycloudflare.com \
  --password probe
```

The password goes through `SSH_ASKPASS`, not `sshpass` — see §7. `--host` is an
argument rather than configuration **because the hostname changes every
session**; a stored one is a hostname to a machine that no longer exists.

Environment variables, if you prefer: `GPUSH_COLAB_HOST`, `GPUSH_COLAB_USER`,
`GPUSH_COLAB_PORT`, `GPUSH_COLAB_PASSWORD`, `GPUSH_COLAB_IDENTITY`,
`GPUSH_COLAB_ASKPASS`, `GPUSH_KAGGLE_CLI`.

---

## 3. The backend interface

This is the contract, in full. Six methods, three phases, and nothing else.
`gpush/backend.py` is the authoritative version.

```
SYNC   plan(tree)               -> Plan      offline; zero network, zero credentials
       push(tree, plan, *, full_resync=False) -> SyncState
                                                  make the remote match the tree
       status()                  -> Status    what the remote is doing right now

RUN    doctor()                 -> Report    is this transport reachable, with what, at what cost
       run(tree, argv, **kw)    -> Run       execute ONE command on the remote device

OUT    fetch(state, dest)       -> [Path]    artefacts home, beside their provenance record
```

**Three properties of that shape matter more than the names.**

**`push` is not `run`.** Push spends bytes and an API call. Run spends
*accelerator quota* — on Kaggle a session against a weekly budget, on a Colab
free tier the session itself, which may also be reclaimed mid-command. The CLI
prints which one it is before doing it. A user should never have to read the
source to know whether a command cost them GPU hours.

**`push` returns, `run` blocks, and `run` does not push.** There is no
"push-and-wait-and-download" verb in the backend. `gpush <backend> run`
composes them, and that composition lives in the core — so a new backend
inherits the ordering, the guards, the gates and the provenance for free, and
cannot accidentally get them in a different order.

**Four things a backend must not do**, each of which would break a property
the core depends on:

- *invent its own fingerprint.* `tree.fingerprint` is already salted with the
  backend name and the target id. A backend that hashes again, differently, has
  broken the thing that makes the no-op check meaningful.
- *promote a baseline to `confirmed`.* Only the core does that, after a run
  completes and its artefacts arrive, and it is the same code for every backend.
- *assume a device.* Colab does not publish which GPU a free session gets.
- *fall back to a local execution "for convenience".* That is how a CPU number
  ends up in a table of GPU numbers.

### Adding a third backend

One module in `gpush/backends/`, subclass `Backend`, `@register` it. The core
does not change — including the CLI, which reads the registry rather than a
list. Plain ssh to a box with a stable hostname, Modal, crabbox: all the same
shape. `gpush/tests/test_plan_is_offline.py` registers a dummy backend and
asserts it appears in `--help` without touching the core, so this claim is
tested rather than asserted.

---

## 4. The subcommands

| command | what it does | costs |
|---|---|---|
| `plan` | the diff, offline | **nothing.** no network, no credentials |
| `doctor` | can this backend be used now, with what | some network, to answer honestly. **no accelerator time** |
| `push` | make the remote match the tree | bytes + API calls. **no accelerator time** |
| `run` | push if needed, then execute one command | **SPENDS GPU QUOTA** |
| `status` | what the remote is doing | nothing |
| `fetch` | artefacts home | nothing |

### The manifest and the fingerprint

The manifest is `git ls-files` ∪ `git ls-files --others --exclude-standard`:
tracked files plus untracked files git does not ignore. That is what a human
means by "the working tree". Build products are gitignored here and stay out.

The fingerprint is `sha256` over the canonical, sorted, content-addressed
serialisation of the whole manifest, **salted with the backend name and the
target id**. So it changes if and only if the file *set* changes or any byte of
any file changes — and a payload pushed to one backend can never be mistaken
for a payload pushed to another. It is not a git hash: untracked files count,
and a dirty working tree is the normal state of this kind of work.

Hashing is incremental the way rsync's is. A file whose `(size, mtime_ns)`
matches the baseline reuses the baseline's sha256; a file that differs — new,
resized, or touched — is read and hashed, which is "the content of every
changed file" in the literal sense.

**`--checksum` and why it exists.** The shortcut is faster and it is not
complete. A test in `gpush/tests/test_incremental.py` constructs the case it
misses: a file edited to the *same length* with its mtime *restored*
(`git checkout` of an old revision, `touch -r`, a copied file). The default
path reports the fingerprint as unchanged; `--checksum` reports it changed. If
that test ever passes by accident, `--checksum` has become pointless.

---

## 5. The guards, and the incident behind each

Every guard runs before any upload, and each can only *refuse*. None of them
auto-repairs a remote, because a tool that silently fixes a payload is a tool
that can silently ship the wrong bytes.

### 5.1 The mass-deletion guard

> Refuses a delete-sync that would prune more than 30% of the previous
> manifest. `--delete-limit` moves the threshold; `--allow-mass-delete` and
> `--full-resync` override it, and the gate then prints `[OVERRIDDEN]` — an
> override that prints nothing is indistinguishable from a guard that did not
> fire.

Delete-sync is a feature (a stale file on the remote is a measurement
attributed to the wrong code) and a hazard (one `rm -r` away from deleting the
remote tree). The threshold is a fraction of the *previous* manifest, so
shrinking the tree to nothing reads as 100%.

Observed live, refusing before upload:

```
guards REFUSED the push
  [REFUSE] mass-deletion guard   would prune 5/11 = 45.5% of the remote
                                 manifest (limit 30%); first: derivation/_fp32.py,
                                 derivation/_util.py, derivation/divide.py ...
  fix: if this is intended, re-run with --allow-mass-delete (or --full-resync
       to reset the remote workdir and re-seed)
```

### 5.2 The payload integrity gate — DECISIONS.md #18

> **The incident.** 2026-09-27. A dataset version was published four seconds
> before the kernel was pushed. The version had not finished being created, so
> it did not mount. The kernel ran against an empty `/kaggle/input`, and all
> 15 battery items recorded `skip`. The run was `COMPLETE`. A GPU session was
> spent producing the string "skip" fifteen times.
> (`notes/DECISIONS.md` #18; `out/kaggle/gpu-v1/SUMMARY.md` records
> `payload sha256: n/a`.)
>
> **Why it survived everything.** The notebook had nothing to compare against.
> An empty payload and a correct one produced identical output, so the only
> symptom was that every probe skipped — which reads like a legitimate result.

Four gates now run between the upload and the run:

1. **title round-trip**, before the upload. Kaggle derives the slug from the
   **title**, not the id, and `kernels push` answers 409 *"Your kernel title
   does not resolve to the specified id"* **after** accepting the body. gpush
   derives the title from the slug (`gpush smoke` → `gpush-smoke`) and checks
   the round-trip. `make_notebook.py` regenerates metadata on every run, so a
   hand-edited title is a 409 waiting to happen.
2. **version created** — `datasets version`, or `datasets create` on first use.
   There is no `datasets update`.
3. **version READY** — the race. Must report `ready` before the kernel is
   pushed, or it does not mount. This is gate 3 of `DECISIONS.md` #18.
4. **published file set and published bytes** — `datasets files` must list
   exactly what gpush wrote, and then the archive is **downloaded back** and
   every member's sha256 compared to the manifest that was packed. This is the
   strong form: it checks content, not names, and it makes "the remote has the
   bytes I just hashed" a verified statement instead of an inference from an
   HTTP 200.

Because they all run inside `push`, `gpush kaggle run` cannot spend a session
on an unverified payload. `--skip-verify` exists and says so in its own output.

### 5.3 The stale-file gate

> **The failure.** A payload carrying a file that is no longer the file the
> working tree holds. The remote executes the stale copy; the number is
> attributed to the current tree.

`payload_completeness()` compares the expected manifest against what is
actually there and refuses on three counts: **missing** (the empty-payload
shape), **extra** (the stale-file shape — an extra member is code nobody asked
for), and **differing** (the same path, different bytes). The comparison is by
content hash, never by name or timestamp.

Verified in `gpush/tests/test_guards.py`, including that the gate is not
vacuous: it passes an honest archive and catches a `STOWAWAY.py` smuggled into
one.

### 5.4 The runner verifies its own payload

The gates above are all local-side. The remote-side half is in the generated
notebook (`gpush/backends/runner_nb.py`), and it is the direct answer to
`payload sha256: n/a`:

1. locate the mount (it has been both `/kaggle/input/<slug>/` and
   `/kaggle/input/datasets/<owner>/<slug>/`)
2. **unpack** `payload.tgz` — Kaggle mounts it as a single *file* and unpacks
   nothing. A runner that assumed an extracted directory found no tree root
   and refused to run; correctly, and expensively. That cost one real session.
3. re-hash every file of the **unpacked tree** and compare to the attestation
   that shipped inside the archive
4. report the device
5. run the command in that resolved tree root
6. `SystemExit(5)` on a mismatch

Hashing the unpacked tree, rather than the archive, is deliberate: it proves the
bytes that are about to be *executed*, not the bytes that were uploaded. The
notebook's verdict comes back as `GPUSH-VERIFY.json` and `gpush run` prints it:

```
runner's payload verdict: MATCHED
```

A run whose verdict is not `MATCHED` is forced to a nonzero exit and its
`provenance.json` says no number from it may be quoted.

### 5.5 The no-op check, and why it is two conditions

> The skip decision is:
> ```
> skip  =  plan.empty  AND  store.is_proven(fingerprint)
> ```
> Both halves. A matching fingerprint with a `pending` baseline is **not** a
> no-op, and that is the entire point.

**Why `pending` exists.** The baseline has two states. A push writes
`pending` — uploaded, unattested. Only a run that **completes** and whose
artefacts arrive promotes it to `confirmed`, and only `confirmed` satisfies the
skip test.

So the #18 sequence becomes: `push` → `pending` → gates refuse to run until
`ready` → run completes → `confirmed`. A dataset version created but never run
leaves a `pending` baseline, and the next `push` re-uploads rather than
skipping. The lost GPU run is not merely less likely; the specific shape of it
is now the state that the tool refuses to treat as done.

Verified live:

```
[pass] no changes, skipping   baseline fingerprint matches but is PENDING: the
                              previous push was never attested by a completed run
                              (this is the shape of the lost run in
                              DECISIONS.md #18), so it is re-pushed
                              [not proven: NOT skipping]
```

### 5.6 The device gate

> A Colab free session is frequently CPU-only, and Kaggle can queue on a machine
> with no NVIDIA device. Either way: a CPU-only box is not a slower GPU run, it
> is a different machine.

`run` reads the device off the machine and, with no GPU, **refuses before the
command runs**. `--allow-cpu` permits it and the result is recorded as a CPU
result. `Device[sm_75]` is **reported, never interpreted**: what sm_75 means for
a given kernel is the study's question, not the transport's. If a backend
silently upgraded "no bf16 tensor cores" into "cannot run", every sm_75 result
would vanish and the comparison the study exists to make would disappear too.

Observed on a real T4:

```
device     Tesla T4  15360 MB  compute capability 7.5  driver 580.159.04
           compute capability sm_75 (below sm_80: no bf16 tensor cores;
           recorded, not interpreted)
```

---

## 6. Incremental sync

### Delete-sync

Remote files that were previously synced and no longer exist locally are
pruned, and the diff says so before anything moves. For Kaggle the tarball is
rebuilt from the manifest so a pruned file cannot survive in the archive, and
gate 5.3 proves it: a file present locally and absent from the manifest would
appear as `EXTRA` and be refused.

Observed live — 5 files pruned, archive rebuilt from 11 members to 6, and the
published bytes re-hashed to confirm:

```
[pass] pack                    6 files, 0.7 KB, sha256 147855bf9566ef83
[pass] published file set      Kaggle lists exactly ['SHA256SUMS', 'payload.tgz']
[pass] published bytes         published archive re-hashed: 6 members match the manifest exactly
```

### `--full-resync`

Resets the remote workdir and re-seeds from scratch. The mass-deletion guard
becomes moot and says so.

On Kaggle the dataset API will not accept a byte-identical version, so a full
resync adds a nonce to the attestation. The nonce changes the archive's bytes
without changing any file's content hash, and therefore **without changing the
fingerprint** — a re-seed is a new version of the same tree, not a new tree.

One measured correction to the belief recorded in `kernels/pack.py`: on
2026-09-29, `kaggle datasets version` **accepted** a byte-identical payload on
kaggle 2.2.4 rather than rejecting it as identical. So the API's "identical"
rejection is not a usable incremental signal, and gpush does not rely on it.
The fingerprint and the `confirmed` state are the signal.

### Determinism, and a bug worth recording

The tarball pins `mtime`, `uid`, `gid`, `uname`, `gname` and mode, and emits
entries in sorted order — but `tarfile.open(..., "w:gz")` builds its `GzipFile`
with **the current time in the gzip header**. Two packs of an identical tree
differed in their first ten bytes, and the sha256 of an unchanged payload moved
on every push. gpush builds `GzipFile` by hand with `mtime=0`. Verified: three
packs spaced 1.1 s apart, one sha256. The sums file is a **sidecar sibling**,
never a member — a file inside the archive cannot name the archive's own hash,
since writing the name changes the bytes, and a hash printed there that does
not verify is worse than none.

### Honest statement of the incrementality

For Kaggle a dataset version is a full snapshot: **there is no partial-upload
primitive**. What gpush provides is the delta *decision*, the delta *report*,
delete-sync, and a real cost of zero for a no-op — not fewer bytes on the wire.
Anything claiming a byte-level delta over a snapshot upload is describing a
diff layered on a snapshot. For the SSH backend it really is a delta transport.

---

## 7. The Colab backend

Deliberately boring, because the machine is a real Linux VM: `tar -xzf -` over
one `ssh`, the command, `tar -czf -` back. Not `rsync` — one code path, no
dependency on `rsync` being present on a disposable box that is rebuilt
constantly, and a single stream cannot half-apply.

**Password auth without a dependency.** `sshpass` would be a new install. It is
not needed: OpenSSH ≥ 8.4 reads a password from `SSH_ASKPASS` when
`SSH_ASKPASS_REQUIRE=force` and there is no tty, which is exactly this case. A
two-line helper script supplies the password, so gpush depends on nothing that
is not already on the machine. A test asserts `sshpass` is *absent*, so the
no-dependency path stays the tested one.

**The three real constraints.**

1. *Disposable, possibly reclaimed mid-run.* Nothing assumes the box survives.
   `run` wraps the command in `timeout` and tees to a remote `run.log` so
   whatever completed before a disconnect is still fetchable. A lost session
   reports `DISCONNECTED`, not a crash.
2. *The hostname changes every session.* `--host` per invocation; the resolved
   host is written into the provenance record. A hostname that changes per
   session is exactly the thing that must be recorded rather than remembered.
3. *The runtime may be CPU-only.* §5.6.

### ⚠️ UNTESTED END TO END

**gpush's Colab path has never been run against a live Colab runtime.** No live
Colab session was available. Concretely, these are **not** exercised:

- the tar-over-ssh transfer, and the remote `mkdir`
- the cloudflared reverse tunnel
- what happens when a session is reclaimed mid-command
- `fetch`'s tar-back and extraction

What **is** tested (17 tests in
`gpush/tests/test_colab_offline.py`): registration on the same six-method
interface; refusal with no host, with the reason; `doctor` failing and
declaring itself untested; correct ssh argv including keepalives and
`BatchMode`; the askpass mechanism actually printing the password; the CPU-only
refusal; that the device probe is stdlib-only and reports "no GPU" correctly on
this Mac; and that the payload passes the same integrity gate.

`gpush colab doctor` on this backend is **a checklist of preconditions, not a
verification that it works**, and it prints that line itself.

---

## 8. Provenance

Every run writes `provenance.json` and `manifest.json` beside its artefacts, in
`out/gpush/runs/<backend>/<slug>/<timestamp>-<fingerprint12>/`. A measurement
that cannot be traced to a tree state is not a measurement.

From a real run:

```json
{
  "target": "tenghuzhang/gpush-smoke-payload",
  "fingerprint": "92935b7bea5de8ee09ea85b362c3d98d07f884dfea4c41532416d82c5baa7f12",
  "files": 5, "bytes": 122569,
  "command": ["python", "derivation/divide.py", "mechanism_sentence"],
  "command_line": "python derivation/divide.py mechanism_sentence",
  "status": "COMPLETE", "exit_code": 0,
  "started": "20260929-015259", "finished": "20260929-015351",
  "compute_capability_reported": "sm_75",
  "remote": {"dataset": "tenghuzhang/gpush-smoke-payload",
             "kernel": "tenghuzhang/gpush-smoke",
             "sha256": "…"},
  "device": {"name": "Tesla T4", "memory_mb": 15360, "compute_cap": "7.5",
             "driver": "580.159.04", "cuda_runtime": "13.0", "nvcc": "12.8",
             "kernel": "6.12.90+", "python": "3.12.13", "count": 2},
  "host": {"platform": "macOS-27.0-arm64-…", "gpush": "1.0.0"},
  "plan": {"added": 0, "modified": 0, "deleted": 0, "unchanged": 5},
  "gates": [ … ],
  "artefacts": [{"path": "GPUSH-VERIFY.json", "bytes": 221, "sha256": "0bea…"}]
}
```

Notes on three fields that were **wrong in the first implementation** and are
fixed, each caught by inspecting a real record rather than by reasoning:

- `bytes` read 0, because `total_bytes` only counted files that had been
  *rehashed* and not those whose hash was reused.
- `compute_capability_reported` read `sm_7.5`; nvidia-smi reports a dotted pair
  and the architecture name drops the dot. It is `sm_75`.
- `remote` read `{}`, because a run whose push was a no-op skip has no
  `SyncState` to draw it from, and the one early run recorded the emptiness
  permanently. There are now three fallbacks — the push that just happened, the
  durable baseline, then the backend's own configured address — and
  `mark_confirmed` merges rather than replaces.

`host` and `device` are separate on purpose: "which Mac pushed this" and "which
GPU ran this" are different questions, and conflating them is how a Metal
number ends up attributed to a T4.

**Provenance is written on failure too.** A run refused by a guard, refused by
a gate, or refused by the device gate still produces a record — the refusal is
the evidence, and the next person to hit the same wall should not have to
rediscover it.

---

## 9. Worked example — Kaggle

```bash
# 1. offline review: no network, no credentials, costs nothing
./gpush/gpush kaggle plan \
  --dataset tenghuzhang/gpush-smoke-payload --kernel tenghuzhang/gpush-smoke \
  --include derivation

# 2. transfer, no accelerator time. Gates run before this returns.
./gpush/gpush kaggle push \
  --dataset tenghuzhang/gpush-smoke-payload --kernel tenghuzhang/gpush-smoke \
  --kaggle-cli .venv/bin/kaggle --include derivation

# 3. run one command. THIS SPENDS GPU QUOTA.
./gpush/gpush kaggle run \
  --dataset tenghuzhang/gpush-smoke-payload --kernel tenghuzhang/gpush-smoke \
  --kaggle-cli .venv/bin/kaggle --include derivation \
  --session 20 --queue-wait 25 --poll 20 \
  -- python derivation/divide.py mechanism_sentence
```

`plan` output — note the fingerprint is already bound to this backend and this
dataset, so it cannot be compared against a `colab` plan:

```
== plan (offline: no network, no credentials) ================================
  tree         5 files, 119.7 KB (from git)
  fingerprint  92935b7bea5de8ee09ea85b362c3d98d07f884dfea4c41532416d82c5baa7f12
  baseline     confirmed  92935b7bea5d  5 files
  change       +0 ~0 -0 =5   (0.0 KB to send)

  no changes: the working tree matches the baseline

  2825 working-tree files are NOT in the payload and cannot be measured remotely:
```

`push` output on the first upload of a tree:

```
== backend gates =============================================================
  [pass  ] title round-trip       'gpush smoke' slugs to 'gpush-smoke'
                                    (derived from the slug, never hand-edited)
  [pass  ] pack                   5 files, 36.8 KB, sha256 5ebd713ba4161b90
  [pass  ] dataset version        dataset created
  [pass  ] dataset READY          dataset reports ready
  [pass  ] published file set     Kaggle lists exactly ['SHA256SUMS', 'payload.tgz']
  [pass  ] published bytes        published archive re-hashed: 5 members match
                                  the manifest exactly
  baseline is now PENDING: uploaded, but not yet attested by a run.
```

`run` output, real, 2026-09-29:

```
== execute on the remote device ==============================================
  status     COMPLETE
  device     Tesla T4  15360 MB  compute capability 7.5  driver 580.159.04
             compute capability sm_75 (below sm_80: no bf16 tensor cores;
             recorded, not interpreted)
  started    20260929-015259    finished 20260929-015351

  runner's payload verdict: MATCHED
  baseline PROMOTED to confirmed. The next push of this tree is a proven no-op.
```

And the second, unchanged invocation — **5 files skipped, zero transferred**,
verified against the live dataset:

```
[pass  ] no changes, skipping   baseline is proven by a completed run

no changes, skipping the transfer: the remote fingerprint 92935b7bea5d is
already proven by a completed run; 5 files, none of them transferred
```

---

## 10. Worked example — Colab

**Untested end to end** (§7). This is the intended sequence, not a transcript of
a successful run.

```bash
# 1. On Colab: paste remote/colab_ssh.py, run it, note the printed hostname.
#    Runtime -> Change runtime type -> GPU FIRST.

# 2. Check preconditions. Reports the device it actually finds.
./gpush/gpush colab doctor \
  --host xxxx-yyyy-zzzz.trycloudflare.com --password probe

# 3. Offline review.
./gpush/gpush colab plan --include derivation --include probes

# 4. Transfer: tar over one ssh pipe, then verify every file's hash ON THE BOX.
./gpush/gpush colab push \
  --host xxxx-yyyy-zzzz.trycloudflare.com --password probe --include derivation

# 5. Run. Refuses on a CPU-only session unless --allow-cpu.
./gpush/gpush colab run \
  --host xxxx-yyyy-zzzz.trycloudflare.com --password probe --include derivation \
  --timeout 1800 \
  -- python derivation/divide.py mechanism_sentence
```

`doctor` on this backend prints, unprompted:

```
  [FAIL] host          no --host; run the remote/colab_ssh.py bootstrap on the
                       Colab side and paste the host
  UNTESTED end to end: gpush's Colab path has never been run against a live
  Colab runtime. Treat `doctor` as a checklist, not as a verification.
```

---

## 11. Tests

```bash
python3 -m unittest discover -s gpush/tests
```

**75 tests, all passing, all offline** — no Kaggle quota, no network.

| file | asserts |
|---|---|
| `test_plan_is_offline.py` | `plan` completes with every socket entry point poisoned and `HOME` pointed at an empty dir; a third backend registered at runtime appears in `--help` with no core change |
| `test_guards.py` | every guard **refuses** and every override **passes and says it was overridden**; the empty-payload and stale-file gates; that the integrity gate is not vacuous on a real archive; the baseline state machine including a corrupt-baseline fallback; tar determinism |
| `test_incremental.py` | the fingerprint moves only when bytes move; hash reuse; one edit rehashes one file; delete-sync; `--checksum` catching the same-size same-mtime edit the shortcut misses; fingerprint bound to backend and target |
| `test_colab_offline.py` | interface parity; refusal with no host; ssh argv; askpass; the CPU-only refusal; the probe is stdlib-only and reports no-GPU correctly on this Mac |
| `test_runner_notebook.py` | the generated notebook parses, has cell ids, carries no unreplaced template token, verifies the **unpacked tree**, and that `validate()` actually fails on a broken cell |

The negative assertions matter most: a guard test that only checks the passing
case proves nothing, because a guard that never refuses also always "passes".

---

## 12. What was actually verified, and how

Everything below was performed. Nothing is transcribed.

**Verified end to end on Kaggle**, against `tenghuzhang/gpush-smoke-payload` /
`tenghuzhang/gpush-smoke` with real credentials, 2026-09-29:

- a first push: dataset created, all five gates passed, published archive
  downloaded and re-hashed, 5/5 members matching
- an unchanged push while the baseline was `pending`: **refused to skip**,
  re-uploaded, gates re-passed
- a GPU run: `COMPLETE`, **Tesla T4, sm_75, 15360 MB, driver 580.159.04, CUDA
  runtime 13.0, nvcc 12.8, 2 devices**, runner's payload verdict `MATCHED`, the
  command executed in the resolved tree root, exit 0
- baseline promoted to `confirmed`
- a third unchanged push: **skipped, 5 files, 0 bytes, 0 uploaded**
- delete-sync: 5 files pruned from the remote, archive rebuilt to 6 members,
  published bytes re-verified
- the mass-deletion guard **refusing** a real 45.5% prune before upload, and the
  `--allow-mass-delete` override passing with `[OVERRIDDEN]` printed
- `--full-resync`: nonce added, bytes changed, content verified identical
- tar determinism: three packs 1.1 s apart, one sha256
- 75 offline unit tests

GPU quota attributable to the kernel runs above: **4.26 h → 4.29 h of 30 h**,
about **2 minutes** for the sessions that were pushed. (The 4.14 h → 4.26 h
step before them is *not* attributable to gpush — at that point no kernel had
been pushed, only dataset versions, and dataset versions cost no accelerator
time; another actor in this repo was running kernels at the same time.) Six
kernel runs in total, two of which failed on the bugs described below and
whose sessions are most of that two minutes.

**Three real bugs the live runs found**, none of which reasoning had caught:

1. **Kaggle mounts `payload.tgz` as a file and unpacks nothing.** The runner
   assumed an extracted directory, found no tree root, and `SystemExit(3)`'d.
   It refused correctly — the gate did its job — but cost a session. Fixed by
   having the runner unpack it and hash the *unpacked tree*.
2. **`kaggle kernels logs` returns a JSON array, one record per stream chunk**,
   printed with a leading comma per line. Read as text, the device probe's
   marker line never appears at line start, so **every** device report came back
   as "unknown" — indistinguishable from "the box died", and both wrong.
3. **The gzip timestamp** (§6), found because the sha256 of an unchanged
   payload moved between two consecutive pushes.

Each was fixed at the cause, not the symptom, and each now has a test or an
assertion that would catch its return.

**Not verified:**

- the entire Colab path (§7)
- a partial-upload benefit on Kaggle — there is none to have, and §6 says so
- behaviour under a genuinely concurrent second writer of the same dataset
- `fetch`'s retry path under a real TLS failure; the retry loop is written and
  exercised only in the sense that a failure keeps what landed

---

## 13. Relationship to the rest of the repo

`gpush/` is **new and standalone**. It duplicates no existing file and modifies
none. The paths it reuses are read-only references, cited so the reasoning is
traceable:

- `remote/kaggle_run.sh` — the proven four-call loop. gpush reimplements it in
  `gpush/backends/kaggle.py` rather than shelling out, because a backend that
  wraps a shell script cannot be reused by a third transport and because the
  gates have to sit between its calls.
- `kernels/pack.py` — the payload-as-versioned-dataset idea, the read-only
  `/kaggle/input` reasoning, and the sidecar sha256. gpush generalises the
  include list into a manifest and fixes the gzip determinism.
- `remote/sync.py`, `remote/sync.sh` — concurrent work on the Kaggle payload
  specifically. **Not touched, and not duplicated:** those are for the study's
  payload; gpush is a general tool that happens to implement the same
  fingerprint-and-guard idea once, for any backend.
- `minecu_ref/kaggle/make_notebook.py` — the origin of the title-from-slug rule
  and the 409. gpush derives the title and checks the round-trip.
- `remote/colab_ssh.py` — the Colab bootstrap. gpush consumes the hostname it
  prints; it does not replace it.
- `remote/preflight.py` — the device report. `gpush/probes.py` is a smaller,
  stdlib-only version of the same idea, because the one gpush needs must also
  run inside a generated notebook cell on a box whose only guarantee is
  `python3`.

## 14. Adding a backend — worked

```python
# gpush/backends/mine.py
from ..backend import Backend, RemoteStatus, Report, Run, SyncState, register
from ..manifest import diff

@register
class MyBackend(Backend):
    name = "mine"
    spends_gpu = True
    transport = "one sentence on what this physically is."

    def plan(self, tree):  return diff(self.store.manifest(), tree)
    def push(self, tree, plan, *, full_resync=False): ...
    def status(self):     return RemoteStatus(phase="IDLE")
    def doctor(self):     return Report(ok=True)
    def run(self, tree, argv, **kw): return Run(ok=True)
    def fetch(self, state, dest): return []

    def describe_target(self): return {"host": "..."}   # optional
```

Add the file. `gpush --list-backends` and `gpush mine plan` now exist. No
change to `cli.py`, `manifest.py`, `state.py`, `gates.py`, `provenance.py` or
`backend.py` — which is the claim, and `test_plan_is_offline.py` checks it by
registering a backend at runtime and looking for it in `--help`.
