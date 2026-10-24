# 21 — The sync loop: what changed, what was pushed, and what it cost

**The problem this note closes.** Code is written on an Apple Mac
(`Device.DEFAULT == METAL`) and every meaningful NVIDIA measurement has to run
on Kaggle's free T4. The transfer path is not the problem: `kernels/pack.py`
builds a payload, `remote/kaggle_run.sh` calls `kaggle datasets create` /
`datasets version`, the notebook attaches the payload through
`dataset_sources` and walks `/kaggle/input`, and the mine.cu baseline went end to
end through it (`notes/17`, `minecu_ref/kaggle/`).

The problem is that the loop was **not incremental** and **not observable**.
Every push repacked and re-uploaded the whole payload, so a one-line edit in
`kda/tiny_kda.py` cost a full upload, and nothing anywhere said what had
changed. Worse, the ways that loop could be *wrong* all fail silently, and this
repo has already paid for two of them.

**What is now in the repo.** Two new files, and one new state file:

| file | what it is |
|---|---|
| `remote/sync.py` | the incremental sync driver: manifest, diff, guards, publish, delegate |
| `remote/sync.sh` | the one command; picks the venv, refuses a too-old python, execs sync.py |
| `remote/last-pushed.json` | the last-pushed manifest, in the repo, so it is diffable and reviewable |

**Label convention, as elsewhere in this repo.** `MEASURED` — a number produced
by running the tool on this host against the live dataset. `[ARTIFACT]` — read
out of a file. Every timing and count in §3 is `MEASURED` here; nothing is
quoted from a log of an earlier run.

---

## 0. The answer, first

1. **A real incremental push works end to end**, and the common case costs
   nothing: with all 25 payload files unchanged, `--sync` detects a no-op,
   creates **no** dataset version, and reports 25 files skipped. It still
   verifies the published bytes by downloading them.
2. **The incrementality is a decision, not a delta transfer, and the difference
   is the API's, not this file's.** A Kaggle dataset version is a full snapshot
   with no partial-upload primitive. So what is incremental is *whether a
   version is created at all*, the *report* of what changed, and the
   *provenance* that ties a result to an exact tree state. rclone would not
   change any of that (no Kaggle backend, no SFTP/WebDAV/S3 endpoint), and
   describing this as a delta upload would be a lie about the wire.
3. **The guards are the point.** Six of them exist because each maps to a
   failure mode that produces a *plausible wrong answer* rather than an error,
   and two of those have already cost this repo a GPU session.

---

## 1. The loop

Three modes, and the distinction between them is the distinction between free
and metered.

```
  edit locally
      |
      v
  remote/sync.sh --check      ZERO network calls, ZERO credentials.
      |                       reads files, prints a diff, refuses on a guard.
      |                       ~0.1 s. This is the mode to run before every push.
      v
  remote/sync.sh --sync       publishes a dataset version ONLY if the diff is
      |                       non-empty. No kernel push, no GPU time, no cost.
      |                       ~0.1 s plan + ~12 s upload, and 0 s if no-op.
      v
  remote/sync.sh --run        sync, then hand off to remote/kaggle_run.sh
                              --skip-pack: build notebooks, validate, push the
                              kernel, poll to a bounded timeout, download,
                              print an index. ONE GPU session.
```

`--run` is the single entry point for the loop. It does not reimplement
`remote/kaggle_run.sh` or `kernels/build_notebooks.py`; it delegates to the
first and lets it call the second. `kernels/pack.py` is *imported* for its
`INCLUDE` list and its builder rather than having a second copy of the include
set written down here, because two copies of an include list is two things to
forget to update, and the whole omission guard depends on that list having
exactly one owner.

### Commands

```bash
remote/sync.sh --check              # offline. no token needed. exit 0 / 1 / 3
remote/sync.sh --sync               # publish only if something changed
remote/sync.sh --run                # sync + kernel push + poll + results
remote/sync.sh --run --only preflight --timeout 25    # a short bounded run
remote/sync.sh --status             # what is published; one API call, no hashing
remote/sync.sh --check --json       # machine-readable plan under out/sync/
remote/sync.sh --check --fail-on-dirty   # exit 3 when changes are pending
remote/sync.sh --adopt-remote       # seed the baseline from what is published
```

### What each mode costs

| mode | network | credentials | GPU time | `MEASURED` wall clock |
|---|---|---|---|---|
| `--check` | **none** | **none** | none | **0.10–0.13 s**, and it is two `stat`s and one `sha256` per file |
| `--sync`, changed | 1 dataset version + 1 status poll + 1 download | yes | none | **19.8–23.1 s** total, of which **11.2–12.0 s** is the upload |
| `--sync`, no-op | 1 status poll + 1 download | yes | none | **7.4 s**, all of it verifying the published bytes |
| `--run` | the above + kernel push, poll, output download | yes | **one GPU session** | see §3.3 |
| `--status` | 1 `datasets files` + 1 `datasets download` | yes | none | ~6 s |

The no-op `--sync` is not free, and that is deliberate: it costs a ~72 KB
download and about 7 s to hash the published tarball and compare it to the
baseline. The alternative is a no-op that *assumes* the remote still agrees, and
"assumed" is how a stale payload becomes a measurement of the wrong code. If
that 7 s is ever unwelcome, `--check` is the mode that skips it, and
`--check` is also the one that needs no token.

---

## 2. The manifest, and what a diff is a diff against

`remote/last-pushed.json` is the baseline, and it lives in the repo rather than
in a dotfile under `$HOME` so that it can be reviewed in a diff. It is written
with sorted keys and two-space indent, one entry per file, so **a one-line
source edit is a one-line manifest diff**:

```json
"probe/kda/cost.py": {
  "bytes": 11220,
  "mode": "0o644",
  "sha256": "c089839d6281...",
  "src": "kda/cost.py"
}
```

Alongside the per-file map it records the payload tarball's own sha256, the
dataset's version file list as the API reported it, the delta counts, and a
`tree` digest: one sha256 over `{arcname: sha256}` for the whole set. That
`tree` value is the handle a result is attributable to, and it is what
`--status` compares the live dataset against.

The diff is over **content**, not over the tarball. The tarball's own bytes are
*not* reproducible — `MEASURED`, packing the same unchanged tree three times:

```
sha256  cc4922ae8ac8b0e81a5c7d8a10539c447a21ad3fbb099e88b391b1880a376ec2
sha256  ed81f72adf1b073190240f1d260526b9f2c13feb549b6466dcb0438a6461b4e2
sha256  ea62eeaba16e0cb7c99b6f998f0672a4fe8ef4da6bf89c5f86b9588324897ccb
```

Three different digests for one tree. The cause is in the gzip header, not in
the tar: `pack.py` fixes every **member's** mtime, uid, gid, mode and uname
(verified: all 25 members report `mtime 0`, `uid 0`, `gid 0`, `mode 420`,
`uname root`), but `tarfile.open(mode="w:gz")` stamps the **gzip header** with
the wall clock:

```
magic 1f8b  cm 8  flg 8  mtime 1790614507  xfl 2  os 255
```

So an "identical" repack is never byte-identical, and a tarball-level no-op
check could never work. This is a real property of the existing pipeline, it is
why the diff is per-file, and it is also why the dataset API's refusal of an
"identical version" is a fallback rather than the mechanism. Per-file sha256 *is*
stable: two consecutive `build_manifest` passes over an unchanged tree produce
the same `tree` id, `84f8af500c1b2c63`, every time.

**`mode` is recorded but does not decide an upload.** `pack.py` writes every
member as `0o644`, so a permission bit cannot change the payload. Treating a
`chmod` as a change would turn a `chmod` into an upload and then into the API's
"identical version" refusal. `MEASURED`:

```
$ chmod 600 kda/cost.py && remote/sync.sh --check
  delta       NO-OP: 0 changed  0 added  0 deleted  25 unchanged
  mode-only   1 file(s) had a permission bit changed. pack.py writes 0o644, so
              the payload bytes are identical and this is not an upload.
```

---

## 3. What was actually run, and what it cost

All of this was performed against the live dataset
`tenghuzhang/mc-cuda-tinygrad-payload` with the token at
`~/.kaggle/access_token`, on 2026-09-29, on this Mac. Quota at the time
(`kaggle quota`): **GPU 4.14 h used, 25.86 h remaining** of 30.

### 3.1 First, an honest correction about what was already published

The published dataset already existed and already matched the local tree on
**19 of 25** files. The 6 that differed were local edits made after the last
manual push. Rather than pretend the first push was from nothing,
`--adopt-remote` seeded the baseline **from the published dataset itself** by
downloading the version and reading its `probe/MANIFEST.json`, so the first diff
was a real diff against what was genuinely deployed.

### 3.2 The incremental push

`--check` first, offline, no credentials:

```
  baseline    f7595f814cb34b37  pushed 2026-09-28T16:53:29Z  (25 files)
  payload     25 files, 236.5 KB of source
  delta       6 changed  0 added  0 deleted  19 unchanged

  CHANGED   probe/kda/perf_kda.py              7326 B  b4ea97dc145b   <- was bf505ca0bcb4
  CHANGED   probe/kda/replay_kda.py            6931 B  6e4dcaf1de7a   <- was 17e9c2be355d
  CHANGED   probe/probes/divide_precision.py   9394 B  7f090f34b287   <- was 8f13b222c502
  CHANGED   probe/probes/fp32_correctness.py   9413 B  ad48a91be898   <- was be693520bf36
  CHANGED   probe/probes/oc/oc_report.py       7918 B  ff947cf0850d   <- was 7a7831fbf3d2
  CHANGED   probe/probes/oc/tiny_oc.py        13300 B  3acf66f5da07   <- was 7f2076a20d68
```

`--sync` then, `MEASURED`:

```
=== publish ===
  payload.tgz  76.7 KB  25 files
  tarball     76.7 KB  sha256 2f36e20224dbf2ab
  dataset     version published
  waiting     for the version to be mountable (this is the race that cost
              DECISIONS.md #18 a session)
  dataset     ready
  upload      12.0s
  cross-check published version matches: 3 files, tarball sha256 verified
              against the live dataset
  baseline    written to remote/last-pushed.json (25 files)
```

**Payload size and file count, `MEASURED`:** 25 files, **236.5 KB of source**,
**76.7 KB** as the gzipped tarball that is actually uploaded. For scale: 6 of
those 25 files changed, so 19 files (about 190 KB of source) were identified as
unchanged and *not* re-sent individually — they travel inside the snapshot,
because the API gives no way to send them separately, and they are excluded from
the *decision* and from the *report*.

### 3.3 The no-op, on the second run

Immediately after, with nothing touched, `--sync` again:

```
  delta       NO-OP: 0 changed  0 added  0 deleted  25 unchanged
  unchanged   25 files skipped, sha256 identical to remote/last-pushed.json
  NO-OP       nothing changed since the last push. No dataset version was created.
              25 files skipped: every sha256 is identical to remote/last-pushed.json
  cross-check the published version still equals the baseline: 2 files, tarball
              sha256 verified by download
```

**25 of 25 files skipped, and no dataset version created.** The 7.4 s went into
verifying that the published bytes still hash to what was pushed.

### 3.4 A one-file change, end to end

To show the incremental path on its own, one comment line was appended to
`kda/cost.py`, checked, pushed, and re-checked — and the file restored
afterwards, with the restore itself pushed so the tree and the dataset agree
again:

| step | delta | outcome |
|---|---|---|
| append 1 line to `kda/cost.py` | 1 changed, 24 unchanged | `--sync` published in 11.2 s upload |
| immediately after | **NO-OP, 25 skipped** | no version created |
| restore the file | 1 changed, 24 unchanged | `--sync` published; tree and dataset agree |

`git status` confirms `kda/cost.py` and the kernel metadata are back to their
original content; `remote/last-pushed.json` records the restored tree.

### 3.5 A real GPU run through `--run`

`--run --only preflight --timeout 25`, which is the cheap end of the battery:

- sync stage: a **no-op**, detected and skipped, 25 files unchanged
- delegated to `remote/kaggle_run.sh --skip-pack`, which built both notebooks,
  ran `kernels/validate_notebooks.py` (**all 1 kernel directory valid**, 6 cells,
  5 code, every code cell compiles), and pushed **kernel version 7**
- the queue held the run at `QUEUED` past the first two polls, which is the
  documented reason a queue deadline must be much larger than the session cap
  (`remote/kaggle_run.sh:30` records a 2026-09-27 kernel still `QUEUED` after
  24 minutes)

The full poll → output → index leg is `remote/kaggle_run.sh`'s job and was not
re-exercised beyond the push here; the run was left to complete on Kaggle's side
rather than holding this session for the battery.

**Timing summary, `MEASURED` on this host:**

| operation | wall clock |
|---|---|
| `--check`, full plan over 25 files | **0.10–0.13 s** |
| `--sync`, no-op path including the download-and-hash verification | **7.4 s** |
| `--sync`, publishing path (plan + pack + upload + ready-wait + cross-check) | **19.8–23.1 s** |
| `--run` up to and including the kernel push | ~40 s of driver time, then **queue-bounded** |

The queue wait is the unbounded part and it is not this file's to fix. It is
bounded here by `--timeout` (25 min in the run above, 90 by default), and the
session itself is capped server-side by `kernels push -t`.

---

## 4. The guards, and the failure each one exists for

Every guard below corresponds to a way this loop can produce a **plausible
wrong answer** rather than an error. The two marked *incident* have already cost
this repo time.

### 4.1 The race, which has already cost a session — *incident*

**The failure.** A dataset version that is still being created does not mount.
The kernel starts, `/kaggle/input` is empty, and every probe records
`payload has no X` as a *skip*. A skip is not an error, so the run completes
`COMPLETE`, looks like a broken notebook, and reports a battery of zeros.

**The incident.** `notes/DECISIONS.md` #18, and it is still the repo's only
recorded **lost run**: `out/kaggle/20260927-231908-gpu/SUMMARY.md` — Tesla T4,
15360 MiB, `payload sha256: n/a`, **all 15** battery items `skip`, 40 s of
battery. `out/kaggle/20260927-231908-gpu/dataset-version.log` ends at "Dataset
version is being created". The payload itself was sound (26 files,
`kernels/payload/payload.sha256` verified, `probes/divide_precision.py` present),
which is exactly what makes it a race and not a build bug. The hardware, the
probe and the quota were all present; the pipeline threw the measurement away.

**The guard.** `--sync` blocks on `kaggle datasets status` until it reads
`ready` before it returns, and `--run` does not delegate until sync has returned.
`MEASURED`, the wait is real and takes effect on every publishing push above.

### 4.2 A stale file in a payload, which is a measurement of the wrong code

**The failure.** A file present in the deployed payload but not in the current
tree, or vice versa. Nothing in the notebook can notice: it verifies that what it
received hashes to what the payload *claimed*, which is self-consistent, and
`kda/tiny_kda.py` will happily run the old version of itself.

**The guard, two directions.**

*Before publishing*: `kaggle datasets files <id> --format json` gives the live
version's own name list. Any name the API still lists that the local pack does
not produce is reported and refuses the push.

*After publishing*: the same list is re-read, and the published
`payload.tgz` is **downloaded and hashed** and compared against the digest the
just-built tarball has. Names and sizes are not enough — the bytes are what the
notebook verifies at run time, and a same-length stale tarball is exactly the
case a size check cannot see.

`dataset-metadata.json` is excluded from the name comparison, and that is
`[ARTIFACT]` from the live dataset: it describes the dataset and is never stored
as one of its files, so including it would report a permanent phantom mismatch.
This was found by the cross-check refusing a push that had in fact succeeded.

**Disagreements are reported, never resolved by picking a side.** If the repo
and the dataset disagree about what is deployed, the tool says both sha256s and
refuses to declare a winner. Choosing the local side would make a stale remote
look fine; choosing the remote side would make a real change look like nothing
happened.

### 4.3 The silent omission

**The failure.** A file in the working tree, in a directory the payload draws
from, that no include rule covers and nothing reports. It is not a broken build;
it is a *wrong build* that looks right.

**The guard, in two parts with different severities**, because conflating them
would make the guard cry wolf on 31 files and be ignored:

- **fatal**: a packed file *imports* or *names* a tree file the payload does not
  carry. This is the real breakage, and it is invisible until a probe dies
  minutes into a GPU session.
- **reported, not fatal**: a file in a payload source directory that is simply
  not packed, with no reason recorded.

The distinction is the point. `MEASURED`, the current tree has **31** such
files (`kda/NOTES.md`, `kda/tpu/*`, `probes/fp32_correctness_fixed.py`,
`remote/gcp_run.sh`, …) and **none of them is imported by anything packed**, so
nothing is broken. A guard that failed on all 31 would be right 31 times and
useless. The list is printed anyway, because the day a result surprises someone,
that list is the first thing to read.

**How the import closure is derived, not transcribed.** For every packed `.py`,
`ast` gives the imported names, and each name is resolved against the search
path *that file actually builds* — which means statically folding the
`sys.path.insert(...)` calls in it. That last part is not decoration:

```python
# kda/perf_kda.py:33-34
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "probes"))
...
from bench import bench_ms          # line 40
```

`probes/bench.py` is not next to `perf_kda.py`. It resolves only because of that
`../probes` insert. A guard that did not model it would either report a false
`bench` (unresolvable, therefore "external, fine") or, worse, a real omission
would hide behind the same misreading. Same story in
`probes/oc/run_oc.py:13-14`, `probes/oc/oc_report.py:22-23`, and
`remote/preflight.py:107-110`, which builds `kda` from
`os.path.dirname(os.path.dirname(__file__))`.

`MEASURED` on the current payload: **29 local imports resolved inside the
payload, 0 missing, 13 external** (`numpy`, `tinygrad`, `torch`, `fla`, and the
stdlib — all pip-installed on the runner by the notebook's own cell). And
`0` unresolved `sys.path` edits, so the guard had the full picture rather than
a partial one.

Verified by construction: writing a probe that imports a helper not in the
payload produces exactly one entry —
`('probes/_sync_probe.py', '_sync_helper', 'probes/_sync_helper.py')` — and
refuses.

**A bug found while building this, worth recording because it was in the
permissive direction.** `__file__` was originally seeded *after* the
constant-folding pass, so `remote/preflight.py`'s two-statement path building
collapsed and its search dir came out as `remote/kda` instead of `kda/`. The
symptom was `ref`, `test_ref` and `tiny_kda` all being classified **external**,
i.e. "not the payload's problem". A guard that is wrong in the direction of
"fine" is worse than no guard, because it reports success. Fixed by seeding
`__file__` first; the resolved count went 26 → 29 and the three false externals
became resolved payload imports.

**The Makefile references.** `probes/oc/Makefile` is not code and there is no
parser for it here, so relative-path tokens are scanned instead, classified into
four kinds. `MEASURED`: 2 resolve to packed files, 0 are built by the Makefile,
2 are provided at run time by the notebook (`../../netherite/blaze/core` is
sparse-cloned, `../../.venv/bin/python` is created by the payload cell), 0
missing. The `../../` is written against the **payload root**, not the tarball's
`probe/` prefix, so the paths are normalised against the payload root or every
reference looks like it stays inside and nothing is ever reported — that was the
first version, and it silently reported zero.

### 4.4 A file whose bytes change during the hash pass

**The failure.** The manifest describes a tree state that never existed, and
every downstream number is attributed to it.

**The guard, in two passes, because one is not enough.** Each file is `stat`ed,
read, and `stat`ed again — if `(size, mtime_ns, ino, dev, mode)` moved, the file
is refused. Then, after the pass, **every** file is re-`stat`ed against its
recorded stamp, because a file can change *after* it was hashed and before the
tarball is built, which would make the shipped tarball differ from the manifest
that justified shipping it.

`MEASURED`, both paths exercised:

```
CASE 1: Refuse: refusing to push: these files changed bytes while they were
        being hashed, so the manifest describes a tree state that did not exist
CASE 2: drift_since -> ['kda/cost.py']
        manifest sha before: c089839d6281  now: 77065f6fe64d
```

The file was restored byte-for-byte in both cases.

### 4.5 A hand-edited kernel title — *incident, twice*

**The failure.** Kaggle derives a kernel's slug from its **title**, and
`kernels push` rejects the upload with **409 Conflict**, "Your kernel title does
not resolve to the specified id". The check is cheap; doing it after assembling
a notebook and spending the budget is not.

**The incident.** `minecu_ref/kaggle/make_notebook.py:355-362` documents it: a
hand-written title was clobbered on every run, because that file *regenerates*
`kernel-metadata.json` each time, and the push then failed with exactly that
409. Its fix is to **derive** the title from the slug and assert the round-trip
locally.

**The guard.** `--run` re-derives the slug from the title with the same
transformation and refuses before delegating. `MEASURED`, by injecting
`"title": "Hand Edited Title"` into `kernels/probe/kernel-metadata.json` and
running `--run`:

```
REFUSED: kernel title 'Hand Edited Title' slugs to 'hand-edited-title', which is
not the id slug 'mc-cuda-tinygrad-probe-battery'. `kaggle kernels push` will
answer 409 ... Derive the title from the slug; do not hand-edit it, because the
generator overwrites it.
```

The metadata was restored afterwards and the round-trip re-verified. Note the
message says *derive from the slug*, not *fix the title*: with a generator that
overwrites the file, a hand-written title is a 409 waiting to happen. That has
now bitten this repo twice, and the fix is to never hand-edit it.

### 4.6 A no-op that becomes an empty dataset version

**The failure.** Publishing a version whose payload is byte-identical to the
current one. Kaggle refuses it ("a version with identical files"), and the
refusal reads like an error in a script that treats a non-zero exit as a failed
push.

**The guard.** The diff decides. With an empty diff, `--sync` **does not call
`datasets version` at all** — §3.3 is the proof: no version, 25 files skipped,
exit 0. `--force` overrides it for the recovery case where the remote and the
repo disagree and you need to republish regardless.

### 4.7 Attribution

**The failure.** A number in a note with no way to say which code produced it.

**The guard.** `--run` copies the manifest into the run's own output directory
as `payload-manifest.json`, alongside a `sync-plan.json` of the plan that
justified the push, and prints the `tree` id and the tarball sha256 next to the
index. A result directory therefore carries the exact tree state that produced
it, without anyone having to remember to record it.

---

## 5. Deletions, stated plainly

A file that leaves the payload is reported explicitly, in its own section, and
split into two cases that mean different things:

```
  DELETED   probe/kda/ghost.py        10 B  000000000000   source kda/ghost.py
                                                                    removed from the tree
```

- **source gone from the tree**: a real deletion. Reported, and it will be in the
  next push.
- **source still in the working tree**: **fatal**. This is not a deletion, it is
  an omission that has been renamed — the tree says the file is current, the
  payload says it is not, and a measurement of the payload is a measurement of
  the wrong code. Same failure mode as §4.2, caught from the other side.

The reason this is worth doing at all: a deleted file cannot announce itself.
The notebook unpacks the payload it is given, verifies the payload against
itself, and runs whatever is in it. Nothing in that chain has any idea what the
working tree contains.

---

## 6. What is *not* claimed here

- **The incrementality is not a delta transfer.** No bytes are saved on the wire
  when something changes: Kaggle's dataset version is a full snapshot and the
  API exposes no partial upload. What is saved is the *upload* when nothing
  changed, the *decision*, the *report*, and the *review*.
- **rclone is not a solution.** It has no Kaggle backend, and Kaggle exposes no
  SFTP, WebDAV or S3 endpoint, so `rclone sync` would be a wrapper around the
  same API this repo already calls — with a second dependency, and no ability to
  see inside a dataset version.
- **No local containers, and no dependency that would need one.** `sync.py`
  imports only the standard library, so `--check` runs on a machine that has
  never seen `kaggle`. Anything requiring nvcc, ptxas or SASS disassembly is
  planned to run server-side on Kaggle, with its artefacts pulled back through
  `kaggle kernels output`. Nothing here was measured on this Mac beyond file
  hashing and HTTP.
- **The full poll → download → index leg of `--run` is `kaggle_run.sh`'s, and it
  has its own history** (`out/kaggle/20260928-014248-gpu` is a complete GPU run
  through it). This note re-exercised the sync half and the push half. The
  battery in §3.5 was left to run on Kaggle's side rather than holding the
  session open; the numbers in §3.5 are the ones this tool produced.
- **The published payload's 6 differing files were pre-existing local edits**,
  not edits this work made. §3.1 says so because "the first push uploaded 25 new
  files" would have been the convenient thing to write and would have been false.

---

## 7. The one thing to remember

Run `--check` before every push. It is free, it needs no token, it takes a tenth
of a second, and it is the only place in this loop where a mistake is cheap.

Everything else in this note is about the case where `--check` was not run.
