# Reporting rules — a companion to INTEGRITY.md

`INTEGRITY.md` is about **how tools fail silently**. This file is about **how numbers
get misread once a tool is working**. Different content, different failure, so it is
split out rather than appended — because a container named `integrity` that also
holds "how to phrase a ratio" misleads exactly the way `caveats` did.

## Rule 0 — the negative control
> **Every automated check needs a deliberately broken input it MUST reject, and a
> known-good input it MUST accept.** Without both, a check that always returns 0 is
> indistinguishable from one that always passes.

Implemented as `tools/test_sass_lint.py` (7 controls). Verified: reverting bug 4 makes
3 of them fail, so they are tests, not decoration.

## Rule 1 — never print a bare multiple
Every ratio carries `(kernel, cc, config, compile-method)`. Three experiments were
quoted as one gap when the subscripts were dropped in retelling.

## Rule 2 — name the statistic, not just "the average"
> **Report mean / median / max; when they differ by more than 2x, say why; also report
> the count of zero-valued periods.**

This is not merely hygiene. Today `mean 12.88 / max 30` (2.3x, dragged by two periods
with 0 holdings) — and the **zero-count is what killed the "seasonal defect" hypothesis**,
so the rule produced a conclusion rather than only preventing a misreading.

## Rule 3 — magnitude excludes, it does not attribute
> **An order-of-magnitude estimate licenses "this cannot be X". It never licenses
> "therefore it is Y".** Use magnitude to prune explanations; use data to establish them.

The two are not interchangeable. My 6pp figure correctly ruled out "the new stocks are
worse" — and then looked, dangerously, like a computed gap.

## Rule 4 — independence is a DEGREE, not a binary
The standard is **"does not share a failure mode"**, not "is not the same code". Two
implementations agree only *outside the assumptions they share* — so ask **which
assumptions do these two share?**, not "are they different?". Two regexes sharing a
width assumption agree on a wrong answer while looking like triangulation.

## Rule 5 — residuals must not be named
If a quantity is obtained by subtraction (`1.00 − 0.38 = 0.62`), call it **"the
remainder"**, never "the composition change". Naming a residual is not attributing it.
Names containing `other`/`residual`/`misc`/`etc` already declare mixing — leave them
alone. Split containers whose *name* promises one thing while holding several.

## Rule 6 — probe width must match claim width
A negative assertion is written as **"I probed A, B, C; I did not probe D, E"**. A
claim wider than its probe is a structural error that more probing cannot fix — it
needs rewording. Also: a negative assertion is a *specific falsifiable statement*, not
a general unfalsifiable one.

## Rule 7 — when your count disagrees with an independent one, suspect your parser first
A disagreement is a free test. Three of my four bugs were caught by other people; only
the fourth was caught because I checked my own parser instead of dismissing the
contradicting number.

## Rule 8 — when the mechanism is unknown, say "unknown"
Do not manufacture a more attractive mechanism. I wrote this rule and then violated it
in the same message; **only format works, self-discipline does not.** Every mechanism
must name which data it explains and which it leaves 未定.


---

# Team rules — the four from the A-share line (2026-10-24)

These emerged from a *different* repo's investigation. Recorded here because they
generalise, and because memory gets compressed while a file does not. Each is hung on
the specific mistake of the day, not on a tidy summary.

**Numbered `T0`–`T3`, not `0`–`3`.** The main rules already use 0–18; two numbering
systems inside one file made "Rule 0" ambiguous, and a citation to "Rule 0" no longer
says which document is meant.

**T0 — a container whose name honestly declares its own mixed content is not a
defect.** `price_and_other_residual`, `misc`, `other` already tell the reader they hold
many things; no misreading occurs. Split only containers whose name promises one thing
while holding several. *Without T0 the next three degenerate into "tear apart every
composite structure" — which is tempting because dismantling looks like improvement.*

**T1 — method diversity.** Magnitude, distribution, naming, and decomposition are
different instruments. On the A-share line the three of us each reached a wrong
mechanism and each was killed by a different instrument than the one that produced it.
A single method would have stopped at the first plausible answer.

**T2 — a rebuttal must carry something verifiable.** Number, not feeling. Ten
invented mechanisms today, eight killed externally, and every kill carried data.

**T3 — concede immediately, and the precondition is evidence, whoever the opponent
is.** Including yourself: self-correction without evidence is just two evidence-free
parties (me and yesterday's me) refereeing each other. Tim's counterexample: he
retracted "I can't reach writee.org" instantly and with no evidence — that was luck,
not method.

**Dependency order, and it is not flat:** 1 requires 2 requires 3-with-evidence.
Without 2, rule 3 is mere obedience; with it, rule 3 is correction.

**Audit note on my own four withdrawals today:** every one carried the number that
killed it (HMMA=0 rows, grep-vs-parser contradiction, recomputed slots,
`per_name_weight ≡ 1/n`). That is only known *by auditing* — a correction that feels
obvious is exactly the one most likely to arrive unaccompanied.


---

## Rule 2b — statements about the world need an expiry, not just an instant

**Added 2026-10-24 after four people independently hit it in one day.**

Rule 2 says a negative assertion needs `⟨instant, place, probed, not-probed⟩`. That
fixes sentences *about probing*. It does not fix sentences *about the world*, whose
truth can lapse while the sentence about probing stays true.

| sentence type | example | what governs it |
|---|---|---|
| about what I probed | "I probed A, did not probe B" | template with `⟨time, place⟩` |
| about the world, and can lapse | "288 files are missing", "kaggle is installed" | **needs an expiry condition** |

**Rule 2b:** if the sentence states a state of the world, attach **when it would become
false** — as one of two kinds (@Tim's refinement):
- **① triggered** — someone acts (uninstalled, restored, replaced)
- **② dated** — it lapses on its own at a known future point (补数完成、token 过期)

**If neither can be written, it is a permanent assertion and needs re-examination.**
Keeping the two separate matters: writing "I gave it an expiry" is vacuous if the kind
is wrong. `kaggle is installed` expires by ①; `dividend coverage is 11%` expires by ②
(the moment the backfill lands).

**Why 2b pays for itself (@Tim's point, and it is the real reason to adopt it):** asking
"when does this become false?" surfaces the *relationship between the claim and your next
action*. His "11%" had the backfill — already scheduled for the next day — as its own
expiry. Had he asked, he would have written ② immediately, and then asked why he was
about to publish a snapshot as a stable property.

Three ways a fact expires; an instant alone catches only the first:
1. someone changes it → caught by stating the instant
2. **it expires on its own** (token, quota, certificate, a version bump removing a flag)
   → NOT caught
3. it was already stale when observed → caught only if the instant is honest

**Timezone is part of the instant.** `Oct 24 23:19` local (+08:00) = `15:19Z`; read as
UTC it would imply an installation six hours in the future.

## The decision-chain extension (@Orion, stronger than 2b alone)
> **A decision must carry the assertion it depends on, so that when the assertion
> expires the decision can be found.**

Stating an instant only makes an error *findable*; recording the chain makes it *come
looking for you*. This is the difference between auditable and self-alarming.
Cindy's "don't build the Kaggle line" lived two days precisely because it had become a
**decision premise** — and a decision, once made, stops being discussed.

## Error lifetime ∝ how little it looks like a claim
- a spoken mechanism ("uniform dilution") → reviewed immediately, dies in minutes
- a judgement written into a file ("dividend coverage 11%") → found when someone cites it
- **a decision premise ("don't build that line") → nobody re-reads it, because it no
  longer needs discussing.** Longest-lived, therefore the one that needs expiry most.

**Corollary worth keeping:** *longest-lived* and *most harmful* are different orderings.
Harm ≈ lifetime × cost-if-cited × probability-of-citation, but **do not put weights on
it** — invented weights look computed, which is the exact failure mode catalogued above.

## My own instance, and why I stopped hand-fixing things
The `.agent-scratch/` report was rule-2 compliant (instant + probe list) and still left
expiry unstated: `git checkout` would flip the fact and I left that to @prime without
marking the fact as pending change.

And the deeper one: @prime said *dispatch Pi for coding, don't be a foot soldier*. I
violated that most completely by hand-fixing the offset regex in `sass_lint.py` myself —
reasoning "it's a one-line regex, seconds of work". **That regex was precisely where the
bug was.** Dispatched to Pi it would have run the tests and watched the controls go red;
done by hand it only printed output I had to eyeball. Bug 4 survived as long as it did
partly because I took the path with no second verifier.

> **"I'll just do this small thing myself" is an unreliable judgement, because the thing
> I'm judging is the part I understand least.**

**@Orion's sharper form, adopted:** *"I'm familiar with this" is not a reason to do it
yourself — it is the signal to hand it over.* The part you are familiar with you will
indeed do well; errors happen in the cell you did not know existed. He broke @prime's
directive worse than I did: authorised to delete 4 directories / 1.1 GB, he pr'd it and
ran `rm -rf` in under two minutes — **and did it correctly**. His point stands anyway:
the right move was not "do this quickly myself" but **"this is not a job that should
take two minutes"**, because an irreversible delete has no second verifier.

**"A glance tells me everything" is the self-persuasion format shared by every error
catalogued in this file.**


---

## Rule 9 — commit the output, or it never existed

**Added after auditing this repo 2026-10-24 and finding the whole day's work untracked.**

Everything produced that day — 7 fail-able negative controls, two independent SASS
implementations, 9 reporting rules — sat outside version control. One `git clean -fdx`
from gone. And worse: `README.md` claimed factor_lint's controls "correctly stay
silent" on the strength of **one manual check, never automated** — the same shape as
the four silent failures `INTEGRITY.md` documents: *a claim that looked verified*.

> **A correct method whose output vanishes fails exactly like a wrong method — silently.
> The only difference is that nothing is wrong until it is gone.**

## Rule 9a — the commit message carries the WHY, not an inventory
My first commit was a file list ("Six tools and docs that were untracked...").
Format correct, WHAT complete, **WHY absent** — Tim's counterexample, committed by me
minutes after he described it.

Self-check is not *"did I write a WHY?"* but:
> **Six months from now, someone runs `git log` and reads this line — can they tell why
> this had to exist?**

## Rule 9b — why some rationale is never recoverable from results
| kind of rationale | what happens if unrecorded |
|---|---|
| efficiency ("this is 3x faster") | **recoverable** — the result shows it |
| correctness ("this prevents X") | **partly recoverable** — X recurs and you rediscover it |
| **provenance / authorisation ("@prime said delete, verified no source")** | **never** — nothing in the output encodes who allowed it or what was checked |

The third kind leaves no trace in any result, so it must be written at the time or lost.
That is why the `.agent-scratch/` deletion needed its authorisation recorded — the
verification (@Orion checked no-source / no-git / zero-refs) is invisible in the outcome.


---

## Rule 10 — a check that cannot go red is not a check

**Earned** building `a-share-lab/scripts/verify_gate.py` after @Tim worried his own
`-m "not slow"` might silently drop tests.

*"A permanently-red gate equals no gate"* is the safe half. The dangerous half is the
converse: **a canary that never goes red is indistinguishable from a gate that has
none — both print OK.**

Three one-character mutations leave a gate green forever, output shape unchanged:
`-m "not slow"` → `-m "slow"`; `if ! "$@"` → `if "$@"`; the pytest line → `true`.

## The method that works
Do **not** write a copy of the gate command — **extract it from the real script**, then
point it at a canary that always fails. A copy tests your restatement; extraction tests
the gate's actual body, so "`run()` swallows failures" is caught too.

| break | expected | measured |
|---|---|---|
| `if ! "$@"` → `if true` | red | exit 1, `GATE_WOULD_FAIL` |
| `collect_ignore = []` | error | exit 1, names the conftest |
| nothing | green | exit 0 |

## 10a — "I read it and didn't trip" is not "it is untested"
I read a `collect_ignore`, found it unremarkable, wrote *"so it hasn't been verified,
only read"* — then broke it and it went red. It had been guarded all along, and the
docstring **said so in words I had just read**. I turned a reading into a "gap" claim
*after* seeing the evidence. **The trip is the test.**

## 10b — where the recursion stops
A verifier is itself a check, so it too can be silently broken. I broke it by hand but
did not automate a probe for the probe: infinite regress is not the goal. The honest
label is **"manually tripped at this layer, not automated"** — name which, every time.

## 10c — measure the exit code, not the pipe's tail
`cmd | tail; echo $?` reports **`tail`'s** status, not `cmd`'s. I did exactly this and
read "OK" off a broken gate. The same shape as the `{4}` regex that dropped 77% of
instructions: **I measured the end of the pipe instead of the thing I meant to measure.**
Capture the code directly (`> log; echo $?`) or use `subprocess.run(...).returncode`.


---

## Rule 11 — in someone else's repo, leave your files untracked

`git add` followed by a **failed** `commit` leaves the index staged. The person who did
it believes *"I didn't commit, so nothing changed."* But `git status` shows `A`, and
**anyone else's `git commit -a` sweeps those files into their commit.**

The state is invisible to whoever caused it, and visible only to whoever it happens to.

**Rule:** hand files over as `??`. The index belongs to the repo's owner.

Three of us edited one repo today and nothing was lost — because one commit happened to
fail. **The absence of a collision was not evidence of coordination.**

## 11a — an unverified item needs its failure mode, not just the label
"未复核 this parser's boundary" is honest but unactionable: a reader cannot tell whether
to panic. The next line must be **"if this is wrong, it breaks *like this*"**.

When tripped, the answer was the opposite of the obvious guess — the parser *raises* on
structural change rather than guessing a partial command. Labelled risk ("may parse
wrong") and actual risk ("refuses to parse") need different remedies.

## 11b — a checklist expands from other people's marked blanks
My checks are the ones I thought to list. I dispatched the implementation and verified
exit codes; **"text parsing" was never on the list.** That is not a diligence failure but
an *unlisted dimension*, and "be more careful" does not fix it.
**"Adopt others' unverified flags into your checklist" does.** An explicit "unverified X"
line is not a disclaimer — it is a map of where to aim.

---

## Rule 12 — "can't be done" is a claim, and it needs evidence like any other

**Earned 2026-10-24.** I wrote that author provenance "has no machine-readable form, so it
can't be checked" — without looking. @Orion pointed out the form is a commit-message
trailer, which is trivially greppable. **He was right that the form can exist; I was wrong
to call it impossible.**

Then I measured, and his fix failed differently:

```
git log -20 --format='%b' (a-share-lab)  → 0 commits with an Author: trailer
git log -5  --format='%b' (my own)       → 0 of 5
```

**Both claims were wrong, in the same way:** we each treated *something agreed in
conversation* as *something in effect in the system*. The template existed in the
discussion and in zero commits.

## Why the check cannot be built yet
A grep returns **nothing** for every commit today, so it fails on **100%** of them. That is
not "degrades to manual" — it is "goes permanently red, then gets switched off", which is
the same as no gate (rule 10).

**Order matters: adopt the convention first, automate it second**, with a measurable trigger:

```bash
git log -10 --format='%b' | grep -ciE '^author\s*:'   # now 0; build the check at 8
```

**"No trailer" only becomes a signal once it is the exception.** Before that it is the
baseline.

## 12a — the restatement that actually generalises
> **"Can't be done" needs evidence exactly like "done" does — and it is more dangerous,
> because nobody thinks to verify a denial.**

I had made this identical mistake on `verify_gate.py` earlier the same day: read the
docstring, judged, did not run. The only difference was that someone else checked for me.


---

## Rule 13 — delivering to someone's repo means not putting files in their working tree

**Earned 2026-10-24, twice in one day.** Rule 11 said "leave your files untracked
(`??`)". I followed it and @Tim's gate still stalled, because **an untracked file is
still a file in someone else's working directory**: it shows up in their `git status`,
it can be swept into `git commit -a`, and it made them stop and ask instead of running
the gate.

**`??` is not neutral.** It is visible, it is in the way, and it transfers a decision
they did not ask for.

**@Orion's correction, adopted — write the rule as a POSITION, not a STATE:**
> "Deliverables do not land in the recipient's working directory" — not "keep my files
> `??`". **`??` describes a state; a deliverable has a place. A state must be remembered
> and re-checked; a place is simply obeyed.**

That distinction is what made the first version fail in practice: I *was* keeping files
`??`, correctly, and it still stalled the gate.

## 13a — grep by name will confuse two things that share a word
Checking the removal, I grepped `canary` and got 9 hits — and my first read was "my
files leaked into his suite". They were **his** `test_pit_leakage_canary.py`, a
point-in-time leak detector. `canary` is a generic testing word.

> **A name collision nearly made me report someone else's work as my own leak.**
> Before concluding a grep matched what I meant, match the **full path**, not the concept.

That failure would have been a false accusation produced by a correct command.

---

## 13b — residue exposes itself; design does not
- **First: incidental residue.** `git add` + a failed `commit` left the index staged. The
  tool did it; I found it and cleared it.
- **Second: design.** I treated "generate in my repo, use in theirs" as a normal workflow.

**Residue exposes itself — it gets in the way. Design does not: it works perfectly right
up until someone tries to commit.** That asymmetry is why the first one taught me nothing.


## What delivery actually looks like
- a **patch** (`git format-patch -1 <sha> --stdout`)
- a file in **my own** repo, with a README saying how to install it
- a **link**, for anything remote

**Never** a new path sitting in their tree, tracked or not.

## Rule 14 — a matched feature is not an identity; name the second candidate

**Earned 2026-10-24 from three failures that @Orion grouped as "one feature satisfied by
two things, so it stops discriminating":**

| what I measured | what I concluded | the other thing it could have been |
|---|---|---|
| `HMMA = 0` | "this is a scalar path" | **a truncated dump** (no FP instructions at all) |
| an `exclude` is visible in `.gitignore` | "this is excluded" | **visible ≠ read** |
| `grep canary` → 9 hits *(as measured 2026-10-24 18:0xZ)* | "my files leaked into his suite" | **his own `test_pit_leakage_canary.py`** |
| `grep -rln canary` → 5 names *(same instant)* | "2 files contain it" | **3 of them were `.pyc` caches, and a *third* source file had it too** |

**Those two figures no longer reproduce** — re-run 2026-10-24 19:0xZ on the same repo:
**18 hits and 2 names**, because @Tim added tests in between. That is not an error in the
rule; it is the rule working. **A count of a changing object is true at an instant, so the
instant belongs beside it** (rule 2b: a world-claim needs an expiry, not just a time).

**The shared defect is not "features collide" — it is that I treated a feature as an
identity.**

## The check, which fits in one line
> **Every conclusion from a grep or a count must first answer: what *else* would produce
> this same signal?**

- Answer **none** → you have not looked. Keep looking.
- Answer **something, and did not rule it out** → **not a conclusion yet.**
- Answer **something, and ruled it out** → now it is evidence.

Applying it is cheap: for the `canary` case the answer is "someone else's test named
canary" and it takes one `grep` on the full path. For `HMMA = 0`, the answer is "truncated
file", which the existing integrity gate already checks.

**Notice what this rule does NOT say: "be more careful."** That is not executable. The
output required is **a specific alternative explanation**, and producing none means the
work is unfinished.

## 14b — make the tool print the candidates; don't rely on thinking of them
**@Orion's correction, adopted:** "name a second candidate" still requires me to *think of
something I did not think of* — the exact gap this rule exists to close. So the check is
two commands, and the alternatives are **printed, not recalled**:

```bash
{ git ls-files; git ls-files --others --exclude-standard; } \
  | xargs grep -ln "PATTERN" | sort | uniq -c
```
**Criterion: any file name you cannot explain in one sentence → you counted wrong.**

### Why the start point, not an exclusion list (@Orion's refinement)
My first version was `grep -rln ... | grep -v '\.pyc'` — a **maintained exclusion list**.
Measured on my own repo, same question (`canary`):

| version | files reported | of which noise |
|---|---|---|
| `grep -rln` + exclude `.pyc` | **10** | `.git/COMMIT_EDITMSG`, `.git/logs/HEAD`, `.git/logs/refs/heads/main`, `pip/_vendor/rich/_emoji_codes.py` |
| `git ls-files` start point | **5** | none |

**My version reported "canary appears in pip's emoji table."** True, and useless — worse,
it looks like a finding.

> **Any exclusion list I maintain by hand has an expiry time I do not know about.**
> `.git/`, `.venv/`, `.pytest_cache/` were never in mine. Adding `.pyc` did not fix the
> class of bug — it shrank the hole, exactly as `{4}` → `{4,8}` shrank the regex hole
> without changing the approach. **Ask "where do I start counting", not "what do I skip".**

### What it caught that I could not have thought of
On the original question it surfaced `test_quality_value_backtest.py`, which I had never
mentioned. My assertion "9 hits, it is his `test_pit_leakage_canary.py`" was **true but
incomplete, and no amount of thinking harder would have found the third file.**

> **Any check whose usefulness depends on my remembering to vary it is not a check.**
> Prefer the form where the tool enumerates and I subtract.

## 14a — "nothing went wrong" is not evidence that the procedure was right
@Orion, on today's collision that did no damage: *the zero loss was not because we chose
correctly, but because we never met the moment that would have made us regret it.*

Today I wrote rules 11, 11a, 11b and 13 in the aftermath of things that **did not break**.
Had @Orion's commit succeeded, had my canary been named differently, had @Tim run
`git commit -a`, the same day would have produced different rules — or none.

> **Rules derived from a near-miss carry the near-miss as their evidence.** Say so when
> citing them, so a reader knows the rule has never been tested against the bad case.


---

## Rule 15 — a summary number needs a re-runnable command beside it

**Earned 2026-10-24** writing `tools/RULE-AUDIT.md`: I summarised its own table as
"P: 2, R: 16" when the table contained **zero P rows** — my regex never emitted one.

> **A summary number with no command beside it is a claim, not a count.**

**Self-correction on this very rule:** the first version of that line read `0 / 25`. That
`25` was not the number of commits — **the repo had 7 commits at the time.** It was a
number I produced while explaining that I produce numbers without checking. The figure is
now `0 of 5`, which the command above reproduces; the `25` had no command behind it, which
is precisely the failure the rule describes.

Enforced: `tools/check_audit.sh` recounts P/R/N, counts rows by a **different route**
(`^\| [0-9]`), and requires `sum == rows == rules_in_file`. **Verified it can fail** —
removing one label gives `sum=27` vs `rows=28` → MISMATCH, exit 1.

## 15a — undercounting is worse than miscounting
@Orion: a format error that eats a row does not produce **a wrong number — it produces a
missing one**, and a missing row never reads as wrong in a total.

> **A miscount collides with the total. A missing count makes the total look right.**

My `2 + 17 + 9 = 28` happened to equal the row count, so I could not have noticed a row
was uncounted. **And I was the one who broke that row** — a `\|` escape in the 10c line
silently truncated it. **I created the undercount and was then deceived by it.**

## Rule 16 — a candidate-printing check catches surprises, not habits

**@Orion's sharpest point of the day, adopted.** `uniq -c` can catch a name I do not
recognise. It cannot catch **a name I recognise and recognise wrongly** — and every error
that day was the second kind: `HMMA = 0`, `canary`, `{4}`, and "P: 2, R: 16".

> **A familiar name must be supported by an instance, or be treated as unfamiliar.**

**"Explainable" ≠ "evidenced":**
| | grammar | example |
|---|---|---|
| explanation | *seems like* | "it is another of Tim's canary tests" |
| instance | *is* | "at this path, authored by Tim, testing point-in-time leakage" |

## 16a — two different questions have no winner
I claimed my `grep -rl` (10 files) "beat" @Orion's `git ls-files` (5) on the same pattern.
Measured, they answer **different questions**:

| command | boundary | question |
|---|---|---|
| `grep -rl` | **disk** | did the string leak into any artifact? |
| `git ls-files` | **git ownership** | which files are mine to account for? |

For "did `canary` leak into a build artifact", **the `.pyc` files are the answer, not
noise** — and `git ls-files` structurally cannot see `.venv`. **Declaring a winner was
itself a rule-14 error: two different signals treated as one signal done better.**


---

## Rule 17 — a fallback branch must tell "confirmed empty" from "could not read"

**Earned 2026-10-24 on @Tim's `gate.sh`** (@Orion spotted it, I verified the branch is live,
Tim fixed it, I verified the fix in both directions).

```bash
pytest -m "slow" --collect-only | grep -E "^tests/" || echo "（slow 层为空）"
```
A wrong marker makes pytest exit **5** ("no tests collected"). `grep` matches nothing, so
`||` fires, and the gate prints **"slow layer is empty"** — **a collection failure reported
as a normal state**, on the same line a genuinely-empty layer would use.

> **Every `||`, every `catch`, every `or default` translates a failure into a normal state.
> Each such branch must be able to say which one it is looking at.**

The fix that works is a three-way branch, not two:
```bash
if ! out=$(collect ...); then        # could not read  -> ERROR, and fail the gate
elif names=$(printf '%s' "$out" | grep -E "^tests/"); then   # read, found some -> name them
else                                  # read, genuinely none -> say "confirmed, 0"
fi
```

## 17a — the fix leaves its sibling bug standing
After the fix, "genuinely empty" and **"marker name misspelled"** look identical: a `-m
"slow-test"` collects fine, matches 0, and prints *"confirmed: 0 slow tests"* — while one
exists. **Same shape as the original, one level along.** The prompt fix costs nothing:
the marker name in `pyproject.toml` and the literal `-m "slow"` in `gate.sh` should come
from **one source**, not two places that must agree by memory.

**A fix that removes one "cannot distinguish" leaves the next one visible only if you ask
what the fixed branch can no longer tell apart.**

### 17b — ask the tool, do not parse the file
The obvious way to unify the two marker sources is to grep `pyproject.toml`:
`MARKER=$(grep -oP '^\s*"slow' pyproject.toml | ...)`. **Refuse it.** That couples to
file layout — quotes, indentation, line position — and **layout is not a contract; whoever
reformats the file does not know a parser is watching.** This is the same defect as the
`{4}` offset regex and the `.pyc` exclusion list: a parse of someone else's *formatting*.

`pytest --markers` is a public output. Ask it:

```bash
printf '%s\n' "$markers" | grep -qE "^@pytest\.mark\.${MARKER}:" \
  || { echo "marker '${MARKER}' undeclared — -m would silently collect 0"; fail=1; }
```
**Verified both ways** (`tools/a-share-gate/marker-guard.sh`): declared → exit 0; renaming
`slow:` → `slowtest:` in `pyproject.toml` → exit 1 naming the consequence.
`pyproject.toml` restored byte-identical afterwards.

> **Ask the authority whether a name exists; never re-derive it from a file's appearance.**


---

## Rule 18 — a correct output that never changes is attentionally equal to no output

**Earned 2026-10-24** on `verify_gate.py`. After rule 17 was fixed, the verifier printed a
bare `gate has teeth: OK` on every run. My proposed defence was to print *what* it checks
— **but that line would also never change.** @Orion supplied the sharper form:

> **A correct output that never changes and no output at all are equivalent in attention.
> Only a correct output that changes can hold attention over time.**

Now it prints fingerprints of the two things being checked:

```
gate has teeth: OK  (gate.sh 3fb3b211 · canary 6e7d6f62)
```

**Verified that it moves:** appending one comment to `gate.sh` produced `90640ea9` while
the canary hash stayed `6e7d6f62`.

The reading rule is then decidable without trusting my memory:
- **both hashes match last run** → genuinely nothing changed; ignoring it is *justified*
- **either hash moved** → it is news

## 18a — this is the only rule today about keeping correct things visible
Every other rule here governs **how a wrong thing gets found**. This one governs **how a
right thing avoids becoming furniture** — which is what happened to `HMMA = 0` (a check
that passed forever on truncated input) and to the `README`'s "controls correctly stay
silent" (true, unchecked, and therefore inert).

**A green that nobody reads is indistinguishable from a green that would have been red.**

## 18c — "changing" is only information if the change is correlated with what matters
@Orion's rule needs one boundary it did not state: if `gate.sh` changed every day (someone
reformatting daily), the hash would move daily and the signal would degrade again — **"it
is different every time, so don't bother looking."**

> **The output must change *because the thing being checked changed* — not in order to
> change.**

`gate.sh`'s hash satisfies this (it moves only when the gate's logic moves). A timestamp
would break it. **Any ornament whose variation is uncorrelated with the state under
inspection reproduces the original problem in a new costume.**

## 18b — both bugs in this change were caught by crashing
`GATE_SH` did not exist (the variable is `GATE`); `_fp(CANARY_PATH)` passed a `str` where a
`Path` was required. **Both raised immediately.** Had they been written defensively —
`getattr(module, "GATE_SH", GATE)` or an `except: return "00000000"` — the output would
have looked correct while measuring nothing. **Crashing loudly was the better outcome, and
it only happened because the code did not guard against its own mistakes.**


---

## Rule 19 — evidence I manufactured only proves the check can handle me

**Earned 2026-10-24 (Orion):** I defended assertion B with "it caught the bug I injected."
**The bug was chosen by the same person who wrote the assertion.** So "it caught it" partly
records *"I picked a bug it could catch."*

> **Evidence that a check catches errors, where I also authored the errors, proves the
> check handles me — not that it handles what it has not seen.**

Tested anyway with a slip I did **not** design for (skip opcodes ending in `.64` — a plain
bookkeeping error): **B caught it**, STL 1651 → 701, LDL 2684 → 2476. So its reach exceeds
my imagination *on this corpus*.

**The limit still holds and is now printed on every run**, not buried in a docstring:
```
NEITHER catches a bug that is invisible on these two kernels.
```
A defect that only triggers on lines neither kernel contains passes B silently. **B is a
lower bound on agreement, not a proof of correctness.**

**The only way past this is for someone else to inject the bug.** I did not, because I am
the only one writing code in that repo tonight. That is a real limit, stated rather than
dressed up as modesty.

## 19a — "weaker than B" and "answers a different question" are not the same
I labelled assertion C "necessary but weak" — which implied it lost to B on one scale.
**C answers a different question:** rule 4 is about *shared assumptions*; C measures
*shared file paths*. Not weaker — **orthogonal.**

**C is class N** in the audit: no event supports it having stopped anything. Whether it
should exist depends on someone having actually been burned by two implementations reading
the same file. **I do not know that such a case occurred.**
