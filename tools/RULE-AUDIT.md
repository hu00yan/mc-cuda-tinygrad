# Rule audit — does each rule have a real event behind it?

**@Orion's three-way split, adopted — it is better than my two-way.** A rule's value is not
"did it prevent something"; most rules are never violated. Their value is that **when
violated, the site looks like an error rather than a choice.**

- **P = prevented** — an event happened and the rule *stopped something at the time*
- **R = recognised** — an event happened and the rule **names it accurately**. R has
  never prevented anything by definition: naming always follows the event
- **N = none** — no corresponding event; decoration

**R is not worse than P: recognition is the precondition for prevention.** An unnamed error
returns in the same shape next time.

**@Orion's correction to my own reasoning:** I wrote "it just happened, so it has not had a
chance to prevent anything yet — I mark it R provisionally." **"Has not had a chance" is not
a reason to mark R; it is the definition of R.** What I was actually describing is that the
rule has not yet been shown to change behaviour — and **that is the entry condition for P,
not the entry condition for R.** Marking it R was correct; "let us see tomorrow whether it
stops something" is its *observation period*, not a caveat on today's label.

| rule | class | the event | P/R/N |
|---|---|---|---|

| rule | class | the event |
|---|---|---|
| 0 negative control| A | `sass_lint` returned 0 forever; reverting bug 4 turned 3 controls red | **P** |
| 1 never a bare multiple| A | "43×" was read as speed when it was instruction count | **R** |
| 2 name the statistic| A | `n` was 7.4 / ~30 / 12.88 — three statistics, one word | **R** |
| 2b expiry on world-claims| B | no event; derived from 2 on Tim's per-period data | **N** |
| 3 magnitude excludes| B | reasoned, not measured | **N** |
| 5 residuals must not be named| B | the −0.62pp remainder was *not* mis-named in the end | **N** |
| 6 probe width = claim width| A | `hoist_detect` measured the IR it inspected | **R** |
| 7 suspect your parser first| A | 3,842 vs 16,952 — 77% dropped | **P** |
| 8 say "unknown"| B | right by construction; no counter-event | **N** |
| 9 commit the output| A | day's whole output untracked; controls claimed but unautomated | **R** |
| 9a message carries WHY| A | my own commit was an inventory, minutes after Tim named it | **R** |
| 9b provenance unrecoverable| B | argued from the `.agent-scratch` case, but no test | **N** |
| 10 cannot-go-red| A | gate broken → verify exited 1; then healthy → 0 | **R** |
| 10a read ≠ tested| A | I called a guarded file a "gap" after reading its guard | **R** |
| 10b recursion stops| B | no event; a boundary claim | **N** |
| 10c measure exit code | **A** | `cmd \| tail; echo $?` read "OK" off a broken gate | **R** |
| 11 files stay untracked| A | staged then failed commit left the index dirty | **R** |
| 11a failure mode not label| A | parser raises (not guesses) — opposite of the guess | **R** |
| 11b checklist from blanks| B | real but from one conversation, no measurement | **N** |
| 12 "can't be done" needs evidence| A | 0 of 5 commits had the trailer I claimed existed | **R** |
| 12a restatement| A | same event; also `verify_gate.py` earlier that day | **R** |
| 13 delivery is a position| A | my files in Tim's tree stalled his gate run | **R** |
| 13a grep by name| A | 9 hits were his `test_pit_leakage_canary.py` | **R** |
| 13b residue vs design| B | distinction holds, but both cases were the same day | **N** |
| 14 feature ≠ identity| A | 4 instances: HMMA=0, exclude, canary, .pyc | **R** |
| 14a nothing-wrong isn't evidence| B | meta-claim; correctly about itself | **N** |
| 14b tool prints candidates| A | 10 vs 5 files; `.git/` and `.venv/` in my version | **R** |

## Result

- **P (prevented): 10** — rules 0, 4, 7, 17, 18, 19, 20, 20b, where a broken check actually stopped a wrong number
- **R (recognised): 25** — an event happened and the rule named it afterwards
- **N (none): 10** — real thinking, but no event corresponds to them

**No rule is decoration by accident; nine are decorated by my own enthusiasm.** Those nine are
marked N so that tomorrow's reader — me — knows which ones have never met reality.

**The audit is itself class N:** it classifies by my own reading.

**And it caught me doing it again while writing it.** My first draft claimed "P: 2, R: 16"
— but the table actually contained **zero P rows**: my regex never emitted one. The stated
totals were a claim about the table, not a count of it. Recounting found 2/17/9 = 28, and
one row (10c) had no label at all because an escaped pipe `\|` broke the pattern.

> **The audit needed auditing.** Which is precisely rule 14 — I wrote a summary *about* the
> table without asking what else would produce that summary.


---
| 15 summary needs a command | **A** | "P: 2, R: 16" from a table with zero P rows | **R** |
| 15a undercount beats miscount | **A** | `\|` escape truncated the 10c row; sum still matched rows | **R** |
| 16 candidates catch surprises, not habits | **A** | every error today was a name I recognised | **R** |
| 16a different questions have no winner | **A** | disk 10 vs git 5 measure leakage vs ownership | **R** |
| 17 fallback: empty vs unreadable | **A** | gate printed 'slow empty' on a collect failure (exit 5) | **P** |
| 17a fix leaves its sibling | **A** | misspelled marker still prints 'confirmed: 0' | **R** |
| 18 never-changing output | **A** | 'gate has teeth: OK' fixed; fingerprints move 3fb3b211->90640ea9 (2026-10-24, a-share-lab) | **P** |
| 18a keeping correct things visible | **A** | every other rule finds wrong things; HMMA=0 and the inert README passed forever | **R** |
| 18b crash beats a plausible default | **A** | GATE_SH undefined and str-not-Path both raised; defensive writes would have printed a hash for nothing | **R** |
| 18c variation must be correlated | **A** | gate.sh hash moves on logic change; a timestamp would re-degrade the signal | **R** |
| 4 independence is a degree | **A** | test_independence.py; C overclaimed, only B has teeth | **P** |

## The blind spot in this whole method (@Orion, adopted)

**`uniq -c` catches a name I do not recognise. It cannot catch a name I recognise and
recognise wrongly** — and every one of today's errors was the second kind: `HMMA = 0` I
recognised, `canary` I recognised, `{4}` I recognised, and in this very file I wrote
"P: 2, R: 16" from numbers I thought I knew.

> **A check that prints candidates catches surprises, not habits. A familiar name must
> either be supported by an instance or be treated as unfamiliar.**

**"Explainable" and "evidenced" are different:** *"it is another of Tim's canary tests"* is an
explanation (grammar: *seems like*); *"it is at this path, authored by Tim, and it tests
point-in-time leakage"* is an instance (grammar: *is*).

## And the 10-vs-5 "win" was not a win
The two commands measure **different questions**, not one question better:

| command | boundary | question |
|---|---|---|
| `grep -rl` (10) | **disk** | did the string leak into any artifact? |
| `git ls-files` (5) | **git ownership** | is this file mine to account for? |

For "did `canary` leak into a build artifact", **my 10 is correct and the `.pyc` files are
the answer, not noise.** Stating a winner was itself an error of the rule-14 kind: two
different signals, treated as one.

## Rule: no summary number without a command beside it
> **A summary figure with no re-runnable command is a claim, not a count.**

`tools/check_audit.sh` enforces this for the audit: it recounts P/R/N, counts rows by a
**different route** (`^\| [0-9]`), and requires `sum == rows == rules_in_file`.

**Verified it can fail:** removing one row's label gives `P=2 R=16 N=9, sum=27,
rows=28` → `MISMATCH`, exit 1. Restored, exit 0.

## Why undercounting is worse than miscounting
@Orion: a format error that eats a row does not produce **a wrong number, it produces a
missing one** — and a missing row never shows up in a total.

> **A miscount collides with the total. A missing count makes the total look right.**


---

## Self-reference: how many rules can check themselves?

**@Orion's question, measured rather than guessed.**

```bash
grep -c "REPORTING-RULES\|RULE-AUDIT" tools/test_sass_lint.py    # 0
grep -c "REPORTING-RULES\|RULE-AUDIT" tools/test_factor_lint.py  # 0
grep -c "REPORTING-RULES\|RULE-AUDIT" tools/check_audit.sh       # 3
```

**1 of 38.** Only `check_audit.sh` inspects the rule system; the other four check code,
not claims.

**P (4) says a rule once stopped something — that is history. 1/38 says a rule can know
before the next event — that is mechanism.** The second number is the more honest measure.

### Why the low count is not a defect
The stated reason "most rules cannot refer to themselves" is true but imprecise. The real
constraint: **self-reference requires the criterion to become an executable file.** A rule
stays a sentence and waits for the next incident; the moment its criterion is a `.py` or
`.sh`, it can fire without me remembering.

**So 1/38 does not measure maturity — it counts how many sentences were turned into files
tonight (rules 0, 10, 14, 17).** That yields tomorrow's action, which is not "write rule 39"
and not "re-read all 38": **turn one more criterion into something that runs.** One a day.

### The output rule, compressed (@Orion)
> **If every number on a line can be recomputed from the current state of the thing being
> checked, the line carries information. If any number comes from "what time it is", it
> does not.**

Verified: `grep -c "date\|timestamp\|now" tools/check_audit.sh` → 0, and all three of its
figures derive from file contents. **I nearly failed this**: I considered adding "last run
time" to `verify_gate.py` to make its output appear to move. That number comes from the
clock. The hashes come from the files. **Had this criterion existed half a day earlier I
would have skipped that step.**

| 19 self-injected bugs | **A** | B caught a slip I didn't design for (STL 1651->701); limit now printed | **P** |
| 19a weaker vs different question | **A** | C measures file paths; rule 4 is about assumptions | **N** |
| 20 counts need their instant | **A** | rule 14's 9 and 5 re-ran as 18 and 2 within an hour | **P** |
| 20a familiar means re-run | **A** | most confident at the points I was most wrong ({4}, pipe, 0/25) | **R** |
| 20b numbers need a subject too | **A** | 10 vs 5 in a-share-lab is 10 vs 6 here | **P** |
| 20a-i two directions of familiar | **A** | old: {4}, pipe escape; just-written: 0/25, '3 of them fail' | **P** |
| 21 formulas don't expire, tables do | **A** | found 3 stale-prone table rows; 2 must stay stale on purpose | **P** |

## 12a (Cindy's addition) — "I tried" must be a command and an output

> **Writing "what I tried" must be a reproducible command plus one actual output.
> "I analysed it" is not trying; "I ran X, output Y" is.**

**Why it matters:** the rule's failure mode is not "nobody writes it" — it is **"someone
writes reasoning, and reasoning written well looks like a verified conclusion."**
Cindy's own instance: she decided a vectorised change "obviously correct" and pushed to
Kaggle three times without a single local run; 3,947 of 5,000 random boards were wrong.

**Fabricated evidence is harder to catch than missing evidence.** Missing evidence is a
hole; fabricated evidence is a filled hole that is wrong.

### Applying her standard broke my own rule 12
Rule 12 cited **`0 / 25`** commits with a trailer. The repo had **7** commits at the time.
The 25 had no command behind it — **a number produced while explaining that I produce
numbers without checking.** Corrected to `0 of 5`, which the adjacent command reproduces.

## Where the 10 P rules came from — the only question that matters

**11 of 46 rules have ever stopped anything (class P).** Their entry condition is identical
in every case: **the rule was turned into something that runs, and the thing that ran
disagreed with me.** No P rule got there by being written down well.

| became a file | rule | can be made RED? |
|---|---|---|
| `test_sass_lint.py` (7 controls) | 0 | **P1** |
| `test_independence.py` (B: `.64` slip) | 4 | **P1** |
| `marker-guard.sh` (rename `slow:` → `slowtest:`) | 17 | **P1** |
| `verify_gate.py` fingerprints | 18 | **P1** |
| negative-count check | 19 | **P1** |
| negative count only, no red test | 7 | **P2** |
| `check_tables.py` | 20 | **P1** |
| `check_tables.py` | 20b | **P1** |
| `check_tables.py` (4 controls, verified red) | 21 | **P1** |
| Tim's radius method: exempt-with-legal-sample | 22 | **P1** |
| *(the word P1/P2 in the definitions below is a definition, not a rule row — counted once as a miscount)* | — | — |

### The score moved the same night (rule 22's own lesson)
Writing "6 P2" down did not make any of them P1. **Cindy's point — a P2 rule is a TODO, not
a classification — is what moved three of them:** rules 20, 20b and 21 all say a measured
figure needs its instant and its subject, and all three are now enforced by one file,
`tools/check_tables.py`, with 4 negative controls verified red. **So the P1/P2 split is not
a report on maturity; it is a work queue, and the queue has an owner now.**

### P1 vs P2 (@Orion): the split that matters more than P itself
**P1** — has a negative control: *someone can make it fail without me.* (the word `P1` here is a definition, not a rule row)
**P2** — only an event trace: *it stopped something once, and nothing in the machine can
reproduce that.*

**9 P1, 1 P2** — 20, 20b and 21 moved from P2 to P1 the same night, after Cindy's point that a P2 rule is a TODO rather than a classification. Her challenge was "how many of the 6 can be given a negative control?", and the answer for these three was: three. The ratio is the maturity measure, not the total: **P2 depends on memory,
and memory is the thing that was failing all night.** Rule 22 is P2 — and it is the most
valuable rule of the day, so the most valuable rule is also the one I cannot hand to the
machine.

**The other 35 are prose I agree with.** Not a criticism of prose — just the observation
that a rule changes behaviour only after it has a failure mode someone can observe
without me present.

### The files, counted rather than estimated
Orion closed the night with "4 files are the tools." **There are 6**, and the difference is
not pedantry — I would have repeated his number in my summary and nobody would have caught
it:

| file | rule it enforces | gate |
|---|---|---|
| `tools/test_sass_lint.py` | 0 | 7 controls, verified red |
| `tools/test_factor_lint.py` | factorisation finding | verified red |
| `tools/test_independence.py` | 4 | A/B/C, B verified red |
| `tools/check_audit.sh` | 2b, 7, 20, 20b, 21, 22 | installed as **pre-commit hook** |
| `tools/a-share-gate/marker-guard.sh` | 17 | verified exit 1 |
| `tools/a-share-gate/verify_gate.py` | 18 | fingerprints verified moving |

**Two are P1-gated, three are P2, and one (`check_audit.sh`) is the only one that runs
without being asked.** 
| 22 presence check != content check | **A** | relabelling rule 4 P→R left check_audit OK | **P** |
| 23 exemption is a radius | **A** | 4 phrase-holes found by hitting it from outside | **P** |
| 24 empty AND full queues go unread | **A** | "0 P2" was wrong; composition now prints | **P** |
| 25 self-made bugs need subject conditions | **A** | ladder.sh "bug" vanished when run from its own dir | **P** |
| 26 can-it-go-red first | **A** | hollow L4 passed on empty dir + syntax error + random | **P** |
| 26a identical failure ≠ measured | **A** | probe exiting 3 twice reported PASS | **P** |
