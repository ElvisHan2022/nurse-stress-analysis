# Conventions

How we work in this repository, and how we write about it. The second half
matters more than the first — the goal of this project is a defensible paper,
and most papers in this area lose credibility in the methods section rather
than in the results.

---

## 1. Repository layout, and what is frozen

| Path | What it is | Can it be rewritten? |
|---|---|---|
| `tasks/` | Every script. One file per audit section or phase. | yes |
| `derived/` | **Frozen deliverables.** Window table, event table, label tables. | **no — see below** |
| `reports/audit/` | Measurements and tables. Regenerable from `tasks/`. | yes, by re-running |
| `figures/audit/` | Figures. Regenerable. | yes, by re-running |
| `Eric/` | The Dryad archive and the exploration notebook. | archive is read-only |
| `reference/` | A clone of the original course repo. | never edit |
| `docs/` | Written documents for people, not code. | yes |

**`derived/` is frozen.** Later phases read those files and never rewrite them.
If a Phase 1 or 2 decision changes, regenerate them deliberately, say so in the
commit message, and note what moved. Silently rewriting a frozen table
invalidates every number computed against the old one, and nothing will error.

**`reports/audit/audit_summary.md` is the source of truth.** Where it and a
plan document disagree, the summary wins, because it is the one whose numbers
were measured over all 609 sessions rather than a diagnostic sample. It
supersedes the Phase 0 tables in `PLAN.md`, `PLAN_v2.md` and `PLAN_v3.md`.

---

## 2. Numbers and provenance

**Every number in a document traces to a script and a commit.** If a figure
appears in prose and cannot be regenerated, it is a claim, not a measurement,
and it should be labelled as one.

**Re-derive before quoting.** Several numbers in the plan documents did not
survive being recomputed — 156 events was really 153 and then 138; a propensity
AUC of 0.728 was an in-sample figure against 0.707 cross-validated; a "34 events
over 60 minutes" figure counted rows the sentence excluded. Assume a quoted
number is stale until you have re-run it.

**State the denominator.** Three different counts describe this dataset and only
two of them are denominators:

- **24,334 windows** — a compute statistic. It belongs nowhere near a confidence
  interval.
- **138 events** — the honest denominator for "can we detect an episode?"
- **10 subjects** — the honest denominator for "does this work on someone new?"

**Report the range, not the point.** Anything computed on the constructed
negatives is reported across all three sampling seeds:

> Event-level recall **0.61** (range 0.58–0.64 across 3 negative-sampling seeds)

**Never average across folds.** Per-fold AUC in this project spans 0.45 to 0.89.
A mean would hide that the model fails completely on two of ten nurses. Report
per fold, and report the low-count folds separately.

**A comparison smaller than the noise floor is not a comparison.** The
seed-to-seed spread is currently ~0.01 AUC. Quote it beside every comparison so
a reader can see whether a difference is real.

---

## 3. Writing conventions

The house style is that **every decision gets explained twice** — once so a
reader outside the project understands it, once in the vocabulary a referee
expects. See `docs/` for worked examples.

### The four registers

**Plain.** What we did and why, with no jargon and no citations. If this doesn't
land with someone outside the project, we don't understand it well enough to
write the technical version yet. Analogies are allowed and encouraged.

**Worked.** The actual numbers from our data. Not an illustration, not a
hypothetical — the real thing, with the file it came from.

**Technical.** The same decision in the register a reviewer expects, so it reads
as deliberate rather than improvised. This is where the terminology and the
citations go.

**The objection.** The honest weakness, written by us. This is the strongest
part of a methods section, not a concession: *"you didn't actually achieve
uniform class balance"* is a far better sentence coming from us than from
Reviewer 2. Every non-trivial decision gets one.

### Rules that follow from that

- **Numbers, not adjectives.** "5.94% of windows" rather than "few windows."
  "Two of ten folds at or below chance" rather than "some variability."
- **Name the choice that produced the number.** When two defensible options give
  different answers, report which one produced the figure being quoted. The
  pooled mean said +9.75 and the median said +0.568 on identical data; only one
  of those sentences is publishable, and it has to say which it is.
- **Say when something is constructed rather than observed.** The negative class
  does not exist in this dataset. Every metric conditioned on it inherits that,
  and the word "constructed" should appear near the first mention.
- **Report null results plainly.** "Heart rate shows no measurable response at
  onset (48.6% of events positive, sign test p = 0.80)" is a finding. Do not
  bury it, and do not soften it into "a weaker response."
- **Hedge only when you cannot measure.** "May be" is acceptable when the data
  genuinely cannot settle a question. It is not acceptable as a substitute for
  running the check.
- **No result without its protocol.** An AUC with no split, no fold count and no
  seed range is not interpretable.

---

## 4. Vocabulary we have settled on

These words get used precisely, because reviewers will read them precisely.

| Term | Means | Does **not** mean |
|---|---|---|
| **Ablation** | Remove one component from the full model; measure the drop. | Any experiment that varies something. |
| **Falsification check** | A deliberately crippled model built from *only* a suspicious signal, which we expect to fail. | An ablation. Opposite direction. |
| **Sensitivity analysis** | Vary a parameter across a grid; see whether conclusions move. | An ablation. |
| **Available ratio** | Eligible negatives ÷ positives. What the data could supply. | What we used. |
| **Achieved ratio** | Negatives actually drawn ÷ positives. | What the data could supply. |
| **`min_run`** | **Beats** in the exploration notebook, **seconds** in `tasks/`. | Interchangeable. State the unit every time. |
| **Positive** | A window ≥50% inside a level-2 event. | Any reported event. |
| **Negative** | A window we *chose* to treat as not-stressed. | An observed absence of stress. |
| **Causal** (of a normalisation) | Uses only preceding data. | Causal inference. |

Two units that have already caused a real disagreement: **`min_run`** (beats vs
seconds) and the **survey severity 0/1/2** versus the **model label 0/1**. Say
which you mean.

---

## 5. Judgment calls

A judgment call is a fork where a competent analyst could defensibly choose
otherwise. They are not errors, and they are the thing this project is
ultimately about, so they get recorded rather than absorbed.

When you make one:

1. Record the default, the alternatives, and one sentence of reasoning.
2. Say whether it will be perturbed in the stability analysis.
3. If it changes a number already quoted somewhere, fix that quote in the same
   commit.

Known calls not yet in the registry: analysis sample rate; the HR `t0` offset;
where the session-length rule sits in the exclusion order; the coverage
tolerance for events with no sensor data; the causal-z floor; and robust versus
mean aggregation. The registry itself (`judgment_calls.yaml`) does not exist yet
and is the preregistration — it should be written and git-tagged before Phase 3
fitting begins.

---

## 6. Environment and running things

```bash
.venv/Scripts/python.exe tasks/<script>.py     # Windows
```

Python 3.11 in `.venv`. Eric's environment is managed by `pixi` and targets
`osx-arm64`; the two are not interchangeable, so a script that runs for one of
us should be checked by the other before its numbers are quoted.

Scripts are self-contained and print their own tables. They read from
`reports/audit/` and `derived/` and write back to the same places. Re-running a
script should reproduce its output exactly — if it doesn't, something
unseeded has crept in.

---

## 7. Commits

- Present tense, specific. "Resolve the HRV disagreement: min_run means beats,
  not seconds" rather than "fix HRV."
- The body carries the numbers. A commit that changes a result states the old
  and new values.
- **No co-author trailers.**
- Never force-push without telling the other person first, and never resolve a
  rewritten history by pulling — that merges the old history back in. Use
  `git fetch` then `git reset --hard origin/main`.

---

## 8. Working with the other person

Eric works primarily in `Eric/nurse_stress_exploration.ipynb`; the phase and
audit scripts live in `tasks/`. Both are authoritative for different things —
the notebook for signal-level diagnostics, the scripts for anything computed
over all 609 sessions.

When the two disagree, **reconcile with a number rather than an argument**.
Every disagreement so far has resolved into a definition: a session-level count
against a window-level one, beats against seconds, in-sample against
cross-validated. Write the reconciliation into `audit_summary.md` so it does not
have to be rediscovered.
