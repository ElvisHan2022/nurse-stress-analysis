# Handover

Written 2026-08-30 at the end of a long session. Read this first; it is meant to
let a new session pick up without the prior conversation.

Repository: `ElvisHan2022/nurse-stress-analysis`, branch `main`, at `f04a43a`.

---

## 1. The immediate situation

A paper is **compiled, anonymised, and ready to submit** to the NewInML
workshop at NeurIPS 2026.

**The deadline is 29 Aug 2026, 11:59 PM Anywhere on Earth, which is 30 Aug at
04:59 PDT.** If that has passed, the work is not wasted: the workshop is
non-archival and the paper can go elsewhere. Check the date before assuming
urgency.

Submission is double-blind via OpenReview. Eligibility requires that no author
has published at a top ML venue (NeurIPS, ICML, ICLR). **Confirm this for both
authors before submitting.** Nobody has verified it.

Ready: `paper/main.pdf`, 12 pages, of which **8 are content** against a limit of
8. References begin on p9, the checklist follows, and neither counts toward the
limit. There is no headroom left.

---

## 2. How to run things

Two environment facts that will otherwise cost an hour.

**Python.** Use the project virtualenv explicitly. There is no activated shell.

```bash
.venv/Scripts/python.exe tasks/<script>.py
```

**LaTeX.** TinyTeX is installed but **not on PATH**. Every Bash call needs the
prefix, because shell environment does not persist between calls:

```bash
export PATH="$PATH:/c/Users/Elvis/AppData/Roaming/TinyTeX/bin/windows"
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

Two passes suffice; the bibliography is a manual `thebibliography`, so bibtex is
not needed. `which pdflatex` returns nothing, which is misleading. `pdfinfo` is
absent entirely; use `pypdf` from the venv instead.

---

## 3. What the paper says

Title: *Nine Percent Labelled: What Has to Be True Before Self-Supervision
Helps.*

The framing, which took several attempts to land: we set out to apply
self-supervised pretraining to a dataset where 90.9% of sensor time is
unlabelled, never got there, and the account of why is the result. Pretraining
needs a negative class the dataset does not contain and a baseline worth
improving on. Establishing both consumed the study.

Headline numbers, all traceable to `reports/audit/` and `derived/`:

| Result | Value |
|---|---|
| EDA responds at onset | 64.3% of 140 episodes, sign test p = 0.0009 |
| Heart rate at onset | 48.6%, p = 0.80, indistinguishable from chance |
| Baseline AUC | 0.7654, seed range 0.0095 |
| Permutation null | 0.4975 ± 0.0101, max 0.5186, observed exceeds every draw |
| Logistic regression | 0.7502, within 0.0153 of boosted |
| Episode recall at 1 alarm/hour | **0.1063**, seed range 0.072 |
| Falsification: ACC-only | 0.513 (chance 0.500) |
| Falsification: time-only | 0.507 |
| Naive negatives | 0.6925 / 0.0652, worse than curated |
| nnPU, prior 0.05 to 0.30 | 0.662 to 0.717 AUC, recall never above 0.065 |

Two findings run against expectation and are the most likely to draw reviewer
attention:

1. **Curation improved rather than inflated.** The naive-vs-curated comparison
   normally detects inflation. Here the curated construction wins by 0.073 AUC.
   The mechanism is asserted (the guard band) and **not measured**; a
   guard-band-only ablation would settle it and has not been run.
2. **A published $F_1 = 0.99$ on this dataset** is noted as difficult to
   reconcile with participant-grouped evaluation, without attributing a cause.
   Keep it that way. We cannot diagnose another group's protocol from a
   published summary.

---

## 4. Where things live

```
paper/
  main.tex            the submission. Anonymous build, no style option.
  main.pdf            compiled, verified anonymous
  checklist.tex       filled: 9 Yes, 1 No, 6 NA, no TODOs
  neurips_2026.sty    the only supported style file
  nips15submit_e.sty  NIPS 2015, supplied but NOT used, explicitly unsupported
  _tmpl/              scaffolding, all marked not-for-submission
    main_named_backup.tex   the non-anonymous version, for camera-ready

derived/              FROZEN. Later phases read these and never rewrite them.
  windows.parquet     24,334 windows, 10 participants, 1,579 positive
  events.parquet      138 episodes with sensor coverage
  labels_seed{0,1,2}.parquet

reports/audit/
  audit_summary.md    THE SOURCE OF TRUTH. Supersedes the Phase 0 tables in
                      PLAN.md, PLAN_v2.md and PLAN_v3.md.
  findings.md         per-section audit detail
  phase3_*.csv        baseline, operating points, PU, controls

tasks/                one script per audit section and phase
docs/
  open_questions.md   what blocks, what is open, what was skipped and why
  explanation_log.md  which concepts took several passes to land
CONTRIBUTING.md       working and writing conventions
```

---

## 5. Conventions that matter

**No `Co-Authored-By` trailers in commit messages.** History was rewritten twice
to remove them. Do not reintroduce them.

**`derived/` is frozen.** Regenerating it invalidates every number computed
against the old version, and nothing will error.

**`audit_summary.md` beats the plan documents** wherever they disagree, because
its numbers were measured over all 609 sessions rather than a diagnostic sample.
Several figures in the plan documents did not survive recomputation: 156 events
was really 153 and then 138; a propensity AUC of 0.728 was in-sample against
0.707 cross-validated.

**Re-derive before quoting.** Assume a number found in prose is stale.

**Denominators.** 24,334 windows is a compute statistic. The denominators are
138 episodes and 10 participants.

**Report per fold, never averaged.** Fold AUC spans 0.388 to 0.885 and two
participants detect nothing.

**Anonymity.** `paper/main.tex` uses `\usepackage{neurips_2026}` with **no
option**. Adding `[preprint]` or `[final]` breaks the double blind. Verify on
the compiled PDF, not the source, using `pypdf` to extract text and check
metadata.

---

## 6. What is open

Blocking or near-blocking:

1. **Eligibility unverified.** Has either author published at NeurIPS, ICML or
   ICLR? Nobody checked.
2. **The guard-band mechanism is asserted, not measured.** One of the two novel
   findings rests on it.
3. **A 21% discrepancy in eligible negative hours**: 265.2 measured in
   `tasks/`, 231.0 in the collaborator's notebook. The negative ratio derives
   from this number.

Open and worth doing:

4. Are the two undetectable participants explicable from measurable properties,
   or arbitrary? One is *below* chance at AUC 0.388, which is not the same as
   uninformative.
5. The reference list is six entries and a domain referee may read it as thin.
   Adding more requires verifying each against Crossref or arXiv first. Nothing
   goes in from memory.
6. `judgment_calls.yaml` still does not exist despite being referenced as the
   preregistration.

Deliberately not attempted, with reasons in `docs/open_questions.md`:
self-supervised pretraining and the label-efficiency sweep, HRV as a baseline
feature, three-class severity, multiple-instance learning.

---

## 7. If the next step is self-supervised pretraining

This is the thing the project actually wanted to do and did not reach. The gate
that stopped it tested the **label-treatment** axis, not the **representation**
axis, so it does not strictly settle the question, and the paper says so.

If attempting it:

- **The deliverable is the label-efficiency curve, not a head-to-head score.**
  Fine-tune pretrained and randomly-initialised encoders at 25, 50, 100 and 138
  episodes and plot recall against label count. Separation at low counts is a
  finding; absence of separation is also a finding. A single head-to-head number
  at 138 episodes is neither.
- **The participant-identity probe is a hard stop.** Compare against **0.768**,
  the accuracy reachable from sensible hand-crafted features, not against chance
  at 0.100. An encoder near 0.95 has learned identity.
- **Pretraining on all participants then evaluating leave-one-out is
  transductive.** Either re-run pretraining inside each fold or label the result
  an upper bound. Pick one and state it.
- Fix the mask span and training duration in advance. Selecting them by
  downstream performance leaks through the selection process.

---

## 8. Traps already hit, so they are not hit again

- A mixed `datetime64[ms]` / `[ns]` pair compared as `int64` silently reported
  100% timezone alignment instead of 86.5%. Normalise resolution explicitly.
- The `causal_z` denominator approaches zero on flat signal and produced
  z-scores up to 1,600. It is floored at 0.0048 and clipped at 10 in
  `audit_common.causal_z_safe`. Five episodes of 140 carried 63% of an apparent
  effect before the fix.
- `min_run` means **beats** in the collaborator's notebook and **seconds** in
  `tasks/`. At 0.75 s per beat these differ by a third. State the unit.
- Session-level HRV availability (48%) is eight times more flattering than
  window-level (5.9%). The window is the modelling substrate.
- A collaborator resolved a force-push by merging rather than resetting, which
  resurrected the old history and duplicated seven commits. After any rewrite:
  `git fetch && git reset --hard origin/main`, never `git pull`.
