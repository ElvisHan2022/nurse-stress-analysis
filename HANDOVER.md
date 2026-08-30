# Handover

Updated 2026-08-30 02:45 PDT, at `de1f57b`. Read this first; it is meant to let
a new session pick up without the prior conversation.

Repository: `ElvisHan2022/nurse-stress-analysis`, branch `main`.

---

## 1. Status: the paper is submittable now

`paper/main.pdf` compiles clean (exit 0, no errors), **7 content pages against a
limit of 8**, references from page 8, checklist filled with no TODOs,
anonymisation verified on the compiled PDF rather than the source.

**Deadline: 29 Aug 11:59 PM Anywhere on Earth = 30 Aug 04:59 PDT.** Check the
clock before assuming there is time. The workshop is non-archival, so if the
deadline has passed the work travels to another venue unchanged.

Venue: NewInML workshop at NeurIPS 2026, double-blind via OpenReview, 2 to 8
pages excluding references.

### Two things nobody has done

1. **Eligibility is unverified.** The workshop requires that no author has
   published at NeurIPS, ICML or ICLR. Confirm for both authors.
2. **Nobody has read Section 7 with human eyes.** The pretraining and severity
   sections were written at 02:00 and never proofread. Tables are
   machine-verified against the CSVs; the sentences are not.

---

## 2. How to run things

Two environment facts that otherwise cost an hour.

```bash
# Python: no activated venv, call it explicitly
.venv/Scripts/python.exe tasks/<script>.py

# LaTeX: TinyTeX is installed but NOT on PATH, and shell env does not persist
export PATH="$PATH:/c/Users/Elvis/AppData/Roaming/TinyTeX/bin/windows"
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

`which pdflatex` returns nothing and is misleading. `pdfinfo` does not exist;
use `pypdf` from the venv. Do not read `main.pdf` in the same command that
compiles it, or you will read a half-written file.

---

## 3. What the paper says

Title: *You Can't Pretrain Your Way Out of a Missing Negative Class: Nurse
Stress Detection from Wearables.*

The arc: 9.1% of the data is labelled, which is the condition self-supervision
exists for. But a pretext task needs a supervised baseline to beat, and building
one needs a negative class this dataset does not contain, because every survey
row marks an episode a nurse flagged. The attrition required to construct both
is the paper. Then pretraining was run anyway and did not help.

| Result | Value |
|---|---|
| Labelled share | 9.1% of 1,251.7 h |
| Cohort after attrition | 10 participants, 138 episodes, 24,334 windows |
| EDA at onset | 64.3% of 140 episodes, sign test p = 0.0009 |
| Heart rate at onset | 48.6%, p = 0.80 |
| Falsification: movement / time | 0.513 / 0.507 against chance 0.500 |
| Baseline AUC | 0.7654, seed range 0.0095 |
| Permutation null | 0.4975 ± 0.0101, max 0.5186 |
| Logistic regression | 0.7502, within 0.0153 of boosted |
| Episode recall at 1 alarm/hour | **0.1063**, seed range 0.072 |
| Naive negatives | 0.6925 (curation *improves*, does not inflate) |
| Pretraining corpus | 1,238 h, 68k windows, probe 0.168 vs 0.768 reference |
| Label-efficiency sweep | probe − random between −0.024 and −0.003 |
| Severity | 0.4964 and 0.5417, at chance |

Two findings a reviewer will focus on. **Curation improved rather than
inflated**, against the usual warning; the mechanism is asserted, not measured.
And **severity fails**, which matters because severity needs no constructed
negatives, so it removes the most natural objection to the detection null.

---

## 4. Where things live

```
paper/
  main.tex          the submission, anonymous build, [nonatbib] only
  main.pdf          compiled, verified
  checklist.tex     filled, no TODOs
  neurips_2026.sty  the only supported style file
  _tmpl/            scaffolding and every edit script, not for submission
    main_named_backup.tex        non-anonymous version, for camera-ready
    main_before_restructure.tex  pre-restructure body

derived/            FROZEN. Later phases read, never rewrite.
  windows.parquet, events.parquet, labels_seed{0,1,2}.parquet
  features.parquet, encoder.pt
  (pretrain_corpus.npz and phase5_windows.npz are gitignored, regenerate)

reports/audit/      every measurement, one CSV per experiment
  audit_summary.md  THE SOURCE OF TRUTH, supersedes the PLAN*.md Phase 0 tables
tasks/              one script per audit section and phase, phase4-6 included
docs/               open_questions.md, explanation_log.md
CONTRIBUTING.md     working and writing conventions
```

---

## 5. Conventions

- **No `Co-Authored-By` trailers in commit messages.** History was rewritten
  twice to remove them.
- **`derived/` is frozen.** Regenerating invalidates every number computed
  against the old version, and nothing will error.
- **`audit_summary.md` beats the plan documents** where they disagree.
- **Re-derive before quoting.** Several plan figures did not survive
  recomputation: 156 events was really 153 then 138; a propensity AUC of 0.728
  was in-sample against 0.707 cross-validated.
- **Denominators**: 24,334 windows is a compute statistic. The denominators are
  138 episodes and 10 participants.
- **Report per fold**, never averaged. Fold AUC spans 0.388 to 0.885 and two
  participants detect nothing.
- **Anonymity**: `\usepackage[nonatbib]{neurips_2026}`. Adding `final` or
  `preprint` breaks the blind. Verify on the compiled PDF with `pypdf`, not the
  source.

---

## 6. Open, in priority order

1. **Proofread Section 7.** Never read by a human.
2. **Eligibility check.** Hard requirement.
3. **The guard-band mechanism is asserted, not measured.** One of the two novel
   findings rests on it. A guard-band-only ablation would settle it cheaply.
4. **Six references** will read as thin to a domain referee, and a paper arguing
   that self-supervision does not help cites no self-supervision-for-physiology
   work. Verify anything new against Crossref or arXiv first; nothing goes in
   from memory.
5. **265.2 h against 231.0 h** for eligible negatives, between `tasks/` and the
   collaborator's notebook. The negative ratio derives from this.
6. **No figures anywhere.** The label-efficiency curve is the obvious one.
7. `judgment_calls.yaml` still does not exist despite being referenced as the
   preregistration.

---

## 7. Traps already hit

- A mixed `datetime64[ms]` / `[ns]` pair compared as `int64` silently reported
  100% timezone alignment instead of 86.5%. Normalise resolution explicitly.
- The `causal_z` denominator approaches zero on flat signal and produced
  z-scores up to 1,600. Floored at 0.0048 and clipped at 10 in
  `audit_common.causal_z_safe`. Five episodes of 140 carried 63% of an apparent
  effect before the fix.
- `min_run` means **beats** in the collaborator's notebook and **seconds** in
  `tasks/`. At 0.75 s per beat these differ by a third.
- Session-level HRV availability (48%) is eight times more flattering than
  window-level (5.9%).
- A collaborator resolved a force-push by merging rather than resetting, which
  resurrected old history and duplicated seven commits. After any rewrite:
  `git fetch && git reset --hard origin/main`, never `git pull`.
- Bash heredocs mangle LaTeX backslashes. Write edit scripts with the Write
  tool into `paper/_tmpl/` and run them; do not inline them in heredocs.
- natbib raised an author-year error on every compile because the bibliography
  is a manual `thebibliography`. Solved with the `nonatbib` style option.

---

## 8. If asked to continue the science

Phases 0 through 6 are complete and all three of 4, 5 and 6 returned nulls. The
honest next steps are not more modelling:

- **Multiple-instance learning** over episode intervals is the one modelling
  idea with a real rationale left. The labels are intervals treated as uniform
  states, and a thirty-minute report is unlikely to describe thirty uniform
  minutes. MIL treats the interval as a bag containing at least one positive
  window and learns which minutes drove it.
- **Per-fold pretraining** would convert the transductive upper bounds in
  Table 4 into honest estimates, at ten times the compute. It would only lower
  them.
- Everything else on the list is data collection, not analysis, and the paper
  says so.
