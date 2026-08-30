# Running log: mental notes and open questions

Kept alongside the draft. Two purposes: nothing gets silently resolved, and the
questions that took several passes to answer are evidence of what the paper must
explain rather than assume (see `explanation_log.md`).

Status key: **OPEN** needs a decision or a measurement · **NOTED** recorded,
no action required · **CLOSED** resolved, with the resolution stated.

---

## Blocking before submission

| # | Item | Status |
|---|---|---|
| 1 | `judgment_calls.yaml` does not exist. It is referenced throughout as the preregistration and is supposed to be git-tagged before fitting. Six calls surfaced during the audit that the drafted registry does not cover: analysis sample rate, HR `t0` offset handling, session-length rule position, coverage tolerance for the 33 uncovered episodes, the causal-z floor, robust-vs-mean aggregation. | **OPEN** |
| 2 | Falsification pass thresholds were qualitative when run ("well below the full model", "near chance"). They should be numeric and fixed against the seed spread before Phase 3, not after. | **OPEN** |
| 3 | `neurips_2025.sty` is not in `paper/`. The draft will not compile until it is downloaded from the conference site. | **OPEN** |
| 4 | Author list, affiliation and anonymisation status for the target venue. Currently written non-anonymous. | **OPEN** |

## Questions the draft raises and does not answer

| # | Question | Why it matters |
|---|---|---|
| 5 | Is the label-noise mechanism for the naive comparator actually the guard band? We infer it from the guard band being the largest structural difference, but we have not ablated the guard band alone. | The "curation improves rather than inflates" claim is one of two novel findings. Its mechanism is currently asserted, not measured. A guard-band-only ablation would settle it and is cheap. |
| 6 | Would a tuned nnPU change the verdict? Ours uses a fixed 120 epochs, full-batch Adam, no inner validation or early stopping. | The paper already discloses this. But a reviewer may reasonably ask, and the honest answer is that we do not know. |
| 7 | Are the two undetectable subjects (5C at AUC 0.388, BG at 0.633) explicable from measurable properties, or arbitrary? | If predictable from episode count, EDA quality, or labelling propensity, that is a finding about who the method fails for. If not, it is unexplained heterogeneity and should be labelled as such. |
| 8 | Does 5C being *below* chance indicate something systematic (inverted response, mislabelled intervals) rather than noise? | Below-chance on a fold is not the same as uninformative. Worth one plot before publication. |
| 9 | Eligible negative hours: 265.2 measured here against 231.0 in a parallel notebook implementation, a 21% gap. | The negative ratio is derived from this number. The gap is unresolved and should be closed or disclosed. |
| 10 | Overlapping *session* pairs: 1 measured here against 7 in the notebook. | Low stakes, still unexplained. |

## Deliberately not attempted

| # | Item | Reason |
|---|---|---|
| 11 | Self-supervised pretraining and the label-efficiency sweep. | Gated on a pre-registered rule that was not met. The gate tested the *label-treatment* axis; representation learning is a different axis and the gate does not strictly bear on it. If attempted later, the label-efficiency sweep (25/50/100/138 episodes, pretrained against random init) is the deliverable, not a head-to-head score, and the subject probe against 0.768 is a mandatory stop condition. |
| 12 | Heart rate variability as a baseline feature. | 5.9% window coverage. Retained as an ablation arm only. |
| 13 | Three-class severity modelling. | 20 medium-severity episodes study-wide, 9 of 15 subjects with none. |
| 14 | Multiple-instance learning over episode intervals. | The most principled response to interval labels treated as uniform states, and out of scope here. Named in the discussion as future work. |

## Resolved, recorded so they are not relitigated

| # | Question | Resolution |
|---|---|---|
| 15 | Window length: 120 s or 300 s? | 120 s. No labelled episode is shorter; 300 s discards 15 episodes; the ultra-short HRV literature supports 120 s independently. **CLOSED** |
| 16 | `min_run` disagreement between collaborators. | Units. One counted beats, the other seconds. At 0.75 s per beat these differ by a third. **CLOSED**, and named in the paper as an instance of the general problem. |
| 17 | HRV viable or not? | Session-level availability (48%) is 8x more flattering than window-level (5.9%). The window is the modelling substrate. **CLOSED** |
| 18 | Negative ratio uniform or per-subject? | Uniform. Class balance moves the decision threshold, and per-fold metrics would not be comparable. Exact uniformity is unattainable; residual spread disclosed. **CLOSED** |
| 19 | Drop subject 8B to raise the ratio ceiling? | No. Removing it moves the ceiling 2.04 to 2.25 because another subject immediately binds, and the minimum negative count per subject does not improve. **CLOSED** |
| 20 | Report mean or median for the onset analysis? | Median and sign test, fixed a priori. The mean reports +9.75 where the median reports +0.568, with five episodes of 140 contributing 63% of the difference. **CLOSED** |

## Writing notes

- Every number in the draft traces to a script in `tasks/` and a commit. Re-derive
  before quoting; several figures in earlier planning documents did not survive
  recomputation.
- Denominators: 24,334 windows is a compute statistic. 138 episodes and
  10 subjects are the denominators.
- Report per fold. Fold AUC spans 0.388 to 0.885 and a mean would conceal that
  two subjects detect nothing.
- All six references verified against Crossref, arXiv, or Consensus before being
  written. None are quoted from memory.
- Prior work reporting F1 0.99 on this dataset is noted as difficult to
  reconcile with subject-grouped evaluation, without attributing a cause. We
  cannot diagnose another group's protocol from a published summary and should
  not appear to.
