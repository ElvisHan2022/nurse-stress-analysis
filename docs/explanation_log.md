# Explanation log

A running record of which concepts needed more than one pass to land, kept
because it is direct evidence of what the paper's methods section has to explain
carefully.

**Why this is worth keeping.** The two people who built this pipeline are the
most motivated readers it will ever have, and they had the author of the code
available to ask. Where *they* needed two or three passes, a reviewer skimming
the methods section at speed will not get it in one. Anything with two or more
entries below is a concept the paper must carry rather than assume.

Entries are dated and quote the question as asked, because the phrasing of a
confused question usually locates the ambiguity better than a summary does.

---

## Ranked by how many passes it took

| Concept | Passes | What was actually unclear |
|---|---|---|
| **Negative-sampling seeds** | **3** | What a seed is; what it is *for*; and a misconception that we pick the best-performing one |
| **`RATIO`** | **3** | What it is; where the number comes from; why uniform rather than per-subject |
| **Ablation vs falsification** | **2** | Which direction each runs in — restated incorrectly both times |
| **Activity matching** | **2** | The mechanism, and why matching rather than filtering |
| **`causal_z` / trailing IQR** | 1 | Every part of it — "what the heck is this" |
| **HRV window length** | 1 | Window length conflated with run-admission threshold |
| **Phase position** | 1 | Which phase we were in and what was blocking |

---

## 2026-08-29 · Seeds (three passes)

> *"what is a seed and what do you mean by 'deterministic'"*

> *"what is the purpose of these seeds?"*

> *"is the intent to create three different datasets ... and we compare them to
> each other to see which seed was the best?"*

**The misconception that matters:** that seeds are candidates to choose between.
They are not — picking the best-scoring seed is selection on the outcome and
would inflate the result by roughly the size of the spread.

**What was missing from the first two explanations:** what you actually *do*
with three tables. Abstract statements about "sampling variance" did not land.
What landed was running the falsification checks across all three seeds and
showing the numbers: a spread of ~0.01 AUC, against an EDA-vs-ACC gap of 0.25.

**For the paper.** Do not define seeds abstractly. State the spread as a
measured quantity, then use it immediately as the threshold for a specific
comparison. The concept is only legible through an example.

---

## 2026-08-29 · The ratio (three passes)

> *"what is this ratio?"*

> *"how is ratio calculated? is it based on potential negatives? ... what is
> 5.04:1, 32:1, 1.9:1, and how was 2.14:1 and 1.72:1 calculated?"*

> *"why are we using 2.14 as the ratio instead of using the ratio that's
> established for each nurse?"*

**Root cause:** two different quantities were both being called "the ratio" —
what the data *could* supply (available) and what we actually *drew* (achieved).
Once separated into two columns, the question resolved immediately.

**What landed:** the worked F1 table showing one fixed model scoring 0.727 in a
2:1 fold and 0.195 in a 32:1 fold. The abstract statement "class balance affects
the threshold" did not land; the arithmetic did.

**Their own restatement, which is the sentence to use in the paper:** *"we want
to craft the negatives so they have the same ratio between the nurses, so that
when we calculate F1 scores you can calculate them in the same way."*

---

## 2026-08-29 · Ablation vs falsification (two passes)

> *"what is the ablation process ... and what we're ablating (the window
> length?)"*

> *"the falsification check where we remove or we test one feature naively to
> see how model performance moves"*

Both restatements had the direction wrong. An ablation *removes* a component
from the full model and expects a small drop. A falsification check builds a
model from *only* the suspicious signal and expects it to fail. Same vocabulary
neighbourhood, opposite construction and opposite expected result.

**For the paper.** Never use "ablation" as a generic word for "experiment that
varies something." `CONTRIBUTING.md` §4 fixes the three terms; use them
precisely and define them at first use.

---

## 2026-08-29 · Activity matching (two passes)

> *"can you explain the activity matching in more simple terms please?"*

The first explanation described the procedure. The second used the drug-trial
analogy — recruiting the treatment group from a cardiology clinic and the
control group from a gym — and that is what landed.

**For the paper.** Lead with the confound, not the procedure. The reader has to
want the fix before the fix is interesting.

---

## 2026-08-29 · Normalisation

> *"what the heck is causal z-denominator and what is this IQR-trailing window
> that you are mentioning?"*

Four separate ideas compressed into one term: z-score, causal windowing, robust
statistics, and the interquartile range. Explaining them in layers worked;
naming the whole thing at once did not.

**For the paper.** Unpack in the order: what a z-score is → why the baseline has
to come from the past only → why median and IQR rather than mean and SD → what
happens when the IQR collapses.

---

## 2026-08-28 · HRV window length

> *"we have a discrepancy on using HRV because my calculations were based at
> 300s as opposed to his interval of 120s"*

Two independent parameters were being treated as one: the **analysis window
length** (how long one prediction covers) and the **run-admission threshold**
(how much clean beat data must sit inside it). A third, unstated choice — what
"available" means — was doing most of the damage.

This one is worth reporting in the paper as a finding rather than a
clarification. It is an instance of the general problem the project is about:
an unstated definition producing a disagreement that looks empirical.

---

## Notes for the methods section

1. **Every concept above needs a worked example, not a definition.** In every
   case the definition failed and the arithmetic worked.
2. **Two terms need units stated at every use:** `min_run` (beats or seconds)
   and "label" (survey severity 0/1/2, or model label 0/1).
3. **Two quantities need distinct names throughout:** available ratio and
   achieved ratio.
4. **The seed spread should appear before its first use as a threshold**, so the
   reader has the noise floor in hand when the first comparison arrives.
5. **Lead each decision with the problem it solves.** The confound before the
   matching, the collapse before the floor, the imbalance before the ratio.
