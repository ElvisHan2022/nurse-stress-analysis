# Start Here — A Tour of the Nurse Stress Project

This is the file to open first. `PROJECT_GUIDE.md` and `CONCEPTS.md` are references you dip
into; this one is meant to be read start to finish, the way a TA would walk you through a
project at a whiteboard, or the way you'd explain it if someone stopped at your poster and
asked "okay, so what did you actually do?"

It covers three things: what the project is, what every file in this folder is for, and
exactly where things stand right now (as of this session, 2026-08-03).

---

## 1. The thirty-second version

Fifteen nurses wore a wristband (an Empatica E4 — the same kind of device as a fitness
tracker, but research-grade) for about a week each during 2020. It recorded heart rate, skin
conductance, skin temperature, and movement continuously. Several times a day, each nurse
filled out a short survey rating how stressed she felt. The question this project asks:
**can you predict a nurse's self-reported stress level from the wristband data alone?**

If the answer is yes, the pitch is straightforward: hospitals could get an early, objective,
continuous signal of staff stress — instead of waiting for someone to burn out and quit —
and route support before things get bad. That's the motivation section of the paper in one
sentence.

The honest answer, which is the actual finding worth presenting, is: **it depends heavily on
whether you build one model for everyone, or a separate model per person.** One model for
everyone does not work well. A model trained on and calibrated to a single nurse works
considerably better. That contrast is the spine of the whole project, and almost every design
decision downstream — how the data gets normalized, how it gets split into train/test,
which model families were even tried — exists to investigate or work around it.

---

## 2. The one idea everything else hangs on

Here's the plain-language version of why "one model for everyone" struggles, because it's
not obvious and it's the thing worth explaining carefully if you're presenting this.

Two nurses can have the exact same heart rate — say, 78 beats per minute — and mean opposite
things. For one nurse, 78 might be an elevated, stressed reading; her resting rate is 60. For
another nurse, 78 might be her calm baseline; she runs high. **The raw number doesn't carry
enough information on its own — you need to know it relative to that specific person's normal
range.**

A model trained by pooling everyone's raw numbers together never gets to learn that, because
the information about "what's normal for this person" was never given to it. It's not a bug
in the model — the model is doing its best with impossible input.

The fix that's been tried is **per-subject normalization**: instead of feeding the model raw
heart rate, you feed it a z-score — "how many standard deviations away from *this nurse's*
own average is this reading right now." That reframes every question from "is this heart
rate high?" to "is this heart rate high *for her*?" It helps, but per the paper's central
finding, it's not enough on its own to rescue a single pooled model. What works considerably
better is going further and fitting a **separate model per nurse**, trained and evaluated
only on that person's data. Section 5 below explains what that trade-off costs you (less
data per model) and what it buys you (a model that only ever has to answer "does this look
normal for you," not "does this look normal for anyone in the study").

That's the **global vs. per-nurse** distinction you'll see everywhere in the file names and
in the paper. It's not two ways of describing the same experiment — they're answering two
different deployment questions:

- **Global**: "A brand-new nurse just started. We have no history on her. Can we still flag
  her stress?" (Tested by holding out an entire nurse the model never trained on.)
- **Per-nurse**: "We know this nurse; we have days of her baseline. Can we flag when today is
  different?" (Tested by holding out one of her days and using her other days as history.)

---

## 3. How this folder is laid out

```
Nurse Stress\
├── START_HERE.md        <- you are here
├── PROJECT_GUIDE.md      <- setup instructions + a file-by-file inventory of the reference repo
├── CONCEPTS.md            <- the deep-dive on every technical idea (windowing, normalization,
│                             metrics, imbalance handling, all nine model architectures)
├── requirements.txt
├── reference\
│   └── MLMAMidtermProject\   <- your friend's original repo, cloned, READ-ONLY
└── my_work\
    └── cnn1d_global.py       <- your first rewrite, annotated, currently the only thing built
```

The split between `reference` and `my_work` is deliberate, not incidental. `reference` is the
friend's repo exactly as they wrote it — you never edit it, so you can always diff your
version against theirs and see exactly what you changed or added. `my_work` is everything
you're building. Right now that's one file; the rest of this document explains what's about
to join it.

---

## 4. File-by-file tour

### 4.1 The three root markdown files, and when to reach for each

| File | Reach for it when... |
|---|---|
| `START_HERE.md` (this one) | You want the narrative — what the project is, why it's built this way, where things stand. Read it front to back once, then use it as a map. |
| `PROJECT_GUIDE.md` | You need to *do* something: install something, find which reference file trains which model, debug an error message, run a script. It's a reference manual, not a story. |
| `CONCEPTS.md` | You need to explain or defend a specific technical choice in depth — why a sliding window, why grouped cross-validation, why macro-F1 instead of accuracy, what each of the nine model architectures is actually doing mathematically. It's the one to reread before a presentation or a defense. |

### 4.2 `reference/MLMAMidtermProject/` — the friend's repo (read-only)

This is the code behind the paper (`MLMA_final_report.pdf`, sitting in your Downloads —
Kwon, Guan, Shrinivasan, Momtaz, JHU EN.520.439). You don't edit anything in here. The two
things worth knowing about it going in:

1. **The filenames lie.** `DT_global.py` (DT = "decision tree") actually trains a random
   forest. `RF_global.py` doesn't train one random forest, it trains twelve — one per nurse
   — and blends their predictions. `lstm_nurse_stress_global.py` is, per its own docstring,
   a *per-nurse* pipeline filed under the global folder. `PROJECT_GUIDE.md` §5 has the true
   contents of every file, because the names will actively mislead you otherwise.
2. **`data/Aditya/` is the one folder that matters.** It holds the 15 per-nurse CSVs at full
   sampling resolution. (`data/Eric/` is a byte-for-byte duplicate; `data/misc/` is an
   earlier, much coarser aggregation.) As of this session, all 15 CSVs are pulled locally —
   the earlier `PROJECT_GUIDE.md` note about only having 3 nurses pulled is now out of date;
   see §6 below.

Two files worth opening if you only open two: `outputs/RF_idealized_model_per_nurse/
RF_idealized_model_per_nurse_results_summary.md` (the paper's headline result, F1 0.859), and
`models/per nurse/RF_idealized_model_per_nurse.py` (the code behind it, and the most
carefully-written file in the repo — see §4.4 below for why).

### 4.3 `my_work/cnn1d_global.py` — the one thing built so far

A faithful, heavily-commented rewrite of the reference repo's `1d_cnn.ipynb`. Explaining it
the way you'd explain it to someone who's never seen it:

**What it's trying to do.** Answer the global question — one model, trained on twelve nurses,
tested on a thirteenth it's never seen — using a neural network that looks at the raw signal
shape rather than a hand-picked summary of it.

**How a "sample" is defined.** Sixty seconds of wristband data at a time (~1,920 individual
readings), sliding forward thirty seconds at a step, so consecutive windows overlap by half.
Each window carries seven numbers per timestep: the four raw channels (movement, skin
conductance, heart rate, temperature) plus three *engineered* channels — the 5-second change
in skin conductance, the 5-second change in heart rate, and a rolling volatility measure of
movement. Those last three exist because the working theory is that **stress shows up as a
rise, not a level** — a heart rate of 95 means nothing without knowing whether the person was
at 65 a minute ago.

**The architecture, in one sentence.** Two convolutional layers slide a small learned filter
across the 60-second window looking for a characteristic local shape (imagine it learning to
recognize "a sharp upward flick in skin conductance," wherever in the minute that flick
happens to occur), then everything gets averaged down to a small dense layer that makes the
final call.

**The clever part, and the part most worth explaining out loud.** After training the
convolutional layers on twelve nurses, the script *freezes* them — locks their learned
weights — and only re-trains the small final layer on a slice of the thirteenth nurse's own
data (chronologically the first 20% of her recording), then tests on the rest. The idea:
the convolutions learn "what a stress signature generally looks like," locked in place, while
the last layer gets to adjust to "what it looks like *for this specific person*." It's a
bridge between the global and per-nurse philosophies rather than a pure instance of either.

**Why it's smaller than the original.** The full version trains on all thirteen usable
nurses and takes 10-15 hours on a laptop CPU. This version defaults to three nurses and caps
windows per nurse, specifically so you can run it in a couple of minutes, confirm your setup
works end to end, and then dial the numbers back up once you trust the pipeline.

### 4.4 The file to actually study closely: `RF_idealized_model_per_nurse.py`

Not yet ported into `my_work`, but worth calling out here because it's the most
methodologically important file in the reference repo, and it's the template the upcoming
per-nurse builds (§6) will follow. Two ideas it gets right that are easy to get wrong:

- **Never tune a decision on the data you'll report a score on.** A model outputs a
  probability, and turning that into "stressed / not stressed" needs a cutoff. It's tempting
  to just try a few cutoffs and report whichever scored best on the test day — but that
  means the test day influenced the model after all, just later in the pipeline. The fix is
  a *three-way* split: train days to fit the model, a separate calibration day to pick the
  cutoff, and the test day touched exactly once, at the very end, to report the number.
- **Throw out folds that can't give you an honest answer.** If a held-out day has only one
  class in it (say, a day where the nurse never reported feeling stressed), no metric
  computed on it means anything. The script checks for a minimum sample count, both classes
  present in train and test, and bounded distribution shift before it accepts a fold.

---

## 5. Reading the paper in five minutes

If you need the conference-abstract version of the actual findings:

- **15 female nurses, ages 30-55**, one week each, during the COVID-19 outbreak, wearing an
  Empatica E4. Self-reported stress via phone survey (0 = none, 1 = medium, 2 = high),
  collapsed to a binary "stressed / not stressed" for modeling, because level 1 was too rare
  to learn and the operational question is binary anyway.
- **Nine model architectures were tried**, splitting into: a linear baseline (logistic
  regression), tree ensembles (decision tree, random forest, XGBoost), neural nets on
  hand-summarized windows (MLP), and sequence models on raw windows (1D CNN, LSTM). Full
  breakdown in `CONCEPTS.md` Part 5.
- **The headline number** is a per-nurse random forest reaching macro-F1 0.859 and accuracy
  0.81, but "idealized" is doing real work in that sentence — it means folds were filtered to
  ones with adequate, balanced data, which is a legitimate way to ask "what's achievable under
  favorable conditions," but is *not* an estimate of what you'd get in an actual deployed
  system that can't discard the inconvenient days.
- **Global models underperformed per-nurse models**, and the paper's explanation is the
  baseline-inconsistency argument from §2 above.
- **Accuracy is close to useless as a metric here**, because roughly 80% of windows are
  labeled "stressed" — a model that always guesses "stressed" scores 80% accuracy while
  learning nothing. This is why the paper leans on macro-F1 instead. (Worth flagging: this
  same fact is *also* the reason the honest-baseline work described in §6 exists — a model
  needs to be shown beating that trivial always-stressed guess, on the metrics that were
  actually reported, before F1 0.859 means what it sounds like it means.)

---

## 6. Where things stand right now, and what's next

This session did three things, in order:

1. **Read the final report PDF** and the existing `PROJECT_GUIDE.md` / `CONCEPTS.md` for
   context (both already thorough — no gaps found).
2. **Pulled the remaining nurse data.** Only 3 of 15 nurse CSVs were real data at the start of
   this session (`15`, `7E`, `8B`); the rest were 133-byte Git LFS pointers. All 15 are now
   pulled (`git lfs pull --include="data/Aditya/*"`, ~935 MB). `my_work/cnn1d_global.py`'s
   `NURSES` list still defaults to just those first three — bump it once you're ready to run
   a fuller sweep.
3. **Designed, but have not yet built**, the next phase of `my_work`: four more algorithms
   (logistic regression, decision tree, random forest, XGBoost), each built for *both* the
   global and per-nurse tracks — something the reference repo doesn't fully have (it has no
   per-nurse decision tree, logistic regression, or XGBoost; only per-nurse random forest,
   MLP, and LSTM exist upstream). The plan, agreed but not yet implemented:
   - A shared 24-number-per-window feature representation (mean, std, min, max, last, and
     slope, across the four raw channels) so all four algorithms — which unlike the CNN
     can't consume a raw sequence — have a common tabular input.
   - A shared **global** evaluation harness (pool N-1 nurses, test on the one left out) and a
     shared **per-nurse** evaluation harness (the calibration-split, fold-filtered LODO
     protocol from §4.4), so each algorithm is a thin script that plugs an estimator into
     a harness that already does the leakage-safe evaluation correctly, rather than eight
     scripts each re-implementing splitting logic slightly differently.
   - **Honest baselines built into every fold**, not added as an afterthought: always-guess-
     stressed, always-guess-not-stressed, and a per-nurse-prior guess, scored alongside every
     real model, on balanced accuracy / macro-F1 / PR-AUC / prevalence — directly motivated by
     the observation in §5 that F1 0.859 needs to be shown beating an 80%-prevalence trivial
     guess before it's a meaningful number, not just a large one.

Two bigger extensions were discussed and deliberately deferred, so they don't quietly balloon
this phase:

   - Going back to the original Dryad source (rather than the merged Kaggle CSV this repo
     uses) to recover heart-rate-variability features from raw inter-beat-interval data,
     which the merged file destroys.
   - Reframing the problem entirely — one-class anomaly detection, multiple-instance
     learning over reported stress *intervals* rather than per-second labels, or
     self-supervised pretraining on the ~11.5M mostly-unlabeled rows.

Both are real, promising directions — they're just separate projects, to be scoped once the
core four-algorithm build produces real numbers to react to.

---

## 7. Plain-English glossary

For the terms in this file and in conversation about it, without the CONCEPTS.md-level detail:

- **Window** — a chunk of continuous time (e.g., 30 or 60 seconds) treated as one data point,
  because a single instantaneous reading can't tell you much on its own.
- **Global model** — one model, trained on data pooled across everyone, tested on a person it
  never saw during training.
- **Per-nurse model** — a separate model per person, trained and evaluated only on that
  person's own data.
- **Normalization / z-score** — rescaling a number to "how many standard deviations from the
  average" instead of its raw value, so it becomes comparable across different people or
  different units.
- **Class imbalance** — when one outcome is far more common than the other (here, ~80%
  "stressed" windows), which makes plain accuracy misleading.
- **F1 score** — a single number balancing "when the model says stressed, is it right"
  (precision) against "does it catch the stressed cases that exist" (recall).
- **Macro-F1** — F1 computed separately for each class, then averaged without weighting by
  how common that class is — so the rare class counts as much as the common one.
- **Baseline** — the simplest possible predictor (e.g., "always guess stressed") that any
  real model has to beat for its score to mean anything.
- **Leakage** — when information from the data you're supposed to be testing on sneaks into
  training or decision-making, which makes a reported score look better than it really is.
- **Fold** — one particular train/test split; cross-validation repeats this several times with
  different splits and averages the results.

---

*If something in this file goes stale — a script gets renamed, a design changes — update this
file alongside the change. It's meant to stay accurate, not to be a one-time snapshot.*
