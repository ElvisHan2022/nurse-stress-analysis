# Concepts Behind the Nurse Stress Project

A companion to `PROJECT_GUIDE.md`. That file tells you what each file in the repo does. This one explains the ideas those files implement, so the code reads as a set of deliberate choices rather than a wall of syntax.

---

## Part 1. What kind of problem is this?

In machine learning vocabulary, this is **supervised binary classification on multivariate time series**. Each of those four words is doing work.

**Supervised** means every training example comes with a correct answer attached. The nurses self-reported their stress through phone surveys, and those reports are the answers. Without them there would be no way to tell the model it was wrong, and the problem would become unsupervised (clustering, anomaly detection).

**Binary** means there are exactly two possible answers: stressed or not stressed. The raw data actually has three levels (0 = none, 1 = medium, 2 = high), so this is a choice the authors made. See Part 2.

**Classification** means the output is a category rather than a number. Predicting "how stressed, on a scale of 0 to 100" would be regression. Predicting "stressed or not" is classification.

**Multivariate time series** means the input is several signals measured simultaneously and repeatedly over time. Four signals (heart rate, skin conductance, temperature, movement), sampled about 33 times per second, for a week.

The last property is what makes this harder than a standard tabular problem. Most introductory machine learning assumes rows are **independent and identically distributed**: shuffling them changes nothing, and any row tells you as much as any other. Here, row 500,001 is nearly identical to row 500,000, because they are 30 milliseconds apart. That single fact drives most of the design decisions in this project, including the sliding window, the grouped cross-validation, and the leakage concerns.

### The core vocabulary

| Term | Meaning here |
|---|---|
| **Sample** (or instance, example) | One thing the model makes a prediction about. In this project a sample is a *window* of about 30 to 60 seconds, not a single sensor reading. |
| **Feature** | One measured quantity describing a sample. `HR`, `EDA`, `mean of EDA over the window`. |
| **Feature vector** | All features for one sample, stacked into a list of numbers. The random forests here use a 24-number vector per sample. |
| **Target** (or label, ground truth, `y`) | The correct answer for a sample. Here, 0 or 1. |
| **Design matrix** (`X`) | All feature vectors stacked. Shape `(n_samples, n_features)`. |
| **Training set** | Data the model learns from. |
| **Test set** | Data held back to measure whether learning generalized. The model must never see it during training. |
| **Fold** | One particular train/test split. Cross-validation rotates through several folds and averages the results. |
| **Parameter** | A number the model learns by itself (a network weight, a split threshold in a tree). |
| **Hyperparameter** | A number *you* choose before training (window length, number of trees, learning rate, dropout rate). |
| **Class imbalance** | One answer is far more common than the other. Roughly 80% of this dataset is labelled stressed. |
| **Generalization** | Performing well on data the model has never seen. The entire point. |
| **Leakage** | Information from the test set influencing training, which inflates the score and makes it meaningless. |

---

## Part 2. The features and the target

### The target

The raw column is `label`, with three values:

| Value | Meaning | Frequency |
|---|---|---|
| 0 | No stress | minority |
| 1 | Medium stress | rare |
| 2 | High stress | majority |

Every model in this repo begins with the same line:

```python
df['label'] = (df['label'] > 0).astype(int)
```

This **binarizes** the target: medium and high collapse into a single "stress" class. Two reasons. First, level 1 is so rare that a three-class model would almost never see it and could not learn it. Second, the operational question ("does this nurse need support right now?") is binary anyway.

The cost is real. The model can no longer distinguish mild strain from acute crisis, which is arguably the distinction a deployed system would most want to make. This is a good example of a modelling decision that improves measurable performance while narrowing what the model is actually useful for.

### The features

Four raw channels come off the Empatica E4 wristband:

| Feature | What it measures | Why it should relate to stress |
|---|---|---|
| `HR` | Heart rate, beats per minute | Sympathetic nervous system activation raises heart rate. The most established physiological stress correlate. |
| `EDA` | Electrodermal activity, skin conductance | Sweat gland activity is controlled almost purely by the sympathetic branch, with no conscious control. Widely treated as the cleanest single stress signal, and the paper's SHAP analysis supports that here. |
| `TEMP` | Skin temperature, °C | Stress causes peripheral vasoconstriction, so blood moves away from extremities and wrist temperature falls. A slower, weaker signal. |
| `acc_mag` | Movement magnitude | Not a stress signal. It is a **confounder control**. Physical exertion raises heart rate and sweating exactly like stress does, so the model needs a channel that lets it tell "sprinting to a code" apart from "sitting and panicking". |

`acc_mag` is itself derived. The raw data has three accelerometer axes (`X`, `Y`, `Z`), which `preprocessing/data_eric.ipynb` collapses:

```python
acc_mag = sqrt(X**2 + Y**2 + Z**2)
```

This is the Euclidean magnitude. It discards direction and keeps only intensity of motion, which is what matters here and costs two columns less.

### The engineered features (1D CNN only)

`1d_cnn.ipynb` adds three more, and the reasoning behind them is the most physiologically interesting part of the project:

```python
df['EDA_slope']  = df['EDA'].diff(periods=160)                      # change over ~5 s
df['HR_slope']   = df['HR'].diff(periods=160)                       # change over ~5 s
df['acc_burst']  = df['acc_mag'].rolling(window=160).std()          # movement volatility
```

**Feature engineering** means constructing new inputs from existing ones because you believe they express the underlying phenomenon more directly. The belief here: acute stress is a *change*, not a *level*. A heart rate of 95 tells you almost nothing without knowing whether the person was at 65 a minute ago or at 110. The first difference captures that trajectory explicitly, so the model does not have to infer it.

This matters because tree-based models cannot compute derivatives on their own. They split on thresholds of features you give them. If you want a tree to reason about change, you must hand it change as a column. Neural networks can in principle learn it, but giving it to them directly saves capacity and data.

---

## Part 3. The sliding window

This is the central idea in the whole project, and the answer to your question.

### The problem it solves

A single sensor reading at one instant is not stress. Heart rate 92 at 14:03:22.15 is meaningless in isolation, since it could be a stressed nurse at rest or a calm nurse walking upstairs. Stress is a **pattern over time**: elevated and rising heart rate, together with increasing skin conductance, sustained over some interval, while movement stays low.

So the unit of prediction cannot be a single row. It has to be a stretch of time.

### The definition

A sliding window carves a continuous recording into overlapping chunks of fixed length. Three hyperparameters define it:

| Hyperparameter | Meaning | Values in this repo |
|---|---|---|
| **Window length** | How much time each sample covers | 60 s in the CNN, 30 s in the random forests, ~6 s in the LSTM |
| **Stride** (step size) | How far forward the window jumps each time | 30 s in the CNN, 5 s in `DT_global.py`, ~1.5 s in the LSTM |
| **Overlap** | `1 - stride/length` | 50% in the CNN, 75% in the LSTM |

In the CNN, at an assumed 32 Hz:

```
WINDOW_SIZE = 32 * 60 = 1920 samples
STEP_SIZE   = 32 * 30 =  960 samples
```

Concretely, on a recording of 100,000 rows:

```
window 0:  rows      0 – 1919      (seconds  0 – 60)
window 1:  rows    960 – 2879      (seconds 30 – 90)
window 2:  rows   1920 – 3839      (seconds 60 – 120)
...
```

Each window becomes one training sample. A recording of 100,000 rows yields roughly 102 windows.

### Labelling a window

A 1920-row window contains 1920 individual labels, which may disagree near a transition. The rule used is **majority vote**:

```python
majority_label = 1 if np.mean(window_labels) >= 0.5 else 0
```

This smooths the boundary where a self-report flips. A window that is 60% stressed becomes a stress window.

### The two things you can do with a window

This is the fork in the road that separates the model families, and it is worth being clear about because it explains why the repo has two visibly different feature representations.

**Option A: summarize the window into a fixed-length vector.**

Compute a handful of statistics per channel and throw the raw sequence away. `RF_global.py`, `DT_global.py`, `XGB_global.py`, and the MLPs all do this:

```python
mean  = w.mean(axis=0)
std   = w.std(axis=0)
minv  = w.min(axis=0)
maxv  = w.max(axis=0)
last  = w[-1]
slope = (w[-1] - w[0]) / (window_steps - 1)
feature_vector = concatenate([mean, std, min, max, last, slope])
```

Six statistics × four channels = **24 numbers**. That is the "24-dimensional feature vector" the paper refers to.

Each statistic answers a question: `mean` asks about the typical level, `std` about variability, `min`/`max` about extremes, `last` about the current state at the moment of prediction, and `slope` about the trend across the window.

This converts a time-series problem into an ordinary tabular problem, which is why random forests and XGBoost can be used at all. They have no concept of time and would be helpless on raw sequences.

The cost is that you have compressed 1920 × 4 = 7,680 numbers down to 24. Anything about the *shape* of the signal is gone. A window that rises smoothly and one that spikes and recovers can have identical means, standard deviations, and endpoints.

**Option B: keep the whole sequence.**

`1d_cnn.ipynb` and the LSTMs feed the raw window in, producing a three-dimensional tensor:

```
(n_windows, timesteps, n_features)
(   ~10600,      1920,           7)
```

Nothing is discarded, and the model can learn shapes. That is precisely what CNNs and LSTMs are for. The cost is memory (about 1.1 GB for one copy of that tensor), compute, and a much greater appetite for training data.

The paper's result is that this extra power did not pay off, because the dataset is too small and too imbalanced to support it.

### Why overlap, and the danger it creates

Overlap exists to manufacture more training examples. A 50% overlap roughly doubles the sample count over non-overlapping windows.

The danger: window 0 and window 1 share 960 of their 1920 rows. They are *not independent samples*. If you shuffle all windows and take a random 20% as a test set, a test window will almost certainly overlap a training window by half. The model has effectively seen the answer. Your reported accuracy will be excellent and completely fictitious.

This is the single most common way time-series machine learning goes wrong, and it is why every evaluation in this repo is grouped by nurse or by day rather than randomly split. See Part 6.

### A caveat specific to this repo

The windows are cut by **row position**, not by timestamp. Preprocessing assigned a synthetic time axis (`row_index * 0.03`) that assumes perfectly continuous recording. In reality there are multi-hour gaps between shifts. So a "60 second" window can span the end of Tuesday's shift and the start of Wednesday's. The per-nurse scripts avoid this by grouping on the calendar date first; the CNN does not.

---

## Part 4. Normalization, and why it is the whole story here

**Z-score normalization** rescales a feature so it has mean 0 and standard deviation 1:

```
z = (x - mean) / std
```

The standard motivation is that models converge faster and are better behaved when features share a scale, since heart rate around 80 and skin conductance around 0.5 are otherwise incomparable in magnitude.

In this project it does something far more important. The question is **what data you compute the mean and standard deviation from**, and the two answers define the paper's central experiment.

**Global normalization** uses statistics pooled across all 15 nurses. A z-score then means "how unusual is this value relative to nurses in general".

**Per-subject normalization** fits a separate `StandardScaler` for each nurse. A z-score then means "how unusual is this value **for this person**".

The paper's core finding is that the first framing is close to hopeless:

> signal baselines were inconsistent across the 15 nurses, meaning one nurse's resting heart rate might be equivalent to another nurse's heart rate during high stress

Under global normalization, identical feature values carry opposite labels depending on whose wrist they came from. No model can resolve that, because the information needed to resolve it was destroyed in preprocessing.

Per-subject normalization converts the question from "is this heart rate high?" to "is this heart rate high *for you*?", which is both answerable and clinically the right question.

There is one further subtlety worth internalizing. The scaler must be **fit on training data only** and then applied to test data. Fitting it on everything lets the test set's mean and variance influence the training pipeline, which is a mild form of leakage. `LSTM_nurse_stress_LODO.py` gets this right by fitting on training days and transforming all days.

---

## Part 5. The model families

Nine architectures were tried. They fall into four groups, and the grouping tells you more than the individual names.

### Group 1: linear baseline

**Logistic regression** (`LogReg_global.ipynb`). Fits a weighted sum of the features, squashes it through a sigmoid to get a probability between 0 and 1, and thresholds at 0.5. The decision boundary is a straight line (a hyperplane in higher dimensions).

Its role is to establish a floor. If a sophisticated model cannot beat logistic regression, the sophistication is not buying anything. Here it is applied to point-wise EDA/HR/TEMP with no windowing at all, so it is the simplest possible thing that could work.

### Group 2: tree ensembles on summarized windows

These consume the 24-dimensional vector from Part 3, Option A.

**Decision tree.** A sequence of yes/no questions on feature thresholds ("is mean EDA above 0.4?"), forming a tree whose leaves are predictions. Interpretable, and prone to memorizing the training set.

**Random forest** (`RF_global.py`, `DT_global.py`, and the per-nurse scripts). Trains hundreds of decision trees, each on a random subsample of rows and a random subset of features at each split, then averages their votes. The randomness decorrelates the trees so their individual errors cancel. This is **bagging**.

Random forests are the workhorse here, and they produce the paper's best result. That is not an accident. They handle small tabular datasets well, need little tuning, are robust to features on different scales, and give feature importances for free. For a 24-feature problem with a few thousand samples, they are close to the right default.

**XGBoost** (`XGB_global.py`). **Gradient boosting**: trees are trained sequentially, and each new tree is fit to the errors the previous ones are still making. Where bagging averages independent guesses, boosting corrects itself iteratively. Usually stronger than a random forest, and correspondingly easier to overfit.

### Group 3: neural networks on summarized windows

**Multilayer perceptron** (`Mlp_global.ipynb`, `Mlp_per_nurse.ipynb`). A stack of fully connected layers with nonlinear activations:

```
24 → 64 → 32 → 1
```

Each layer computes a weighted sum of the previous layer and passes it through ReLU (which sets negatives to zero). The final layer produces one number, sigmoid-squashed into a probability.

An MLP can represent curved decision boundaries that logistic regression cannot. It has no notion of order or time, so feeding it the 24-dim vector is appropriate, but feeding it a raw sequence would not be.

Two regularization techniques appear here. **Dropout** (0.3) randomly zeroes 30% of neurons each training step, forcing the network not to depend on any single pathway. **Early stopping** halts training when validation performance stops improving, so the model does not continue into memorization.

### Group 4: sequence models on raw windows

These consume the 3-D tensor from Part 3, Option B.

**1D convolutional neural network** (`1d_cnn.ipynb`). A convolution slides a small learned filter (kernel size 5, so five timesteps wide) across the time axis, computing a response at every position. The filter learns to fire on a particular local shape, perhaps a sharp EDA upstroke. Because the same filter is applied everywhere, the model learns the pattern once and detects it anywhere in the window. This is **translation invariance**, and it is why CNNs need far fewer parameters than an MLP over the same input.

The architecture stacks two convolutional blocks, then applies `GlobalAveragePooling1D`, which averages each filter's response across all 1920 positions. That deliberately throws away *where* the pattern occurred and keeps only *whether* it occurred, which is the right inductive bias when a stress signature could appear at any point in the minute.

`BatchNormalization` between layers keeps activations at a stable scale so training does not diverge.

**Long short-term memory network** (the LSTM files). A recurrent network processes the sequence one timestep at a time, carrying a hidden state forward. The LSTM adds learned gates that control what to keep, what to forget, and what to output, which lets it retain information across long stretches where a plain recurrent network would lose it.

Only the final hidden state is passed to the classifier. The interpretation is that the network reads the whole window and compresses it into a summary of what it saw.

CNN and LSTM encode different assumptions. The CNN asks "does a characteristic local shape appear anywhere here?". The LSTM asks "what is the state of this signal after watching the whole sequence unfold in order?". Both are defensible for physiological data, which is why both were tried.

### Group 5: ensembles

**`RF_global.py` and `XGB_global.py`** are misleadingly named. Both train **one model per nurse**, then blend the models' probability outputs, weighting each by how well it scored on a calibration split. This is a hedge between global and per-nurse: personalized models, aggregated for a global prediction.

**`Branched_Ensemble_global.py`** runs two parallel feature streams over each window. Branch A carries level statistics (mean, std, min, max, last, slope). Branch B carries dynamics and volatility (quartiles, interquartile range). Each branch feeds its own learners, and the two are fused with a calibrated mixing weight. The premise is that "how high is the signal" and "how erratic is the signal" are different kinds of evidence and deserve separate treatment before being combined.

---

## Part 6. Global vs per-nurse, and how these models are evaluated

### The experimental contrast

**Global** models pool data from all nurses and are tested by holding out an entire nurse (**Leave-One-Subject-Out**, LOSO, also written LONO for leave-one-nurse-out). The question: *can a model trained on other people predict stress in a person it has never met?*

**Per-nurse** models fit a separate model for each nurse, normalized on that nurse's own data, and are tested by holding out an entire day (**Leave-One-Day-Out**, LODO). The question: *given a personal baseline, can we detect departure from it?*

The paper's answer is that global fails and per-nurse works, for the reason in Part 4. Every result it actually defends comes from the per-nurse family.

`1d_cnn.ipynb` attempts a bridge between the two, and its protocol is worth understanding as a technique in its own right. **Transfer learning**: train the CNN on 12 nurses, then freeze the convolutional layers and re-fit only the dense head on the first 20% of the held-out nurse's recording, then score the remaining 80%. The convolutions learn what a stress signature looks like in general and are locked; the head learns what it looks like in this specific person. The split must be chronological, since a random split would let the model see the future.

### Why grouped splits instead of a random split

A random 80/20 split is standard practice and would be catastrophically wrong here, for two reasons stacked on top of each other. Overlapping windows share raw samples (Part 3). And even non-overlapping windows from the same nurse on the same day share that nurse's baseline physiology, calibration, and shift context.

Holding out whole nurses or whole days is what makes the reported number mean something. The rule of thumb generalizes: **split along the axis you want to generalize across.** If the deployment scenario is "new nurse, no history", hold out nurses. If it is "known nurse, new shift", hold out days.

### Metrics

**Accuracy** is the fraction of correct predictions, and it is close to useless here. About 80% of windows are labelled stress, so a model that outputs "stressed" unconditionally scores 0.80 while being worthless. This is the **accuracy paradox** under class imbalance, and it is why the paper explicitly declines to use accuracy as a primary metric.

The alternatives, for the stress class:

| Metric | Definition | Question it answers |
|---|---|---|
| **Precision** | TP / (TP + FP) | When the model says "stressed", how often is it right? |
| **Recall** (sensitivity) | TP / (TP + FN) | Of all genuinely stressed windows, how many did we catch? |
| **F1** | harmonic mean of precision and recall | A single number balancing the two. Harmonic rather than arithmetic so that one bad component drags the score down. |
| **Macro-F1** | F1 computed per class, then averaged unweighted | Forces the rare class to count as much as the common one. The primary metric in this paper. |
| **Balanced accuracy** | mean of per-class recall | Same spirit as macro-F1, immune to imbalance. |
| **PR-AUC** | area under the precision-recall curve | Summarizes performance across all thresholds. Preferred over ROC-AUC when classes are heavily imbalanced. |
| **ROC-AUC** | area under the true-positive vs false-positive curve | Threshold-independent, but can look flatteringly high under imbalance. |

The precision/recall tradeoff has a real clinical shape here. High recall means few stressed nurses missed, at the cost of false alarms that erode trust in the system. High precision means every alert is credible, at the cost of missing people who needed support. Which to favor is a deployment decision, not a statistical one.

### The decision threshold, and the paper's most important methodological point

A classifier does not output a class. It outputs a **probability**. Converting that to a decision requires a threshold, conventionally 0.5.

Under 80/20 imbalance, 0.5 is a poor choice, and moving the threshold can improve macro-F1 substantially. So it is natural to tune it.

The trap: if you tune the threshold by checking which value scores best **on the test day**, you have used the test set to make a modelling decision. The test set is no longer held out, and the reported score is optimistic. The paper calls this "implicit validation leakage", and it is exactly the difference between `RF_standard_model_per_nurse.py` and `RF_idealized_model_per_nurse.py`.

The fix is a **three-way split**:

```
train days        →  fit the model
calibration day   →  choose the threshold
test day          →  report the score, touched exactly once
```

This is the general pattern for any decision made after training: it needs its own data. Understanding why is probably the most transferable thing in this repository.

### What "idealized" means

`RF_idealized_model_per_nurse.py` produces the headline result (macro-F1 0.859, accuracy 0.81), and the qualifier is doing heavy lifting. Beyond the calibration split, it discards folds that fail quality checks: minimum sample counts, both classes present in train and test, and bounded class-distribution shift between splits.

That is a legitimate way to answer "what could this method achieve if the data collection were adequate?", and the paper is explicit that this is the intent. It is **not** an estimate of deployed performance, because in deployment you cannot discard the inconvenient days. Read 0.859 as an upper bound under favorable conditions.

---

## Part 7. Handling class imbalance

Four techniques appear in this repo, and they intervene at different points.

**Class weights.** Multiply the loss for minority-class errors so the model is penalized more for getting them wrong. `compute_class_weight('balanced', ...)` in the CNN, `class_weight='balanced_subsample'` in the random forests. Cheapest option, no data thrown away.

**Weighted sampling.** `WeightedRandomSampler` in the per-nurse LSTM assigns each window a draw probability inversely proportional to its class frequency, so each training batch is roughly balanced regardless of the underlying distribution. The paper uses this alongside class weights because imbalance was severe at the single-day level, and a batch drawn from an almost entirely stressed day would otherwise show the model no counterexamples at all.

**Undersampling.** Discard majority-class examples until the ratio improves. Present in the CNN as `moderate_undersample()` but commented out, and available in `DT_global.py` as `--balance-train undersample`. Simple, and it throws away real data.

**Focal loss.** Down-weights examples the model already classifies confidently, concentrating training on hard cases. Appears commented out in `build_1d_cnn`, an approach they tried and dropped.

**SMOTE** (Synthetic Minority Oversampling Technique) appears in `LogReg_global.ipynb`. It generates synthetic minority examples by interpolating between real neighbors. Worth noting that SMOTE is questionable on time-series windows, since interpolating between two physiological states can produce a window that is not physiologically possible.

---

## Part 8. Reading the paper section by section

| Paper section | What to take from it |
|---|---|
| **Abstract** | The whole arc in eight sentences. Global models fail, per-nurse models are variable, idealized per-nurse RF reaches 0.859. |
| **I. Introduction** | Motivation. Self-report is biased and retrospective; wearables are continuous and objective. Establishes HRV and EDA as literature-supported stress markers. |
| **II.A Data Description** | 15 female nurses, ages 30–55, Empatica E4, one week each, during COVID-19. 11.5 M rows, 9 columns. Note the sentence that the dataset does not record *when* stress was reported. That single gap explains much of the label noise. |
| **II.B Preprocessing** | Four limitations and the response to each: variable-length recordings, mostly missing labels, inconsistent baselines (answered by per-subject z-scoring), and three accelerometer axes (answered by `acc_mag`). Also documents the switch away from 60-second aggregation to raw readings, on the grounds that aggregation left too few informative features. |
| **II.C Model Families** | The global vs per-nurse contrast in two paragraphs. Part 6 above. |
| **II.D Global Models** | Which architectures, and what input representation each consumed. |
| **II.E Per-Nurse Models** | The most technically detailed section. E.2 on the LSTM is worth close reading: the 20–80% stress-ratio day filter, the 200-sample windows with stride 50, the `time_progress` feature, and the doubled imbalance handling. |
| **III. Results and Discussion** | Day-grouped cross-validation, why accuracy was rejected, why three nurses were excluded (CE and EG lack both classes, 6D has one day). The EDA-distribution analysis in Figure 5 is the paper's one attempt at explaining *why* certain nurses were hard, and the authors correctly flag it as suggestive rather than established. |
| **IV. Conclusions** | The honest limitations: severe imbalance driven by the pandemic collection window, inter-nurse variability, only 15 subjects, all female. |

---

## Part 9. What to take away methodologically

Five ideas here transfer well beyond this dataset:

1. **The unit of prediction is a design choice.** Nothing forced a window to be the sample. Choosing 60 seconds rather than 5 or 600 was a hypothesis about the timescale at which stress expresses itself physiologically.

2. **Feature representation determines which models are even available.** Summarizing a window into 24 numbers is what makes random forests applicable. Keeping the sequence is what makes CNNs applicable. The representation decision precedes and constrains the model decision.

3. **Split along the axis you want to generalize across.** Random splits are the default in tutorials and are wrong for grouped or temporal data.

4. **Every decision made after training needs its own data.** Threshold tuning, model selection, early stopping. Each one consumes a piece of the evaluation budget, and reusing the test set for any of them invalidates the number you report.

5. **State what a number is an estimate of.** The 0.859 is real, and it is an estimate of achievable performance under filtered, favorable conditions. It is not an estimate of what would happen on a hospital ward. The paper is careful about this, and that care is the difference between a credible result and an overclaim.
