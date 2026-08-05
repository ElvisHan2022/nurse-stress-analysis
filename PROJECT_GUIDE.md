# Nurse Stress Detection — Local Setup and Project Map

Reference repo: https://github.com/adishri8/MLMAMidtermProject
Paper: `MLMA_final_report.pdf` (Kwon, Guan, Shrinivasan, Momtaz — JHU EN.520.439)
Target machine folder: `C:\Users\Elvis\Documents\Nurse Stress`

---

## 1. Answer to your question: is this the right way to proceed?

Yes, cloning locally is the right instinct. Three refinements will save you a lot of pain.

**a) Separate "reference" from "my work."**
If you clone into the folder you then edit, you lose the ability to diff your version against theirs, and `git status` fills with noise. Keep the friend's repo read-only and write your recreation next to it.

**b) Do not pull all the data on the first clone.**
Every `data/*/processed_nurse_*.csv` is a Git LFS pointer, not a real file. A blind `git clone` gives you 133-byte stubs, and every script will fail with a confusing pandas parse error. The full data payload is about **935 MB**. Pull two or three nurses first, confirm the pipeline runs end to end, then pull the rest.

**c) Do not try to reproduce the full LOSO run before you understand it.**
`1d_cnn.ipynb` trains 13 separate global CNNs (one per held-out nurse) over roughly 10,600 windows of shape 1920 × 7. On a laptop CPU that is on the order of **10 to 15 hours** and peaks somewhere around 4 to 6 GB of RAM. See §7 for how to shrink it to a two-minute loop while you learn the code.

One more thing worth knowing before you read anything: **the filenames in this repo do not reliably describe the contents.** `DT_global.py` trains a random forest. `lstm_nurse_stress_global.py` is documented as a per-nurse pipeline. `RF_global.py` is a per-nurse ensemble rather than a single pooled model. §5 gives the actual contents of each file.

---

## 2. Recommended folder layout

```
C:\Users\Elvis\Documents\Nurse Stress\
├── PROJECT_GUIDE.md              <- this file
├── requirements.txt
├── reference\
│   └── MLMAMidtermProject\       <- friend's repo, read-only
└── my_work\
    └── cnn1d_global.py           <- annotated local rewrite of 1d_cnn.ipynb
```

---

## 3. Setup (PowerShell, run from `C:\Users\Elvis\Documents\Nurse Stress`)

### 3.1 Install Git LFS

The repo will not work without it. Check first:

```powershell
git lfs version
```

If that errors, install it:

```powershell
winget install --id GitHub.GitLFS -e
git lfs install
```

### 3.2 Clone without downloading the 935 MB of CSVs

```powershell
mkdir reference
cd reference
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/adishri8/MLMAMidtermProject.git
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE
cd MLMAMidtermProject
```

This takes about 590 MB (the `.joblib` and `.pt` model artifacts are ordinary Git objects, so they do come down). The CSVs are still pointers at this stage.

### 3.3 Pull only the nurses you need

Start with three:

```powershell
git lfs pull --include="data/Aditya/processed_nurse_15.csv"
git lfs pull --include="data/Aditya/processed_nurse_7E.csv"
git lfs pull --include="data/Aditya/processed_nurse_8B.csv"
```

Verify a real file arrived rather than a pointer:

```powershell
Get-Content data\Aditya\processed_nurse_15.csv -TotalCount 3
```

You should see `datetime,time,acc_mag,EDA,HR,TEMP,label` and two data rows. If you instead see `version https://git-lfs.github.com/spec/v1`, LFS did not run.

Later, when you want everything:

```powershell
git lfs pull --include="data/Aditya/*"
```

### 3.4 Python environment

```powershell
cd C:\Users\Elvis\Documents\Nurse Stress
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name nurse-stress --display-name "Nurse Stress"
```

Python 3.11 is the safe choice. TensorFlow's Windows wheels lag the newest Python releases, and this project needs both TensorFlow (CNN, initial LSTM) and PyTorch (MLP, main LSTM).

Note that TensorFlow 2.11 and later have no native GPU support on Windows. Everything here runs on CPU unless you move to WSL2.

### 3.5 Open in VS Code

```powershell
code C:\Users\Elvis\Documents\Nurse Stress
```

Install the Python and Jupyter extensions, then `Ctrl+Shift+P` → *Python: Select Interpreter* → the `.venv` you just made. In any notebook, pick the "Nurse Stress" kernel in the top right.

---

## 4. Project structure

### 4.1 `data/` — three folders, two of which are identical

| Folder | Per-file size | Total | What it is |
|---|---|---|---|
| `data/Aditya/` | 20 to 122 MB | ~935 MB | Raw-resolution per-nurse CSVs. **This is the one every script points at.** |
| `data/Eric/` | identical | ~935 MB | Byte-for-byte identical to `Aditya` (same LFS object IDs on all 15 files). |
| `data/misc/` | 30 to 173 KB | ~1.3 MB | ~700× smaller. Almost certainly the earlier 60-second-window aggregation described in §II.B of the paper. |

Two consequences:

- The paper says "This initial preprocessing version is under `Eric/`." That is no longer true of the current repo state. `Eric` and `Aditya` now hold the same raw-resolution data, and `misc` holds the aggregated version. Do not trust the paper's folder attribution.
- You never need to pull `data/Eric/`. It is a duplicate. Some notebooks point at it (`Mlp_global.ipynb`, `lstm_classifier_initial.ipynb`); just repoint them to `data/Aditya/`.

Schema of a `processed_nurse_XX.csv`:

| Column | Meaning |
|---|---|
| `datetime` | Real wall-clock timestamp from the Empatica E4 |
| `time` | **Synthetic** elapsed seconds, `row_index * 0.03`, restarting at 0 per nurse |
| `acc_mag` | `sqrt(X² + Y² + Z²)`, the three accelerometer axes collapsed to one magnitude |
| `EDA` | Electrodermal activity (skin conductance) |
| `HR` | Heart rate, BPM |
| `TEMP` | Skin temperature, °C |
| `label` | 0 = no stress, 1 = medium, 2 = high. Every model binarizes to `label > 0`. |

The `time` column deserves attention. It is generated by `preprocessing/data_eric.ipynb` as a pure row counter times 0.03 s, so it assumes perfectly continuous 33.3 Hz sampling. Real recordings have gaps of hours between shifts. Any windowing that indexes by row position, which includes `1d_cnn.ipynb`, can therefore produce a "60-second" window that actually straddles two different days. The per-nurse scripts avoid this by grouping on `datetime.dt.date` first.

### 4.2 `preprocessing/`

| File | Purpose |
|---|---|
| `data_eric.ipynb` | The one that matters. Reads `merged_data.csv.zip`, groups by nurse `id`, builds `time` and `acc_mag`, drops `X/Y/Z`, writes the 15 `processed_nurse_*.csv` files. Run this and you regenerate `data/Aditya/` yourself. |
| `preprocessing_eda.ipynb` | Exploratory plots, 1.6 MB of stored figure output. Source of Figures 1, 4, 5. |
| `raw_data_vis.ipynb` | Raw signal visualization. |

### 4.3 `outputs/` — results already computed

Pre-existing artifacts, so you can inspect results before running anything. Useful for checking your recreation against theirs.

- `outputs/RF_idealized_model_per_nurse/` — the headline result (macro-F1 0.859, accuracy 0.81). Read `RF_idealized_model_per_nurse_results_summary.md` first.
- `outputs/sliding_window_rf/` — global RF, with `stressed/` and `unstressed/` subruns.
- `outputs/LSTM_model_global/` — 12 PyTorch checkpoints, one per nurse.
- `outputs/*_shap*/` — SHAP feature-importance tables.
- `outputs/compiled_results_table.csv` — Table 2 of the paper.

### 4.4 Root-level leftovers

`mlma_project.ipynb` and `random_forest.ipynb` are earlier midterm-era scratch work. `merged_data.csv.zip` is duplicated at the root and in `raw_data/`. You can ignore all of these.

---

## 5. What each model actually is

### 5.1 The dividing line: global vs per-nurse

This is the central experimental contrast in the paper, so it is worth stating precisely.

**Global** means one model is fit on data pooled across nurses, and evaluation holds out an entire nurse (Leave-One-Nurse-Out). The question being asked is: *can a model trained on other people predict stress in a person it has never seen?* The paper's answer is essentially no. Resting heart rate for one nurse sits where high-stress heart rate sits for another, so identical feature values carry opposite labels across subjects.

**Per-nurse** means a separate model is fit for each nurse, normalized only on that nurse's own data, and evaluation holds out an entire day (Leave-One-Day-Out). The question becomes: *given a personal baseline, can we detect deviation from it?* This works considerably better, and every result the paper actually defends comes from this family.

### 5.2 `models/global/`

| File | Framework | Input representation | Evaluation | Notes |
|---|---|---|---|---|
| `LogReg_global.ipynb` | sklearn | Point-wise EDA, HR, TEMP. No windowing. | Group split by nurse id | Weakest baseline. The only file that fetches data from Kaggle via `kagglehub` instead of reading `data/`. Later cells experiment with `class_weight='balanced'`, upsampling, and SMOTE. |
| `DT_global.py` | sklearn | 30 s sliding windows → 24-dim vector (mean, std, min, max, last, slope × 4 channels) | Leave-one-nurse-day-out | **Named DT, actually a RandomForestClassifier.** The most production-shaped script in the repo: full `argparse`, a `--positive-class stressed\|unstressed` switch, fold-quality filters, and `--threshold-strategy balanced_acc`. It writes `outputs/sliding_window_rf/`. |
| `RF_global.py` | sklearn | Same 24-dim windows | Held-out nurse windows | Despite the name, this is a **per-nurse ensemble**: it fits one RF per nurse, weights each by its calibration balanced accuracy, and averages probabilities. Saved as `rf_eric_sliding_window.joblib`. |
| `XGB_global.py` | xgboost | Same 24-dim windows | Held-out validation nurses | Same personalized-aggregation pattern as `RF_global.py`, with XGBoost as the base learner and a calibration-tuned decision threshold. |
| `Branched_Ensemble_global.py` | sklearn | Two parallel 24-dim streams per window | Held-out nurses | Branch A = level statistics (mean, std, min, max, last, slope). Branch B = dynamics and volatility (quartiles, IQR, and similar). Base learners are `HistGradientBoostingClassifier` and `RandomForestClassifier`; the two branches are fused with a calibrated mixing weight. The most elaborate global architecture. |
| `Mlp_global.ipynb` | PyTorch | Same 24-dim windows | Leave-One-Nurse-Out folds | `StressPredictor`: 24 → 64 → 32 → 1, ReLU, dropout 0.3, sigmoid. Also holds the label-distribution plots. Points at `data/Eric/`. |
| `lstm_classifier_initial.ipynb` | TensorFlow | 128-timestep sequences, 4 raw features | `GroupKFold` by nurse | Exploratory first LSTM. Stacked LSTM 64 → 32, dropout 0.25/0.20, dense 16, sigmoid. Points at `data/Eric/`. |
| `lstm_nurse_stress_global.py` | PyTorch | 200-timestep sequences (~6 s at 33 Hz), stride 50, 5 features including `time_progress` | Day-based splits | Docstring reads "Per-Nurse LSTM Pipeline" and it saves one checkpoint per nurse (`outputs/LSTM_model_global/nurse_XX_lstm.pt`). Structurally per-nurse, filed under global. 2 layers × 128 hidden, dropout 0.3, Adam 1e-3, 30 epochs with early stopping on macro-F1. |
| `1d_cnn.ipynb` | TensorFlow | 60 s windows at assumed 32 Hz → 1920 timesteps × 7 features, 30 s step | Leave-One-Subject-Out with transfer learning | Your starting point. Detailed in §6. |

### 5.3 `models/per nurse/`

| File | What it does |
|---|---|
| `RF_idealized_model_per_nurse.py` | **The most important file in the repo.** Produces the paper's headline F1 of 0.859. Also the shared library: it defines `load_nurse_csv`, `compute_nurse_normalization`, `build_windows_for_day`, `generate_day_combos`, `has_min_class_mix`, and `fit_and_predict`, which the other two RF scripts import. "Idealized" refers to three constraints: folds are rejected unless they meet minimum sample counts and contain both classes in train and test; class distribution shift between splits is bounded; and the decision threshold is tuned on a separate calibration split rather than on the held-out test day. That last point is the methodological fix. Tuning the threshold on the test day, which the standard version effectively does, leaks validation information and inflates the score. |
| `RF_standard_model_per_nurse.py` | The legacy baseline it is compared against. Fixed 0.5 threshold, looser fold filtering. Imports everything from `RF_idealized_model_per_nurse`. |
| `SHAP_importance_RF_model_per_nurse.py` | SHAP values over the idealized folds. Also imports from `RF_idealized_model_per_nurse`. This is where the paper's claim about EDA mattering most comes from. |
| `LSTM_nurse_stress_LODO.py` | Per-nurse LSTM under strict Leave-One-Day-Out. Discards any day whose stress ratio falls outside 20–80% so that folds contain both classes. Handles imbalance twice over: a `WeightedRandomSampler` for batch composition, plus `CrossEntropyLoss` with inverse-frequency weights and a 2× boost on the minority no-stress class, capped at 20. |
| `lstm_nurse_stress_loo.py` | A leave-one-out variant of the same pipeline. |
| `Mlp_per_nurse.ipynb` | Per-nurse MLP on point-wise samples with day-grouped splits. **Written for Google Colab**: it calls `drive.mount('/content/drive')`. That cell must be removed to run locally. |

Because the two RF scripts and the SHAP script use plain `from RF_idealized_model_per_nurse import ...`, you must run them with `models/per nurse/` on the import path:

```powershell
cd "reference\MLMAMidtermProject\models\per nurse"
python RF_idealized_model_per_nurse.py --data-dir "..\..\data\Aditya"
```

---

## 6. Walkthrough of `1d_cnn.ipynb`

The notebook has 18 cells. Here is what each block does and why.

**Cells 3, 5 — imports and hyperparameters**

```python
FS = 32                          # assumed sampling rate, Hz
WINDOW_SEC = 60                  # each example covers 60 seconds
STEP_SEC   = 30                  # windows overlap by 50%
WINDOW_SIZE = FS * WINDOW_SEC    # = 1920 timesteps
STEP_SIZE   = FS * STEP_SEC      # = 960 timesteps
FEATURES = ['acc_mag','EDA','HR','TEMP','EDA_slope','HR_slope','acc_burst']
```

Note the inconsistency: preprocessing built `time` at 0.03 s per sample, which is 33.3 Hz, while this cell assumes 32 Hz. Each "60 second" window is really about 57.6 seconds. Harmless for the model, worth knowing when you describe the method.

**Cell 7 — `load_and_normalize_data`**

Four things happen per nurse, in order:

1. Nurses `CE` and `EG` are excluded. The paper explains why: neither has both classes represented, so they cannot be scored.
2. `label` is binarized to `(label > 0)`. Medium and high stress collapse into one positive class.
3. Three features are engineered on top of the four raw channels:
   - `EDA_slope = EDA.diff(periods=160)` — change in skin conductance over the previous 160 samples, roughly 5 seconds.
   - `HR_slope` — same for heart rate.
   - `acc_burst = acc_mag.rolling(160).std()` — short-window movement volatility, a proxy for bursts of physical activity.

   These carry the physiological reasoning of the whole project. Acute stress shows up as a *rise* in EDA and HR, not as a particular absolute level, so the derivative is the informative quantity.
4. `StandardScaler` is fit **per nurse**, on that nurse's own data. This is the subject-wise z-scoring that makes cross-subject pooling even nominally possible.

One item to verify yourself: if any `label` values are `NaN`, then `(NaN > 0)` evaluates to `False` and those rows silently become no-stress. Run `df['label'].isna().sum()` on one nurse before you trust the label counts.

**Cell 9 — `create_windows`**

Slides a 1920-sample window forward in 960-sample steps, producing a 3-D tensor of shape `(n_windows, 1920, 7)`. Each window's label is the majority vote of its 1920 sample labels. Because indexing is by row position rather than by `datetime`, windows can cross day boundaries; see §4.1.

**Cell 11 — `moderate_undersample`**

Present but unused. The calls are commented out in cell 15. The notebook handles imbalance with class weights instead.

**Cell 13 — `build_1d_cnn`**

```
Input(1920, 7)
Conv1D(32, kernel=5, relu) → BatchNorm → MaxPool(2)
Conv1D(64, kernel=3, relu) → BatchNorm
GlobalAveragePooling1D()
Dense(64, relu, name='dense_1') → Dropout(0.4)
Dense(1, sigmoid, name='output_layer')
```

The convolutional layers learn local temporal motifs, short shapes in the signal lasting a fraction of a second. `GlobalAveragePooling1D` then averages across all 1920 positions, which makes the representation invariant to *where* in the minute the motif occurred and keeps the parameter count small. Loss is `binary_crossentropy`; a commented-out `BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)` shows an imbalance strategy they tried and dropped.

**Cell 15 — `run_loso_pipeline`**

This is the part worth reading closely, because it is not a plain LOSO loop. For each of the 13 nurses:

1. Build training windows from the other 12 nurses and concatenate them.
2. Compute balanced class weights, then train the CNN for 10 epochs at batch size 64.
3. Split the held-out nurse chronologically: first 20% for fine-tuning, remaining 80% for evaluation.
4. **Freeze every `Conv1D` layer** and recompile with Adam at `lr=1e-4`.
5. Fine-tune the unfrozen dense head on the held-out nurse's first 20%, but only if that slice contains at least 5 examples of each class.
6. Evaluate on the final 80%.

Conceptually: the convolutional stack learns *what a stress signature looks like in general* and is then locked; the dense head is re-fit to *what stress looks like in this particular person*. The chronological split is what makes this legitimate. Fine-tuning on a random 20% would let the model see the future.

**Cell 17 — execution**

```python
subjects_data = load_and_normalize_data("/content/MLMAMidtermProject/data/Aditya")
```

A Colab path. This is the single change required to run locally, though see §7 for why you should also shrink the workload.

---

## 7. Running it locally

Full LOSO across 13 nurses is roughly 10,600 windows of 1920 × 7. In float64 that is about 1.1 GB for one copy, and the loop holds a list plus a concatenated array simultaneously, so peak usage lands around 4 to 6 GB. Thirteen folds at 10 epochs each on a CPU is plausibly 10 to 15 hours.

`my_work\cnn1d_global.py` is an annotated rewrite that keeps the logic identical and makes three changes so you can iterate:

1. A `NURSES` list at the top, defaulting to three nurses instead of thirteen.
2. Windows built in `float32` into a preallocated array, which halves memory and removes the list-of-arrays spike.
3. `EPOCHS` and `MAX_WINDOWS_PER_NURSE` as top-level constants.

Run it with the defaults first. It should finish in a couple of minutes and tell you whether your environment, data paths, and LFS pull are all correct. Once that works, raise the numbers.

```powershell
cd C:\Users\Elvis\Documents\Nurse Stress
.\.venv\Scripts\Activate.ps1
python my_work\cnn1d_global.py
```

---

## 8. Known issues to expect

| Symptom | Cause | Fix |
|---|---|---|
| `ParserError` or a one-row DataFrame from `read_csv` | The CSV is still a 133-byte LFS pointer | `git lfs pull --include="data/Aditya/*"` |
| `FileNotFoundError: /content/...` | Colab path hardcoded in `1d_cnn.ipynb` | Repoint to a local path |
| `ModuleNotFoundError: google.colab` | `Mlp_per_nurse.ipynb` mounts Drive | Delete that cell |
| `ModuleNotFoundError: RF_idealized_model_per_nurse` | Sibling import in the per-nurse RF scripts | `cd` into `models\per nurse` before running |
| `MemoryError` in `create_windows` | Full LOSO in float64 | Use `my_work\cnn1d_global.py`, or reduce `NURSES` |
| No GPU detected | TF ≥ 2.11 dropped native Windows GPU support | Accept CPU, or move to WSL2 |
| `data/Eric` files fail to load | Never pulled from LFS | Skip it; it duplicates `data/Aditya` |

---

## 9. Suggested reading order

1. `preprocessing/data_eric.ipynb` — see the data get built. Three cells.
2. `models/global/1d_cnn.ipynb` alongside §6 above.
3. `outputs/RF_idealized_model_per_nurse/RF_idealized_model_per_nurse_results_summary.md` — the result the paper actually defends.
4. `models/per nurse/RF_idealized_model_per_nurse.py` — the fold-filtering and calibration-split logic. This is the most methodologically careful code in the project and the part most worth internalizing.
