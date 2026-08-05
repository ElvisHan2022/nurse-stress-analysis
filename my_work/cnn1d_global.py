"""
Global 1D CNN for nurse stress detection - local, annotated rewrite.

This is a faithful port of reference/MLMAMidtermProject/models/global/1d_cnn.ipynb
with three practical changes so it runs on a laptop:

  1. NURSES is a short list by default (3 nurses, not 13), so a full
     leave-one-subject-out sweep finishes in minutes rather than hours.
  2. Windows are built as float32 into a preallocated array. The original
     appended to a Python list and called np.array() at the end, which
     briefly holds two full copies in float64.
  3. EPOCHS and MAX_WINDOWS_PER_NURSE are top-level knobs.

The modelling logic (features, normalization, architecture, freeze-and-
fine-tune protocol) is unchanged. Read this alongside section 6 of
PROJECT_GUIDE.md.

Run:
    cd C:\\Users\\Elvis\\Documents\\Nurse Stress
    .\\.venv\\Scripts\\Activate.ps1
    python my_work\\cnn1d_global.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential

# ============================================================================
# CONFIGURATION
# ============================================================================

# Where the per-nurse CSVs live. Adjust if you cloned somewhere else.
DATA_DIR = Path(__file__).resolve().parents[1] / "reference" / "MLMAMidtermProject" / "data" / "Aditya"

# Start small. Add IDs as you pull more data with `git lfs pull`.
# The full set, minus the three the paper drops:
#   15 5C 6B 7A 7E 83 8B 94 BG DF E4 F5
# Dropped upstream: CE and EG (only one class present), 6D (single day only).
NURSES = ["15", "7E", "8B"]

# Sampling. Preprocessing stamped `time` at 0.03 s per row, i.e. 33.3 Hz.
# The original notebook assumes 32 Hz, so a "60 s" window is really ~57.6 s.
# Kept at 32 for parity with the reference implementation.
FS = 32
WINDOW_SEC = 60
STEP_SEC = 30

WINDOW_SIZE = FS * WINDOW_SEC   # 1920 timesteps per example
STEP_SIZE = FS * STEP_SEC       # 960 timesteps between window starts (50% overlap)

# Four raw channels plus three engineered ones. See build_features().
FEATURES = ["acc_mag", "EDA", "HR", "TEMP", "EDA_slope", "HR_slope", "acc_burst"]
TARGET = "label"

# Derivative/volatility lookback: 160 samples ~= 5 seconds at 32 Hz.
SLOPE_LAG = 160

EPOCHS = 3                      # reference uses 10
BATCH_SIZE = 64
FINETUNE_FRAC = 0.20            # chronological head of the held-out nurse
MIN_FINETUNE_PER_CLASS = 5      # skip fine-tuning if the head is single-class

# Cap windows per nurse so a first run stays fast. Set to None for all of them.
MAX_WINDOWS_PER_NURSE = 400

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================================
# 1. LOADING, FEATURE ENGINEERING, PER-SUBJECT NORMALIZATION
# ============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three engineered channels and binarize the label.

    Why these three:
      EDA_slope / HR_slope  Acute stress shows up as a *rise* in skin
                            conductance and heart rate, not as a particular
                            absolute level. The first difference over ~5 s
                            captures that rise directly.
      acc_burst             Rolling standard deviation of movement magnitude.
                            Physical exertion also raises HR and EDA, so this
                            gives the model a channel that lets it tell
                            "running down a corridor" apart from "stressed".
    """
    df = df.copy()

    # 0 = no stress, 1 = medium, 2 = high  ->  0 = no stress, 1 = stress.
    df[TARGET] = (df[TARGET] > 0).astype(np.int8)

    df["EDA_slope"] = df["EDA"].diff(periods=SLOPE_LAG).fillna(0)
    df["HR_slope"] = df["HR"].diff(periods=SLOPE_LAG).fillna(0)
    df["acc_burst"] = (
        df["acc_mag"].rolling(window=SLOPE_LAG, min_periods=1).std().fillna(0)
    )
    return df


def load_and_normalize(data_dir: Path, nurses: list[str]) -> dict[str, pd.DataFrame]:
    """Load each nurse's CSV and z-score it against *its own* statistics.

    This per-subject scaling is the crux of the whole global-model idea.
    One nurse's resting heart rate can equal another's high-stress heart rate,
    so raw values are not comparable across people. Centring each nurse on
    their own mean turns the features into "how far from your normal is this",
    which is at least in principle comparable.

    The paper's finding is that even this is not enough, and that per-nurse
    models outperform the global ones. Worth holding in mind while reading.
    """
    subjects: dict[str, pd.DataFrame] = {}

    for nurse_id in nurses:
        path = data_dir / f"processed_nurse_{nurse_id}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Pull it with:\n"
                f'  git lfs pull --include="data/Aditya/processed_nurse_{nurse_id}.csv"'
            )

        # A Git LFS pointer is ~133 bytes and will parse as a nonsense DataFrame.
        if path.stat().st_size < 10_000:
            raise RuntimeError(
                f"{path.name} is only {path.stat().st_size} bytes, which means it is "
                f"still an LFS pointer rather than real data. Run:\n"
                f'  git lfs pull --include="data/Aditya/*"'
            )

        df = pd.read_csv(path)

        n_missing = int(df[TARGET].isna().sum())
        if n_missing:
            # (NaN > 0) evaluates to False, so unlabelled rows would silently
            # become "no stress". Flag it rather than hide it.
            print(f"  [warn] nurse {nurse_id}: {n_missing:,} rows have no label")

        df = build_features(df)

        scaler = StandardScaler()
        df[FEATURES] = scaler.fit_transform(df[FEATURES]).astype(np.float32)

        subjects[nurse_id] = df
        stress_rate = float(df[TARGET].mean())
        print(f"  nurse {nurse_id}: {len(df):,} rows, {stress_rate:.1%} stress")

    return subjects


# ============================================================================
# 2. SLIDING WINDOWS
# ============================================================================

def create_windows(df: pd.DataFrame, max_windows: int | None = None):
    """Turn a continuous recording into a (n_windows, 1920, 7) tensor.

    Each window's label is the majority vote over its 1920 sample labels,
    which smooths the noisy boundary where a self-report transitions.

    Caveat worth knowing: indexing is by row position, not by `datetime`.
    Because preprocessing stamped a synthetic continuous time axis, a window
    can straddle a multi-hour gap between two shifts. The per-nurse scripts in
    the reference repo avoid this by grouping on the calendar date first.
    """
    data = df[FEATURES].to_numpy(dtype=np.float32)
    labels = df[TARGET].to_numpy(dtype=np.int8)

    starts = list(range(0, len(data) - WINDOW_SIZE, STEP_SIZE))
    if max_windows is not None and len(starts) > max_windows:
        # Evenly spaced subsample keeps coverage across the whole recording,
        # rather than truncating to the first N windows.
        idx = np.linspace(0, len(starts) - 1, max_windows).astype(int)
        starts = [starts[i] for i in idx]

    n = len(starts)
    if n == 0:
        return (
            np.empty((0, WINDOW_SIZE, len(FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
        )

    # Preallocate. The original appended to a list then called np.array(),
    # which peaks at roughly twice the memory.
    X = np.empty((n, WINDOW_SIZE, len(FEATURES)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)

    for i, start in enumerate(starts):
        end = start + WINDOW_SIZE
        X[i] = data[start:end]
        y[i] = 1 if labels[start:end].mean() >= 0.5 else 0

    return X, y


# ============================================================================
# 3. ARCHITECTURE
# ============================================================================

def build_1d_cnn(input_shape: tuple[int, int]) -> Sequential:
    """Two convolutional blocks, global average pooling, small dense head.

    Conv1D learns short local motifs in the signal. GlobalAveragePooling1D
    then averages over all 1920 positions, which makes the representation
    invariant to *where* in the minute a motif occurred and keeps the
    parameter count low enough for ~10k training examples.

    Named layers matter: run_loso() freezes by layer type, and the dense head
    is what gets fine-tuned per subject.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=5, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu", name="dense_1"),
        Dropout(0.4),
        Dense(1, activation="sigmoid", name="output_layer"),
    ])

    # The reference also tried focal loss and left it commented out:
    #   tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ============================================================================
# 4. LEAVE-ONE-SUBJECT-OUT WITH TRANSFER LEARNING
# ============================================================================

def run_loso(subjects: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Train globally, then adapt the head to each held-out nurse.

    Per fold:
      1. Pool windows from every other nurse and train the full CNN.
      2. Split the held-out nurse chronologically into a 20% head and an
         80% tail.
      3. Freeze the Conv1D layers and re-fit only the dense head on that 20%.
      4. Score on the untouched 80%.

    The intuition: convolutional filters learn what a stress signature looks
    like in general and are then locked; the dense head learns what stress
    looks like in this specific person. The split must be chronological, not
    random, or the model gets to see the future.
    """
    subject_ids = list(subjects.keys())
    rows = []

    for test_subject in subject_ids:
        tf.keras.backend.clear_session()
        print(f"\n{'=' * 64}\nFold: holding out nurse {test_subject}\n{'=' * 64}")

        # --- 1. pooled training set from the other nurses -------------------
        X_parts, y_parts = [], []
        for subj in subject_ids:
            if subj == test_subject:
                continue
            Xs, ys = create_windows(subjects[subj], MAX_WINDOWS_PER_NURSE)
            X_parts.append(Xs)
            y_parts.append(ys)

        X_train = np.concatenate(X_parts)
        y_train = np.concatenate(y_parts)
        del X_parts, y_parts

        if len(np.unique(y_train)) < 2:
            print("  skipped: training pool has only one class")
            continue

        # Inverse-frequency weighting. The dataset is roughly 80% stress, so
        # an unweighted model can score ~0.80 accuracy by always predicting 1.
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {0: weights[0], 1: weights[1]}
        print(f"  train windows: {X_train.shape}, class weights: "
              f"{ {k: round(v, 3) for k, v in class_weights.items()} }")

        model = build_1d_cnn(input_shape=(WINDOW_SIZE, len(FEATURES)))
        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            verbose=1,
        )
        del X_train, y_train

        # --- 2. chronological split of the held-out nurse -------------------
        X_test, y_test = create_windows(subjects[test_subject], MAX_WINDOWS_PER_NURSE)
        split_idx = int(len(X_test) * FINETUNE_FRAC)
        X_ft, y_ft = X_test[:split_idx], y_test[:split_idx]
        X_eval, y_eval = X_test[split_idx:], y_test[split_idx:]

        # --- 3. freeze convolutions, fine-tune the head ---------------------
        for layer in model.layers:
            if isinstance(layer, Conv1D):
                layer.trainable = False

        # Low learning rate: the head is being nudged, not retrained.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
        )

        has_both = (
            len(y_ft) > 0
            and int(np.sum(y_ft == 0)) >= MIN_FINETUNE_PER_CLASS
            and int(np.sum(y_ft == 1)) >= MIN_FINETUNE_PER_CLASS
        )
        if has_both:
            ft_w = compute_class_weight(
                class_weight="balanced", classes=np.unique(y_ft), y=y_ft
            )
            print(f"  fine-tuning head on {len(X_ft)} windows")
            model.fit(
                X_ft, y_ft,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                class_weight={0: ft_w[0], 1: ft_w[1]},
                verbose=0,
            )
        else:
            print("  fine-tuning skipped: head slice lacks both classes")

        # --- 4. evaluate ----------------------------------------------------
        if len(y_eval) == 0 or len(np.unique(y_eval)) < 2:
            print("  eval skipped: held-out tail has only one class")
            continue

        prob = model.predict(X_eval, verbose=0).ravel()
        pred = (prob >= 0.5).astype(np.int8)

        precision, recall, _ = precision_recall_curve(y_eval, prob)
        rows.append({
            "nurse": test_subject,
            "n_eval": len(y_eval),
            "stress_rate": round(float(y_eval.mean()), 3),
            "accuracy": round(accuracy_score(y_eval, pred), 4),
            "f1_macro": round(f1_score(y_eval, pred, average="macro"), 4),
            "f1_stress": round(f1_score(y_eval, pred, pos_label=1, zero_division=0), 4),
            "pr_auc": round(auc(recall, precision), 4),
        })
        print(f"  {rows[-1]}")
        print(f"  confusion matrix:\n{confusion_matrix(y_eval, pred)}")

        del X_test, y_test, X_ft, y_ft, X_eval, y_eval

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs visible: {tf.config.list_physical_devices('GPU')}")
    print(f"Data directory: {DATA_DIR}\n")

    print("Loading and normalizing...")
    subjects = load_and_normalize(DATA_DIR, NURSES)

    results = run_loso(subjects)

    print(f"\n{'=' * 64}\nPER-NURSE RESULTS\n{'=' * 64}")
    if results.empty:
        print("No fold produced a scorable result. Add more nurses to NURSES.")
        return

    print(results.to_string(index=False))
    print("\nMean across folds:")
    print(results[["accuracy", "f1_macro", "f1_stress", "pr_auc"]].mean().round(4).to_string())

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cnn1d_global_loso.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
