"""PHASE 2 - RATIO sensitivity, and the drop-8B variant.

Option 1: run the falsification battery at both defensible ratios and sort the
          findings into "survives the perturbation" and "doesn't".
Option 4: 8B is the subject forcing the ceiling down (its top activity bin has
          13 positives and 7 candidates) AND it already fails the >=100
          negative-count check. Removing it raises the achievable ratio for
          everyone else. Costs 13 events and one fold.

RATIO is a parameter, so this is a sensitivity analysis, not an ablation.
Nothing is removed from the model.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, LOCAL_TZ, load_survey, banner

WINDOW_S, N_BINS, SEEDS = 120, 10, [0, 1, 2]
FEATS = {
    "ACC-only": ["acc_mean", "acc_sd"],
    "Time-only": ["hour"],
    "EDA-only": ["eda_med", "eda_floor"],
    "Full": ["acc_mean", "acc_sd", "eda_med", "eda_floor", "hour", "min_into_session"],
}


def build(W, ratio, seed, drop=()):
    """Draw negatives at `ratio`, matched within subject and activity bin."""
    W = W[~W.subject.isin(drop)].copy()
    POS = W[W.label == 1]
    edges = POS.acc_mean.quantile(np.linspace(0, 1, N_BINS + 1)).values.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    W["abin"] = pd.cut(W.acc_mean, edges, labels=False)
    POS, CAND = W[W.label == 1], W[W.eligible_neg]
    picks = []
    for sub, gp in POS.groupby("subject"):
        want = (gp.abin.value_counts() * ratio).round().astype(int)
        pool = CAND[CAND.subject == sub]
        for b, n in want.items():
            avail = pool[pool.abin == b]
            take = min(int(n), len(avail))
            if take:
                picks.append(avail.sample(take, random_state=seed))
    NEG = pd.concat(picks) if picks else CAND.iloc[:0]
    return pd.concat([POS.assign(label=1), NEG.assign(label=0)])


def ceiling(W, drop=()):
    """Highest ratio every retained subject could in principle reach."""
    W = W[~W.subject.isin(drop)].copy()
    POS = W[W.label == 1]
    edges = POS.acc_mean.quantile(np.linspace(0, 1, N_BINS + 1)).values.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    W["abin"] = pd.cut(W.acc_mean, edges, labels=False)
    POS, CAND = W[W.label == 1], W[W.eligible_neg]
    out = {}
    for sub, gp in POS.groupby("subject"):
        pool = CAND[CAND.subject == sub]
        pb, ab = gp.abin.value_counts(), pool.abin.value_counts()
        out[sub] = sum(int(ab.get(b, 0)) for b in pb.index) / len(gp)
    return pd.Series(out)


def evaluate(D):
    D = D.copy()
    D["hour"] = D.start_utc.dt.tz_convert(LOCAL_TZ).dt.hour
    D["min_into_session"] = D.w * 2.0
    y, g = D.label.values.astype(int), D.subject.values
    out = {}
    for name, cols in FEATS.items():
        X = D[cols].to_numpy(dtype=float)
        oof = np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=.1,
                                               random_state=0, class_weight="balanced")
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        ok = np.isfinite(oof)
        out[name] = roc_auc_score(y[ok], oof[ok])
    return out


def main():
    banner("PHASE 2 - RATIO SENSITIVITY (option 1) AND DROP-8B (option 4)")

    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True)
    A = pd.read_parquet(os.path.join(OUT, "a8_windows_eligible.parquet"))
    W = W.merge(A[["subject", "session", "w", "eligible_neg"]],
                on=["subject", "session", "w"], how="left")
    W["eligible_neg"] = W.eligible_neg.fillna(False) & (W.label == 0)

    banner("Achievable ceiling, with and without 8B")
    c_all, c_no8b = ceiling(W), ceiling(W, drop=("8B",))
    C = pd.DataFrame({"all 10": c_all.round(2), "without 8B": c_no8b.round(2)})
    print()
    print(C.to_string())
    print(f"\n  uniform ceiling, all 10 subjects : {c_all.min():.2f}  (set by {c_all.idxmin()})")
    print(f"  uniform ceiling, without 8B      : {c_no8b.min():.2f}  (set by {c_no8b.idxmin()})")

    ARMS = [
        ("all 10, r=1.31", 1.31, ()),
        ("all 10, r=2.14", 2.14, ()),
        ("drop 8B, r=1.31", 1.31, ("8B",)),
        ("drop 8B, r=2.25", 2.25, ("8B",)),
    ]

    banner("Falsification battery, every arm x 3 seeds")
    rows, comp = [], []
    for label, ratio, drop in ARMS:
        per_seed = {k: [] for k in FEATS}
        counts = None
        for seed in SEEDS:
            D = build(W, ratio, seed, drop)
            if counts is None:
                counts = (int((D.label == 1).sum()), int((D.label == 0).sum()),
                          D.subject.nunique(),
                          D[D.label == 0].groupby("subject").size().min(),
                          (D[D.label == 0].groupby("subject").size() /
                           D[D.label == 1].groupby("subject").size()))
            for k, v in evaluate(D).items():
                per_seed[k].append(v)
        r = {"arm": label, "subj": counts[2], "pos": counts[0], "neg": counts[1],
             "min neg/subj": int(counts[3]),
             "balance spread": f"{counts[4].min():.2f}-{counts[4].max():.2f}",
             "spread x": round(counts[4].max() / counts[4].min(), 2)}
        for k in FEATS:
            r[k] = round(float(np.mean(per_seed[k])), 4)
            r[k + " rng"] = round(float(np.ptp(per_seed[k])), 4)
        rows.append(r)
        comp.append({"arm": label,
                     "EDA-ACC": [round(per_seed["EDA-only"][i] - per_seed["ACC-only"][i], 4)
                                 for i in range(3)],
                     "EDA-Full": [round(per_seed["EDA-only"][i] - per_seed["Full"][i], 4)
                                  for i in range(3)]})
    R = pd.DataFrame(rows).set_index("arm")

    print("\n--- cohort and balance ---")
    print(R[["subj", "pos", "neg", "min neg/subj", "balance spread", "spread x"]].to_string())
    print("\n--- AUC, mean of 3 seeds (rng = seed-to-seed range) ---")
    print(R[[c for c in R.columns if c in FEATS or c.endswith(" rng")]].to_string())

    banner("Which findings survive the perturbation?")
    print("\nA finding survives if the effect exceeds the seed range in EVERY arm.\n")
    for c in comp:
        eda_acc, eda_full = c["EDA-ACC"], c["EDA-Full"]
        rng_acc = R.loc[c["arm"], "ACC-only rng"]
        rng_full = R.loc[c["arm"], "Full rng"]
        print(f"  {c['arm']}")
        print(f"    EDA-only > ACC-only : {eda_acc}   vs seed rng {rng_acc:.4f}  "
              f"-> {'SURVIVES' if min(eda_acc) > rng_acc else 'inside noise'}")
        print(f"    EDA-only > Full     : {eda_full}   vs seed rng {rng_full:.4f}  "
              f"-> {'SURVIVES' if min(eda_full) > rng_full else 'INSIDE NOISE'}")

    print("\n--- falsification pass check (chance = 0.500) ---")
    for a in R.index:
        acc, tod = R.loc[a, "ACC-only"], R.loc[a, "Time-only"]
        print(f"  {a:18} ACC-only {acc:.4f}  Time-only {tod:.4f}   "
              f"{'PASS' if abs(acc-.5) < .05 and abs(tod-.5) < .05 else 'CHECK'}")

    R.to_csv(os.path.join(OUT, "phase2_ratio_sensitivity.csv"))
    print(f"\nwrote {os.path.join(OUT, 'phase2_ratio_sensitivity.csv')}")


if __name__ == "__main__":
    main()
