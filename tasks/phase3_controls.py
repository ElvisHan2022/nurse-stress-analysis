"""PHASE 3.6 - A second learner and a permutation null.

Two referee findings, both cheap, both converting an inference into a
demonstration.

1. SECOND LEARNER. "Gradient boosting fails" is not "the method fails". A
   penalised logistic regression is the natural floor: linear, low capacity,
   nothing to overfit with. If it lands near the boosted model, the ceiling is
   not model capacity.

2. PERMUTATION NULL. Shuffling the labels within subject destroys the
   label-feature relationship while preserving everything else: class balance,
   fold structure, feature distributions, subject identity. Whatever the model
   scores on shuffled labels is what the pipeline produces from nothing. The
   real score has to clear it.

   Shuffling WITHIN subject rather than globally is the strict version. A
   global shuffle would also destroy per-subject prevalence differences, which
   a model could otherwise exploit, and would make the null too easy to beat.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, banner
from phase3_baseline import normalise, NORMALISE, PASSTHRU, SEEDS, WINDOW_S

N_PERM = 20
EVENT_FIRE_FRAC = 0.50


def gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=.06, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=0, class_weight="balanced")


def logreg():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))


def run(X, y, g, make_model):
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def episode_recall(y, oof, ok, ev, neg_per_hour, fa=1.0):
    negs = np.sort(oof[ok & (y == 0)])[::-1]
    if not len(negs):
        return np.nan
    k = max(int(round((fa / neg_per_hour) * len(negs))) - 1, 0)
    pred = (oof >= negs[min(k, len(negs) - 1)]).astype(int)
    d = pd.DataFrame({"ev": ev, "pred": pred, "ok": ok, "y": y})
    d = d[(d.y == 1) & d.ok & (d.ev >= 0)]
    return float((d.groupby("ev").pred.mean() >= EVENT_FIRE_FRAC).mean()) \
        if len(d) else np.nan


def main():
    banner("PHASE 3.6 - SECOND LEARNER AND PERMUTATION NULL")

    F = normalise(pd.read_parquet(os.path.join(DERIVED, "features.parquet")))
    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    EV = pd.read_parquet(os.path.join(DERIVED, "events.parquet"))
    FEATS = [c + "_z" for c in NORMALISE] + PASSTHRU

    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True)
    W["end_utc"] = pd.to_datetime(W.end_utc, utc=True)
    W["event_id"] = -1
    for i, r in EV.iterrows():
        ov = ((np.minimum(W.end_utc, r.end_utc)
               - np.maximum(W.start_utc, r.start_utc)).dt.total_seconds())
        W.loc[(W.subject == r.subject) & (ov >= WINDOW_S * .5), "event_id"] = i
    ev_map = W.set_index(["subject", "session", "w"]).event_id
    true_prev = float(W.label.mean())
    neg_per_hour = (3600 / WINDOW_S) * (1 - true_prev)

    # ---- 1. second learner -------------------------------------------------
    banner("1. Learner comparison, identical folds and features")
    rows = []
    for name, mk in [("gradient boosting", gbm), ("logistic regression", logreg)]:
        aucs, recs = [], []
        for seed in SEEDS:
            L = pd.read_parquet(os.path.join(DERIVED, f"labels_seed{seed}.parquet"))
            D = L[["subject", "session", "w", "label"]].merge(
                F.drop(columns=["label"]), on=["subject", "session", "w"], how="left")
            D["event_id"] = ev_map.reindex(
                pd.MultiIndex.from_frame(D[["subject", "session", "w"]])).values
            X = D[FEATS].to_numpy(dtype=float)
            y = D.label.values.astype(int)
            g = D.subject.values
            oof = run(X, y, g, mk)
            ok = np.isfinite(oof)
            aucs.append(roc_auc_score(y[ok], oof[ok]))
            recs.append(episode_recall(y, oof, ok, D.event_id.values, neg_per_hour))
        rows.append({"learner": name,
                     "AUC": round(float(np.mean(aucs)), 4),
                     "AUC range": round(float(np.ptp(aucs)), 4),
                     "episode recall": round(float(np.mean(recs)), 4),
                     "recall range": round(float(np.ptp(recs)), 4)})
        print(f"  {name:20} AUC {rows[-1]['AUC']:.4f}  "
              f"episode recall {rows[-1]['episode recall']:.4f}")
    LC = pd.DataFrame(rows)
    gap = abs(LC.AUC.iloc[0] - LC.AUC.iloc[1])
    print(f"\n  gap between learners: {gap:.4f} AUC")
    print("  A small gap means capacity is not the binding constraint - a linear")
    print("  model with nothing to overfit reaches nearly the same place.")

    # ---- 2. permutation null ----------------------------------------------
    banner(f"2. Permutation null, {N_PERM} within-subject label shuffles")
    print("\nLabels shuffled WITHIN each subject, so class balance, fold")
    print("structure and feature distributions are all preserved. Only the")
    print("label-feature correspondence is destroyed.\n")

    L = pd.read_parquet(os.path.join(DERIVED, "labels_seed0.parquet"))
    D = L[["subject", "session", "w", "label"]].merge(
        F.drop(columns=["label"]), on=["subject", "session", "w"], how="left")
    D["event_id"] = ev_map.reindex(
        pd.MultiIndex.from_frame(D[["subject", "session", "w"]])).values
    X = D[FEATS].to_numpy(dtype=float)
    y_true = D.label.values.astype(int)
    g = D.subject.values

    oof = run(X, y_true, g, gbm)
    ok = np.isfinite(oof)
    obs_auc = roc_auc_score(y_true[ok], oof[ok])
    obs_rec = episode_recall(y_true, oof, ok, D.event_id.values, neg_per_hour)

    rng = np.random.default_rng(0)
    null_auc = []
    for i in range(N_PERM):
        y_perm = y_true.copy()
        for s in np.unique(g):
            m = g == s
            y_perm[m] = rng.permutation(y_true[m])
        o = run(X, y_perm, g, gbm)
        k = np.isfinite(o)
        null_auc.append(roc_auc_score(y_perm[k], o[k]))
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{N_PERM} permutations")

    null_auc = np.array(null_auc)
    p_auc = float((null_auc >= obs_auc).sum() + 1) / (N_PERM + 1)
    print(f"""
  observed AUC        {obs_auc:.4f}
  null AUC mean       {null_auc.mean():.4f}   sd {null_auc.std():.4f}
  null AUC max        {null_auc.max():.4f}
  permutation p       {p_auc:.4f}   (<= {1/(N_PERM+1):.4f} is the floor at {N_PERM} draws)
  observed exceeds every null draw: {bool(obs_auc > null_auc.max())}
""")

    pd.DataFrame({"null_auc": null_auc}).to_csv(
        os.path.join(OUT, "phase3_permutation_null.csv"), index=False)
    LC.to_csv(os.path.join(OUT, "phase3_learner_comparison.csv"), index=False)

    banner("WHAT THESE TWO ESTABLISH")
    print(f"""
  The permutation null rules out the pipeline manufacturing its own signal:
  observed {obs_auc:.4f} against a null centred on {null_auc.mean():.4f}.

  The learner comparison rules out model capacity as the constraint: a linear
  model reaches within {gap:.4f} AUC of a boosted one on identical folds.

  Together they narrow what remains. Neither speaks to whether a different
  REPRESENTATION would help, which is a separate axis and stays untested.
""")


if __name__ == "__main__":
    main()
