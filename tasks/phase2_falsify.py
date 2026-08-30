"""PHASE 2.3 - Falsification checks. Models we WANT to fail.

These are not ablations in the "remove an ingredient" sense. Each one is a
deliberately crippled model built from a single suspicious signal. If it does
well, the negative construction leaked and the headline result would be
measuring the wrong thing.

  ACC-only        did we build a pedometer?
  Time-of-day     did we build a shift-schedule detector?
  Subject-ID      how much subject identity is sitting in the features?
  Neg count       does every subject have enough negatives to score?

Same learner as the planned baseline (gradient boosting) so a poor result means
"no signal here", not "I used a weaker model".
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import DERIVED, OUT, LOCAL_TZ, banner

SEED = 0
MIN_NEG_PER_SUBJECT = 100


def loso_auc(X, y, groups):
    """Leave-one-subject-out, pooled AUC plus per-fold."""
    oof = np.full(len(y), np.nan)
    per = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.1, random_state=0,
            class_weight="balanced")
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        per.append({"subject": groups[te][0], "n": int(te.size),
                    "pos": int(y[te].sum()), "auc": round(roc_auc_score(y[te], p), 3)})
    ok = np.isfinite(oof)
    return roc_auc_score(y[ok], oof[ok]), pd.DataFrame(per)


def main():
    banner("PHASE 2.3 - FALSIFICATION CHECKS")

    L = pd.read_parquet(os.path.join(DERIVED, f"labels_seed{SEED}.parquet"))
    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True).astype("datetime64[ns, UTC]")
    D = L.merge(W.drop(columns=["label"]), on=["subject", "session", "w"], how="left")

    D["hour"] = D.start_utc.dt.tz_convert(LOCAL_TZ).dt.hour
    D["min_into_session"] = D.w * 2.0
    y = D.label.values.astype(int)
    g = D.subject.values
    print(f"\n{len(D):,} windows | {y.sum():,} positive | "
          f"{D.subject.nunique()} subjects")

    # ---- seeds really do differ -------------------------------------------
    banner("Sanity: do the seeds draw different windows?")
    keys = []
    for s in [0, 1, 2]:
        d = pd.read_parquet(os.path.join(DERIVED, f"labels_seed{s}.parquet"))
        keys.append(set(map(tuple, d[d.label == 0][["subject", "session", "w"]].values)))
    print(f"\n  negatives per seed: {[len(k) for k in keys]}")
    print(f"  shared by all three: {len(keys[0] & keys[1] & keys[2]):,}")
    print(f"  unique to seed 0:    {len(keys[0] - keys[1] - keys[2]):,}")
    print("\n  Counts are identical by construction (min(want, available)); it is")
    print("  WHICH windows get drawn that varies. The spread shows up in the")
    print("  model score, not in the table sizes.")

    FEATS = {
        "ACC-only": ["acc_mean", "acc_sd"],
        "Time-of-day-only": ["hour"],
        "Time + session position": ["hour", "min_into_session"],
        "EDA-only": ["eda_med", "eda_floor"],
        "Full (minimal)": ["acc_mean", "acc_sd", "eda_med", "eda_floor",
                           "hour", "min_into_session"],
    }

    banner("Stress detection from each feature set (LOSO)")
    rows = []
    per_fold = {}
    for name, cols in FEATS.items():
        X = D[cols].to_numpy(dtype=float)
        auc, per = loso_auc(X, y, g)
        rows.append({"feature set": name, "n_features": len(cols),
                     "LOSO AUC": round(auc, 4)})
        per_fold[name] = per
        print(f"  {name:26} AUC {auc:.4f}")

    R = pd.DataFrame(rows)
    full = R.loc[R["feature set"] == "Full (minimal)", "LOSO AUC"].iloc[0]

    banner("Pass conditions")
    acc = R.loc[R["feature set"] == "ACC-only", "LOSO AUC"].iloc[0]
    tod = R.loc[R["feature set"] == "Time-of-day-only", "LOSO AUC"].iloc[0]
    print(f"""
  ACC-only        {acc:.4f}   vs full {full:.4f}
                  pass = well below full. gap = {full-acc:+.4f}
  Time-of-day     {tod:.4f}
                  pass = near chance (0.50). distance = {abs(tod-.5):.4f}
""")
    verdicts = []
    verdicts.append(("ACC-only well below full", acc < full - 0.05))
    verdicts.append(("Time-of-day near chance", abs(tod - .5) < 0.10))
    for n, ok in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")

    # ---- subject-ID recoverability ----------------------------------------
    banner("Subject identity in the features (the Phase 4 ceiling)")
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    Xs = D[FEATS["Full (minimal)"]].to_numpy(dtype=float)
    ys = LabelEncoder().fit_transform(D.subject.values)
    m = HistGradientBoostingClassifier(max_iter=150, random_state=0)
    acc_id = cross_val_score(m, Xs, ys, cv=StratifiedKFold(3, shuffle=True,
                                                           random_state=0),
                             scoring="accuracy").mean()
    print(f"\n  subject-ID accuracy from features: {acc_id:.4f}")
    print(f"  chance (1/{D.subject.nunique()}):                  "
          f"{1/D.subject.nunique():.4f}")
    print(f"  -> Phase 4's embedding probe must be compared against "
          f"{acc_id:.3f}, not against chance.")

    # ---- negative counts ---------------------------------------------------
    banner("Negative count per subject (pass = >=100)")
    nc = D[D.label == 0].groupby("subject").size().rename("negatives")
    nc = nc.to_frame()
    nc["positives"] = D[D.label == 1].groupby("subject").size()
    nc["ratio"] = (nc.negatives / nc.positives).round(2)
    nc["pass"] = np.where(nc.negatives >= MIN_NEG_PER_SUBJECT, "ok", "FLAG")
    print()
    print(nc.sort_values("negatives").to_string())
    flagged = list(nc[nc["pass"] == "FLAG"].index)
    print(f"\n  flagged: {flagged if flagged else 'none'}")

    R.to_csv(os.path.join(OUT, "phase2_falsification.csv"), index=False)
    nc.to_csv(os.path.join(OUT, "phase2_negative_counts.csv"))
    print("\n--- per-fold AUC, full minimal feature set ---")
    print(per_fold["Full (minimal)"].to_string(index=False))


if __name__ == "__main__":
    main()
