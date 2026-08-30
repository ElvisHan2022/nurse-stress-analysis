"""PHASE 3.2-3.3 - Normalisation and the baseline model.

3.2  causal_z_safe within session. Trailing 60-min median and IQR, so it uses
     only the past: leakage-free, and reproducible at deployment where there is
     no future. The IQR denominator is floored and the output clipped (section
     0.1) - without that a flat stretch produced |z| up to 1,600.

3.3  Gradient boosting, balanced class weights, leave-one-subject-out.
     This is the number everything later has to beat, and there is a real
     chance it wins outright.

Reporting rules from CONTRIBUTING.md, applied here rather than afterwards:
  - the majority-class baseline sits beside every model number
  - per-fold, never averaged, with counts
  - across all three negative-sampling seeds, range quoted
  - event-level recall at a fixed false-alarm rate is the primary metric;
    AUC is secondary because it is prevalence-invariant and therefore does not
    reflect the operating point anyone would deploy
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, banner, causal_z_safe

SEEDS = [0, 1, 2]
WINDOW_S = 120
FA_PER_HOUR = 1.0          # the operating point: 1 false alarm per worn hour
EVENT_FIRE_FRAC = 0.50     # an event counts as detected if >=50% of its windows fire

NORMALISE = ["eda_tonic_mean", "eda_phasic_sd", "eda_scr_amp",
             "hr_mean", "hr_max", "acc_mag_mean", "acc_p2p"]
PASSTHRU = ["eda_tonic_slope", "eda_scr_count", "hr_sd", "hr_slope",
            "hr_delta30", "temp_slope", "temp_delta30", "acc_mag_sd",
            "acc_frac_move", "hour", "min_into_session"]


def normalise(F):
    """3.2 - causal z within session on the level-like channels."""
    F = F.sort_values(["session", "w"]).copy()
    for c in NORMALISE:
        out = np.full(len(F), np.nan)
        for sess, g in F.groupby("session", sort=False):
            s = pd.Series(g[c].values,
                          index=pd.RangeIndex(len(g)) * WINDOW_S)
            s.index = pd.to_timedelta(s.index, unit="s") + pd.Timestamp("2020-01-01")
            z = causal_z_safe(s, window="60min", min_periods=3)
            out[F.index.get_indexer(g.index)] = z.values
        F[c + "_z"] = out
    return F


def main():
    banner("PHASE 3.2-3.3 - NORMALISATION AND BASELINE")

    F = pd.read_parquet(os.path.join(DERIVED, "features.parquet"))
    print(f"\n{len(F):,} windows, {F.subject.nunique()} subjects")

    print("\n3.2 causal_z_safe within session on level-like channels...")
    F = normalise(F)
    zc = [c + "_z" for c in NORMALISE]
    print(f"    normalised {len(zc)} channels; "
          f"|z| max = {F[zc].abs().max().max():.2f} (clip is 10)")
    print(f"    NaN from the warm-up period: "
          f"{100*F[zc].isna().mean().mean():.2f}% of values")

    FEATS = zc + PASSTHRU
    print(f"\nbaseline feature set: {len(FEATS)} features")

    banner("3.3 Baseline - gradient boosting, LOSO, 3 seeds")

    per_seed, per_fold_all = [], []
    for seed in SEEDS:
        L = pd.read_parquet(os.path.join(DERIVED, f"labels_seed{seed}.parquet"))
        D = L[["subject", "session", "w", "label"]].merge(
            F.drop(columns=["label"]), on=["subject", "session", "w"], how="left")
        X = D[FEATS].to_numpy(dtype=float)
        y = D.label.values.astype(int)
        g = D.subject.values

        oof = np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            if len(np.unique(y[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=.06, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=0, class_weight="balanced")
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]

        ok = np.isfinite(oof)
        auc = roc_auc_score(y[ok], oof[ok])
        ap = average_precision_score(y[ok], oof[ok])
        prev = y[ok].mean()

        # threshold set to hit FA_PER_HOUR on the NEGATIVE windows only
        neg_scores = np.sort(oof[ok & (y == 0)])[::-1]
        neg_hours = (ok & (y == 0)).sum() * WINDOW_S / 3600
        n_allowed = int(FA_PER_HOUR * neg_hours)
        thr = neg_scores[min(n_allowed, len(neg_scores) - 1)]
        pred = (oof >= thr).astype(int)
        rec = pred[ok & (y == 1)].mean()

        per_seed.append({"seed": seed, "AUC": round(auc, 4),
                         "PR-AUC": round(ap, 4), "prevalence": round(prev, 4),
                         "recall@1FA/h": round(rec, 4),
                         "F1": round(f1_score(y[ok], pred[ok]), 4)})

        for s in np.unique(g):
            m = (g == s) & ok
            if len(np.unique(y[m])) < 2:
                continue
            per_fold_all.append({
                "seed": seed, "subject": s, "n": int(m.sum()),
                "pos": int(y[m].sum()),
                "AUC": round(roc_auc_score(y[m], oof[m]), 3),
                "recall@1FA/h": round(pred[m & (y == 1)].mean(), 3)})

    P = pd.DataFrame(per_seed)
    print()
    print(P.to_string(index=False))

    banner("Baselines it must beat")
    prev = P.prevalence.mean()
    print(f"""
  always-negative     accuracy {1-prev:.4f}   recall 0.000   F1 0.000
  always-positive     accuracy {prev:.4f}   recall 1.000   F1 {2*prev/(1+prev):.4f}
  random (AUC)                 0.5000
  MODEL               AUC {P.AUC.mean():.4f} (range {P.AUC.max()-P.AUC.min():.4f})
                      recall@1FA/h {P['recall@1FA/h'].mean():.4f}
                      PR-AUC {P['PR-AUC'].mean():.4f} vs prevalence {prev:.4f}
""")
    lift = P["PR-AUC"].mean() / prev
    print(f"  PR-AUC lift over prevalence: {lift:.2f}x")

    banner("Per fold - never averaged")
    PF = pd.DataFrame(per_fold_all)
    piv = PF.pivot_table(index="subject", columns="seed", values="AUC")
    piv.columns = [f"seed{c}" for c in piv.columns]
    piv["mean"] = piv.mean(axis=1).round(3)
    piv["range"] = (piv[[c for c in piv.columns if c.startswith("seed")]].max(axis=1)
                    - piv[[c for c in piv.columns if c.startswith("seed")]].min(axis=1)).round(3)
    piv["pos"] = PF[PF.seed == 0].set_index("subject")["pos"]
    print()
    print(piv.sort_values("mean").to_string())
    below = piv[piv["mean"] <= 0.55].index.tolist()
    print(f"\n  folds at or below 0.55: {below if below else 'none'}")
    print(f"  fold spread: {piv['mean'].min():.3f} to {piv['mean'].max():.3f}")

    P.to_csv(os.path.join(OUT, "phase3_baseline.csv"), index=False)
    PF.to_csv(os.path.join(OUT, "phase3_per_fold.csv"), index=False)
    print(f"\nwrote reports/audit/phase3_baseline.csv and phase3_per_fold.csv")


if __name__ == "__main__":
    main()
