"""PHASE 3.3b - The metrics the plan actually asks for.

Two corrections to a first pass at the baseline:

1. FALSE ALARMS PER WORN HOUR IS NOT DIRECTLY ESTIMABLE on the constructed set.
   The scored sample is 43.6% positive; real worn time is roughly 6%. Counting
   false alarms per hour of SAMPLED negative time therefore understates the real
   rate by about 7x. The fix is to fix the operating point on the FALSE POSITIVE
   RATE, which is prevalence-invariant, and then translate that into a real-time
   alarm rate using the true window prevalence.

2. EVENT-LEVEL RECALL IS THE PRIMARY METRIC and was not computed. An event is
   detected if at least k% of its windows fire. Window recall and event recall
   are different quantities and the plan headlines the second.

Everything is reported per fold across three seeds, beside the trivial baselines.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, banner
from phase3_baseline import normalise, NORMALISE, PASSTHRU, SEEDS, WINDOW_S

EVENT_FIRE_FRAC = 0.50
TARGET_FA_PER_HOUR = [0.5, 1.0, 2.0]
WINDOWS_PER_HOUR = 3600 / WINDOW_S          # 30


def main():
    banner("PHASE 3.3b - EVENT-LEVEL METRICS AT A REAL OPERATING POINT")

    F = pd.read_parquet(os.path.join(DERIVED, "features.parquet"))
    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    EV = pd.read_parquet(os.path.join(DERIVED, "events.parquet"))
    F = normalise(F)
    FEATS = [c + "_z" for c in NORMALISE] + PASSTHRU

    # true window-level prevalence on ALL worn windows, not the sampled set
    true_prev = float(W.label.mean())
    neg_per_hour = WINDOWS_PER_HOUR * (1 - true_prev)
    print(f"\ntrue window prevalence over all {len(W):,} windows : {true_prev:.4f}")
    print(f"negative windows per worn hour                    : {neg_per_hour:.1f}")
    print("\nTo allow F false alarms per worn hour, the false positive rate must")
    print("be F / (negative windows per hour):")
    for f in TARGET_FA_PER_HOUR:
        print(f"    {f:>4} FA/h  ->  FPR = {f/neg_per_hour:.4f}")

    # map each positive window to its event
    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True)
    W["end_utc"] = pd.to_datetime(W.end_utc, utc=True)
    W["event_id"] = -1
    for i, r in EV.iterrows():
        ov = ((np.minimum(W.end_utc, r.end_utc) - np.maximum(W.start_utc, r.start_utc))
              .dt.total_seconds())
        m = (W.subject == r.subject) & (ov >= WINDOW_S * 0.5)
        W.loc[m, "event_id"] = i
    ev_map = W.set_index(["subject", "session", "w"]).event_id
    n_ev = int((W.event_id >= 0).groupby(W.event_id).ngroups)
    print(f"\nevents with at least one positive window: "
          f"{W[W.event_id>=0].event_id.nunique()} of {len(EV)}")

    rows, fold_rows = [], []
    for seed in SEEDS:
        L = pd.read_parquet(os.path.join(DERIVED, f"labels_seed{seed}.parquet"))
        D = L[["subject", "session", "w", "label"]].merge(
            F.drop(columns=["label"]), on=["subject", "session", "w"], how="left")
        D["event_id"] = ev_map.reindex(
            pd.MultiIndex.from_frame(D[["subject", "session", "w"]])).values
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

        for fa in TARGET_FA_PER_HOUR:
            fpr_target = fa / neg_per_hour
            negs = np.sort(oof[ok & (y == 0)])[::-1]
            k = max(int(round(fpr_target * len(negs))) - 1, 0)
            thr = negs[min(k, len(negs) - 1)]
            pred = (oof >= thr).astype(int)

            win_rec = pred[ok & (y == 1)].mean()
            actual_fpr = pred[ok & (y == 0)].mean()

            # event-level: an event fires if >=50% of its positive windows fire
            dfp = pd.DataFrame({"ev": D.event_id.values, "pred": pred, "ok": ok,
                                "y": y})
            dfp = dfp[(dfp.y == 1) & dfp.ok & (dfp.ev >= 0)]
            per_ev = dfp.groupby("ev").pred.mean()
            ev_rec = float((per_ev >= EVENT_FIRE_FRAC).mean())
            rows.append({"seed": seed, "target FA/h": fa,
                         "FPR": round(actual_fpr, 4),
                         "window recall": round(win_rec, 4),
                         "EVENT recall": round(ev_rec, 4),
                         "n events": len(per_ev), "AUC": round(auc, 4),
                         "PR-AUC": round(ap, 4)})

            if fa == 1.0:
                for s in np.unique(g):
                    m = (g == s) & ok
                    d2 = dfp[D.subject.values[dfp.index] == s] if False else None
                    sub = pd.DataFrame({"ev": D.event_id.values[m],
                                        "pred": pred[m], "y": y[m]})
                    sub = sub[(sub.y == 1) & (sub.ev >= 0)]
                    pe = sub.groupby("ev").pred.mean()
                    fold_rows.append({
                        "seed": seed, "subject": s,
                        "AUC": round(roc_auc_score(y[m], oof[m]), 3)
                        if len(np.unique(y[m])) == 2 else np.nan,
                        "events": len(pe),
                        "event recall": round(float((pe >= EVENT_FIRE_FRAC).mean()), 3)
                        if len(pe) else np.nan})

    R = pd.DataFrame(rows)
    banner("Operating points, 3 seeds")
    print()
    print(R.groupby("target FA/h").agg(
        FPR=("FPR", "mean"),
        window_recall=("window recall", "mean"),
        EVENT_recall=("EVENT recall", "mean"),
        event_recall_range=("EVENT recall", lambda x: round(np.ptp(x), 4)),
        n_events=("n events", "first"),
        AUC=("AUC", "mean")).round(4).to_string())

    banner("Against the trivial baselines")
    print(f"""
  always-positive       event recall 1.000   at FPR 1.000  (alarms constantly)
  random at same FPR    event recall ~ FPR
  MODEL at 1 FA/h       event recall {R[R['target FA/h']==1]['EVENT recall'].mean():.4f}
                        at FPR {R[R['target FA/h']==1].FPR.mean():.4f}
  AUC {R.AUC.mean():.4f} (seed range {R.AUC.max()-R.AUC.min():.4f})
  PR-AUC {R['PR-AUC'].mean():.4f}
""")

    banner("Per fold at 1 FA/h - never averaged")
    PF = pd.DataFrame(fold_rows)
    piv = PF.pivot_table(index="subject", values=["AUC", "event recall", "events"],
                         aggfunc={"AUC": "mean", "event recall": "mean",
                                  "events": "first"}).round(3)
    print()
    print(piv.sort_values("AUC").to_string())

    R.to_csv(os.path.join(OUT, "phase3_operating_points.csv"), index=False)
    PF.to_csv(os.path.join(OUT, "phase3_event_per_fold.csv"), index=False)
    print(f"\nwrote phase3_operating_points.csv and phase3_event_per_fold.csv")


if __name__ == "__main__":
    main()
