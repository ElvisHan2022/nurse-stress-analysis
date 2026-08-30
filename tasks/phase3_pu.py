"""PHASE 3.4-3.5 - nnPU and the naive comparator.

Three models, identical folds, identical windows, identical features. The only
thing that changes is HOW THE LABELS ARE USED.

  3.3 BASELINE   positives vs the curated, activity-matched negatives.
                 Treats the constructed negatives as if they were observed.

  3.4 nnPU       positives vs the FULL eligible pool, treated as unlabelled
                 rather than negative. Uses the non-negative risk correction
                 (Kiryo et al. 2017) so the loss cannot go negative when the
                 model gets confident. Fitted subject-stratified, because
                 labelling propensity varies by subject (audit A7, AUC 0.630
                 for the subject component) and a single pooled correction is
                 not defensible.
                 pi is BRACKETED, not estimated - it is not identifiable from
                 positive-unlabelled data. Anchored on the observed 6.49%
                 window prevalence, with under-reporting implying the truth
                 sits above it.

  3.5 NAIVE      positives vs ALL unlabelled time called negative. No same-day
                 rule, no guard band, no activity matching. Reported BESIDE the
                 curated result, never used to seed or select it - its errors
                 are systematically placed and bootstrapping from it would
                 launder the bias.

THE GATE, written down before the numbers are seen:

  Proceed to Phase 4 only if 3.4 or 3.5 moves event recall at 1 alarm/hour by
  more than the seed range of the baseline (0.0724). Otherwise the bottleneck
  is the data, not the method, and a learned representation will not fix it.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, DERIVED, banner
from phase3_baseline import normalise, NORMALISE, PASSTHRU, SEEDS, WINDOW_S

EVENT_FIRE_FRAC = 0.50
FA_PER_HOUR = 1.0
PI_GRID = [0.05, 0.10, 0.20, 0.30]
BASELINE_SEED_RANGE = 0.0724          # the gate threshold, fixed in advance


# ---------------------------------------------------------------- nnPU ----

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def nnpu_fit(Xp, Xu, pi, epochs=120, lr=1e-3, seed=0):
    """Non-negative PU risk (Kiryo et al. 2017).

    R = pi*E_p[l(+)] + max(0, E_u[l(-)] - pi*E_p[l(-)])

    The max(0, .) is the whole point: the unlabelled set contains positives, so
    the naive negative-risk term can go negative once the model is confident,
    and an uncorrected objective then drives itself to overfit.
    """
    torch.manual_seed(seed)
    d = Xp.shape[1]
    m = MLP(d)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    xp = torch.tensor(Xp, dtype=torch.float32)
    xu = torch.tensor(Xu, dtype=torch.float32)
    sig = nn.functional.softplus                     # l(z) = log(1+exp(-z))
    for _ in range(epochs):
        opt.zero_grad()
        gp, gu = m(xp), m(xu)
        r_p_pos = sig(-gp).mean()                    # positives called positive
        r_p_neg = sig(gp).mean()                     # positives called negative
        r_u_neg = sig(gu).mean()                     # unlabelled called negative
        neg_risk = r_u_neg - pi * r_p_neg
        loss = pi * r_p_pos + (neg_risk if neg_risk >= 0 else -neg_risk * 0.0)
        if neg_risk < 0:                             # gradient ascent on the
            loss = -neg_risk                         # violating term only
        loss.backward()
        opt.step()
    m.eval()
    return m


# ----------------------------------------------------------- evaluation ----

def operating_point(y, score, ok, ev_id, neg_per_hour, fa=FA_PER_HOUR):
    """Threshold on FPR, then report window and event recall."""
    fpr_target = fa / neg_per_hour
    negs = np.sort(score[ok & (y == 0)])[::-1]
    if len(negs) == 0:
        return np.nan, np.nan, np.nan
    k = max(int(round(fpr_target * len(negs))) - 1, 0)
    thr = negs[min(k, len(negs) - 1)]
    pred = (score >= thr).astype(int)
    win_rec = pred[ok & (y == 1)].mean()
    d = pd.DataFrame({"ev": ev_id, "pred": pred, "ok": ok, "y": y})
    d = d[(d.y == 1) & d.ok & (d.ev >= 0)]
    ev_rec = float((d.groupby("ev").pred.mean() >= EVENT_FIRE_FRAC).mean()) \
        if len(d) else np.nan
    return pred[ok & (y == 0)].mean(), win_rec, ev_rec


def main():
    banner("PHASE 3.4-3.5 - nnPU AND THE NAIVE COMPARATOR")

    F = normalise(pd.read_parquet(os.path.join(DERIVED, "features.parquet")))
    W = pd.read_parquet(os.path.join(DERIVED, "windows.parquet"))
    EV = pd.read_parquet(os.path.join(DERIVED, "events.parquet"))
    A = pd.read_parquet(os.path.join(OUT, "a8_windows_eligible.parquet"))
    FEATS = [c + "_z" for c in NORMALISE] + PASSTHRU

    W["start_utc"] = pd.to_datetime(W.start_utc, utc=True)
    W["end_utc"] = pd.to_datetime(W.end_utc, utc=True)
    W["event_id"] = -1
    for i, r in EV.iterrows():
        ov = ((np.minimum(W.end_utc, r.end_utc)
               - np.maximum(W.start_utc, r.start_utc)).dt.total_seconds())
        W.loc[(W.subject == r.subject) & (ov >= WINDOW_S * .5), "event_id"] = i
    W = W.merge(A[["subject", "session", "w", "eligible_neg"]],
                on=["subject", "session", "w"], how="left")
    W["eligible_neg"] = W.eligible_neg.fillna(False) & (W.label == 0)

    true_prev = float(W.label.mean())
    neg_per_hour = (3600 / WINDOW_S) * (1 - true_prev)
    print(f"\ntrue window prevalence {true_prev:.4f}  ->  "
          f"1 FA/h = FPR {1/neg_per_hour:.4f}")

    FULL = W.merge(F.drop(columns=["label"]), on=["subject", "session", "w"],
                   how="left")
    rows = []

    # ------------------------------------------------ 3.5 NAIVE ------------
    banner("3.5 Naive comparator - all unlabelled time called negative")
    D = FULL.copy()
    X = D[FEATS].to_numpy(dtype=float)
    y = D.label.values.astype(int)
    g = D.subject.values
    print(f"\n{len(D):,} windows, {y.sum():,} positive "
          f"({100*y.mean():.2f}%) - no matching, no guard band, no same-day rule")
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=0, class_weight="balanced")
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    fpr, wr, er = operating_point(y, oof, ok, D.event_id.values, neg_per_hour)
    rows.append({"model": "3.5 naive (all unlabelled = neg)", "seed": "-",
                 "pi": "-", "AUC": round(roc_auc_score(y[ok], oof[ok]), 4),
                 "PR-AUC": round(average_precision_score(y[ok], oof[ok]), 4),
                 "FPR": round(fpr, 4), "window recall": round(wr, 4),
                 "EVENT recall": round(er, 4)})
    print(f"  AUC {rows[-1]['AUC']}   EVENT recall {rows[-1]['EVENT recall']}")

    # ------------------------------------------------ 3.4 nnPU ------------
    banner("3.4 nnPU, subject-stratified, pi bracketed")
    print("\nPositives = level-2 windows. Unlabelled = the FULL eligible pool,")
    print("not the matched subset. pi is not identifiable, so it is bracketed.\n")

    POS = FULL[FULL.label == 1]
    UNL = FULL[FULL.eligible_neg]
    print(f"  positives {len(POS):,}   unlabelled pool {len(UNL):,}")

    for pi in PI_GRID:
        for seed in SEEDS[:1]:                       # pi grid is the sweep here
            oof = np.full(len(FULL), np.nan)
            for tr_s, te_s in LeaveOneGroupOut().split(
                    np.zeros(FULL.subject.nunique()),
                    groups=np.arange(FULL.subject.nunique())):
                pass
            subs = np.array(sorted(FULL.subject.unique()))
            for held in subs:
                tr = FULL[(FULL.subject != held) &
                          ((FULL.label == 1) | FULL.eligible_neg)]
                te_mask = (FULL.subject == held).values
                sc = StandardScaler().fit(tr[FEATS].to_numpy(dtype=float))
                Xp = sc.transform(tr[tr.label == 1][FEATS].to_numpy(dtype=float))
                Xu = sc.transform(tr[tr.eligible_neg][FEATS].to_numpy(dtype=float))
                Xp = np.nan_to_num(Xp); Xu = np.nan_to_num(Xu)
                mdl = nnpu_fit(Xp, Xu, pi, seed=seed)
                Xt = np.nan_to_num(sc.transform(
                    FULL.loc[te_mask, FEATS].to_numpy(dtype=float)))
                with torch.no_grad():
                    oof[te_mask] = mdl(torch.tensor(Xt, dtype=torch.float32)).numpy()
            y = FULL.label.values.astype(int)
            ok = np.isfinite(oof)
            fpr, wr, er = operating_point(y, oof, ok, FULL.event_id.values,
                                          neg_per_hour)
            rows.append({"model": "3.4 nnPU", "seed": seed, "pi": pi,
                         "AUC": round(roc_auc_score(y[ok], oof[ok]), 4),
                         "PR-AUC": round(average_precision_score(y[ok], oof[ok]), 4),
                         "FPR": round(fpr, 4), "window recall": round(wr, 4),
                         "EVENT recall": round(er, 4)})
            print(f"  pi={pi:<5} AUC {rows[-1]['AUC']:.4f}   "
                  f"EVENT recall {rows[-1]['EVENT recall']:.4f}")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "phase3_pu_naive.csv"), index=False)

    # ------------------------------------------------ THE GATE ------------
    banner("THE GATE")
    base_er = 0.1063
    print(f"""
  Pre-committed rule, written before these numbers existed:

    Proceed to Phase 4 only if 3.4 or 3.5 moves event recall at 1 alarm/hour
    by MORE than the baseline seed range of {BASELINE_SEED_RANGE:.4f}.

  Phase 3.3 baseline event recall @ 1 FA/h : {base_er:.4f}
  Threshold to clear                       : {base_er + BASELINE_SEED_RANGE:.4f}
""")
    best = R.loc[R["EVENT recall"].idxmax()]
    print(R[["model", "pi", "AUC", "FPR", "window recall", "EVENT recall"]]
          .to_string(index=False))
    delta = best["EVENT recall"] - base_er
    print(f"\n  best arm : {best['model']} (pi={best['pi']})")
    print(f"  event recall {best['EVENT recall']:.4f}  vs baseline {base_er:.4f}"
          f"   delta {delta:+.4f}")
    print(f"\n  GATE: {'PASSED - proceed to Phase 4' if delta > BASELINE_SEED_RANGE else 'NOT PASSED - the bottleneck is the data, not the method'}")
    print(f"\nwrote {os.path.join(OUT, 'phase3_pu_naive.csv')}")


if __name__ == "__main__":
    main()
