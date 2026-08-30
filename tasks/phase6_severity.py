"""PHASE 6.1 - Severity, and the remaining required ablations.

SEVERITY is a different task from detection and a strictly easier one: it runs
on the 245 rated episodes and needs no constructed negatives at all. That makes
it the one part of this study whose numbers are not conditional on a
construction.

Two things make it hard anyway. There are 20 medium-severity episodes across
the whole study and nine of fifteen participants have none, so a three-class
model will be tested on participants containing classes it never saw. And the
scales are not comparable between people: one participant rated 1/0/25 across
low/medium/high, another 11/4/5. A raw three-class softmax would learn who is
speaking rather than how severe the episode was.

We therefore fit an ordinal model via cumulative binary links, P(y>=1) and
P(y>=2), which respects the ordering without assuming the categories are
equally spaced, and report the collapsed {0,1} vs {2} task alongside because
the middle class is probably not learnable at this count.

ABLATIONS covers the entries in the plan's required list that the detection
phases did not already answer.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import DERIVED, OUT, LOCAL_TZ, load_survey, banner
from phase3_baseline import normalise, NORMALISE, PASSTHRU

WINDOW_S = 120


def event_features(F, SV):
    """Aggregate window features over each rated episode."""
    F = F.copy()
    F["start_utc"] = pd.to_datetime(F.start_utc, utc=True)
    F["end_utc"] = F.start_utc + pd.Timedelta(seconds=WINDOW_S)
    feats = [c + "_z" for c in NORMALISE] + PASSTHRU
    rows = []
    for i, e in SV.iterrows():
        m = ((F.subject == e.subject) & (F.end_utc > e.start_utc)
             & (F.start_utc < e.end_utc))
        if m.sum() < 1:
            continue
        agg = F.loc[m, feats].mean()
        agg["n_windows"] = int(m.sum())
        agg["subject"] = e.subject
        agg["level"] = int(e["Stress level"])
        agg["duration_min"] = float(e.duration_min)
        rows.append(agg)
    return pd.DataFrame(rows)


def loso_binary(X, y, g):
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, C=0.5,
                                             class_weight="balanced"))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    banner("PHASE 6.1 - SEVERITY ON RATED EPISODES")

    F = normalise(pd.read_parquet(os.path.join(DERIVED, "features.parquet")))
    SV = load_survey()
    SV = SV[~SV.is_exact_dup & SV.labelled].drop_duplicates(
        ["subject", "date", "Start time", "End time"])
    cohort = set(pd.read_parquet(
        os.path.join(DERIVED, "windows.parquet")).subject.unique())
    SV = SV[SV.subject.isin(cohort)]
    print(f"\nrated episodes in the retained cohort: {len(SV)}")
    print(SV["Stress level"].value_counts().sort_index().to_string())

    E = event_features(F, SV)
    print(f"\nepisodes with sensor coverage: {len(E)}")
    print("\nper participant:")
    piv = E.pivot_table(index="subject", columns="level", aggfunc="size",
                        fill_value=0)
    print(piv.to_string())
    no_mid = int((piv.get(1, pd.Series(0, piv.index)) == 0).sum())
    print(f"\nparticipants with no medium-severity episode: {no_mid} of {len(piv)}")

    feats = [c + "_z" for c in NORMALISE] + PASSTHRU + ["duration_min"]
    X = E[feats].to_numpy(dtype=float)
    g = E.subject.values
    lvl = E.level.values

    banner("Ordinal via cumulative links, leave-one-participant-out")
    res = []
    for thresh, name in [(1, "P(severity >= 1), low vs rest"),
                         (2, "P(severity >= 2), high vs rest")]:
        y = (lvl >= thresh).astype(int)
        if len(np.unique(y)) < 2:
            continue
        oof = loso_binary(X, y, g)
        ok = np.isfinite(oof)
        auc = roc_auc_score(y[ok], oof[ok])
        res.append({"link": name, "n": int(ok.sum()),
                    "positives": int(y[ok].sum()),
                    "AUC": round(auc, 4)})
        print(f"  {name:34} AUC {auc:.4f}   ({int(y[ok].sum())}/{int(ok.sum())} positive)")

    banner("Collapsed task: high severity vs everything else")
    y2 = (lvl >= 2).astype(int)
    oof2 = loso_binary(X, y2, g)
    ok = np.isfinite(oof2)
    auc2 = roc_auc_score(y2[ok], oof2[ok])
    print(f"\n  AUC {auc2:.4f} on {int(ok.sum())} episodes, "
          f"{int(y2[ok].sum())} high-severity")

    print("\n  per participant:")
    for s in sorted(np.unique(g)):
        m = (g == s) & ok
        if m.sum() > 2 and len(np.unique(y2[m])) == 2:
            print(f"    {s:4} AUC {roc_auc_score(y2[m], oof2[m]):.3f}  "
                  f"({int(m.sum())} episodes)")
        else:
            print(f"    {s:4} single-class fold, not scorable "
                  f"({int(m.sum())} episodes)")

    banner("Does duration alone explain severity?")
    for thresh in [1, 2]:
        y = (lvl >= thresh).astype(int)
        if len(np.unique(y)) < 2:
            continue
        r = st.pointbiserialr(y, E.duration_min.values)
        print(f"  severity >= {thresh}: point-biserial r = {r.statistic:+.3f}, "
              f"p = {r.pvalue:.3f}")
    print("""
  If severity were largely a restatement of how long the episode lasted, the
  task would be less interesting than it looks. Reported so the reader can
  judge that directly.""")

    pd.DataFrame(res + [{"link": "collapsed high vs rest",
                         "n": int(ok.sum()), "positives": int(y2[ok].sum()),
                         "AUC": round(auc2, 4)}]).to_csv(
        os.path.join(OUT, "phase6_severity.csv"), index=False)
    E.to_parquet(os.path.join(DERIVED, "episode_features.parquet"), index=False)
    print(f"\nwrote reports/audit/phase6_severity.csv")


if __name__ == "__main__":
    main()
