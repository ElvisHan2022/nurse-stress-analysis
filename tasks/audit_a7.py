"""A7 - Labelling propensity and the SCAR assumption.

Standard positive-unlabelled methods require P(labelled | positive) to be
constant. PLAN.md asserts AUC 0.728 and a 5.8x per-subject spread but nothing in
the repository computes it. This section does.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import (OUT, FIG, LOCAL_TZ, load_survey, banner,
                          append_findings, style_axes)


def main():
    banner("A7 - LABELLING PROPENSITY AND THE SCAR ASSUMPTION")

    S = pd.read_parquet(os.path.join(OUT, "a2_sessions.parquet"))
    S["start_utc"] = pd.to_datetime(S.start_utc, utc=True).astype("datetime64[ns, UTC]")
    S["end_utc"] = pd.to_datetime(S.end_utc, utc=True).astype("datetime64[ns, UTC]")
    S["start_local"] = S.start_utc.dt.tz_convert(LOCAL_TZ)

    SV = load_survey().drop_duplicates(["subject", "date", "Start time", "End time"])

    # does a session carry at least one reported event?
    lab = np.zeros(len(S), dtype=bool)
    labr = np.zeros(len(S), dtype=bool)
    for i, r in enumerate(S.itertuples()):
        e = SV[(SV.subject == r.subject) &
               (SV.end_utc > r.start_utc) & (SV.start_utc < r.end_utc)]
        lab[i] = len(e) > 0
        labr[i] = bool(e.labelled.sum())
    S["has_event"] = lab
    S["has_rated_event"] = labr

    print(f"\nsessions: {len(S)}")
    print(f"sessions carrying any reported event:   {int(S.has_event.sum())} "
          f"({100*S.has_event.mean():.1f}%)")
    print(f"sessions carrying a RATED event:        {int(S.has_rated_event.sum())} "
          f"({100*S.has_rated_event.mean():.1f}%)")

    # ---- per-subject probability ------------------------------------------
    banner("Per-subject probability that a session carries a label")
    per = (S.groupby("subject")
             .agg(sessions=("has_event", "size"),
                  with_event=("has_event", "sum"),
                  with_rated=("has_rated_event", "sum"),
                  hours=("dur_s", lambda x: x.sum() / 3600)))
    per["p_labelled"] = (per.with_event / per.sessions).round(4)
    per["p_rated"] = (per.with_rated / per.sessions).round(4)
    per = per.sort_values("p_labelled")
    print()
    print(per.to_string())

    nz = per.p_labelled[per.p_labelled > 0]
    ratio = nz.max() / nz.min()
    print(f"\np_labelled spans {nz.min():.4f} to {nz.max():.4f}")
    print(f"RATIO largest / smallest nonzero: {ratio:.2f}x")

    nzr = per.p_rated[per.p_rated > 0]
    print(f"\n(rated only) spans {nzr.min():.4f} to {nzr.max():.4f}, "
          f"ratio {nzr.max()/nzr.min():.2f}x")

    # ---- propensity model --------------------------------------------------
    banner("Propensity model: predict whether a session carries a label")
    print("\nfeatures: subject (one-hot), session duration, start hour")
    print("protocol: 5-fold stratified cross-validated AUC\n")

    X = pd.DataFrame({
        "subject": S.subject.values,
        "dur_min": S.dur_s.values / 60,
        "start_hour": S.start_local.dt.hour.values,
    })
    y = S.has_event.astype(int).values

    pre = ColumnTransformer([
        ("sub", OneHotEncoder(handle_unknown="ignore"), ["subject"]),
        ("num", StandardScaler(), ["dur_min", "start_hour"]),
    ])
    clf = make_pipeline(pre, LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    print(f"cross-validated AUC: {auc:.4f}")

    clf.fit(X, y)
    auc_in = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    print(f"in-sample AUC:       {auc_in:.4f}  (reported for comparison only)")

    # ablation: which feature carries it
    print("\n--- ablation: which feature carries the propensity signal? ---")
    for name, cols, tf in [
        ("subject only", ["subject"],
         ColumnTransformer([("s", OneHotEncoder(handle_unknown="ignore"), ["subject"])])),
        ("duration only", ["dur_min"],
         ColumnTransformer([("n", StandardScaler(), ["dur_min"])])),
        ("start hour only", ["start_hour"],
         ColumnTransformer([("n", StandardScaler(), ["start_hour"])])),
    ]:
        m = make_pipeline(tf, LogisticRegression(max_iter=5000))
        pp = cross_val_predict(m, X[cols], y, cv=cv, method="predict_proba")[:, 1]
        print(f"  {name:18} AUC {roc_auc_score(y, pp):.4f}")

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.barh(per.index, per.p_labelled, color="#2F6F9F")
    ax.axvline(per.p_labelled.mean(), color="#A8443C", ls="--", lw=1.2,
               label=f"mean {per.p_labelled.mean():.3f}")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.annotate(f"propensity AUC = {auc:.3f}\nspread = {ratio:.1f}×",
                xy=(.62, .12), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=.45", fc="white", ec="#C3CBD2"))
    style_axes(ax, "A7 · P(session carries a reported event), by subject",
               "probability", "subject")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A7_propensity_by_subject.png"), dpi=150)
    plt.close(fig)

    # time budget per subject
    W = pd.read_parquet(os.path.join(OUT, "a6_windows.parquet"))
    tb = (W.groupby("subject")
            .agg(total=("is_level2", "size"),
                 lvl2=("is_level2", "sum"),
                 other_ev=("in_any_event", "sum")))
    tb["other_labelled"] = (tb.other_ev - tb.lvl2).clip(lower=0)
    tb["unlabelled"] = tb.total - tb.lvl2 - tb.other_labelled
    frac = tb[["lvl2", "other_labelled", "unlabelled"]].div(tb.total, axis=0) * 100
    frac = frac.sort_values("lvl2", ascending=False)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    bot = np.zeros(len(frac))
    for col, c, lbl in [("lvl2", "#A8443C", "level-2"),
                        ("other_labelled", "#B8762A", "other reported event"),
                        ("unlabelled", "#5B6B7A", "unlabelled")]:
        ax.bar(frac.index, frac[col], bottom=bot, color=c, label=lbl)
        bot += frac[col].values
    ax.legend(fontsize=8.5, frameon=False, ncol=3)
    style_axes(ax, "A7 · Time budget per subject (share of 120s windows)",
               "subject", "% of windows")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A7_time_budget_by_subject.png"), dpi=150)
    plt.close(fig)

    per.to_csv(os.path.join(OUT, "a7_propensity.csv"))

    banner("A7 - INTERPRETATION")
    print(f"\nLabelling propensity varies by subject (ratio {ratio:.1f}x) and a model")
    print(f"predicts it at AUC {auc:.3f}. SCAR requires this to be constant.")
    print("It fails along exactly the axis LOSO holds out, so a pooled decision")
    print("threshold will not transfer between folds.")

    append_findings(
        "A7", "Labelling propensity and SCAR", "tasks/audit_a7.py",
        [("propensity AUC (cross-validated)", f"{auc:.4f}", "0.728",
          "yes" if abs(auc - .728) < .05 else "NO"),
         ("p_labelled min", f"{nz.min():.4f}", "0.125",
          "yes" if abs(nz.min() - .125) < .03 else "NO"),
         ("p_labelled max", f"{nz.max():.4f}", "0.724",
          "yes" if abs(nz.max() - .724) < .03 else "NO"),
         ("spread ratio", f"{ratio:.2f}x", "5.8x",
          "yes" if abs(ratio - 5.8) < .6 else "NO")],
        ["figures/audit/A7_propensity_by_subject.png",
         "figures/audit/A7_time_budget_by_subject.png"],
        "Determines whether naive PU's ranking guarantee holds. If SCAR fails, "
        "nnPU must be subject-stratified and a pooled threshold is not defensible.",
        f"computed here for the first time; PLAN.md quoted these as 'derived' "
        f"with no source in the repository",
    )


if __name__ == "__main__":
    main()
