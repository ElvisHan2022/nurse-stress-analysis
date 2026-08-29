"""A11 - Near-floor EDA subjects (closes a gap the audit missed).

Plan v2 flags DF, 7E, CE and EG as having median skin conductance of
0.07-0.10 uS, at or below the sensor floor, and makes dropping them a live
decision worth 20 of 178 level-2 events. The A1-A10 audit never checked this.

This computes the per-subject EDA quality picture on the full archive rather
than v2's 45-session diagnostic sample, and re-runs the Phase 2 negative budget
under each exclusion option so the decision is made against numbers.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import OUT, FIG, banner, append_findings, style_axes

FLOOR_US = 0.05
AT_RISK = ["DF", "7E", "CE", "EG"]
WINDOW_S = 120


def main():
    banner("A11 - NEAR-FLOOR EDA SUBJECTS")

    E = pd.read_parquet(os.path.join(OUT, "a9_eda_stats.parquet"))
    W = pd.read_parquet(os.path.join(OUT, "a8_windows_eligible.parquet"))
    W = W.merge(E, on=["session", "w"], how="left")
    print(f"\nwindows with EDA statistics: {W.eda_med.notna().sum():,} of {len(W):,}")

    # ---- per-subject EDA quality, full archive -----------------------------
    banner("Per-subject EDA quality (all 609 sessions, not a 45-session sample)")
    Q = (W.groupby("subject")
           .agg(windows=("eda_med", "size"),
                eda_median=("eda_med", "median"),
                eda_p10=("eda_med", lambda x: x.quantile(.10)),
                eda_p90=("eda_med", lambda x: x.quantile(.90)),
                floor_frac=("eda_floor", "mean"))
           .round(4))
    Q["below_floor_pct"] = (100 * W.groupby("subject").eda_med
                            .apply(lambda x: (x < FLOOR_US).mean())).round(1)
    Q["v2_flag"] = ["AT RISK" if s in AT_RISK else "" for s in Q.index]
    Q = Q.sort_values("eda_median")
    print()
    print(Q.to_string())

    v2_medians = {"DF": 0.07, "7E": 0.09, "CE": 0.10, "EG": 0.10}
    print("\n--- v2's flagged subjects, its figure against the full archive ---")
    for s, v in v2_medians.items():
        mine = Q.loc[s, "eda_median"] if s in Q.index else np.nan
        agree = "yes" if abs(mine - v) < 0.05 else "NO"
        print(f"  {s}: v2 says {v:.2f} uS, full archive says {mine:.3f} uS   match={agree}")

    print(f"\nper-subject median EDA range: {Q.eda_median.min():.3f} to "
          f"{Q.eda_median.max():.3f} uS  ({Q.eda_median.max()/max(Q.eda_median.min(),1e-9):.0f}x)")

    # ---- does near-floor EDA actually cost anything? -----------------------
    banner("Does a low EDA median mean the signal is dead, or just small?")
    print("\nA low median is only disqualifying if the signal is also flat. A small")
    print("but VARYING signal still carries information once causally normalised.\n")

    V = (W.dropna(subset=["eda_med"]).groupby("subject").eda_med
           .agg(median="median", iqr=lambda x: x.quantile(.75) - x.quantile(.25)))
    V["rel_variation"] = (V.iqr / V["median"].replace(0, np.nan)).round(2)
    V["v2_flag"] = ["AT RISK" if s in AT_RISK else "" for s in V.index]
    print(V.sort_values("rel_variation", ascending=False).round(4).to_string())
    print("\nrel_variation = IQR / median. A subject with a low median but healthy")
    print("relative variation is usable; one with low median AND low variation is not.")

    # ---- Phase 2 budget under each exclusion option ------------------------
    banner("Phase 2 negative budget under each exclusion option")
    ev2 = W.groupby("subject").is_level2.sum()
    neg = W.groupby("subject").eligible_neg.sum()

    options = [
        ("keep all 15", []),
        ("drop 6D only (v2 rule 8)", ["6D"]),
        ("drop 6D + all four at-risk", ["6D"] + AT_RISK),
        ("drop 6D + DF, 7E only", ["6D", "DF", "7E"]),
    ]
    rows = []
    for label, drop in options:
        keep = [s for s in ev2.index if s not in drop]
        pos = int(ev2[keep].sum())
        ng = int(neg[keep].sum())
        # per-subject achievable ratio floor, ignoring subjects with no positives
        ratios = (neg[keep] / ev2[keep].replace(0, np.nan)).dropna()
        rows.append({
            "option": label,
            "subjects": len(keep),
            "level2 windows": pos,
            "eligible neg windows": ng,
            "neg hours": round(ng * WINDOW_S / 3600, 1),
            "aggregate ratio": round(ng / max(pos, 1), 2),
            "min achievable": round(ratios.min(), 2) if len(ratios) else np.nan,
            "subjects < 3:1": int((ratios < 3).sum()),
        })
    B = pd.DataFrame(rows)
    print()
    print(B.to_string(index=False))

    ev_counts = pd.read_csv(os.path.join(OUT, "a8_achievable_ratio.csv"),
                            index_col="subject").events
    lost = int(ev_counts[AT_RISK].sum())
    print(f"\nDropping all four at-risk subjects costs {lost} level-2 EVENTS "
          f"of {int(ev_counts.sum())} ({100*lost/ev_counts.sum():.0f}%).")
    print("v2 estimates 20 of 178. Both are material; make the call explicitly.")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = ["#A8443C" if s in AT_RISK else "#2F6F9F" for s in Q.index]
    ax.barh(Q.index, Q.eda_median, color=cols)
    ax.axvline(FLOOR_US, color="#17212B", ls="--", lw=1.3,
               label=f"sensor floor {FLOOR_US} uS")
    ax.axvline(0.2, color="#B8762A", ls=":", lw=1.2, label="typical resting lower bound")
    ax.set_xscale("log")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    style_axes(ax, "A11 · Median skin conductance per subject (red = flagged by plan v2)",
               "median EDA (µS, log scale)", "subject")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A11_eda_floor_by_subject.png"), dpi=150)
    plt.close(fig)

    Q.to_csv(os.path.join(OUT, "a11_eda_quality.csv"))
    B.to_csv(os.path.join(OUT, "a11_exclusion_budget.csv"), index=False)

    append_findings(
        "A11", "Near-floor EDA subjects", "tasks/audit_a11.py",
        [(f"median EDA, {s}", f"{Q.loc[s,'eda_median']:.3f} uS",
          f"{v2_medians[s]:.2f} (v2)",
          "yes" if abs(Q.loc[s, "eda_median"] - v2_medians[s]) < .05 else "NO")
         for s in AT_RISK if s in Q.index] +
        [("per-subject median EDA range",
          f"{Q.eda_median.min():.3f}-{Q.eda_median.max():.3f} uS", "26x (v2)",
          f"{Q.eda_median.max()/max(Q.eda_median.min(),1e-9):.0f}x here"),
         ("events lost if all four dropped", lost, "20 (v2)",
          "yes" if abs(lost - 20) <= 3 else "NO")],
        ["figures/audit/A11_eda_floor_by_subject.png"],
        "Closes the gap between the A1-A10 audit and plan v2 section 1.5. Feeds "
        "the Phase 1 rule-9 decision and the Phase 2 budget.",
        "the audit never checked EDA signal quality per subject; v2 caught it",
    )


if __name__ == "__main__":
    main()
