"""A3 - Survey audit.

Structure, duplicates, level distribution, event durations, overlaps.
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
from audit_common import (OUT, FIG, SURVEY, load_survey, banner, show,
                          append_findings, style_axes)

LEVEL_NAME = {0.0: "low (0)", 1.0: "medium (1)", 2.0: "high (2)"}


def main():
    banner("A3 - SURVEY AUDIT")

    raw = pd.read_excel(SURVEY, sheet_name=0)
    print(f"\nraw shape: {raw.shape}")
    print("\n--- columns (repr, to expose whitespace and newlines) ---")
    for c in raw.columns:
        print(f"  {c!r}")

    print("\n--- dtypes ---")
    print(raw.dtypes.to_string())

    show("raw.head(10)", raw, 10)

    print("\n--- ID column dtype trap ---")
    print("python types present in ID:",
          sorted({type(v).__name__ for v in raw['ID']}))

    SV = load_survey()
    print(f"\nrows before dropping exact duplicates: {len(SV)}")
    n_exact = int(SV.is_exact_dup.sum())
    print(f"exact duplicate rows:                  {n_exact}")
    print(f"rows after:                            {len(SV) - n_exact}")

    print("\n--- duplicate rows (all copies shown) ---")
    dups = SV[SV.is_dup][["subject", "date", "Start time", "End time",
                          "Stress level"]].sort_values(
        ["subject", "date", "Start time"])
    print(dups.to_string() if len(dups) else "  none")

    # ---- level distribution -----------------------------------------------
    print("\n--- Stress level value_counts(dropna=False), unrated included ---")
    vc = SV["Stress level"].value_counts(dropna=False)
    for k, v in vc.items():
        name = "UNRATED ('na')" if pd.isna(k) else LEVEL_NAME.get(k, str(k))
        print(f"  {name:16} {v:4d}")
    n_unrated = int(SV["Stress level"].isna().sum())
    n_rated = int(SV.labelled.sum())
    print(f"\n  rated total  {n_rated}")
    print(f"  unrated      {n_unrated}")

    print("\n--- level by subject ---")
    piv = SV.pivot_table(index="subject", columns="Stress level",
                         aggfunc="size", fill_value=0)
    piv.columns = [LEVEL_NAME.get(c, str(c)) for c in piv.columns]
    piv["events"] = SV.groupby("subject").size()
    piv["rated"] = SV.groupby("subject").labelled.sum()
    piv["unrated"] = piv.events - piv.rated
    print(piv.to_string())
    n_no_med = int((piv.get("medium (1)", pd.Series(0, piv.index)) == 0).sum())
    print(f"\nsubjects with zero medium(1) events: {n_no_med} of {len(piv)}")

    # ---- durations ---------------------------------------------------------
    R = SV[SV.labelled].copy()
    print("\n--- duration_min.describe() by level (rated events only) ---")
    dd = R.groupby("Stress level").duration_min.describe(
        percentiles=[.25, .5, .75, .9]).round(2)
    dd.index = [LEVEL_NAME.get(i, str(i)) for i in dd.index]
    print(dd.to_string())

    n_over60 = int((R.duration_min > 60).sum())
    print(f"\nrated events over 60 min: {n_over60}")
    print(f"maximum rated duration:   {R.duration_min.max():.0f} min")
    print(f"rated events <= 2 min:    {int((R.duration_min <= 2).sum())}")
    print(f"events crossing midnight: {int(SV.crossed_midnight.sum())}")

    # ---- overlaps ----------------------------------------------------------
    ov = []
    for sub, g in SV.sort_values("start_local").groupby("subject"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i + 1, "start_local"] < g.loc[i, "end_local"]:
                ov.append({
                    "subject": sub,
                    "a_start": g.loc[i, "start_local"], "a_end": g.loc[i, "end_local"],
                    "a_lvl": g.loc[i, "Stress level"],
                    "b_start": g.loc[i + 1, "start_local"],
                    "b_end": g.loc[i + 1, "end_local"],
                    "b_lvl": g.loc[i + 1, "Stress level"],
                })
    OV = pd.DataFrame(ov)
    print(f"\n--- overlapping event pairs within subject: {len(OV)} ---")
    if len(OV):
        print(OV.to_string(index=False))

    # The mixed-dtype ID column (str for '5C', int for 15) cannot be written to
    # parquet as-is. This is trap 4 from the exploration notebook, live.
    SV_out = SV.copy()
    for c in SV_out.columns:
        if SV_out[c].dtype == object and c not in ("subject",):
            SV_out[c] = SV_out[c].astype(str)
    SV_out.to_parquet(os.path.join(OUT, "a3_survey.parquet"))

    # AUDIT.md expects 34 events over 60 min. Resolve which population that is.
    print("\n--- events over 60 min, by population ---")
    for label, pop in [("rated only (n=%d)" % len(R), R),
                       ("all events (n=%d)" % len(SV), SV)]:
        print(f"  {label:24} over 60 min: {int((pop.duration_min > 60).sum()):3d}"
              f"   max: {pop.duration_min.max():.0f} min")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = {0.0: "#3F7D62", 1.0: "#B8762A", 2.0: "#A8443C"}
    for lvl, g in R.groupby("Stress level"):
        x = np.sort(g.duration_min.values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=1.8,
                color=colors.get(lvl, "#5B6B7A"),
                label=f"{LEVEL_NAME.get(lvl, lvl)}  (n={len(g)})")
    ax.axvline(60, color="#17212B", ls="--", lw=1)
    ax.annotate(f"60 min\n{n_over60} events above", (60, .12), fontsize=8.5,
                xytext=(6, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    style_axes(ax, "A3 · Event duration ECDF by reported stress level",
               "event duration (minutes, log scale)",
               "fraction of events at or below")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A3_duration_ecdf_by_level.png"), dpi=150)
    plt.close(fig)

    # ---- expectations ------------------------------------------------------
    banner("A3 EXPECTED vs OBSERVED")
    lv = SV["Stress level"].value_counts()
    checks = [
        ("survey rows", len(SV), 358),
        ("exact duplicates", n_exact, 3),
        ("unrated", n_unrated, 113),
        ("level 0 (low)", int(lv.get(0.0, 0)), 46),
        ("level 1 (medium)", int(lv.get(1.0, 0)), 20),
        ("level 2 (high)", int(lv.get(2.0, 0)), 179),
        ("events > 60 min", n_over60, 34),
        ("max duration (min)", round(R.duration_min.max()), 323),
        ("overlapping pairs", len(OV), 12),
    ]
    rows = []
    for q, o, e in checks:
        m = "yes" if o == e else "NO"
        print(f"  {q:22} observed {o:>6}   expected {e:>6}   {m}")
        rows.append((q, o, e, m))

    append_findings(
        "A3", "Survey audit", "tasks/audit_a3.py", rows,
        ["figures/audit/A3_duration_ecdf_by_level.png"],
        "Fixes the label inventory and the effective positive count. Feeds JC06 "
        "(unrated events), JC07 (level encoding), JC08 (overlaps), JC10 (max duration).",
        "none" if all(r[3] == "yes" for r in rows) else
        "; ".join(f"{r[0]}: {r[1]} vs {r[2]}" for r in rows if r[3] == "NO"),
    )


if __name__ == "__main__":
    main()
