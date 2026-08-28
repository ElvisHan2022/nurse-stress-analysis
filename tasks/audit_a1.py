"""A1 - Inventory and headers, plus the sample-rate feasibility analysis.

Question: do I have what I think I have, and at what rates?

Sampling rates are read from row 2 of each file. Published descriptions of this
dataset give BVP at 72 Hz and TEMP at 10 Hz, which contradicts the device
specification of 64 Hz and 4 Hz. The header settles it for this extraction.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_common import (ARCHIVE, OUT, ALL_FILES, SIGNALS, CANDIDATE_RATES,
                          read_e4_header, session_dirs, banner, show,
                          append_findings)


def main():
    banner("A1 - INVENTORY AND HEADERS")

    sessions = session_dirs()
    subjects = sorted({os.path.basename(os.path.dirname(s)) for s in sessions})

    print(f"\nsubjects ({len(subjects)}): {' '.join(subjects)}")
    print(f"session directories: {len(sessions)}")

    rows = []
    for sd in sessions:
        subject = os.path.basename(os.path.dirname(sd))
        rec = {"subject": subject, "session": os.path.basename(sd), "path": sd}
        for name in ALL_FILES:
            p = os.path.join(sd, f"{name}.csv")
            rec[f"has_{name}"] = os.path.exists(p)
            rec[f"bytes_{name}"] = os.path.getsize(p) if os.path.exists(p) else 0
        for name in SIGNALS:
            p = os.path.join(sd, f"{name}.csv")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    t0, fs = read_e4_header(p)
                    rec[f"t0_{name}"], rec[f"fs_{name}"] = t0, fs
                except Exception as e:
                    rec[f"t0_{name}"], rec[f"fs_{name}"] = np.nan, np.nan
                    print(f"  [warn] header unreadable: {sd}/{name}.csv ({e})")
            else:
                rec[f"t0_{name}"], rec[f"fs_{name}"] = np.nan, np.nan
        rows.append(rec)

    INV = pd.DataFrame(rows)
    INV.to_parquet(os.path.join(OUT, "a1_inventory.parquet"))

    show("inventory.head()", INV[["subject", "session"] +
                                 [f"fs_{s}" for s in SIGNALS]], 10)

    # ---- file presence -----------------------------------------------------
    print("\n--- fraction of sessions containing each file ---")
    pres = pd.Series({n: INV[f"has_{n}"].mean() for n in ALL_FILES})
    empty = pd.Series({n: (INV[f"bytes_{n}"] == 0).mean() for n in ALL_FILES})
    P = pd.DataFrame({"present": pres, "empty_if_present": empty}).round(4)
    print(P.to_string())

    # ---- observed sampling rates ------------------------------------------
    print("\n--- observed distinct sampling rates per signal (from row 2) ---")
    rate_rows = []
    for s in SIGNALS:
        vc = INV[f"fs_{s}"].dropna().value_counts()
        rate_rows.append({
            "signal": s,
            "n_distinct": int(vc.size),
            "rates": ", ".join(f"{k:g}Hz x{v}" for k, v in vc.items()),
        })
    RATES = pd.DataFrame(rate_rows)
    print(RATES.to_string(index=False))

    multi_rate = RATES[RATES.n_distinct > 1]

    # ---- t0 spread within a session ---------------------------------------
    print("\n--- t0 spread across signals within a session (seconds) ---")
    t0cols = [f"t0_{s}" for s in SIGNALS]
    spread = INV[t0cols].max(axis=1) - INV[t0cols].min(axis=1)
    print(spread.describe().round(3).to_string())
    print(f"\nsessions with nonzero t0 spread: {int((spread > 0).sum())} / {len(INV)}")

    print("\n--- t0 offset of each signal relative to EDA (seconds) ---")
    off = pd.DataFrame({s: (INV[f"t0_{s}"] - INV["t0_EDA"]) for s in SIGNALS})
    print(off.describe().loc[["mean", "min", "50%", "max"]].round(3).to_string())

    # ---- sample-rate feasibility ------------------------------------------
    # Which candidate analysis rates are supported natively by which channel.
    banner("A1b - ANALYSIS RATE FEASIBILITY")
    native = {s: float(INV[f"fs_{s}"].dropna().mode().iloc[0]) for s in SIGNALS}
    print("\nnative rates observed:", {k: f"{v:g}Hz" for k, v in native.items()})

    feas = []
    for target in CANDIDATE_RATES:
        rec = {"target_Hz": target}
        for s in SIGNALS:
            nat = native[s]
            if nat >= target:
                rec[s] = "downsample" if nat > target else "native"
            else:
                rec[s] = f"UPSAMPLE x{target/nat:g}"
        rec["channels_real"] = sum(native[s] >= target for s in SIGNALS)
        feas.append(rec)
    FEAS = pd.DataFrame(feas)
    print("\n--- what each candidate analysis rate can actually carry ---")
    print(FEAS.to_string(index=False))

    print("\nReading: 'native' means the file is already at that rate. 'downsample'")
    print("means real information is being aggregated away but every output sample")
    print("is backed by measurements. 'UPSAMPLE' means the channel is being")
    print("forward-filled or interpolated - the extra samples carry no new")
    print("information and any within-window variance they produce is an artifact.")

    # Cost of each rate, in samples per worn hour across the 5 sampled signals.
    print("\n--- storage/compute cost per session-hour at each rate ---")
    cost = []
    for target in CANDIDATE_RATES:
        per_hour = target * 3600 * (len(SIGNALS) + 2)   # ACC contributes 3 axes
        cost.append({"target_Hz": target,
                     "samples_per_hour_all_channels": f"{per_hour:,}",
                     "rel_to_1Hz": f"{target:g}x"})
    print(pd.DataFrame(cost).to_string(index=False))

    json.dump({"native_rates": native,
               "candidate_rates": CANDIDATE_RATES,
               "feasibility": FEAS.to_dict("records")},
              open(os.path.join(OUT, "a1_rates.json"), "w"), indent=2)

    # ---- stop condition ----------------------------------------------------
    stop = []
    if len(multi_rate):
        stop.append(f"multiple distinct rates for: {list(multi_rate.signal)}")
    if int((spread > 0).sum()) > 0:
        stop.append(f"{int((spread > 0).sum())} sessions have nonzero t0 spread")

    banner("A1 STOP CONDITION")
    if stop:
        print("TRIGGERED:")
        for s in stop:
            print("  -", s)
    else:
        print("not triggered")

    append_findings(
        "A1", "Inventory and headers",
        "tasks/audit_a1.py",
        [("subjects", len(subjects), 15, "yes" if len(subjects) == 15 else "no"),
         ("session directories", len(sessions), 609,
          "yes" if len(sessions) == 609 else "no"),
         ("distinct rates per signal", "; ".join(
             f"{r.signal}={r.n_distinct}" for r in RATES.itertuples()),
          "1 each", "yes" if not len(multi_rate) else "no"),
         ("sessions with nonzero t0 spread", int((spread > 0).sum()), 0,
          "no" if int((spread > 0).sum()) else "yes"),
         ("HR t0 offset vs EDA (median s)", f"{off['HR'].median():.1f}",
          "10 (documented)", "see notes")],
        ["(none - tabular section)"],
        "Fixes the native rate of every channel, which sets the ceiling for the "
        "analysis-rate choice. Feeds the resampling decision (JC14).",
        "; ".join(stop) if stop else "none",
    )


if __name__ == "__main__":
    main()
