"""Shared helpers for the AUDIT.md sections.

Read-only with respect to the archive. Nothing here writes into Eric/Stress_dataset.
"""
from __future__ import annotations

import os
import re
import glob
import datetime as dt

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "Eric", "Stress_dataset")
SURVEY = os.path.join(ROOT, "Eric", "SurveyResults.xlsx")
OUT = os.path.join(ROOT, "reports", "audit")
FIG = os.path.join(ROOT, "figures", "audit")
DERIVED = os.path.join(ROOT, "derived")

for _d in (OUT, FIG, DERIVED):
    os.makedirs(_d, exist_ok=True)

SIGNALS = ["ACC", "BVP", "EDA", "HR", "TEMP"]      # sampled files with the 2-row header
ALL_FILES = SIGNALS + ["IBI", "tags"]

LOCAL_TZ = "America/Chicago"

# Candidate analysis rates the pipeline could target (Hz).
CANDIDATE_RATES = [1, 4, 8, 16, 32]


def read_e4_header(path):
    """Row 1 = session start (unix epoch, UTC). Row 2 = sample rate (Hz).

    ACC repeats each value three times, comma separated, so take field 0.
    """
    with open(path) as f:
        t0 = float(f.readline().split(",")[0])
        fs = float(f.readline().split(",")[0])
    return t0, fs


def count_data_rows(path):
    """Number of data rows, excluding the 2 header rows. Single streaming pass."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(n - 2, 0)


def read_signal(path, name):
    """EDA/TEMP/HR/BVP -> Series; ACC -> 3-col DataFrame in g. UTC-indexed."""
    t0, fs = read_e4_header(path)
    if name == "ACC":
        v = np.atleast_2d(np.loadtxt(path, skiprows=2, delimiter=","))
        idx = pd.to_datetime(t0 + np.arange(len(v)) / fs, unit="s", utc=True)
        return pd.DataFrame(v / 64.0, index=idx, columns=["acc_x", "acc_y", "acc_z"])
    v = np.atleast_1d(np.loadtxt(path, skiprows=2))
    idx = pd.to_datetime(t0 + np.arange(len(v)) / fs, unit="s", utc=True)
    return pd.Series(v, index=idx, name=name.lower())


def read_ibi(path):
    """Row 1 = '<t0>, IBI' (text). Later rows: elapsed_s_since_t0, interval_s."""
    with open(path) as f:
        t0 = float(f.readline().split(",")[0])
    try:
        v = np.atleast_2d(np.loadtxt(path, skiprows=1, delimiter=","))
    except Exception:
        return pd.Series(dtype=float, name="ibi")
    if v.size == 0:
        return pd.Series(dtype=float, name="ibi")
    return pd.Series(
        v[:, 1], name="ibi",
        index=pd.to_datetime(t0 + v[:, 0], unit="s", utc=True),
    )


def session_dirs(archive=ARCHIVE):
    """Every directory containing an EDA.csv, sorted."""
    return sorted(os.path.dirname(p)
                  for p in glob.glob(os.path.join(archive, "*", "*", "EDA.csv")))


# ---------------------------------------------------------------- survey ----

FACTORS = [
    "COVID related", "Treating a covid patient", "Patient in Crisis",
    "Patient or patient's family", "Doctors or colleagues",
    "Administration, lab, pharmacy, radiology, or other ancilliary services",
    "Increased Workload", "Technology related stress", "Lack of supplies",
    "Documentation", "Competency related stress",
    "Saftey (physical or physiological threats)",
    "Work Environment - Physical or others: work processes or procedures",
]


def load_survey(path=SURVEY, tz=LOCAL_TZ):
    """Survey events with both naive-local and UTC timestamps attached.

    Handles the documented traps: mixed-dtype ID, 'na' string sentinel,
    a column name ending in a newline, and events crossing midnight.
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df["subject"] = df["ID"].astype(str).str.strip()

    def mk(d, t):
        return pd.Timestamp(dt.datetime.combine(pd.Timestamp(d).date(), t))

    df["start_naive"] = [mk(d, t) for d, t in zip(df["date"], df["Start time"])]
    df["end_naive"] = [mk(d, t) for d, t in zip(df["date"], df["End time"])]
    df["crossed_midnight"] = df.end_naive < df.start_naive
    df.loc[df.crossed_midnight, "end_naive"] += pd.Timedelta(days=1)

    for c in ["start_naive", "end_naive"]:
        loc = df[c].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
        df[c.replace("_naive", "_local")] = loc
        df[c.replace("_naive", "_utc")] = loc.dt.tz_convert("UTC")

    for c in ["Stress level"] + [re.sub(r"\s+", " ", f).strip() for f in FACTORS]:
        if c in df:
            df[c] = pd.to_numeric(
                df[c].replace({"na": np.nan, "NA": np.nan}), errors="coerce")

    df["labelled"] = df["Stress level"].notna()
    df["duration_min"] = (df.end_utc - df.start_utc).dt.total_seconds() / 60
    df["is_dup"] = df.duplicated(
        ["subject", "date", "Start time", "End time"], keep=False)
    df["is_exact_dup"] = df.duplicated(
        ["subject", "date", "Start time", "End time", "Stress level"], keep="first")
    return df


# ------------------------------------------------------------- reporting ----

def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(name, obj, n=15):
    """Print an actual object, truncated by rows rather than described in prose."""
    print(f"\n--- {name} ---")
    if isinstance(obj, pd.DataFrame):
        with pd.option_context("display.width", 200, "display.max_columns", 60):
            print(obj.head(n).to_string())
            if len(obj) > n:
                print(f"... ({len(obj):,} rows total)")
    elif isinstance(obj, pd.Series):
        print(obj.head(n).to_string())
        if len(obj) > n:
            print(f"... ({len(obj):,} entries total)")
    else:
        print(obj)


def append_findings(section, name, script, numbers, figures, changes, surprises):
    """Append one block to reports/audit/findings.md using the AUDIT.md template."""
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n### {section} · {name}\n",
             f"**Ran:** `{script}` at {ts}\n",
             "**Numbers**\n",
             "| Quantity | Observed | Expected | Match |",
             "|---|---|---|---|"]
    for q, o, e, m in numbers:
        lines.append(f"| {q} | {o} | {e} | {m} |")
    figs = ", ".join(f"`{f}`" for f in figures) if figures else "none"
    lines += [f"\n**Figures:** {figs}\n",
              f"**What this changes.** {changes}\n",
              f"**Surprises.** {surprises}\n"]
    with open(os.path.join(OUT, "findings.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=.3, linewidth=.6)
    ax.tick_params(labelsize=8)
