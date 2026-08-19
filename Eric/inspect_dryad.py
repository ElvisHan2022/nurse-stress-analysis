#!/usr/bin/env python3
"""
inspect_dryad.py

Dumps cheap structural metadata from the unzipped Dryad nurse-stress dataset
(doi:10.5061/dryad.5hqbzkh6f) so it can be reviewed without uploading GBs of
signal data. Reads only file sizes, line counts, and the first/last few lines
of each signal file. No sample data beyond ~8 lines per file leaves your disk.

Usage:
    pixi run python inspect_dryad.py /path/to/unzipped/Stress_dataset

Writes ./dataset_manifest.txt (typically a few hundred KB of plain text).
Stdlib only -- no dependencies.
"""

import os
import sys

OUT = "dataset_manifest.txt"
SIGNALS = ["EDA.csv", "HR.csv", "TEMP.csv", "IBI.csv", "BVP.csv", "ACC.csv",
           "tags.csv", "info.txt"]
MAXLEN = 300          # truncate very long lines
HEAD_N = 6            # lines from the top of each signal file
TAIL_N = 2            # lines from the bottom
N_SAMPLE_SESSIONS = 2  # inspect this many session folders, from different subjects


def head(path, n=HEAD_N):
    out = []
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.append(line.rstrip("\n")[:MAXLEN])
    return out


def tail_and_count(path, n=TAIL_N):
    """Single pass: total line count plus the last n lines."""
    count = 0
    buf = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            count += 1
            buf.append(line.rstrip("\n")[:MAXLEN])
            if len(buf) > n:
                buf.pop(0)
    return count, buf


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = os.path.abspath(sys.argv[1])
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)

    with open(OUT, "w") as o:
        def w(s=""):
            o.write(str(s) + "\n")

        w("# dataset_manifest.txt")
        w(f"# root: {root}")
        w()

        # ---- Section 1: full file manifest -----------------------------
        # Per-file sizes across every subject double as the coverage census:
        # hours of signal per subject is roughly proportional to EDA.csv size.
        w("=== 1. FILE MANIFEST (relative_path,bytes) ===")
        n_files = 0
        total_bytes = 0
        ext_counts = {}
        sessions = []          # dirs containing an EDA.csv
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()
            if "EDA.csv" in filenames:
                sessions.append(dirpath)
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    sz = -1
                w(f"{rel},{sz}")
                n_files += 1
                total_bytes += max(sz, 0)
                ext = os.path.splitext(fn)[1].lower() or "(none)"
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
        w()
        w(f"# total files: {n_files}")
        w(f"# total bytes: {total_bytes}")
        w(f"# extensions:  {dict(sorted(ext_counts.items()))}")
        w(f"# session folders (containing EDA.csv): {len(sessions)}")
        w()

        # ---- Section 2: top-level layout -------------------------------
        w("=== 2. TOP-LEVEL ENTRIES ===")
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            kind = "DIR " if os.path.isdir(p) else "FILE"
            w(f"{kind} {name}")
        w()

        # ---- Section 3: sample sessions --------------------------------
        # Pick sessions from *different* top-level subject folders so schema
        # consistency across subjects can be checked.
        picked = []
        seen_subjects = set()
        for s in sessions:
            rel = os.path.relpath(s, root)
            subject = rel.split(os.sep)[0]
            if subject not in seen_subjects:
                seen_subjects.add(subject)
                picked.append(s)
            if len(picked) >= N_SAMPLE_SESSIONS:
                break

        if not picked:
            w("!! No EDA.csv found anywhere under the path provided.")
            w("!! Check that you passed the unzipped Stress_dataset directory.")
        for s in picked:
            w(f"=== 3. SAMPLE SESSION: {os.path.relpath(s, root)} ===")
            w(f"    files present: {sorted(os.listdir(s))}")
            w()
            for fn in SIGNALS:
                p = os.path.join(s, fn)
                if not os.path.exists(p):
                    w(f"--- {fn}: ABSENT ---")
                    w()
                    continue
                n_lines, tl = tail_and_count(p)
                w(f"--- {fn} ({os.path.getsize(p)} bytes, {n_lines} lines) ---")
                w(f"  first {HEAD_N} lines:")
                for L in head(p):
                    w("    " + L)
                w(f"  last {TAIL_N} lines:")
                for L in tl:
                    w("    " + L)
                w()

    print(f"Wrote {OUT} ({os.path.getsize(OUT)} bytes)")
    print("Review it, then upload it.")


if __name__ == "__main__":
    main()
