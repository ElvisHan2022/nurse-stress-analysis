"""Build a measurement-only copy of the paper to obtain a page count.

The official neurips_2026.sty is not present, so this substitutes a stub that
reproduces only the page geometry and type size. It also removes two things the
stub cannot support: microtype font expansion (needs scalable fonts) and the
tabular-based \\And author macro.

The output is a ruler. It is not the submission and must never be sent anywhere.
"""
import io
import os
import re

SRC = os.path.join("paper", "main.tex")
DST = os.path.join("paper", "_tmpl", "measure.tex")

FLAT_AUTHOR = "\\author{Elvis Han \\and Eric Guan \\\\ Johns Hopkins University}"


def main():
    s = io.open(SRC, encoding="utf-8").read()

    s = s.replace("\\usepackage[preprint]{neurips_2026}",
                  "\\usepackage{measure_only}")
    s = s.replace("\\usepackage{microtype}", "")

    # Replace the author block literally, avoiding regex escape handling.
    start = s.find("\\author{%")
    if start != -1:
        end = s.find("\n}", start)
        s = s[:start] + FLAT_AUTHOR + s[end + 2:]

    io.open(DST, "w", encoding="utf-8", newline="\n").write(s)
    print(f"wrote {DST}")
    for probe in ["measure_only", "microtype", "\\And"]:
        print(f"  {probe:14} present: {probe in s}")


if __name__ == "__main__":
    main()
