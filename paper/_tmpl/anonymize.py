"""Anonymise main.tex for double-blind submission and scan for leaks.

NewInML at NeurIPS 2026 is double-blind via OpenReview and requires fully
anonymised submissions. Two changes are needed:

  1. The [preprint] option produces a NON-anonymous document with a
     "Preprint. Work in progress." footer. For an anonymous submission the
     style file takes NO option, which makes it emit the placeholder
     "Anonymous Author(s) / Affiliation / Address / email" block.

  2. The author block itself must carry no names, emails or institution.

The scan afterwards is the part that matters. Anonymisation failures are
usually not the author block, which is obvious, but a stray acknowledgement,
a repository URL, or a self-citation written in the first person.
"""
import io
import os
import re

SRC = os.path.join("paper", "main.tex")

ANON_AUTHOR = r"""\author{%
  Anonymous Author(s) \\
  Affiliation \\
  Address \\
  \texttt{email} \\
}"""

# Strings that must not survive into a double-blind submission.
LEAKS = [
    "Elvis", "Han", "Eric", "Guan", "ehan20", "guan8zhi",
    "Johns Hopkins", "jh.edu", "gmail",
    "github.com", "ElvisHan2022", "nurse-stress-analysis",
]


def main():
    s = io.open(SRC, encoding="utf-8").read()

    # 1. anonymous build: no style option
    before = s
    s = s.replace("\\usepackage[preprint]{neurips_2026}",
                  "\\usepackage{neurips_2026}")
    print("  style option -> anonymous" if s != before
          else "  style option  already anonymous")

    # 2. author block
    start = s.find("\\author{%")
    if start != -1:
        end = s.find("\n}", start) + 2
        s = s[:start] + ANON_AUTHOR + s[end:]
        print("  author block -> anonymised")

    # 3. acknowledgements name nobody, but the section itself can be a tell
    s = s.replace(
        "\\subsubsection*{Acknowledgments}\n"
        "We thank the authors of the source dataset for making it publicly "
        "available.",
        "% Acknowledgments omitted for double-blind review.")

    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)

    # 4. the scan
    print("\n  leak scan:")
    found = False
    for token in LEAKS:
        hits = [i + 1 for i, line in enumerate(s.split("\n"))
                if token.lower() in line.lower()]
        if hits:
            found = True
            print(f"    LEAK  {token!r} on line(s) {hits}")
    if not found:
        print("    clean, no identifying string found")

    # 5. confirm the anonymous option really took
    assert "[preprint]" not in s, "preprint option still present"
    assert "[final]" not in s, "final option present"
    print("\n  style options verified absent (anonymous build)")


if __name__ == "__main__":
    main()
