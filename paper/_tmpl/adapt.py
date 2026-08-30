"""Point main.tex at the correct NeurIPS 2026 style file and remove the
dependencies that a minimal TeX tree may not carry.

Run from the repository root:  python paper/_tmpl/adapt.py
"""
import io
import os
import re

SRC = os.path.join("paper", "main.tex")

OLD_PRE = r"""\usepackage[preprint]{neurips_2025}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{siunitx}"""

NEW_PRE = r"""% The only supported style file for NeurIPS 2026 is neurips_2026.sty,
% obtainable from neurips.cc. Options: final, preprint, nonatbib.
% Use [preprint] for arXiv, no option for anonymous submission.
\usepackage[preprint]{neurips_2026}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}   % required by the instructions for table rules
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{graphicx}"""

# unit, replacement suffix. siunitx is dropped so the file builds on a
# minimal tree; the instructions do not require it.
UNITS = [
    (r"\percent",           r"\%"),
    (r"\hour",              r" hours"),
    (r"\hertz",             r" Hz"),
    (r"\second",            r" s"),
    (r"\micro\siemens",     r" $\mu$S"),
]

SI_RE = re.compile(r"\\SI\{([^}]*)\}\{([^}]*)\}")


def si_sub(m):
    val, unit = m.group(1), m.group(2)
    for pat, rep in UNITS:
        if unit == pat:
            return val + rep
    return val + " " + unit


def main():
    s = io.open(SRC, encoding="utf-8").read()

    if OLD_PRE in s:
        s = s.replace(OLD_PRE, NEW_PRE)
        print("  preamble  -> neurips_2026")
    else:
        print("  preamble  already adapted, skipped")

    n_si = len(SI_RE.findall(s))
    s = SI_RE.sub(si_sub, s)
    print(f"  units     -> {n_si} \\SI calls inlined")

    n_doi = len(re.findall(r"\\doi\{", s))
    s = re.sub(r"\\newblock \\doi\{([^}]*)\}\.", r"\\newblock DOI: \1.", s)
    print(f"  doi       -> {n_doi} \\doi calls inlined")

    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)

    leftover_si = len(SI_RE.findall(s))
    leftover_doi = len(re.findall(r"\\doi\{", s))
    print(f"\n  remaining \\SI:  {leftover_si}")
    print(f"  remaining \\doi: {leftover_doi}")
    assert leftover_si == 0 and leftover_doi == 0, "adaptation incomplete"
    print("  ok")


if __name__ == "__main__":
    main()
