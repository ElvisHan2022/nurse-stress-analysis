# -*- coding: utf-8 -*-
r"""Drop the last editorialising bold: the stopping-rule verdict.

Neither model paper bolds a sentence of prose. The verdict carries itself.
"""
import io

SRC = "paper/main.tex"

OLD = "\\textbf{The rule was not met, and every alternative is\nworse than the baseline rather than merely insufficiently better.}"
NEW = "The rule was not met, and every alternative is\nworse than the baseline rather than merely insufficiently better."

s = io.open(SRC, encoding="utf-8").read()
if OLD in s:
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s.replace(OLD, NEW, 1))
    print("applied")
else:
    print("NOT FOUND; nearby text follows")
    i = s.find("The rule was not met")
    print(repr(s[i - 90:i + 130]))
