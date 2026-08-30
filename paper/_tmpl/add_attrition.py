"""Add the attrition table and expand the introduction's context.

The attrition table is the single element that most helps a reader follow the
study design: each row removes something for a stated reason, and the three
totals at the bottom make the denominator argument visible rather than
asserted.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

OLD_ANCHOR = ("\\textbf{What survives is 10 participants, 24{,}334 windows, "
              "and 138\nhigh-severity episodes.} Of the three counts")

TABLE = r"""\begin{table}[t]
\caption{What the study is left with. Each row removes something for a stated
reason, and the count is what the next stage actually operates on. The three
totals are the same data described three ways; only two of them are
denominators.}
\label{tab:attrition}
\centering
\begin{tabular}{llr}
\toprule
Step & Reason & Remaining \\
\midrule
Recording sessions            & as deposited                     & 609 \\
\quad after the length filter & shorter than one analysis window & 517 \\
\addlinespace
Survey rows                   & as deposited                     & 358 \\
\quad rated                   & a severity was recorded          & 245 \\
\quad high severity only      & the binary task                  & 179 \\
\quad after merging overlaps  & twelve overlapping pairs         & 153 \\
\quad with sensor coverage    & a recording exists at that time  & \textbf{138} \\
\addlinespace
Participants                  & as deposited                     & 15 \\
\quad after exclusions        & no negatives, or a flat channel  & \textbf{10} \\
\midrule
\multicolumn{2}{l}{Windows, a compute statistic} & 24{,}334 \\
\multicolumn{2}{l}{\textbf{Episodes}, the denominator for detection} & \textbf{138} \\
\multicolumn{2}{l}{\textbf{Participants}, for generalising to a new person} & \textbf{10} \\
\bottomrule
\end{tabular}
\end{table}

"""

NEW_ANCHOR = ("\\textbf{What survives is 10 participants, 24{,}334 windows, "
              "and 138\nhigh-severity episodes} (Table \\ref{tab:attrition}). "
              "Of the three counts")

OLD_INTRO = """week at a time, and skin conductance in particular is driven almost entirely by
the sympathetic nervous system with little voluntary control."""

NEW_INTRO = """week at a time, and skin conductance in particular is driven almost entirely by
the sympathetic nervous system with little voluntary control, which is why it
has long been the channel of choice for laboratory work on arousal. Moving that
measurement from the laboratory onto a hospital ward changes the problem in
ways that are easy to underestimate. The wrist is in constant motion, and the
same physical exertion that raises heart rate and sweating during a stressful
episode raises them during an unremarkable one. Ground truth stops being an
experimenter-controlled stimulus and becomes whatever the participant remembers
to report, when she has a moment to report it."""


def main():
    s = io.open(SRC, encoding="utf-8").read()

    assert OLD_ANCHOR in s, "attrition anchor not found"
    s = s.replace(OLD_ANCHOR, TABLE + NEW_ANCHOR)
    print("  attrition table inserted")

    assert OLD_INTRO in s, "intro anchor not found"
    s = s.replace(OLD_INTRO, NEW_INTRO)
    print("  introduction context expanded")

    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)


if __name__ == "__main__":
    main()
