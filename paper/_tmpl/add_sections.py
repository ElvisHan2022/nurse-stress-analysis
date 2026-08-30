"""Add the two short sections the NeurIPS checklist requires evidence for.

Compute resources (checklist Q8) and broader impacts (Q10). Both are genuinely
warranted here rather than box-ticking: the compute figure is small enough to
be worth stating, and the application is workplace monitoring of employees,
which has a real misuse path the paper should name.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

COMPUTE = r"""
\paragraph{Compute.} All experiments run on a single laptop CPU with no GPU.
Feature extraction over 387 sessions takes \SI{107}{\second}; the complete set
of reported experiments, including the twenty-draw permutation null and the
four-arm ratio sensitivity, completes in under an hour. The audit that precedes
them reads the \SI{3.5}{\giga\byte} archive once per section and is likewise
CPU-bound. No experiment reported here was preceded by a larger unreported
sweep.
"""

IMPACTS = r"""
\section{Broader impacts}

The application is monitoring of employees during work, and that is worth
naming plainly. A reliable stress detector deployed in a hospital could direct
support toward staff who need it, which is the motivating case. The same
instrument could as easily support surveillance, scheduling penalties, or
insurance decisions, and the people measured are in a weak position to refuse.
Consent to be monitored by an employer is not freely given in the way consent
to a research study is.

Our result bears on this directly. A detector recovering one episode in ten at
a deployable alarm rate is not fit for any of those uses, including the
benevolent one, and two of ten participants are undetectable at any usable
threshold. We would regard deployment on the strength of results at this level
as harmful, and we note that a published figure of $F_1 = 0.99$ on this same
dataset could be read as licensing exactly that. Reporting the achievable
number honestly is the contribution most relevant to impact here.

The cohort is ten female nurses at one hospital during a pandemic. Any
performance claim generalised beyond that population would be unsupported, and
differential performance across groups cannot be assessed at this sample size.
"""


def main():
    s = io.open(SRC, encoding="utf-8").read()

    # Compute paragraph goes at the end of the baseline section, before the
    # subsection on controls.
    anchor = "\\subsection{Two controls}"
    if anchor in s and "\\paragraph{Compute.}" not in s:
        s = s.replace(anchor, COMPUTE.strip() + "\n\n" + anchor)
        print("  compute paragraph added")
    else:
        print("  compute paragraph skipped")

    # Broader impacts goes after Limitations, before Conclusion.
    anchor2 = "\\section{Conclusion}"
    if anchor2 in s and "\\section{Broader impacts}" not in s:
        s = s.replace(anchor2, IMPACTS.strip() + "\n\n" + anchor2)
        print("  broader impacts section added")
    else:
        print("  broader impacts skipped")

    # siunitx was removed from the preamble, so inline these units too.
    s = s.replace(r"\SI{107}{\second}", "107 s")
    s = s.replace(r"\SI{3.5}{\giga\byte}", "3.5 GB")

    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)
    assert "\\SI{" not in s, "stray \\SI call remains"
    print("  units inlined, no \\SI remaining")


if __name__ == "__main__":
    main()
