"""Fold Phases 4, 5 and 6 into the paper.

The draft was written when pretraining had not been run. Section 7.2 speculated
about what it would need; that section is now replaced by what it did. The
abstract and Section 7.1 both said the representation axis was untested, and
both change.

Net space: Section 7.2 was 16 lines of speculation and becomes a results
section with a table, so most of the cost is absorbed. There was one page of
headroom after the appendix move.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

# ---------------------------------------------------------------- abstract --
OLD_ABS = """data, and a natural pretext task. We never got there, and this paper is an
account of why that is the more useful result. Pretraining requires two things
this dataset does not supply. It requires a negative class, and no participant
ever recorded being unstressed: all 358 survey entries mark an episode a nurse
flagged, and the rating is its severity. It also requires a baseline worth
improving on. Establishing both consumed the study."""

NEW_ABS = """data, and a natural pretext task. Establishing the prerequisites consumed the
study, and then pretraining did not help. Pretraining requires a negative
class, and no participant ever recorded being unstressed: all 358 survey
entries mark an episode a nurse flagged, and the rating is its severity."""

# ------------------------------------------------------------ abstract tail --
OLD_TAIL = """treatments of the labels, including non-negative positive-unlabelled risk
estimation, all fail to improve on it. We stopped at a threshold fixed in
advance. Two findings run against expectation: careful negative construction
improves rather than inflates performance, and a published accuracy figure on
this dataset is difficult to reconcile with subject-grouped evaluation."""

NEW_TAIL = """treatments of the labels, including non-negative positive-unlabelled risk
estimation, all fail to improve on it. Masked-reconstruction pretraining over
1,238 hours then fails as well: a pretrained encoder matches a randomly
initialised one at every label count from 25 to 138 episodes, and both trail
the hand-crafted baseline by $0.11$ AUC. Severity prediction, which needs no
constructed negatives at all, is also at chance. Two findings run against
expectation: careful negative construction improves rather than inflates
performance, and a published accuracy figure on this dataset is difficult to
reconcile with subject-grouped evaluation."""

# --------------------------------------------------------------- section 7.1 --
OLD_71 = """\\textbf{What we have eliminated is the label-treatment axis, not the
representation axis.} Our decision rule tested whether a different way of
\\emph{using} the labels helps. Self-supervised pretraining asks a different
question: whether a different \\emph{representation} of the signal helps. The
rule we fixed does not settle that, and we state this rather than treating the
null as broader than it is.

What we can say is narrower and still useful. Within the feature set and
learner space we searched, the results do not move; the constraint is not
capacity, not the comparison group, and not the pipeline. The remaining
candidates are the representation and the data. The data-side candidates are
concrete: 138 episodes from 10 participants; labels that are retrospective
self-reports of intervals, treated here as uniform states although a
thirty-minute report is unlikely to describe thirty uniform minutes; a
constructed rather than observed negative class; and one responsive channel out
of four."""

NEW_71 = """Section \\ref{sec:ssl} closes the remaining axis. A learned representation does
not move the result either, and neither does severity prediction, which needs
no constructed negatives at all.

What remains is the data: 138 episodes from 10 participants; labels that are
retrospective self-reports of intervals, treated here as uniform states
although a thirty-minute report is unlikely to describe thirty uniform
minutes; a constructed rather than observed negative class; and one responsive
channel out of four."""

# ------------------------------- section 7.2 becomes the Phase 4-6 results ----
OLD_72 = """\\subsection{What this implies for the pretraining we did not run}

A label-efficiency comparison remains the informative experiment, and it is
worth stating what it would need. The measurement is not whether a pretrained
encoder beats hand-crafted features outright, which is unlikely at 138
episodes, but whether it reaches a given level with fewer of them. That curve
would separate a label-supply constraint from a representation constraint
directly.

Two preconditions follow from our results. The participant-identity probe must
be compared against $0.768$, the accuracy reachable from sensible hand-crafted
features, and not against chance; an encoder scoring near $0.95$ would have
learned identity rather than physiology. And pretraining on all participants
before a leave-one-participant-out evaluation is transductive, so it must
either be re-run inside each fold or be labelled as an upper bound.

"""

NEW_SSL = """"""   # section moved before the Discussion; see INSERT below

# ---------------------------------------------------------- new section 7 ----
INSERT = """\\section{Does a learned representation help?}
\\label{sec:ssl}

The decision rule in Section \\ref{sec:pu} tested how the labels are
\\emph{used}. Self-supervision asks a different question: whether a different
\\emph{representation} helps. We ran it, past our own stopping point, because
the two axes are not the same and a null on one does not settle the other.

\\paragraph{Pretraining.} A dilated 1D convolutional encoder is trained by
masked reconstruction over all 609 sessions and all 15 participants, including
the five excluded from the supervised cohort, since the pretext task needs no
labels. The corpus is 1,238 usable hours at 1 Hz across five causally
normalised channels, yielding 68k windows. Mask spans of 30 to 120 s are fixed
a priori rather than tuned on downstream performance, which would leak through
the selection process. Contrastive learning was rejected deliberately: its
standard augmentations include amplitude scaling, and electrodermal amplitude
is the signal Section \\ref{sec:labels} identified.

\\paragraph{The identity gate.} A linear probe on frozen embeddings recovers
participant identity at $0.168$, against chance $0.067$ and the $0.768$
reachable from hand-crafted features. The encoder did not learn identity, so
the phase proceeds. The competing reading, that $0.168$ indicates a weak
representation rather than a clean one, is settled by the sweep below.

\\begin{table}[t]
\\caption{Label-efficiency sweep. Identical folds and windows to Table
\\ref{tab:baseline}. Training positives are restricted to a sampled subset of
episodes; the test fold is never subsampled. All values are transductive upper
bounds, since the encoder saw every participant during pretraining.}
\\label{tab:ssl}
\\centering
\\begin{tabular}{lrrrr}
\\toprule
Episodes used & Linear probe & Full fine-tune & Random init & Probe $-$ random \\\\
\\midrule
25  & 0.6157 & 0.6382 & 0.6398 & $-0.024$ \\\\
50  & 0.6342 & 0.6438 & 0.6367 & $-0.003$ \\\\
100 & 0.6486 & 0.6602 & 0.6604 & $-0.012$ \\\\
138 & 0.6467 & 0.6570 & 0.6495 & $-0.003$ \\\\
\\midrule
\\multicolumn{4}{l}{\\emph{Hand-crafted baseline, Section \\ref{sec:baseline}}}
  & \\textbf{0.7654} \\\\
\\bottomrule
\\end{tabular}
\\end{table}

\\paragraph{No label efficiency, and no benefit.} Table \\ref{tab:ssl} gives
two negatives. Pretrained minus randomly initialised is between $-0.024$ and
$-0.003$ at every label count, so there is no separation at low counts, which
was the entire hypothesis. And every arm trails the hand-crafted baseline by
roughly $0.11$ AUC.

That the randomly initialised arm matches the pretrained one is the
informative part. The architecture is not the problem, reaching $0.65$ from
random weights; the pretext task simply added nothing to it. This also
resolves the ambiguity in the identity probe: the representation is weak
rather than clean. And because the encoder saw every participant, these are
upper bounds, so the honest comparison is less favourable still.

\\paragraph{Severity, where no negatives are constructed.} Detection could in
principle be blamed on our constructed comparison group. Severity cannot: it is
a three-level ordinal task over the 186 rated episodes with sensor coverage,
and it needs no negatives at all. We fit cumulative binary links rather than a
three-class softmax, because 17 medium-severity episodes across the cohort and
six of ten participants with none make the middle class unlearnable, and
because the rating scales are not comparable between people.

Both links sit at chance: $0.4964$ for severity $\\geq 1$ and $0.5417$ for
severity $\\geq 2$, with per-participant values from $0.300$ to $0.808$. Nor is
severity a restatement of episode length, since the point-biserial correlation
with duration is $-0.03$ and $-0.06$, neither significant. The one task in this
study whose numbers are not conditional on a construction fails too, which
removes the negative construction as the explanation for the rest.

"""

# ---------------------------------------------------------------- conclusion --
OLD_CONC = """We report this as a result rather than a failure, and we are explicit about its
boundary: we have eliminated the label-treatment axis, not the representation
axis. What a future study most needs from this one is not a score but a
measurement of what was missing, and the most useful next experiment is a
label-efficiency curve rather than another point estimate."""

NEW_CONC = """Pretraining over 1,238 unlabelled hours does not move it either, and neither
does severity prediction, which needs no constructed negatives at all. We
report this as a result rather than a failure. What a future study most needs
from this one is not a score but a measurement of what was missing, and the
answer is labels: explicit non-stress intervals, finer onset resolution, and
electrodermal signal quality good enough that a participant's only responsive
channel is not flat."""


def main():
    s = io.open(SRC, encoding="utf-8").read()
    n = 0
    for name, old, new in [
        ("abstract head", OLD_ABS, NEW_ABS),
        ("abstract tail", OLD_TAIL, NEW_TAIL),
        ("section 7.1", OLD_71, NEW_71),
        ("section 7.2 removed", OLD_72, ""),
        ("conclusion", OLD_CONC, NEW_CONC),
    ]:
        if old in s:
            s = s.replace(old, new)
            n += 1
            print(f"  {name:24} applied")
        else:
            print(f"  {name:24} NOT FOUND")

    # new section goes immediately before the Discussion
    anchor = "\\section{Discussion}"
    if anchor in s and "\\label{sec:ssl}" not in s:
        s = s.replace(anchor, INSERT + anchor)
        n += 1
        print("  ssl section             inserted before Discussion")

    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)
    print(f"\n  {n}/6 edits applied")


if __name__ == "__main__":
    main()
