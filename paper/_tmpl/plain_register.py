# -*- coding: utf-8 -*-
r"""Move the paper into the register of the two model papers.

Vaswani et al. and Wang et al. both open with continuous prose: no bolded
paragraph labels inside the introduction, no bolded numbers, no rhetorical
signposting. The structure is carried by ordinary sentences in order, and the
significance is left to the facts. Attention Is All You Need bolds nothing in
its abstract and nothing in its introduction; its only inline labels are
descriptive nouns ("Encoder:", "Decoder:").

Changes:

1. The introduction loses all four \paragraph labels and becomes six flowing
   paragraphs. The contributions fold into the closing paragraph as a plain
   description of what each section does, which is what both model papers do.
2. The abstract loses its four \textbf spans and its sentence fragment opener.
3. Three sentence-level \textbf spans in the body become plain text. Bold that
   marks a definition stays, since that is the one use both papers make of it.
4. "The gap is the finding." was the one editorialising paragraph label left.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

OLD_ABSTRACT = r"""Nine percent of this dataset is labelled. That is the condition
self-supervised learning exists for, and it was what we set out to do: pretrain
on the unlabelled remainder, fine-tune on the little that is annotated. But a
pretext task cannot be evaluated without a supervised baseline to beat, and
this dataset supplies neither a baseline nor the negative class one would need,
because no nurse ever recorded being unstressed. All 358 survey entries mark an
episode a nurse flagged; the number attached is its severity, not its presence.
This paper is the attrition process required to construct both, reported as the
result. We start from 609 recordings and 15 participants and arrive at 138
episodes from 10, testing at each step whether what survived is real rather
than manufactured. The labels do mark a physiological state, but in one channel
only: skin conductance rises at reported onset in \textbf{64.3\%} of episodes
($p=0.0009$), while heart rate rises in \textbf{48.6\%}, which is what a coin
flip produces ($p=0.80$). The constructed comparison group survives
falsification, detecting neither movement nor shift schedule (both AUC
$0.51$), and the baseline built on it clears a within-participant permutation
null decisively ($0.765$ against $0.498\pm0.010$) while still recovering only
\textbf{10.6\%} of episodes at one false alarm per worn hour. We then ran the
pretraining anyway. Over 1,238 unlabelled hours it matches random
initialisation at every label count from 25 to 138 episodes, and both trail the
hand-crafted baseline by $0.11$ AUC. Severity prediction, the one task here
needing no constructed negatives, is also at chance. On this dataset the
constraint is the label supply, and pretraining as we implemented it does not
substitute for it."""

NEW_ABSTRACT = r"""Nine percent of this dataset is labelled, which is the condition
self-supervised learning exists for, and pretraining on the unlabelled
remainder was the experiment we set out to run. A pretext task cannot be
evaluated without a supervised baseline to beat, and this dataset supplies
neither a baseline nor the negative class one would need, because no nurse ever
recorded being unstressed. All 358 survey entries mark an episode a nurse
flagged, and the number attached is its severity, not its presence. This paper
reports the attrition required to construct both. We start from 609 recordings
and 15 participants and arrive at 138 episodes from 10, testing at each step
whether what survived is real rather than manufactured. The labels do mark a
physiological state, but in one channel only: skin conductance rises at
reported onset in 64.3\% of episodes ($p=0.0009$), while heart rate rises in
48.6\%, which is what a coin flip produces ($p=0.80$). The constructed
comparison group survives falsification, detecting neither movement nor shift
schedule (both AUC $0.51$), and the baseline built on it clears a
within-participant permutation null ($0.765$ against $0.498\pm0.010$) while
recovering only 10.6\% of episodes at one false alarm per worn hour.
Pretraining over 1,238 unlabelled hours then matches random initialisation at
every label count from 25 to 138 episodes, and both trail the hand-crafted
baseline by $0.11$ AUC. Severity prediction, the one task here needing no
constructed negatives, is also at chance. On this dataset the constraint is the
label supply, and pretraining as we implemented it does not substitute for it."""

OLD_INTRO_START = r"""Nurse attrition and clinical error are both costly, and occupational stress is
routinely implicated in each; it is the motivation the authors of this dataset
give for collecting it \cite{hosseini2022}. The case for measuring stress
continuously is therefore both an occupational-health and a patient-safety
argument. The instrument normally used, a retrospective questionnaire, is a
poor one: it is administered after the fact, it depends on recall, and it
cannot resolve when during a shift the stress occurred."""

NEW_INTRO_START = r"""Nurse attrition and clinical error are both costly, and occupational stress is
routinely implicated in each; it is the motivation the authors of this dataset
give for collecting it \cite{hosseini2022}. The instrument normally used to
measure that stress is a retrospective questionnaire, which is administered
after the fact, depends on recall, and cannot resolve when during a shift the
stress occurred."""

# The dataset sentence moves to the end of paragraph two, so the paragraph
# ends on the thing the rest of the paper uses.
OLD_P2_TAIL = r"""Ground truth stops being an
experimenter-controlled stimulus and becomes whatever the participant remembers
to report, when she has a moment to report it. A public dataset
now exists for testing the idea. Hosseini et al.\ \cite{hosseini2022} recorded
15 nurses for one week each during the COVID-19 outbreak, with periodic
smartphone-administered stress reports alongside the sensor streams."""

NEW_P2_TAIL = r"""Ground truth stops being an
experimenter-controlled stimulus and becomes whatever the participant remembers
to report, when she has a moment to report it. Hosseini et al.\
\cite{hosseini2022} recorded 15 nurses for one week each during the COVID-19
outbreak, with periodic smartphone-administered stress reports alongside the
sensor streams, and the data are public."""

# Everything from the first \paragraph to the end of the contributions list is
# replaced wholesale by continuous prose.
OLD_INTRO_REST = r"""\paragraph{What we set out to do.} Only \textbf{9.1\%} of the 1{,}251.7 recorded
hours falls inside a reported episode. The other \textbf{90.9\%} is unlabelled
physiology, which is precisely the regime self-supervised learning was
developed for: learn the structure of ordinary physiology from everything, then
fine-tune on the little that is annotated. Our intent was that experiment.

\paragraph{What stopped us.} Two prerequisites, neither of which the dataset
supplies. A pretext task is only meaningful against a supervised baseline worth
improving on, and no such baseline existed for this dataset that we could
trust. Building one requires a negative class, and there is none: every survey
row marks an episode a participant flagged, so the annotation records severity,
not presence. The comparison group has to be constructed from unlabelled time,
and constructing it defensibly turned out to be the entire study.

\paragraph{Why the groundwork is the contribution.} Published results on this
dataset are strong. Mathur et al.\ \cite{mathur2024} report a weighted $F_1$ of
$0.99$ for two-level classification. We do not reproduce anything close to it,
and the reason we think it is worth reporting our own attrition in full is that
each step of that attrition is a place where a number can be inflated without
anyone noticing: adjacent analysis windows are near-duplicates of one another,
the negative class is invented, and the effective sample size is 138 episodes
from 10 people rather than the 24{,}334 windows those episodes generate.

We therefore report the study as a sequence of eliminations. Does the label
mark anything physiological (Section \ref{sec:labels})? Does the constructed
comparison group leak a confound (Section \ref{sec:negatives})? How far do
standard features and a standard learner get, and can the pipeline manufacture
its own signal (Section \ref{sec:baseline})? Does a different treatment of the
labels help (Section \ref{sec:pu})? Does a learned representation (Section
\ref{sec:ssl})? Each answer narrows what is left, and what is left at the end
is the data.

\paragraph{Contributions.} An onset analysis showing that one of four channels
responds and the others do not, which reorders the feature priorities for this
sensor. A negative-class construction with falsification probes, and the
finding that careful construction \emph{improves} rather than inflates
performance, against the usual warning. A decision rule fixed before results
were seen, and the measurement that answered it. And a label-efficiency sweep
establishing that pretraining over 1{,}238 unlabelled hours buys nothing here,
reported with the comparison biased in its favour."""

NEW_INTRO_REST = r"""Only 9.1\% of the 1{,}251.7 recorded hours falls inside a reported episode. The
remaining 90.9\% is unlabelled physiology, which is the regime self-supervised
learning was developed for: learn the structure of ordinary physiology from
everything, then fine-tune on the little that is annotated. That was the
experiment we set out to run.

Two prerequisites stopped us, neither of which the dataset supplies. A pretext
task is only meaningful against a supervised baseline worth improving on, and
no baseline we could trust existed for this dataset. Building one requires a
negative class, and there is none: every survey row marks an episode a
participant flagged, so the annotation records severity, not presence. The
comparison group has to be constructed from unlabelled time, and constructing
it defensibly turned out to be the study.

Published results on this dataset are strong. Mathur et al.\ \cite{mathur2024}
report a weighted $F_1$ of $0.99$ for two-level classification. We do not
reproduce anything close to it, and each step of the attrition that separates
us from it is a place where a number can be inflated without anyone noticing:
adjacent analysis windows are near-duplicates of one another, the negative
class is invented, and the effective sample size is 138 episodes from 10 people
rather than the 24{,}334 windows those episodes generate.

We therefore report the study as a sequence of eliminations, each narrowing
what is left. Section \ref{sec:labels} asks whether the labels mark anything
physiological, and finds that one of four channels responds at onset while the
others do not, which reorders the feature priorities for this sensor. Section
\ref{sec:negatives} constructs the comparison group with falsification probes
and finds that careful construction improves rather than inflates performance,
against the usual warning. Section \ref{sec:baseline} measures how far standard
features and a standard learner get, and whether the pipeline can manufacture
its own signal. Section \ref{sec:pu} tests three treatments of the labels
against a rule fixed before results were seen. Section \ref{sec:ssl} tests a
learned representation over 1{,}238 unlabelled hours, with the comparison
biased in its favour. What is left at the end is the data."""

EDITS = [
    (OLD_ABSTRACT, NEW_ABSTRACT),
    (OLD_INTRO_START, NEW_INTRO_START),
    (OLD_P2_TAIL, NEW_P2_TAIL),
    (OLD_INTRO_REST, NEW_INTRO_REST),

    # sentence-level bold that editorialises rather than defines
    (r"""\textbf{What survives is 10 participants, 24{,}334 windows, and 138
high-severity episodes} (Table \ref{tab:attrition}).""",
     r"""What survives is 10 participants, 24{,}334 windows, and 138
high-severity episodes (Table \ref{tab:attrition})."""),

    (r"""\textbf{The rule was not met, and every alternative is worse than the baseline rather than merely insufficiently better.}""",
     r"""The rule was not met, and every alternative is worse than the baseline
rather than merely insufficiently better."""),

    (r"""\textbf{The one task whose numbers are not conditional on a
construction fails too}, which removes the negative construction as the
explanation for the rest.""",
     r"""The one task whose numbers are not conditional on a
construction fails too, which removes the negative construction as the
explanation for the rest."""),

    # the last editorialising paragraph label
    (r"""\paragraph{The gap is the finding.} An AUC of $0.765$ means the model reliably""",
     r"""\paragraph{From AUC to the operating point.} An AUC of $0.765$ means the model reliably"""),
]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    bad = 0
    for i, (old, new) in enumerate(EDITS, 1):
        if old in s:
            s = s.replace(old, new, 1)
            print("  edit %2d: applied" % i)
        else:
            bad += 1
            print("  edit %2d: NOT FOUND <-- %s" % (i, old[:64].replace("\n", " ")))
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)
    print("\n%d of %d applied" % (len(EDITS) - bad, len(EDITS)))


if __name__ == "__main__":
    main()
