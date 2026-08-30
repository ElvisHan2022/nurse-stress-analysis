# -*- coding: utf-8 -*-
r"""Settle the flaws a referee would call critical or major.

Each was verified against the CSVs in reports/audit/ before being touched.

1. CRITICAL. Table 3's last column was headed "Full" and reads 0.749, while
   Table 4's baseline reads 0.7654. They are different feature sets: the
   falsification probes use a deliberately small six-feature set. Two numbers
   for what looks like one arm is the kind of thing that costs a paper its
   credibility on a first read.
2. CRITICAL. "17 medium-severity episodes" is wrong twice. level==1 is 14
   (186 episodes: 31 at level 0, 14 at level 1, 141 at level 2). The related
   claim, six of ten participants with none, checks out: 4 of 10 have one.
3. MAJOR. The identity gate compares a fifteen-way probe (chance 0.067) with
   a ten-way one (chance 0.100) as if the accuracies were commensurable.
   They are not. Lift over chance is.
4. MAJOR. "7,955 candidates, or 265.2 hours" double-counts: windows overlap by
   half, so that is 265.2 window-hours over 132.6 distinct hours.
5. MAJOR. The opening sentence asserted two empirical claims with no citation.
6. MAJOR. The onset analysis uses 140 episodes; the rest of the paper uses 138
   and 186. The 140 was never explained.
7. MINOR. Two skin features beat the six-feature set, which corroborates the
   one-channel finding and went unremarked; and the skin-only AUC collides
   numerically with the identity accuracy, which reads as an error.
8. MINOR. "Six eliminations" introduces seven.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

EDITS = [
    # 5. the uncited opening
    (r"""Nurses leave the profession at rates that track measured occupational stress,
and stressed clinicians make more errors. The case for measuring stress
continuously is therefore both an occupational-health and a patient-safety
argument.""",
     r"""Nurse attrition and clinical error are both costly, and occupational stress is
routinely implicated in each; it is the motivation the authors of this dataset
give for collecting it \cite{hosseini2022}. The case for measuring stress
continuously is therefore both an occupational-health and a patient-safety
argument."""),

    # 4. window-hours are not hours
    (r"""It is also the most expensive condition,
cutting 24{,}334 windows to 16{,}214. All four together leave 7{,}955
candidates, or 265.2 hours.""",
     r"""It is also the most expensive condition,
cutting 24{,}334 windows to 16{,}214. All four together leave 7{,}955 candidate
windows. Because windows overlap by half, that is 265.2 window-hours over 132.6
hours of distinct recording, and we quote the distinct figure wherever coverage
rather than sample size is meant."""),

    # 6. where 140 comes from
    (r"""We
align every episode at its reported onset, normalise each channel against the
participant's own preceding hour, and compare the ten minutes after onset
against the thirty before.""",
     r"""We
align every episode at its reported onset, normalise each channel against the
participant's own preceding hour, and compare the ten minutes after onset
against the thirty before. This admits 140 high-severity episodes, those with
recording on both sides of onset. It is not the 138 of Table
\ref{tab:attrition}, which additionally require analysis windows surviving the
negative construction, and the two counts are not interchangeable."""),

    # 1. the column that contradicts the baseline
    (r"""\caption{Falsification probes and ratio sensitivity, mean of three seeds. All
probes pass in all arms. AUC is prevalence-invariant, so arms with different
denominators remain comparable.}""",
     r"""\caption{Falsification probes and ratio sensitivity, mean of three seeds. All
probes pass in all arms. AUC is prevalence-invariant, so arms with different
denominators remain comparable. Every column uses a deliberately small
six-feature probe set, two of acceleration, two of skin conductance, hour and
position in session; these values are therefore not comparable with the
18-feature baseline of Table \ref{tab:baseline}, and what matters is the
contrast between columns within a row.}"""),

    (r"""Arm & Participants & Balance spread & Movement only & Time only & Skin only & Full \\""",
     r"""Arm & Participants & Balance spread & Movement only & Time only & Skin only & All six \\"""),

    # 7 and 3. what the probes actually say
    (r"""A model given only acceleration reaches $0.513$ and one given only the hour of
day reaches $0.507$, against chance $0.500$, in all four arms. The comparison
group leaks neither movement nor schedule. Participant identity is recoverable
from the same features at $0.768$ against chance $0.100$, which we record as a
measurement rather than a failure, since physiology genuinely differs between
people; it is the reference any learned representation must later be compared
against.""",
     r"""A model given only acceleration reaches $0.513$ and one given only the hour of
day reaches $0.507$, against chance $0.500$, in all four arms. The comparison
group leaks neither movement nor schedule. Two skin-conductance features on
their own reach $0.768$, above the six-feature set at $0.749$, which is the
Section \ref{sec:labels} result arriving a second time: adding the unresponsive
channels to this probe does not help it.

Participant identity is recoverable from those same six features at an accuracy
of $0.768$, against a chance rate of $0.100$ over ten participants. Its
agreement to three decimals with the skin-only AUC in the same row is a
coincidence between two unrelated quantities. We record the identity figure as
a measurement rather than a failure, since physiology genuinely differs between
people; it is the reference any learned representation must later be compared
against."""),

    (r"""A linear probe on frozen embeddings recovers
participant identity at $0.168$, against chance $0.067$ and the $0.768$
reachable from hand-crafted features. The encoder did not learn identity, so
the phase proceeds.""",
     r"""A linear probe on frozen embeddings recovers
participant identity at $0.168$ against a chance rate of $0.067$, the encoder
having been trained across all fifteen participants, while the hand-crafted
reference of $0.768$ is a ten-way problem at chance $0.100$. The accuracies are
not commensurable and we compare them as lift over chance: $2.5\times$ for the
encoder against $7.7\times$ for the features. The encoder did not learn
identity, so the phase proceeds."""),

    # 2. the medium-severity count, twice
    (r"""because 17
medium-severity episodes across the cohort and six of ten participants with
none make the middle class unlearnable""",
     r"""because 14
medium-severity episodes across the cohort and six of ten participants with
none make the middle class unlearnable"""),

    (r"""The severity task rests on 17
medium-severity episodes, which is why we report cumulative links and a
collapsed task rather than a three-class model.""",
     r"""The severity task rests on 14
medium-severity episodes, which is why we report cumulative links and a
collapsed task rather than a three-class model."""),

    # 8. count what is listed
    (r"""Six eliminations, in the order we ran them.""",
     r"""Seven eliminations, in the order we ran them."""),

    (r"""A learned
representation over 1{,}238 unlabelled hours does not improve on it either, and
neither does severity, which needs no constructed negatives.""",
     r"""A learned
representation over 1{,}238 unlabelled hours does not improve on it either. And
severity, which needs no constructed negatives at all, is at chance."""),
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
            print("  edit %2d: NOT FOUND <-- %s" % (i, old[:60].replace("\n", " ")))
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)
    print("\n%d of %d applied" % (len(EDITS) - bad, len(EDITS)))


if __name__ == "__main__":
    main()
