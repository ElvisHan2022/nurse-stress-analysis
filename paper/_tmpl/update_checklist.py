"""Bring the checklist and the limitations section in line with Phases 4-6.

Three checklist answers referenced a boundary the paper no longer has, and the
limitations section did not cover the pretraining arm at all. A checklist that
describes a previous draft is worse than no checklist, because it reads as
carelessness in exactly the place a reviewer checks for it.
"""
import io
import os

CHK = os.path.join("paper", "checklist.tex")
SRC = os.path.join("paper", "main.tex")

CHK_EDITS = [
    # Q1 claims
    ("""Justification: The abstract states the negative result and its boundary explicitly: we eliminated the label-treatment axis and not the representation axis, and Section 7.1 repeats this rather than generalising past it. The intended contribution, an account of the prerequisites for self-supervision on sparsely labelled physiological data, matches what is delivered. Every quantity quoted in the abstract appears with its seed range where one exists.""",
     """Justification: The abstract reports three nulls in the order the paper establishes them: label treatments, learned representations, and severity. Section 7 does not generalise past them, and the transductive status of the pretraining results is stated in the abstract, in Section 7, and in the limitations rather than only once. Every quantity quoted in the abstract appears with its seed range where one exists."""),

    # Q2 limitations
    ("""the two-learner search space, the 33 episodes with no sensor coverage, the two normalisation constants chosen rather than derived, the underpowered ten-fold comparisons, and the resolution floor of a twenty-draw permutation test.""",
     """the two-learner search space, the 33 episodes with no sensor coverage, the two normalisation constants chosen rather than derived, the underpowered ten-fold comparisons, the resolution floor of a twenty-draw permutation test, and, for the pretraining arm, its transductive design, its single architecture and training budget, and the fact that a weak representation and a clean one are distinguished only by the downstream sweep."""),

    # Q6 settings
    ("""Choices fixed before results were seen are identified as such, including the episode-detection fraction and the decision rule in Section 6.""",
     """Section 7 gives the pretraining corpus, the encoder architecture, the mask-span range and that it was fixed a priori, the number of epochs, and the three fine-tuning arms with their learning rates. Choices fixed before results were seen are identified as such, including the episode-detection fraction, the decision rule in Section 6, and the mask span."""),

    # Q8 compute
    ("""gives the feature-extraction time over 387 sessions, and states that the complete set of reported experiments finishes in under an hour.""",
     """gives the feature-extraction time over 387 sessions, the pretraining time over 68k windows, and the runtime of the label-efficiency sweep, and states that the complete set of reported experiments finishes in about half an hour beyond feature extraction."""),
]

LIM_OLD = """The permutation null uses twenty
draws, so its $p$-value floor is $0.048$; the observed value exceeds every draw
but a finer resolution would require more permutations."""

LIM_NEW = """The permutation null uses twenty
draws, so its $p$-value floor is $0.048$; the observed value exceeds every draw
but a finer resolution would require more permutations.

The pretraining result carries its own qualifications. The encoder saw every
participant, so Table \\ref{tab:ssl} reports transductive upper bounds rather
than estimates for an unseen person; per-fold pretraining is the clean
alternative and costs ten times the compute. One architecture and one training
budget were tried, and a larger encoder or longer schedule is untested. A
representation that is weak and one that is genuinely free of participant
identity produce the same low probe accuracy, and we separate them only by the
downstream sweep, which is indirect. The severity task is limited by 17
medium-severity episodes across the cohort, which is why we report cumulative
links and a collapsed task rather than a three-class model."""

COMPUTE_OLD = """the complete set
of reported experiments, including the twenty-draw permutation null and the
four-arm ratio sensitivity, completes in under an hour."""

COMPUTE_NEW = """the complete set
of reported experiments completes in about half an hour beyond feature
extraction. Pretraining over 68k windows takes 192 s for twelve epochs, and the
label-efficiency sweep, which is 120 model fits, takes roughly thirteen
minutes."""


def apply(path, edits, label):
    s = io.open(path, encoding="utf-8").read()
    for i, (old, new) in enumerate(edits, 1):
        if old in s:
            s = s.replace(old, new)
            print(f"  {label} edit {i}: applied")
        else:
            print(f"  {label} edit {i}: NOT FOUND")
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)


def main():
    apply(CHK, CHK_EDITS, "checklist")
    apply(SRC, [(LIM_OLD, LIM_NEW), (COMPUTE_OLD, COMPUTE_NEW)], "paper")


if __name__ == "__main__":
    main()
