"""The three referee fixes. All localised, none require re-running anything.

1. The abstract's closing sentence generalised past one encoder at one budget.
   Validated as a CRITICAL overclaim and narrowed to what was tested.
2. Table 4 quoted no seed range while every other table did, and the deltas it
   reports are smaller than the baseline's own seed spread. Stated plainly.
3. Severity uses 186 episodes and detection uses 138. The difference is real
   and explicable, and went unexplained.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

EDITS = [
    # 1. the overclaim
    ("""Severity prediction, the one task
here needing no constructed negatives, is also at chance. The constraint is the
label supply, and no amount of unlabelled data substitutes for it.""",
     """Severity prediction, the one task
here needing no constructed negatives, is also at chance. On this dataset the
constraint is the label supply, and pretraining as we implemented it does not
substitute for it."""),

    # 2. the missing noise floor on the sweep
    ("""Table \\ref{tab:ssl} gives two negatives. Pretrained minus randomly initialised
sits between $-0.024$ and $-0.003$ at every label count, so there is no
separation at low counts, which was the entire hypothesis.""",
     """Table \\ref{tab:ssl} gives two negatives, and they are not equally strong.
Pretrained minus randomly initialised sits between $-0.024$ and $-0.003$ at
every label count. Three of those four differences are smaller than the
baseline's own seed spread of $0.0095$, so the honest statement is that no
separation is \\emph{detectable} at this resolution rather than that none
exists; each cell is a single run, and resolving a difference this small would
need repeated draws we did not make. The direction is consistent across all
four counts and never favours pretraining, which is what makes the absence of a
low-count advantage, the entire hypothesis, informative."""),

    # 3. the two denominators
    ("""Severity cannot: it
runs on the 186 rated episodes with sensor coverage and needs no negatives at
all.""",
     """Severity cannot: it
runs on 186 episodes rather than the 138 used for detection, because the
detection task additionally requires each episode to survive the negative
construction and the high-severity filter, while severity keeps the low- and
medium-severity episodes it needs and requires no negatives at all."""),
]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    for i, (old, new) in enumerate(EDITS, 1):
        if old in s:
            s = s.replace(old, new)
            print(f"  fix {i}: applied")
        else:
            print(f"  fix {i}: NOT FOUND")
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)


if __name__ == "__main__":
    main()
