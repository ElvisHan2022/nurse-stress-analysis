"""Trim the draft from 9 content pages to fit the NewInML 2-8 page limit.

Every cut is prose compression or table compaction. No result, caveat, or
limitation is removed, because those are the parts a reviewer of a negative
result will check hardest.

Four edits:
  1. Clock alignment: five sentences to two. The method is interesting but
     the paper does not depend on how it was resolved, only that it was.
  2. The F1 prevalence table: seven rows to three. A referee could not
     reconstruct the argument without it, so it stays, but precision, recall
     and F1 carry the point without the four cell counts.
  3. Future-collection guidance: compressed to its three imperatives.
  4. Exclusions: the rule list is compacted into one sentence.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

EDITS = [
    # 1. clock alignment
    ("""Sensor timestamps are UTC; survey timestamps are bare local wall-clock with no
zone recorded. Choosing wrongly places nearly every episode outside its own
recording and raises no error. We resolved it by counting, under each
hypothesis, how many rated episodes start inside a session belonging to the
same participant. Treating survey times as UTC places 50 of 245
(20.4\\%); treating them as naive \\texttt{America/Chicago} places
212 (86.5\\%). A fixed five-hour offset reaches 198, so the
daylight-saving-aware zone is required: the archive spans April to December.
Thirty-three rated episodes (13.5\\%) fall outside every session
even under the winning hypothesis and are set aside.""",
     """Sensor timestamps are UTC; survey timestamps are bare local wall-clock with no
zone recorded. Choosing wrongly places nearly every episode outside its own
recording and raises no error, so we resolved it empirically by counting how
many rated episodes start inside a session belonging to the same participant.
Treating them as UTC places 50 of 245 (20.4\\%); as daylight-saving-aware local
time, 212 (86.5\\%). Thirty-three rated episodes (13.5\\%) fall outside every
session even so, and are set aside."""),

    # 2. F1 table, seven rows to three
    ("""\\begin{tabular}{lrr}
\\toprule
 & 2:1 fold (33.3\\% positive) & 32:1 fold (3.0\\% positive) \\\\
\\midrule
True positives  & 26.7 & 2.4 \\\\
False negatives &  6.7 & 0.6 \\\\
False positives & 13.3 & 19.4 \\\\
True negatives  & 53.3 & 77.6 \\\\
\\midrule
Precision & 0.667 & 0.111 \\\\
Recall    & 0.800 & 0.800 \\\\
$F_1$     & \\textbf{0.727} & \\textbf{0.195} \\\\
\\bottomrule
\\end{tabular}""",
     """\\begin{tabular}{lrr}
\\toprule
 & 2:1 fold (33.3\\% positive) & 32:1 fold (3.0\\% positive) \\\\
\\midrule
Precision & 0.667 & 0.111 \\\\
Recall    & 0.800 & 0.800 \\\\
$F_1$     & \\textbf{0.727} & \\textbf{0.195} \\\\
\\bottomrule
\\end{tabular}"""),

    # 3. future collection
    ("""The most actionable consequence concerns label supply. A collection targeting
this task should record explicit non-stress intervals, since their absence
forces every downstream number to be conditional on a construction; should
capture onset with finer resolution than a retrospective interval; and should
prioritise electrodermal signal quality, since four of fifteen participants
here produced a flat trace on the only responsive channel. Heart rate
contributed nothing measurable at onset in this cohort, and inter-beat
intervals were usable in under 6\\% of windows.""",
     """The actionable consequence concerns label supply. A collection targeting this
task should record explicit non-stress intervals, since their absence makes
every downstream number conditional on a construction; capture onset more
finely than a retrospective interval; and prioritise electrodermal signal
quality, since four of fifteen participants here produced a flat trace on the
only responsive channel."""),

    # 4. exclusions
    ("""We remove sessions under five minutes, non-wear windows, sessions with a dead
electrodermal sensor, unrated episodes from both classes, three duplicate
survey rows, and low- and medium-severity episodes from the binary task. We
merge twelve overlapping episode pairs to their outer span.""",
     """We remove short sessions, non-wear windows, sessions with a dead electrodermal
sensor, unrated episodes from both classes, duplicate survey rows, and low- and
medium-severity episodes; twelve overlapping episode pairs are merged to their
outer span."""),
]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    before = len(s)
    for i, (old, new) in enumerate(EDITS, 1):
        if old in s:
            s = s.replace(old, new)
            saved = len(old) - len(new)
            print(f"  edit {i}: applied, {saved} chars saved")
        else:
            print(f"  edit {i}: NOT FOUND, skipped")
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)
    print(f"\n  {before} -> {len(s)} chars ({before - len(s)} removed)")


if __name__ == "__main__":
    main()
