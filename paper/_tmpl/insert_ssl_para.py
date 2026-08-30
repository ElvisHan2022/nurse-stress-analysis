# -*- coding: utf-8 -*-
r"""The two paragraphs edit 2 of related_work.py could not anchor.

Same content; the anchor is rewrapped to match the file as it now stands.
"""
import io

SRC = "paper/main.tex"

ANCHOR = """comparison group has to be constructed from unlabelled time, and constructing
it defensibly turned out to be the study."""

NEW = ANCHOR + r"""

Almost all wearable stress detection is supervised. A model is shown windows
labelled stressed or not and learns the mapping, which works when the protocol
supplies both labels. Self-supervised learning is the standard answer when
labels are scarce, and in medicine it is established rather than novel; a
review of the area \cite{krishnan2022} spans electronic health records, medical
images, and bioelectrical signals. The weight of that work sits in imaging,
where a systematic review of classification alone screened 412 studies
\cite{huang2023}. Ambulatory wearable physiology has attracted far less of it,
and occupational stress recorded by spontaneous self-report has attracted
almost none. That gap is what made this dataset look like an opportunity, and
this paper is what we found when we tried to prepare it for the task.

The pretext tasks themselves are mature. Masked reconstruction
\cite{zerveas2021} and contrastive learning over augmented views
\cite{yue2021} are the two dominant families for time series, patch-based
encoders \cite{nie2022} are now common, and both families have been applied to
free-living wearable recordings \cite{spathis2020, tang2020}. Two of our
choices follow. We use masked reconstruction, because the standard contrastive
augmentations include amplitude scaling and skin conductance amplitude is the
one signal Section \ref{sec:labels} finds responsive. And we keep hand-crafted
features as the reference throughout, because strong classical baselines remain
hard to beat on time series \cite{dempster2020, fawaz2019, fawaz2020} and are
easy to leave out in a way that flatters a deep model \cite{zeng2022}."""

s = io.open(SRC, encoding="utf-8").read()
assert ANCHOR in s, "anchor missing"
io.open(SRC, "w", encoding="utf-8", newline="\n").write(s.replace(ANCHOR, NEW, 1))
print("applied")
