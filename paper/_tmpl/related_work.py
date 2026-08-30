# -*- coding: utf-8 -*-
r"""Give the introduction its related work, using only verified references.

Fourteen new citations, every one checked before being written. The seven with
DOIs went through the Crossref REST API, the five preprints through the arXiv
API, and the two healthcare reviews through PubMed, in each case confirming
title, first author, year and venue against the identifier. One candidate was
discarded on the way: arXiv 1907.11065 is "DropAttention: A Regularization
Method for Fully-Connected Self-Attention Networks", not the wearable
self-supervision paper it was going in as, which is exactly the failure mode
this check exists to catch. Preprints are cited as preprints; no venue is
claimed that a registry did not confirm.

Three paragraphs go in.

The first places the study in the wearable stress literature and makes the
point the paper turns on: Healey and Picard, WESAD and Sano and Picard all
supply a non-stress condition by design, and a dataset built on spontaneous
self-report does not. That reframes the negative-class construction as the
field's standing assumption being unavailable rather than a quirk of our
pipeline.

The second is supervised versus self-supervised in the detection space. The
claim it was drafted from, that few researchers have tried self-supervision on
unlabelled healthcare data, does not survive checking: PubMed returns 243
papers with the phrase in the title in a medical context, and the area has a
Nature Biomedical Engineering review. What does survive is the narrower and
more useful version. The work concentrates in imaging, where a single
systematic review of classification screened 412 studies; restricting the same
search to wearable or ambulatory physiological signals returns 30. So the
paragraph says established-but-concentrated, not neglected.

The third is methodology, and earns its place by forcing two of our choices:
masked reconstruction over contrastive learning, because contrastive
augmentations scale amplitude and amplitude is the responsive signal here; and
keeping a hand-crafted baseline, because classical baselines stay hard to beat.
"""
import io
import os

SRC = os.path.join("paper", "main.tex")

# --- paragraph one, after the dataset sentence ------------------------------
ANCHOR_A = r"""outbreak, with periodic smartphone-administered stress reports alongside the
sensor streams, and the data are public."""

NEW_A = ANCHOR_A + r"""

Wearable stress detection is an established field, and nearly all of it rests
on protocols that supply a non-stress condition by design. Healey and Picard
\cite{healey2005} recorded drivers across rest, highway and city segments,
where the rest segment is the comparison group. WESAD \cite{schmidt2018}
alternates a stress condition with a neutral baseline and an amusement
condition for the same reason. Sano and Picard \cite{sano2013} moved to
ambulatory recording of daily life, but collected ratings on a fixed schedule
rather than only when a participant chose to report. The designed condition is
what makes the resulting data trainable, and it is the thing a dataset built on
spontaneous self-report does not have: participants record episodes, not their
absence. Published work on this cohort inherits the problem. Mathur et al.\
\cite{mathur2024} report a weighted $F_1$ of $0.99$ for two-level
classification on it."""

# --- paragraphs two and three, after the prerequisites paragraph ------------
ANCHOR_B = r"""The comparison group has to
be constructed from unlabelled time, and constructing it defensibly turned out
to be the study."""

NEW_B = ANCHOR_B + r"""

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

# --- Mathur now arrives earlier, so drop the second introduction of it ------
OLD_C = r"""Published results on this dataset are strong. Mathur et al.\ \cite{mathur2024}
report a weighted $F_1$ of $0.99$ for two-level classification. We do not
reproduce anything close to it, and each step of the attrition that separates
us from it is a place where a number can be inflated without anyone noticing:"""

NEW_C = r"""We do not reproduce anything close to that figure, and each step of the
attrition that separates us from it is a place where a number can be inflated
without anyone noticing:"""

# --- bibliography, kept alphabetical by key ---------------------------------
BIB_DEMPSTER = r"""\bibitem{dempster2020}
A.~Dempster, F.~Petitjean, and G.~I. Webb.
\newblock ROCKET: exceptionally fast and accurate time series classification
  using random convolutional kernels.
\newblock \emph{Data Mining and Knowledge Discovery}, 2020.
\newblock DOI: 10.1007/s10618-020-00701-z.

"""

BIB_FGH = r"""\bibitem{fawaz2019}
H.~Ismail Fawaz, G.~Forestier, J.~Weber, L.~Idoumghar, and P.-A. Muller.
\newblock Deep learning for time series classification: a review.
\newblock \emph{Data Mining and Knowledge Discovery}, 2019.
\newblock DOI: 10.1007/s10618-019-00619-1.

\bibitem{fawaz2020}
H.~Ismail Fawaz, B.~Lucas, G.~Forestier, C.~Pelletier, D.~F. Schmidt,
  J.~Weber, G.~I. Webb, L.~Idoumghar, P.-A. Muller, and F.~Petitjean.
\newblock InceptionTime: finding AlexNet for time series classification.
\newblock \emph{Data Mining and Knowledge Discovery}, 2020.
\newblock DOI: 10.1007/s10618-020-00710-y.

\bibitem{healey2005}
J.~A. Healey and R.~W. Picard.
\newblock Detecting stress during real-world driving tasks using physiological
  sensors.
\newblock \emph{IEEE Transactions on Intelligent Transportation Systems}, 2005.
\newblock DOI: 10.1109/TITS.2005.848368.

"""

BIB_HUANG = r"""\bibitem{huang2023}
S.-C. Huang, A.~Pareek, M.~Jensen, M.~P. Lungren, S.~Yeung, and
  A.~S. Chaudhari.
\newblock Self-supervised learning for medical image classification: a
  systematic review and implementation guidelines.
\newblock \emph{npj Digital Medicine}, 6:74, 2023.
\newblock DOI: 10.1038/s41746-023-00811-0.

"""

BIB_KRISHNAN = r"""\bibitem{krishnan2022}
R.~Krishnan, P.~Rajpurkar, and E.~J. Topol.
\newblock Self-supervised learning in medicine and healthcare.
\newblock \emph{Nature Biomedical Engineering}, 6(12):1346--1352, 2022.
\newblock DOI: 10.1038/s41551-022-00914-1.

"""

BIB_NIE = r"""\bibitem{nie2022}
Y.~Nie, N.~H. Nguyen, P.~Sinthong, and J.~Kalagnanam.
\newblock A time series is worth 64 words: long-term forecasting with
  transformers.
\newblock arXiv:2211.14730, 2022.

"""

BIB_TAIL = r"""\bibitem{sano2013}
A.~Sano and R.~W. Picard.
\newblock Stress recognition using wearable sensors and mobile phones.
\newblock In \emph{2013 Humaine Association Conference on Affective Computing
  and Intelligent Interaction}, 2013.
\newblock DOI: 10.1109/ACII.2013.117.

\bibitem{schmidt2018}
P.~Schmidt, A.~Reiss, R.~Duerichen, C.~Marberger, and K.~Van~Laerhoven.
\newblock Introducing WESAD, a multimodal dataset for wearable stress and
  affect detection.
\newblock In \emph{Proceedings of the 20th ACM International Conference on
  Multimodal Interaction}, 2018.
\newblock DOI: 10.1145/3242969.3242985.

\bibitem{spathis2020}
D.~Spathis, I.~Perez-Pozuelo, S.~Brage, N.~J. Wareham, and C.~Mascolo.
\newblock Self-supervised transfer learning of physiological representations
  from free-living wearable data.
\newblock arXiv:2011.12121, 2020.

\bibitem{tang2020}
C.~I. Tang, I.~Perez-Pozuelo, D.~Spathis, and C.~Mascolo.
\newblock Exploring contrastive learning in human activity recognition for
  healthcare.
\newblock arXiv:2011.11542, 2020.

\bibitem{yue2021}
Z.~Yue, Y.~Wang, J.~Duan, T.~Yang, C.~Huang, Y.~Tong, and B.~Xu.
\newblock TS2Vec: towards universal representation of time series.
\newblock arXiv:2106.10466, 2021.

\bibitem{zeng2022}
A.~Zeng, M.~Chen, L.~Zhang, and Q.~Xu.
\newblock Are transformers effective for time series forecasting?
\newblock arXiv:2205.13504, 2022.

\bibitem{zerveas2021}
G.~Zerveas, S.~Jayaraman, D.~Patel, A.~Bhamidipaty, and C.~Eickhoff.
\newblock A transformer-based framework for multivariate time series
  representation learning.
\newblock In \emph{Proceedings of the 27th ACM SIGKDD Conference on Knowledge
  Discovery and Data Mining}, 2021.
\newblock DOI: 10.1145/3447548.3467401.

\end{thebibliography}"""

EDITS = [
    (ANCHOR_A, NEW_A),
    (ANCHOR_B, NEW_B),
    (OLD_C, NEW_C),
    (r"\begin{thebibliography}{9}", r"\begin{thebibliography}{20}"),
    (r"\bibitem{elkan2008}", BIB_DEMPSTER + r"\bibitem{elkan2008}"),
    (r"\bibitem{hosseini2022}", BIB_FGH + r"\bibitem{hosseini2022}"),
    (r"\bibitem{kiryo2017}", BIB_HUANG + r"\bibitem{kiryo2017}"),
    (r"\bibitem{mathur2024}", BIB_KRISHNAN + r"\bibitem{mathur2024}"),
    (r"\bibitem{orini2023}", BIB_NIE + r"\bibitem{orini2023}"),
    (r"\end{thebibliography}", BIB_TAIL),
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
