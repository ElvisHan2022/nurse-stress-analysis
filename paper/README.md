# Paper draft

`main.tex` — the draft. NeurIPS 2025 preprint format.
`checklist.tex` — the required NeurIPS checklist, not yet filled in.

## To compile

`neurips_2025.sty` is not vendored here. Download it from the conference site
and place it in this directory, then:

    pdflatex main && pdflatex main

## Provenance

Every number traces to a script in `../tasks/` and a frozen artifact in
`../derived/` or `../reports/audit/`. Verified against source before writing:
see `../docs/open_questions.md`.

All six references were checked against Crossref, the arXiv API, or Consensus.
None are quoted from memory.
