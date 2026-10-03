# JOSS: the submission is the repository root

There are no files in this folder on purpose. The JOSS submission is the repository itself,
not a document you upload.

## What gets submitted

The whole repository at the parent of this folder:

```
aurora/                    74 modules, ~11,500 lines
tests/                     69 tests across 7 files
docs/                      architecture.md, methodology.md
paper/paper.md             the software paper, 1,504 words
paper/paper.bib            18 references
LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md, CITATION.cff
.github/workflows/         CI plus the JOSS paper compiler
```

## What reviewers actually do

Two reviewers work through a public checklist on GitHub. They install the package, run the
tests, read the docs, and judge whether other researchers could use this. They do **not**
assess whether your forecasting results are correct. That question belongs to CJSJ.

The checklist: <https://joss.theoj.org/about#reviewer_guidelines>

## Why the JOSS paper and the CJSJ paper are different documents

JOSS does not accept full-length research papers. `paper/paper.md` is 1,504 words about the
software: what it does, why existing packages do not cover it, how it is designed, and
evidence it is fit for research use. The CJSJ paper is 21 pages arguing a research finding.
Submitting the same text to both would get the JOSS one rejected.

## Instructions

`UPLOAD_TO_GITHUB.md` at the repository root. Three placeholders to fill first: your ORCID in
`paper/paper.md`, a contact email in `CODE_OF_CONDUCT.md`, and your username in the README
clone URL.

## Timing

File the provisional patent first. A public GitHub repository is a public disclosure.
