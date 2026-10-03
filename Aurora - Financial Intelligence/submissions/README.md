# Three submissions, three audiences

Everything in this project folder feeds one of three destinations. They are separate
submissions with separate reviewers, separate rules, and one hard ordering constraint between
them. This file is the map.

| | Goes to | Reviews | Public? | Lives in |
|---|---|---|---|---|
| **1. Patent** | USPTO, via a patent attorney | whether the *mechanisms* are novel and non-obvious | **No.** Confidential until filed | `01-patent-CONFIDENTIAL/` |
| **2. JOSS** | Journal of Open Source Software | whether the *software* is well-built and usable | Yes, fully open | the repository root |
| **3. CJSJ** | Columbia Junior Science Journal | whether the *research* is sound | Yes, on acceptance | `03-cjsj-research-paper/` |

They are not three versions of one thing. A patent claims a mechanism. A software paper
documents a tool. A research paper reports a finding. The same project produces all three, and
each reviewer cares about something the other two do not.

---

## The one rule that constrains everything

**File the provisional patent before JOSS or CJSJ publishes.**

The United States gives an inventor a twelve-month grace period after their own public
disclosure. Europe, China and most other jurisdictions give none: any public disclosure before
the filing date destroys novelty permanently and cannot be undone.

Both a JOSS submission and a CJSJ acceptance are public disclosures.

So the order is:

```
   file provisional  →  submit to JOSS and CJSJ  →  (within 12 months) decide on the full patent
```

Doing it the other way costs nothing in the US and everything everywhere else. The delay is
a few weeks. The loss is permanent.

One caveat worth checking now: if the research paper is already posted on SSRN, that clock has
already started and non-US rights may already be gone. Establish the SSRN posting date before
paying for any international filing.

---

## 1. Patent

**Folder:** `01-patent-CONFIDENTIAL/`
**Status:** pre-filing draft, not yet filed
**Audience:** a registered patent attorney first, then a USPTO examiner

Contains the invention disclosure: the mechanisms, their mathematics, code exhibits, measured
evidence of reduction to practice, a claim skeleton, a prior-art map, and a subject-matter
eligibility analysis.

**What is being claimed** is four mechanisms, not the model's accuracy:

1. the point-in-time availability graph with the net-lag composition check
2. conformal calibration partitioned by a *filtered* latent state
3. a training penalty on prediction magnitude rather than prediction error
4. the truncation-invariance test for verifying non-anticipation

The disclosure deliberately does **not** claim superior forecast accuracy, because the study
shows classical baselines remain competitive and that record is public. A specification
contradicted by the applicant's own published results invites a finding of inequitable
conduct, which voids an entire patent rather than one claim.

**This folder is excluded from git** and must not go into the public repository. It contains
the claim strategy and a candid assessment of which claims are weak, written for your
attorney's eyes.

**Next step:** engage a registered patent attorney and commission a professional prior-art
search, focused on conditional and Mondrian conformal prediction, which is the identified area
of greatest anticipation risk.

---

## 2. JOSS

**Folder:** the repository root, not a subfolder
**Status:** ready to push, three placeholders to fill
**Audience:** two software reviewers working a public checklist on GitHub

JOSS reviews **software**, not findings. Reviewers install the package, run the tests, read the
documentation, and ask whether other researchers could use this. They do not evaluate whether
your forecasting results are correct.

The submission is the whole repository: `aurora/`, `tests/`, `docs/`, `paper/paper.md`,
`paper/paper.bib`, plus LICENSE, CONTRIBUTING, CODE_OF_CONDUCT and CITATION.cff.

`paper/paper.md` is 1,504 words and describes the *software*. It is a different document from
the CJSJ paper, which is 21 pages and describes the *study*. That distinction is required:
JOSS explicitly does not accept a full-length research paper, and a submission that duplicates
one gets sent back.

Step-by-step instructions are in `UPLOAD_TO_GITHUB.md` at the repository root.

---

## 3. CJSJ

**Folder:** `03-cjsj-research-paper/`
**Status:** complete manuscript, check journal formatting before submitting
**Audience:** journal reviewers assessing the research

`Aurora_Paper_FINAL.pdf`, 21 pages, is the research contribution: the comparative study of
five model families over 7,303 trading days, the calibration result, the stability result, the
selection-instability finding, and the two negative results.

This is where the *findings* are argued and defended. The reviewers here will question the
methodology, the statistics, and whether the conclusions follow from the evidence, which is
exactly what the paper was built to withstand.

**Before submitting, check three things against CJSJ's own guidelines:**

1. **Preprint policy.** If the paper is already on SSRN, confirm CJSJ accepts submissions with
   an existing preprint. Most journals do; some do not. This is the item most likely to cause
   a problem.
2. **Formatting.** Page limits, citation style, anonymisation for review, figure
   specifications. The current PDF is formatted as a two-column research paper, which may or
   may not match.
3. **Author eligibility.** Confirm the eligibility criteria and any institutional
   endorsement requirement.

---

## What else is in this project folder

| Item | Belongs to | Note |
|---|---|---|
| `aurora/`, `tests/`, `docs/` | JOSS | the software being submitted |
| `paper/paper.md`, `paper/paper.bib` | JOSS | the software paper |
| `RSMoE_study/`, `RSMoE_study_v2/` | CJSJ | run outputs and figures backing the study. 49 MB, excluded from git. Attach to a GitHub Release or archive on Zenodo if you need them citable |
| `RSMoE_package.tar.gz`, `RSMoE_v2_package.tar.gz` | CJSJ | 23 MB study snapshots, excluded from git |
| `experiments/`, `notebooks/`, `scripts/` | JOSS | shipped with the software |
| `_superseded/` | nothing | earlier drafts of the research paper, kept for provenance only. Do not submit these anywhere |

---

## AI usage, which each destination treats differently

- **JOSS** requires a disclosure section. `paper/paper.md` has one. Read it and correct it
  against what you know to be true.
- **CJSJ** will have its own policy. Check it and comply. Journals increasingly ask.
- **The patent** has a distinct question: inventorship. Under current USPTO guidance an
  inventor must be a natural person who made a significant contribution to the conception of
  the claimed invention. Discuss with your attorney how the work was done, honestly. This
  affects whether a patent is valid, so it is not a question to guess at.
