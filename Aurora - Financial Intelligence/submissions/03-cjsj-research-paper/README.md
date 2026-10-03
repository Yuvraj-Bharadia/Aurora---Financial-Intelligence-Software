# CJSJ: the research paper

## Contents

- `Aurora_Paper_FINAL.pdf` — 21 pages. The comparative study.

## What this paper argues

Five model families compared over 7,303 trading days of daily index data, 1990 to 2019, under
one pre-registered protocol: five expanding-window folds, three horizons, three random seeds,
purge-and-embargo, and significance testing corrected across the entire comparison family
rather than per table.

The findings:

- The integrated architecture is the most accurate of the learned models, significant at
  Holm-adjusted p < 0.05 against every one of them.
- It is roughly eight times more stable across random restarts than the adversarial baseline,
  which bears on whether the single-seed comparisons standard in this field can support the
  differences they report.
- Its regime-conditioned prediction intervals are calibrated: 90.3% and 90.0% empirical
  coverage against a 90% target, with width adapting across market states by a factor of 1.88.
- Across fifteen fold-horizon cells, the model ranking first on validation ranked first on
  test in one. This is the most transferable finding, and it offers a concrete explanation for
  why published comparisons in this field disagree with each other.

And two negative results reported in full: classical univariate baselines remained competitive
with every learned model including this one, and cost-aware abstention paid at only one of
three horizons.

## Why the negative results stay in

The code ships with the paper, so any reviewer can re-run it and find them. A paper whose
claims its own repository contradicts is worse than one that reports the limits plainly. The
negative results are also the part a reviewer who knows the field will find most credible,
because almost nobody publishes them.

## Check before submitting

1. **Preprint policy.** If this is already on SSRN, confirm CJSJ accepts submissions with an
   existing preprint. Most journals do, some do not, and this is the likeliest problem.
2. **Formatting.** Page limits, citation style, whether review is anonymised, figure
   requirements. The PDF is currently two-column research format.
3. **Eligibility.** Author criteria and any institutional endorsement requirement.
4. **AI policy.** CJSJ will have one. Read it and comply.

## Timing

File the provisional patent first. Publication is a public disclosure.

## Do not submit the drafts in `_superseded/`

Two earlier versions of this study are archived there for provenance. One still has
`[Author Name]` on the title page. Neither should go anywhere.
