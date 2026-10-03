---
title: 'Aurora: A regime-switching forecasting framework with verifiable causal integrity and calibrated uncertainty'
tags:
  - Python
  - time series
  - forecasting
  - finance
  - conformal prediction
  - hidden Markov models
  - reproducibility
authors:
  - name: Yuvraj Bharadia
    orcid: 0009-0006-0192-2383
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, United Arab Emirates
    index: 1
date: 29 September 2026
bibliography: paper.bib
---

# Summary

Forecasts of financial time series are usually reported as a single number, with no statement
of how much confidence that number deserves and no way for a reader to check that the model
was not accidentally shown information from the future. Both omissions matter. A forecast
without an honest uncertainty range cannot support a decision, and a model that has seen the
future will report excellent accuracy while being useless in practice.

`Aurora` is a Python framework for building and evaluating forecasting models that addresses
both problems directly. It first estimates what kind of market conditions currently hold,
distinguishing calm periods from turbulent ones, and then conditions everything downstream on
that estimate: which predictors matter, how several candidate forecasters are combined, how
wide the uncertainty range should be, and whether acting on the forecast is worth the cost of
doing so. Crucially, the market-state estimate is computed using only information available at
the time, and the framework ships with executable tests that verify this property exactly
rather than asserting it in prose.

The framework is intended for researchers who need a forecasting harness where methodological
errors are detectable rather than silent. It provides a point-in-time data layer that records
when each input genuinely became knowable, walk-forward evaluation with horizon-aware purging,
a library of econometric, machine-learning and deep-learning forecasters behind one interface,
regime-conditioned conformal prediction intervals, and an evaluation layer that emits
significance tests and overfitting diagnostics alongside every result.

# Statement of need

The comparative literature on machine learning for financial forecasting is internally
inconsistent. Systematic reviews report that neural architectures outperform classical
statistical benchmarks with near-unanimity [@sezer2020; @jiang2021], while large-scale
forecasting competitions report close to the opposite across tens of thousands of series
[@makridakis2018]. Both bodies of work are carefully conducted. The disagreement is the
problem.

Several mechanisms can produce a favourable result without any underlying predictive
advantage, and none of them raises an error at runtime. Macroeconomic series are published
with a lag and subsequently revised, so joining them on their reference date rather than their
release date grants a model information it could not have had [@croushore2001;
@lopezdeprado2018]. Latent-state labels derived from a smoothed posterior condition on the
entire sample, including observations after the labelled date. Architectures whose results
vary substantially with random initialisation are routinely compared at a single seed. Studies
reporting the best of several models across several horizons adjust significance within tables
rather than across the full comparison family [@bailey2014; @bailey2016].

`Aurora` exists because no available package makes these errors structurally difficult. It is
designed for quantitative finance researchers, econometricians, and methodologists studying
forecast evaluation, and its central design commitment is that every causal claim the
framework makes is enforced by a test that fails when the claim is violated.

# State of the field

Several mature packages address parts of this problem. `statsmodels` [@seabold2010] provides
classical time-series models and Markov-switching estimation. `sktime` [@loning2019] and
`darts` [@herzen2022] supply unified forecasting interfaces with backtesting utilities.
`hmmlearn` implements hidden Markov models, and `Aurora` builds on it. `MAPIE` [@cordier2023]
and `crepes` [@bostrom2022] provide conformal prediction wrappers around arbitrary estimators.

Each solves its own problem well, and `Aurora` was built rather than contributed to them for
three reasons.

First, the components do not compose in the way this research question requires. Conformal
wrappers calibrate on a pooled residual set; conditioning calibration on a latent state
inferred online, with a documented fallback when a state is sparsely populated, is not
expressible through their interfaces. Combining a regime estimator from one package with a
conformal wrapper from another leaves the causal relationship between them unverified, which
is precisely the property under study.

Second, no existing package treats non-anticipation as testable. Backtesting utilities in
`sktime` and `darts` partition data correctly at the split level, but nothing verifies that a
latent-state estimator embedded in a pipeline is forward-only. This distinction is concrete
rather than theoretical: `hmmlearn`'s `predict_proba` returns the smoothed posterior from the
forward-backward algorithm, which conditions on the entire sample. `Aurora` therefore
implements a separate forward-only filter in `aurora.regimes.online_controller` and passes
only that posterior downstream, with a test asserting bit-exact truncation invariance and a
companion control test confirming that the smoothed posterior does not satisfy the same
property.

Third, the framework is built for ablation rather than deployment. Its layers are
independently switchable, so measuring what each component contributes is a supported
operation rather than a manual exercise. General-purpose forecasting libraries reasonably
optimise for fitting one good model; this framework optimises for measuring the contribution
of each part.

# Software design

`Aurora` is organised as a layered pipeline, each stage independently switchable.

The data layer (`aurora.data.availability`) attaches availability metadata to every feature,
comprising a release lag, revision lag, event offset and release hour, from which a
first-availability timestamp is computed. A validation constraint asserts strictly positive net
lag after offset composition, because a system can store release lags correctly and still
eliminate them through a compensating offset elsewhere. Fold construction enforces an embargo
of at least the forecast horizon, since a target at index $i$ depends on prices through $i+h$.

The regime layer estimates market state with a Gaussian hidden Markov model in the tradition of
@hamilton1989. Only the forward-filtered posterior is exposed downstream:

$$\gamma_t(k) \propto b_k(y_t) \sum_{k'} \gamma_{t-1}(k')\, A_{k'k}$$

Because this recursion reads only the previous forward variable and the current observation,
appending later observations cannot alter any floating-point operation contributing to
$\gamma_t$. This is what makes the truncation-invariance test exact rather than approximate.

The forecasting layer wraps econometric models (ARIMA, GARCH), gradient-boosted trees, and
deep architectures (LSTM, Transformer) behind a common interface, with regime-conditioned
interaction features and a training objective that penalises predicted move magnitudes
exceeding a rolling realised bound:

$$\mathcal{L} = \frac{1}{n}\sum_t (\hat{\rho}_t - \rho_t)^2 + \beta \frac{1}{n}\sum_t
\max\left(0, |\hat{\rho}_t| - B_t\right)$$

The penalty argument is the prediction magnitude, not the residual. This distinction is the
mechanism: penalising residuals reweights the loss, whereas penalising implausible predictions
restricts the effective hypothesis class.

The ensemble layer (`aurora.ensembles.oof_router`) fits a gate over the expert pool on strictly
out-of-fold predictions, with the squared-error term normalised by the target variance. That
normalisation is load-bearing rather than cosmetic: unnormalised squared error on daily log
returns is of order $10^{-4}$, which leaves the gradient reaching the softmax logits negligible
against the regularisers, and the gate then remains frozen at initialisation while silently
emitting a plausible equal-weight ensemble.

Split-conformal calibration [@vovk2005; @lei2018] in `aurora.evaluation.conformal` is performed
within each filtered state, with the finite-sample quantile correction and a pooled fallback
below a minimum count. The portfolio layer converts a calibrated forecast into a position or an
abstention, thresholding expected edge net of square-root market impact [@almgren2001] against
a multiple of the conformal half-width. The evaluation layer emits Diebold-Mariano tests
[@diebold1995] with the @harvey1997 small-sample correction, Holm-Bonferroni adjustment across
the full comparison family, and overfitting diagnostics.

# Research impact statement

`Aurora` has been used to produce a complete comparative study of five model families over
thirty years of daily index data, covering 7,303 trading days across five expanding-window
folds, three horizons and three random seeds. That study is the framework's primary evidence of
fitness for research use, and its findings are reproducible from the released code.

Three results illustrate what the framework makes measurable. Its regime-conditioned conformal
intervals attained 90.3% and 90.0% empirical coverage against a 90% nominal target at one-day
and one-week horizons, with interval width adapting across market states by a factor of 1.88;
replacing this layer with an unconditional interval cost 11.7 percentage points of coverage.
Across-seed dispersion at the one-month horizon was 2.02 index points for the integrated
architecture against 15.44 for an adversarial baseline, a measurement with direct implications
for whether the single-seed comparisons standard in this literature can support the differences
they report. And across fifteen fold-horizon cells, the model ranking first on validation data
ranked first on test data in one cell, with a mean rank correlation of 0.517.

The framework also made two negative results measurable and reportable: classical univariate
baselines remained competitive with every learned model tested, and cost-aware abstention
produced positive risk-adjusted returns net of modelled costs at only one of three horizons. A
harness that surfaces such results rather than concealing them is the contribution being
offered.

The component we consider most transferable is the truncation-invariance test, which is
domain-independent and applies to any forward-only sequential model whose causal integrity must
be demonstrated rather than asserted. The framework comprises roughly 11,500 lines of Python
across 74 modules, is installable from source, is covered by 69 tests run against three Python
versions on Linux, macOS and Windows in continuous integration, and is documented with an
architecture guide and a methodology reference.

# AI usage disclosure

No Generative AI assistance was used during development of the software and writing of the paper.

# Acknowledgements

The author received no financial support for this work.

# References
