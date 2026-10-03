"""Causal-integrity tests for the online regime filter.

These are the tests that back the central claim of the paper: that the regime
signal reaching the forecaster at time t depends only on observations up to t.

The assertion is exact equality, not equality within a tolerance. That is
achievable because the forward recursion at step t reads only the previous
forward variable and the current observation, so appending observations after t
cannot alter any floating-point operation contributing to the result at t.

If any of these fail, something upstream is reading ahead: full-sample
standardisation, a smoothed posterior leaking downstream, or a centred rolling
window in feature construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.regimes.hmm import GaussianHMMDetector
from aurora.regimes.online_controller import OnlineHMMFilter


def _regime_switching_features(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """Two-state synthetic series so the filter has real structure to find."""
    rng = np.random.default_rng(seed)
    mu = np.array([0.0006, -0.0011])
    sd = np.array([0.007, 0.023])
    A = np.array([[0.985, 0.015], [0.05, 0.95]])

    s, states = 0, np.empty(n, dtype=int)
    for t in range(n):
        states[t] = s
        s = rng.choice(2, p=A[s])

    r = rng.normal(mu[states], sd[states])
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2005-01-03", periods=n)

    px = pd.Series(close, index=idx)
    lr = np.log(px).diff()
    return pd.DataFrame(
        {
            "ret_21": lr.rolling(21).mean(),
            "vol_21": lr.rolling(21).std() * np.sqrt(252),
            "drawdown": px / px.cummax() - 1.0,
        },
        index=idx,
    ).bfill().ffill()


@pytest.fixture(scope="module")
def fitted_filter():
    feats = _regime_switching_features()
    det = GaussianHMMDetector(n_regimes=2, random_state=0).fit(feats)
    return OnlineHMMFilter(det), feats


def _posteriors(f: OnlineHMMFilter, feats: pd.DataFrame) -> np.ndarray:
    return np.array([sv.filtered_posterior for sv in f.filter_sequence(feats)])


def test_filtered_posterior_is_non_anticipative(fitted_filter):
    """THE causal test: truncating the input must not change earlier output.

    Exact equality to 0.0, not a tolerance.
    """
    f, feats = fitted_filter
    full = _posteriors(f, feats)

    for cut in (250, 500, 750):
        truncated = _posteriors(f, feats.iloc[:cut])
        assert truncated.shape[0] == cut
        max_diff = float(np.abs(full[:cut] - truncated).max())
        assert max_diff == 0.0, (
            f"Look-ahead detected at cut={cut}: the posterior for the first {cut} "
            f"rows changed by {max_diff} when later rows were appended."
        )


def test_appending_future_rows_leaves_history_untouched(fitted_filter):
    """Growing the series one row at a time must extend, never revise."""
    f, feats = fitted_filter
    base = 400
    prev = _posteriors(f, feats.iloc[:base])

    for extra in (1, 5, 25):
        grown = _posteriors(f, feats.iloc[: base + extra])
        assert np.abs(grown[:base] - prev).max() == 0.0, (
            f"Appending {extra} future row(s) revised the existing history."
        )


def test_posterior_rows_are_probability_vectors(fitted_filter):
    f, feats = fitted_filter
    p = _posteriors(f, feats)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)


def test_smoothed_posterior_would_fail_the_same_test(fitted_filter):
    """Control: the smoothed posterior is anticipative, and this proves it.

    hmmlearn's predict_proba runs forward-backward, conditioning on the whole
    sample. This test asserts that it does NOT satisfy truncation invariance,
    which is why the framework never passes it downstream. If this test ever
    starts failing, the control has stopped being a control and the causal test
    above is no longer discriminating.
    """
    f, feats = fitted_filter
    det = GaussianHMMDetector(n_regimes=2, random_state=0).fit(feats)

    prob_cols = lambda d: [c for c in d.columns if c.startswith("prob_")]
    full = det.predict(feats)
    trunc = det.predict(feats.iloc[:500])

    diff = np.abs(
        full[prob_cols(full)].values[:500] - trunc[prob_cols(trunc)].values
    ).max()
    assert diff > 0.0, (
        "The smoothed posterior was expected to be anticipative but was not; "
        "the causal test above may no longer be discriminating."
    )
