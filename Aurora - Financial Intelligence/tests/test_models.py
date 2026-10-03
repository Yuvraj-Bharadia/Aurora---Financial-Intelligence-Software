"""
Unit tests for all Aurora forecasting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.models.econometric.arima import ARIMAForecaster
from aurora.models.econometric.garch import GARCHForecaster
from aurora.models.ml.gradient_boosting import (
    LightGBMForecaster, RandomForestForecaster, XGBoostForecaster,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split(features, returns):
    """80/20 train/test split."""
    n = len(features)
    cut = int(n * 0.8)
    return (
        features.iloc[:cut], returns.iloc[:cut],
        features.iloc[cut:], returns.iloc[cut:],
    )


# ── ARIMA ──────────────────────────────────────────────────────────────────────

class TestARIMAForecaster:
    def test_fit_predict(self, synthetic_features, synthetic_returns):
        X_tr, y_tr, X_te, _ = _split(synthetic_features, synthetic_returns)
        model = ARIMAForecaster(horizon=1)
        model.fit(pd.DataFrame(index=y_tr.index), y_tr)
        preds = model.predict(pd.DataFrame(index=y_tr.index[-10:]))
        assert len(preds) > 0
        assert np.isfinite(preds).all()

    def test_predict_interval(self, synthetic_features, synthetic_returns):
        X_tr, y_tr, X_te, _ = _split(synthetic_features, synthetic_returns)
        model = ARIMAForecaster(horizon=1)
        model.fit(pd.DataFrame(index=y_tr.index), y_tr)
        lo, hi = model.predict_interval(pd.DataFrame(index=y_tr.index[-10:]))
        assert (hi >= lo).all()


# ── GARCH ──────────────────────────────────────────────────────────────────────

class TestGARCHForecaster:
    def test_fit_predict(self, synthetic_returns):
        model = GARCHForecaster(horizon=5)
        model.fit(pd.DataFrame(index=synthetic_returns.index[:500]),
                  synthetic_returns.iloc[:500])
        preds = model.predict(pd.DataFrame())
        assert len(preds) == 5
        assert (preds >= 0).all()  # volatility is non-negative

    def test_conditional_vol(self, synthetic_returns):
        model = GARCHForecaster(horizon=1)
        model.fit(pd.DataFrame(index=synthetic_returns.index[:500]),
                  synthetic_returns.iloc[:500])
        cv = model.conditional_volatility()
        assert len(cv) > 0
        assert (cv >= 0).all()


# ── XGBoost ────────────────────────────────────────────────────────────────────

class TestXGBoostForecaster:
    def test_fit_predict_shape(self, synthetic_features, synthetic_returns):
        X_tr, y_tr, X_te, y_te = _split(synthetic_features, synthetic_returns)
        model = XGBoostForecaster(horizon=1, n_estimators=50)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(X_te)
        assert np.isfinite(preds).all()

    def test_feature_importance(self, synthetic_features, synthetic_returns):
        X_tr, y_tr, _, _ = _split(synthetic_features, synthetic_returns)
        model = XGBoostForecaster(horizon=1, n_estimators=50)
        model.fit(X_tr, y_tr)
        fi = model.feature_importance()
        assert len(fi) > 0
        assert (fi >= 0).all()

    def test_save_load_roundtrip(self, synthetic_features, synthetic_returns, tmp_path):
        X_tr, y_tr, X_te, _ = _split(synthetic_features, synthetic_returns)
        model = XGBoostForecaster(horizon=1, n_estimators=50)
        model.fit(X_tr, y_tr)
        path = tmp_path / "xgb.pkl"
        model.save(path)
        loaded = XGBoostForecaster.load(path)
        preds_orig = model.predict(X_te)
        preds_loaded = loaded.predict(X_te)
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


# ── LightGBM ───────────────────────────────────────────────────────────────────

class TestLightGBMForecaster:
    def test_fit_predict(self, synthetic_features, synthetic_returns):
        X_tr, y_tr, X_te, _ = _split(synthetic_features, synthetic_returns)
        model = LightGBMForecaster(horizon=1, n_estimators=50)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(X_te)


# ── Random Forest ──────────────────────────────────────────────────────────────

class TestRandomForestForecaster:
    def test_fit_predict(self, synthetic_features, synthetic_returns):
        X_tr, y_tr, X_te, _ = _split(synthetic_features, synthetic_returns)
        model = RandomForestForecaster(horizon=1, n_estimators=50)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(X_te)

    def test_predict_interval(self, synthetic_features, synthetic_returns):
        X_tr, y_tr, X_te, _ = _split(synthetic_features, synthetic_returns)
        model = RandomForestForecaster(horizon=1, n_estimators=50)
        model.fit(X_tr, y_tr)
        lo, hi = model.predict_interval(X_te)
        assert len(lo) == len(X_te)
        assert (hi >= lo).all()
