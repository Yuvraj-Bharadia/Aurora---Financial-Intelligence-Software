"""
Unit and integration tests for the Aurora regime detection engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.regimes.hmm import GaussianHMMDetector
from aurora.regimes.detector import RegimeDetector


class TestGaussianHMM:
    def test_fit_returns_self(self, synthetic_features):
        hmm = GaussianHMMDetector(n_regimes=3)
        result = hmm.fit(synthetic_features.iloc[:, :5])
        assert result is hmm
        assert hmm.model is not None

    def test_predict_shape(self, synthetic_features):
        feats = synthetic_features.iloc[:, :5]
        hmm = GaussianHMMDetector(n_regimes=3)
        hmm.fit(feats)
        preds = hmm.predict(feats)
        assert len(preds) == len(feats)
        assert "regime" in preds.columns
        assert "regime_id" in preds.columns

    def test_regime_ids_valid(self, synthetic_features):
        feats = synthetic_features.iloc[:, :5]
        hmm = GaussianHMMDetector(n_regimes=3)
        hmm.fit(feats)
        preds = hmm.predict(feats)
        assert preds["regime_id"].between(0, 2).all()

    def test_transition_matrix_rows_sum_to_one(self, synthetic_features):
        feats = synthetic_features.iloc[:, :5]
        hmm = GaussianHMMDetector(n_regimes=3)
        hmm.fit(feats)
        trans = hmm.transition_matrix()
        row_sums = trans.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_confidence_bounded(self, synthetic_features):
        feats = synthetic_features.iloc[:, :5]
        hmm = GaussianHMMDetector(n_regimes=3)
        hmm.fit(feats)
        preds = hmm.predict(feats)
        conf = hmm.regime_confidence(preds)
        assert (conf >= 0).all() and (conf <= 1).all()


class TestRegimeDetector:
    def test_fit_and_predict(self, synthetic_features, synthetic_returns):
        feats = synthetic_features.reindex(synthetic_returns.index)
        detector = RegimeDetector(method="hmm", n_regimes=3)
        detector.fit(feats, synthetic_returns)
        result = detector.predict(feats, synthetic_returns)
        assert "regime" in result.columns
        assert len(result) > 0

    def test_regime_labels_are_strings(self, synthetic_features, synthetic_returns):
        feats = synthetic_features.reindex(synthetic_returns.index)
        detector = RegimeDetector(method="hmm", n_regimes=3)
        detector.fit(feats, synthetic_returns)
        result = detector.predict(feats, synthetic_returns)
        assert result["regime"].dtype == object

    def test_save_load_roundtrip(self, synthetic_features, synthetic_returns, tmp_path):
        feats = synthetic_features.reindex(synthetic_returns.index)
        detector = RegimeDetector(method="hmm", n_regimes=3)
        detector.fit(feats, synthetic_returns)
        save_path = tmp_path / "detector.pkl"
        detector.save(save_path)
        loaded = RegimeDetector.load(save_path)
        result = loaded.predict(feats, synthetic_returns)
        assert "regime" in result.columns

    def test_regime_summary_structure(self, synthetic_features, synthetic_returns):
        feats = synthetic_features.reindex(synthetic_returns.index)
        detector = RegimeDetector(method="hmm", n_regimes=3)
        detector.fit(feats, synthetic_returns)
        regimes = detector.predict(feats, synthetic_returns)
        summary = detector.regime_summary(regimes)
        assert "mean_duration" in summary.columns
        assert "pct_time" in summary.columns
        assert summary["pct_time"].sum() == pytest.approx(1.0, abs=0.01)
