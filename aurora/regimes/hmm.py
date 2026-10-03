"""
Hidden Markov Model regime detection.

Fits a Gaussian HMM to market features and assigns each trading day
to a latent regime (bull, bear, high-volatility, sideways).

The HMM is initialized with k-means++ to improve convergence stability.
We assign regime labels by matching each HMM state to its economic
interpretation based on mean return and mean volatility.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


# Canonical regime label assignment heuristics
# After fitting, states are labeled by (mean_return_rank, volatility_rank)
_REGIME_LABEL_MAP = {
    (1, 0): "bull",          # high return, low vol
    (0, 0): "sideways",      # low return, low vol
    (1, 1): "high_volatility",  # high return, high vol
    (0, 1): "bear",          # low return, high vol
}


class GaussianHMMDetector:
    """
    Gaussian Hidden Markov Model for regime detection.

    Attributes:
        n_regimes: Number of hidden states (default: 4).
        covariance_type: HMM covariance structure.
        model: Fitted hmmlearn GaussianHMM instance.
        scaler: StandardScaler fitted on training data.
        regime_labels: Maps integer state → regime name.
    """

    def __init__(
        self,
        n_regimes: int | None = None,
        covariance_type: str | None = None,
        n_iter: int | None = None,
        tol: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes or settings.regimes.n_regimes
        self.covariance_type = covariance_type or settings.regimes.hmm_covariance_type
        self.n_iter = n_iter or settings.regimes.hmm_n_iter
        self.tol = tol or settings.regimes.hmm_tol
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: hmm.GaussianHMM | None = None
        self.regime_labels: dict[int, str] = {}
        self._feature_names: list[str] = []

    def fit(self, features: pd.DataFrame) -> "GaussianHMMDetector":
        """
        Fit the HMM on a feature matrix.

        Args:
            features: DataFrame of features (rows = trading days).
                      NaN rows are dropped before fitting.

        Returns:
            self (for method chaining).
        """
        clean = features.dropna()
        if len(clean) < self.n_regimes * 10:
            raise ValueError(
                f"Insufficient data: {len(clean)} rows for {self.n_regimes} regimes"
            )

        self._feature_names = list(clean.columns)
        X = self.scaler.fit_transform(clean.values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
                init_params="kmeans",
            )
            self.model.fit(X)

        logger.info(
            f"HMM fitted: {self.n_regimes} regimes | "
            f"log-likelihood={self.model.score(X):.2f}"
        )
        self._assign_labels(clean)
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Assign regimes and compute state probabilities.

        Args:
            features: Feature DataFrame aligned to the trading calendar.

        Returns:
            DataFrame with columns:
            - regime_id: Integer state (0 … n_regimes-1)
            - regime: Regime name string
            - prob_bull, prob_bear, …: Posterior state probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Align to fitted features, handle NaN via forward-fill
        feat = features[self._feature_names].ffill().bfill()
        X = self.scaler.transform(feat.values)

        states = self.model.predict(X)
        log_probs = self.model.predict_proba(X)

        result = pd.DataFrame(index=features.index)
        result["regime_id"] = states
        result["regime"] = [self.regime_labels.get(s, f"state_{s}") for s in states]

        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f"state_{i}")
            result[f"prob_{label}"] = log_probs[:, i]

        return result

    def regime_confidence(self, probabilities: pd.DataFrame) -> pd.Series:
        """Maximum posterior probability (confidence in the predicted regime)."""
        prob_cols = [c for c in probabilities.columns if c.startswith("prob_")]
        return probabilities[prob_cols].max(axis=1).rename("regime_confidence")

    def transition_matrix(self) -> pd.DataFrame:
        """Return the estimated regime transition probability matrix."""
        if self.model is None:
            raise RuntimeError("Not fitted.")
        labels = [self.regime_labels.get(i, f"state_{i}") for i in range(self.n_regimes)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels,
        )

    def _assign_labels(self, features: pd.DataFrame) -> None:
        """
        Label each HMM state by economic interpretation.

        States are ranked by:
        - Mean of the first feature column (proxy for returns)
        - Mean of the second feature column (proxy for volatility)
        """
        if self.model is None:
            return

        means = self.model.means_  # shape (n_states, n_features)
        ret_col = 0
        vol_col = 1 if means.shape[1] > 1 else 0

        ret_rank = (means[:, ret_col] > np.median(means[:, ret_col])).astype(int)
        vol_rank = (means[:, vol_col] > np.median(means[:, vol_col])).astype(int)

        labels = settings.regimes.regime_labels
        for state in range(self.n_regimes):
            key = (ret_rank[state], vol_rank[state])
            # Fall back to sequential label if map lookup misses
            self.regime_labels[state] = _REGIME_LABEL_MAP.get(
                key, labels[state % len(labels)]
            )

        logger.info(f"Regime labels assigned: {self.regime_labels}")

    def state_statistics(self, features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-regime descriptive statistics.

        Returns:
            DataFrame with mean, std, and count per regime.
        """
        combined = features.join(regimes[["regime"]], how="inner")
        stats = combined.groupby("regime").agg(["mean", "std", "count"])
        return stats


class DifferentiableHMMDetector:
    """
    Layer 3 — PyTorch-differentiable HMM detector.

    Wraps the _DifferentiableHMM from QCPACForecaster so the regime
    assignments and HRT features can be trained end-to-end with the
    rest of the QCPAC pipeline via a single backward pass.

    Exposes the same fit/predict interface as GaussianHMMDetector so it
    can be used as a drop-in replacement anywhere GaussianHMMDetector
    is accepted.
    """

    def __init__(
        self,
        n_regimes: int | None = None,
        seq_len: int = 20,
        n_iter: int = 200,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.n_regimes = n_regimes or settings.regimes.n_regimes
        self.seq_len = seq_len
        self.n_iter = n_iter
        self.lr = lr
        self.device = device
        self.scaler = StandardScaler()
        self.regime_labels: dict[int, str] = {}
        self._diff_hmm = None
        self._n_features: int = 0

    def fit(self, features: pd.DataFrame) -> "DifferentiableHMMDetector":
        import torch
        import torch.optim as optim

        try:
            from aurora.models.deep_learning.qcpac import _DifferentiableHMM
        except ImportError as e:
            raise ImportError("QCPAC module required for DifferentiableHMMDetector") from e

        clean = features.dropna()
        self._n_features = clean.shape[1]
        X_scaled = self.scaler.fit_transform(clean.values)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        self._diff_hmm = _DifferentiableHMM(
            n_states=self.n_regimes, n_features=self._n_features
        ).to(self.device)

        optimizer = optim.Adam(self._diff_hmm.parameters(), lr=self.lr)

        for step in range(self.n_iter):
            optimizer.zero_grad()
            posteriors = self._diff_hmm(X_t.unsqueeze(0))  # (1, T, n_states)
            # Maximize entropy encourages regime separation
            entropy = -(posteriors * (posteriors + 1e-9).log()).sum(-1).mean()
            loss = -entropy
            loss.backward()
            optimizer.step()

        # Assign labels based on fitted means
        means = self._diff_hmm.means.detach().cpu().numpy()
        ret_col = 0
        vol_col = 1 if means.shape[1] > 1 else 0
        ret_rank = (means[:, ret_col] > np.median(means[:, ret_col])).astype(int)
        vol_rank = (means[:, vol_col] > np.median(means[:, vol_col])).astype(int)
        labels = settings.regimes.regime_labels
        for state in range(self.n_regimes):
            key = (int(ret_rank[state]), int(vol_rank[state]))
            self.regime_labels[state] = _REGIME_LABEL_MAP.get(
                key, labels[state % len(labels)]
            )

        logger.info(f"DifferentiableHMM fitted: labels={self.regime_labels}")
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        import torch

        if self._diff_hmm is None:
            raise RuntimeError("Not fitted. Call fit() first.")

        feat = features.ffill().bfill()
        feat_cols = feat.columns[:self._n_features]
        X_scaled = self.scaler.transform(feat[feat_cols].values)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            posteriors = self._diff_hmm(X_t.unsqueeze(0))[0]  # (T, n_states)
        posteriors_np = posteriors.cpu().numpy()

        states = posteriors_np.argmax(axis=1)
        result = pd.DataFrame(index=features.index)
        result["regime_id"] = states
        result["regime"] = [self.regime_labels.get(int(s), f"state_{s}") for s in states]
        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f"state_{i}")
            result[f"prob_{label}"] = posteriors_np[:, i]
        return result

    def regime_confidence(self, probabilities: pd.DataFrame) -> pd.Series:
        prob_cols = [c for c in probabilities.columns if c.startswith("prob_")]
        return probabilities[prob_cols].max(axis=1).rename("regime_confidence")

    def transition_matrix(self) -> pd.DataFrame:
        import torch
        if self._diff_hmm is None:
            raise RuntimeError("Not fitted.")
        A = torch.softmax(self._diff_hmm.transition_logits, dim=-1).detach().cpu().numpy()
        labels = [self.regime_labels.get(i, f"state_{i}") for i in range(self.n_regimes)]
        return pd.DataFrame(A, index=labels, columns=labels)
