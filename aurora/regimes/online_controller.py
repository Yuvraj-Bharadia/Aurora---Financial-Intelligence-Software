"""
Layer 1 — Online Regime-State Controller.

Replaces hmmlearn's smoothed posterior (which uses future observations)
with a causal forward-pass-only filtered posterior:

    P(regime_t | observations_1 … observations_t)

The RegimeStateVector enriches the scalar regime label with transition risk,
data freshness, market liquidity, and entropy — enabling downstream components
to make principled decisions about position sizing and trade execution.

The NoTradeGate converts the state vector into a discrete risk mode:
FULL_TRADE, REDUCED, or NO_TRADE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from aurora.utils.logging import get_logger

logger = get_logger(__name__)

RiskMode = Literal["FULL_TRADE", "REDUCED", "NO_TRADE"]


@dataclass
class RegimeStateVector:
    """
    Rich regime state at a single point in time.

    Attributes
    ----------
    timestamp       : bar timestamp
    filtered_posterior : P(regime | past observations) — never smoothed
    regime_id       : argmax of filtered_posterior
    regime_label    : string label mapped from regime_id
    transition_risk : 1 - P(stay in current regime next step)
    state_uncertainty : Shannon entropy of filtered_posterior (bits)
    data_freshness  : fraction of features passing causal guard at this bar
    liquidity_score : 0–1; 1 = highly liquid, 0 = crisis liquidity
    risk_mode       : FULL_TRADE / REDUCED / NO_TRADE
    """

    timestamp: pd.Timestamp
    filtered_posterior: np.ndarray
    regime_id: int
    regime_label: str
    transition_risk: float
    state_uncertainty: float
    data_freshness: float = 1.0
    liquidity_score: float = 1.0
    risk_mode: RiskMode = "FULL_TRADE"

    @property
    def confidence(self) -> float:
        return float(self.filtered_posterior.max())


class OnlineHMMFilter:
    """
    Runs the HMM forward algorithm step-by-step, producing the filtered
    posterior at each time step without using any future information.

    This is a wrapper around a *fitted* GaussianHMMDetector (hmmlearn).
    It extracts the transition matrix and emission parameters from the
    fitted hmmlearn model and reimplements only the forward pass.

    Parameters
    ----------
    hmm_detector : fitted GaussianHMMDetector instance
    regime_labels : dict mapping state_id → regime name
    """

    def __init__(
        self,
        hmm_detector,  # GaussianHMMDetector — avoid circular import
        regime_labels: dict[int, str] | None = None,
    ) -> None:
        if hmm_detector.model is None:
            raise RuntimeError("HMM detector must be fitted before creating OnlineHMMFilter")

        model = hmm_detector.model
        self._A = model.transmat_.astype(np.float64)           # (n_states, n_states)
        self._means = model.means_.astype(np.float64)          # (n_states, n_features)
        self._covars = model.covars_.astype(np.float64)        # shape depends on covariance_type
        self._covar_type = model.covariance_type
        self._n_states = model.n_components
        self._scaler = hmm_detector.scaler
        self._regime_labels = regime_labels or hmm_detector.regime_labels

        # Uniform initial distribution
        self._log_prior = np.log(np.ones(self._n_states) / self._n_states)

    # ── Public API ─────────────────────────────────────────────────────────

    def filter_sequence(
        self,
        features: pd.DataFrame,
        freshness_series: pd.Series | None = None,
        liquidity_series: pd.Series | None = None,
    ) -> list[RegimeStateVector]:
        """
        Run the forward algorithm over the full sequence and return one
        RegimeStateVector per row.

        Parameters
        ----------
        features : scaled or raw feature DataFrame (will be scaled internally)
        freshness_series : optional per-row data freshness score [0, 1]
        liquidity_series : optional per-row liquidity score [0, 1]
        """
        feat = features.ffill().bfill()
        X = self._scaler.transform(feat.values)

        log_alpha = self._log_prior.copy()
        results: list[RegimeStateVector] = []

        for t, (idx, row) in enumerate(zip(feat.index, X)):
            log_emission = self._log_emission(row)

            if t == 0:
                log_alpha = self._log_prior + log_emission
            else:
                # log-sum-exp over previous states weighted by transition probs
                log_alpha_mat = log_alpha[:, None] + np.log(self._A + 1e-300)
                log_alpha = np.logaddexp.reduce(log_alpha_mat, axis=0) + log_emission

            # Normalise → filtered posterior
            log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha)
            posterior = np.exp(log_alpha_norm)

            regime_id = int(np.argmax(posterior))
            label = self._regime_labels.get(regime_id, f"state_{regime_id}")

            # Transition risk: probability of leaving current regime
            p_stay = float(self._A[regime_id, regime_id])
            t_risk = 1.0 - p_stay

            # Entropy in bits
            entropy = float(-np.sum(posterior * np.log2(posterior + 1e-300)))

            freshness = float(freshness_series.iloc[t]) if freshness_series is not None else 1.0
            liquidity = float(liquidity_series.iloc[t]) if liquidity_series is not None else 1.0

            rsv = RegimeStateVector(
                timestamp=idx,
                filtered_posterior=posterior,
                regime_id=regime_id,
                regime_label=label,
                transition_risk=t_risk,
                state_uncertainty=entropy,
                data_freshness=freshness,
                liquidity_score=liquidity,
            )
            results.append(rsv)

        return results

    def filter_step(
        self,
        obs: np.ndarray,
        log_alpha_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Single-step online update.

        Returns (log_alpha_t, posterior_t) where log_alpha_t is passed
        back on the next call.
        """
        obs_scaled = self._scaler.transform(obs.reshape(1, -1))[0]
        log_emission = self._log_emission(obs_scaled)
        log_alpha_mat = log_alpha_prev[:, None] + np.log(self._A + 1e-300)
        log_alpha = np.logaddexp.reduce(log_alpha_mat, axis=0) + log_emission
        log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha)
        return log_alpha, np.exp(log_alpha_norm)

    def to_dataframe(self, state_vectors: list[RegimeStateVector]) -> pd.DataFrame:
        """Convert list of RegimeStateVectors to a DataFrame (for downstream use)."""
        rows = []
        n_states = self._n_states
        for sv in state_vectors:
            row = {
                "regime": sv.regime_label,
                "regime_id": sv.regime_id,
                "regime_confidence": sv.confidence,
                "transition_risk": sv.transition_risk,
                "state_uncertainty": sv.state_uncertainty,
                "data_freshness": sv.data_freshness,
                "liquidity_score": sv.liquidity_score,
                "risk_mode": sv.risk_mode,
            }
            for i in range(n_states):
                label = self._regime_labels.get(i, f"state_{i}")
                row[f"prob_{label}"] = float(sv.filtered_posterior[i])
            rows.append(row)
        return pd.DataFrame(rows, index=[sv.timestamp for sv in state_vectors])

    # ── Internals ──────────────────────────────────────────────────────────

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """Log p(obs | state) for all states under Gaussian emissions."""
        log_probs = np.zeros(self._n_states)
        for s in range(self._n_states):
            mean = self._means[s]
            if self._covar_type == "full":
                cov = self._covars[s]
            elif self._covar_type == "diag":
                cov = np.diag(self._covars[s])
            elif self._covar_type == "spherical":
                cov = np.eye(len(mean)) * self._covars[s]
            else:  # tied
                cov = self._covars[0]

            diff = obs - mean
            try:
                sign, logdet = np.linalg.slogdet(cov)
                cov_inv = np.linalg.inv(cov + np.eye(len(mean)) * 1e-6)
                maha = float(diff @ cov_inv @ diff)
                k = len(mean)
                log_probs[s] = -0.5 * (k * np.log(2 * np.pi) + logdet + maha)
            except np.linalg.LinAlgError:
                log_probs[s] = -1e9
        return log_probs


class NoTradeGate:
    """
    Converts a RegimeStateVector into a RiskMode signal.

    Rules (all thresholds configurable):
    - NO_TRADE    : confidence < min_confidence  OR  transition_risk > max_transition_risk
                   OR  data_freshness < min_freshness  OR  liquidity_score < min_liquidity
    - REDUCED     : any single threshold is borderline (within the buffer zone)
    - FULL_TRADE  : all metrics clear the thresholds plus buffer

    The size multipliers attached to each mode feed into the execution
    optimizer (Layer 5) to scale position sizes.
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        max_transition_risk: float = 0.40,
        min_freshness: float = 0.70,
        min_liquidity: float = 0.30,
        buffer_fraction: float = 0.10,   # ±10% buffer around each threshold
        reduced_size_multiplier: float = 0.50,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_transition_risk = max_transition_risk
        self.min_freshness = min_freshness
        self.min_liquidity = min_liquidity
        self.buffer = buffer_fraction
        self.reduced_size = reduced_size_multiplier

    def evaluate(self, sv: RegimeStateVector) -> RegimeStateVector:
        """Assign risk_mode to the state vector in-place and return it."""
        hard_no_trade = (
            sv.confidence < self.min_confidence
            or sv.transition_risk > self.max_transition_risk
            or sv.data_freshness < self.min_freshness
            or sv.liquidity_score < self.min_liquidity
        )
        if hard_no_trade:
            sv.risk_mode = "NO_TRADE"
            return sv

        # Buffer zone: any metric within [threshold * (1 - buffer), threshold * (1 + buffer)]
        borderline = (
            sv.confidence < self.min_confidence * (1 + self.buffer)
            or sv.transition_risk > self.max_transition_risk * (1 - self.buffer)
            or sv.data_freshness < self.min_freshness * (1 + self.buffer)
            or sv.liquidity_score < self.min_liquidity * (1 + self.buffer)
        )
        sv.risk_mode = "REDUCED" if borderline else "FULL_TRADE"
        return sv

    def evaluate_series(
        self, state_vectors: list[RegimeStateVector]
    ) -> list[RegimeStateVector]:
        return [self.evaluate(sv) for sv in state_vectors]

    def size_multiplier(self, risk_mode: RiskMode) -> float:
        if risk_mode == "NO_TRADE":
            return 0.0
        if risk_mode == "REDUCED":
            return self.reduced_size
        return 1.0

    def to_series(self, state_vectors: list[RegimeStateVector]) -> pd.Series:
        """Return a Series of size multipliers indexed by timestamp."""
        return pd.Series(
            {sv.timestamp: self.size_multiplier(sv.risk_mode) for sv in state_vectors},
            name="size_multiplier",
        )
