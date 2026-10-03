"""
Portfolio optimization using Mean-Variance and Risk-Parity methods.

Implements:
- Mean-Variance Optimization (Markowitz 1952)
- Maximum Sharpe Ratio portfolio (tangency portfolio)
- Minimum Variance portfolio
- Risk Parity (equal-risk contribution)
- Hierarchical Risk Parity (HRP)
- Black-Litterman with regime views
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer.

    Args:
        method: Optimization method.
        risk_free_rate: Annual risk-free rate.
        max_weight: Maximum position weight per asset.
        min_weight: Minimum position weight (use 0 for long-only).
    """

    def __init__(
        self,
        method: Literal[
            "max_sharpe", "min_variance", "risk_parity", "hrp", "equal_weight"
        ] = "max_sharpe",
        risk_free_rate: float | None = None,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
        annualization: int = 252,
    ) -> None:
        self.method = method
        self.risk_free_rate = risk_free_rate or settings.backtest.risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.annualization = annualization

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        regime_views: dict[str, float] | None = None,
    ) -> pd.Series:
        """
        Compute optimal portfolio weights.

        Args:
            returns: Historical return matrix (rows=days, cols=assets).
            expected_returns: Expected return forecasts per asset.
            regime_views: Black-Litterman regime-conditioned views
                         (asset → expected return adjustment).

        Returns:
            Portfolio weights as a Series (sums to 1).
        """
        returns = returns.dropna()
        cov = returns.cov() * self.annualization
        mu = expected_returns if expected_returns is not None else (
            returns.mean() * self.annualization
        )

        if regime_views:
            mu = self._apply_bl_views(mu, cov, regime_views)

        if self.method == "equal_weight":
            n = len(returns.columns)
            return pd.Series(1.0 / n, index=returns.columns)

        elif self.method == "min_variance":
            return self._min_variance(cov)

        elif self.method == "max_sharpe":
            return self._max_sharpe(mu, cov)

        elif self.method == "risk_parity":
            return self._risk_parity(cov)

        elif self.method == "hrp":
            return self._hrp(returns, cov)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ── Optimization methods ───────────────────────────────────────────────────

    def _min_variance(self, cov: pd.DataFrame) -> pd.Series:
        """Global minimum variance portfolio."""
        n = len(cov)
        w0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n

        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ cov.values @ w)

        result = minimize(portfolio_variance, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 1000})
        weights = result.x
        return pd.Series(weights / weights.sum(), index=cov.index)

    def _max_sharpe(self, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
        """Maximum Sharpe ratio (tangency) portfolio."""
        n = len(cov)
        w0 = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        rf = self.risk_free_rate

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu.values)
            port_vol = float(np.sqrt(w @ cov.values @ w))
            return -(port_ret - rf) / (port_vol + 1e-9)

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 1000})
        weights = result.x
        return pd.Series(weights / weights.sum(), index=cov.index)

    def _risk_parity(self, cov: pd.DataFrame) -> pd.Series:
        """
        Equal Risk Contribution portfolio.

        Iterative algorithm: each asset contributes equally to total portfolio risk.
        """
        n = len(cov)
        w = np.ones(n) / n

        def _risk_contributions(w: np.ndarray) -> np.ndarray:
            sigma = np.sqrt(w @ cov.values @ w)
            mrc = cov.values @ w  # marginal risk contributions
            return w * mrc / (sigma + 1e-9)

        def objective(w: np.ndarray) -> float:
            rc = _risk_contributions(w)
            rc_target = np.ones(n) / n * rc.sum()
            return float(np.sum((rc - rc_target) ** 2))

        bounds = [(1e-4, self.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        result = minimize(objective, w, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 2000})
        weights = result.x
        return pd.Series(weights / weights.sum(), index=cov.index)

    def _hrp(self, returns: pd.DataFrame, cov: pd.DataFrame) -> pd.Series:
        """
        Hierarchical Risk Parity (López de Prado 2016).

        Uses hierarchical clustering on the correlation matrix to build a
        diversified portfolio that respects asset clusters.
        """
        corr = returns.corr()
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist.values), method="ward")
        sorted_idx = leaves_list(link)
        sorted_tickers = corr.index[sorted_idx].tolist()

        weights = pd.Series(1.0, index=sorted_tickers)
        items = [sorted_tickers]

        while items:
            items_next = []
            for cluster in items:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]
                left_var = self._cluster_variance(weights[left], cov.loc[left, left])
                right_var = self._cluster_variance(weights[right], cov.loc[right, right])
                alpha = 1 - left_var / (left_var + right_var)
                weights[left] *= alpha
                weights[right] *= 1 - alpha
                items_next.extend([left, right])
            items = items_next

        return (weights / weights.sum()).reindex(cov.index).fillna(0)

    @staticmethod
    def _cluster_variance(w: pd.Series, cov: pd.DataFrame) -> float:
        """Variance of a sub-portfolio."""
        w_arr = w.values / w.sum()
        return float(w_arr @ cov.values @ w_arr)

    def _apply_bl_views(
        self, mu: pd.Series, cov: pd.DataFrame, views: dict[str, float]
    ) -> pd.Series:
        """
        Black-Litterman view integration (simplified).

        Blends equilibrium returns with analyst views proportional
        to view confidence (τ = 0.05 scaling).
        """
        tau = 0.05
        omega = np.diag([tau * cov.loc[k, k] for k in views if k in mu.index])
        P = np.zeros((len(views), len(mu)))
        q = np.zeros(len(views))
        valid_views = [(k, v) for k, v in views.items() if k in mu.index]

        for i, (ticker, expected) in enumerate(valid_views):
            col_idx = list(mu.index).index(ticker)
            P[i, col_idx] = 1.0
            q[i] = expected

        if len(valid_views) == 0:
            return mu

        sigma = cov.values
        inv_sigma = np.linalg.pinv(tau * sigma)
        inv_omega = np.linalg.pinv(omega)
        bl_cov = np.linalg.pinv(inv_sigma + P.T @ inv_omega @ P)
        bl_mu = bl_cov @ (inv_sigma @ mu.values + P.T @ inv_omega @ q)
        return pd.Series(bl_mu, index=mu.index)
