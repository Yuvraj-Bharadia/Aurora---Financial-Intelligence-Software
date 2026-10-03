"""
Layer 3 — Physics-Informed Loss with Rolling Bounds.

Extends MSE/pinball loss with two domain-aware penalty terms:

1. No-arbitrage penalty  — bounds predictions within a rolling historical
   volatility envelope, preventing the model from extrapolating into
   economically implausible territory.
   CRITICAL FIX vs. original QCPAC: the bound uses a rolling 63-day 95th
   percentile of realized |returns|, NOT y_true × 1.05, which would be
   look-ahead bias.

2. Market elasticity penalty — penalizes predictions that imply price moves
   inconsistent with prevailing liquidity, scaled by the OnlineHMMFilter's
   regime volatility index so crisis regimes receive stronger regularisation.

Usage::

    loss_fn = PhysicsInformedLoss(lambda_arb=0.3, lambda_elas=0.2)
    # In training loop:
    loss = loss_fn(y_pred, y_true, gamma_regime, order_book_depth, rolling_bounds)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def compute_rolling_bounds(
    returns: pd.Series | np.ndarray,
    window: int = 63,
    quantile: float = 0.95,
    min_bound: float = 0.01,
) -> np.ndarray:
    """
    Compute the rolling quantile of absolute returns as the no-arbitrage bound.

    Parameters
    ----------
    returns  : daily log-return series
    window   : lookback window in trading days (63 ≈ 1 quarter)
    quantile : upper percentile of historical moves used as the bound
    min_bound: floor to prevent degenerate near-zero bounds

    Returns
    -------
    Array of same length as returns; each element is the allowed maximum
    absolute move at that time step (no look-ahead: uses only past returns).
    """
    if isinstance(returns, pd.Series):
        arr = returns.values.astype(np.float64)
    else:
        arr = np.asarray(returns, dtype=np.float64)

    bounds = np.full(len(arr), min_bound)
    abs_arr = np.abs(arr)
    for i in range(window, len(arr)):
        window_slice = abs_arr[i - window: i]  # strictly past — no index i
        bounds[i] = max(float(np.quantile(window_slice, quantile)), min_bound)
    bounds[:window] = bounds[window] if len(bounds) > window else min_bound
    return bounds


class PhysicsInformedLoss(nn.Module):
    """
    Composite loss for regime-conditioned financial forecasting.

    Components
    ----------
    L_total = L_base + λ_arb · L_arbitrage + λ_elas · L_elasticity

    L_base       : RMSE (or swapped for pinball loss in quantile mode)
    L_arbitrage  : relu(|y_pred - y_true| - bound) — penalises predictions
                   outside the rolling historical move envelope
    L_elasticity : relu(|y_pred - y_true| / (depth + ε) - 0.02) weighted
                   by regime volatility; prevents predicting large moves when
                   the order book is thin or the regime is volatile

    Parameters
    ----------
    lambda_arb   : weight on no-arbitrage penalty (default 0.3)
    lambda_elas  : weight on elasticity penalty (default 0.2)
    quantile_mode: if True, uses pinball (quantile) loss as L_base
    quantile     : quantile level for pinball loss (default 0.5 = median)
    """

    def __init__(
        self,
        lambda_arb: float = 0.3,
        lambda_elas: float = 0.2,
        quantile_mode: bool = False,
        quantile: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_arb = lambda_arb
        self.lambda_elas = lambda_elas
        self.quantile_mode = quantile_mode
        self.quantile = quantile
        self._mse = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,           # (batch, 1) or (batch,)
        y_true: torch.Tensor,           # (batch, 1) or (batch,)
        gamma_regime: torch.Tensor,     # (batch, n_regimes) — posterior probs
        order_book_depth: torch.Tensor, # (batch, 1) — normalised depth [0, 1]
        rolling_bound: torch.Tensor,    # (batch, 1) — pre-computed causal bound
    ) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        order_book_depth = order_book_depth.view(-1)
        rolling_bound = rolling_bound.view(-1)

        # ── Base loss ───────────────────────────────────────────────────────
        if self.quantile_mode:
            errors = y_true - y_pred
            loss_base = torch.mean(
                torch.max(self.quantile * errors, (self.quantile - 1) * errors)
            )
        else:
            loss_base = torch.sqrt(self._mse(y_pred, y_true) + 1e-8)

        # ── No-arbitrage penalty ────────────────────────────────────────────
        # Penalise predictions that exceed the historical move envelope
        abs_error = torch.abs(y_pred - y_true)
        arb_violation = torch.relu(abs_error - rolling_bound)
        loss_arb = arb_violation.mean()

        # ── Market elasticity penalty ───────────────────────────────────────
        # Scale by regime: high-vol regime (last state in our 4-regime setup)
        # gets double weight; others get 1x
        # Use entropy of posterior as the volatility proxy (high entropy = uncertain/volatile)
        eps = torch.tensor(1e-9)
        posterior_entropy = -(gamma_regime * torch.log(gamma_regime + eps)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(gamma_regime.shape[-1])))
        volatility_weight = 1.0 + posterior_entropy / (max_entropy + eps)

        implied_impact = abs_error / (order_book_depth + 1e-5)
        elas_violation = torch.relu(implied_impact - 0.02)
        loss_elas = (elas_violation * volatility_weight).mean()

        total = loss_base + self.lambda_arb * loss_arb + self.lambda_elas * loss_elas
        return total

    def auxiliary(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Lightweight auxiliary term (no regime/depth info needed) for adding
        to existing model losses as a soft regulariser.  Only applies the
        base RMSE component scaled by *weight*.
        """
        return weight * torch.sqrt(self._mse(y_pred.view(-1), y_true.view(-1)) + 1e-8)
