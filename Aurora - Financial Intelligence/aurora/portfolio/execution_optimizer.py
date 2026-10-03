"""
Layer 5 — Execution-Aware Allocation Optimizer.

Replaces Aurora's post-hoc cost deduction with a pre-trade cost model
that is embedded in the optimization objective itself:

    maximize  E[net return]
    subject to
        ∑ |w - w_prev| · (spread/2 + impact(w)) ≤ turnover_budget
        -max_short ≤ w_i ≤ max_long  ∀ i
        ∑ w_i = 1  (or 0 for market-neutral)
        w_i = 0 where NoTradeGate = NO_TRADE

Market impact model (Almgren-Chriss square-root):
    impact_i = impact_coeff · σ_i · √(|Δw_i| · AUM / ADV_i)

Capital is allocated ONLY when:
    |expected net return| > commission + slippage + calibrated VaR downside

Usage::

    opt = ExecutionAwareOptimizer(initial_capital=1_000_000)
    result = opt.optimize(
        forecasts={"SPY": 0.003, "QQQ": -0.001},
        uncertainties={"SPY": 0.008, "QQQ": 0.010},
        spreads={"SPY": 0.0001, "QQQ": 0.0001},
        adv_fracs={"SPY": 0.001, "QQQ": 0.002},
        vols={"SPY": 0.012, "QQQ": 0.018},
        prev_weights={"SPY": 0.5, "QQQ": 0.5},
        risk_modes={"SPY": "FULL_TRADE", "QQQ": "REDUCED"},
        var_95={"SPY": -0.015, "QQQ": -0.022},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from aurora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AllocationResult:
    """
    Output of ExecutionAwareOptimizer.

    Attributes
    ----------
    weights         : final portfolio weights {asset → weight}
    expected_net    : expected return net of all costs (per asset)
    estimated_costs : estimated round-trip cost per asset
    size_multipliers: NoTradeGate multipliers applied
    turnover        : one-way portfolio turnover vs prev_weights
    eligible        : assets that passed the edge filter
    """

    weights: dict[str, float]
    expected_net: dict[str, float]
    estimated_costs: dict[str, float]
    size_multipliers: dict[str, float]
    turnover: float = 0.0
    eligible: list[str] = field(default_factory=list)


class ExecutionCostEstimator:
    """
    Estimates per-asset execution cost using Almgren-Chriss impact model.

    Parameters
    ----------
    commission_bps : one-way commission in basis points
    slippage_bps   : one-way slippage (half-spread) in basis points
    impact_coeff   : Almgren-Chriss permanent impact coefficient (typical: 0.1)
    """

    def __init__(
        self,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        impact_coeff: float = 0.1,
    ) -> None:
        self.commission = commission_bps / 10_000
        self.slippage = slippage_bps / 10_000
        self.impact_coeff = impact_coeff

    def round_trip_cost(
        self,
        ticker: str,
        delta_weight: float,
        sigma: float,
        adv_frac: float,
        borrow_available: bool = True,
        is_short: bool = False,
    ) -> float:
        """
        Estimate total round-trip cost for a weight change.

        Parameters
        ----------
        delta_weight    : change in portfolio weight (|Δw|)
        sigma           : daily return volatility of the asset
        adv_frac        : |order| / ADV (what fraction of daily volume we are)
        borrow_available: whether the stock can be shorted
        is_short        : True if the trade is a short

        Returns
        -------
        Estimated one-way cost as a fraction of portfolio value.
        """
        if is_short and not borrow_available:
            return 1.0  # Prohibitive — can't short this stock

        # Base transaction costs
        one_way_base = self.commission + self.slippage

        # Almgren-Chriss square-root market impact
        abs_delta = abs(delta_weight)
        impact = self.impact_coeff * sigma * np.sqrt(max(adv_frac * abs_delta, 1e-8))

        # Borrow cost for shorts (annualized, prorated to 1 day)
        borrow_daily = 0.0
        if is_short:
            borrow_annual = 0.01  # typical 100bps/yr borrow cost
            borrow_daily = borrow_annual / 252

        total_one_way = one_way_base + impact + borrow_daily
        return total_one_way

    def net_expected_return(
        self,
        gross_return: float,
        delta_weight: float,
        sigma: float,
        adv_frac: float,
        is_short: bool = False,
        borrow_available: bool = True,
    ) -> float:
        """Expected return net of round-trip execution costs."""
        cost = self.round_trip_cost(
            "", delta_weight, sigma, adv_frac, borrow_available, is_short
        )
        # Round-trip deducted from gross
        return gross_return - 2 * cost


class ExecutionAwareOptimizer:
    """
    Portfolio optimizer that integrates execution costs and regime-based
    risk mode gates into the weight computation.

    Optimization is done analytically (closed-form Kelly / cost-adjusted
    mean-variance) rather than requiring cvxpy, keeping it fast and
    dependency-light.

    Parameters
    ----------
    initial_capital   : AUM for computing market impact
    commission_bps    : one-way commission in basis points
    slippage_bps      : one-way spread cost in basis points
    impact_coeff      : Almgren-Chriss coefficient
    max_position      : maximum single-asset weight
    max_short         : maximum short weight (positive number → |short| limit)
    turnover_budget   : maximum one-way portfolio turnover per rebalance
    risk_free_rate    : daily risk-free rate (annualized / 252)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        impact_coeff: float = 0.1,
        max_position: float = 0.20,
        max_short: float = 0.10,
        turnover_budget: float = 0.30,
        risk_free_rate: float = 0.04 / 252,
    ) -> None:
        self._cost = ExecutionCostEstimator(commission_bps, slippage_bps, impact_coeff)
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.max_short = max_short
        self.turnover_budget = turnover_budget
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        forecasts: dict[str, float],
        uncertainties: dict[str, float],
        spreads: dict[str, float],
        adv_fracs: dict[str, float],
        vols: dict[str, float],
        prev_weights: dict[str, float] | None = None,
        risk_modes: dict[str, str] | None = None,
        var_95: dict[str, float] | None = None,
        borrow_available: dict[str, bool] | None = None,
        neutral: bool = False,
    ) -> AllocationResult:
        """
        Compute execution-aware portfolio weights.

        Parameters
        ----------
        forecasts       : expected return per asset
        uncertainties   : half-width of calibrated interval per asset
        spreads         : bid-ask spread per asset (as fraction)
        adv_fracs       : estimated order_size / ADV per asset
        vols            : daily return volatility per asset
        prev_weights    : previous portfolio weights (for turnover penalty)
        risk_modes      : NoTradeGate output per asset
        var_95          : calibrated 95% VaR per asset (negative = loss)
        borrow_available: can we short this asset?
        neutral         : if True, enforce zero net exposure
        """
        assets = list(forecasts.keys())
        n = len(assets)
        if n == 0:
            return AllocationResult({}, {}, {}, {})

        prev = prev_weights or {a: 0.0 for a in assets}
        modes = risk_modes or {a: "FULL_TRADE" for a in assets}
        var = var_95 or {a: float("-inf") for a in assets}
        borrow = borrow_available or {a: True for a in assets}

        size_mult = {
            a: 0.0 if modes.get(a) == "NO_TRADE" else (0.5 if modes.get(a) == "REDUCED" else 1.0)
            for a in assets
        }

        # ── Compute net expected return for each asset ───────────────────
        net_ret: dict[str, float] = {}
        costs: dict[str, float] = {}
        eligible: list[str] = []

        for a in assets:
            f = forecasts[a]
            sigma = vols.get(a, 0.01)
            adv_f = adv_fracs.get(a, 0.001)
            spread = spreads.get(a, 0.0)
            is_short = f < 0
            prev_w = prev.get(a, 0.0)
            delta_w = abs(f / max(max(abs(f) for f in forecasts.values()), 1e-8) * self.max_position - prev_w)

            c = self._cost.round_trip_cost(a, delta_w, sigma, adv_f, borrow.get(a, True), is_short)
            net = self._cost.net_expected_return(f, delta_w, sigma, adv_f, is_short, borrow.get(a, True))
            net_ret[a] = net
            costs[a] = c

            # Edge filter: only trade when net return > calibrated downside risk
            downside = abs(var.get(a, -0.05))
            min_edge = c * 2 + downside
            if abs(net) > min_edge and size_mult[a] > 0:
                eligible.append(a)

        if not eligible:
            logger.info("[ExecutionOptimizer] No assets passed edge filter — flat portfolio")
            return AllocationResult(
                weights={a: 0.0 for a in assets},
                expected_net=net_ret,
                estimated_costs=costs,
                size_multipliers=size_mult,
                eligible=[],
            )

        # ── Signal-proportional sizing (risk-adjusted) ───────────────────
        raw_scores: dict[str, float] = {}
        for a in eligible:
            sigma = vols.get(a, 0.01)
            # Kelly fraction: μ / σ², clipped to max_position
            kelly = net_ret[a] / (sigma ** 2 + 1e-8)
            raw_scores[a] = np.clip(kelly, -self.max_position, self.max_position)

        # Apply size multipliers
        scaled = {a: raw_scores[a] * size_mult[a] for a in eligible}

        # Enforce turnover budget
        weights: dict[str, float] = {a: 0.0 for a in assets}
        total_turnover = sum(abs(scaled.get(a, 0) - prev.get(a, 0)) for a in eligible)
        if total_turnover > self.turnover_budget and total_turnover > 0:
            scale = self.turnover_budget / total_turnover
            scaled = {a: prev.get(a, 0) + (scaled[a] - prev.get(a, 0)) * scale for a in eligible}

        for a in eligible:
            w = np.clip(scaled[a], -self.max_short, self.max_position)
            weights[a] = float(w)

        # Enforce market neutrality if requested
        if neutral:
            net_exp = sum(weights.values())
            if abs(net_exp) > 1e-6:
                adjustment = net_exp / max(len(eligible), 1)
                for a in eligible:
                    weights[a] -= adjustment

        # One-way turnover
        turnover = sum(abs(weights.get(a, 0) - prev.get(a, 0)) for a in assets) / 2

        return AllocationResult(
            weights=weights,
            expected_net=net_ret,
            estimated_costs=costs,
            size_multipliers=size_mult,
            turnover=float(turnover),
            eligible=eligible,
        )

    def optimize_from_series(
        self,
        forecast_series: pd.Series,
        vol_series: pd.Series,
        prev_weights: pd.Series | None = None,
        risk_mode_series: pd.Series | None = None,
    ) -> pd.Series:
        """
        Convenience method: given a time series of forecasts, return a time
        series of portfolio weights (single asset, varying through time).
        """
        weights_out: list[float] = []
        prev_w: dict[str, float] = {"asset": 0.0}

        for ts in forecast_series.index:
            f = float(forecast_series.loc[ts])
            sigma = float(vol_series.loc[ts]) if ts in vol_series.index else 0.01
            mode = str(risk_mode_series.loc[ts]) if risk_mode_series is not None and ts in risk_mode_series.index else "FULL_TRADE"

            result = self.optimize(
                forecasts={"asset": f},
                uncertainties={"asset": sigma},
                spreads={"asset": 0.0001},
                adv_fracs={"asset": 0.001},
                vols={"asset": sigma},
                prev_weights=prev_w,
                risk_modes={"asset": mode},
            )
            w = result.weights.get("asset", 0.0)
            weights_out.append(w)
            prev_w = {"asset": w}

        return pd.Series(weights_out, index=forecast_series.index, name="weight")
