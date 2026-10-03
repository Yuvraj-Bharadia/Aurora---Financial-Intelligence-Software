"""
Layer 4 — Regime-Conditioned Conformal Calibration.

Applies split conformal prediction independently within each market regime.
Unlike raw MC-Dropout or quantile regression intervals, conformal calibration
provides a mathematical coverage guarantee:

    P(y_true ∈ [ŷ - q, ŷ + q]) ≥ 1 - α  (marginally over the calibration set)

And when applied per-regime:

    P(y_true ∈ interval | regime = r) ≥ 1 - α  (conditional on regime)

Four separate forecast heads are calibrated independently:
  - return     : point return forecast
  - direction  : binary up/down classification
  - volatility : realized volatility estimate
  - tail_risk  : extreme quantile (VaR-like)

Usage::

    cal = RegimeConformalCalibrator(alpha=0.10)
    cal.fit(oof_preds, actuals, regimes_series)
    intervals = cal.predict(new_preds, new_regimes)
    print(cal.coverage_report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from aurora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibratedInterval:
    """
    A calibrated prediction interval for one time step.

    Attributes
    ----------
    timestamp    : bar date
    regime       : regime label at this time step
    point        : point forecast (return)
    lower        : calibrated lower bound (1 - α/2 coverage)
    upper        : calibrated upper bound (1 - α/2 coverage)
    direction    : probability of positive return
    vol_forecast : calibrated volatility estimate
    var_95       : calibrated 95% VaR (negative number — a loss)
    half_width   : upper - point (for sizing decisions)
    covered      : True if actual fell inside [lower, upper] (set post-hoc)
    """

    timestamp: pd.Timestamp
    regime: str
    point: float
    lower: float
    upper: float
    direction: float = 0.5
    vol_forecast: float = float("nan")
    var_95: float = float("nan")
    half_width: float = float("nan")
    covered: Optional[bool] = None

    def __post_init__(self) -> None:
        self.half_width = self.upper - self.point


class RegimeConformalCalibrator:
    """
    Split conformal calibration stratified by market regime.

    Calibration procedure
    ---------------------
    1. Collect out-of-fold (OOF) predictions and actuals alongside
       their regime labels (strictly no look-ahead).
    2. For each regime, compute conformity scores:
           s_i = |actual_i - predicted_i|   (for return head)
    3. Compute the (1 - α)(1 + 1/n)-th quantile of the per-regime scores.
       This q̂ is the conformal width for that regime.
    4. At test time: interval = [ŷ - q̂_regime, ŷ + q̂_regime].

    Parameters
    ----------
    alpha        : miscoverage level (0.10 → 90% coverage)
    min_calib_n  : minimum samples per regime to compute a valid quantile
    fallback_to_global : if a regime has too few calibration samples,
                         use the global (all-regime) quantile
    """

    def __init__(
        self,
        alpha: float = 0.10,
        min_calib_n: int = 30,
        fallback_to_global: bool = True,
    ) -> None:
        self.alpha = alpha
        self.min_calib_n = min_calib_n
        self.fallback_to_global = fallback_to_global

        # Calibration quantiles per regime (fitted by fit())
        self._q_return: dict[str, float] = {}
        self._q_vol: dict[str, float] = {}
        self._q_var: dict[str, float] = {}
        self._q_global_return: float = float("nan")
        self._q_global_vol: float = float("nan")

        # Coverage tracking
        self._coverage_log: list[dict] = []
        self._fitted = False

    # ── Fitting ────────────────────────────────────────────────────────────

    def fit(
        self,
        oof_return_preds: pd.Series,
        oof_vol_preds: pd.Series,
        actuals: pd.Series,
        realized_vols: pd.Series,
        regimes: pd.Series,
    ) -> "RegimeConformalCalibrator":
        """
        Fit conformal quantiles from out-of-fold predictions.

        Parameters
        ----------
        oof_return_preds : OOF point return forecasts (DatetimeIndex)
        oof_vol_preds    : OOF volatility forecasts
        actuals          : realized returns
        realized_vols    : realized 5-day realized volatility
        regimes          : regime label per date
        """
        common = (
            oof_return_preds.index
            .intersection(actuals.index)
            .intersection(regimes.index)
        )
        ret_pred = oof_return_preds.reindex(common)
        vol_pred = oof_vol_preds.reindex(common)
        actual = actuals.reindex(common)
        real_vol = realized_vols.reindex(common)
        regime = regimes.reindex(common)

        # Global quantile (fallback)
        ret_scores = (actual - ret_pred).abs().values
        self._q_global_return = self._conformal_quantile(ret_scores, self.alpha)

        vol_scores = (real_vol - vol_pred).abs().dropna().values
        self._q_global_vol = self._conformal_quantile(vol_scores, self.alpha)

        for reg_name in regime.dropna().unique():
            mask = regime == reg_name
            if mask.sum() < self.min_calib_n:
                logger.warning(
                    f"[Conformal] Regime '{reg_name}' has only {mask.sum()} "
                    f"calibration samples (min={self.min_calib_n}) — will use global fallback"
                )
                continue

            r_scores = (actual[mask] - ret_pred[mask]).abs().values
            self._q_return[reg_name] = self._conformal_quantile(r_scores, self.alpha)

            v_scores = (real_vol[mask] - vol_pred[mask]).abs().dropna().values
            if len(v_scores) >= self.min_calib_n:
                self._q_vol[reg_name] = self._conformal_quantile(v_scores, self.alpha)

            # VaR calibration: tail losses
            tail_losses = actual[mask][actual[mask] < 0].values
            if len(tail_losses) >= 10:
                self._q_var[reg_name] = float(np.quantile(tail_losses, self.alpha))

            logger.info(
                f"[Conformal] {reg_name}: q_return={self._q_return[reg_name]:.4f}"
                + (f" q_vol={self._q_vol.get(reg_name, float('nan')):.4f}"
                   if reg_name in self._q_vol else "")
            )

        self._fitted = True
        return self

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(
        self,
        return_preds: pd.Series,
        vol_preds: pd.Series,
        regimes: pd.Series,
        direction_probs: pd.Series | None = None,
    ) -> list[CalibratedInterval]:
        """
        Wrap point forecasts in calibrated intervals.

        Parameters
        ----------
        return_preds    : point return forecast (DatetimeIndex)
        vol_preds       : volatility forecast
        regimes         : current regime label per date
        direction_probs : P(return > 0) per date; if None defaults to 0.5
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        intervals: list[CalibratedInterval] = []
        common = return_preds.index.intersection(regimes.index)

        for ts in common:
            reg = str(regimes.loc[ts])
            q_ret = self._q_return.get(
                reg, self._q_global_return if self.fallback_to_global else float("nan")
            )
            q_vol = self._q_vol.get(reg, self._q_global_vol)
            q_var = self._q_var.get(reg, float("nan"))

            pt = float(return_preds.loc[ts])
            vol = float(vol_preds.loc[ts]) if ts in vol_preds.index else float("nan")
            dir_p = float(direction_probs.loc[ts]) if direction_probs is not None and ts in direction_probs.index else 0.5

            intervals.append(CalibratedInterval(
                timestamp=ts,
                regime=reg,
                point=pt,
                lower=pt - q_ret,
                upper=pt + q_ret,
                direction=dir_p,
                vol_forecast=vol + q_vol if not np.isnan(vol) else float("nan"),
                var_95=q_var,
            ))

        return intervals

    def update_coverage(
        self, intervals: list[CalibratedInterval], actuals: pd.Series
    ) -> "RegimeConformalCalibrator":
        """Mark each interval as covered or not after realisation."""
        actual_map = actuals.to_dict()
        for ci in intervals:
            if ci.timestamp in actual_map:
                act = float(actual_map[ci.timestamp])
                ci.covered = ci.lower <= act <= ci.upper
                self._coverage_log.append({
                    "timestamp": ci.timestamp,
                    "regime": ci.regime,
                    "covered": ci.covered,
                    "width": ci.upper - ci.lower,
                })
        return self

    def coverage_report(self) -> pd.DataFrame:
        """
        Empirical coverage rate by regime.
        Should be ≥ (1 - alpha) for valid calibration.
        """
        if not self._coverage_log:
            return pd.DataFrame(columns=["regime", "coverage", "mean_width", "n"])
        df = pd.DataFrame(self._coverage_log)
        report = df.groupby("regime").agg(
            coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            n=("covered", "count"),
        ).round(4)
        target = 1 - self.alpha
        report["target_coverage"] = target
        report["calibrated"] = report["coverage"] >= target - 0.02  # 2% tolerance
        return report

    # ── Sizing hint ─────────────────────────────────────────────────────────

    def edge_exceeds_costs(
        self,
        interval: CalibratedInterval,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
    ) -> bool:
        """
        Return True only when the expected edge (|point forecast|) exceeds
        total round-trip costs plus the calibrated downside risk.

        This is the capital allocation gate: no position when not getting paid.
        """
        total_cost_bps = (commission_bps + slippage_bps) * 2  # round-trip
        total_cost = total_cost_bps / 10_000

        downside_risk = abs(interval.lower) if interval.point >= 0 else abs(interval.upper)
        min_required_edge = total_cost + downside_risk
        return abs(interval.point) > min_required_edge

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """
        Finite-sample corrected conformal quantile:
        q̂ = ⌈(1 - α)(n + 1)⌉ / n -th quantile.
        """
        n = len(scores)
        if n == 0:
            return float("nan")
        level = min((1 - alpha) * (1 + 1 / n), 1.0)
        return float(np.quantile(scores, level))
