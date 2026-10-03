"""
Macroeconomic feature engineering.

Transforms raw FRED macro series into model-ready features:
yield curve shape, regime-relevant spreads, and normalized indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurora.utils.logging import get_logger

logger = get_logger(__name__)

# FRED series IDs used in feature construction
_VIX = "VIXCLS"
_T10Y = "GS10"
_T2Y = "GS2"
_T3M = "TB3MS"
_FED_FUNDS = "FEDFUNDS"
_CPI = "CPIAUCSL"
_UNEMP = "UNRATE"
_WTI = "DCOILWTICO"
_USD = "DTWEXBGS"
_M2 = "M2SL"
_IG_SPREAD = "BAA10Y"


def yield_curve_slope(macro: pd.DataFrame) -> pd.Series:
    """10Y − 2Y Treasury yield spread (inversion → recession signal)."""
    if _T10Y in macro.columns and _T2Y in macro.columns:
        return (macro[_T10Y] - macro[_T2Y]).rename("yield_curve_10y2y")
    return pd.Series(dtype=float, name="yield_curve_10y2y")


def yield_curve_short(macro: pd.DataFrame) -> pd.Series:
    """10Y − 3M Treasury yield spread."""
    if _T10Y in macro.columns and _T3M in macro.columns:
        return (macro[_T10Y] - macro[_T3M]).rename("yield_curve_10y3m")
    return pd.Series(dtype=float, name="yield_curve_10y3m")


def real_rate(macro: pd.DataFrame) -> pd.Series:
    """
    Approximate real interest rate: Fed Funds − YoY CPI inflation.

    Negative real rates → accommodative monetary policy.
    """
    if _FED_FUNDS in macro.columns and _CPI in macro.columns:
        cpi_yoy = macro[_CPI].pct_change(252) * 100
        return (macro[_FED_FUNDS] - cpi_yoy).rename("real_rate")
    return pd.Series(dtype=float, name="real_rate")


def vix_regime(macro: pd.DataFrame) -> pd.DataFrame:
    """
    VIX-derived features: level, z-score, and quantile bucket.

    VIX > 30 historically marks high-fear / crisis regimes.
    """
    if _VIX not in macro.columns:
        return pd.DataFrame()

    vix = macro[_VIX]
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    vix_quantile = vix.rolling(252).rank(pct=True)

    return pd.DataFrame({
        "vix": vix,
        "vix_z": vix_z,
        "vix_quantile": vix_quantile,
        "vix_log": np.log(vix.clip(lower=1)),
        "vix_change_5d": vix.pct_change(5),
        "vix_change_21d": vix.pct_change(21),
    })


def credit_conditions(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Credit market conditions: IG/HY spreads and their changes.

    Widening spreads → tightening financial conditions.
    """
    frames: dict[str, pd.Series] = {}
    if _IG_SPREAD in macro.columns:
        ig = macro[_IG_SPREAD]
        frames["ig_spread"] = ig
        frames["ig_spread_chg_5d"] = ig.diff(5)
        frames["ig_spread_chg_21d"] = ig.diff(21)
    return pd.DataFrame(frames)


def commodity_features(macro: pd.DataFrame) -> pd.DataFrame:
    """WTI oil and USD index features (risk appetite proxies)."""
    frames: dict[str, pd.Series] = {}
    if _WTI in macro.columns:
        wti = macro[_WTI]
        frames["wti"] = np.log(wti.clip(lower=0.1))
        frames["wti_ret_21d"] = wti.pct_change(21)
    if _USD in macro.columns:
        usd = macro[_USD]
        frames["usd_index"] = usd
        frames["usd_ret_21d"] = usd.pct_change(21)
    return pd.DataFrame(frames)


def money_supply_features(macro: pd.DataFrame) -> pd.DataFrame:
    """M2 money supply growth — leading indicator of liquidity."""
    frames: dict[str, pd.Series] = {}
    if _M2 in macro.columns:
        m2 = macro[_M2]
        frames["m2_yoy"] = m2.pct_change(252) * 100
        frames["m2_mom"] = m2.pct_change(21) * 100
    return pd.DataFrame(frames)


def build_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw macro panel into model-ready features.

    Args:
        macro: Wide DataFrame (DatetimeIndex × FRED series).

    Returns:
        Feature DataFrame with engineered macro signals.
    """
    if macro.empty:
        logger.warning("Macro panel is empty — skipping macro features")
        return pd.DataFrame()

    parts = [
        yield_curve_slope(macro),
        yield_curve_short(macro),
        real_rate(macro),
        vix_regime(macro),
        credit_conditions(macro),
        commodity_features(macro),
        money_supply_features(macro),
    ]

    non_empty = [p for p in parts if not (isinstance(p, pd.DataFrame) and p.empty)
                 and not (isinstance(p, pd.Series) and p.empty)]
    if not non_empty:
        return pd.DataFrame()

    result = pd.concat(non_empty, axis=1)
    return result.sort_index()
