"""
Feature engineering orchestration pipeline.

Combines price, technical, statistical, and macro features
into a single model-ready feature matrix per ticker.

Aurora v2 additions:
  - CausalGuard enforces data availability at every step (Layer 0)
  - Optional HRT layer fuses regime posteriors into feature space (Layer 2)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from aurora.configs.config import settings
from aurora.data.availability import CausalGuard
from aurora.data.schemas import MarketDataBundle
from aurora.features.macro_features import build_macro_features
from aurora.features.price_features import build_price_features
from aurora.features.statistical_features import build_statistical_features
from aurora.features.technical_indicators import build_technical_features
from aurora.utils.logging import get_logger
from aurora.utils.validation import validate_features

logger = get_logger(__name__)

_COMPRESSION = "snappy"


class FeaturePipeline:
    """
    Orchestrates feature generation for the full asset universe.

    Aurora v2: integrates CausalGuard (Layer 0) to prevent look-ahead,
    and optionally applies the HRT layer (Layer 2) to fuse regime posteriors.

    Usage::

        pipeline = FeaturePipeline(enable_causal_guard=True)
        features = pipeline.run(bundle, market_ticker="SPY")
        # features: dict[ticker → feature DataFrame]
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        dropna_threshold: float = 0.5,
        enable_causal_guard: bool = True,
        enable_hrt: bool = False,            # set True after regime fitting
        hrt_n_qubits: int = 32,
    ) -> None:
        self._output_dir = output_dir or (settings.data.data_dir / "features")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._dropna_threshold = dropna_threshold
        self._causal_guard = CausalGuard.default() if enable_causal_guard else None
        self._enable_hrt = enable_hrt
        self._hrt_n_qubits = hrt_n_qubits
        self._hrt_layer = None    # set via attach_hrt()
        self._regime_df: pd.DataFrame | None = None

    def attach_hrt(self, hrt_layer, regime_df: pd.DataFrame) -> "FeaturePipeline":
        """
        Attach a pre-trained HRTLayer and its corresponding regime DataFrame.
        Once attached, run() will append HRT features to every ticker's output.
        """
        self._hrt_layer = hrt_layer
        self._regime_df = regime_df
        self._enable_hrt = True
        return self

    def run(
        self,
        bundle: MarketDataBundle,
        market_ticker: str = "SPY",
        persist: bool = True,
        prediction_time: pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate features for every ticker in the bundle.

        Args:
            bundle: MarketDataBundle with OHLCV and macro data.
            market_ticker: Ticker to use as the market benchmark for beta/corr.
            persist: Write Parquet files to disk.
            prediction_time: UTC timestamp of the prediction; if provided,
                             CausalGuard filters out unavailable feature values.

        Returns:
            Dict mapping ticker → feature DataFrame.
        """
        macro_feats = build_macro_features(bundle.macro)
        market_rets: pd.Series | None = None
        if market_ticker in bundle.ohlcv and not bundle.ohlcv[market_ticker].empty:
            market_rets = np.log(
                bundle.ohlcv[market_ticker]["close"]
                / bundle.ohlcv[market_ticker]["close"].shift(1)
            ).rename("market_ret")

        feature_map: dict[str, pd.DataFrame] = {}

        for ticker, ohlcv in bundle.ohlcv.items():
            if ohlcv.empty:
                logger.warning(f"[Features] Skipping {ticker}: empty OHLCV")
                continue

            try:
                df = self._build_ticker_features(ticker, ohlcv, macro_feats, market_rets)

                # Layer 0: causal guard
                if self._causal_guard is not None and prediction_time is not None:
                    df = self._causal_guard.filter(df, prediction_time)

                validate_features(df)

                # Layer 2: HRT fusion
                if self._enable_hrt and self._hrt_layer is not None and self._regime_df is not None:
                    from aurora.features.hrt_layer import apply_hrt
                    df = apply_hrt(df, self._regime_df, self._hrt_n_qubits, self._hrt_layer)

                if persist:
                    self._save(ticker, df)
                feature_map[ticker] = df
                logger.info(
                    f"[Features] {ticker}: {df.shape[0]} rows × {df.shape[1]} features"
                )
            except Exception as exc:
                logger.error(f"[Features] Failed {ticker}: {exc}")

        return feature_map

    def _build_ticker_features(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        macro_feats: pd.DataFrame,
        market_rets: pd.Series | None,
    ) -> pd.DataFrame:
        """Assemble all feature groups for a single ticker."""
        close = ohlcv["close"]
        log_ret = np.log(close / close.shift(1))

        price_f = build_price_features(ohlcv)
        tech_f = build_technical_features(ohlcv)
        stat_f = build_statistical_features(
            returns=log_ret,
            prices=close,
            market_returns=market_rets,
        )

        frames = [price_f, tech_f, stat_f]
        if not macro_feats.empty:
            aligned_macro = macro_feats.reindex(ohlcv.index, method="ffill")
            frames.append(aligned_macro)

        df = pd.concat(frames, axis=1)

        # Drop columns with too many NaNs, then forward-fill residual gaps
        nan_frac = df.isnull().mean()
        keep_cols = nan_frac[nan_frac <= self._dropna_threshold].index
        df = df[keep_cols].ffill().bfill()

        # Replace infinities with NaN then forward-fill
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        return df

    def _save(self, ticker: str, df: pd.DataFrame) -> None:
        path = self._output_dir / f"{ticker}.parquet"
        df.to_parquet(path, compression=_COMPRESSION)

    def load(self, ticker: str) -> pd.DataFrame:
        """Load pre-computed features for a ticker."""
        path = self._output_dir / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No feature file for {ticker} at {path}")
        return pd.read_parquet(path)

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all pre-computed feature files."""
        result: dict[str, pd.DataFrame] = {}
        for p in self._output_dir.glob("*.parquet"):
            result[p.stem] = pd.read_parquet(p)
        return result
