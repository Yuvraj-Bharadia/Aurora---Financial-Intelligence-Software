"""
Central configuration management for Aurora.

All sub-systems read their defaults from this module.
Settings can be overridden via environment variables or YAML config files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Directory roots ───────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT_DIR / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
CACHE_DIR = OUTPUTS_DIR / "cache"
MODEL_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"


class DataConfig(BaseSettings):
    """Data ingestion and storage settings."""

    model_config = SettingsConfigDict(env_prefix="AURORA_DATA_")

    data_dir: Path = DATA_DIR
    cache_dir: Path = CACHE_DIR
    cache_ttl_hours: int = 24
    max_retries: int = 3
    retry_backoff: float = 2.0
    request_timeout: int = 30

    # Default universe
    equity_tickers: list[str] = Field(
        default=[
            "SPY", "QQQ", "IWM", "DIA", "VTI",
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "GS", "BAC",
        ]
    )
    macro_series: list[str] = Field(
        default=[
            "VIXCLS",       # CBOE VIX
            "GS10",         # 10-year Treasury yield
            "GS2",          # 2-year Treasury yield
            "DCOILWTICO",   # WTI crude oil
            "DTWEXBGS",     # USD broad index
            "UNRATE",       # Unemployment rate
            "CPIAUCSL",     # CPI
            "FEDFUNDS",     # Federal funds rate
        ]
    )
    default_start: str = "2010-01-01"
    default_end: str = "2024-12-31"


class FeatureConfig(BaseSettings):
    """Feature engineering settings."""

    model_config = SettingsConfigDict(env_prefix="AURORA_FEAT_")

    lookback_windows: list[int] = Field(default=[5, 10, 20, 60, 120, 252])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    hurst_lags: int = 100
    garch_p: int = 1
    garch_q: int = 1


class RegimeConfig(BaseSettings):
    """Regime detection settings."""

    model_config = SettingsConfigDict(env_prefix="AURORA_REGIME_")

    n_regimes: int = 4
    hmm_covariance_type: Literal["full", "diag", "spherical", "tied"] = "full"
    hmm_n_iter: int = 200
    hmm_tol: float = 1e-4
    regime_labels: list[str] = Field(
        default=["bull", "bear", "high_volatility", "sideways"]
    )
    min_regime_duration: int = 5  # trading days
    confidence_threshold: float = 0.6


class ModelConfig(BaseSettings):
    """Forecasting model defaults."""

    model_config = SettingsConfigDict(env_prefix="AURORA_MODEL_")

    model_dir: Path = MODEL_DIR
    forecast_horizons: list[int] = Field(default=[1, 5, 10, 21])
    default_horizon: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio implied as 1 - train - val

    # Deep learning
    sequence_length: int = 60
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    dropout: float = 0.2
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 8
    patience: int = 15
    grad_clip: float = 1.0
    use_gpu: bool = True

    # ARIMA
    arima_max_p: int = 5
    arima_max_q: int = 5
    arima_max_d: int = 2

    # Random Forest
    rf_n_estimators: int = 500
    rf_max_depth: int | None = None
    rf_n_jobs: int = -1

    # XGBoost / LightGBM
    gbm_n_estimators: int = 1000
    gbm_learning_rate: float = 0.05
    gbm_max_depth: int = 6
    gbm_subsample: float = 0.8
    gbm_early_stopping_rounds: int = 50


class EnsembleConfig(BaseSettings):
    """Ensemble and regime-switching settings."""

    model_config = SettingsConfigDict(env_prefix="AURORA_ENS_")

    method: Literal["weighted", "stacking", "bayesian", "regime_adaptive"] = "regime_adaptive"
    min_model_weight: float = 0.05
    weight_lookback: int = 60  # days of recent performance for weighting
    rebalance_frequency: int = 21  # trading days


class BacktestConfig(BaseSettings):
    """Backtesting engine settings."""

    model_config = SettingsConfigDict(env_prefix="AURORA_BT_")

    initial_capital: float = 1_000_000.0
    commission_bps: float = 5.0      # basis points per trade
    slippage_bps: float = 2.0        # basis points
    max_position_size: float = 0.20  # fraction of portfolio
    risk_free_rate: float = 0.04     # annualized
    annualization_factor: int = 252


class SentimentConfig(BaseSettings):
    """Sentiment analysis settings."""

    model_config = SettingsConfigDict(env_prefix="AURORA_SENT_")

    finbert_model: str = "ProsusAI/finbert"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    sentiment_lookback: int = 3   # days for rolling sentiment
    min_articles: int = 3


class APIConfig(BaseSettings):
    """FastAPI service settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    secret_key: str = "change-me-in-production"
    debug: bool = False


class AuroraConfig(BaseSettings):
    """Master configuration aggregating all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    regimes: RegimeConfig = Field(default_factory=RegimeConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AuroraConfig":
        """Load configuration from a YAML file, merging with env vars."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)

    def ensure_directories(self) -> None:
        """Create all required output directories."""
        for d in [DATA_DIR, CACHE_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
settings = AuroraConfig()
settings.ensure_directories()
