# Aurora System Architecture

## Overview

Aurora (Adaptive Unified Regime-Oriented Research Architecture) is an institutional-grade
quantitative research framework for non-stationary financial markets. It integrates
probabilistic regime detection with adaptive model selection to produce forecasts that
are robust across varying market conditions.

---

## Architectural Layers

```
┌─────────────────────────────────────────────────────────┐
│                    AURORA FRAMEWORK                      │
├──────────┬──────────────┬──────────────┬────────────────┤
│  DATA    │   FEATURE    │   REGIME     │   FORECASTING  │
│ LAYER    │  ENGINEERING │  DETECTION   │   ENGINE       │
├──────────┼──────────────┼──────────────┼────────────────┤
│ Yahoo    │ Price feats  │ Gaussian HMM │ ARIMA / GARCH  │
│ Finance  │ Technical    │ Markov       │ Random Forest  │
│ FRED     │ Statistical  │ Switching    │ XGBoost/LGBM   │
│ News API │ Macro        │ Bayesian     │ LSTM / GRU     │
│ Reddit   │ Sentiment    │ Ensemble     │ Transformer    │
├──────────┴──────────────┴──────────────┴────────────────┤
│               REGIME-ADAPTIVE ENSEMBLE                   │
│   Regime → Optimal Model Mapping + Confidence Weighting │
├─────────────────────────────────────────────────────────┤
│        BACKTESTING & PORTFOLIO OPTIMIZATION             │
│   Transaction costs · Slippage · Risk parity · HRP     │
├─────────────────────────────────────────────────────────┤
│        EVALUATION & STATISTICAL VALIDATION              │
│   Walk-forward CV · Diebold-Mariano · Sharpe / Calmar  │
├─────────────────────────────────────────────────────────┤
│              API / DASHBOARD / NOTEBOOKS                 │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Regime-Adaptivity as a First-Class Citizen
Every component — feature engineering, model selection, position sizing —
is aware of the current market regime. Regime detection uses an ensemble of
Gaussian HMM and Markov Switching models to produce probabilistic regime labels
rather than hard classifications.

### 2. Temporal Integrity
The system enforces strict temporal ordering throughout:
- Features use only past information (no look-ahead).
- Walk-forward validation (never random shuffling) for model selection.
- Training windows expand or roll — never peek at future data.

### 3. Uncertainty Quantification
All forecasts come with prediction intervals:
- ARIMA: analytical confidence intervals.
- GARCH: simulation-based variance bands.
- LSTM/GRU: MC-Dropout epistemic uncertainty.
- Transformer: quantile regression (pinball loss).

### 4. Modularity and Configurability
Every model, pipeline stage, and ensemble method is independently configurable
via `aurora/configs/config.py` and environment variables, enabling clean
ablation studies and parameter sensitivity analysis.

---

## Data Flow

```
Raw Data (Yahoo, FRED, News)
        │
        ▼
  IngestionPipeline
  (validates, normalises, caches)
        │
        ▼
  MarketDataBundle
  (prices, OHLCV, macro, news)
        │
        ▼
  FeaturePipeline
  (price + technical + statistical + macro + sentiment)
        │
        ▼
  RegimeDetector.fit(features, returns)
  → RegimeLabels (bull / bear / high_vol / sideways)
        │
  ┌─────┴──────┐
  │            │
  ▼            ▼
Individual   Regime-Adaptive
 Models      Ensemble Weights
  (fit per    (learned per regime
  fold)        via walk-forward)
        │
        ▼
  RegimeAdaptiveEnsemble.predict(features, regimes)
  → forecast, forecast_lower, forecast_upper
        │
        ▼
  BacktestEngine / PortfolioOptimizer
  → Sharpe, CAGR, MaxDD, attribution
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `aurora.ingestion` | Data adapters (Yahoo, FRED, News, Reddit) |
| `aurora.features` | Price, technical, statistical, macro feature engineering |
| `aurora.regimes` | HMM and Markov Switching regime detection |
| `aurora.models` | Econometric, ML, and deep learning forecasters |
| `aurora.ensembles` | Regime-adaptive ensemble weighting |
| `aurora.sentiment` | FinBERT NLP pipeline and temporal aggregation |
| `aurora.evaluation` | Walk-forward CV, forecast and trading metrics |
| `aurora.backtesting` | Event-driven backtest with transaction costs |
| `aurora.portfolio` | MVO, risk parity, HRP portfolio optimisation |
| `aurora.optimization` | Optuna hyperparameter search |
| `aurora.visualization` | Plotly charts: regimes, forecasts, performance |
| `aurora.api` | FastAPI REST service |
| `aurora.dashboard` | Streamlit research dashboard |
| `aurora.pipelines` | Training and inference orchestration |
