# 🌌 Aurora — Adaptive Unified Regime-Oriented Research Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io)
[![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen.svg)](tests/)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Contributor%20Covenant-blue.svg)](CODE_OF_CONDUCT.md)

**Aurora** is an institutional-grade quantitative research framework for **regime-adaptive
financial forecasting** in non-stationary markets. It combines hidden Markov model regime
detection, a diverse suite of econometric, ML, and deep learning forecasters, FinBERT
sentiment analysis, and a regime-conditioned ensemble to produce forecasts that adapt
dynamically to changing market states.

---

## Core Research Hypothesis

> Financial markets exhibit hidden regime structures. Different predictive models perform
> optimally under different market conditions. A dynamic ensemble conditioned on latent
> regimes can outperform static architectures across MSE, directional accuracy, and
> risk-adjusted returns.

---

## Architecture at a Glance

```
Data (Alpha Vantage · FRED · News · Reddit)
    ↓
Feature Engineering (price · technical · statistical · macro · sentiment)
    ↓
Regime Detection (HMM + Markov Switching ensemble)
    ↓
Per-Regime Forecasters (ARIMA · GARCH · XGBoost · LSTM · Transformer)
    ↓
Regime-Adaptive Ensemble (confidence-weighted dynamic model selection)
    ↓
Backtesting + Portfolio Optimization + Risk Analytics
    ↓
FastAPI Service  |  Streamlit Dashboard  |  MLflow Tracking
```

---

## Feature Highlights

| Component | Details |
|-----------|---------|
| **Data** | Alpha Vantage (daily adjusted OHLCV); FRED macro panel; NewsAPI + Reddit sentiment |
| **Regime Detection** | Gaussian HMM (4 states) + Hamilton Markov Switching; ensemble confidence arbitration |
| **Feature Engineering** | 100+ features: log returns, realized vol, Garman-Klass, RSI, MACD, BB, Hurst, autocorrelation, VIX, yield curve |
| **Forecasters** | ARIMA/SARIMA, GARCH/EGARCH, Random Forest, XGBoost, LightGBM, LSTM (MC-Dropout), BiLSTM, GRU, Transformer (quantile) |
| **Ensemble** | Regime-conditioned softmax weighting learned from walk-forward RMSE; confidence scaling |
| **Sentiment** | FinBERT sentiment scoring, daily aggregation, sentiment momentum, surprise signals |
| **Backtesting** | Long-only, long-short, market-neutral, volatility-targeting; commissions + slippage |
| **Portfolio** | Mean-variance, max Sharpe, risk parity, Hierarchical Risk Parity (HRP), Black-Litterman |
| **Evaluation** | Walk-forward CV, Diebold-Mariano test, Mincer-Zarnowitz rationality test |
| **Optimization** | Optuna TPE/CMA-ES; objectives: RMSE, Sharpe, Calmar |
| **API** | FastAPI: `/forecast`, `/regime`, `/backtest` |
| **Dashboard** | Streamlit: regime overlay, forecast comparison, risk analytics |
| **Tracking** | MLflow experiment logging; Docker + docker-compose |

---

## Quickstart

### 1. Install

```bash
# Clone the repository
git clone https://github.com/your-org/aurora-financial-intelligence
cd aurora-financial-intelligence

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (FRED, NewsAPI, etc.)
```

### 3. Train a model

```bash
aurora-train --ticker SPY --start 2010-01-01 --end 2023-12-31
```

### 4. Run a backtest

```bash
aurora-backtest --ticker SPY --strategy long_short --start 2018-01-01 --end 2023-12-31
```

### 5. Start the API

```bash
aurora-serve
# API docs at http://localhost:8000/docs
```

### 6. Launch the dashboard

```bash
streamlit run aurora/dashboard/app.py
# Dashboard at http://localhost:8501
```

### 7. Docker (full stack)

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Repository Structure

```
aurora/
├── aurora/
│   ├── configs/          # Pydantic settings (AuroraConfig)
│   ├── data/             # Data schemas and types
│   ├── ingestion/        # Alpha Vantage, FRED, News, Reddit adapters
│   ├── features/         # Price, technical, statistical, macro features
│   ├── regimes/          # HMM, Markov Switching, RegimeDetector
│   ├── models/
│   │   ├── econometric/  # ARIMA, GARCH
│   │   ├── ml/           # Random Forest, XGBoost, LightGBM
│   │   └── deep_learning/# LSTM, GRU, BiLSTM, Transformer
│   ├── ensembles/        # RegimeAdaptiveEnsemble
│   ├── sentiment/        # FinBERT scorer, daily aggregator
│   ├── forecasting/      # AuroraForecaster orchestrator
│   ├── evaluation/       # Metrics, WalkForwardEvaluator, DM test
│   ├── backtesting/      # BacktestEngine with cost modelling
│   ├── portfolio/        # MVO, risk parity, HRP optimiser
│   ├── optimization/     # Optuna hyperparameter search
│   ├── visualization/    # Plotly regime, forecast, performance charts
│   ├── api/              # FastAPI application + routes
│   ├── dashboard/        # Streamlit research dashboard
│   ├── pipelines/        # Training + inference orchestrators
│   └── utils/            # Logging, caching, validation
├── notebooks/            # EDA, regime, backtest, validation notebooks
├── experiments/          # 6 reproducible research experiments
├── tests/                # Unit and integration tests (pytest)
├── docs/                 # Architecture, methodology, API docs
├── scripts/              # CLI scripts
├── docker/               # Dockerfile + docker-compose
└── .github/workflows/    # CI: lint, type-check, test, docker build
```

---

## Running Experiments

```bash
# Experiment 1: Static vs Adaptive
python experiments/exp_01_static_vs_adaptive.py

# Experiment 3: Sentiment-Enhanced Forecasting
python experiments/exp_03_sentiment_enhanced.py
```

Results are logged to MLflow (`outputs/mlflow.db`). View with:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_regimes.py -v

# With coverage
pytest tests/ --cov=aurora --cov-report=html
```

---

## Research Papers & References

1. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*, 57(2), 357–384.
2. Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy. *JBES*, 13(3), 253–263.
3. López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample. *JPM*, 42(4), 59–69.
4. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.
5. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 30*.
6. Lim, B., et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *Int. J. Forecasting*, 37(4), 1748–1764.

---

## Causal integrity

Aurora's central methodological claim is that the regime signal reaching a forecaster at time
`t` depends only on observations up to `t`. That claim is enforced by test, not asserted in
prose.

`hmmlearn`'s `predict_proba` returns the **smoothed** posterior from the forward-backward
algorithm, which conditions on the entire sample. Aurora therefore implements a separate
forward-only filter in `aurora/regimes/online_controller.py` and passes only that posterior
downstream.

```bash
pytest tests/test_non_anticipation.py -v
```

The decisive assertion is exact equality, not equality within a tolerance:

```python
max_diff = np.abs(full[:cut] - truncated).max()
assert max_diff == 0.0
```

Truncating the input series must not change any earlier output. A companion control test
confirms that the smoothed posterior does **not** satisfy this property, so the causal test
stays discriminating. If either starts behaving differently, something upstream has begun
reading ahead: full-sample standardisation, a smoothed posterior leaking into features, or a
centred rolling window.

---

## Citing Aurora

If you use Aurora in your research, please cite it. See [CITATION.cff](CITATION.cff), and the
software paper in [`paper/paper.md`](paper/paper.md).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).
Reports of suspected look-ahead contamination or calibration failures take priority over
feature requests.

---

## This folder holds three separate submissions

The patent, the JOSS software paper and the CJSJ research paper are different submissions to
different audiences with different rules, and one hard ordering constraint between them.
See [`submissions/README.md`](submissions/README.md) for the full map.

In short: **this repository is the JOSS submission.** The research paper lives in
`submissions/03-cjsj-research-paper/`. The patent disclosure is confidential, excluded from
git, and must be filed before either of the other two is published.

---

## Scope and limitations

Aurora is a research instrument for studying forecast construction and evaluation. It is not
investment advice and not a trading system.

The published validation covers one equity index over 1990 to 2019. Results on other
instruments, markets or periods have not been established. In that study, classical univariate
baselines remained competitive with every learned model tested, including this one, and
cost-aware abstention produced positive risk-adjusted returns net of modelled costs at only one
of three horizons. Both findings are reported in the paper rather than omitted.

---

## License

MIT License — see [LICENSE](LICENSE).
