"""
Notebook 07 — Backtesting & Performance Attribution
====================================================
Demonstrates the Aurora backtesting engine:
- Strategy comparison: long-only vs long-short vs vol-targeting
- Regime-attributed performance
- Risk analytics (VaR, CVaR, drawdown)
- Portfolio optimisation comparison
"""

# %% [markdown]
# # Aurora — Notebook 07: Backtesting

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from aurora.backtesting.engine import BacktestEngine
from aurora.evaluation.metrics import trading_metrics, sharpe_ratio
from aurora.visualization.forecast_plots import (
    plot_cumulative_returns, plot_drawdown, plot_rolling_metrics,
)

# %% [markdown]
# ## 1. Load Data and Run Forecasts

# %%
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.features.pipeline import FeaturePipeline

ingestion = IngestionPipeline()
bundle = ingestion.run(tickers=["SPY"], start="2015-01-01", end="2023-12-31")
feat_pipeline = FeaturePipeline()
features = feat_pipeline.run(bundle)["SPY"]
ohlcv = bundle.ohlcv["SPY"]
close = ohlcv["close"].reindex(features.index)
returns = np.log(close / close.shift(1)).dropna()
features = features.reindex(returns.index)

# Simple momentum signal as forecast proxy
signal = returns.rolling(20).mean().dropna()
prices = close.reindex(signal.index)

# %% [markdown]
# ## 2. Run Multiple Strategy Backtests

# %%
engine = BacktestEngine(initial_capital=1_000_000, commission_bps=5, slippage_bps=2)
strategy_results = {}

for strategy in ["long_only", "long_short", "volatility_targeting"]:
    result = engine.run(signal, prices, strategy=strategy)
    strategy_results[strategy] = result
    m = result.metrics
    print(f"\n{strategy.upper()}")
    print(f"  Sharpe:       {m['sharpe']:.3f}")
    print(f"  CAGR:         {m['cagr']:.2%}")
    print(f"  Max Drawdown: {m['max_drawdown']:.2%}")
    print(f"  Hit Ratio:    {m['hit_ratio']:.2%}")

# Buy & Hold benchmark
bh_returns = returns.reindex(signal.index)

# %% [markdown]
# ## 3. Cumulative Returns Comparison

# %%
returns_dict = {strat: res.returns for strat, res in strategy_results.items()}
returns_dict["Buy & Hold"] = bh_returns
plot_cumulative_returns(returns_dict, title="Strategy Cumulative Returns (2015–2023)").show()

# %% [markdown]
# ## 4. Drawdown Analysis

# %%
plot_drawdown(returns_dict).show()

# %% [markdown]
# ## 5. Rolling Sharpe Ratio

# %%
plot_rolling_metrics(
    {k: v for k, v in returns_dict.items() if k != "Buy & Hold"},
    metric="sharpe", window=63,
    title="Rolling 63-Day Sharpe Ratio",
).show()

# %% [markdown]
# ## 6. Risk Analytics (VaR / CVaR)

# %%
long_short_rets = strategy_results["long_short"].returns
print("\n=== Risk Analytics — Long/Short Strategy ===")
for conf in [0.95, 0.99]:
    var = np.percentile(long_short_rets, (1 - conf) * 100)
    cvar = long_short_rets[long_short_rets <= var].mean()
    print(f"  VaR {conf:.0%}:  {var:.4f} ({var*100:.2f}%)")
    print(f"  CVaR {conf:.0%}: {cvar:.4f} ({cvar*100:.2f}%)")

# %% [markdown]
# ## 7. Regime Performance Attribution

# %%
from aurora.regimes.detector import RegimeDetector

detector = RegimeDetector(method="hmm", n_regimes=4)
detector.fit(features, returns)
regimes = detector.predict(features, returns)

ls_rets = strategy_results["long_short"].returns
regime_col = regimes["regime"].reindex(ls_rets.index).ffill()

print("\n=== Regime-Attributed Performance ===")
regime_perf = {}
for regime_name in regime_col.unique():
    mask = regime_col == regime_name
    r = ls_rets[mask]
    if len(r) > 5:
        regime_perf[regime_name] = {
            "n_days": len(r),
            "ann_return": r.mean() * 252,
            "ann_vol": r.std() * np.sqrt(252),
            "sharpe": sharpe_ratio(r),
            "total_return": (1 + r).prod() - 1,
        }
        print(f"\n  {regime_name.upper()} ({len(r)} days):")
        print(f"    Ann. Return: {regime_perf[regime_name]['ann_return']:.1%}")
        print(f"    Ann. Vol:    {regime_perf[regime_name]['ann_vol']:.1%}")
        print(f"    Sharpe:      {regime_perf[regime_name]['sharpe']:.3f}")

print("\n✅ Backtesting analysis complete.")
