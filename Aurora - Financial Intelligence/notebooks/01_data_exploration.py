"""
Notebook 01 — Data Exploration
================================
Run as a Jupyter notebook or plain Python script.

Explores:
- OHLCV data quality and statistics
- Return distributions and stylised facts
- Macro indicator visualisation
- Correlation structure
"""

# %% [markdown]
# # Aurora — Notebook 01: Data Exploration
# This notebook demonstrates the Aurora data ingestion pipeline and performs
# exploratory data analysis on the loaded market data.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aurora.ingestion.pipeline import IngestionPipeline
from aurora.configs.config import settings

# %% [markdown]
# ## 1. Data Ingestion

# %%
print("Ingesting market data (using cache if available)…")
pipeline = IngestionPipeline()
bundle = pipeline.run(
    tickers=["SPY", "QQQ", "IWM"],
    start="2015-01-01",
    end="2023-12-31",
)

spy_ohlcv = bundle.ohlcv["SPY"]
print(f"SPY OHLCV shape: {spy_ohlcv.shape}")
print(spy_ohlcv.describe().round(2))

# %% [markdown]
# ## 2. Return Statistics & Stylised Facts

# %%
close = spy_ohlcv["close"]
log_ret = np.log(close / close.shift(1)).dropna()

print("\n=== SPY Daily Log Return Statistics ===")
stats = pd.Series({
    "Mean (ann.)":   log_ret.mean() * 252,
    "Std (ann.)":    log_ret.std() * np.sqrt(252),
    "Skewness":      log_ret.skew(),
    "Kurtosis":      log_ret.kurt(),
    "Min":           log_ret.min(),
    "Max":           log_ret.max(),
    "5th pctile":    log_ret.quantile(0.05),
    "95th pctile":   log_ret.quantile(0.95),
}).round(4)
print(stats.to_string())

# %%
# Return distribution
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=log_ret.values, nbinsx=100,
    name="Daily Returns", marker_color="#3498db", opacity=0.7,
))

# Overlay normal distribution
from scipy.stats import norm
x_norm = np.linspace(log_ret.min(), log_ret.max(), 300)
y_norm = norm.pdf(x_norm, log_ret.mean(), log_ret.std()) * len(log_ret) * (log_ret.max() - log_ret.min()) / 100
fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines",
                         line=dict(color="red", width=2), name="Normal Fit"))
fig.update_layout(title="SPY Daily Return Distribution", template="plotly_white",
                  xaxis_title="Log Return", height=400)
fig.show()

# %% [markdown]
# ## 3. Autocorrelation (Volatility Clustering)

# %%
sq_ret = log_ret ** 2
ac_sq = [sq_ret.autocorr(lag=i) for i in range(1, 31)]

fig = go.Figure(go.Bar(x=list(range(1, 31)), y=ac_sq, marker_color="#e74c3c"))
fig.add_hline(y=1.96/np.sqrt(len(sq_ret)), line_dash="dash", line_color="gray")
fig.add_hline(y=-1.96/np.sqrt(len(sq_ret)), line_dash="dash", line_color="gray")
fig.update_layout(title="Autocorrelation of Squared Returns (Volatility Clustering)",
                  xaxis_title="Lag", yaxis_title="ACF", template="plotly_white", height=400)
fig.show()

# %% [markdown]
# ## 4. Multi-Asset Correlation Matrix

# %%
all_close = pd.concat(
    {t: bundle.ohlcv[t]["close"] for t in ["SPY", "QQQ", "IWM"] if t in bundle.ohlcv},
    axis=1
)
all_rets = np.log(all_close / all_close.shift(1)).dropna()
corr = all_rets.corr()

fig = go.Figure(go.Heatmap(
    z=corr.values, x=list(corr.columns), y=list(corr.index),
    colorscale="RdBu_r", zmin=-1, zmax=1,
    text=corr.round(3).values, texttemplate="%{text}",
))
fig.update_layout(title="Return Correlation Matrix", template="plotly_white", height=400)
fig.show()

# %% [markdown]
# ## 5. Macro Indicators

# %%
if not bundle.macro.empty:
    macro = bundle.macro.loc[:, bundle.macro.columns.str.contains("VIXCLS|GS10|GS2")]
    fig = make_subplots(rows=len(macro.columns), cols=1, shared_xaxes=True)
    for i, col in enumerate(macro.columns, 1):
        fig.add_trace(go.Scatter(x=macro.index, y=macro[col], mode="lines", name=col), row=i, col=1)
    fig.update_layout(title="Key Macro Indicators", template="plotly_white", height=600)
    fig.show()

print("\n✅ Data exploration complete.")
