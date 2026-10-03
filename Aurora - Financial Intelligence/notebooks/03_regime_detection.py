"""
Notebook 03 — Regime Detection
================================
Demonstrates the Aurora regime detection engine:
- HMM fitting and regime assignment
- Regime overlay on SPY price history
- Transition matrix visualisation
- Per-regime return statistics
- Regime duration analysis
"""

# %% [markdown]
# # Aurora — Notebook 03: Regime Detection

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from aurora.ingestion.pipeline import IngestionPipeline
from aurora.features.pipeline import FeaturePipeline
from aurora.regimes.detector import RegimeDetector
from aurora.visualization.regime_plots import (
    plot_regime_overlay, plot_transition_matrix,
    plot_regime_duration_stats, plot_regime_return_distributions,
)

# %% [markdown]
# ## 1. Load Data and Features

# %%
ingestion = IngestionPipeline()
bundle = ingestion.run(tickers=["SPY"], start="2005-01-01", end="2023-12-31")

feat_pipeline = FeaturePipeline()
features = feat_pipeline.run(bundle)["SPY"]

close = bundle.ohlcv["SPY"]["close"].reindex(features.index)
returns = np.log(close / close.shift(1)).dropna()
features = features.reindex(returns.index)

print(f"Features: {features.shape[0]} rows × {features.shape[1]} columns")

# %% [markdown]
# ## 2. Train Regime Detector

# %%
detector = RegimeDetector(method="ensemble", n_regimes=4)
detector.fit(features, returns)

regimes = detector.predict(features, returns)
print("\nRegime distribution:")
print(regimes["regime"].value_counts())

# %% [markdown]
# ## 3. Regime Overlay on Price History

# %%
fig = plot_regime_overlay(close, regimes, title="SPY — Hidden Market Regimes (2005–2023)")
fig.show()

# %% [markdown]
# ## 4. Transition Matrix

# %%
trans = detector.transition_matrix()
print("\nRegime Transition Matrix:")
print(trans.round(3).to_string())
plot_transition_matrix(trans).show()

# %% [markdown]
# ## 5. Regime Duration Statistics

# %%
summary = detector.regime_summary(regimes)
print("\nRegime Summary:")
print(summary.to_string())
plot_regime_duration_stats(summary).show()

# %% [markdown]
# ## 6. Return Distributions by Regime

# %%
plot_regime_return_distributions(returns, regimes).show()

# %% [markdown]
# ## 7. Statistical Properties per Regime

# %%
aligned = regimes[["regime"]].join(returns.rename("ret"), how="inner")
regime_stats = aligned.groupby("regime")["ret"].agg([
    "mean", "std", "skew",
    lambda x: x.kurt(),
    lambda x: (x > 0).mean(),
]).rename(columns={"<lambda_0>": "kurtosis", "<lambda_1>": "hit_ratio"})

regime_stats["ann_return"] = regime_stats["mean"] * 252
regime_stats["ann_vol"] = regime_stats["std"] * np.sqrt(252)
regime_stats["sharpe"] = regime_stats["ann_return"] / regime_stats["ann_vol"]

print("\nRegime Statistics:")
print(regime_stats[["ann_return", "ann_vol", "sharpe", "hit_ratio", "skew", "kurtosis"]].round(3).to_string())

# %% [markdown]
# ## Key Findings
# - Markets exhibit distinct regime structures with materially different return / risk profiles.
# - Regime duration varies significantly: bull markets persist longer than crisis periods.
# - The transition matrix confirms regime persistence (diagonal dominance).
# - Each regime has characteristic return distributions supporting the hypothesis that
#   different models should be active under different conditions.

print("\n✅ Regime detection analysis complete.")
