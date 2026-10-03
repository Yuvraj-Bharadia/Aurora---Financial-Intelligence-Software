"""
Aurora Quantitative Research Dashboard.

A Streamlit application providing:
  - Live regime detection overlay
  - Multi-model forecast comparison
  - Sentiment tracking
  - Portfolio simulation
  - Risk analytics
  - Rolling performance metrics

Launch with:
    streamlit run aurora/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from aurora.configs.config import settings
from aurora.visualization.regime_plots import (
    plot_regime_overlay, plot_transition_matrix,
    plot_regime_duration_stats, plot_regime_return_distributions,
)
from aurora.visualization.forecast_plots import (
    plot_forecast_vs_actual, plot_model_comparison,
    plot_rolling_metrics, plot_cumulative_returns,
    plot_drawdown, plot_feature_importance,
)
from aurora.utils.logging import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Aurora — Financial Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/Aurora-v0.1.0-blueviolet?style=flat-square", width=180)
    st.title("⚙️ Controls")
    ticker = st.selectbox("Ticker", options=settings.data.equity_tickers, index=0)
    start_date = st.date_input("Start Date", value=pd.Timestamp("2018-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp("2023-12-31"))
    strategy = st.selectbox(
        "Backtest Strategy",
        ["long_only", "long_short", "market_neutral", "volatility_targeting"],
        index=1,
    )
    run_live = st.button("🚀 Run Analysis", type="primary")

    st.divider()
    st.caption("Aurora — Adaptive Unified Regime-Oriented Research Architecture")


# ── Helper: load cached data ───────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading market data…")
def load_data(ticker_: str, start_: str, end_: str) -> dict:
    """Fetch and cache market data + features + regimes."""
    try:
        from aurora.ingestion.pipeline import IngestionPipeline
        from aurora.features.pipeline import FeaturePipeline
        from aurora.regimes.detector import RegimeDetector

        ingestion = IngestionPipeline()
        bundle = ingestion.run(tickers=[ticker_], start=start_, end=end_)
        feat_pipeline = FeaturePipeline()
        features = feat_pipeline.run(bundle).get(ticker_, pd.DataFrame())
        ohlcv = bundle.ohlcv.get(ticker_, pd.DataFrame())
        if ohlcv.empty or features.empty:
            return {}

        close = ohlcv["close"].reindex(features.index)
        returns = np.log(close / close.shift(1)).dropna()
        features = features.reindex(returns.index)

        model_dir = settings.models.model_dir / ticker_
        if (model_dir / "regime_detector.pkl").exists():
            detector = RegimeDetector.load(model_dir / "regime_detector.pkl")
        else:
            detector = RegimeDetector(method="hmm")
            detector.fit(features, returns)

        regimes = detector.predict(features, returns)
        return {
            "close": close, "returns": returns,
            "features": features, "regimes": regimes,
            "detector": detector,
        }
    except Exception as exc:
        st.error(f"Data loading failed: {exc}")
        return {}


@st.cache_data(ttl=3600, show_spinner="Running backtest…")
def run_backtest(ticker_: str, start_: str, end_: str, strategy_: str) -> dict:
    try:
        from aurora.backtesting.engine import BacktestEngine
        data = load_data(ticker_, start_, end_)
        if not data:
            return {}
        engine = BacktestEngine()
        simple_signal = data["returns"].shift(1).fillna(0)
        result = engine.run(
            simple_signal, data["close"],
            strategy=strategy_,  # type: ignore
            regimes=data["regimes"],
        )
        return {"result": result, "returns": result.returns}
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        return {}


# ── Main layout ────────────────────────────────────────────────────────────────

st.title("🌌 Aurora — Adaptive Unified Regime-Oriented Research Architecture")

if not run_live:
    st.info("Configure the controls in the sidebar and click **Run Analysis** to begin.")
    st.stop()

data = load_data(ticker, str(start_date), str(end_date))
if not data:
    st.error("Failed to load data. Check your API keys and network connection.")
    st.stop()

close = data["close"]
returns = data["returns"]
regimes = data["regimes"]
features = data["features"]
detector = data["detector"]

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab_overview, tab_regimes, tab_forecast, tab_backtest, tab_risk, tab_features = st.tabs([
    "📊 Overview", "🎯 Regimes", "🔮 Forecasts", "📈 Backtest", "⚠️ Risk", "🔬 Features",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader(f"Market Overview — {ticker}")

    col1, col2, col3, col4 = st.columns(4)
    current_regime = str(regimes["regime"].iloc[-1])
    regime_conf = float(regimes.get("regime_confidence", pd.Series([1.0])).iloc[-1])
    total_return = float((1 + returns).prod() - 1)
    ann_vol = float(returns.std() * np.sqrt(252))

    col1.metric("Current Regime", current_regime.replace("_", " ").title())
    col2.metric("Regime Confidence", f"{regime_conf:.1%}")
    col3.metric("Total Return", f"{total_return:.1%}")
    col4.metric("Annualized Vol", f"{ann_vol:.1%}")

    st.plotly_chart(
        plot_regime_overlay(close, regimes, title=f"{ticker} — Regime Overlay"),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: REGIMES
# ─────────────────────────────────────────────────────────────────────────────
with tab_regimes:
    st.subheader("Regime Analysis")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Transition Matrix")
        trans = detector.transition_matrix()
        st.plotly_chart(plot_transition_matrix(trans), use_container_width=True)

    with col_r:
        st.subheader("Regime Duration Statistics")
        summary = detector.regime_summary(regimes)
        st.plotly_chart(plot_regime_duration_stats(summary), use_container_width=True)

    st.subheader("Return Distributions by Regime")
    st.plotly_chart(
        plot_regime_return_distributions(returns, regimes),
        use_container_width=True,
    )

    st.subheader("Regime Summary Table")
    st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: FORECASTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.subheader("Forecast Comparison")
    model_dir = settings.models.model_dir / ticker
    if not (model_dir / "ensemble.pkl").exists():
        st.warning(
            f"No trained model found for **{ticker}**. "
            "Run the training pipeline first:\n\n"
            "```\naurora-train --ticker " + ticker + "\n```"
        )
    else:
        try:
            from aurora.forecasting.forecaster import AuroraForecaster
            forecaster = AuroraForecaster.load(model_dir)
            fc_df = forecaster.forecast(features, returns)
            if not fc_df.empty:
                st.plotly_chart(
                    plot_forecast_vs_actual(
                        returns.tail(252),
                        fc_df["forecast"].tail(252),
                        fc_df.get("forecast_lower", pd.Series()).tail(252),
                        fc_df.get("forecast_upper", pd.Series()).tail(252),
                    ),
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Forecast error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader(f"Backtest — {strategy} strategy")
    bt_data = run_backtest(ticker, str(start_date), str(end_date), strategy)

    if bt_data:
        result = bt_data["result"]
        m = result.metrics

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sharpe",      f"{m.get('sharpe', 0):.3f}")
        c2.metric("Sortino",     f"{m.get('sortino', 0):.3f}")
        c3.metric("CAGR",        f"{m.get('cagr', 0):.1%}")
        c4.metric("Max Drawdown",f"{m.get('max_drawdown', 0):.1%}")
        c5.metric("Hit Ratio",   f"{m.get('hit_ratio', 0):.1%}")

        strat_rets = result.returns
        buy_hold_rets = returns.reindex(strat_rets.index)
        st.plotly_chart(
            plot_cumulative_returns(
                {"Aurora Strategy": strat_rets, "Buy & Hold": buy_hold_rets},
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_drawdown({"Aurora Strategy": strat_rets, "Buy & Hold": buy_hold_rets}),
            use_container_width=True,
        )

        if result.regime_returns:
            st.subheader("Regime Return Attribution")
            regime_metrics = {
                reg: {
                    "sharpe": float(
                        np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252)
                    ) if len(r) > 1 else 0.0,
                    "mean_return": float(np.mean(r) * 252),
                    "n_days": len(r),
                }
                for reg, r in result.regime_returns.items()
            }
            st.dataframe(
                pd.DataFrame(regime_metrics).T.round(3),
                use_container_width=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: RISK
# ─────────────────────────────────────────────────────────────────────────────
with tab_risk:
    st.subheader("Risk Analytics")

    # VaR / CVaR
    confidence_levels = [0.95, 0.99]
    var_table = {}
    for conf in confidence_levels:
        var = float(np.percentile(returns.dropna(), (1 - conf) * 100))
        cvar = float(returns[returns <= var].mean())
        var_table[f"VaR {conf:.0%}"] = var
        var_table[f"CVaR {conf:.0%}"] = cvar

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 95%",  f"{var_table['VaR 0.95']:.2%}")
    c2.metric("CVaR 95%", f"{var_table['CVaR 0.95']:.2%}")
    c3.metric("VaR 99%",  f"{var_table['VaR 0.99']:.2%}")
    c4.metric("CVaR 99%", f"{var_table['CVaR 0.99']:.2%}")

    st.plotly_chart(
        plot_rolling_metrics({"Strategy": returns}, metric="sharpe", window=63),
        use_container_width=True,
    )

    # Return histogram with VaR lines
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=returns.values, nbinsx=80,
        name="Daily Returns", marker_color="#3498db", opacity=0.7,
    ))
    hist_fig.add_vline(x=var_table["VaR 0.95"], line_dash="dash",
                       line_color="orange", annotation_text="VaR 95%")
    hist_fig.add_vline(x=var_table["VaR 0.99"], line_dash="dash",
                       line_color="red", annotation_text="VaR 99%")
    hist_fig.update_layout(
        title="Return Distribution", xaxis_title="Daily Return",
        template="plotly_white", height=400,
    )
    st.plotly_chart(hist_fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: FEATURES
# ─────────────────────────────────────────────────────────────────────────────
with tab_features:
    st.subheader("Feature Engineering")
    model_dir = settings.models.model_dir / ticker

    if (model_dir / "xgboost.pkl").exists():
        try:
            from aurora.models.ml.gradient_boosting import XGBoostForecaster
            xgb = XGBoostForecaster.load(model_dir / "xgboost.pkl")
            importance = xgb.feature_importance()
            st.plotly_chart(
                plot_feature_importance(importance, title=f"{ticker} — XGBoost Feature Importance"),
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"Feature importance requires a trained XGBoost model: {exc}")
    else:
        # Show raw feature correlations with returns as a proxy
        st.info("Train the model to see SHAP-based feature importance.")
        numeric_cols = features.select_dtypes(include=np.number).columns
        corrs = features[numeric_cols].corrwith(returns).dropna().abs().sort_values(ascending=False)
        st.plotly_chart(
            plot_feature_importance(corrs.head(30), title="Feature–Return |Correlation|"),
            use_container_width=True,
        )

    st.subheader("Feature Statistics")
    st.dataframe(
        features.describe().round(4),
        use_container_width=True,
        height=300,
    )
