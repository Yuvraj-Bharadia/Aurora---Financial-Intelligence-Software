"""
Forecast visualization utilities.

Generates Plotly figures for:
- Forecast vs actual with confidence intervals
- Multi-horizon forecast fans
- Model comparison charts
- Attention weight heatmaps
- SHAP feature importance waterfalls
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_forecast_vs_actual(
    actual: pd.Series,
    forecast: pd.Series,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    regime: pd.Series | None = None,
    title: str = "Forecast vs Actual",
) -> go.Figure:
    """
    Overlay actual returns with forecast line and optional uncertainty band.

    Args:
        actual: Realised returns.
        forecast: Point forecast series.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
        regime: Regime series for background colouring.
        title: Chart title.
    """
    fig = go.Figure()

    # ── Confidence band ────────────────────────────────────────────────────
    if lower is not None and upper is not None:
        lo = lower.reindex(forecast.index)
        hi = upper.reindex(forecast.index)
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(hi.values) + list(lo.values[::-1]),
            fill="toself",
            fillcolor="rgba(52,152,219,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Interval",
            hoverinfo="skip",
        ))

    # ── Actual ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual.values,
        mode="lines", name="Actual",
        line=dict(color="#2c3e50", width=1.5),
    ))

    # ── Forecast ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        mode="lines", name="Forecast",
        line=dict(color="#3498db", width=2, dash="dot"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


def plot_model_comparison(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    title: str = "Model Forecast Comparison",
) -> go.Figure:
    """Compare multiple model forecasts against actual on one chart."""
    palette = px.colors.qualitative.Set2
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual.index, y=actual.values,
        mode="lines", name="Actual",
        line=dict(color="#2c3e50", width=2),
    ))

    for i, (model_name, fc) in enumerate(forecasts.items()):
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values,
            mode="lines",
            name=model_name,
            line=dict(color=palette[i % len(palette)], width=1.5, dash="dot"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def plot_rolling_metrics(
    returns_dict: dict[str, pd.Series],
    metric: str = "sharpe",
    window: int = 63,
    risk_free_rate: float = 0.04,
    title: str | None = None,
) -> go.Figure:
    """
    Rolling performance metric for multiple strategies.

    Args:
        returns_dict: Dict of strategy name → daily returns.
        metric: 'sharpe', 'sortino', or 'drawdown'.
        window: Rolling window in trading days.
    """
    from aurora.evaluation.metrics import sharpe_ratio, sortino_ratio, max_drawdown

    palette = px.colors.qualitative.Set2
    title = title or f"Rolling {metric.title()} ({window}d window)"
    fig = go.Figure()

    for i, (name, rets) in enumerate(returns_dict.items()):
        if metric == "sharpe":
            series = rets.rolling(window).apply(
                lambda r: sharpe_ratio(r, risk_free_rate), raw=False
            )
        elif metric == "sortino":
            series = rets.rolling(window).apply(
                lambda r: sortino_ratio(r, risk_free_rate), raw=False
            )
        else:  # drawdown
            cum = (1 + rets).cumprod()
            series = cum.rolling(window).apply(
                lambda x: max_drawdown(x), raw=True
            )

        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=1.8),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=metric.title(),
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    return fig


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 25,
    title: str = "Feature Importance",
) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    top = importance.nlargest(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index.tolist(),
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Importance"),
        ),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        template="plotly_white",
        height=max(400, top_n * 22),
        margin=dict(l=180),
    )
    return fig


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    log_scale: bool = False,
) -> go.Figure:
    """Cumulative return curves for multiple strategies."""
    palette = px.colors.qualitative.Set2
    fig = go.Figure()

    for i, (name, rets) in enumerate(returns_dict.items()):
        cum = (1 + rets.dropna()).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100,
            mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        yaxis_type="log" if log_scale else "linear",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


def plot_drawdown(returns_dict: dict[str, pd.Series], title: str = "Drawdown") -> go.Figure:
    """Drawdown chart for one or more strategies."""
    palette = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, (name, rets) in enumerate(returns_dict.items()):
        cum = (1 + rets.dropna()).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            mode="lines", name=name,
            fill="tozeroy",
            line=dict(color=palette[i % len(palette)], width=1.5),
            fillcolor=palette[i % len(palette)].replace(")", ",0.15)").replace("rgb", "rgba"),
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_white", height=400, hovermode="x unified",
    )
    return fig
