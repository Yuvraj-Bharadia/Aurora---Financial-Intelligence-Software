"""
Regime visualization utilities.

Generates Plotly figures showing:
- Latent regime states over price history
- Regime transition heatmaps
- Regime duration statistics
- Transition probability matrices
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Canonical colour palette for up to 6 regimes
REGIME_COLORS: dict[str, str] = {
    "bull":            "#2ecc71",   # green
    "bear":            "#e74c3c",   # red
    "high_volatility": "#f39c12",   # amber
    "sideways":        "#3498db",   # blue
    "crisis":          "#8e44ad",   # purple
    "recovery":        "#1abc9c",   # teal
}
_DEFAULT_COLOR = "#95a5a6"


def plot_regime_overlay(
    prices: pd.Series,
    regimes: pd.DataFrame,
    title: str = "Market Regimes — Price with Regime Overlay",
) -> go.Figure:
    """
    Price chart with colour-coded regime background bands.

    Args:
        prices: Close price series (DatetimeIndex).
        regimes: DataFrame with 'regime' column (output of RegimeDetector).
        title: Figure title.

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # ── Price line ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=prices.index, y=prices.values,
            mode="lines", name="Price",
            line=dict(color="#2c3e50", width=1.5),
        ),
        row=1, col=1,
    )

    # ── Regime bands ───────────────────────────────────────────────────────
    aligned = regimes["regime"].reindex(prices.index).ffill()
    regime_changes = aligned[aligned != aligned.shift()].dropna()
    starts = list(regime_changes.index)
    ends = list(regime_changes.index[1:]) + [prices.index[-1]]

    for start, end, reg in zip(starts, ends, regime_changes.values):
        color = REGIME_COLORS.get(str(reg), _DEFAULT_COLOR)
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, opacity=0.12,
            layer="below", line_width=0,
            row=1, col=1,
        )

    # ── Regime confidence bar ──────────────────────────────────────────────
    if "regime_confidence" in regimes.columns:
        conf = regimes["regime_confidence"].reindex(prices.index)
        colors = [
            REGIME_COLORS.get(str(r), _DEFAULT_COLOR)
            for r in aligned
        ]
        fig.add_trace(
            go.Bar(
                x=conf.index, y=conf.values,
                name="Regime Confidence",
                marker_color=colors,
                showlegend=False,
            ),
            row=2, col=1,
        )

    # ── Legend entries for each regime ────────────────────────────────────
    for regime_name, color in REGIME_COLORS.items():
        if regime_name in aligned.values:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol="square"),
                    name=regime_name.replace("_", " ").title(),
                    showlegend=True,
                ),
                row=1, col=1,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis2_title="Date",
        yaxis_title="Price",
        yaxis2_title="Confidence",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", y=-0.12),
        hovermode="x unified",
    )
    return fig


def plot_transition_matrix(
    transition_matrix: pd.DataFrame,
    title: str = "Regime Transition Probability Matrix",
) -> go.Figure:
    """Heatmap of the HMM regime transition probability matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=transition_matrix.values,
            x=list(transition_matrix.columns),
            y=list(transition_matrix.index),
            colorscale="Blues",
            text=np.round(transition_matrix.values, 3),
            texttemplate="%{text}",
            colorbar=dict(title="Probability"),
            zmin=0, zmax=1,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_regime_duration_stats(
    regime_summary: pd.DataFrame,
    title: str = "Regime Duration Statistics",
) -> go.Figure:
    """Bar charts showing mean duration and time allocation per regime."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Mean Duration (days)", "% of Total Time"],
    )
    regimes = regime_summary.index.tolist()
    colors = [REGIME_COLORS.get(r, _DEFAULT_COLOR) for r in regimes]

    fig.add_trace(
        go.Bar(
            x=regimes,
            y=regime_summary["mean_duration"].values,
            marker_color=colors,
            name="Mean Duration",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=regimes,
            y=(regime_summary["pct_time"].values * 100),
            marker_color=colors,
            name="% Time",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        yaxis2_title="% of Time",
        yaxis_title="Days",
    )
    return fig


def plot_regime_return_distributions(
    returns: pd.Series,
    regimes: pd.DataFrame,
    title: str = "Return Distributions by Regime",
) -> go.Figure:
    """Violin plots of return distributions per regime."""
    aligned = regimes["regime"].reindex(returns.index).ffill().dropna()
    fig = go.Figure()
    for reg in aligned.unique():
        color = REGIME_COLORS.get(str(reg), _DEFAULT_COLOR)
        r = returns[aligned == reg].dropna()
        fig.add_trace(
            go.Violin(
                y=r.values,
                name=str(reg).replace("_", " ").title(),
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                line_color=color,
                opacity=0.7,
            )
        )
    fig.update_layout(
        title=title,
        yaxis_title="Daily Return",
        template="plotly_white",
        height=500,
        showlegend=True,
    )
    return fig
