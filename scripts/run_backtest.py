"""
CLI script: run a full backtest for a trained Aurora model.

Usage:
    python scripts/run_backtest.py --ticker SPY --strategy long_short
    aurora-backtest --ticker SPY --start 2018-01-01 --end 2023-12-31
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from aurora.backtesting.engine import BacktestEngine
from aurora.configs.config import settings
from aurora.features.pipeline import FeaturePipeline
from aurora.forecasting.forecaster import AuroraForecaster
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer(help="Aurora backtesting CLI")


@app.command()
def main(
    ticker: str = typer.Option("SPY", help="Ticker symbol"),
    start: str = typer.Option("2018-01-01", help="Backtest start date"),
    end: str = typer.Option("2023-12-31", help="Backtest end date"),
    strategy: str = typer.Option("long_short", help="Trading strategy"),
    model_dir: str | None = typer.Option(None, help="Model directory"),
    commission_bps: float = typer.Option(5.0),
    slippage_bps: float = typer.Option(2.0),
    initial_capital: float = typer.Option(1_000_000.0),
    output_csv: str | None = typer.Option(None, help="Save returns to CSV"),
) -> None:
    """Run a backtest using a trained Aurora model."""
    model_path = Path(model_dir) if model_dir else (settings.models.model_dir / ticker)

    if not (model_path / "ensemble.pkl").exists():
        typer.echo(f"❌ No trained model at {model_path}. Run aurora-train first.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading model from {model_path}…")
    forecaster = AuroraForecaster.load(model_path)

    typer.echo("Ingesting data…")
    import numpy as np
    ingestion = IngestionPipeline()
    bundle = ingestion.run(tickers=[ticker], start=start, end=end)

    typer.echo("Building features…")
    feat_pipeline = FeaturePipeline()
    features = feat_pipeline.run(bundle).get(ticker)
    if features is None or features.empty:
        typer.echo("❌ Feature generation failed.", err=True)
        raise typer.Exit(1)

    ohlcv = bundle.ohlcv[ticker]
    close = ohlcv["close"].reindex(features.index)
    returns = np.log(close / close.shift(1)).dropna()
    features = features.reindex(returns.index)

    typer.echo("Generating forecasts…")
    forecast_df = forecaster.forecast(features, returns)
    regimes = forecaster.regime_detector.predict(features, returns)

    typer.echo("Running backtest…")
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    result = engine.run(
        forecast_df["forecast"].dropna(),
        close.reindex(forecast_df.index),
        strategy=strategy,  # type: ignore
        regimes=regimes,
    )

    typer.echo("\n" + "=" * 50)
    typer.echo(f"  AURORA BACKTEST RESULTS — {ticker}  ")
    typer.echo("=" * 50)
    m = result.metrics
    typer.echo(f"  Sharpe Ratio:    {m.get('sharpe', 0):.4f}")
    typer.echo(f"  Sortino Ratio:   {m.get('sortino', 0):.4f}")
    typer.echo(f"  CAGR:            {m.get('cagr', 0):.2%}")
    typer.echo(f"  Max Drawdown:    {m.get('max_drawdown', 0):.2%}")
    typer.echo(f"  Hit Ratio:       {m.get('hit_ratio', 0):.2%}")
    typer.echo(f"  Ann. Volatility: {m.get('ann_volatility', 0):.2%}")
    typer.echo(f"  Calmar Ratio:    {m.get('calmar', 0):.4f}")
    typer.echo(f"  Turnover (ann.): {m.get('turnover', 0):.2f}x")
    typer.echo(f"  Total Costs:     {m.get('total_costs_bps', 0):.1f} bps")
    typer.echo("=" * 50)

    if output_csv:
        result.returns.to_csv(output_csv)
        typer.echo(f"\n📁 Returns saved to {output_csv}")


if __name__ == "__main__":
    app()
