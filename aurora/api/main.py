"""
Aurora FastAPI application.

Start with:
    uvicorn aurora.api.main:app --reload --host 0.0.0.0 --port 8000

Or via the CLI:
    aurora-serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aurora import __version__
from aurora.api.routes.backtest import router as backtest_router
from aurora.api.routes.forecast import router as forecast_router
from aurora.api.routes.regime import router as regime_router
from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown lifecycle."""
    logger.info(f"Aurora API v{__version__} starting on {settings.api.host}:{settings.api.port}")
    yield
    logger.info("Aurora API shutting down")


app = FastAPI(
    title="Aurora Financial Intelligence API",
    description=(
        "Regime-adaptive forecasting, sentiment analysis, and "
        "portfolio backtesting for institutional quantitative research."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(forecast_router)
app.include_router(regime_router)
app.include_router(backtest_router)


@app.get("/", tags=["Health"])
async def root() -> JSONResponse:
    return JSONResponse({"aurora": "ok", "version": __version__})


@app.get("/health", tags=["Health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "healthy", "version": __version__})


def serve() -> None:
    """CLI entrypoint: aurora-serve."""
    uvicorn.run(
        "aurora.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    serve()
