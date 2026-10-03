# ── Aurora Production Dockerfile ──────────────────────────────────────────────
# Multi-stage build: builder installs heavy ML dependencies once;
# runtime image is lean and reproducible.

# ── Stage 1: base Python image ────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps required by scientific Python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        libgomp1 \
        libffi-dev \
        libssl-dev \
        libblas-dev \
        liblapack-dev \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*


# ── Stage 2: dependency builder ───────────────────────────────────────────────
FROM base AS builder

COPY pyproject.toml .

# Install all Python dependencies into a prefix we can copy
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install \
        "torch>=2.2.0" --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefix=/install ".[dev]" 2>/dev/null || \
    pip install --prefix=/install \
        yfinance fredapi requests aiohttp httpx \
        pandas numpy polars pyarrow \
        statsmodels arch hmmlearn \
        scikit-learn xgboost lightgbm catboost \
        transformers sentence-transformers \
        optuna mlflow \
        fastapi uvicorn pydantic pydantic-settings \
        plotly matplotlib seaborn streamlit \
        loguru rich typer python-dotenv joblib diskcache tqdm tenacity pyyaml \
        pytest pytest-asyncio pytest-cov


# ── Stage 3: runtime ──────────────────────────────────────────────────────────
FROM base AS runtime

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

RUN pip install --no-deps -e . 2>/dev/null || true

# Create output directories
RUN mkdir -p outputs/{data,cache,models,results,figures}

# Non-root user for security
RUN useradd --create-home --shell /bin/bash aurora
RUN chown -R aurora:aurora /app
USER aurora

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "aurora.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
