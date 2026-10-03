"""
Disk-based caching utilities using diskcache.

Provides a decorator and a context-manager cache for expensive operations
such as data ingestion, feature computation, and model inference.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import diskcache

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class AuroraCache:
    """Thin wrapper around diskcache.Cache with TTL and namespace support."""

    def __init__(
        self,
        namespace: str = "default",
        cache_dir: Path | None = None,
        ttl: int | None = None,
        size_limit: int = 10 * 1024**3,  # 10 GB
    ) -> None:
        cache_dir = cache_dir or settings.data.cache_dir
        self._path = cache_dir / namespace
        self._path.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(self._path), size_limit=size_limit)
        self._default_ttl = ttl or settings.data.cache_ttl_hours * 3600

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expire = ttl if ttl is not None else self._default_ttl
        self._cache.set(key, value, expire=expire)

    def delete(self, key: str) -> None:
        self._cache.delete(key)

    def clear(self) -> None:
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    @staticmethod
    def make_key(*args: Any, **kwargs: Any) -> str:
        """Deterministic cache key from arbitrary arguments."""
        payload = json.dumps({"args": list(args), "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


# Module-level shared caches
_caches: dict[str, AuroraCache] = {}


def get_cache(namespace: str = "default") -> AuroraCache:
    """Retrieve (or create) a named cache."""
    if namespace not in _caches:
        _caches[namespace] = AuroraCache(namespace=namespace)
    return _caches[namespace]


def cached(
    namespace: str = "default",
    ttl: int | None = None,
    ignore_args: list[str] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to cache function results on disk.

    Args:
        namespace: Cache partition name.
        ttl: Time-to-live in seconds; defaults to config value.
        ignore_args: Parameter names to exclude from the cache key.
    """
    ignore = set(ignore_args or [])

    def decorator(fn: F) -> F:
        cache = get_cache(namespace)
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            key_data = {k: v for k, v in bound.arguments.items() if k not in ignore}
            key = f"{fn.__qualname__}:{AuroraCache.make_key(**key_data)}"

            if key in cache:
                logger.debug(f"Cache hit: {fn.__qualname__}")
                return cache.get(key)

            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            cache.set(key, result, ttl=ttl)
            logger.debug(f"Cache miss, computed in {elapsed:.2f}s: {fn.__qualname__}")
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
