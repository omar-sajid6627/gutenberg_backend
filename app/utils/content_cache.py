"""Book text cached in memory and on disk for an hour."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, book_file

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache"
CACHE_EXPIRATION = 60 * 60

_memory_cache: dict[int, dict[str, Any]] = {}


def _cache_file(book_id: int) -> Path:
    return book_file(CACHE_DIR, book_id, "content.json")


def get_cached_content(book_id: int) -> dict[str, Any] | None:
    entry = _memory_cache.get(book_id)
    if entry is None:
        cache_file = _cache_file(book_id)
        if not cache_file.exists():
            return None
        try:
            entry = json.loads(cache_file.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable content cache %s: %s", cache_file, exc)
            return None
        _memory_cache[book_id] = entry

    if time.time() - entry.get("timestamp", 0) >= CACHE_EXPIRATION:
        _memory_cache.pop(book_id, None)
        return None
    return entry["content"]


def save_content_to_cache(book_id: int, content: dict[str, Any]) -> None:
    entry = {"content": content, "timestamp": time.time()}
    _memory_cache[book_id] = entry
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_file(book_id).write_text(json.dumps(entry))
    except OSError as exc:
        logger.warning("Could not write content cache for book %s: %s", book_id, exc)


def clear_cache(book_id: int | None = None) -> None:
    """Forget one book's text, or every book's."""
    if book_id is not None:
        _memory_cache.pop(book_id, None)
        _cache_file(book_id).unlink(missing_ok=True)
        return
    _memory_cache.clear()
    for cache_file in CACHE_DIR.glob("book_*_content.json"):
        cache_file.unlink(missing_ok=True)
