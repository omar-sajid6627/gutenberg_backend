from __future__ import annotations

from typing import Any

from ..scraper import fetch_book_content
from .content_cache import get_cached_content, save_content_to_cache


def get_book_content(book_id: int) -> dict[str, Any] | None:
    """A book's paginated text, from the cache or fetched from gutenberg.org."""
    content = get_cached_content(book_id)
    if content:
        return content
    content = fetch_book_content(book_id)
    if content:
        save_content_to_cache(book_id, content)
    return content
