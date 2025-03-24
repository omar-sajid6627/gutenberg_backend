"""
Splitting a book into overlapping passages, and caching them per book.

The cache used to be a pickle of passages plus embedding vectors. Unpickling
runs whatever code the file contains, so anything able to write into the cache
directory could execute code inside the API. The vectors were never read
either: question answering only ever used the passage text, and the embedding
model's package (fastembed) was not in the requirements, so building a new
cache failed. Passages are now stored as plain JSON, and old `.pkl` files are
ignored.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.config import DATA_DIR, book_file
from app.utils.local_llm import content_words
from app.utils.sentences import split_sentences

logger = logging.getLogger(__name__)

CHUNKS_DIR = DATA_DIR / "chunks"
MAX_CHUNKS = 50
# Beyond this, chunking a whole novel only to sample 50 passages is waste.
MAX_TEXT_CHARS = 450_000


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 100) -> list[str]:
    """
    Split text into passages of at most `chunk_size` words, on sentence
    boundaries, each sharing up to `overlap` words with the previous one.
    At most MAX_CHUNKS passages are kept, spread evenly over the text.
    """
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in split_sentences(text):
        sentence_length = len(sentence.split())
        if current and current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current))
            kept: list[str] = []
            kept_length = 0
            for previous in reversed(current):
                words = len(previous.split())
                if kept_length + words > overlap:
                    break
                kept.insert(0, previous)
                kept_length += words
            current, current_length = kept, kept_length

        current.append(sentence)
        current_length += sentence_length

    if current:
        chunks.append(" ".join(current))

    if len(chunks) > MAX_CHUNKS:
        step = len(chunks) / MAX_CHUNKS
        chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS - 1)] + [chunks[-1]]
    return chunks


def get_chunk_info(chunks: list[str]) -> dict:
    sizes = [len(chunk.split()) for chunk in chunks]
    return {
        "total_chunks": len(chunks),
        "total_words": sum(sizes),
        "average_chunk_size": sum(sizes) / len(sizes) if sizes else 0,
        "chunk_sizes": sizes,
    }


def _chunks_file(book_id: int) -> Path:
    return book_file(CHUNKS_DIR, book_id, "chunks.json")


def save_chunks(book_id: int, chunks: list[str]) -> Path:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    path = _chunks_file(book_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"book_id": book_id, "chunks": chunks}))
    os.replace(tmp, path)
    return path


def load_chunks(book_id: int) -> list[str] | None:
    """Cached passages for a book, or None if there are none or the file is malformed."""
    path = _chunks_file(book_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable chunk cache %s: %s", path, exc)
        return None
    chunks = data.get("chunks") if isinstance(data, dict) else None
    if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        logger.warning("Ignoring malformed chunk cache %s", path)
        return None
    return chunks


def get_book_chunks(book_id: int, text: str) -> list[str]:
    """Passages for a book: from the cache, or chunked from `text` and cached."""
    cached = load_chunks(book_id)
    if cached is not None:
        return cached
    if len(text) > MAX_TEXT_CHARS:
        third = MAX_TEXT_CHARS // 3
        middle = (len(text) - third) // 2
        text = "\n\n".join((text[:third], text[middle : middle + third], text[-third:]))
    chunks = chunk_text(text)
    save_chunks(book_id, chunks)
    return chunks


def select_passages(chunks: list[str], query: str, limit: int = 3) -> list[str]:
    """
    The passages sharing the most words with the question, in book order.

    Context used to be the opening, middle and closing passages whatever was
    asked, so a question about chapter twenty was answered from chapter one.
    If nothing overlaps, those three are still the fallback.
    """
    if len(chunks) <= limit:
        return list(chunks)
    query_words = content_words(query)
    scores = [len(query_words & content_words(chunk)) for chunk in chunks]
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    picked = [i for i in ranked[:limit] if scores[i] > 0]
    if not picked:
        picked = [0, len(chunks) // 2, len(chunks) - 1]
    return [chunks[i] for i in sorted(set(picked))]
