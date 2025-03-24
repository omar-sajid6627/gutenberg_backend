"""Settings read from the environment, and where cached files live."""

from __future__ import annotations

import os
from pathlib import Path

# Relative to the project, not the working directory. Serverless hosts only
# allow writes under /tmp, so DATA_DIR is overridable (vercel.json sets it).
DATA_DIR = Path(os.getenv("DATA_DIR") or Path(__file__).resolve().parent.parent / "data")


def book_file(directory: Path, book_id: int, suffix: str) -> Path:
    """
    The cache file for a book. Only positive integers are accepted, so a book
    ID can never carry a path separator or `..` into a filename.
    """
    if type(book_id) is not int or book_id <= 0:
        raise ValueError(f"book_id must be a positive int, got {book_id!r}")
    return directory / f"book_{book_id}_{suffix}"
