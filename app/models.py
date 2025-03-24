from __future__ import annotations

from pydantic import BaseModel


class Book(BaseModel):
    id: str
    title: str
    author: str
    language: str
    category: str
    ebook_no: str
    release_date: str
    summary: str
    cover_url: str | None = None
    downloads: int = 0
    reading_ease_score: float | None = None
    content: str | None = None
