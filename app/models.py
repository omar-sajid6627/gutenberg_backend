from pydantic import BaseModel
from typing import List, Optional

class Book(BaseModel):
    id: str
    title: str
    author: str
    language: str
    category: str
    ebook_no: str
    release_date: str
    summary: str
    cover_url: Optional[str] = None
    downloads: int = 0
    reading_ease_score: Optional[float] = None
    content: Optional[str] = None

class BookList(BaseModel):
    books: List[Book]
    total: int
    page: int
    per_page: int 