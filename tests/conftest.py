"""
Shared fixtures. Nothing in the suite reaches gutenberg.org: every test gets
a stand-in HTTP session that serves only the pages a test registers, and its
own empty data directory.
"""

from __future__ import annotations

import pytest

from app import scraper
from app.utils import content_cache, sentiment_analysis, text_chunking

SAMPLE_TEXT = (
    "The Project Gutenberg eBook of The Lighthouse Keeper\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK THE LIGHTHOUSE KEEPER ***\n\n"
    "CHAPTER I.\n\n"
    "Mara kept the lighthouse on the northern rock for eleven winters. "
    "Every evening she climbed the stairs and lit the great lamp. "
    "The sailors trusted the light, and the light never failed them. "
    "When the storm came in the twelfth winter, the lamp went dark. "
    "Mara carried a lantern to the top of the tower and held it until dawn. "
    "The fishing boats came home safely, guided by her small flame. "
    "In the morning the village gathered on the shore to thank her. "
    "She said only that the sea had been kind, and went to sleep. "
    "_Years later_ the children of the village still told the story of that night. "
    "The lighthouse stands there today, and its lamp is bright.\n\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK THE LIGHTHOUSE KEEPER ***\n\n"
    "Licence text that must not appear in the pages.\n"
)

SAMPLE_HTML = """
<html><body>
<table>
  <tr><th>Author</th><td><a rel="marcrel:aut" href="/ebooks/author/1">Wren, Elsie</a></td></tr>
  <tr><th>Title</th><td itemprop="headline">The Lighthouse Keeper</td></tr>
  <tr><th>Note</th><td>Reading ease score: 82.4 (6th grade). Easy to read.</td></tr>
  <tr property="dcterms:language"><th>Language</th><td>English</td></tr>
  <tr><th>Subject</th><td>Lighthouses -- Fiction</td></tr>
  <tr><th>Subject</th><td>Storms -- Fiction</td></tr>
  <tr property="dcterms:issued"><th>Release Date</th><td>Jan 1, 2001</td></tr>
  <tr><th>Downloads</th><td itemprop="interactionCount">1234 downloads in the last 30 days.</td></tr>
</table>
<img class="cover-art" src="/cache/epub/99999/pg99999.cover.medium.jpg">
<div class="summary-text-container">A keeper holds a lantern through a storm.</div>
</body></html>
"""


class FakeResponse:
    def __init__(self, status_code: int, body: bytes, headers: dict | None = None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.encoding = "utf-8"

    def iter_content(self, size: int):
        for i in range(0, len(self.body), size):
            yield self.body[i : i + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeGutenberg:
    """Serves registered URLs; everything else is a 404."""

    def __init__(self):
        self.routes: dict[str, FakeResponse] = {}
        self.requests: list[tuple[str, object]] = []

    def serve(self, url: str, body: str | bytes, status: int = 200, headers: dict | None = None):
        data = body.encode() if isinstance(body, str) else body
        self.routes[url] = FakeResponse(status, data, headers)

    def serve_book(self, book_id: int, text: str = SAMPLE_TEXT, html: str = SAMPLE_HTML):
        self.serve(f"{scraper.GUTENBERG}/cache/epub/{book_id}/pg{book_id}.txt", text)
        self.serve(f"{scraper.GUTENBERG}/ebooks/{book_id}", html)

    def get(self, url, stream=False, timeout=None):
        self.requests.append((url, timeout))
        return self.routes.get(url, FakeResponse(404, b""))


@pytest.fixture(autouse=True)
def gutenberg(monkeypatch) -> FakeGutenberg:
    fake = FakeGutenberg()
    monkeypatch.setattr(scraper, "SESSION", fake)
    return fake


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(content_cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sentiment_analysis, "SENTIMENT_DIR", tmp_path / "sentiment")
    monkeypatch.setattr(text_chunking, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(content_cache, "_memory_cache", {})
    monkeypatch.setattr(sentiment_analysis, "_sentiment_cache", {})
    return tmp_path
