"""Fetching book metadata and text from gutenberg.org."""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

from .models import Book

logger = logging.getLogger(__name__)

GUTENBERG = "https://www.gutenberg.org"
# (connect, read) seconds. Without a timeout a stalled connection held a
# worker thread forever.
TIMEOUT = (5, 30)
# The longest Gutenberg texts are a few MB; anything far beyond that is not a
# book, and reading it into memory unbounded is a denial-of-service lever.
MAX_TEXT_BYTES = 20 * 1024 * 1024
MAX_HTML_BYTES = 2 * 1024 * 1024
WORDS_PER_PAGE = 500

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "gutenberg-backend (+https://www.gutenberg.org/policy/robot_access.html)"

START_MARKER = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*")
END_MARKER = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG")


class TooLarge(Exception):
    pass


def _get_text(url: str, max_bytes: int) -> str | None:
    """GET `url` as text, or None on any HTTP or network failure."""
    try:
        with SESSION.get(url, stream=True, timeout=TIMEOUT) as response:
            if response.status_code != 200:
                return None
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > max_bytes:
                raise TooLarge(url)
            body = bytearray()
            for block in response.iter_content(64 * 1024):
                body.extend(block)
                if len(body) > max_bytes:
                    raise TooLarge(url)
            return body.decode(response.encoding or "utf-8", errors="replace")
    except TooLarge:
        logger.warning("Refusing %s: larger than %d bytes", url, max_bytes)
    except requests.RequestException as exc:
        logger.warning("Fetching %s failed: %s", url, exc)
    return None


def text_urls(book_id: int) -> list[str]:
    return [
        f"{GUTENBERG}/cache/epub/{book_id}/pg{book_id}.txt",
        f"{GUTENBERG}/files/{book_id}/{book_id}-0.txt",
    ]


def strip_boilerplate(raw: str) -> str:
    """Drop the Project Gutenberg licence header and footer, if present."""
    start, end = START_MARKER.search(raw), END_MARKER.search(raw)
    if start and end and start.end() < end.start():
        return raw[start.end() : end.start()]
    return raw


def paginate(text: str, words_per_page: int = WORDS_PER_PAGE) -> list[str]:
    # Underscores mark italics in Gutenberg plain text.
    words = text.replace("_", "").split()
    return [" ".join(words[i : i + words_per_page]) for i in range(0, len(words), words_per_page)]


def fetch_book_content(book_id: int) -> dict | None:
    """The book's text split into pages, or None if it could not be fetched."""
    for url in text_urls(book_id):
        raw = _get_text(url, MAX_TEXT_BYTES)
        if raw:
            pages = paginate(strip_boilerplate(raw))
            if pages:
                return {"pages": pages, "total_pages": len(pages)}
    return None


def _row_value(soup: BeautifulSoup, prop: str) -> str | None:
    row = soup.find("tr", property=prop)
    cell = row.find("td") if row else None
    return cell.get_text(strip=True) if cell else None


def _table_values(soup: BeautifulSoup, heading: str) -> list[str]:
    values = []
    for th in soup.find_all("th", string=heading):
        td = th.find_next_sibling("td")
        if td:
            values.append(td.get_text(strip=True))
    return values


def scrape_gutenberg_book(html: str) -> dict:
    """Parse the fields we use from a gutenberg.org /ebooks/<id> page."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("td", itemprop="headline")
    author = soup.find("a", rel="marcrel:aut")

    cover_url = None
    cover = soup.find("img", class_="cover-art")
    if cover and cover.get("src"):
        src = cover["src"]
        cover_url = src if src.startswith("http") else GUTENBERG + src

    summary = soup.find(class_="summary-text-container")

    downloads = 0
    downloads_cell = soup.find("td", itemprop="interactionCount")
    if downloads_cell:
        match = re.search(r"\d+", downloads_cell.get_text())
        downloads = int(match.group()) if match else 0

    notes = _table_values(soup, "Note")
    reading_ease_score = None
    for note in notes:
        match = re.search(r"Reading ease score:\s*([\d.]+)", note)
        if match:
            reading_ease_score = float(match.group(1))
            break

    return {
        "title": title.get_text(strip=True) if title else "N/A",
        "author": author.get_text(strip=True) if author else "Author not found",
        "cover_url": cover_url,
        "summary": summary.get_text(strip=True) if summary else "N/A",
        "notes": notes,
        "subjects": _table_values(soup, "Subject"),
        "downloads": downloads,
        "language": _row_value(soup, "dcterms:language") or "N/A",
        "release_date": _row_value(soup, "dcterms:issued") or "N/A",
        "reading_ease_score": reading_ease_score,
    }


def fetch_book(book_id: int) -> Book | None:
    """Metadata for a book, or None if gutenberg.org has no page for it."""
    html = _get_text(f"{GUTENBERG}/ebooks/{book_id}", MAX_HTML_BYTES)
    if not html:
        return None
    metadata = scrape_gutenberg_book(html)
    return Book(
        id=str(book_id),
        title=metadata["title"],
        author=metadata["author"],
        language=metadata["language"],
        category=", ".join(metadata["subjects"]),
        ebook_no=str(book_id),
        release_date=metadata["release_date"],
        summary=metadata["summary"],
        cover_url=metadata["cover_url"],
        downloads=metadata["downloads"],
        reading_ease_score=metadata["reading_ease_score"],
    )
