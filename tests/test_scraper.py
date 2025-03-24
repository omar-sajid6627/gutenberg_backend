"""Scraping gutenberg.org, against the stand-in session from conftest."""

from __future__ import annotations

import requests

from app import scraper
from tests.conftest import SAMPLE_TEXT


def test_metadata_is_parsed(gutenberg):
    gutenberg.serve_book(99999)
    book = scraper.fetch_book(99999)
    assert book.title == "The Lighthouse Keeper"
    assert book.author == "Wren, Elsie"
    assert book.language == "English"
    assert book.category == "Lighthouses -- Fiction, Storms -- Fiction"
    assert book.downloads == 1234
    assert book.reading_ease_score == 82.4
    assert book.cover_url == "https://www.gutenberg.org/cache/epub/99999/pg99999.cover.medium.jpg"


def test_missing_book_is_none():
    assert scraper.fetch_book(1) is None
    assert scraper.fetch_book_content(1) is None


def test_content_drops_the_licence_and_paginates(gutenberg):
    gutenberg.serve_book(99999)
    content = scraper.fetch_book_content(99999)
    text = " ".join(content["pages"])
    assert text.startswith("CHAPTER I.")
    assert "Licence text" not in text and "START OF" not in text
    assert "_" not in text
    assert content["total_pages"] == len(content["pages"]) == 1


def test_pages_hold_a_fixed_number_of_words():
    pages = scraper.paginate("word " * 1201, words_per_page=500)
    assert [len(p.split()) for p in pages] == [500, 500, 201]


def test_falls_back_to_the_older_files_url(gutenberg):
    gutenberg.serve(f"{scraper.GUTENBERG}/files/77/77-0.txt", SAMPLE_TEXT)
    assert scraper.fetch_book_content(77)["total_pages"] == 1


def test_every_request_has_a_timeout(gutenberg):
    gutenberg.serve_book(99999)
    scraper.fetch_book(99999)
    scraper.fetch_book_content(99999)
    assert gutenberg.requests and all(timeout for _, timeout in gutenberg.requests)


def test_oversized_download_is_refused(gutenberg, monkeypatch):
    monkeypatch.setattr(scraper, "MAX_TEXT_BYTES", 1000)
    gutenberg.serve_book(5, text="word " * 1000)
    assert scraper.fetch_book_content(5) is None


def test_declared_oversize_is_refused_before_reading(gutenberg):
    gutenberg.serve(f"{scraper.GUTENBERG}/ebooks/6", "<html></html>", headers={"Content-Length": str(10**9)})
    assert scraper.fetch_book(6) is None


def test_network_errors_become_none(monkeypatch):
    class Broken:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("no route")

    monkeypatch.setattr(scraper, "SESSION", Broken())
    assert scraper.fetch_book(3) is None
