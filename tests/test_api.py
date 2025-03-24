"""HTTP-level tests. Gutenberg is the stand-in from conftest; no network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from app.config import book_file
from app.utils.local_llm import LocalLLMHandler


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "llm_handler", LocalLLMHandler())
    with TestClient(main.app) as test_client:
        yield test_client


class TestBookIds:
    @pytest.mark.parametrize("bad", ["abc", "0", "-1", "1.5", "%2E%2E%2Fetc", "99999999999"])
    def test_non_positive_integer_ids_are_rejected(self, client, gutenberg, bad):
        for path in (f"/books/{bad}", f"/books/{bad}/content", f"/books/{bad}/readability"):
            assert client.get(path).status_code in (404, 422), path
        assert gutenberg.requests == []

    def test_cache_filenames_only_accept_positive_ints(self, tmp_path):
        assert book_file(tmp_path, 12, "content.json").name == "book_12_content.json"
        for bad in ("12", "../12", 0, -3, True, 1.0):
            with pytest.raises(ValueError):
                book_file(tmp_path, bad, "content.json")

    def test_delete_cache_rejects_traversal(self, client):
        assert client.delete("/cache/..%2F..%2Fmain").status_code in (404, 422)


def test_book_metadata(client, gutenberg):
    gutenberg.serve_book(99999)
    body = client.get("/books/99999").json()
    assert body["title"] == "The Lighthouse Keeper"
    assert body["id"] == "99999"


def test_unknown_book_is_404(client):
    assert client.get("/books/123").status_code == 404


def test_content_is_cached_after_the_first_fetch(client, gutenberg):
    gutenberg.serve_book(99999)
    first = client.get("/books/99999/content").json()
    fetches = len(gutenberg.requests)
    assert client.get("/books/99999/content").json() == first
    assert len(gutenberg.requests) == fetches
