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


def test_readability_is_measured_from_the_text(client, gutenberg):
    gutenberg.serve_book(99999, text="The cat sat on the mat. The dog ran to the park. " * 30)
    body = client.get("/books/99999/readability").json()
    assert body["book_id"] == 99999
    assert body["reading_ease_score"] > 70


class TestAsk:
    def test_answers_are_quoted_from_the_book(self, client, gutenberg):
        gutenberg.serve_book(99999)
        body = client.post("/books/99999/ask", json={"query": "What did Mara carry up the tower?"}).json()
        assert "Mara carried a lantern to the top of the tower" in body["response"]
        assert body["book_id"] == 99999

    def test_a_missing_book_is_404_not_500(self, client):
        response = client.post("/books/4242/ask", json={"query": "Who is the hero?"})
        assert response.status_code == 404

    def test_passages_are_cached_for_the_next_question(self, client, gutenberg, data_dir):
        gutenberg.serve_book(99999)
        client.post("/books/99999/ask", json={"query": "storm?"})
        assert (data_dir / "chunks" / "book_99999_chunks.json").exists()

    @pytest.mark.parametrize(
        "payload", [{"query": ""}, {"query": "x" * 1001}, {"query": "ok", "temperature": 5}]
    )
    def test_invalid_requests_are_rejected(self, client, payload):
        assert client.post("/books/1/ask", json=payload).status_code == 422


class TestSentiment:
    @pytest.fixture(autouse=True)
    def fake_nlp(self, monkeypatch):
        from app.utils import sentiment_analysis as sa

        nlp = sa.NLPResources(
            frozenset({"the"}),
            str,
            lambda s: {"pos": 0.5, "neg": 0.0, "neu": 0.5, "compound": 0.6},
        )
        monkeypatch.setattr(sa, "nlp_resources", lambda: nlp)
        monkeypatch.setattr(main, "nlp_resources", lambda: nlp)

    def test_first_call_processes_then_the_result_is_served(self, client, gutenberg):
        gutenberg.serve_book(99999)
        first = client.get("/books/99999/sentiment").json()
        assert first["status"] == "processing"
        # TestClient runs background tasks before returning, so it is done now.
        second = client.get("/books/99999/sentiment").json()
        assert second["status"] == "completed"
        assert second["overall_sentiment"] == "positive"

    def test_polling_during_analysis_does_not_start_another(self, client, gutenberg, monkeypatch):
        gutenberg.serve_book(99999)
        monkeypatch.setattr(main, "_sentiment_jobs", {99999})
        started = []
        monkeypatch.setattr(main, "_run_sentiment", lambda *a: started.append(a))
        assert client.get("/books/99999/sentiment").json()["status"] == "processing"
        assert started == []

    def test_missing_nltk_data_is_503(self, client, gutenberg, monkeypatch):
        from app.utils.sentiment_analysis import MissingNLTKData

        def missing():
            raise MissingNLTKData("NLTK data is missing")

        monkeypatch.setattr(main, "nlp_resources", missing)
        gutenberg.serve_book(99999)
        assert client.get("/books/99999/sentiment").status_code == 503


class FakeCommunicate:
    calls: list = []

    def __init__(self, text, voice, rate=None):
        FakeCommunicate.calls.append(text)

    async def stream(self):
        yield {"type": "WordBoundary"}
        yield {"type": "audio", "data": b"ID3fake"}
        yield {"type": "audio", "data": b"-mp3"}


class TestTTS:
    def test_audio_is_returned_without_temp_files(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main.edge_tts, "Communicate", FakeCommunicate)
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        response = client.post("/tts", json={"text": "Hello there.", "chunk_index": 2, "total_chunks": 5})
        assert response.status_code == 200
        assert response.content == b"ID3fake-mp3"
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.headers["x-chunk-index"] == "2"
        assert list(tmp_path.iterdir()) == []

    def test_oversized_text_is_rejected_before_synthesis(self, client, monkeypatch):
        FakeCommunicate.calls.clear()
        monkeypatch.setattr(main.edge_tts, "Communicate", FakeCommunicate)
        assert client.post("/tts", json={"text": "a" * (main.MAX_TTS_CHARS + 1)}).status_code == 422
        assert FakeCommunicate.calls == []

    def test_service_failure_is_502_without_internals(self, client, monkeypatch):
        class Broken(FakeCommunicate):
            async def stream(self):
                raise RuntimeError("wss://internal-endpoint refused")
                yield

        monkeypatch.setattr(main.edge_tts, "Communicate", Broken)
        response = client.post("/tts", json={"text": "Hello."})
        assert response.status_code == 502
        assert "internal-endpoint" not in response.text

    def test_text_is_split_on_sentences(self, client):
        text = "First sentence here. " * 100
        body = client.post("/tts/chunks", json={"text": text}).json()
        assert body["total_chunks"] == len(body["chunks"]) > 1
        assert all(c.endswith(".") for c in body["chunks"])


def test_cors_allows_any_origin_without_credentials(client):
    response = client.get("/health", headers={"Origin": "https://reader.example"})
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_the_empty_search_stub_is_gone(client):
    assert client.get("/books/search/whale").status_code in (404, 422)


def test_clearing_one_books_cache(client, gutenberg, data_dir):
    gutenberg.serve_book(99999)
    client.get("/books/99999/content")
    assert (data_dir / "cache" / "book_99999_content.json").exists()
    assert client.delete("/cache/99999").status_code == 200
    assert not (data_dir / "cache" / "book_99999_content.json").exists()
