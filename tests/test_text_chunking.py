"""
Passage chunking and the per-book passage cache.

The cache was a pickle; loading one runs arbitrary code. These pin that the
store is plain JSON, that a planted pickle is never opened, and that a
malformed file is ignored rather than trusted.
"""

from __future__ import annotations

import ast
import inspect
import json
import pickle
from pathlib import Path

import pytest

from app.utils import text_chunking as tc


@pytest.fixture
def isolated_store(data_dir):
    store = data_dir / "chunks"
    store.mkdir()
    return store


def sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} has six words." for i in range(n))


class TestStore:
    def test_round_trip(self):
        tc.save_chunks(42, ["first passage.", "second passage."])
        assert tc.load_chunks(42) == ["first passage.", "second passage."]

    def test_the_file_is_plain_json(self, isolated_store):
        path = tc.save_chunks(42, ["only passage."])
        assert json.loads(path.read_text())["chunks"] == ["only passage."]

    def test_missing_cache_is_none(self):
        assert tc.load_chunks(7) is None

    @pytest.mark.parametrize("payload", ["not json", '{"chunks": [1, 2]}', "[]", '{"chunks": "text"}'])
    def test_malformed_cache_is_ignored(self, isolated_store, payload):
        (isolated_store / "book_9_chunks.json").write_text(payload)
        assert tc.load_chunks(9) is None

    def test_a_planted_pickle_is_never_loaded(self, isolated_store, tmp_path_factory):
        sentinel = tmp_path_factory.mktemp("sentinel") / "executed"

        class Payload:
            def __reduce__(self):
                return (Path.touch, (sentinel,))

        for name in ("book_5_embeddings.pkl", "book_5_chunks.json"):
            (isolated_store / name).write_bytes(pickle.dumps(Payload()))

        assert tc.load_chunks(5) is None
        assert tc.get_book_chunks(5, sentences(3))
        assert not sentinel.exists()

    def test_the_module_does_not_import_pickle(self):
        tree = ast.parse(inspect.getsource(tc))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not imported & {"pickle", "dill", "joblib", "cloudpickle"}


class TestChunking:
    def test_empty_text(self):
        assert tc.chunk_text("") == []

    def test_chunks_respect_the_word_limit(self):
        chunks = tc.chunk_text(sentences(200), chunk_size=60, overlap=0)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 60 for c in chunks)

    def test_consecutive_chunks_overlap(self):
        chunks = tc.chunk_text(sentences(60), chunk_size=30, overlap=12)
        assert chunks[0].split(". ")[-1] in chunks[1]

    def test_long_books_are_capped_and_keep_the_ending(self):
        text = sentences(3000)
        chunks = tc.chunk_text(text, chunk_size=30, overlap=0)
        assert len(chunks) == tc.MAX_CHUNKS
        assert chunks[-1].endswith("Sentence number 2999 has six words.")

    def test_get_book_chunks_caches_its_result(self, isolated_store):
        first = tc.get_book_chunks(3, sentences(10))
        assert (isolated_store / "book_3_chunks.json").exists()
        assert tc.get_book_chunks(3, "completely different text.") == first


class TestPassageSelection:
    CHUNKS = [
        "The voyage began in spring with a calm sea.",
        "They mended nets and talked about the harvest.",
        "A storm broke the mast and the lighthouse lamp went dark.",
        "Weeks of quiet trading followed in the southern ports.",
        "At last the ship came home and the crew were paid.",
    ]

    def test_the_passage_that_matches_the_question_is_chosen(self):
        picked = tc.select_passages(self.CHUNKS, "What happened during the storm?", limit=1)
        assert picked == [self.CHUNKS[2]]

    def test_selected_passages_stay_in_book_order(self):
        picked = tc.select_passages(self.CHUNKS, "storm ship home", limit=2)
        assert picked == [self.CHUNKS[2], self.CHUNKS[4]]

    def test_no_overlap_falls_back_to_opening_middle_and_end(self):
        picked = tc.select_passages(self.CHUNKS, "quantum chromodynamics", limit=3)
        assert picked == [self.CHUNKS[0], self.CHUNKS[2], self.CHUNKS[4]]

    def test_short_books_are_used_whole(self):
        assert tc.select_passages(self.CHUNKS[:2], "anything") == self.CHUNKS[:2]
