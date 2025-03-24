"""
Tests for extractive answering without a Groq key.

`llm_handler` crashed at import without a key — `os.environ[...] = None` raises
TypeError — so the entire API was unimportable. These pin that the offline path
works and, more importantly, that it never invents text.
"""

from __future__ import annotations

import pytest

from app.utils.local_llm import (
    LocalLLMHandler,
    answer_from_context,
    summarise,
    usable_api_key,
)

PASSAGE = (
    "Call me Ishmael. Some years ago, never mind how long precisely, having "
    "little or no money in my purse, I thought I would sail about a little and "
    "see the watery part of the world. Whenever I find myself growing grim "
    "about the mouth, I account it high time to get to sea as soon as I can. "
    "The Pequod sailed south toward the whaling grounds of the Pacific."
)


class TestGrounding:
    def test_every_quoted_line_appears_in_the_source(self):
        answer = answer_from_context("Why does the narrator go to sea?", PASSAGE)
        quoted = [ln[2:].strip() for ln in answer.splitlines() if ln.startswith("> ")]
        assert quoted, "nothing was quoted"
        for line in quoted:
            assert line in PASSAGE, f"invented text: {line!r}"

    def test_it_says_it_did_not_generate_the_text(self):
        answer = answer_from_context("Why go to sea?", PASSAGE)
        assert "without a language model" in answer.lower()

    def test_a_question_sharing_no_vocabulary_is_admitted(self):
        answer = answer_from_context("quantum chromodynamics lattice", PASSAGE)
        assert "nothing" in answer.lower() or "not" in answer.lower()
        assert ">" not in answer.split("_Answered")[0]

    def test_empty_context_is_reported_not_answered(self):
        assert "no text" in answer_from_context("anything", "").lower()


class TestSummary:
    def test_it_returns_the_opening_of_the_text(self):
        assert "Call me Ishmael" in summarise(PASSAGE)

    def test_it_does_not_claim_to_be_a_summary(self):
        # Compressing meaning needs a model; pretending otherwise would
        # misrepresent what the offline mode can do.
        assert "not a written summary" in summarise(PASSAGE).lower()


class TestHandler:
    async def test_it_answers_from_context(self):
        result = await LocalLLMHandler().generate_response("Why go to sea?", PASSAGE)
        assert "Ishmael" in result or "sea" in result.lower()

    async def test_a_summary_request_is_routed_to_the_summariser(self):
        result = await LocalLLMHandler().generate_response("Give me a summary", PASSAGE)
        assert "not a written summary" in result.lower()

    async def test_no_context_is_admitted_rather_than_answered(self):
        result = await LocalLLMHandler().generate_response("What is the capital of France?")
        assert "no book text" in result.lower()


class TestKeySelection:
    def test_absent_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert usable_api_key() is None

    @pytest.mark.parametrize("value", ["your_groq_api_key_here", "changeme", "   ", "<key>"])
    def test_placeholders_count_as_absent(self, monkeypatch, value):
        monkeypatch.setenv("GROQ_API_KEY", value)
        assert usable_api_key() is None

    def test_a_real_key_is_returned(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_realkeyvalue123")
        assert usable_api_key() == "gsk_realkeyvalue123"
