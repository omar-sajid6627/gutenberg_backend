"""The Groq-backed handler, with a stand-in model instead of the network."""

from __future__ import annotations

import asyncio

from app.utils import llm_handler as lh


class FakeChat:
    def __init__(self, reply="The lamp went dark.", error=None, delay=0):
        self.reply, self.error, self.delay = reply, error, delay
        self.calls = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error

        class Message:
            content = self.reply

        return Message()


async def test_context_is_put_in_the_prompt_and_temperature_passed():
    chat = FakeChat()
    answer = await lh.LLMHandler(llm=chat).generate_response("What happened?", "A storm came.", 0.7)
    assert answer == "The lamp went dark."
    messages, kwargs = chat.calls[0]
    assert "Context: A storm came." in messages[0].content
    assert kwargs["temperature"] == 0.7


async def test_errors_are_logged_not_returned():
    chat = FakeChat(error=RuntimeError("401 invalid key gsk_secret"))
    answer = await lh.LLMHandler(llm=chat).generate_response("q", "c")
    assert "gsk_secret" not in answer and "error" in answer


async def test_timeouts_are_reported(monkeypatch):
    monkeypatch.setattr(lh, "API_TIMEOUT", 0.01)
    answer = await lh.LLMHandler(llm=FakeChat(delay=1)).generate_response("q", "c")
    assert "timed out" in answer


def test_without_a_key_the_local_handler_is_chosen(monkeypatch):
    monkeypatch.setattr(lh, "GROQ_API_KEY", None)
    assert lh._build_handler().is_local


def test_a_key_selects_groq(monkeypatch):
    class Groq:
        is_local = False

    monkeypatch.setattr(lh, "GROQ_API_KEY", "gsk_testvalue")
    monkeypatch.setattr(lh, "LLMHandler", Groq)
    assert not lh._build_handler().is_local


def test_a_key_that_fails_to_initialise_falls_back(monkeypatch):
    def broken():
        raise ValueError("bad key")

    monkeypatch.setattr(lh, "GROQ_API_KEY", "gsk_testvalue")
    monkeypatch.setattr(lh, "LLMHandler", broken)
    assert lh._build_handler().is_local
