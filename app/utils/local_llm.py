"""
Extractive answering, used when no GROQ_API_KEY is configured.

`llm_handler` crashed on import without a key:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY   # TypeError: str expected, not NoneType

so the whole API was unimportable — not degraded, unimportable — and the error
said nothing about a missing key. Everything worth reviewing here (the
Gutenberg scraper, the chunker, the sentiment analysis, the caching) works
without a hosted model.

**Extractive on purpose.** These are questions about the text of a specific
book, and the answer is supposed to come from that book. A stand-in that
invented prose would be demonstrating exactly the failure a
retrieval-over-documents system exists to prevent. Every sentence returned here
appears verbatim in the passage it was given.
"""

from __future__ import annotations

import os
import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "him", "his", "how", "i",
    "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "them", "there", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")
_SUFFIXES = ("ing", "ies", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            base = word[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return word


def content_words(text: str) -> set[str]:
    """Stemmed, lower-cased words of three letters or more, minus stopwords."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 2}


def _sentences(text: str, min_length: int = 15) -> list[str]:
    """
    Split into sentences, dropping fragments.

    The floor was 30 characters, which discarded short but complete sentences:
    "Call me Ishmael." is sixteen. It exists to drop headings and Gutenberg
    page furniture ("CHAPTER 1.", "THE END."), not real prose. Fifteen keeps
    the shortest real sentences while still dropping most of the furniture —
    it is a heuristic, and it will occasionally get both wrong.
    """
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]
    return [s for s in parts if len(s) >= min_length]


def score(sentence: str, query_tokens: set[str]) -> float:
    """Overlap with the question, normalised so long sentences do not win by length."""
    tokens = content_words(sentence)
    if not tokens:
        return 0.0
    overlap = len(tokens & query_tokens)
    return 0.0 if overlap == 0 else overlap / (len(tokens) ** 0.5)


def answer_from_context(query: str, context: str, max_sentences: int = 4) -> str:
    """Answer a question by quoting the passages that address it."""
    if not context or not context.strip():
        return (
            "No text from this book was available to answer from. The book "
            "content may still be downloading."
        )

    query_tokens = content_words(query)
    scored = [(score(s, query_tokens), s) for s in _sentences(context)]
    scored = [(sc, s) for sc, s in scored if sc > 0]
    scored.sort(key=lambda pair: -pair[0])

    if not scored:
        return (
            "Nothing in the retrieved passages uses the words in that question, "
            "so there is nothing to quote. Try phrasing it with words the book "
            "itself would use.\n\n"
            "_Answered without a language model: passages are selected from the "
            "book's text, not generated. Set GROQ_API_KEY for a written answer._"
        )

    chosen = [s for _, s in scored[:max_sentences]]
    body = "\n\n".join(f"> {s}" for s in chosen)
    return (
        "From the book's own text:\n\n"
        f"{body}\n\n"
        "_Answered without a language model: passages are selected from the "
        "book's text, not generated. Set GROQ_API_KEY for a written answer._"
    )


def summarise(context: str, max_sentences: int = 6) -> str:
    """
    A lead summary: the opening sentences of the passage.

    Deliberately not an abstractive summary. Compressing meaning is exactly
    what needs a model, and faking it would misrepresent what the offline mode
    can do.
    """
    sentences = _sentences(context)
    if not sentences:
        return "No text was available to summarise."
    lead = " ".join(sentences[:max_sentences])
    return (
        f"{lead}\n\n"
        "_This is the opening of the text, not a written summary. Summarising "
        "needs a language model; set GROQ_API_KEY to generate one._"
    )


class LocalLLMHandler:
    """Same async surface as LLMHandler, no network."""

    is_local = True

    async def generate_response(
        self, query: str, context: str | None = None, temperature: float = 0
    ) -> str:
        if context is None:
            return (
                "No book text was supplied with this question, and there is no "
                "language model configured to answer it from general knowledge."
            )
        # `\bsummar\b` never matched "summary" or "summarise" — the trailing
        # word boundary requires the word to *end* there.
        if re.search(r"\b(summar\w*|overview|synopsis|what is this book about)", query, re.I):
            return summarise(context)
        return answer_from_context(query, context)


def usable_api_key(name: str = "GROQ_API_KEY") -> str | None:
    """The key, or None when absent or an obvious placeholder."""
    value = (os.getenv(name) or "").strip()
    if not value:
        return None
    lowered = value.lower()
    if any(m in lowered for m in ("your_", "your-", "changeme", "xxx", "<", "api_key_here")):
        return None
    return value
