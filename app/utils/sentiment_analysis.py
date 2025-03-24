"""
Book-level sentiment (NLTK VADER) and word frequencies for a word cloud.

NLTK data is loaded on first use and never downloaded here. The module used to
call nltk.download() at import time, so the whole API failed to import on a
machine without outbound network access. Install the data once with
`python scripts/download_nltk_data.py`; the Docker image does that at build
time.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, book_file
from app.utils.sentences import split_sentences

logger = logging.getLogger(__name__)

SENTIMENT_DIR = DATA_DIR / "sentiment"
SENTIMENT_CACHE_EXPIRATION = 12 * 60 * 60

# Full novels run to millions of characters; VADER and the lemmatiser are
# pure Python, so analysis works on an evenly spread sample instead.
MAX_CHARS = 1_000_000
MAX_TOKENS = 50_000
MAX_SENTENCES = 300
TOP_WORDS = 150

NON_LETTERS = re.compile(r"[^a-z\s]")

_sentiment_cache: dict[int, dict[str, Any]] = {}


class MissingNLTKData(RuntimeError):
    """The NLTK corpora this module needs are not installed."""


@dataclass(frozen=True)
class NLPResources:
    stop_words: frozenset[str]
    lemmatize: Callable[[str], str]
    polarity_scores: Callable[[str], dict[str, float]]


@lru_cache(maxsize=1)
def nlp_resources() -> NLPResources:
    from nltk.corpus import stopwords
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.stem import WordNetLemmatizer

    try:
        stop_words = frozenset(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("books")  # forces the lazy wordnet load now
        analyzer = SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise MissingNLTKData(
            "NLTK data is missing; run `python scripts/download_nltk_data.py`."
        ) from exc
    return NLPResources(stop_words, lemmatizer.lemmatize, analyzer.polarity_scores)


def _sample(text: str, limit: int) -> str:
    """Beginning, middle and end of `text`, `limit` characters in total."""
    if len(text) <= limit:
        return text
    part = limit // 3
    middle = (len(text) - part) // 2
    return "\n\n".join((text[:part], text[middle : middle + part], text[-part:]))


def _cache_file(book_id: int) -> Path:
    return book_file(SENTIMENT_DIR, book_id, "sentiment.json")


def get_cached_sentiment(book_id: int) -> dict[str, Any] | None:
    entry = _sentiment_cache.get(book_id)
    if entry is None:
        cache_file = _cache_file(book_id)
        if not cache_file.exists():
            return None
        try:
            entry = json.loads(cache_file.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable sentiment cache %s: %s", cache_file, exc)
            return None
        _sentiment_cache[book_id] = entry

    if time.time() - entry.get("timestamp", 0) >= SENTIMENT_CACHE_EXPIRATION:
        _sentiment_cache.pop(book_id, None)
        return None
    result = entry["result"]
    result.setdefault("status", "completed")
    return result


def save_sentiment_to_cache(book_id: int, result: dict[str, Any]) -> None:
    entry = {"result": result, "timestamp": time.time()}
    _sentiment_cache[book_id] = entry
    try:
        SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
        _cache_file(book_id).write_text(json.dumps(entry))
    except OSError as exc:
        logger.warning("Could not write sentiment cache for book %s: %s", book_id, exc)


def preprocess_text(content: str, nlp: NLPResources) -> list[str]:
    """Lower-cased, stopword-free, lemmatised tokens of three letters or more."""
    words = NON_LETTERS.sub("", content.lower()).split()[:MAX_TOKENS]
    return [nlp.lemmatize(w) for w in words if len(w) > 2 and w not in nlp.stop_words]


def analyze_sentiment(content: str, nlp: NLPResources) -> dict[str, float]:
    """Mean VADER scores over an evenly spaced sample of sentences."""
    sentences = split_sentences(content)
    if not sentences:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    step = max(1, len(sentences) // MAX_SENTENCES)
    sentences = sentences[::step][:MAX_SENTENCES]
    scores = [nlp.polarity_scores(s) for s in sentences]
    return {key: sum(s[key] for s in scores) / len(scores) for key in ("neg", "neu", "pos", "compound")}


def get_word_frequencies(tokens: list[str]) -> list[dict[str, Any]]:
    return [{"text": word, "value": count} for word, count in Counter(tokens).most_common(TOP_WORDS)]


def overall_label(compound: float) -> str:
    """VADER's conventional thresholds."""
    if compound < -0.05:
        return "negative"
    if compound <= 0.05:
        return "neutral"
    return "positive"


def analyze_book_sentiment(content: str, book_id: int) -> dict[str, Any]:
    cached = get_cached_sentiment(book_id)
    if cached:
        return cached

    nlp = nlp_resources()
    started = time.monotonic()
    content = _sample(content, MAX_CHARS)
    tokens = preprocess_text(content, nlp)
    word_data = get_word_frequencies(tokens)
    scores = analyze_sentiment(content, nlp)
    overall = overall_label(scores["compound"])

    result = {
        "book_id": book_id,
        "sentiment": {
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": scores["compound"],
            "overall": overall,
        },
        "overall_sentiment": overall,
        "wordcloud_data": {
            "words": word_data,
            "word_count": len(word_data),
            "total_words_analyzed": len(tokens),
        },
        "status": "completed",
        "message": "Sentiment analysis completed successfully",
    }
    logger.info("Sentiment for book %s took %.1fs", book_id, time.monotonic() - started)
    save_sentiment_to_cache(book_id, result)
    return result
