"""
Sentiment analysis, run against stand-in NLTK resources.

The real VADER lexicon and WordNet are downloads; the tests must not depend on
them being installed or on network access to fetch them.
"""

from __future__ import annotations

import pytest

from app.utils import sentiment_analysis as sa
from app.utils.sentences import split_sentences

REAL_NLP_RESOURCES = sa.nlp_resources

POSITIVE = {"happy", "love", "wonderful", "delight"}
NEGATIVE = {"dreadful", "hate", "grief", "misery"}


def fake_polarity(sentence: str) -> dict[str, float]:
    words = sentence.lower().replace(".", "").split()
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    total = max(len(words), 1)
    compound = (pos - neg) / max(pos + neg, 1)
    return {"pos": pos / total, "neg": neg / total, "neu": 1 - (pos + neg) / total, "compound": compound}


FAKE_NLP = sa.NLPResources(
    stop_words=frozenset({"the", "and", "was", "she", "her", "with"}),
    lemmatize=lambda w: w[:-1] if w.endswith("s") and len(w) > 4 else w,
    polarity_scores=fake_polarity,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "SENTIMENT_DIR", tmp_path)
    monkeypatch.setattr(sa, "_sentiment_cache", {})
    monkeypatch.setattr(sa, "nlp_resources", lambda: FAKE_NLP)


def test_happy_text_is_positive():
    result = sa.analyze_book_sentiment("She was happy. Love and delight filled the house. " * 20, "11")
    assert result["overall_sentiment"] == "positive"
    assert result["status"] == "completed"


def test_grim_text_is_negative():
    result = sa.analyze_book_sentiment("Dreadful grief came. Misery and hate remained. " * 20, "12")
    assert result["overall_sentiment"] == "negative"


def test_word_frequencies_skip_stopwords_and_are_ranked():
    result = sa.analyze_book_sentiment("The garden and the gardens. The garden was green. " * 5, "13")
    words = result["wordcloud_data"]["words"]
    assert words[0] == {"text": "garden", "value": 15}
    assert all(w["text"] not in FAKE_NLP.stop_words for w in words)


def test_results_are_cached_to_disk(tmp_path):
    sa.analyze_book_sentiment("Wonderful days were had by all of them. " * 10, "14")
    assert (tmp_path / "book_14_sentiment.json").exists()
    sa._sentiment_cache.clear()
    assert sa.get_cached_sentiment("14")["overall_sentiment"] == "positive"


def test_sentence_sample_is_capped():
    calls = []
    nlp = sa.NLPResources(frozenset(), str, lambda s: calls.append(s) or fake_polarity(s))
    sa.analyze_sentiment("A short line here. " * 2000, nlp)
    assert len(calls) <= sa.MAX_SENTENCES


def test_long_text_is_sampled_from_beginning_middle_and_end():
    text = "a" * 1000 + "b" * 1000 + "c" * 1000
    sample = sa._sample(text, 300)
    assert sample.startswith("a") and "b" in sample and sample.endswith("c")


def test_missing_nltk_data_is_reported_not_downloaded(tmp_path, monkeypatch):
    import nltk

    monkeypatch.setattr(nltk.data, "path", [str(tmp_path)])
    monkeypatch.setattr(nltk, "download", lambda *a, **k: pytest.fail("tried to download"))
    REAL_NLP_RESOURCES.cache_clear()
    with pytest.raises(sa.MissingNLTKData):
        REAL_NLP_RESOURCES()


def test_sentences_ending_inside_quotes_are_split():
    assert split_sentences('"Off with her head!" said the Queen. "Who are you?" asked Alice.') == [
        '"Off with her head!" said the Queen.',
        '"Who are you?" asked Alice.',
    ]
