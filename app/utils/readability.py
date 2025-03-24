"""
Flesch Reading Ease, computed from the book's actual text.

The frontend previously displayed a "Reading Ease" score that was invented:

    // Generate a score between 50 and 100
    return 50 + (hashNum % 51)

It hashed the book's ID and derived a number from it, then labelled the result
"Easy" or "Difficult". A reader had no way to tell that the figure was not a
measurement of anything. Every book got a stable, plausible, meaningless score.

Flesch Reading Ease is a closed-form measure over the text:

    206.835 - 1.015 x (words / sentences) - 84.6 x (syllables / words)

Higher is easier: 90-100 reads like children's fiction, 30-50 like academic
prose, below 30 is very hard going. It is a crude instrument — it counts
syllables and sentence lengths, and knows nothing about vocabulary or subject
matter — but it is a real function of the text, and it is reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VOWEL_GROUPS = re.compile(r"[aeiouy]+")
# A sentence terminator may be followed by a closing quote or bracket before
# the whitespace: `"Off with her head!" said the Queen.` Without allowing for
# that, dialogue-heavy books lose most of their sentence boundaries — Alice in
# Wonderland measured 29 words per sentence, which is not remotely true of it.
SENTENCE_END = re.compile(r"""[.!?]+["'”’)\]]*(?:\s|$)""")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def count_syllables(word: str) -> int:
    """
    Approximate syllable count for an English word.

    Vowel groups, minus a silent trailing 'e', floored at one. This is the
    standard heuristic and it is wrong on plenty of words ("queue", "rhythm");
    over thousands of words the errors largely wash out, which is the only
    reason a readability score computed this way means anything at all.
    """
    word = word.lower().strip("'-")
    if not word:
        return 0

    groups = VOWEL_GROUPS.findall(word)
    count = len(groups)

    # A trailing 'e' is usually silent ("make"), unless it is the only vowel
    # group ("the") or follows a consonant+le ("table").
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1

    return max(count, 1)


def _first_n_words(text: str, n: int) -> str:
    """The prefix of `text` containing its first `n` words, boundaries intact."""
    count = 0
    for match in WORD.finditer(text):
        count += 1
        if count == n:
            return text[: match.end()]
    return text


@dataclass
class Readability:
    score: float
    label: str
    words: int
    sentences: int
    syllables: int
    words_per_sentence: float
    syllables_per_word: float

    def as_dict(self) -> dict:
        return {
            "reading_ease_score": round(self.score, 1),
            "reading_ease_label": self.label,
            "words": self.words,
            "sentences": self.sentences,
            "words_per_sentence": round(self.words_per_sentence, 1),
            "syllables_per_word": round(self.syllables_per_word, 2),
        }


# Lower bound of each conventional Flesch band, easiest first.
BANDS = (
    (90, "Very easy"),
    (80, "Easy"),
    (70, "Fairly easy"),
    (60, "Standard"),
    (50, "Fairly difficult"),
    (30, "Difficult"),
)


def label_for(score: float) -> str:
    for floor, label in BANDS:
        if score >= floor:
            return label
    return "Very difficult"


def flesch_reading_ease(text: str, sample_words: int = 20000) -> Readability | None:
    """
    Score a passage. Returns None when there is not enough text to measure.

    Returning None rather than a default is the point: a missing measurement
    must be distinguishable from a real one, which is exactly what the
    fabricated score destroyed.
    """
    if not text or not text.strip():
        return None

    if len(WORD.findall(text[:200_000])) < 100:
        return None

    # Words and sentences must be counted over the *same* span of text.
    # Counting words from one sample and sentences from a character slice of
    # the original gave Alice in Wonderland 30 words per sentence — it has
    # short sentences, and the two spans simply did not correspond.
    sample = _first_n_words(text, sample_words)
    words = WORD.findall(sample)
    if len(words) < 100:
        return None

    sentence_count = max(len(SENTENCE_END.findall(sample)), 1)

    syllables = sum(count_syllables(w) for w in words)

    words_per_sentence = len(words) / sentence_count
    syllables_per_word = syllables / len(words)

    score = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    # The formula is unbounded; the scale it is interpreted on is 0-100.
    score = max(0.0, min(100.0, score))

    return Readability(
        score=score,
        label=label_for(score),
        words=len(words),
        sentences=sentence_count,
        syllables=syllables,
        words_per_sentence=words_per_sentence,
        syllables_per_word=syllables_per_word,
    )
