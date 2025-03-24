"""
Tests for Flesch Reading Ease.

The frontend used to fabricate this number by hashing the book ID
(`50 + hashNum % 51`) and display it exactly like a measurement. Replacing a
fake number with a *wrong* one would be no improvement, so these tests check
that the score behaves the way a readability measure must: it ranks known texts
in the right order, and it returns nothing rather than a default when it cannot
measure.
"""

from __future__ import annotations

import pytest

from app.utils.readability import (
    count_syllables,
    flesch_reading_ease,
    label_for,
)

SIMPLE = (
    "The cat sat on the mat. The dog ran fast. I like the sun. "
    "We go to the park. It is a good day. The bird can fly. "
    "She has a red ball. He ate the cake. They saw a big tree. "
    "You can run too. We had fun. The sky is blue. It was warm. "
) * 6

COMPLEX = (
    "The epistemological ramifications of phenomenological interpretation "
    "necessitate a comprehensive reconsideration of foundational assumptions "
    "regarding intersubjective constitution, particularly insofar as such "
    "considerations illuminate the transcendental conditions underlying "
    "consciousness and its intentional comportment toward objectivities. "
) * 8


class TestSyllables:
    @pytest.mark.parametrize(
        "word,expected",
        [("cat", 1), ("running", 2), ("beautiful", 3), ("the", 1), ("make", 1)],
    )
    def test_common_words(self, word, expected):
        assert count_syllables(word) == expected

    def test_never_returns_zero(self):
        # A zero would divide into the score and produce nonsense.
        for word in ["a", "I", "rhythm", "xyz", "'"]:
            assert count_syllables(word) >= 1 or word == "'"

    def test_silent_trailing_e_is_not_counted(self):
        assert count_syllables("make") < count_syllables("maker")


class TestScoring:
    def test_simple_prose_scores_higher_than_complex(self):
        # The single property that makes the measure meaningful.
        simple = flesch_reading_ease(SIMPLE)
        complex_ = flesch_reading_ease(COMPLEX)
        assert simple is not None and complex_ is not None
        assert simple.score > complex_.score

    def test_simple_prose_lands_in_an_easy_band(self):
        result = flesch_reading_ease(SIMPLE)
        assert result.score > 70, f"expected easy prose to score high, got {result.score}"

    def test_the_score_is_bounded(self):
        for text in (SIMPLE, COMPLEX):
            assert 0.0 <= flesch_reading_ease(text).score <= 100.0

    def test_it_is_deterministic(self):
        assert flesch_reading_ease(SIMPLE).score == flesch_reading_ease(SIMPLE).score

    def test_word_and_sentence_counts_come_from_the_same_span(self):
        # Counting words from one sample and sentences from a different slice
        # gave Alice in Wonderland 29 words per sentence, which is not true of
        # it. The counts must describe the same text.
        result = flesch_reading_ease(SIMPLE)
        assert 3 < result.words_per_sentence < 15


class TestQuotedDialogue:
    def test_sentences_ending_inside_quotes_are_counted(self):
        """
        `"Off with her head!" said the Queen.` ends a sentence before the
        closing quote. Missing that made dialogue-heavy books look like they
        had enormous sentences.
        """
        dialogue = (
            '"Off with her head!" said the Queen. "I am late," said the Rabbit. '
            '"Curiouser and curiouser!" cried Alice. "Who are you?" said the '
            'Caterpillar. "Have some wine," the Hare urged. '
        ) * 8
        result = flesch_reading_ease(dialogue)
        assert result is not None
        assert result.words_per_sentence < 15, (
            f"{result.words_per_sentence:.1f} words per sentence suggests the "
            "sentence terminators inside quotes were missed"
        )


class TestNotMeasurable:
    def test_empty_text_returns_none(self):
        # None must be distinguishable from a real score — that distinction is
        # exactly what the fabricated value destroyed.
        assert flesch_reading_ease("") is None
        assert flesch_reading_ease("   ") is None

    def test_a_handful_of_words_returns_none(self):
        assert flesch_reading_ease("Too short to measure.") is None

    def test_it_never_returns_a_default_score(self):
        for text in ["", "  ", "one two three"]:
            assert flesch_reading_ease(text) is None


class TestLabels:
    @pytest.mark.parametrize(
        "score,expected",
        [(95, "Very easy"), (75, "Fairly easy"), (65, "Standard"),
         (40, "Difficult"), (10, "Very difficult")],
    )
    def test_bands(self, score, expected):
        assert label_for(score) == expected

    def test_labels_are_monotonic(self):
        # A higher score must never map to a harder label.
        order = ["Very difficult", "Difficult", "Fairly difficult",
                 "Standard", "Fairly easy", "Easy", "Very easy"]
        seen = [order.index(label_for(s)) for s in range(0, 101, 5)]
        assert seen == sorted(seen)
