"""Sentence splitting without NLTK's punkt model."""

from __future__ import annotations

import re

# A sentence ends at . ! or ?, possibly inside a closing quote or bracket, and
# the next one starts with a capital. That keeps `"Off with her head!" said the
# Queen.` whole. Python's lookbehind must be fixed-width, hence two branches.
SENTENCE_BOUNDARY = re.compile(
    r"""(?:(?<=[.!?])|(?<=[.!?]["'”’)\]]))\s+(?=["'“‘(\[]?[A-Z0-9])"""
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_BOUNDARY.split(text or "") if s.strip()]
