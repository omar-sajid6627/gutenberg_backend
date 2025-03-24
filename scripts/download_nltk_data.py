"""
Install the NLTK data the sentiment endpoint needs.

The API never downloads anything at import or request time; run this once
with network access (the Docker image runs it at build time):

    python scripts/download_nltk_data.py [--dir PATH]

Without --dir, NLTK's default location is used; set NLTK_DATA to point the
API at a custom one.
"""

from __future__ import annotations

import argparse

import nltk

PACKAGES = ("stopwords", "vader_lexicon", "wordnet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", default=None, help="download directory")
    args = parser.parse_args()
    for package in PACKAGES:
        nltk.download(package, download_dir=args.dir, quiet=True, raise_on_error=True)
        print(f"installed {package}")


if __name__ == "__main__":
    main()
