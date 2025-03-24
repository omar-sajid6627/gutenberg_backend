# Gutenberg API

The backend for [Project Gutenberg Explorer](../Gutenberg-web-app): fetches
books from gutenberg.org, chunks them, answers questions about them, scores
their sentiment, measures their readability, and reads them aloud.

FastAPI, LangChain, NLTK, edge-tts.

**It runs with no API key.** That was the first thing to fix: the module that
talks to Groq crashed *at import* when the key was unset, so the entire API was
unimportable — not degraded, unimportable.

---

## Run it

```bash
docker compose up -d --build
```

The API is on **http://localhost:8600**, docs at `/docs`.

```bash
# Metadata
curl http://localhost:8600/books/2701

# Flesch Reading Ease, computed from the text
curl http://localhost:8600/books/2701/readability

# Ask a question — answered from the book with no key configured
curl -X POST http://localhost:8600/books/2701/ask \
  -H 'Content-Type: application/json' \
  -d '{"book_id":"2701","query":"Why does the narrator go to sea?"}'
```

To run it with the frontend, use the compose file in `../Gutenberg-web-app`,
which starts both.

---

## The import-time crash

```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
```

With no key, `os.getenv` returns `None` and assigning `None` into `os.environ`
raises `TypeError: str expected, not NoneType`. Because the handler was
constructed as a module-level singleton, that single line made the whole
application fail to start, with an error that named neither the key nor the
cause.

The key is now validated first — including recognising placeholder values like
`your_groq_api_key_here`, which are non-empty and would otherwise select the
hosted path and fail later with a 401 — and a local handler takes over when
there isn't one.

**Answers are extractive offline, on purpose.** These are questions about the
text of a specific book, and the answer is supposed to come from that book. A
stand-in that invented prose would be demonstrating the exact failure a
retrieval-over-documents system exists to prevent. Every line it returns
appears verbatim in the passage it was given, and it says so.

---

## Readability

Added because the frontend was displaying a number it had made up. See the
[frontend README](../Gutenberg-web-app) for that story; the fix lives here.

```
book                     score  band              words/sentence
Alice in Wonderland       78.9  Fairly easy               17.3
Moby Dick                 69.5  Standard                  15.8
Pride and Prejudice       64.8  Standard                  16.7
Frankenstein              54.3  Fairly difficult          23.3
```

Flesch Reading Ease is `206.835 − 1.015 × (words/sentences) − 84.6 ×
(syllables/words)`. It is a crude instrument — it counts syllables and sentence
lengths and knows nothing about vocabulary or subject matter — but it is a real
function of the text and it is reproducible.

Two things had to be right for those numbers to mean anything:

**Words and sentences must be counted over the same span.** Counting words from
one sample and sentences from a character slice of the original gave Alice in
Wonderland 29 words per sentence, which is not remotely true of it.

**Sentences can end inside quotation marks.** `"Off with her head!" said the
Queen.` terminates before the closing quote. Without allowing for that,
dialogue-heavy books lose most of their sentence boundaries — which is what
produced Alice's implausible number.

`null` means "not measurable", and is returned rather than a default. That
distinction is the entire point.

---

## What else was wrong

**The dependency list was a 73-package `pip freeze`,** including `wordcloud`,
`matplotlib`, `textblob`, `seaborn` and `pandas` — none imported anywhere.
`wordcloud` has no wheel for this platform and needs gcc to build, so an unused
dependency was what broke the image build.

**There was no Dockerfile.** NLTK corpora were also downloaded at first use,
turning a cold start into a silent network call that fails outright on a
machine with no outbound access. They are fetched at build time now.

---

## Tests

```bash
docker compose run --rm --no-deps api python -m pytest tests/ -q
```

37 tests, from none:

- **Readability ranks known texts correctly** — simple prose above complex —
  and returns `None` rather than a default when it cannot measure.
- **Sentences ending inside quotes are counted**, pinning the fix above.
- **Extractive answers quote only text that is in the source**, and say they
  were not generated.
- **Placeholder API keys count as absent.**

Two bugs surfaced while writing them: a 30-character minimum sentence length
silently discarded "Call me Ishmael." (sixteen characters), and the summary
keyword pattern `\bsummar\b` never matched the word "summary" — the trailing
word boundary requires the word to end there.

---

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /books/{id}` | Metadata from gutenberg.org |
| `GET /books/{id}/content` | Paginated text |
| `GET /books/{id}/readability` | Flesch Reading Ease, or `null` |
| `POST /books/{id}/ask` | Question answering over the text |
| `GET /books/{id}/sentiment` | Sentiment + word frequencies |
| `POST /tts` | Read aloud via edge-tts |
| `GET /health` | Liveness |

## Known limitations

- **Sentiment analysis takes about a minute** on a full novel and runs in a
  background task; the first request returns `status: "processing"`.
- **Retrieval is semantic, sentence selection is lexical.** The chunker embeds
  passages, but the offline answerer picks sentences by word overlap. A
  sentence can answer a question while sharing no words with it, and this will
  miss it.
- **Readability is measured on the first 20,000 words**, not the whole book.
- **No caching layer beyond the in-process content cache**, so a restart
  re-downloads.
