"""HTTP API over Project Gutenberg: metadata, text, Q&A, sentiment, read-aloud."""

from __future__ import annotations

import logging
import os
import time
from typing import Annotated, Any

import edge_tts
from fastapi import BackgroundTasks, FastAPI, HTTPException, Path
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from mangum import Mangum
from pydantic import BaseModel, Field

from app.models import Book
from app.scraper import fetch_book
from app.utils import get_book_content
from app.utils.content_cache import clear_cache
from app.utils.llm_handler import llm_handler
from app.utils.readability import flesch_reading_ease
from app.utils.sentences import split_sentences
from app.utils.sentiment_analysis import (
    MissingNLTKData,
    analyze_book_sentiment,
    get_cached_sentiment,
    nlp_resources,
)
from app.utils.text_chunking import get_book_chunks, load_chunks

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gutenberg_api")

TTS_VOICE = "en-US-JennyNeural"
# edge-tts is a free Microsoft service; the frontend sends ~1000-character
# chunks, so anything far larger is either a mistake or abuse.
MAX_TTS_CHARS = 5000
READABILITY_PAGES = 40

app = FastAPI(
    title="Gutenberg API",
    description="API for accessing Project Gutenberg books",
    version="1.0.0",
)

# Comma-separated list of allowed browser origins; "*" by default. No
# endpoint uses cookies, so credentials stay off (they cannot be combined with
# a wildcard origin anyway).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gutenberg ebook numbers are positive integers. Typing the path parameter as
# one means FastAPI rejects anything else with 422 before it reaches a cache
# filename built from it.
BookId = Annotated[int, Path(gt=0, le=10_000_000)]


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)
    chunk_index: int = 0
    total_chunks: int = 1


class ChunkRequest(BaseModel):
    text: str = Field(max_length=2_000_000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    # Ignored; the book is the one in the URL. Kept so existing clients validate.
    book_id: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)


def split_text_into_chunks(text: str, chunk_size: int = 1000) -> list[str]:
    """Group whole sentences into pieces of roughly `chunk_size` characters."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(text):
        if current and size + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def require_content(book_id: int) -> dict[str, Any]:
    content = get_book_content(book_id)
    if not content or not content.get("pages"):
        raise HTTPException(status_code=404, detail="Book content not found")
    return content


# Endpoints that fetch from gutenberg.org or read the disk are plain `def`, so
# FastAPI runs them in its threadpool. They used to be `async def` around
# blocking `requests` calls, which stalled every other request meanwhile.

@app.get("/")
def root():
    return {"message": "Welcome to Gutenberg API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/books/{book_id}", response_model=Book)
def get_book(book_id: BookId):
    book = fetch_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@app.get("/books/{book_id}/content")
def get_book_content_endpoint(book_id: BookId):
    return require_content(book_id)


@app.get("/books/{book_id}/readability")
def get_readability(book_id: BookId):
    """Flesch Reading Ease of the opening pages; null scores mean "not measurable"."""
    pages = require_content(book_id)["pages"]
    result = flesch_reading_ease("\n".join(pages[:READABILITY_PAGES]))
    if result is None:
        return {
            "book_id": book_id,
            "reading_ease_score": None,
            "reading_ease_label": None,
            "detail": "Not enough text to measure readability.",
        }
    return {"book_id": book_id, **result.as_dict()}


def context_for_question(book_id: int) -> str:
    chunks = load_chunks(book_id)
    if chunks is None:
        chunks = get_book_chunks(book_id, "\n\n".join(require_content(book_id)["pages"]))
    # Opening, middle and closing passages.
    picked = sorted({0, len(chunks) // 2, len(chunks) - 1}) if chunks else []
    return "\n\n".join(chunks[i] for i in picked)


@app.post("/books/{book_id}/ask")
async def ask_about_book(book_id: BookId, request: QueryRequest):
    started = time.monotonic()
    context = await run_in_threadpool(context_for_question, book_id)
    response = await llm_handler.generate_response(
        query=request.query, context=context, temperature=request.temperature
    )
    return {
        "query": request.query,
        "response": response,
        "book_id": book_id,
        "processing_time": f"{time.monotonic() - started:.2f} seconds",
    }


_sentiment_jobs: set[int] = set()


def _run_sentiment(content: str, book_id: int) -> None:
    try:
        analyze_book_sentiment(content, book_id)
    except Exception:
        logger.exception("Sentiment analysis failed for book %s", book_id)
    finally:
        _sentiment_jobs.discard(book_id)


@app.get("/books/{book_id}/sentiment")
@app.post("/books/{book_id}/sentiment")
def analyze_book_sentiment_endpoint(book_id: BookId, background_tasks: BackgroundTasks):
    """
    Sentiment and word frequencies. Analysis takes a while on a full novel, so
    the first call starts it in the background and answers "processing"; poll
    until "completed".
    """
    cached = get_cached_sentiment(book_id)
    if cached:
        return cached

    if book_id not in _sentiment_jobs:
        content = require_content(book_id)
        try:
            nlp_resources()
        except MissingNLTKData as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        # Without this, every poll while the analysis ran started another one.
        _sentiment_jobs.add(book_id)
        background_tasks.add_task(_run_sentiment, "\n\n".join(content["pages"]), book_id)

    return {
        "book_id": book_id,
        "sentiment": {"positive": 0, "negative": 0, "neutral": 1.0, "compound": 0, "overall": "neutral"},
        "overall_sentiment": "neutral",
        "wordcloud_data": {"words": [], "word_count": 0, "total_words_analyzed": 0},
        "status": "processing",
        "message": "Sentiment analysis is being processed in the background. "
        "Please check again in a few moments.",
    }


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """MP3 of the text, synthesised by edge-tts and returned from memory."""
    audio = bytearray()
    try:
        async for chunk in edge_tts.Communicate(request.text, TTS_VOICE, rate="+0%").stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
    except Exception as exc:
        logger.exception("Text-to-speech failed")
        raise HTTPException(status_code=502, detail="Text-to-speech service unavailable") from exc
    if not audio:
        raise HTTPException(status_code=502, detail="Text-to-speech returned no audio")
    return Response(
        content=bytes(audio),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename=speech_chunk_{request.chunk_index}.mp3",
            "X-Chunk-Index": str(request.chunk_index),
            "X-Total-Chunks": str(request.total_chunks),
        },
    )


@app.post("/tts/chunks")
def get_text_chunks(request: ChunkRequest):
    """The text split into pieces small enough for /tts."""
    chunks = split_text_into_chunks(request.text)
    return {"chunks": chunks, "total_chunks": len(chunks)}


@app.delete("/cache/{book_id}")
def clear_book_cache(book_id: BookId):
    clear_cache(book_id)
    return {"message": f"Cache cleared for book {book_id}"}


@app.delete("/cache")
def clear_all_cache():
    clear_cache()
    return {"message": "All book content cache cleared"}


# Entry point for AWS Lambda; Vercel and uvicorn serve `app` directly.
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
