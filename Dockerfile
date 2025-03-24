FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/app/.nltk

WORKDIR /app

COPY requirements.txt requirements.freeze.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements.freeze.txt

# NLTK data (stopwords, VADER lexicon, WordNet) comes from NLTK's own
# download server, not PyPI. The API never fetches it at runtime; without it
# only the sentiment endpoint is unavailable (503). Build with
# --build-arg NLTK_DOWNLOAD=0 to skip it.
ARG NLTK_DOWNLOAD=1
COPY scripts/download_nltk_data.py scripts/
RUN if [ "$NLTK_DOWNLOAD" = "1" ]; then python scripts/download_nltk_data.py --dir "$NLTK_DATA"; fi

COPY main.py ./
COPY app ./app

# Code stays root-owned and read-only; only the cache directory is writable.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown app:app /app/data
USER app

EXPOSE 8000


FROM base AS test
USER root
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pytest.ini ruff.toml ./
COPY tests ./tests
USER app
CMD ["sh", "-c", "python -m pytest && ruff check --no-cache ."]


FROM base AS runtime
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
