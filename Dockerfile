FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/app/.nltk

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# NLTK corpora are downloaded at first use by default, which turns a cold
# start into a silent network call — and fails outright on a machine with no
# outbound access. Fetch them at build time instead.
RUN python -c "import nltk; [nltk.download(p, download_dir='/app/.nltk', quiet=True) for p in ('punkt','punkt_tab','vader_lexicon','stopwords')]"

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/data /app/.nltk \
 && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
