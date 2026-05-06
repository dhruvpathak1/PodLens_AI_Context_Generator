# syntax=docker/dockerfile:1
# PodLens: React SPA (built) + FastAPI (Whisper, entities, enrichment).

FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts index.html ./
COPY public ./public
COPY src ./src
# Same-origin /api/* in the browser — no VITE_TRANSCRIBE_URL needed.
RUN npm run build

FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt ./server/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir openai-whisper \
    && pip install --no-cache-dir -r server/requirements.txt

COPY server ./server
COPY --from=frontend /build/dist ./server/static

ENV PYTHONUNBUFFERED=1 \
    STATIC_DIST_DIR=/app/server/static \
    WHISPER_DOWNLOAD_ROOT=/data/whisper_models \
    TORCH_HOME=/data/torch_home \
    TRANSCRIPTS_DIR=/data/transcripts \
    ENTITY_JSON_DIR=/data/entity_exports

RUN mkdir -p /data/whisper_models /data/torch_home /data/transcripts /data/entity_exports

WORKDIR /app/server

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
