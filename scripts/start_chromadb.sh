#!/usr/bin/env sh
set -eu

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

CHROMA_HOST="${CHROMA_HTTP_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_HTTP_PORT:-8000}"
CHROMA_PATH="${CHROMA_PATH:-./data/chromadb}"

mkdir -p "$CHROMA_PATH"

exec chroma run --host "$CHROMA_HOST" --port "$CHROMA_PORT" --path "$CHROMA_PATH"
