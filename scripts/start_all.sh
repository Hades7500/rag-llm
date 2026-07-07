#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
elif [ -f .env.example ]; then
  cp .env.example .env
  set -a
  . ./.env
  set +a
  echo "Created .env from .env.example. Edit OPEN_WEBUI_APPIMAGE and WEBUI_SECRET_KEY if needed."
fi

OPEN_WEBUI_APPIMAGE="${OPEN_WEBUI_APPIMAGE:-}"
OLLAMA_BASE_URLS="${OLLAMA_BASE_URLS:-http://127.0.0.1:11434}"
CHROMA_HTTP_HOST="${CHROMA_HTTP_HOST:-127.0.0.1}"
CHROMA_HTTP_PORT="${CHROMA_HTTP_PORT:-8000}"
CHROMA_PATH="${CHROMA_PATH:-./data/chromadb}"
DATA_DIR="${DATA_DIR:-./data/open-webui}"
CHAT_MODEL="${CHAT_MODEL:-llama3.2:3b}"
RAG_EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-nomic-embed-text}"
LOG_DIR="${LOG_DIR:-./logs}"

STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

OLLAMA_PID=""
CHROMA_PID=""
OPEN_WEBUI_PID=""
STREAMLIT_PID=""
STARTED_OLLAMA=0

mkdir -p "$LOG_DIR" "$CHROMA_PATH" "$DATA_DIR"

cleanup() {
  echo ""
  echo "Stopping services..."

  for pid in "$STREAMLIT_PID" "$OPEN_WEBUI_PID" "$CHROMA_PID"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  # Stop Ollama: either the process we launched, or the systemd service.
  if [ "$STARTED_OLLAMA" = "1" ] && [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID" 2>/dev/null || true
  elif systemctl is-active --quiet ollama 2>/dev/null; then
    echo "Stopping Ollama systemd service..."
    systemctl stop ollama 2>/dev/null \
      || sudo systemctl stop ollama 2>/dev/null \
      || echo "  Could not stop Ollama automatically. Run: sudo systemctl stop ollama"
  fi

  wait 2>/dev/null || true
  echo "Stopped."
}

trap cleanup INT TERM EXIT

http_ok() {
  url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "$url" >/dev/null 2>&1
  else
    python3 - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(sys.argv[1], timeout=2).read()
PY
  fi
}

wait_for_http() {
  name="$1"
  url="$2"
  attempts="${3:-60}"

  i=1
  while [ "$i" -le "$attempts" ]; do
    if http_ok "$url"; then
      echo "$name is ready: $url"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done

  echo "Timed out waiting for $name at $url"
  return 1
}

chroma_ok() {
  base_url="http://$CHROMA_HTTP_HOST:$CHROMA_HTTP_PORT"
  http_ok "$base_url/api/v1/heartbeat" || http_ok "$base_url/api/v2/heartbeat"
}

wait_for_chromadb() {
  attempts="${1:-60}"
  i=1
  while [ "$i" -le "$attempts" ]; do
    if chroma_ok; then
      echo "ChromaDB is ready: http://$CHROMA_HTTP_HOST:$CHROMA_HTTP_PORT"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done

  echo "Timed out waiting for ChromaDB at http://$CHROMA_HTTP_HOST:$CHROMA_HTTP_PORT"
  return 1
}

ensure_venv() {
  if [ ! -x .venv/bin/python ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
  fi

  if ! .venv/bin/python -c "import chromadb; import streamlit" >/dev/null 2>&1; then
    echo "Installing Python requirements..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
  fi
}

ensure_appimage() {
  # Open WebUI AppImage is optional. Skip if not configured.
  if [ -z "$OPEN_WEBUI_APPIMAGE" ]; then
    return 0
  fi

  if [ ! -f "$OPEN_WEBUI_APPIMAGE" ]; then
    echo "Warning: OPEN_WEBUI_APPIMAGE set but file not found: $OPEN_WEBUI_APPIMAGE (skipping)"
    OPEN_WEBUI_APPIMAGE=""
    return 0
  fi

  if [ ! -x "$OPEN_WEBUI_APPIMAGE" ]; then
    echo "Making AppImage executable: $OPEN_WEBUI_APPIMAGE"
    chmod +x "$OPEN_WEBUI_APPIMAGE"
  fi
}

start_ollama() {
  if http_ok "$OLLAMA_BASE_URLS/api/tags"; then
    echo "Ollama is already running: $OLLAMA_BASE_URLS"
    return 0
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama command not found. Install Ollama first: https://ollama.com/download"
    exit 1
  fi

  echo "Starting Ollama... logs: $LOG_DIR/ollama.log"
  ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
  OLLAMA_PID="$!"
  STARTED_OLLAMA=1
  wait_for_http "Ollama" "$OLLAMA_BASE_URLS/api/tags" 60
}

ensure_ollama_model() {
  model="$1"
  if ollama list | awk '{print $1}' | grep -Fx "$model" >/dev/null 2>&1; then
    echo "Ollama model already present: $model"
  else
    echo "Pulling Ollama model: $model"
    ollama pull "$model"
  fi
}

start_chromadb() {
  if chroma_ok; then
    echo "ChromaDB is already running: http://$CHROMA_HTTP_HOST:$CHROMA_HTTP_PORT"
    return 0
  fi

  echo "Starting ChromaDB... logs: $LOG_DIR/chromadb.log"
  .venv/bin/chroma run --host "$CHROMA_HTTP_HOST" --port "$CHROMA_HTTP_PORT" --path "$CHROMA_PATH" > "$LOG_DIR/chromadb.log" 2>&1 &
  CHROMA_PID="$!"
  wait_for_chromadb 60
}

start_open_webui() {
  # Optional — only runs if OPEN_WEBUI_APPIMAGE is set and valid.
  if [ -z "$OPEN_WEBUI_APPIMAGE" ]; then
    return 0
  fi

  echo "Starting Open WebUI AppImage... logs: $LOG_DIR/open-webui.log"

  export DATA_DIR
  export WEBUI_NAME="${WEBUI_NAME:-Local RAG}"
  export WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY:-change-me-generate-a-long-random-string}"
  export OLLAMA_BASE_URLS
  export VECTOR_DB="${VECTOR_DB:-chroma}"
  export CHROMA_HTTP_HOST
  export CHROMA_HTTP_PORT
  export CHROMA_TENANT="${CHROMA_TENANT:-default_tenant}"
  export CHROMA_DATABASE="${CHROMA_DATABASE:-default_database}"
  export RAG_EMBEDDING_ENGINE="${RAG_EMBEDDING_ENGINE:-ollama}"
  export RAG_EMBEDDING_MODEL

  "$OPEN_WEBUI_APPIMAGE" > "$LOG_DIR/open-webui.log" 2>&1 &
  OPEN_WEBUI_PID="$!"
}

start_streamlit() {
  if http_ok "http://127.0.0.1:$STREAMLIT_PORT/_stcore/health"; then
    echo "Streamlit is already running at http://127.0.0.1:$STREAMLIT_PORT"
    return 0
  fi

  echo "Starting Streamlit... logs: $LOG_DIR/streamlit.log"
  .venv/bin/streamlit run app/main.py \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    > "$LOG_DIR/streamlit.log" 2>&1 &
  STREAMLIT_PID="$!"
  wait_for_http "Streamlit" "http://127.0.0.1:$STREAMLIT_PORT/_stcore/health" 60
}

ensure_appimage
ensure_venv
start_ollama
ensure_ollama_model "$CHAT_MODEL"
ensure_ollama_model "$RAG_EMBEDDING_MODEL"
start_chromadb
start_streamlit
start_open_webui

cat <<EOF

RAG stack is running.

Streamlit UI:  http://127.0.0.1:$STREAMLIT_PORT
EOF

if [ -n "$OPEN_WEBUI_APPIMAGE" ]; then
  echo "Open WebUI:    http://127.0.0.1:8080"
fi

cat <<EOF

Models:
  Chat model:      $CHAT_MODEL
  Embedding model: $RAG_EMBEDDING_MODEL  (embedding-only, do not use for chat)

Logs:
  Ollama:     $LOG_DIR/ollama.log
  ChromaDB:   $LOG_DIR/chromadb.log
  Streamlit:  $LOG_DIR/streamlit.log

Press Ctrl-C to stop.
EOF

wait
