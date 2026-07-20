#!/bin/sh
set -eu

CHAT_MODEL="${CHAT_MODEL:-llama3.2:3b}"
RAG_EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-nomic-embed-text}"
OLLAMA_PULL_RETRIES="${OLLAMA_PULL_RETRIES:-5}"

/bin/ollama serve &
SERVE_PID=$!

echo "Waiting for Ollama API..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

model_present() {
  ollama list | awk 'NR>1 {print $1}' | grep -Fx "$1" >/dev/null 2>&1
}

pull_model() {
  label="$1"
  model="$2"

  if model_present "$model"; then
    echo "$label model already present: $model"
    return 0
  fi

  attempt=1
  while [ "$attempt" -le "$OLLAMA_PULL_RETRIES" ]; do
    echo "Pulling $label model: $model (attempt $attempt/$OLLAMA_PULL_RETRIES)"
    if ollama pull "$model"; then
      return 0
    fi

    if [ "$attempt" -lt "$OLLAMA_PULL_RETRIES" ]; then
      sleep_for=$((attempt * 5))
      echo "Pull failed. Retrying in ${sleep_for}s..."
      sleep "$sleep_for"
    fi
    attempt=$((attempt + 1))
  done

  echo "Failed to pull $label model after $OLLAMA_PULL_RETRIES attempts: $model"
  echo "Check Docker/container DNS and internet access, then run: docker compose up ollama"
  return 1
}

pull_model "chat" "$CHAT_MODEL"
pull_model "embedding" "$RAG_EMBEDDING_MODEL"

echo "Ollama is ready."
wait "$SERVE_PID"
