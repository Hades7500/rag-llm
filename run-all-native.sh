#!/usr/bin/env sh
set -eu

cat <<'MSG'
Start the full local RAG stack with one command:

  sh scripts/start_all.sh

That script starts/checks:
  - Ollama
  - required Ollama models
  - ChromaDB
  - Open WebUI AppImage

Before running it, set OPEN_WEBUI_APPIMAGE in .env:

  OPEN_WEBUI_APPIMAGE=/home/abhinav/Downloads/Open-WebUI.AppImage

Open WebUI is usually available at:
  http://127.0.0.1:8080
MSG
