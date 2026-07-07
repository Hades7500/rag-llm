#!/usr/bin/env sh
set -eu

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ "${OPEN_WEBUI_APPIMAGE:-}" = "" ]; then
  echo "Set OPEN_WEBUI_APPIMAGE to your Open WebUI AppImage path."
  echo "Example: OPEN_WEBUI_APPIMAGE=~/Downloads/Open-WebUI.AppImage ./scripts/start_openwebui_appimage.sh"
  exit 1
fi

export DATA_DIR="${DATA_DIR:-./data/open-webui}"
mkdir -p "$DATA_DIR"

exec "$OPEN_WEBUI_APPIMAGE"
