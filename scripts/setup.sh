#!/usr/bin/env sh
set -eu

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit WEBUI_SECRET_KEY before production use."
fi

echo "Setup complete. Activate with: . .venv/bin/activate"
