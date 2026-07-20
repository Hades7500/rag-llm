FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SKIP_VENV=1 \
    SKIP_OLLAMA=1 \
    SKIP_CHROMA=1 \
    FOREGROUND=1 \
    STREAMLIT_BIN=streamlit \
    STREAMLIT_ADDRESS=0.0.0.0 \
    STREAMLIT_PORT=8501

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/
COPY scripts/start_all.sh ./scripts/start_all.sh
RUN chmod +x ./scripts/start_all.sh

EXPOSE 8501

CMD ["./scripts/start_all.sh"]
