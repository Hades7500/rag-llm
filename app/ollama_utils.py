import json
import os
from typing import Generator

import requests
from chromadb import Documents, EmbeddingFunction, Embeddings


def get_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def is_embed_model(name: str) -> bool:
    """Heuristic: models whose name contains 'embed' are embedding-only."""
    return "embed" in name.lower()


class OllamaEmbedding(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function backed by Ollama /api/embed."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def __call__(self, input: Documents) -> Embeddings:
        resp = requests.post(
            f"{self._base_url}/api/embed",
            json={
                "model": self._model,
                "input": list(input),
                "options": {"num_thread": os.cpu_count() or 4},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# Seconds to wait for Ollama to establish a connection.
_CONNECT_TIMEOUT = 10
# Seconds allowed between any two received bytes while streaming.
# Large context or slow models can take a long time to produce the first token.
_READ_TIMEOUT = 3000


def stream_chat(
    model: str,
    messages: list[dict],
    system: str = "",
    base_url: str = "http://127.0.0.1:11434",
) -> Generator[str, None, None]:
    """Stream chat completion from Ollama. Yields text chunks."""
    payload_messages = (
        [{"role": "system", "content": system}] + messages
        if system.strip()
        else list(messages)
    )
    payload = {"model": model, "messages": payload_messages, "stream": True}

    with requests.post(
        f"{base_url}/api/chat",
        json=payload,
        stream=True,
        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done"):
                break
