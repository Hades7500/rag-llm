"""ChromaDB HTTP-client helpers."""

import chromadb


def get_client(host: str = "127.0.0.1", port: int = 8000) -> chromadb.HttpClient:
    return chromadb.HttpClient(host=host, port=port)


def list_collections(client: chromadb.HttpClient) -> list[str]:
    return sorted(col.name for col in client.list_collections())


def get_or_create_collection(client, name: str, embed_fn):
    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
    )


def delete_collection(client, name: str) -> None:
    client.delete_collection(name=name)


def collection_count(col) -> int:
    return col.count()


def upsert_documents(
    col,
    texts: list[str],
    ids: list[str],
    metadatas: list[dict],
    embeddings: list | None = None,
    batch_size: int = 500,
) -> None:
    """Upsert chunks into ChromaDB.

    Pass pre-computed ``embeddings`` to skip calling the embedding model again.
    ``batch_size`` caps each individual upsert call to avoid very large payloads
    to the ChromaDB HTTP server; it no longer controls Ollama batch size.
    """
    for i in range(0, len(texts), batch_size):
        sl = slice(i, i + batch_size)
        kwargs: dict = {
            "documents": texts[sl],
            "ids": ids[sl],
            "metadatas": metadatas[sl],
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings[sl]
        col.upsert(**kwargs)


def query_collection(
    col,
    query: str,
    n_results: int = 3,
    score_threshold: float = 1.0,
) -> tuple[list[str], list[float]]:
    """Return (chunks, distances) for the top-n most relevant results.

    Only chunks whose L2 distance is <= score_threshold are returned.
    Lower distance = more similar.  Typical range for nomic-embed-text:
      0.0 – 0.5  very similar
      0.5 – 1.0  related
      1.0 – 1.5  loosely related
      > 1.5      probably irrelevant
    """
    total = col.count()
    if total == 0:
        return [], []
    results = col.query(
        query_texts=[query],
        n_results=min(n_results, total),
        include=["documents", "distances"],
    )
    docs = results.get("documents", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []
    filtered = [
        (doc, dist) for doc, dist in zip(docs, dists) if dist <= score_threshold
    ]
    if not filtered:
        return [], []
    out_docs, out_dists = zip(*filtered)
    return list(out_docs), list(out_dists)
