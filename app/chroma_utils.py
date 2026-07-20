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
    min_results: int = 1,
) -> tuple[list[str], list[float]]:
    """Return (chunks, distances) for the top-n most relevant results.

    Always keep at least ``min_results`` top matches. Chroma distance scales can
    vary across embedding models and versions, so using the threshold as a hard
    gate can hide facts that are clearly present in small PDFs.
    """
    total = col.count()
    if total == 0:
        return [], []
    n = min(max(n_results, min_results), total)
    results = col.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "distances"],
    )
    docs = results.get("documents", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []
    ranked = list(zip(docs, dists))
    if not ranked:
        return [], []

    keep = ranked[:min_results]
    keep.extend(
        (doc, dist)
        for doc, dist in ranked[min_results:]
        if dist <= score_threshold
    )
    out_docs, out_dists = zip(*keep)
    return list(out_docs), list(out_dists)
