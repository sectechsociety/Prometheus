import os

from .embeddings import batch_generate_embeddings, generate_embedding

COLLECTION_NAME = "prometheus_guidelines"

# Prompt-type retrieval boost: prioritize guidelines matching detected prompt_type
MODEL_PRIORITY_BOOST = 0.2  # 20% boost for matching prompt_type


def init_client(persist: bool = True):
    # Lazy-import chromadb so environments without the dependency (e.g., CI unit tests) still import this module
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "chromadb is required for vector store operations. Install backend/requirements.txt "
            "in production or run tests with RAG disabled/mocked."
        ) from exc

    # Prefer explicit env override, support both CHROMA_DATA_PATH and CHROMA_DB_PATH for compatibility
    persist_dir = (
        os.getenv("CHROMA_DATA_PATH") or os.getenv("CHROMA_DB_PATH") or "services/ingest/chroma_db"
    )
    settings = Settings(persist_directory=persist_dir, is_persistent=True)
    client = chromadb.Client(settings=settings)
    return client


def get_or_create_collection(client):
    try:
        coll = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        coll = client.create_collection(name=COLLECTION_NAME)
    return coll


def add_documents(doc_texts: list[str], metadatas: list[dict], ids: list[str] | None = None):
    client = init_client()
    coll = get_or_create_collection(client)
    # generate embeddings in batch
    embs = batch_generate_embeddings(doc_texts)
    coll.add(documents=doc_texts, metadatas=metadatas, ids=ids, embeddings=embs)
    # persist if client supports it
    try:
        client.persist()
    except Exception:
        pass


def search(
    query: str,
    top_k: int = 5,
    prompt_type: str | None = None,
    prompt_type_only: bool = False,
) -> dict:
    """Search with prompt-type prioritization (no user-selected model needed)."""
    client = init_client()
    coll = get_or_create_collection(client)
    query_emb = generate_embedding(query)

    if prompt_type_only and prompt_type:
        # Strict filtering: only return guidelines for this prompt type
        filter_dict = {"prompt_type": prompt_type}
        results = coll.query(query_embeddings=[query_emb], n_results=top_k, where=filter_dict)
    else:
        # Soft filtering: retrieve more results, then re-rank with model priority
        fetch_k = min(top_k * 3, 50)  # Over-fetch for re-ranking, cap at 50
        results = coll.query(query_embeddings=[query_emb], n_results=fetch_k)

        if prompt_type:
            # Re-rank: boost scores for matching prompt_type
            results = _rerank_by_prompt_type(results, prompt_type, top_k)

    return results


def _rerank_by_prompt_type(results: dict, prompt_type: str, top_k: int) -> dict:
    """Re-rank results to prioritize guidelines matching the detected prompt_type."""
    if not results["ids"] or not results["ids"][0]:
        return results

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Calculate boosted scores
    scored_items = []
    for i, (id_, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances, strict=False)):
        base_score = 1 / (1 + dist)  # Convert distance to similarity

        # Boost if prompt_type matches
        if meta.get("prompt_type") == prompt_type:
            boosted_score = base_score * (1 + MODEL_PRIORITY_BOOST)
        else:
            boosted_score = base_score

        scored_items.append(
            {
                "id": id_,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": boosted_score,
                "original_rank": i,
            }
        )

    # Re-sort by boosted score (descending)
    scored_items.sort(key=lambda x: x["score"], reverse=True)

    # Take top_k and reconstruct results format
    top_items = scored_items[:top_k]

    return {
        "ids": [[item["id"] for item in top_items]],
        "documents": [[item["document"] for item in top_items]],
        "metadatas": [[item["metadata"] for item in top_items]],
        "distances": [[item["distance"] for item in top_items]],
        "scores": [[item["score"] for item in top_items]],  # Add explicit scores
    }


def get_model_statistics() -> dict[str, int]:
    """Return count of guidelines per prompt_type for monitoring."""
    client = init_client()
    coll = get_or_create_collection(client)

    all_docs = coll.get()
    stats: dict[str, int] = {"total": len(all_docs["ids"])}

    for meta in all_docs.get("metadatas", []):
        ptype = meta.get("prompt_type", "unknown")
        stats[ptype] = stats.get(ptype, 0) + 1

    return stats


class VectorStore:
    """Wrapper class for ChromaDB vector store operations."""

    def __init__(self):
        self.client = init_client()
        self.collection = get_or_create_collection(self.client)

    def get_collection(self):
        """Get the ChromaDB collection."""
        return self.collection

    def search(
        self, query: str, top_k: int = 5, prompt_type: str | None = None, prompt_type_only: bool = False
    ):
        """Search for similar documents by prompt type."""
        return search(query=query, top_k=top_k, prompt_type=prompt_type, prompt_type_only=prompt_type_only)

    def add_documents(
        self, doc_texts: list[str], metadatas: list[dict], ids: list[str] | None = None
    ):
        """Add documents to the collection."""
        return add_documents(doc_texts, metadatas, ids)


# Singleton instance
_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Get singleton vector store instance.

    Returns:
        VectorStore instance
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()

    return _vector_store_instance
