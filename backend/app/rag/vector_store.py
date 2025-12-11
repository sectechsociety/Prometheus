import os

from .embeddings import batch_generate_embeddings, generate_embedding

COLLECTION_NAME = "prometheus_guidelines"

# Model-specific retrieval boost: prioritize exact model matches
MODEL_PRIORITY_BOOST = 0.2  # 20% boost for matching target_model


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
    query: str, top_k: int = 5, target_model: str | None = None, model_specific_only: bool = False
) -> dict:
    """
    Search with model-specific prioritization.

    Args:
        query: User's raw prompt
        top_k: Number of results to return
        target_model: Target model (ChatGPT, Gemini, Claude)
        model_specific_only: If True, only return guidelines for target_model

    Returns:
        Results dict with model-aware scoring
    """
    client = init_client()
    coll = get_or_create_collection(client)
    query_emb = generate_embedding(query)

    if model_specific_only and target_model:
        # Strict filtering: only return guidelines for this model
        filter_dict = {"target_model": target_model}
        results = coll.query(query_embeddings=[query_emb], n_results=top_k, where=filter_dict)
    else:
        # Soft filtering: retrieve more results, then re-rank with model priority
        fetch_k = min(top_k * 3, 50)  # Over-fetch for re-ranking, cap at 50
        results = coll.query(query_embeddings=[query_emb], n_results=fetch_k)

        if target_model:
            # Re-rank: boost scores for matching target_model
            results = _rerank_by_model(results, target_model, top_k)

    return results


def _rerank_by_model(results: dict, target_model: str, top_k: int) -> dict:
    """
    Re-rank results to prioritize target_model guidelines.

    Strategy:
    1. Convert distances to similarity scores
    2. Apply boost to matching models
    3. Re-sort and return top_k
    """
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

        # Boost if model matches
        if meta.get("target_model") == target_model:
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
    """Return count of guidelines per model for monitoring."""
    client = init_client()
    coll = get_or_create_collection(client)

    all_docs = coll.get()
    stats = {"ChatGPT": 0, "Gemini": 0, "Claude": 0, "total": len(all_docs["ids"])}

    for meta in all_docs["metadatas"]:
        model = meta.get("target_model", "unknown")
        if model in stats:
            stats[model] += 1

    return stats


class VectorStore:
    """Wrapper class for ChromaDB vector store operations."""

    def __init__(self):
        self.client = init_client()
        self.collection = get_or_create_collection(self.client)

    def get_collection(self):
        """Get the ChromaDB collection."""
        return self.collection

    def search(self, query: str, top_k: int = 5, target_model: str | None = None):
        """Search for similar documents."""
        return search(query, top_k, target_model)

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
