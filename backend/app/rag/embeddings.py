# Lazily load model to avoid heavy import on module import
_MODEL: object | None = None


def get_model(name: str = "all-MiniLM-L6-v2"):
    """Load sentence-transformer model lazily.

    Import inside the function so tests can run without the heavy dependency installed.
    """
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install backend/requirements.txt in production environments."
            ) from exc

        _MODEL = SentenceTransformer(name)
    return _MODEL


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    model = get_model(model_name)
    emb = model.encode(text, show_progress_bar=False)
    return emb.tolist() if hasattr(emb, "tolist") else emb  # type: ignore


def batch_generate_embeddings(
    texts: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> list[list[float]]:
    model = get_model(model_name)
    embs = model.encode(texts, show_progress_bar=True)
    # ensure list of lists
    return [e.tolist() if hasattr(e, "tolist") else e for e in embs]
