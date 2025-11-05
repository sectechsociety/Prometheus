from typing import List

from sentence_transformers import SentenceTransformer

# Lazily load model to avoid heavy import on module import
_MODEL = None


def get_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    model = get_model(model_name)
    emb = model.encode(text, show_progress_bar=False)
    return emb.tolist() if hasattr(emb, 'tolist') else emb


def batch_generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    model = get_model(model_name)
    embs = model.encode(texts, show_progress_bar=True)
    # ensure list of lists
    return [e.tolist() if hasattr(e, 'tolist') else e for e in embs]
