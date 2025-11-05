from typing import List, Optional
import chromadb
from chromadb.config import Settings

from .embeddings import generate_embedding, batch_generate_embeddings

# Use default client (in-memory/persistent per chroma configuration). For local dev
# this will work; for production you can adapt to a persistent client per Chroma docs.
COLLECTION_NAME = "prometheus_guidelines"


def init_client(persist: bool = True):
    # Use persistent directory so data survives across processes during development
    persist_dir = "services/ingest/chroma_db"
    settings = Settings(persist_directory=persist_dir, is_persistent=True)
    client = chromadb.Client(settings=settings)
    return client


def get_or_create_collection(client):
    try:
        coll = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        coll = client.create_collection(name=COLLECTION_NAME)
    return coll


def add_documents(doc_texts: List[str], metadatas: List[dict], ids: Optional[List[str]] = None):
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


def search(query: str, top_k: int = 5, target_model: Optional[str] = None):
    client = init_client()
    coll = get_or_create_collection(client)
    query_emb = generate_embedding(query)
    # optional filter
    filter = {"target_model": target_model} if target_model else None
    results = coll.query(query_embeddings=[query_emb], n_results=top_k, where=filter)
    return results
