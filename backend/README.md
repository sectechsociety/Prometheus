FastAPI backend for Prometheus

Structure:
- app/
  - main.py
  - rag/
    - embeddings.py (Sentence-Transformer wrapper)
    - vector_store.py (ChromaDB initialization, add, search)
    - retriever.py (retrieve_context + format_context)
  - api/
  - core/
- requirements.txt
- Dockerfile

Retrieval behavior (RAG)
- Vector store: ChromaDB persisted at services/ingest/chroma_db with collection name prometheus_guidelines
- Embeddings: sentence-transformers all-MiniLM-L6-v2 (CPU)
- Filtering: You can filter by metadata target_model (ChatGPT, Gemini, Claude)
- API surface:
  - vector_store.search(query: str, top_k: int = 5, target_model: Optional[str] = None) -> dict
  - retriever.retrieve_context(query: str, target_model: Optional[str], top_k: int = 5) -> List[RetrievedChunk]
  - retriever.format_context(items: List[RetrievedChunk], max_chars: int = 2000) -> str

Tuning guidance
- Typical top_k: 5 is a good default. Try 3 for tighter context windows; 10 if the query is broad or noisy.
- Score interpretation: score ≈ 1 - distance (higher is better). Distances come from the vector DB query.

Quick test (CLI)
Run retrieval locally (assuming DB populated via python -m backend.app.rag.populate_db):

```bash
python -m backend.app.rag.retriever --query "Explain machine learning" --top-k 5 --print-context

python -m backend.app.rag.retriever --query "Summarize a research paper" --top-k 5 --print-context

# With model filter
python -m backend.app.rag.retriever --query "Summarize a research paper" --target-model ChatGPT --top-k 5
```

Integration notes
- The /augment endpoint will call retrieve_context(...) to assemble a short guidelines block, then pass it with the raw prompt to the generator.
- Keep retrieval calls asynchronous if you add networked components; the current Chroma client runs locally/in-process.
