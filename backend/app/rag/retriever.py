from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .vector_store import search


@dataclass
class RetrievedChunk:
    id: Optional[str]
    text: str
    score: Optional[float]
    distance: Optional[float]
    metadata: Dict[str, Any]


def _build_items(results: Dict[str, Any]) -> List[RetrievedChunk]:
    docs = (results or {}).get("documents", [[]])
    metas = (results or {}).get("metadatas", [[]])
    ids = (results or {}).get("ids", [[]])
    dists = (results or {}).get("distances", [[]])

    if not docs or not isinstance(docs, list):
        return []

    docs0 = docs[0] if len(docs) > 0 else []
    metas0 = metas[0] if len(metas) > 0 else []
    ids0 = ids[0] if len(ids) > 0 else []
    dists0 = dists[0] if len(dists) > 0 else []

    items: List[RetrievedChunk] = []
    for i, text in enumerate(docs0):
        meta = metas0[i] if i < len(metas0) else {}
        _id = ids0[i] if i < len(ids0) else None
        dist = dists0[i] if i < len(dists0) else None
        # Convert distance to a similarity-like score; keep bounded in (0,1]
        # Use 1/(1+distance) which monotonically decreases with distance
        if isinstance(dist, (int, float)):
            score = 1.0 / (1.0 + float(dist))
        else:
            score = None
        items.append(RetrievedChunk(id=_id, text=text, score=score, distance=dist, metadata=meta))
    return items


def retrieve_context(query: str, target_model: Optional[str] = None, top_k: int = 5) -> List[RetrievedChunk]:
    """
    Retrieve top-k relevant guideline chunks for a query.

    Contract:
    - Input: query (str), optional target_model filter, top_k (int)
    - Output: List[RetrievedChunk] ordered from most to least relevant
    - Error modes: returns empty list on no results or storage errors
    """
    res = search(query=query, top_k=top_k, target_model=target_model)
    return _build_items(res)


def format_context(items: List[RetrievedChunk], max_chars: int = 2000) -> str:
    """
    Build a compact context string to feed into generation. Truncates to max_chars.
    """
    lines: List[str] = []
    for idx, it in enumerate(items, start=1):
        src = it.metadata.get("source") if isinstance(it.metadata, dict) else None
        model = it.metadata.get("target_model") if isinstance(it.metadata, dict) else None
        head = f"[{idx}] src={src or 'unknown'} model={model or '-'} id={it.id or '-'}"
        lines.append(head)
        lines.append(it.text.strip())
        lines.append("")
    ctx = "\n".join(lines).strip()
    if len(ctx) > max_chars:
        ctx = ctx[: max_chars - 3].rstrip() + "..."
    return ctx


if __name__ == "__main__":
    # Simple CLI for local testing
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve relevant guideline chunks")
    parser.add_argument("--query", required=True, help="User query to retrieve context for")
    parser.add_argument("--target-model", default=None, help="Optional target model filter (e.g., ChatGPT)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--print-context", action="store_true", help="Print formatted context block")
    args = parser.parse_args()

    chunks = retrieve_context(args.query, target_model=args.target_model, top_k=args.top_k)
    print(f"Results: {len(chunks)} items (top_k={args.top_k}, target_model={args.target_model})")
    for i, ch in enumerate(chunks, start=1):
        print(f"#{i} id={ch.id} score={ch.score:.4f} dist={ch.distance:.4f} model={ch.metadata.get('target_model')} src={ch.metadata.get('source')}")
    if args.print_context:
        print("\n--- Context ---\n")
        print(format_context(chunks))
