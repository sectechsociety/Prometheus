from dataclasses import dataclass
from typing import Any

from .vector_store import get_model_statistics, search


@dataclass
class RetrievedChunk:
    id: str | None
    text: str
    score: float | None
    distance: float | None
    metadata: dict[str, Any]


def _build_items(results: dict[str, Any]) -> list[RetrievedChunk]:
    docs = (results or {}).get("documents", [[]])
    metas = (results or {}).get("metadatas", [[]])
    ids = (results or {}).get("ids", [[]])
    dists = (results or {}).get("distances", [[]])
    scores = (results or {}).get("scores", [[]])  # From re-ranking

    if not docs or not isinstance(docs, list):
        return []

    docs0 = docs[0] if len(docs) > 0 else []
    metas0 = metas[0] if len(metas) > 0 else []
    ids0 = ids[0] if len(ids) > 0 else []
    dists0 = dists[0] if len(dists) > 0 else []
    scores0 = scores[0] if scores and len(scores) > 0 else []

    items: list[RetrievedChunk] = []
    for i, text in enumerate(docs0):
        meta = metas0[i] if i < len(metas0) else {}
        _id = ids0[i] if i < len(ids0) else None
        dist = dists0[i] if i < len(dists0) else None

        # Use explicit score from re-ranking if available, else compute
        if scores0 and i < len(scores0):
            score = scores0[i]
        elif isinstance(dist, (int, float)):
            score = 1.0 / (1.0 + float(dist))
        else:
            score = None

        items.append(RetrievedChunk(id=_id, text=text, score=score, distance=dist, metadata=meta))
    return items


def retrieve_context(
    query: str, prompt_type: str | None = None, top_k: int = 5, prompt_type_only: bool = False
) -> list[RetrievedChunk]:
    """
    Retrieve model-specific context for prompt enhancement.

    Args:
        query: User's raw prompt
        prompt_type: Detected prompt type (code, analysis, explain, creative, summarize, troubleshoot)
        top_k: Number of guidelines to retrieve
        prompt_type_only: If True, only use guidelines for prompt_type

    Returns:
        List of relevant guideline chunks with metadata

    Example:
        >>> chunks = retrieve_context(
        ...     "Explain quantum computing",
        ...     prompt_type="analysis",
        ...     top_k=5
        ... )
        >>> print(f"Retrieved {len(chunks)} type-prioritized guidelines")
    """
    res = search(
        query=query, top_k=top_k, prompt_type=prompt_type, prompt_type_only=prompt_type_only
    )
    return _build_items(res)


def format_context(
    items: list[RetrievedChunk], target_model: str | None = None, max_chars: int = 2000
) -> str:
    """
    Format retrieved guidelines into model-specific prompt context.

    Different models prefer different context formats:
    - ChatGPT: Conversational bullets with examples
    - Claude: XML-structured sections
    - Gemini: Concise numbered list
    """
    if not items:
        return ""

    # Model-specific formatting
    if target_model == "Claude":
        # Claude loves XML tags
        context_parts = ["<guidelines>"]
        for chunk in items:
            if len("\n".join(context_parts)) > max_chars:
                break
            src = chunk.metadata.get("source", "unknown")
            context_parts.append(f"<guideline source='{src}'>\n{chunk.text}\n</guideline>")
        context_parts.append("</guidelines>")
        return "\n".join(context_parts)

    elif target_model == "Gemini":
        # Gemini prefers concise, action-oriented
        context_parts = ["**Enhancement Guidelines:**"]
        for i, chunk in enumerate(items, 1):
            if len("\n".join(context_parts)) > max_chars:
                break
            # Extract key points (first sentence or up to 100 chars)
            summary = chunk.text.split(".")[0][:100]
            if not summary.endswith("."):
                summary += "..."
            context_parts.append(f"{i}. {summary}")
        return "\n".join(context_parts)

    else:  # ChatGPT or default
        # ChatGPT handles longer, example-rich context well
        context_parts = ["Here are relevant prompt engineering guidelines:\n"]
        for i, chunk in enumerate(items, 1):
            if len("\n".join(context_parts)) > max_chars:
                break
            src = chunk.metadata.get("source", "guidelines")
            context_parts.append(f"**Guideline {i}** (from {src}):\n{chunk.text}\n")
        return "\n".join(context_parts)


def get_retrieval_stats() -> dict[str, int]:
    """Get statistics about the knowledge base for monitoring."""
    return get_model_statistics()


if __name__ == "__main__":
    # CLI testing with model-specific examples
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve model-specific guideline chunks")
    parser.add_argument("--query", required=True, help="User query to retrieve context for")
    parser.add_argument(
        "--target-model", default=None, help="Target model (ChatGPT, Gemini, Claude)"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--print-context", action="store_true", help="Print formatted context block"
    )
    parser.add_argument(
        "--model-only", action="store_true", help="Only retrieve guidelines for target model"
    )
    args = parser.parse_args()

    print(f"\n🔍 Retrieving for: {args.query}")
    print(f"🎯 Target Model: {args.target_model or 'All'}")
    print(f"📊 Top-K: {args.top_k}")
    print(f"🎚️  Model-only filter: {args.model_only}\n")

    chunks = retrieve_context(
        args.query,
        target_model=args.target_model,
        top_k=args.top_k,
        model_specific_only=args.model_only,
    )

    print(f"✅ Retrieved {len(chunks)} chunks\n")
    for i, ch in enumerate(chunks, start=1):
        model = ch.metadata.get("target_model")
        src = ch.metadata.get("source", "unknown")[:40]
        print(f"#{i} score={ch.score:.4f} dist={ch.distance:.4f} model={model} src={src}...")

    if args.print_context:
        print("\n📝 Formatted Context:\n")
        context = format_context(chunks, args.target_model)
        print(context)

    print("\n📊 Knowledge Base Stats:")
    print(get_retrieval_stats())
