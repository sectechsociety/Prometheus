import json
import os

from .vector_store import add_documents
from .prompt_classifier import classify_prompt

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "services",
    "ingest",
    "data",
    "all_guidelines.jsonl",
)


def load_items(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(obj)
    return out


def run(populate_all: bool = True, limit: int | None = None):
    items = load_items(DATA_PATH)
    if limit:
        items = items[:limit]

    texts = [it["input_prompt"] for it in items]
    metadatas = []

    for it in items:
        cls = classify_prompt(it.get("input_prompt", ""))
        metadatas.append(
            {
                "source": it.get("source"),
                "chunk_id": it.get("chunk_id"),
                "created_at": it.get("created_at"),
                "prompt_type": cls.prompt_type,
                "target_model": it.get("target_model"),  # kept for backward compatibility
            }
        )
    ids = [it.get("chunk_id") for it in items]

    print(f"Adding {len(texts)} documents to ChromaDB...")
    add_documents(texts, metadatas, ids=ids)
    print("Done")


if __name__ == "__main__":
    # default: populate all
    run()
