"""Ingest pipeline utilities

This script provides simple functions to:
- load raw HTML/text from files or URLs
- clean HTML and normalize text
- chunk text on sentence boundaries
- export JSONL with metadata for fine-tuning/RAG

Usage examples (local):
  python services/ingest/ingest.py --source-dir ./services/ingest/raw_html --out ./services/ingest/data/initial_dataset.jsonl

The script is intentionally small and dependency-light (beautifulsoup4 recommended).
"""

import argparse
import html
import json
import os
import re
from datetime import datetime
from uuid import uuid4

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


def read_file(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def clean_html(raw_html: str) -> str:
    """Strip tags, remove scripts/styles and return cleaned text."""
    if BeautifulSoup is None:
        # very small fallback: remove tags naively
        text = re.sub(r"<script.*?>.*?</script>", " ", raw_html, flags=re.S | re.I)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    soup = BeautifulSoup(raw_html, "lxml")
    for s in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        s.decompose()
    text = soup.get_text(separator=" ")
    text = html.unescape(text)
    # remove long whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip()
    # basic boilerplate removal
    text = re.sub(r"(advertisement|subscribe|read more|cookie)", "", text, flags=re.I)
    return text


def normalize_text(text: str) -> str:
    # collapse whitespace, tidy punctuation spacing
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Chunk on sentence boundaries up to max_chars (heuristic)."""
    if not text:
        return []
    # split on sentence enders
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    chunks = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) + 1 > max_chars:
            if cur:
                chunks.append(" ".join(cur).strip())
            cur = [s]
            cur_len = len(s)
        else:
            cur.append(s)
            cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks


def dedupe_and_filter(chunks: list[str], min_len: int = 40) -> list[str]:
    seen = set()
    out = []
    for c in chunks:
        key = c[:200]
        if key in seen:
            continue
        seen.add(key)
        if len(c) < min_len:
            continue
        out.append(c)
    return out


def make_item(
    chunk: str, source: str = "local", url: str | None = None, target_model: str = "ChatGPT"
) -> dict:
    return {
        "input_prompt": chunk,
        "enhanced_prompt": None,
        "source": source,
        "url": url,
        "chunk_id": uuid4().hex[:12],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "target_model": target_model,
        "tags": [],
    }


def export_jsonl(items: list[dict], outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as fh:
        for itm in items:
            fh.write(json.dumps(itm, ensure_ascii=False) + "\n")


def ingest_from_dir(
    source_dir: str, outpath: str, max_chars: int = 800, target_model: str = "ChatGPT"
):
    files = []
    for root, _, filenames in os.walk(source_dir):
        for fn in filenames:
            if fn.lower().endswith((".html", ".htm", ".txt")):
                files.append(os.path.join(root, fn))

    items = []
    for fp in files:
        raw = read_file(fp)
        cleaned = clean_html(raw) if fp.lower().endswith((".html", ".htm")) else raw
        cleaned = normalize_text(cleaned)
        chunks = chunk_text(cleaned, max_chars=max_chars)
        chunks = dedupe_and_filter(chunks)
        for c in chunks:
            itm = make_item(c, source=os.path.basename(fp), url=None, target_model=target_model)
            items.append(itm)

    export_jsonl(items, outpath)
    print(f"Wrote {len(items)} items to {outpath}")


def main():
    p = argparse.ArgumentParser(
        description="Simple ingest: clean HTML/text and export JSONL chunks"
    )
    p.add_argument("--source-dir", "-s", required=True, help="Directory with .html/.htm/.txt files")
    p.add_argument("--out", "-o", required=True, help="Output JSONL path")
    p.add_argument("--max-chars", type=int, default=800)
    p.add_argument("--target-model", default="ChatGPT")
    args = p.parse_args()
    ingest_from_dir(
        args.source_dir, args.out, max_chars=args.max_chars, target_model=args.target_model
    )


if __name__ == "__main__":
    main()
