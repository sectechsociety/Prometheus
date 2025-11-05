import json
import os
from typing import List

from .vector_store import add_documents, init_client, get_or_create_collection


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'services', 'ingest', 'data', 'all_guidelines.jsonl')


def load_items(path: str) -> List[dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(obj)
    return out


def run(populate_all: bool = True, limit: int = None):
    items = load_items(DATA_PATH)
    if limit:
        items = items[:limit]

    texts = [it['input_prompt'] for it in items]
    metadatas = [{
        'source': it.get('source'),
        'chunk_id': it.get('chunk_id'),
        'created_at': it.get('created_at'),
        'target_model': it.get('target_model')
    } for it in items]
    ids = [it.get('chunk_id') for it in items]

    print(f'Adding {len(texts)} documents to ChromaDB...')
    add_documents(texts, metadatas, ids=ids)
    print('Done')


if __name__ == '__main__':
    # default: populate all
    run()
