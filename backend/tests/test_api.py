from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure backend package is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as main


class DummyModel:
    is_mock = True

    def enhance_prompt(
        self,
        raw_prompt: str,
        target_model: str = "ChatGPT",
        rag_context=None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ):
        # Return simple deterministic variations for testing
        return [f"enhanced:{raw_prompt}:{i}" for i in range(num_return_sequences)]


def _dummy_store():
    class _Collection:
        def count(self):
            return 10

    class _Store:
        def get_collection(self):
            return _Collection()

    return _Store()


@pytest.fixture()
def client(monkeypatch):
    # Ensure heavyweight deps are mocked before TestClient triggers startup events
    monkeypatch.setattr(main, "get_model", lambda: DummyModel(), raising=False)
    monkeypatch.setattr(main, "retrieve_context", lambda *_, **__: [], raising=False)
    monkeypatch.setattr(main, "format_context", lambda *_, **__: "", raising=False)
    monkeypatch.setattr(main, "get_vector_store", _dummy_store, raising=False)

    with TestClient(main.app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in {"healthy", "degraded", "unhealthy"}
    assert data["model"]["ready"] is True
    # RAG may be mocked or unavailable in lightweight test env; just ensure field exists
    assert "available" in data["rag"]


def test_augment_basic(client):
    payload = {
        "raw_prompt": "Explain machine learning",
        "target_model": "ChatGPT",
        "num_variations": 2,
        "use_rag": False,
    }
    resp = client.post("/augment", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_prompt"] == payload["raw_prompt"]
    assert len(data["enhanced_prompts"]) == 2
    assert data["model_type"] == "mock"
    assert data["rag_context_used"] is False


def test_augment_validates_model_name(client):
    payload = {"raw_prompt": "Test", "target_model": "InvalidModel"}
    resp = client.post("/augment", json=payload)
    # Pydantic validation should fail
    assert resp.status_code == 422
