## Project Prometheus — Agent instructions

This file gives concise, actionable information an AI coding agent needs to be productive in this repository.

Quick summary
- Monorepo with three main pieces: `backend/` (FastAPI prototype), `frontend/` (Vite + React), and `services/ingest/` (ingestion pipeline placeholder).
- Development commonly uses Docker Compose (`docker-compose.yml`) which mounts source into containers for live-editing.

How to run (developer-first)
- Docker Compose (recommended for full stack): `docker-compose up --build` — exposes backend on port 8000 and frontend on 5173. There's a VS Code task named "docker-compose up" that runs this command.
- Backend only (fast iterate):
  - cd `backend`
  - Create a venv and install: `.venv/bin/python -m pip install -r requirements.txt`
  - Run the dev server: `.venv/bin/uvicorn app.main:app --reload --port 8000`
- Frontend only:
  - cd `frontend`
  - `npm install` then `npm run dev` (Vite dev server on port 5173)

Key integration points & where to hook features
- `backend/app/main.py` — current prototype: defines the POST `/augment` endpoint and Pydantic models `AugmentRequest` and `AugmentResponse`. When implementing RAG or model calls, this is the primary place to integrate.
  - Request shape: `{ "raw_prompt": "...", "target_model": "model-name" }`
  - Response shape: `{ "enhanced_prompts": ["..."] }`
- `services/ingest/ingest.py` — placeholder for ingestion pipeline. Use this to implement knowledge ingestion and vector indexing for the RAG pipeline.
- External pieces expected but currently not implemented: Vector DB (e.g., Milvus/Weaviate/Chroma) and the fine-tuned model / model-serving endpoint. Keep those integrations behind clear adapter classes so tests/mocks can replace them.

Project-specific conventions & patterns
- Minimal prototype style: many modules are single-file prototypes (e.g., `backend/app/main.py`). New functionality should follow the existing structure: small, explicit modules under `backend/app/` (add `api/`, `core/` subpackages if needed).
- Use Pydantic request/response models for public endpoints (the project already returns `response_model` for `/augment`). Prefer explicit types in new endpoints.
- Docker Compose mounts local directories into containers for live reload. Expect that changes to local files will be visible inside containers.

Files to inspect first (most helpful)
- `README.md` — high-level architecture and workflow.
- `backend/README.md` — backend notes and structure.
- `backend/app/main.py` — main API endpoint to extend.
- `backend/requirements.txt` — python deps for the backend.
- `frontend/package.json` and `frontend/src/` — frontend dev/build commands and UI entry points.
- `docker-compose.yml` — compose configuration and service ports/volumes.

Practical examples for common tasks
- Add a new endpoint to support a different transform:
  1. Create a new Pydantic model in `backend/app/schemas.py` (or next to `main.py`).
  2. Implement the handler in `backend/app/main.py` and include `response_model=...`.
  3. If needed, add local config to `docker-compose.yml` and update `backend/README.md`.
- Wire RAG retrieval:
  - Implement ingestion to populate vectors in `services/ingest/` and add a `retriever` adapter used by the backend.
  - Keep retrieval calls asynchronous and isolated behind an interface (so the endpoint remains testable).

Developer workflows & commands to use now
- Full stack (rebuild images): `docker-compose up --build`
- Backend dev without Docker: see Backend only section above (`uvicorn ... --reload`).
- Frontend dev: `cd frontend && npm install && npm run dev`

Notes & gotchas
- The repository is a prototype; many TODOs exist (see `backend/app/main.py` where RAG + fine-tuned model are TODO). Prefer small incremental changes.
- No test framework is present — if you add tests, include instructions in `backend/README.md` or a new `tests/` directory.

If anything here is unclear or you'd like more examples (small PR layout, preferred testing approach, or a suggested adapter shape for the Vector DB), tell me which area to expand and I will iterate.
