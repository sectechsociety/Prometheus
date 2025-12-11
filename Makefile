PYTHON ?= python3
PIP ?= pip
FRONTEND_DIR := frontend
BACKEND_DIR := backend

.PHONY: install lint lint-backend lint-frontend format format-frontend format-backend dev backend frontend test

install:
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	cd $(FRONTEND_DIR) && npm install

lint: lint-backend lint-frontend

lint-backend:
	$(PYTHON) -m ruff check backend/app services/ingest
	$(PYTHON) -m mypy backend/app services/ingest

lint-frontend:
	cd $(FRONTEND_DIR) && npm run lint

format: format-backend format-frontend

format-backend:
	$(PYTHON) -m ruff format backend/app services/ingest

format-frontend:
	cd $(FRONTEND_DIR) && npm run format

test:
	cd $(BACKEND_DIR) && pytest

backend:
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload --port 8000

frontend:
	cd $(FRONTEND_DIR) && npm run dev
