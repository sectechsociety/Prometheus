# Changelog

## [0.1.0] - 2025-12-11

### Added
- **Professional Standards**: Implemented comprehensive linting and formatting.
  - **Backend**: Integrated `ruff` (linting/formatting) and `mypy` (static type checking).
  - **Frontend**: Verified `eslint` and `prettier` configurations.
- **CI/CD**: Added GitHub Actions workflow (`.github/workflows/ci.yml`) to run tests, linting, and type checking on every push.
- **Prompt Enhancement**: Improved the `/augment` endpoint with model-specific strategies (ChatGPT, Claude, Gemini).
- **Developer Experience**:
  - Added `Makefile` for common tasks (`make lint`, `make format`, `make test`).
  - Added `backend/requirements-dev.txt` for development dependencies.
  - Configured `pre-commit` hooks.

### Fixed
- **CORS Issue**: Fixed backend to allow requests from frontend on port 5174.
- **Syntax Errors**: Fixed f-string backslash issues for Python < 3.12 compatibility.
- **Type Errors**: Resolved various type mismatches in the backend.
