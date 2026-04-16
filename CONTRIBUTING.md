# Contributing to Ed4All

## Development Setup

```bash
git clone <repo-url>
cd Ed4All
python -m venv venv
source venv/bin/activate
pip install -e ".[full]"
```

### Optional System Dependencies (for DART)

DART's PDF conversion pipeline uses external tools. These are only needed if you work on DART:

```bash
# Tesseract OCR — used for OCR-based PDF text extraction
sudo apt install tesseract-ocr

# poppler-utils — provides pdftotext, pdfinfo, etc.
sudo apt install poppler-utils
```

## Running Tests

```bash
# All tests
pytest

# Single component
pytest Trainforge/tests/ -v
pytest MCP/tests/ -v

# With coverage
pytest --cov

# Integration tests only
pytest -m integration
```

## Code Style

Lint with ruff:

```bash
ruff check .
```

No auto-formatter is enforced. Just pass `ruff check`.

## Commit Messages

Use imperative mood in the subject line:

- **Good:** `Add batch retry logic for Trainforge`
- **Bad:** `Added batch retry logic for Trainforge`

Reference issues with `#N` when applicable:

```
Fix assessment validator crash on empty input (#42)
```

Keep the subject under 72 characters. Use the body for context if the change isn't obvious.

## Pull Request Process

1. Branch from `main`. Name your branch descriptively (e.g., `fix/trainforge-bloom-validation`).
2. Make sure all tests pass locally before opening a PR.
3. In the PR body, describe **what** changed and **why**. Link related issues.
4. One approval required to merge. Squash-merge preferred for single-purpose branches.

## Project Structure

```
Ed4All/
├── DART/           # PDF to accessible HTML conversion
├── Courseforge/     # Course content generation and IMSCC packaging
├── Trainforge/     # Assessment generation via RAG training
├── LibV2/          # Course content repository and retrieval engine
├── MCP/            # FastMCP server exposing tool endpoints
├── orchestrator/   # Workflow execution and agent coordination
├── cli/            # CLI entry point (ed4all command)
├── lib/            # Shared libraries, validators, decision capture
├── config/         # Workflow and agent configuration (YAML)
├── schemas/        # JSON schemas for validation
├── state/          # Runtime state and progress tracking
└── ci/             # CI integrity checks
```

## Documentation

Each component maintains its own `CLAUDE.md` with component-specific guidance:

- `DART/CLAUDE.md` — conversion pipeline, WCAG requirements
- `Courseforge/CLAUDE.md` — content generation, IMSCC packaging
- `Trainforge/CLAUDE.md` — assessment generation, Bloom's alignment
- `LibV2/CLAUDE.md` — repository structure, retrieval engine

The root `CLAUDE.md` covers the orchestration protocol, MCP tools, workflow definitions, and cross-component coordination.

## Decision Capture

If your change involves AI-driven decisions (content generation, assessment creation, remediation choices), log decisions to `training-captures/` using `lib.decision_capture.DecisionCapture`. Every decision needs a rationale of at least 20 characters. See `CLAUDE.md` for the full protocol.

## Adding a New MCP Tool

1. Create the tool function in the appropriate module under `MCP/tools/`.
2. Register it in `MCP/server.py`.
3. Add tests in `MCP/tests/`.
4. Document the tool in the root `CLAUDE.md` tool reference table.

## Adding a New Workflow

1. Define phases and concurrency limits in `config/workflows.yaml`.
2. Register agents in `config/agents.yaml`.
3. Implement phase handlers in `orchestrator/`.
4. Add validation gates if the workflow produces artifacts that need quality checks.
