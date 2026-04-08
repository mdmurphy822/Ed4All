# Ed4All

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Unified orchestration for accessible educational content generation.**

## Components

| Component | Purpose |
|-----------|---------|
| **DART** | PDF to accessible HTML (WCAG 2.2 AA) |
| **Courseforge** | Course generation & IMSCC packaging |
| **Trainforge** | Assessment-based RAG training |
| **LibV2** | Educational content repository |
| **MCP Server** | Unified tool orchestration |

## Quick Start

### Prerequisites

- Python 3.9+
- (Optional) Tesseract OCR for PDF processing
- (Optional) poppler-utils for PDF extraction

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Ed4All.git
cd Ed4All

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[full]"
```

### Start MCP Server

```bash
cd MCP && python server.py
```

### Run Pipeline

```bash
ed4all textbook-to-course textbook.pdf -n COURSE_101
```

## Workflows

| Workflow | Description |
|----------|-------------|
| `textbook_to_course` | Full pipeline: PDF -> Course -> Assessments |
| `course_generation` | Generate course from objectives |
| `intake_remediation` | Import and fix existing IMSCC |
| `batch_dart` | Batch PDF conversion |

## Project Structure

```
Ed4All/
├── DART/                    # PDF to accessible HTML conversion
├── Courseforge/             # Course content generation & packaging
├── Trainforge/              # Assessment-based RAG training
├── LibV2/                   # Course content repository
├── MCP/                     # FastMCP server and tools
├── orchestrator/            # Multi-terminal coordination
├── lib/                     # Shared libraries
├── config/                  # Workflow & agent configs
├── schemas/                 # JSON schemas for validation
├── state/                   # Shared state & progress tracking
└── training-captures/       # Decision capture output
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run unit tests only
pytest -m unit

# Run specific component tests
pytest Courseforge/scripts/tests/
```

## Documentation

- [DART](DART/CLAUDE.md) - PDF conversion
- [Courseforge](Courseforge/CLAUDE.md) - Course generation
- [Trainforge](Trainforge/CLAUDE.md) - Assessment training
- [LibV2](LibV2/CLAUDE.md) - Content repository
- [Orchestrator Protocol](CLAUDE.md) - Main orchestration guide

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE)
