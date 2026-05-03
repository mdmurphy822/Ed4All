# ed4all-chunker

Canonical chunker for the Ed4All pipeline (DART + IMSCC). Produces v4
chunk envelopes consumed by Trainforge synthesis, the LibV2 RAG corpus,
and downstream SLM training surfaces.

## Status

**WIP — chunker functions land in Subtasks 2-4.** This Phase 7a
Subtask 1 commit ships only the package skeleton (`pyproject.toml`,
empty `__init__.py`, empty test scaffolding). The boilerplate
detector lifts in Subtask 2, helper functions in Subtask 3, and the
canonical `chunk_content` / `chunk_text_block` entry points in
Subtask 4. The smoke test suite lands in Subtask 5.

## Usage (post-Subtask-4)

```python
from ed4all_chunker import chunk_content

chunks = chunk_content(
    parsed_items=[...],     # DART or IMSCC parsed-page list
    course_code="PHYS_101",
    boilerplate_spans=[...],
)
```

## Development

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

## Reference

- Plan: `plans/phase7_chunker_dual_chunkset.md`
- Source modules being lifted:
  - `Trainforge/rag/boilerplate_detector.py` → `ed4all_chunker/boilerplate.py`
  - `Trainforge/process_course.py` chunking helpers → `ed4all_chunker/helpers.py`
    + `ed4all_chunker/chunker.py`
