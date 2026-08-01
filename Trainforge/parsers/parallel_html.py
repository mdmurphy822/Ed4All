"""
Parallel HTML parse worker

A process-pool worker that turns one staged accessible-HTML file into the
parsed page item the chunking phase feeds to ``Trainforge.chunker.chunk_content``.
The module exists so a pool can reference the callable **by qualified name** —
``spawn`` pickles a function as ``(module, qualname)``, so the worker cannot be
a closure, a bound method, or a lambda defined at the call site.

Design constraints, each load-bearing:

1. **Plain-string payload.** ``(html_path, staging_root_or_None)``. No ``Path``
   objects, no parser instance, no config mapping — a payload that carries only
   ``str`` and ``None`` pickles in a fixed handful of bytes and cannot drag
   parent state into a child.
2. **Fresh parser per call.** ``HTMLContentParser`` has no ``__init__`` and
   holds no instance state (see ``PARSER_IS_STATELESS`` below), so constructing
   one per call is free and removes any question of cross-file bleed inside a
   long-lived worker process.
3. **Typed error envelope — the worker never raises for a bad input file.**
   Under a pool, an exception raised in a worker is re-raised at the point the
   ordered ``map`` iterator is consumed, which aborts the iterator and discards
   every already-completed parse after that index. A per-file read/parse failure
   must therefore come back as *data*; the parent decides policy. Note that
   ``UnicodeDecodeError`` is a ``ValueError``, **not** an ``OSError`` — a font
   or image asset named ``*.html`` fails on decode, not on read.
4. **No ``raw_html`` in the returned envelope.** The parsed envelope is roughly
   a third of the pickled size of the same envelope carrying the source markup,
   and a whole staged corpus round-tripping through a pool draws that traffic
   twice (serialize + deserialize) from the same memory-bandwidth pool the GPU
   uses. The parent re-reads the source bytes from the (already page-cached)
   file to reconstruct the full item.
5. **The path is never resolved.** Staging is symlink-mode by default, and
   ``item_path`` is derived from ``relative_to(staging_root)`` guarded by
   ``is_relative_to`` on the *unresolved* path. Resolving would send every item
   through the bare-filename fallback and silently rewrite ``source.item_path``
   on every emitted chunk.

The worker reads no environment variables and emits no log records: a child
process does not inherit the parent's logging configuration, and a knob read in
a worker is a knob whose resolved value never appears in the phase log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Trainforge.parsers.html_content_parser import HTMLContentParser

#: ``HTMLContentParser`` declares no ``__init__`` and assigns no instance
#: attributes; every ``parse`` call builds its own extractor locally. That
#: undocumented invariant is what makes both the current single-instance reuse
#: and this module's construct-per-call safe, and it is pinned by
#: ``test_parallel_html_worker.py::test_parser_holds_no_instance_state``.
PARSER_IS_STATELESS = True

#: Item keys the worker populates. The chunking phase adds exactly the keys in
#: ``PARENT_SUPPLIED_ITEM_KEYS`` to reconstruct the full page item; together the
#: two tuples are the complete item shape the chunker walks.
WORKER_ITEM_KEYS: Tuple[str, ...] = (
    "item_id",
    "item_path",
    "title",
    "resource_type",
    "module_id",
    "module_title",
    "week_num",
    "word_count",
    "sections",
    "learning_objectives",
    "key_concepts",
    "interactive_components",
    "page_id",
    "misconceptions",
    "suggested_assessment_types",
    "courseforge_metadata",
    "objective_refs",
    "source_references",
)

#: Deliberately withheld from the envelope (constraint 4) and re-read from disk
#: by the parent.
PARENT_SUPPLIED_ITEM_KEYS: Tuple[str, ...] = ("raw_html",)

#: Exception classes that describe a *bad input file* rather than a defect in
#: this code. Only these become an error envelope. Anything else propagates so
#: the phase fails loudly instead of shrinking the corpus silently — deliberately
#: NOT a bare ``except Exception``.
_INPUT_FAILURES = (OSError, UnicodeDecodeError, ValueError)


def is_error(envelope: Dict[str, Any]) -> bool:
    """True when ``envelope`` reports a per-file failure rather than an item."""
    return envelope.get("error") is not None


def parse_html_path(payload: Tuple[str, Optional[str]]) -> Dict[str, Any]:
    """Read and parse one staged HTML file into a page item envelope.

    Args:
        payload: ``(html_path, staging_root)`` as plain strings. ``staging_root``
            is ``None`` when the caller has no staging directory, in which case
            ``item_path`` falls back to the bare filename.

    Returns:
        ``{"html_path": <str>, "item": {...}, "error": None}`` on success, or
        ``{"html_path": <str>, "item": None, "error": {...}}`` when the file
        could not be read or decoded. ``error`` carries ``type`` (the exception
        class name), ``message``, ``path`` and ``stage`` (``"read"`` or
        ``"parse"``) so the caller can register a named drop reason without
        re-classifying a string.

        The success envelope carries every key in ``WORKER_ITEM_KEYS`` and none
        of ``PARENT_SUPPLIED_ITEM_KEYS``.
    """
    html_path, staging_root = payload

    try:
        # No ``.resolve()`` anywhere on this path (constraint 5).
        html_text = Path(html_path).read_text(encoding="utf-8")
    except _INPUT_FAILURES as exc:
        return _error_envelope(html_path, exc, stage="read")

    try:
        parsed = HTMLContentParser().parse(html_text)
    except _INPUT_FAILURES as exc:
        return _error_envelope(html_path, exc, stage="parse")

    return {
        "html_path": html_path,
        "item": _build_item(parsed, html_path, staging_root),
        "error": None,
    }


def _build_item(
    parsed: Any,
    html_path: str,
    staging_root: Optional[str],
) -> Dict[str, Any]:
    """Project a ``ParsedHTMLModule`` into the phase's page-item shape.

    Mirrors the chunking phase's item construction exactly, minus ``raw_html``.
    ``sections`` and ``learning_objectives`` stay as the module-level
    ``ContentSection`` / ``LearningObjective`` dataclasses the chunker expects;
    both are picklable by qualified name.
    """
    path = Path(html_path)
    slug = path.stem.lower().replace(" ", "-")
    title = parsed.title or slug

    return {
        "item_id": slug,
        "item_path": _item_path(path, staging_root),
        "title": title,
        "resource_type": "page",
        "module_id": slug,
        "module_title": title,
        "week_num": 0,
        "word_count": parsed.word_count,
        "sections": parsed.sections,
        "learning_objectives": parsed.learning_objectives,
        "key_concepts": parsed.key_concepts,
        "interactive_components": parsed.interactive_components,
        "page_id": parsed.page_id,
        "misconceptions": parsed.misconceptions,
        "suggested_assessment_types": parsed.suggested_assessment_types,
        "courseforge_metadata": parsed.metadata.get("courseforge"),
        "objective_refs": parsed.objective_refs,
        "source_references": parsed.source_references,
    }


def _item_path(path: Path, staging_root: Optional[str]) -> str:
    """Staging-relative path, or the bare filename when it is not under staging.

    Both the containment test and the relative projection run against the
    UNRESOLVED path: staged files are symlinks into the converter's output tree,
    so a resolved path is never under the staging root and every item would
    collapse to ``path.name``.
    """
    if staging_root is None:
        return path.name
    staging_dir = Path(staging_root)
    if path.is_relative_to(staging_dir):
        return str(path.relative_to(staging_dir))
    return path.name


def _error_envelope(html_path: str, exc: BaseException, *, stage: str) -> Dict[str, Any]:
    """Build the typed failure envelope (constraint 3)."""
    return {
        "html_path": html_path,
        "item": None,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "path": html_path,
            "stage": stage,
        },
    }


__all__: List[str] = [
    "PARENT_SUPPLIED_ITEM_KEYS",
    "PARSER_IS_STATELESS",
    "WORKER_ITEM_KEYS",
    "is_error",
    "parse_html_path",
]
