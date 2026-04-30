"""Chunk-ID → human-readable label resolver.

Audit 2026-04-30 found that eval probes in `faithfulness.py`,
`holdout_builder.py`, and `invariants.py` interpolated raw chunk-IDs
(`shacl_551_chunk_NNNNN`) into question text. The model can't
semantically reason about an opaque ID, so it echoes the literal back
into its answer (1441 chunk-id token matches in the cc07cc76 eval
report). The faithfulness classifier then scores those echoes as
ambiguous → 0/22 correct → faithfulness=0 on the adapter+RAG setup.

This module owns the single mapping from `chunk_id` to a clean label
the model can reason about (the chunk's `summary`, or the first ~80
characters of its `text` as a fallback). Probe templates substitute
the label in place of the raw ID so the question is semantically
answerable.

Usage:

    resolver = ChunkLabelResolver.from_course(course_path)
    label = resolver.label_for("rdf_shacl_551_chunk_00270")
    # → "Validating SHACL property shapes against RDF data"
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


_CHUNK_ID_PATTERN = re.compile(r"\bchunk_\d+\b|\b\w*_chunk_\d+\b")
_LABEL_MAX_CHARS = 80
_LABEL_FALLBACK = "an unnamed chunk"

_CORPUS_CANDIDATES = (
    "corpus/chunks.jsonl",
    "chunks.jsonl",
)


@dataclass
class ChunkLabelResolver:
    """Map `chunk_id` → human-readable label.

    Constructed once per eval run; the underlying dict is small
    (one entry per chunk, ~hundreds of entries even on large
    corpora) so memory is not a concern.
    """

    labels: Dict[str, str]

    @classmethod
    def from_course(cls, course_path: Path) -> "ChunkLabelResolver":
        """Load chunks.jsonl and build the chunk_id → label map.

        Falls back to an empty resolver (label_for returns the fallback
        string) when the corpus file is missing — eval still runs but
        probes carry the generic "an unnamed chunk" label.
        """
        course_path = Path(course_path)
        for candidate in _CORPUS_CANDIDATES:
            path = course_path / candidate
            if path.exists():
                return cls.from_jsonl(path)
        logger.warning(
            "ChunkLabelResolver: no chunks.jsonl found under %s; "
            "probes will use the generic fallback label.",
            course_path,
        )
        return cls(labels={})

    @classmethod
    def from_jsonl(cls, path: Path) -> "ChunkLabelResolver":
        labels: Dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk_id = record.get("id") or record.get("chunk_id")
                    if not chunk_id:
                        continue
                    labels[chunk_id] = _derive_label(record)
        except OSError as exc:
            logger.warning(
                "ChunkLabelResolver: failed to read %s (%s); "
                "probes will use the generic fallback label.",
                path, exc,
            )
        return cls(labels=labels)

    def label_for(self, chunk_id: str) -> str:
        """Return the chunk's label, or a generic fallback when unknown.

        The fallback is intentionally bland — never the chunk_id —
        because returning the ID defeats the entire purpose of the
        resolver.
        """
        if not chunk_id:
            return _LABEL_FALLBACK
        return self.labels.get(chunk_id, _LABEL_FALLBACK)

    def is_chunk_id(self, value: str) -> bool:
        """True when ``value`` looks like a chunk-ID literal.

        The probe templates substitute graph-edge sources/targets,
        which may be chunk-IDs OR concept-IDs (e.g. ``CO-18``). Only
        chunk-IDs need scrubbing.
        """
        if not isinstance(value, str):
            return False
        return bool(_CHUNK_ID_PATTERN.search(value))

    def scrub(self, value: str) -> str:
        """Replace ``value`` with its label IFF it's a chunk-ID;
        otherwise return ``value`` unchanged.

        Use this in probe templates instead of raw substitution so
        non-chunk references (concept IDs, learning outcomes) flow
        through unmodified.
        """
        if self.is_chunk_id(value):
            return self.label_for(value)
        return value


def _derive_label(record: Dict[str, object]) -> str:
    """Pick the cleanest available label from a chunk record.

    Priority: ``summary`` > first sentence of ``text`` > truncated
    ``text``. Empty/None fields fall through. Always returns a
    non-empty string (falls back to the generic placeholder).
    """
    summary = record.get("summary")
    if isinstance(summary, str) and summary.strip():
        return _truncate(summary.strip())

    text = record.get("text")
    if isinstance(text, str) and text.strip():
        # First sentence (period / question / exclamation), or whole text
        # if no sentence break in the first _LABEL_MAX_CHARS.
        first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
        return _truncate(first)

    return _LABEL_FALLBACK


def _truncate(s: str) -> str:
    """Cap at _LABEL_MAX_CHARS with an ellipsis."""
    s = " ".join(s.split())  # normalize whitespace
    if len(s) <= _LABEL_MAX_CHARS:
        return s
    return s[: _LABEL_MAX_CHARS - 1].rstrip() + "…"


__all__ = ["ChunkLabelResolver"]
