"""Provenance-resolution top-level aggregator.

Reports whether course-page citations ultimately resolve back to their book
source — a key evaluation metric for "did the generated course actually cite
the corpus it claims to teach from?".

The aggregator walks the LibV2 course's IMSCC chunkset
(``<libv2_course>/imscc_chunks/chunks.jsonl``). For each chunk it extracts the
``data-cf-source-ids="..."`` attribute values from the chunk's ``html`` field
(comma-separated ``dart:{stem}#{anchor}`` tokens). An **empty-string** attr is
the sanctioned Wave-27 boilerplate contract (a fail-closed re-rolled block
legitimately emits none) and counts as *no provenance* — never a resolution
failure.

Three ratios are surfaced:

* ``chunks_with_provenance`` — chunks carrying >=1 non-empty token / total
  chunks.
* ``source_ids_anchor_resolved`` — distinct tokens whose ``{anchor}`` exists as
  a ``data-dart-block-id`` attribute in the staged DART/vendor accessible HTML
  for ``{stem}`` / distinct tokens. Skipped (never fabricated as 0s) when no
  staging-HTML root resolves — the field becomes
  ``{"skipped": "no_staging_dir"}``.
* ``source_ids_book_chunk_resolved`` — distinct tokens carried by >=1 book-side
  dart chunk's ``source.source_references[].sourceId``
  (``<libv2_course>/dart_chunks/chunks.jsonl``) / distinct tokens.

Also emitted: the raw counts behind every ratio, capped (<=25) lists of
unresolved tokens, a per-``module_id`` breakdown (Counter) of provenance-free
chunks, ``course_slug`` / ``run_id`` / ``generated_at`` / ``staging_dir_used``,
and a ``notes`` field explaining the empty-provenance semantics.

Output: ``<libv2_course>/quality/provenance_resolution_report.json`` (quality/
is created if missing), with a ``<project_path>/provenance_resolution_report.json``
fallback when no LibV2 course dir resolves.

Cheap pure-regex/JSON work — no LLM, no embedding, no decision surface.
Best-effort posture: missing inputs degrade gracefully and
``WorkflowRunner`` wraps the call in try/except so an aggregator failure does
NOT change ``final_status``; the per-phase reports remain the source of truth.

Schema: :file:`schemas/aggregators/provenance_resolution.schema.json`
(Draft 2020-12, ``additionalProperties: false``).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# Cap on the unresolved-token sample lists emitted per ratio.
_UNRESOLVED_CAP = 25

_NOTES = (
    "data-cf-source-ids tokens have the shape dart:{stem}#{anchor}. An "
    "empty-string data-cf-source-ids attr is the sanctioned Wave-27 "
    "boilerplate contract (a fail-closed re-rolled block legitimately emits no "
    "provenance) and counts as no-provenance, never a resolution failure. "
    "anchor resolution checks {anchor} against data-dart-block-id in the "
    "staged {stem}.html; book-chunk resolution checks the token against the "
    "book-side dart_chunks source.source_references[].sourceId set."
)

# ``data-cf-source-ids="..."`` — capture the raw (possibly empty) attr value.
_ATTR_RE = re.compile(r'data-cf-source-ids="([^"]*)"')
# ``data-dart-block-id="..."`` in the staged accessible HTML.
_BLOCK_ID_RE = re.compile(r'data-dart-block-id="([^"]*)"')


def _split_token(token: str) -> Optional[Tuple[str, str]]:
    """Split a ``dart:{stem}#{anchor}`` token into ``(stem, anchor)``.

    Returns ``None`` when the token is not a well-formed dart source token.
    """
    if not token.startswith("dart:"):
        return None
    rest = token[len("dart:"):]
    stem, sep, anchor = rest.partition("#")
    if not sep or not stem or not anchor:
        return None
    return stem, anchor


class ProvenanceResolutionAggregator:
    """Report whether course-page citations resolve to their book source.

    Parameters
    ----------
    phase_outputs:
        ``WorkflowRunner.run_workflow``'s accumulated ``phase_outputs`` map.
        Read best-effort for the ``libv2_archival.course_dir`` and
        ``staging.staging_dir`` signals; never modified.
    course_slug:
        Operator-facing course slug so cross-run diffs key cleanly.
    run_id:
        Workflow run ID.
    libv2_course_path:
        Optional override for the LibV2 course directory
        (``LibV2/courses/<slug>``). When unset it resolves lazily from
        ``phase_outputs.libv2_archival.course_dir``.
    staging_dir:
        Optional override for the staged accessible-HTML root. When unset it
        resolves from ``phase_outputs.staging.staging_dir``. When neither
        resolves, the anchor-resolution metric is SKIPPED (not fabricated).
    """

    def __init__(
        self,
        phase_outputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        course_slug: str = "",
        run_id: str = "",
        libv2_course_path: Optional[Path] = None,
        staging_dir: Optional[Path] = None,
    ) -> None:
        self.phase_outputs = phase_outputs or {}
        self.course_slug = course_slug or ""
        self.run_id = run_id or ""
        self.libv2_course_path = (
            Path(libv2_course_path) if libv2_course_path else None
        )
        self.staging_dir = Path(staging_dir) if staging_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Optional[Dict[str, Any]]:
        """Build the canonical provenance-resolution report dict.

        Returns ``None`` (skip, no raise) when the IMSCC chunkset can't be
        located — there is nothing to report against.
        """
        chunks_path = self._resolve_imscc_chunks_path()
        if chunks_path is None or not chunks_path.exists():
            logger.info(
                "provenance_resolution: no imscc chunkset resolvable "
                "(course_slug=%s, run_id=%s); skipping aggregator",
                self.course_slug, self.run_id,
            )
            return None

        chunks = self._read_jsonl(chunks_path)

        total_chunks = 0
        chunks_with_prov = 0
        distinct_tokens: Set[str] = set()
        provenance_free_by_module: Counter = Counter()

        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            total_chunks += 1
            tokens = self._chunk_tokens(chunk)
            if tokens:
                chunks_with_prov += 1
                distinct_tokens.update(tokens)
            else:
                module_id = self._chunk_module_id(chunk)
                provenance_free_by_module[module_id] += 1

        sorted_tokens = sorted(distinct_tokens)

        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "course_slug": self.course_slug,
            "run_id": self.run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "staging_dir_used": None,
            "distinct_source_id_count": len(sorted_tokens),
            "chunks_with_provenance": self._ratio_block(
                chunks_with_prov, total_chunks
            ),
            "source_ids_anchor_resolved": self._anchor_resolution(sorted_tokens),
            "source_ids_book_chunk_resolved": self._book_chunk_resolution(
                sorted_tokens
            ),
            "provenance_free_by_module": dict(
                sorted(provenance_free_by_module.items())
            ),
            "notes": _NOTES,
        }

        staging_root = self._resolve_staging_dir()
        if staging_root is not None:
            report["staging_dir_used"] = str(staging_root)

        return report

    def write(self, output_path: Path) -> Optional[Path]:
        """Serialise :meth:`build` output to ``output_path`` (deterministic).

        Returns the resolved absolute path on success, or ``None`` when the
        build was skipped (no imscc chunkset). Raises ``OSError`` only on a
        filesystem failure — the workflow-runner caller wraps the call in
        try/except so an aggregator failure never aborts the workflow.
        """
        report = self.build()
        if report is None:
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    # ------------------------------------------------------------------
    # Metric builders
    # ------------------------------------------------------------------
    @staticmethod
    def _ratio_block(count: int, total: int) -> Dict[str, Any]:
        """Build a ``{count, total, ratio}`` block (ratio null when total==0)."""
        ratio = round(count / total, 4) if total else None
        return {"count": count, "total": total, "ratio": ratio}

    def _anchor_resolution(self, tokens: List[str]) -> Dict[str, Any]:
        """Resolve each distinct token's ``{anchor}`` against the staged HTML.

        Skipped (``{"skipped": "no_staging_dir"}``) when no staging-HTML root
        resolves — we never fabricate 0s for a metric we couldn't measure.
        """
        staging_root = self._resolve_staging_dir()
        if staging_root is None:
            return {"skipped": "no_staging_dir"}

        anchors_by_stem: Dict[str, Set[str]] = {}

        def _anchors(stem: str) -> Set[str]:
            if stem in anchors_by_stem:
                return anchors_by_stem[stem]
            html_path = staging_root / f"{stem}.html"
            found: Set[str] = set()
            try:
                text = html_path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            if text:
                found = set(_BLOCK_ID_RE.findall(text))
            anchors_by_stem[stem] = found
            return found

        resolved = 0
        unresolved: List[str] = []
        for token in tokens:
            parts = _split_token(token)
            if parts is not None and parts[1] in _anchors(parts[0]):
                resolved += 1
            else:
                unresolved.append(token)

        block = self._ratio_block(resolved, len(tokens))
        block["unresolved"] = unresolved[:_UNRESOLVED_CAP]
        return block

    def _book_chunk_resolution(self, tokens: List[str]) -> Dict[str, Any]:
        """Resolve each distinct token against the book-side dart-chunk set."""
        book_tokens = self._book_chunk_source_ids()
        resolved = 0
        unresolved: List[str] = []
        for token in tokens:
            if token in book_tokens:
                resolved += 1
            else:
                unresolved.append(token)
        block = self._ratio_block(resolved, len(tokens))
        block["unresolved"] = unresolved[:_UNRESOLVED_CAP]
        return block

    def _book_chunk_source_ids(self) -> Set[str]:
        """Collect every ``source.source_references[].sourceId`` from dart_chunks."""
        out: Set[str] = set()
        course_path = self._resolve_libv2_course_path()
        if course_path is None:
            return out
        dart_path = course_path / "dart_chunks" / "chunks.jsonl"
        if not dart_path.exists():
            return out
        for chunk in self._read_jsonl(dart_path):
            source = chunk.get("source") if isinstance(chunk, Mapping) else None
            refs = source.get("source_references") if isinstance(source, Mapping) else None
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if isinstance(ref, Mapping):
                    sid = ref.get("sourceId")
                    if isinstance(sid, str) and sid.strip():
                        out.add(sid.strip())
        return out

    # ------------------------------------------------------------------
    # Chunk-level extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_tokens(chunk: Mapping[str, Any]) -> List[str]:
        """Extract all non-empty dart source tokens from a chunk's html.

        An empty-string ``data-cf-source-ids`` attr contributes nothing (the
        Wave-27 boilerplate contract). Tokens are stripped + de-duped within
        the chunk (order-preserving).
        """
        html = chunk.get("html")
        if not isinstance(html, str) or not html:
            return []
        out: List[str] = []
        seen: Set[str] = set()
        for attr_val in _ATTR_RE.findall(html):
            for token in attr_val.split(","):
                token = token.strip()
                if token and token not in seen:
                    seen.add(token)
                    out.append(token)
        return out

    @staticmethod
    def _chunk_module_id(chunk: Mapping[str, Any]) -> str:
        """Resolve a chunk's ``source.module_id`` (falls back to 'unknown')."""
        source = chunk.get("source")
        if isinstance(source, Mapping):
            mid = source.get("module_id")
            if isinstance(mid, str) and mid.strip():
                return mid.strip()
        return "unknown"

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    def _resolve_libv2_course_path(self) -> Optional[Path]:
        """Resolve the LibV2 course dir from explicit override or phase output."""
        if self.libv2_course_path is not None:
            return self.libv2_course_path
        archival = self.phase_outputs.get("libv2_archival") or {}
        course_dir = archival.get("course_dir")
        if course_dir:
            return Path(course_dir)
        return None

    def _resolve_imscc_chunks_path(self) -> Optional[Path]:
        """Resolve ``<libv2_course>/imscc_chunks/chunks.jsonl``."""
        course_path = self._resolve_libv2_course_path()
        if course_path is None:
            return None
        return course_path / "imscc_chunks" / "chunks.jsonl"

    def _resolve_staging_dir(self) -> Optional[Path]:
        """Resolve the staged accessible-HTML root (explicit / phase output).

        Returns ``None`` (→ anchor metric skipped) when neither the explicit
        override nor ``phase_outputs.staging.staging_dir`` resolves to an
        extant directory.
        """
        candidate: Optional[Path] = None
        if self.staging_dir is not None:
            candidate = self.staging_dir
        else:
            staging = self.phase_outputs.get("staging") or {}
            staging_dir = staging.get("staging_dir")
            if staging_dir:
                candidate = Path(staging_dir)
        if candidate is not None and candidate.is_dir():
            return candidate
        return None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        """Best-effort JSONL read; skips malformed rows, returns the rest."""
        out: List[Dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("provenance_resolution: cannot read %s: %s", path, exc)
            return out
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out


__all__ = [
    "ProvenanceResolutionAggregator",
    "SCHEMA_VERSION",
]
