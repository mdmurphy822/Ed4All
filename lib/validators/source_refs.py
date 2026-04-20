"""PageSourceRefValidator — Wave 9 source-reference integrity gate.

Wave 9 lands per-page source attribution in Courseforge: pages optionally
carry ``sourceReferences[]`` in their JSON-LD block and ``data-cf-source-ids``
attributes in HTML. This validator cross-checks every emitted ``sourceId``
against the staging manifest (Wave 8) so bad routing or hallucinated IDs
cannot silently pass packaging into Trainforge / LibV2.

Critical-severity gate: emitted ``sourceId`` that does not resolve against
the staging manifest / provenance sidecars fails the gate. Graceful
fallback: when ``source_module_map.json`` is empty or absent, no ``sourceId``
values are expected and the validator passes clean (empty input = no emit,
no check needed).

Referenced by: ``config/workflows.yaml`` →
``textbook_to_course.content_generation.validation_gates[source_refs]``
(Wave 9). This is the emit-side counterpart to Wave 8's DART block-ID
minting — the ``sourceId`` pattern is validated by the
``SourceReference`` schema itself; this validator catches the orthogonal
"ID doesn't exist in staging" failure mode.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult


# Matches the canonical shape: dart:{slug}#{block_id}
# (lowercase slug/block, kept in sync with schemas/knowledge/source_reference.schema.json).
_SOURCE_ID_RE = re.compile(r"^dart:[a-z0-9_-]+#[a-z0-9_-]+$")

# Extract data-cf-source-ids values from emitted HTML. The attribute holds
# a comma-separated slug list — same shape as data-cf-key-terms.
_DATA_CF_SOURCE_IDS_RE = re.compile(
    r'data-cf-source-ids\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)

# Extract data-cf-source-primary single-id values when present.
_DATA_CF_SOURCE_PRIMARY_RE = re.compile(
    r'data-cf-source-primary\s*=\s*(["\'])([^"\']*)\1',
    re.IGNORECASE,
)

# Pull application/ld+json blocks out of HTML for sourceReferences extraction.
_JSON_LD_RE = re.compile(
    r'<script\b[^>]*type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


class PageSourceRefValidator:
    """Validates that every emitted sourceId resolves against staging.

    Inputs (any combination; all optional):
        page_paths: iterable of HTML file paths to scan.
        html_contents: list of ``{"path": ..., "html": ...}`` records.
        staging_dir: path to the run's staging directory produced by
            ``stage_dart_outputs`` (Wave 8). When provided, valid block IDs
            are harvested from the role-tagged manifest + provenance
            sidecars.
        source_module_map_path: path to ``source_module_map.json`` (the
            output of the Wave 9 source-router agent). Used to detect the
            backward-compat no-op path (empty map -> no refs expected).
        valid_source_ids: pre-computed iterable of valid ``dart:slug#id``
            strings. Overrides ``staging_dir`` harvesting; useful for tests.
    """

    name = "source_refs"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "source_refs")
        issues: List[GateIssue] = []

        valid_ids = self._collect_valid_ids(inputs)

        # Graceful fallback: empty source_module_map -> no refs expected.
        # We still scan pages for stray sourceIds (catches hallucination)
        # but an empty valid_ids set combined with an empty emitted set
        # means the gate passes silently.
        map_is_empty = self._source_map_is_empty(inputs)

        emitted_ids, emit_errors = self._collect_emitted_ids(inputs)
        for err in emit_errors:
            issues.append(err)

        if not emitted_ids:
            # Nothing to check. Backward-compat path.
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=issues,
            )

        # When the map is explicitly empty but the emitter produced
        # sourceIds anyway, that's a routing bug — treat as critical so
        # we don't silently propagate hallucinated IDs.
        if map_is_empty and not valid_ids:
            for entry in sorted(emitted_ids):
                page, sid = entry
                issues.append(GateIssue(
                    severity="critical",
                    code="UNEXPECTED_SOURCE_ID",
                    message=(
                        f"Page emitted sourceId {sid!r} but source_module_map.json "
                        "is empty. Either populate the map or drop the emit."
                    ),
                    location=page,
                ))

        for entry in sorted(emitted_ids):
            page, sid = entry
            if not _SOURCE_ID_RE.match(sid):
                issues.append(GateIssue(
                    severity="critical",
                    code="INVALID_SOURCE_ID_SHAPE",
                    message=(
                        f"sourceId {sid!r} does not match the canonical "
                        "dart:{slug}#{block_id} shape."
                    ),
                    location=page,
                    suggestion=(
                        "Only emit IDs produced by DART staging — see "
                        "schemas/knowledge/source_reference.schema.json."
                    ),
                ))
                continue
            if valid_ids and sid not in valid_ids:
                issues.append(GateIssue(
                    severity="critical",
                    code="UNRESOLVED_SOURCE_ID",
                    message=(
                        f"sourceId {sid!r} does not resolve against the "
                        "staging manifest / provenance sidecars."
                    ),
                    location=page,
                    suggestion=(
                        "Check that stage_dart_outputs produced a "
                        "provenance_sidecar entry declaring this block ID, "
                        "or that source_module_map.json points at a real block."
                    ),
                ))

        critical = [i for i in issues if i.severity == "critical"]
        # Score is the fraction of emitted IDs that resolved cleanly.
        bad_entries = len(critical)
        total = max(1, len(emitted_ids))
        score = max(0.0, 1.0 - bad_entries / total)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=len(critical) == 0,
            score=score,
            issues=issues,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _collect_valid_ids(self, inputs: Dict[str, Any]) -> Set[str]:
        """Resolve the valid sourceId universe from the inputs.

        Resolution order:
            1. Explicit ``valid_source_ids`` iterable (tests + callers who
               pre-compute the set).
            2. ``staging_dir`` — harvest IDs from the role-tagged manifest
               + every provenance_sidecar (``*_synthesized.json``).
        """
        pre = inputs.get("valid_source_ids")
        if pre is not None:
            return {str(s) for s in pre}

        staging_dir_arg = inputs.get("staging_dir")
        if not staging_dir_arg:
            return set()

        staging_dir = Path(staging_dir_arg)
        if not staging_dir.exists() or not staging_dir.is_dir():
            return set()

        valid: Set[str] = set()
        manifest_path = staging_dir / "staging_manifest.json"
        sidecar_names: List[str] = []
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in manifest.get("files", []) or []:
                    if entry.get("role") == "provenance_sidecar":
                        sidecar_names.append(entry.get("path", ""))
            except (OSError, json.JSONDecodeError):
                pass

        # Fall back to discovery when manifest is missing or empty.
        if not sidecar_names:
            sidecar_names = [
                p.name for p in staging_dir.glob("*_synthesized.json")
            ]

        for name in sidecar_names:
            sidecar = staging_dir / name
            if not sidecar.exists():
                continue
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for sid in _iter_sidecar_block_ids(data):
                valid.add(sid)

        return valid

    def _source_map_is_empty(self, inputs: Dict[str, Any]) -> bool:
        path_arg = inputs.get("source_module_map_path")
        if not path_arg:
            # No path provided — treat as "unknown", not empty. Caller
            # may still have seeded valid_source_ids directly.
            return False
        path = Path(path_arg)
        if not path.exists():
            return True
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if not data:
            return True
        # Nested shape: {week: {page: {primary: [...], contributing: [...]}}}
        for week_entries in data.values():
            if isinstance(week_entries, dict):
                for page_entry in week_entries.values():
                    if isinstance(page_entry, dict):
                        if page_entry.get("primary") or page_entry.get("contributing"):
                            return False
        return True

    def _collect_emitted_ids(
        self, inputs: Dict[str, Any]
    ) -> "tuple[Set[tuple[str, str]], List[GateIssue]]":
        """Scan every page for sourceIds emitted in JSON-LD or data-cf-*.

        Returns a set of (page_location, source_id) pairs plus a list of
        issues raised during scanning (e.g. malformed JSON-LD).
        """
        emitted: Set[tuple[str, str]] = set()
        scan_issues: List[GateIssue] = []

        records: List[tuple[str, str]] = []  # (location, html)

        for entry in inputs.get("html_contents") or []:
            loc = str(entry.get("path", "<inline>"))
            html = entry.get("html", "") or ""
            records.append((loc, html))

        for raw_path in inputs.get("page_paths") or []:
            path = Path(raw_path)
            if not path.exists():
                scan_issues.append(GateIssue(
                    severity="warning",
                    code="PAGE_NOT_FOUND",
                    message=f"Page not found while scanning source refs: {path}",
                    location=str(path),
                ))
                continue
            try:
                records.append((str(path), path.read_text(encoding="utf-8")))
            except OSError as exc:
                scan_issues.append(GateIssue(
                    severity="warning",
                    code="PAGE_READ_ERROR",
                    message=f"Failed to read page: {exc}",
                    location=str(path),
                ))

        for loc, html in records:
            # JSON-LD sourceReferences (page + section level)
            for match in _JSON_LD_RE.finditer(html):
                raw = match.group(1).strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    scan_issues.append(GateIssue(
                        severity="warning",
                        code="INVALID_JSON_LD",
                        message=f"JSON-LD block failed to parse: {exc}",
                        location=loc,
                    ))
                    continue
                for sid in _iter_jsonld_source_ids(data):
                    emitted.add((loc, sid))

            # HTML attributes: data-cf-source-ids + data-cf-source-primary
            for attr_match in _DATA_CF_SOURCE_IDS_RE.finditer(html):
                for raw_sid in attr_match.group(2).split(","):
                    sid = raw_sid.strip()
                    if sid:
                        emitted.add((loc, sid))
            for attr_match in _DATA_CF_SOURCE_PRIMARY_RE.finditer(html):
                sid = attr_match.group(2).strip()
                if sid:
                    emitted.add((loc, sid))

        return emitted, scan_issues


# ---------------------------------------------------------------------- #
# Module-level helpers (public for unit tests that bypass the validator)
# ---------------------------------------------------------------------- #


def _iter_jsonld_source_ids(data: Any) -> Iterable[str]:
    """Walk a JSON-LD payload and yield every sourceReferences sourceId.

    Handles the Wave 9 shape: ``sourceReferences`` at page level and inside
    every ``sections[]`` entry. Silently tolerates nested shapes so legacy
    emitters don't trip the scan.
    """
    if isinstance(data, dict):
        refs = data.get("sourceReferences")
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict):
                    sid = ref.get("sourceId")
                    if isinstance(sid, str) and sid:
                        yield sid
        # Recurse into sections (and anywhere else sourceReferences might
        # nest, to stay forward-compat with additional carrier objects).
        for value in data.values():
            yield from _iter_jsonld_source_ids(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_source_ids(item)


def _iter_sidecar_block_ids(data: Any) -> Iterable[str]:
    """Walk a Wave 8 ``*_synthesized.json`` sidecar and yield valid sourceIds.

    Recognized shapes:

    - Top-level ``campus_code`` + ``sections[]`` (multi-source synthesizer
      output) → emit ``dart:{slug}#{section_id}`` and
      ``dart:{slug}#{block_id}`` for every leaf block.
    - Top-level ``document_slug`` override (set by future ingestors) —
      prefer it when present.
    - Any nested ``block_id`` key encountered while walking; paired with
      the closest surrounding slug.
    """
    if not isinstance(data, dict):
        return

    slug = _resolve_doc_slug(data)
    if not slug:
        return

    sections = data.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_id = section.get("section_id")
            if isinstance(section_id, str) and section_id:
                yield f"dart:{slug}#{section_id}"
            # Block IDs live anywhere under data[]
            for block_id in _iter_nested_block_ids(section):
                yield f"dart:{slug}#{block_id}"


def _iter_nested_block_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        bid = value.get("block_id")
        if isinstance(bid, str) and bid:
            yield bid
        for nested in value.values():
            yield from _iter_nested_block_ids(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_block_ids(item)


def _resolve_doc_slug(data: Dict[str, Any]) -> Optional[str]:
    """Pick the canonical doc slug from a synthesized sidecar.

    Prefers an explicit ``document_slug`` override; falls back to the
    ``campus_code`` field emitted by ``multi_source_interpreter`` today.
    The returned value is lower-cased / slugified in the same way as
    ``DART.multi_source_interpreter._document_slug``.
    """
    explicit = data.get("document_slug")
    if isinstance(explicit, str) and explicit.strip():
        return _slugify_doc(explicit)
    code = data.get("campus_code")
    if isinstance(code, str) and code.strip():
        return _slugify_doc(code)
    return None


def _slugify_doc(code: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", code).strip("_").lower()
    return slug or "document"
