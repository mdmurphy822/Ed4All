"""ContentGroundingValidator (Wave 31 — new).

Addresses the "empty generated course" defect observed on the
``SIM_RUN_01`` run. 48 weekly pages shipped with < 80 words each, empty
objectives lists, and the same activity prompt copy-pasted 12 times.
Nothing in the pre-Wave-31 QA caught that as "empty content" — the
content validators all rubber-stamped the output because it *had* HTML
structure.

This validator asserts that generated Courseforge content actually
traces back to DART source. For every non-trivial paragraph in every
page, we require a ``data-cf-source-ids`` attribute on the element or
one of its ancestors (Wave 27 emits these). If the attribute is present,
we also verify the source ID resolves to a known ``data-dart-block-id``
in the staged DART HTML.

Failure modes caught
--------------------

* **Ungrounded paragraph** — a substantive ``<p>`` / ``<li>`` with no
  ancestor carrying ``data-cf-source-ids``. If ≥ 50% of a page's
  non-trivial paragraphs are ungrounded, the page fails critical.
* **Unresolved source ID** — ``data-cf-source-ids`` references a block
  that does not appear in any staged DART synthesized sidecar or HTML.
* **Empty page** — a weekly page with zero paragraphs > 30 words.
  Aggregated: if ≥ 25% of pages are empty → critical; if < 25% but > 0
  → warning.

Referenced by: ``config/workflows.yaml`` →
``textbook_to_course.content_generation.validation_gates[content_grounding]``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# H3 W6a: orchestration-phase decision-capture (Pattern A — one emit
# per validate() call).
def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    blocks_audited: int,
    blocks_grounded: int,
    grounding_rate: Optional[float],
    blocks_with_attribute_walk_failures: int,
    pages_audited: int,
    empty_pages: int,
    resolution_enabled: bool,
) -> None:
    """Emit one ``content_grounding_check`` decision per validate() call."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rate_str = (
        f"{grounding_rate:.3f}" if grounding_rate is not None else "n/a"
    )
    rationale = (
        f"Content-grounding orchestration check: "
        f"pages_audited={pages_audited}, "
        f"empty_pages={empty_pages}, "
        f"blocks_audited={blocks_audited}, "
        f"blocks_grounded={blocks_grounded}, "
        f"grounding_rate={rate_str}, "
        f"blocks_with_attribute_walk_failures={blocks_with_attribute_walk_failures}, "
        f"resolution_enabled={resolution_enabled}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="content_grounding_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on content_grounding_check: %s",
            exc,
        )

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


# Non-trivial paragraph threshold (words). Short paragraphs like
# "Chapter 3" aren't expected to carry source attribution.
NON_TRIVIAL_WORD_FLOOR = 30

# Per-page critical threshold: fraction of ungrounded non-trivial paragraphs.
PAGE_CRITICAL_UNGROUNDED_FRACTION = 0.5

# Aggregate empty-page thresholds.
AGGREGATE_EMPTY_CRITICAL_FRACTION = 0.25

# Regex to extract block IDs from DART source HTML / synthesized JSON.
_DART_BLOCK_ID_RE = re.compile(
    r'data-dart-block-id\s*=\s*(["\'])([^"\']+)\1',
    re.IGNORECASE,
)

# FIX 1 — structural-page grounding exemption. The auto-generated learning
# objectives map (``course_overview/learning_objectives.html``) renders CO
# objective statements as ``<li>`` carrying no ``data-cf-source-ids`` —
# legitimately, because an objective statement is a synthesized learning target
# with no single DART source block. Mirrors the ``_COGNITIVE_LOAD_EXEMPT_TYPES``
# precedent in ``lib/validators/content.py``. TIGHT by basename ONLY: the week
# ``*_overview`` / ``*_summary`` pages ARE grounded and MUST keep being checked,
# so this set is intentionally a single basename, not a path prefix or suffix.
_GROUNDING_EXEMPT_BASENAMES = frozenset({"learning_objectives.html"})

# FIX 2 — chunk-id namespace recognizer. The ``ED4ALL_KEY_TERMS_PAGE`` feature
# stamps REAL chunk ids (shape ``{course}_chunk_NNNNN``, see
# ``Trainforge/chunker/chunker.py::_generate_chunk_id``) as
# ``data-cf-source-ids`` on its vocab cards in the canonical
# ``dart:{slug}#{course}_chunk_NNNNN`` form — deep-linking to a chunk's
# ``item_path``. Those ids live in the chunk-id namespace, NOT the DART
# block-id universe, so they are resolved by exact membership against the run's
# harvested chunkset (the primary path). The pattern below is used ONLY for the
# harvest-unavailable fallback demotion (a genuinely-bogus DART block id has no
# ``_chunk_N`` suffix, so it can never match here and stays critical).
_CHUNK_ID_RE = re.compile(r"_chunk_\d+$")


class ContentGroundingValidator:
    """Verifies Courseforge content traces back to DART source blocks.

    Expected inputs:
        page_paths: iterable of HTML file paths (Courseforge-generated).
        staging_dir: Path to the run's Courseforge staging dir
            (produced by ``stage_dart_outputs``). Used to harvest the
            universe of valid DART block IDs. Optional — when absent,
            we still validate that source-id attributes EXIST but
            don't check whether they resolve.
        valid_block_ids: optional pre-computed iterable of valid
            block IDs for tests that don't want to build a staging dir.
        content_dir: alternative to page_paths — a directory walked for
            all ``.html`` pages.
    """

    name = "content_grounding"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "content_grounding")
        capture = inputs.get("decision_capture")
        if capture is None:
            capture = inputs.get("capture")
        issues: List[GateIssue] = []

        if BeautifulSoup is None:
            _emit_decision(
                capture,
                passed=False,
                code="MISSING_DEPENDENCY",
                blocks_audited=0,
                blocks_grounded=0,
                grounding_rate=None,
                blocks_with_attribute_walk_failures=0,
                pages_audited=0,
                empty_pages=0,
                resolution_enabled=False,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_DEPENDENCY",
                    message="BeautifulSoup is required for ContentGroundingValidator",
                )],
            )

        page_paths = self._collect_page_paths(inputs)
        if not page_paths:
            _emit_decision(
                capture,
                passed=True,
                code="NO_PAGES_TO_SCAN",
                blocks_audited=0,
                blocks_grounded=0,
                grounding_rate=None,
                blocks_with_attribute_walk_failures=0,
                pages_audited=0,
                empty_pages=0,
                resolution_enabled=False,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code="NO_PAGES_TO_SCAN",
                    message="No Courseforge page paths supplied — skipping grounding check.",
                )],
            )

        valid_block_ids = self._resolve_valid_block_ids(inputs)
        # FIX 2 — also harvest the run's chunk-id universe so legitimately
        # chunk-grounded key_terms source-ids (``dart:{slug}#{course}_chunk_N``)
        # resolve instead of categorically failing UNRESOLVED_SOURCE_ID.
        valid_chunk_ids = self._resolve_valid_chunk_ids(inputs)
        resolution_enabled = bool(valid_block_ids) or bool(valid_chunk_ids)

        per_page_stats: List[Dict[str, Any]] = []
        empty_pages: List[str] = []
        critical_pages: List[str] = []
        # FIX 1 — pages exempt from the per-page grounding check (the auto-
        # generated objectives-list structural page) are excluded from BOTH the
        # per-page checks AND the empty-page-fraction denominator.
        exempt_pages = 0

        for page_path in page_paths:
            # FIX 1 — skip the structural objectives-list page entirely (its
            # ungrounded objective <li>s are legitimate, not a content defect).
            if self._is_grounding_exempt_page(page_path):
                exempt_pages += 1
                continue

            stats = self._analyze_page(
                page_path, valid_block_ids, valid_chunk_ids, resolution_enabled
            )
            per_page_stats.append(stats)

            if stats["is_empty"]:
                empty_pages.append(str(page_path))
                continue
            # FIX 2 — chunk-namespace source-ids that could not be resolved
            # against a harvested chunkset (harvest unavailable) demote to a
            # warning rather than a categorical critical failure.
            for soft in stats.get("soft_unresolved_ids", [])[:3]:
                issues.append(GateIssue(
                    severity="warning",
                    code="UNRESOLVED_CHUNK_SOURCE_ID",
                    message=(
                        f"data-cf-source-ids references chunk-namespace id "
                        f"{soft!r} which could not be confirmed against the "
                        f"run chunkset (chunkset unavailable to harvest); "
                        f"treating as warning, not a grounding failure."
                    ),
                    location=str(page_path),
                ))
            if stats["ungrounded_fraction"] >= PAGE_CRITICAL_UNGROUNDED_FRACTION:
                critical_pages.append(str(page_path))
                issues.append(GateIssue(
                    severity="critical",
                    code="PAGE_UNGROUNDED",
                    message=(
                        f"{stats['ungrounded_paragraphs']} of "
                        f"{stats['non_trivial_paragraphs']} non-trivial paragraphs "
                        f"({stats['ungrounded_fraction']:.0%}) carry no "
                        f"data-cf-source-ids — page content is not grounded in "
                        f"DART source."
                    ),
                    location=str(page_path),
                    suggestion=(
                        "Ensure the content-generator copies "
                        "data-cf-source-ids from the source_module_map onto "
                        "every generated content paragraph."
                    ),
                ))
            # Unresolved source-IDs on this page.
            for unresolved in stats["unresolved_ids"][:3]:
                issues.append(GateIssue(
                    severity="critical",
                    code="UNRESOLVED_SOURCE_ID",
                    message=(
                        f"data-cf-source-ids references {unresolved!r} but "
                        f"that block ID does not appear in any staged DART "
                        f"synthesized sidecar."
                    ),
                    location=str(page_path),
                ))

        # Aggregate empty-page analysis. FIX 1 — exempt structural pages are not
        # part of the denominator (they are never "content" pages to begin with).
        total_pages = len(page_paths) - exempt_pages
        empty_fraction = len(empty_pages) / total_pages if total_pages else 0.0
        if empty_fraction >= AGGREGATE_EMPTY_CRITICAL_FRACTION:
            issues.append(GateIssue(
                severity="critical",
                code="AGGREGATE_EMPTY_PAGES",
                message=(
                    f"{len(empty_pages)} of {total_pages} pages "
                    f"({empty_fraction:.0%}) contain zero non-trivial paragraphs "
                    f"(>{NON_TRIVIAL_WORD_FLOOR} words each). "
                    f"This isn't a course — it's an empty template."
                ),
                suggestion=(
                    "Re-run content_generation with real DART source material "
                    "and check that the content-generator is producing body "
                    "paragraphs (not just objective lists + headings)."
                ),
                location=",".join(empty_pages[:5]),
            ))
        elif empty_pages:
            issues.append(GateIssue(
                severity="warning",
                code="SOME_EMPTY_PAGES",
                message=(
                    f"{len(empty_pages)} of {total_pages} pages "
                    f"({empty_fraction:.0%}) contain zero non-trivial paragraphs."
                ),
                location=",".join(empty_pages[:5]),
            ))

        # Compute overall score.
        total_paragraphs = sum(s["non_trivial_paragraphs"] for s in per_page_stats)
        total_ungrounded = sum(s["ungrounded_paragraphs"] for s in per_page_stats)
        if total_paragraphs == 0:
            score = 0.0 if empty_fraction >= AGGREGATE_EMPTY_CRITICAL_FRACTION else 0.5
        else:
            score = max(0.0, 1.0 - total_ungrounded / total_paragraphs)

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0

        # Pick a representative failure code for the rationale.
        failure_code: Optional[str] = None
        if not passed:
            failure_code = next(
                (i.code for i in issues if i.severity == "critical"), None
            )
        grounded = total_paragraphs - total_ungrounded
        grounding_rate = (
            grounded / total_paragraphs if total_paragraphs > 0 else None
        )
        attribute_walk_failures = total_ungrounded
        _emit_decision(
            capture,
            passed=passed,
            code=failure_code,
            blocks_audited=total_paragraphs,
            blocks_grounded=grounded,
            grounding_rate=grounding_rate,
            blocks_with_attribute_walk_failures=attribute_walk_failures,
            pages_audited=total_pages,
            empty_pages=len(empty_pages),
            resolution_enabled=resolution_enabled,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _collect_page_paths(self, inputs: Dict[str, Any]) -> List[Path]:
        result: List[Path] = []
        paths = inputs.get("page_paths")
        if paths:
            for p in paths:
                path = Path(p) if not isinstance(p, Path) else p
                if path.exists() and path.is_file():
                    result.append(path)
        content_dir = inputs.get("content_dir")
        if not result and content_dir:
            cd = Path(content_dir)
            if cd.exists():
                result.extend(sorted(cd.rglob("*.html")))
        return result

    def _resolve_valid_block_ids(self, inputs: Dict[str, Any]) -> Set[str]:
        pre = inputs.get("valid_block_ids")
        if pre is not None:
            return {str(b) for b in pre}

        valid: Set[str] = set()
        staging_arg = inputs.get("staging_dir")
        if not staging_arg:
            return valid
        staging_dir = Path(staging_arg)
        if not staging_dir.exists():
            return valid

        # Scan every HTML file in staging for data-dart-block-id.
        for html_path in staging_dir.rglob("*.html"):
            try:
                content = html_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in _DART_BLOCK_ID_RE.finditer(content):
                raw_block_id = match.group(2).strip()
                # Normalize to the canonical dart:{slug}#{block_id} shape.
                slug = html_path.stem.lower().replace(" ", "-")
                valid.add(f"dart:{slug}#{raw_block_id}")
                # Also allow the bare block_id for tests that use it directly.
                valid.add(raw_block_id)

        # Also scan synthesized sidecars for block_id fields.
        for sidecar_path in staging_dir.rglob("*_synthesized.json"):
            try:
                import json as _json
                data = _json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for bid in self._iter_sidecar_block_ids(data, sidecar_path.stem):
                valid.add(bid)

        return valid

    @staticmethod
    def _iter_sidecar_block_ids(data: Any, stem: str) -> Iterable[str]:
        if isinstance(data, dict):
            for key, val in data.items():
                if key in ("block_id", "section_id") and isinstance(val, str):
                    # Normalize stem: strip trailing "_synthesized" if present.
                    normalized_stem = stem.replace("_synthesized", "").lower().replace(" ", "-")
                    yield f"dart:{normalized_stem}#{val}"
                    yield val
                elif isinstance(val, (dict, list)):
                    yield from ContentGroundingValidator._iter_sidecar_block_ids(val, stem)
        elif isinstance(data, list):
            for item in data:
                yield from ContentGroundingValidator._iter_sidecar_block_ids(item, stem)

    @staticmethod
    def _is_grounding_exempt_page(page_path: Path) -> bool:
        """True for the auto-generated objectives-list structural page (FIX 1).

        TIGHT: matches ONLY the ``learning_objectives.html`` basename (the page
        the packager renders into ``course_overview/``). Week ``*_overview`` /
        ``*_summary`` content pages are deliberately NOT matched — they carry
        real grounded prose and must keep being checked.
        """
        return page_path.name.lower() in _GROUNDING_EXEMPT_BASENAMES

    @staticmethod
    def _resolve_dart_chunks_path(inputs: Dict[str, Any]) -> Optional[Path]:
        """Resolve the run's ``dart_chunks/chunks.jsonl`` path for FIX 2.

        The ``run_dart_chunking`` phase (which runs before content_generation)
        emits ``dart_chunks_path`` in its output; the gate input builder
        forwards it here. Returns ``None`` when no path is supplied.
        """
        raw = inputs.get("dart_chunks_path") or inputs.get("chunks_path")
        if isinstance(raw, Path):
            return raw
        if isinstance(raw, str) and raw.strip():
            return Path(raw.strip())
        return None

    def _resolve_valid_chunk_ids(self, inputs: Dict[str, Any]) -> Set[str]:
        """Harvest the run's bare chunk ids from the chunkset (FIX 2).

        The ``ED4ALL_KEY_TERMS_PAGE`` feature deep-links vocab cards to real
        chunks via ``data-cf-source-ids="dart:{slug}#{chunk_id}"``. Those
        chunk ids resolve against the run's chunkset (the SAME
        ``dart_chunks/chunks.jsonl`` the feature reads), not the DART block-id
        universe, so they are harvested separately and matched by their
        post-``#`` component.

        Source order: explicit ``valid_chunk_ids`` (tests) > the chunkset at
        ``dart_chunks_path``. A missing / unreadable chunkset yields an empty
        set (the validator then falls back to pattern-demotion of the
        chunk-namespace unresolved case to a warning).
        """
        pre = inputs.get("valid_chunk_ids")
        if pre is not None:
            return {str(c) for c in pre}

        out: Set[str] = set()
        path = self._resolve_dart_chunks_path(inputs)
        if path is None or not path.exists():
            return out
        import json as _json

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    cid = obj.get("id") or obj.get("chunk_id")
                    if isinstance(cid, str) and cid.strip():
                        out.add(cid.strip())
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning(
                "content_grounding: failed to read chunkset at %s: %s",
                path, exc,
            )
        return out

    def _analyze_page(
        self,
        page_path: Path,
        valid_block_ids: Set[str],
        valid_chunk_ids: Set[str],
        resolution_enabled: bool,
    ) -> Dict[str, Any]:
        """Per-page stats: non-trivial paragraph count + grounding coverage."""
        try:
            html = page_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return {
                "path": str(page_path),
                "is_empty": True,
                "non_trivial_paragraphs": 0,
                "ungrounded_paragraphs": 0,
                "ungrounded_fraction": 0.0,
                "unresolved_ids": [],
                "soft_unresolved_ids": [],
            }

        soup = BeautifulSoup(html, "html.parser")

        # Strip nav/header/footer so their paragraphs don't pollute counts.
        for tag in soup.find_all(["nav", "header", "footer"]):
            tag.decompose()

        candidate_elements = soup.find_all(
            ["p", "li", "figcaption", "blockquote"]
        )

        non_trivial = 0
        ungrounded = 0
        unresolved_ids: List[str] = []
        soft_unresolved_ids: List[str] = []
        for el in candidate_elements:
            text = el.get_text(separator=" ", strip=True)
            word_count = len(text.split())
            if word_count < NON_TRIVIAL_WORD_FLOOR:
                continue
            non_trivial += 1

            # Look for data-cf-source-ids on element or any ancestor.
            source_ids_attr = self._find_source_ids(el)
            if not source_ids_attr:
                ungrounded += 1
                continue
            # Parse comma-separated IDs.
            ids = [s.strip() for s in source_ids_attr.split(",") if s.strip()]
            if not ids:
                ungrounded += 1
                continue
            if resolution_enabled:
                for sid in ids:
                    if sid in valid_block_ids:
                        continue
                    # FIX 2 — chunk-id namespace resolution. The key_terms
                    # feature stamps ``dart:{slug}#{chunk_id}``; extract the
                    # post-``#`` chunk-id component (or the bare id) and resolve
                    # it by EXACT membership against the harvested chunkset. This
                    # is tight: a genuinely-bogus chunk-shaped id is not in the
                    # set, so it still flags critical.
                    cid = sid.split("#", 1)[1] if "#" in sid else sid
                    if valid_chunk_ids and cid in valid_chunk_ids:
                        continue
                    # FIX 2 fallback — when the chunkset is unavailable to
                    # harvest, demote ONLY the chunk-id namespace (``_chunk_N``
                    # suffix) to a soft warning. A bogus DART block id never
                    # matches this pattern, so its unresolved case stays
                    # critical (detection unweakened).
                    if not valid_chunk_ids and _CHUNK_ID_RE.search(cid):
                        soft_unresolved_ids.append(sid)
                        continue
                    unresolved_ids.append(sid)

        ungrounded_fraction = ungrounded / non_trivial if non_trivial else 0.0
        return {
            "path": str(page_path),
            "is_empty": non_trivial == 0,
            "non_trivial_paragraphs": non_trivial,
            "ungrounded_paragraphs": ungrounded,
            "ungrounded_fraction": ungrounded_fraction,
            "unresolved_ids": unresolved_ids,
            "soft_unresolved_ids": soft_unresolved_ids,
        }

    @staticmethod
    def _find_source_ids(element) -> Optional[str]:
        """Walk element + ancestors for the first ``data-cf-source-ids``."""
        cur = element
        while cur is not None and hasattr(cur, "get"):
            val = cur.get("data-cf-source-ids")
            if val:
                return val
            cur = cur.parent
        return None


def _build_content_grounding(phase_outputs: Dict[str, Any], workflow_params: Dict[str, Any]):
    """Gate input builder (moved into the module so it's self-contained)."""
    pages: List[str] = []
    # Pull from content_generation.content_paths.
    cg = phase_outputs.get("content_generation") or {}
    cps = cg.get("content_paths")
    if isinstance(cps, str) and cps:
        pages.extend(p.strip() for p in cps.split(",") if p.strip())

    # Walk content_dir as fallback.
    if not pages:
        for phase_data in phase_outputs.values():
            if not isinstance(phase_data, dict):
                continue
            cd = phase_data.get("content_dir")
            if isinstance(cd, str) and cd:
                cdp = Path(cd)
                if cdp.exists():
                    pages.extend(str(p) for p in sorted(cdp.rglob("*.html")))
                    break

    staging = None
    for phase_data in phase_outputs.values():
        if not isinstance(phase_data, dict):
            continue
        sd = phase_data.get("staging_dir")
        if isinstance(sd, str) and sd:
            staging = sd
            break

    # FIX 2 — forward the run's DART chunkset so chunk-namespace key_terms
    # source-ids resolve. The ``run_dart_chunking`` phase emits
    # ``dart_chunks_path``; scan all phase outputs for it (it runs before
    # content_generation, so it is present in the accumulated phase outputs).
    dart_chunks_path = None
    for phase_data in phase_outputs.values():
        if not isinstance(phase_data, dict):
            continue
        dcp = phase_data.get("dart_chunks_path")
        if isinstance(dcp, str) and dcp:
            dart_chunks_path = dcp
            break

    inputs: Dict[str, Any] = {"page_paths": pages}
    if staging:
        inputs["staging_dir"] = staging
    if dart_chunks_path:
        inputs["dart_chunks_path"] = dart_chunks_path
    if not pages:
        return inputs, ["page_paths"]
    return inputs, []


__all__ = ["ContentGroundingValidator", "_build_content_grounding"]
