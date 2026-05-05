"""Per-gate input routing (Wave 23 Sub-task A).

Pre-Wave-23, ``TaskExecutor.execute_phase`` invoked
``ValidationGateManager.run_phase_gates`` with a single generic blob
(``{'artifacts': ..., 'results': ...}``) regardless of which validator
was about to run. Every real validator expects a bespoke shape
(``html_path``, ``content_dir``, ``imscc_path``, ``page_paths`` + friends,
``manifest_path`` + ``course_dir``, ...), so in practice every gate
either returned an error issue ("MISSING_CONTENT_DIR" /
"EMPTY_CONTENT" / ...), which — because most gates were configured
at ``severity: warning / on_fail: warn`` — still let the phase pass.
Other gates wired at ``severity: critical`` happened to be unused.

This module is the single source of truth for mapping a phase's
accumulated outputs + workflow-level params into the per-validator
input shape. It's data-driven: each validator dotted path maps to a
small builder that inspects the phase outputs + workflow params and
returns a ready-to-use kwargs dict. Adding a new validator is a
one-line registry edit.

Contract
--------

A builder returns ``(inputs, required_missing)``:

* ``inputs``: the kwargs dict to hand to ``validator.validate(...)``.
* ``required_missing``: list of input-key names that the validator
  needs but weren't available. If the list is non-empty, the gate
  must be marked ``skipped=True`` with a structured reason — not
  silently passed or silently failed.

Builders never raise. They return the missing-key list on any failure
path so the caller can log structured skip reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #


def _find_content_dir(phase_outputs: Dict[str, Any]) -> Optional[Path]:
    """Locate a content_dir candidate from accumulated phase outputs.

    Courseforge's content-generation phase emits ``content_paths`` as a
    comma-joined list of generated HTML paths under a
    ``.../content/`` directory. The ``content_dir`` is the common
    parent. When the phase exposes a ``project_path`` (pre-Wave-8
    shape) we prefer ``project_path / "content"`` to match the
    packager's layout.
    """
    # Preferred: explicit content_dir key wherever it appears.
    for phase_data in phase_outputs.values():
        if not isinstance(phase_data, dict):
            continue
        cd = phase_data.get("content_dir")
        if isinstance(cd, str) and cd:
            return Path(cd)

    # Derive from content_generation.content_paths
    cg = phase_outputs.get("content_generation") or {}
    content_paths = cg.get("content_paths")
    if isinstance(content_paths, str) and content_paths:
        # comma-joined list; take the first existing parent
        for p in content_paths.split(","):
            cand = Path(p.strip())
            if cand.exists():
                # Walk up until we find "content/" directory or project root
                for parent in [cand.parent, *cand.parents]:
                    if parent.name == "content":
                        return parent
                return cand.parent
        # fallback: just return the parent of the first path
        first = content_paths.split(",")[0].strip()
        if first:
            return Path(first).parent

    # Derive from objective_extraction.project_path
    oe = phase_outputs.get("objective_extraction") or {}
    project_path = oe.get("project_path")
    if isinstance(project_path, str) and project_path:
        content_dir = Path(project_path) / "content"
        if content_dir.exists():
            return content_dir

    return None


def _walk_html_paths(content_dir: Path) -> List[Path]:
    """Return all .html files under content_dir (deterministic order)."""
    if not content_dir or not content_dir.exists():
        return []
    return sorted(content_dir.rglob("*.html"))


def _first_html_path(phase_outputs: Dict[str, Any]) -> Optional[Path]:
    """Locate a single html_path candidate for validators that need one.

    DART output paths surface as ``output_path`` (single) or
    ``output_paths`` (comma-joined). Falls back to walking the
    discovered content_dir when no DART outputs are present.
    """
    dc = phase_outputs.get("dart_conversion") or {}
    op = dc.get("output_path")
    if isinstance(op, str) and op:
        return Path(op)
    ops = dc.get("output_paths")
    if isinstance(ops, str) and ops:
        first = ops.split(",")[0].strip()
        if first:
            return Path(first)

    cd = _find_content_dir(phase_outputs)
    htmls = _walk_html_paths(cd) if cd else []
    return htmls[0] if htmls else None


def _all_html_paths(phase_outputs: Dict[str, Any]) -> List[str]:
    """Return a list of HTML page paths derivable from phase outputs."""
    dc = phase_outputs.get("dart_conversion") or {}
    ops = dc.get("output_paths")
    if isinstance(ops, str) and ops:
        return [p.strip() for p in ops.split(",") if p.strip()]
    op = dc.get("output_path")
    if isinstance(op, str) and op:
        return [op]

    cg = phase_outputs.get("content_generation") or {}
    cps = cg.get("content_paths")
    if isinstance(cps, str) and cps:
        return [p.strip() for p in cps.split(",") if p.strip()]

    cd = _find_content_dir(phase_outputs)
    return [str(p) for p in _walk_html_paths(cd)] if cd else []


def _locate(phase_outputs: Dict[str, Any], *keys: str) -> Optional[str]:
    """Find the first non-empty str value matching any key across all phases."""
    for phase_data in phase_outputs.values():
        if not isinstance(phase_data, dict):
            continue
        for key in keys:
            val = phase_data.get(key)
            if isinstance(val, str) and val:
                return val
    return None


# ---------------------------------------------------------------------- #
# Per-validator builders
# ---------------------------------------------------------------------- #


BuilderResult = Tuple[Dict[str, Any], List[str]]


def _build_content_structure(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    html = _first_html_path(phase_outputs)
    if html and html.exists():
        return {"html_path": str(html)}, []
    # No HTML available — must be skipped, not passed.
    return {}, ["html_path"]


def _build_page_objectives(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    content_dir = _find_content_dir(phase_outputs)
    if content_dir is None:
        return {}, ["content_dir"]
    inputs: Dict[str, Any] = {"content_dir": str(content_dir)}
    # Forward objectives_path when the workflow surfaced one.
    op = workflow_params.get("objectives_path")
    if isinstance(op, str) and op:
        inputs["objectives_path"] = op
    return inputs, []


def _build_source_refs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    staging = _locate(phase_outputs, "staging_dir")
    smm = _locate(phase_outputs, "source_module_map_path")
    pages = _all_html_paths(phase_outputs)
    inputs: Dict[str, Any] = {"page_paths": pages}
    if staging:
        inputs["staging_dir"] = staging
    if smm:
        inputs["source_module_map_path"] = smm
    # page_paths is the required input — source_refs validator gracefully
    # handles empty pages at pass, but if we literally have no pages,
    # we can't assert anything, so mark as skipped.
    if not pages:
        return inputs, ["page_paths"]
    return inputs, []


def _build_imscc(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    # imscc path lives under packaging.package_path or workflow_params.imscc_path
    imscc = _locate(phase_outputs, "imscc_path", "package_path", "libv2_package_path")
    if not imscc:
        imscc = workflow_params.get("imscc_path")
    if not imscc:
        return {}, ["imscc_path"]
    return {"imscc_path": imscc}, []


def _build_wcag(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    # WCAGValidator.validate(html: str, file_path: str=...) is a positional
    # signature, but the gate manager passes kwargs. We deliberately expose
    # html_path so a shim (see executor.py) can call .validate_file for us.
    html = _first_html_path(phase_outputs)
    if html and html.exists():
        return {"html_path": str(html)}, []
    return {}, ["html_path"]


def _build_oscqr(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 31 OSCQRValidator: forward course_path / content_dir + course.json + imscc.

    The Wave 31 implementation inspects the whole course artifact:
    weekly HTML pages, course.json (for assessments), and optionally
    the IMSCC package.
    """
    inputs: Dict[str, Any] = {}
    # Prefer content_dir from content_generation; fall back to course_dir
    # from packaging/archival.
    content_dir = _find_content_dir(phase_outputs)
    if content_dir is not None:
        inputs["content_dir"] = str(content_dir)
    course_path = _locate(phase_outputs, "course_dir", "project_path")
    if course_path:
        inputs["course_path"] = course_path
    # Forward IMSCC path when packaging has completed.
    imscc = _locate(phase_outputs, "package_path", "imscc_path", "libv2_package_path")
    if imscc:
        inputs["imscc_path"] = imscc
    # Explicit course.json path if the planner surfaced one.
    cj = _locate(phase_outputs, "course_json_path", "synthesized_objectives_path")
    if cj:
        inputs["course_json_path"] = cj
    # Objectives still flow through for downstream item alignment.
    objectives = workflow_params.get("objectives_path")
    if objectives:
        inputs["objectives_path"] = objectives
    return inputs, []


def _build_dart_markers(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 29: batch-aware DART markers resolution.

    Pre-Wave-29 the builder only returned a single ``html_path``; when
    the DART phase emitted multiple HTML files (batch corpora) only the
    first file was validated. Now we surface the full list as
    ``html_paths`` alongside a representative ``html_path`` so:

    * the validator's single-file entrypoint still works (back-compat)
    * an aggregating caller can walk ``html_paths`` to validate every
      emitted file.

    Reaches through a broader set of phase-output keys so staged copies
    (``staging.html_paths``) and batch emits
    (``dart_conversion.output_paths``) both surface.
    """
    all_paths = _all_html_paths(phase_outputs)
    existing = [Path(p) for p in all_paths if Path(p).exists()]
    if not existing:
        # One last fallback: try the single html_path helper (walks
        # content_dir when DART outputs are absent).
        single = _first_html_path(phase_outputs)
        if single and single.exists():
            existing = [single]
    if not existing:
        return {}, ["html_path"]

    inputs: Dict[str, Any] = {
        "html_path": str(existing[0]),
        "html_paths": [str(p) for p in existing],
    }
    return inputs, []


def _build_assessment_quality(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 29: check file existence / non-empty before handing off.

    Pre-Wave-29 the builder returned any string path the phase
    surfaced, letting the validator crash with
    ``json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)``
    when the path pointed at an empty or absent file (a common outcome
    when ``--no-assessments`` was half-honoured, or when Trainforge
    phase bailed early without writing the assessments file).

    Now we:

    * resolve the candidate path as before,
    * verify it exists AND is non-empty,
    * return ``(None, ['ASSESSMENTS_FILE_MISSING'])`` when it isn't so
      the gate is marked skipped with a structured reason rather than
      crashing on ``json.loads``.
    """
    path_str = _locate(
        phase_outputs,
        "assessments_path",
        "assessment_path",
        "output_path",
        "assessment_id",  # trainforge fallback
    )
    if not path_str:
        return {}, ["ASSESSMENTS_FILE_MISSING"]

    try:
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return {}, ["ASSESSMENTS_FILE_MISSING"]
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size <= 0:
            return {}, ["ASSESSMENTS_FILE_MISSING"]
    except (OSError, ValueError, TypeError):
        return {}, ["ASSESSMENTS_FILE_MISSING"]

    return {"assessment_path": str(path)}, []


def _build_bloom_alignment(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    path = _locate(phase_outputs, "assessment_path", "output_path")
    if not path:
        return {}, ["assessment_path"]
    return {"assessment_path": path}, []


def _build_leak_check(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    # LeakCheckValidator needs assessment_data dict; the executor can't
    # reconstitute that from file paths cheaply, so we skip when the
    # caller hasn't pre-loaded it into workflow_params.assessment_data.
    data = workflow_params.get("assessment_data")
    if isinstance(data, dict):
        return {"assessment_data": data}, []
    # Try to load from assessment path as best effort.
    path = _locate(phase_outputs, "assessment_path", "output_path")
    if path:
        try:
            import json as _json
            p = Path(path)
            if p.exists():
                return {"assessment_data": _json.loads(p.read_text(encoding="utf-8"))}, []
        except (OSError, ValueError, TypeError):
            pass
    return {}, ["assessment_data"]


def _build_final_quality(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    # Same shape as assessment_quality for now.
    return _build_assessment_quality(phase_outputs, workflow_params)


def _build_content_facts(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    # Works on chunks_path or an in-memory chunks list.
    chunks_path = _locate(phase_outputs, "chunks_path")
    if chunks_path:
        return {"chunks_path": chunks_path}, []
    return {}, ["chunks_path"]


def _build_question_quality(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    # Same dependency as leak_check — needs assessment_data.
    return _build_leak_check(phase_outputs, workflow_params)


def _build_libv2_manifest(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 29: derive ``manifest_path`` from ``course_dir`` when absent.

    Pre-Wave-29 the builder only looked for an explicit ``manifest_path``
    key in phase outputs. The ``libv2_archival`` phase emits
    ``course_dir`` (the archived course root) and guarantees
    ``manifest.json`` sits inside — so when ``manifest_path`` isn't
    surfaced explicitly we derive it as ``course_dir/manifest.json``.
    """
    manifest = _locate(phase_outputs, "manifest_path")
    course_dir = _locate(phase_outputs, "course_dir")

    if not manifest and course_dir:
        try:
            derived = Path(course_dir) / "manifest.json"
            if derived.exists():
                manifest = str(derived)
        except (OSError, ValueError, TypeError):
            pass

    if not manifest:
        return {}, ["manifest_path"]

    inputs: Dict[str, Any] = {"manifest_path": manifest}
    if course_dir:
        inputs["course_dir"] = course_dir
    return inputs, []


# ---------------------------------------------------------------------- #
# W1: Phase 3 / 3.5 / 4 Block-input + statistical-tier builders
# ---------------------------------------------------------------------- #


def _accepted_block_fields() -> frozenset:
    """Mirror the accepted-fields set in
    ``MCP/tools/pipeline_tools.py::_run_post_rewrite_validation``.

    Single source of truth for the hydration projection — every Block
    field the JSONL emit can carry. Unknown keys are silently dropped.
    """
    return frozenset({
        "block_id", "block_type", "page_id", "sequence", "content",
        "template_type", "key_terms", "objective_ids",
        "bloom_level", "bloom_verb", "bloom_range",
        "bloom_levels", "bloom_verbs", "cognitive_domain",
        "teaching_role", "content_type_label", "purpose",
        "component", "source_ids", "source_primary",
        "source_references", "content_hash",
        "validation_attempts", "escalation_marker",
    })


def _hydrate_blocks_from_path(blocks_path: Path) -> List[Any]:
    """Deserialise a ``blocks_*_path`` JSONL/JSON file into a List[Block].

    Mirrors ``_run_post_rewrite_validation::_entry_to_block`` so the
    workflow runner and the gate-input router agree on the projection
    rules. Malformed entries are dropped (logged at WARNING). Missing
    file → empty list.
    """
    try:
        from Courseforge.scripts.blocks import Block  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to import Courseforge.scripts.blocks.Block "
            "for hydration: %s",
            exc,
        )
        return []

    if blocks_path is None or not blocks_path.exists():
        return []

    try:
        raw_text = blocks_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", blocks_path, exc)
        return []

    raw_entries: List[Any] = []
    try:
        if blocks_path.suffix == ".jsonl":
            import json as _json
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw_entries.append(_json.loads(line))
        else:
            import json as _json
            parsed = _json.loads(raw_text)
            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, dict):
                inner = parsed.get("blocks")
                if isinstance(inner, list):
                    raw_entries = inner
    except (ValueError, OSError) as exc:
        logger.warning("Failed to parse %s: %s", blocks_path, exc)
        return []

    accepted = _accepted_block_fields()
    tuple_fields = {
        "key_terms", "objective_ids", "bloom_levels",
        "bloom_verbs", "source_ids", "source_references",
    }
    blocks: List[Any] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        cleaned: Dict[str, Any] = {}
        for k, v in entry.items():
            if k not in accepted:
                continue
            if k in tuple_fields and isinstance(v, list):
                if k == "source_references":
                    v = tuple(
                        dict(r) if isinstance(r, dict) else r for r in v
                    )
                else:
                    v = tuple(v)
            cleaned[k] = v
        if "block_id" not in cleaned or "block_type" not in cleaned:
            continue
        cleaned.setdefault("page_id", cleaned.get("block_id", ""))
        cleaned.setdefault("sequence", 0)
        cleaned.setdefault("content", "")
        try:
            blocks.append(Block(**cleaned))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Skipping malformed block entry block_id=%r: %s",
                entry.get("block_id"),
                exc,
            )
    return blocks


def _resolve_blocks_path_for_gate(
    gate_id: str,
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> Optional[Path]:
    """Pick the canonical blocks-source for a given gate_id.

    ``outline_*`` gates read ``content_generation_outline.blocks_outline_path``
    (Phase 3 inter-tier seam). ``rewrite_*`` gates read
    ``content_generation_rewrite.blocks_final_path`` (Phase 3.5 post-
    rewrite seam). Both fall back to explicit workflow_params overrides
    when the upstream phase output isn't present (e.g. Phase 5 stage
    subcommand re-runs).
    """
    gid = gate_id or ""
    is_rewrite = gid.startswith("rewrite_")
    if is_rewrite:
        cgr = phase_outputs.get("content_generation_rewrite") or {}
        candidate = (
            cgr.get("blocks_final_path")
            or workflow_params.get("blocks_final_path")
        )
    else:
        cgo = phase_outputs.get("content_generation_outline") or {}
        candidate = (
            cgo.get("blocks_outline_path")
            or workflow_params.get("blocks_outline_path")
        )
    if not candidate:
        return None
    try:
        return Path(candidate)
    except (TypeError, ValueError):
        return None


def _resolve_objectives_path(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> Optional[str]:
    """Find the canonical synthesized_objectives.json (course_planning emit)."""
    op = workflow_params.get("objectives_path")
    if isinstance(op, str) and op:
        return op
    located = _locate(
        phase_outputs,
        "objectives_path",
        "synthesized_objectives_path",
    )
    if located:
        return located
    # Derive from objective_extraction.project_path -> Courseforge exports.
    oe = phase_outputs.get("objective_extraction") or {}
    project_path = oe.get("project_path")
    if isinstance(project_path, str) and project_path:
        derived = (
            Path(project_path)
            / "01_learning_objectives"
            / "synthesized_objectives.json"
        )
        if derived.exists():
            return str(derived)
    return None


def _resolve_staging_manifest_path(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> Optional[str]:
    """Locate the DART staging_manifest.json BlockSourceRefValidator wants."""
    explicit = _locate(phase_outputs, "manifest_path")
    if explicit:
        # Prefer staging-side manifest_path; libv2_archival also emits
        # manifest_path but for the course manifest. Distinguish by
        # filename when possible.
        if explicit.endswith("staging_manifest.json"):
            return explicit
    staging = phase_outputs.get("staging") or {}
    mp = staging.get("manifest_path")
    if isinstance(mp, str) and mp:
        return mp
    staging_dir = (
        staging.get("staging_dir")
        or _locate(phase_outputs, "staging_dir")
        or workflow_params.get("staging_dir")
    )
    if isinstance(staging_dir, str) and staging_dir:
        derived = Path(staging_dir) / "staging_manifest.json"
        if derived.exists():
            return str(derived)
        return str(derived)  # validator handles missing-file gracefully
    return None


def _build_block_input(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
    *,
    gate_id: str = "",
) -> BuilderResult:
    """Group A — Block-input builder for the four ``Block*Validator``s.

    Surfaces ``{blocks, objectives_path?, manifest_path?, valid_objective_ids?,
    valid_source_ids?}`` so all four ``Courseforge.router.inter_tier_gates``
    Block validators see the inputs their ``validate()`` paths consult.

    Distinguishes ``outline_*`` vs ``rewrite_*`` gates via gate_id
    prefix and routes blocks_path resolution accordingly. The Phase
    ``inter_tier_validation`` / ``post_rewrite_validation`` helpers
    pass an explicit ``gate_id`` per validator dispatch, but the
    register layer doesn't carry the gate_id directly — we read it
    out of ``inputs.gate_id`` after the executor merges
    ``gate.config`` (see ``executor.py:1442``). Until then we infer
    from phase_outputs presence: rewrite path wins when both exist.
    """
    blocks_path = _resolve_blocks_path_for_gate(
        gate_id, phase_outputs, workflow_params,
    )
    if blocks_path is None:
        # Fall back: prefer rewrite-tier emit when present (post-rewrite
        # phase has both paths in phase_outputs), else outline.
        cgr = phase_outputs.get("content_generation_rewrite") or {}
        cgo = phase_outputs.get("content_generation_outline") or {}
        candidate = (
            cgr.get("blocks_final_path")
            or cgo.get("blocks_outline_path")
            or workflow_params.get("blocks_final_path")
            or workflow_params.get("blocks_outline_path")
        )
        if candidate:
            try:
                blocks_path = Path(candidate)
            except (TypeError, ValueError):
                blocks_path = None
    if blocks_path is None:
        return {}, ["blocks_outline_path|blocks_final_path"]
    if not blocks_path.exists():
        return {}, [f"blocks_path:{blocks_path}"]

    blocks = _hydrate_blocks_from_path(blocks_path)
    if not blocks:
        return {}, ["blocks (hydration produced 0 entries)"]

    inputs: Dict[str, Any] = {"blocks": blocks}

    objectives_path = _resolve_objectives_path(phase_outputs, workflow_params)
    if objectives_path:
        inputs["objectives_path"] = objectives_path

    seeded_objectives = workflow_params.get("valid_objective_ids")
    if seeded_objectives is not None:
        inputs["valid_objective_ids"] = seeded_objectives

    manifest_path = _resolve_staging_manifest_path(phase_outputs, workflow_params)
    if manifest_path:
        inputs["manifest_path"] = manifest_path
        # BlockSourceRefValidator looks at ``staging_dir`` only via
        # ``manifest_path`` resolution; surface staging_dir too for
        # downstream callers / debugging parity.
        staging = phase_outputs.get("staging") or {}
        sd = staging.get("staging_dir") or _locate(phase_outputs, "staging_dir")
        if sd:
            inputs["staging_dir"] = sd

    seeded_sources = workflow_params.get("valid_source_ids")
    if seeded_sources is not None:
        inputs["valid_source_ids"] = seeded_sources

    return inputs, []


def _build_block_input_outline(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Outline-tier registration shim — pins gate_id prefix to outline."""
    return _build_block_input(
        phase_outputs, workflow_params, gate_id="outline_",
    )


def _build_block_input_rewrite(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Rewrite-tier registration shim — pins gate_id prefix to rewrite."""
    return _build_block_input(
        phase_outputs, workflow_params, gate_id="rewrite_",
    )


def _build_rewrite_block_input(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Group B — Rewrite-emit shape / sentence-grounding builder.

    Adds ``source_chunks`` (a Dict[sourceId, chunk_text] mapping) on
    top of the Group A surface so ``RewriteSourceGroundingValidator``
    can compute per-sentence cosine grounding. ``RewriteHtmlShapeValidator``
    only consumes ``blocks`` and ignores the extra keys.

    The source_chunks mapping is rebuilt from the staging manifest +
    sidecar JSON files when present; when the staging surface is
    unavailable the validator's no-grounding-source path emits a
    warning per block (passed=True), so the absence is non-fatal.
    """
    inputs, missing = _build_block_input_rewrite(phase_outputs, workflow_params)
    if missing:
        return inputs, missing

    # Best-effort source_chunks rebuild from sidecar files alongside
    # the staging manifest. The sentence-grounding validator handles
    # an empty mapping gracefully (warning, passed=True per block).
    chunks_lookup: Dict[str, str] = {}
    manifest_path_str = inputs.get("manifest_path")
    if isinstance(manifest_path_str, str) and manifest_path_str:
        try:
            import json as _json
            mp = Path(manifest_path_str)
            if mp.exists():
                manifest = _json.loads(mp.read_text(encoding="utf-8"))
                files = manifest.get("files", []) or []
                if isinstance(files, list):
                    for entry in files:
                        if not isinstance(entry, dict):
                            continue
                        sid = entry.get("source_id") or entry.get("sourceId")
                        text = entry.get("text") or entry.get("plain_text")
                        if isinstance(sid, str) and isinstance(text, str):
                            chunks_lookup[sid] = text
        except (OSError, ValueError, TypeError) as exc:
            logger.debug(
                "rewrite-block source_chunks rebuild from %s failed: %s",
                manifest_path_str, exc,
            )
    if chunks_lookup:
        inputs["source_chunks"] = chunks_lookup
    return inputs, []


def _build_block_only_input(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Group C — block-only input for ``CourseforgeOutlineShaclValidator``.

    The SHACL validator's ``_coerce_block_payloads`` accepts either
    ``inputs['blocks']`` (preferred) or ``inputs['blocks_path']``. We
    surface ``blocks`` as Block dataclass instances; the validator
    silently skips non-dict entries via the dict / str dispatch in
    ``_extract_jsonld_blocks``, but Phase 4 PoC contract is
    informational severity so a partial drop just yields a warning.
    """
    # Prefer rewrite-tier blocks (post_rewrite_validation::rewrite_shacl)
    # when present, else outline-tier (inter_tier_validation::outline_shacl).
    inputs, missing = _build_block_input(
        phase_outputs, workflow_params, gate_id="rewrite_shacl",
    )
    if missing:
        # Try outline path explicitly.
        inputs, missing = _build_block_input(
            phase_outputs, workflow_params, gate_id="outline_shacl",
        )
        if missing:
            return {}, missing
    # Strip non-essential keys; SHACL validator only reads "blocks" /
    # "blocks_path".
    blocks_only: Dict[str, Any] = {"blocks": inputs.get("blocks", [])}
    return blocks_only, []


def _build_block_statistical_input(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Group D — Phase-4 statistical-tier builder.

    Surfaces ``{blocks, objectives_path}``. The executor merges
    ``gate.config`` (and therefore ``gate.config.thresholds``) into
    the inputs dict at ``executor.py:1442`` so the per-validator
    threshold dial flows through unchanged. Each validator additionally
    accepts ``objective_statements`` / ``concept_definitions`` /
    ``paraphrase_fn`` / ``embedder`` overrides via ``inputs.*``; the
    Phase 4 PoC contract degrades to ``passed=True`` warnings when
    those auxiliaries aren't wired.
    """
    inputs, missing = _build_block_input(
        phase_outputs, workflow_params, gate_id="rewrite_",
    )
    if missing:
        inputs, missing = _build_block_input(
            phase_outputs, workflow_params, gate_id="outline_",
        )
        if missing:
            return {}, missing
    # Statistical-tier validators consume only ``blocks`` +
    # ``objectives_path`` + the threshold inputs the executor merges
    # in via gate.config. Drop manifest/staging so the validator
    # doesn't see unrelated keys.
    pruned: Dict[str, Any] = {"blocks": inputs.get("blocks", [])}
    if inputs.get("objectives_path"):
        pruned["objectives_path"] = inputs["objectives_path"]
    return pruned, []


def _build_degraded_chunk_input(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Group E — fail-loud structured-skip for a YAML mis-pointing.

    The legacy chunk-shape ``CurieAnchoringValidator`` /
    ``ContentTypeValidator`` are wired at the Phase 3
    ``content_generation_outline`` validation_gates by a YAML mistake
    (the Block-shape variants live in
    ``Courseforge.router.inter_tier_gates``). Emitting a non-empty
    missing-list here surfaces the mismatch as a structured skip
    (``GATE_SKIPPED_MISSING_INPUTS``) instead of letting the no-builder
    fallthrough silently pass. W4 corrects the YAML; this builder is
    safety against future drift.
    """
    return {}, ["wrong_validator_class"]


def _build_assessment_objective_alignment(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 24: assessments path + chunks path builder.

    The Trainforge phase emits ``output_path`` for the assessments.json
    and produces ``chunks.jsonl`` under ``{trainforge_dir}/corpus/``.
    The trainforge_dir is the parent of the IMSCC's project dir —
    derive it conservatively from the assessments output path.
    """
    assessments = _locate(
        phase_outputs, "assessments_path", "assessment_path", "output_path",
    )
    if not assessments:
        return {}, ["assessments_path"]

    inputs: Dict[str, Any] = {"assessments_path": assessments}

    # Chunks live at ``{trainforge_dir}/imscc_chunks/chunks.jsonl``
    # (Phase 7c rename of the legacy ``corpus/`` directory). If the
    # phase output surfaces chunks_path explicitly, prefer that.
    chunks = _locate(phase_outputs, "chunks_path")
    if not chunks:
        # Derive from assessments path: walk up to find an
        # imscc_chunks/ (or legacy corpus/) sibling with chunks.jsonl,
        # or fallback to the same directory.
        try:
            ap = Path(assessments)
            for parent in [ap.parent, *ap.parents]:
                # Phase 7c: prefer imscc_chunks/, fall back to corpus/.
                for subdir in ("imscc_chunks", "corpus"):
                    candidate = parent / subdir / "chunks.jsonl"
                    if candidate.exists():
                        chunks = str(candidate)
                        break
                if chunks:
                    break
                # Also check a sibling chunks.jsonl.
                sib = parent / "chunks.jsonl"
                if sib.exists():
                    chunks = str(sib)
                    break
        except (OSError, ValueError):
            pass

    if not chunks:
        # Wave 29: fall back to the LibV2-archived corpus when
        # Trainforge didn't surface chunks_path directly. The
        # libv2_archival phase emits ``course_dir`` (and sometimes
        # ``course_slug``) for the archived course root; the
        # canonical location is
        # ``LibV2/courses/{slug}/imscc_chunks/chunks.jsonl`` post-Phase
        # 7c (or the legacy ``corpus/chunks.jsonl``) per
        # ``lib/libv2_storage.py``.
        archive_dir = _locate(phase_outputs, "course_dir")
        if archive_dir:
            try:
                from lib.libv2_storage import resolve_imscc_chunks_path
                candidate = resolve_imscc_chunks_path(
                    Path(archive_dir), "chunks.jsonl"
                )
                if candidate.exists():
                    chunks = str(candidate)
            except (OSError, ValueError, TypeError, ImportError):
                pass

    if chunks:
        inputs["chunks_path"] = chunks
        return inputs, []
    return inputs, ["chunks_path"]


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #


BuilderFn = Callable[[Dict[str, Any], Dict[str, Any]], BuilderResult]


@dataclass
class GateInputRouter:
    """Dispatches validator dotted paths to their input builders.

    Keyed on the validator's dotted import path (as it appears in
    ``config/workflows.yaml::validation_gates[].validator``). Adding a
    new validator is a single-line registry entry — no executor edits
    required.
    """

    builders: Dict[str, BuilderFn] = field(default_factory=dict)

    def register(self, validator_path: str, builder: BuilderFn) -> None:
        self.builders[validator_path] = builder

    def build(
        self,
        validator_path: str,
        phase_outputs: Dict[str, Any],
        workflow_params: Dict[str, Any],
    ) -> BuilderResult:
        """Look up + run the builder; return ({}, []) fallthrough on miss.

        Unknown validators fall through to the fallback ``artifacts``
        blob — this is the pre-Wave-23 behaviour and preserves graceful
        degradation when someone wires a new validator in YAML before
        registering a builder. The executor logs a warning when this
        happens so the drift is observable.
        """
        fn = self.builders.get(validator_path)
        if fn is None:
            logger.warning(
                "No gate-input builder registered for validator %s; "
                "falling back to artifacts blob (gate may skip)",
                validator_path,
            )
            return {}, ["__no_builder_registered__"]
        try:
            return fn(phase_outputs, workflow_params)
        except Exception as exc:  # noqa: BLE001 - builders never raise by contract
            logger.warning(
                "Gate-input builder %s raised: %s; marking gate as skipped",
                validator_path,
                exc,
            )
            return {}, ["__builder_error__"]


def default_router() -> GateInputRouter:
    """Return a router pre-populated with every validator shipping today."""
    r = GateInputRouter()
    r.register(
        "lib.validators.content.ContentStructureValidator",
        _build_content_structure,
    )
    r.register(
        "lib.validators.page_objectives.PageObjectivesValidator",
        _build_page_objectives,
    )
    # Phase 4 PoC: SHACL parallel of page_objectives. Reuses the
    # Python-validator's input contract (content_dir + objectives_path)
    # verbatim so workflow-config drift is impossible.
    r.register(
        "lib.validators.shacl_runner.PageObjectivesShaclValidator",
        _build_page_objectives,
    )
    r.register(
        "lib.validators.source_refs.PageSourceRefValidator",
        _build_source_refs,
    )
    r.register(
        "lib.validators.imscc.IMSCCValidator",
        _build_imscc,
    )
    r.register(
        "lib.validators.imscc.IMSCCParseValidator",
        _build_imscc,
    )
    r.register(
        "DART.pdf_converter.wcag_validator.WCAGValidator",
        _build_wcag,
    )
    r.register(
        "lib.validators.oscqr.OSCQRValidator",
        _build_oscqr,
    )
    r.register(
        "lib.validators.dart_markers.DartMarkersValidator",
        _build_dart_markers,
    )
    r.register(
        "lib.validators.assessment.AssessmentQualityValidator",
        _build_assessment_quality,
    )
    r.register(
        "lib.validators.assessment.FinalQualityValidator",
        _build_final_quality,
    )
    r.register(
        "lib.validators.bloom.BloomAlignmentValidator",
        _build_bloom_alignment,
    )
    r.register(
        "lib.validators.leak_check.LeakCheckValidator",
        _build_leak_check,
    )
    r.register(
        "lib.validators.content_facts.ContentFactValidator",
        _build_content_facts,
    )
    r.register(
        "lib.validators.question_quality.QuestionQualityValidator",
        _build_question_quality,
    )
    r.register(
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
        _build_libv2_manifest,
    )
    # Wave 78: packet integrity validator (gates the libv2_archival
    # phase fail-closed). Reuses the same input shape as the manifest
    # validator (course_dir + manifest_path) — the validator's gate
    # adapter resolves archive_root from either.
    r.register(
        "lib.validators.libv2_packet_integrity.PacketIntegrityValidator",
        _build_libv2_manifest,
    )
    r.register(
        "lib.validators.assessment_objective_alignment.AssessmentObjectiveAlignmentValidator",
        _build_assessment_objective_alignment,
    )
    # Wave 31: content grounding — verifies Courseforge content traces
    # back to DART source blocks. The builder lives in the validator
    # module so routing stays co-located with the check.
    try:
        from lib.validators.content_grounding import _build_content_grounding
        r.register(
            "lib.validators.content_grounding.ContentGroundingValidator",
            _build_content_grounding,
        )
    except ImportError:  # pragma: no cover
        # Keep router functional even when the validator import fails.
        logger.warning("content_grounding validator import failed")

    # ------------------------------------------------------------------ #
    # W1 — Phase 3 / 3.5 / 4 Courseforge two-pass validator wiring.
    # Closes the no-builder fallthrough that stamped these gates
    # passed=True via waiver_info["skipped"]="true". 13 validators
    # split into five input-shape groups; one helper per group.
    # ------------------------------------------------------------------ #

    # Group A — four Block-input validators (rewrite_* gates pull
    # blocks_final_path; outline-tier seam reuses the same builder via
    # the outline_* shim).
    r.register(
        "Courseforge.router.inter_tier_gates.BlockCurieAnchoringValidator",
        _build_block_input_rewrite,
    )
    r.register(
        "Courseforge.router.inter_tier_gates.BlockContentTypeValidator",
        _build_block_input_rewrite,
    )
    r.register(
        "Courseforge.router.inter_tier_gates.BlockPageObjectivesValidator",
        _build_block_input_rewrite,
    )
    r.register(
        "Courseforge.router.inter_tier_gates.BlockSourceRefValidator",
        _build_block_input_rewrite,
    )
    # Worker W7: assessment_item payload-shape gate. Same Block-input
    # surface as the four Block*Validators above (filters to
    # block_type == "assessment_item" internally), so it reuses the
    # rewrite-tier shim — the inter_tier_validation seam falls through
    # to the outline-tier path inside ``_build_block_input`` when only
    # blocks_outline_path is present.
    r.register(
        "lib.validators.assessment_item_payload.BlockAssessmentItemPayloadValidator",
        _build_block_input_rewrite,
    )

    # Group B — Rewrite-emit shape + sentence-grounding gates. Reuse
    # the rewrite-tier Block surface and additionally surface
    # source_chunks from the staging manifest.
    r.register(
        "lib.validators.rewrite_html_shape.RewriteHtmlShapeValidator",
        _build_block_input_rewrite,
    )
    r.register(
        "lib.validators.rewrite_source_grounding.RewriteSourceGroundingValidator",
        _build_rewrite_block_input,
    )

    # Group C — Block-only SHACL validator (one binding wired at both
    # outline and rewrite seams in YAML; same builder routes both).
    r.register(
        "lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator",
        _build_block_only_input,
    )

    # Group D — Phase-4 statistical-tier validators (objective ↔
    # assessment cosine; concept ↔ example cosine; objective paraphrase
    # roundtrip cosine; BERT-ensemble Bloom disagreement). Each is
    # wired symmetrically at outline + rewrite seams; gate.config
    # thresholds flow through via the executor's :1442 merge.
    r.register(
        "lib.validators.objective_assessment_similarity.ObjectiveAssessmentSimilarityValidator",
        _build_block_statistical_input,
    )
    r.register(
        "lib.validators.concept_example_similarity.ConceptExampleSimilarityValidator",
        _build_block_statistical_input,
    )
    r.register(
        "lib.validators.objective_roundtrip_similarity.ObjectiveRoundtripSimilarityValidator",
        _build_block_statistical_input,
    )
    r.register(
        "lib.validators.bloom_classifier_disagreement.BloomClassifierDisagreementValidator",
        _build_block_statistical_input,
    )

    # Group E — degraded fail-loud entries. The chunk-shape
    # CurieAnchoringValidator / ContentTypeValidator are wired at the
    # Phase 3 outline gates by a YAML misnomer (the Block-shape
    # variants live under Courseforge.router.inter_tier_gates). W4
    # corrects the YAML; until then these entries surface the mismatch
    # as a structured GATE_SKIPPED_MISSING_INPUTS skip rather than a
    # silent no-builder pass. After W4 the YAML stops pointing here,
    # but the registrations stay as fail-loud safety against drift.
    r.register(
        "lib.validators.curie_anchoring.CurieAnchoringValidator",
        _build_degraded_chunk_input,
    )
    r.register(
        "lib.validators.content_type.ContentTypeValidator",
        _build_degraded_chunk_input,
    )

    return r


__all__ = [
    "BuilderFn",
    "BuilderResult",
    "GateInputRouter",
    "default_router",
]
