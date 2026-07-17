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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Repo root: this file is ``<repo>/MCP/hardening/gate_input_routing.py``,
# so ``parents[2]`` is the Ed4All root. Used by the on-disk
# domain-vocabulary fallback in ``_resolve_minted_curie_map`` (R5).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #


# Canonical Courseforge content layouts, in priority order. The
# textbook_to_course pipeline emits generated weekly pages under
# ``<project_export>/03_content_development/week_NN/*.html`` (the
# Courseforge two-pass packager's layout). Legacy / packaged courses
# place flattened HTML directly under ``<project_export>/content/``.
# The disk-glob fallback in ``_find_content_dir`` / ``_all_html_paths``
# tries each subdir against an export root resolved from phase outputs.
_CONTENT_EXPORT_SUBDIRS = ("03_content_development", "content")


def _find_project_export_dir(
    phase_outputs: Dict[str, Any],
    workflow_params: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Resolve the Courseforge project-export root from available signals.

    Used by the disk-glob fallback in :func:`_find_content_dir` /
    :func:`_all_html_paths` when no ``content_dir`` / ``content_paths``
    surfaced. The export root is the directory that contains the
    canonical ``NN_<stage>/`` subdirs (``01_learning_objectives/``,
    ``03_content_development/``, ...).

    Resolution chain (high → low); first extant directory wins:

    * ``objective_extraction.project_path`` (the canonical export root
      threaded by the staging / extraction phases).
    * any ``project_path`` / ``project_export`` / ``export_dir`` /
      ``project_dir`` / ``project_export_dir`` key surfaced by ANY
      phase output (``_locate``).
    * the same keys from ``workflow_params``.
    """
    candidates: List[str] = []

    oe = phase_outputs.get("objective_extraction") or {}
    pp = oe.get("project_path")
    if isinstance(pp, str) and pp:
        candidates.append(pp)

    located = _locate(
        phase_outputs,
        "project_path",
        "project_export",
        "export_dir",
        "project_dir",
        "project_export_dir",
    )
    if located:
        candidates.append(located)

    if workflow_params:
        for key in (
            "project_path",
            "project_export",
            "export_dir",
            "project_dir",
            "project_export_dir",
        ):
            val = workflow_params.get(key)
            if isinstance(val, str) and val:
                candidates.append(val)

    for cand in candidates:
        try:
            p = Path(cand)
        except (TypeError, ValueError):
            continue
        if p.exists() and p.is_dir():
            return p

    return None


def _glob_content_dir_from_export(export_dir: Path) -> Optional[Path]:
    """Return the first canonical content subdir under ``export_dir``.

    Tries each entry in :data:`_CONTENT_EXPORT_SUBDIRS`; a subdir wins
    when it exists AND contains at least one ``*.html`` file at any
    depth (handles both the flat ``content/*.html`` layout and the
    nested ``03_content_development/week_NN/*.html`` layout).
    """
    if not export_dir or not export_dir.is_dir():
        return None
    for sub in _CONTENT_EXPORT_SUBDIRS:
        cand = export_dir / sub
        try:
            if cand.is_dir() and next(cand.rglob("*.html"), None) is not None:
                return cand
        except OSError:
            continue
    return None


def _find_content_dir(
    phase_outputs: Dict[str, Any],
    workflow_params: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Locate a content_dir candidate from accumulated phase outputs.

    Courseforge's content-generation phase emits ``content_paths`` as a
    comma-joined list of generated HTML paths under a
    ``.../content/`` directory. The ``content_dir`` is the common
    parent. When the phase exposes a ``project_path`` (pre-Wave-8
    shape) we prefer ``project_path / "content"`` to match the
    packager's layout.

    Disk-glob fallback (additive, backward-compatible): when none of
    the existing resolution arms match — which happens for
    ``textbook_to_course`` runs whose generated pages live at
    ``<export>/03_content_development/week_NN/*.html`` and whose
    content-generation phase output carries no ``content_paths`` (e.g.
    when the phase is dispatched to subagents) — derive the project
    export root from phase outputs / ``workflow_params`` and glob the
    canonical content subdirs (``03_content_development/`` then legacy
    ``content/``). The fallback fires ONLY when the legacy arms return
    ``None``, so existing runs are byte-identical.
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

    # Disk-glob FALLBACK — reached only when every arm above missed.
    # Resolve the project export root and probe the canonical content
    # subdirs. Handles the textbook_to_course
    # ``<export>/03_content_development/week_NN/*.html`` layout that the
    # legacy arms can't see when content_paths isn't surfaced.
    export_dir = _find_project_export_dir(phase_outputs, workflow_params)
    if export_dir is not None:
        globbed = _glob_content_dir_from_export(export_dir)
        if globbed is not None:
            return globbed

    return None


def _walk_html_paths(content_dir: Path) -> List[Path]:
    """Return all .html files under content_dir (deterministic order)."""
    if not content_dir or not content_dir.exists():
        return []
    return sorted(content_dir.rglob("*.html"))


def _first_html_path(
    phase_outputs: Dict[str, Any],
    workflow_params: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Locate a single html_path candidate for validators that need one.

    DART output paths surface as ``output_path`` (single) or
    ``output_paths`` (comma-joined). Falls back to walking the
    discovered content_dir when no DART outputs are present.
    ``workflow_params`` (optional) is threaded into ``_find_content_dir``
    so the disk-glob export-root fallback can fire.
    """
    # DART->semantik purge Stage 1 (dual-READ): accept the future
    # ``semantik_conversion`` phase-output key as an alias of ``dart_conversion``.
    dc = (
        phase_outputs.get("dart_conversion")
        or phase_outputs.get("semantik_conversion")
        or {}
    )
    op = dc.get("output_path")
    if isinstance(op, str) and op:
        return Path(op)
    ops = dc.get("output_paths")
    if isinstance(ops, str) and ops:
        first = ops.split(",")[0].strip()
        if first:
            return Path(first)

    cd = _find_content_dir(phase_outputs, workflow_params)
    htmls = _walk_html_paths(cd) if cd else []
    return htmls[0] if htmls else None


def _all_html_paths(
    phase_outputs: Dict[str, Any],
    workflow_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return a list of HTML page paths derivable from phase outputs.

    ``workflow_params`` (optional) is threaded into ``_find_content_dir``
    so the disk-glob export-root fallback fires when neither DART
    outputs nor ``content_generation.content_paths`` surfaced — e.g. a
    ``textbook_to_course`` run whose generated pages live under
    ``<export>/03_content_development/week_NN/*.html`` and whose
    content-generation phase output carries no content paths.
    """
    # Prefer generated COURSE CONTENT pages when they exist. This must come
    # BEFORE the DART-source arm: content-phase gates (source_refs,
    # content_grounding) need the content-generator pages (which carry
    # ``data-cf-source-ids``), NOT the DART staged HTML (``data-dart-*``).
    # At the dart_conversion phase no content dir resolves yet, so the DART
    # arm below still serves the dart_markers gate — order is safe.
    cg = phase_outputs.get("content_generation") or {}
    cps = cg.get("content_paths")
    if isinstance(cps, str) and cps:
        return [p.strip() for p in cps.split(",") if p.strip()]

    cd = _find_content_dir(phase_outputs, workflow_params)
    if cd:
        walked = [str(p) for p in _walk_html_paths(cd)]
        if walked:
            return walked

    # Fallback: DART conversion outputs (serves the DART-phase dart_markers
    # gate, and content-less workflows). DART->semantik purge Stage 1
    # (dual-READ): accept the future ``semantik_conversion`` phase-output key.
    dc = (
        phase_outputs.get("dart_conversion")
        or phase_outputs.get("semantik_conversion")
        or {}
    )
    ops = dc.get("output_paths")
    if isinstance(ops, str) and ops:
        return [p.strip() for p in ops.split(",") if p.strip()]
    op = dc.get("output_path")
    if isinstance(op, str) and op:
        return [op]
    return []


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


def _chunking_chunks_path(chunking: Dict[str, Any]) -> Optional[str]:
    """Dual-read the ``chunking`` phase's chunkset-path envelope key.

    DART->semantik naming purge Stage 3c: the ``chunking`` emitter surfaces the
    ratified ``semantik_chunks_path`` key; fall back to the legacy
    ``dart_chunks_path`` so a run whose chunking phase_outputs were checkpointed
    under the old key (pre-3c resume sidecar) still resolves. Mirrors the
    on-disk ``resolve_imscc_chunks_dir`` dual-read for the emit-dir rename.
    """
    for key in ("semantik_chunks_path", "dart_chunks_path"):
        val = chunking.get(key)
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
    html = _first_html_path(phase_outputs, workflow_params)
    if html and html.exists():
        return {"html_path": str(html)}, []
    # No HTML available — must be skipped, not passed.
    return {}, ["html_path"]


def _build_page_objectives(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    content_dir = _find_content_dir(phase_outputs, workflow_params)
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
    pages = _all_html_paths(phase_outputs, workflow_params)
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
    html = _first_html_path(phase_outputs, workflow_params)
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
    content_dir = _find_content_dir(phase_outputs, workflow_params)
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
    all_paths = _all_html_paths(phase_outputs, workflow_params)
    existing = [Path(p) for p in all_paths if Path(p).exists()]
    if not existing:
        # One last fallback: try the single html_path helper (walks
        # content_dir when DART outputs are absent).
        single = _first_html_path(phase_outputs, workflow_params)
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
        # FR-INT-05 / FR-A11Y-03 — additive Optional Block fields the
        # CalloutStructureValidator / InteractionFeedbackValidator read off
        # hydrated blocks. Default None → byte-stable when absent from the JSONL.
        "feedback", "option_feedback", "callout_kind",
        # FR-INT-04 — B10 three-move discussion protocol fields the
        # B10ProtocolValidator reads. Default None → byte-stable when absent.
        "discussion_protocol", "discussion_bloom_verb",
        # FR-INT-03 — B11 predict-then-reveal calibration fields the
        # InteractionFeedbackValidator (REFLECTION_NO_CAPTURE) +
        # AnatomySlotPresenceValidator (benchmark-in-feedback) read. Default None
        # → byte-stable when absent from the JSONL.
        "prediction_prompt", "reveal_content", "calibration_feedback",
        # IB1 anatomy slots the AnatomySlotPresenceValidator reads (the benchmark
        # check inspects the feedback slot). heading/purpose_tag/interaction/
        # transition complete the six-slot read surface; default None → byte-stable.
        "heading", "purpose_tag", "interaction", "transition",
    })


def _touches_from_jsonl_entry(entry: Dict[str, Any]) -> tuple:
    """Reconstruct the ``touched_by`` Touch tuple from a JSONL entry.

    Mirrors ``MCP/tools/pipeline_tools.py::_touches_from_entry`` so the
    workflow-runner gate-input router and the standalone validation handlers
    rehydrate the touch chain identically — without it, blocks hydrated here
    for the manifest-completeness gate carry no provenance and the per-block
    synthesis manifest would emit ``synthesis.tiers:[]``. Tolerant: a
    malformed Touch dict is dropped, not fatal. Empty tuple when absent.
    """
    raw = entry.get("touched_by")
    if not isinstance(raw, list):
        return ()
    try:
        from Courseforge.scripts.blocks import Touch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return ()
    out = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        try:
            out.append(Touch(
                model=d.get("model", ""),
                provider=d.get("provider", ""),
                tier=d.get("tier", ""),
                timestamp=d.get("timestamp", ""),
                decision_capture_id=d.get("decision_capture_id", ""),
                purpose=d.get("purpose", ""),
            ))
        except (TypeError, ValueError):
            continue
    return tuple(out)


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
        _touches = _touches_from_jsonl_entry(entry)
        if _touches:
            cleaned["touched_by"] = _touches
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


def _build_chunkset_drift(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 3 W3.D — input builder for ChunksetDriftValidator.

    Surfaces ``{dart_chunks_path, imscc_chunks_path, course_path}``.
    The validator compares two chunksets emitted by the chunking +
    imscc_chunking phases respectively, then writes a sidecar
    ``drift_report.json`` next to the LibV2 course root.

    Resolution chain (high → low):

    * Explicit ``dart_chunks_path`` / ``imscc_chunks_path`` keys in
      phase outputs (the chunking / imscc_chunking phases emit these).
    * Derive both from the libv2_archival ``course_dir`` (Wave 23
      contract). Canonical post-Phase-7c layout:
      ``<course_dir>/dart_chunks/chunks.jsonl`` +
      ``<course_dir>/imscc_chunks/chunks.jsonl``.
    * Fall back to the Phase-7c shim ``<course_dir>/corpus/chunks.jsonl``
      for the IMSCC side on legacy archives the
      ``backfill_dart_chunks.py`` migration hasn't touched yet
      (lib.libv2_storage.resolve_imscc_chunks_path handles the
      shim warning + fallback).

    The validator's ``MISSING_CHUNKSET`` arm fires when either path
    fails to resolve to an extant file — that's the validator's
    responsibility, not the router's. We surface non-existent
    candidate paths so the validator can emit a structured warning
    rather than letting the router silently skip the gate.
    """
    inputs: Dict[str, Any] = {}

    # Course-path preference order: libv2_archival.course_dir > any
    # explicit course_dir key in phase outputs > workflow_params.
    course_dir_str = _locate(phase_outputs, "course_dir")
    if not course_dir_str:
        course_dir_str = workflow_params.get("course_dir")
    course_dir = Path(course_dir_str) if course_dir_str else None
    if course_dir is not None:
        inputs["course_path"] = str(course_dir)

    # DART chunkset resolution.
    dart_chunks_path: Optional[str] = None
    chunking = phase_outputs.get("chunking") or {}
    candidate = _chunking_chunks_path(chunking)
    if isinstance(candidate, str) and candidate:
        dart_chunks_path = candidate
    if not dart_chunks_path and course_dir is not None:
        # DART->semantik purge Stage 3c: prefer the ratified on-disk
        # ``semantik_chunks/`` dir (NEW conversions), falling back to the legacy
        # ``dart_chunks/`` (un-migrated archives). This branch is DART-lineage
        # specific — it must NOT resolve an ``imscc_chunks/`` sibling on a
        # dual-chunkset course, so it does not route through the imscc-first
        # ``resolve_imscc_chunks_dir``. The final fallback (neither dir present)
        # keeps the canonical ratified path so a clean FileNotFoundError names it.
        semantik_candidate = course_dir / "semantik_chunks" / "chunks.jsonl"
        dart_candidate = course_dir / "dart_chunks" / "chunks.jsonl"
        if semantik_candidate.exists():
            dart_chunks_path = str(semantik_candidate)
        elif dart_candidate.exists():
            dart_chunks_path = str(dart_candidate)
        else:
            dart_chunks_path = str(semantik_candidate)

    # IMSCC chunkset resolution.
    imscc_chunks_path: Optional[str] = None
    imscc_chunking = phase_outputs.get("imscc_chunking") or {}
    candidate = imscc_chunking.get("imscc_chunks_path")
    if isinstance(candidate, str) and candidate:
        imscc_chunks_path = candidate
    if not imscc_chunks_path:
        # Fall back to chunks_path emitted by other phases.
        located = _locate(phase_outputs, "imscc_chunks_path")
        if located:
            imscc_chunks_path = located
    if not imscc_chunks_path and course_dir is not None:
        try:
            from lib.libv2_storage import resolve_imscc_chunks_path
            imscc_chunks_path = str(
                resolve_imscc_chunks_path(course_dir, "chunks.jsonl")
            )
        except (ImportError, OSError, ValueError, TypeError):
            # Fall through to the canonical post-Phase-7c path.
            imscc_chunks_path = str(
                course_dir / "imscc_chunks" / "chunks.jsonl"
            )

    missing: List[str] = []
    if dart_chunks_path:
        inputs["dart_chunks_path"] = dart_chunks_path
    else:
        missing.append("dart_chunks_path")
    if imscc_chunks_path:
        inputs["imscc_chunks_path"] = imscc_chunks_path
    else:
        missing.append("imscc_chunks_path")

    return inputs, missing


def _build_objective_source_refs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 1.6 W1.6.C — input builder for ObjectiveSourceRefValidator.

    Surfaces ``{synthesized_objectives_path, textbook_structure_path?,
    dart_chunks_manifest_path?, require_to_attribution?}`` so the
    validator's ``validate()`` path sees the inputs the per-objective
    walk consults.

    Resolution chain:

    * ``synthesized_objectives_path`` — required. Pulls from
      ``phase_outputs.course_planning.synthesized_objectives_path``
      (or ``workflow_params.objectives_path`` as the override) via
      :func:`_resolve_objectives_path`. Missing → builder returns the
      structured ``synthesized_objectives_path`` missing-key marker so
      the executor surfaces a GATE_SKIPPED_MISSING_INPUTS skip rather
      than crashing the validator.
    * ``textbook_structure_path`` — optional. Pulls from
      ``phase_outputs.objective_extraction.textbook_structure_path``;
      absent → graceful-degrade arm of the validator engages
      (``OBJECTIVE_SOURCE_REFS_NO_UNIVERSE`` warning when chunks
      universe is also absent).
    * ``dart_chunks_manifest_path`` — optional. The chunking phase
      emits ``dart_chunks_path`` (the path to ``chunks.jsonl``); the
      validator's contract is the SIBLING ``manifest.json`` in the
      same directory, so we derive that from the chunks-jsonl parent.
      Absent / unreadable → chunks-id resolution skipped on a per-LO
      basis (the validator handles this gracefully).
    * ``require_to_attribution`` — optional, default false. Surfaces
      from gate config (the meta-schema accepts a ``config:`` block at
      the gate level which the executor merges into inputs at
      ``MCP/core/executor.py:1442``). Wave 1.6 day-1 default keeps TOs
      from firing OBJECTIVE_MISSING_SOURCE_REFS.
    """
    inputs: Dict[str, Any] = {}

    objectives_path = _resolve_objectives_path(phase_outputs, workflow_params)
    if not objectives_path:
        # Mark structured-skip so the executor surfaces the missing
        # path through GATE_SKIPPED_MISSING_INPUTS rather than letting
        # the validator return a critical-severity miss against an
        # empty input.
        return {}, ["synthesized_objectives_path"]
    inputs["synthesized_objectives_path"] = objectives_path

    # textbook_structure_path — pulled directly from objective_extraction.
    oe = phase_outputs.get("objective_extraction") or {}
    ts_path = oe.get("textbook_structure_path")
    if isinstance(ts_path, str) and ts_path:
        inputs["textbook_structure_path"] = ts_path

    # dart_chunks_manifest_path — derive from chunking.dart_chunks_path
    # (the validator wants the manifest sidecar; chunks.jsonl sits in
    # the same dir).
    chunking = phase_outputs.get("chunking") or {}
    chunks_jsonl_raw = _chunking_chunks_path(chunking)
    if isinstance(chunks_jsonl_raw, str) and chunks_jsonl_raw:
        try:
            chunks_jsonl_path = Path(chunks_jsonl_raw)
            manifest_candidate = chunks_jsonl_path.parent / "manifest.json"
            inputs["dart_chunks_manifest_path"] = str(manifest_candidate)
        except (TypeError, ValueError):
            pass

    return inputs, []


def _build_manifest_completeness(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """W3 — input builder for ManifestCompletenessValidator.

    Surfaces ``{manifest_path, dart_chunks_manifest_path?,
    content_development_dir?}`` so the validator's RESOLUTION walk sees the
    §1.3a sidecar + the DART chunkset resolution universe. Models on
    :func:`_build_objective_source_refs` (which derives the same
    ``dart_chunks_manifest_path`` from ``chunking.dart_chunks_path``).

    Resolution chain:

    * ``manifest_path`` — the ``block_synthesis_manifest.jsonl`` sidecar the
      rewrite-tier emit writes at
      ``<content_dir>/block_synthesis_manifest.jsonl``. Derived from the resolved
      content_dir (the disk-glob export-root fallback handles the
      ``03_content_development/`` layout). The validator handles a missing
      manifest itself (graceful pass when no blocks; ``MANIFEST_FILE_MISSING``
      critical when ``03_content_development/`` is populated), so we ALWAYS
      surface a candidate ``manifest_path`` even when the file isn't present yet
      — never a structured skip — so the anti-silent-degradation guard can fire.
    * ``content_development_dir`` — the content_dir itself, so the validator's
      ``MANIFEST_FILE_MISSING`` guard knows blocks were produced.
    * ``dart_chunks_manifest_path`` — the sibling ``manifest.json`` of the DART
      chunkset ``chunks.jsonl`` (``chunking.dart_chunks_path`` parent), the
      resolution universe. Absent → the validator's empty-universe guard fires
      only when the manifest declares ids.
    """
    inputs: Dict[str, Any] = {}

    content_dir = _find_content_dir(phase_outputs, workflow_params)
    if content_dir is None:
        # No content dir resolves yet (dry-run / pre-content). Structured skip;
        # the validator's graceful-pass arm (no manifest + no blocks) is the
        # safety net for runs that reach the gate with neither.
        return {}, ["content_dir"]

    inputs["content_development_dir"] = str(content_dir)
    inputs["manifest_path"] = str(
        content_dir / "block_synthesis_manifest.jsonl"
    )

    # dart_chunks_manifest_path — same derivation as _build_objective_source_refs.
    chunking = phase_outputs.get("chunking") or {}
    chunks_jsonl_raw = _chunking_chunks_path(chunking)
    if not (isinstance(chunks_jsonl_raw, str) and chunks_jsonl_raw):
        chunks_jsonl_raw = _locate(phase_outputs, "semantik_chunks_path", "dart_chunks_path")
    if isinstance(chunks_jsonl_raw, str) and chunks_jsonl_raw:
        try:
            manifest_candidate = Path(chunks_jsonl_raw).parent / "manifest.json"
            inputs["dart_chunks_manifest_path"] = str(manifest_candidate)
        except (TypeError, ValueError):
            pass

    return inputs, []


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

    # IB4.6 — thread the ED4ALL_BLOCK_A11Y resolution into the Block-input
    # surface so RewriteHtmlShapeValidator's per-block a11y sub-check (IB4.1)
    # fires only when the flag is on. Harmless for the other Block*Validators
    # sharing this builder — they ignore the key. Default OFF → byte-stable.
    try:
        from lib.generation.block_a11y import resolve_block_a11y

        inputs["block_a11y_enabled"] = resolve_block_a11y()
    except Exception:  # noqa: BLE001 — never let the resolver import break routing
        inputs["block_a11y_enabled"] = False

    # IB5.8 — thread the ED4ALL_NEW_BLOCK_TYPES resolution into the Block-input
    # surface so RewriteHtmlShapeValidator's IB5 B04/B06 a11y-shape arms (IB5.7)
    # fire only when the flag is on. Harmless for the other Block*Validators
    # sharing this builder — they ignore the key. Default OFF → byte-stable.
    try:
        from lib.generation.new_block_types import resolve_new_block_types

        inputs["new_block_types_enabled"] = resolve_new_block_types()
    except Exception:  # noqa: BLE001 — never let the resolver import break routing
        inputs["new_block_types_enabled"] = False

    # FR-INT-03 — thread the ED4ALL_REFLECTION_CALIBRATION resolution into the
    # Block-input surface so the InteractionFeedbackValidator's REFLECTION_NO_
    # CAPTURE arm + the AnatomySlotPresenceValidator's benchmark-in-feedback
    # check fire only when the flag is on. Harmless for the other validators
    # sharing this builder — they ignore the key. Default OFF → byte-stable.
    try:
        from lib.generation.reflection_calibration import (
            resolve_reflection_calibration,
        )

        inputs["reflection_calibration_enabled"] = resolve_reflection_calibration()
    except Exception:  # noqa: BLE001 — never let the resolver import break routing
        inputs["reflection_calibration_enabled"] = False

    # Mayer-CTML — thread the ED4ALL_MAYER_CTML resolution into the Block-input
    # surface so MayerCtmlValidator's signaling/contiguity/redundancy/segmenting
    # arms fire only when the flag is on. Harmless for the other validators
    # sharing this builder — they ignore the key. Default OFF → byte-stable.
    try:
        from lib.validators.mayer_ctml import resolve_mayer_ctml

        inputs["mayer_ctml_enabled"] = resolve_mayer_ctml()
    except Exception:  # noqa: BLE001 — never let the resolver import break routing
        inputs["mayer_ctml_enabled"] = False

    # recall_self_check — thread the ED4ALL_RECALL_SELF_CHECK resolution into the
    # Block-input surface so RecallSelfCheckFormatValidator fires only when the
    # flag is on. Harmless for the other validators sharing this builder — they
    # ignore the key. Default OFF → byte-stable.
    try:
        from lib.generation.recall_self_check import resolve_recall_self_check

        inputs["recall_self_check_enabled"] = resolve_recall_self_check()
    except Exception:  # noqa: BLE001 — never let the resolver import break routing
        inputs["recall_self_check_enabled"] = False

    # misconception_rich — thread the ED4ALL_MISCONCEPTION_RICH resolution into
    # the Block-input surface so MisconceptionProductiveFailureValidator fires
    # only when the flag is on. Harmless for the other validators sharing this
    # builder — they ignore the key. Default OFF → byte-stable.
    try:
        from lib.generation.misconception_rich import resolve_misconception_rich

        inputs["misconception_rich_enabled"] = resolve_misconception_rich()
    except Exception:  # noqa: BLE001 — never let the resolver import break routing
        inputs["misconception_rich_enabled"] = False

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


def _resolve_minted_curie_map(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build the minted-CURIE map from the Stage-3 domain vocabulary.

    R5 — delegates to the single canonical resolver
    :func:`lib.ontology.curie_discovery.resolve_minted_curie_map`, which
    INCLUDES the on-disk fallback. Before R5 this copy lacked that
    fallback while the ``pipeline_tools`` validation-handler copy had
    it, so a resumed / stage-subcommand run (no
    ``phase_outputs.concept_extraction`` thread) gave the same logical
    gate two different verdicts. Both call sites now resolve identically.

    Returns ``None`` when no vocabulary path is threaded / on disk (RDF
    / legacy corpora) or when the file is missing / unparseable — the
    consuming gate then runs its legacy literal-token anchoring.
    """
    from lib.ontology.curie_discovery import resolve_minted_curie_map

    ce = phase_outputs.get("concept_extraction") or {}
    candidate = ce.get("domain_concept_vocabulary_path")
    if not isinstance(candidate, str) or not candidate:
        candidate = _locate(phase_outputs, "domain_concept_vocabulary_path")

    course_id = workflow_params.get("course_name") or workflow_params.get(
        "course_code"
    )
    # On-disk fallback inputs: derive the slug (lower + _/space -> -,
    # mirroring _run_concept_extraction) and resolve the LibV2 root
    # (per-run param > ED4ALL_LIBV2_ROOT env > in-tree default).
    course_slug = ""
    if isinstance(course_id, str) and course_id:
        course_slug = course_id.lower().replace("_", "-").replace(" ", "-")
    libv2_root_raw = (
        workflow_params.get("libv2_root")
        or os.environ.get("ED4ALL_LIBV2_ROOT", "").strip()
    )
    if libv2_root_raw:
        libv2_root: Optional[Path] = Path(libv2_root_raw)
    else:
        libv2_root = _PROJECT_ROOT / "LibV2"

    return resolve_minted_curie_map(
        threaded_path=candidate if isinstance(candidate, str) else None,
        course_id=course_id,
        course_slug=course_slug or None,
        libv2_root=libv2_root,
    )


def _build_block_curie_anchoring_input(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Builder for ``BlockCurieAnchoringValidator`` (outline + rewrite).

    Reuses the Group-A rewrite-tier Block surface, then threads in the
    optional ``minted_curie_map`` so the validator can anchor a minted
    (prose-corpus) CURIE via vocabulary surface forms. When no domain
    vocabulary exists, ``minted_curie_map`` is omitted and the validator
    runs byte-identical legacy literal-token anchoring.
    """
    inputs, missing = _build_block_input_rewrite(phase_outputs, workflow_params)
    if missing:
        return inputs, missing
    minted = _resolve_minted_curie_map(phase_outputs, workflow_params)
    if minted:
        inputs["minted_curie_map"] = minted
    # Thread the per-block grounded source-chunk universe (the
    # ``outline_chunks.json`` sidecar emitted by
    # ``_run_content_generation_outline``) so the validator can anchor a
    # minted CURIE via its concept's surface form appearing in the block's
    # REAL provenance, not only its key_claims prose. This mirrors the
    # ``_run_inter_tier_validation`` handler's threading so the phase-level
    # gate and the handler report stay consistent. Absent/unreadable
    # sidecar → no-op (anchoring runs over key_claims only; ungrounded
    # blocks still fail closed).
    ocp = _locate(phase_outputs, "outline_chunks_path")
    if isinstance(ocp, str) and ocp:
        sidecar = Path(ocp)
        if sidecar.exists():
            try:
                import json as _json
                sc_map = _json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(sc_map, dict):
                    inputs["source_chunks_by_block_id"] = sc_map
            except (OSError, ValueError) as exc:
                logger.debug(
                    "curie-anchoring builder: outline_chunks.json load "
                    "failed at %s (%s); anchoring over key_claims only.",
                    ocp, exc,
                )
    # Objective-refs-concept anchoring: thread the {objective_id: statement}
    # map from synthesized_objectives.json so the validator can anchor a
    # minted CURIE via its concept surface form appearing in the statement
    # of an objective the block DECLARES. Absent/unreadable → no-op
    # (anchoring runs over key_claims + source_chunk_text only; byte-
    # identical to the pre-fix contract, including every RDF/legacy run).
    obj_statements = _resolve_objective_statements_map(
        phase_outputs, workflow_params
    )
    if obj_statements:
        inputs["objective_statements"] = obj_statements
    return inputs, []


def _resolve_objective_statements_map(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> Dict[str, str]:
    """Load ``{objective_id: statement}`` from synthesized_objectives.json.

    Resolves the canonical objectives path via
    :func:`_resolve_objectives_path` and flattens terminal + chapter
    objectives via :func:`lib.validators.abcd_objective._flatten_objectives`
    (the same loader the statistical-tier input builder uses). Returns an
    empty map when no objectives file resolves / is unparseable — the
    objective-refs anchoring surface is then unavailable (graceful).
    """
    objectives_path = _resolve_objectives_path(phase_outputs, workflow_params)
    if not objectives_path:
        return {}
    try:
        import json as _json
        from lib.validators.abcd_objective import _flatten_objectives

        path = Path(objectives_path)
        if not path.exists():
            return {}
        payload = _json.loads(path.read_text(encoding="utf-8"))
        flat = _flatten_objectives(payload)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        logger.debug(
            "curie-anchoring builder: objective_statements load failed "
            "from %s (%s); objective-refs anchoring unavailable.",
            objectives_path, exc,
        )
        return {}
    statements: Dict[str, str] = {}
    for lo in flat or []:
        if not isinstance(lo, dict):
            continue
        lo_id = lo.get("id") or lo.get("objective_id")
        if not isinstance(lo_id, str) or not lo_id:
            continue
        stmt = lo.get("statement") or lo.get("text")
        if isinstance(stmt, str) and stmt.strip():
            statements[lo_id] = stmt.strip()
    return statements


def _build_source_chunks_from_dart_jsonl(
    phase_outputs: Dict[str, Any],
) -> Dict[str, str]:
    """W4 — id→chunk-TEXT map from the DART chunkset ``chunks.jsonl``.

    The staging manifest's ``files[].text`` (the legacy
    :func:`_build_rewrite_block_input` source) is best-effort and is often
    ABSENT on a live run, so the NLI grounding gates (``claim_support`` /
    ``block_prose_entailment``) silently degrade to no-grounding-source
    warnings against an empty premise map. The DART chunkset ``chunks.jsonl``
    is the AUTHORITATIVE chunk-body source: it carries the full chunk ``text``
    keyed by both the chunk top-level ``id`` and every
    ``source.source_references[].sourceId`` (the ``dart:{slug}#{block_id}``
    shape a block's ``source_ids`` / ``source_references[]`` use).

    Resolves ``dart_chunks_path`` from ``phase_outputs.chunking.dart_chunks_path``
    (or any phase via :func:`_locate`), reads the JSONL, and maps each
    ``sourceId`` + chunk ``id`` → the chunk ``text``/``body``. Best-effort:
    returns an empty map when the path is absent / unreadable so the gates'
    graceful-degrade path stays intact.
    """
    chunking = phase_outputs.get("chunking") or {}
    dart_chunks_path = _chunking_chunks_path(chunking)
    if not isinstance(dart_chunks_path, str) or not dart_chunks_path:
        dart_chunks_path = _locate(phase_outputs, "semantik_chunks_path", "dart_chunks_path")
    if not isinstance(dart_chunks_path, str) or not dart_chunks_path:
        return {}
    chunks_jsonl = Path(dart_chunks_path)
    if not chunks_jsonl.exists():
        return {}

    text_map: Dict[str, str] = {}
    try:
        import json as _json
        with chunks_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                body = chunk.get("text") or chunk.get("body")
                if not isinstance(body, str) or not body.strip():
                    continue
                chunk_id = chunk.get("id")
                if isinstance(chunk_id, str) and chunk_id.strip():
                    text_map.setdefault(chunk_id.strip(), body)
                source = chunk.get("source")
                if not isinstance(source, dict):
                    continue
                src_refs = source.get("source_references") or []
                if not isinstance(src_refs, list):
                    continue
                for ref in src_refs:
                    if not isinstance(ref, dict):
                        continue
                    sid = ref.get("sourceId")
                    if isinstance(sid, str) and sid.strip():
                        text_map.setdefault(sid.strip(), body)
    except OSError as exc:
        logger.debug(
            "rewrite-block source_chunks rebuild from chunks.jsonl %s "
            "failed: %s",
            dart_chunks_path, exc,
        )
        return {}
    return text_map


def _build_chunk_provenance_index_from_dart_jsonl(
    phase_outputs: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Build the ``{provenance_ref -> [chunk_id, ...]}`` reverse index for the
    opt-in ``ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE`` gate-side resolution.

    Resolves the SAME chunkset ``chunks.jsonl`` path as
    :func:`_build_source_chunks_from_dart_jsonl` and delegates to
    ``block_prose_entailment.load_chunk_provenance_index``. Rewrite-tier blocks
    cite ``semantik:{slug}#anchor`` provenance refs, not
    ``{course}_chunk_NNNNN`` chunk ids; this index lets
    ``BlockProseEntailmentValidator`` map an unresolved ref → the chunk ids that
    carry it (a section-level ref → ALL of the section's chunks). Threaded
    UNCONDITIONALLY (cheap, same JSONL) — inert unless the validator's env flag
    is on. Best-effort: absent / unreadable path → empty index.
    """
    chunking = phase_outputs.get("chunking") or {}
    dart_chunks_path = _chunking_chunks_path(chunking)
    if not isinstance(dart_chunks_path, str) or not dart_chunks_path:
        dart_chunks_path = _locate(
            phase_outputs, "semantik_chunks_path", "dart_chunks_path"
        )
    if not isinstance(dart_chunks_path, str) or not dart_chunks_path:
        return {}
    try:
        from lib.validators.block_prose_entailment import (
            load_chunk_provenance_index,
        )

        return load_chunk_provenance_index(dart_chunks_path)
    except Exception as exc:  # noqa: BLE001 — best-effort, never break the build
        logger.debug(
            "chunk_provenance_index build from %s failed: %s",
            dart_chunks_path, exc,
        )
        return {}


def _build_rewrite_block_input(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Group B — Rewrite-emit shape / sentence-grounding + NLI builder.

    Adds ``source_chunks`` (a Dict[sourceId, chunk_text] mapping) on
    top of the Group A surface so ``RewriteSourceGroundingValidator``
    (cosine), ``ClaimSupportValidator`` (per-claim NLI), and
    ``BlockProseEntailmentValidator`` (full-prose NLI) can resolve each
    block's cited chunk text. ``RewriteHtmlShapeValidator`` only consumes
    ``blocks`` and ignores the extra keys.

    The source_chunks mapping is rebuilt from TWO sources, AUTHORITATIVE
    first (W4 §3.5):

    1. The DART chunkset ``chunks.jsonl`` (``_build_source_chunks_from_dart_jsonl``)
       — the full chunk-body source keyed on every
       ``source.source_references[].sourceId``. This is the fix for the §0.1
       silent-no-op bug: the staging manifest often lacks per-source bodies, so
       the NLI premise map was empty and every NLI gate degraded to
       no-grounding-source warnings.
    2. The staging manifest's ``files[].{text,plain_text}`` (legacy best-effort)
       — layered on top WITHOUT overwriting a chunks.jsonl body (the chunkset
       body is authoritative).

    When neither surface is available the validators' no-grounding-source path
    emits a warning per block (passed=True), so the absence is non-fatal.
    """
    inputs, missing = _build_block_input_rewrite(phase_outputs, workflow_params)
    if missing:
        return inputs, missing

    # W4 §3.5 — authoritative chunk bodies from the DART chunkset first.
    chunks_lookup: Dict[str, str] = _build_source_chunks_from_dart_jsonl(
        phase_outputs
    )

    # Legacy best-effort: layer the staging manifest's per-source bodies on
    # top WITHOUT overwriting a chunks.jsonl body (setdefault).
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
                            chunks_lookup.setdefault(sid, text)
        except (OSError, ValueError, TypeError) as exc:
            logger.debug(
                "rewrite-block source_chunks rebuild from %s failed: %s",
                manifest_path_str, exc,
            )
    if chunks_lookup:
        inputs["source_chunks"] = chunks_lookup

    # Opt-in gate-side provenance resolution (ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE)
    # — thread the {provenance_ref -> [chunk_id]} reverse index UNCONDITIONALLY;
    # BlockProseEntailmentValidator consults it only when its env flag is on.
    provenance_index = _build_chunk_provenance_index_from_dart_jsonl(phase_outputs)
    if provenance_index:
        inputs["chunk_provenance_index"] = provenance_index
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

    Surfaces ``{blocks, objectives_path, objective_statements?,
    objectives?}``. The executor merges ``gate.config`` (and therefore
    ``gate.config.thresholds``) into the inputs dict at
    ``executor.py:1442`` so the per-validator threshold dial flows
    through unchanged. Each validator additionally accepts
    ``objective_statements`` / ``concept_definitions`` /
    ``paraphrase_fn`` / ``embedder`` overrides via ``inputs.*``; the
    Phase 4 PoC contract degrades to ``passed=True`` warnings when
    those auxiliaries aren't wired.

    GPT Feedback v2 Wave 1.7 W1.7.C — Drift B fix: pre-Wave-1.7 the
    builder only emitted ``{blocks, objectives_path}``, so the existing
    ``ObjectiveAssessmentSimilarityValidator`` silently degraded to
    ``OBJECTIVE_STATEMENT_UNRESOLVED`` warnings on every block since
    Phase 4 PoC. Wave 1.7 fixes this by loading
    ``synthesized_objectives.json`` from ``objectives_path``,
    flattening via :func:`lib.validators.abcd_objective._flatten_objectives`,
    and surfacing both:

    * ``objective_statements: Dict[str, str]`` — convenience map
      ``{lo.id: lo.statement}`` consumed by the Wave-N2 similarity
      validators.
    * ``objectives: Dict[str, Dict[str, Any]]`` — full LO dicts keyed
      by id (with ``bloom_level`` / ``bloom_verb`` / ``statement``)
      consumed by the new
      :class:`lib.validators.block_objective_delivery.BlockObjectiveDeliveryValidator`.

    Best-effort: any failure to read / parse the objectives JSON is
    logged at debug and falls through to the legacy two-key shape so
    the gate path stays alive even on a malformed objectives surface.
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
    # ``objectives_path`` + ``objective_statements`` + ``objectives``
    # + the threshold inputs the executor merges in via gate.config.
    # Drop manifest/staging so the validator doesn't see unrelated keys.
    pruned: Dict[str, Any] = {"blocks": inputs.get("blocks", [])}
    objectives_path_raw = inputs.get("objectives_path")
    if objectives_path_raw:
        pruned["objectives_path"] = objectives_path_raw
        # Wave 1.7 W1.7.C — Drift B fix. Load + flatten + surface the
        # objective_statements + objectives maps so every statistical-
        # tier validator (and the new BlockObjectiveDeliveryValidator)
        # sees populated inputs.
        try:
            import json as _json
            from lib.validators.abcd_objective import (
                _flatten_objectives,
            )

            objectives_path = Path(objectives_path_raw)
            if objectives_path.exists():
                payload = _json.loads(
                    objectives_path.read_text(encoding="utf-8")
                )
                flat = _flatten_objectives(payload)
                statements: Dict[str, str] = {}
                full_dicts: Dict[str, Dict[str, Any]] = {}
                for lo in flat:
                    if not isinstance(lo, dict):
                        continue
                    lo_id_raw = lo.get("id") or lo.get("objective_id")
                    if not isinstance(lo_id_raw, str) or not lo_id_raw:
                        continue
                    statement_raw = lo.get("statement") or lo.get("text")
                    if isinstance(statement_raw, str) and statement_raw.strip():
                        statements[lo_id_raw] = statement_raw.strip()
                    full_dicts[lo_id_raw] = dict(lo)
                if statements:
                    pruned["objective_statements"] = statements
                if full_dicts:
                    pruned["objectives"] = full_dicts
        except (OSError, ValueError, TypeError, ImportError) as exc:
            logger.debug(
                "block_statistical_input: failed to populate "
                "objective_statements/objectives from %s: %s",
                objectives_path_raw, exc,
            )
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


def _build_chunk_health(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Input builder for ChunkHealthValidator (pre-synthesis chunk-health gate).

    Surfaces ``{chunks_path, textbook_structure_path?}`` so the validator can
    audit the emitted chunkset + structure the course is about to be
    synthesized from. Wired on ``textbook_to_course::objective_extraction`` (it
    runs AFTER ``chunking`` + the extractor, BEFORE ``course_planning``), so at
    gate time both phase outputs are on ``phase_outputs``:

    * ``chunks_path`` — the ``chunking`` phase's ratified
      ``semantik_chunks_path`` (legacy ``dart_chunks_path``) via
      :func:`_chunking_chunks_path`; falls back to any ``*chunks_path`` key.
    * ``textbook_structure_path`` — the ``objective_extraction`` phase's
      declared YAML output (optional; the validator skips its S-class structure
      checks when absent).

    The validator is opt-in (``ED4ALL_CHUNK_HEALTH_GATE``, default OFF) and
    skips-with-pass BEFORE touching inputs when the flag is unset, so we NEVER
    emit a missing-key marker for ``textbook_structure_path`` (its absence is a
    graceful skip, not a gate skip). We DO surface a missing ``chunks_path`` so
    the gate is structured-skipped rather than crashing when no chunkset
    resolved at all on a default-off run; when the flag is ON the validator's
    own fail-closed arm handles an absent chunkset.
    """
    inputs: Dict[str, Any] = {}

    chunking = phase_outputs.get("chunking") or {}
    chunks_path = _chunking_chunks_path(chunking)
    if not (isinstance(chunks_path, str) and chunks_path):
        chunks_path = _locate(
            phase_outputs, "semantik_chunks_path", "dart_chunks_path", "chunks_path"
        )
    if isinstance(chunks_path, str) and chunks_path:
        inputs["chunks_path"] = chunks_path

    oe = phase_outputs.get("objective_extraction") or {}
    ts_path = oe.get("textbook_structure_path")
    if not (isinstance(ts_path, str) and ts_path):
        ts_path = _locate(phase_outputs, "textbook_structure_path")
    if isinstance(ts_path, str) and ts_path:
        inputs["textbook_structure_path"] = ts_path

    # Always let the validator RUN (it reads ED4ALL_CHUNK_HEALTH_GATE itself and
    # skips-with-pass when OFF). Never mark a missing textbook_structure_path —
    # its absence is a graceful skip inside the validator, not a gate skip.
    return inputs, []


def _build_chunkset_manifest_inputs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave2-I9 — input builder for ChunksetManifestValidator.

    The validator's ``validate()`` reads ``inputs["chunkset_manifest_path"]``
    (see ``lib/validators/chunkset_manifest.py:213``). This gate fires
    symmetrically at two phases:

    * ``chunking`` (DART) — emits ``dart_chunks_path``; manifest.json
      sits in the same directory.
    * ``imscc_chunking`` (IMSCC) — emits ``imscc_chunks_path``; manifest
      sits beside chunks.jsonl.

    Resolution chain (high → low):

    * Explicit ``manifest_path`` key in either chunking phase output
      (Phase 7c emits this directly even though the YAML
      ``outputs:`` block doesn't yet declare it).
    * Derive ``<chunks_dir>/manifest.json`` from the sibling
      ``dart_chunks_path`` (DART) or ``imscc_chunks_path`` (IMSCC).

    Pre-Wave2-I9 the validator silently skipped with
    ``__no_builder_registered__`` because no builder was registered;
    the gate ran ``passed=True warning-severity`` (gate config
    ``severity: warning``, ``on_fail: warn``, ``on_error: warn``),
    so the manifest was never inspected. Finding 5 / Wave2-I9.
    """
    chunking = phase_outputs.get("chunking") or {}
    imscc_chunking = phase_outputs.get("imscc_chunking") or {}

    # Prefer an explicit manifest_path emitted by the phase tool (the
    # _run_dart_chunking / _run_imscc_chunking helpers both emit it
    # even though YAML outputs:block doesn't currently declare it).
    manifest_path = (
        chunking.get("manifest_path")
        or imscc_chunking.get("manifest_path")
    )

    # Derive from the sibling chunks.jsonl path. DART takes priority;
    # if both are present the chunking phase fired first so its
    # manifest is the one tested at the chunking-phase gate. The
    # imscc_chunking gate runs later and re-reads the IMSCC manifest
    # at that point (phase_outputs.chunking won't have a manifest
    # for the IMSCC sibling, so we won't mis-route).
    if not manifest_path:
        chunks_path_raw = (
            _chunking_chunks_path(chunking)
            or imscc_chunking.get("imscc_chunks_path")
        )
        if isinstance(chunks_path_raw, str) and chunks_path_raw:
            try:
                chunks_jsonl = Path(chunks_path_raw)
                manifest_path = str(chunks_jsonl.parent / "manifest.json")
            except (TypeError, ValueError):
                manifest_path = None

    if not manifest_path:
        return {}, ["chunkset_manifest_path"]

    return {"chunkset_manifest_path": manifest_path}, []


def _build_chunk_wcag_status(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """IB4.2 — input builder for ChunkWcagStatusValidator.

    Surfaces the chunkset JSONL path so the validator can audit the data-only
    ``wcag_block_status`` / ``figure_alt`` chunk fields. Fires symmetrically at
    two phases:

    * ``chunking`` (DART) — emits ``dart_chunks_path``.
    * ``imscc_chunking`` (IMSCC) — emits ``imscc_chunks_path``.

    Input-starved (no path resolved) is NOT a missing-input failure: the
    validator's own ``WCAG_FIELDS_ABSENT`` arm warns + passes (warning-day-1,
    can't break a run), so we surface whatever resolves and never return a
    missing-input list that would mark the gate failed.
    """
    chunking = phase_outputs.get("chunking") or {}
    imscc_chunking = phase_outputs.get("imscc_chunking") or {}

    inputs: Dict[str, Any] = {}
    dart_path = _chunking_chunks_path(chunking) or _locate(
        phase_outputs, "semantik_chunks_path", "dart_chunks_path"
    )
    if isinstance(dart_path, str) and dart_path:
        inputs["dart_chunks_path"] = dart_path
    imscc_path = imscc_chunking.get("imscc_chunks_path") or _locate(
        phase_outputs, "imscc_chunks_path"
    )
    if isinstance(imscc_path, str) and imscc_path:
        inputs["imscc_chunks_path"] = imscc_path

    return inputs, []


def _build_qti_well_formed(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """W10 §7 — input builder for QtiWellFormedValidator.

    The validator's ``validate()`` reads ``inputs["qti_dir"]`` (see
    ``lib/validators/qti_well_formed.py:285``) — a directory under which it
    globs ``*.xml`` and round-trips / XSD-validates each QTI document. The
    ``assessment_synthesis`` phase (§2.4 Option A) writes those XML files to
    ``<export>/06_assessments/``.

    Resolution chain (high → low); first existing directory wins, then a
    derived candidate is surfaced even if absent so the validator can emit
    its own structured ``QTI_NO_INPUT`` / non-directory issue rather than the
    router silently skipping the gate:

    * Explicit ``qti_dir`` / ``assessment_dir`` key emitted by the
      ``assessment_synthesis`` phase (the handler emits these directly).
    * ``<project_export_root>/06_assessments`` — the canonical layout. The
      export root is resolved via :func:`_find_project_export_dir` (the same
      ``objective_extraction.project_path`` / ``project_*`` resolution the
      content-dir disk-glob fallback uses).
    """
    # Explicit phase-output key wins.
    explicit = _locate(phase_outputs, "qti_dir", "assessment_dir")
    if isinstance(explicit, str) and explicit:
        return {"qti_dir": explicit}, []

    export_dir = _find_project_export_dir(phase_outputs, workflow_params)
    if export_dir is not None:
        candidate = export_dir / "06_assessments"
        # Surface the canonical candidate even when it isn't on disk yet —
        # the validator's own directory check emits the structured issue,
        # mirroring the manifest_completeness "always surface" contract.
        return {"qti_dir": str(candidate)}, []

    return {}, ["qti_dir"]


def _build_concept_graph_inputs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave2-I9 — input builder for ConceptGraphValidator.

    The validator's ``validate()`` reads ``inputs["concept_graph_path"]``
    (see ``lib/validators/concept_graph.py:218``). Phase
    ``concept_extraction`` declares ``concept_graph_path`` as a YAML
    ``outputs:`` key (``config/workflows.yaml:873``), so it surfaces
    directly in ``phase_outputs["concept_extraction"]``.

    Optional ``min_nodes`` / ``min_edge_types`` thresholds flow
    through ``gate.config`` via the executor's
    ``executor.py:1442`` merge — the builder doesn't need to surface
    them. Pre-Wave2-I9 the gate skipped silently with
    ``__no_builder_registered__``.
    """
    ce = phase_outputs.get("concept_extraction") or {}
    candidate = ce.get("concept_graph_path")
    if not candidate:
        # Fallback: scan any phase that surfaced this key. Mirrors
        # the resilience of other builders that use `_locate`.
        candidate = _locate(phase_outputs, "concept_graph_path")
    if not isinstance(candidate, str) or not candidate:
        return {}, ["concept_graph_path"]
    return {"concept_graph_path": candidate}, []


def _build_kg_quality_inputs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Activate-the-dormant-gate builder for KGQualityValidator.

    The ``kg_quality_report`` gate fires at the
    ``textbook_to_course::libv2_archival`` phase, declared
    ``severity: critical`` / ``on_fail: block`` / ``on_error:
    fail_closed`` in ``config/workflows.yaml``. Pre-activation NO builder
    was registered, so ``GateInputRouter.build`` returned
    ``({}, ["__no_builder_registered__"])`` and the executor stamped the
    gate ``GATE_SKIPPED_MISSING_INPUTS`` with ``passed=True`` — the gate
    NEVER ran. This builder routes the inputs the validator's
    ``validate()`` consumes so the gate fires and fails closed when its
    graph inputs are missing.

    The validator (``lib/validators/kg_quality.py``) requires five
    non-empty inputs (``course_slug`` / ``run_id`` / ``output_dir`` /
    ``concept_graph_path`` / ``semantic_graph_path``) and treats any
    missing one as a critical fail-closed
    (``KG_QUALITY_PEDAGOGY_GRAPH_MISSING``) — that is the whole point of
    activation: a libv2_archival run with no concept / semantic graph
    must NOT ship an empty KG to LibV2.

    Resolution chain (graph is the load-bearing input):

    * ``semantic_graph_path`` — the SEMANTIC graph
      (``concept_graph_semantic.json``). The ``concept_extraction``
      phase emits ``concept_graph_path`` pointing AT the semantic graph
      (``<course_dir>/concept_graph/concept_graph_semantic.json`` — see
      ``MCP/tools/pipeline_tools.py::_run_concept_extraction`` :10412 /
      :10536), so we route that value to ``semantic_graph_path``. When
      the phase output isn't surfaced (resumed / stage-subcommand run)
      we derive it from the libv2_archival ``course_dir`` against the
      same canonical locations the archival code probes
      (``concept_graph/`` then legacy ``graph/`` /
      ``imscc_chunks/`` / ``corpus/``).
    * ``concept_graph_path`` — the ASSERTED graph
      (``concept_graph.json``). Many corpora ship
      only the semantic graph; the reporter degrades a missing asserted
      graph to ``{}`` internally (``_load_json(...) or {}``), so we
      surface a best-effort sibling path (``<graph_dir>/concept_graph.json``)
      rather than leaving it empty — the validator's missing-input check
      requires a non-empty STRING, and the reporter tolerates the file's
      absence.
    * ``course_slug`` — ``libv2_archival.course_slug`` >
      ``concept_extraction.course_slug`` > any phase's ``course_slug`` >
      ``workflow_params.course_name``.
    * ``run_id`` — ``workflow_params.run_id`` > env ``ED4ALL_RUN_ID``.
    * ``output_dir`` — the LibV2 course ``quality/`` directory (the
      canonical home of ``kg_quality_report.json``), derived from
      ``course_dir``.

    Threshold floors ARE enforced. The YAML ``threshold:`` block
    (``min_completeness`` / ``min_consistency`` / ``min_accuracy`` /
    ``min_coverage``) is forwarded into the validator's inputs by
    ``ValidationGateManager.run_gate`` (it merges ``gate.threshold`` into
    the inputs dict alongside ``gate.config``, via ``setdefault``). The
    validator reads each floor from its inputs; ``_apply_thresholds``
    still handles only the result-level keys
    (``max_critical_issues`` / ``max_issues`` / ``min_score`` /
    ``required_score``). A metric breach emits a critical
    ``KG_QUALITY_<DIM>_BELOW_THRESHOLD`` GateIssue and inverts
    ``passed`` so the critical/block gate refuses to ship a degraded KG;
    a missing/malformed graph or reporter exception likewise fails the
    gate closed. See the ``docs/validation/gates.md`` note.

    Double-consensus guard: the validator re-runs
    ``EdgeConsensusAggregator`` to attenuate the consistency axis by
    ``(1 - contradiction_rate)``. ``build()`` is deterministic over the
    same semantic graph, so the re-run is idempotent — it reproduces the
    canonical ``concept_graph/edge_consensus_report.json`` content
    byte-for-byte (non-divergent). We do NOT overwrite that canonical
    sibling: the validator writes its consensus copy under ``output_dir``
    (``quality/``), leaving the authoring-time sibling untouched.
    """
    # --- course_slug
    course_slug = (
        _locate(phase_outputs, "course_slug")
        or workflow_params.get("course_name")
        or workflow_params.get("course_code")
    )

    # --- course_dir (anchors output_dir + the on-disk graph fallback)
    course_dir_raw = _locate(phase_outputs, "course_dir") or workflow_params.get(
        "course_dir"
    )
    course_dir = Path(course_dir_raw) if course_dir_raw else None

    # --- run_id
    run_id = workflow_params.get("run_id") or os.environ.get(
        "ED4ALL_RUN_ID", ""
    ).strip()

    # --- semantic_graph_path: concept_extraction emits concept_graph_path
    #     pointing AT the semantic graph.
    ce = phase_outputs.get("concept_extraction") or {}
    semantic_graph_path: Optional[str] = None
    candidate = ce.get("concept_graph_path")
    if isinstance(candidate, str) and candidate:
        semantic_graph_path = candidate
    if not semantic_graph_path:
        located = _locate(phase_outputs, "semantic_graph_path")
        if located:
            semantic_graph_path = located
    if not semantic_graph_path:
        located = _locate(phase_outputs, "concept_graph_path")
        if located:
            semantic_graph_path = located
    if not semantic_graph_path and course_dir is not None:
        # Disk-derive fallback (resumed / stage-subcommand run with no
        # concept_extraction phase output). Probe the same canonical
        # locations the archival code uses; surface ONLY an EXTANT file.
        # A genuinely-absent graph must leave semantic_graph_path empty so
        # the validator's KG_QUALITY_PEDAGOGY_GRAPH_MISSING arm fires
        # (critical/block) — that is the fail-closed point of activation.
        for sub in ("concept_graph", "graph", "imscc_chunks", "corpus"):
            cand = course_dir / sub / "concept_graph_semantic.json"
            if cand.exists():
                semantic_graph_path = str(cand)
                break

    # --- concept_graph_path: the asserted concept_graph.json (sibling of
    #     the semantic graph). Best-effort — the reporter degrades a
    #     missing asserted graph to {} internally, so we never block on
    #     its absence, but we surface a non-empty STRING so the
    #     validator's missing-input check passes.
    concept_graph_path: Optional[str] = None
    located_asserted = _locate(phase_outputs, "asserted_concept_graph_path")
    if located_asserted:
        concept_graph_path = located_asserted
    if not concept_graph_path and semantic_graph_path:
        try:
            sib = Path(semantic_graph_path).parent / "concept_graph.json"
            concept_graph_path = str(sib)
        except (TypeError, ValueError):
            concept_graph_path = None

    # --- output_dir: canonical quality/ home of kg_quality_report.json.
    output_dir: Optional[str] = None
    if course_dir is not None:
        output_dir = str(course_dir / "quality")

    inputs: Dict[str, Any] = {}
    missing: List[str] = []
    if course_slug:
        inputs["course_slug"] = str(course_slug)
    else:
        missing.append("course_slug")
    if run_id:
        inputs["run_id"] = str(run_id)
    else:
        missing.append("run_id")
    if output_dir:
        inputs["output_dir"] = output_dir
    else:
        missing.append("output_dir")
    if concept_graph_path:
        inputs["concept_graph_path"] = concept_graph_path
    else:
        missing.append("concept_graph_path")
    if semantic_graph_path:
        inputs["semantic_graph_path"] = semantic_graph_path
    else:
        missing.append("semantic_graph_path")

    # IMPORTANT: do NOT short-circuit to a structured router-skip when
    # graph inputs are unresolvable. The whole point of activation is to
    # let the validator's own KG_QUALITY_PEDAGOGY_GRAPH_MISSING
    # fail-closed arm fire on a missing graph (critical/block). Returning
    # a non-empty missing-list here would route to
    # GATE_SKIPPED_MISSING_INPUTS (passed=True) and re-dormant the gate.
    # We therefore pass whatever resolved and let the validator
    # adjudicate — surfacing required-input markers ONLY for the
    # non-graph context keys (course_slug / run_id / output_dir) would
    # also skip the gate, so we surface NO missing keys and rely on the
    # validator's fail-closed contract.
    return inputs, []


def _build_domain_concept_vocabulary_inputs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Stage-3 (Wave C) — input builder for DomainConceptVocabularyValidator.

    The validator's ``validate()`` reads
    ``inputs["domain_concept_vocabulary_path"]`` (or the in-memory
    ``domain_concept_vocabulary`` dict) and skips-with-pass when neither
    is supplied — the ``ABCD_MISSING`` graceful-degrade contract.

    Phase ``concept_extraction`` declares ``domain_concept_vocabulary_path``
    as an OPTIONAL YAML ``outputs:`` key, written only when
    ``TEXTBOOK_SYNTHESIS_PROVIDER`` is set. On a default-off run the key
    is absent — so this builder returns an EMPTY inputs dict with NO
    missing-key list, letting the validator run its no-op skip-with-pass
    rather than the executor marking the gate ``GATE_SKIPPED_MISSING_INPUTS``.
    """
    ce = phase_outputs.get("concept_extraction") or {}
    candidate = ce.get("domain_concept_vocabulary_path")
    if not candidate:
        candidate = _locate(phase_outputs, "domain_concept_vocabulary_path")
    if not isinstance(candidate, str) or not candidate:
        # Default-off run: no vocabulary artifact. Empty inputs + no
        # missing keys → the validator's graceful-degrade no-op pass.
        return {}, []
    return {"domain_concept_vocabulary_path": candidate}, []


def _build_chapter_objective_coverage_inputs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Stage-2 (Wave D) — input builder for ChapterObjectiveCoverageValidator.

    The validator (``lib/validators/chapter_objective_coverage.py``)
    cross-checks every ``chapters[].id`` with non-empty ``chapter_text``
    against the synthesized chapter objectives, and asserts the
    reconciled ``terminal_objectives[]`` is non-empty. It reads:

    * ``inputs["textbook_structure_path"]`` — source of ``chapters[]``
      (each chapter carries ``id`` + the Stage-2 ``chapter_text``).
      Pulled from ``phase_outputs.objective_extraction.textbook_structure_path``.
    * ``inputs["synthesized_objectives_path"]`` — the Stage-2 output
      (``chapter_objectives`` + reconciled ``terminal_objectives``).
      Routed through :func:`_resolve_objectives_path`.

    Graceful-degrade contract: on a default-off run
    (``TEXTBOOK_SYNTHESIS_PROVIDER`` unset) the
    ``textbook_structure.json`` carries no ``chapter_text``, so the
    validator's own no-op skip-with-pass fires. We therefore return an
    EMPTY missing-key list even when an input is unresolvable — the
    validator handles the absence — so the executor lets the validator
    run rather than marking the gate ``GATE_SKIPPED_MISSING_INPUTS``.
    """
    inputs: Dict[str, Any] = {}

    oe = phase_outputs.get("objective_extraction") or {}
    ts_path = oe.get("textbook_structure_path")
    if not (isinstance(ts_path, str) and ts_path):
        ts_path = _locate(phase_outputs, "textbook_structure_path")
    if isinstance(ts_path, str) and ts_path:
        inputs["textbook_structure_path"] = ts_path

    objectives_path = _resolve_objectives_path(phase_outputs, workflow_params)
    if objectives_path:
        inputs["synthesized_objectives_path"] = objectives_path

    # I3 — route the DART chunkset path so the source_coverage gate's chunk arm
    # (pure set-membership, embedding-free) can measure the chunk-level citation
    # gap. The chunking phase emits ``dart_chunks_path`` (chunks.jsonl). Optional
    # — the chunk arm no-op-passes on absence (measurement guardrail).
    chunking = phase_outputs.get("chunking") or {}
    dart_chunks_path = _chunking_chunks_path(chunking)
    if not (isinstance(dart_chunks_path, str) and dart_chunks_path):
        dart_chunks_path = _locate(phase_outputs, "semantik_chunks_path", "dart_chunks_path")
    if isinstance(dart_chunks_path, str) and dart_chunks_path:
        inputs["dart_chunks_path"] = dart_chunks_path

    # No missing-key markers: all inputs are optional from the
    # router's POV — the validator skips-with-pass on absence.
    return inputs, []


def _build_textbook_outline_inputs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Three-stage textbook synthesis (Wave A/B) — input builder for
    TextbookOutlineValidator.

    The validator (``lib/validators/textbook_structure.py``) audits the
    Stage-1 enrichment keys (``semantic_outline`` /
    ``draft_terminal_objectives``) that ``_extract_textbook_structure``
    folds into ``textbook_structure.json`` when
    ``TEXTBOOK_SYNTHESIS_PROVIDER`` is set. Its ``_coerce_structure``
    resolution chain reads (in priority order)
    ``inputs["textbook_structure"]`` (an in-memory dict) >
    ``inputs["textbook_structure_path"]`` (an on-disk JSON the validator
    loads). Phase ``objective_extraction`` declares
    ``textbook_structure_path`` as a YAML ``outputs:`` key
    (``config/workflows.yaml``), so it surfaces directly in
    ``phase_outputs["objective_extraction"]`` — mirroring the resolution
    in :func:`_build_chapter_objective_coverage_inputs`.

    Graceful-degrade contract: on a default-off run
    (``TEXTBOOK_SYNTHESIS_PROVIDER`` unset) the ``textbook_structure.json``
    carries neither enrichment key, so the validator's own no-op
    skip-with-pass fires. We therefore return an EMPTY missing-key list
    even when the path is unresolvable — the validator handles the
    absence (``_coerce_structure`` returns ``(None, None)`` → skip-with-
    pass) — so the executor lets the validator RUN rather than stamping
    the gate ``GATE_SKIPPED_MISSING_INPUTS``. The gate is wired
    ``severity: critical`` / ``on_fail: block`` at
    ``textbook_to_course::objective_extraction::
    textbook_outline_enrichment``; pre-registration NO builder existed,
    so the router returned ``__no_builder_registered__`` and the gate
    skipped with a warning on every run (gate_input_routing.py:1836-1842).
    """
    oe = phase_outputs.get("objective_extraction") or {}
    ts_path = oe.get("textbook_structure_path")
    if not (isinstance(ts_path, str) and ts_path):
        ts_path = _locate(phase_outputs, "textbook_structure_path")
    if isinstance(ts_path, str) and ts_path:
        return {"textbook_structure_path": ts_path}, []
    # Path unresolvable: the validator skips-with-pass on absence, so
    # let it run (no missing-key marker) rather than skip the gate.
    return {}, []


def _build_abcd_objective_inputs(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave2-I9 — input builder for AbcdObjectiveValidator.

    The validator's ``_coerce_objectives`` resolution chain reads
    (in priority order) ``inputs["objectives"]`` >
    ``inputs["synthesized_objectives_path"]`` (see
    ``lib/validators/abcd_objective.py:179``). Phase
    ``course_planning`` declares ``synthesized_objectives_path`` as
    a YAML ``outputs:`` key (``config/workflows.yaml:938``), so we
    route through the canonical
    :func:`_resolve_objectives_path` helper to keep the resolution
    surface consistent with every other synthesized-objectives
    consumer (course_planning emit > workflow_params override >
    derived from objective_extraction.project_path).

    Pre-Wave2-I9 the gate skipped silently with
    ``__no_builder_registered__`` despite being wired as
    ``abcd_verb_alignment`` at the course_planning phase.
    """
    objectives_path = _resolve_objectives_path(phase_outputs, workflow_params)
    if not objectives_path:
        return {}, ["synthesized_objectives_path"]
    return {"synthesized_objectives_path": objectives_path}, []


def _build_assessment_objective_alignment(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Wave 24: assessments path + chunks path builder.

    The Trainforge phase emits ``output_path`` for the assessments.json
    and produces ``chunks.jsonl`` under ``{trainforge_dir}/corpus/``.
    The trainforge_dir is the parent of the IMSCC's project dir —
    derive it conservatively from the assessments output path.

    **Wave 5 W5.E**: also surface
    ``phase_outputs.course_planning.synthesized_objectives_path`` (or
    any other phase that emits the canonical synthesized objectives) so
    the validator can union the synthesized objectives' ``id`` set into
    the chunks-side ``learning_outcome_refs`` resolution surface.
    Closes Finding F: a chunks-side empty ``learning_outcome_refs``
    used to trigger phantom-ref criticality even when the assessment
    objective IS in the synthesized objectives. Routed via the
    canonical ``_resolve_objectives_path`` helper so the resolution
    chain matches the rest of the synthesized-objectives consumers
    (course_planning emit > workflow_params override > derived from
    objective_extraction.project_path).

    The path is surfaced as ``inputs["synthesized_objectives_path"]``
    when resolvable; absent when the workflow doesn't emit it
    (``rag_training`` legacy workflow), preserving byte-identical
    pre-W5.E behaviour.
    """
    assessments = _locate(
        phase_outputs, "assessments_path", "assessment_path", "output_path",
    )
    if not assessments:
        return {}, ["assessments_path"]

    inputs: Dict[str, Any] = {"assessments_path": assessments}

    # Wave 5 W5.E: surface synthesized_objectives_path when available.
    # Optional input — missing is fine; the validator falls back to the
    # chunks-only resolution surface byte-identically with the pre-W5.E
    # behaviour.
    synthesized = _resolve_objectives_path(phase_outputs, workflow_params)
    if synthesized:
        inputs["synthesized_objectives_path"] = synthesized

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


def _build_discussion_assignment_grounding(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """C3-2 — input builder for DiscussionAssignmentGroundingValidator.

    Per-TYPE grounding for the B10 discussion + assignment items. Surfaces
    ``{assessments_path, chunks_path, synthesized_objectives_path}`` —
    the SAME three inputs :func:`_build_assessment_objective_alignment`
    resolves, because the per-type validator reuses that validator's
    chunk-side ``learning_outcome_refs`` resolution surface (no new model
    load). When the upstream phase surfaces the in-memory discussion /
    assignment item lists directly (``discussion_items`` /
    ``assignment_items``), those are forwarded too; otherwise the validator
    reconstructs the item set from the ``06_assessments`` manifest at
    ``assessments_path``.

    Resolution mirrors the alignment builder exactly:

    * ``assessments_path`` — required (the 06_assessments manifest the
      validator reconstructs discussion/assignment items from). Absent →
      structured skip.
    * ``chunks_path`` — the grounding-resolution surface (every chunk's
      ``learning_outcome_refs`` + id). Derived from the assessments path /
      LibV2 archive exactly as the alignment builder does. Absent → the
      validator's graceful GROUNDING_INPUTS_UNAVAILABLE skip.
    * ``synthesized_objectives_path`` — optional union arm (W5.E parity).
    """
    # Reuse the alignment builder's path resolution verbatim — it resolves
    # assessments_path + chunks_path + synthesized_objectives_path with the
    # same fallback chain (assessments output -> imscc_chunks/corpus sibling
    # -> LibV2 archive). The per-type validator consumes the identical shape.
    inputs, missing = _build_assessment_objective_alignment(
        phase_outputs, workflow_params
    )

    # Forward the in-memory item lists when an upstream phase surfaced them
    # directly (optional fast path; the validator otherwise reconstructs them
    # from the manifest at assessments_path). phase_outputs is keyed by phase
    # name, so walk the nested per-phase dicts (mirrors _locate).
    for key in ("discussion_items", "assignment_items"):
        for phase_data in phase_outputs.values():
            if isinstance(phase_data, dict) and isinstance(
                phase_data.get(key), list
            ):
                inputs[key] = phase_data[key]
                break

    # chunks_path is the grounding surface but NOT load-bearing for the skip
    # decision — the validator graceful-skips (passed=True) when it's absent.
    # Only a missing assessments_path is a structured skip here.
    if "assessments_path" not in inputs:
        return inputs, ["assessments_path"]
    return inputs, []


def _build_cumulative_assessment(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """FR-COURSE-03 — input builder for CumulativeAssessmentValidator.

    Surfaces ``{assessments_path, synthesized_objectives_path}``. Mirrors
    :func:`_build_assessment_objective_alignment`'s assessment-path resolution
    (the assessment_synthesis phase emits ``assessments_path`` /
    ``assessment_path`` / ``output_path``) and reuses the canonical
    :func:`_resolve_objectives_path` for the TO universe. The validator
    establishes the terminal-objective count from the objectives doc and is a
    strict no-op when the course has < 4 TOs, so the objectives path is the
    load-bearing input.

    Resolution:

    * ``assessments_path`` — required. Absent → structured skip.
    * ``synthesized_objectives_path`` — required for the TO universe; absent →
      structured skip (the validator's OBJECTIVES_UNAVAILABLE arm is the
      safety net, but a missing path is a skip, not a silent pass).
    """
    assessments = _locate(
        phase_outputs, "assessments_path", "assessment_path", "output_path",
    )
    if not assessments:
        return {}, ["assessments_path"]

    inputs: Dict[str, Any] = {"assessments_path": assessments}

    synthesized = _resolve_objectives_path(phase_outputs, workflow_params)
    if synthesized:
        inputs["synthesized_objectives_path"] = synthesized
        return inputs, []
    return inputs, ["synthesized_objectives_path"]


def _build_course_level_qa(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """FR-COURSE-01 — input builder for CourseLevelQaValidator.

    The course-level §6.5 emergent-quality gate is the BROADEST builder on the
    post_rewrite_validation seam: it composes the FULL rewrite-tier ``blocks``
    set (the load-bearing signal — the per-page block-type distribution drives
    the interaction-mix + integration-close + coverage signals) PLUS the
    optional 06_assessments manifest PLUS the synthesized objectives universe
    PLUS the rollup it self-sufficiently recomputes from ``blocks``.

    Resolution:

    * ``blocks`` — required (reuses the rewrite-tier Block surface, the same
      hydration the IB6 rubric / rollup gates consume). Absent → skip.
    * ``synthesized_objectives_path`` — optional (Dimension A cumulative-TO
      coverage). When absent the coverage signal is skipped, not invented.
    * ``assessments_path`` — optional supplementary student↔instructor signal
      (the 06_assessments manifest). Usually absent at post_rewrite_validation
      because assessment_synthesis runs LATER; the block-derived B14 signal is
      the primary student↔instructor surface, so absence is graceful.

    The validator no-ops byte-stable when ED4ALL_BLOCK_QUALITY_RUBRIC is unset
    (reads the flag itself), so this builder always populates whatever it can.
    """
    inputs, missing = _build_block_input_rewrite(phase_outputs, workflow_params)
    if missing:
        # No blocks → the validator's BLOCKS_UNAVAILABLE arm is the safety net,
        # but a missing block set is a structured skip, not a silent pass.
        return inputs, missing

    synthesized = _resolve_objectives_path(phase_outputs, workflow_params)
    if synthesized:
        inputs["synthesized_objectives_path"] = synthesized

    # Optional 06_assessments manifest (supplementary student↔instructor signal).
    assessments = _locate(
        phase_outputs, "assessments_path", "assessment_path", "qti_dir", "output_path",
    )
    if assessments:
        inputs["assessments_path"] = assessments

    return inputs, []


def _build_bloom_distribution(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Input builder for BloomDistributionValidator.

    Mirrors :func:`_build_course_level_qa`: surfaces the rewrite-tier ``blocks``
    set (the SECONDARY corroborating ``target_bloom`` signal, surfaced in
    metadata only) PLUS the synthesized objectives universe (the PRIMARY signal —
    the course's intended cognitive demand). The objectives path drives the gate;
    blocks are corroborating, so a missing block set is NOT a hard skip — the
    validator can still audit the objective Bloom mix.

    The validator no-ops byte-stable when ED4ALL_BLOOM_DISTRIBUTION is unset
    (reads the flag itself), so this builder always populates whatever it can.
    """
    inputs, missing = _build_block_input_rewrite(phase_outputs, workflow_params)
    # Blocks are a SECONDARY signal; if absent, keep going with objectives only.
    if missing:
        inputs = {}
    synthesized = _resolve_objectives_path(phase_outputs, workflow_params)
    if synthesized:
        inputs["synthesized_objectives_path"] = synthesized
    return inputs, []


def _build_prereq_sequencing(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """Input builder for PrereqSequencingValidator (inter_tier_validation).

    Surfaces the synthesized objectives universe (the emitted TO order +
    ``key_concepts``) and the archived ``concept_graph_semantic.json`` path so
    the validator can project the TO->TO prerequisite graph. Both are optional:
    the validator graceful-skips (``passed=True``) when either is absent, so a
    missing graph / objectives is never a hard failure.

    The validator no-ops byte-stable when ED4ALL_PREREQ_SEQUENCING is unset
    (reads the flag itself), so this builder always populates whatever it can.
    """
    inputs: Dict[str, Any] = {}
    synthesized = _resolve_objectives_path(phase_outputs, workflow_params)
    if synthesized:
        inputs["synthesized_objectives_path"] = synthesized
        # Prefer the POST-sequencing TO order (the emitted order) when the
        # outline phase persisted it. The prereq sequencer reorders TOs in
        # memory and does NOT rewrite synthesized_objectives.json, so auditing
        # the planning artifact would flag prerequisites the sequencer fixed.
        # The sidecar carries the full reordered terminal_objectives list (with
        # ``key_concepts``), which the validator prefers over the path.
        sequenced = Path(synthesized).with_name(
            "synthesized_objectives.sequenced.json"
        )
        if sequenced.exists() and sequenced.is_file():
            try:
                seq_doc = json.loads(sequenced.read_text(encoding="utf-8"))
                seq_tos = (seq_doc or {}).get("terminal_objectives")
                if isinstance(seq_tos, list) and seq_tos:
                    inputs["terminal_objectives"] = [
                        t for t in seq_tos if isinstance(t, dict)
                    ]
            except (OSError, ValueError, TypeError):
                pass
    # Prefer the typed SEMANTIC graph (carries the ``prerequisite`` edges) over
    # the untyped concept_graph.json; fall back to whatever concept graph path
    # is threaded. A non-semantic graph simply projects zero prereq edges →
    # graceful NO_EDGES skip.
    graph = _locate(
        phase_outputs,
        "concept_graph_semantic_path",
        "concept_graph_path",
    )
    if graph:
        sib = Path(graph).with_name("concept_graph_semantic.json")
        if sib.exists():
            graph = str(sib)
        inputs["concept_graph_path"] = graph
    return inputs, []


def _build_cross_week_spacing(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    """C3-6 — input builder for CrossWeekSpacingValidator.

    The cross-week distributed-practice gate needs the FULL rewrite-tier
    ``blocks`` set — the load-bearing signal, since the per-block objective
    ids / concept tags + their ``week_NN`` page-id grouping are what drive the
    cross-week distribution measurement. Reuses the broadest post_rewrite Block
    builder (the same hydration the IB6 rubric / rollup / course_level_qa gates
    consume), so a missing block set is a structured skip, not a silent pass.

    The validator no-ops byte-stable when ED4ALL_BLOCK_QUALITY_RUBRIC is unset
    (reads the flag itself), so this builder always populates whatever it can.
    """
    return _build_block_input_rewrite(phase_outputs, workflow_params)


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
        "lib.validators.wcag.WCAGValidator",
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
        "lib.validators.bloom.alignment.BloomAlignmentValidator",
        _build_bloom_alignment,
    )
    # W-D10 T10.1 back-compat: re-register under the legacy flat path so
    # any caller still passing the pre-subpackage dotted path resolves
    # to the same builder. Drop alongside the shim removal in the next
    # minor version.
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
        "lib.validators.libv2.manifest.LibV2ManifestValidator",
        _build_libv2_manifest,
    )
    # W-D10 T10.1 back-compat alias for the pre-subpackage flat path.
    r.register(
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
        _build_libv2_manifest,
    )
    # Wave 78: packet integrity validator (gates the libv2_archival
    # phase fail-closed). Reuses the same input shape as the manifest
    # validator (course_dir + manifest_path) — the validator's gate
    # adapter resolves archive_root from either.
    r.register(
        "lib.validators.libv2.packet_integrity.PacketIntegrityValidator",
        _build_libv2_manifest,
    )
    # W-D10 T10.1 back-compat alias for the pre-subpackage flat path.
    r.register(
        "lib.validators.libv2_packet_integrity.PacketIntegrityValidator",
        _build_libv2_manifest,
    )
    # Honest IRT difficulty-calibration scaffold: the DifficultyProvenance
    # validator reads the archived chunkset from course_dir, so it reuses the
    # manifest builder's {course_dir, manifest_path} input shape (the
    # validator resolves <course_dir>/{imscc,dart}_chunks/chunks.jsonl).
    r.register(
        "lib.validators.difficulty_provenance.DifficultyProvenanceValidator",
        _build_libv2_manifest,
    )
    # "True full course" archival-completeness gate. Inspects the archived
    # course dir (chunks + vector index integrity), so it reuses the same
    # {course_dir, manifest_path} input shape as the manifest validator —
    # the validator resolves <course_dir>/{imscc,dart}_chunks/chunks.jsonl
    # and <course_dir>/vector_index/.
    r.register(
        "lib.validators.libv2.course_completeness.CourseCompletenessValidator",
        _build_libv2_manifest,
    )
    r.register(
        "lib.validators.assessment_objective_alignment.AssessmentObjectiveAlignmentValidator",
        _build_assessment_objective_alignment,
    )
    # C3-2 — DiscussionAssignmentGroundingValidator: per-TYPE grounding for
    # the B10 discussion + assignment items (warning day-1). Replaces the
    # AssessmentObjectiveAlignmentValidator stand-in on the
    # discussion_assignment_grounded gate. Reuses the alignment builder's
    # {assessments_path, chunks_path, synthesized_objectives_path} resolution.
    r.register(
        "lib.validators.discussion_assignment_grounding.DiscussionAssignmentGroundingValidator",
        _build_discussion_assignment_grounding,
    )
    # FR-COURSE-03 — CumulativeAssessmentValidator audits whether the final
    # graded assessment (B14) spans >= 2 TOs when the course has >= 4 TOs.
    # Surfaces {assessments_path, synthesized_objectives_path}; strict no-op
    # when the course has < 4 TOs. Mirrors the assessment_objective_alignment
    # builder's path resolution.
    r.register(
        "lib.validators.cumulative_assessment.CumulativeAssessmentValidator",
        _build_cumulative_assessment,
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

    # Anti-silent-template guard: blocks the deterministic generate_week
    # template emitter from passing as real LLM content authoring.
    try:
        from lib.validators.content_authorship import _build_content_authorship
        r.register(
            "lib.validators.content_authorship.ContentAuthorshipValidator",
            _build_content_authorship,
        )
    except ImportError:  # pragma: no cover
        logger.warning("content_authorship validator import failed")

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
        _build_block_curie_anchoring_input,
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
    # CB5a — stub-example safety-net gate (post_rewrite_validation only).
    # Pure HTML/text; needs only the rewrite-tier ``blocks`` surface, so it
    # reuses the same rewrite-tier Block-input shim as the four
    # Block*Validators above.
    r.register(
        "lib.validators.example_completeness.ExampleCompletenessValidator",
        _build_block_input_rewrite,
    )
    # IB6.6 — block->module->course quality-rollup GATE (FR-07/13, framework
    # §6.5). Self-sufficient: it scores the rewrite-tier ``blocks`` surface
    # with the canonical IB6.1 rubric scorer and rolls the scores up via the
    # BlockQualityRollupAggregator, returning passed iff course_pass. Reuses
    # the same rewrite-tier Block-input shim as the rubric gate above; no-ops
    # byte-stable when ED4ALL_BLOCK_QUALITY_RUBRIC is unset.
    r.register(
        "lib.validators.block_quality_rollup.BlockQualityRollupValidator",
        _build_block_input_rewrite,
    )
    # FR-COURSE-01 — course-level §6.5 emergent-quality QA gate. The BROADEST
    # post_rewrite_validation builder: surfaces the full rewrite-tier ``blocks``
    # set + the optional synthesized objectives universe + the optional
    # 06_assessments manifest so the validator can COMPOSE the
    # interaction-mix (OSCQR item 34), integration-close, cumulative-TO-coverage,
    # and retrieval-rhythm signals (it self-sufficiently recomputes the rollup
    # from ``blocks``). No-op + byte-stable when ED4ALL_BLOCK_QUALITY_RUBRIC is
    # unset (the validator reads the flag itself).
    r.register(
        "lib.validators.course_level_qa.CourseLevelQaValidator",
        _build_course_level_qa,
    )
    # Course-level Bloom-distribution-vs-target-curve gate. Surfaces the
    # synthesized objectives universe (PRIMARY signal — the course's intended
    # cognitive demand) + the optional rewrite-tier ``blocks`` (SECONDARY
    # ``target_bloom`` corroboration). No-op + byte-stable when
    # ED4ALL_BLOOM_DISTRIBUTION is unset (the validator reads the flag itself).
    r.register(
        "lib.validators.bloom_distribution.BloomDistributionValidator",
        _build_bloom_distribution,
    )
    # Prerequisite-sequencing gate (inter_tier_validation). Surfaces the
    # synthesized objectives universe (emitted TO order + key_concepts) + the
    # archived concept_graph_semantic.json so the validator can audit
    # dependent-before-prerequisite ordering. Graceful skip when either is
    # absent; no-op + byte-stable when ED4ALL_PREREQ_SEQUENCING is unset.
    r.register(
        "lib.validators.prereq_sequencing.PrereqSequencingValidator",
        _build_prereq_sequencing,
    )
    # C3-6 — course-level CROSS-WEEK distributed-practice (spacing) gate. Reuses
    # the broadest post_rewrite Block surface: the per-block objective ids /
    # concept tags + their ``week_NN`` page-id grouping drive the cross-week
    # distribution measurement. No-op + byte-stable when
    # ED4ALL_BLOCK_QUALITY_RUBRIC is unset (the validator reads the flag itself).
    r.register(
        "lib.validators.cross_week_spacing.CrossWeekSpacingValidator",
        _build_cross_week_spacing,
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
    # Investigation a6d6291b — the four distractor-quality validators are
    # wired at the outline_* + rewrite_* assessment seams in workflows.yaml
    # but had NO builder registered here. On the executor's phase-level gate
    # path that returned __no_builder_registered__, so the executor marked
    # them SKIPPED with passed=True and they NEVER ran on assessment blocks.
    # Each filters internally to block_type == "assessment_item", so all four
    # need the same rewrite-tier Block surface as the W7 gate above. Three of
    # them consume only ``inputs['blocks']``; DistractorStructuralValidator
    # additionally reads an OPTIONAL ``inputs['source_chunks']`` to enable its
    # distractor-not-entailed-by-source entailment sub-check, so it uses the
    # Group-B rewrite-block + source_chunks builder (graceful-degrades to the
    # structural-only check when no chunk bodies resolve).
    r.register(
        "lib.validators.distractor_plausibility.DistractorPlausibilityValidator",
        _build_block_input_rewrite,
    )
    r.register(
        "lib.validators.distractor_misconception_alignment.DistractorMisconceptionAlignmentValidator",
        _build_block_input_rewrite,
    )
    r.register(
        "lib.validators.padded_distractor.PaddedDistractorValidator",
        _build_block_input_rewrite,
    )
    r.register(
        "lib.validators.distractor_structural.DistractorStructuralValidator",
        _build_rewrite_block_input,
    )
    # Net-new numeric-equivalence gate (a6d6291b follow-up). The four
    # distractor validators above do token-Jaccard distinctness only, so a
    # distractor mathematically EQUAL to the correct answer (4/6 vs 2/3 →
    # Jaccard 0.0 → "distinct") sails through. This gate parses each option
    # to a fractions.Fraction and fails any distractor that equals the
    # correct answer's value. Filters internally to block_type ==
    # "assessment_item" and consumes only ``inputs['blocks']``, so it reuses
    # the same rewrite-tier Block surface as the distractor gates above.
    r.register(
        "lib.validators.assessment_numeric_equivalence.AssessmentNumericEquivalenceValidator",
        _build_block_input_rewrite,
    )

    # Group B — Rewrite-emit shape + sentence-grounding gates. Reuse
    # the rewrite-tier Block surface and additionally surface
    # source_chunks from the staging manifest.
    r.register(
        "lib.validators.rewrite_html_shape.RewriteHtmlShapeValidator",
        _build_block_input_rewrite,
    )
    # IB4.5 — UdlCoverageValidator audits the UDL multiple-means coverage of the
    # Block batch (derives on read). Consumes only ``inputs['blocks']`` so it
    # reuses the rewrite-tier Block surface (falls through to outline-tier
    # inside _build_block_input when only blocks_outline_path is present).
    r.register(
        "lib.validators.udl_coverage.UdlCoverageValidator",
        _build_block_input_rewrite,
    )
    # B15 — ResourceLinkPurposeValidator audits WCAG 2.4.4 descriptive-link-text
    # on ``resources`` blocks. Consumes only ``inputs['blocks']`` so it reuses
    # the rewrite-tier Block surface (no-ops + byte-stable when
    # ED4ALL_NEW_BLOCK_TYPES is unset; the B15 type is only selectable then).
    r.register(
        "lib.validators.resource_link_purpose.ResourceLinkPurposeValidator",
        _build_block_input_rewrite,
    )
    # FR-INT-02 — B08SequenceValidator audits the guided-practice sequence
    # (follows a worked_example) + fade_state presence on ``guided_practice``
    # blocks. Consumes only ``inputs['blocks']`` (IN ORDER) so it reuses the
    # rewrite-tier Block surface (no-ops + byte-stable when ED4ALL_NEW_BLOCK_TYPES
    # is unset; the B08 first-class type is only selectable then).
    r.register(
        "lib.validators.b08_sequence.B08SequenceValidator",
        _build_block_input_rewrite,
    )
    # FR-INT-06 — B09DebriefValidator audits the mandatory case/scenario debrief
    # in the transition/consolidate slot on ``scenario`` blocks. Consumes only
    # ``inputs['blocks']`` so it reuses the rewrite-tier Block surface (no-ops +
    # byte-stable when ED4ALL_NEW_BLOCK_TYPES is unset; the same flag the B09
    # scenario_mode / debrief render scaffold rides).
    r.register(
        "lib.validators.b09_debrief.B09DebriefValidator",
        _build_block_input_rewrite,
    )
    # FR-INT-04 — B10ProtocolValidator audits the three-move discussion protocol
    # (post -> respond -> synthesize) on ``discussion_prompt`` blocks. Consumes
    # only ``inputs['blocks']`` so it reuses the rewrite-tier Block surface
    # (no-ops + byte-stable when ED4ALL_NEW_BLOCK_TYPES is unset; the same flag
    # the B10 three-move render scaffold rides).
    r.register(
        "lib.validators.b10_protocol.B10ProtocolValidator",
        _build_block_input_rewrite,
    )
    # FR-A11Y-02 — InteractiveA11yValidator audits WCAG 2.1.1/2.5.7
    # (drag-only-no-keyboard) + 1.4.1 (colour-only signalling) on interaction
    # blocks. Consumes only ``inputs['blocks']`` (+ the threaded
    # ``block_a11y_enabled``) so it reuses the rewrite-tier Block surface
    # (no-ops + byte-stable when ED4ALL_BLOCK_A11Y is unset).
    r.register(
        "lib.validators.interactive_a11y.InteractiveA11yValidator",
        _build_block_input_rewrite,
    )
    # FR-A11Y-03 — CalloutStructureValidator audits typed B12 callouts (WCAG
    # 1.4.1 non-color coding + body-overflow + motion). Consumes only
    # ``inputs['blocks']`` so it reuses the rewrite-tier Block surface (no-ops +
    # byte-stable when ED4ALL_CALLOUT_TYPED is unset — the same flag that makes
    # the typed callout renderer emit the redundant label/icon/border).
    r.register(
        "lib.validators.callout_structure.CalloutStructureValidator",
        _build_block_input_rewrite,
    )
    # W1.5 — KeyTermsDefinitionQualityValidator audits key-terms vocab cards
    # (template_type == "key_terms" or block_type == "vocab_card") for circular /
    # too-long / not-distinct glossary definitions. Consumes only
    # ``inputs['blocks']`` so it reuses the rewrite-tier Block surface (reads its
    # own ED4ALL_KEYTERM_DEF_QUALITY flag + vocab_card body ceiling; no-ops +
    # byte-stable when the flag is unset).
    r.register(
        "lib.validators.key_terms_definition_quality.KeyTermsDefinitionQualityValidator",
        _build_block_input_rewrite,
    )
    # Mayer-CTML — MayerCtmlValidator audits text+visual blocks (B04/B06/B05 or
    # any block whose rendered HTML carries <figure>/<img>/<video>) over a
    # precision-first subset of Mayer's CTML principles (signaling / spatial
    # contiguity / redundancy / segmenting). Consumes only ``inputs['blocks']``
    # (+ the threaded ``mayer_ctml_enabled``) so it reuses the rewrite-tier Block
    # surface (no-ops + byte-stable when ED4ALL_MAYER_CTML is unset).
    r.register(
        "lib.validators.mayer_ctml.MayerCtmlValidator",
        _build_block_input_rewrite,
    )
    # recall_self_check — RecallSelfCheckFormatValidator audits free-recall /
    # cloze self_check_question blocks (no pre-enumerated options + no inline
    # answer reveal). Consumes only ``inputs['blocks']`` (+ the threaded
    # ``recall_self_check_enabled``) so it reuses the rewrite-tier Block surface
    # (no-ops + byte-stable when ED4ALL_RECALL_SELF_CHECK is unset).
    r.register(
        "lib.validators.recall_self_check.RecallSelfCheckFormatValidator",
        _build_block_input_rewrite,
    )
    # misconception_productive_failure — MisconceptionProductiveFailureValidator
    # audits B03/B12 misconception blocks for a named faulty model + a
    # predict-then-reveal-then-reconcile productive-failure scaffold. Consumes
    # only ``inputs['blocks']`` (+ the threaded ``misconception_rich_enabled``)
    # so it reuses the rewrite-tier Block surface (no-ops + byte-stable when
    # ED4ALL_MISCONCEPTION_RICH is unset).
    r.register(
        "lib.validators.misconception_productive_failure.MisconceptionProductiveFailureValidator",
        _build_block_input_rewrite,
    )
    # FR-COURSE-02 — BlockSequenceOrderValidator audits the worked-example ->
    # guided-practice fade gradient, check spacing, and scenario-opens-TO on
    # each content module. Consumes only ``inputs['blocks']`` so it reuses the
    # outline-tier Block surface (mirrors retrieval_presence — runs warning-
    # day-1 regardless of any flag).
    r.register(
        "lib.validators.block_sequence_order.BlockSequenceOrderValidator",
        _build_block_input_outline,
    )
    r.register(
        "lib.validators.rewrite_source_grounding.RewriteSourceGroundingValidator",
        _build_rewrite_block_input,
    )
    # W4 — full-prose NLI entailment gate (post_rewrite_validation). Reuses
    # the same rewrite-block + source_chunks builder so the NLI premise map is
    # populated from the authoritative DART chunkset chunks.jsonl.
    r.register(
        "lib.validators.block_prose_entailment.BlockProseEntailmentValidator",
        _build_rewrite_block_input,
    )
    # W4 §0.1 FIX — claim_support was NEVER registered, so the executor's
    # __no_builder_registered__ contract ran it with source_chunks={} (a
    # silent no-op: every claim hit the empty-premise branch). Wire the same
    # rewrite-block + source_chunks builder so the per-claim NLI premise map is
    # populated. REQUIRED before the (DEFERRED) claim_support critical flip —
    # promoting it critical without this would fail-close every two-pass run
    # against an empty premise.
    r.register(
        "lib.validators.claim_support.ClaimSupportValidator",
        _build_rewrite_block_input,
    )
    # Numeric-literal grounding gate — the fabrication control for NUMERIC /
    # math content the NLI gate (block_prose_entailment) cannot provide.
    # Established this session (content-block-quality iters 5/5b):
    # DeBERTa-v3-mnli is NUMBER-BLIND (scores a fabricated worked-example input
    # ABOVE every grounded math claim), and groundedness._is_computational
    # EXEMPTS such claims, so the NLI gate gives zero numeric fabrication
    # protection. This gate cross-checks each prose fraction's num/denom pair
    # against the block's cited source under OCR-tolerant containment. Reuses
    # the same rewrite-block + source_chunks builder so the source-text premise
    # is populated from the authoritative DART chunkset chunks.jsonl.
    r.register(
        "lib.validators.numeric_literal_grounding.NumericLiteralGroundingValidator",
        _build_rewrite_block_input,
    )
    # Symbolic worked-example verification gate — the CORRECTNESS control for the
    # MATH in worked examples that provenance / number-blind NLI / numeric-literal
    # grounding cannot provide (it re-checks the DERIVATION under sympy, not the
    # grounding of individual numbers). Consumes ONLY ``inputs['blocks']`` (the
    # symbolic re-check is internal to the block's own HTML — no source premise),
    # so it reuses the rewrite-tier Block-input shim (NOT the +source_chunks
    # builder the numeric-grounding gate needs). Falls through to the outline-tier
    # path inside _build_block_input when only blocks_outline_path is present.
    r.register(
        "lib.validators.worked_example_math.WorkedExampleMathValidator",
        _build_block_input_rewrite,
    )
    # Deterministic rewrite-tier content lint — the STRUCTURAL-shape control for
    # leaked authoring machinery (escaped/namespaced/custom-element pseudo-tags,
    # \text{} slug leftovers, generic numbered-publisher-apparatus cross-refs,
    # doubled-term / bare-slug definition glue) the semantic gates (NLI /
    # numeric / symbolic-math) are blind to. Regex-only over the block's own
    # rendered HTML — no source premise — so it reuses the rewrite-tier
    # Block-input shim (NOT the +source_chunks builder), mirroring
    # worked_example_math. Issues carry the block id so courseforge-rewrite
    # --block-ids can re-roll exactly the leaking blocks.
    r.register(
        "lib.validators.rewrite_content_lint.RewriteContentLintValidator",
        _build_block_input_rewrite,
    )

    # Gap #11 near-dup anchor-example gate — flags one worked example (its
    # normalized number-sequence signature) recurring across >= N blocks in a
    # module. Wired at inter_tier_validation (outline tier), so it reuses the
    # outline-tier Block-input shim (``inputs['blocks']`` only; no source
    # premise). Issues carry the module key in GateIssue.location.
    r.register(
        "lib.validators.near_dup_example.NearDupExampleValidator",
        _build_block_input_outline,
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
        "lib.validators.bloom.classifier_disagreement.BloomClassifierDisagreementValidator",
        _build_block_statistical_input,
    )
    # W-D10 T10.1 back-compat alias for the pre-subpackage flat path.
    r.register(
        "lib.validators.bloom_classifier_disagreement.BloomClassifierDisagreementValidator",
        _build_block_statistical_input,
    )
    # GPT Feedback v2 Wave 1.7 W1.7.C — tri-axis per-block-per-objective
    # delivery gate. Reuses the statistical-tier builder so the Drift B
    # fix (objective_statements + objectives surfacing from
    # synthesized_objectives.json) flows through unchanged.
    r.register(
        "lib.validators.block_objective_delivery.BlockObjectiveDeliveryValidator",
        _build_block_statistical_input,
    )
    # IB3.4 / IB3.5 — constructive-alignment keystone gates (anchored
    # rubric on Evaluate/Create scored blocks; triangle completeness per
    # objective). Both consume the Block surface (``inputs['blocks']`` +
    # ``inputs['objectives']``) and handle the OUTLINE-tier dict-content
    # block shape, so the statistical-tier builder — which surfaces both
    # blocks and the full objectives map (rewrite-tier-then-outline-tier
    # fallthrough) — feeds them at both the outline + rewrite seams. The
    # FR-11 anchored-rubric producer authors Block.anchored_rubric at the
    # outline phase, so the rubric is present by inter_tier_validation.
    r.register(
        "lib.validators.alignment.anchored_rubric.AnchoredRubricValidator",
        _build_block_statistical_input,
    )
    r.register(
        "lib.validators.alignment.triangle_completeness.TriangleCompletenessValidator",
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

    # Wave 1.6 W1.6.C — per-objective source-attribution gate at
    # course_planning. Builder resolves synthesized_objectives_path,
    # textbook_structure_path, and the DART chunkset manifest sidecar
    # from upstream phase outputs. ``require_to_attribution`` surfaces
    # via gate ``config:`` block.
    r.register(
        "lib.validators.objective_source_refs.ObjectiveSourceRefValidator",
        _build_objective_source_refs,
    )
    # W4 — LO-statement NLI entailment gate at course_planning. IDENTICAL
    # inputs to objective_source_refs (synthesized_objectives_path +
    # dart_chunks_manifest_path), so reuse the same builder; the validator's
    # own id→text loader reads the sibling chunks.jsonl for premise text.
    r.register(
        "lib.validators.objective_entailment.ObjectiveEntailmentValidator",
        _build_objective_source_refs,
    )

    # W3 — per-block synthesis-manifest RESOLUTION gate at content_generation +
    # post_rewrite_validation. Builder resolves the block_synthesis_manifest.jsonl
    # sidecar from the resolved content_dir + the DART chunkset manifest.json
    # (resolution universe) from chunking.dart_chunks_path. Pre-registration NO
    # builder was wired, so the executor returned __no_builder_registered__ and
    # the gate skipped silently — the manifest emission would be unvalidated.
    r.register(
        "lib.validators.manifest_completeness.ManifestCompletenessValidator",
        _build_manifest_completeness,
    )

    # Wave 3 W3.D — DART vs. IMSCC chunkset drift detector at
    # libv2_archival. Builder resolves dart_chunks_path /
    # imscc_chunks_path from the chunking + imscc_chunking phase
    # outputs (when present); falls back to deterministic LibV2 paths
    # under <course_dir>/dart_chunks/chunks.jsonl and
    # <course_dir>/imscc_chunks/chunks.jsonl (or the Phase-7c-shim
    # legacy <course_dir>/corpus/chunks.jsonl) when the phase outputs
    # don't surface them directly.
    r.register(
        "lib.validators.chunkset_drift.ChunksetDriftValidator",
        _build_chunkset_drift,
    )

    # Wave2-I9 — Finding 5: three validators silently skipped via
    # ``__no_builder_registered__`` because the router didn't know how
    # to derive their required inputs from phase outputs. The gate ran
    # ``passed=True warning-severity`` (a no-op), preventing the
    # validator from ever inspecting the artifact.
    #
    # ChunksetManifestValidator fires at both ``chunking`` (DART) and
    # ``imscc_chunking`` (IMSCC) phases; the builder handles both by
    # deriving ``<chunks_dir>/manifest.json`` from the sibling
    # chunks.jsonl path.
    r.register(
        "lib.validators.chunkset_manifest.ChunksetManifestValidator",
        _build_chunkset_manifest_inputs,
    )
    # IB4.2 — ChunkWcagStatusValidator gates the data-only chunk WCAG fields
    # (wcag_block_status / figure_alt) at chunking (DART) + imscc_chunking
    # (IMSCC). Needs the chunkset JSONL path; warning-day-1.
    r.register(
        "lib.validators.chunk_wcag_status.ChunkWcagStatusValidator",
        _build_chunk_wcag_status,
    )
    # W10 §7 — QtiWellFormedValidator fires at the ``assessment_synthesis``
    # phase and needs ``inputs["qti_dir"] = <export>/06_assessments``. The
    # builder derives that from the resolved project export root (or an
    # explicit qti_dir/assessment_dir phase output).
    r.register(
        "lib.validators.qti_well_formed.QtiWellFormedValidator",
        _build_qti_well_formed,
    )
    # W7.3 — SynthesizedQuizDistractorValidator fires at the
    # ``assessment_synthesis`` phase (both course_generation and
    # textbook_to_course) as the warning-severity
    # ``synthesized_quiz_distractor`` gate. It audits every synthesized MCQ's
    # distractors (equals-key / duplicate / too-few / placeholder) — the QTI
    # item shape the authored-block distractor_* gates never reach. Its input
    # contract is IDENTICAL to QtiWellFormedValidator (it globs
    # ``inputs["qti_dir"] = <export>/06_assessments`` for ``*.xml``), so it
    # reuses that builder verbatim; no new builder. Warning day-1; deferred
    # critical-flip. Flip-wave-2 measurement: ZERO observations across the
    # discovered calibration corpora (the 06_assessments QTI artifact only
    # exists on runs that emit synthesized assessments) — the blocker is a
    # gate-never-exercised data gap needing >=2 assessment-emitting runs, NOT a
    # missing corpus or an unbuilt harness.
    r.register(
        "lib.validators.synthesized_quiz_distractor.SynthesizedQuizDistractorValidator",
        _build_qti_well_formed,
    )
    # ConceptGraphValidator fires at ``concept_extraction`` and needs
    # the path to concept_graph_semantic.json (a declared YAML output).
    r.register(
        "lib.validators.concept_graph.ConceptGraphValidator",
        _build_concept_graph_inputs,
    )
    # AbcdObjectiveValidator fires at ``course_planning`` as
    # ``abcd_verb_alignment`` and needs synthesized_objectives.json.
    r.register(
        "lib.validators.abcd_objective.AbcdObjectiveValidator",
        _build_abcd_objective_inputs,
    )
    # DomainConceptVocabularyValidator fires at ``concept_extraction``
    # as ``domain_concept_vocabulary`` (Stage 3, Wave C) and needs the
    # optional domain_concept_vocabulary.json path.
    r.register(
        "lib.validators.domain_concept_vocabulary.DomainConceptVocabularyValidator",
        _build_domain_concept_vocabulary_inputs,
    )
    # ChapterObjectiveCoverageValidator fires at ``course_planning`` as
    # ``chapter_objective_coverage`` (Stage 2, Wave D) and needs the
    # textbook_structure (for chapters[].chapter_text) + the
    # synthesized_objectives (for chapter_objectives + reconciled
    # terminal_objectives). Skips-with-pass on a default-off run.
    r.register(
        "lib.validators.chapter_objective_coverage.ChapterObjectiveCoverageValidator",
        _build_chapter_objective_coverage_inputs,
    )
    # TerminalObjectiveCoverageValidator fires at ``course_planning`` as
    # the fail-fast ``terminal_objective_coverage`` gate. It consumes the
    # SAME two inputs as ChapterObjectiveCoverageValidator — the
    # synthesized_objectives (for terminal_objectives + chapter_objectives
    # roll-up) + the textbook_structure (to adjudicate CO-less courses) —
    # so it reuses that builder verbatim. The validator graceful-degrades
    # when either is absent.
    r.register(
        "lib.validators.terminal_objective_coverage.TerminalObjectiveCoverageValidator",
        _build_chapter_objective_coverage_inputs,
    )
    # WS3 — CoTerminalAlignmentValidator fires at ``course_planning`` as the
    # warning-severity ``co_terminal_alignment`` gate. It recomputes
    # cosine(co.statement, assigned_to.statement) per chapter objective to
    # close the structural-roll-up silent-pass loophole. It consumes the
    # SAME two inputs as the two coverage validators above
    # (synthesized_objectives for the CO/TO statements; textbook_structure
    # accepted-and-ignored for builder-shape compat), so it reuses that
    # builder verbatim — no new builder.
    r.register(
        "lib.validators.co_terminal_alignment.CoTerminalAlignmentValidator",
        _build_chapter_objective_coverage_inputs,
    )
    # WS6a — SourceCoverageValidator fires at ``course_planning`` as the
    # warning-severity ``source_coverage`` gate. It embeds each
    # content-bearing textbook section and asserts ≥1 synthesized objective
    # (CO or TO) covers it above a cosine floor — a measurement guardrail
    # for an objectives set that misses source material. It consumes the
    # SAME two inputs as the coverage validators above (synthesized_objectives
    # for the CO/TO statements; textbook_structure for the chapters[].sections[]
    # it audits — both surfaced by this builder), so it reuses that builder
    # verbatim — no new builder.
    r.register(
        "lib.validators.source_coverage.SourceCoverageValidator",
        _build_chapter_objective_coverage_inputs,
    )
    # W7.5 (M-TO) — TerminalObjectiveSourceGroundingValidator fires at
    # ``course_planning`` as the opt-in (``ED4ALL_TO_SOURCE_GROUNDING``)
    # warning-severity ``terminal_objective_source_grounding`` gate. It embeds
    # each TO statement + the source chunks its cluster cites and flags a TO
    # whose best supporting chunk is below the cosine floor (the
    # coherent-but-hallucinated-TO class). It reads the SAME two inputs this
    # builder already surfaces — ``synthesized_objectives_path`` (for the TO/CO
    # statements + source_refs/source_chunk_ids the cluster->chunk map is
    # reconstructed from) and ``dart_chunks_path`` (the source-chunk text
    # universe) — so it reuses that builder verbatim; no new builder. Default
    # OFF → the validator's own no-op skip-with-pass fires.
    r.register(
        "lib.validators.terminal_objective_source_grounding.TerminalObjectiveSourceGroundingValidator",
        _build_chapter_objective_coverage_inputs,
    )
    # W2 Defect B — ObjectiveSpecificityValidator fires at ``course_planning`` as
    # the opt-in (``ED4ALL_OBJECTIVE_SPECIFICITY``) warning-severity
    # ``objective_specificity`` gate (wired AFTER ``objective_entailment``). Its
    # three deterministic checks (V1 content-residual vacuity, V2 vague-object,
    # V3 source-token recall) read the SAME two inputs this builder already
    # surfaces — ``synthesized_objectives_path`` (the CO statements +
    # source_chunk_ids/source_refs) and ``dart_chunks_path`` (the source-chunk
    # text universe for V3) — so it reuses that builder verbatim; no new builder.
    # Default OFF → the validator's own no-op skip-with-pass fires.
    r.register(
        "lib.validators.objective_specificity.ObjectiveSpecificityValidator",
        _build_chapter_objective_coverage_inputs,
    )
    # Three-stage textbook synthesis (Wave A/B): TextbookOutlineValidator
    # fires at ``textbook_to_course::objective_extraction`` as the
    # critical / block ``textbook_outline_enrichment`` gate. Pre-
    # registration NO builder was wired, so the router returned
    # ``__no_builder_registered__`` and the gate skipped with a warning on
    # every run. The builder routes textbook_structure_path from the
    # objective_extraction phase output (a declared YAML output); the
    # validator skips-with-pass when the Stage-1 enrichment keys are
    # absent (default-off runs).
    r.register(
        "lib.validators.textbook_structure.TextbookOutlineValidator",
        _build_textbook_outline_inputs,
    )

    # Pre-synthesis chunk-health gate: ChunkHealthValidator fires at
    # ``textbook_to_course::objective_extraction`` (AFTER chunking + the
    # extractor, BEFORE course_planning) as the opt-in
    # ``chunk_health`` gate. The builder routes the emitted chunkset
    # (chunking.semantik_chunks_path) + textbook_structure_path so the
    # validator can audit the chunkset the course is about to be synthesized
    # from. Opt-in via ED4ALL_CHUNK_HEALTH_GATE (default OFF); the validator
    # skips-with-pass BEFORE touching inputs when the flag is unset, so a
    # default-off run is byte-identical.
    r.register(
        "lib.validators.chunk_health.ChunkHealthValidator",
        _build_chunk_health,
    )

    # Activate-the-dormant-gate: KGQualityValidator fires at
    # ``textbook_to_course::libv2_archival`` as the critical / block /
    # fail_closed ``kg_quality_report`` gate. Pre-activation NO builder
    # was registered, so the router returned
    # ``__no_builder_registered__`` and the executor stamped the gate
    # GATE_SKIPPED_MISSING_INPUTS (passed=True) — it NEVER ran. The
    # builder routes course_slug / run_id / output_dir (LibV2 quality/) +
    # the semantic graph (concept_extraction.concept_graph_path points at
    # concept_graph_semantic.json) + a best-effort asserted-graph sibling.
    # A genuinely-missing graph leaves semantic_graph_path empty so the
    # validator's KG_QUALITY_PEDAGOGY_GRAPH_MISSING arm fails closed.
    # Threshold semantics unchanged (metric breach → warning, passed=True;
    # only missing/malformed graph or reporter exception blocks). See
    # docs/validation/gates.md.
    r.register(
        "lib.validators.kg_quality.KGQualityValidator",
        _build_kg_quality_inputs,
    )

    # ------------------------------------------------------------------ #
    # IB6 / IB7 keystone block-quality gates — latent no-builder skip.
    #
    # These validators are wired at ``inter_tier_validation`` +
    # ``post_rewrite_validation`` (and the assessment seams) in
    # config/workflows.yaml but had NO builder registered here, so the
    # executor returned ``__no_builder_registered__`` and EVERY one of
    # them skipped silently on a real run (the log line
    # "No gate-input builder registered for validator
    # lib.validators.block_quality_rubric.BlockQualityRubricValidator ...
    # missing inputs: __no_builder_registered__"). The IB6 keystone
    # rubric, anatomy slot-presence, interaction-feedback, QA-checklist,
    # Bloom type-range, cognitive-load, retrieval-presence,
    # instructional-depth, Bloom structural-enforcement, and
    # assessment-retrieval-grounding gates NEVER fired.
    #
    # Every one of these reads ``inputs['blocks']`` (the rewrite-tier
    # Block set is the load-bearing signal). The rubric/QA gates also
    # read OPTIONAL ``gate_results_by_block`` / ``spacing_by_block`` and
    # COMPOSE upstream signals when present (they degrade byte-stable to
    # an empty-signal pass when absent — never recompute). So the broadest
    # rewrite-tier Block-input shim (``_build_block_input_rewrite``, which
    # also threads the IB4/IB5/FR-INT-03 flag resolutions the anatomy /
    # interaction gates consult) is the correct input shape for all of the
    # blocks-only gates. ``assessment_retrieval_grounding`` additionally
    # needs the per-source chunk-body map (``chunks_lookup`` / fallback
    # ``source_chunks``), so it routes to ``_build_rewrite_block_input``
    # which layers ``source_chunks`` on top of the same Block surface.
    #
    # All of these reuse the shim with NO gate_id discrimination, so the
    # single registration covers both the inter_tier (outline-tier) and
    # post_rewrite (rewrite-tier) seams — the shim's blocks_path resolver
    # prefers the rewrite-tier emit when present and falls back to the
    # outline-tier emit, so the inter_tier seam resolves the outline
    # blocks correctly.

    # IB6.4 — per-block D2 cognitive-load body ceiling. blocks-only.
    r.register(
        "lib.validators.content.BlockCognitiveLoadValidator",
        _build_block_input_rewrite,
    )
    # IB6 — anatomy six-slot presence (reads reflection_calibration_enabled,
    # threaded by the shim). blocks-only.
    r.register(
        "lib.validators.anatomy_slot_presence.AnatomySlotPresenceValidator",
        _build_block_input_rewrite,
    )
    # IB6.3 — interaction→feedback contract (reads the OPTIONAL
    # distractor_signals_by_block + reflection_calibration_enabled, both
    # graceful-degrade). blocks-only.
    r.register(
        "lib.validators.interaction_feedback.InteractionFeedbackValidator",
        _build_block_input_rewrite,
    )
    # IB6 keystone — eight-dimension 0-3 block-quality rubric. COMPOSES the
    # OPTIONAL upstream ``gate_results_by_block`` signal map (degrades to an
    # empty-signal pass when absent — never recomputes / loads a model).
    # blocks-only via the shim.
    r.register(
        "lib.validators.block_quality_rubric.BlockQualityRubricValidator",
        _build_block_input_rewrite,
    )
    # IB6 — 15-point QA checklist. COMPOSES the OPTIONAL
    # ``gate_results_by_block`` + ``spacing_by_block`` maps (both
    # graceful-degrade). blocks-only via the shim.
    r.register(
        "lib.validators.qa_checklist.QaChecklistValidator",
        _build_block_input_rewrite,
    )
    # IB7.5b — retrieval-presence (spaced-checkpoint) gate. blocks-only,
    # runs warning-day-1 regardless of any flag.
    r.register(
        "lib.validators.retrieval_presence.RetrievalPresenceValidator",
        _build_block_input_rewrite,
    )
    # IB7.6c — per-type Bloom×type-range gate (reads OPTIONAL
    # ``bloom_ceilings`` override, merged via gate.config). blocks-only.
    r.register(
        "lib.validators.bloom_type_range.BloomTypeRangeValidator",
        _build_block_input_rewrite,
    )
    # W7 — per-page instructional-depth (pedagogical density) floors.
    # blocks-only (reads OPTIONAL ``thresholds`` override via gate.config).
    # Wired at outline_instructional_depth + rewrite_instructional_depth.
    r.register(
        "lib.validators.instructional_depth.InstructionalDepthValidator",
        _build_block_input_rewrite,
    )
    # W6 — deterministic Bloom structural-enforcement on assessment_item
    # stems. blocks-only. Wired at outline_bloom_structural_enforcement +
    # rewrite_bloom_structural_enforcement.
    r.register(
        "lib.validators.bloom.structural_enforcement.BloomStructuralEnforcementValidator",
        _build_block_input_rewrite,
    )
    # W2 — assessment answerability / retrieval-grounding. Needs the Block
    # set PLUS the per-source chunk-body map (``chunks_lookup`` →
    # fallback ``source_chunks``) so it can overlap each assessment_item
    # answer against its referenced source chunk. Routes to the rewrite-
    # block + source_chunks builder. Wired at
    # outline_assessment_retrieval_grounding +
    # rewrite_assessment_retrieval_grounding.
    r.register(
        "lib.validators.assessment_retrieval_grounding.AssessmentRetrievalGroundingValidator",
        _build_rewrite_block_input,
    )

    return r


__all__ = [
    "BuilderFn",
    "BuilderResult",
    "GateInputRouter",
    "default_router",
]
