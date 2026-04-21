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
    # OSCQRValidator is stubbed — just forward the discovered course_path.
    course_path = _locate(phase_outputs, "course_dir", "project_path")
    inputs: Dict[str, Any] = {}
    if course_path:
        inputs["course_path"] = course_path
    objectives = workflow_params.get("objectives_path")
    if objectives:
        inputs["objectives"] = objectives
    return inputs, []


def _build_dart_markers(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    html = _first_html_path(phase_outputs)
    if html and html.exists():
        return {"html_path": str(html)}, []
    return {}, ["html_path"]


def _build_assessment_quality(
    phase_outputs: Dict[str, Any],
    workflow_params: Dict[str, Any],
) -> BuilderResult:
    path = _locate(
        phase_outputs,
        "assessment_path",
        "output_path",
        "assessment_id",  # trainforge fallback
    )
    if not path:
        return {}, ["assessment_path"]
    return {"assessment_path": path}, []


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
    manifest = _locate(phase_outputs, "manifest_path")
    if not manifest:
        return {}, ["manifest_path"]
    inputs: Dict[str, Any] = {"manifest_path": manifest}
    course_dir = _locate(phase_outputs, "course_dir")
    if course_dir:
        inputs["course_dir"] = course_dir
    return inputs, []


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
    return r


__all__ = [
    "BuilderFn",
    "BuilderResult",
    "GateInputRouter",
    "default_router",
]
