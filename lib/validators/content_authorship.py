"""ContentAuthorshipValidator — anti-silent-template guard for content_generation.

The Courseforge ``content_generation`` phase can emit pages three ways:

* ``llm_provider``        — in-process ``ContentGeneratorProvider`` authored the
                            pages (``COURSEFORGE_PROVIDER`` set).
* ``two_pass_router``     — the outline→validate→rewrite ``CourseforgeRouter``
                            authored them (``COURSEFORGE_TWO_PASS=true``).
* ``template_deterministic`` — the deterministic ``generate_week`` template
                            emitter produced them (NO LLM).

The template path is a *fallback*: we pretty much never want it to fire in a
real run, because it slices source paragraphs into a styled shell rather than
having an LLM author grounded educational content. Before this gate, a template
result was byte-indistinguishable from LLM-authored content and every gate
rubber-stamped it — so an operator could (accidentally) service the
content-generator mailbox tasks by running the template tool and the pipeline
would never notice.

``_generate_course_content`` now stamps an honest
``content_generation_provenance.json`` recording which path ran plus the
operator's LLM-authoring *intent* (derived from ``COURSEFORGE_PROVIDER`` /
``COURSEFORGE_TWO_PASS`` / ``ED4ALL_AGENT_DISPATCH``). This validator reads it:

* template fired **and** LLM authoring was intended  -> **critical / block**
  (unless ``COURSEFORGE_ALLOW_TEMPLATE_EMITTER=1`` opts out).
* template fired with no LLM intent (offline/unconfigured smoke run) ->
  **warning** (loud flag, non-blocking).
* LLM/router authored, or no provenance file present -> **pass**.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _envflag(name: str) -> bool:
    import os
    return os.environ.get(name, "").strip().lower() in _TRUTHY


class ContentAuthorshipValidator:
    """Blocks the deterministic template emitter from passing as LLM content.

    Expected inputs (any combination):
        provenance_path: path to ``content_generation_provenance.json``.
        generator_mode / template_fallback_fired / llm_authoring_intended /
            allow_template_emitter: inline fields (tests, or surfaced from the
            content_generation phase output) — used when no file is given.
    """

    name = "content_authorship"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "content_authorship")
        prov = self._load_provenance(inputs)

        # No provenance at all -> nothing template-emitted to flag. A real
        # LLM/subagent worker that authored pages without invoking the
        # template tool legitimately leaves no provenance file behind.
        if prov is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[],
            )

        template_fired = bool(prov.get("template_fallback_fired"))
        if not template_fired:
            # Partial runtime degradation: provider/router authored some
            # pages but silently fell back to the template floor on others.
            # Not a block (real LLM content exists), but surface it loudly.
            if prov.get("llm_authoring_degraded"):
                issue = GateIssue(
                    severity="warning",
                    code="PARTIAL_TEMPLATE_FALLBACK",
                    message=(
                        "content_generation partially degraded to the "
                        "deterministic template floor: "
                        f"{prov.get('llm_authored_pages')} page(s) LLM-authored, "
                        f"{prov.get('template_fallback_pages')} page(s) fell back. "
                        "Inspect provider health (network / budget / parse misses)."
                    ),
                    location=str(inputs.get("provenance_path") or ""),
                    suggestion=(
                        "Re-run content_generation after resolving the provider "
                        "failures so every page is LLM-authored."
                    ),
                )
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=True,
                    issues=[issue],
                )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[],
            )

        # Template fired. Decide block vs warn. Trust the provenance file's
        # captured intent, but let a live env override re-assert intent (so a
        # stale provenance file can't downgrade a genuinely-intended run).
        intended = bool(prov.get("llm_authoring_intended")) or (
            _envflag("ED4ALL_AGENT_DISPATCH")
            or _envflag("COURSEFORGE_TWO_PASS")
            or bool(str(prov.get("courseforge_provider") or "").strip())
        )
        allowed = bool(prov.get("allow_template_emitter")) or _envflag(
            "COURSEFORGE_ALLOW_TEMPLATE_EMITTER"
        )
        mode = prov.get("generator_mode", "template_deterministic")
        pages = prov.get("page_count")

        base_msg = (
            "content_generation used the deterministic template emitter "
            f"(generator_mode={mode!r}, pages={pages}) instead of a real LLM "
            "content-generator. The template slices source paragraphs into a "
            "styled shell rather than authoring grounded educational content."
        )
        suggestion = (
            "Set COURSEFORGE_PROVIDER (anthropic/together/local), or "
            "COURSEFORGE_TWO_PASS=true, or service the content-generator "
            "mailbox agent_tasks with real LLM subagents. To intentionally "
            "permit the template emitter (offline smoke run), set "
            "COURSEFORGE_ALLOW_TEMPLATE_EMITTER=1."
        )

        if intended and not allowed:
            issue = GateIssue(
                severity="critical",
                code="TEMPLATE_EMITTER_IN_REAL_RUN",
                message=(
                    base_msg + " LLM authoring was the configured intent, so "
                    "this template result is rejected."
                ),
                location=str(inputs.get("provenance_path") or ""),
                suggestion=suggestion,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[issue],
            )

        # Template fired but either explicitly allowed or no LLM intent
        # detected — flag loudly, don't block.
        code = (
            "TEMPLATE_EMITTER_ALLOWED" if allowed
            else "TEMPLATE_EMITTER_NO_LLM_INTENT"
        )
        issue = GateIssue(
            severity="warning",
            code=code,
            message=(
                base_msg + (
                    " Explicitly allowed via COURSEFORGE_ALLOW_TEMPLATE_EMITTER."
                    if allowed else
                    " No LLM provider/router/dispatch was configured, so the "
                    "template was the only available emitter."
                )
            ),
            location=str(inputs.get("provenance_path") or ""),
            suggestion=suggestion,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            issues=[issue],
        )

    @staticmethod
    def _load_provenance(inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Inline fields win when no explicit path is given (tests).
        path = inputs.get("provenance_path")
        if path:
            p = Path(path)
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:  # noqa: BLE001
                    logger.warning(
                        "ContentAuthorshipValidator: unreadable provenance "
                        "%s: %s", path, exc,
                    )
                    return None
            return None
        if "template_fallback_fired" in inputs or "generator_mode" in inputs:
            return {
                "generator_mode": inputs.get("generator_mode"),
                "template_fallback_fired": inputs.get("template_fallback_fired"),
                "llm_authoring_intended": inputs.get("llm_authoring_intended"),
                "allow_template_emitter": inputs.get("allow_template_emitter"),
                "courseforge_provider": inputs.get("courseforge_provider"),
                "page_count": inputs.get("page_count"),
                "llm_authoring_degraded": inputs.get("llm_authoring_degraded"),
                "llm_authored_pages": inputs.get("llm_authored_pages"),
                "template_fallback_pages": inputs.get("template_fallback_pages"),
            }
        return None


def _build_content_authorship(
    phase_outputs: Dict[str, Any], workflow_params: Dict[str, Any]
):
    """Gate input builder: locate ``content_generation_provenance.json``.

    Prefers the explicit path surfaced by the content_generation phase output;
    falls back to globbing the project export dir.
    """
    cg = phase_outputs.get("content_generation") or {}
    prov_path = cg.get("content_generation_provenance_path")

    if not prov_path:
        # Fall back: derive from content_dir's parent (the export root).
        content_dir = cg.get("content_dir")
        if not content_dir:
            for phase_data in phase_outputs.values():
                if isinstance(phase_data, dict) and phase_data.get("content_dir"):
                    content_dir = phase_data["content_dir"]
                    break
        if content_dir:
            cand = Path(content_dir).parent / "content_generation_provenance.json"
            if cand.exists():
                prov_path = str(cand)

    inputs: Dict[str, Any] = {}
    # Always forward inline fields when present (cheap, and lets the gate run
    # even if the file write failed).
    for k in (
        "generator_mode", "template_fallback_fired", "llm_authoring_intended",
        "allow_template_emitter", "courseforge_provider", "page_count",
        "llm_authoring_degraded", "llm_authored_pages", "template_fallback_pages",
    ):
        if k in cg:
            inputs[k] = cg[k]
    if prov_path:
        inputs["provenance_path"] = prov_path

    # The gate is intentionally permissive about missing inputs: with neither
    # a provenance file nor inline fields it passes (no template signal). So we
    # never declare a required-missing key — the validator handles absence.
    return inputs, []
