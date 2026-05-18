"""Stage-1 textbook-outline enrichment validator.

Three-stage large-LLM textbook synthesis architecture, Wave A — see
``plans/textbook-llm-synthesis-3stage-2026-05.md`` §8 row 1.

``TextbookOutlineValidator`` gates the Stage-1 enrichment keys that
``_extract_textbook_structure`` folds into ``textbook_structure.json``
when ``TEXTBOOK_SYNTHESIS_PROVIDER`` is set — ``semantic_outline`` and
``draft_terminal_objectives``. Floor checks (plan §8):

1. ``semantic_outline.themes[]`` non-empty — code ``OUTLINE_THEMES_EMPTY``.
2. every theme ``chapter_ids[]`` resolves against ``chapters[].id`` —
   code ``OUTLINE_THEME_CHAPTER_UNRESOLVED``.
3. ``draft_terminal_objectives[]`` non-empty — code
   ``OUTLINE_DRAFT_TO_EMPTY``.
4. each draft TO has a non-empty ``statement`` + a valid ``bloom_level``
   — code ``OUTLINE_DRAFT_TO_MALFORMED``.

**Graceful-degrade contract (the ``ABCD_MISSING`` pattern).** When the
enrichment keys are absent — a default-off run, where
``TEXTBOOK_SYNTHESIS_PROVIDER`` is unset and the deterministic
``SemanticStructureExtractor`` emitted today's byte-identical
``textbook_structure.json`` — the validator returns a no-op
``passed=True``. Default-off runs are unaffected.

Day-1 severity is **warning** (the standing calibration-debt pattern;
promote to critical after a clean-corpus calibration run). The validator
emits warning-severity issues so the gate logs but does not block.

Inputs contract:

* ``inputs["textbook_structure"]`` — the in-memory
  ``textbook_structure.json`` dict. Either this or
  ``inputs["textbook_structure_path"]`` must be supplied.
* ``inputs["textbook_structure_path"]`` — alternative; an on-disk
  ``textbook_structure.json`` the validator loads.
* ``inputs["decision_capture"]`` — optional ``DecisionCapture``; the
  validator emits one ``content_structure_check`` event per run. It
  does NOT mint a validator-specific decision_type
  (``docs/architecture/decision-capture.md``).
* ``inputs["gate_id"]`` — optional gate-ID override.

Cross-references:

* ``schemas/knowledge/textbook_structure_enrichment.schema.json`` — the
  canonical shape this validator's floor checks mirror.
* ``schemas/events/decision_event.schema.json::decision_type.enum`` —
  the ``content_structure_check`` member this validator emits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


#: Canonical 6-value Bloom-level set (mirrors objectives_v1.schema.json
#: and textbook_structure_enrichment.schema.json).
_VALID_BLOOM_LEVELS: frozenset = frozenset(
    {"remember", "understand", "apply", "analyze", "evaluate", "create"}
)

#: Cap emitted issues so a uniformly-broken outline doesn't drown the
#: report. Mirrors the cap used by ``AbcdObjectiveValidator``.
_ISSUE_LIST_CAP: int = 50

#: Validator-capture decision_type. An EXISTING enum member — validators
#: do not mint bespoke types (docs/architecture/decision-capture.md).
_DECISION_TYPE: str = "content_structure_check"


def _emit_decision(
    capture: Any,
    *,
    decision: str,
    rationale: str,
    context: Optional[str] = None,
) -> None:
    """Emit one DecisionCapture event, swallowing capture-side errors.

    Capture is observability, not control flow — a capture failure must
    not break the gate result.
    """
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.warning(
            "TextbookOutlineValidator: decision_capture.log_decision "
            "failed: %s",
            exc,
        )


def _coerce_structure(
    inputs: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve the textbook_structure dict from ``inputs``.

    Priority: ``inputs['textbook_structure']`` >
    ``inputs['textbook_structure_path']``. Returns
    ``(structure, error_issue)``; ``error_issue`` is non-None only on a
    structural failure (path missing / unparseable).
    """
    explicit = inputs.get("textbook_structure")
    if explicit is not None:
        if isinstance(explicit, Mapping):
            return dict(explicit), None
        return None, GateIssue(
            severity="critical",
            code="OUTLINE_STRUCTURE_MALFORMED",
            message=(
                f"inputs['textbook_structure'] is "
                f"{type(explicit).__name__!r}, not a mapping."
            ),
        )

    path_raw = inputs.get("textbook_structure_path")
    if not path_raw:
        return None, None

    p = Path(path_raw)
    if not p.exists():
        return None, GateIssue(
            severity="critical",
            code="OUTLINE_STRUCTURE_PATH_MISSING",
            message=(
                f"textbook_structure_path {str(p)!r} does not exist; "
                f"cannot audit Stage-1 outline enrichment."
            ),
            location=str(p),
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, GateIssue(
            severity="critical",
            code="OUTLINE_STRUCTURE_PATH_UNREADABLE",
            message=(
                f"Failed to parse textbook_structure_path {str(p)!r}: "
                f"{exc.__class__.__name__}: {exc}"
            ),
            location=str(p),
        )
    if not isinstance(data, Mapping):
        return None, GateIssue(
            severity="critical",
            code="OUTLINE_STRUCTURE_MALFORMED",
            message=(
                f"textbook_structure_path {str(p)!r} did not parse to a "
                f"JSON object (got {type(data).__name__!r})."
            ),
            location=str(p),
        )
    return dict(data), None


def _chapter_ids(structure: Mapping[str, Any]) -> set:
    """Pull the set of declared chapter IDs from a textbook_structure.

    Tolerates a missing / non-list ``chapters`` key (returns an empty
    set — the unresolved-chapter check then flags every theme).
    """
    out: set = set()
    chapters = structure.get("chapters")
    if not isinstance(chapters, list):
        return out
    for ch in chapters:
        if not isinstance(ch, Mapping):
            continue
        cid = ch.get("id")
        if isinstance(cid, str) and cid.strip():
            out.add(cid.strip())
    return out


class TextbookOutlineValidator:
    """Stage-1 textbook-outline enrichment gate.

    Wired into ``textbook_to_course::objective_extraction::
    textbook_outline_enrichment`` (gate wiring is Wave B, not this wave).
    Skips-with-pass when the Stage-1 enrichment keys are absent so
    default-off runs are unaffected.
    """

    name = "textbook_outline"
    version = "0.1.0"  # Wave A — three-stage textbook synthesis

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        structure, err = _coerce_structure(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Graceful-degrade: enrichment keys absent → no-op pass. A
        # default-off run (TEXTBOOK_SYNTHESIS_PROVIDER unset) carries
        # neither key, so this validator must not flag it.
        semantic_outline = (structure or {}).get("semantic_outline")
        draft_tos = (structure or {}).get("draft_terminal_objectives")
        if structure is None or (
            semantic_outline is None and draft_tos is None
        ):
            _emit_decision(
                capture,
                decision="skip-with-pass: Stage-1 enrichment keys absent",
                rationale=(
                    "TextbookOutlineValidator found no `semantic_outline` "
                    "and no `draft_terminal_objectives` on the "
                    "textbook_structure — this is a default-off run "
                    "(TEXTBOOK_SYNTHESIS_PROVIDER unset). Returning a "
                    "no-op pass per the ABCD_MISSING graceful-degrade "
                    "contract so default-off runs are unaffected."
                ),
                context="enrichment_absent=true",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        issues: List[GateIssue] = []

        # --- Floor check 1 + 2: semantic_outline.themes ----------------
        themes: List[Any] = []
        if isinstance(semantic_outline, Mapping):
            raw_themes = semantic_outline.get("themes")
            if isinstance(raw_themes, list):
                themes = raw_themes

        if not themes:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="OUTLINE_THEMES_EMPTY",
                    message=(
                        "semantic_outline.themes[] is empty or absent; "
                        "Stage-1 synthesis produced no cross-chapter "
                        "themes."
                    ),
                    location="semantic_outline.themes",
                    suggestion=(
                        "Re-run Stage 1 — the outline LLM call should "
                        "emit at least one theme spanning the textbook."
                    ),
                )
            )
        else:
            valid_chapter_ids = _chapter_ids(structure)
            for idx, theme in enumerate(themes):
                if not isinstance(theme, Mapping):
                    continue
                title = theme.get("title")
                theme_label = (
                    str(title).strip()
                    if isinstance(title, str) and title.strip()
                    else f"theme[{idx}]"
                )
                chapter_ids = theme.get("chapter_ids")
                if not isinstance(chapter_ids, list):
                    chapter_ids = []
                for cid in chapter_ids:
                    if not isinstance(cid, str) or not cid.strip():
                        continue
                    if cid.strip() not in valid_chapter_ids:
                        if len(issues) < _ISSUE_LIST_CAP:
                            issues.append(
                                GateIssue(
                                    severity="warning",
                                    code="OUTLINE_THEME_CHAPTER_UNRESOLVED",
                                    message=(
                                        f"Theme {theme_label!r} references "
                                        f"chapter_id {cid.strip()!r} which "
                                        f"does not resolve against "
                                        f"chapters[].id "
                                        f"({len(valid_chapter_ids)} "
                                        f"chapter IDs declared)."
                                    ),
                                    location=(
                                        f"semantic_outline.themes[{idx}]"
                                        ".chapter_ids"
                                    ),
                                    suggestion=(
                                        "Stage-1 should only reference "
                                        "chapter IDs that exist on "
                                        "textbook_structure.chapters[]."
                                    ),
                                )
                            )

        # --- Floor check 3 + 4: draft_terminal_objectives --------------
        draft_to_list: List[Any] = (
            draft_tos if isinstance(draft_tos, list) else []
        )
        if not draft_to_list:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="OUTLINE_DRAFT_TO_EMPTY",
                    message=(
                        "draft_terminal_objectives[] is empty or absent; "
                        "Stage-1 synthesis produced no draft terminal "
                        "objectives."
                    ),
                    location="draft_terminal_objectives",
                    suggestion=(
                        "Re-run Stage 1 — the outline LLM call should "
                        "emit at least one draft TO-NN objective."
                    ),
                )
            )
        else:
            for idx, to in enumerate(draft_to_list):
                if not isinstance(to, Mapping):
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="OUTLINE_DRAFT_TO_MALFORMED",
                                message=(
                                    f"draft_terminal_objectives[{idx}] is "
                                    f"{type(to).__name__!r}, not a mapping."
                                ),
                                location=f"draft_terminal_objectives[{idx}]",
                            )
                        )
                    continue
                to_id = (
                    str(to.get("id")).strip()
                    if to.get("id")
                    else f"draft_terminal_objectives[{idx}]"
                )
                statement = to.get("statement")
                bloom = to.get("bloom_level")
                problems: List[str] = []
                if not isinstance(statement, str) or not statement.strip():
                    problems.append("empty/absent `statement`")
                if (
                    not isinstance(bloom, str)
                    or bloom.strip().lower() not in _VALID_BLOOM_LEVELS
                ):
                    problems.append(
                        f"invalid `bloom_level` (got {bloom!r}; "
                        f"expected one of "
                        f"{sorted(_VALID_BLOOM_LEVELS)})"
                    )
                if problems and len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="OUTLINE_DRAFT_TO_MALFORMED",
                            message=(
                                f"Draft TO {to_id!r} is malformed: "
                                f"{'; '.join(problems)}."
                            ),
                            location=f"draft_terminal_objectives[{idx}]",
                            suggestion=(
                                "Each draft TO must carry a non-empty "
                                "`statement` and a Bloom-valid "
                                "`bloom_level`."
                            ),
                        )
                    )

        # Day-1 warning severity: warnings never fail-close the gate.
        # Only a structural critical issue (path errors above) fails it.
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        warning_count = sum(1 for i in issues if i.severity == "warning")

        _emit_decision(
            capture,
            decision=(
                f"Audited Stage-1 outline enrichment: "
                f"{len(themes)} theme(s), "
                f"{len(draft_to_list)} draft TO(s); "
                f"{warning_count} warning issue(s)."
            ),
            rationale=(
                f"TextbookOutlineValidator ran the §8-row-1 floor checks "
                f"over the Stage-1 enrichment keys: themes non-empty, "
                f"theme chapter_ids resolved against chapters[].id, "
                f"draft_terminal_objectives non-empty + each TO "
                f"statement/bloom_level valid. Emitted {warning_count} "
                f"warning issue(s) across "
                f"{len(themes)} theme(s) and {len(draft_to_list)} "
                f"draft TO(s); passed={passed}."
            ),
            context=(
                f"themes={len(themes)}; "
                f"draft_tos={len(draft_to_list)}; "
                f"warnings={warning_count}; passed={passed}"
            ),
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=1.0 if passed else 0.0,
            issues=issues,
        )


__all__ = [
    "TextbookOutlineValidator",
]
