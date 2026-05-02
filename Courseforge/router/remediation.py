#!/usr/bin/env python3
"""Generalized remediation prompt-suffix builder for the two-pass router.

Phase 3.5 Subtasks 1-4 (per `plans/phase3_5_post_rewrite_validation.md`
§A). Sibling to :mod:`Courseforge.router.router` (the self-consistency
loop that consumes these helpers between candidate iterations) and to
:mod:`Courseforge.router.inter_tier_gates` (the four ``Block*Validator``
classes whose ``GateResult`` outputs feed the
:func:`_append_remediation_for_gates` builder).

Roadmap context (root ``CLAUDE.md`` § Phase 3 outline-rewrite two-pass
router): both the outline tier and the rewrite tier go through a
post-emit validator chain. On a failing chain, the offending block is
re-rolled — but pre-Phase-3.5 the re-roll re-issued the SAME prompt,
which is wasted budget when the failure mode is deterministic
(missing CURIEs, wrong ``content_type`` enum value, missing source
refs, missing objective refs). Phase 3.5 wires the failure context
into the next prompt as a remediation suffix so the model sees what
went wrong and the directive to fix it.

Public surface:

- :func:`_append_remediation_for_gates(prompt, failures)` — general
  per-failure suffix builder. Iterates the failed ``GateResult`` list,
  looks up a per-gate-id directive, and appends one block per failure.
- :func:`_append_preserve_remediation(prompt, missing_tokens, in_keys)`
  — preserve-token specialization (CURIE / fact / ref preservation).
  The rewrite tier consumes this via the CURIE-preservation gate.
- :func:`_missing_preserve_tokens(content, tokens, in_keys)` — token-
  presence detector. Accepts ``content: Any`` (str or dict) per the
  Subtask 3 generalization: when str, searches the string body; when
  dict, searches the keys named in ``in_keys`` (default ``("body",)``).

Direct port of the
:func:`Trainforge.generators._local_provider.LocalSynthesisProvider._missing_preserve_tokens`
+ ``_append_preserve_remediation`` precedent
(`Trainforge/generators/_local_provider.py:548-583`), generalised to
accept the rewrite tier's HTML-string outputs as well as the legacy
dict shape.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult


# ---------------------------------------------------------------------------
# Per-failure-mode remediation directives table (Subtask 1, pre-resolved
# decision #4).
# ---------------------------------------------------------------------------
#
# Keys are the canonical ``gate_id`` values for the eight inter-tier
# gates (4 outline + 4 rewrite, mirrored across both tiers). Values are
# short imperative directives (~80 chars) the remediation builder
# emits as the "Correct by:" line per failure block. Cross-tier
# overlap is intentional — when both tiers share a failure mode the
# directive copy stays the same.
#
# Source of truth for the gate_id naming: the Phase 3 inter-tier-
# gates module (``Courseforge/router/inter_tier_gates.py``) and the
# new ``post_rewrite_validation`` workflow phase
# (``config/workflows.yaml::post_rewrite_validation``, Phase 3.5
# Subtask 10).

_REMEDIATION_DIRECTIVES_BY_GATE_ID: Dict[str, str] = {
    # Outline-tier gates (consume Block.content as a dict).
    "outline_curie_anchoring": (
        "Preserve every CURIE verbatim. Re-emit the JSON object "
        "including all source-declared CURIEs in 'curies'."
    ),
    "outline_content_type": (
        "Set 'content_type' to one of the canonical 8 enum values "
        "(see schemas/taxonomies/content_type.schema.json)."
    ),
    "outline_page_objectives": (
        "Populate 'objective_refs' with at least one canonical "
        "TO-NN / CO-NN learning objective ID."
    ),
    "outline_source_refs": (
        "Populate 'source_refs' with at least one DART sourceId "
        "from the supplied source-chunk grounding list."
    ),
    # Rewrite-tier gates (consume Block.content as an HTML string).
    "rewrite_curie_anchoring": (
        "Preserve every CURIE verbatim in the rendered HTML body "
        "(exact characters, colon and case intact)."
    ),
    "rewrite_content_type": (
        "Stamp data-cf-content-type on the section/heading wrapper "
        "with one of the canonical 8 enum values."
    ),
    "rewrite_page_objectives": (
        "Stamp data-cf-objective-id on the rendered HTML for each "
        "canonical TO-NN / CO-NN objective ref."
    ),
    "rewrite_source_refs": (
        "Stamp data-cf-source-ids on the section/heading wrapper "
        "with at least one DART sourceId."
    ),
}


# ---------------------------------------------------------------------------
# Stub functions (filled in by Subtasks 2, 3).
# ---------------------------------------------------------------------------


# Per-failure issue-message truncation budget (Subtask 2). Bounds the
# remediation suffix size when a validator emits very long error
# messages (e.g. structural violations enumerating dozens of missing
# refs). 200 chars keeps the suffix readable for the model without
# exhausting its prompt budget on a multi-failure block.
_MAX_ISSUE_MESSAGE_CHARS = 200


def _truncate_message(message: str, max_chars: int = _MAX_ISSUE_MESSAGE_CHARS) -> str:
    """Truncate ``message`` at ``max_chars`` chars with an ellipsis suffix.

    Used by :func:`_append_remediation_for_gates` to bound issue-message
    size in the remediation suffix. Empty / falsy input passes through
    unchanged.
    """
    if not message:
        return message
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 1].rstrip() + "…"


def _format_failure_block(result: GateResult) -> str:
    """Render one failed ``GateResult`` as a remediation block.

    Per pre-resolved decision #4: each failure renders as a
    `\\n- [<gate_id>] <issue.message>\\n  Correct by: <directive>` line
    per issue in the result. The directive is looked up in
    :data:`_REMEDIATION_DIRECTIVES_BY_GATE_ID`; unknown gate IDs fall
    back to a generic "Re-emit correctly per the {validator_name}
    contract" string so the suffix is never silent on a failure.

    When the GateResult carries no issues (defensive — a failed result
    SHOULD always carry at least one issue) the block still renders one
    line referencing the gate_id alone so the model still gets the
    remediation directive.
    """
    directive = _REMEDIATION_DIRECTIVES_BY_GATE_ID.get(
        result.gate_id,
        f"Re-emit correctly per the {result.validator_name} contract.",
    )
    if not result.issues:
        return (
            f"\n- [{result.gate_id}] gate failed (no issue detail)."
            f"\n  Correct by: {directive}"
        )
    lines: List[str] = []
    for issue in result.issues:
        # ``issue`` may be a GateIssue dataclass or a plain dict (the
        # GateResult.to_dict roundtrip path). Accept both.
        if isinstance(issue, GateIssue):
            message = issue.message or ""
        elif isinstance(issue, dict):
            message = str(issue.get("message", "") or "")
        else:
            message = str(issue)
        truncated = _truncate_message(message)
        lines.append(
            f"\n- [{result.gate_id}] {truncated}"
            f"\n  Correct by: {directive}"
        )
    return "".join(lines)


def _append_remediation_for_gates(
    prompt: str, failures: Sequence[GateResult]
) -> str:
    """Append a per-failure remediation block to ``prompt``.

    Iterates ``failures``; for each ``GateResult`` whose action is not
    ``"pass"`` (or whose ``passed`` flag is False when ``action`` is
    ``None``), emits a remediation block per :func:`_format_failure_block`.
    Pass-action results are no-ops — they don't carry a failure to
    correct.

    The appended suffix opens with the canonical
    ``"Your previous attempt failed validation:"`` header so the model
    has an unambiguous signal that the prompt is mid-iteration. When
    ``failures`` contains no actionable results the prompt passes
    through unchanged.

    Returns ``prompt`` (unchanged) when no actionable failures exist;
    otherwise returns ``prompt + "\\n\\n<header><blocks>"``.
    """
    actionable: List[GateResult] = []
    for result in failures:
        # Pre-resolved decision #4: pass-action results don't trigger
        # remediation. The router still passes them in (as the chain
        # output), so filter here rather than at every call site.
        action = result.action
        if action == "pass":
            continue
        # Legacy validators leave ``action=None``; treat them as
        # actionable when ``passed`` is False (the
        # ``GateResult.derive_default_action`` fallback semantics).
        if action is None and result.passed:
            continue
        actionable.append(result)
    if not actionable:
        return prompt
    blocks = [_format_failure_block(r) for r in actionable]
    suffix = "\n\nYour previous attempt failed validation:" + "".join(blocks)
    return prompt + suffix


def _append_preserve_remediation(
    prompt: str,
    missing_tokens: Sequence[str],
    in_keys: Tuple[str, ...] = ("body",),
) -> str:
    """Append a remediation directive naming the missing tokens.

    Filled in by Subtask 3 (port from
    ``Courseforge/generators/_rewrite_provider.py``).
    """
    raise NotImplementedError("Subtask 3 deliverable")


def _missing_preserve_tokens(
    content: Any,
    tokens: Sequence[str],
    in_keys: Tuple[str, ...] = ("body",),
) -> List[str]:
    """Return the subset of ``tokens`` that don't appear in ``content``.

    Filled in by Subtask 3 (port from
    ``Courseforge/generators/_rewrite_provider.py``). Accepts
    ``content: Any`` per the Subtask 3 generalization: str searches
    the full body; dict searches the keys named in ``in_keys``.
    """
    raise NotImplementedError("Subtask 3 deliverable")


__all__ = [
    "_REMEDIATION_DIRECTIVES_BY_GATE_ID",
    "_append_preserve_remediation",
    "_append_remediation_for_gates",
    "_missing_preserve_tokens",
]
