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

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        "Each preserved CURIE must appear in pedagogical voice — "
        "inside <code>, as a definitional <span data-cf-term=...>, "
        "or in a sentence introducing the prefix/vocabulary/namespace. "
        "Do NOT stuff CURIEs into attribute values or invented "
        "triple examples."
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


# ---------------------------------------------------------------------------
# Wave 1.7 W1.7.D — per-issue-code directive dispatch
# ---------------------------------------------------------------------------
#
# Wave 1.7 introduces three issue codes from
# ``lib/validators/block_objective_delivery.py::BlockObjectiveDeliveryValidator``:
#
#  - ``BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED`` (NLI entailment miss)
#  - ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` (Bloom-level gap >= 2)
#  - ``BLOCK_OBJECTIVE_VERB_ABSENT`` (no verb-synonym in prose)
#
# These codes carry per-pair context (objective_id, scores, levels,
# verbs) embedded in the GateIssue.message f-string. The Wave 1.7
# directives interpolate that context so the rewrite-tier model sees a
# concrete remediation signal — name the failing objective_id, the
# scoring deltas, the missing verb synonyms, etc.
#
# The :data:`_WAVE17_OBJECTIVE_DELIVERY_CODES` set drives both the
# per-issue dispatch in :func:`_format_failure_block` AND the
# rewrite-tier "only Wave 1.7 codes" detector in
# :meth:`Courseforge.router.router.CourseforgeRouter._gate_results_only_block_objective_failures`
# (canonical source for both surfaces; mirrors Wave 1.5 W1.5.C's
# per_claim escalation-marker fallback at
# ``CourseforgeRouter._gate_results_only_per_claim_failures``).

_CODE_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED: str = (
    "BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED"
)
_CODE_BLOCK_OBJECTIVE_BLOOM_UNDERMET: str = "BLOCK_OBJECTIVE_BLOOM_UNDERMET"
_CODE_BLOCK_OBJECTIVE_VERB_ABSENT: str = "BLOCK_OBJECTIVE_VERB_ABSENT"

_WAVE17_OBJECTIVE_DELIVERY_CODES: frozenset = frozenset(
    {
        _CODE_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED,
        _CODE_BLOCK_OBJECTIVE_BLOOM_UNDERMET,
        _CODE_BLOCK_OBJECTIVE_VERB_ABSENT,
    }
)


#: Truncation budget for ``objective_statement`` text interpolated into
#: the Wave 1.7 STATEMENT_UNDERSUPPORTED directive. Keeps the
#: per-failure suffix bounded when an objective statement is
#: pathologically long. Mirrors Wave 1.5 W1.5.D's per-claim citation
#: truncation pattern (~80 chars).
_MAX_OBJECTIVE_STATEMENT_CHARS: int = 80


# Regexes that pull structured context out of the validator's
# f-string-formatted GateIssue messages. The validator at
# ``lib/validators/block_objective_delivery.py`` uses Python's ``!r``
# repr formatting, which produces single-quoted string forms — the
# regexes accept both single + double quotes for resilience against
# future repr-format changes.
_RE_QUOTED_ID = r"['\"]([^'\"]+)['\"]"
_RE_OBJECTIVE_ID = re.compile(r"objective\s+" + _RE_QUOTED_ID)
_RE_ENTAILMENT_SCORE = re.compile(r"NLI entailment\s*=\s*([0-9.]+)")
_RE_ENTAILMENT_FLOOR = re.compile(
    r"NLI entailment\s*=\s*[0-9.]+\s*\(floor\s*([0-9.]+)\)"
)
_RE_OBSERVED_BLOOM = re.compile(r"bloom_level\s*=\s*" + _RE_QUOTED_ID)
_RE_BLOOM_GAP = re.compile(r"is\s+(\d+)\s+levels below")
_RE_DECLARED_BLOOM = re.compile(
    r"declared objective\s+" + _RE_QUOTED_ID + r"'s bloom_level\s*=\s*"
    + _RE_QUOTED_ID
)
_RE_BLOOM_VERB = re.compile(r"bloom_verb\s*=\s*" + _RE_QUOTED_ID)
_RE_VERB_SYNONYMS_PREVIEW = re.compile(r"\(e\.g\.\s+([^)]+)\)")


def _truncate_statement(
    statement: str, max_chars: int = _MAX_OBJECTIVE_STATEMENT_CHARS
) -> str:
    """Truncate an objective_statement at ``max_chars`` with ellipsis.

    Mirrors :func:`_truncate_message` but uses the smaller Wave 1.7
    statement budget so the directive stays readable at the per-claim
    citation pattern Wave 1.5 W1.5.D introduced.
    """
    if not statement:
        return statement
    if len(statement) <= max_chars:
        return statement
    return statement[: max_chars - 1].rstrip() + "…"


def _extract_first_match(pattern: "re.Pattern[str]", message: str) -> Optional[str]:
    """Best-effort regex extraction; returns first capture group or None."""
    if not message:
        return None
    m = pattern.search(message)
    if m is None:
        return None
    return m.group(1)


def _build_wave17_statement_undersupported_directive(
    *,
    message: str,
    objective_statements: Optional[Dict[str, str]],
) -> Optional[str]:
    """Build the W1.7.D STATEMENT_UNDERSUPPORTED remediation directive.

    Parses the validator's GateIssue.message to extract the
    ``first_failing_objective_id``, ``entailment_score``, and
    ``entailment_floor`` fields the directive interpolates. Looks up
    ``objective_statement`` in the optional ``objective_statements`` map
    keyed by objective_id; truncates at
    :data:`_MAX_OBJECTIVE_STATEMENT_CHARS`. When the statement isn't
    available (caller didn't pass the map / id not in map) the directive
    emits ``"<unavailable>"`` so the per-failure block still renders
    a useful instruction.
    """
    obj_id = _extract_first_match(_RE_OBJECTIVE_ID, message)
    score_str = _extract_first_match(_RE_ENTAILMENT_SCORE, message)
    floor_str = _extract_first_match(_RE_ENTAILMENT_FLOOR, message)
    if obj_id is None or score_str is None or floor_str is None:
        return None
    try:
        entailment_score = float(score_str)
        entailment_floor = float(floor_str)
    except (TypeError, ValueError):
        return None
    statement = None
    if objective_statements:
        raw_stmt = objective_statements.get(obj_id)
        if isinstance(raw_stmt, str) and raw_stmt.strip():
            statement = _truncate_statement(raw_stmt.strip())
    statement_render = statement if statement else "<unavailable>"
    return (
        f"On the prior attempt, the block's prose did not "
        f"semantically support the declared objective "
        f"{obj_id!r}: \"{statement_render}\". "
        f"NLI entailment was {entailment_score:.2f} (floor "
        f"{entailment_floor:.2f}). Re-emit prose that explicitly "
        f"covers the objective's behavioral outcome."
    )


def _build_wave17_bloom_undermet_directive(
    *,
    message: str,
) -> Optional[str]:
    """Build the W1.7.D BLOOM_UNDERMET remediation directive.

    Parses ``observed_bloom``, ``bloom_gap``, ``first_failing_objective_id``,
    and ``declared_bloom`` out of the validator message. The Bloom
    levels are rendered exactly as the validator emitted them
    (lowercased canonical Bloom enum values).
    """
    observed = _extract_first_match(_RE_OBSERVED_BLOOM, message)
    gap_str = _extract_first_match(_RE_BLOOM_GAP, message)
    declared_match = _RE_DECLARED_BLOOM.search(message or "")
    if (
        observed is None
        or gap_str is None
        or declared_match is None
    ):
        return None
    try:
        bloom_gap = int(gap_str)
    except (TypeError, ValueError):
        return None
    obj_id = declared_match.group(1)
    declared_bloom = declared_match.group(2)
    return (
        f"On the prior attempt, the block's bloom_level "
        f"({observed!r}) was {bloom_gap} levels below the "
        f"declared objective {obj_id!r}'s "
        f"bloom_level ({declared_bloom!r}). Re-emit at or above "
        f"the declared level — scaffold up to the declared "
        f"cognitive demand."
    )


def _build_wave17_verb_absent_directive(
    *,
    message: str,
) -> Optional[str]:
    """Build the W1.7.D VERB_ABSENT remediation directive.

    Parses ``bloom_verb`` and the synonym preview list (the validator's
    ``e.g. <comma-list>`` clause) out of the message. The synonyms_csv
    is rendered as the validator's truncated 8-synonym preview — the
    full set isn't recoverable from the message, but the preview is
    sufficient for the model to pivot to a synonym voice.
    """
    bloom_verb = _extract_first_match(_RE_BLOOM_VERB, message)
    synonyms_csv = _extract_first_match(_RE_VERB_SYNONYMS_PREVIEW, message)
    if bloom_verb is None or synonyms_csv is None:
        return None
    return (
        f"On the prior attempt, no synonym of the objective's "
        f"bloom_verb {bloom_verb!r} appeared in the block prose. "
        f"Re-emit prose using {bloom_verb!r} or one of its "
        f"Bloom-level synonyms ({synonyms_csv}) at least once."
    )


def _build_wave17_directive(
    *,
    code: str,
    message: str,
    objective_statements: Optional[Dict[str, str]],
) -> Optional[str]:
    """Dispatch on a Wave 1.7 issue code → matching directive builder.

    Returns ``None`` when the issue code isn't a Wave 1.7 code OR when
    the message-parse fails (caller falls back to the gate_id directive
    table so the suffix is never silent).
    """
    if code == _CODE_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED:
        return _build_wave17_statement_undersupported_directive(
            message=message,
            objective_statements=objective_statements,
        )
    if code == _CODE_BLOCK_OBJECTIVE_BLOOM_UNDERMET:
        return _build_wave17_bloom_undermet_directive(message=message)
    if code == _CODE_BLOCK_OBJECTIVE_VERB_ABSENT:
        return _build_wave17_verb_absent_directive(message=message)
    return None


def _format_failure_block(
    result: GateResult,
    *,
    objective_statements: Optional[Dict[str, str]] = None,
) -> str:
    """Render one failed ``GateResult`` as a remediation block.

    Per pre-resolved decision #4: each failure renders as a
    `\\n- [<gate_id>] <issue.message>\\n  Correct by: <directive>` line
    per issue in the result. The directive is looked up in this order:

    1. **Per-issue Wave 1.7 dispatch** (Wave 1.7 W1.7.D) — when
       ``issue.code`` is one of the Wave 1.7 block-objective delivery
       codes (:data:`_WAVE17_OBJECTIVE_DELIVERY_CODES`), build a
       per-axis directive interpolating the parsed message context
       (objective_id, scores, Bloom levels, verb synonyms) plus the
       objective_statement looked up in ``objective_statements`` (when
       available). Keeps the per-failure block actionable instead of
       falling back to a generic "stamp data-cf-objective-id" rewrite
       directive that doesn't address the pedagogical-delivery miss.
    2. **Per-gate-id directive table** lookup
       (:data:`_REMEDIATION_DIRECTIVES_BY_GATE_ID`).
    3. **Generic "Re-emit correctly per the {validator_name} contract"**
       fallback so the suffix is never silent on a failure.

    When the GateResult carries no issues (defensive — a failed result
    SHOULD always carry at least one issue) the block still renders one
    line referencing the gate_id alone so the model still gets the
    gate_id-table directive.
    """
    gate_directive = _REMEDIATION_DIRECTIVES_BY_GATE_ID.get(
        result.gate_id,
        f"Re-emit correctly per the {result.validator_name} contract.",
    )
    if not result.issues:
        return (
            f"\n- [{result.gate_id}] gate failed (no issue detail)."
            f"\n  Correct by: {gate_directive}"
        )
    lines: List[str] = []
    for issue in result.issues:
        # ``issue`` may be a GateIssue dataclass or a plain dict (the
        # GateResult.to_dict roundtrip path). Accept both.
        if isinstance(issue, GateIssue):
            message = issue.message or ""
            code = issue.code or ""
        elif isinstance(issue, dict):
            message = str(issue.get("message", "") or "")
            code = str(issue.get("code", "") or "")
        else:
            message = str(issue)
            code = ""
        truncated = _truncate_message(message)
        directive = gate_directive
        if code in _WAVE17_OBJECTIVE_DELIVERY_CODES:
            wave17_directive = _build_wave17_directive(
                code=code,
                message=message,
                objective_statements=objective_statements,
            )
            if wave17_directive is not None:
                directive = wave17_directive
        lines.append(
            f"\n- [{result.gate_id}] {truncated}"
            f"\n  Correct by: {directive}"
        )
    return "".join(lines)


def _append_remediation_for_gates(
    prompt: str,
    failures: Sequence[GateResult],
    *,
    objective_statements: Optional[Dict[str, str]] = None,
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

    ``objective_statements`` is the optional Wave 1.7 W1.7.D
    enrichment kwarg — a ``Dict[objective_id, statement]`` map the
    per-issue dispatch path uses to interpolate the failing objective's
    behavioral-outcome statement into the
    ``BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED`` directive. The
    rewrite-tier router builds this map from the same ``objectives``
    arg it threads through ``_run_validator_chain``; default ``None``
    keeps every pre-Wave-1.7 caller byte-stable.

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
    blocks = [
        _format_failure_block(
            r, objective_statements=objective_statements,
        )
        for r in actionable
    ]
    suffix = "\n\nYour previous attempt failed validation:" + "".join(blocks)
    return prompt + suffix


def _missing_preserve_tokens(
    content: Any,
    tokens: Sequence[str],
    in_keys: Tuple[str, ...] = ("body",),
) -> List[str]:
    """Return the subset of ``tokens`` that don't appear in ``content``.

    Direct port of
    :func:`Trainforge.generators._local_provider.LocalSynthesisProvider._missing_preserve_tokens`
    (`Trainforge/generators/_local_provider.py:548-564`), generalised
    to accept ``content: Any`` per the Subtask 3 contract:

    - When ``content`` is a string, searches the full string for each
      token via substring match (the rewrite-tier HTML emit shape:
      a CURIE preserved in any tag, attribute value, or inline prose
      counts as preserved).
    - When ``content`` is a dict, concatenates the values at each key
      named in ``in_keys`` (default ``("body",)``) into a haystack and
      substring-matches each token (the Trainforge instruction-pair /
      preference-pair shape; falls back to ``""`` when a key is absent).
    - When ``content`` is anything else, ``str(content)`` is used as
      the haystack so callers passing list-of-strings or other
      non-canonical shapes still get a deterministic answer.

    Empty ``tokens`` or empty ``in_keys`` (for dict content) returns
    the empty list — there is nothing to enforce.

    Substring match — exact-character preservation is the contract.
    The Trainforge precedent and the rewrite-tier CURIE-preservation
    gate both rely on this semantics so a CURIE survives across any
    HTML structural shape.
    """
    if not tokens:
        return []
    if isinstance(content, str):
        haystack = content
    elif isinstance(content, dict):
        if not in_keys:
            return []
        haystack = " ".join(
            str(content.get(k, "") or "") for k in in_keys
        )
    else:
        haystack = str(content) if content is not None else ""
    missing: List[str] = []
    for tok in tokens:
        if tok and tok not in haystack:
            missing.append(tok)
    return missing


def _append_preserve_remediation(
    prompt: str,
    missing_tokens: Sequence[str],
    in_keys: Tuple[str, ...] = ("body",),
) -> str:
    """Append a remediation directive naming the missing tokens.

    Direct port of
    :func:`Trainforge.generators._local_provider.LocalSynthesisProvider._append_preserve_remediation`
    (`Trainforge/generators/_local_provider.py:566-583`), generalised
    to operate on a string prompt (the Courseforge rewrite-tier shape)
    rather than a list-of-dict messages payload (the Trainforge shape).
    The wording deliberately preserves the canonical
    ``"did not include the required"`` phrase so the rewrite-provider
    test suite's substring-match keeps passing across the move.

    ``in_keys`` is purely informational here — the field name is
    interpolated into the directive so the model knows where the
    tokens were supposed to land. The default ``("body",)`` matches
    the rewrite tier's HTML-body emit shape; the Trainforge dict-
    content path passes ``("prompt", "completion")`` etc.

    Returns ``prompt`` unchanged when ``missing_tokens`` is empty —
    no remediation is needed.
    """
    if not missing_tokens:
        return prompt
    token_list = ", ".join(repr(t) for t in missing_tokens)
    field_list = " and ".join(in_keys) if in_keys else "the response"
    remediation = (
        f"\n\nThe prior response did not include the required "
        f"tokens {token_list} verbatim in {field_list}. Rewrite the "
        f"response so each of those tokens appears VERBATIM (exact "
        f"characters, with the colon and case intact). Output the "
        f"same response shape — no preamble, no markdown."
    )
    return prompt + remediation


__all__ = [
    "_CODE_BLOCK_OBJECTIVE_BLOOM_UNDERMET",
    "_CODE_BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED",
    "_CODE_BLOCK_OBJECTIVE_VERB_ABSENT",
    "_MAX_OBJECTIVE_STATEMENT_CHARS",
    "_REMEDIATION_DIRECTIVES_BY_GATE_ID",
    "_WAVE17_OBJECTIVE_DELIVERY_CODES",
    "_append_preserve_remediation",
    "_append_remediation_for_gates",
    "_build_wave17_directive",
    "_missing_preserve_tokens",
]
