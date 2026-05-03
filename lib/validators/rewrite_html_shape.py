"""Post-rewrite HTML-shape sentinel.

Critical-severity gate at ``post_rewrite_validation`` that closes the
``{"div": {...}}`` regression class — a rewrite-tier emit that is
JSON-wrapped (or markdown-fenced, or otherwise not a bare HTML body
fragment) sailing into packaging because every other validator's
HTML-strip + regex-match path accidentally accepts the inner-quoted
attribute strings as legitimate-looking signal.

Contract per ``plans/qwen7b-courseforge-fixes-2026-05-followup.md``
§3.2:

- **Not JSON-wrapped.** First non-whitespace char must be ``<``. A
  leading ``{`` / ``[`` / triple-backtick / ``<!DOCTYPE`` triggers
  ``code="REWRITE_NOT_HTML_BODY_FRAGMENT"`` with ``action="regenerate"``.
- **Not JSON-stringified.** The full content string must NOT
  ``json.loads`` to a dict / list — sentinel for the
  ``{"div": {...}}`` regression that the regex-based gates miss.
- **Parses cleanly via stdlib ``html.parser``.** Tracks open / close
  tag balance + records whether any recognised body tag opened. A
  parse failure or unbalanced tag stack triggers
  ``code="REWRITE_HTML_PARSE_FAIL"``.
- **Required ``data-cf-*`` attributes per block_type.** Mirrors the
  emit contract in ``Courseforge/scripts/blocks.py::Block.to_html_attrs``;
  a missing required attribute triggers
  ``code="REWRITE_MISSING_REQUIRED_ATTR"``.

Decision-capture: emits one ``rewrite_html_shape_check`` decision per
block evaluated. Rationale interpolates dynamic signals (block_id,
block_type, content length, parser tags seen, the failing attribute
when applicable). Strict-mode-on-unknown-decision-types wired to the
shape gate — the new ``decision_type`` value is added to
``schemas/events/decision_event.schema.json::decision_type.enum`` so
``DECISION_VALIDATION_STRICT=true`` runs don't fail closed on the
first emit.

References:
    - ``Courseforge/scripts/blocks.py::Block.to_html_attrs`` —
      canonical emit shape per block_type (the required-attrs map
      below mirrors that surface).
    - ``Courseforge/router/inter_tier_gates.py`` — sibling Block-input
      validators that this gate complements at the post-rewrite seam.
    - ``lib/validators/source_refs.py`` — Wave 9 emit-side
      counterpart of the manifest-resolution gate this complements.
"""

from __future__ import annotations

import json
import logging
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Block import bridge (mirror of inter_tier_gates / Phase 4 validators).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


# Cap per-block issue list so a uniformly-broken rewrite batch doesn't
# drown the gate report.
_ISSUE_LIST_CAP: int = 50

# Recognised body-level HTML tags. The parser tracks whether any of
# these opened — a rewrite-tier emit that strips down to plain text
# without a single body tag is functionally equivalent to JSON-wrap
# from the renderer's perspective.
_BODY_TAGS: frozenset = frozenset(
    {
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ol", "ul", "li", "section", "article", "div", "span",
        "strong", "em", "code", "pre", "blockquote", "figure",
        "figcaption", "table", "thead", "tbody", "tr", "td", "th",
        "a", "br", "img", "details", "summary", "dl", "dt", "dd",
        "small", "sub", "sup", "i", "b",
    }
)

# HTML void elements that don't require a closing tag — exclude them
# from the open-stack when the parser sees a self-closing or
# unclosed-by-spec start tag.
_VOID_TAGS: frozenset = frozenset(
    {
        "br", "img", "hr", "input", "meta", "link", "area",
        "base", "col", "embed", "param", "source", "track", "wbr",
    }
)

# Per-block-type required ``data-cf-*`` attribute map. Mirrors the
# emit contract at ``Courseforge/scripts/blocks.py::Block.to_html_attrs``
# (see ``_objective_attrs``, ``_self_check_question_attrs``,
# ``_activity_attrs``, ``_content_section_attrs``, etc.). Block types
# whose canonical emit shape does NOT carry data-cf-* attributes (e.g.
# ``misconception`` emits via JSON-LD only; ``assessment_item`` lives
# in QTI XML — but the rewrite tier does emit data-cf-* on these for
# consumability) are still required to carry the universal
# ``data-cf-block-id`` so the post-rewrite report can cross-reference
# the JSON-LD blocks[] projection.
_REQUIRED_ATTRS: Dict[str, Tuple[str, ...]] = {
    # Objective list items carry the canonical TO-NN / CO-NN reference
    # plus Bloom metadata.
    "objective": ("data-cf-block-id", "data-cf-objective-id", "data-cf-bloom-level"),
    # Concept and example sections share the heading content-section
    # attribute shape (data-cf-content-type + data-cf-key-terms).
    "concept": ("data-cf-block-id", "data-cf-content-type", "data-cf-key-terms"),
    "example": ("data-cf-block-id", "data-cf-content-type"),
    "explanation": ("data-cf-block-id", "data-cf-content-type"),
    "summary_takeaway": ("data-cf-block-id", "data-cf-content-type"),
    # Assessment items: rewrite-tier emit carries objective_ref +
    # bloom_level on the block wrapper for consumability before QTI
    # serialisation.
    "assessment_item": (
        "data-cf-block-id", "data-cf-objective-ref", "data-cf-bloom-level",
    ),
    # Misconception emits via JSON-LD (no data-cf-* attribute on the
    # wrapper itself per blocks.py:393-397) — only the universal
    # block_id is required.
    "misconception": ("data-cf-block-id",),
    # Self-check / activity / flip-card components carry component +
    # purpose + bloom (per the emit helpers in blocks.py).
    "self_check_question": (
        "data-cf-block-id", "data-cf-component", "data-cf-purpose",
        "data-cf-bloom-level",
    ),
    "activity": (
        "data-cf-block-id", "data-cf-component", "data-cf-purpose",
        "data-cf-bloom-level",
    ),
    "flip_card_grid": ("data-cf-block-id", "data-cf-component", "data-cf-purpose"),
    # Wrapper-only blocks (prereq_set, callout, recap, prompts) — only
    # the universal block_id is mandatory; the renderer adds
    # source-id attrs when grounding is present, but those are
    # optional per blocks.py:391.
    "callout": ("data-cf-block-id",),
    "prereq_set": ("data-cf-block-id",),
    "reflection_prompt": ("data-cf-block-id",),
    "discussion_prompt": ("data-cf-block-id",),
    "recap": ("data-cf-block-id",),
    "chrome": ("data-cf-block-id",),
}

# Block types where the body-tag check is relaxed because the canonical
# emit is short-form (one or two tagged spans / list items rather than
# a paragraph or heading). ``summary_takeaway`` for example may emit a
# ``<li>`` or a single short ``<p>`` — both legitimate.
_SHORT_FORM_BLOCK_TYPES: frozenset = frozenset(
    {"summary_takeaway", "recap", "reflection_prompt", "discussion_prompt"}
)


class _ShapeParser(HTMLParser):
    """Stdlib HTML parser that tracks tag balance + body-tag presence.

    Records every start tag (excluding void elements) on an open-stack
    and pops on every end tag. ``unbalanced`` becomes True if a pop
    happens against an empty stack OR if the final stack is non-empty.
    ``saw_body_tag`` becomes True the first time a recognised
    ``_BODY_TAGS`` element opens. ``found_attrs`` is a set of every
    ``data-cf-*`` attribute name the parser saw (case-folded), used by
    the required-attribute check.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._open_stack: List[str] = []
        self.unbalanced: bool = False
        self.saw_body_tag: bool = False
        self.found_attrs: Set[str] = set()
        self.parse_error: Optional[str] = None
        self.tags_seen: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        self.tags_seen.append(tag)
        if tag in _BODY_TAGS:
            self.saw_body_tag = True
        # Track data-cf-* attributes. The parser lowercases attr names
        # by default, so we don't need a case-fold pass here.
        for attr_name, _attr_value in attrs:
            if attr_name and attr_name.startswith("data-cf-"):
                self.found_attrs.add(attr_name)
        if tag not in _VOID_TAGS:
            self._open_stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        # Self-closing form ``<tag />``. Treat as void — record body /
        # attrs but don't push on the stack.
        tag = tag.lower()
        self.tags_seen.append(tag)
        if tag in _BODY_TAGS:
            self.saw_body_tag = True
        for attr_name, _attr_value in attrs:
            if attr_name and attr_name.startswith("data-cf-"):
                self.found_attrs.add(attr_name)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            # Spec-violating end tag for a void element — ignore but
            # don't fail (browsers tolerate this).
            return
        if not self._open_stack:
            self.unbalanced = True
            return
        # Pop until we find the matching open tag, marking unbalanced
        # if we have to skip over mismatched intermediates.
        if self._open_stack[-1] == tag:
            self._open_stack.pop()
        elif tag in self._open_stack:
            # Tag is somewhere in the stack — pop everything down to it
            # and mark as unbalanced (HTML is malformed even if
            # browsers recover).
            while self._open_stack and self._open_stack[-1] != tag:
                self._open_stack.pop()
                self.unbalanced = True
            if self._open_stack:
                self._open_stack.pop()
        else:
            # Closing tag that was never opened.
            self.unbalanced = True

    def error(self, message: str) -> None:  # pragma: no cover — stdlib never calls in py3
        self.parse_error = message

    def finalize(self) -> None:
        """Mark unbalanced if the open-stack is non-empty after feed."""
        if self._open_stack:
            self.unbalanced = True


def _coerce_blocks(inputs: Dict[str, Any]) -> Tuple[List[Block], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs["blocks"]`` (sibling helper)."""
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _is_json_wrapped(content: str) -> bool:
    """True when ``content`` round-trips through json.loads to a dict / list.

    This is the explicit sentinel for the ``{"div": {...}}`` regression
    where the rewrite-tier model serialises HTML as a JSON object instead
    of emitting it bare. The leading-char check at the call site catches
    the obvious cases; this helper is the second-pass parse for cases
    where the emit happens to start with whitespace before the brace.
    """
    stripped = content.strip()
    if not stripped:
        return False
    if stripped[0] not in "{[":
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, (dict, list))


def _emit_decision(
    capture: Any,
    block: Block,
    *,
    passed: bool,
    code: Optional[str],
    content_length: int,
    tags_seen: List[str],
    failing_attr: Optional[str] = None,
) -> None:
    """Emit one ``rewrite_html_shape_check`` decision per block.

    ``capture`` is the optional ``DecisionCapture`` instance the caller
    threads through. ``None`` means decision capture is not wired
    (e.g. unit tests that don't seed a capture); we silently skip.
    Rationale interpolates dynamic signals so the audit trail is
    replayable post-hoc.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rationale_parts: List[str] = [
        f"block_id={block.block_id}",
        f"block_type={block.block_type}",
        f"content_length={content_length}",
        f"body_tags_seen={len(tags_seen)}",
    ]
    if not passed:
        rationale_parts.append(f"failure_code={code}")
    if failing_attr:
        rationale_parts.append(f"missing_attr={failing_attr}")
    if tags_seen:
        # Cap to first 8 tags so the rationale stays readable.
        rationale_parts.append(f"first_tags={','.join(tags_seen[:8])}")
    rationale = (
        f"Post-rewrite HTML-shape check on Block {block.block_id!r}: "
        f"{', '.join(rationale_parts)}."
    )
    try:
        capture.log_decision(
            decision_type="rewrite_html_shape_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — never let capture wiring kill the gate
        logger.debug(
            "DecisionCapture.log_decision raised on rewrite_html_shape_check: %s",
            exc,
        )


class RewriteHtmlShapeValidator:
    """Post-rewrite HTML-shape critical sentinel.

    Iterates every Block in ``inputs["blocks"]`` whose ``content`` is a
    string (rewrite-tier shape; outline-tier dict-content blocks skip
    silently) and runs the four-part shape contract above. Any block
    that fails any check emits a critical GateIssue with
    ``action="regenerate"`` so the rewrite-tier router consumes it as
    a regen signal.

    Optional decision-capture: the validator looks for a
    ``decision_capture`` instance in ``inputs`` (DecisionCapture from
    ``lib.decision_capture``); when present, one
    ``rewrite_html_shape_check`` decision fires per block evaluated.
    Tests can opt out by omitting the key.
    """

    name = "rewrite_html_shape"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0

        for block in blocks:
            content = block.content
            # Outline-tier blocks (dict content) skip silently — the
            # post-rewrite seam only audits string content.
            if not isinstance(content, str):
                continue
            audited += 1

            content_length = len(content)
            stripped = content.lstrip()

            # Empty content — fail with parse error.
            if not stripped:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_HTML_PARSE_FAIL",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} "
                            f"emitted empty content; expected an HTML body "
                            f"fragment."
                        ),
                        location=block.block_id,
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_HTML_PARSE_FAIL",
                    content_length=content_length, tags_seen=[],
                )
                continue

            # 1. Not JSON-wrapped (leading-char check + json.loads round-trip).
            first_char = stripped[0]
            json_wrapped = first_char in "{["
            if first_char == "<" and stripped.lower().startswith("<!doctype"):
                # DOCTYPE preamble is a full HTML document, not a body
                # fragment — fail the shape check.
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} emitted a "
                            f"full HTML document (<!DOCTYPE ...>); expected a "
                            f"bare body fragment."
                        ),
                        location=block.block_id,
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                    content_length=content_length, tags_seen=[],
                )
                continue
            if stripped.startswith("```"):
                # Markdown-fenced HTML.
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} emitted "
                            f"markdown-fenced content (leading triple-backtick); "
                            f"expected a bare HTML body fragment."
                        ),
                        location=block.block_id,
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                    content_length=content_length, tags_seen=[],
                )
                continue
            if json_wrapped:
                # Leading brace / bracket — definitely not bare HTML.
                # Distinguish JSON-stringified payload (round-trips to
                # dict / list) from leading-brace garbage.
                if _is_json_wrapped(stripped):
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="critical",
                            code="REWRITE_JSON_WRAPPED_HTML",
                            message=(
                                f"Rewrite-tier Block {block.block_id!r} emitted "
                                f"JSON-stringified content (round-trips to a "
                                f"dict / list); expected a bare HTML body "
                                f"fragment."
                            ),
                            location=block.block_id,
                            suggestion=(
                                "The model serialised the HTML as a JSON object "
                                "(e.g. {\"div\": {...}}). Re-prompt the rewrite "
                                "tier to emit raw HTML without JSON wrapping."
                            ),
                        ))
                    _emit_decision(
                        capture, block,
                        passed=False, code="REWRITE_JSON_WRAPPED_HTML",
                        content_length=content_length, tags_seen=[],
                    )
                else:
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="critical",
                            code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                            message=(
                                f"Rewrite-tier Block {block.block_id!r} emitted "
                                f"non-HTML content (leading {first_char!r}); "
                                f"expected a bare HTML body fragment."
                            ),
                            location=block.block_id,
                        ))
                    _emit_decision(
                        capture, block,
                        passed=False, code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                        content_length=content_length, tags_seen=[],
                    )
                continue
            if first_char != "<":
                # Plain text without HTML — fail.
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} emitted "
                            f"plain text (leading {first_char!r}); expected an "
                            f"HTML body fragment starting with '<'."
                        ),
                        location=block.block_id,
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_NOT_HTML_BODY_FRAGMENT",
                    content_length=content_length, tags_seen=[],
                )
                continue

            # 2. Parse via stdlib HTMLParser; track open/close balance
            # + body-tag presence + data-cf-* attribute set.
            parser = _ShapeParser()
            try:
                parser.feed(stripped)
                parser.close()
                parser.finalize()
            except Exception as exc:  # noqa: BLE001 — stdlib parser is permissive but we wrap defensively
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_HTML_PARSE_FAIL",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} failed to "
                            f"parse via stdlib HTMLParser: {exc}"
                        ),
                        location=block.block_id,
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_HTML_PARSE_FAIL",
                    content_length=content_length, tags_seen=parser.tags_seen,
                )
                continue

            short_form_ok = block.block_type in _SHORT_FORM_BLOCK_TYPES
            if not parser.saw_body_tag and not short_form_ok:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_HTML_PARSE_FAIL",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} emit "
                            f"contained no recognised body tag (p / h2 / ul / "
                            f"section / div / etc.); expected an HTML body "
                            f"fragment."
                        ),
                        location=block.block_id,
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_HTML_PARSE_FAIL",
                    content_length=content_length, tags_seen=parser.tags_seen,
                )
                continue

            if parser.unbalanced:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_HTML_PARSE_FAIL",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} emit had "
                            f"unbalanced HTML tags; the open-stack was not "
                            f"empty after parse."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Re-prompt the rewrite tier to close every opened "
                            "tag. Common cause: nested <p> elements without "
                            "a closing </p>."
                        ),
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_HTML_PARSE_FAIL",
                    content_length=content_length, tags_seen=parser.tags_seen,
                )
                continue

            # 3. Required data-cf-* attributes per block_type. Missing
            # ANY required attr fails the gate.
            required = _REQUIRED_ATTRS.get(block.block_type, ())
            missing: List[str] = [
                attr for attr in required if attr not in parser.found_attrs
            ]
            if missing:
                # Emit one GateIssue per missing attribute (capped).
                first_missing = missing[0]
                for attr in missing:
                    if len(issues) >= _ISSUE_LIST_CAP:
                        break
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_MISSING_REQUIRED_ATTR",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} (block_type="
                            f"{block.block_type!r}) is missing the required "
                            f"attribute {attr!r}."
                        ),
                        location=block.block_id,
                        suggestion=(
                            f"The canonical emit shape for {block.block_type!r} "
                            f"requires {attr!r} on the block-bearing wrapper. "
                            f"Re-prompt the rewrite tier to stamp every "
                            f"required data-cf-* attribute."
                        ),
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_MISSING_REQUIRED_ATTR",
                    content_length=content_length, tags_seen=parser.tags_seen,
                    failing_attr=first_missing,
                )
                continue

            # All four checks passed.
            passed_count += 1
            _emit_decision(
                capture, block,
                passed=True, code=None,
                content_length=content_length, tags_seen=parser.tags_seen,
            )

        passed = len(issues) == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


__all__ = ["RewriteHtmlShapeValidator"]
