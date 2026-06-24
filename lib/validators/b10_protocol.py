"""FR-INT-04 — B10ProtocolValidator (three-move discussion protocol).

A real B10 discussion is NOT a single open-ended prompt — the framework's B10
contract is a **three-move protocol**: an initial POST, a required RESPOND to
peers, and a SYNTHESIZE move that consolidates the thread (the move that drives
the discussion up to Create). A discussion block that ships only one prompt,
with no required-response and no synthesis move, has COLLAPSED the protocol.

This validator audits every ``discussion_prompt`` block (framework B10) and
flags ``DISCUSSION_PROTOCOL_COLLAPSED`` when the three-move structure is absent.

It resolves the three moves from BOTH block shapes (mirroring the
``ResourceLinkPurposeValidator`` dual-path posture):

* the structured ``discussion_protocol`` field (``{"post", "respond",
  "synthesize"}``) is authoritative when populated — a non-empty value for each
  of the three moves clears the check.
* otherwise fall back to the block ``content`` (dict ``{"initial_post",
  "replies"/"respond", "synthesize"/"synthesis"}`` OR rewrite-tier HTML) and
  scan for the required-response + synthesize signals — a discussion with an
  initial post but no peer-response requirement and no synthesis move fires.

Gating (default OFF, byte-stable): a no-op (``passed=True`` + an info issue)
unless ``ED4ALL_NEW_BLOCK_TYPES`` is truthy — the same flag the B10 three-move
render scaffold rides, so a default run never sees this gate fire (mirrors the
B15 ``resource_link_purpose`` no-op posture). When the flag is on, fires
WARNING-severity issues (``action="regenerate"``) — warning-day-1 with a
deferred critical-flip.

Inputs (``inputs`` dict):

    blocks: List[Block | dict]
        Outline- or rewrite-tier blocks. Only ``discussion_prompt`` blocks are
        audited.
    new_block_types_enabled: Optional[bool]
        Override the ``ED4ALL_NEW_BLOCK_TYPES`` resolution (testing seam).
    decision_capture / capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance; one
        ``content_structure_check`` event per validate.
    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to ``"b10_protocol"``).

References:
    - ``lib/validators/resource_link_purpose.py`` — sibling B-code block-type
      gate (structure mirrored: flag-gated no-op, ``_ISSUE_LIST_CAP``,
      capture-emit, resolve-override seam, dual dict/str content path).
    - ``lib/generation/new_block_types.py`` — ``ED4ALL_NEW_BLOCK_TYPES`` gate.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap on per-validate issue list to avoid runaway emit on discussion-dense corpora.
_ISSUE_LIST_CAP: int = 50

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

#: Lexical signals that a peer-RESPOND move is required (str/HTML fallback).
_RESPOND_RE = re.compile(
    r"\b(?:repl(?:y|ies)|respond(?:ing)?|response|peer|classmate|"
    r"two\s+(?:of\s+your\s+)?(?:peers|classmates))\b",
    re.IGNORECASE,
)

#: Lexical signals that a SYNTHESIZE move is present (str/HTML fallback).
_SYNTHESIZE_RE = re.compile(
    r"\b(?:synthesi[sz]e|synthesis|consolidat|summari[sz]e\s+the\s+(?:thread|"
    r"discussion)|reconcil|integrate\s+(?:the\s+)?(?:views|perspectives|"
    r"viewpoints))\b",
    re.IGNORECASE,
)


def _strip_text(s: Any) -> str:
    """Strip HTML tags + collapse whitespace; ``""`` for non-string/empty."""
    if not isinstance(s, str) or not s:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()


def _nonempty(value: Any) -> bool:
    """True iff ``value`` is a non-blank string."""
    return isinstance(value, str) and bool(value.strip())


class B10ProtocolValidator:
    """FR-INT-04 — three-move discussion-protocol gate for B10 blocks."""

    name = "b10_protocol"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("new_block_types_enabled")
        if enabled is None:
            from lib.generation.new_block_types import resolve_new_block_types

            enabled = resolve_new_block_types()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="B10_PROTOCOL_DISABLED",
                        message=(
                            "ED4ALL_NEW_BLOCK_TYPES unset — B10 discussion-"
                            "protocol check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"b10_protocol": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        discussion_blocks = 0
        collapsed = 0
        per_block: Dict[str, Dict[str, Any]] = {}

        for idx, block in enumerate(blocks):
            if _block_type_of(block) != "discussion_prompt":
                continue
            discussion_blocks += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            has_post, has_respond, has_synthesize = _resolve_moves(block)

            block_collapsed = not (has_post and has_respond and has_synthesize)
            if block_collapsed:
                collapsed += 1
                missing = [
                    name
                    for name, present in (
                        ("post", has_post),
                        ("respond", has_respond),
                        ("synthesize", has_synthesize),
                    )
                    if not present
                ]
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="DISCUSSION_PROTOCOL_COLLAPSED",
                            message=(
                                f"Discussion block {block_id!r} has collapsed the "
                                f"B10 three-move protocol — missing the "
                                f"{', '.join(missing)} move(s). A real discussion "
                                f"is post -> respond-to-peers -> synthesize, not a "
                                f"single prompt (framework B10)."
                            ),
                            location=block_id,
                            suggestion=(
                                "Add the missing move(s): a required peer-response "
                                "step and a synthesis move that consolidates the "
                                "thread (drives the discussion to Create)."
                            ),
                        )
                    )
            per_block[block_id] = {
                "has_post": has_post,
                "has_respond": has_respond,
                "has_synthesize": has_synthesize,
                "collapsed": block_collapsed,
            }

        self._emit_decision(
            capture,
            discussion_blocks=discussion_blocks,
            collapsed=collapsed,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action="regenerate" if collapsed else None,
            metadata={
                "b10_protocol": {
                    "enabled": True,
                    "discussion_blocks": discussion_blocks,
                    "collapsed": collapsed,
                    "per_block": per_block,
                }
            },
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        discussion_blocks: int,
        collapsed: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=f"b10_protocol_collapsed={collapsed}",
                rationale=(
                    f"FR-INT-04 B10 three-move protocol audit over "
                    f"{discussion_blocks} discussion block(s): {collapsed} "
                    f"collapsed the post->respond->synthesize protocol into a "
                    f"single prompt (missing required-response / synthesis move)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on b10_protocol: %s",
                exc,
            )


def _block_attr(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    if hasattr(block, key):
        return getattr(block, key)
    return None


def _block_type_of(block: Any) -> str:
    bt = _block_attr(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


def _resolve_moves(block: Any) -> "tuple[bool, bool, bool]":
    """Resolve ``(has_post, has_respond, has_synthesize)`` for a B10 block.

    Precedence: the structured ``discussion_protocol`` field is authoritative
    when populated (a non-empty string for a move = present); otherwise fall
    back to the block ``content`` (dict keys OR rewrite-tier HTML lexical scan).
    """
    proto = _block_attr(block, "discussion_protocol")
    if isinstance(proto, dict) and proto:
        post = _nonempty(proto.get("post")) or _nonempty(proto.get("initial_post"))
        respond = _nonempty(proto.get("respond")) or _nonempty(proto.get("replies"))
        synth = _nonempty(proto.get("synthesize")) or _nonempty(
            proto.get("synthesis")
        )
        return post, respond, synth

    content = _block_attr(block, "content")
    if isinstance(content, dict):
        post = _nonempty(content.get("initial_post")) or _nonempty(
            content.get("post")
        ) or _nonempty(content.get("prompt"))
        respond = _nonempty(content.get("replies")) or _nonempty(
            content.get("respond")
        )
        synth = _nonempty(content.get("synthesize")) or _nonempty(
            content.get("synthesis")
        )
        # If a respond/synthesize key is absent, scan any prose for the signal.
        prose = " ".join(
            _strip_text(v) for v in content.values() if isinstance(v, str)
        )
        if not respond and prose:
            respond = bool(_RESPOND_RE.search(prose))
        if not synth and prose:
            synth = bool(_SYNTHESIZE_RE.search(prose))
        return post, respond, synth

    if isinstance(content, str):
        text = _strip_text(content)
        post = bool(text)  # an authored discussion body is the post move
        respond = bool(_RESPOND_RE.search(text))
        synth = bool(_SYNTHESIZE_RE.search(text))
        return post, respond, synth

    return False, False, False


__all__ = ["B10ProtocolValidator"]
