"""B15 — ResourceLinkPurposeValidator (WCAG 2.4.4 Link Purpose).

The B15 ``resources`` block type (Resources / Further Reading) is the one
framework-aligned block whose entire payload is OUTBOUND links. WCAG 2.4.4
Link Purpose (In Context) requires the purpose of each link to be determinable
from the link text. This validator audits every link inside a ``resources``
block and flags a link whose text is non-descriptive — a bare URL, an empty
anchor, or a generic filler phrase ("click here", "read more", "here", "link",
"this") — emitting ``RESOURCE_LINK_PURPOSE_UNCLEAR``.

It runs over BOTH block shapes (mirroring the ``Block*Validator`` dual-path
posture):

* dict content (outline tier) — reads ``content["links"]`` (a list of
  ``{"url", "title"/"text"/"label", "annotation"}`` dicts, or bare URL
  strings). The descriptive text is the resolved ``title``.
* str content (rewrite-tier HTML) — parses every ``<a ...>text</a>`` and audits
  the visible anchor text.

Gating (default OFF, byte-stable): a no-op (``passed=True`` + an info issue)
unless ``ED4ALL_NEW_BLOCK_TYPES`` is truthy — the same flag that makes the B15
``resources`` type selectable / renderable, so a default run never sees this
gate fire (mirrors the IB5 new-block-type posture and the IB6 rubric-flag
no-op). When the flag is on, fires WARNING-severity issues
(``action="regenerate"``) — warning-day-1 with a deferred critical-flip.

Inputs (``inputs`` dict):

    blocks: List[Block | dict]
        Outline- or rewrite-tier blocks. Only ``resources`` blocks are audited.
    new_block_types_enabled: Optional[bool]
        Override the ``ED4ALL_NEW_BLOCK_TYPES`` resolution (testing seam).
    decision_capture / capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance; one
        ``content_structure_check`` event per validate.
    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"resource_link_purpose"``).

References:
    - WCAG 2.2 SC 2.4.4 Link Purpose (In Context).
    - ``lib/validators/bloom_type_range.py`` — sibling block-type gate
      (structure mirrored: empty-input graceful pass, ``_ISSUE_LIST_CAP``,
      capture-emit, resolve-override seam).
    - ``lib/generation/new_block_types.py`` — ``ED4ALL_NEW_BLOCK_TYPES`` gate.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap on per-validate issue list to avoid runaway emit on link-dense corpora.
_ISSUE_LIST_CAP: int = 50

#: Generic, non-descriptive anchor phrases that fail 2.4.4 on their own.
#: Matched against the WHOLE stripped lowercased link text.
_GENERIC_LINK_PHRASES: frozenset = frozenset(
    {
        "click here",
        "click",
        "here",
        "read more",
        "more",
        "learn more",
        "link",
        "this link",
        "this",
        "go",
        "download",
        "view",
        "see more",
        "details",
    }
)

#: A bare URL used AS the link text (http(s)://… or www.…). 2.4.4 wants a human
#: title, not the raw address.
_BARE_URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)

#: Extract each ``<a ...>inner</a>`` from rewrite-tier HTML (non-greedy inner).
_ANCHOR_RE = re.compile(r"<a\b[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_text(s: Any) -> str:
    """Strip HTML tags + collapse whitespace; ``""`` for non-string/empty."""
    if not isinstance(s, str) or not s:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()


def _is_unclear_link_text(text: str) -> bool:
    """True iff ``text`` is non-descriptive per WCAG 2.4.4."""
    t = text.strip()
    if not t:
        return True
    if _BARE_URL_RE.match(t):
        return True
    return t.lower() in _GENERIC_LINK_PHRASES


class ResourceLinkPurposeValidator:
    """B15 — WCAG 2.4.4 descriptive-link-text gate for ``resources`` blocks."""

    name = "resource_link_purpose"
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
                        code="RESOURCE_LINK_PURPOSE_DISABLED",
                        message=(
                            "ED4ALL_NEW_BLOCK_TYPES unset — resource link-purpose "
                            "check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"resource_link_purpose": {"enabled": False}},
            )

        blocks = inputs.get("blocks") or []
        issues: List[GateIssue] = []
        resources_blocks = 0
        links_audited = 0
        unclear = 0
        per_block: Dict[str, Dict[str, Any]] = {}

        for idx, block in enumerate(blocks):
            if _block_type_of(block) != "resources":
                continue
            resources_blocks += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            link_texts = _extract_link_texts(block)
            block_unclear = 0
            for text in link_texts:
                links_audited += 1
                if _is_unclear_link_text(text):
                    unclear += 1
                    block_unclear += 1
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="RESOURCE_LINK_PURPOSE_UNCLEAR",
                                message=(
                                    f"Resources block {block_id!r} has a link "
                                    f"whose text is non-descriptive "
                                    f"({text[:60]!r}) — WCAG 2.4.4 requires the "
                                    f"link purpose be clear from the link text "
                                    f"(no bare URL / 'click here' / 'read more')."
                                ),
                                location=block_id,
                                suggestion=(
                                    "Give the link descriptive text (the "
                                    "resource title), not a bare URL or filler."
                                ),
                            )
                        )
            per_block[block_id] = {
                "links_audited": len(link_texts),
                "links_unclear": block_unclear,
            }

        self._emit_decision(
            capture,
            resources_blocks=resources_blocks,
            links_audited=links_audited,
            unclear=unclear,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action="regenerate" if unclear else None,
            metadata={
                "resource_link_purpose": {
                    "enabled": True,
                    "resources_blocks": resources_blocks,
                    "links_audited": links_audited,
                    "links_unclear": unclear,
                    "per_block": per_block,
                }
            },
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        resources_blocks: int,
        links_audited: int,
        unclear: int,
    ) -> None:
        if capture is None:
            return
        try:
            capture.log_decision(
                decision_type="content_structure_check",
                decision=f"resource_link_purpose_unclear={unclear}",
                rationale=(
                    f"WCAG 2.4.4 link-purpose audit over {resources_blocks} B15 "
                    f"resources block(s): {links_audited} links checked, "
                    f"{unclear} non-descriptive (bare URL / generic filler)."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "resource_link_purpose: %s",
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


def _extract_link_texts(block: Any) -> List[str]:
    """Resolve every link's DESCRIPTIVE text from a resources block.

    dict content -> the resolved per-link title (falling back to the URL when
    no title resolves, so a titleless link is correctly flagged unclear).
    str content -> the visible anchor text of every ``<a>`` (an empty anchor or
    a bare-URL anchor is preserved verbatim so the audit flags it).
    """
    content = _block_attr(block, "content")
    texts: List[str] = []
    if isinstance(content, dict):
        raw_links = content.get("links") or []
    elif isinstance(content, (list, tuple)):
        raw_links = content
    elif isinstance(content, str):
        for inner in _ANCHOR_RE.findall(content):
            texts.append(_strip_text(inner))
        return texts
    else:
        return texts

    for link in raw_links:
        if isinstance(link, dict):
            title = str(
                link.get("title")
                or link.get("text")
                or link.get("label")
                or ""
            ).strip()
            url = str(link.get("url") or link.get("href") or "").strip()
            texts.append(title or url)
        else:
            texts.append(str(link or "").strip())
    return texts


__all__ = ["ResourceLinkPurposeValidator"]
