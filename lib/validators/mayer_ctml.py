"""Mayer CTML — MayerCtmlValidator (multimedia-learning structural check).

Mayer's Cognitive Theory of Multimedia Learning (CTML) names 12 evidence-based
multimedia-design principles. The per-block UDL surface
(``Courseforge/scripts/blocks.py::_derive_udl_coverage`` →
``lib/validators/udl_coverage.py::UdlCoverageValidator``) measures only the three
UDL *networks* (representation-mode count, response formats, engagement
affordance); NO Mayer CTML principle is checked anywhere, and the B04 / B06 / B05
renderers' caption / transcript / audio-description / long-description markup is
never audited for multimedia-learning quality.

This validator adds a precision-first, deterministic, **static-HTML-only** subset
of Mayer's principles over text+visual blocks. It is read-only: it invents no
ids/citations/sources, prunes/keeps nothing, and mutates no blocks.

In-scope principles (deterministically reachable from rendered HTML):

* SIGNALING (``CTML_NO_SIGNALING``) — a visual block with substantive body but
  no ``<figcaption>`` / heading / emphasis cue to direct attention.
* SPATIAL CONTIGUITY (``CTML_CAPTION_NOT_ADJACENT``) — an ``<img>``/``<video>``
  is present but its descriptive caption is absent or rendered outside the
  enclosing ``<figure>`` (corresponding words + picture should be near).
* REDUNDANCY (``CTML_REDUNDANT_NARRATION``) — the transcript / audio-description
  text is normalized-equal to the figcaption or to each other (identical
  on-screen text + narration; precision-first EXACT normalized equality only).
* SEGMENTING (``CTML_NOT_SEGMENTED``) — a ``multimedia`` / ``worked_example``
  body over the per-type body ceiling (``resolve_body_char_ceiling``) with zero
  segment markers (no ``<details>`` / ``<ol>`` / ``<ul>`` / sub-heading).

OUT OF SCOPE — the remaining 8 CTML principles need audio / timing / page-render
context a static-HTML validator cannot see, so they are NOT checked here (this
mirrors ``lib/validators/block_a11y.py``'s structural-only keyboard rationale):

* temporal contiguity (narration/animation synchrony — needs timing),
* pre-training (prior-knowledge sequencing — needs cross-page context),
* modality (spoken vs on-screen words — needs an audio channel),
* personalization (conversational tone — a prose-quality judgement),
* voice (human vs machine narration — needs the audio asset),
* image (omit redundant static speaker image — needs the asset),
* multimedia (words+pictures over words alone — a design-time choice),
* coherence / extraneous-motion — already covered by ``block_a11y`` reduced
  motion; this validator never duplicates that check.

Gating (default OFF, byte-stable): a no-op (``passed=True`` + a single
``MAYER_CTML_DISABLED`` info issue) unless ``ED4ALL_MAYER_CTML`` is truthy. No
renderer change exists, so emit / ``contentHash`` / snapshots stay byte-identical
when off (mirrors ``ED4ALL_CALLOUT_TYPED`` / ``ED4ALL_BLOCK_QUALITY_RUBRIC``).
When on, fires WARNING-severity issues (``action="regenerate"``) — warning-day-1
with a ``# TODO(calibration)`` deferred critical-flip (WS3/W4 pattern; NOT IB3).

Reachability note (for future maintainers): the ``multimedia`` / ``diagram`` /
``worked_example`` types only emit via the dynamic planner under
``ED4ALL_NEW_BLOCK_TYPES`` (default OFF), so on a default course run this gate
audits zero blocks and silently no-ops — the gate provides measurement signal
only when that flag is also on. The regression suite therefore builds Block
fixtures DIRECTLY (no course slug / path / run).

Inputs (``inputs`` dict):

    blocks: List[Block | dict]
        Outline- or rewrite-tier blocks. Only text+visual blocks are audited.
    mayer_ctml_enabled: Optional[bool]
        Override the ``ED4ALL_MAYER_CTML`` resolution (testing/routing seam).
    body_char_ceiling: Optional[int]
        Override the per-type body ceiling (testing seam).
    decision_capture / capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance; one ``mayer_ctml_check``
        event per validate.
    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to ``"mayer_ctml"``).

References:
    - R. E. Mayer, *Multimedia Learning* (CTML 12 principles).
    - ``lib/validators/callout_structure.py`` — the structural template cloned
      (flag-gated default-OFF, dual dict/str path, ``_ISSUE_LIST_CAP``,
      decision-capture emit, resolve-override seam, warning-day-1).
    - ``lib/validators/_block_rubric_helpers.py::resolve_body_char_ceiling``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

#: Cap on per-validate issue list to avoid runaway emit on visual-dense corpora.
_ISSUE_LIST_CAP: int = 50

#: ED4ALL_MAYER_CTML gate.
_MAYER_CTML_ENV = "ED4ALL_MAYER_CTML"
_TRUTHY: frozenset = frozenset({"1", "true", "yes", "on"})

#: Block types that are always audited (B04 multimedia / B06 diagram /
#: B05 worked_example). Other blocks are audited only if their rendered HTML
#: carries a visual element.
_VISUAL_BLOCK_TYPES: frozenset = frozenset({"multimedia", "diagram", "worked_example"})

#: The SEGMENTING check applies to media / worked-example bodies (time-based or
#: multi-step content that should be broken into learner-paced segments).
_SEGMENTING_BLOCK_TYPES: frozenset = frozenset({"multimedia", "worked_example"})

#: Substantive-body floor (chars) below which the SIGNALING check is skipped — a
#: tiny caption-only figure does not need extra signaling.
_SIGNALING_BODY_FLOOR: int = 40

_VISUAL_TAG_RE = re.compile(r"<(?:img|video|svg)\b", re.IGNORECASE)
_FIGURE_REGION_RE = re.compile(r"<figure\b[^>]*>.*?</figure>", re.IGNORECASE | re.DOTALL)
_FIGCAPTION_RE = re.compile(r"<figcaption\b[^>]*>(.*?)</figcaption>", re.IGNORECASE | re.DOTALL)
_HEADING_RE = re.compile(r"<h[1-6]\b", re.IGNORECASE)
_EMPHASIS_RE = re.compile(r"<(?:strong|em|b|mark)\b", re.IGNORECASE)
#: Segment markers: a paced/segmented body carries one of these.
_SEGMENT_MARKER_RE = re.compile(r"<(?:details|ol|ul|h[2-6])\b", re.IGNORECASE)
_TRANSCRIPT_RE = re.compile(
    r"<details\b[^>]*\bdata-cf-transcript\b[^>]*>(.*?)</details>",
    re.IGNORECASE | re.DOTALL,
)
_AUDIO_DESC_RE = re.compile(
    r'<p\b[^>]*class="[^"]*audio-description[^"]*"[^>]*>(.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)

#: The renderer prefixes the audio-description body with this chrome label
#: (``<p class="audio-description">Audio description: ...``); it is stripped
#: before the redundancy comparison so the LABEL never defeats text equality.
_AUDIO_DESC_LABEL_RE = re.compile(r"^audio description:\s*", re.IGNORECASE)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_text(s: Any) -> str:
    """Strip HTML tags + collapse whitespace; ``""`` for non-string/empty."""
    if not isinstance(s, str) or not s:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()


def _normalize(s: str) -> str:
    """Lowercase + tag-strip + ws-collapse for EXACT redundancy comparison."""
    return _strip_text(s).lower()


def _mayer_ctml_enabled() -> bool:
    """True iff ``ED4ALL_MAYER_CTML`` is truthy (read each call)."""
    return os.environ.get(_MAYER_CTML_ENV, "").strip().lower() in _TRUTHY


def resolve_mayer_ctml() -> bool:
    """Public resolver for ``ED4ALL_MAYER_CTML`` (routing-layer seam).

    Mirrors ``lib.generation.reflection_calibration.resolve_reflection_calibration``
    so ``MCP/hardening/gate_input_routing.py`` can thread the flag into the
    Block-input surface without re-reading the env at the call site.
    """
    return _mayer_ctml_enabled()


def _block_attr(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    if hasattr(block, key):
        return getattr(block, key)
    return None


def _block_type_of(block: Any) -> str:
    bt = _block_attr(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


def _dict_body_text(content: Dict[str, Any]) -> str:
    """Join string values + string list-items one level deep (dict-shape body)."""
    parts: List[str] = []
    for value in content.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(i) for i in value if isinstance(i, str))
    return _strip_text(" ".join(parts))


#: Keys whose truthy presence signals a caption / heading / attention cue.
_DICT_SIGNAL_KEYS: frozenset = frozenset(
    {"caption", "figcaption", "heading", "title", "label"}
)
#: List-valued keys whose >=2-entry presence signals a segmented body.
_DICT_SEGMENT_KEYS: frozenset = frozenset(
    {"steps", "segments", "sections", "items", "parts"}
)


class MayerCtmlValidator:
    """Mayer CTML structural multimedia-learning gate (precision-first subset)."""

    name = "mayer_ctml"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        from lib.validators._block_rubric_helpers import resolve_body_char_ceiling

        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        enabled = inputs.get("mayer_ctml_enabled")
        if enabled is None:
            enabled = _mayer_ctml_enabled()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="MAYER_CTML_DISABLED",
                        message=(
                            "ED4ALL_MAYER_CTML unset — Mayer-CTML multimedia "
                            "check skipped (byte-stable)."
                        ),
                    )
                ],
                metadata={"mayer_ctml": {"enabled": False}},
            )

        raw = inputs.get("blocks")
        if not isinstance(raw, list):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="MISSING_BLOCKS_INPUT",
                        message=(
                            "inputs['blocks'] absent/not a list; skipping "
                            "Mayer-CTML audit."
                        ),
                    )
                ],
                metadata={"mayer_ctml": {"enabled": True, "audited": 0}},
            )

        override_ceiling = inputs.get("body_char_ceiling")
        issues: List[GateIssue] = []
        audited = 0
        flagged = 0
        principle_counts: Dict[str, int] = {
            "CTML_NO_SIGNALING": 0,
            "CTML_CAPTION_NOT_ADJACENT": 0,
            "CTML_REDUNDANT_NARRATION": 0,
            "CTML_NOT_SEGMENTED": 0,
        }
        per_block: Dict[str, Dict[str, Any]] = {}

        for idx, block in enumerate(raw):
            block_type = _block_type_of(block)
            content = _block_attr(block, "content")
            raw_html = content if isinstance(content, str) else ""

            is_visual_type = block_type in _VISUAL_BLOCK_TYPES
            has_visual_markup = bool(raw_html) and bool(_VISUAL_TAG_RE.search(raw_html))
            if not (is_visual_type or has_visual_markup):
                continue  # non-visual block — not audited.

            audited += 1
            block_id = str(_block_attr(block, "block_id") or f"block-{idx}")
            ceiling = resolve_body_char_ceiling(override_ceiling, block_type)
            codes: List[str] = []

            if isinstance(content, str):
                codes.extend(self._audit_str(content, block_type, ceiling))
            elif isinstance(content, dict):
                codes.extend(self._audit_dict(content, block_type, ceiling))

            for code in codes:
                flagged += 1
                principle_counts[code] = principle_counts.get(code, 0) + 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(self._issue_for(code, block_id, ceiling))

            per_block[block_id] = {"block_type": block_type, "issues": codes}

        self._emit_decision(
            capture,
            audited=audited,
            flagged=flagged,
            principle_counts=principle_counts,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,  # warning-day-1
            issues=issues,
            action="regenerate" if flagged else None,
            metadata={
                "mayer_ctml": {
                    "enabled": True,
                    "audited": audited,
                    "flagged": flagged,
                    "principle_counts": principle_counts,
                    "per_block": per_block,
                }
            },
        )

    # ----- per-shape principle audits -------------------------------------

    @classmethod
    def _audit_str(cls, html: str, block_type: str, ceiling: int) -> List[str]:
        """Audit a rendered-HTML (str) block over all four principles."""
        codes: List[str] = []
        body_text = _strip_text(html)
        has_visual = bool(_VISUAL_TAG_RE.search(html))
        figcaptions = [m.group(1) for m in _FIGCAPTION_RE.finditer(html)]
        has_caption = bool(figcaptions)

        # SIGNALING — substantive body, no figcaption/heading/emphasis cue.
        if len(body_text) >= _SIGNALING_BODY_FLOOR:
            if not (
                has_caption
                or _HEADING_RE.search(html)
                or _EMPHASIS_RE.search(html)
            ):
                codes.append("CTML_NO_SIGNALING")

        # SPATIAL CONTIGUITY — a visual whose caption is absent or outside the
        # enclosing <figure>. Precision-first: flag only when a visual exists
        # but no <figure> region carries BOTH a visual and a figcaption.
        if has_visual:
            adjacent = any(
                _VISUAL_TAG_RE.search(region) and _FIGCAPTION_RE.search(region)
                for region in _FIGURE_REGION_RE.findall(html)
            )
            if not adjacent:
                codes.append("CTML_CAPTION_NOT_ADJACENT")

        # REDUNDANCY — transcript / audio-description normalized-equal to the
        # figcaption or to each other (identical narration + on-screen text).
        narrations: List[str] = []
        for regex, strip_label in (
            (_TRANSCRIPT_RE, False),
            (_AUDIO_DESC_RE, True),
        ):
            for m in regex.finditer(html):
                norm = _normalize(m.group(1))
                if strip_label:
                    norm = _AUDIO_DESC_LABEL_RE.sub("", norm).strip()
                if norm:
                    narrations.append(norm)
        caption_norms = [_normalize(c) for c in figcaptions if _normalize(c)]
        candidate_texts = narrations + caption_norms
        if len(candidate_texts) >= 2 and len(set(candidate_texts)) < len(
            candidate_texts
        ):
            codes.append("CTML_REDUNDANT_NARRATION")

        # SEGMENTING — long media/worked-example body, no segment markers.
        if block_type in _SEGMENTING_BLOCK_TYPES and len(body_text) > ceiling:
            if not _SEGMENT_MARKER_RE.search(html):
                codes.append("CTML_NOT_SEGMENTED")

        return codes

    @classmethod
    def _audit_dict(
        cls, content: Dict[str, Any], block_type: str, ceiling: int
    ) -> List[str]:
        """Audit an outline-tier (dict) block.

        Only SIGNALING + SEGMENTING are dict-inspectable; the str-only
        SPATIAL CONTIGUITY / REDUNDANCY principles need rendered markup and are
        skipped on the dict path.
        """
        codes: List[str] = []
        body_text = _dict_body_text(content)

        # SIGNALING — substantive body, no caption/heading/title cue key.
        if len(body_text) >= _SIGNALING_BODY_FLOOR:
            has_cue = any(
                bool(content.get(key)) for key in _DICT_SIGNAL_KEYS
            )
            if not has_cue:
                codes.append("CTML_NO_SIGNALING")

        # SEGMENTING — long body, no >=2-entry segment list.
        if block_type in _SEGMENTING_BLOCK_TYPES and len(body_text) > ceiling:
            segmented = any(
                isinstance(content.get(key), (list, tuple))
                and len(content.get(key)) >= 2
                for key in _DICT_SEGMENT_KEYS
            )
            if not segmented:
                codes.append("CTML_NOT_SEGMENTED")

        return codes

    # ----- issue + decision emit ------------------------------------------

    @staticmethod
    def _issue_for(code: str, block_id: str, ceiling: int) -> GateIssue:
        messages = {
            "CTML_NO_SIGNALING": (
                f"Visual block {block_id!r} has substantive body but no "
                f"<figcaption>/heading/emphasis cue — Mayer SIGNALING: direct "
                f"the learner's attention to the essential material."
            ),
            "CTML_CAPTION_NOT_ADJACENT": (
                f"Visual block {block_id!r} carries an <img>/<video> with no "
                f"caption inside its enclosing <figure> — Mayer SPATIAL "
                f"CONTIGUITY: words and the picture they describe must be near."
            ),
            "CTML_REDUNDANT_NARRATION": (
                f"Visual block {block_id!r} repeats the same text as transcript "
                f"+ audio-description/figcaption — Mayer REDUNDANCY: identical "
                f"on-screen text and narration overload the verbal channel."
            ),
            "CTML_NOT_SEGMENTED": (
                f"Media/worked-example block {block_id!r} body exceeds the "
                f"{ceiling}-char ceiling with no segment markers "
                f"(<details>/<ol>/<ul>/sub-heading) — Mayer SEGMENTING: break "
                f"continuous material into learner-paced segments."
            ),
        }
        suggestions = {
            "CTML_NO_SIGNALING": "Add a figcaption / heading / emphasis cue.",
            "CTML_CAPTION_NOT_ADJACENT": "Place the caption inside the <figure>.",
            "CTML_REDUNDANT_NARRATION": "Differentiate transcript vs description.",
            "CTML_NOT_SEGMENTED": "Break the body into <details>/list segments.",
        }
        return GateIssue(
            severity="warning",
            code=code,
            message=messages.get(code, code),
            location=block_id,
            suggestion=suggestions.get(code),
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        audited: int,
        flagged: int,
        principle_counts: Dict[str, int],
    ) -> None:
        if capture is None:
            return
        try:
            counts = ", ".join(f"{k}={v}" for k, v in sorted(principle_counts.items()))
            capture.log_decision(
                decision_type="mayer_ctml_check",
                decision=f"mayer_ctml_flagged={flagged}",
                rationale=(
                    f"Mayer-CTML multimedia audit over {audited} text+visual "
                    f"block(s): {flagged} principle flag(s) [{counts}]."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on mayer_ctml: %s", exc
            )


__all__ = ["MayerCtmlValidator", "resolve_mayer_ctml"]
