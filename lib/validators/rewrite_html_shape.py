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

# IB4 — single source of truth for the generic-link-text list (2.4.4). Imported
# from the existing page-level WCAGValidator so the per-block check and the
# packaging gate agree. Defensive fallback keeps this module importable if the
# DART package layout shifts (the SemantiK migration touches that tree).
try:  # pragma: no cover — import-shape guard
    from DART.pdf_converter.wcag_validator import WCAGValidator as _WCAGValidator

    _GENERIC_LINK_TEXT: Tuple[str, ...] = tuple(
        t.strip().lower() for t in _WCAGValidator.GENERIC_LINK_TEXT
    )
except Exception:  # noqa: BLE001
    _GENERIC_LINK_TEXT = (
        "click here", "read more", "learn more", "more",
        "here", "link", "this", "page", "info",
    )

logger = logging.getLogger(__name__)

# IB4 — per-block WCAG 2.2 AA contract block-type partitions.
# Figure-bearing types whose <img> must carry alt-or-decorative (1.1.1).
_A11Y_FIGURE_BLOCK_TYPES: frozenset = frozenset(
    {"concept", "example", "explanation", "diagram"}
)
# Interactive types whose custom controls must be keyboard-operable
# (2.1.1 / 4.1.2). A native <details>/<summary> reveal passes for free.
_A11Y_INTERACTIVE_BLOCK_TYPES: frozenset = frozenset(
    {"self_check_question", "activity", "flip_card_grid"}
)
# B04 time-based media (lands fully in IB5; the contract ships dormant here).
_A11Y_MEDIA_BLOCK_TYPES: frozenset = frozenset({"multimedia"})

# ARIA roles that mark an element as an interactive control for the visible-
# focus (2.4.7) check. A custom <div role="button"> that suppresses its focus
# outline is as broken as a native <button> that does.
_INTERACTIVE_ARIA_ROLES: frozenset = frozenset(
    {
        "button", "link", "checkbox", "radio", "switch", "tab", "menuitem",
        "menuitemcheckbox", "menuitemradio", "option", "slider", "spinbutton",
        "textbox", "combobox", "listbox", "treeitem",
    }
)


def _suppresses_focus_outline(style: str) -> bool:
    """True iff an inline ``style`` removes the focus outline with no replacement.

    WCAG 2.4.7 (visible focus). A bare ``outline:none`` / ``outline:0`` on an
    interactive element kills the keyboard-focus ring. It is only a violation
    when NO replacement focus indicator is declared in the SAME inline style —
    a ``box-shadow``, a non-zero ``outline-width`` / ``outline-color`` /
    ``outline-style`` declaration, a ``border`` change, or an explicit
    focus-styling marker. ``style`` is expected already lower-cased.
    """
    compact = style.replace(" ", "")
    suppresses = "outline:none" in compact or "outline:0" in compact
    if not suppresses:
        return False
    # A replacement focus indicator in the same inline style rescues it.
    replacement = (
        "box-shadow:" in compact
        or "border:" in compact
        or "border-" in compact
        or "outline-style:" in compact
        or "outline-color:" in compact
        or "outline-width:" in compact
        or "outline-offset:" in compact
    )
    return not replacement


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
        # Issue I6 instruction-palette-v2 structural elements.
        "aside", "caption",
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
#
# Public surface: imported by
# ``Courseforge/generators/_rewrite_provider.py`` so the rewrite
# prompt enumerates the same attributes the gate enforces. Single
# source of truth; do NOT duplicate the table.
REQUIRED_ATTRS: Dict[str, Tuple[str, ...]] = {
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
    # Issue I6 instruction-palette-v2: table / key_idea carry the
    # content-type attribute (stamped on the rewrite path); acronym is a
    # wrapper-only block carrying only the universal block_id. The
    # element-shape contract (caption + scoped th / matched dl / aside) is
    # enforced by ``_check_palette_v2_shape`` below, NOT by these attrs.
    "table": ("data-cf-block-id", "data-cf-content-type"),
    "acronym": ("data-cf-block-id",),
    "key_idea": ("data-cf-block-id", "data-cf-content-type"),
    # IB5 framework-aligned pedagogical block types. Wrapper-only at the attr
    # surface (the a11y / shape richness is enforced by the IB5 arms in
    # _check_ib5_a11y_shape, not by data-cf-* attrs) — only the universal
    # block_id is required; diagram additionally carries the content-type
    # attribute the rewrite contract stamps.
    "hook": ("data-cf-block-id",),
    "multimedia": ("data-cf-block-id",),
    "worked_example": ("data-cf-block-id",),
    "diagram": ("data-cf-block-id",),
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
        # Issue I6 instruction-palette-v2 structural-shape tracking.
        self.saw_caption: bool = False
        self.saw_scoped_th: bool = False
        self.dt_count: int = 0
        self.dd_count: int = 0
        self.saw_dl: bool = False
        self.saw_aside: bool = False
        # IB4 per-block WCAG 2.2 AA contract tracking (1.1.1 / 2.1.1 /
        # 4.1.2 / 2.4.4). All default-empty so a parse over a block with no
        # a11y-relevant markup yields no a11y findings.
        self.saw_img: bool = False
        # True only if EVERY <img> seen carried a non-empty alt OR was marked
        # decorative (role="presentation" / aria-hidden="true"). Starts True
        # (vacuously satisfied) and is cleared the first time a bare <img>
        # without alt / decorative-marker is seen.
        self.img_alt_present: bool = True
        # Custom interactive control (data-cf-component-bearing div/span, OR a
        # click-bearing non-native element). Per-element keyboard-operability
        # facts: tabindex=0 + an ARIA role + an accessible name.
        self.saw_custom_interactive: bool = False
        self.interactive_tabindex: bool = False
        self.interactive_role: bool = False
        self.interactive_name: bool = False
        # WCAG 2.1.1 keyboard OPERATION (vs mere focusability). A static HTML
        # parser can't run JS keydown handlers, so the contract is: a custom
        # (non-native) control that LOOKS interactive (click-bearing or a
        # data-cf-component) MUST declare a keyboard affordance — a documented
        # key-handling marker the renderer emits (``data-cf-keys`` /
        # ``data-cf-keyboard`` / ``onkeydown`` / ``onkeyup`` / ``onkeypress``)
        # OR be a native interactive element (which is keyboard-operable for
        # free). Cleared the first time a click-bearing custom control with no
        # such affordance is seen.
        self.saw_clickonly_custom_control: bool = False
        # WCAG 2.4.7 visible focus. ``focus_suppressed`` becomes True the first
        # time an interactive (native OR custom) element carries an inline
        # ``outline:none`` / ``outline:0`` in its ``style`` attribute WITHOUT a
        # replacement focus indicator (a ``box-shadow`` / ``outline-*`` /
        # ``border`` declaration in the same inline style, or a focus-styling
        # data-marker).
        self.focus_suppressed: bool = False
        # WCAG 1.2.5 audio description: a <track kind="descriptions"> on the
        # media element, OR a described/transcript alternative.
        self.media_audio_desc: bool = False
        # Native <details>/<summary> reveal — keyboard-operable for free.
        self.saw_details_summary: bool = False
        # Anchor text accumulation for the descriptive-link-text check (2.4.4).
        self.link_texts: List[str] = []
        self._open_anchor_text: Optional[List[str]] = None
        # B04 multimedia (IB5 lands the block; the contract ships dormant here).
        self.media_track: bool = False
        self.media_transcript: bool = False
        # IB5 — additional B04/B06 shape flags activated by the IB5 a11y arms.
        # media controls (learner pause/segment) + a media element present.
        self.media_controls: bool = False
        self.saw_media_element: bool = False
        # B06 diagram: a long-description <details> (or aria-describedby target)
        # AND a <table> data-equivalent.
        self.saw_long_desc_details: bool = False
        self.saw_data_table: bool = False

    def _track_palette_v2(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        """Record the element-shape facts the I6 palette-v2 shape check reads."""
        if tag == "caption":
            self.saw_caption = True
        elif tag == "th":
            scope = ""
            for attr_name, attr_value in attrs:
                if attr_name == "scope" and isinstance(attr_value, str):
                    scope = attr_value.strip().lower()
            if scope in ("col", "row", "colgroup", "rowgroup"):
                self.saw_scoped_th = True
        elif tag == "dl":
            self.saw_dl = True
        elif tag == "dt":
            self.dt_count += 1
        elif tag == "dd":
            self.dd_count += 1
        elif tag == "aside":
            self.saw_aside = True

    def _track_a11y(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        """Record the per-block WCAG 2.2 AA facts the IB4 a11y check reads.

        Deterministic element-level tracking for: alt text on <img> (1.1.1),
        keyboard-operable custom controls (2.1.1 / 4.1.2), native <details>/
        <summary> reveal (keyboard-operable for free), descriptive link text
        (2.4.4), and the B04 multimedia captions/transcript stack (dormant
        until IB5 lands the block type).
        """
        attr_map: Dict[str, str] = {}
        for attr_name, attr_value in attrs:
            if attr_name:
                attr_map[attr_name.lower()] = (
                    attr_value if isinstance(attr_value, str) else ""
                )

        if tag == "img":
            self.saw_img = True
            alt = attr_map.get("alt")
            role = (attr_map.get("role") or "").strip().lower()
            aria_hidden = (attr_map.get("aria-hidden") or "").strip().lower()
            decorative = role == "presentation" or aria_hidden == "true"
            has_alt = alt is not None and alt.strip() != ""
            if not (has_alt or decorative):
                self.img_alt_present = False

        if tag in ("details", "summary"):
            self.saw_details_summary = True

        # Custom interactive control: a data-cf-component-bearing element, OR a
        # click-bearing element that is not a native interactive element.
        is_native_interactive = tag in ("button", "a", "details", "summary", "input", "select", "textarea")
        has_component = "data-cf-component" in attr_map
        has_click = "onclick" in attr_map
        # A documented keyboard affordance the renderer emits for a custom
        # control: a native key-handler attribute OR a data-marker enumerating
        # the keys the control honors (Enter/Space/Arrow). (WCAG 2.1.1)
        has_keyboard_affordance = (
            "onkeydown" in attr_map
            or "onkeyup" in attr_map
            or "onkeypress" in attr_map
            or "data-cf-keys" in attr_map
            or "data-cf-keyboard" in attr_map
        )
        if has_component or (has_click and not is_native_interactive):
            self.saw_custom_interactive = True
            tabindex = (attr_map.get("tabindex") or "").strip()
            if tabindex == "0":
                self.interactive_tabindex = True
            if (attr_map.get("role") or "").strip():
                self.interactive_role = True
            if (attr_map.get("aria-label") or "").strip() or (
                attr_map.get("aria-labelledby") or ""
            ).strip():
                self.interactive_name = True
            # 2.1.1 keyboard OPERATION: a custom control that responds to a
            # pointer click (or is a non-native component) but is NOT a native
            # interactive element and declares NO keyboard affordance can't be
            # operated from the keyboard. A native <details>/<summary> reveal
            # elsewhere in the block satisfies the contract for free, so the
            # caller exempts blocks that carry one.
            if not is_native_interactive and not has_keyboard_affordance:
                self.saw_clickonly_custom_control = True

        # 2.4.7 visible focus: an inline outline:none / outline:0 on an
        # interactive element (native OR custom) WITHOUT a replacement focus
        # indicator suppresses the focus ring. A custom non-native control with
        # tabindex / role is interactive; a native interactive element always
        # is.
        is_interactive_el = (
            is_native_interactive
            or has_component
            or has_click
            or (attr_map.get("tabindex") or "").strip() == "0"
            or (attr_map.get("role") or "").strip() in _INTERACTIVE_ARIA_ROLES
        )
        if is_interactive_el:
            style = (attr_map.get("style") or "").lower()
            if style and _suppresses_focus_outline(style):
                self.focus_suppressed = True

        if tag == "track":
            kind = (attr_map.get("kind") or "").strip().lower()
            if kind == "captions":
                self.media_track = True
            # 1.2.5 audio description: a <track kind="descriptions"> is the
            # native AD affordance.
            if kind == "descriptions":
                self.media_audio_desc = True
        if "data-cf-transcript" in attr_map:
            self.media_transcript = True
        # A described/transcript alternative also satisfies the audio-
        # description obligation (1.2.5) for a media-alternative-for-prerecorded
        # delivery. Three recognised AD affordances (matching what the renderer
        # actually emits): an explicit data-marker, OR a non-empty
        # ``class="audio-description"`` element (the B04 renderer's
        # ``<p class="audio-description">Audio description: …</p>`` skeleton).
        if "data-cf-audio-description" in attr_map:
            self.media_audio_desc = True
        cls = (attr_map.get("class") or "").lower()
        if "audio-description" in cls.split():
            self.media_audio_desc = True
        # IB5 B04 — media element + controls (learner pause/segment, 1.4.2/etc).
        if tag in ("video", "audio"):
            self.saw_media_element = True
            if "controls" in attr_map:
                self.media_controls = True
        # IB5 B06 — long-description <details> + a <table> data-equivalent. A
        # <details> anywhere in a diagram body satisfies the long-desc reveal;
        # an aria-describedby on any element is the alternate long-desc target.
        if tag == "details":
            self.saw_long_desc_details = True
        if "aria-describedby" in attr_map:
            self.saw_long_desc_details = True
        if tag == "table":
            self.saw_data_table = True

        if tag == "a" and "href" in attr_map:
            # Begin accumulating this anchor's visible text for the 2.4.4 check.
            self._open_anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._open_anchor_text is not None and data:
            self._open_anchor_text.append(data)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        self.tags_seen.append(tag)
        if tag in _BODY_TAGS:
            self.saw_body_tag = True
        self._track_palette_v2(tag, attrs)
        self._track_a11y(tag, attrs)
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
        self._track_a11y(tag, attrs)
        for attr_name, _attr_value in attrs:
            if attr_name and attr_name.startswith("data-cf-"):
                self.found_attrs.add(attr_name)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._open_anchor_text is not None:
            text = "".join(self._open_anchor_text).strip()
            self.link_texts.append(text)
            self._open_anchor_text = None
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


def _is_unshipped_escalation_tombstone(block: Block) -> bool:
    """True only for a marker-bearing block that never ships.

    Mirrors ``Courseforge.router.inter_tier_gates.
    _is_unshipped_escalation_tombstone``: the skip predicate is the
    EXACT ship-exclusion predicate the packager applies at HTML emit
    (``MCP/tools/pipeline_tools.py``'s per-page emit filter at
    ``pipeline_tools.py:5226``):

        ``escalation_marker is not None and not (content or "").strip()``

    A marker + EMPTY block is a tombstone the packager filters out of the
    shipped IMSCC, so re-auditing its degenerate HTML at the post-rewrite
    seam is a false positive → skip (True). A marker + NON-EMPTY block is
    a *salvaged* block (escalated-rewrite salvage path,
    ``_rewrite_provider.py::_apply_rewrite_touch`` via
    ``dataclasses.replace``, marker preserved) that DOES ship and per the
    design comment at ``pipeline_tools.py:5135`` MUST be audited here →
    return False so the caller audits it.

    Keep this predicate in lockstep with the packaging emit filter at
    ``pipeline_tools.py:5226``. The IMSCC-side backstop
    (``lib/validators/imscc.py::IMSCCValidator._check_escalated_blocks_absent``,
    code ``ESCALATED_BLOCK_IN_IMSCC``) only confirms no tombstone leaked
    into shipped HTML; it is warning-severity in `textbook_to_course` and
    input-starved in orchestrated runs (needs ``blocks_final_path`` +
    shipped HTML threaded in), so it is NOT a critical backstop — this
    lockstep predicate is what keeps salvaged blocks on the audit path.
    """
    marker = getattr(block, "escalation_marker", None)
    if marker is None or marker == "":
        return False
    content = getattr(block, "content", None)
    if isinstance(content, str):
        body = content
    elif content is None:
        body = ""
    else:
        # Non-str content (outline-tier dict) is skipped by the caller's
        # isinstance(content, str) guard anyway; never a shipped HTML
        # tombstone.
        return False
    return not body.strip()


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


def _check_palette_v2_shape(
    block_type: str, parser: "_ShapeParser"
) -> Optional[str]:
    """Return a shape-failure message for the I6 palette-v2 block types.

    Issue I6 added three WCAG-shaped structural block types. Beyond the
    generic body-tag / balance / required-attr checks the validator already
    runs, each carries a specific element contract:

    - ``table``: a real ``<table>`` MUST carry a ``<caption>`` AND at least
      one scoped ``<th scope="col"|"row">`` header cell (the accessible-table
      contract).
    - ``acronym``: the letter→term mapping MUST be a ``<dl>`` with matching
      ``<dt>`` / ``<dd>`` pairs (equal, non-zero counts).
    - ``key_idea``: the block MUST be an ``<aside>`` (semantically set apart
      for assistive technology), not a bare ``<div>`` / ``<p>``.

    Returns ``None`` when the block_type is not a palette-v2 type or the
    shape is satisfied; otherwise a short human-readable failure reason.
    """
    if block_type == "table":
        if not parser.saw_caption:
            return "table block is missing a <caption> element"
        if not parser.saw_scoped_th:
            return (
                "table block has no scoped header cell "
                "(<th scope=\"col\"|\"row\">)"
            )
        return None
    if block_type == "acronym":
        if not parser.saw_dl:
            return "acronym block has no <dl> letter→term mapping"
        if parser.dt_count == 0 or parser.dd_count == 0:
            return (
                "acronym block <dl> has no <dt>/<dd> pairs "
                f"(dt={parser.dt_count}, dd={parser.dd_count})"
            )
        if parser.dt_count != parser.dd_count:
            return (
                "acronym block <dl> has mismatched <dt>/<dd> counts "
                f"(dt={parser.dt_count}, dd={parser.dd_count}); each letter "
                "<dt> needs exactly one expansion <dd>"
            )
        return None
    if block_type == "key_idea":
        if not parser.saw_aside:
            return "key_idea block must be an <aside>, not a <div>/<p>"
        return None
    return None


def _check_block_a11y_contract(
    block_type: str, parser: "_ShapeParser", content: str
) -> List[Tuple[str, str]]:
    """Return ``[(code, reason), ...]`` of per-block WCAG 2.2 AA failures.

    IB4.1 — per-block-type WCAG 2.2 AA contract (the §4.2 baseline + the
    block-by-block playbook). Reads the deterministic facts the
    :class:`_ShapeParser` tracked (``_track_a11y``); does NOT re-parse.
    Returns one ``(code, reason)`` tuple per failed obligation (empty list when
    the block satisfies its per-type contract). Every issue is WARNING-severity
    at the call site (non-blocking day-1, deferred critical-flip).

    Obligations by block-type partition:

    * Figure-bearing (concept / example / explanation / diagram): every
      ``<img>`` MUST carry a non-empty ``alt`` OR be marked decorative
      (``role="presentation"`` / ``aria-hidden="true"``). (1.1.1)
    * Interactive (self_check_question / activity / flip_card_grid): a custom
      control (``data-cf-component``-bearing div/span, or a click-bearing
      non-native element) MUST be keyboard-FOCUSABLE — ``tabindex="0"`` AND an
      ARIA ``role`` AND an accessible ``name`` — AND keyboard-OPERABLE: either
      a native interactive element (``<button>``/``<a href>``/``<input>``/
      ``<details>``/``<summary>``) OR a documented keyboard affordance the
      renderer emits (``data-cf-keys`` / ``data-cf-keyboard`` /
      ``onkeydown``/``onkeyup``/``onkeypress``). A click-only custom
      ``<div role="button">`` with no key handler is operable by mouse only →
      ``BLOCK_KEYBOARD_OPERABLE`` (2.1.1). A native ``<details>``/``<summary>``
      reveal in the block satisfies the operability contract for free.
    * Any interactive element: an inline ``outline:none`` / ``outline:0`` with
      no replacement focus indicator suppresses the keyboard-focus ring →
      ``BLOCK_FOCUS_VISIBLE`` (2.4.7). Applies to ANY block.
    * Any block: a ``<a href>`` MUST NOT use generic link text
      (reuses ``WCAGValidator.GENERIC_LINK_TEXT``). (2.4.4)
    * ``multimedia`` (B04, dormant until IB5): MUST carry a
      ``<track kind="captions">`` AND a transcript anchor.

    The dedicated B04 time-based-media stack (captions / audio-description /
    transcript / controls, each with its own code) is checked by
    :func:`_check_ib5_a11y_shape` under ``ED4ALL_NEW_BLOCK_TYPES``; this
    figure-partition multimedia check stays the IB4 baseline for the
    ``ED4ALL_BLOCK_A11Y`` flag.
    """
    findings: List[Tuple[str, str]] = []

    # 2.4.4 — descriptive link text (applies to ANY block carrying an <a href>).
    for text in parser.link_texts:
        normalized = text.strip().lower()
        if normalized and normalized in _GENERIC_LINK_TEXT:
            findings.append((
                "REWRITE_BLOCK_A11Y_CONTRACT",
                f"non-descriptive link text: {text!r}",
            ))
            break

    # 1.1.1 — figure-bearing blocks: every <img> alt-or-decorative.
    if block_type in _A11Y_FIGURE_BLOCK_TYPES and parser.saw_img:
        if not parser.img_alt_present:
            findings.append((
                "REWRITE_BLOCK_A11Y_CONTRACT",
                "<img> with no alt text and not marked decorative",
            ))

    # 4.1.2 — interactive blocks: a custom control must be name/role/value
    # complete (keyboard-FOCUSABLE). A native <details>/<summary> reveal passes
    # for free.
    if block_type in _A11Y_INTERACTIVE_BLOCK_TYPES:
        if parser.saw_custom_interactive and not parser.saw_details_summary:
            if not (
                parser.interactive_tabindex
                and parser.interactive_role
                and parser.interactive_name
            ):
                findings.append((
                    "REWRITE_BLOCK_A11Y_CONTRACT",
                    "interactive component is not keyboard-operable "
                    "(missing tabindex=0 + role + accessible name)",
                ))

    # 2.1.1 — keyboard OPERATION. A custom non-native control that responds to
    # a pointer click (or is a data-cf-component) but declares NO keyboard
    # affordance and is NOT a native interactive element can only be operated
    # with a mouse. A native <details>/<summary> reveal in the block is the
    # keyboard-operable escape hatch. Applies to ANY block (interactivity isn't
    # confined to the interactive-block partition once a click handler appears).
    if parser.saw_clickonly_custom_control and not parser.saw_details_summary:
        findings.append((
            "BLOCK_KEYBOARD_OPERABLE",
            "custom control is operable by mouse only (a click-bearing "
            "non-native element with no keyboard affordance: needs a native "
            "interactive element, an onkeydown/onkeyup/onkeypress handler, or "
            "a data-cf-keys / data-cf-keyboard marker)",
        ))

    # 2.4.7 — visible focus. An interactive element with an inline outline:none
    # / outline:0 and no replacement focus indicator hides the focus ring.
    if parser.focus_suppressed:
        findings.append((
            "BLOCK_FOCUS_VISIBLE",
            "interactive element suppresses its focus outline "
            "(outline:none / outline:0) with no replacement focus indicator "
            "(box-shadow / border / outline-* declaration)",
        ))

    # B04 multimedia time-based-media stack (IB4 baseline; the richer
    # per-piece IB5 check rides ED4ALL_NEW_BLOCK_TYPES).
    if block_type in _A11Y_MEDIA_BLOCK_TYPES:
        if not (parser.media_track and parser.media_transcript):
            findings.append((
                "REWRITE_BLOCK_A11Y_CONTRACT",
                "time-based media missing captions/transcript stack",
            ))

    return findings


def _check_ib5_a11y_shape(
    block_type: str, parser: "_ShapeParser"
) -> List[Tuple[str, str]]:
    """Return ``[(code, reason), ...]`` of IB5 B04/B06 a11y-shape failures.

    IB5.7 — the type-specific a11y contracts for the two NEW types that carry a
    structural a11y obligation:

    * ``multimedia`` (B04): the MANDATORY time-based-media stack — each piece is
      checked and flagged with its OWN code so an operator can see exactly which
      part of the rendered skeleton is incomplete / empty-stub:

      - ``MULTIMEDIA_CONTROLS_MISSING`` — no native player ``controls`` on the
        ``<video>``/``<audio>`` element (learner pause/segment).
      - ``MULTIMEDIA_CAPTIONS_MISSING`` — no ``<track kind="captions">`` (1.2.4).
      - ``MULTIMEDIA_AUDIO_DESC_MISSING`` — no audio-description affordance:
        ``<track kind="descriptions">`` OR a ``data-cf-audio-description``
        described/transcript alternative (1.2.5).
      - ``MULTIMEDIA_TRANSCRIPT_MISSING`` — no transcript affordance
        (``<details data-cf-transcript>``).

    * ``diagram`` (B06): a structured long-description (``<details>`` reveal or
      an ``aria-describedby`` target) AND a ``<table>`` data-equivalent so the
      spatial relationships are available non-visually
      (``REWRITE_IB5_A11Y_CONTRACT``).

    Returns an empty list when the block_type is not an IB5 structural type or
    its contract is satisfied; otherwise one ``(code, reason)`` per missing
    piece. WARNING-severity surface (NOT a critical parse-fail).
    ``# TODO(calibration)`` — flip to critical only after a ≥2-corpus FP
    measurement.
    """
    findings: List[Tuple[str, str]] = []
    if block_type == "multimedia":
        # The renderer emits the skeleton; the validator confirms it's complete
        # AND non-empty-stub. A media element MUST be present — captions /
        # AD / controls live ON it, so a multimedia block with no <video>/
        # <audio> element at all is an empty stub that fails every piece.
        if not parser.media_controls:
            findings.append((
                "MULTIMEDIA_CONTROLS_MISSING",
                "time-based media has no native player controls "
                "(controls attribute on the <video>/<audio> element)",
            ))
        if not parser.media_track:
            findings.append((
                "MULTIMEDIA_CAPTIONS_MISSING",
                'time-based media has no captions track '
                '(<track kind="captions">) — WCAG 1.2.4',
            ))
        if not parser.media_audio_desc:
            findings.append((
                "MULTIMEDIA_AUDIO_DESC_MISSING",
                "time-based media has no audio-description affordance "
                '(<track kind="descriptions"> or a data-cf-audio-description '
                "described/transcript alternative) — WCAG 1.2.5",
            ))
        if not parser.media_transcript:
            findings.append((
                "MULTIMEDIA_TRANSCRIPT_MISSING",
                "time-based media has no transcript affordance "
                "(<details data-cf-transcript>)",
            ))
        return findings
    if block_type == "diagram":
        missing_d: List[str] = []
        if not parser.saw_long_desc_details:
            missing_d.append(
                "a structured long-description (<details> or aria-describedby)"
            )
        if not parser.saw_data_table:
            missing_d.append("a <table> data-equivalent")
        if missing_d:
            findings.append((
                "REWRITE_IB5_A11Y_CONTRACT",
                "diagram missing " + " and ".join(missing_d),
            ))
        return findings
    return findings


def _emit_a11y_decision(
    capture: Any,
    block: Block,
    *,
    n_representations: int,
    reason: Optional[str],
    saw_img: bool,
    interactive_role: bool,
    link_text_flagged: bool,
) -> None:
    """Emit one ``rewrite_block_a11y_check`` decision per audited block (IB4.7).

    Deterministic-check capture (no LLM call → no model/provider in the
    rationale), same class as the ``rewrite_html_shape_check`` event. Rationale
    interpolates dynamic signals so the audit trail is replayable post-hoc.
    ``capture`` is ``None`` when decision capture isn't wired → silent skip.
    """
    if capture is None:
        return
    decision = "passed" if reason is None else f"failed:{reason}"
    rationale = (
        f"Per-block WCAG 2.2 AA + UDL contract check on Block "
        f"{block.block_id!r}: block_type={block.block_type}, "
        f"n_representations={n_representations}, "
        f"a11y_reason={reason or 'none'}, saw_img={saw_img}, "
        f"interactive_role={interactive_role}, "
        f"link_text_flagged={link_text_flagged}, flag=ED4ALL_BLOCK_A11Y."
    )
    try:
        capture.log_decision(
            decision_type="rewrite_block_a11y_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — never let capture wiring kill the gate
        logger.debug(
            "DecisionCapture.log_decision raised on rewrite_block_a11y_check: %s",
            exc,
        )


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
        # IB4.1 — per-block WCAG 2.2 AA contract sub-check. Gated on the
        # block_a11y_enabled input the env resolver threads in (IB4.6); off →
        # the sub-check is a complete no-op so every existing snapshot / pass
        # is byte-identical.
        block_a11y_enabled = bool(inputs.get("block_a11y_enabled"))
        # IB5.7 — per-block B04/B06 a11y-shape arms. Gated on the
        # new_block_types_enabled input the env resolver threads in (IB5.8); off
        # → the IB5 arms are a complete no-op so every existing snapshot / pass
        # is byte-identical. The four IB5 types are never even constructed on a
        # flag-off run, so these arms are dead there.
        new_block_types_enabled = bool(inputs.get("new_block_types_enabled"))

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
            # Skip only an UNSHIPPED escalation tombstone (marker + empty
            # content): the packager filters it out of the IMSCC, so
            # auditing it is a false positive. A salvaged block (marker +
            # content) DOES ship and MUST stay on the audit path.
            # Predicate mirrors pipeline_tools.py:5226.
            if _is_unshipped_escalation_tombstone(block):
                continue
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

            # 2b. Issue I6 instruction-palette-v2 element-shape contract
            # (caption + scoped th / matched dl pairs / aside). Only fires
            # for the three palette-v2 block types; a no-op for every other
            # block_type. A shape miss is a structural defect → regenerate.
            palette_v2_fail = _check_palette_v2_shape(block.block_type, parser)
            if palette_v2_fail is not None:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="REWRITE_BLOCK_SHAPE_INVALID",
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} (block_type="
                            f"{block.block_type!r}) failed its element-shape "
                            f"contract: {palette_v2_fail}."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Re-prompt the rewrite tier to emit the canonical "
                            "WCAG markup for this block type (table: "
                            "<caption> + scoped <th>; acronym: <dl> with "
                            "matched <dt>/<dd>; key_idea: <aside>)."
                        ),
                    ))
                _emit_decision(
                    capture, block,
                    passed=False, code="REWRITE_BLOCK_SHAPE_INVALID",
                    content_length=content_length, tags_seen=parser.tags_seen,
                )
                continue

            # 2c. IB4.1 — per-block WCAG 2.2 AA contract (alt text / keyboard-
            # operable interaction + name/role/value / descriptive link text /
            # B04 captions+transcript). WARNING severity (does NOT promote the
            # critical REWRITE_BLOCK_SHAPE_INVALID path) and NON-terminal (no
            # `continue` — the block keeps flowing through the critical
            # required-attr check). A complete no-op when block_a11y_enabled is
            # False (byte-stable). Emits one rewrite_block_a11y_check decision
            # per audited block (IB4.7).
            if block_a11y_enabled:
                a11y_findings = _check_block_a11y_contract(
                    block.block_type, parser, content
                )
                link_flagged = any(
                    reason.startswith("non-descriptive link")
                    for _code, reason in a11y_findings
                )
                # Emit one WARNING per failed obligation. Distinct codes per
                # WCAG criterion: REWRITE_BLOCK_A11Y_CONTRACT (1.1.1 / 4.1.2 /
                # 2.4.4 / B04 baseline), BLOCK_KEYBOARD_OPERABLE (2.1.1),
                # BLOCK_FOCUS_VISIBLE (2.4.7). All warning-day-1 with a deferred
                # critical-flip (# TODO(calibration): flip after a ≥2-corpus FP
                # measurement, mirroring REWRITE_BLOCK_A11Y_CONTRACT).
                for code, reason in a11y_findings:
                    if len(issues) >= _ISSUE_LIST_CAP:
                        break
                    issues.append(GateIssue(
                        severity="warning",
                        code=code,
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} (block_type="
                            f"{block.block_type!r}) failed its per-block WCAG 2.2 "
                            f"AA contract: {reason}."
                        ),
                        location=block.block_id,
                    ))
                _emit_a11y_decision(
                    capture, block,
                    n_representations=getattr(block, "n_representations", 0),
                    reason=("; ".join(r for _c, r in a11y_findings) or None),
                    saw_img=parser.saw_img,
                    interactive_role=parser.interactive_role,
                    link_text_flagged=link_flagged,
                )

            # 2d. IB5.7 — per-block B04/B06 a11y-shape contracts (multimedia
            # captions/AD/transcript/controls stack + diagram long-desc /
            # data-table). WARNING severity (does NOT promote the critical
            # REWRITE_BLOCK_SHAPE_INVALID path) and NON-terminal (no `continue`)
            # — honors §3.3 warning-day-1 for NEW checks without demoting the
            # existing critical I6 / parse-fail behavior. A complete no-op when
            # new_block_types_enabled is False (byte-stable). Emits a
            # rewrite_html_shape_check decision (reuses the existing event; no
            # NEW call site). # TODO(calibration): flip to critical after a
            # ≥2-corpus FP measurement.
            if new_block_types_enabled:
                ib5_findings = _check_ib5_a11y_shape(block.block_type, parser)
                # One WARNING per missing piece. B04 multimedia uses per-piece
                # codes (MULTIMEDIA_CAPTIONS_MISSING / _AUDIO_DESC_MISSING /
                # _TRANSCRIPT_MISSING / _CONTROLS_MISSING) so an operator sees
                # exactly which part of the rendered media skeleton is an
                # empty stub; B06 diagram keeps the combined
                # REWRITE_IB5_A11Y_CONTRACT code.
                for code, reason in ib5_findings:
                    if len(issues) >= _ISSUE_LIST_CAP:
                        break
                    issues.append(GateIssue(
                        severity="warning",
                        code=code,
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} (block_type="
                            f"{block.block_type!r}) failed its IB5 a11y-shape "
                            f"contract: {reason}."
                        ),
                        location=block.block_id,
                    ))
                    _emit_decision(
                        capture, block,
                        passed=False, code=code,
                        content_length=content_length, tags_seen=parser.tags_seen,
                    )

            # 3. Required data-cf-* attributes per block_type. Missing
            # ANY required attr fails the gate.
            required = REQUIRED_ATTRS.get(block.block_type, ())
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

        # IB4.1 — the gate fails (blocks / regenerates) ONLY on CRITICAL issues.
        # The new REWRITE_BLOCK_A11Y_CONTRACT is warning-severity and
        # non-terminal day-1, so it never flips passed → False (the deferred
        # critical-flip is tracked in config/workflows.yaml). Before IB4 every
        # issue was critical, so this is byte-equivalent to the prior
        # `len(issues) == 0` for any flag-off run.
        critical_issues = sum(1 for i in issues if i.severity == "critical")
        passed = critical_issues == 0
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
