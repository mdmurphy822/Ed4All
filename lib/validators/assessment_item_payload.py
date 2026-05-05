"""Worker W7 — assessment_item payload-shape validator.

Closes the W7 regression class: pre-W7 the outline-tier per-block JSON
schema required only ``stem`` + ``answer_key`` for ``assessment_item``
Blocks (`Courseforge/generators/_outline_provider.py:431-438`), and the
four ``Block*Validator``s in ``Courseforge/router/inter_tier_gates.py``
only validated generic CURIE / content_type / objective_ref / source_id
shape. So a model could emit a "valid" assessment_item carrying one (or
zero) distractor(s) and the rewrite tier would happily ship it.

This validator gates the per-Block payload of every ``assessment_item``
Block end-to-end, mirroring the four shape-discriminating
``Block*Validator``s' two-mode dispatch:

- **Outline mode** (``isinstance(block.content, dict)``): walks the
  dict-side payload (``content["distractors"]``, ``content["correct_answer_index"]``,
  per-distractor ``text`` / ``misconception_ref``).
- **Rewrite mode** (``isinstance(block.content, str)``): scrapes the
  HTML body for ``<li data-cf-distractor-index="N">`` siblings under an
  ``<ol>`` / ``<ul>``; asserts >=2 entries and contiguous 0-based indices.

GateIssue codes (all severity ``critical``):

- ``ASSESSMENT_ITEM_MISSING_DISTRACTORS`` — no ``distractors`` field
  (outline) / fewer than 2 ``<li data-cf-distractor-index>`` siblings
  (rewrite).
- ``ASSESSMENT_ITEM_INVALID_MISCONCEPTION_REF`` — a distractor has a
  ``misconception_ref`` that doesn't match
  ``^[A-Z]{2,}-\\d{2,}#m\\d+$`` (outline-only — rewrite-tier HTML
  doesn't carry the ref).
- ``ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE`` —
  ``correct_answer_index`` < 0 or >= len(distractors) (outline-only).
- ``ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING`` — a distractor has no
  ``text`` field or empty ``text`` (outline) / a
  ``<li data-cf-distractor-index>`` element with no body text
  (rewrite).

Failure ``action`` is always ``"regenerate"``: a rewrite-tier re-roll
can fix every payload-shape miss (the model just needs to be told to
emit the right structure). Block-side ``action="block"`` is reserved
for structural references that resolve against an external manifest
(objective_id / source_id) — the assessment_item payload has no such
external resolution requirement.

Wired symmetrically at the inter-tier and post-rewrite seams via
``MCP/hardening/gate_input_routing.py``'s ``_build_block_input_outline``
/ ``_build_block_input_rewrite`` shims; per-block-type pinning via
``Courseforge/config/block_routing.yaml`` is unaffected.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# ``blocks.py`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# router / inter_tier_gates import bridge so ``from blocks import Block``
# resolves regardless of how this module is loaded.
_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover — import-bridge tested via the test suite
    from blocks import Block  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    Block = None  # type: ignore[assignment,misc]


# Cap per-block issues so a uniformly broken assessment batch doesn't
# drown the gate report. Mirrors ``_ISSUE_LIST_CAP`` in inter_tier_gates.
_ISSUE_LIST_CAP = 50

# Misconception CURIE pattern (mirrors $defs.AssessmentItem in
# schemas/knowledge/courseforge_jsonld_v1.schema.json + the outline
# provider's per-distractor schema).
_MISCONCEPTION_REF_RE = re.compile(r"^[A-Z]{2,}-\d{2,}#m\d+$")

# Rewrite-mode HTML scraper. Captures every
# ``<li data-cf-distractor-index="N">`` element with the captured
# index. Quotes normalised at emit; accept both single- and double-
# quoted forms for forward-compat with future rendering changes.
_DATA_CF_DISTRACTOR_INDEX_LI_RE = re.compile(
    r'<li[^>]*data-cf-distractor-index=["\'](\d+)["\'][^>]*>(.*?)</li>',
    re.IGNORECASE | re.DOTALL,
)

# Cheap text-presence check: an <li ...></li> with at least one
# non-whitespace, non-tag character between the opening and closing tag.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """Strip tags + collapse whitespace (mirrors inter_tier_gates._strip_html)."""
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _coerce_blocks(inputs: Dict[str, Any]) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs['blocks']``.

    Mirrors ``inter_tier_gates._coerce_blocks`` so the same shape
    contract applies. Returns ``(blocks, error_issue)``; non-None
    error_issue is wrapped into a ``passed=False, action="regenerate"``
    GateResult by the caller.
    """
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


def _audit_outline_block(
    block: Any,
    issues: List[GateIssue],
) -> bool:
    """Audit one outline-tier (dict-content) assessment_item Block.

    Returns True iff every payload check passes. Side-effect: appends
    GateIssue entries to ``issues`` on each miss (capped at
    ``_ISSUE_LIST_CAP``).
    """
    content = block.content
    if not isinstance(content, dict):  # defensive: only outline path
        return True

    distractors = content.get("distractors")
    if not isinstance(distractors, list) or len(distractors) < 2:
        if len(issues) < _ISSUE_LIST_CAP:
            issues.append(GateIssue(
                severity="critical",
                code="ASSESSMENT_ITEM_MISSING_DISTRACTORS",
                message=(
                    f"assessment_item Block {block.block_id!r} requires "
                    f"a `distractors` array with at least 2 entries; "
                    f"got {distractors!r}."
                ),
                location=block.block_id,
                suggestion=(
                    "Re-roll the outline tier with the per-block JSON "
                    "schema's distractors[] requirement (see "
                    "Courseforge/generators/_outline_provider.py "
                    "assessment_item branch)."
                ),
            ))
        # If distractors is wholly missing/wrong shape, the index
        # range / per-entry checks below would crash on len(...) /
        # iteration; bail this block here and report the rest of the
        # batch.
        return False

    block_passed = True

    # Per-distractor text + misconception_ref checks.
    for idx, entry in enumerate(distractors):
        if not isinstance(entry, dict):
            block_passed = False
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(GateIssue(
                    severity="critical",
                    code="ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING",
                    message=(
                        f"assessment_item Block {block.block_id!r} "
                        f"distractor[{idx}] is not an object; got "
                        f"{type(entry).__name__}."
                    ),
                    location=block.block_id,
                ))
            continue
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            block_passed = False
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(GateIssue(
                    severity="critical",
                    code="ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING",
                    message=(
                        f"assessment_item Block {block.block_id!r} "
                        f"distractor[{idx}] has no `text` (or empty); "
                        f"got {text!r}."
                    ),
                    location=block.block_id,
                ))
        ref = entry.get("misconception_ref")
        if ref is not None:
            if not isinstance(ref, str) or not _MISCONCEPTION_REF_RE.match(ref):
                block_passed = False
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="ASSESSMENT_ITEM_INVALID_MISCONCEPTION_REF",
                        message=(
                            f"assessment_item Block {block.block_id!r} "
                            f"distractor[{idx}].misconception_ref={ref!r} "
                            f"does not match the canonical pattern "
                            f"^[A-Z]{{2,}}-\\d{{2,}}#m\\d+$."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "misconception_ref is OPTIONAL — drop it "
                            "or correct to the canonical Misconception "
                            "CURIE (e.g., TO-01#m1)."
                        ),
                    ))

    # correct_answer_index range check.
    cai = content.get("correct_answer_index")
    if not isinstance(cai, int) or isinstance(cai, bool):
        block_passed = False
        if len(issues) < _ISSUE_LIST_CAP:
            issues.append(GateIssue(
                severity="critical",
                code="ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE",
                message=(
                    f"assessment_item Block {block.block_id!r} "
                    f"correct_answer_index must be an integer; got "
                    f"{cai!r} ({type(cai).__name__})."
                ),
                location=block.block_id,
            ))
    elif cai < 0 or cai >= len(distractors):
        block_passed = False
        if len(issues) < _ISSUE_LIST_CAP:
            issues.append(GateIssue(
                severity="critical",
                code="ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE",
                message=(
                    f"assessment_item Block {block.block_id!r} "
                    f"correct_answer_index={cai} is out of range; "
                    f"distractors has {len(distractors)} entries "
                    f"(valid: 0..{len(distractors) - 1})."
                ),
                location=block.block_id,
            ))

    return block_passed


def _audit_rewrite_block(
    block: Any,
    issues: List[GateIssue],
) -> bool:
    """Audit one rewrite-tier (str-content) assessment_item Block.

    Scrapes ``<li data-cf-distractor-index="N">`` siblings and asserts
    >=2 entries with contiguous 0-based indices, and that each <li>
    carries a non-empty body. The rewrite-tier HTML doesn't carry
    misconception_ref or correct_answer_index attributes (those are
    structural fields on the outline tier's dict payload), so only
    ASSESSMENT_ITEM_MISSING_DISTRACTORS and
    ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING are reachable from here.
    """
    content = block.content
    if not isinstance(content, str):  # defensive: only rewrite path
        return True

    matches = list(_DATA_CF_DISTRACTOR_INDEX_LI_RE.finditer(content))
    if len(matches) < 2:
        if len(issues) < _ISSUE_LIST_CAP:
            issues.append(GateIssue(
                severity="critical",
                code="ASSESSMENT_ITEM_MISSING_DISTRACTORS",
                message=(
                    f"assessment_item Block {block.block_id!r} rewrite-"
                    f"tier HTML has fewer than 2 <li data-cf-distractor-"
                    f"index='N'> siblings (got {len(matches)})."
                ),
                location=block.block_id,
                suggestion=(
                    "Re-roll the rewrite tier; ensure the HTML emit "
                    "wraps each distractor in <li data-cf-distractor-"
                    "index='N'>...</li> with at least 2 entries."
                ),
            ))
        return False

    block_passed = True
    seen_indices: Set[int] = set()
    for match in matches:
        try:
            idx = int(match.group(1))
        except (TypeError, ValueError):
            continue
        seen_indices.add(idx)
        body_text = _strip_html(match.group(2) or "")
        if not body_text:
            block_passed = False
            if len(issues) < _ISSUE_LIST_CAP:
                issues.append(GateIssue(
                    severity="critical",
                    code="ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING",
                    message=(
                        f"assessment_item Block {block.block_id!r} "
                        f"<li data-cf-distractor-index='{idx}'> has "
                        f"no body text after HTML strip."
                    ),
                    location=block.block_id,
                ))

    # Indices must be contiguous from 0 (mirrors the
    # correct_answer_index range check on the dict path: a non-
    # contiguous run breaks the cross-walk between the outline-tier
    # ``correct_answer_index`` integer and the rewrite-tier <li>
    # ordering).
    expected = set(range(len(matches)))
    if seen_indices != expected:
        block_passed = False
        if len(issues) < _ISSUE_LIST_CAP:
            issues.append(GateIssue(
                severity="critical",
                code="ASSESSMENT_ITEM_CORRECT_INDEX_OUT_OF_RANGE",
                message=(
                    f"assessment_item Block {block.block_id!r} rewrite-"
                    f"tier <li data-cf-distractor-index> values are not "
                    f"contiguous from 0; got {sorted(seen_indices)!r}, "
                    f"expected {sorted(expected)!r}."
                ),
                location=block.block_id,
            ))

    return block_passed


class BlockAssessmentItemPayloadValidator:
    """Worker W7 assessment_item payload-shape gate.

    Walks ``inputs['blocks']``, filters to ``block.block_type ==
    "assessment_item"``, and validates payload shape. Non-assessment_item
    blocks are silently skipped (no-op). Mirrors the validate() shape
    of the four sibling Block validators in
    ``Courseforge/router/inter_tier_gates.py`` so the existing
    ``_build_block_input_outline`` / ``_build_block_input_rewrite``
    builders in ``MCP/hardening/gate_input_routing.py`` wire it cleanly
    without a dedicated builder.
    """

    name = "block_assessment_item_payload"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
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
            block_type = getattr(block, "block_type", None)
            if block_type != "assessment_item":
                continue

            content = getattr(block, "content", None)
            if isinstance(content, dict):
                audited += 1
                if _audit_outline_block(block, issues):
                    passed_count += 1
            elif isinstance(content, str):
                audited += 1
                if _audit_rewrite_block(block, issues):
                    passed_count += 1
            else:
                # Non-dict / non-str content — nothing to audit.
                continue

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


__all__ = ["BlockAssessmentItemPayloadValidator"]
