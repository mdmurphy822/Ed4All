"""``NumericLiteralGroundingValidator`` tests — the math-fabrication control.

Covers the per-block numeric-literal grounding gate (the fabrication control
for NUMERIC / math content the number-blind NLI gate cannot provide). This is a
PURE text/regex validator (no embeddings, no NLI), so unlike
``test_block_prose_entailment`` these tests need no model stub and are NOT
slow-marked.

Test cases:

1. Keystone: fabricated worked-example INPUT (``-40/88``, absent from source)
   FLAGS (``NUMERIC_LITERAL_UNGROUNDED``).
2. OCR-flattened grounded block (source stores ``− 55 84``) PASSES.
3. Computed-result-not-in-source behaves per the measured rule
   (count-of-absent-fractions ≥ 1; a grounded input + a source-absent result
   under the default threshold of 1 flags — documents the measured rule and the
   ``min_absent_fractions`` knob that relaxes it).
4. Graceful-degrade: no cited source → warning ``NUMERIC_NO_GROUNDING_SOURCE``,
   ``passed=True``.
5. Shadow path computes without gating.
6. Decision-capture fires with dynamic per-block signals.
7. Skip set (``assessment_item`` / ``objective`` skipped).
8. Block with no fraction literals → no-op pass (not audited, no event).
9. The two REAL 7B fabrication blocks flag against the corpus; the grounded
   blocks pass (mirrors the module's real-data measurement, run only when the
   fixture corpus is present).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.utils import strip_html_to_text  # noqa: E402
from lib.validators.numeric_literal_grounding import (  # noqa: E402
    NumericLiteralGroundingValidator,
    _CODE_NUMERIC_FABRICATION,
    _CODE_NUMERIC_NO_GROUNDING_SOURCE,
    _DECISION_TYPE,
    extract_fractions,
    fraction_in_source,
)

_SID = "dart:slug#blk_0"


def _make_block(
    *,
    block_id: str = "page_01#explanation_0",
    block_type: str = "explanation",
    content: Any = None,
    source_ids: Tuple[str, ...] = (_SID,),
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content if content is not None else "<p>placeholder.</p>",
        source_ids=source_ids,
        source_references=tuple({"sourceId": sid} for sid in source_ids),
    )


class _CaptureSpy:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------- #
# Unit-level extraction / containment
# --------------------------------------------------------------------- #


def test_extract_fractions_latex_and_inline() -> None:
    fr = extract_fractions(r"simplifying \(-\frac{40}{88}\) = -\frac{5}{11}")
    assert ("40", "88") in fr and ("5", "11") in fr
    fr2 = extract_fractions("inline -6/8 = -3/4 here")
    assert ("6", "8") in fr2 and ("3", "4") in fr2


def test_ocr_tolerant_containment() -> None:
    # The real chunk stores -55/84 as the OCR digit-group pair "− 55 84".
    src = "Are there any common factors? No. − 55 84 Table 1.21"
    assert fraction_in_source(("55", "84"), src) is True
    # 40/88 appears nowhere as an adjacent pair (the fabrication).
    assert fraction_in_source(("40", "88"), src) is False
    # Digit-boundary guard: 5/8 must not match inside 55/84.
    assert fraction_in_source(("5", "8"), src) is False


# --------------------------------------------------------------------- #
# Test 1: Keystone — fabricated worked-example INPUT flags
# --------------------------------------------------------------------- #


def test_fabricated_input_flags() -> None:
    # Source has the grounded fractions but NOT 40/88.
    source = "Worked: − 11 12 · 5 7 = − 55 84. Also − 6 8 = − 3 4."
    block = _make_block(
        content=(
            r"<p>simplifying \(-\frac{40}{88}\) = -\frac{5}{11}\) by dividing "
            r"out 8. Also \(-\frac{6}{8} = -\frac{3}{4}\).</p>"
        ),
    )
    res = NumericLiteralGroundingValidator().validate(
        {"blocks": [block], "source_chunks": {_SID: source}}
    )
    codes = [i.code for i in res.issues]
    assert _CODE_NUMERIC_FABRICATION in codes
    # Not shadow → critical severity → passed flips False, action regenerate.
    assert res.passed is False
    assert res.action == "regenerate"
    # The message names the absent fraction.
    fab = next(i for i in res.issues if i.code == _CODE_NUMERIC_FABRICATION)
    assert "40/88" in fab.message


# --------------------------------------------------------------------- #
# Test 2: OCR-flattened grounded block PASSES
# --------------------------------------------------------------------- #


def test_ocr_flattened_grounded_passes() -> None:
    source = "Reduce: Are there common factors? No. − 55 84 Table 1.21"
    block = _make_block(
        block_type="concept",
        content=r"<p>The product is \(-\frac{55}{84}\).</p>",
    )
    res = NumericLiteralGroundingValidator().validate(
        {"blocks": [block], "source_chunks": {_SID: source}}
    )
    assert res.passed is True
    assert not [i for i in res.issues if i.code == _CODE_NUMERIC_FABRICATION]


# --------------------------------------------------------------------- #
# Test 3: computed-result-not-in-source behaves per the measured rule
# --------------------------------------------------------------------- #


def test_computed_result_absent_under_default_rule_flags() -> None:
    """Default rule is count-of-absent-fractions >= 1. A grounded INPUT plus a
    source-absent RESULT flags under the default. Raising
    ``min_absent_fractions`` (the calibration knob) relaxes it — documents the
    measured rule + the ratio-floor escape hatch."""
    # Source has the inputs 11/12 and 5/7 but NOT the result 55/84.
    source = "Multiply − 11 12 by 5 7."
    block = _make_block(
        block_type="example",
        content=r"<p>\(-\frac{11}{12} \cdot \frac{5}{7} = -\frac{55}{84}\).</p>",
    )
    inp = {"blocks": [block], "source_chunks": {_SID: source}}
    flagged = NumericLiteralGroundingValidator().validate(dict(inp))
    assert _CODE_NUMERIC_FABRICATION in [i.code for i in flagged.issues]
    # Relax the threshold to 2 → a single source-absent result no longer flags.
    relaxed = NumericLiteralGroundingValidator().validate(
        {**inp, "min_absent_fractions": 2}
    )
    assert _CODE_NUMERIC_FABRICATION not in [i.code for i in relaxed.issues]
    assert relaxed.passed is True


# --------------------------------------------------------------------- #
# Test 4: graceful-degrade — no cited source
# --------------------------------------------------------------------- #


def test_no_grounding_source_warns_and_passes() -> None:
    block = _make_block(content=r"<p>\(-\frac{40}{88}\)</p>")
    res = NumericLiteralGroundingValidator().validate(
        {"blocks": [block], "source_chunks": {}}
    )
    assert res.passed is True
    codes = [i.code for i in res.issues]
    assert _CODE_NUMERIC_NO_GROUNDING_SOURCE in codes
    assert all(i.severity == "warning" for i in res.issues)


# --------------------------------------------------------------------- #
# Test 5: shadow computes without gating
# --------------------------------------------------------------------- #


def test_shadow_computes_without_gating() -> None:
    source = "grounded text containing no fabricated fraction pair"
    block = _make_block(content=r"<p>\(-\frac{40}{88}\)</p>")
    res = NumericLiteralGroundingValidator().validate(
        {"blocks": [block], "source_chunks": {_SID: source}, "shadow": True}
    )
    # Issue still computed (fabrication detected) but warning-severity.
    fab = [i for i in res.issues if i.code == _CODE_NUMERIC_FABRICATION]
    assert fab and fab[0].severity == "warning"
    # passed stays True and action is NOT regenerate during shadow.
    assert res.passed is True
    assert res.action != "regenerate"


# --------------------------------------------------------------------- #
# Test 6: decision-capture fires with dynamic signals
# --------------------------------------------------------------------- #


def test_decision_capture_fires() -> None:
    source = "grounded with no fab pair"
    block = _make_block(content=r"<p>\(-\frac{40}{88}\)</p>")
    capture = _CaptureSpy()
    NumericLiteralGroundingValidator().validate(
        {
            "blocks": [block],
            "source_chunks": {_SID: source},
            "decision_capture": capture,
            "shadow": True,
        }
    )
    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision_type"] == _DECISION_TYPE
    rationale = ev["rationale"]
    # Dynamic per-block signals present + >= 20 chars.
    assert len(rationale) >= 20
    assert block.block_id in rationale
    assert "40/88" in rationale
    assert "source_absent_ratio" in rationale


# --------------------------------------------------------------------- #
# Test 7: skip set
# --------------------------------------------------------------------- #


def test_skip_set_skips_assessment_and_objective() -> None:
    source = "no fab pair here"
    blocks = [
        _make_block(
            block_id="p#assessment_0",
            block_type="assessment_item",
            content=r"<p>\(-\frac{40}{88}\)</p>",
        ),
        _make_block(
            block_id="p#objective_0",
            block_type="objective",
            content=r"<p>\(-\frac{40}{88}\)</p>",
        ),
    ]
    res = NumericLiteralGroundingValidator().validate(
        {"blocks": blocks, "source_chunks": {_SID: source}}
    )
    # Both skipped silently → no fabrication issues, no events, audited == 0.
    assert not [i for i in res.issues if i.code == _CODE_NUMERIC_FABRICATION]
    assert res.passed is True
    assert res.score == 1.0


# --------------------------------------------------------------------- #
# Test 8: no fraction literals → no-op pass
# --------------------------------------------------------------------- #


def test_no_fractions_noop_pass() -> None:
    capture = _CaptureSpy()
    block = _make_block(
        block_type="concept",
        content="<p>Photosynthesis converts light into chemical energy.</p>",
    )
    res = NumericLiteralGroundingValidator().validate(
        {
            "blocks": [block],
            "source_chunks": {_SID: "any source"},
            "decision_capture": capture,
        }
    )
    assert res.passed is True
    assert not res.issues
    # Non-numeric blocks are skip-silent: no event emitted.
    assert capture.events == []


# --------------------------------------------------------------------- #
# Test 9: real-data regression (fixture corpus present)
# --------------------------------------------------------------------- #

_ALG_BLOCKS = _REPO_ROOT / "inputs" / "contentgen" / "alg_blocks.json"


def _fixture_corpus_slugs() -> set:
    """Course slugs the ``alg_blocks.json`` fixture was measured against.

    Derived from the fixture's own ``data-cf-source-ids`` (e.g.
    ``<course_slug>_chunk_00043``) — the identity comes from the
    fixture DATA, not a hardcoded slug in test code — so the regression binds
    to the SAME corpus it was calibrated against rather than whatever course
    happens to sort first on disk. The documented flagged-set (exactly the two
    -40/88 fabrication blocks) is only reproducible against that corpus; any
    other course's text grounds/flags different literals.
    """
    import re

    if not _ALG_BLOCKS.exists():
        return set()
    try:
        data = json.loads(_ALG_BLOCKS.read_text())
    except (OSError, ValueError):
        return set()
    slugs: set = set()
    for block in data if isinstance(data, list) else []:
        html = block.get("rewrite_html", "") if isinstance(block, dict) else ""
        for group in re.findall(r'data-cf-source-ids="([^"]+)"', html):
            for tok in group.split():
                stem = re.sub(r"_chunk_.*$", "", tok.strip().rstrip(","))
                if stem:
                    slugs.add(stem.replace("_", "-"))
    return slugs


def _discover_chunks_jsonl() -> Optional[Path]:
    """The ``dart_chunks/chunks.jsonl`` of the corpus the fixture references.

    Resolves the LibV2 root via the ``ED4ALL_LIBV2_ROOT`` convention
    (``lib.paths.libv2_path``) and binds specifically to the course named in
    the fixture's ``data-cf-source-ids`` (slug-free — discovered from fixture
    data). Returns ``None`` when that calibrated corpus is absent so the
    regression SKIPS rather than mis-firing against an unrelated course.
    """
    from lib.paths import libv2_path

    courses_root = libv2_path() / "courses"
    if not courses_root.is_dir():
        return None
    for slug in sorted(_fixture_corpus_slugs()):
        chunks = courses_root / slug / "dart_chunks" / "chunks.jsonl"
        if chunks.is_file():
            return chunks
    return None


_ALG_CHUNKS = _discover_chunks_jsonl()


@pytest.mark.skipif(
    not (_ALG_BLOCKS.exists() and _ALG_CHUNKS is not None and _ALG_CHUNKS.exists()),
    reason="no LibV2 course dart_chunks/chunks.jsonl + alg_blocks.json fixture corpus present",
)
def test_real_corpus_separates_fabrication_from_grounded() -> None:
    """The two REAL 7B fabrication blocks (explanation + misconception, both
    carrying -40/88) flag against the real corpus; the grounded blocks pass.
    Mirrors the module's documented real-data measurement."""
    blocks_raw = json.loads(_ALG_BLOCKS.read_text())
    corpus = " ".join(
        json.loads(line)["text"] for line in _ALG_CHUNKS.read_text().splitlines() if line.strip()
    )
    sid = "dart:corpus#all"
    blocks = []
    for i, b in enumerate(blocks_raw):
        blocks.append(
            _make_block(
                block_id=f"page#{b['block_type']}_{i}",
                block_type=b["block_type"],
                content=b["rewrite_html"],
                source_ids=(sid,),
            )
        )
    res = NumericLiteralGroundingValidator().validate(
        {"blocks": blocks, "source_chunks": {sid: corpus}, "shadow": True}
    )
    flagged_locs = {
        i.location for i in res.issues if i.code == _CODE_NUMERIC_FABRICATION
    }
    # Exactly the two fabrication blocks flag.
    assert flagged_locs == {"page#explanation_1", "page#misconception_3"}
