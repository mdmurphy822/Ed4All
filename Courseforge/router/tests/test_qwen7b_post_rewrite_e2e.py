"""End-to-end regression test for the post-rewrite validator chain.

Closes plan §5 of ``plans/qwen7b-courseforge-fixes-2026-05-followup.md``.
Three assertions land here, exercising the two new post-rewrite gates
(:class:`RewriteHtmlShapeValidator` and
:class:`RewriteSourceGroundingValidator`) end-to-end against the
recorded Qwen-2.5-7B-Q4 surfaces fixture at
``runtime/qwen_test/surfaces.json``.

Plan §5 contracts (verbatim):

1. **Replay JSON-wrapped emit → reject.** Load
   ``runtime/qwen_test/surfaces.json::rewrite[2].html`` (the recorded
   ``{"div": {...}}`` content). Construct a Block with that as
   ``content``, run through the post-rewrite validator chain (incl. the
   new ``rewrite_html_shape`` gate). Assert the gate fires
   ``action="regenerate"`` with ``code="REWRITE_NOT_HTML_BODY_FRAGMENT"``.

   Note on observed code: the implemented validator subdivides the
   leading-brace failure into ``REWRITE_JSON_WRAPPED_HTML`` (when the
   content round-trips through ``json.loads`` to a dict / list) vs.
   ``REWRITE_NOT_HTML_BODY_FRAGMENT`` (any other leading non-``<``).
   The plan-canonical regression input — a fully-formed
   ``{"div": {...}}`` payload — round-trips and therefore fires the
   ``REWRITE_JSON_WRAPPED_HTML`` code in practice. This test accepts
   either code (both critical, both ``action="regenerate"``) so the
   contract holds regardless of which path the validator's
   leading-brace branch took.

   Fixture caveat: the on-disk fixture's current ``rewrite[2].html``
   field captures the post-hardening clean emit, not the original
   regression. The plan §2 records the original verbatim
   (``{"div": {"class": "assessment-item", "content": "<p>...</p>"}}``);
   this test uses that recorded literal as the regression input.

2. **Paraphrase-of-source rewrite → accept.** Hand-author a Block
   whose content is bare HTML, parses clean, has all required
   ``data-cf-*`` attrs, and whose prose is a paraphrase of a known
   source chunk. Assert every gate ``passed=True, action=None``.

3. **Hallucinated-content rewrite → reject via grounding gate.**
   Hand-author a Block whose content is bare valid HTML with the
   required attrs but whose prose is fabricated (e.g. talks about
   blockchain when the source is RDF triples). Assert the
   ``rewrite_source_grounding`` gate fires ``action="regenerate"``
   with ``code="REWRITE_SENTENCE_GROUNDING_LOW"``.

Mirrors the import + fixture-loader conventions of the predecessor
test :mod:`Courseforge.router.tests.test_capability_tier`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.rewrite_html_shape import (  # noqa: E402
    RewriteHtmlShapeValidator,
)
from lib.validators.rewrite_source_grounding import (  # noqa: E402
    RewriteSourceGroundingValidator,
)


_SURFACES_PATH = PROJECT_ROOT / "runtime" / "qwen_test" / "surfaces.json"


# --------------------------------------------------------------------- #
# Fixture loaders + helpers
# --------------------------------------------------------------------- #


def _load_surfaces() -> Dict[str, Any]:
    """Load the recorded Qwen-7B-Q4 surfaces or skip the suite.

    Mirrors the predecessor pattern at
    ``test_capability_tier.py::_load_surfaces``.
    """
    if not _SURFACES_PATH.exists():
        pytest.skip(
            f"runtime/qwen_test/surfaces.json absent at {_SURFACES_PATH}"
        )
    return json.loads(_SURFACES_PATH.read_text())


# Plan §2 records the original JSON-wrapped regression verbatim. The
# on-disk fixture has since been refreshed with the post-hardening
# clean emit; the historical regression payload is reconstructed here
# so the gate-replay assertion holds.
_REGRESSION_JSON_WRAPPED_HTML = json.dumps({
    "div": {
        "class": "assessment-item",
        "content": (
            "<p>What are the three components of an RDF triple?</p>"
            "<ol>"
            "<li>subject, predicate, object</li>"
            "<li>subject, object, predicate</li>"
            "<li>predicate, subject, object</li>"
            "</ol>"
        ),
    }
})


# Plan §5 case 2 grounding source (paraphrase target).
_RDF_SOURCE_CHUNK = (
    "An RDF triple is composed of three components: a subject identifying "
    "the resource being described, a predicate naming the property or "
    "relationship, and an object holding the value or related resource. "
    "Together these three components form the smallest unit of meaning in "
    "the Resource Description Framework graph model. Subjects and predicates "
    "are always IRIs while objects can be IRIs, blank nodes, or literals."
)

# Plan §5 case 2 paraphrase-of-source rewrite content. Bare HTML body
# fragment, all required attrs for ``concept`` block_type:
# ``data-cf-block-id``, ``data-cf-content-type``, ``data-cf-key-terms``.
_PARAPHRASE_HTML = (
    '<section data-cf-block-id="page_week_1#concept_rdf_triple_0" '
    'data-cf-content-type="concept" '
    'data-cf-key-terms="rdf,triple,subject,predicate,object">'
    "<h2>Components of an RDF Triple</h2>"
    "<p>The Resource Description Framework triple is built from three "
    "components: the subject identifying which resource the statement "
    "describes, the predicate naming the relationship or property, and "
    "the object holding the related resource or literal value. "
    "Together these three components encode the smallest meaningful "
    "statement in any RDF graph. The subject and predicate must always "
    "be IRIs, while the object can be an IRI, a blank node, or a literal "
    "value.</p>"
    "</section>"
)

# Plan §5 case 3 hallucinated rewrite content. Bare HTML body fragment,
# all required attrs for ``concept`` block_type, but prose is about
# blockchain rather than RDF triples.
_HALLUCINATED_HTML = (
    '<section data-cf-block-id="page_week_1#concept_rdf_triple_0" '
    'data-cf-content-type="concept" '
    'data-cf-key-terms="rdf,triple,subject,predicate,object">'
    "<h2>Components of an RDF Triple</h2>"
    "<p>Blockchain consensus mechanisms validate distributed ledger "
    "entries among participating network nodes everywhere across the "
    "globe. Smart contracts execute deterministic state transitions on "
    "virtual machines without any trusted central authority involvement. "
    "Cryptocurrency wallets manage private keys for transaction signing "
    "across multiple blockchain networks for individual users.</p>"
    "</section>"
)


def _make_assessment_item_block(content: str) -> Block:
    """Build the assessment_item Block referenced by plan §5 case 1."""
    return Block(
        block_id="page_week_1#assessment_item_rdf_triple_0",
        block_type="assessment_item",
        page_id="page_week_1",
        sequence=0,
        content=content,
        objective_ids=("CO-01",),
        bloom_level="remember",
    )


def _make_concept_block(content: str) -> Block:
    """Build the concept Block used by plan §5 cases 2 and 3.

    The grounding gate skips ``assessment_item`` per its content-type
    skip list, so the paraphrase / hallucinated assertions exercise
    the ``concept`` block_type — which carries all three required
    ``data-cf-*`` attributes the shape gate audits.
    """
    return Block(
        block_id="page_week_1#concept_rdf_triple_0",
        block_type="concept",
        page_id="page_week_1",
        sequence=0,
        content=content,
        content_type_label="definition",
        source_ids=("dart:rdf-primer#blk_3",),
    )


# --------------------------------------------------------------------- #
# Stub embedder (deterministic prefix-match → high cosine; orthogonal
# default vector for everything else). Mirrors the
# ``test_rewrite_source_grounding._StubEmbedder`` pattern verbatim so
# we don't need to load sentence-transformers in CI.
# --------------------------------------------------------------------- #


_GROUNDED_VECTOR = [1.0, 0.0, 0.0, 0.0]
_HALLUCINATED_VECTOR = [0.0, 1.0, 0.0, 0.0]
_DEFAULT_VECTOR = [0.0, 0.0, 0.0, 1.0]


class _StubEmbedder:
    """Longest-prefix-match deterministic embedder."""

    def __init__(self, vector_map: Dict[str, List[float]]) -> None:
        self.vector_map = vector_map
        self.calls: List[str] = []

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        self.calls.append(text)
        match_key = ""
        for key in self.vector_map:
            if text.startswith(key) and len(key) > len(match_key):
                match_key = key
        if match_key:
            return self.vector_map[match_key]
        return list(_DEFAULT_VECTOR)


def _grounded_embedder() -> _StubEmbedder:
    """Stub mapping the paraphrase + source sentences to the same vector."""
    return _StubEmbedder({
        # Source chunk sentence prefixes.
        "An RDF triple": _GROUNDED_VECTOR,
        "Together these three": _GROUNDED_VECTOR,
        "Subjects and predicates": _GROUNDED_VECTOR,
        # Paraphrase rewrite sentence prefixes.
        "The Resource Description": _GROUNDED_VECTOR,
        "The subject and predicate": _GROUNDED_VECTOR,
    })


def _hallucinated_embedder() -> _StubEmbedder:
    """Stub mapping source to grounded; hallucinated prose to orthogonal."""
    return _StubEmbedder({
        # Source vectors.
        "An RDF triple": _GROUNDED_VECTOR,
        "Together these three": _GROUNDED_VECTOR,
        "Subjects and predicates": _GROUNDED_VECTOR,
        # Hallucinated rewrite sentences map to an orthogonal vector.
        "Blockchain consensus": _HALLUCINATED_VECTOR,
        "Smart contracts": _HALLUCINATED_VECTOR,
        "Cryptocurrency wallets": _HALLUCINATED_VECTOR,
    })


# --------------------------------------------------------------------- #
# Plan §5 case 1: replay JSON-wrapped emit → reject
# --------------------------------------------------------------------- #


def test_replay_json_wrapped_rewrite_emit_rejected_by_html_shape_gate() -> None:
    """Plan §5 case 1.

    Replay the recorded JSON-wrapped ``{"div": {...}}`` rewrite emit
    (originally captured at ``surfaces.json::rewrite[2].html`` before
    the post-followup hardening commits refreshed the fixture) through
    the post-rewrite chain. The new ``rewrite_html_shape`` gate must
    fire ``action="regenerate"`` with a critical-severity code
    flagging the non-HTML body fragment.

    Plan §5 names ``REWRITE_NOT_HTML_BODY_FRAGMENT`` as the expected
    code; the implemented validator branches the leading-brace failure
    into either ``REWRITE_JSON_WRAPPED_HTML`` (round-trips to a dict /
    list — the exact regression class plan §2 calls out) or
    ``REWRITE_NOT_HTML_BODY_FRAGMENT`` (any other leading non-``<``).
    Both are critical with ``action="regenerate"``; this test accepts
    either to keep the test contract aligned with the implemented
    behaviour without softening the gate's fail-closed posture.
    """
    # Touch the fixture so the suite skips when the fixture is absent
    # (mirrors test_capability_tier conventions). Fixture content is
    # not consumed directly here; the regression input is reconstructed
    # from the plan §2 verbatim record.
    _load_surfaces()

    block = _make_assessment_item_block(_REGRESSION_JSON_WRAPPED_HTML)

    result = RewriteHtmlShapeValidator().validate({"blocks": [block]})

    assert result.passed is False
    assert result.action == "regenerate"
    critical_codes = [
        issue.code for issue in result.issues if issue.severity == "critical"
    ]
    # Plan §5 names REWRITE_NOT_HTML_BODY_FRAGMENT; the validator
    # subdivides into REWRITE_JSON_WRAPPED_HTML for the dict-roundtrip
    # case. Either is the right fail-closed signal for this regression.
    assert any(
        code in {"REWRITE_NOT_HTML_BODY_FRAGMENT", "REWRITE_JSON_WRAPPED_HTML"}
        for code in critical_codes
    ), (
        f"expected REWRITE_NOT_HTML_BODY_FRAGMENT or "
        f"REWRITE_JSON_WRAPPED_HTML; got {critical_codes!r}"
    )


# --------------------------------------------------------------------- #
# Plan §5 case 2: paraphrase-of-source rewrite → accept
# --------------------------------------------------------------------- #


def test_paraphrase_of_source_rewrite_passes_post_rewrite_chain() -> None:
    """Plan §5 case 2.

    Hand-author a concept Block whose content is bare HTML with all
    required ``data-cf-*`` attrs and whose prose paraphrases a known
    source chunk. The post-rewrite chain (shape + grounding) must pass
    every gate with ``passed=True, action=None``.
    """
    block = _make_concept_block(_PARAPHRASE_HTML)
    source_chunks = {"dart:rdf-primer#blk_3": _RDF_SOURCE_CHUNK}

    shape_result = RewriteHtmlShapeValidator().validate({"blocks": [block]})
    grounding_result = RewriteSourceGroundingValidator(
        embedder=_grounded_embedder(),
    ).validate({
        "blocks": [block],
        "source_chunks": source_chunks,
    })

    assert shape_result.passed is True
    assert shape_result.action is None
    assert all(
        issue.severity != "critical" for issue in shape_result.issues
    )

    assert grounding_result.passed is True
    assert grounding_result.action is None
    assert all(
        issue.severity != "critical" for issue in grounding_result.issues
    )


# --------------------------------------------------------------------- #
# Plan §5 case 3: hallucinated-content rewrite → reject via grounding
# --------------------------------------------------------------------- #


def test_hallucinated_rewrite_rejected_by_source_grounding_gate() -> None:
    """Plan §5 case 3.

    Hand-author a concept Block whose content is bare valid HTML with
    all required ``data-cf-*`` attrs but whose prose is fabricated
    (blockchain content where the source is RDF triples). The
    ``rewrite_source_grounding`` gate must fire ``action="regenerate"``
    with critical code ``REWRITE_SENTENCE_GROUNDING_LOW``.

    The shape gate must NOT fail — the input is structurally valid
    HTML; the only issue is semantic drift, which is exactly what
    the grounding gate exists to catch.
    """
    block = _make_concept_block(_HALLUCINATED_HTML)
    source_chunks = {"dart:rdf-primer#blk_3": _RDF_SOURCE_CHUNK}

    shape_result = RewriteHtmlShapeValidator().validate({"blocks": [block]})
    grounding_result = RewriteSourceGroundingValidator(
        embedder=_hallucinated_embedder(),
    ).validate({
        "blocks": [block],
        "source_chunks": source_chunks,
    })

    # Shape gate accepts the structurally valid input.
    assert shape_result.passed is True
    assert shape_result.action is None

    # Grounding gate rejects the fabricated prose.
    assert grounding_result.passed is False
    assert grounding_result.action == "regenerate"
    critical_codes = [
        issue.code
        for issue in grounding_result.issues
        if issue.severity == "critical"
    ]
    assert "REWRITE_SENTENCE_GROUNDING_LOW" in critical_codes
