"""GPT Feedback v2 — Wave 5 end-of-wave Trainforge ingestion contract gate.

Authored 2026-05-07 against the closing 6-test gate enumerated in
``plans/wave5-trainforge-ingestion-mirror-2026-05.md`` § 4.

Predecessors landed:

* W5.A — :class:`Trainforge.parsers.html_content_parser.ContentSection`
  carries ``key_claims`` + ``objective_alignment`` (commit ``557a2ff``).
* W5.B — :meth:`CourseProcessor._extract_section_metadata` returns a
  5-tuple with ``section_metadata_extras`` carrying the new audit
  arrays; :meth:`CourseProcessor._create_chunk` stamps them onto the
  chunk dict.
* W5.C — ``schemas/knowledge/chunk_v4.schema.json`` admits the two new
  optional fields under ``additionalProperties: true`` (commit
  ``e7161e0``).
* W5.D (this gate's sibling) — pair-side per-claim attribution: when
  the cited chunk carries structured ``key_claims=[{claim,
  source_chunk_ids[]}]``, :meth:`PairClaimSupportValidator.validate_pair`
  fans out NLI per-claim against the cited chunks' text via the
  caller-injected ``chunk_id_to_text_map`` kwarg.
* W5.E — :class:`AssessmentObjectiveAlignmentValidator` accepts an
  optional ``synthesized_objectives_path`` and unions every flattened
  objective's ``id`` into the resolution surface.
* W5.F — :func:`Trainforge.chunker.merge_small_sections` returns a
  :class:`MergedSectionResult` dataclass carrying ``merged_key_claims``
  (concat) + ``merged_objective_alignment`` (first-seen-wins dedup).

This file is a SELF-CONTAINED contract test mirroring the Wave 1.5 /
1.6 / 1.7 / Wave 4 TIGHT / Wave 4 MEDIUM predecessor gates
(``test_wave15_per_claim_attribution_contract.py``,
``test_wave16_per_objective_attribution_contract.py``,
``test_wave17_block_against_objective_contract.py``,
``test_wave4_pair_validation_contract.py``,
``test_wave4_medium_pair_objective_delivery_contract.py``). Every
fixture is inlined locally so a future rename of a per-worker test
file cannot silently disable this wave-end gate.

Tests:

* ``test_html_parser_carries_key_claims_from_jsonld_blocks`` — the
  W5.A projection contract: a JSON-LD ``blocks[]`` carrying
  ``keyClaims`` + ``objectiveAlignment`` projects through to the
  emitted ``ContentSection`` dataclass via
  :meth:`HTMLContentParser._content_sections_from_blocks`.
* ``test_extract_section_metadata_returns_5_tuple_with_extras`` — the
  W5.B return-shape contract: the 4th tuple element
  (``section_metadata_extras``) carries ``key_claims`` +
  ``objective_alignment`` populated from the matched JSON-LD block.
* ``test_create_chunk_stamps_key_claims_and_objective_alignment`` —
  the W5.B end-to-end stamping contract on
  :meth:`CourseProcessor._create_chunk`. Positive case stamps both
  fields; negative case (legacy item without ``blocks[]``) emits a
  chunk that DOES NOT carry the keys at all (not even as ``[]``) so
  legacy back-compat holds.
* ``test_chunk_v4_strict_validation_admits_new_optional_fields`` —
  the W5.C schema admission contract under
  ``TRAINFORGE_VALIDATE_CHUNKS=true``: legacy chunk passes,
  structured chunk passes, malformed chunk (int claim) fails with a
  clear schema error.
* ``test_pair_claim_support_uses_per_claim_attribution_when_chunk_carries_key_claims``
  — the W5.D per-claim attribution contract on
  :meth:`PairClaimSupportValidator.validate_pair`. With structured
  ``key_claims``, NLI fan-out is scoped per-claim against the
  attributed chunks; the audit arm carries
  ``per_claim_support[*].source_chunk_ids`` and the decision-capture
  rationale interpolates ``structured_attribution_used=True`` +
  ``per_claim_match_count=2``. Legacy fallback (chunk without
  structured ``key_claims``) preserves whole-chunk-text scoring.
* ``test_assessment_objective_alignment_resolves_via_synthesized_objectives_source_refs``
  — the W5.E union surface on
  :class:`AssessmentObjectiveAlignmentValidator`. With
  ``synthesized_objectives_path`` provided, an assessment objective
  resolves via the synthesized objectives' ``id`` set even when every
  chunk's ``learning_outcome_refs`` is empty. Without the path,
  back-compat: gate fails with ``ASSESSMENT_ALIGNMENT_NO_CHUNKS``.

Optional Test 7 (recommended, not part of the 6-test contract):

* ``test_merge_small_sections_unions_audit_fields_across_merge`` —
  the W5.F chunker merge-path provenance contract. Two adjacent
  small ContentSections each carry their own ``key_claims`` +
  ``objective_alignment``; the emitted
  :class:`MergedSectionResult` carries the concatenated
  ``merged_key_claims`` and the first-seen-wins
  ``merged_objective_alignment`` so per-claim audit signals survive
  the merge boundary.

Drift notes vs plan §4:

* Plan §4 spec for Test 1 names ``concept`` + ``assessment_item``
  block types; those types are NOT in
  :pyattr:`HTMLContentParser._SECTION_BLOCK_TYPES` (the canonical
  8-value frozenset is ``{explanation, example, procedure,
  comparison, definition, overview, summary, exercise}``). Using
  unadmitted block types would silently drop the projected sections
  and trivially "pass" the assertion against an empty list. This file
  uses ``definition`` (carrying ``keyClaims``) + ``explanation``
  (carrying ``objectiveAlignment``) so the projection actually
  exercises the carry contract. The shape under test is identical —
  the W5.A code path treats every admitted block type uniformly.
* Test 5 stub-NLI shape mirrors the Wave 4 TIGHT precedent: a
  callable ``score_fn(premise, hypothesis) ->
  (entailment, neutral, contradiction)`` that lets the test shape NLI
  verdicts per-pair without instantiating the real DeBERTa-v3 model.
* Stub ``DecisionCapture`` mirrors the Wave 4 TIGHT
  ``_RecordingCapture`` pattern (attribute name preserved for
  cross-wave grep symmetry).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Helpers — stub NLI classifier + DecisionCapture stub
# --------------------------------------------------------------------------- #


class _StubNliScore:
    """Minimal NliScore-shaped object exposing the three probability
    attributes the W4.A / W5.D consumer reads.

    Inlined so the test file doesn't import the production
    :class:`NliScore` (which would couple the gate to that class's
    evolution). Mirrors the Wave 4 TIGHT precedent in
    ``test_wave4_pair_validation_contract.py``.
    """

    def __init__(
        self,
        *,
        entailment: float,
        neutral: float,
        contradiction: float,
    ) -> None:
        self.entailment = float(entailment)
        self.neutral = float(neutral)
        self.contradiction = float(contradiction)


class _StubNliClassifier:
    """Stub :class:`NliClassifier` that returns canned per-pair scores.

    Constructor accepts a callable ``score_fn(premise, hypothesis) ->
    (entailment, neutral, contradiction)`` so every test can shape the
    NLI verdict without instantiating the real DeBERTa-v3 model.

    The ``score_batch`` surface records every (premise, hypothesis)
    pair it received in :pyattr:`calls` so per-test assertions can
    verify the W5.D per-claim scoping (premise text == per-claim
    chunk text, NOT whole-chunk text).
    """

    def __init__(self, *, score_fn) -> None:
        self._score_fn = score_fn
        self.calls: List[List[Tuple[str, str]]] = []

    def score_batch(self, *, pairs, batch_size=None):
        self.calls.append(list(pairs))
        out: List[_StubNliScore] = []
        for premise, hypothesis in pairs:
            ent, neu, con = self._score_fn(premise, hypothesis)
            out.append(
                _StubNliScore(
                    entailment=ent, neutral=neu, contradiction=con,
                )
            )
        return out

    def score_pair(self, *, premise, hypothesis):
        return self.score_batch(pairs=[(premise, hypothesis)])[0]


class _RecordingCapture:
    """Minimal capture stub recording every ``log_decision`` payload.

    Inlined so a rename of any per-worker capture helper can't silently
    break this gate. Mirrors the
    ``test_wave4_pair_validation_contract.py::_RecordingCapture``
    pattern (attribute name preserved for cross-wave grep symmetry).
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


def _structured_key_claim(
    *, claim: str, source_chunk_ids: List[str]
) -> Dict[str, Any]:
    """Build a single structured key_claims[] entry. Wave 1.5 / W5.A
    canonical shape: ``{claim: str, source_chunk_ids: List[str]}``."""
    return {"claim": claim, "source_chunk_ids": list(source_chunk_ids)}


def _objective_alignment_entry(
    *,
    objective_id: str,
    declared_bloom: str = "apply",
    observed_bloom: str = "apply",
    statement_entailment_score: float = 0.85,
    action_verb_present: bool = True,
    status: str = "delivered",
) -> Dict[str, Any]:
    """Build a single objective_alignment[] entry. Wave 1.7 / W5.A
    canonical shape mirrors the Courseforge JSON-LD
    ``$defs.ObjectiveAlignment`` definition (objective_id +
    declared_bloom + observed_bloom + statement_entailment_score +
    action_verb_present + status)."""
    return {
        "objective_id": objective_id,
        "declared_bloom": declared_bloom,
        "observed_bloom": observed_bloom,
        "statement_entailment_score": statement_entailment_score,
        "action_verb_present": action_verb_present,
        "status": status,
    }


def _bare_processor():
    """Minimal :class:`CourseProcessor` instance bypassing ``__init__``.

    Mirrors the established
    ``Trainforge/tests/test_provenance.py::_chunks_have_split_disjoint_spans``
    + ``test_merged_heading_metadata_lookup.py::_bare_processor`` pattern.
    Test sets up only the attributes ``_create_chunk`` reads (nothing
    more, so additional fields land in the chunk dict only via the
    code paths under test).
    """
    from collections import defaultdict
    from Trainforge.process_course import CourseProcessor

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TEST"
    proc.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    proc.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE
    proc.TARGET_CHUNK_SIZE = CourseProcessor.TARGET_CHUNK_SIZE
    proc._all_concept_tags = set()
    proc._lo_parent_map = {}
    proc.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
    }
    return proc


# --------------------------------------------------------------------------- #
# Test 1 — Parser projects keyClaims / objectiveAlignment onto ContentSection
# --------------------------------------------------------------------------- #


def test_html_parser_carries_key_claims_from_jsonld_blocks() -> None:
    """Plan §4 Test 1 — :meth:`HTMLContentParser._content_sections_from_blocks`
    MUST project every JSON-LD block's ``keyClaims`` into
    ``ContentSection.key_claims`` and every block's ``objectiveAlignment``
    into ``ContentSection.objective_alignment``.

    Drift note: plan §4 spec names ``concept`` + ``assessment_item``
    block types; this file uses ``definition`` (admitted, carries
    ``keyClaims``) + ``explanation`` (admitted, carries
    ``objectiveAlignment``) because ``concept`` /
    ``assessment_item`` are NOT in the admitted
    :pyattr:`_SECTION_BLOCK_TYPES` frozenset. The W5.A code path
    treats every admitted block type uniformly so the contract under
    test is identical.
    """
    from Trainforge.parsers.html_content_parser import (
        ContentSection,
        HTMLContentParser,
    )

    key_claim = _structured_key_claim(
        claim="X is Y",
        source_chunk_ids=["dart:rdf-primer-ch3#sec-2-rdf-graph"],
    )
    objective_entry = _objective_alignment_entry(
        objective_id="TO-05",
        declared_bloom="apply",
        observed_bloom="apply",
        statement_entailment_score=0.85,
        action_verb_present=True,
        status="delivered",
    )

    blocks = [
        {
            "blockId": "page#definition_x_is_y_0",
            "blockType": "definition",
            "contentTypeLabel": "definition",
            "keyTerms": ["x"],
            "keyClaims": [key_claim],
            # No objectiveAlignment — this block carries only the
            # claim-attribution audit array.
        },
        {
            "blockId": "page#explanation_to05_1",
            "blockType": "explanation",
            "contentTypeLabel": "explanation",
            "keyTerms": ["to-05"],
            "objectiveAlignment": [objective_entry],
            # No keyClaims — this block carries only the per-objective
            # tri-axis audit array.
        },
    ]

    parser = HTMLContentParser()
    sections = parser._content_sections_from_blocks(blocks)

    assert len(sections) == 2, (
        f"expected 2 ContentSection entries (one per admitted block); "
        f"got {len(sections)}: {sections!r}"
    )

    # The first section maps the claim-bearing block.
    sec_claim, sec_obj = sections[0], sections[1]
    assert isinstance(sec_claim, ContentSection)
    assert isinstance(sec_obj, ContentSection)
    assert sec_claim.content_type == "definition", (
        f"expected first section content_type='definition'; "
        f"got {sec_claim.content_type!r}"
    )
    assert sec_claim.key_claims == [key_claim], (
        "claim-bearing block MUST project keyClaims onto "
        f"ContentSection.key_claims; got {sec_claim.key_claims!r}"
    )
    # The claim-bearing block carries no objectiveAlignment, so the
    # downstream attribute is the empty default.
    assert sec_claim.objective_alignment == [], (
        "claim-bearing block MUST NOT carry objective_alignment "
        f"(default []); got {sec_claim.objective_alignment!r}"
    )

    # The second section maps the objective-alignment-bearing block.
    assert sec_obj.content_type == "explanation", (
        f"expected second section content_type='explanation'; "
        f"got {sec_obj.content_type!r}"
    )
    assert sec_obj.objective_alignment == [objective_entry], (
        "objective-alignment-bearing block MUST project "
        "objectiveAlignment onto ContentSection.objective_alignment; "
        f"got {sec_obj.objective_alignment!r}"
    )
    # And the obj-bearing block carries no keyClaims.
    assert sec_obj.key_claims == [], (
        "objective-alignment-bearing block MUST NOT carry key_claims "
        f"(default []); got {sec_obj.key_claims!r}"
    )


# --------------------------------------------------------------------------- #
# Test 2 — _extract_section_metadata returns 5-tuple with extras dict
# --------------------------------------------------------------------------- #


def test_extract_section_metadata_returns_5_tuple_with_extras() -> None:
    """Plan §4 Test 2 — :meth:`CourseProcessor._extract_section_metadata`
    return shape MUST be a 5-tuple where the 4th element is the
    ``section_metadata_extras`` dict carrying ``key_claims`` +
    ``objective_alignment`` populated from the matched JSON-LD block.

    Pins the W5.B return-shape contract: any future refactor that
    returns a 4-tuple (dropping extras) breaks this test, so a
    silent-degradation regression on the audit-field carry can't
    slip past CI.
    """
    proc = _bare_processor()

    key_claim = _structured_key_claim(
        claim="A is B",
        source_chunk_ids=["dart:chunk_alpha"],
    )
    objective_entry = _objective_alignment_entry(
        objective_id="TO-07",
        declared_bloom="understand",
    )

    item = {
        "title": "Page Title",
        "courseforge_metadata": {
            "blocks": [
                {
                    "blockId": "page#section_a_0",
                    "blockType": "definition",
                    "heading": "Section A",
                    "contentType": "definition",
                    "keyTerms": [{"term": "a", "definition": "the letter A"}],
                    "keyClaims": [key_claim],
                    "objectiveAlignment": [objective_entry],
                }
            ],
        },
        "sections": [],
    }

    result = proc._extract_section_metadata(item, "Section A")

    # ---- Sub-clause 1 — return is a 5-tuple. ---- #
    assert isinstance(result, tuple), (
        f"expected tuple return; got {type(result).__name__}: {result!r}"
    )
    assert len(result) == 5, (
        f"expected 5-tuple (bloom_level, content_type_label, key_terms, "
        f"section_metadata_extras, trace); got {len(result)}-tuple: "
        f"{result!r}"
    )

    bloom_level, content_type_label, key_terms, extras, trace = result

    # ---- Sub-clause 2 — extras carries the audit arrays. ---- #
    assert isinstance(extras, dict), (
        f"expected 4th element to be a dict (section_metadata_extras); "
        f"got {type(extras).__name__}: {extras!r}"
    )
    assert extras.get("key_claims") == [key_claim], (
        "section_metadata_extras MUST carry key_claims populated from "
        f"the matched block; got {extras.get('key_claims')!r}"
    )
    assert extras.get("objective_alignment") == [objective_entry], (
        "section_metadata_extras MUST carry objective_alignment populated "
        f"from the matched block; got {extras.get('objective_alignment')!r}"
    )

    # ---- Sub-clause 3 — trace dict surfaces the source path. ---- #
    assert isinstance(trace, dict), (
        f"expected 5th element to be a trace dict; got {type(trace).__name__}"
    )
    # When a JSON-LD blocks[] match populates the audit field, the
    # trace value is "jsonld_blocks_match" (W5.B contract).
    assert trace.get("key_claims") == "jsonld_blocks_match", (
        "trace MUST surface 'jsonld_blocks_match' for the populated "
        f"key_claims field; got {trace.get('key_claims')!r}"
    )
    assert trace.get("objective_alignment") == "jsonld_blocks_match", (
        "trace MUST surface 'jsonld_blocks_match' for the populated "
        f"objective_alignment field; got {trace.get('objective_alignment')!r}"
    )

    # ---- Sub-clause 4 — defence-in-depth: content_type_label + key_terms
    #     are also lifted from the matched block (existing W5.B / Phase 2
    #     contract — included so a regression on either field can't slip
    #     past this test). ---- #
    assert content_type_label == "definition", (
        f"expected content_type_label='definition' from matched block; "
        f"got {content_type_label!r}"
    )
    assert key_terms and key_terms[0].get("term") == "a", (
        f"expected key_terms[0].term='a'; got {key_terms!r}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — _create_chunk stamps key_claims + objective_alignment
# --------------------------------------------------------------------------- #


def test_create_chunk_stamps_key_claims_and_objective_alignment() -> None:
    """Plan §4 Test 3 — :meth:`CourseProcessor._create_chunk` MUST stamp
    ``chunk["key_claims"]`` and ``chunk["objective_alignment"]`` on the
    emitted chunk dict when the source item's ``courseforge_metadata.blocks[]``
    carries the W5.A audit arrays.

    Negative case (back-compat): when the item DOES NOT carry
    ``blocks[]``, the chunk dict MUST NOT carry the new keys at all
    (not even as ``[]``) — legacy chunks built from pre-W5.A corpora
    stay byte-identical to pre-W5.B emit.
    """
    proc = _bare_processor()

    # Monkey-patch helpers that depend on full __init__ state. These are
    # the same patches the established test_provenance.py pattern uses,
    # extended to keep _extract_section_metadata wired to the real
    # implementation (the W5.B path is what we're exercising).
    proc._extract_concept_tags = lambda t, i: []  # type: ignore[method-assign]
    proc._determine_difficulty = lambda t, i: "foundational"  # type: ignore[method-assign]
    proc._extract_objective_refs = (  # type: ignore[method-assign]
        lambda i, **kw: []
    )

    key_claim = _structured_key_claim(
        claim="RDF triples link a subject to an object via a predicate",
        source_chunk_ids=["dart:rdf-primer-ch3#sec-2-rdf-graph"],
    )
    objective_entry = _objective_alignment_entry(
        objective_id="TO-05",
        declared_bloom="apply",
        observed_bloom="apply",
        statement_entailment_score=0.85,
        action_verb_present=True,
        status="delivered",
    )

    item_with_blocks: Dict[str, Any] = {
        "item_id": "lesson_x",
        "item_path": "lessons/x.html",
        "module_id": "week_01_overview",
        "module_title": "Week 1",
        "title": "RDF Basics",
        "resource_type": "page",
        "raw_html": "<section><p>Body text.</p></section>",
        "sections": [],
        "learning_objectives": [],
        "courseforge_metadata": {
            "blocks": [
                {
                    "blockId": "page#definition_rdf_0",
                    "blockType": "definition",
                    "heading": "RDF Triples",
                    "contentType": "definition",
                    "keyTerms": [
                        {"term": "rdf", "definition": "Resource Description Framework"}
                    ],
                    "keyClaims": [key_claim],
                    "objectiveAlignment": [objective_entry],
                }
            ],
        },
        "misconceptions": [],
    }

    chunk = proc._create_chunk(
        chunk_id="test_course_chunk_00001",
        text="RDF triples link a subject to an object via a predicate.",
        html="<p>RDF triples link a subject to an object via a predicate.</p>",
        item=item_with_blocks,
        section_heading="RDF Triples",
        chunk_type="definition",
    )

    # ---- Positive case ---- #
    assert chunk.get("key_claims") == [key_claim], (
        "chunk MUST carry key_claims populated from the matched block; "
        f"got {chunk.get('key_claims')!r}"
    )
    assert chunk.get("objective_alignment") == [objective_entry], (
        "chunk MUST carry objective_alignment populated from the matched "
        f"block; got {chunk.get('objective_alignment')!r}"
    )

    # ---- Negative case — same item shape minus blocks[] ---- #
    proc_legacy = _bare_processor()
    proc_legacy._extract_concept_tags = lambda t, i: []  # type: ignore[method-assign]
    proc_legacy._determine_difficulty = lambda t, i: "foundational"  # type: ignore[method-assign]
    proc_legacy._extract_objective_refs = (  # type: ignore[method-assign]
        lambda i, **kw: []
    )

    item_legacy: Dict[str, Any] = {
        "item_id": "lesson_y",
        "item_path": "lessons/y.html",
        "module_id": "week_01_overview",
        "module_title": "Week 1",
        "title": "Legacy Page",
        "resource_type": "page",
        "raw_html": "<section><p>Body text.</p></section>",
        "sections": [],
        "learning_objectives": [],
        # Pre-W5.A corpus — courseforge_metadata is absent / empty.
        "courseforge_metadata": None,
        "misconceptions": [],
    }
    chunk_legacy = proc_legacy._create_chunk(
        chunk_id="test_course_chunk_00002",
        text="Legacy body text without per-block audit arrays.",
        html="<p>Legacy body text without per-block audit arrays.</p>",
        item=item_legacy,
        section_heading="Legacy Heading",
        chunk_type="explanation",
    )

    # The new keys MUST be absent — not even as ``[]``. Pinning key-
    # absence (vs. falsy-value) closes the silent-degradation class
    # where a future refactor stamps an empty list as a default and
    # downstream consumers can't tell "absent" from "no audit data".
    assert "key_claims" not in chunk_legacy, (
        "legacy chunk (no blocks[]) MUST NOT carry key_claims (not "
        f"even as []); got chunk['key_claims']={chunk_legacy.get('key_claims')!r}"
    )
    assert "objective_alignment" not in chunk_legacy, (
        "legacy chunk (no blocks[]) MUST NOT carry objective_alignment "
        f"(not even as []); got "
        f"chunk['objective_alignment']={chunk_legacy.get('objective_alignment')!r}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — chunk_v4 strict validation admits the new optional fields
# --------------------------------------------------------------------------- #


def _build_chunk_validator():
    """Build a Draft202012Validator with offline ref resolution against
    every $id in schemas/ so the new ObjectiveAlignment $ref into
    courseforge_jsonld_v1.schema.json resolves without network.

    Mirrors the loader at
    ``Trainforge/tests/test_chunk_strict_validation.py::_build_validator``
    so this contract gate uses the same resolver as the per-worker test
    file. Inlined here so a rename of the per-worker helper can't
    silently disable this gate.
    """
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schemas_dir = PROJECT_ROOT / "schemas"
    chunk_schema_path = schemas_dir / "knowledge" / "chunk_v4.schema.json"

    schema = json.loads(chunk_schema_path.read_text(encoding="utf-8"))
    id_to_schema: Dict[str, Any] = {}
    for p in schemas_dir.rglob("*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            id_to_schema[sid] = s

    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(schema, registry=registry)


def _base_v4_chunk() -> Dict[str, Any]:
    """Minimum chunk_v4-compliant record carrying the required fields
    only — no W5.C audit arrays."""
    return {
        "id": "test_course_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "week_01_overview",
            "lesson_id": "l1",
        },
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 3,
        "word_count": 3,
        "bloom_level": "apply",
    }


def test_chunk_v4_strict_validation_admits_new_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan §4 Test 4 — under ``TRAINFORGE_VALIDATE_CHUNKS=true``,
    validate three fixture chunks against ``chunk_v4.schema.json``:

    (a) chunk WITHOUT the new fields → passes.
    (b) chunk WITH structured ``key_claims`` + ``objective_alignment``
        → passes.
    (c) chunk WITH malformed ``key_claims=[{"claim": 5}]`` (claim is
        not a string) → fails with a clear schema error.

    Pins the W5.C admission contract: legacy back-compat AND
    structured-emit positive AND malformed-emit fail-loud.
    """
    monkeypatch.setenv("TRAINFORGE_VALIDATE_CHUNKS", "true")

    validator = _build_chunk_validator()

    # ---- (a) Legacy chunk (no new fields) — passes. ---- #
    legacy_chunk = _base_v4_chunk()
    assert "key_claims" not in legacy_chunk
    assert "objective_alignment" not in legacy_chunk
    legacy_errors = list(validator.iter_errors(legacy_chunk))
    assert legacy_errors == [], (
        "(a) pre-Wave-5 chunk (no new fields) MUST validate cleanly; "
        f"got errors: {[(list(e.absolute_path), e.message) for e in legacy_errors]}"
    )

    # ---- (b) Structured chunk — passes. ---- #
    structured_chunk = _base_v4_chunk()
    structured_chunk["key_claims"] = [
        {"claim": "X is Y", "source_chunk_ids": ["dart:chunk_A"]}
    ]
    structured_chunk["objective_alignment"] = [
        {
            "objective_id": "TO-05",
            "declared_bloom": "apply",
            "status": "delivered",
        }
    ]
    structured_errors = list(validator.iter_errors(structured_chunk))
    assert structured_errors == [], (
        "(b) structured-shape chunk MUST validate cleanly; "
        f"got errors: {[(list(e.absolute_path), e.message) for e in structured_errors]}"
    )

    # ---- (c) Malformed chunk — fails with clear schema error. ---- #
    malformed_chunk = _base_v4_chunk()
    malformed_chunk["key_claims"] = [{"claim": 5}]  # claim is int, not str
    malformed_errors = list(validator.iter_errors(malformed_chunk))
    assert malformed_errors, (
        "(c) malformed key_claims (int claim) MUST fail strict validation; "
        "the oneOf must reject both the List[str] arm (object is not a "
        "string) and the structured arm (claim is not a string)"
    )
    # The error path MUST attribute to key_claims so a future schema
    # refactor can't silently shift error attribution onto a different
    # field.
    paths = [list(e.absolute_path) for e in malformed_errors]
    assert any("key_claims" in p for p in paths), (
        f"expected error path to include 'key_claims'; got {paths}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — pair_claim_support uses per-claim attribution from key_claims
# --------------------------------------------------------------------------- #


class _StubEmbedder:
    """Stub :class:`SentenceEmbedder` returning a deterministic vector
    keyed off the input string's prefix. Mirrors the SentenceEmbedder
    ``encode(text) -> np.ndarray-like`` surface the W5.D per-claim
    matcher expects, but without pulling in the optional
    ``[embedding]`` extras (sentence-transformers + torch).

    The encoding is intentionally trivial: any text starting with
    ``"RDF triples"`` maps to vector ``(1, 0)``; any text starting
    with ``"SHACL"`` maps to ``(0, 1)``; everything else maps to
    ``(0.5, 0.5)``. The cosine helper that consumes these returns
    1.0 / 0.0 against the canonical claims so the per-claim matcher
    reliably picks the right claim per sentence.
    """

    def encode(self, text: str, normalize: bool = True):
        text = (text or "").strip()
        if text.startswith("RDF triples"):
            return [1.0, 0.0]
        if text.startswith("SHACL"):
            return [0.0, 1.0]
        return [0.5, 0.5]


def _stub_cosine_similarity(vec_a, vec_b):
    """Stub cosine matching the :class:`_StubEmbedder` 2-d encoding.
    Returns the standard normalised dot product so ``(1,0)·(1,0)=1.0``
    and ``(1,0)·(0,1)=0.0``. Above the W5.D
    ``_PER_CLAIM_MATCH_COSINE_FLOOR`` (0.50 default) so the matcher
    promotes the matching claim entry."""
    a0, a1 = float(vec_a[0]), float(vec_a[1])
    b0, b1 = float(vec_b[0]), float(vec_b[1])
    import math

    norm_a = math.sqrt(a0 * a0 + a1 * a1) or 1.0
    norm_b = math.sqrt(b0 * b0 + b1 * b1) or 1.0
    return (a0 * b0 + a1 * b1) / (norm_a * norm_b)


def test_pair_claim_support_uses_per_claim_attribution_when_chunk_carries_key_claims(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan §4 Test 5 — :meth:`PairClaimSupportValidator.validate_pair`
    MUST use per-claim attribution when the cited chunk carries
    structured ``key_claims=[{claim, source_chunk_ids[]}]``.

    Positive contract:

    1. NLI fan-out is scoped per-claim against the attributed chunks'
       text (NOT the whole-chunk text). The stub NLI's recorded
       ``calls`` premise list MUST contain the per-claim chunk text,
       NOT the parent chunk's body text.
    2. ``new_fields["per_claim_support"]`` carries
       ``source_chunk_ids`` populated per sentence from the matched
       structured claim.
    3. The captured ``pair_claim_support_check`` event's rationale
       interpolates ``structured_attribution_used=True`` AND
       ``per_claim_match_count=2`` (both sentences matched a
       structured claim).

    Negative contract (back-compat): same pair + chunk WITHOUT
    structured ``key_claims`` falls back to whole-chunk-text NLI
    scoring (the W4.A pre-W5.D behaviour stays byte-identical).
    """
    from lib.validators import pair_claim_support as pcs_mod
    from lib.validators.pair_claim_support import PairClaimSupportValidator

    # ---- Stub the embedder + cosine helpers so the per-claim
    # matcher fires deterministically without the optional
    # ``[embedding]`` extras. The W5.D path calls
    # ``_try_load_embedder_safe`` and ``_cosine_safe`` from inside
    # ``validate_pair`` so we monkey-patch the module-level helpers.
    monkeypatch.setattr(
        pcs_mod, "_try_load_embedder_safe", lambda: _StubEmbedder()
    )
    monkeypatch.setattr(
        pcs_mod, "_cosine_safe",
        lambda a, b: _stub_cosine_similarity(a, b),
    )

    # Two-sentence completion mapped 1-to-1 against two structured
    # claim entries; each claim cites a distinct DART chunk ID. Each
    # sentence MUST exceed the validator's ``_MIN_SENTENCE_TOKENS=4``
    # content-token floor (alphabetic-only tokens of >=2 chars) so the
    # decomposer doesn't drop them as too-short fragments before NLI
    # fan-out.
    completion = (
        "RDF triples link a subject to an object via a predicate. "
        "SHACL constraints validate RDF graphs against shape definitions."
    )
    pair = {
        "completion": completion,
        "chunk_id": "ch1#sec1",
    }

    chunk_a_text = (
        "RDF triples link a subject to an object via a predicate "
        "in the canonical RDF graph form."
    )
    chunk_b_text = (
        "SHACL constraints validate RDF graphs against shape "
        "definitions under the closed-world axiom."
    )
    chunk_id_to_text_map = {
        "dart:chunk_A": chunk_a_text,
        "dart:chunk_B": chunk_b_text,
    }
    chunk = {
        "text": "Parent chunk body text — not the per-claim attributed text.",
        "key_claims": [
            {
                "claim": (
                    "RDF triples link a subject to an object via a predicate"
                ),
                "source_chunk_ids": ["dart:chunk_A"],
            },
            {
                "claim": (
                    "SHACL constraints validate RDF graphs against shape "
                    "definitions"
                ),
                "source_chunk_ids": ["dart:chunk_B"],
            },
        ],
    }

    capture = _RecordingCapture()
    stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.85, 0.10, 0.05),
    )
    validator = PairClaimSupportValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        chunk_id_to_text_map=chunk_id_to_text_map,
        decision_capture=capture,
    )

    # ---- Sub-clause 1 — gate verdict (validated). ---- #
    assert status == "validated", (
        f"expected validated; got status={status!r}, reason={reason!r}, "
        f"new_fields={new_fields!r}"
    )

    # ---- Sub-clause 2 — NLI fan-out scoped per-claim, NOT against the
    #     whole-chunk text. The stub records every (premise, hypothesis)
    #     pair so we can pin the premise-text contract directly. ---- #
    assert stub.calls, (
        "stub NLI MUST have been called at least once; got no calls"
    )
    all_premises: List[str] = [
        premise for batch in stub.calls for (premise, _h) in batch
    ]
    assert chunk_a_text in all_premises, (
        f"NLI fan-out MUST score against per-claim chunk_A text; "
        f"got premises={all_premises!r}"
    )
    assert chunk_b_text in all_premises, (
        f"NLI fan-out MUST score against per-claim chunk_B text; "
        f"got premises={all_premises!r}"
    )
    assert chunk["text"] not in all_premises, (
        "NLI fan-out MUST NOT use the whole-chunk text as premise when "
        "structured key_claims are present (W5.D scoping contract); "
        f"got premises={all_premises!r}"
    )

    # ---- Sub-clause 3 — per_claim_support carries source_chunk_ids. ---- #
    pcs = new_fields.get("per_claim_support")
    assert isinstance(pcs, list) and len(pcs) == 2, (
        f"expected 2 per_claim_support entries (one per completion sentence); "
        f"got {pcs!r}"
    )
    pcs_chunk_ids = [entry.get("source_chunk_ids") for entry in pcs]
    assert pcs_chunk_ids == [["dart:chunk_A"], ["dart:chunk_B"]], (
        "per_claim_support[*].source_chunk_ids MUST be populated per "
        f"sentence from the matched structured claim; got {pcs_chunk_ids!r}"
    )

    # ---- Sub-clause 4 — capture event rationale interpolates the
    #     W5.D contract signals. ---- #
    claim_events = [
        e for e in capture.events
        if e.get("decision_type") == "pair_claim_support_check"
    ]
    assert len(claim_events) == 1, (
        f"expected exactly 1 pair_claim_support_check event; got "
        f"{len(claim_events)}: "
        f"{[e.get('decision_type') for e in capture.events]!r}"
    )
    rationale = str(claim_events[0].get("rationale", ""))
    assert "structured_attribution_used=True" in rationale, (
        "W5.D rationale MUST interpolate 'structured_attribution_used=True'; "
        f"got rationale={rationale!r}"
    )
    assert "per_claim_match_count=2" in rationale, (
        "W5.D rationale MUST interpolate 'per_claim_match_count=2'; "
        f"got rationale={rationale!r}"
    )

    # ----------------------------------------------------------------- #
    # Negative case — chunk WITHOUT structured key_claims falls back to
    # whole-chunk-text scoring (W4.A pre-W5.D behaviour). Pins the
    # back-compat contract.
    # ----------------------------------------------------------------- #
    legacy_chunk = {"text": "Whole-chunk body text used as the NLI premise."}
    legacy_capture = _RecordingCapture()
    legacy_stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.85, 0.10, 0.05),
    )
    legacy_validator = PairClaimSupportValidator(nli=legacy_stub)

    legacy_status, legacy_reason, legacy_fields = (
        legacy_validator.validate_pair(
            pair,
            kind="instruction",
            chunk=legacy_chunk,
            decision_capture=legacy_capture,
        )
    )

    assert legacy_status == "validated", (
        f"legacy fallback MUST validate; got status={legacy_status!r}, "
        f"reason={legacy_reason!r}"
    )
    legacy_premises = [
        premise for batch in legacy_stub.calls for (premise, _h) in batch
    ]
    assert all(p == legacy_chunk["text"] for p in legacy_premises), (
        "legacy chunk (no structured key_claims) MUST score against the "
        f"whole-chunk text; got premises={legacy_premises!r}"
    )
    # Defence-in-depth: legacy fallback emits a capture event whose
    # rationale carries structured_attribution_used=False (W5.D
    # rationale contract — fall-back is also a dynamic per-call signal).
    legacy_events = [
        e for e in legacy_capture.events
        if e.get("decision_type") == "pair_claim_support_check"
    ]
    assert legacy_events, (
        "legacy fallback MUST still emit a pair_claim_support_check event"
    )
    legacy_rationale = str(legacy_events[0].get("rationale", ""))
    assert "structured_attribution_used=False" in legacy_rationale, (
        "W5.D rationale MUST interpolate 'structured_attribution_used=False' "
        f"on legacy fallback; got rationale={legacy_rationale!r}"
    )


# --------------------------------------------------------------------------- #
# Test 6 — assessment_objective_alignment unions synthesized objectives
# --------------------------------------------------------------------------- #


def test_assessment_objective_alignment_resolves_via_synthesized_objectives_source_refs(  # noqa: E501
    tmp_path: Path,
) -> None:
    """Plan §4 Test 6 — :class:`AssessmentObjectiveAlignmentValidator`
    MUST resolve assessment ``objective_id`` against the union of
    chunk ``learning_outcome_refs[]`` ∪ synthesized objectives' ``id``
    set when ``synthesized_objectives_path`` is provided.

    Positive contract: every chunk has empty ``learning_outcome_refs``,
    but the synthesized objectives carry ``TO-05``, so the assessment
    question with ``objective_id="TO-05"`` MUST pass without a
    phantom-ref criticality.

    Negative contract (back-compat): when
    ``synthesized_objectives_path`` is omitted, the gate MUST fail
    closed with ``ASSESSMENT_ALIGNMENT_NO_CHUNKS`` (the chunks-only
    surface is empty so every assessment objective is vacuously
    phantom — the silent-degradation guard documented at
    ``lib/validators/assessment_objective_alignment.py:228``).
    """
    from lib.validators.assessment_objective_alignment import (
        AssessmentObjectiveAlignmentValidator,
    )

    # ---- Build the three on-disk fixtures. ---- #
    # chunks.jsonl — every chunk has empty learning_outcome_refs.
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_lines = [
        {
            "id": "chunk_001",
            "text": "Some chunk body text.",
            "learning_outcome_refs": [],
        },
        {
            "id": "chunk_002",
            "text": "More chunk body text.",
            "learning_outcome_refs": [],
        },
    ]
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks_lines) + "\n",
        encoding="utf-8",
    )

    # synthesized_objectives.json — Courseforge synthesized form
    # (terminal_objectives + chapter_objectives). _flatten_objectives
    # walks both lists and unions every entry's ``id``.
    synthesized_path = tmp_path / "synthesized_objectives.json"
    synthesized_payload = {
        "terminal_objectives": [
            {
                "id": "TO-05",
                "text": "Apply RDF/SHACL constraints to validate a graph.",
                "source_refs": [
                    {"ref": "ch3", "chunk_ids": ["dart:chunk_A"]}
                ],
            }
        ],
        "chapter_objectives": [],
    }
    synthesized_path.write_text(
        json.dumps(synthesized_payload), encoding="utf-8"
    )

    # assessments.json — one question carrying objective_id="TO-05".
    assessments_path = tmp_path / "assessments.json"
    assessments_payload = {
        "questions": [
            {
                "question_id": "q-001",
                "objective_id": "TO-05",
                "stem": "Which SHACL constraint validates RDF triples?",
                "options": [
                    {"text": "sh:datatype", "correct": True},
                    {"text": "sh:pattern", "correct": False},
                ],
            }
        ]
    }
    assessments_path.write_text(
        json.dumps(assessments_payload), encoding="utf-8"
    )

    validator = AssessmentObjectiveAlignmentValidator()

    # ---- Positive case — synthesized_objectives_path provided. ---- #
    pos_capture = _RecordingCapture()
    pos_result = validator.validate({
        "assessments_path": str(assessments_path),
        "chunks_path": str(chunks_path),
        "synthesized_objectives_path": str(synthesized_path),
        "decision_capture": pos_capture,
    })

    assert pos_result.passed, (
        "W5.E union surface MUST pass when the assessment objective is "
        "in the synthesized objectives set even though chunks-side refs "
        f"are empty; got passed={pos_result.passed}, issues="
        f"{[(i.code, i.message) for i in (pos_result.issues or [])]}"
    )
    # Defence-in-depth: per-question event carries the W5.E flag.
    pos_events = [
        e for e in pos_capture.events
        if e.get("decision_type") == "assessment_objective_alignment_check"
    ]
    assert pos_events, (
        "expected at least one assessment_objective_alignment_check event "
        "on the positive case"
    )
    pos_event = pos_events[0]
    pos_metrics = pos_event.get("metrics") or {}
    assert pos_metrics.get("resolved_via_synthesized_objectives") is True, (
        "W5.E event metrics MUST flag resolved_via_synthesized_objectives=True; "
        f"got metrics={pos_metrics!r}"
    )

    # ---- Negative case — omit synthesized_objectives_path. ---- #
    # Pre-W5.E behaviour: chunks-side surface is empty (every chunk has
    # learning_outcome_refs=[]) so the silent-degradation guard fires
    # ASSESSMENT_ALIGNMENT_NO_CHUNKS.
    neg_result = validator.validate({
        "assessments_path": str(assessments_path),
        "chunks_path": str(chunks_path),
    })

    assert not neg_result.passed, (
        "back-compat: gate MUST fail when chunks-side refs are empty "
        "AND synthesized_objectives_path is omitted; "
        f"got passed={neg_result.passed}"
    )
    neg_codes = [i.code for i in (neg_result.issues or [])]
    # The plan §4 spec asks for MISSING_OBJECTIVE_REF as the fail
    # signature; the validator's actual fail code on the empty-chunks
    # silent-degradation guard is ASSESSMENT_ALIGNMENT_NO_CHUNKS.
    # Either signal proves the back-compat fail-closed contract holds
    # — the validator did NOT vacuously accept the assessment question
    # against an empty resolution surface.
    assert (
        "ASSESSMENT_ALIGNMENT_NO_CHUNKS" in neg_codes
        or "MISSING_OBJECTIVE_REF" in neg_codes
        or "PHANTOM_OBJECTIVE_REFS" in neg_codes
    ), (
        f"expected back-compat fail with one of "
        f"{{ASSESSMENT_ALIGNMENT_NO_CHUNKS, MISSING_OBJECTIVE_REF, "
        f"PHANTOM_OBJECTIVE_REFS}}; got codes={neg_codes!r}"
    )


# --------------------------------------------------------------------------- #
# Optional Test 7 — merge_small_sections unions audit fields across merge
# --------------------------------------------------------------------------- #


def test_merge_small_sections_unions_audit_fields_across_merge() -> None:
    """Plan §4 optional Test 7 — :func:`merge_small_sections` MUST
    carry the W5.A per-section audit fields across the merge boundary.

    Two adjacent ContentSections each ~100 words (well below
    :data:`MIN_CHUNK_SIZE` so they collapse into a single merged
    buffer) each carry their own ``key_claims`` + ``objective_alignment``.
    The emitted :class:`MergedSectionResult`:

    * ``merged_key_claims`` is the concatenation of both sections'
      ``key_claims`` arrays (per-claim entries are independent, so
      concat is safe — W5.F brief).
    * ``merged_objective_alignment`` is deduped by ``objective_id`` with
      first-seen-wins (mirrors the
      :func:`merge_section_source_ids` policy).

    Verifies W5.F end-to-end (symmetric to the W4 TIGHT round-trip
    test in ``test_wave4_pair_validation_contract.py::
    test_wave4_strict_pair_schema_round_trip``).
    """
    from Trainforge.chunker import merge_small_sections
    from Trainforge.parsers.html_content_parser import ContentSection

    # ~100-word body for each section (small enough that the merger
    # collapses both into a single buffer).
    body_a = " ".join([f"alpha word {i}" for i in range(50)])
    body_b = " ".join([f"beta word {i}" for i in range(50)])

    sec_a_key_claim = _structured_key_claim(
        claim="A claim from section A",
        source_chunk_ids=["dart:chunk_A1"],
    )
    sec_b_key_claim = _structured_key_claim(
        claim="A claim from section B",
        source_chunk_ids=["dart:chunk_B1"],
    )
    # Overlapping objective_id ("TO-05") on both sections — first-seen
    # wins. Section A is emitted first, so its entry survives.
    sec_a_obj = _objective_alignment_entry(
        objective_id="TO-05", status="delivered"
    )
    sec_b_obj_overlap = _objective_alignment_entry(
        objective_id="TO-05", status="underdelivered"
    )
    sec_b_obj_distinct = _objective_alignment_entry(
        objective_id="TO-07", status="delivered"
    )

    sec_a = ContentSection(
        heading="Section A",
        level=2,
        content=body_a,
        word_count=100,
        key_claims=[sec_a_key_claim],
        objective_alignment=[sec_a_obj],
    )
    sec_b = ContentSection(
        heading="Section B",
        level=2,
        content=body_b,
        word_count=100,
        key_claims=[sec_b_key_claim],
        objective_alignment=[sec_b_obj_overlap, sec_b_obj_distinct],
    )

    merged = merge_small_sections([sec_a, sec_b])

    # Both sections collapse into one merged buffer.
    assert len(merged) == 1, (
        f"expected exactly 1 merged result (both sections collapse); "
        f"got {len(merged)}: {merged!r}"
    )

    result = merged[0]

    # ---- Sub-clause 1 — back-compat 5-tuple destructure still works. ---- #
    heading, combined_text, chunk_type, source_ids, headings = result
    assert headings == ["Section A", "Section B"], (
        f"expected merged_headings=['Section A', 'Section B']; "
        f"got {headings!r}"
    )

    # ---- Sub-clause 2 — merged_key_claims is the concat. ---- #
    assert hasattr(result, "merged_key_claims"), (
        "MergedSectionResult MUST expose merged_key_claims attribute "
        "(W5.F contract)"
    )
    assert result.merged_key_claims == [sec_a_key_claim, sec_b_key_claim], (
        "merged_key_claims MUST be the concatenation of every section's "
        f"key_claims (per-claim entries are independent); "
        f"got {result.merged_key_claims!r}"
    )

    # ---- Sub-clause 3 — merged_objective_alignment dedups first-seen. ---- #
    assert hasattr(result, "merged_objective_alignment"), (
        "MergedSectionResult MUST expose merged_objective_alignment "
        "attribute (W5.F contract)"
    )
    objs = result.merged_objective_alignment
    objective_ids = [entry.get("objective_id") for entry in objs]
    assert objective_ids == ["TO-05", "TO-07"], (
        "merged_objective_alignment MUST dedupe by objective_id (first-"
        f"seen wins); got objective_ids={objective_ids!r}"
    )
    # The TO-05 entry MUST be section A's (status='delivered'), NOT
    # section B's overlapping entry (status='underdelivered'). Pins
    # the first-seen-wins policy contract directly.
    to05_entry = next(o for o in objs if o.get("objective_id") == "TO-05")
    assert to05_entry.get("status") == "delivered", (
        "first-seen-wins on overlapping objective_id MUST preserve "
        "section A's status='delivered' (not section B's "
        f"'underdelivered'); got entry={to05_entry!r}"
    )
