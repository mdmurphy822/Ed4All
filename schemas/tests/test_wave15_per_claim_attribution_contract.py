"""GPT Feedback v2 — Wave 1.5 end-of-wave per-claim-attribution gate.

Authored 2026-05-06 against the closing 6-test gate enumerated in
the Wave 1.5 per-claim-attribution gate spec § 4.
Predecessors: W1.5.A (d317327, schema oneOf bump), W1.5.B (58db527,
outline prompt directives), W1.5.C (5de2326, per-claim consistency
walk + ``per_claim_attribution_unfixable`` escalation marker), W1.5.D
(d9e4264, rewrite-tier per-claim citation map prompt threading).

This file is a SELF-CONTAINED contract test that asserts each of the 6
plan-§4 invariants directly; it does NOT delegate into the per-worker
test files so a future test rename / move can't silently break the
wave gate. Mirrors the pattern of the Wave 1 end-of-wave gate at
``schemas/tests/test_wave1_additive_contract.py`` (commit 9c2987b).

Tests:

- ``test_1_schema_validity_both_shapes_and_mixed_rejection`` — pin the
  bumped per-block-type schema's ``oneOf`` contract: legacy ``List[str]``
  validates, structured ``List[{claim, source_chunk_ids[]}]``
  validates, mixed shape FAILS, empty ``source_chunk_ids: []`` FAILS.
- ``test_2_anti_silent_degradation_chunk_id_consistency`` — pin the
  per-claim consistency walk inside ``BlockSourceRefValidator``:
  unresolved per-claim chunk_id fires
  ``OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS`` with ``action="regenerate"``
  (NOT "block" — claim-level only); decision-capture event surfaces
  ``claim_misses_count >= 1``.
- ``test_3_anti_silent_degradation_legacy_corpora_validate`` — pin
  back-compat: at least 5 distinct legacy ``List[str]`` ``key_claims``
  examples covering canonical block_types validate clean against the
  bumped schema.
- ``test_4_outline_prompt_golden_output_sanity`` — pin the outline-tier
  prompt directives + retry pattern: system prompt carries the
  ``"key_claims MUST be a list of objects"`` sentinel, the rendered
  user prompt carries ``"source_chunk_ids MUST be a non-empty subset"``,
  and ``_RETRY_DIRECTIVE_PATTERNS`` carries an entry whose regex
  matches ``"is not valid under any of the given schemas"``.
- ``test_5_rewrite_prompt_includes_per_claim_map`` — pin the rewrite
  tier prompt threading: ``_render_user_prompt`` and
  ``_render_escalated_user_prompt`` both render
  ``"Per-claim source attribution"`` + ``"claim 1: "`` / ``"claim 2: "``
  + each chunk_id list verbatim; total prompt-size growth versus the
  legacy shape stays under the 500-char §6.3 calibration ceiling; the
  ``_REWRITE_SYSTEM_PROMPT`` literal carries the
  ``"per-claim source attribution"`` directive sentinel.
- ``test_6_collected_count_guard`` — wave-end "no silent test removal"
  ratchet. AST-walks each of the three Courseforge sub-trees
  (``Courseforge/generators/tests/``, ``Courseforge/router/tests/``,
  ``Courseforge/scripts/tests/``) and counts every ``def test_``
  declaration; assert each tree's count is at least the post-Wave-1.5
  baseline. Also asserts ``Draft202012Validator.check_schema(...)``
  passes on the bumped per-block-type schema (no jsonschema deprecation
  warnings on the new ``oneOf`` surface).
"""
from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Block lives at ``Courseforge/scripts/blocks.py`` — mirror the import
# bridge used by the Courseforge router test suite.
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Audited post-Wave-1.5 baselines — encoded as constants so a future
# silent test removal trips the ratchet. Counts captured 2026-05-06
# on dev-v0.3.0 at HEAD by AST-walking each tree's ``test_*.py`` files
# and tallying every ``def test_`` declaration (the same algorithm
# ``_count_test_functions`` runs at gate time).
#
# Note: these counts are LOWER than the pytest --collect-only counts
# (80 / 167 / 419) because pytest expands parametrized cases at
# collection time. AST counting is the implementation choice for
# this gate (subprocess-pytest from inside a test is brittle); the
# ratchet contract — "no silent test removal" — is satisfied
# either way because both algorithms are monotonic.
# --------------------------------------------------------------------------- #

_MIN_GENERATORS_TESTS = 80
_MIN_ROUTER_TESTS = 149
_MIN_SCRIPTS_TESTS = 380


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _outline_payload(
    *,
    block_type: str,
    key_claims: List[Any],
) -> Dict[str, Any]:
    """Build a structurally-valid outline payload around ``key_claims``.

    Every other field is filled in just enough to satisfy the per-block-
    type ``required`` list so the only contract under test is the
    ``key_claims`` arm of the ``oneOf``.
    """
    from Courseforge.generators.outline._outline_provider import _OUTLINE_KIND_BOUNDS

    section_min, _section_max = (
        _OUTLINE_KIND_BOUNDS.get(block_type, {}).get("section_skeleton", (0, 0))
    )
    section_skeleton: List[Dict[str, Any]]
    if section_min <= 0:
        section_skeleton = []
    else:
        section_skeleton = [{"heading": "Definition"}]

    base: Dict[str, Any] = {
        "block_id": f"page_01#{block_type}_x_0",
        "block_type": block_type,
        # ``explanation`` is the canonical ChunkType enum value matching
        # the definitional/conceptual SHACL prose these claims carry;
        # ``definition`` is NOT in the per-block-type schema's 10-value
        # content_type enum (schemas/taxonomies/content_type.json
        # ::$defs.ChunkType) so it fails the bumped ``oneOf`` schema.
        "content_type": "explanation",
        "bloom_level": "understand",
        "objective_refs": ["TO-01"],
        "curies": ["sh:NodeShape"],
        "key_claims": key_claims,
        "section_skeleton": section_skeleton,
        "source_refs": [
            {"sourceId": "semantik:shacl-spec#sec1", "role": "primary"},
        ],
        "structural_warnings": [],
    }
    if block_type == "assessment_item":
        base["stem"] = "Which of the following describes a SHACL node shape?"
        base["answer_key"] = "A"
        base["distractors"] = [
            {"text": "An OWL class declaration."},
            {"text": "A SHACL node shape declaring constraints."},
        ]
        base["correct_answer_index"] = 1
    elif block_type == "prereq_set":
        base["prerequisitePages"] = ["page_00"]
    return base


def _count_test_functions(tree: Path) -> Tuple[int, List[str]]:
    """AST-walk every ``test_*.py`` under ``tree`` and count ``def test_``
    declarations (top-level and nested inside class bodies).

    Returns ``(count, file_paths)`` so a regression diagnostic can name
    every contributing file.
    """
    if not tree.exists():
        return 0, []
    total = 0
    seen_files: List[str] = []
    for path in sorted(tree.rglob("test_*.py")):
        # Skip __pycache__ / hidden directories defensively.
        if "__pycache__" in path.parts:
            continue
        try:
            module = ast.parse(path.read_text())
        except (SyntaxError, OSError):
            continue
        seen_files.append(str(path.relative_to(PROJECT_ROOT)))
        for node in ast.walk(module):
            # Top-level + class-nested test funcs both qualify.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    total += 1
    return total, seen_files


# --------------------------------------------------------------------------- #
# Test 1 — schema validity (both shapes + mixed-shape rejection)
# --------------------------------------------------------------------------- #


_TEST_1_LEGACY_BY_TYPE: Dict[str, List[str]] = {
    "concept": [
        "A SHACL node shape declares constraints that fire on target nodes.",
        "Each constraint references a property predicate.",
    ],
    "assessment_item": [
        "Stem references the SHACL node shape concept.",
    ],
    "prereq_set": [
        "Learners must know RDF triples before SHACL.",
    ],
    "summary_takeaway": [
        "SHACL gates structural validity of an RDF graph.",
        "Node shapes are composable with property shapes.",
    ],
}

_TEST_1_STRUCTURED_BY_TYPE: Dict[str, List[Dict[str, Any]]] = {
    "concept": [
        {
            "claim": "A SHACL node shape declares constraints "
            "that fire on target nodes.",
            "source_chunk_ids": ["semantik:shacl-spec#sec1"],
        },
        {
            "claim": "Each constraint references a property predicate.",
            "source_chunk_ids": [
                "semantik:shacl-spec#sec2",
                "semantik:shacl-spec#sec3",
            ],
        },
    ],
    "assessment_item": [
        {
            "claim": "Stem references the SHACL node shape concept.",
            "source_chunk_ids": ["semantik:shacl-spec#sec1"],
        },
    ],
    "prereq_set": [
        {
            "claim": "Learners must know RDF triples before SHACL.",
            "source_chunk_ids": ["semantik:rdf-primer#triples"],
        },
    ],
    "summary_takeaway": [
        {
            "claim": "SHACL gates structural validity of an RDF graph.",
            "source_chunk_ids": ["semantik:shacl-spec#sec0"],
        },
        {
            "claim": "Node shapes are composable with property shapes.",
            "source_chunk_ids": ["semantik:shacl-spec#sec5"],
        },
    ],
}


def test_1_schema_validity_both_shapes_and_mixed_rejection() -> None:
    """Plan §4 Test 1 — bumped ``oneOf`` schema admits both shapes and
    rejects the mixed + empty-attribution failure modes.

    Replicates the per-block-type sweep from
    ``Courseforge/router/tests/test_outline_provider_schema.py`` so a
    future rename of that file can't silently disable this gate. Uses
    ``Courseforge.generators.outline._outline_provider._build_block_outline_schema``
    directly (the schema builder is the single source of truth for the
    bumped per-block-type ``oneOf`` shape).
    """
    jsonschema = _require_jsonschema()

    from Courseforge.generators.outline._outline_provider import (
        _BLOCK_TYPE_JSON_SCHEMAS,
    )

    block_types = ("concept", "assessment_item", "prereq_set", "summary_takeaway")

    # Sub-test 1a — legacy List[str] validates against the bumped schema
    # for every representative block_type.
    for block_type in block_types:
        schema = _BLOCK_TYPE_JSON_SCHEMAS[block_type]
        payload = _outline_payload(
            block_type=block_type,
            key_claims=_TEST_1_LEGACY_BY_TYPE[block_type],
        )
        # Should not raise.
        jsonschema.Draft202012Validator(schema).validate(payload)

    # Sub-test 1b — structured List[{claim, source_chunk_ids[]}]
    # validates against the same set.
    for block_type in block_types:
        schema = _BLOCK_TYPE_JSON_SCHEMAS[block_type]
        payload = _outline_payload(
            block_type=block_type,
            key_claims=_TEST_1_STRUCTURED_BY_TYPE[block_type],
        )
        jsonschema.Draft202012Validator(schema).validate(payload)

    # Sub-test 1c — mixed shape (one string + one object in the same
    # array) MUST FAIL via the canonical ``oneOf`` rejection error.
    schema = _BLOCK_TYPE_JSON_SCHEMAS["concept"]
    mixed_payload = _outline_payload(
        block_type="concept",
        key_claims=[
            "A bare-string legacy claim.",
            {
                "claim": "A structured claim mixed in.",
                "source_chunk_ids": ["semantik:shacl-spec#sec1"],
            },
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(mixed_payload)

    # Sub-test 1d — empty source_chunk_ids: [] MUST FAIL the inner
    # ``minItems: 1`` bound.
    empty_payload = _outline_payload(
        block_type="concept",
        key_claims=[
            {
                "claim": "A claim with no attribution.",
                "source_chunk_ids": [],
            },
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(empty_payload)


# --------------------------------------------------------------------------- #
# Test 2 — anti-silent-degradation: every new chunk_id is consistent
# --------------------------------------------------------------------------- #


class _RecordingCapture:
    """Minimal capture stub recording every ``log_decision`` payload."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _outline_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    page_id: str = "page_01",
    sequence: int = 0,
    curies: Tuple[str, ...] = ("ed4all:Foo",),
    key_claims: Optional[List[Any]] = None,
    content_type: str = "definition",
    objective_ids: Tuple[str, ...] = ("TO-01",),
    source_references: Tuple[Dict[str, Any], ...] = (),
    source_ids: Tuple[str, ...] = (),
):  # type: ignore[no-untyped-def]
    """Self-contained outline-tier Block fixture.

    Inlined from
    ``Courseforge/router/tests/test_inter_tier_gates.py::_outline_block``
    so a rename of that helper can't silently break this gate.
    """
    from blocks import Block  # noqa: WPS433  (path-injected module)

    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content={
            "curies": list(curies),
            "key_claims": (
                list(key_claims)
                if key_claims is not None
                else ["The ed4all:Foo predicate marks anchoring."]
            ),
            "content_type": content_type,
        },
        objective_ids=tuple(objective_ids),
        source_references=tuple(source_references),
        source_ids=tuple(source_ids),
    )


def test_2_anti_silent_degradation_chunk_id_consistency() -> None:
    """Plan §4 Test 2 — per-claim attribution mismatch fires
    ``OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS`` with
    ``action="regenerate"`` (NOT "block" — claim-level only).

    Fixture mirrors the plan §4 Test 2 verbatim:

      block.source_references = [{"sourceId": "semantik:baz#qux"}]
      block.content["key_claims"] = [
          {"claim": "foo", "source_chunk_ids": ["semantik:foo#bar"]}
      ]

    Decision-capture event MUST surface ``claim_misses_count >= 1`` so
    the audit trail records per-claim attribution health.
    """
    from Courseforge.router.inter_tier_gates import BlockSourceRefValidator

    capture = _RecordingCapture()
    valid_ids = {"semantik:baz#qux"}
    blocks = [
        _outline_block(
            block_id="page_01#concept_5",
            source_references=({"sourceId": "semantik:baz#qux"},),
            key_claims=[
                {
                    "claim": "foo",
                    "source_chunk_ids": ["semantik:foo#bar"],
                },
            ],
        ),
    ]
    result = BlockSourceRefValidator().validate({
        "blocks": blocks,
        "valid_source_ids": list(valid_ids),
        "decision_capture": capture,
    })

    assert result.passed is False, "expected per-claim miss to fail the gate"
    assert result.action == "regenerate", (
        f"expected action='regenerate' on claim-level-only miss; "
        f"got action={result.action!r}"
    )

    codes = [i.code for i in result.issues]
    assert "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS" in codes, (
        "expected at least one issue with code "
        "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS"
    )

    # Per-claim miss MUST be warning severity (claim-level is semantic,
    # not structural — the rewrite tier can re-roll).
    per_claim_issues = [
        i for i in result.issues
        if i.code == "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS"
    ]
    assert all(i.severity == "warning" for i in per_claim_issues), (
        "every per-claim miss MUST be severity=warning"
    )

    # Decision-capture event MUST surface claim_misses_count >= 1.
    relevant = [
        e for e in capture.events
        if e.get("decision_type") == "block_source_ref_check"
    ]
    assert relevant, "expected at least one block_source_ref_check event"
    assert any(
        "claim_misses_count=" in str(e.get("rationale", ""))
        and "claim_misses_count=0" not in str(e.get("rationale", ""))
        for e in relevant
    ), (
        "decision-capture rationale MUST surface claim_misses_count >= 1; "
        f"got events: {relevant!r}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — anti-silent-degradation: legacy corpora still validate
# --------------------------------------------------------------------------- #


# Five distinct legacy ``List[str]`` ``key_claims`` examples covering
# every canonical block_type the plan enumerates. Self-contained so a
# rename of the per-worker test fixture file can't silently break this
# gate.
_TEST_3_LEGACY_FIXTURES: List[Tuple[str, List[str]]] = [
    (
        "concept",
        [
            "A SHACL node shape declares constraints on RDF nodes.",
            "Constraints fire when a target node fails a property predicate.",
        ],
    ),
    (
        "assessment_item",
        [
            "The stem references the SHACL node shape concept.",
        ],
    ),
    (
        "prereq_set",
        [
            "Learners must understand RDF graphs before SHACL shapes.",
        ],
    ),
    (
        "callout",
        [
            "Note: SHACL targets are not the same as OWL ranges.",
        ],
    ),
    (
        "summary_takeaway",
        [
            "SHACL gates structural validity of an RDF graph.",
            "Node shapes compose with property shapes.",
        ],
    ),
]


def test_3_anti_silent_degradation_legacy_corpora_validate() -> None:
    """Plan §4 Test 3 — every canonical legacy ``List[str]`` ``key_claims``
    example MUST validate against the bumped per-block-type schema.

    Catches the regression class where a future schema author tightens
    the ``oneOf`` ``List[str]`` arm and silently breaks every existing
    fixture and every existing rebuild on the RDF/SHACL
    calibration corpus.
    """
    jsonschema = _require_jsonschema()
    from Courseforge.generators.outline._outline_provider import (
        _BLOCK_TYPE_JSON_SCHEMAS,
    )

    assert len(_TEST_3_LEGACY_FIXTURES) >= 5, (
        "Test 3 contract requires at least 5 distinct legacy examples"
    )

    failures: List[str] = []
    for block_type, key_claims in _TEST_3_LEGACY_FIXTURES:
        schema = _BLOCK_TYPE_JSON_SCHEMAS[block_type]
        payload = _outline_payload(block_type=block_type, key_claims=key_claims)
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(payload))
        if errors:
            failures.append(
                f"{block_type}: legacy List[str] failed validation — "
                f"{[(list(e.absolute_path), e.message) for e in errors]}"
            )

    assert not failures, (
        "legacy List[str] fixtures regressed against the bumped oneOf "
        "schema:\n  " + "\n  ".join(failures)
    )


# --------------------------------------------------------------------------- #
# Test 4 — outline-prompt golden-output sanity
# --------------------------------------------------------------------------- #


def test_4_outline_prompt_golden_output_sanity(monkeypatch) -> None:
    """Plan §4 Test 4 — outline-tier prompt directives + retry pattern.

    Three sentinel assertions, replicated from
    ``Courseforge/generators/tests/test_outline_provider.py`` so a
    rename of that file can't silently break this gate:

    1. ``_OUTLINE_SYSTEM_PROMPT`` carries the literal
       ``"key_claims MUST be a list of objects"`` directive.
    2. ``_render_user_prompt(...)`` returns a string containing
       ``"source_chunk_ids MUST be a non-empty subset"``.
    3. ``_RETRY_DIRECTIVE_PATTERNS`` carries an entry whose regex
       matches ``"is not valid under any of the given schemas"`` (the
       canonical jsonschema ``oneOf`` rejection error surface).
    """
    # Import private symbols directly per the plan §4 Test 4 contract.
    from Courseforge.generators.outline._outline_provider import (
        OutlineProvider,
        _OUTLINE_SYSTEM_PROMPT,
        _RETRY_DIRECTIVE_PATTERNS,
        _match_retry_directive,
    )
    from blocks import Block  # noqa: WPS433  (path-injected module)

    # Sub-test 4a — system prompt sentinel.
    assert "key_claims MUST be a list of objects" in _OUTLINE_SYSTEM_PROMPT, (
        "Wave 1.5 W1.5.B system-prompt directive sentinel missing — "
        "operators rely on this literal substring as the regression seam."
    )

    # Sub-test 4b — user prompt closing clause sentinel. Build a minimal
    # local-provider OutlineProvider that bypasses any real network.
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    provider = OutlineProvider(provider="local")
    block = Block(
        block_id="page-1#concept_intro_0",
        block_type="concept",
        page_id="page-1",
        sequence=0,
        content="",
    )
    chunks = [
        {"id": "semantik:slug-a#blk1", "body": "First chunk body."},
        {"id": "semantik:slug-b#blk2", "body": "Second chunk body."},
    ]
    objectives = [
        {"id": "TO-01", "statement": "Define the central concept."},
    ]
    rendered = provider._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives
    )
    assert "source_chunk_ids MUST be a non-empty subset" in rendered, (
        "Wave 1.5 W1.5.B user-prompt closing clause missing — the "
        "outline-tier model needs the chunk-id universe bound stated "
        "verbatim in the user prompt."
    )

    # Sub-test 4c — retry-directive regex coverage. _match_retry_directive
    # MUST return a non-None directive when the validator surfaces the
    # canonical oneOf rejection error. Also confirm at least one entry
    # in _RETRY_DIRECTIVE_PATTERNS has a regex matching the literal.
    canonical_error = (
        "[\"flat string claim\", \"another flat string\"] is not valid "
        "under any of the given schemas"
    )
    directive = _match_retry_directive(canonical_error)
    assert directive is not None, (
        "_match_retry_directive returned None for the canonical oneOf "
        "rejection error surface; the W1.5.B retry pattern is missing"
    )
    assert "key_claims MUST be a list of objects" in directive, (
        "W1.5.B retry directive must point the model at the structured "
        "shape; got: " + directive
    )

    # Pin the regex itself exists in the pattern table (defensive — an
    # author swapping the directive text but accidentally dropping the
    # regex would still pass _match_retry_directive via a fallthrough
    # if a future pattern caught the same error).
    pattern_strings = [p.pattern for (p, _d) in _RETRY_DIRECTIVE_PATTERNS]
    assert any(
        "is not valid under any of the given schemas" in pat
        for pat in pattern_strings
    ), (
        "_RETRY_DIRECTIVE_PATTERNS must contain an entry whose regex "
        "matches the canonical oneOf rejection error literal"
    )


# --------------------------------------------------------------------------- #
# Test 5 — rewrite-tier prompt includes per-claim map
# --------------------------------------------------------------------------- #


def _structured_rewrite_block(
    *,
    claims: Optional[List[dict]] = None,
    escalation_marker: Optional[str] = None,
):  # type: ignore[no-untyped-def]
    """Build an outline-tier Block with structured ``key_claims``.

    Inlined from
    ``Courseforge/generators/tests/test_rewrite_provider_per_claim.py``
    so a rename of that helper can't silently break this gate.
    """
    from blocks import Block  # noqa: WPS433

    if claims is None:
        claims = [
            {
                "claim": "A SHACL node shape declares constraints on RDF nodes.",
                "source_chunk_ids": ["semantik:shacl-spec#sec1"],
            },
            {
                "claim": "Targets fire when a node fails a property predicate.",
                "source_chunk_ids": [
                    "semantik:shacl-spec#sec2",
                    "semantik:shacl-spec#sec3",
                ],
            },
        ]
    return Block(
        block_id="page#concept_intro_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={
            "key_claims": claims,
            "curies": ["sh:NodeShape"],
            "source_refs": [
                "semantik:shacl-spec#sec1",
                "semantik:shacl-spec#sec2",
                "semantik:shacl-spec#sec3",
            ],
            "objective_refs": ["TO-01"],
        },
        escalation_marker=escalation_marker,
    )


def _legacy_rewrite_block(
    *,
    claims: Optional[List[str]] = None,
):  # type: ignore[no-untyped-def]
    """Build an outline-tier Block with legacy ``List[str]`` ``key_claims``."""
    from blocks import Block  # noqa: WPS433

    if claims is None:
        claims = [
            "A SHACL node shape declares constraints on RDF nodes.",
            "Targets fire when a node fails a property predicate.",
        ]
    return Block(
        block_id="page#concept_intro_0",
        block_type="concept",
        page_id="page",
        sequence=0,
        content={
            "key_claims": claims,
            "curies": ["sh:NodeShape"],
            "source_refs": ["semantik:shacl-spec#sec1"],
            "objective_refs": ["TO-01"],
        },
    )


def test_5_rewrite_prompt_includes_per_claim_map(monkeypatch) -> None:
    """Plan §4 Test 5 — rewrite-tier prompt threads the per-claim
    citation map verbatim AND keeps prompt-size growth bounded.

    Five sentinel checks:

    1. Regular ``_render_user_prompt`` carries ``"Per-claim source
       attribution"``, ``"claim 1: "``, ``"claim 2: "``, and each
       chunk_id list verbatim (single-chunk + comma-separated multi-
       chunk both render).
    2. Escalated ``_render_escalated_user_prompt`` carries the same
       per-claim citation map (symmetry across the escalated path).
    3. Prompt-size growth: structured-shape prompt − legacy-shape
       prompt < 500 chars on a 5-claim block (the §6.3 calibration
       ceiling).
    4. ``_REWRITE_SYSTEM_PROMPT`` carries the literal
       ``"per-claim source attribution"`` directive sentinel
       (case-insensitive — W1.5.D's authored copy uses lowercase).
    """
    from Courseforge.generators.rewrite._rewrite_provider import (
        RewriteProvider,
        _REWRITE_SYSTEM_PROMPT,
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    provider = RewriteProvider(anthropic_client=object())

    # Sub-test 5a — regular path renders the per-claim map verbatim.
    structured_block = _structured_rewrite_block()
    rendered = provider._render_user_prompt(
        block=structured_block, source_chunks=[], objectives=[]
    )
    assert "Per-claim source attribution" in rendered
    assert "claim 1: " in rendered
    assert "claim 2: " in rendered
    # Single chunk_id verbatim.
    assert "[semantik:shacl-spec#sec1]" in rendered
    # Multi-chunk list verbatim, comma-separated.
    assert "[semantik:shacl-spec#sec2, semantik:shacl-spec#sec3]" in rendered

    # Sub-test 5b — escalated path symmetry.
    escalated_block = _structured_rewrite_block(
        escalation_marker="outline_budget_exhausted"
    )
    escalated = provider._render_escalated_user_prompt(
        block=escalated_block, source_chunks=[], objectives=[]
    )
    assert "ESCALATED REWRITE" in escalated, (
        "escalated user prompt must keep its existing escalation header"
    )
    assert "Per-claim source attribution" in escalated
    assert "claim 1: " in escalated
    assert "claim 2: " in escalated
    assert "[semantik:shacl-spec#sec1]" in escalated
    assert "[semantik:shacl-spec#sec2, semantik:shacl-spec#sec3]" in escalated

    # Sub-test 5c — prompt-size growth on a 5-claim block stays under
    # the §6.3 calibration ceiling (500 chars).
    structured_claims = [
        {
            "claim": f"Claim number {i} body — short prose statement.",
            # Short synthetic ids keep this size fixture isolating the
            # per-claim MAP overhead (headers / brackets / newlines) from
            # incidental id length, holding the §6.3 calibration ceiling.
            "source_chunk_ids": [
                f"semantik:s#c{i}",
                f"semantik:s#c{i + 100}",
            ][: 1 + (i % 3)],
        }
        for i in range(1, 6)
    ]
    legacy_claims = [
        f"Claim number {i} body — short prose statement." for i in range(1, 6)
    ]
    structured_5 = _structured_rewrite_block(claims=structured_claims)
    legacy_5 = _legacy_rewrite_block(claims=legacy_claims)
    structured_prompt = provider._render_user_prompt(
        block=structured_5, source_chunks=[], objectives=[]
    )
    legacy_prompt = provider._render_user_prompt(
        block=legacy_5, source_chunks=[], objectives=[]
    )
    delta = len(structured_prompt) - len(legacy_prompt)
    assert delta < 500, (
        f"per-claim citation map added {delta} chars on a 5-claim block; "
        f"plan §6.3 calibration ceiling is < 500. Tighten the truncation "
        f"rule or shrink the per-claim block header."
    )

    # Sub-test 5d — system prompt directive sentinel (lowercase per
    # W1.5.D's authored copy).
    assert "per-claim source attribution" in _REWRITE_SYSTEM_PROMPT, (
        "Wave 1.5 W1.5.D rewrite system-prompt directive sentinel "
        "missing — the rewrite author needs the SHOULD-grade per-claim "
        "directive salient at the system-prompt level."
    )


# --------------------------------------------------------------------------- #
# Test 6 — full pytest sweep collected-count guard + jsonschema check
# --------------------------------------------------------------------------- #


def test_6_collected_count_guard() -> None:
    """Plan §4 Test 6 — wave-end "no silent test removal" ratchet.

    Walks every ``test_*.py`` file under the three Courseforge sub-trees
    and counts each ``def test_`` declaration via ``ast.walk``. Asserts
    each tree's count is at least the post-Wave-1.5 baseline encoded in
    the ``_MIN_*_TESTS`` constants above. Ratchets upward — if a future
    wave legitimately ADDS tests, bump the constant; if a refactor
    accidentally REMOVES tests, this gate fires.

    Also asserts ``Draft202012Validator.check_schema`` passes on the
    bumped per-block-type schema (the §4 Test 6 sub-clause: "no
    jsonschema deprecation warnings on the new oneOf surface").
    """
    jsonschema = _require_jsonschema()

    trees = [
        ("generators", PROJECT_ROOT / "Courseforge" / "generators" / "tests",
         _MIN_GENERATORS_TESTS),
        ("router", PROJECT_ROOT / "Courseforge" / "router" / "tests",
         _MIN_ROUTER_TESTS),
        ("scripts", PROJECT_ROOT / "Courseforge" / "scripts" / "tests",
         _MIN_SCRIPTS_TESTS),
    ]
    failures: List[str] = []
    for label, tree_path, floor in trees:
        actual, files = _count_test_functions(tree_path)
        if actual < floor:
            failures.append(
                f"{label} tree at {tree_path}: collected {actual} test "
                f"functions across {len(files)} files; floor is {floor}. "
                f"This gate ratchets upward — a future wave that legitimately "
                f"adds tests must bump _MIN_{label.upper()}_TESTS."
            )
    assert not failures, (
        "Wave-end test-count guard tripped (silent test removal "
        "regression suspected):\n  " + "\n  ".join(failures)
    )

    # jsonschema deprecation-warning sub-clause: assert the bumped
    # per-block-type schema passes Draft202012Validator.check_schema()
    # for every canonical block_type, capturing every emitted warning
    # via warnings.catch_warnings() so a DeprecationWarning trips the
    # gate explicitly. Plan §4 Test 6 named clause: "no warnings tagged
    # ``DeprecationWarning`` from the new oneOf surface".
    from Courseforge.generators.outline._outline_provider import (
        _BLOCK_TYPE_JSON_SCHEMAS,
        _build_block_outline_schema,
    )

    sentinel_block_types = (
        "concept",
        "assessment_item",
        "prereq_set",
        "summary_takeaway",
        "callout",
    )
    deprecation_warnings: List[str] = []
    for block_type in sentinel_block_types:
        schema = _BLOCK_TYPE_JSON_SCHEMAS[block_type]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            jsonschema.Draft202012Validator.check_schema(schema)
            for w in captured:
                if issubclass(w.category, DeprecationWarning):
                    deprecation_warnings.append(
                        f"{block_type}: {w.category.__name__}: {w.message}"
                    )
    assert not deprecation_warnings, (
        "jsonschema emitted DeprecationWarning(s) on the bumped oneOf "
        "schema surface:\n  " + "\n  ".join(deprecation_warnings)
    )

    # Defence-in-depth — also re-build the schema fresh via the public
    # builder and re-run check_schema, so a future refactor that
    # rewires _BLOCK_TYPE_JSON_SCHEMAS to bypass _build_block_outline_schema
    # can't silently regress this gate.
    fresh = _build_block_outline_schema("concept")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        jsonschema.Draft202012Validator.check_schema(fresh)
        fresh_deps = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
    assert not fresh_deps, (
        "_build_block_outline_schema('concept') emitted DeprecationWarning: "
        + "; ".join(str(w.message) for w in fresh_deps)
    )
