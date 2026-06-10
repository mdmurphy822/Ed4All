"""GPT Feedback v2 — Wave 1.7 end-of-wave block-against-objective gate.

Authored 2026-05-06 against the closing 6-test gate enumerated in
the Wave 1.7 block-against-objective gate spec § 4.

Predecessors landed:

* W1.7.A at 1666f4f — ``Block.objective_alignment`` audit field +
  bumped JSON-LD ``$defs.Block`` / ``$defs.ObjectiveAlignment``.
* W1.7.B at 9b3fa56 — outline + rewrite prompts surface the Bloom
  triple ``[Bloom: <level>, verb: <verb>]`` per objective +
  behavioral-outcome system-prompt directives.
* W1.7.C at 6fea5c2 — :class:`lib.validators.block_objective_delivery.
  BlockObjectiveDeliveryValidator` (tri-axis NLI / Bloom / verb) plus
  Drift A/B fixes (gate-input routing) and the
  :class:`lib.classifiers.nli_classifier.NliClassifier` stub.
* W1.7.D at 6dbbad9 — rewrite-tier remediation suffix dispatch on the
  three Wave 1.7 issue codes plus the
  ``per_claim_attribution_unfixable``-style escalation marker.

This file is a SELF-CONTAINED contract test mirroring the Wave 1.5 +
Wave 1.6 predecessor gates (``test_wave15_per_claim_attribution_contract.py``,
``test_wave16_per_objective_attribution_contract.py``). Every fixture
is inlined locally so a future rename of a per-worker test file cannot
silently disable the wave-end gate.

Drift notes vs plan §4:

* Plan §4 Test 2 reads "``result.passed is False``", but the W1.7.C
  validator ships every Wave 1.7 GateIssue at ``severity="warning"``
  by construction (Day-1 contract — promotion to critical is a
  Wave 3 follow-up after RDF/SHACL calibration corpus calibration). The
  ``passed`` flag stays ``True``; the regeneration signal the
  router actually consumes is ``action="regenerate"`` plus the
  warning-severity GateIssue code. We pin the ``action`` + code +
  rationale interpolation here per the plan §4 Test 2 explicit
  contract amendment in the W1.7 brief. Same drift Wave 1.5 / 1.6
  wave-end gates absorbed.
* Plan §4 Test 5 reads "Use the existing
  ``_run_rewrite_with_failing_validator`` test harness"; the actual
  W1.7.D dispatch surface is
  :func:`Courseforge.router.remediation._build_wave17_directive`
  (and its three per-axis sub-builders). We test the dispatch
  directly per the W1.7 brief — this is more durable than mocking
  the rewrite-provider call path.

Tests:

* ``test_1_jsonld_schema_admits_new_and_legacy_shapes`` — pin the
  bumped ``$defs.Block`` / ``$defs.ObjectiveAlignment`` Draft-2020-12
  schema: legacy shape (no ``objectiveAlignment``) validates,
  populated shape validates, status-out-of-enum FAILS, missing
  ``objective_id`` FAILS, ``statement_entailment_score`` outside
  [0, 1] FAILS.
* ``test_2_anti_silent_degradation_intentional_bloom_mismatch_fires``
  — ``BlockObjectiveDeliveryValidator`` fires
  ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` on a gap-4 fixture; rationale
  interpolates ``bloom_gap=4`` + ``declared_bloom='create'`` +
  ``observed_bloom='remember'``.
* ``test_3_legacy_blocks_skip_bloom_axis_cleanly`` — pre-Wave-1
  Block with ``observed_bloom_level=None`` skips the Bloom axis;
  the Bloom-axis decision-capture event carries
  ``status=unverifiable``; the verb axis still runs and emits a
  decision event.
* ``test_4_outline_and_rewrite_prompts_carry_bloom_triple_verbatim``
  — both ``OutlineProvider._render_user_prompt`` and
  ``RewriteProvider._render_user_prompt`` render the verbatim
  ``"[Bloom: create, verb: design]"`` substring + the objective
  statement; both system prompts carry their respective
  behavioral-outcome directives.
* ``test_5_remediation_dispatch_emits_distinct_per_code_directives``
  — each of the three Wave 1.7 issue codes produces a distinct
  remediation suffix substring keyed off the validator's
  f-string-formatted GateIssue.message context (objective_id,
  scores, Bloom levels, verb synonyms).
* ``test_6_collected_count_guard_and_check_schema_clean`` —
  wave-end "no silent test removal" ratchet across four trees
  (``lib/validators/tests/``, ``schemas/tests/``,
  ``Courseforge/router/tests/``, ``Courseforge/generators/tests/``);
  ``Draft202012Validator.check_schema(...)`` clean on the bumped
  ``$defs.Block`` AND ``$defs.ObjectiveAlignment``; AST-walk
  cross-check that the four pre-existing statistical-tier gate
  test files still carry their post-Wave-1.7 baseline counts.
"""
from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Block lives at ``Courseforge/scripts/blocks.py`` — mirror the import
# bridge used by the Courseforge router test suite + Wave 1.5/1.6 gates.
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Audited post-Wave-1.7 baselines — encoded as constants so a future
# silent test removal trips the ratchet. Counts captured 2026-05-06 on
# dev-v0.3.0 at HEAD by AST-walking each tree's ``test_*.py`` files
# and tallying every ``def test_`` declaration (the same algorithm
# ``_count_test_functions`` runs at gate time). Mirrors the
# AST-vs-collect-only contract noted in the Wave 1.5 / 1.6 gates: AST
# counts are LOWER than ``pytest --collect-only`` counts because
# pytest expands parametrized cases at collection time. Both
# algorithms are monotonic so the ratchet contract holds either way.
# --------------------------------------------------------------------------- #

_MIN_VALIDATORS_TESTS = 479
_MIN_SCHEMAS_TESTS = 339
_MIN_ROUTER_TESTS = 154
_MIN_GENERATORS_TESTS = 87

#: Per-file floors for the four pre-existing statistical-tier gate
#: test suites — Test 6 sub-clause: confirm Drift-B (W1.7.C
#: gate-input routing fix) didn't silently remove a test from any of
#: the four files the Drift-B fix touched the input shape of.
_MIN_STATISTICAL_TIER_TEST_COUNTS: Dict[str, int] = {
    "lib/validators/tests/test_objective_assessment_similarity.py": 9,
    "lib/validators/tests/test_concept_example_similarity.py": 11,
    "lib/validators/tests/test_objective_roundtrip_similarity.py": 12,
    "lib/validators/tests/test_bloom_classifier_disagreement.py": 14,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _load_jsonld_schema_doc() -> Dict[str, Any]:
    """Read the bumped ``courseforge_jsonld_v1.schema.json`` from disk."""
    schema_path = (
        PROJECT_ROOT
        / "schemas"
        / "knowledge"
        / "courseforge_jsonld_v1.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _build_block_def_validator(jsonschema_mod, schema_doc):
    """Return a Draft-2020-12 validator scoped to ``$defs.Block``.

    Wraps the Block sub-schema in a thin envelope so ``$ref`` references
    inside the schema (e.g. to ``$defs.ObjectiveAlignment``,
    ``$defs.Touch``, ``bloom_verbs.schema.json``) resolve through the
    parent document.

    The returned validator validates a single Block payload directly
    via ``validator.validate(payload)``.
    """
    # Build a tiny envelope schema that inlines ``$defs`` from the parent
    # document so ``#/$defs/...`` refs resolve, and whose top-level
    # ``$ref`` points at the Block sub-schema. This avoids depending on
    # jsonschema's $RefResolver API surface (which churned across
    # 4.0 → 4.18 — Draft-2020-12-friendly Registry pattern is the
    # cross-version-stable path).
    envelope = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "#/$defs/Block",
        "$defs": schema_doc.get("$defs", {}),
    }
    return jsonschema_mod.Draft202012Validator(envelope)


def _block_payload(
    *,
    objective_alignment=None,
    drop_alignment: bool = True,
) -> Dict[str, Any]:
    """Build a minimally-valid Block JSON-LD payload.

    ``drop_alignment=True`` (default) omits ``objectiveAlignment``
    entirely (legacy / pre-Wave-1.7 shape).
    """
    base: Dict[str, Any] = {
        "blockId": "page_01#concept_intro_0",
        "blockType": "concept",
        "sequence": 0,
    }
    if not drop_alignment:
        base["objectiveAlignment"] = objective_alignment
    return base


def _alignment_entry(
    *,
    objective_id: str = "CO-08",
    declared_bloom: str = "create",
    status: str = "delivered",
    observed_bloom: Any = "create",
    statement_entailment_score: Any = 0.85,
    action_verb_present: bool = True,
    drop_keys: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Build a minimally-valid ObjectiveAlignment entry."""
    entry: Dict[str, Any] = {
        "objective_id": objective_id,
        "declared_bloom": declared_bloom,
        "status": status,
        "observed_bloom": observed_bloom,
        "statement_entailment_score": statement_entailment_score,
        "action_verb_present": action_verb_present,
    }
    for k in drop_keys:
        entry.pop(k, None)
    return entry


def _count_test_functions(tree: Path) -> Tuple[int, List[str]]:
    """AST-walk every ``test_*.py`` under ``tree`` and count ``def test_``
    declarations.

    Mirrors the helper in ``test_wave15_per_claim_attribution_contract.py``
    + ``test_wave16_per_objective_attribution_contract.py`` so the
    algorithm is identical between gates and the count contract is
    genuinely monotonic across waves.
    """
    if not tree.exists():
        return 0, []
    total = 0
    seen_files: List[str] = []
    for path in sorted(tree.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            module = ast.parse(path.read_text())
        except (SyntaxError, OSError):
            continue
        seen_files.append(str(path.relative_to(PROJECT_ROOT)))
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    total += 1
    return total, seen_files


def _count_tests_in_file(path: Path) -> int:
    """AST-walk one ``test_*.py`` file and count ``def test_``."""
    if not path.exists():
        return 0
    try:
        module = ast.parse(path.read_text())
    except (SyntaxError, OSError):
        return 0
    return sum(
        1
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


class _RecordingCapture:
    """Minimal capture stub recording every ``log_decision`` payload.

    Inlined so a rename of any per-worker capture helper can't silently
    break this gate. Mirrors the
    ``test_wave15_per_claim_attribution_contract.py::_RecordingCapture``
    pattern (attribute name preserved for cross-wave grep symmetry).
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Test 1 — JSON-LD schema admits new + legacy shapes; rejects malformed
# --------------------------------------------------------------------------- #


def test_1_jsonld_schema_admits_new_and_legacy_shapes() -> None:
    """Plan §4 Test 1 — bumped ``$defs.Block`` /
    ``$defs.ObjectiveAlignment`` contract.

    Five sub-clauses:

    * 1a. Legacy Block payload (no ``objectiveAlignment`` field at all)
      validates clean.
    * 1b. Block payload with a non-empty ``objectiveAlignment[]`` of
      well-formed entries validates clean.
    * 1c. ``status`` outside the canonical 4-value enum
      (``delivered`` / ``underdelivered`` / ``verb_only`` /
      ``unverifiable``) FAILS validation.
    * 1d. Entry missing required ``objective_id`` FAILS.
    * 1e. ``statement_entailment_score`` outside ``[0, 1]`` FAILS.
    """
    jsonschema = _require_jsonschema()
    schema_doc = _load_jsonld_schema_doc()
    validator = _build_block_def_validator(jsonschema, schema_doc)

    # Sub-test 1a — legacy Block, no objectiveAlignment field.
    legacy = _block_payload(drop_alignment=True)
    validator.validate(legacy)

    # Sub-test 1b — populated objectiveAlignment validates.
    populated = _block_payload(
        drop_alignment=False,
        objective_alignment=[
            _alignment_entry(
                objective_id="CO-08",
                declared_bloom="create",
                status="delivered",
                observed_bloom="create",
                statement_entailment_score=0.92,
                action_verb_present=True,
            ),
            _alignment_entry(
                objective_id="TO-01",
                declared_bloom="apply",
                status="underdelivered",
                observed_bloom="remember",
                statement_entailment_score=0.30,
                action_verb_present=False,
            ),
        ],
    )
    validator.validate(populated)

    # Sub-test 1c — status outside the canonical 4-value enum FAILS.
    bad_status = _block_payload(
        drop_alignment=False,
        objective_alignment=[
            _alignment_entry(status="bogus_status_value"),
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad_status)

    # Sub-test 1d — missing required objective_id FAILS.
    missing_obj_id = _block_payload(
        drop_alignment=False,
        objective_alignment=[
            _alignment_entry(drop_keys=("objective_id",)),
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(missing_obj_id)

    # Sub-test 1e — statement_entailment_score outside [0, 1] FAILS.
    score_too_high = _block_payload(
        drop_alignment=False,
        objective_alignment=[
            _alignment_entry(statement_entailment_score=1.5),
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(score_too_high)

    score_negative = _block_payload(
        drop_alignment=False,
        objective_alignment=[
            _alignment_entry(statement_entailment_score=-0.1),
        ],
    )
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(score_negative)


# --------------------------------------------------------------------------- #
# Test 2 — anti-silent-degradation: intentional Bloom mismatch fires
# --------------------------------------------------------------------------- #


def test_2_anti_silent_degradation_intentional_bloom_mismatch_fires() -> None:
    """Plan §4 Test 2 — gap-5 Bloom mismatch fires
    ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` with ``action="regenerate"``.

    Drift A vs plan §4 Test 2: the W1.7.C validator ships every Wave 1.7
    GateIssue at warning severity by construction (Day-1 contract,
    plan §6.1). The ``passed`` flag stays True; the regeneration signal
    is ``action="regenerate"`` plus the warning-severity GateIssue
    code. Pin both per the W1.7 brief amendment — same drift Wave 1.5
    / 1.6 wave-end gates absorbed.

    Drift B vs plan §4 Test 2: plan brief reads ``bloom_gap=4``, but
    the canonical Bloom enum (``lib/ontology/bloom.BLOOM_LEVELS``)
    is 6 levels (``remember`` idx 0 … ``create`` idx 5), so the
    declared-vs-observed gap for ``create`` → ``remember`` is 5, not
    4. The validator interpolates the actual numeric gap into the
    rationale; we pin the truth, not the brief.
    """
    from blocks import Block  # noqa: WPS433  (path-injected module)
    from lib.validators.block_objective_delivery import (
        BlockObjectiveDeliveryValidator,
        _CODE_BLOOM_UNDERMET,
    )

    block = Block(
        block_id="page_01#concept_rdfs_subclass_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "statement": (
                "RDF Schema defines rdfs:subClassOf as a property."
            ),
        },
        objective_ids=("CO-08",),
        observed_bloom_level="remember",
    )
    objectives = {
        "CO-08": {
            "id": "CO-08",
            "bloom_level": "create",
            "bloom_verb": "construct",
            "statement": (
                "Construct subclass and subproperty hierarchies in Turtle."
            ),
        },
    }

    capture = _RecordingCapture()
    # No NLI override — get_or_load() returns None per the W1.7.C stub
    # so the entailment axis silently graceful-degrades; Bloom + verb
    # axes still run (which is exactly what this test is exercising).
    validator = BlockObjectiveDeliveryValidator()
    result = validator.validate({
        "blocks": [block],
        "objectives": objectives,
        "decision_capture": capture,
    })

    # Drift vs plan §4: Wave 1.7 ships warning-only; ``passed`` stays
    # True. The router-consumed signal is ``action="regenerate"``.
    assert result.action == "regenerate", (
        f"expected action='regenerate' on Bloom-gap=4 miss; "
        f"got action={result.action!r}"
    )
    codes = [i.code for i in result.issues]
    assert _CODE_BLOOM_UNDERMET in codes, (
        f"expected at least one issue with code {_CODE_BLOOM_UNDERMET!r}; "
        f"got codes={codes!r}"
    )

    # Decision-capture event MUST carry bloom_gap=4 +
    # declared_bloom='create' + observed_bloom='remember' in the
    # rationale. Validator interpolates these via the
    # ``_emit_decision`` helper at lib/validators/block_objective_delivery.py
    # `:316-329`.
    relevant = [
        e for e in capture.events
        if e.get("decision_type") == "block_objective_delivery_check"
        and "CO-08" in str(e.get("rationale", ""))
    ]
    assert relevant, (
        "expected at least one block_objective_delivery_check event "
        "for CO-08; got events: "
        f"{[e.get('decision_type') for e in capture.events]!r}"
    )
    bloom_event_rationales = [str(e.get("rationale", "")) for e in relevant]
    assert any("bloom_gap=5" in r for r in bloom_event_rationales), (
        "decision-capture rationale MUST surface bloom_gap=5 "
        "(create-idx-5 minus remember-idx-0 = 5; plan brief said "
        "4 but the canonical Bloom enum has 6 levels — pin the "
        "truth, not the brief); got rationales: "
        f"{bloom_event_rationales!r}"
    )
    assert any("declared_bloom='create'" in r for r in bloom_event_rationales), (
        "decision-capture rationale MUST surface "
        "declared_bloom='create'; "
        f"got rationales: {bloom_event_rationales!r}"
    )
    assert any(
        "observed_bloom='remember'" in r for r in bloom_event_rationales
    ), (
        "decision-capture rationale MUST surface "
        "observed_bloom='remember'; "
        f"got rationales: {bloom_event_rationales!r}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — anti-silent-degradation: legacy blocks skip Bloom axis cleanly
# --------------------------------------------------------------------------- #


def test_3_legacy_blocks_skip_bloom_axis_cleanly() -> None:
    """Plan §4 Test 3 — pre-Wave-1 Block with ``observed_bloom_level=None``
    silently skips the Bloom axis.

    Catches the regression where a future author tightens the Bloom
    check to require ``observed_bloom_level`` be non-None — silently
    breaking every existing corpus.

    Block prose contains ``"design"`` (a ``create``-level synonym) so
    the verb axis fires and passes. NLI loader graceful-degrades
    (W1.7.C stub returns None; entailment axis emits the
    ``BLOCK_OBJECTIVE_NLI_DEPS_MISSING`` warning + ``passed=True,
    action=None``). Net outcome: result.passed=True, no
    ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` issue, the per-pair
    decision-capture event carries ``status=unverifiable`` (because
    the Bloom axis SKIPPED while no axis fired a real miss).
    """
    from blocks import Block  # noqa: WPS433
    from lib.validators.block_objective_delivery import (
        BlockObjectiveDeliveryValidator,
        _CODE_BLOOM_UNDERMET,
    )

    # Prose contains "design" — a canonical ``create``-level Bloom
    # synonym — so the verb axis passes. NO observed_bloom_level
    # attached → Bloom axis silently skips.
    block = Block(
        block_id="page_01#concept_legacy_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "statement": (
                "Design RDF Schema vocabularies that describe subclass "
                "and subproperty hierarchies in Turtle."
            ),
        },
        objective_ids=("CO-08",),
        observed_bloom_level=None,
    )
    objectives = {
        "CO-08": {
            "id": "CO-08",
            "bloom_level": "create",
            "bloom_verb": "design",
            "statement": (
                "Design subclass and subproperty hierarchies in Turtle."
            ),
        },
    }

    capture = _RecordingCapture()
    validator = BlockObjectiveDeliveryValidator()
    result = validator.validate({
        "blocks": [block],
        "objectives": objectives,
        "decision_capture": capture,
    })

    # Result MUST pass: Bloom axis skipped, verb axis passed,
    # entailment axis graceful-degraded (NLI=None).
    assert result.passed is True, (
        f"expected legacy block (observed_bloom_level=None) to pass; "
        f"got result.passed={result.passed!r}, action={result.action!r}, "
        f"issue_codes={[i.code for i in result.issues]!r}"
    )

    # Bloom axis MUST NOT fire — silently skipped on the legacy block.
    codes = [i.code for i in result.issues]
    assert _CODE_BLOOM_UNDERMET not in codes, (
        f"Bloom axis must silently skip when observed_bloom_level=None; "
        f"got codes={codes!r}"
    )

    # Per-pair decision-capture event MUST carry status=unverifiable
    # (the Bloom axis skipped while no axis fired a real miss; status
    # resolution per ``_resolve_status`` falls through to
    # ``_STATUS_UNVERIFIABLE`` when ``has_skip and not any(real_misses)``).
    pair_events = [
        e for e in capture.events
        if e.get("decision_type") == "block_objective_delivery_check"
        and "CO-08" in str(e.get("rationale", ""))
    ]
    assert pair_events, (
        "expected at least one per-pair "
        "block_objective_delivery_check event for CO-08; "
        f"got: {[e.get('decision_type') for e in capture.events]!r}"
    )
    rationales = [str(e.get("rationale", "")) for e in pair_events]
    assert any("status=unverifiable" in r for r in rationales), (
        "per-pair decision-capture event MUST carry "
        "status=unverifiable when Bloom axis silently skipped while "
        "no axis fired a real miss; got rationales: "
        f"{rationales!r}"
    )

    # Defence-in-depth — the verb axis MUST have run on the same pair.
    # The ``verb_match_count=`` interpolation is present per the
    # ``_emit_decision`` helper (str(verb_match_count) when not None).
    # Block prose contains "design" → at least one match.
    assert any("verb_match_count=" in r for r in rationales), (
        "verb-axis decision-capture event MUST carry "
        "verb_match_count= interpolation (the verb axis still runs "
        "even when Bloom is skipped); got rationales: "
        f"{rationales!r}"
    )
    assert any(
        "verb_match_count=0" not in r and "verb_match_count=" in r
        for r in rationales
    ), (
        "the test fixture's prose contains 'design' (a canonical "
        "create-level Bloom synonym) so verb_match_count MUST be > 0; "
        f"got rationales: {rationales!r}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — outline + rewrite prompts contain Bloom triple verbatim
# --------------------------------------------------------------------------- #


def test_4_outline_and_rewrite_prompts_carry_bloom_triple_verbatim(
    monkeypatch,
) -> None:
    """Plan §4 Test 4 — both outline and rewrite tier prompts surface
    the ``[Bloom: <level>, verb: <verb>]`` triple verbatim plus their
    behavioral-outcome system-prompt directives.

    Four sentinel checks:

    * 4a. ``OutlineProvider._render_user_prompt`` returns a string
      containing ``"[Bloom: create, verb: design]"`` AND the objective
      statement verbatim.
    * 4b. ``RewriteProvider._render_user_prompt`` ditto.
    * 4c. ``_OUTLINE_SYSTEM_PROMPT`` carries the
      ``"MUST be at or above the declared Bloom"`` directive sentinel.
    * 4d. ``_REWRITE_SYSTEM_PROMPT`` carries the
      ``"MUST teach the BEHAVIORAL OUTCOME"`` directive sentinel.
    """
    from blocks import Block  # noqa: WPS433
    from Courseforge.generators._outline_provider import (
        OutlineProvider,
        _OUTLINE_SYSTEM_PROMPT,
    )
    from Courseforge.generators._rewrite_provider import (
        RewriteProvider,
        _REWRITE_SYSTEM_PROMPT,
    )

    # Sub-test 4c — outline system prompt directive sentinel.
    assert "MUST be at or above the declared Bloom" in _OUTLINE_SYSTEM_PROMPT, (
        "Wave 1.7 W1.7.B outline-system-prompt behavioral-outcome "
        "directive sentinel 'MUST be at or above the declared Bloom' "
        "missing — the outline-tier model needs the Bloom-floor "
        "directive surfaced verbatim at the system-prompt level."
    )

    # Sub-test 4d — rewrite system prompt directive sentinel.
    assert "MUST teach the BEHAVIORAL OUTCOME" in _REWRITE_SYSTEM_PROMPT, (
        "Wave 1.7 W1.7.B rewrite-system-prompt behavioral-outcome "
        "directive sentinel 'MUST teach the BEHAVIORAL OUTCOME' "
        "missing — the rewrite-tier model needs the behavioral-"
        "outcome directive surfaced verbatim at the system-prompt "
        "level."
    )

    # Build the fixture for sub-tests 4a + 4b.
    block = Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "key_claims": [
                "RDF Schema defines a vocabulary for class hierarchies."
            ],
            "curies": ["rdfs:subClassOf"],
            "source_refs": ["dart:rdfs-spec#sec1"],
            "objective_refs": ["CO-08"],
        },
    )
    chunks = [
        {"id": "dart:rdfs-spec#sec1", "body": "RDFS subclass body."},
    ]
    objective_statement = (
        "Design RDF Schema vocabularies for subclass hierarchies."
    )
    objectives = [
        {
            "id": "CO-08",
            "statement": objective_statement,
            "bloom_level": "create",
            "bloom_verb": "design",
        },
    ]

    # Sub-test 4a — outline-tier user-prompt rendering.
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    outline_provider = OutlineProvider(provider="local")
    rendered_outline = outline_provider._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives,
    )
    assert "[Bloom: create, verb: design]" in rendered_outline, (
        "Wave 1.7 W1.7.B outline-tier _render_user_prompt MUST surface "
        "the verbatim '[Bloom: create, verb: design]' triple inline "
        "with each objective. The 7B-class outline model has no "
        "structural way to recover the declared cognitive demand "
        "without this inline annotation."
    )
    assert objective_statement in rendered_outline, (
        "Wave 1.7 W1.7.B outline-tier _render_user_prompt MUST render "
        "the objective statement verbatim alongside the Bloom triple."
    )

    # Sub-test 4b — rewrite-tier user-prompt rendering. Use the
    # Wave 1.5 Test 5 pattern: api_key env var + an inert
    # ``anthropic_client=object()`` to bypass any real network call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    rewrite_provider = RewriteProvider(anthropic_client=object())
    rendered_rewrite = rewrite_provider._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives,
    )
    assert "[Bloom: create, verb: design]" in rendered_rewrite, (
        "Wave 1.7 W1.7.B rewrite-tier _render_user_prompt MUST surface "
        "the verbatim '[Bloom: create, verb: design]' triple inline "
        "with each objective. Symmetric with the outline-tier rendering."
    )
    assert objective_statement in rendered_rewrite, (
        "Wave 1.7 W1.7.B rewrite-tier _render_user_prompt MUST render "
        "the objective statement verbatim alongside the Bloom triple."
    )


# --------------------------------------------------------------------------- #
# Test 5 — rewrite remediation: each Wave 1.7 issue code injects its own suffix
# --------------------------------------------------------------------------- #


def test_5_remediation_dispatch_emits_distinct_per_code_directives() -> None:
    """Plan §4 Test 5 — each Wave 1.7 issue code produces a distinct
    remediation-suffix directive substring.

    Source surface: :func:`Courseforge.router.remediation.
    _append_remediation_for_gates` plus its
    :func:`_format_failure_block` per-issue dispatch helper. Tests
    drive the suffix builder directly via three synthetic GateResults
    whose issues each carry one of the three Wave 1.7 codes plus a
    realistic message body matching the validator's f-string format
    (the regex extractors in ``remediation.py`` parse against this
    format).

    Asserts each code produces a DISTINCT directive substring:

    * ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` → contains
      ``"bloom_level"`` + ``"levels below"`` + the bloom_gap value.
    * ``BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED`` → contains
      ``"did not"`` + ``"semantically support"`` + the entailment
      score.
    * ``BLOCK_OBJECTIVE_VERB_ABSENT`` → contains ``"no synonym"`` +
      the bloom_verb value.

    Defence-in-depth: the three suffixes are MUTUALLY exclusive on
    these substrings — the BLOOM directive does NOT contain
    ``"semantically support"`` etc. so a future copy-shuffle that
    accidentally collapses two directives into one trips this gate.
    """
    from MCP.hardening.validation_gates import GateIssue, GateResult
    from Courseforge.router.remediation import _append_remediation_for_gates

    base_prompt = "Author the rendered HTML body for this block now."

    # ------------------------------------------------------------------
    # Sub-test 5a — BLOCK_OBJECTIVE_BLOOM_UNDERMET dispatch.
    # ------------------------------------------------------------------
    # Validator emits messages of the form:
    #   "Block 'page_01#concept_intro_0' (block_type='concept') "
    #   "bloom_level='remember' is 4 levels below the declared "
    #   "objective 'CO-08''s bloom_level='create'. ..."
    # Per ``_RE_OBSERVED_BLOOM`` / ``_RE_BLOOM_GAP`` /
    # ``_RE_DECLARED_BLOOM`` regex shapes in remediation.py.
    bloom_message = (
        "Block 'page_01#concept_intro_0' (block_type='concept') "
        "bloom_level='remember' is 4 levels below the declared "
        "objective 'CO-08''s bloom_level='create'. Re-emit at or "
        "above the declared level — scaffold up to the declared "
        "cognitive demand."
    )
    bloom_failure = GateResult(
        gate_id="rewrite_block_objective_delivery",
        validator_name="block_objective_delivery",
        validator_version="1.0.0",
        passed=True,  # Wave 1.7 ships warning-only; passed stays True.
        issues=[
            GateIssue(
                severity="warning",
                code="BLOCK_OBJECTIVE_BLOOM_UNDERMET",
                message=bloom_message,
            ),
        ],
        action="regenerate",
    )
    bloom_suffix = _append_remediation_for_gates(
        base_prompt, [bloom_failure],
    )
    assert bloom_suffix != base_prompt, (
        "BLOCK_OBJECTIVE_BLOOM_UNDERMET MUST trigger a non-empty "
        "remediation suffix; got byte-identical base prompt back."
    )
    # Required substrings for the BLOOM directive.
    assert "bloom_level" in bloom_suffix
    assert "levels below" in bloom_suffix
    assert "4" in bloom_suffix, (
        "BLOOM_UNDERMET directive MUST surface the parsed bloom_gap "
        "value verbatim (4); got: " + bloom_suffix
    )
    assert "'create'" in bloom_suffix, (
        "BLOOM_UNDERMET directive MUST name the declared_bloom "
        "level verbatim (create); got: " + bloom_suffix
    )
    assert "'remember'" in bloom_suffix, (
        "BLOOM_UNDERMET directive MUST name the observed_bloom "
        "level verbatim (remember); got: " + bloom_suffix
    )

    # ------------------------------------------------------------------
    # Sub-test 5b — BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED dispatch.
    # ------------------------------------------------------------------
    # Validator emits messages of the form:
    #   "Block 'page_01#concept_intro_0' (block_type='concept') "
    #   "prose did not entail objective 'CO-08'. NLI entailment="
    #   "0.1234 (floor 0.4000); contradiction=0.7800 (floor 0.5000)."
    # Per ``_RE_OBJECTIVE_ID`` / ``_RE_ENTAILMENT_SCORE`` /
    # ``_RE_ENTAILMENT_FLOOR`` regex shapes in remediation.py.
    statement_message = (
        "Block 'page_01#concept_intro_0' (block_type='concept') "
        "prose did not entail objective 'CO-08'. NLI entailment="
        "0.1234 (floor 0.4000); contradiction=0.7800 (floor 0.5000)."
    )
    statement_failure = GateResult(
        gate_id="rewrite_block_objective_delivery",
        validator_name="block_objective_delivery",
        validator_version="1.0.0",
        passed=True,
        issues=[
            GateIssue(
                severity="warning",
                code="BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED",
                message=statement_message,
            ),
        ],
        action="regenerate",
    )
    statement_suffix = _append_remediation_for_gates(
        base_prompt, [statement_failure],
        objective_statements={
            "CO-08": (
                "Construct subclass and subproperty hierarchies "
                "in Turtle."
            ),
        },
    )
    assert statement_suffix != base_prompt, (
        "BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED MUST trigger a "
        "non-empty remediation suffix; got byte-identical base "
        "prompt back."
    )
    assert "did not" in statement_suffix
    assert "semantically support" in statement_suffix
    assert "0.12" in statement_suffix, (
        "STATEMENT_UNDERSUPPORTED directive MUST surface the parsed "
        "entailment_score (0.12 after %.2f formatting); got: "
        + statement_suffix
    )
    assert "0.40" in statement_suffix, (
        "STATEMENT_UNDERSUPPORTED directive MUST surface the parsed "
        "entailment_floor (0.40 after %.2f formatting); got: "
        + statement_suffix
    )
    assert "'CO-08'" in statement_suffix, (
        "STATEMENT_UNDERSUPPORTED directive MUST name the failing "
        "objective_id verbatim (CO-08); got: " + statement_suffix
    )
    assert (
        "Construct subclass and subproperty hierarchies in Turtle."
        in statement_suffix
    ), (
        "STATEMENT_UNDERSUPPORTED directive MUST interpolate the "
        "objective statement from the objective_statements map; "
        "got: " + statement_suffix
    )

    # ------------------------------------------------------------------
    # Sub-test 5c — BLOCK_OBJECTIVE_VERB_ABSENT dispatch.
    # ------------------------------------------------------------------
    # Validator emits messages of the form:
    #   "Block 'page_01#concept_intro_0' (block_type='concept') "
    #   "prose contains no synonym of the objective 'CO-08''s "
    #   "bloom_verb='construct'. Re-emit prose using 'construct' "
    #   "or one of the 'create'-level synonyms (e.g. compose, "
    #   "construct, create, design, develop, formulate, generate, "
    #   "invent)."
    # Per ``_RE_BLOOM_VERB`` / ``_RE_VERB_SYNONYMS_PREVIEW`` regex
    # shapes in remediation.py.
    verb_message = (
        "Block 'page_01#concept_intro_0' (block_type='concept') "
        "prose contains no synonym of the objective 'CO-08''s "
        "bloom_verb='construct'. Re-emit prose using 'construct' "
        "or one of the 'create'-level synonyms (e.g. compose, "
        "construct, create, design, develop, formulate, generate, "
        "invent)."
    )
    verb_failure = GateResult(
        gate_id="rewrite_block_objective_delivery",
        validator_name="block_objective_delivery",
        validator_version="1.0.0",
        passed=True,
        issues=[
            GateIssue(
                severity="warning",
                code="BLOCK_OBJECTIVE_VERB_ABSENT",
                message=verb_message,
            ),
        ],
        action="regenerate",
    )
    verb_suffix = _append_remediation_for_gates(
        base_prompt, [verb_failure],
    )
    assert verb_suffix != base_prompt, (
        "BLOCK_OBJECTIVE_VERB_ABSENT MUST trigger a non-empty "
        "remediation suffix; got byte-identical base prompt back."
    )
    assert "no synonym" in verb_suffix
    assert "'construct'" in verb_suffix, (
        "VERB_ABSENT directive MUST name the bloom_verb verbatim "
        "(construct); got: " + verb_suffix
    )

    # ------------------------------------------------------------------
    # Sub-test 5d — directives are MUTUALLY exclusive on their key
    # discriminating substrings. A future copy-shuffle that collapses
    # two directives into one trips this gate.
    # ------------------------------------------------------------------
    assert "semantically support" not in bloom_suffix, (
        "BLOOM_UNDERMET directive must NOT contain "
        "'semantically support' (that is the STATEMENT directive's "
        "discriminator)."
    )
    assert "no synonym" not in bloom_suffix, (
        "BLOOM_UNDERMET directive must NOT contain 'no synonym' "
        "(that is the VERB directive's discriminator)."
    )
    assert "no synonym" not in statement_suffix, (
        "STATEMENT_UNDERSUPPORTED directive must NOT contain "
        "'no synonym' (that is the VERB directive's discriminator)."
    )
    assert "levels below" not in statement_suffix, (
        "STATEMENT_UNDERSUPPORTED directive must NOT contain "
        "'levels below' (that is the BLOOM directive's discriminator)."
    )
    assert "semantically support" not in verb_suffix, (
        "VERB_ABSENT directive must NOT contain 'semantically "
        "support' (that is the STATEMENT directive's discriminator)."
    )
    assert "levels below" not in verb_suffix, (
        "VERB_ABSENT directive must NOT contain 'levels below' "
        "(that is the BLOOM directive's discriminator)."
    )


# --------------------------------------------------------------------------- #
# Test 6 — collected-count guard + DeprecationWarning-clean
# --------------------------------------------------------------------------- #


def test_6_collected_count_guard_and_check_schema_clean() -> None:
    """Plan §4 Test 6 — wave-end "no silent test removal" ratchet PLUS
    Draft202012Validator.check_schema cleanliness on the bumped Block /
    ObjectiveAlignment schema.

    Walks every ``test_*.py`` under the four trees Wave 1.7 ratchets
    and counts each ``def test_`` declaration via ``ast.walk``. Asserts
    each tree's count is at least the post-Wave-1.7 baseline encoded in
    the ``_MIN_*_TESTS`` constants above. Ratchets upward — if a future
    wave legitimately ADDS tests, bump the constant; if a refactor
    accidentally REMOVES tests, this gate fires.

    Also asserts:

    * ``Draft202012Validator.check_schema`` passes on the bumped
      ``$defs.Block`` AND ``$defs.ObjectiveAlignment`` definitions
      with ZERO ``DeprecationWarning``s captured.
    * The four pre-existing statistical-tier gate test files still
      carry their post-Wave-1.7 baseline counts (Drift-B fix
      additivity check — the W1.7.C
      ``_build_block_statistical_input`` change must not have
      silently removed any test from those files).
    """
    jsonschema = _require_jsonschema()

    trees = [
        ("validators", PROJECT_ROOT / "lib" / "validators" / "tests",
         _MIN_VALIDATORS_TESTS),
        ("schemas", PROJECT_ROOT / "schemas" / "tests",
         _MIN_SCHEMAS_TESTS),
        ("router", PROJECT_ROOT / "Courseforge" / "router" / "tests",
         _MIN_ROUTER_TESTS),
        ("generators", PROJECT_ROOT / "Courseforge" / "generators" / "tests",
         _MIN_GENERATORS_TESTS),
    ]
    failures: List[str] = []
    for label, tree_path, floor in trees:
        actual, files = _count_test_functions(tree_path)
        if actual < floor:
            failures.append(
                f"{label} tree at {tree_path}: collected {actual} test "
                f"functions across {len(files)} files; floor is {floor}. "
                f"This gate ratchets upward — a future wave that "
                f"legitimately adds tests must bump _MIN_{label.upper()}_TESTS."
            )
    assert not failures, (
        "Wave-end test-count guard tripped (silent test removal "
        "regression suspected):\n  " + "\n  ".join(failures)
    )

    # Per-file floors for the four pre-existing statistical-tier gate
    # test files — Drift-B additivity check.
    per_file_failures: List[str] = []
    for rel_path, floor in _MIN_STATISTICAL_TIER_TEST_COUNTS.items():
        path = PROJECT_ROOT / rel_path
        actual = _count_tests_in_file(path)
        if actual < floor:
            per_file_failures.append(
                f"{rel_path}: collected {actual} test functions; "
                f"floor is {floor}. Drift-B fix in W1.7.C should be "
                f"additive — a regression would suggest a removed test."
            )
    assert not per_file_failures, (
        "Wave-end statistical-tier per-file ratchet tripped:\n  "
        + "\n  ".join(per_file_failures)
    )

    # check_schema sub-clause: capture every emitted warning and assert
    # ZERO are DeprecationWarnings on the bumped Block /
    # ObjectiveAlignment schema. Plan §4 Test 6 named clause: "no
    # warnings tagged DeprecationWarning from the new
    # objective_alignment field surface".
    schema_doc = _load_jsonld_schema_doc()
    deprecation_warnings: List[str] = []

    for def_name in ("Block", "ObjectiveAlignment"):
        # Wrap each sub-schema in a Draft-2020-12 envelope that inlines
        # the parent's ``$defs`` so internal ``$ref`` references
        # resolve. This matches how ``_build_block_def_validator``
        # builds the validator above.
        envelope = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": f"#/$defs/{def_name}",
            "$defs": schema_doc.get("$defs", {}),
        }
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            jsonschema.Draft202012Validator.check_schema(envelope)
            for w in captured:
                if issubclass(w.category, DeprecationWarning):
                    deprecation_warnings.append(
                        f"$defs.{def_name}: "
                        f"{w.category.__name__}: {w.message}"
                    )
    assert not deprecation_warnings, (
        "jsonschema emitted DeprecationWarning(s) on the bumped "
        "objectiveAlignment schema surface:\n  "
        + "\n  ".join(deprecation_warnings)
    )

    # Defence-in-depth — also re-check the WHOLE document so a future
    # refactor that splits the new defs out but leaves the per-def
    # shape intact still trips the gate if a deprecation surfaces at
    # the document level.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        jsonschema.Draft202012Validator.check_schema(schema_doc)
        whole_doc_deps = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
    assert not whole_doc_deps, (
        "Draft202012Validator.check_schema(whole-doc) emitted "
        "DeprecationWarning(s): "
        + "; ".join(str(w.message) for w in whole_doc_deps)
    )
