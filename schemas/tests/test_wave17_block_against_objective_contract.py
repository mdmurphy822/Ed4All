"""End-of-wave block-against-objective gate.

SELF-CONTAINED contract test: every fixture is inlined locally so a
future rename of a per-worker test file cannot silently disable this
gate.

Two contracts that read as surprising and are deliberate:

* ``BlockObjectiveDeliveryValidator`` ships every issue at
  ``severity="warning"``, so ``result.passed`` stays ``True`` even on a
  real miss. The signal the router consumes is ``action="regenerate"``
  plus the issue code — asserting on ``passed`` here would test nothing.
* The remediation dispatch is exercised through
  :func:`Courseforge.router.remediation._append_remediation_for_gates`
  directly rather than by mocking the rewrite-provider call path; the
  regex extractors in ``remediation.py`` parse the validator's
  f-string-formatted ``GateIssue.message``, so the synthetic messages
  below must keep that exact shape.

Tests:

* ``test_1_jsonld_schema_admits_new_and_legacy_shapes`` — ``$defs.Block``
  / ``$defs.ObjectiveAlignment``: legacy shape (no
  ``objectiveAlignment``) validates, populated shape validates,
  status-out-of-enum / missing ``objective_id`` /
  ``statement_entailment_score`` outside [0, 1] all FAIL.
* ``test_2_anti_silent_degradation_intentional_bloom_mismatch_fires``
  — ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` fires with the numeric gap and
  both Bloom levels interpolated into the decision-capture rationale.
* ``test_3_legacy_blocks_skip_bloom_axis_cleanly`` — a Block with
  ``observed_bloom_level=None`` skips the Bloom axis and records
  ``status=unverifiable``; the verb axis still runs.
* ``test_4_outline_and_rewrite_prompts_carry_bloom_triple_verbatim``
  — both tiers render the ``[Bloom: <level>, verb: <verb>]`` triple and
  the objective statement; both system prompts carry their
  behavioral-outcome directives.
* ``test_5_remediation_dispatch_emits_distinct_per_code_directives``
  — each issue code produces a distinct remediation suffix.
* ``test_6_collected_count_guard_and_check_schema_clean`` — test-count
  ratchet across four trees, ``check_schema`` cleanliness on the Block /
  ObjectiveAlignment surface, plus per-file floors on the four
  statistical-tier gate test files.
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

# Block lives at ``Courseforge/scripts/blocks.py``, which is not an
# importable package — bridge the directory onto sys.path.
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Per-tree test-count floors — a silent test removal trips the ratchet.
# Counts come from AST-walking each tree's ``test_*.py`` and tallying
# ``def test_`` declarations (the algorithm ``_count_test_functions``
# runs at gate time). AST counts are LOWER than
# ``pytest --collect-only`` counts (pytest expands parametrized cases at
# collection time), so the two must never be mixed across gates; each is
# monotonic on its own.
# --------------------------------------------------------------------------- #

_MIN_VALIDATORS_TESTS = 479
_MIN_SCHEMAS_TESTS = 339
_MIN_ROUTER_TESTS = 154
_MIN_GENERATORS_TESTS = 87

#: Per-file floors for the four statistical-tier gate test suites whose
#: gate-input shape the tri-axis validator changed — the tree-level
#: ratchet above would not notice a removal in one file offset by
#: additions elsewhere in the same tree.
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
    # Inline the parent's ``$defs`` into a tiny envelope whose top-level
    # ``$ref`` points at the Block sub-schema, so ``#/$defs/...`` refs
    # resolve. Deliberately avoids jsonschema's $RefResolver API, whose
    # surface is not stable across the 4.x line.
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
    entirely — the legacy shape.
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

    Must stay byte-identical to the counting helper in the sibling
    wave-end gates — mixing counting algorithms breaks the monotonic
    ratchet contract across gates.
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
    break this gate.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Test 1 — JSON-LD schema admits new + legacy shapes; rejects malformed
# --------------------------------------------------------------------------- #


def test_1_jsonld_schema_admits_new_and_legacy_shapes() -> None:
    """``$defs.Block`` / ``$defs.ObjectiveAlignment`` schema contract.

    * 1a. Legacy Block payload (no ``objectiveAlignment`` field at all)
      validates clean — the field must stay optional or every
      pre-existing corpus fails validation.
    * 1b. Block payload with a non-empty ``objectiveAlignment[]`` of
      well-formed entries validates clean.
    * 1c. ``status`` outside the canonical 4-value enum
      (``delivered`` / ``underdelivered`` / ``verb_only`` /
      ``unverifiable``) FAILS.
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
    """A declared-vs-observed Bloom mismatch fires
    ``BLOCK_OBJECTIVE_BLOOM_UNDERMET`` with ``action="regenerate"``.

    The validator emits at warning severity, so ``passed`` stays True —
    ``action`` plus the issue code are the only load-bearing signals.

    The expected gap is 5, not 4: the canonical Bloom enum
    (``lib/ontology/bloom.BLOOM_LEVELS``) has 6 levels (``remember``
    idx 0 … ``create`` idx 5), so ``create`` → ``remember`` spans 5.
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
    # No NLI override: the loader returns None, so the entailment axis
    # graceful-degrades and the Bloom + verb axes — what this test
    # exercises — still run.
    validator = BlockObjectiveDeliveryValidator()
    result = validator.validate({
        "blocks": [block],
        "objectives": objectives,
        "decision_capture": capture,
    })

    # Warning-only validator: ``passed`` stays True, so ``action`` is
    # the router-consumed signal.
    assert result.action == "regenerate", (
        f"expected action='regenerate' on a Bloom-gap miss; "
        f"got action={result.action!r}"
    )
    codes = [i.code for i in result.issues]
    assert _CODE_BLOOM_UNDERMET in codes, (
        f"expected at least one issue with code {_CODE_BLOOM_UNDERMET!r}; "
        f"got codes={codes!r}"
    )

    # The rationale must interpolate the numeric gap and both Bloom
    # levels (via the validator's ``_emit_decision`` helper) so an audit
    # can replay the decision without re-running the validator.
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
        "(create idx 5 minus remember idx 0 over the 6-level "
        "canonical Bloom enum); got rationales: "
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
    """A Block with ``observed_bloom_level=None`` skips the Bloom axis.

    Catches the regression where an author tightens the Bloom check to
    require ``observed_bloom_level`` be non-None — which would fail
    every legacy corpus, none of which carries the field.

    Block prose contains ``"design"`` (a ``create``-level synonym) so
    the verb axis fires and passes; the NLI loader graceful-degrades to
    a warning. Net outcome: ``passed=True``, no
    ``BLOCK_OBJECTIVE_BLOOM_UNDERMET``, and a per-pair decision-capture
    event carrying ``status=unverifiable`` — a skipped axis with no real
    miss must not be recorded as a pass.
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
    # ``verb_match_count=`` is only interpolated when the axis ran, and
    # the fixture prose contains "design", so the count must be > 0.
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
    """Both tiers surface the ``[Bloom: <level>, verb: <verb>]`` triple
    verbatim plus their behavioral-outcome system-prompt directives.

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
        "outline system prompt is missing the 'MUST be at or above the "
        "declared Bloom' directive — the outline-tier model needs the "
        "Bloom floor surfaced verbatim at the system-prompt level."
    )

    # Sub-test 4d — rewrite system prompt directive sentinel.
    assert "MUST teach the BEHAVIORAL OUTCOME" in _REWRITE_SYSTEM_PROMPT, (
        "rewrite system prompt is missing the 'MUST teach the "
        "BEHAVIORAL OUTCOME' directive — the rewrite-tier model needs "
        "it surfaced verbatim at the system-prompt level."
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
            "source_refs": ["semantik:rdfs-spec#sec1"],
            "objective_refs": ["CO-08"],
        },
    )
    chunks = [
        {"id": "semantik:rdfs-spec#sec1", "body": "RDFS subclass body."},
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
        "outline-tier _render_user_prompt MUST surface the verbatim "
        "'[Bloom: create, verb: design]' triple inline with each "
        "objective. A 7B-class outline model has no structural way to "
        "recover the declared cognitive demand without this inline "
        "annotation."
    )
    assert objective_statement in rendered_outline, (
        "outline-tier _render_user_prompt MUST render the objective "
        "statement verbatim alongside the Bloom triple."
    )

    # Sub-test 4b — rewrite-tier user-prompt rendering. The api_key env
    # var plus an inert ``anthropic_client=object()`` satisfy
    # construction without any real network call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    rewrite_provider = RewriteProvider(anthropic_client=object())
    rendered_rewrite = rewrite_provider._render_user_prompt(
        block=block, source_chunks=chunks, objectives=objectives,
    )
    assert "[Bloom: create, verb: design]" in rendered_rewrite, (
        "rewrite-tier _render_user_prompt MUST surface the verbatim "
        "'[Bloom: create, verb: design]' triple inline with each "
        "objective, symmetric with the outline tier."
    )
    assert objective_statement in rendered_rewrite, (
        "rewrite-tier _render_user_prompt MUST render the objective "
        "statement verbatim alongside the Bloom triple."
    )


# --------------------------------------------------------------------------- #
# Test 5 — rewrite remediation: each issue code injects its own suffix
# --------------------------------------------------------------------------- #


def test_5_remediation_dispatch_emits_distinct_per_code_directives() -> None:
    """Each issue code produces a distinct remediation-suffix directive.

    Drives :func:`Courseforge.router.remediation.
    _append_remediation_for_gates` directly via three synthetic
    GateResults. Each synthetic message must match the validator's
    f-string format verbatim — the regex extractors in
    ``remediation.py`` parse against that shape, so a loosely-worded
    fixture would silently exercise nothing.

    Each code produces a DISTINCT directive substring:

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
    # Message shape must match ``_RE_OBSERVED_BLOOM`` / ``_RE_BLOOM_GAP``
    # / ``_RE_DECLARED_BLOOM`` in remediation.py.
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
        passed=True,  # warning-only validator; passed stays True
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
    # Message shape must match ``_RE_OBJECTIVE_ID`` /
    # ``_RE_ENTAILMENT_SCORE`` / ``_RE_ENTAILMENT_FLOOR`` in
    # remediation.py.
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
    # Message shape must match ``_RE_BLOOM_VERB`` /
    # ``_RE_VERB_SYNONYMS_PREVIEW`` in remediation.py.
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
    """Test-count ratchet + ``check_schema`` cleanliness.

    Counts ``def test_`` declarations under four trees and asserts each
    is at or above its ``_MIN_*_TESTS`` floor. The ratchet only moves up
    — adding tests means bumping the constant; a refactor that removes
    tests trips the gate.

    Also asserts:

    * ``Draft202012Validator.check_schema`` passes on ``$defs.Block``
      AND ``$defs.ObjectiveAlignment`` with ZERO
      ``DeprecationWarning``s — a deprecated construct still LOADS, so
      without this the only symptom is a warning at every consumer's
      validation time.
    * The four statistical-tier gate test files still meet their
      per-file floors; the tree-level ratchet alone would miss a
      removal in one file offset by additions elsewhere in that tree.
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

    # Per-file floors for the four statistical-tier gate test files.
    per_file_failures: List[str] = []
    for rel_path, floor in _MIN_STATISTICAL_TIER_TEST_COUNTS.items():
        path = PROJECT_ROOT / rel_path
        actual = _count_tests_in_file(path)
        if actual < floor:
            per_file_failures.append(
                f"{rel_path}: collected {actual} test functions; "
                f"floor is {floor} — a test was removed from this file."
            )
    assert not per_file_failures, (
        "Wave-end statistical-tier per-file ratchet tripped:\n  "
        + "\n  ".join(per_file_failures)
    )

    # Capture every emitted warning and assert ZERO are
    # DeprecationWarnings on the Block / ObjectiveAlignment surface.
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
        "jsonschema emitted DeprecationWarning(s) on the "
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
