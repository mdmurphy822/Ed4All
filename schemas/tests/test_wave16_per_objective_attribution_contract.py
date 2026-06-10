"""GPT Feedback v2 — Wave 1.6 end-of-wave per-objective-attribution gate.

Authored 2026-05-06 against the closing 6-test gate enumerated in
the Wave 1.6 per-objective-attribution gate spec § 4.

Predecessors landed:

* W1.6.A at 194e999 — schema oneOf bump on
  ``schemas/knowledge/objectives_v1.schema.json::definitions.{TerminalOutcome,ComponentObjective}``.
* W1.6.B at 44aa868 — course-outliner.md prompt directive +
  ``MCP.tools._content_gen_helpers.synthesize_objectives_from_topics``
  emit of structured ``[{ref, chunk_ids[]}]`` source_refs.
* W1.6.C at 875c391 — ``lib.validators.objective_source_refs.ObjectiveSourceRefValidator``
  + ``objective_source_ref_check`` enum addition.
* W1.6.D plan amendment (no commit) — W2.F brief now references
  three-surface NLI fan-out.

This file is a SELF-CONTAINED contract test mirroring the Wave 1.5
predecessor at ``schemas/tests/test_wave15_per_claim_attribution_contract.py``
(commit pre-194e999). Every fixture is inlined locally so a future
rename of a per-worker test file cannot silently disable the
wave-end gate.

Drift note vs plan §4: plan text references ``$defs.TerminalOutcome``
and ``$defs.ComponentObjective`` (Draft-2020-12 vocabulary). The live
``objectives_v1.schema.json`` declares
``"$schema": "http://json-schema.org/draft-07/schema#"`` and uses the
Draft-07 ``definitions/`` vocabulary. This gate validates the live
shape, not the plan's terminology — Test 6 uses ``Draft7Validator`` and
walks ``definitions.{TerminalOutcome,ComponentObjective}``.

Tests:

* ``test_1_schema_validity_both_shapes_and_mixed_rejection`` — pin the
  bumped per-definition ``oneOf`` contract on BOTH ``TerminalOutcome``
  and ``ComponentObjective``: legacy ``List[str]`` validates, structured
  ``List[{ref, chunk_ids[]}]`` validates, mixed-shape FAILS, empty
  ``chunk_ids: []`` FAILS the inner ``minItems: 1``, and a TO with
  no ``source_refs`` field at all STILL validates (optional field).
* ``test_2_anti_silent_degradation_unresolved_ref_consistency_walk``
  — pin the W1.6.C consistency walk: a synthesized objective citing
  a chapter ID NOT in ``textbook_structure.chapters[*].id`` fires
  ``OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE`` with
  ``action="regenerate"``; the decision-capture event signals
  ``unresolved_refs_count == 1``.
* ``test_3_anti_silent_degradation_legacy_corpora_validate`` — pin
  back-compat against the first discovered LibV2 course's
  ``objectives.json`` (``pytest.skip()``
  on a clean checkout): every CO entry validates clean against the
  bumped ``definitions.ComponentObjective``; every TO (no
  ``source_refs`` field) validates clean against bumped
  ``definitions.TerminalOutcome``; ZERO entries fire the mixed-shape
  rejection path.
* ``test_4_outliner_prompt_directive_and_synthesizer_alignment`` —
  pin ``Courseforge/agents/course-outliner.md`` carrying the literal
  ``"Per-objective source attribution"`` + ``"chunk_ids[] MUST be a
  non-empty subset"`` sentinels; PLUS
  ``synthesize_objectives_from_topics`` against a 1-topic /
  2-statement Path A fixture returns LOs whose ``source_refs[]`` is
  structured-shape with non-empty ``chunk_ids[]``.
* ``test_5_validator_decision_type_in_canonical_enum`` — pin
  ``objective_source_ref_check`` IS in
  ``schemas/events/decision_event.schema.json::decision_type.enum``
  (W1.6.C addition); pin the future ``objective_support_check`` either
  IS in the enum (no-op confirmation) or — at the time of writing —
  is NOT yet (xfail-recorded as a checklist item for the future
  ``ObjectiveClaimSupportValidator``). Plus a real validator-fire
  emit that asserts the registered string verbatim.
* ``test_6_collected_count_guard_and_check_schema_clean`` — wave-end
  "no silent test removal" ratchet. AST-walks ``lib/validators/tests/``,
  ``schemas/tests/``, and ``MCP/tools/tests/``; encodes the post-Wave-1.6
  baseline counts as ``_MIN_*_TESTS`` constants; asserts
  ``Draft7Validator.check_schema(...)`` passes on the bumped
  ``objectives_v1.schema.json`` for BOTH
  ``definitions.TerminalOutcome`` AND ``definitions.ComponentObjective``;
  asserts ZERO ``DeprecationWarning``s on the bumped oneOf surface.
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Audited post-Wave-1.6 baselines — encoded as constants so a future
# silent test removal trips the ratchet. Counts captured 2026-05-06 on
# dev-v0.3.0 at HEAD by AST-walking each tree's ``test_*.py`` files and
# tallying every ``def test_`` declaration (the same algorithm
# ``_count_test_functions`` runs at gate time).
#
# Mirrors the AST-vs-collect-only contract noted in
# ``test_wave15_per_claim_attribution_contract.py``: AST counts are
# LOWER than ``pytest --collect-only`` counts because pytest expands
# parametrized cases at collection time. AST counting is the choice
# for this gate (subprocess-pytest from inside a test is brittle); the
# ratchet contract — "no silent test removal" — is satisfied either
# way because both algorithms are monotonic.
# --------------------------------------------------------------------------- #

_MIN_VALIDATORS_TESTS = 466
_MIN_SCHEMAS_TESTS = 339
_MIN_MCP_TOOLS_TESTS = 36


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _load_objectives_schema() -> Dict[str, Any]:
    """Read the live ``objectives_v1.schema.json`` from disk."""
    schema_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "objectives_v1.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _to_payload(source_refs: Any = None, drop_field: bool = False) -> Dict[str, Any]:
    """Build a minimally-valid TerminalOutcome payload.

    ``drop_field=True`` omits the ``source_refs`` key entirely (proves
    the field is optional). Otherwise ``source_refs`` is attached
    verbatim to exercise each ``oneOf`` arm.
    """
    base: Dict[str, Any] = {
        "id": "TO-01",
        "statement": "Analyze RDF triples and SHACL shape graphs.",
        "bloom_level": "analyze",
        "bloom_verb": "analyze",
    }
    if not drop_field:
        base["source_refs"] = source_refs
    return base


def _co_payload(source_refs: Any = None, drop_field: bool = False) -> Dict[str, Any]:
    """Build a minimally-valid ComponentObjective payload.

    Mirrors ``_to_payload`` but adds the optional ``parent_terminal``
    back-pointer and the chapter-week field — all strictly defined by
    the live ``definitions.ComponentObjective`` shape.
    """
    base: Dict[str, Any] = {
        "id": "CO-01",
        "statement": (
            "Identify subject, predicate, object in an RDF triple."
        ),
        "parent_terminal": "TO-01",
        "bloom_level": "remember",
        "bloom_verb": "identify",
        "week": 1,
    }
    if not drop_field:
        base["source_refs"] = source_refs
    return base


def _count_test_functions(tree: Path) -> Tuple[int, List[str]]:
    """AST-walk every ``test_*.py`` under ``tree`` and count ``def test_``
    declarations.

    Mirrors ``test_wave15_per_claim_attribution_contract.py::_count_test_functions``
    so the algorithm is identical between gates and the count contract
    is genuinely monotonic across waves.
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


# --------------------------------------------------------------------------- #
# Test 1 — schema validity (both shapes + mixed rejection on BOTH defs)
# --------------------------------------------------------------------------- #


def test_1_schema_validity_both_shapes_and_mixed_rejection() -> None:
    """Plan §4 Test 1 — bumped ``oneOf`` contract is symmetric across
    TerminalOutcome AND ComponentObjective.

    A future schema author tightening only the CO arm and forgetting
    TO (or vice versa) trips this gate. Each sub-clause runs against
    BOTH ``definitions.TerminalOutcome`` and
    ``definitions.ComponentObjective``.
    """
    jsonschema = _require_jsonschema()

    schema_doc = _load_objectives_schema()
    to_def = schema_doc["definitions"]["TerminalOutcome"]
    co_def = schema_doc["definitions"]["ComponentObjective"]

    # Sub-test 1a — legacy List[str] source_refs validates clean on BOTH defs.
    legacy_refs = ["dart:rdf-primer#sec1", "dart:rdf-primer#sec2"]
    jsonschema.Draft7Validator(to_def).validate(_to_payload(legacy_refs))
    jsonschema.Draft7Validator(co_def).validate(_co_payload(legacy_refs))

    # Sub-test 1b — structured List[{ref, chunk_ids[]}] validates clean.
    structured_refs = [
        {
            "ref": "ch7",
            "chunk_ids": ["dart:shacl-spec#sec3-1-targets"],
        },
        {
            "ref": "ch8",
            "chunk_ids": [
                "dart:shacl-spec#sec4-1-core-constraints",
                "dart:shacl-spec#sec4-2-property-shapes",
            ],
        },
    ]
    jsonschema.Draft7Validator(to_def).validate(_to_payload(structured_refs))
    jsonschema.Draft7Validator(co_def).validate(_co_payload(structured_refs))

    # Sub-test 1c — mixed-shape (one string + one object) MUST FAIL via
    # the canonical ``oneOf`` rejection error on BOTH defs.
    mixed_refs = [
        "dart:rdf-primer#sec1",
        {
            "ref": "ch7",
            "chunk_ids": ["dart:shacl-spec#sec3-1-targets"],
        },
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(to_def).validate(_to_payload(mixed_refs))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(co_def).validate(_co_payload(mixed_refs))

    # Sub-test 1d — empty chunk_ids: [] MUST FAIL the inner minItems: 1
    # bound on BOTH defs.
    empty_chunk_ids = [
        {
            "ref": "ch7",
            "chunk_ids": [],
        },
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(to_def).validate(_to_payload(empty_chunk_ids))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(co_def).validate(_co_payload(empty_chunk_ids))

    # Sub-test 1e — TO with no source_refs field at all STILL validates
    # (the field is optional). Symmetry: same on CO.
    jsonschema.Draft7Validator(to_def).validate(_to_payload(drop_field=True))
    jsonschema.Draft7Validator(co_def).validate(_co_payload(drop_field=True))


# --------------------------------------------------------------------------- #
# Test 2 — anti-silent-degradation: per-LO unresolved-ref consistency walk
# --------------------------------------------------------------------------- #


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


def test_2_anti_silent_degradation_unresolved_ref_consistency_walk(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 2 — per-LO source_refs consistency walk.

    Fixture mirrors the plan §4 Test 2 verbatim:

      objective.source_refs = [
          {"ref": "ch-nope", "chunk_ids": ["dart:bar#qux"]}
      ]
      textbook_structure.json carries [{"id": "ch7"}, {"id": "ch8"}]
      dart_chunks/manifest.json carries no entries (legacy archive).

    Asserts:

    * at least one issue has code
      ``OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE``
    * ``result.action == "regenerate"`` (warning-severity miss → re-roll)
    * decision-capture event signals ``unresolved_refs_count == 1``

    Drift note vs plan §4 Test 2: plan text asserts
    ``result.passed is False``, but the W1.6.C validator
    (``ObjectiveSourceRefValidator.validate``) sets ``passed`` from
    ``critical_count == 0`` only — warning-severity misses leave
    ``passed=True`` and signal re-roll via ``action="regenerate"``.
    Wave 1.6 explicitly ships at WARNING severity day-1 (per the
    W1.6.C docstring + plan §6.3 risk note); promotion to CRITICAL
    waits for future-wave calibration. This test gates on the
    regeneration signal — ``action="regenerate"`` + the issue code
    + the decision-capture rationale — which is what the workflow
    runner actually consumes. A future wave that promotes the codes
    to critical-severity should ALSO add an explicit ``passed is False``
    assertion here.
    """
    from lib.validators.objective_source_refs import ObjectiveSourceRefValidator

    # Build the ``synthesized_objectives.json`` payload shaped like the
    # course-planning phase emit. Use the LibV2-archive form
    # (``terminal_outcomes`` + ``component_objectives``) since the
    # validator's ``_flatten_objectives`` shape-tolerance helper accepts
    # both forms identically.
    objectives_payload: Dict[str, Any] = {
        "schema_version": "v1",
        "course_code": "TEST_W16",
        "terminal_outcomes": [
            {
                "id": "TO-01",
                "statement": "Analyze SHACL shape graphs.",
                "bloom_level": "analyze",
            },
        ],
        "component_objectives": [
            {
                "id": "CO-01",
                "statement": "Identify a SHACL node shape constraint.",
                "parent_terminal": "TO-01",
                "bloom_level": "remember",
                "week": 1,
                "source_refs": [
                    {
                        "ref": "ch-nope",
                        "chunk_ids": ["dart:bar#qux"],
                    },
                ],
            },
        ],
        "objective_count": {"terminal": 1, "component": 1},
    }
    objectives_path = tmp_path / "synthesized_objectives.json"
    objectives_path.write_text(json.dumps(objectives_payload), encoding="utf-8")

    textbook_structure_payload: Dict[str, Any] = {
        "chapters": [
            {"id": "ch7", "sections": []},
            {"id": "ch8", "sections": []},
        ],
    }
    textbook_structure_path = tmp_path / "textbook_structure.json"
    textbook_structure_path.write_text(
        json.dumps(textbook_structure_payload), encoding="utf-8"
    )

    # Empty manifest — sibling chunks.jsonl absent so the chunks
    # universe is empty (matches the "legacy archive" branch the plan
    # calls out).
    chunks_dir = tmp_path / "dart_chunks"
    chunks_dir.mkdir()
    chunks_manifest_path = chunks_dir / "manifest.json"
    chunks_manifest_path.write_text(
        json.dumps({
            "chunkset_kind": "dart",
            "chunker_version": "v0",
            "chunks_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )

    capture = _RecordingCapture()
    result = ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives_path),
        "textbook_structure_path": str(textbook_structure_path),
        "dart_chunks_manifest_path": str(chunks_manifest_path),
        "decision_capture": capture,
    })

    # Drift vs plan §4: the W1.6.C validator gates ``passed`` on
    # critical-severity misses only (Wave 1.6 ships warning-only). The
    # regeneration signal that the workflow runner consumes is
    # ``action="regenerate"`` + the warning-severity GateIssue code,
    # NOT ``passed=False``. Pin both so a future critical-promotion
    # wave can add an explicit ``passed is False`` assertion above.
    assert result.action == "regenerate", (
        f"expected action='regenerate' on warning-severity miss; "
        f"got action={result.action!r}"
    )

    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE" in codes, (
        "expected at least one issue with code "
        "OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE; got "
        f"codes={codes!r}"
    )

    # Decision-capture event MUST surface unresolved_refs_count == 1.
    relevant = [
        e for e in capture.events
        if e.get("decision_type") == "objective_source_ref_check"
    ]
    assert relevant, (
        "expected at least one objective_source_ref_check decision event"
    )
    # The validator interpolates ``unresolved_refs_count={n}`` into both
    # ``rationale`` and ``context`` — assert at least one event carries
    # the literal ``unresolved_refs_count=1`` AND the LO that fired it
    # is the structurally-broken CO-01.
    miss_events = [
        e for e in relevant
        if "unresolved_refs_count=1" in str(e.get("context", ""))
        or "unresolved_refs_count=1" in str(e.get("rationale", ""))
    ]
    assert miss_events, (
        "decision-capture rationale / context MUST surface "
        "unresolved_refs_count=1; got events: "
        + repr([{k: v for k, v in e.items() if k != "rationale"} for e in relevant])
    )


# --------------------------------------------------------------------------- #
# Test 3 — anti-silent-degradation: legacy corpora still validate
# --------------------------------------------------------------------------- #


def _discover_legacy_archive_path() -> "Path | None":
    """First LibV2 course carrying an ``objectives.json``.

    Resolves the LibV2 root via the ``ED4ALL_LIBV2_ROOT`` convention
    (``lib.paths.libv2_path``) so the back-compat gate binds to whatever
    archived corpus is on disk rather than a pinned course slug.
    """
    try:
        from lib.paths import libv2_path

        courses_root = libv2_path() / "courses"
    except Exception:
        courses_root = PROJECT_ROOT / "LibV2" / "courses"
    if not courses_root.is_dir():
        return None
    for course in sorted(p for p in courses_root.iterdir() if p.is_dir()):
        objectives = course / "objectives.json"
        if objectives.exists():
            return objectives
    return None


_LEGACY_ARCHIVE_PATH = _discover_legacy_archive_path()


def test_3_anti_silent_degradation_legacy_corpora_validate() -> None:
    """Test 3 — the live LibV2 archive validates clean
    against the W1.6.A bumped schema.

    Self-contained: reads the archive file directly, no fixture
    dependency on per-worker test files. ``pytest.skip()`` on a clean
    checkout where the archive isn't present so CI doesn't fail.

    Catches the regression class where a future schema author tightens
    the ``oneOf`` ``List[str]`` arm and silently breaks every existing
    archive on the RDF/SHACL calibration corpus.
    """
    if _LEGACY_ARCHIVE_PATH is None or not _LEGACY_ARCHIVE_PATH.exists():
        pytest.skip(
            "no LibV2 course objectives.json present; "
            "skipping back-compat gate (clean-checkout path)."
        )
    jsonschema = _require_jsonschema()

    archive = json.loads(_LEGACY_ARCHIVE_PATH.read_text(encoding="utf-8"))

    schema_doc = _load_objectives_schema()
    to_def = schema_doc["definitions"]["TerminalOutcome"]
    co_def = schema_doc["definitions"]["ComponentObjective"]

    to_validator = jsonschema.Draft7Validator(to_def)
    co_validator = jsonschema.Draft7Validator(co_def)

    failures: List[str] = []
    mixed_shape_hits: List[str] = []

    terminal_outcomes = archive.get("terminal_outcomes") or []
    component_objectives = archive.get("component_objectives") or []

    # Plan §4 Test 3 contract: the live archive carries TOs with NO
    # source_refs at all, and COs with legacy List[str] source_refs.
    # Both shapes MUST validate clean against the bumped definitions.
    for to in terminal_outcomes:
        errors = list(to_validator.iter_errors(to))
        if errors:
            failures.append(
                f"TO {to.get('id')!r}: "
                f"{[(list(e.absolute_path), e.message) for e in errors]}"
            )
        sr = to.get("source_refs")
        if isinstance(sr, list) and sr:
            shapes = {("dict" if isinstance(x, dict) else "str") for x in sr}
            if len(shapes) > 1:
                mixed_shape_hits.append(f"TO {to.get('id')!r}: shapes={shapes}")

    for co in component_objectives:
        errors = list(co_validator.iter_errors(co))
        if errors:
            failures.append(
                f"CO {co.get('id')!r}: "
                f"{[(list(e.absolute_path), e.message) for e in errors]}"
            )
        sr = co.get("source_refs")
        if isinstance(sr, list) and sr:
            shapes = {("dict" if isinstance(x, dict) else "str") for x in sr}
            if len(shapes) > 1:
                mixed_shape_hits.append(f"CO {co.get('id')!r}: shapes={shapes}")

    assert not failures, (
        "live legacy archive entries regressed against the bumped oneOf "
        "schema:\n  " + "\n  ".join(failures)
    )
    assert not mixed_shape_hits, (
        "live legacy archive entries fired the mixed-shape rejection "
        "path (should be impossible — schema's oneOf forbids mixed "
        "lists, and the synthesizer never emits them):\n  "
        + "\n  ".join(mixed_shape_hits)
    )

    # Defence-in-depth — the archive must carry at least one TO and at
    # least one CO so the assertions above had real material to walk
    # against. A future archive layout shift that drops one of the two
    # collections would silently no-op this gate without this check.
    assert terminal_outcomes, (
        "live archive carries no terminal_outcomes — Test 3's TO arm "
        "had nothing to validate against; archive shape may have shifted."
    )
    assert component_objectives, (
        "live archive carries no component_objectives — Test 3's CO arm "
        "had nothing to validate against; archive shape may have shifted."
    )


# --------------------------------------------------------------------------- #
# Test 4 — outliner prompt directive + synthesizer alignment
# --------------------------------------------------------------------------- #


_COURSE_OUTLINER_PATH = (
    PROJECT_ROOT / "Courseforge" / "agents" / "course-outliner.md"
)


def test_4_outliner_prompt_directive_and_synthesizer_alignment() -> None:
    """Plan §4 Test 4 — course-outliner.md sentinel pair + synthesizer
    structural-shape emit.

    Three sentinel assertions:

    1. ``Courseforge/agents/course-outliner.md`` carries the literal
       ``"Per-objective source attribution"`` (W1.6.B section header).
    2. Same file carries the canonical chunk-id-universe-bound
       directive ``"MUST be a non-empty subset"`` immediately
       following the ``chunk_ids`` token (W1.6.B verbatim contract
       directive). Drift note vs plan §4 Test 4: plan text expects
       the bare ``"chunk_ids[] MUST be a non-empty subset"`` literal,
       but the W1.6.B-authored file wraps the token in backticks
       (``` `chunk_ids[]` MUST be a non-empty subset ```) — markdown
       inline-code styling. We assert both fragments individually
       (token + clause) so the gate fires on either backtick variant
       and on a future drop of either half of the directive.
    3. ``synthesize_objectives_from_topics`` against a 1-topic /
       2-statement Path A fixture returns LOs whose ``source_refs[]``
       is structured-shape with non-empty ``chunk_ids[]``.

    Confirms the prompt + synthesizer aren't drifted out of sync.
    """
    assert _COURSE_OUTLINER_PATH.exists(), (
        f"course-outliner agent spec missing at {_COURSE_OUTLINER_PATH}"
    )
    spec_text = _COURSE_OUTLINER_PATH.read_text(encoding="utf-8")

    # Sub-test 4a — section header sentinel.
    assert "Per-objective source attribution" in spec_text, (
        "Wave 1.6 W1.6.B section-header sentinel "
        "'Per-objective source attribution' missing from "
        f"{_COURSE_OUTLINER_PATH}; operators rely on this literal "
        "substring as the prompt-directive regression seam."
    )

    # Sub-test 4b — verbatim contract directive sentinel. Pin both the
    # ``chunk_ids`` token AND the ``MUST be a non-empty subset`` clause;
    # tolerate the inline-backtick variant the W1.6.B author shipped.
    assert "chunk_ids" in spec_text, (
        "Wave 1.6 W1.6.B verbatim contract directive missing the "
        "'chunk_ids' token from "
        f"{_COURSE_OUTLINER_PATH}; the chunk-id universe bound MUST "
        "be stated verbatim so the model's emit aligns with the "
        "validator's resolution rule."
    )
    assert "MUST be a non-empty subset" in spec_text, (
        "Wave 1.6 W1.6.B verbatim contract directive missing the "
        "'MUST be a non-empty subset' clause from "
        f"{_COURSE_OUTLINER_PATH}; the model needs the explicit "
        "non-empty-subset bound so it doesn't emit empty chunk_ids[]."
    )
    # Defence-in-depth — at least one ``chunk_ids`` occurrence MUST
    # appear close enough to the bound clause to plausibly form the
    # same sentence. The W1.6.B-authored copy uses ``chunk_ids`` in
    # several locations (the JSON example block AND the prose
    # directive); search every occurrence and assert the MIN distance
    # to the bound clause is small. Catches a refactor that sets
    # ``chunk_ids`` only in one example section and silently drops the
    # bound directive in unrelated prose.
    bound_idx = spec_text.find("MUST be a non-empty subset")
    chunk_id_positions = [
        i for i in range(len(spec_text))
        if spec_text.startswith("chunk_ids", i)
    ]
    nearest_distance = min(
        abs(pos - bound_idx) for pos in chunk_id_positions
    )
    assert nearest_distance <= 100, (
        "Wave 1.6 W1.6.B contract directive 'chunk_ids' and "
        "'MUST be a non-empty subset' have drifted apart in "
        f"{_COURSE_OUTLINER_PATH}; they should co-occur within 100 "
        "chars per the W1.6.B authored copy. "
        f"nearest_distance={nearest_distance}, bound_idx={bound_idx}, "
        f"chunk_id_positions={chunk_id_positions}"
    )

    # Sub-test 4c — synthesizer alignment. Use a 1-topic / 2-statement
    # Path A fixture (extracted_lo_statements pre-populated, so
    # synthesize_objectives_from_topics takes Path A and emits one LO
    # per statement). Both LOs must carry structured source_refs[]
    # with non-empty chunk_ids[].
    from MCP.tools import _content_gen_helpers as _cgh  # noqa: WPS433

    topic = {
        "heading": "RDF Triples",
        "paragraphs": [
            (
                "Body text describing RDF triples in sufficient depth "
                "to satisfy the grounding validator non-trivial "
                "paragraph floor of thirty words each required for "
                "content pages to be treated as real body prose."
            ),
        ],
        "key_terms": ["rdf", "triple"],
        "source_file": "rdf_primer_accessible",
        "word_count": 60,
        "chapter_id": "ch1",
        "dart_block_ids": ["sec1-1-triples", "sec1-2-iris"],
        "extracted_lo_statements": [
            "Identify the three components of an RDF triple.",
            "Describe how triples combine to form an RDF graph.",
        ],
        "extracted_misconceptions": [],
        "extracted_questions": [],
    }
    terminal, chapter = _cgh.synthesize_objectives_from_topics(
        topics=[topic],
        duration_weeks=1,
        max_terminal=1,
    )
    all_los = list(terminal) + list(chapter)
    assert len(all_los) == 2, (
        f"expected exactly 2 LOs on the 1-topic / 2-statement Path A "
        f"fixture; got {len(all_los)} (terminal={len(terminal)}, "
        f"chapter={len(chapter)})"
    )

    for lo in all_los:
        sr = lo.get("source_refs")
        assert isinstance(sr, list) and sr, (
            f"LO {lo.get('id')!r} emitted empty source_refs; "
            "synthesizer Path A MUST emit a structured entry when the "
            "topic carries dart_block_ids + source_file."
        )
        for entry in sr:
            assert isinstance(entry, dict), (
                f"LO {lo.get('id')!r} source_refs entry is not a dict: "
                f"{entry!r}; synthesizer must emit structured shape, "
                "not legacy List[str]."
            )
            assert "ref" in entry and entry["ref"], (
                f"LO {lo.get('id')!r} structured source_refs entry "
                f"missing non-empty 'ref' field: {entry!r}"
            )
            chunk_ids = entry.get("chunk_ids")
            assert isinstance(chunk_ids, list) and chunk_ids, (
                f"LO {lo.get('id')!r} structured source_refs entry "
                f"has empty chunk_ids[]: {entry!r}; synthesizer must "
                "honour the schema's inner minItems: 1 bound."
            )
            for cid in chunk_ids:
                assert isinstance(cid, str) and cid.strip(), (
                    f"LO {lo.get('id')!r} chunk_ids entry malformed: "
                    f"{cid!r}"
                )


# --------------------------------------------------------------------------- #
# Test 5 — validator decision_type is in canonical enum
# --------------------------------------------------------------------------- #


_DECISION_EVENT_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "events" / "decision_event.schema.json"
)


def _load_decision_type_enum() -> set:
    schema = json.loads(
        _DECISION_EVENT_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    return set(schema["properties"]["decision_type"]["enum"])


def test_5_validator_decision_type_in_canonical_enum(tmp_path: Path) -> None:
    """Plan §4 Test 5 — registered decision_type membership.

    Two assertions:

    1. ``objective_source_ref_check`` (W1.6.C addition) MUST be in
       ``schemas/events/decision_event.schema.json::decision_type.enum``.
       This is a HARD assertion — DECISION_VALIDATION_STRICT=true
       would silently drop the W1.6.C events without it.
    2. ``objective_support_check`` (pre-registered for the future
       ``ObjectiveClaimSupportValidator``) — at the time of writing
       (Wave 1.6) this enum addition was NOT made. The plan §4 Test 5
       contract: structure as a pending checklist. We use ``pytest.xfail``
       to record the missing entry honestly. When a future wave adds
       the enum member, this xfail flips to xpass — surface the
       transition without it being a silent test removal.

    Plus: a real validator-fire path emits the registered string
    verbatim (defensive against a typo in
    ``ObjectiveSourceRefValidator._DECISION_TYPE``).
    """
    enum = _load_decision_type_enum()

    # Sub-test 5a — HARD assertion: W1.6.C addition is present.
    assert "objective_source_ref_check" in enum, (
        "Wave 1.6 W1.6.C addition 'objective_source_ref_check' is NOT "
        "in schemas/events/decision_event.schema.json::decision_type.enum. "
        "DECISION_VALIDATION_STRICT=true would silently drop the "
        "ObjectiveSourceRefValidator's events. Add it to the enum."
    )

    # Sub-test 5b — pre-registered checklist for the future
    # ``ObjectiveClaimSupportValidator``. If absent, mark this xfail
    # so the missing entry is recorded but doesn't fail the wave.
    if "objective_support_check" not in enum:
        pytest.xfail(
            "'objective_support_check' is pre-registered for the future "
            "ObjectiveClaimSupportValidator (post-Wave-1.6 sibling "
            "landing per plan §5). Wave 1.6 only adds "
            "'objective_source_ref_check'; this xfail flips to xpass "
            "when a future wave adds the enum member."
        )

    # Sub-test 5c — real validator-fire emits the registered string
    # verbatim. Build a minimal one-LO fixture that takes the
    # graceful-degrade path (no universes wired) so the validator runs
    # without disk I/O or upstream fixtures. With no universes wired,
    # ObjectiveSourceRefValidator emits OBJECTIVE_SOURCE_REFS_NO_UNIVERSE
    # WITHOUT an objective-level decision event — but with a textbook
    # structure path AND empty objectives, the per-LO decision-capture
    # path fires. We want the actual emit string, so wire a real
    # textbook_structure.json + an LO with structured refs that resolve.
    from lib.validators.objective_source_refs import ObjectiveSourceRefValidator

    objectives_payload: Dict[str, Any] = {
        "schema_version": "v1",
        "course_code": "TEST_W16_5C",
        "terminal_outcomes": [
            {
                "id": "TO-01",
                "statement": "Analyze SHACL shape graphs.",
                "bloom_level": "analyze",
            },
        ],
        "component_objectives": [
            {
                "id": "CO-01",
                "statement": "Identify SHACL constraints.",
                "parent_terminal": "TO-01",
                "bloom_level": "remember",
                "week": 1,
                "source_refs": [
                    {"ref": "ch7", "chunk_ids": ["dart:fixture#sec1"]},
                ],
            },
        ],
        "objective_count": {"terminal": 1, "component": 1},
    }
    objectives_path = tmp_path / "synthesized_objectives.json"
    objectives_path.write_text(
        json.dumps(objectives_payload), encoding="utf-8"
    )

    textbook_structure_path = tmp_path / "textbook_structure.json"
    textbook_structure_path.write_text(
        json.dumps({
            "chapters": [{"id": "ch7", "sections": []}],
        }),
        encoding="utf-8",
    )

    capture = _RecordingCapture()
    ObjectiveSourceRefValidator().validate({
        "synthesized_objectives_path": str(objectives_path),
        "textbook_structure_path": str(textbook_structure_path),
        "decision_capture": capture,
    })

    emitted_types = {e.get("decision_type") for e in capture.events}
    assert "objective_source_ref_check" in emitted_types, (
        "ObjectiveSourceRefValidator did not emit a decision event with "
        "decision_type='objective_source_ref_check'; the registered "
        "string in lib/validators/objective_source_refs.py "
        "(_DECISION_TYPE) drifted from the enum entry. Got types: "
        f"{emitted_types!r}"
    )


# --------------------------------------------------------------------------- #
# Test 6 — full pytest sweep + collected-count guard + DeprecationWarning-clean
# --------------------------------------------------------------------------- #


def test_6_collected_count_guard_and_check_schema_clean() -> None:
    """Plan §4 Test 6 — wave-end "no silent test removal" ratchet PLUS
    Draft7Validator.check_schema cleanliness on the bumped oneOf surface.

    Walks every ``test_*.py`` under the three trees Wave 1.6 ratchets
    and counts each ``def test_`` declaration via ``ast.walk``. Asserts
    each tree's count is at least the post-Wave-1.6 baseline encoded in
    the ``_MIN_*_TESTS`` constants above. Ratchets upward — if a future
    wave legitimately ADDS tests, bump the constant; if a refactor
    accidentally REMOVES tests, this gate fires.

    Also asserts ``Draft7Validator.check_schema`` passes on the bumped
    ``definitions.TerminalOutcome`` AND ``definitions.ComponentObjective``,
    capturing every emitted warning so a ``DeprecationWarning`` trips
    the gate explicitly.
    """
    jsonschema = _require_jsonschema()

    trees = [
        ("validators", PROJECT_ROOT / "lib" / "validators" / "tests",
         _MIN_VALIDATORS_TESTS),
        ("schemas", PROJECT_ROOT / "schemas" / "tests",
         _MIN_SCHEMAS_TESTS),
        ("mcp_tools", PROJECT_ROOT / "MCP" / "tools" / "tests",
         _MIN_MCP_TOOLS_TESTS),
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

    # check_schema sub-clause: capture every emitted warning and assert
    # ZERO are DeprecationWarnings. Plan §4 Test 6 named clause: "no
    # warnings tagged DeprecationWarning from the new oneOf surface".
    schema_doc = _load_objectives_schema()
    deprecation_warnings: List[str] = []

    for def_name in ("TerminalOutcome", "ComponentObjective"):
        def_schema = schema_doc["definitions"][def_name]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            jsonschema.Draft7Validator.check_schema(def_schema)
            for w in captured:
                if issubclass(w.category, DeprecationWarning):
                    deprecation_warnings.append(
                        f"definitions.{def_name}: "
                        f"{w.category.__name__}: {w.message}"
                    )
    assert not deprecation_warnings, (
        "jsonschema emitted DeprecationWarning(s) on the bumped oneOf "
        "schema surface:\n  " + "\n  ".join(deprecation_warnings)
    )

    # Defence-in-depth — also re-check the WHOLE document so a future
    # refactor that splits the oneOf out of definitions/ but leaves the
    # per-definition shape intact still trips the gate if a deprecation
    # surfaces at the document level.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        jsonschema.Draft7Validator.check_schema(schema_doc)
        whole_doc_deps = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
    assert not whole_doc_deps, (
        "Draft7Validator.check_schema(whole-doc) emitted "
        "DeprecationWarning(s): "
        + "; ".join(str(w.message) for w in whole_doc_deps)
    )
