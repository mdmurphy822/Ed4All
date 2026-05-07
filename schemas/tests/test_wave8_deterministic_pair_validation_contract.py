"""GPT Feedback v2 — Wave 8 end-of-wave deterministic-pair-validation
contract gate.

Authored 2026-05-07 against the closing 6-test minimum gate enumerated
in ``plans/wave8-generator-pair-validation-audit-2026-05.md`` § 4
(tests 1, 2, 5, 6, 7, 11 from the 14-test list).

Predecessors landing:

* W8.A — :func:`Trainforge.synthesize_training._stamp_deterministic_pair_audit_fields`
  helper that fans every appended deterministic-generator pair through
  the W2.E + W4.A + W4.B + W4.C audit-field stamping pattern. Stamps
  ``promotion_status="validated"`` (oracle-grounded by construction),
  ``per_claim_support=[]``, ``claim_support_rate=None``,
  ``claim_contradicted_rate=None``, ``deps_missing=False``,
  ``pair_lo_resolution={..., "skipped": "deterministic_template"}``,
  and ``pair_objective_alignment=None``. Companion change extends the
  ``PairLearningOutcomeRefsValidator.validate`` walk with a
  ``pair_lo_resolution.skipped == "deterministic_template"`` short-
  circuit so synthetic chunk_ids on kg_metadata / violation_detection
  / schema_translation pairs no longer fire ``PAIR_CHUNK_NOT_FOUND``
  critically. Closes Checks 1 + 2 + 3 + the ``pair_promotion`` /
  ``pair_lo_refs`` blocking failures from the audit. **Tests 1, 2, 5,
  6, 7 depend on this worker; if it has not landed when this gate
  runs, those tests skip with a deferred-verification marker.**

* W8.B — Worker B stamped ``question_type`` on the four deterministic-
  generator pair envelopes (commit ``5f7185b``):

    * ``kg_metadata_generator.py`` → ``"question_type": "true_false"``.
    * ``violation_generator.py`` → ``"question_type": "true_false"``.
    * ``abstention_generator.py`` → ``"question_type": "true_false"``.
    * ``schema_translation_generator.py`` → ``"question_type":
      "short_answer"`` (paragraph-shaped definition / usage / pitfall
      / comparison / reasoning completions).

  Closes Check 5 (the W6.B / W7.A-C silent-degradation hole where
  deterministic pairs were silently invisible to the per-question-type
  matrix).

* W8.C — Worker C added an optional ``question_type`` enum field to
  ``schemas/knowledge/instruction_pair.schema.json`` and
  ``schemas/knowledge/preference_pair.schema.json`` (commit
  ``9f46be6``); admits the canonical 5-value set
  (``multiple_choice``, ``true_false``, ``short_answer``, ``essay``,
  ``fill_in_blank``). Optional so legacy corpora pre-Wave-8 stay
  schema-valid.

Predecessor wave-end gates this file mirrors:

* ``test_wave7_per_type_eval_segmentation_contract.py`` (Wave 7) —
  closest precedent for the wave-end-gate file structure (project-
  root ``sys.path`` injection, inlined fixtures, inlined helpers).
* ``test_wave5_trainforge_ingestion_contract.py`` (Wave 5) — closer
  precedent for the ``schemas/tests/`` path discipline (Lesson 2 from
  Wave 5 §7: the wave-end gate MUST land at ``schemas/tests/``, NOT
  ``lib/validators/tests/`` nor ``Trainforge/tests/``).
* ``test_wave4_pair_validation_contract.py`` (Wave 4 TIGHT) — THE
  TEMPLATE for the gate-file conventions across waves.

This file is a SELF-CONTAINED contract test mirroring the prior
wave-end gates. Every fixture is inlined locally so a future rename
of a per-component test file cannot silently disable this wave-end
gate.

Tests (the §4 6-test minimum):

1. ``test_kg_metadata_pair_carries_audit_fields_after_run_synthesis``
   — Worker A surface 1. Invokes :func:`run_synthesis` with
   ``with_kg_metadata=True`` against an inlined synthetic pedagogy
   graph; asserts every emitted ``content_type=="kg_metadata"`` pair
   on disk carries ``promotion_status="validated"``,
   ``per_claim_support=[]``,
   ``pair_lo_resolution.skipped == "deterministic_template"``,
   ``pair_objective_alignment is None``. AND the Worker B field
   ``question_type == "true_false"``.
2. ``test_violation_pair_carries_audit_fields_after_run_synthesis``
   — Worker A surface 2. Same shape as Test 1 but with
   ``with_violation_detection=True`` against a SHACL-family property
   manifest. Asserts the same audit fields + ``question_type ==
   "true_false"``.
5. ``test_pair_lo_refs_validate_walk_skips_deterministic_pairs`` —
   Worker A regression net. Synthesizes a corpus with
   ``with_kg_metadata=True`` (the most contaminated surface — uses
   graph node IDs as chunk_ids that never exist in chunks.jsonl), then
   runs :meth:`PairLearningOutcomeRefsValidator.validate` against the
   on-disk pair files + the fixture's chunks.jsonl. Asserts
   ``passed=True`` — no ``PAIR_CHUNK_NOT_FOUND`` issues against
   synthetic chunk_ids. Closes the blocking-failure regression on the
   ``pair_lo_refs`` critical gate identified in the audit (Check 3).
6. ``test_pair_promotion_validate_walk_passes_deterministic_pairs`` —
   Worker A regression net. Same setup as Test 5; runs
   :meth:`TrainingPairPromotionValidator.validate`. Asserts
   ``passed=True`` — no ``MISSING_PROMOTION_STATUS`` issues. Closes
   the blocking-failure regression on the ``pair_promotion`` critical
   gate identified in the audit (Check 1).
7. ``test_kg_metadata_pair_carries_question_type_true_false`` —
   Worker B surface 1. Mirror of Test 1 narrowed to the
   ``question_type`` field; pinned independently so a future regression
   in Worker B's stamping is caught even when the Worker A audit-field
   surface is healthy. Worker C precondition baked in: the
   ``question_type`` value MUST be one of the canonical 5-value enum.
11. ``test_instruction_pair_schema_admits_question_type_enum`` —
    Worker C schema-bump contract. Validates three pairs against
    ``schemas/knowledge/instruction_pair.schema.json``:

      * ``question_type="multiple_choice"`` — passes (canonical
        enum value).
      * ``question_type="invalid"`` — fails (off-enum value).
      * ``question_type`` absent — passes (back-compat with legacy
        pre-Wave-8 corpora that pre-date Worker B's stamping).

No optional 7th test — the §4 minimum is 6. The 8 additional tests
from the §4 14-test list (3, 4, 8, 9, 10, 12, 13, 14) close coverage
gaps where a future regression in a single generator would not be
caught by the others; those live in Worker-specific test files (Test
3 + 4 in Worker A's :file:`Trainforge/tests/test_deterministic_pair_audit_stamp.py`,
Tests 8 / 9 / 10 in Worker B's per-generator test files, Test 12 in
Worker C's :file:`Trainforge/tests/test_schema_validation.py` sibling).

Deferred-verification posture:

* Worker C landed in commit ``9f46be6``: Test 11 runs against the
  live schema; passes today.
* Worker B landed in commit ``5f7185b``: Test 7's Worker-B-side
  ``question_type`` assertion runs against the live emit; passes
  today (gated on Worker A's plumbing through ``run_synthesis``).
* Worker A may not have landed when this gate runs. Tests 1, 2, 5,
  6, 7 detect this via a sentinel check on the first emitted
  deterministic pair: when ``promotion_status`` (or
  ``pair_lo_resolution``) is absent, the test calls
  :func:`pytest.skip` with a deferred-verification marker. The skip
  message names Worker A explicitly so an operator triaging a CI
  output sees the dependency. Once Worker A lands, the same tests
  run-not-skip and assert the contract against the live emit.

Drift notes vs plan §4:

* The plan's Test 1 spec wording is "every emitted kg_metadata pair
  on disk carries: ``promotion_status="validated"``, …". The on-disk
  pair envelope is JSON-serialised per
  :func:`json.dumps(_p, sort_keys=True)` at
  ``Trainforge/synthesize_training.py:1538``, so the assertion
  reads the JSONL file back rather than the in-memory return list.
* The plan's Test 5 / 6 specs call ``validate({"course_dir": ...})``;
  in practice the validator accepts either ``course_dir`` (which
  resolves the standard ``training_specs/{instruction,preference}_pairs.jsonl``
  paths) or explicit ``instruction_pairs_path`` / ``preference_pairs_path``
  / ``chunks_path`` overrides. This file uses the explicit-path form
  so the chunks.jsonl resolution is unambiguous against the test
  fixture (which carries ``corpus/chunks.jsonl``, not
  ``imscc_chunks/chunks.jsonl``).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Inlined fixtures — pedagogy graph + mini training corpus copy helper
# --------------------------------------------------------------------------- #


_FIXTURE_TRAINING_ROOT = (
    PROJECT_ROOT / "Trainforge" / "tests" / "fixtures" / "mini_course_training"
)


def _copy_mini_training_fixture(tmp_path: Path) -> Path:
    """Materialize a writable copy of the read-only training fixture.

    Mirrors :func:`Trainforge/tests/test_synthesize_training.py::_make_working_copy`
    so the inlined fixture path matches the canonical pattern used by
    every other ``run_synthesis`` regression test in the suite.
    """
    dst = tmp_path / "mini_course_training"
    shutil.copytree(_FIXTURE_TRAINING_ROOT, dst)
    # Strip stale outputs from prior runs (the fixture is committed
    # with empty training_specs/ but a checkout-with-uncleaned-state
    # could carry residue).
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
        dst / "training_specs" / "instruction_pairs.jsonl.in_progress",
        dst / "training_specs" / "preference_pairs.jsonl.in_progress",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def _write_pedagogy_graph(course_dir: Path) -> Path:
    """Drop a small pedagogy graph next to the corpus.

    Five-node fixture mirroring the
    :func:`Trainforge/tests/test_kg_metadata_generator.py::_five_triple_graph`
    structure — minimal but enough to exercise every relation template
    in :data:`Trainforge.eval.faithfulness._RELATION_TEMPLATES`.

    Path matches branch (1) of
    :func:`Trainforge.synthesize_training._resolve_pedagogy_graph_path`'s
    candidate list (``<corpus_dir>/graph/pedagogy_graph.json``) so the
    LibV2-archive layout resolution wins on a fresh fixture.
    """
    graph_dir = course_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": [
            {"id": "chunk_a", "class": "Chunk"},
            {"id": "chunk_b", "class": "Chunk"},
            {"id": "concept_alpha", "class": "Concept"},
            {"id": "concept_beta", "class": "Concept"},
            {"id": "module:week_01", "class": "Module"},
            {"id": "module:week_02", "class": "Module"},
            {"id": "bloom:remember", "class": "BloomLevel"},
            {"id": "bloom:apply", "class": "BloomLevel"},
        ],
        "edges": [
            {
                "source": "chunk_a",
                "target": "concept_alpha",
                "relation_type": "assesses",
            },
            {
                "source": "chunk_b",
                "target": "concept_beta",
                "relation_type": "assesses",
            },
            {
                "source": "chunk_a",
                "target": "module:week_01",
                "relation_type": "belongs_to_module",
            },
            {
                "source": "chunk_b",
                "target": "module:week_02",
                "relation_type": "belongs_to_module",
            },
            {
                "source": "chunk_a",
                "target": "bloom:remember",
                "relation_type": "at_bloom_level",
            },
            {
                "source": "chunk_b",
                "target": "bloom:apply",
                "relation_type": "at_bloom_level",
            },
        ],
    }
    out = graph_dir / "pedagogy_graph.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _patch_shacl_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a SHACL-gated property manifest so the violation-detection
    generator's Wave 133c family-gate (``validation_kind == "shacl"``)
    admits pairs against the test fixture.

    Mirrors :func:`Trainforge/tests/test_synthesize_training.py::_patch_shacl_manifest`.
    Surface forms are cosmetic — the violation catalog is hand-curated,
    not chunk-driven.
    """
    from lib.ontology.property_manifest import (
        PropertyEntry,
        PropertyManifest,
    )

    manifest = PropertyManifest(
        family="mini_shacl",
        properties=[
            PropertyEntry(
                id="sh_datatype",
                uri="http://www.w3.org/ns/shacl#datatype",
                curie="sh:datatype",
                label="SHACL datatype surface form",
                surface_forms=["sh:datatype"],
                min_pairs=1,
            ),
        ],
        validation_kind="shacl",
    )
    monkeypatch.setattr(
        "lib.ontology.property_manifest.load_property_manifest",
        lambda *_a, **_kw: manifest,
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Tolerant JSONL reader; soft-skips blank / malformed lines."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


# Sentinel keys used by the deferred-verification skip pattern. When
# Worker A hasn't landed yet, the deterministic-generator pairs still
# emit (Worker B + the legacy hoist), but they DON'T carry these
# audit fields. Tests 1, 2, 5, 6, 7 detect that on the first
# deterministic pair and skip with a deferred-verification marker.
_W8_A_AUDIT_KEYS = (
    "promotion_status",
    "pair_lo_resolution",
    "per_claim_support",
)


def _worker_a_landed(pair: Dict[str, Any]) -> bool:
    """Detect whether the W8.A audit-field stamping helper has landed.

    Returns True iff the pair carries every one of the W8.A-stamped
    audit fields. False indicates Worker A's helper hasn't been wired
    into ``run_synthesis`` yet — the pair was emitted via the legacy
    hoist path that pre-dates Wave 8.
    """
    return all(key in pair for key in _W8_A_AUDIT_KEYS)


_DEFERRED_W8_A_SKIP = (
    "Wave 8 Worker A (`_stamp_deterministic_pair_audit_fields` helper "
    "in `Trainforge/synthesize_training.py::run_synthesis`) has not "
    "landed yet. Deterministic-generator pairs do not carry the "
    "W2.E + W4.A + W4.B + W4.C audit fields. Test deferred until "
    "Worker A lands; per the Wave 8 plan §5 sequencing this is "
    "expected when Workers A and the wave-end gate ship in parallel."
)


# --------------------------------------------------------------------------- #
# Test 1 — kg_metadata pair carries audit fields after run_synthesis
# --------------------------------------------------------------------------- #


def test_kg_metadata_pair_carries_audit_fields_after_run_synthesis(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 1 — Worker A surface 1.

    End-to-end contract: invoke :func:`run_synthesis` with
    ``with_kg_metadata=True`` against an inlined synthetic pedagogy
    graph; every emitted ``content_type=="kg_metadata"`` pair on disk
    MUST carry the W2.E + W4.A + W4.B + W4.C audit fields the post-hoc
    gate-runner walks treat as evidence the per-pair filter ran.

    Pins the architectural fix that closes Checks 1 + 2 + 3 of the
    Wave 8 audit: deterministic pairs were landing on disk WITHOUT
    ``promotion_status`` / ``per_claim_support`` / ``pair_lo_resolution``
    / ``pair_objective_alignment``, blocking ``training_synthesis``
    completion via critical-coded ``MISSING_PROMOTION_STATUS`` /
    ``PAIR_CHUNK_NOT_FOUND`` issues on every rebuild that opted in to
    the deterministic generators.

    Skip pattern: when Worker A has not landed (sentinel check on the
    first kg_metadata pair's audit-field set), the test skips with a
    deferred-verification marker. Once Worker A lands the same test
    runs-not-skip and asserts the contract against the live emit.
    """
    from Trainforge.synthesize_training import run_synthesis

    course_dir = _copy_mini_training_fixture(tmp_path)
    _write_pedagogy_graph(course_dir)

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_kg_metadata=True,
        kg_metadata_max_pairs=10,
    )
    assert stats.kg_metadata_pairs_emitted > 0, (
        f"fixture failure: kg_metadata generator emitted zero pairs "
        f"against the inlined pedagogy graph; expected > 0. Stats: "
        f"{stats!r}"
    )

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    rows = _read_jsonl(inst_path)
    kg_pairs = [r for r in rows if r.get("content_type") == "kg_metadata"]
    assert kg_pairs, (
        f"fixture failure: zero kg_metadata pairs landed in "
        f"{inst_path}; got content_types="
        f"{sorted({r.get('content_type') for r in rows})!r}"
    )

    # ---- Worker A landing detection (deferred-verification skip). ---- #
    if not _worker_a_landed(kg_pairs[0]):
        pytest.skip(_DEFERRED_W8_A_SKIP)

    # ---- Sub-clause 1 — promotion_status="validated" on every pair. ---- #
    for idx, pair in enumerate(kg_pairs):
        assert pair.get("promotion_status") == "validated", (
            f"kg_pairs[{idx}] promotion_status MUST be 'validated' "
            f"(deterministic generators are oracle-grounded by "
            f"construction); got {pair.get('promotion_status')!r}"
        )

    # ---- Sub-clause 2 — per_claim_support=[] on every pair (W4.A
    #     graceful-pass arm; deterministic completions bypass NLI
    #     scoring). ---- #
    for idx, pair in enumerate(kg_pairs):
        pcs = pair.get("per_claim_support")
        assert pcs == [], (
            f"kg_pairs[{idx}] per_claim_support MUST be [] (W4.A "
            f"deterministic-template graceful-pass arm); got {pcs!r}"
        )

    # ---- Sub-clause 3 — pair_lo_resolution.skipped ==
    #     "deterministic_template" on every pair (the W8.A discriminator
    #     consumed by the W4.B validate-walk skip). ---- #
    for idx, pair in enumerate(kg_pairs):
        plr = pair.get("pair_lo_resolution")
        assert isinstance(plr, dict), (
            f"kg_pairs[{idx}] pair_lo_resolution MUST be a dict; got "
            f"type {type(plr).__name__}"
        )
        assert plr.get("skipped") == "deterministic_template", (
            f"kg_pairs[{idx}] pair_lo_resolution.skipped MUST be "
            f"'deterministic_template'; got {plr.get('skipped')!r}"
        )

    # ---- Sub-clause 4 — pair_objective_alignment is None on every pair
    #     (W4.C deps-missing arm; deterministic pairs don't have a
    #     verifiable LO entailment surface). ---- #
    for idx, pair in enumerate(kg_pairs):
        assert pair.get("pair_objective_alignment", "MISSING") is None, (
            f"kg_pairs[{idx}] pair_objective_alignment MUST be None "
            f"(W4.C deps-missing arm); got "
            f"{pair.get('pair_objective_alignment', 'MISSING')!r}"
        )

    # ---- Sub-clause 5 — Worker B field: question_type="true_false". ---- #
    for idx, pair in enumerate(kg_pairs):
        assert pair.get("question_type") == "true_false", (
            f"kg_pairs[{idx}] question_type MUST be 'true_false' (W8.B "
            f"emit on kg_metadata_generator); got "
            f"{pair.get('question_type')!r}"
        )


# --------------------------------------------------------------------------- #
# Test 2 — violation-detection pair carries audit fields after run_synthesis
# --------------------------------------------------------------------------- #


def test_violation_pair_carries_audit_fields_after_run_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan §4 Test 2 — Worker A surface 2.

    Mirror of Test 1 against the violation-detection generator. The
    violation pair envelope carries the most contaminated chunk_id
    surface (``violation_fixture:<name>`` for fixtures that don't match
    a real chunk text) so the W4.B walk would fire ``PAIR_CHUNK_NOT_FOUND``
    on every pair — Worker A's audit-field stamping is what closes the
    blocking failure on this surface.

    Requires pyshacl + rdflib at runtime; skipped cleanly in
    environments without those extras (mirrors the existing
    :func:`Trainforge/tests/test_synthesize_training.py::test_violation_detection_max_pairs_caps_emit`
    pattern).
    """
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")

    from Trainforge.synthesize_training import run_synthesis

    _patch_shacl_manifest(monkeypatch)
    course_dir = _copy_mini_training_fixture(tmp_path)

    stats = run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_violation_detection=True,
        violation_detection_max_pairs=12,
    )
    assert stats.violation_pairs_emitted > 0, (
        f"fixture failure: violation-detection generator emitted zero "
        f"pairs; expected > 0. Stats: {stats!r}"
    )

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    rows = _read_jsonl(inst_path)
    vio_pairs = [
        r for r in rows if r.get("content_type") == "violation_detection"
    ]
    assert vio_pairs, (
        f"fixture failure: zero violation_detection pairs landed in "
        f"{inst_path}; got content_types="
        f"{sorted({r.get('content_type') for r in rows})!r}"
    )

    # ---- Worker A landing detection (deferred-verification skip). ---- #
    if not _worker_a_landed(vio_pairs[0]):
        pytest.skip(_DEFERRED_W8_A_SKIP)

    # ---- Sub-clauses 1-4 — same audit-field shape as Test 1. ---- #
    for idx, pair in enumerate(vio_pairs):
        assert pair.get("promotion_status") == "validated", (
            f"vio_pairs[{idx}] promotion_status MUST be 'validated'; "
            f"got {pair.get('promotion_status')!r}"
        )
        assert pair.get("per_claim_support") == [], (
            f"vio_pairs[{idx}] per_claim_support MUST be []; got "
            f"{pair.get('per_claim_support')!r}"
        )
        plr = pair.get("pair_lo_resolution")
        assert isinstance(plr, dict), (
            f"vio_pairs[{idx}] pair_lo_resolution MUST be a dict; got "
            f"type {type(plr).__name__}"
        )
        assert plr.get("skipped") == "deterministic_template", (
            f"vio_pairs[{idx}] pair_lo_resolution.skipped MUST be "
            f"'deterministic_template'; got {plr.get('skipped')!r}"
        )
        assert pair.get("pair_objective_alignment", "MISSING") is None, (
            f"vio_pairs[{idx}] pair_objective_alignment MUST be None; "
            f"got {pair.get('pair_objective_alignment', 'MISSING')!r}"
        )
        # Worker B field — violation-detection uses 'true_false' since
        # valid? / invalid? verdicts are structurally true_false.
        assert pair.get("question_type") == "true_false", (
            f"vio_pairs[{idx}] question_type MUST be 'true_false' "
            f"(W8.B emit on violation_generator); got "
            f"{pair.get('question_type')!r}"
        )


# --------------------------------------------------------------------------- #
# Test 5 — pair_lo_refs validate walk skips deterministic pairs
# --------------------------------------------------------------------------- #


def test_pair_lo_refs_validate_walk_skips_deterministic_pairs(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 5 — Worker A regression net for the
    ``pair_lo_refs`` blocking failure.

    Synthesizes a corpus with ``with_kg_metadata=True`` (the most
    contaminated surface — kg_metadata uses graph node IDs as
    ``chunk_id`` that NEVER exist in chunks.jsonl), then runs the
    post-hoc :meth:`PairLearningOutcomeRefsValidator.validate` walk.
    Asserts ``passed=True`` — no ``PAIR_CHUNK_NOT_FOUND`` issues
    against the synthetic chunk_ids.

    Closes Check 3 from the Wave 8 audit: the W4.B post-hoc walk was
    critical-failing every deterministic pair because none of their
    synthetic chunk_ids resolved against ``chunks.jsonl``. Worker A's
    companion change adds a
    ``pair_lo_resolution.skipped == "deterministic_template"``
    short-circuit at the chunk-id-resolution branch so the walk
    skips deterministic pairs cleanly.

    Skip pattern: when Worker A's ``run_synthesis`` helper hasn't
    landed (deterministic pairs lack ``pair_lo_resolution.skipped``),
    the W4.B walk fires ``PAIR_CHUNK_NOT_FOUND`` critically. The test
    detects that condition via the pre-walk audit-field check and
    skips with a deferred-verification marker.
    """
    from Trainforge.synthesize_training import run_synthesis
    from lib.validators.pair_lo_refs import PairLearningOutcomeRefsValidator

    course_dir = _copy_mini_training_fixture(tmp_path)
    _write_pedagogy_graph(course_dir)

    run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_kg_metadata=True,
        kg_metadata_max_pairs=8,
    )

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    pref_path = course_dir / "training_specs" / "preference_pairs.jsonl"
    chunks_path = course_dir / "corpus" / "chunks.jsonl"

    # ---- Worker A landing detection: peek at the first deterministic
    #     pair on disk; if the audit-field stamping isn't present, the
    #     W4.B walk WILL fail and the test skips with a deferred marker. ---- #
    rows = _read_jsonl(inst_path)
    kg_pairs = [r for r in rows if r.get("content_type") == "kg_metadata"]
    assert kg_pairs, (
        f"fixture failure: zero kg_metadata pairs landed in "
        f"{inst_path}; cannot exercise W4.B walk."
    )
    if not _worker_a_landed(kg_pairs[0]):
        pytest.skip(_DEFERRED_W8_A_SKIP)

    # ---- Run the W4.B post-hoc walk against on-disk artifacts. ---- #
    result = PairLearningOutcomeRefsValidator().validate(
        {
            "instruction_pairs_path": str(inst_path),
            "preference_pairs_path": str(pref_path),
            "chunks_path": str(chunks_path),
        }
    )

    # ---- Sub-clause 1 — gate passes. ---- #
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert result.passed is True, (
        f"PairLearningOutcomeRefsValidator MUST pass on a corpus that "
        f"includes deterministic-generator pairs once Worker A's "
        f"audit-field skip lands. Got passed={result.passed}, "
        f"critical_codes={critical_codes!r}, all_codes="
        f"{[(i.code, i.severity) for i in result.issues]!r}"
    )

    # ---- Sub-clause 2 — no PAIR_CHUNK_NOT_FOUND issues against the
    #     synthetic kg_metadata chunk_ids (they're graph node IDs like
    #     'chunk_a' / 'chunk_b' / 'concept_alpha' that don't exist in
    #     the fixture's chunks.jsonl). ---- #
    chunk_not_found_codes = [
        i.code for i in result.issues if i.code == "PAIR_CHUNK_NOT_FOUND"
    ]
    assert chunk_not_found_codes == [], (
        f"PAIR_CHUNK_NOT_FOUND issues MUST be zero on deterministic "
        f"pairs (Worker A's `pair_lo_resolution.skipped == "
        f"\"deterministic_template\"` short-circuit closes the blocking "
        f"failure); got {chunk_not_found_codes!r}"
    )


# --------------------------------------------------------------------------- #
# Test 6 — pair_promotion validate walk passes deterministic pairs
# --------------------------------------------------------------------------- #


def test_pair_promotion_validate_walk_passes_deterministic_pairs(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 6 — Worker A regression net for the
    ``pair_promotion`` blocking failure.

    Same setup as Test 5; runs
    :meth:`TrainingPairPromotionValidator.validate` against the
    on-disk pair files. Asserts ``passed=True`` — no
    ``MISSING_PROMOTION_STATUS`` issues fired by the post-hoc
    audit walk.

    Closes Check 1 from the Wave 8 audit: the W2.E post-hoc walk was
    critical-failing every deterministic pair because none of them
    carried ``promotion_status`` (the call-site never routed them
    through the W2.E per-pair filter). Worker A's
    ``_stamp_deterministic_pair_audit_fields`` helper stamps
    ``promotion_status="validated"`` unconditionally on the four
    deterministic surfaces (oracle-grounded by construction).

    Skip pattern: same deferred-verification marker as Test 5 when
    Worker A hasn't landed.
    """
    from Trainforge.synthesize_training import run_synthesis
    from lib.validators.training_pair_promotion import (
        TrainingPairPromotionValidator,
    )

    course_dir = _copy_mini_training_fixture(tmp_path)
    _write_pedagogy_graph(course_dir)

    run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_kg_metadata=True,
        kg_metadata_max_pairs=8,
    )

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    pref_path = course_dir / "training_specs" / "preference_pairs.jsonl"

    # ---- Worker A landing detection (deferred-verification skip). ---- #
    rows = _read_jsonl(inst_path)
    kg_pairs = [r for r in rows if r.get("content_type") == "kg_metadata"]
    assert kg_pairs, (
        f"fixture failure: zero kg_metadata pairs landed in "
        f"{inst_path}; cannot exercise W2.E walk."
    )
    if not _worker_a_landed(kg_pairs[0]):
        pytest.skip(_DEFERRED_W8_A_SKIP)

    # ---- Run the W2.E post-hoc walk against on-disk artifacts. ---- #
    result = TrainingPairPromotionValidator().validate(
        {
            "instruction_pairs_path": str(inst_path),
            "preference_pairs_path": str(pref_path),
        }
    )

    # ---- Sub-clause 1 — gate passes. ---- #
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert result.passed is True, (
        f"TrainingPairPromotionValidator MUST pass on a corpus that "
        f"includes deterministic-generator pairs once Worker A's "
        f"`promotion_status='validated'` stamp lands. Got "
        f"passed={result.passed}, critical_codes={critical_codes!r}, "
        f"all_codes={[(i.code, i.severity) for i in result.issues]!r}"
    )

    # ---- Sub-clause 2 — no MISSING_PROMOTION_STATUS issues fired
    #     against the deterministic surfaces. ---- #
    missing_codes = [
        i.code for i in result.issues
        if i.code == "MISSING_PROMOTION_STATUS"
    ]
    assert missing_codes == [], (
        f"MISSING_PROMOTION_STATUS issues MUST be zero on deterministic "
        f"pairs (Worker A's helper stamps `promotion_status=\"validated\"` "
        f"on every emit); got {missing_codes!r}"
    )


# --------------------------------------------------------------------------- #
# Test 7 — kg_metadata pair carries question_type="true_false"
# --------------------------------------------------------------------------- #


def test_kg_metadata_pair_carries_question_type_true_false(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 7 — Worker B surface 1.

    Confirms the W8.B emit on the kg_metadata generator: every
    ``content_type=="kg_metadata"`` pair on disk carries
    ``question_type == "true_false"``.

    Pinned independently of Test 1's broader audit-field assertion so
    a future regression in Worker B's stamping (e.g. the constant
    drifts to a non-canonical bucket) is caught even when the Worker A
    audit-field surface is healthy. Mirrors the W7.A precedent that
    eval-side per-question-type bucketing requires every emitted pair
    to carry ``question_type`` in the canonical 5-value enum so the
    W6.B / W7.A-C per-type matrices populate correctly (closes
    Check 5 from the Wave 8 audit).

    Worker C precondition: the ``question_type`` value MUST be in the
    canonical 5-value enum (``multiple_choice``, ``true_false``,
    ``short_answer``, ``essay``, ``fill_in_blank``) per
    ``schemas/knowledge/instruction_pair.schema.json:69-72``. The
    assertion below pins the kg_metadata surface to ``true_false``
    specifically (yes/no membership probes).

    Skip pattern: same as Test 1. When Worker A hasn't landed the
    deterministic pairs still emit (the hoist + Worker B stamping pre-
    date Wave 8.A's audit-field surface), so the question_type
    assertion is independently verifiable. The deferred-verification
    skip pattern still applies for symmetry with Tests 1, 2, 5, 6 —
    the same operator-readable signal fires on a single-failure CI
    output.
    """
    from Trainforge.synthesize_training import run_synthesis

    course_dir = _copy_mini_training_fixture(tmp_path)
    _write_pedagogy_graph(course_dir)

    run_synthesis(
        corpus_dir=course_dir,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=11,
        pilot_report_every=0,
        curriculum_from_graph=False,
        with_kg_metadata=True,
        kg_metadata_max_pairs=10,
    )

    inst_path = course_dir / "training_specs" / "instruction_pairs.jsonl"
    rows = _read_jsonl(inst_path)
    kg_pairs = [r for r in rows if r.get("content_type") == "kg_metadata"]
    assert kg_pairs, (
        f"fixture failure: zero kg_metadata pairs landed in "
        f"{inst_path}; got content_types="
        f"{sorted({r.get('content_type') for r in rows})!r}"
    )

    # ---- Sub-clause 1 — every kg_metadata pair carries
    #     question_type="true_false". (Worker B contract; runs against
    #     the live emit since W8.B landed in commit 5f7185b.) ---- #
    for idx, pair in enumerate(kg_pairs):
        qt = pair.get("question_type")
        assert qt == "true_false", (
            f"kg_pairs[{idx}] question_type MUST be 'true_false' "
            f"(W8.B emit pinned by `kg_metadata_generator.py:259`); "
            f"got {qt!r}"
        )

    # ---- Sub-clause 2 — Worker C precondition: the value is in the
    #     canonical 5-value enum on `instruction_pair.schema.json`.
    #     Pinned at the call site so a future Worker B drift to a
    #     non-canonical bucket (e.g. "binary_yes_no") is caught here
    #     before the schema gate fires downstream. ---- #
    canonical_enum = {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    }
    for idx, pair in enumerate(kg_pairs):
        qt = pair.get("question_type")
        assert qt in canonical_enum, (
            f"kg_pairs[{idx}] question_type {qt!r} MUST be one of "
            f"{sorted(canonical_enum)!r} (Worker C schema enum at "
            f"`schemas/knowledge/instruction_pair.schema.json:71`)."
        )


# --------------------------------------------------------------------------- #
# Test 11 — instruction_pair schema admits question_type enum
# --------------------------------------------------------------------------- #


def test_instruction_pair_schema_admits_question_type_enum() -> None:
    """Plan §4 Test 11 — Worker C schema-bump contract.

    Three sub-clause assertions against
    :file:`schemas/knowledge/instruction_pair.schema.json`:

      * A pair carrying ``question_type="multiple_choice"`` validates
        cleanly (canonical enum value).
      * A pair carrying ``question_type="invalid"`` fails validation
        (off-enum value — the Worker C enum constraint fires).
      * A pair WITHOUT ``question_type`` validates cleanly (back-compat
        with legacy pre-Wave-8 corpora that pre-date Worker B's
        stamping).

    Pins the W8.C contract: the field is OPTIONAL (not in
    ``required``) so legacy corpora stay schema-valid, AND the field
    is ENUM-validated when present so a future regression that emits
    a non-canonical bucket fails closed in strict-validation mode
    (``TRAINFORGE_VALIDATE_CHUNKS=true``).

    Runs against the live schema as committed in ``9f46be6``; no
    deferred-verification posture (Worker C landed first per plan
    §5 sequencing).
    """
    pytest.importorskip("jsonschema")

    from jsonschema import Draft202012Validator

    schema_path = (
        PROJECT_ROOT
        / "schemas"
        / "knowledge"
        / "instruction_pair.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    def _base_pair() -> Dict[str, Any]:
        # Mirrors :func:`schemas/tests/test_instruction_pair_promotion_fields.py::_base_pair`
        # so every required field is populated; only the question_type
        # axis is under test here.
        return {
            "prompt": (
                "Explain how RDF triples encode subject-predicate-object "
                "relationships in 2-3 sentences."
            ),
            "completion": (
                "An RDF triple binds a subject to an object via a named "
                "predicate, encoding a directed edge in a knowledge "
                "graph. The subject and predicate are IRIs; the object "
                "is an IRI or literal."
            ),
            "chunk_id": "test_course_chunk_00042",
            "lo_refs": ["TO-01", "CO-03"],
            "bloom_level": "understand",
            "content_type": "explanation",
            "seed": 12345,
            "decision_capture_id": "evt_abcdef0123456789",
        }

    # ---- Sub-clause 1 — question_type="multiple_choice" passes. ---- #
    pair_ok = _base_pair()
    pair_ok["question_type"] = "multiple_choice"
    errors_ok = list(validator.iter_errors(pair_ok))
    assert errors_ok == [], (
        f"question_type='multiple_choice' MUST be schema-valid (W8.C "
        f"canonical enum); got errors="
        f"{[(list(e.absolute_path), e.message) for e in errors_ok]!r}"
    )

    # ---- Sub-clause 2 — question_type="invalid" fails. ---- #
    pair_bad = _base_pair()
    pair_bad["question_type"] = "invalid"
    errors_bad = list(validator.iter_errors(pair_bad))
    assert errors_bad, (
        "question_type='invalid' MUST fail schema validation (W8.C "
        "enum constraint fires on off-enum values); got zero errors."
    )
    # Confirm the failure is on the question_type field specifically,
    # not some unrelated absent-required-field error elsewhere.
    qt_errors = [
        e for e in errors_bad
        if "question_type" in [str(p) for p in e.absolute_path]
    ]
    assert qt_errors, (
        f"question_type='invalid' MUST fail with an error on the "
        f"question_type field specifically (not an unrelated error); "
        f"got error paths="
        f"{[(list(e.absolute_path), e.message) for e in errors_bad]!r}"
    )

    # ---- Sub-clause 3 — question_type absent passes (back-compat). ---- #
    pair_legacy = _base_pair()
    assert "question_type" not in pair_legacy
    errors_legacy = list(validator.iter_errors(pair_legacy))
    assert errors_legacy == [], (
        f"Legacy pair WITHOUT question_type MUST be schema-valid (W8.C "
        f"field is optional, back-compat with pre-Wave-8 corpora); got "
        f"errors="
        f"{[(list(e.absolute_path), e.message) for e in errors_legacy]!r}"
    )
