"""Wave 2 end-of-wave contract gate: validators + aggregators.

Covers the four Wave 2 net-new validators (DistractorStructuralValidator,
PaddedDistractorValidator, TrainingPairPromotionValidator,
ClaimSupportValidator) plus the two post-loop aggregators.

Every fixture is inlined locally rather than imported from a per-worker test
module, so renaming any of those modules cannot silently disable this gate.

Two deliberate deviations from the plan this gate was written against:

* Test 5 loops the 10 deliberately-bad pairs through ``validate_pair``
  individually because the promotion filter is per-pair, not batch-aware.
  Rejection counts come from the test's own Counter — the production stats
  live on ``synthesize_training.SynthesisStats`` and are not reachable from
  the validator.
* Test 6 asserts the mock NLI was actually CALLED rather than trying to
  assert a vacuous pass. A validator that short-circuits around NLI dispatch
  produces the same passed/action signature as a genuine pass, so call count
  is the only signal that distinguishes them.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Block lives at Courseforge/scripts/blocks.py, which is not an importable
# package root — put its dir on sys.path so `from blocks import Block` works.
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Per-tree test-count floors, so a silent test removal trips the ratchet.
# Counted by AST-walking each tree's ``test_*.py`` and tallying ``def test_``
# declarations — the same algorithm ``_count_test_functions`` runs at gate
# time. These are deliberately NOT ``pytest --collect-only`` counts, which are
# higher because pytest expands parametrized cases; both are monotonic, so the
# ratchet holds either way, but the two must not be mixed.
# --------------------------------------------------------------------------- #

_MIN_VALIDATORS_TESTS = 490
_MIN_AGGREGATORS_TESTS = 43
_MIN_TRAINFORGE_TESTS = 1637
_MIN_SCHEMAS_TESTS = 345


# --------------------------------------------------------------------------- #
# Net-new validator inventory: (label, per-validator-test relative path,
# decision_type strings the validator MUST emit).
# Aggregators are tracked separately because their emit surface is the
# workflow-runner aggregator, not a per-validator file.
# --------------------------------------------------------------------------- #

_W2_VALIDATORS: List[Dict[str, Any]] = [
    {
        "label": "W2.C DistractorStructuralValidator",
        "test_path": "lib/validators/tests/test_distractor_structural.py",
        "decision_types": ["distractor_structural_check"],
    },
    {
        "label": "W2.D PaddedDistractorValidator",
        "test_path": "lib/validators/tests/test_padded_distractor.py",
        "decision_types": ["padded_distractor_check"],
    },
    {
        "label": "W2.E TrainingPairPromotionValidator",
        "test_path": "lib/validators/tests/test_training_pair_promotion.py",
        "decision_types": ["training_pair_promotion_check"],
    },
    {
        "label": "W2.F ClaimSupportValidator",
        "test_path": "lib/validators/tests/test_claim_support.py",
        "decision_types": ["claim_support_check"],
    },
]

# Every decision_type the schema enum MUST admit: the four validators, the
# two aggregators, and the generator-side ``distractor_padding_skipped`` event
# emitted from ``assessment_generator._generate_multiple_choice`` when the
# padded-fallback path is short-circuited.
_W2_REQUIRED_DECISION_TYPES: Tuple[str, ...] = (
    "distractor_structural_check",         # W2.C
    "padded_distractor_check",             # W2.D regression-net validator
    "distractor_padding_skipped",          # W2.D generator-side skip
    "training_pair_promotion_check",       # W2.E
    "claim_support_check",                 # W2.F
    "courseforge_validation_aggregated",   # W2.A
    "trainforge_quality_aggregated",       # W2.B
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _count_test_functions(tree: Path) -> Tuple[int, List[str]]:
    """AST-walk every ``test_*.py`` under ``tree`` and count ``def test_``
    declarations.

    Duplicated verbatim across the sibling wave gates so every gate's count
    contract is measured the same way and stays comparable.
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


def _load_decision_event_enum() -> set:
    """Read ``schemas/events/decision_event.schema.json`` and return the
    full ``properties.decision_type.enum`` value as a set."""
    schema_path = (
        PROJECT_ROOT
        / "schemas"
        / "events"
        / "decision_event.schema.json"
    )
    doc = json.loads(schema_path.read_text(encoding="utf-8"))
    enum_list = (
        doc.get("properties", {})
        .get("decision_type", {})
        .get("enum", [])
    )
    return set(enum_list)


def _ast_test_function_source(test_path: Path) -> str:
    """Return the raw source of every ``def test_*`` function in ``test_path``.

    Scoped to test-function bodies so module-level fixtures don't pollute the
    capture-wiring grep. On parse error, returns the whole file: over-matching
    is preferable to missing a real wiring assertion.
    """
    if not test_path.exists():
        return ""
    try:
        text = test_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        module = ast.parse(text)
    except SyntaxError:
        return text
    chunks: List[str] = []
    lines = text.splitlines(keepends=True)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                # ast nodes carry 1-based line numbers; end_lineno is
                # the inclusive end. Slice the source verbatim so
                # grep-style assertions match.
                start = node.lineno - 1
                end = getattr(node, "end_lineno", None) or start + 1
                chunks.append("".join(lines[start:end]))
    if not chunks:
        # Defensive: if AST didn't surface any test functions (e.g.
        # exotic decorators), fall back to the whole source. Test 1
        # is then permissive but not silent.
        return text
    return "\n".join(chunks)


class _RecordingCapture:
    """Minimal capture stub recording every ``log_decision`` payload.

    Inlined so a rename of any per-worker capture helper can't silently break
    this gate. Name kept identical across the sibling wave gates so they stay
    greppable as one family.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Test 1 — every new validator has at least one decision-emit test
# --------------------------------------------------------------------------- #


_CAPTURE_WIRING_NEEDLES: Tuple[str, ...] = (
    "_emit_decision",
    "decision_capture",
    "RecordingCapture",
    "log_decision",
    "capture.log_decision",
)


def test_1_each_w2_validator_has_capture_wiring_test() -> None:
    """Every Wave 2 validator has at least one decision-emit test.

    For each net-new validator, AST-walk its per-validator test file's
    ``def test_*`` bodies and assert at least one references a capture-wiring
    needle. Fires when a refactor drops the capture-wiring assertion.
    """
    failures: List[str] = []
    for entry in _W2_VALIDATORS:
        test_path = PROJECT_ROOT / entry["test_path"]
        if not test_path.is_file():
            failures.append(
                f"{entry['label']}: per-validator test file is missing "
                f"({entry['test_path']}). The wave-end gate requires "
                f"each Wave 2 net-new validator carry a sibling test "
                f"file with a capture-wiring assertion."
            )
            continue
        source = _ast_test_function_source(test_path)
        if not any(needle in source for needle in _CAPTURE_WIRING_NEEDLES):
            failures.append(
                f"{entry['label']}: no capture-wiring assertion found "
                f"in {entry['test_path']} ``def test_*`` function "
                f"bodies. Searched for: "
                f"{list(_CAPTURE_WIRING_NEEDLES)!r}. Mirror the precedents "
                "at ``lib/validators/tests/test_chunk_corpus_capture_wiring.py`` "
                "and ``lib/validators/tests/test_wcag_capture_wiring.py``."
            )
    assert not failures, (
        "Wave 2 capture-wiring contract violated:\n  " + "\n  ".join(failures)
    )


# --------------------------------------------------------------------------- #
# Test 2 — every new validator's decision_type is in the schema enum
# --------------------------------------------------------------------------- #


def test_2_w2_decision_types_in_schema_enum() -> None:
    """Every Wave 2 decision_type is a member of the schema enum.

    Catches the silent-degrade class where a validator emits a fresh
    decision_type that ``DECISION_VALIDATION_STRICT=true`` drops on the
    floor — the audit trail loses the event with no warning.
    """
    enum_set = _load_decision_event_enum()
    missing: List[str] = []
    for required in _W2_REQUIRED_DECISION_TYPES:
        if required not in enum_set:
            missing.append(required)
    assert not missing, (
        "Wave 2 decision_type enum members missing from "
        "schemas/events/decision_event.schema.json::properties."
        f"decision_type.enum: {missing!r}. Add each missing entry to "
        "the alphabetised enum list before closing Wave 2."
    )


# --------------------------------------------------------------------------- #
# Test 3 — anti-silent-degradation: aggregator output IS consumed
# --------------------------------------------------------------------------- #


def test_3_trainforge_assessment_quality_report_has_consumer() -> None:
    """The assessment-quality aggregator's output is actually consumed.

    Greps the repo for ``trainforge_assessment_quality_report.json`` and
    requires >= 2 hits: the writing aggregator AND at least one non-test
    reader. Catches the "we wrote a report nobody reads" silent-success
    class, where the aggregator keeps passing while its output is dead.
    """
    target = "trainforge_assessment_quality_report.json"
    # Prefer ``rg`` if available (faster + ignores binaries cleanly);
    # fall back to ``grep -rln``. Both should land the same set on a
    # clean checkout.
    cmd = [
        "grep", "-rln",
        "--include=*.py",
        "--include=*.md",
        "--include=*.yaml",
        "--exclude-dir=__pycache__",
        "--exclude-dir=.git",
        target,
        str(PROJECT_ROOT),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.fail(
            f"grep -rln failed to execute against {PROJECT_ROOT}: {exc!r}"
        )

    hits = [
        line.strip() for line in (proc.stdout or "").splitlines()
        if line.strip()
    ]

    # Partition into writer surface + non-test readers + tests.
    aggregator_writers = [
        h for h in hits
        if "lib/aggregators/trainforge_assessment_quality_report.py" in h
        or h.endswith("lib/aggregators/__init__.py")
    ]
    test_files = [
        h for h in hits
        if "/tests/" in h
        or h.startswith(str(PROJECT_ROOT / "schemas" / "tests"))
        or "test_" in Path(h).name
    ]
    workflow_writer = [
        h for h in hits
        if "MCP/core/workflow_runner.py" in h
    ]
    non_test_readers = [
        h for h in hits
        if h not in test_files
        and h not in aggregator_writers
        and h not in workflow_writer
        # README / CLAUDE.md mentions are documentation, not consumers.
        and not h.endswith("CLAUDE.md")
    ]

    # Writer presence is guaranteed by the W2.B landing; assert it
    # explicitly so a refactor that silently moved the writer file
    # trips this gate too.
    assert aggregator_writers or workflow_writer, (
        "Wave 2 W2.B writer surface is missing — expected at least one "
        "of ``lib/aggregators/trainforge_assessment_quality_report.py`` "
        "or ``MCP/core/workflow_runner.py`` to reference "
        f"{target!r}; got hits: {hits!r}"
    )

    if not non_test_readers:
        # Plan §Test 3 escape hatch: Wave 3 G1 is the canonical
        # consumer but doesn't exist yet. The Wave 2-era sibling read
        # in ``MCP/tools/trainforge_tools.py`` is the documented
        # alternative. If neither is present, fail with a clear
        # remediation pointer.
        pytest.fail(
            "Wave 2 W2.B aggregator emits "
            f"``{target}`` but no non-test, non-writer surface reads "
            "it. Plan §Test 3 requires either (a) the Wave 3 master "
            "promotion-chain aggregator G1 OR (b) a sibling read in "
            "``MCP/tools/trainforge_tools.py::get_trainforge_status`` "
            "(the pre-existing ``eval_report.json`` parallel read at "
            ":1056 establishes the pattern). Wire one or the other "
            "before closing Wave 2.\n"
            f"All grep hits: {hits!r}\n"
            f"Aggregator writers: {aggregator_writers!r}\n"
            f"Workflow writer: {workflow_writer!r}\n"
            f"Test files: {test_files!r}\n"
            f"Non-test, non-writer surfaces: {non_test_readers!r}"
        )

    # Total surface count: writer(s) + reader(s) >= 2.
    assert len(aggregator_writers) + len(workflow_writer) + len(non_test_readers) >= 2, (
        f"Expected at least 2 references to ``{target}`` (writer + "
        f"reader); got {hits!r}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — anti-silent-degradation: padded-distractor regression
# --------------------------------------------------------------------------- #


def test_4_padded_distractor_fallback_eliminated() -> None:
    """The padded-distractor fallback stays dead, and its skip stays visible.

    Constructs a chunk that would have triggered the removed
    ``while len(distractors) < 3:`` padding fallback (one extractable key
    term, zero misconceptions → fewer than 3 plausible distractors resolve),
    runs the generator end-to-end, and asserts:

    * NO emitted distractor matches
      ``r"^(A |An )?(concept|topic|item|element) unrelated to "``.
    * At least one ``SkippedItem`` with
      ``reason="padded_distractor_fallback_eliminated"`` appears in
      ``AssessmentData.skipped_items[]``.
    * The generator-side ``distractor_padding_skipped`` decision event
      lands in the capture stream.

    Catches BOTH the silent-skip regression (padding removed without a
    SkippedItem emit) and the silent-reintroduction regression (the padding
    template comes back).
    """
    import re

    from Trainforge.generators.assessment_generator import (
        AssessmentGenerator,
        SkippedItem,
    )

    # Force the precondition: the chunk yields exactly ONE definition via the
    # ``X is defined as Y`` pattern, no misconceptions, and no other definable
    # terms. The removed code path padded the second + third distractor with
    # the literal ``"A concept unrelated to {term} in this context"`` template.
    chunk_text = (
        "Provenance is defined as the documented chain of custody for a "
        "data artifact across its lifecycle in a research workflow. "
        "Provenance applies broadly to scientific computing."
    )
    chunk = {
        "id": "synthetic_chunk_w2d",
        "text": chunk_text,
        "concept_tags": ["provenance"],
        # Empty misconceptions[] ensures the misconception-distractor
        # strategy yields zero distractors.
        "misconceptions": [],
    }

    capture = _RecordingCapture()
    generator = AssessmentGenerator(
        capture=capture,
        check_leaks=False,
    )

    # Drive the MCQ path. ``bloom_levels=["remember"]`` is the
    # canonical level for the "Which best describes <term>?" stem
    # template.
    assessment = generator.generate(
        course_code="WAVE2_GATE",
        objective_ids=["TO-01"],
        bloom_levels=["remember"],
        question_count=3,
        source_chunks=[chunk],
    )

    # Assertion 1 — NO distractor matches the canonical pad pattern.
    pad_re = re.compile(
        r"^(?:A |An )?(?:concept|topic|item|element) unrelated to ",
        re.IGNORECASE,
    )
    pad_hits: List[Tuple[str, str]] = []
    for q in assessment.questions:
        for choice in q.choices or ():
            text = str(choice.get("text", "")).strip()
            # Strip a single ``<p>...</p>`` wrapper for robustness.
            stripped = re.sub(r"^<[^>]+>\s*", "", text)
            stripped = re.sub(r"\s*</[^>]+>$", "", stripped)
            if pad_re.search(stripped):
                pad_hits.append((q.question_id, stripped))
    assert not pad_hits, (
        "Pre-W2.D padded-distractor template re-introduced into "
        f"AssessmentGenerator output: {pad_hits!r}. The W2.D fix at "
        "Trainforge/generators/assessment_generator.py "
        "(``_generate_multiple_choice``) MUST short-circuit to a "
        "SkippedItem when fewer than 3 plausible distractors resolve."
    )

    # Assertion 2 — at least one SkippedItem with the canonical reason.
    padded_skips = [
        s for s in assessment.skipped_items
        if isinstance(s, SkippedItem)
        and s.reason == "padded_distractor_fallback_eliminated"
    ]
    assert padded_skips, (
        "Expected at least one SkippedItem with "
        "reason='padded_distractor_fallback_eliminated' in "
        f"AssessmentData.skipped_items[]; got skipped_items="
        f"{[s.to_dict() for s in assessment.skipped_items]!r}, "
        f"questions_emitted={len(assessment.questions)}. The W2.D "
        "post-fallback path MUST emit a SkippedItem with this reason "
        "string so the silent-skip class is closed."
    )

    # Assertion 3 — the generator-side decision-capture event lands.
    distractor_padding_events = [
        e for e in capture.events
        if e.get("decision_type") == "distractor_padding_skipped"
    ]
    assert distractor_padding_events, (
        "Expected at least one ``distractor_padding_skipped`` "
        "decision-capture event from the W2.D generator-side fix; "
        f"got decision_types={[e.get('decision_type') for e in capture.events]!r}. "
        "The audit trail MUST surface the count of insufficient-"
        "distractor questions instead of silently corrupting the corpus."
    )

    # Defence-in-depth — the rationale MUST interpolate the resolved
    # distractor count + the canonical reason string so an operator
    # replaying the JSONL audit can attribute each skip.
    rationales = [
        str(e.get("rationale", "")) for e in distractor_padding_events
    ]
    assert any(
        "padded_distractor_fallback_eliminated" in r for r in rationales
    ), (
        "distractor_padding_skipped decision rationale MUST surface "
        "the canonical reason string 'padded_distractor_fallback_"
        f"eliminated'; got rationales={rationales!r}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — anti-silent-degradation: per-pair promotion drop is visible
# --------------------------------------------------------------------------- #


def test_5_per_pair_promotion_drop_is_visible() -> None:
    """Every per-pair promotion drop carries a visible rejection reason.

    Runs 10 deliberately-bad pairs covering the rejection criteria through
    ``validate_pair`` and asserts:

    * The number of pairs marked ``promotion_status="rejected"`` matches
      the count of injected bad pairs.
    * For each rejected pair, ``rejection_reason`` is populated AND
      matches the expected enum value.
    * A per-rejection-reason Counter shows the expected distribution
      (3 placeholder_residue, 2 source_free_generation, etc.).

    Catches the silent-drop class where pairs vanish with no audit log.
    """
    from lib.validators.training_pair_promotion import (
        TrainingPairPromotionValidator,
    )

    # Stub embedder + bloom classifier inlined so this gate doesn't depend on
    # the per-worker fixtures, which could be renamed without notice.

    class _StubEmbedder:
        """Bag-of-tokens cosine embedder; no ML deps."""

        def __init__(self) -> None:
            self._vocab: Dict[str, int] = {}

        def _tokenise(self, text: str) -> List[str]:
            return [t.lower() for t in text.split() if t.strip()]

        def encode(self, text: str) -> List[float]:
            tokens = self._tokenise(text)
            for tok in tokens:
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)
            vec = [0.0] * len(self._vocab)
            for tok in tokens:
                vec[self._vocab[tok]] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            return vec

    class _StubBloomEnsemble:
        def __init__(
            self, level: str = "remember", score: float = 0.85
        ) -> None:
            self._level = level
            self._score = score

        def classify(self, text: str) -> Dict[str, Any]:
            return {
                "winner_level": self._level,
                "winner_score": self._score,
                "dispersion": 0.0,
                "per_member": [],
            }

    # The chunk text has substantial overlap with the well-formed
    # prompts so unanswerable_stem only fires on the deliberately
    # alien prompts.
    chunk_text = (
        "RDF is a declarative graph data model. "
        "An RDF triple has the form subject predicate object. "
        "RDF is the W3C standard for representing knowledge as graphs."
    )
    chunk = {"id": "rdf_shacl_chunk_00001", "text": chunk_text}

    def _good_instruction_pair(**overrides: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "chunk_id": "rdf_shacl_chunk_00001",
            "source_chunk_id": "rdf_shacl_chunk_00001",
            "prompt": "What is the structure of an RDF triple in the source?",
            "completion": (
                "An RDF triple is composed of a subject, a predicate, "
                "and an object — the canonical declarative graph form."
            ),
            "bloom_level": "remember",
            "content_type": "definition",
            "lo_refs": ["TO-01"],
            "seed": 42,
            "decision_capture_id": "evt_0001",
            "rationale": (
                "Definition recall: the source explicitly states the "
                "subject predicate object structure as the canonical "
                "RDF triple form."
            ),
        }
        base.update(overrides)
        return base

    def _good_preference_pair(**overrides: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "chunk_id": "rdf_shacl_chunk_00001",
            "source_chunk_id": "rdf_shacl_chunk_00001",
            "prompt": "Which best describes the structure of an RDF triple?",
            "chosen": (
                "An RDF triple is composed of a subject, a predicate, "
                "and an object — the declarative graph form."
            ),
            "rejected": (
                "An RDF triple is a procedural function-call signature "
                "with named arguments evaluated left-to-right at runtime."
            ),
            "bloom_level": "understand",
            "content_type": "definition",
            "lo_refs": ["TO-01"],
            "seed": 42,
            "decision_capture_id": "evt_0001",
            "rationale": (
                "Comparative reasoning: the chosen completion preserves "
                "the declarative subject predicate object form while "
                "the rejected encodes a procedural-language misconception."
            ),
        }
        base.update(overrides)
        return base

    # Build the 10 bad pairs + their expected rejection reasons.
    bad_pairs: List[Tuple[Dict[str, Any], str, str]] = []  # (pair, kind, reason)

    # 3 placeholder_residue (instruction kind) — placeholder text in
    # completion matches the W2.E ASSESSMENT_PLACEHOLDER_PATTERNS regexes.
    for marker in (
        "Correct answer based on content of TO-01.",
        "Correct answer based on content for TO-02.",
        "Correct answer based on content of objective_3.",
    ):
        bad_pairs.append((
            _good_instruction_pair(completion=marker),
            "instruction",
            "placeholder_residue",
        ))

    # 2 source_free_generation — drop source_chunk_id entirely.
    for _ in range(2):
        bad = _good_instruction_pair()
        bad.pop("source_chunk_id", None)
        bad_pairs.append((bad, "instruction", "source_free_generation"))

    # 2 generic_rationale — repetitive single-token rationale below
    # the 0.30 richness floor.
    repetitive_rationale = (
        "because " * 20
    ).strip()
    for _ in range(2):
        bad_pairs.append((
            _good_instruction_pair(rationale=repetitive_rationale),
            "instruction",
            "generic_rationale",
        ))

    # 2 weak_distractor (preference kind) — chosen ≈ rejected.
    near_dup_text = (
        "An RDF triple has subject predicate object as its declarative "
        "graph data structure W3C standard form."
    )
    for _ in range(2):
        bad_pairs.append((
            _good_preference_pair(chosen=near_dup_text, rejected=near_dup_text),
            "preference",
            "weak_distractor",
        ))

    # 1 low_bloom_alignment — declared "remember" but the stub
    # ensemble winner is "create" with score above the 0.40 floor.
    bad_pairs.append((
        _good_instruction_pair(bloom_level="remember"),
        "instruction",
        "low_bloom_alignment",
    ))

    assert len(bad_pairs) == 10, (
        f"Test design bug: expected 10 bad pairs; got {len(bad_pairs)}"
    )

    # Two validator instances: the default ensemble agrees on "remember" (so
    # no Bloom rejection), while the second is wired with a "create"
    # classifier so the low_bloom_alignment pair actually fires.
    default_validator = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(level="remember", score=0.85),
    )
    bloom_disagree_validator = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(level="create", score=0.80),
    )

    rejected_count = 0
    rejection_reasons_observed = Counter()
    capture = _RecordingCapture()

    for pair, kind, expected_reason in bad_pairs:
        if expected_reason == "low_bloom_alignment":
            v = bloom_disagree_validator
        else:
            v = default_validator
        status, reason, _new_fields = v.validate_pair(
            pair, kind=kind, chunk=chunk, decision_capture=capture
        )
        assert status == "rejected", (
            f"Pair expecting reason {expected_reason!r} (kind={kind}) "
            f"unexpectedly validated: status={status!r}, reason={reason!r}, "
            f"pair={pair!r}"
        )
        assert reason == expected_reason, (
            f"rejection_reason mismatch: expected {expected_reason!r}, "
            f"got {reason!r} (kind={kind}, pair={pair!r})"
        )
        rejected_count += 1
        rejection_reasons_observed[reason] += 1

    # All 10 rejected — no silent drops.
    assert rejected_count == 10, (
        f"Expected 10 rejections; observed {rejected_count}. The "
        "silent-drop class is the regression this gate catches — every "
        "rejected pair MUST carry a non-empty rejection_reason."
    )

    # Per-reason distribution matches the injected design.
    expected_distribution = Counter({
        "placeholder_residue": 3,
        "source_free_generation": 2,
        "generic_rationale": 2,
        "weak_distractor": 2,
        "low_bloom_alignment": 1,
    })
    assert rejection_reasons_observed == expected_distribution, (
        f"Rejection-reason distribution drift: expected "
        f"{dict(expected_distribution)!r}, got "
        f"{dict(rejection_reasons_observed)!r}. Either the validator "
        "is silently mis-categorising rejections OR a fixture is "
        "tripping a different criterion than designed."
    )

    # Decision-capture stream MUST carry one event per pair (10 total).
    promotion_events = [
        e for e in capture.events
        if e.get("decision_type") == "training_pair_promotion_check"
    ]
    assert len(promotion_events) == 10, (
        f"Expected 10 training_pair_promotion_check decision events "
        f"(one per validate_pair call); got {len(promotion_events)}. "
        "Every per-pair filter call MUST emit exactly one audit event."
    )
    rejected_events = [
        e for e in promotion_events
        if str(e.get("decision", "")).startswith("rejected:")
    ]
    assert len(rejected_events) == 10, (
        f"Expected 10 rejected promotion events; got {len(rejected_events)}. "
        "The audit trail must visibly surface every drop."
    )


# --------------------------------------------------------------------------- #
# Test 6 — claim-support regression
# --------------------------------------------------------------------------- #


class _StubNliClassifier:
    """Drop-in stub for ``NliClassifier``. Records every ``score_pair`` /
    ``score_batch`` call so tests can assert the validator is NOT
    short-circuiting around the NLI surface."""

    def __init__(self, score_for_all: Any) -> None:
        # ``score_for_all`` may be a NliScore instance (used
        # uniformly) or a dict mapping (premise, hypothesis) → NliScore.
        self._uniform = score_for_all
        self.batch_calls: List[List[Tuple[str, str]]] = []
        self.pair_calls: List[Tuple[str, str]] = []

    def score_pair(self, *, premise: str, hypothesis: str):
        self.pair_calls.append((premise, hypothesis))
        if isinstance(self._uniform, dict):
            return self._uniform.get(
                (premise, hypothesis), self._uniform.get("__default__")
            )
        return self._uniform

    def score_batch(self, *, pairs):
        self.batch_calls.append(list(pairs))
        if isinstance(self._uniform, dict):
            return [
                self._uniform.get(
                    pair, self._uniform.get("__default__")
                )
                for pair in pairs
            ]
        return [self._uniform for _ in pairs]


def _build_claim_support_block(
    *,
    block_id: str = "page_01#concept_rdf_0",
    claims: List[Any],
    source_id: str = "dart:rdf_intro#blk_0",
):
    """Build a Block with ``key_claims`` populated + a single source ref."""
    from blocks import Block  # noqa: WPS433

    return Block(
        block_id=block_id,
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "statement": "RDF intro block.",
            "key_claims": claims,
        },
        source_ids=(source_id,),
        source_references=({"sourceId": source_id},),
    )


def test_6_claim_support_contradiction_and_entailment_regression() -> None:
    """Claim-support: contradiction fires, entailment passes, NLI is called.

    Three fixtures:

    * Fixture A (contradiction): block claims "RDF is a procedural
      language", cited chunk states "RDF is a declarative graph data
      model". Mock NLI returns entailment=0.05 / contradiction=0.85.
      Asserts ``passed=True`` (Wave-1.7 warning-only contract is
      symmetric here — every issue ships warning-severity), ``action=
      "regenerate"``, and ``CONTRADICTED_CLAIM`` issue code present.
    * Fixture B (entailed): block claim entailed by the cited chunk.
      Mock NLI returns entailment=0.85. Asserts ``passed=True``,
      ``action=None``, no issues.
    * Fixture C (silent-degrade detection): mock NLI returns
      entailment=0.99 for every pair (the silent-degrade scenario).
      Asserts the validator IS calling NLI (call count > 0) so a
      future short-circuit that returns vacuous-pass without ever
      invoking NLI trips this gate.
    """
    from lib.classifiers.nli_classifier import NliScore
    from lib.validators.claim_support import (
        ClaimSupportValidator,
        _CODE_CONTRADICTED_CLAIM,
    )

    # ---- Fixture A — contradiction ----
    contradicted_claim = "RDF is a procedural language."
    cited_chunk_a = "RDF is a declarative graph data model."
    block_a = _build_claim_support_block(claims=[contradicted_claim])
    contradicted_score = NliScore(
        entailment=0.05, neutral=0.10, contradiction=0.85
    )
    nli_a = _StubNliClassifier(score_for_all=contradicted_score)
    validator_a = ClaimSupportValidator(nli=nli_a)
    result_a = validator_a.validate({
        "blocks": [block_a],
        "source_chunks": {"dart:rdf_intro#blk_0": cited_chunk_a},
    })

    # passed stays True (every issue is warning-severity); the router consumes
    # action="regenerate" instead.
    assert result_a.passed is True, (
        "ClaimSupportValidator ships every Wave 2 W2.F issue at "
        "warning severity; passed must stay True. Got "
        f"passed={result_a.passed!r}, action={result_a.action!r}"
    )
    assert result_a.action == "regenerate", (
        "Expected action='regenerate' on contradiction fixture; got "
        f"action={result_a.action!r}, codes="
        f"{[i.code for i in result_a.issues]!r}"
    )
    codes_a = [i.code for i in result_a.issues]
    assert _CODE_CONTRADICTED_CLAIM in codes_a, (
        f"Expected at least one issue with code "
        f"{_CODE_CONTRADICTED_CLAIM!r} on contradiction fixture; "
        f"got codes={codes_a!r}"
    )
    # NLI MUST have been called — guards against silent short-circuit.
    assert nli_a.batch_calls, (
        "Validator did NOT invoke NLI on contradiction fixture; "
        "either short-circuiting (silent degrade) OR the validator "
        "is not threading the NLI override correctly."
    )

    # ---- Fixture B — entailment ----
    entailed_claim = "RDF triples consist of subject-predicate-object."
    cited_chunk_b = (
        "Each RDF statement is a triple of subject, predicate, and object."
    )
    block_b = _build_claim_support_block(
        block_id="page_01#concept_rdf_1",
        claims=[entailed_claim],
    )
    entailed_score = NliScore(
        entailment=0.85, neutral=0.10, contradiction=0.05
    )
    nli_b = _StubNliClassifier(score_for_all=entailed_score)
    validator_b = ClaimSupportValidator(nli=nli_b)
    result_b = validator_b.validate({
        "blocks": [block_b],
        "source_chunks": {"dart:rdf_intro#blk_0": cited_chunk_b},
    })
    assert result_b.passed is True
    assert result_b.action is None, (
        f"Expected action=None on entailment fixture; got "
        f"action={result_b.action!r}, codes="
        f"{[i.code for i in result_b.issues]!r}"
    )
    # No critical / regenerate-action issues; warning-only NLI deps
    # missing should NOT surface here because we injected a real stub.
    failure_codes = [
        i.code for i in result_b.issues
        if i.code == _CODE_CONTRADICTED_CLAIM
    ]
    assert not failure_codes, (
        f"Entailment fixture unexpectedly fired issue codes "
        f"{failure_codes!r}; got all codes="
        f"{[i.code for i in result_b.issues]!r}"
    )
    assert nli_b.batch_calls, (
        "Validator did NOT invoke NLI on entailment fixture; same "
        "short-circuit guard as fixture A."
    )

    # ---- Fixture C — silent-degrade detection ----
    # Mock NLI returns entailment=0.99 for everything. A vacuous pass and a
    # short-circuit produce the same passed/action signature, so the only
    # distinguishing assertion is that NLI was actually invoked.
    always_entailed = NliScore(
        entailment=0.99, neutral=0.005, contradiction=0.005
    )
    nli_c = _StubNliClassifier(score_for_all=always_entailed)
    block_c = _build_claim_support_block(
        block_id="page_01#concept_rdf_2",
        claims=[contradicted_claim],  # Same prose as fixture A.
    )
    validator_c = ClaimSupportValidator(nli=nli_c)
    result_c = validator_c.validate({
        "blocks": [block_c],
        # Same chunk as fixture A — but the mock NLI ignores semantics
        # and returns entailment=0.99 unconditionally.
        "source_chunks": {"dart:rdf_intro#blk_0": cited_chunk_a},
    })
    # Vacuous pass is the expected outcome of a silent-degrade NLI
    # (entailment >= 0.7 → bucketed as entailed).
    assert result_c.passed is True
    assert result_c.action is None, (
        "With a silent-degrade NLI returning entailment=0.99, the "
        "validator SHOULD vacuously pass — but only because NLI was "
        "actually called. If a future refactor adds an early return "
        "around NLI dispatch (e.g. for empty key_claims), we'd see "
        f"the same passed=True / action=None signature, masking the "
        f"regression. Got action={result_c.action!r}, codes="
        f"{[i.code for i in result_c.issues]!r}"
    )
    # Critical anti-short-circuit guard: the mock MUST have been called
    # at least once. If it wasn't, the validator returned a
    # vacuous-pass without ever consulting NLI, which is the silent-
    # degrade class the plan calls out.
    assert nli_c.batch_calls, (
        "Wave 2 W2.F silent-degrade guard tripped: "
        "ClaimSupportValidator returned passed=True / action=None "
        "WITHOUT ever invoking the injected NLI mock. This is "
        "exactly the regression the plan §Test 6 silent-degrade "
        "scenario calls out — a future refactor that early-returns "
        "around NLI dispatch would make every claim-support gate "
        "vacuously pass without scoring a single claim. Mock call "
        f"counts: batch_calls={len(nli_c.batch_calls)}, "
        f"pair_calls={len(nli_c.pair_calls)}."
    )
    # The single claim was fanned out against the single chunk → at
    # least one (premise, hypothesis) pair must appear in the batch
    # calls. Pin the count so a future fan-out regression that drops
    # the claim is caught.
    flat_pairs = [
        pair for batch in nli_c.batch_calls for pair in batch
    ]
    assert len(flat_pairs) >= 1, (
        "Validator invoked NLI but the (premise, hypothesis) batch "
        f"is empty: batch_calls={nli_c.batch_calls!r}. Fan-out "
        "regression?"
    )


# --------------------------------------------------------------------------- #
# Test 7 — collected-count guard + DeprecationWarning-clean
# --------------------------------------------------------------------------- #


def test_7_collected_count_guard_and_check_schema_clean() -> None:
    """Test-count ratchet across four trees + JSON-LD schema cleanliness.

    AST-walks four test trees and asserts each is at or above its
    ``_MIN_*_TESTS`` floor. The ratchet only moves up: legitimately adding
    tests means bumping the constant; a refactor that removes tests fires
    this gate.

    Also asserts ``Draft202012Validator.check_schema(...)`` runs clean (no
    ``DeprecationWarning``) on ``courseforge_jsonld_v1.schema.json`` — a
    sentinel against a future edit introducing a deprecated field shape.
    """
    jsonschema = pytest.importorskip("jsonschema")

    trees = [
        ("validators",
         PROJECT_ROOT / "lib" / "validators" / "tests",
         _MIN_VALIDATORS_TESTS),
        ("aggregators",
         PROJECT_ROOT / "lib" / "aggregators" / "tests",
         _MIN_AGGREGATORS_TESTS),
        ("trainforge",
         PROJECT_ROOT / "Trainforge" / "tests",
         _MIN_TRAINFORGE_TESTS),
        ("schemas",
         PROJECT_ROOT / "schemas" / "tests",
         _MIN_SCHEMAS_TESTS),
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

    # Draft202012 cleanliness sentinel on the JSON-LD schema — catches a
    # future edit that silently introduces a deprecated field shape.
    schema_path = (
        PROJECT_ROOT
        / "schemas"
        / "knowledge"
        / "courseforge_jsonld_v1.schema.json"
    )
    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        jsonschema.Draft202012Validator.check_schema(schema_doc)
        deprecation_warnings = [
            w for w in captured
            if issubclass(w.category, DeprecationWarning)
        ]
    assert not deprecation_warnings, (
        "Draft202012Validator.check_schema(courseforge_jsonld_v1) "
        "emitted DeprecationWarning(s): "
        + "; ".join(str(w.message) for w in deprecation_warnings)
    )
