"""Wave 135c — CurieAnchoringValidator regression tests.

Pins the binary per-pair CURIE anchoring sentinel that replaces
Wave 130b's mean-retention metric. Healthy Wave 135b force-injection
keeps the per-pair anchoring rate at ~1.00 by construction; a broken
injector drops the rate to whatever the natural paraphrase rate is.

Cases:

* **Positive (anchored):** every paraphrase pair contains ≥1 source
  CURIE → rate=1.0, gate passes.
* **Negative (injector regression):** 4/5 pairs anchored, 1 dropped
  → rate=0.8 < 0.95, gate fails closed.
* **Skip-deterministic:** 5 paraphrase pairs (anchored) + 5
  deterministic pairs (unanchored, but skipped) → rate=1.0, passes.
* **Threshold override:** override config to 0.5 → corpus at 0.6 passes
  (would fail at the default 0.95).

U4 (post-376b64f) — structured ``chunk["curies"]`` consumption:

* **Force-injected-then-stripped regression:** a chunk whose CURIEs
  were stripped from ``text`` by 376b64f but harvested into
  ``chunk["curies"]`` is now ELIGIBLE; unanchored pairs fail the gate
  instead of vanishing into ``skipped_no_curies``.
* **Forced metrics surfaced:** ``forced_source_chunks`` /
  ``forced_curie_block_rate`` appear on ``result.metadata``.
* **Genuinely CURIE-free chunk:** still counted in
  ``skipped_no_curies`` — semantics preserved.
* **Legacy chunk:** no ``curies`` field → text-regex fallback.
* **Advisory knob:** ``max_forced_curie_block_rate`` defaults to 1.0
  (never blocks); a lowered knob emits a warning, not a failure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pytest

from lib.validators.curie_anchoring import CurieAnchoringValidator


# --------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------- #


def _write_corpus(course_dir: Path, chunks: Iterable[dict]) -> Path:
    p = course_dir / "corpus" / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n",
        encoding="utf-8",
    )
    return p


def _write_pairs(course_dir: Path, rows: Iterable[dict]) -> Path:
    p = course_dir / "training_specs" / "instruction_pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return p


def _chunk_text_with_curies(curies: List[str]) -> str:
    """Compose chunk text with the listed CURIEs woven into prose."""
    sentences = [
        f"The constraint {c} appears here as a typed predicate."
        for c in curies
    ]
    return (
        "Background prose without any CURIE tokens. "
        + " ".join(sentences)
        + " The vocabulary is fixed."
    )


# --------------------------------------------------------------- #
# Positive: anchored corpus → rate=1.0, passes
# --------------------------------------------------------------- #


def test_passes_when_every_pair_anchors_at_least_one_curie(
    tmp_path: Path,
) -> None:
    """5 chunks × 2 CURIEs each, every paraphrase pair contains ≥1
    source CURIE → pair_anchoring_rate=1.0, gate passes."""
    course = tmp_path / "course"
    chunks: List[dict] = []
    pairs: List[dict] = []
    for i in range(5):
        curies = [f"sh:Shape{i}A", f"rdfs:label{i}B"]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        # Two paraphrase pairs per chunk — each anchors at least one
        # source CURIE in its body.
        for j in range(2):
            anchor = curies[j % 2]
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": f"paraphrase.def_{j}",
                "prompt": (
                    f"Define for a learner: introduce the relevant "
                    f"constraint (variant {j})."
                ),
                "completion": (
                    f"In SHACL the {anchor} predicate types the node, "
                    f"and the validator interprets it as a typed shape."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []


# --------------------------------------------------------------- #
# Negative: injector regression → rate=0.8 < 0.95, fails closed
# --------------------------------------------------------------- #


def test_fails_closed_when_anchoring_rate_below_threshold(
    tmp_path: Path,
) -> None:
    """5 chunks × 2 CURIEs, 4 of 5 paraphrase pairs anchor, 1 has zero
    source CURIE in body → rate=0.8 < 0.95, gate fails."""
    course = tmp_path / "course"
    chunks: List[dict] = []
    pairs: List[dict] = []
    for i in range(5):
        curies = [f"sh:Shape{i}A", f"rdfs:label{i}B"]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        if i == 4:
            # Injector regression: pair body contains zero source
            # CURIEs.
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Briefly explain the constraint shape.",
                "completion": (
                    "It declares structural validation behaviour for "
                    "typed nodes without naming the predicate."
                ),
            })
        else:
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Define the constraint and its predicate.",
                "completion": (
                    f"The {curies[0]} shape types the node and the "
                    f"validator follows the declared constraint."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is False
    assert result.score == pytest.approx(0.8)
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert "PAIR_ANCHORING_BELOW_THRESHOLD" in critical_codes
    report = next(
        i for i in result.issues if i.code == "PAIR_ANCHORING_REPORT"
    )
    payload = json.loads(report.message)
    assert payload["total_eligible_pairs"] == 5
    assert payload["anchored_count"] == 4
    assert payload["unanchored_count"] == 1
    assert payload["pair_anchoring_rate"] == pytest.approx(0.8)


# --------------------------------------------------------------- #
# Skip-deterministic: oracle-grounded pairs are excluded from audit
# --------------------------------------------------------------- #


def test_skips_deterministic_template_ids(tmp_path: Path) -> None:
    """5 paraphrase pairs (all anchored) + 5 deterministic-prefixed
    pairs (unanchored). Only paraphrase pairs count → rate=1.0, gate
    passes despite the deterministic pairs' missing CURIEs."""
    course = tmp_path / "course"
    curies = ["sh:NodeShape", "rdfs:label"]
    chunks = [{"id": "c1", "text": _chunk_text_with_curies(curies)}]
    pairs: List[dict] = []
    # 5 clean paraphrase pairs (anchored).
    for j in range(5):
        pairs.append({
            "chunk_id": "c1",
            "template_id": "paraphrase.def_0",
            "prompt": (
                f"Define the constraint shape (variant {j})."
            ),
            "completion": (
                f"In SHACL, {curies[0]} types the node and pairs with "
                f"{curies[1]} as the human-readable label."
            ),
        })
    # 5 deterministic-prefixed pairs that DROP all CURIEs (would tank
    # the anchoring rate if counted; must be skipped).
    deterministic_prefixes = [
        "kg_metadata.entity",
        "violation_detection.shape",
        "abstention.unsupported",
        "schema_translation.translate",
        "kg_metadata.lookup",
    ]
    for prefix in deterministic_prefixes:
        pairs.append({
            "chunk_id": "c1",
            "template_id": prefix,
            "prompt": (
                "Answer the structured question without ontology "
                "tokens (deterministic generator)."
            ),
            "completion": (
                "The expected answer omits ontology-prefixed terms by "
                "construction in this generator family."
            ),
        })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []
    report = next(
        (i for i in result.issues
         if i.code == "PAIR_ANCHORING_REPORT"),
        None,
    )
    if report is not None:
        payload = json.loads(report.message)
        assert payload["skipped_deterministic"] == 5
        assert payload["total_eligible_pairs"] == 5


# --------------------------------------------------------------- #
# Threshold override: relaxed gate passes a corpus that would fail
# at the default 0.95.
# --------------------------------------------------------------- #


def test_threshold_override_via_inputs(tmp_path: Path) -> None:
    """Override min_pair_anchoring_rate to 0.5; a corpus at 0.6
    passes the relaxed gate but fails the default 0.95."""
    course = tmp_path / "course"
    chunks: List[dict] = []
    pairs: List[dict] = []
    for i in range(5):
        curies = [f"sh:Shape{i}A", f"rdfs:label{i}B"]
        chunks.append({
            "id": f"c{i}",
            "text": _chunk_text_with_curies(curies),
        })
        # 3 of 5 chunks contribute anchored pairs (60%); the other 2
        # produce unanchored pairs.
        if i < 3:
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Describe the constraint and its label.",
                "completion": (
                    f"The {curies[0]} predicate types the node and "
                    f"binds to the {curies[1]} annotation."
                ),
            })
        else:
            pairs.append({
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.def_0",
                "prompt": "Briefly summarize the validation behaviour.",
                "completion": (
                    "It refines validation of typed nodes through a "
                    "declarative predicate without naming it."
                ),
            })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    strict = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert strict.passed is False
    assert strict.score == pytest.approx(0.6)

    relaxed = CurieAnchoringValidator().validate({
        "course_dir": str(course),
        "thresholds": {"min_pair_anchoring_rate": 0.5},
    })
    assert relaxed.passed is True
    assert relaxed.score == pytest.approx(0.6)


# --------------------------------------------------------------- #
# Missing-input fail-closed cases
# --------------------------------------------------------------- #


def test_missing_inputs_fails_critical() -> None:
    """No course_dir / training_specs_dir / instruction_pairs_path on
    inputs → gate fails closed with MISSING_INPUTS."""
    result = CurieAnchoringValidator().validate({})
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "MISSING_INPUTS" in codes


def test_missing_pair_file_fails_critical(tmp_path: Path) -> None:
    """No instruction_pairs.jsonl on disk → gate fails closed."""
    course = tmp_path / "course"
    _write_corpus(course, [{"id": "c1", "text": "no curies here"}])
    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "INSTRUCTION_PAIRS_NOT_FOUND" in codes


def test_missing_chunks_file_fails_critical(tmp_path: Path) -> None:
    """No corpus chunks.jsonl on disk → gate fails closed."""
    course = tmp_path / "course"
    _write_pairs(course, [{
        "chunk_id": "c1",
        "template_id": "paraphrase.def_0",
        "prompt": "p", "completion": "c",
    }])
    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is False
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "CHUNKS_NOT_FOUND" in codes


def test_passes_when_no_auditable_pairs(tmp_path: Path) -> None:
    """Corpus with chunks that have zero CURIEs → no pairs audited
    → gate passes with NO_AUDITABLE_PAIRS info issue."""
    course = tmp_path / "course"
    _write_corpus(course, [
        {"id": "c1", "text": "Plain text without ontology-prefixed terms."},
    ])
    _write_pairs(course, [
        {
            "chunk_id": "c1",
            "template_id": "paraphrase.def_0",
            "prompt": "Summarize the chunk.",
            "completion": "It contains plain prose.",
        },
    ])
    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert result.passed is True
    info_codes = [i.code for i in result.issues if i.severity == "info"]
    assert "NO_AUDITABLE_PAIRS" in info_codes


# --------------------------------------------------------------- #
# U4 — structured chunk["curies"] consumption (post-376b64f)
# --------------------------------------------------------------- #


def test_force_injected_stripped_chunk_now_fails_closed(
    tmp_path: Path,
) -> None:
    """REGRESSION TEST for the U4 bug.

    Commit 376b64f makes the chunker strip force-injected
    ``data-cf-curie`` spans out of ``chunk["text"]``. A sibling unit
    has the chunker harvest those CURIEs into the structured
    ``chunk["curies"]`` / ``chunk["forced_curies"]`` fields BEFORE the
    strip.

    Here a chunk carries ``curies: ["ns:concept"]`` (forced) but its
    ``text`` does NOT contain ``ns:concept`` (stripped), and the
    instruction pairs' bodies also do NOT contain it (a real
    rewrite-tier preservation failure).

    Pre-fix the validator built its source-CURIE map from
    ``chunk["text"]`` only → the chunk had zero source CURIEs → every
    pair hit ``skipped_no_curies`` and was excluded from the
    denominator → the gate passed at rate 1.0 over a non-
    representative sample.

    Post-fix the validator reads ``chunk["curies"]`` as the
    authoritative source-CURIE set → the pairs are now ELIGIBLE,
    anchor nothing → the gate FAILS closed.
    """
    course = tmp_path / "course"
    # text deliberately omits ns:concept — 376b64f stripped the span.
    chunks = [{
        "id": "c1",
        "text": "Background prose with the canonical token removed.",
        "curies": ["ns:concept"],
        "forced_curies": ["ns:concept"],
    }]
    # Pair bodies also omit ns:concept — genuine anchoring failure.
    pairs = [{
        "chunk_id": "c1",
        "template_id": "paraphrase.def_0",
        "prompt": "Explain the constraint.",
        "completion": (
            "It declares validation behaviour without naming the "
            "canonical predicate."
        ),
    }]
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert "PAIR_ANCHORING_BELOW_THRESHOLD" in critical_codes
    report = next(
        i for i in result.issues if i.code == "PAIR_ANCHORING_REPORT"
    )
    payload = json.loads(report.message)
    # Pre-fix: total_eligible_pairs==0 (all skipped_no_curies), pass.
    assert payload["total_eligible_pairs"] == 1
    assert payload["skipped_no_curies"] == 0
    assert payload["anchored_count"] == 0


def test_declared_curies_anchor_and_expose_forced_metrics(
    tmp_path: Path,
) -> None:
    """A chunk with a structured ``curies`` field whose pairs DO
    contain the CURIE → gate passes; forced_source_chunks /
    forced_curie_block_rate surface in result.metadata."""
    course = tmp_path / "course"
    chunks = [{
        "id": "c1",
        # text omits the CURIE; the structured field is authoritative.
        "text": "Prose without the canonical token.",
        "curies": ["ns:concept"],
        "forced_curies": ["ns:concept"],
    }]
    pairs = [{
        "chunk_id": "c1",
        "template_id": "paraphrase.def_0",
        "prompt": "Define the constraint.",
        "completion": (
            "The ns:concept predicate types the node for the "
            "validator."
        ),
    }]
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    assert result.metadata is not None
    assert result.metadata["forced_source_chunks"] == 1
    assert result.metadata["distinct_eligible_chunks"] == 1
    assert result.metadata["forced_curie_block_rate"] == pytest.approx(1.0)
    # The structured report also carries the forced sub-signal.
    report = next(
        i for i in result.issues if i.code == "PAIR_ANCHORING_REPORT"
    )
    payload = json.loads(report.message)
    assert payload["forced_source_chunks"] == 1
    assert payload["forced_curie_block_rate"] == pytest.approx(1.0)


def test_genuinely_curie_free_chunk_still_skipped(
    tmp_path: Path,
) -> None:
    """A chunk with NO ``curies`` field and no CURIE in ``text`` is
    genuinely CURIE-free → its pairs still increment
    skipped_no_curies and are excluded; gate passes. The
    skipped_no_curies semantics are preserved — only force-injected
    chunks change from skipped → eligible."""
    course = tmp_path / "course"
    chunks = [{
        "id": "c1",
        "text": "Plain prose without any ontology-prefixed terms.",
    }]
    pairs = [{
        "chunk_id": "c1",
        "template_id": "paraphrase.def_0",
        "prompt": "Summarize the chunk.",
        "completion": "It contains plain prose with no predicates.",
    }]
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    assert result.passed is True
    info_codes = [i.code for i in result.issues if i.severity == "info"]
    assert "NO_AUDITABLE_PAIRS" in info_codes


def test_legacy_chunk_without_curies_field_uses_text_fallback(
    tmp_path: Path,
) -> None:
    """A legacy chunk with NO ``curies`` field but a real CURIE woven
    into ``text`` → the text-regex fallback still resolves the source
    CURIE; pair anchoring is evaluated as before."""
    course = tmp_path / "course"
    # No `curies` key — predates the structured field.
    chunks = [{
        "id": "c1",
        "text": _chunk_text_with_curies(["sh:NodeShape"]),
    }]
    # Anchored pair (body contains the legacy text CURIE).
    anchored_pair = {
        "chunk_id": "c1",
        "template_id": "paraphrase.def_0",
        "prompt": "Define the shape.",
        "completion": (
            "In SHACL the sh:NodeShape predicate types the node."
        ),
    }
    # Unanchored pair (body drops the CURIE).
    unanchored_pair = {
        "chunk_id": "c1",
        "template_id": "paraphrase.def_1",
        "prompt": "Summarize the validation behaviour.",
        "completion": (
            "It refines validation without naming the predicate."
        ),
    }
    _write_corpus(course, chunks)
    _write_pairs(course, [anchored_pair, unanchored_pair])

    result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )

    # 1/2 anchored → 0.5 < default 0.95 → fails (fallback resolved
    # the source CURIE, so the pairs WERE eligible).
    assert result.passed is False
    assert result.score == pytest.approx(0.5)
    report = next(
        i for i in result.issues if i.code == "PAIR_ANCHORING_REPORT"
    )
    payload = json.loads(report.message)
    assert payload["total_eligible_pairs"] == 2
    assert payload["anchored_count"] == 1
    # No structured forced field on a legacy chunk → zero forced.
    assert payload["forced_source_chunks"] == 0


def test_max_forced_curie_block_rate_is_advisory_by_default(
    tmp_path: Path,
) -> None:
    """A corpus of force-injected chunks whose pairs all anchor →
    gate PASSES under the default max_forced_curie_block_rate (1.0,
    advisory). Lowering the knob below the actual rate surfaces a
    FORCED_CURIE_BLOCKS_PRESENT warning issue without flipping the
    verdict to fail (warning severity, never blocks)."""
    course = tmp_path / "course"
    chunks: List[dict] = []
    pairs: List[dict] = []
    for i in range(4):
        curie = f"ns:concept{i}"
        chunks.append({
            "id": f"c{i}",
            "text": "Prose without the canonical token.",
            "curies": [curie],
            "forced_curies": [curie],
        })
        pairs.append({
            "chunk_id": f"c{i}",
            "template_id": "paraphrase.def_0",
            "prompt": "Define the constraint.",
            "completion": (
                f"The {curie} predicate types the node for the "
                f"validator."
            ),
        })
    _write_corpus(course, chunks)
    _write_pairs(course, pairs)

    # Default knob (1.0): advisory — gate passes, no warning issue.
    default_result = CurieAnchoringValidator().validate(
        {"course_dir": str(course)}
    )
    assert default_result.passed is True
    assert default_result.score == pytest.approx(1.0)
    default_codes = [i.code for i in default_result.issues]
    assert "FORCED_CURIE_BLOCKS_PRESENT" not in default_codes

    # Explicitly-lowered knob: rate 1.0 exceeds 0.5 → warning issue
    # is present, but the verdict still PASSES (warning severity).
    lowered_result = CurieAnchoringValidator().validate({
        "course_dir": str(course),
        "thresholds": {"max_forced_curie_block_rate": 0.5},
    })
    assert lowered_result.passed is True
    warning_codes = [
        i.code for i in lowered_result.issues
        if i.severity == "warning"
    ]
    assert "FORCED_CURIE_BLOCKS_PRESENT" in warning_codes


# --------------------------------------------------------------- #
# Phase 3 Subtask 51 — Block-list dispatch path
# --------------------------------------------------------------- #


def _make_outline_block(
    block_id: str = "page_01#concept_intro_0",
    curies=("ed4all:Foo",),
    key_claims=None,
):
    """Minimal Block fixture for the outline-tier dispatch path.

    Imports inline so the legacy instruction-pair tests don't pay the
    blocks.py sys.path cost. Mirrors the bridge in
    Courseforge/router/router.py.
    """
    import sys
    from pathlib import Path

    scripts_dir = (
        Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
    )
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from blocks import Block  # noqa: E402

    return Block(
        block_id=block_id,
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={
            "curies": list(curies),
            "key_claims": list(
                key_claims
                if key_claims is not None
                else ["The ed4all:Foo predicate marks anchoring."]
            ),
            "content_type": "definition",
        },
    )


def test_validate_blocks_returns_regenerate_action_on_curie_miss():
    """Outline-tier Block declaring CURIEs that don't appear in
    key_claims fails closed with action='regenerate' (Phase 4 §1
    mapping — semantic miss the rewrite tier could fix on a re-roll).
    """
    blocks = [
        _make_outline_block(
            block_id="page_01#concept_a_0",
            curies=("ed4all:Foo",),
            key_claims=["The ed4all:Foo predicate marks anchoring."],
        ),
        # Negative case: declares ed4all:Bar but key_claims has no
        # CURIE that resolves.
        _make_outline_block(
            block_id="page_01#concept_b_1",
            curies=("ed4all:Bar",),
            key_claims=["This claim has no anchoring CURIE."],
        ),
    ]
    result = CurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "OUTLINE_BLOCK_CURIE_NOT_ANCHORED" in codes
    # Score reflects 1/2 anchored.
    assert result.score == 0.5


def test_validate_blocks_passes_when_all_curies_anchored():
    """Every Block has at least one CURIE present in key_claims →
    rate=1.0, action remains None (pass)."""
    blocks = [
        _make_outline_block(
            curies=("ed4all:Foo",),
            key_claims=["The ed4all:Foo predicate marks anchoring."],
        ),
    ]
    result = CurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is True
    assert result.action is None
    assert result.score == 1.0


def test_validate_blocks_skips_rewrite_tier_html_string_blocks():
    """A rewrite-tier Block (block.content as string) is silently
    skipped per Phase 3.5 scope split — outline-only audit. With no
    auditable blocks the gate passes with NO_AUDITABLE_BLOCKS info.
    """
    import sys
    from pathlib import Path

    scripts_dir = (
        Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
    )
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from blocks import Block  # noqa: E402

    blocks = [
        Block(
            block_id="page_01#concept_html_0",
            block_type="concept",
            page_id="page_01",
            sequence=0,
            content="<p>Rewrite-tier HTML.</p>",
        ),
    ]
    result = CurieAnchoringValidator().validate({"blocks": blocks})
    assert result.passed is True
    info_codes = [i.code for i in result.issues if i.severity == "info"]
    assert "NO_AUDITABLE_BLOCKS" in info_codes
