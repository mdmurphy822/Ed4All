"""WI-16 — `RejectedClaimEntailmentValidator.validate()` tests.

Pins the single gate-runner surface: walks `preference_pairs.jsonl`,
filters to `source="misconception"` pairs, and NLI-scores each
`rejected` side against its cited chunk's text (the "source window").
Never blocks — `passed` is asserted `True` in every scenario,
including the flagged-finding and infra-degrade arms.

Mirror of `lib/validators/tests/test_claim_support.py`'s stub-NLI +
graceful-degrade test shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.validators.pair.rejected_claim_entailment import (  # noqa: E402
    RejectedClaimEntailmentValidator,
    _CODE_INVALID_CHUNKS,
    _CODE_NLI_DEPS_MISSING,
    _CODE_NO_CHUNKS,
    _CODE_READ_ERROR,
    _CODE_REJECTED_ENTAILED_BY_SOURCE,
)


class _StubNli:
    """Drop-in `NliClassifier` stand-in returning a fixed score for
    every `score_pair` call. Records calls so fan-out can be asserted."""

    def __init__(self, entailment: float, contradiction: float = 0.05) -> None:
        neutral = max(0.0, 1.0 - entailment - contradiction)
        self._score = NliScore(
            entailment=entailment, neutral=neutral, contradiction=contradiction,
        )
        self.calls: list[tuple[str, str]] = []

    def score_pair(self, *, premise: str, hypothesis: str) -> NliScore:
        self.calls.append((premise, hypothesis))
        return self._score


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _misconception_pair(
    chunk_id: str = "c1",
    rejected: str = "This claim is drawn from the chunk's misconceptions.",
    **extra,
) -> dict:
    base = {
        "prompt": "x" * 50,
        "chosen": "y" * 60,
        "rejected": rejected,
        "chunk_id": chunk_id,
        "source": "misconception",
        "rejected_source": "misconception",
        "misconception_id": "mc_001",
        "lo_refs": ["TO-01"],
        "seed": 17,
    }
    base.update(extra)
    return base


# --------------------------------------------------------------------- #
# Smoke + no-op arms
# --------------------------------------------------------------------- #


def test_validator_instantiates() -> None:
    validator = RejectedClaimEntailmentValidator()
    assert validator is not None
    assert callable(getattr(validator, "validate", None))


def test_empty_inputs_no_op_pass() -> None:
    """No preference_pairs_path / course_dir resolvable → no-op pass,
    never a Python exception, never a block."""
    result = RejectedClaimEntailmentValidator().validate({})
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []


def test_missing_preference_pairs_file_no_op_pass(tmp_path: Path) -> None:
    result = RejectedClaimEntailmentValidator().validate({
        "preference_pairs_path": str(tmp_path / "does_not_exist.jsonl"),
    })
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []


def test_no_misconception_sourced_pairs_no_op_pass(tmp_path: Path) -> None:
    """Every pair is `rule_synthesized` → nothing to score; not even
    the NLI classifier gets touched."""
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(pref, [
        _misconception_pair(source="rule_synthesized", rejected_source="rule_synthesized"),
    ])
    stub = _StubNli(entailment=0.99)  # would flag if it were ever called

    result = RejectedClaimEntailmentValidator(nli=stub).validate({
        "preference_pairs_path": str(pref),
        "chunks_path": str(tmp_path / "chunks.jsonl"),
    })
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []
    assert stub.calls == []


# --------------------------------------------------------------------- #
# Core behavior: entailed-rejected flag
# --------------------------------------------------------------------- #


def test_low_entailment_rejected_passes_clean(tmp_path: Path) -> None:
    """rejected side NOT entailed by source → no finding, score 1.0."""
    chunks = tmp_path / "chunks.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [
        {"chunk_id": "c1", "text": "Momentum is conserved in an isolated system."},
    ])
    _write_jsonl(pref, [_misconception_pair(chunk_id="c1")])

    stub = _StubNli(entailment=0.10, contradiction=0.80)
    result = RejectedClaimEntailmentValidator(nli=stub).validate({
        "preference_pairs_path": str(pref),
        "chunks_path": str(chunks),
    })

    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []
    assert len(stub.calls) == 1


def test_high_entailment_rejected_flags_warning_never_blocks(tmp_path: Path) -> None:
    """rejected side IS entailed by its own cited source → flagged as
    a warning, but the gate still reports passed=True (never blocks)."""
    chunks = tmp_path / "chunks.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [
        {"chunk_id": "c1", "text": "Momentum is conserved in an isolated system."},
    ])
    _write_jsonl(pref, [
        _misconception_pair(
            chunk_id="c1",
            rejected="Momentum is conserved in an isolated system.",
        ),
    ])

    stub = _StubNli(entailment=0.93, contradiction=0.02)
    result = RejectedClaimEntailmentValidator(nli=stub).validate({
        "preference_pairs_path": str(pref),
        "chunks_path": str(chunks),
    })

    assert result.passed is True  # warning severity — never blocks
    assert result.action is None
    assert result.score < 1.0

    flagged = [i for i in result.issues if i.code == _CODE_REJECTED_ENTAILED_BY_SOURCE]
    assert len(flagged) == 1
    assert flagged[0].severity == "warning"
    assert "c1" in flagged[0].message
    assert "0.93" in flagged[0].message
    assert flagged[0].location == str(pref)


def test_entailment_floor_is_configurable(tmp_path: Path) -> None:
    """A pair scoring 0.60 entailment does NOT flag against the
    default 0.70 floor, but DOES flag when the floor is lowered."""
    chunks = tmp_path / "chunks.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [{"chunk_id": "c1", "text": "Some source text."}])
    _write_jsonl(pref, [_misconception_pair(chunk_id="c1")])

    stub = _StubNli(entailment=0.60, contradiction=0.10)

    default_result = RejectedClaimEntailmentValidator(nli=stub).validate({
        "preference_pairs_path": str(pref), "chunks_path": str(chunks),
    })
    assert default_result.issues == []

    lowered_result = RejectedClaimEntailmentValidator(
        nli=stub, entailment_floor=0.5,
    ).validate({
        "preference_pairs_path": str(pref), "chunks_path": str(chunks),
    })
    assert len(lowered_result.issues) == 1
    assert lowered_result.passed is True


def test_multiple_pairs_only_misconception_sourced_scored(tmp_path: Path) -> None:
    """A rule_synthesized pair sitting next to a misconception pair in
    the same file is skipped even though the stub would flag it."""
    chunks = tmp_path / "chunks.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [
        {"chunk_id": "c1", "text": "Source one."},
        {"chunk_id": "c2", "text": "Source two."},
    ])
    _write_jsonl(pref, [
        _misconception_pair(chunk_id="c1", rejected="Source one."),
        _misconception_pair(
            chunk_id="c2", rejected="Something unrelated entirely.",
            source="rule_synthesized", rejected_source="rule_synthesized",
        ),
    ])

    stub = _StubNli(entailment=0.95, contradiction=0.01)
    result = RejectedClaimEntailmentValidator(nli=stub).validate({
        "preference_pairs_path": str(pref), "chunks_path": str(chunks),
    })

    assert result.passed is True
    assert len(stub.calls) == 1  # only the misconception-sourced pair scored
    assert stub.calls[0] == ("Source one.", "Source one.")
    assert result.metadata["misconception_pairs"] == 1
    assert result.metadata["scored"] == 1
    assert result.metadata["flagged_entailed_rejected"] == 1


# --------------------------------------------------------------------- #
# Infra / resolution degrade arms
# --------------------------------------------------------------------- #


def test_unresolvable_source_window_skips_pair_silently(tmp_path: Path) -> None:
    """chunk_id absent from chunks.jsonl → pair simply isn't scored
    (still passed=True, no finding, no crash)."""
    chunks = tmp_path / "chunks.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [{"chunk_id": "other", "text": "Unrelated."}])
    _write_jsonl(pref, [_misconception_pair(chunk_id="c1")])

    stub = _StubNli(entailment=0.99)
    result = RejectedClaimEntailmentValidator(nli=stub).validate({
        "preference_pairs_path": str(pref), "chunks_path": str(chunks),
    })

    assert result.passed is True
    assert result.issues == []
    assert stub.calls == []
    assert result.metadata["scored"] == 0


def test_no_chunks_file_emits_warning(tmp_path: Path) -> None:
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(pref, [_misconception_pair()])

    result = RejectedClaimEntailmentValidator(nli=_StubNli(0.99)).validate({
        "preference_pairs_path": str(pref),
        "chunks_path": str(tmp_path / "missing_chunks.jsonl"),
    })

    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert _CODE_NO_CHUNKS in codes
    assert all(i.severity == "warning" for i in result.issues)


def test_unreadable_preference_pairs_file_emits_warning(tmp_path: Path) -> None:
    """A non-UTF-8 preference_pairs.jsonl degrades to a warning issue
    (the pairs-side twin of the invalid-chunks arm below)."""
    pref = tmp_path / "preference_pairs.jsonl"
    pref.write_bytes(b"\xff\xfe not valid utf-8 jsonl at all \x00\x01")

    result = RejectedClaimEntailmentValidator(nli=_StubNli(0.99)).validate({
        "preference_pairs_path": str(pref),
        "chunks_path": str(tmp_path / "chunks.jsonl"),
    })

    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert _CODE_READ_ERROR in codes
    assert all(i.severity == "warning" for i in result.issues)


def test_invalid_chunks_file_emits_warning(tmp_path: Path) -> None:
    pref = tmp_path / "preference_pairs.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(pref, [_misconception_pair()])
    chunks.write_bytes(b"\xff\xfe not valid utf-8 jsonl at all \x00\x01")

    result = RejectedClaimEntailmentValidator(nli=_StubNli(0.99)).validate({
        "preference_pairs_path": str(pref),
        "chunks_path": str(chunks),
    })

    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert _CODE_INVALID_CHUNKS in codes


# --------------------------------------------------------------------- #
# Graceful-degrade contract (NLI deps missing)
# --------------------------------------------------------------------- #


def test_nli_deps_missing_emits_warning_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.classifiers import nli_classifier as nli_mod

    monkeypatch.setattr(
        nli_mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)

    chunks = tmp_path / "chunks.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [{"chunk_id": "c1", "text": "Some source."}])
    _write_jsonl(pref, [_misconception_pair(chunk_id="c1")])

    validator = RejectedClaimEntailmentValidator()  # nli=None → real loader
    result = validator.validate({
        "preference_pairs_path": str(pref), "chunks_path": str(chunks),
    })

    deps_issues = [i for i in result.issues if i.code == _CODE_NLI_DEPS_MISSING]
    assert len(deps_issues) == 1
    assert deps_issues[0].severity == "warning"
    assert result.passed is True
    assert result.action is None


def test_strict_mode_with_missing_deps_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.classifiers import nli_classifier as nli_mod

    monkeypatch.setattr(
        nli_mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    chunks = tmp_path / "chunks.jsonl"
    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(chunks, [{"chunk_id": "c1", "text": "Some source."}])
    _write_jsonl(pref, [_misconception_pair(chunk_id="c1")])

    validator = RejectedClaimEntailmentValidator()  # nli=None
    with pytest.raises(RuntimeError) as excinfo:
        validator.validate({
            "preference_pairs_path": str(pref), "chunks_path": str(chunks),
        })
    assert "TRAINFORGE_REQUIRE_EMBEDDINGS" in str(excinfo.value)


def test_no_deps_missing_check_when_no_misconception_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deps-missing / strict-mode raise never fires when the corpus has
    no misconception-sourced pairs — the NLI classifier is never
    consulted (mirrors the `stub.calls == []` no-op assertion above,
    exercised here through the real (monkeypatched-absent) loader)."""
    from lib.classifiers import nli_classifier as nli_mod

    monkeypatch.setattr(
        nli_mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    pref = tmp_path / "preference_pairs.jsonl"
    _write_jsonl(pref, [
        _misconception_pair(source="rule_synthesized", rejected_source="rule_synthesized"),
    ])

    validator = RejectedClaimEntailmentValidator()
    result = validator.validate({"preference_pairs_path": str(pref)})
    assert result.passed is True
    assert result.score == 1.0
