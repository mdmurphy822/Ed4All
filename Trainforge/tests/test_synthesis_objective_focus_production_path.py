"""Objective focus must reach the PRODUCTION synthesis entry point.

The regression this pins: ``_focus_chunk_on_objective`` gated the canonical
objective focus on ``TRAINFORGE_STAGED_SYNTHESIS_V4`` alone, while
``_pair_eligibility_for_mode`` ran the staged ``pair_eligibility`` whenever
EITHER staged contract was selected.  A ``micro-v1`` run therefore handed
``pair_eligibility`` an UNFOCUSED chunk, so every chunk carrying
``learning_outcome_refs`` came back ``missing_canonical_objective_focus`` and
the whole corpus emitted ZERO pairs while the process exited 0.

Every prior eligibility measurement was taken from in-process harnesses that
called ``focus_chunk_on_canonical_objective`` directly, so nothing caught it.
These tests therefore drive ``run_synthesis_from_libv2`` — the entry point the
CLI and the ``training_synthesis`` workflow phase both use — against an
archive built inline, and assert on the emitted disposition sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from Trainforge.synthesis.synthesize_training import (
    _focus_chunk_on_objective,
    _pair_eligibility_for_mode,
    run_synthesis_from_libv2,
    staged_objective_contract_enabled,
)

_STAGED_ENV = (
    "TRAINFORGE_STAGED_SYNTHESIS_V4",
    "TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1",
)


@pytest.fixture(autouse=True)
def _isolated_staged_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither staged contract leaks in from the ambient environment."""
    for name in _STAGED_ENV:
        monkeypatch.delenv(name, raising=False)


def _objectives_doc() -> Dict[str, Any]:
    """A LibV2-shape objectives artifact (``terminal_outcomes``)."""
    return {
        "course_code": "TEST_101",
        "terminal_outcomes": [
            {
                "id": "TO-01",
                "statement": (
                    "Analyze how a solvent's polarity governs the "
                    "solubility of an ionic compound."
                ),
                "bloom_level": "analyze",
                "component_objectives": [
                    {
                        "id": "CO-01",
                        "statement": (
                            "Explain why a polar solvent dissolves an ionic "
                            "compound more readily than a nonpolar solvent "
                            "does."
                        ),
                        "bloom_level": "understand",
                    },
                    {
                        "id": "CO-02",
                        "statement": (
                            "Describe how a solvent's dielectric constant "
                            "weakens the attraction between dissolved ions."
                        ),
                        "bloom_level": "understand",
                    },
                ],
            },
        ],
    }


def _chunks() -> List[Dict[str, Any]]:
    """Chunks whose prose genuinely evidences their declared objectives.

    The refs are lowercase and the artifact ids are uppercase, matching the
    real archives; ``focus_chunk_on_canonical_objective`` normalises both.
    """
    polarity_text = (
        "A polar solvent dissolves an ionic compound more readily than a "
        "nonpolar solvent does. Water molecules carry a partial negative "
        "charge on oxygen and a partial positive charge on each hydrogen, so "
        "they orient themselves around a dissolved ion and stabilise it. A "
        "nonpolar solvent such as hexane offers no comparable charge "
        "separation, so it cannot stabilise the separated ions and the ionic "
        "compound stays undissolved. This is why sodium chloride dissolves "
        "in water but not in hexane."
    )
    dielectric_text = (
        "The dielectric constant of a solvent measures how strongly that "
        "solvent weakens the attraction between two dissolved ions. Water has "
        "a high dielectric constant, so the attraction between a dissolved "
        "sodium ion and a dissolved chloride ion falls to a small fraction of "
        "its value in a vacuum. A solvent with a low dielectric constant "
        "weakens that attraction far less, so the ions stay paired and the "
        "compound remains largely undissolved."
    )
    return [
        {
            "id": "test_chunk_00001",
            "text": polarity_text,
            "chunk_type": "explanation",
            "learning_outcome_refs": ["to-01", "co-01"],
            "bloom_level": "understand",
            "concept_tags": ["polarity", "solubility"],
            "word_count": len(polarity_text.split()),
        },
        {
            "id": "test_chunk_00002",
            "text": dielectric_text,
            "chunk_type": "explanation",
            "learning_outcome_refs": ["to-01", "co-02"],
            "bloom_level": "understand",
            "concept_tags": ["dielectric constant", "ion pairing"],
            "word_count": len(dielectric_text.split()),
        },
    ]


def _build_archive(
    root: Path,
    slug: str = "objective-focus-fixture",
    *,
    with_objectives: bool = True,
) -> Path:
    """Materialise a minimal LibV2 course archive and return the LibV2 root."""
    courses = root / "courses"
    course_dir = courses / slug
    chunks_dir = course_dir / "imscc_chunks"
    chunks_dir.mkdir(parents=True)
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in _chunks():
            handle.write(json.dumps(chunk, sort_keys=True) + "\n")
    if with_objectives:
        (course_dir / "objectives.json").write_text(
            json.dumps(_objectives_doc(), indent=2), encoding="utf-8",
        )
    return courses


def _dispositions(output_dir: Path) -> List[Dict[str, Any]]:
    path = output_dir / "synthesis_dispositions.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ineligible_reasons(output_dir: Path, *, kind: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in _dispositions(output_dir):
        if record.get("kind") != kind:
            continue
        if record.get("disposition") != "ineligible":
            continue
        reason = str(record.get("reason") or record.get("rejection_reason"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _eligible_unit_count(output_dir: Path, *, kind: str) -> int:
    """Count units that PASSED eligibility and reached the provider.

    ``synthesis_dispositions.jsonl`` journals only the TERMINAL negative
    outcomes (``ineligible`` before dispatch, ``rejected`` after it), so an
    accepted unit leaves no disposition row and must be counted from the
    emitted pairs file instead.  Eligible == emitted + rejected.

    Counting this way keeps the assertion about OBJECTIVE RESOLUTION and
    independent of whether the mock provider's templated text survives the
    downstream claim-support / promotion quality gates — a rejected pair still
    proves eligibility admitted its chunk.
    """
    emitted_path = output_dir / f"{kind}_pairs.jsonl"
    emitted = 0
    if emitted_path.exists():
        emitted = sum(
            1 for line in emitted_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rejected = sum(
        1 for record in _dispositions(output_dir)
        if record.get("kind") == kind
        and record.get("disposition") == "rejected"
    )
    return emitted + rejected


@pytest.mark.parametrize("staged_env", _STAGED_ENV)
def test_staged_contract_focuses_objectives_through_libv2_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_env: str,
) -> None:
    """Both staged contracts must resolve objectives at the real entry point.

    ``missing_canonical_objective_focus`` on a chunk whose refs DO resolve is
    the whole-corpus-zero signature; a passing run must never emit it.
    """
    monkeypatch.setenv(staged_env, "true")
    libv2_root = _build_archive(tmp_path)
    output_dir = tmp_path / "out" / staged_env
    output_dir.mkdir(parents=True)

    run_synthesis_from_libv2(
        "objective-focus-fixture",
        libv2_root=libv2_root,
        output_dir=output_dir,
        provider="mock",
        synthesis_pairs_checkpoint_path=None,
    )

    reasons = _ineligible_reasons(output_dir, kind="instruction")
    assert "missing_canonical_objective_focus" not in reasons, (
        "the focus seam did not run before the eligibility seam under "
        f"{staged_env}; residual reasons: {reasons}"
    )
    assert "authoritative_objectives_unavailable" not in reasons, (
        f"objectives never reached the resolver under {staged_env}; "
        f"residual reasons: {reasons}"
    )
    assert _eligible_unit_count(output_dir, kind="instruction") > 0, (
        f"zero instruction-eligible chunks under {staged_env}; "
        f"residual reasons: {reasons}"
    )


@pytest.mark.parametrize("staged_env", _STAGED_ENV)
def test_focus_and_eligibility_seams_agree_on_every_staged_contract(
    monkeypatch: pytest.MonkeyPatch, staged_env: str,
) -> None:
    """The two seams share one predicate, so they cannot drift apart again.

    A unit-level companion to the entry-point test above: whenever
    ``_pair_eligibility_for_mode`` routes to the staged ``pair_eligibility``,
    ``_focus_chunk_on_objective`` must already have attached the canonical
    focus.
    """
    monkeypatch.setenv(staged_env, "true")
    assert staged_objective_contract_enabled() is True

    objectives = {
        "co-01": {
            "statement": (
                "Explain why a polar solvent dissolves an ionic compound "
                "more readily than a nonpolar solvent does."
            ),
            "bloom_level": "understand",
        },
    }
    chunk = _chunks()[0]
    focused = _focus_chunk_on_objective(chunk, seed=7, objectives=objectives)

    assert focused is not chunk, (
        f"{staged_env} left the chunk unfocused; pair_eligibility would "
        "report missing_canonical_objective_focus for the whole corpus"
    )
    assert focused.get("synthesis_focus_objective"), (
        "the staged contract requires a resolved canonical objective focus"
    )
    eligibility = _pair_eligibility_for_mode(focused, kind="instruction")
    assert eligibility.reason != "missing_canonical_objective_focus"
    assert eligibility.eligible, eligibility.reason


def test_legacy_path_stays_unfocused_and_unconditionally_admitted() -> None:
    """No staged contract selected -> historical chunk view, byte-identical."""
    assert staged_objective_contract_enabled() is False
    chunk = _chunks()[0]
    focused = _focus_chunk_on_objective(chunk, seed=7, objectives={})
    assert focused is chunk
    assert _pair_eligibility_for_mode(focused, kind="instruction").eligible


@pytest.mark.parametrize("staged_env", _STAGED_ENV)
def test_missing_objectives_artifact_fails_loudly_under_staged_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_env: str,
) -> None:
    """A staged run without the artifact must raise, not emit zero pairs.

    Defaulting an objective would manufacture training data with no real
    objective binding, so the only honest outcome is a loud failure naming
    the producing phase.
    """
    monkeypatch.setenv(staged_env, "true")
    libv2_root = _build_archive(tmp_path, with_objectives=False)
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="course_planning"):
        run_synthesis_from_libv2(
            "objective-focus-fixture",
            libv2_root=libv2_root,
            output_dir=output_dir,
            provider="mock",
            synthesis_pairs_checkpoint_path=None,
        )
