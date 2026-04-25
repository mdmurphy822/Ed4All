"""Tests for ``ed4all libv2 generate-quiz`` (Wave 77 Worker γ)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.commands.libv2_validate_packet import libv2_group


# ---------------------------------------------------------------------- #
# Fixture builders
# ---------------------------------------------------------------------- #


_BLOOMS = ["remember", "understand", "apply", "analyze", "create", "evaluate"]


def _make_assessment_chunk(
    *,
    idx: int,
    bloom: str,
    difficulty: str = "intermediate",
    los: list[str] | None = None,
    concept_tags: list[str] | None = None,
    misconceptions: list[dict] | None = None,
    text: str | None = None,
) -> dict:
    """Build a synthetic assessment_item chunk that round-trips through
    the engine. Uses a "Show answer X." block so the heuristic
    splitter has something to chew on."""
    body = text or (
        f"Question {idx}: which option is correct? "
        f"Option Alpha-{idx} Option Bravo-{idx} Option Charlie-{idx} "
        f"Option Delta-{idx} "
        f"Show answer B. Because option B aligns with concept {idx}."
    )
    return {
        "id": f"chunk_{idx:03d}",
        "schema_version": "v4",
        "chunk_type": "assessment_item",
        "text": body,
        "bloom_level": bloom,
        "difficulty": difficulty,
        "learning_outcome_refs": list(los or []),
        "concept_tags": list(concept_tags or []),
        "misconceptions": list(misconceptions or []),
    }


def _make_explanation_chunk(
    *,
    idx: int,
    misconceptions: list[dict],
    los: list[str] | None = None,
    concept_tags: list[str] | None = None,
) -> dict:
    """Explanation chunk that carries misconceptions tied to a
    learning-outcome / concept set. Used to verify the misconception
    distractor harvest."""
    return {
        "id": f"expl_{idx:03d}",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Some explanation text.",
        "bloom_level": "understand",
        "difficulty": "foundational",
        "learning_outcome_refs": list(los or []),
        "concept_tags": list(concept_tags or []),
        "misconceptions": list(misconceptions),
    }


def _build_archive(
    courses_root: Path,
    slug: str,
    chunks: list[dict],
) -> Path:
    """Write a minimal archive containing only ``corpus/chunks.json``.

    ``QuizGenerator`` only reads chunks; the rest of the archive
    layout is irrelevant here."""
    root = courses_root / slug
    (root / "corpus").mkdir(parents=True)
    (root / "corpus" / "chunks.json").write_text(
        json.dumps(chunks), encoding="utf-8"
    )
    return root


def _build_distribution_archive(courses_root: Path, slug: str) -> Path:
    """Build an archive with enough items to satisfy ``{remember:3,
    understand:4, apply:3}`` (plus extras to verify sampling).
    """
    chunks: list[dict] = []
    counts = {"remember": 5, "understand": 6, "apply": 5, "analyze": 2}
    idx = 0
    for level, n in counts.items():
        for _ in range(n):
            idx += 1
            chunks.append(
                _make_assessment_chunk(
                    idx=idx,
                    bloom=level,
                    los=[f"co-{idx:02d}"],
                    concept_tags=[f"concept-{idx}"],
                )
            )
    return _build_archive(courses_root, slug, chunks)


def _build_misconception_archive(courses_root: Path, slug: str) -> Path:
    """Archive where assessment items share concept_tags and
    learning_outcome_refs with explanation chunks that carry
    misconceptions — so the harvest can match."""
    shared_concept = "shared-concept-x"
    shared_outcome = "co-99"

    chunks: list[dict] = []
    # Assessment items, all 'apply' so we can sample 2 of them.
    for idx in (1, 2, 3):
        chunks.append(
            _make_assessment_chunk(
                idx=idx,
                bloom="apply",
                los=[shared_outcome],
                concept_tags=[shared_concept],
            )
        )
    # Misconceptions live on an explanation chunk that overlaps on
    # concept_tags + LOs — the harvest rule should pick them up.
    chunks.append(
        _make_explanation_chunk(
            idx=10,
            los=[shared_outcome],
            concept_tags=[shared_concept],
            misconceptions=[
                {
                    "misconception": "RDF triples are like SQL rows.",
                    "correction": "Triples are first-class facts, not rows.",
                },
                {
                    "misconception": "An IRI must always resolve to a webpage.",
                    "correction": "IRIs are identifiers, not addresses.",
                },
                {
                    "misconception": "Blank node labels are global.",
                    "correction": "Blank node labels are document-local.",
                },
            ],
        )
    )
    return _build_archive(courses_root, slug, chunks)


# ---------------------------------------------------------------------- #
# CLI tests
# ---------------------------------------------------------------------- #


def test_cli_help_lists_generate_quiz():
    runner = CliRunner()
    result = runner.invoke(libv2_group, ["generate-quiz", "--help"])
    assert result.exit_code == 0
    assert "--slug" in result.output
    assert "--bloom-mix" in result.output
    assert "--use-misconceptions-as-distractors" in result.output
    assert "--seed" in result.output


def test_bloom_mix_produces_exact_distribution(tmp_path: Path):
    """`--bloom-mix '{"remember":3,"understand":4,"apply":3}'` -> 10 items."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"remember":3,"understand":4,"apply":3}',
            "--seed",
            "42",
            "--courses-root",
            str(courses_root),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    quiz = json.loads(result.output)
    items = quiz["items"]
    assert len(items) == 10

    by_bloom: dict[str, int] = {}
    for item in items:
        by_bloom[item["bloom_level"]] = by_bloom.get(item["bloom_level"], 0) + 1
    assert by_bloom == {"remember": 3, "understand": 4, "apply": 3}


def test_bloom_mix_shortage_fails_loud(tmp_path: Path):
    """Asking for more 'create' items than the corpus has raises an error."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")
    # The distribution archive has 0 'create' items.

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"create":2}',
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 2, result.output
    # Specific shortage report (not a generic stack trace)
    assert "Bloom mix exceeds available" in result.output
    assert "create" in result.output
    assert "requested=2" in result.output
    assert "available=0" in result.output


def test_misconceptions_become_distractors(tmp_path: Path):
    """At least one distractor matches a real misconception statement."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_misconception_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"apply":2}',
            "--use-misconceptions-as-distractors",
            "--num-distractors",
            "3",
            "--seed",
            "7",
            "--courses-root",
            str(courses_root),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    quiz = json.loads(result.output)
    assert quiz["use_misconceptions_as_distractors"] is True

    canonical_mc_set = {
        "RDF triples are like SQL rows.",
        "An IRI must always resolve to a webpage.",
        "Blank node labels are global.",
    }
    matches = 0
    for item in quiz["items"]:
        for d in item.get("misconception_distractors", []):
            assert d["source"] == "misconception"
            if d["text"] in canonical_mc_set:
                matches += 1
    assert matches >= 1, (
        "Expected ≥1 distractor to match a real misconception statement; "
        f"got 0. Quiz: {json.dumps(quiz, indent=2)}"
    )
    # Reporting counter sanity-check.
    assert quiz["distractor_source_counts"]["misconception"] >= 1


def test_seed_produces_identical_output(tmp_path: Path):
    """Running with --seed 42 twice produces byte-identical output."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")

    runner = CliRunner()
    args = [
        "generate-quiz",
        "--slug",
        "demo-course",
        "--bloom-mix",
        '{"remember":2,"understand":3,"apply":2}',
        "--seed",
        "42",
        "--courses-root",
        str(courses_root),
        "--format",
        "json",
    ]
    result_a = runner.invoke(libv2_group, args)
    result_b = runner.invoke(libv2_group, args)
    assert result_a.exit_code == 0, result_a.output
    assert result_b.exit_code == 0, result_b.output
    assert result_a.output == result_b.output


def test_qti_format_emits_valid_xml(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"remember":1,"understand":2}',
            "--seed",
            "1",
            "--courses-root",
            str(courses_root),
            "--format",
            "qti",
        ],
    )
    assert result.exit_code == 0, result.output
    # Must parse as XML.
    root = ET.fromstring(result.output)
    # Root tag carries the QTI 1.2 namespace.
    assert root.tag.endswith("questestinterop")
    ns = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
    assessment = root.find(f"{{{ns}}}assessment")
    assert assessment is not None
    section = assessment.find(f"{{{ns}}}section")
    assert section is not None
    items = section.findall(f"{{{ns}}}item")
    # 3 sampled items × 1 question each (heuristic single-question split)
    assert len(items) >= 3


def test_md_format_emits_readable_markdown(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"remember":1,"apply":1}',
            "--seed",
            "9",
            "--courses-root",
            str(courses_root),
            "--format",
            "md",
        ],
    )
    assert result.exit_code == 0, result.output
    md = result.output
    assert md.startswith("# Quiz: demo-course")
    assert "## Item 1." in md
    assert "## Item 2." in md
    assert "Bloom mix" in md
    # Heuristic splitter detects the "Show answer B." marker — at
    # least one item should report the correct letter.
    assert "_Correct_: **B**" in md


def test_imscc_format_writes_valid_zip(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")
    output = tmp_path / "out" / "quiz.imscc"

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"apply":2}',
            "--seed",
            "11",
            "--courses-root",
            str(courses_root),
            "--format",
            "imscc",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "imsmanifest.xml" in names
        assert "quiz/quiz.xml" in names
        # Manifest parses as XML.
        ET.fromstring(zf.read("imsmanifest.xml").decode("utf-8"))
        # Quiz XML parses as QTI.
        ET.fromstring(zf.read("quiz/quiz.xml").decode("utf-8"))


def test_imscc_format_requires_output(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"apply":1}',
            "--courses-root",
            str(courses_root),
            "--format",
            "imscc",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "requires --output" in result.output


def test_outcomes_filter_restricts_pool(tmp_path: Path):
    """Filtering by --outcomes shrinks the eligible pool and can
    reduce a previously-satisfiable bloom_mix into a shortage."""
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")
    # Only 1 chunk has co-01 — asking for 2 of any level filtered to
    # co-01 must shortage-fail.

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            '{"remember":2}',
            "--outcomes",
            "co-01",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 2
    assert "available=" in result.output


def test_archive_not_found_returns_clear_error(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "missing-slug",
            "--bloom-mix",
            '{"apply":1}',
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "not found" in result.output.lower()


def test_invalid_bloom_mix_json_rejected(tmp_path: Path):
    courses_root = tmp_path / "courses"
    courses_root.mkdir()
    _build_distribution_archive(courses_root, "demo-course")

    runner = CliRunner()
    result = runner.invoke(
        libv2_group,
        [
            "generate-quiz",
            "--slug",
            "demo-course",
            "--bloom-mix",
            "not-json",
            "--courses-root",
            str(courses_root),
        ],
    )
    assert result.exit_code != 0
    # click.BadParameter routes to stderr with our explanatory message.
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "valid JSON" in combined
