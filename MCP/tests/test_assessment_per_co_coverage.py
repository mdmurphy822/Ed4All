"""Per-CO assessment coverage (owner-directed 2026-07-19).

End-to-end regression net for the strict archival coverage contract:
every TO + CO in the synthesized-objectives universe must end up with an
``assessment_item`` CHUNK whose ``learning_outcome_refs`` cover it
(``lib/validators/libv2/packet_integrity.py`` — OBJECTIVE_NO_ASSESSMENT /
UNCOVERED_TERMINAL_OUTCOME, gate_id ``packet_integrity_strict``).

Four load-bearing seams, each tested here:

1. ``AssessmentGenerator.generate(ensure_objective_coverage=True)`` — the
   coverage-first distribution emits >=1 item per objective (the legacy
   nested distribution exhausts question_count on the first 1-2).
2. ``run_assessment_synthesis`` — the W10 quiz tier grows each weekly
   quiz's item budget to >=1 item per (TO + child COs), iterates the FULL
   CO universe (learning_outcomes-only + orphan COs included), and each
   emitted QTI ``<item>`` carries its objective_id in the ``title`` attr
   (the metadata the chunk harvest reads).
3. ``run_imscc_chunking`` — harvests one ``assessment_item`` chunk per
   objective-anchored QTI item from the packaged ``06_assessments/*.xml``
   with gate-resolvable (lowercase-normalized) ``learning_outcome_refs``.
4. ``PacketIntegrityValidator`` (strict coverage) — passes on a synthetic
   archive built from the harvested chunks; fails without them (negative
   control guarding the test's own sensitivity).

NO LLM, NO GPU — the quiz tier is deterministic. No course slugs; all
fixtures under tmp_path.
"""

from __future__ import annotations

import asyncio
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools as pt  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    _harvest_qti_assessment_chunks,
    _index_objective_records,
    _parse_qti_assessment_items,
)
from Trainforge.generators.assessment_generator import (  # noqa: E402
    AssessmentGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TERM_SENTENCES = [
    "A monomial is a polynomial that contains exactly one term of interest.",
    "A binomial is a polynomial that contains exactly two separate terms.",
    "A trinomial is a polynomial that contains exactly three separate terms.",
    "A coefficient is a numerical factor multiplying the variable in a term.",
    "An exponent is a superscript number counting repeated multiplication.",
    "A variable is a symbol standing in for an unknown numeric quantity.",
    "A constant is a fixed value that never changes within an expression.",
]


def _fixture_chunks(course: str = "demo") -> list:
    """Chunks rich enough for the deterministic MC builder (terms +
    >=3 distinct distractor definitions)."""
    chunks = []
    for i, sentence in enumerate(_TERM_SENTENCES, start=1):
        chunks.append({
            "id": f"{course}_chunk_{i:05d}",
            "chunk_type": "explanation",
            "text": f"{sentence} Practice recognizing this in expressions.",
            "learning_outcome_refs": [],
            "concept_tags": [],
        })
    return chunks


def _fixture_objectives_doc() -> dict:
    """2 TOs / 5 COs. CO-04 exists ONLY in learning_outcomes (no chapter
    entry, no terminal backlink -> orphan guard). CO-05 is a chapter CO
    WITHOUT a terminal_id (orphan within chapter form)."""
    return {
        "course_name": "PERCOCOV",
        "duration_weeks": 2,
        "terminal_objectives": [
            {"id": "TO-01",
             "statement": "Learners will classify polynomial expressions.",
             "child_co_ids": ["CO-01", "CO-02"]},
            {"id": "TO-02",
             "statement": "Learners will evaluate algebraic expressions.",
             "child_co_ids": ["CO-03"]},
        ],
        "chapter_objectives": [
            {"chapter": "ch1", "objectives": [
                {"id": "CO-01", "terminal_id": "TO-01",
                 "statement": "Identify monomials among polynomial forms.",
                 "source_chunk_ids": ["percocov_chunk_00001"]},
                {"id": "CO-02", "terminal_id": "TO-01",
                 "statement": "Distinguish binomials from trinomials.",
                 "source_chunk_ids": ["percocov_chunk_00002"]},
            ]},
            {"chapter": "ch2", "objectives": [
                {"id": "CO-03", "terminal_id": "TO-02",
                 "statement": "Evaluate expressions containing exponents.",
                 "source_chunk_ids": ["percocov_chunk_00005"]},
                {"id": "CO-05",
                 "statement": "Recognize constants inside expressions."},
            ]},
        ],
        "learning_outcomes": [
            {"id": "TO-01", "hierarchy_level": "terminal",
             "statement": "Learners will classify polynomial expressions."},
            {"id": "TO-02", "hierarchy_level": "terminal",
             "statement": "Learners will evaluate algebraic expressions."},
            {"id": "CO-04", "hierarchy_level": "chapter",
             "statement": "Interpret coefficients within algebraic terms."},
        ],
    }


_ALL_OBJECTIVE_IDS = {"TO-01", "TO-02", "CO-01", "CO-02", "CO-03", "CO-04", "CO-05"}
_ALL_CO_IDS = {"CO-01", "CO-02", "CO-03", "CO-04", "CO-05"}


@pytest.fixture()
def registry():
    return _build_tool_registry()


@pytest.fixture(autouse=True)
def _isolate_captures(tmp_path, monkeypatch):
    """Keep decision-capture mirrors out of the repo tree."""
    monkeypatch.setenv(
        "ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "captures")
    )


# ---------------------------------------------------------------------------
# 0. _index_objective_records — full-universe harvest.
# ---------------------------------------------------------------------------


def test_index_objective_records_includes_learning_outcomes_universe():
    to_map, co_map = _index_objective_records(_fixture_objectives_doc())
    assert set(to_map) == {"TO-01", "TO-02"}
    assert set(co_map) == _ALL_CO_IDS
    # Richer chapter records win over the flat learning_outcomes echo.
    assert co_map["CO-01"].get("terminal_id") == "TO-01"


# ---------------------------------------------------------------------------
# 1. Coverage-first generator distribution.
# ---------------------------------------------------------------------------


def test_generate_coverage_mode_covers_every_objective():
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    obj_ids = ["TO-01", "CO-01", "CO-02", "CO-03", "CO-04", "CO-05"]
    assessment = gen.generate(
        course_code="DEMO",
        objective_ids=obj_ids,
        bloom_levels=["remember", "understand", "apply"],
        question_count=max(5, len(obj_ids)),
        source_chunks=_fixture_chunks(),
        ensure_objective_coverage=True,
    )
    covered = {q.objective_id for q in assessment.questions}
    assert covered == set(obj_ids)
    assert len(assessment.questions) >= len(obj_ids)


def test_generate_legacy_mode_byte_compatible_distribution():
    """Default (flag off) keeps the legacy nested distribution: 5 items
    across 6 objectives x 3 blooms exhaust on the first two objectives."""
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    obj_ids = ["TO-01", "CO-01", "CO-02", "CO-03", "CO-04", "CO-05"]
    assessment = gen.generate(
        course_code="DEMO",
        objective_ids=obj_ids,
        bloom_levels=["remember", "understand", "apply"],
        question_count=5,
        source_chunks=_fixture_chunks(),
    )
    assert len(assessment.questions) <= 5
    covered = {q.objective_id for q in assessment.questions}
    assert covered != set(obj_ids)  # legacy mode does NOT cover all six


def test_generate_coverage_mode_scoped_chunks_fall_back_to_full_pool():
    """A CO scoped to a starved chunk subset still gets an item via the
    full-pool fallback (coverage beats scoping)."""
    gen = AssessmentGenerator(capture=None, check_leaks=False)
    starved = [{
        "id": "demo_chunk_09999",
        "chunk_type": "explanation",
        "text": "Too thin.",
        "learning_outcome_refs": [],
        "concept_tags": [],
    }]
    assessment = gen.generate(
        course_code="DEMO",
        objective_ids=["CO-01"],
        bloom_levels=["understand"],
        question_count=1,
        source_chunks=_fixture_chunks(),
        ensure_objective_coverage=True,
        chunks_by_objective={"CO-01": starved},
    )
    assert {q.objective_id for q in assessment.questions} == {"CO-01"}


# ---------------------------------------------------------------------------
# 2. W10 quiz tier — one QTI <item title=objective_id> per TO + CO.
# ---------------------------------------------------------------------------


def _write_fixture_export(tmp_path: Path, course: str) -> Path:
    exports_root = tmp_path / "exports"
    proj = exports_root / f"PROJ-{course}-20260719000000"
    obj_dir = proj / "01_learning_objectives"
    obj_dir.mkdir(parents=True)
    (obj_dir / "synthesized_objectives.json").write_text(
        json.dumps(_fixture_objectives_doc()), encoding="utf-8"
    )
    (proj / "project_config.json").write_text(
        json.dumps({"course_name": course}), encoding="utf-8"
    )
    return exports_root


def _write_fixture_chunkset(tmp_path: Path, course: str) -> Path:
    chunks_path = tmp_path / "dart_chunks" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in _fixture_chunks(course.lower()):
            fh.write(json.dumps(c) + "\n")
    return chunks_path


def _emitted_quiz_item_titles(assessments_dir: Path) -> set:
    titles = set()
    for xml_path in sorted(assessments_dir.glob("*.xml")):
        _title, items = _parse_qti_assessment_items(
            xml_path.read_text(encoding="utf-8")
        )
        for item in items:
            if item.get("objective_ref"):
                titles.add(item["objective_ref"])
    return titles


def _run_assessment_synthesis_fixture(registry, tmp_path, monkeypatch, course):
    exports_root = _write_fixture_export(tmp_path, course)
    chunks_path = _write_fixture_chunkset(tmp_path, course)
    monkeypatch.setattr(pt, "courseforge_exports_dir", lambda: exports_root)
    tool = registry["run_assessment_synthesis"]
    result = json.loads(asyncio.run(tool(
        project_id=f"PROJ-{course}-20260719000000",
        course_name=course,
        dart_chunks_path=str(chunks_path),
        question_count=2,
    )))
    assert result.get("success") is True, result
    return Path(result["assessments_dir"])


def test_quiz_tier_emits_item_per_co(registry, tmp_path, monkeypatch):
    course = "PERCOCOV"
    assessments_dir = _run_assessment_synthesis_fixture(
        registry, tmp_path, monkeypatch, course
    )
    titles = _emitted_quiz_item_titles(assessments_dir)
    missing = _ALL_OBJECTIVE_IDS - titles
    assert not missing, f"objectives without an emitted quiz item: {missing}"
    # Weekly structure preserved: one quiz doc per TO/week.
    manifest = json.loads(
        (assessments_dir / "manifest.json").read_text(encoding="utf-8")
    )
    quiz_entries = [
        e for e in manifest["assessments"] if e.get("type") == "qti"
    ]
    assert len(quiz_entries) == 2


# ---------------------------------------------------------------------------
# 3. imscc_chunking QTI harvest -> assessment_item chunks.
# ---------------------------------------------------------------------------


def _build_fixture_imscc(tmp_path: Path, assessments_dir: Path) -> Path:
    imscc = tmp_path / "package.imscc"
    page_html = (
        "<html><head><title>Week 1</title></head><body>"
        "<h1>Polynomials</h1><p>A monomial is a polynomial that contains "
        "exactly one term of interest.</p></body></html>"
    )
    with zipfile.ZipFile(imscc, "w") as zf:
        zf.writestr("week_01/content.html", page_html)
        for xml_path in sorted(assessments_dir.glob("*.xml")):
            zf.writestr(f"06_assessments/{xml_path.name}", xml_path.read_text())
    return imscc


def _harvested_assessment_chunks(chunks_path: Path) -> list:
    out = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        if chunk.get("chunk_type") == "assessment_item":
            out.append(chunk)
    return out


def test_imscc_chunking_harvests_per_co_assessment_chunks(
    registry, tmp_path, monkeypatch
):
    course = "PERCOCOV"
    assessments_dir = _run_assessment_synthesis_fixture(
        registry, tmp_path, monkeypatch, course
    )
    imscc = _build_fixture_imscc(tmp_path, assessments_dir)
    libv2_root = tmp_path / "libv2"
    tool = registry["run_imscc_chunking"]
    result = json.loads(asyncio.run(tool(
        course_name=course,
        imscc_path=str(imscc),
        libv2_root=str(libv2_root),
    )))
    assert result.get("success") is True, result
    assert result.get("qti_assessment_chunks_count", 0) > 0
    chunks_path = Path(result["imscc_chunks_path"])
    assessment_chunks = _harvested_assessment_chunks(chunks_path)
    assert assessment_chunks, "no assessment_item chunks harvested"

    # Every TO + CO covered by some chunk's refs, lowercase-normalized
    # exactly as the strict gate compares (universe is lowercase).
    covered = set()
    for chunk in assessment_chunks:
        refs = chunk.get("learning_outcome_refs") or []
        assert refs, f"unanchored assessment chunk {chunk.get('id')}"
        for ref in refs:
            covered.add(ref.strip().lower())
    expected = {o.lower() for o in _ALL_OBJECTIVE_IDS}
    missing = expected - covered
    assert not missing, f"objectives without assessment-chunk refs: {missing}"

    # id uniqueness + canonical v4 id pattern.
    all_ids = [
        json.loads(line)["id"]
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(all_ids) == len(set(all_ids))
    import re
    for cid in all_ids:
        assert re.match(r"^[a-z][a-z0-9_]*_chunk_(\d{5}|[0-9a-f]{16})$", cid), cid


def test_harvest_skips_non_qti_and_unanchored_items(tmp_path):
    """imsdt/assignment docs and items without a canonical LO title never
    become chunks (anti-fabrication / anti-UNANCHORED_ASSESSMENT)."""
    from Courseforge.scripts.qti_emitter import (
        assessment_to_qti,
        discussion_to_imsdt,
    )
    quiz_xml = assessment_to_qti({
        "assessment_id": "ASM-X",
        "title": "Week 1 Quiz",
        "questions": [
            {"question_id": "q-1", "question_type": "multiple_choice",
             "stem": "<p>Pick one.</p>", "objective_id": "CO-01",
             "bloom_level": "understand",
             "choices": [
                 {"id": "A", "text": "Right", "is_correct": True},
                 {"id": "B", "text": "Wrong", "is_correct": False},
             ], "correct_answer": "A"},
            # No canonical LO ref in the title -> must be skipped.
            {"question_id": "q-2", "question_type": "essay",
             "stem": "<p>Explain.</p>", "objective_id": "not-an-lo-id",
             "bloom_level": "apply", "choices": []},
        ],
    })
    disc_xml = discussion_to_imsdt({
        "discussion_id": "DISC-TO-01", "objective_id": "TO-01",
        "title": "W1 Discussion", "text": "Discuss."})

    emitted = []

    def _fake_create_chunk(**kwargs):
        emitted.append(kwargs)
        return {
            "id": kwargs["chunk_id"],
            "chunk_type": kwargs["chunk_type"],
            "learning_outcome_refs": kwargs["item"]["objective_refs"],
        }

    chunks = _harvest_qti_assessment_chunks(
        [
            {"path": "06_assessments/week_01_quiz.xml", "content": quiz_xml},
            {"path": "06_assessments/week_01_discussion.xml", "content": disc_xml},
        ],
        create_chunk=_fake_create_chunk,
        existing_chunks=[{"id": "demo_chunk_00007"}],
        course_code="DEMO",
    )
    assert len(chunks) == 1
    assert chunks[0]["learning_outcome_refs"] == ["CO-01"]
    # id sequence continues after the existing max suffix.
    assert chunks[0]["id"] == "demo_chunk_00008"
    assert emitted[0]["chunk_type"] == "assessment_item"


# ---------------------------------------------------------------------------
# 4. Strict packet-integrity coverage gate on a synthetic archive.
# ---------------------------------------------------------------------------

_COVERAGE_CODES = {"OBJECTIVE_NO_ASSESSMENT", "UNCOVERED_TERMINAL_OUTCOME"}


def _write_archive(tmp_path: Path, chunks: list) -> Path:
    archive = tmp_path / "archive"
    (archive / "imscc_chunks").mkdir(parents=True)
    with (archive / "imscc_chunks" / "chunks.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    objectives = {
        "terminal_outcomes": [
            {"id": "TO-01", "statement": "Classify polynomials."},
            {"id": "TO-02", "statement": "Evaluate expressions."},
        ],
        "component_objectives": [
            {"id": cid, "parent_terminal": parent, "statement": f"CO {cid}"}
            for cid, parent in [
                ("CO-01", "to-01"), ("CO-02", "to-01"), ("CO-03", "to-02"),
                ("CO-04", "to-01"), ("CO-05", "to-02"),
            ]
        ],
    }
    (archive / "objectives.json").write_text(
        json.dumps(objectives), encoding="utf-8"
    )
    return archive


def _teaching_chunks() -> list:
    return [
        {"id": f"percocov_chunk_1{i:04d}", "chunk_type": "explanation",
         "text": f"Teaching text for {ref}.",
         "learning_outcome_refs": [ref]}
        for i, ref in enumerate(
            ["to-01", "to-02", "co-01", "co-02", "co-03", "co-04", "co-05"]
        )
    ]


def _assessment_chunks() -> list:
    return [
        {"id": f"percocov_chunk_2{i:04d}", "chunk_type": "assessment_item",
         "text": f"Quiz item for {ref}.",
         "learning_outcome_refs": [ref]}
        for i, ref in enumerate(
            ["to-01", "to-02", "co-01", "co-02", "co-03", "co-04", "co-05"]
        )
    ]


def test_packet_integrity_strict_passes_with_per_co_assessment_chunks(tmp_path):
    from lib.validators.libv2.packet_integrity import PacketIntegrityValidator
    archive = _write_archive(
        tmp_path, _teaching_chunks() + _assessment_chunks()
    )
    validator = PacketIntegrityValidator(strict_coverage=True)
    result = validator.validate(archive)
    coverage_issues = [
        i for i in result.issues if i.issue_code in _COVERAGE_CODES
    ]
    assert not coverage_issues, [
        (i.issue_code, i.message) for i in coverage_issues
    ]


def test_packet_integrity_strict_fails_without_per_co_coverage(tmp_path):
    """Negative control: TO-only assessment refs (tonight's production
    shape) leave every CO uncovered -> the gate MUST fire."""
    from lib.validators.libv2.packet_integrity import PacketIntegrityValidator
    to_only_assessments = [
        {"id": "percocov_chunk_30000", "chunk_type": "assessment_item",
         "text": "Quiz item for to-01.", "learning_outcome_refs": ["to-01"]},
        {"id": "percocov_chunk_30001", "chunk_type": "assessment_item",
         "text": "Quiz item for to-02.", "learning_outcome_refs": ["to-02"]},
    ]
    archive = _write_archive(
        tmp_path, _teaching_chunks() + to_only_assessments
    )
    validator = PacketIntegrityValidator(strict_coverage=True)
    result = validator.validate(archive)
    fired = {
        i.context.get("objective_id")
        for i in result.issues
        if i.issue_code == "OBJECTIVE_NO_ASSESSMENT"
    }
    assert {"co-01", "co-02", "co-03", "co-04", "co-05"} <= fired


# ---------------------------------------------------------------------------
# 5. qti_well_formed stays green on the per-CO quiz docs.
# ---------------------------------------------------------------------------


def test_qti_well_formed_green_on_per_co_quiz(registry, tmp_path, monkeypatch):
    from lib.validators.qti_well_formed import QtiWellFormedValidator
    course = "PERCOCOV"
    assessments_dir = _run_assessment_synthesis_fixture(
        registry, tmp_path, monkeypatch, course
    )
    validator = QtiWellFormedValidator()
    result = validator.validate({"qti_dir": str(assessments_dir)})
    critical = [i for i in result.issues if i.severity == "critical"]
    assert result.passed, [(i.code, i.message) for i in critical]
