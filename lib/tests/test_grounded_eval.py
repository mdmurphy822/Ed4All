"""Tests for the WS3 grounded-answer eval harness (D8).

CI-safe: a fake grounded-answer pipeline (``answer_fn=``) is injected so the
harness never imports the real E6 wave-B ``grounded_answer`` module (which has
not landed). One test asserts the import-guard raises ``PipelineUnavailable``
when no ``answer_fn`` is injected and the real module is absent.

The fixture mini-course is copied into a tmp LibV2 layout (``ED4ALL_LIBV2_ROOT``
monkeypatched — the WS1/WS2 fixture pattern) so ``load_gold_set`` resolves the
pinned chunkset and its sha pin.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lib.retrieval.grounded_eval import (
    PipelineUnavailable,
    _sample_size,
    _seeded_order,
    run_grounded_eval,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "retrieval" / "mini_course"
)


# ===========================================================================
# Fixture LibV2 layout
# ===========================================================================

@pytest.fixture
def libv2_course(tmp_path, monkeypatch):
    """Copy the mini-course fixture into a tmp LibV2 course dir."""
    slug = "mini-retrieval-101"
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / slug
    shutil.copytree(FIXTURE, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return tmp_path, slug, course_dir


# ===========================================================================
# Fake pipeline (GroundedAnswer-shaped, duck-typed)
# ===========================================================================

class _FakeCitation:
    def __init__(self, chunk_id, anchor_status="resolved_exact",
                 page_label="P", text_quote="q"):
        self.chunk_id = chunk_id
        self.anchor_status = anchor_status
        self.page_label = page_label
        self.text_quote = text_quote

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "anchor_status": self.anchor_status,
            "page_label": self.page_label,
            "text_quote": self.text_quote,
        }


class _FakeAnswer:
    def __init__(self, *, status, answer_text, citations, groundedness=None,
                 latency_ms=12.5):
        self.status = status
        self.answer_text = answer_text
        self.citations = citations
        self.groundedness = groundedness
        self.latency_ms = latency_ms
        self.model_id = "fake-model"
        self.prompt_version = "ws3.v1"
        self.confidence = {"policy_version": "ws3.v0-uncalibrated"}


def _gold_answer_fn(grounded_block=None):
    """Build a deterministic fake ``answer_course_question``.

    Answers gold questions with a citation to the gold-relevant primary chunk;
    refuses on refusal probes (probe text never matches a gold question).
    """
    # Map gold question_text → (primary chunk_id) from the fixture.
    answers_by_text = {
        "What does a vector store index?": "mini_alpha_chunk_001",
        "How is retrieval quality commonly measured?": "mini_alpha_chunk_003",
        "Where does the course cover chunking strategies?": "mini_beta_chunk_005",
    }

    def _fn(repo_root, course_slug, query, *, engine="lexical", limit=8,
            client=None, refusal_policy=None, with_groundedness=False,
            capture=None, **kwargs):
        cid = answers_by_text.get(query)
        if cid is None:
            # Refusal probe → low-confidence refusal.
            return _FakeAnswer(
                status="refused_low_confidence",
                answer_text=None,
                citations=[],
            )
        grounded = grounded_block if with_groundedness else None
        return _FakeAnswer(
            status="answered",
            answer_text=f"Answer about {cid}.",
            citations=[_FakeCitation(cid)],
            groundedness=grounded,
        )

    return _fn


_GROUNDED_OK = {
    "available": True,
    "groundedness_rate": 1.0,
    "scored_count": 2,
    "unsupported_count": 0,
    "contradicted_count": 0,
    "claims": [
        {"claim_text": "c", "verdict": "entailed", "entailment": 0.9,
         "contradiction": 0.05, "best_chunk_id": "x"},
    ],
}


# ===========================================================================
# Report shape + headline keys
# ===========================================================================

def test_report_shape_and_headline(libv2_course):
    repo_root, slug, _ = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_gold_answer_fn(_GROUNDED_OK),
        with_groundedness=True, write=False,
    )
    assert report["schema_version"] == "1.2"
    assert report["course_slug"] == slug
    assert report["engine"] == "lexical"
    assert report["model_id"] == "fake-model"
    assert report["prompt_version"] == "ws3.v1"
    assert report["gold"]["chunks_sha256"]  # pinned
    assert report["gold"]["chunkset_kind"] == "dart"

    headline = report["headline"]
    for key in (
        "answer_rate", "citation_resolution_rate", "citation_precision",
        "citation_precision_primary", "groundedness_rate_mean",
        "unsupported_claim_rate", "refusal", "latency_ms",
    ):
        assert key in headline
    # 3 answerable questions, all answered → answer_rate 1.0
    assert headline["answer_rate"] == 1.0
    # every emitted citation resolved + relevant primary
    assert headline["citation_resolution_rate"] == 1.0
    assert headline["citation_precision"] == 1.0
    assert headline["citation_precision_primary"] == 1.0
    # groundedness available → mean is a float, not None
    assert headline["groundedness_rate_mean"] == 1.0
    # 3 fixture probes, all refused
    assert headline["refusal"]["n_probes"] == 3
    assert headline["refusal"]["refusal_recall"] == 1.0
    assert headline["refusal"]["false_refusals_on_gold"] == 0
    assert report["blocked"] == {"invalid_citation": 0, "citation_gate": 0}


def test_per_question_rows(libv2_course):
    repo_root, slug, _ = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_gold_answer_fn(_GROUNDED_OK),
        with_groundedness=True, write=False,
    )
    assert len(report["questions"]) == 3
    row = report["questions"][0]
    # v1.0 base row fields are always present; citation_population is the only
    # P4-additive field on a v1.0 fixture (no expected_key_points / parts).
    base = {
        "question_id", "status", "n_citations", "citations_resolved",
        "citation_relevant_primary", "groundedness_rate", "latency_ms",
    }
    assert base <= set(row)
    assert set(row) - base <= {"citation_population"}
    # answered → the population breakdown is attached.
    assert "citation_population" in row
    assert row["status"] == "answered"
    assert row["n_citations"] == 1
    assert row["citations_resolved"] == 1
    assert row["citation_relevant_primary"] == 1


def test_groundedness_null_when_unavailable(libv2_course):
    repo_root, slug, _ = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_gold_answer_fn(grounded_block=None),
        with_groundedness=False, write=False,
    )
    assert report["headline"]["groundedness_rate_mean"] is None
    assert report["headline"]["unsupported_claim_rate"] is None
    for row in report["questions"]:
        assert row["groundedness_rate"] is None


# ===========================================================================
# chunkset_kind threading from the gold pin (E6/E7 fix)
# ===========================================================================

def test_chunkset_kind_from_gold_pin_reaches_pipeline(libv2_course):
    """run_grounded_eval threads the gold set's chunkset.kind pin into every
    answer_course_question call (gold questions AND refusal probes).

    Fail-without-fix: the pre-fix harness never passed chunkset_kind, so the
    pipeline fell back to its directory-presence guess instead of the gold's
    authoritative pin.
    """
    repo_root, slug, _ = libv2_course
    seen_kinds = []
    inner = _gold_answer_fn(_GROUNDED_OK)

    def _spy_fn(*args, **kwargs):
        seen_kinds.append(kwargs.get("chunkset_kind", "__ABSENT__"))
        return inner(*args, **kwargs)

    run_grounded_eval(
        repo_root, slug, engine="semantic", answer_fn=_spy_fn,
        with_groundedness=True, write=False,
    )
    # The mini fixture gold set pins chunkset.kind == "dart"; every call must
    # carry it (never absent, never the directory guess).
    assert seen_kinds  # at least the 3 gold questions + 3 probes
    assert all(k == "dart" for k in seen_kinds)


# ===========================================================================
# Determinism
# ===========================================================================

def _strip_volatile(report):
    r = dict(report)
    r.pop("generated_at", None)
    r.pop("_written", None)
    r.pop("_review_sample", None)
    for row in r.get("questions", []):
        row.pop("latency_ms", None)
    r.get("headline", {}).pop("latency_ms", None)
    return r


def test_two_runs_deterministic(libv2_course):
    repo_root, slug, _ = libv2_course
    a = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_gold_answer_fn(_GROUNDED_OK), write=False,
    )
    b = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_gold_answer_fn(_GROUNDED_OK), write=False,
    )
    assert _strip_volatile(a) == _strip_volatile(b)


# ===========================================================================
# Artifact emission
# ===========================================================================

def test_writes_report_and_review_sample(libv2_course):
    import json

    repo_root, slug, course_dir = libv2_course
    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_gold_answer_fn(_GROUNDED_OK), write=True,
    )
    eval_dir = course_dir / "retrieval_eval"
    reports = list(eval_dir.glob("grounded_answer_eval_*.json"))
    assert len(reports) == 1
    review = eval_dir / "groundedness_review_sample.json"
    assert review.exists()
    # filename-disjoint from gold_set / refusal_probes
    assert (eval_dir / "gold_set.json").exists()
    assert (eval_dir / "refusal_probes.json").exists()

    written = report["_written"]
    assert Path(written["report_path"]).exists()

    review_doc = json.loads(review.read_text())
    assert review_doc["course_slug"] == slug
    assert review_doc["n_questions"] == 3
    # n = max(10, 20%) capped at question count = 3
    assert review_doc["n_sampled"] == 3
    sample = review_doc["samples"][0]
    assert set(sample["reviewer_fields"]) == {
        "claims_all_supported", "citation_correct", "refusal_correct", "notes",
    }
    assert sample["reviewer_fields"]["claims_all_supported"] is None


# ===========================================================================
# Review-sample seeded determinism
# ===========================================================================

def test_review_sample_seeded_order_deterministic():
    qids = [f"gq-{i:04d}" for i in range(50)]
    order_a = _seeded_order(qids, "course-x")
    order_b = _seeded_order(qids, "course-x")
    assert order_a == order_b
    # Different slug → (very likely) different order, still deterministic.
    order_c = _seeded_order(qids, "course-y")
    assert order_c == _seeded_order(qids, "course-y")
    assert order_a != order_c  # seeded by slug


def test_sample_size_rule():
    assert _sample_size(3) == 3          # capped at question count
    assert _sample_size(50) == 10        # max(10, 20% of 50)=10
    assert _sample_size(100) == 20       # 20% of 100
    assert _sample_size(8) == 8          # capped


# ===========================================================================
# Critical gold-set issue → hard error (WS1 contract)
# ===========================================================================

def test_critical_gold_issue_raises(libv2_course):
    repo_root, slug, course_dir = libv2_course
    # Tamper the chunkset so the pinned sha no longer matches → critical.
    chunks = course_dir / "dart_chunks" / "chunks.jsonl"
    chunks.write_text(chunks.read_text() + '\n{"id":"x","text":"tamper"}\n')
    with pytest.raises(RuntimeError) as exc:
        run_grounded_eval(
            repo_root, slug, engine="lexical",
            answer_fn=_gold_answer_fn(_GROUNDED_OK), write=False,
        )
    assert "critical" in str(exc.value).lower()


# ===========================================================================
# Import-guard: pipeline absent → PipelineUnavailable
# ===========================================================================

def test_pipeline_unavailable_when_grounded_answer_absent(libv2_course, monkeypatch):
    """When no answer_fn is injected and grounded_answer is unimportable,
    run_grounded_eval raises PipelineUnavailable naming the dependency."""
    import builtins

    repo_root, slug, _ = libv2_course
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "lib.retrieval.grounded_answer" or (
            name.endswith("grounded_answer") and "lib.retrieval" in name
        ):
            raise ImportError("simulated: grounded_answer not landed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(PipelineUnavailable) as exc:
        run_grounded_eval(repo_root, slug, engine="lexical", write=False)
    assert "grounded_answer" in str(exc.value)
    assert "answer_course_question" in str(exc.value)


def test_pipeline_unavailable_is_notimplementederror():
    assert issubclass(PipelineUnavailable, NotImplementedError)


# ===========================================================================
# Milestone targets — measure-then-pin floors (pinned 2026-06-10)
# ===========================================================================

def test_milestone_targets_exist_and_are_sane():
    """The pinned milestone constants exist, carry the pin date, and every
    target is a sane proportion (0..1). citation_resolution_rate is pinned at
    exactly 1.0 (all measured runs resolved every emitted citation), and the
    ceiling set names exactly the lower-is-better metric."""
    from lib.retrieval.grounded_eval import (
        MILESTONE_CEILINGS,
        MILESTONE_TARGETS,
        MILESTONE_TARGETS_PINNED_AT,
    )

    assert MILESTONE_TARGETS_PINNED_AT == "2026-06-12"
    expected_keys = {
        "answer_rate",
        "citation_resolution_rate",
        "citation_precision",
        "groundedness_rate_mean",
        "unsupported_claim_rate",
        "refusal_recall",
        "refusal_precision",
    }
    assert set(MILESTONE_TARGETS) == expected_keys
    for key, target in MILESTONE_TARGETS.items():
        assert 0.0 < target <= 1.0, f"{key} target out of range: {target}"
    assert MILESTONE_TARGETS["citation_resolution_rate"] == 1.0
    # The only ceiling (lower is better) is unsupported_claim_rate; it must be
    # a known target key.
    assert MILESTONE_CEILINGS == {"unsupported_claim_rate"}
    assert MILESTONE_CEILINGS <= set(MILESTONE_TARGETS)


try:
    from lib.paths import libv2_path

    _LIBV2_COURSES = libv2_path() / "courses"
except Exception:  # pragma: no cover - defensive
    _LIBV2_COURSES = Path(__file__).resolve().parents[2] / "LibV2" / "courses"


def _discover_measured_courses():
    """Discover real courses with a stored grounded_answer_eval artifact.

    Course data dirs are gitignored user data; tests must not name slugs.
    Glob whatever is present, skip cleanly when none exist.
    """
    if not _LIBV2_COURSES.is_dir():
        return []
    # dict.fromkeys → de-dup courses with multiple dated artifacts, order-stable.
    slugs = {
        p.parent.parent.name: None
        for p in sorted(
            _LIBV2_COURSES.glob("*/retrieval_eval/grounded_answer_eval_*.json")
        )
    }
    return list(slugs)


def _latest_eval_artifact(slug: str):
    eval_dir = _LIBV2_COURSES / slug / "retrieval_eval"
    candidates = sorted(eval_dir.glob("grounded_answer_eval_*.json"))
    return candidates[-1] if candidates else None


_MEASURED_COURSES = _discover_measured_courses()


#: The gold basis the 2026-06-12 MILESTONE_TARGETS re-pin was measured against:
#: a frozen gold v1.1 set (schema_version "1.1"). The plan §4 calls a gold-basis
#: change a hard COMPARABILITY BOUNDARY — an artifact measured against an OLDER
#: gold basis (the pre-scale-up 10q/9-probe seed, gold schema < 1.1) cannot be
#: held to floors re-pinned to the new basis (citation_precision moves via gold
#: denominator semantics; refusal moves to the new hard-probe basis). Such
#: artifacts are SKIPPED here (not xfailed, not asserted) until they are
#: re-measured against the v1.1 gold — the honest tier-2 behavior for a
#: comparability boundary, mirroring how a sha-mismatched gold set fails closed.
_TARGETS_GOLD_BASIS_SCHEMA = "1.1"


def _artifact_predates_targets_gold_basis(doc: dict) -> bool:
    """True when the eval artifact was measured against a gold basis older than
    the one MILESTONE_TARGETS is pinned to (gold schema_version < 1.1, or a
    pre-v1.1 report that records no v1.1 gold pin)."""
    report_schema = str(doc.get("schema_version") or "")
    gold = doc.get("gold") if isinstance(doc.get("gold"), dict) else {}
    gold_schema = str(gold.get("schema_version") or "")
    # A v1.1 gold pin is the marker of the new basis; anything lacking it
    # (older report schema 1.0, or a gold block with no/old schema_version) is
    # the pre-scale-up seed basis and not comparable to the re-pinned floors.
    if gold_schema == _TARGETS_GOLD_BASIS_SCHEMA:
        return False
    if report_schema and report_schema < _TARGETS_GOLD_BASIS_SCHEMA:
        return True
    # No v1.1 gold pin recorded → treat as pre-basis (stale).
    return gold_schema < _TARGETS_GOLD_BASIS_SCHEMA


@pytest.mark.skipif(
    not _MEASURED_COURSES,
    reason="no stored grounded_answer_eval_*.json under LibV2/courses/*",
)
@pytest.mark.parametrize("slug", _MEASURED_COURSES)
def test_stored_eval_artifacts_meet_milestone_targets(slug):
    """Tier-2: the latest stored grounded_answer_eval_*.json for each measured
    course meets every pinned target — floors via >=, the unsupported-claim
    ceiling via <= — proving the targets are evidence-met today
    (measure-then-pin), not aspirational. Skips cleanly when LibV2 course data
    is absent so CI without committed artifacts still passes.

    SKIPS (does not assert) a course whose latest artifact predates the gold
    basis the targets were re-pinned to (2026-06-12, frozen gold v1.1): the
    plan §4 comparability boundary means an old-basis artifact's metrics are not
    on the same axis as the new floors. Such a course re-enters the gate the
    moment it is re-measured against the v1.1 gold (honest, no blanket xfail)."""
    import json

    from lib.retrieval.grounded_eval import MILESTONE_CEILINGS, MILESTONE_TARGETS

    path = _latest_eval_artifact(slug)
    if path is None:
        pytest.skip(f"no stored grounded_answer_eval artifact for {slug}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if _artifact_predates_targets_gold_basis(doc):
        pytest.skip(
            f"{slug}: latest artifact ({path.name}) predates the 2026-06-12 "
            f"frozen-gold v1.1 basis the milestone targets are pinned to "
            f"(comparability boundary, plan §4) — re-measure against v1.1 gold "
            f"to re-enter the gate"
        )
    headline = doc["headline"]
    refusal = headline["refusal"]

    measured = {
        "answer_rate": headline["answer_rate"],
        "citation_resolution_rate": headline["citation_resolution_rate"],
        "citation_precision": headline["citation_precision"],
        "groundedness_rate_mean": headline["groundedness_rate_mean"],
        "unsupported_claim_rate": headline["unsupported_claim_rate"],
        "refusal_recall": refusal["refusal_recall"],
        "refusal_precision": refusal["refusal_precision"],
    }
    for key, target in MILESTONE_TARGETS.items():
        value = measured[key]
        assert value is not None, f"{slug}: {key} missing from {path.name}"
        if key in MILESTONE_CEILINGS:
            assert value <= target, (
                f"{slug}: {key}={value} above pinned ceiling {target} "
                f"({path.name})"
            )
        else:
            assert value >= target, (
                f"{slug}: {key}={value} below pinned floor {target} "
                f"({path.name})"
            )


# ===========================================================================
# Eval containment — per-question composer exhaustion (Fix 3)
# ===========================================================================

def _exhausting_answer_fn(fail_query, exc):
    """answer_fn that RAISES `exc` for `fail_query`, answers others normally."""
    inner = _gold_answer_fn(_GROUNDED_OK)

    def _fn(repo_root, course_slug, query, **kwargs):
        if query == fail_query:
            raise exc
        return inner(repo_root, course_slug, query, **kwargs)

    return _fn


def test_composer_exhaustion_is_contained_per_question(libv2_course):
    """A composer error on ONE gold question is recorded as composer_exhausted;
    the OTHER questions are still evaluated and the artifact still writes."""
    from lib.retrieval.answer_composer import AnswerComposeError

    repo_root, slug, course_dir = libv2_course
    fail_q = "What does a vector store index?"
    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_exhausting_answer_fn(
            fail_q, AnswerComposeError("parse exhausted")
        ),
        with_groundedness=True, write=True,
    )
    # Per-question status recorded.
    statuses = {r["question_id"]: r["status"] for r in report["questions"]}
    exhausted = [s for s in statuses.values() if s == "composer_exhausted"]
    assert len(exhausted) == 1
    # The other two gold questions answered normally.
    assert sorted(statuses.values()).count("answered") == 2
    # Aggregate honesty: exhausted question is NOT-answered (out of
    # answered_count) but stays in the answer_rate denominator (3 questions).
    assert report["headline"]["answer_rate"] == pytest.approx(2 / 3)
    assert report["composer_exhausted"] == 1
    # NOT counted as a refusal.
    assert report["headline"]["refusal"]["false_refusals_on_gold"] == 0
    # Artifact written despite the per-question failure.
    assert Path(report["_written"]["report_path"]).exists()


def test_truncation_error_is_contained_per_question(libv2_course):
    from lib.retrieval.answer_backend import PromptTruncatedError

    repo_root, slug, _ = libv2_course
    fail_q = "How is retrieval quality commonly measured?"
    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_exhausting_answer_fn(
            fail_q, PromptTruncatedError("head truncated")
        ),
        write=False,
    )
    statuses = [r["status"] for r in report["questions"]]
    assert statuses.count("composer_exhausted") == 1
    assert report["composer_exhausted"] == 1


def test_backend_unavailable_is_NOT_contained(libv2_course):
    """A systemic backend outage must abort the eval (NOT be swallowed per
    question) — otherwise every metric silently hollows out."""
    from lib.retrieval.answer_backend import AnswerBackendUnavailable

    repo_root, slug, _ = libv2_course

    def _all_unavailable(repo_root, course_slug, query, **kwargs):
        raise AnswerBackendUnavailable("server down")

    with pytest.raises(AnswerBackendUnavailable):
        run_grounded_eval(
            repo_root, slug, engine="lexical",
            answer_fn=_all_unavailable, write=False,
        )


def test_cli_exit_zero_with_artifact_on_composer_exhaustion(libv2_course):
    """The CLI/`run_grounded_eval` returns the report (exit 0 path) with the
    artifact written even when a question's composer exhausted — exit stays 0
    unless something systemic fails."""
    from lib.retrieval.answer_composer import InvalidCitationError

    repo_root, slug, course_dir = libv2_course
    fail_q = "Where does the course cover chunking strategies?"
    report = run_grounded_eval(
        repo_root, slug, engine="lexical",
        answer_fn=_exhausting_answer_fn(
            fail_q, InvalidCitationError("persistent bad ids")
        ),
        write=True,
    )
    eval_dir = course_dir / "retrieval_eval"
    assert list(eval_dir.glob("grounded_answer_eval_*.json"))
    # The review sample still builds over all three questions.
    assert report["_review_sample"]["n_questions"] == 3
