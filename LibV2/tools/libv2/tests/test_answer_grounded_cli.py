"""WS3 Wave C — CliRunner tests for the grounded-answer CLI surface.

Covers ``libv2 answer-grounded`` (the single entry point), plus the thin
delegations ``libv2 answer-eval`` and ``libv2 refusal-calibrate``.

The pipeline (``lib.retrieval.grounded_answer.answer_course_question``) is
stubbed via monkeypatch on the symbol the CLI imports — NO model calls, NO
local server, NO vector index. Coverage:

- happy path (answered) — exit 0, plain-text rendering of answer + citations.
- refusal path — exit 2, no answer text, reason surfaced.
- blocked citation-gate path — exit 3, answer withheld, failing citation shown.
- ``--json`` shape — full ``GroundedAnswer.to_dict()`` round-trips.
- typed-error exits — AnswerBackendUnavailable + AnswerProviderNotLocal (exit 1
  with operator guidance), SemanticIndexMissing (exit 1, build guidance).
- unknown engine rejected by click.Choice (exit 2 — usage error).
- unknown course fails closed (exit 1).
- ``--engine auto`` resolves semantic when an index manifest exists, else lexical.
- delegations: answer-eval / refusal-calibrate forward argv + exit code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import LibV2.tools.libv2.cli as cli  # noqa: E402
from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402
from lib.retrieval.grounded_answer import (  # noqa: E402
    Citation,
    GroundedAnswer,
)

_SLUG = "mini-answer-101"


def _make_course(repo_root: Path, *, with_index: bool = False) -> Path:
    cdir = repo_root / "courses" / _SLUG
    (cdir / "semantik_chunks").mkdir(parents=True)
    (cdir / "semantik_chunks" / "chunks.jsonl").write_text("{}\n")
    (cdir / "manifest.json").write_text(json.dumps({"classification": {}}))
    if with_index:
        from LibV2.tools.libv2.vector_index import (
            MANIFEST_FILENAME,
            VECTOR_INDEX_DIRNAME,
        )

        idx = cdir / VECTOR_INDEX_DIRNAME
        idx.mkdir(parents=True)
        (idx / MANIFEST_FILENAME).write_text(json.dumps({"chunkset_kind": "semantik"}))
    return cdir


def _invoke(repo_root: Path, *args: str):
    return CliRunner().invoke(libv2_main, ["--repo", str(repo_root), *args])


def _answered(*, warnings=None) -> GroundedAnswer:
    return GroundedAnswer(
        status="answered",
        query="q",
        course_slug=_SLUG,
        engine="lexical",
        answer_text="A SHACL NodeShape constrains nodes in the data graph.",
        citations=[
            Citation(
                chunk_id="chunk_001",
                item_path="module1/page.html",
                section_heading="Node Shapes",
                module_id="m1",
                page_label="Node Shapes",
                anchor_status="resolved_exact",
                source_path="module1/page.html",
                text_quote="A NodeShape constrains nodes.",
                link_target={"kind": "course_page", "item_path": "module1/page.html"},
            )
        ],
        refusal=None,
        confidence={"signals": {"top_score": 0.9}, "policy_version": "ws3.v0-uncalibrated"},
        groundedness=None,
        warnings=list(warnings or []),
        model_id="qwen2.5:7b-instruct-q4_K_M",
        prompt_version="ws3.v1",
        generated_at="2026-06-09T00:00:00Z",
        latency_ms=42.0,
    )


def _refused() -> GroundedAnswer:
    return GroundedAnswer(
        status="refused_low_confidence",
        query="q",
        course_slug=_SLUG,
        engine="lexical",
        answer_text=None,
        citations=[],
        refusal={
            "refuse": True,
            "reason_code": "low_confidence",
            "signals": {"top_score": 0.11},
            "policy_version": "ws3.v0-uncalibrated",
        },
        confidence={"signals": {"top_score": 0.11}, "policy_version": "ws3.v0-uncalibrated"},
        groundedness=None,
        warnings=[],
        model_id=None,
        prompt_version=None,
        generated_at="2026-06-09T00:00:00Z",
        latency_ms=3.0,
    )


def _blocked() -> GroundedAnswer:
    return GroundedAnswer(
        status="blocked_citation_gate",
        query="q",
        course_slug=_SLUG,
        engine="lexical",
        answer_text=None,  # withheld
        citations=[
            Citation(
                chunk_id="chunk_fabricated",
                item_path="module1/page.html",
                section_heading=None,
                module_id="m1",
                page_label="Page",
                anchor_status="span_fabricated",
                source_path=None,
                text_quote=None,
                link_target={},
            )
        ],
        refusal=None,
        confidence={"signals": {"top_score": 0.8}, "policy_version": "ws3.v0-uncalibrated"},
        groundedness=None,
        warnings=[],
        model_id="qwen2.5:7b-instruct-q4_K_M",
        prompt_version="ws3.v1",
        generated_at="2026-06-09T00:00:00Z",
        latency_ms=50.0,
    )


def _patch(monkeypatch, fn):
    """Replace the symbol the CLI command imports inside its body."""
    monkeypatch.setattr(
        "lib.retrieval.grounded_answer.answer_course_question", fn
    )


class TestHappyPath:
    def test_answered_text_rendering(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _answered())
        res = _invoke(tmp_path, "answer-grounded", "what is a node shape?", "--course", _SLUG)
        assert res.exit_code == 0, res.output
        assert "status: answered" in res.output
        assert "A SHACL NodeShape" in res.output
        assert "chunk_001" in res.output
        assert "Node Shapes" in res.output
        assert "resolved_exact" in res.output

    def test_answered_with_warnings_exit_zero(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        ans = _answered(warnings=["contradicted_claim"])
        ans.status = "answered_with_warnings"
        _patch(monkeypatch, lambda *a, **k: ans)
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG)
        assert res.exit_code == 0, res.output
        assert "contradicted_claim" in res.output

    def test_passes_engine_limit_through(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        seen = {}

        def fake(repo_root, slug, query, **kwargs):
            seen.update(kwargs)
            seen["slug"] = slug
            return _answered()

        _patch(monkeypatch, fake)
        res = _invoke(
            tmp_path, "answer-grounded", "q", "--course", _SLUG,
            "--engine", "lexical", "--limit", "5", "--with-groundedness",
        )
        assert res.exit_code == 0, res.output
        assert seen["slug"] == _SLUG
        assert seen["engine"] == "lexical"
        assert seen["limit"] == 5
        assert seen["with_groundedness"] is True
        # A DecisionCapture is always wired by the CLI command.
        assert seen["capture"] is not None


class TestRefusalAndBlocked:
    def test_refused_exit_two(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _refused())
        res = _invoke(tmp_path, "answer-grounded", "off topic?", "--course", _SLUG)
        assert res.exit_code == 2, res.output
        assert "Refused" in res.output
        assert "low_confidence" in res.output
        # No fabricated answer leaks into a refusal.
        assert "NodeShape" not in res.output

    def test_blocked_citation_gate_exit_three(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _blocked())
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG)
        assert res.exit_code == 3, res.output
        assert "Blocked" in res.output
        assert "chunk_fabricated" in res.output
        assert "span_fabricated" in res.output


class TestJson:
    def test_json_full_shape(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _answered())
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG, "--json")
        assert res.exit_code == 0, res.output
        doc = json.loads(res.output)
        assert doc["status"] == "answered"
        assert doc["answer_text"].startswith("A SHACL NodeShape")
        assert doc["citations"][0]["chunk_id"] == "chunk_001"
        assert doc["citations"][0]["anchor_status"] == "resolved_exact"
        assert doc["model_id"] == "qwen2.5:7b-instruct-q4_K_M"
        assert doc["prompt_version"] == "ws3.v1"

    def test_json_refused(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _refused())
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG, "--json")
        assert res.exit_code == 2, res.output
        doc = json.loads(res.output)
        assert doc["status"] == "refused_low_confidence"
        assert doc["answer_text"] is None
        assert doc["refusal"]["reason_code"] == "low_confidence"


class TestTypedErrorExits:
    def test_backend_unavailable_exit_one(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        from lib.retrieval.answer_backend import AnswerBackendUnavailable

        def boom(*a, **k):
            raise AnswerBackendUnavailable("connection refused at localhost:11434")

        _patch(monkeypatch, boom)
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG)
        assert res.exit_code == 1, res.output
        assert "AnswerBackendUnavailable" in res.output
        assert "ED4ALL_ANSWER_PROVIDER" in res.output

    def test_provider_not_local_exit_one(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        from lib.retrieval.answer_backend import AnswerProviderNotLocal

        def boom(*a, **k):
            raise AnswerProviderNotLocal("resolved base_url is not loopback")

        _patch(monkeypatch, boom)
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG)
        assert res.exit_code == 1, res.output
        assert "AnswerProviderNotLocal" in res.output

    def test_semantic_index_missing_exit_one(self, monkeypatch, tmp_path):
        pytest.importorskip("numpy")  # imports vector_index.SemanticIndexMissing (needs [embedding])
        _make_course(tmp_path)
        from LibV2.tools.libv2.vector_index import SemanticIndexMissing

        def boom(*a, **k):
            raise SemanticIndexMissing(
                "no vector index; run `libv2 vector-index build --course ...`"
            )

        _patch(monkeypatch, boom)
        res = _invoke(
            tmp_path, "answer-grounded", "q", "--course", _SLUG, "--engine", "semantic"
        )
        assert res.exit_code == 1, res.output
        assert "SemanticIndexMissing" in res.output
        assert "vector-index build" in res.output


class TestEngineAutoAndValidation:
    def test_auto_resolves_semantic_when_index_present(self, monkeypatch, tmp_path):
        pytest.importorskip("numpy")  # _make_course(with_index) imports vector_index (needs [embedding])
        _make_course(tmp_path, with_index=True)
        seen = {}
        _patch(monkeypatch, lambda *a, **k: (seen.update(k), _answered())[1])
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG, "--engine", "auto")
        assert res.exit_code == 0, res.output
        assert seen["engine"] == "semantic"

    def test_auto_resolves_lexical_without_index(self, monkeypatch, tmp_path):
        _make_course(tmp_path, with_index=False)
        seen = {}
        _patch(monkeypatch, lambda *a, **k: (seen.update(k), _answered())[1])
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG, "--engine", "auto")
        assert res.exit_code == 0, res.output
        assert seen["engine"] == "lexical"

    def test_unknown_engine_rejected(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _answered())
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG, "--engine", "banana")
        assert res.exit_code == 2  # click usage error
        assert "banana" in res.output or "Invalid value" in res.output

    def test_unknown_course_fails(self, monkeypatch, tmp_path):
        (tmp_path / "courses").mkdir(parents=True)
        _patch(monkeypatch, lambda *a, **k: _answered())
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", "no-such")
        assert res.exit_code == 1
        assert "course not found" in res.output


class TestQueryLog:
    def test_log_flag_persists_answer(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _answered())
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG, "--log")
        assert res.exit_code == 0, res.output
        records = list((tmp_path / "courses" / _SLUG / "queries").glob("*.json"))
        # index sidecar + record; at least one record carries the grounded answer.
        answered = [
            r for r in records
            if json.loads(r.read_text()).get("answered_by", "").startswith("grounded:")
        ]
        assert answered, [r.name for r in records]

    def test_no_log_default(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        _patch(monkeypatch, lambda *a, **k: _answered())
        res = _invoke(tmp_path, "answer-grounded", "q", "--course", _SLUG)
        assert res.exit_code == 0, res.output
        assert not (tmp_path / "courses" / _SLUG / "queries").exists()


class TestDelegations:
    def test_answer_eval_forwards_and_returns_exit(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 3

        monkeypatch.setattr("lib.retrieval.grounded_eval.main", fake_main)
        res = _invoke(
            tmp_path, "answer-eval", "--course", _SLUG, "--engine", "lexical",
            "--no-groundedness",
        )
        assert res.exit_code == 3
        assert "--course" in captured["argv"]
        assert _SLUG in captured["argv"]
        assert "--no-groundedness" in captured["argv"]
        assert "--repo-root" in captured["argv"]

    def test_refusal_calibrate_forwards_and_returns_exit(self, monkeypatch, tmp_path):
        _make_course(tmp_path)
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        monkeypatch.setattr("lib.retrieval.refusal.main", fake_main)
        res = _invoke(
            tmp_path, "refusal-calibrate", "--course", _SLUG, "--engine", "semantic",
            "--no-write",
        )
        assert res.exit_code == 0
        assert "--calibrate" in captured["argv"]
        assert "--course" in captured["argv"]
        assert "semantic" in captured["argv"]
        assert "--no-write" in captured["argv"]
