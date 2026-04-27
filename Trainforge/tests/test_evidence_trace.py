"""Wave 103 - tests for the evidence-trace writer + classifier.

Exercises:
* TraceWriter writes parseable JSONL with all 11 fields per row.
* extract_citations pulls bracketed chunk-ids from free text.
* classify_failure_mode maps the four canonical failure conditions to
  the right label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_EXPECTED_FIELDS = {
    "probe_id", "setup", "retrieval_method", "prompt", "retrieved_chunks",
    "ground_truth_chunk_id", "retrieved_at_top_k", "model_output",
    "extracted_citations", "cited_correct_chunk", "answer_correct",
    "failure_mode",
}


def test_trace_writer_emits_jsonl_with_all_fields(tmp_path):
    from Trainforge.eval.evidence_trace import (
        EvidenceTrace, TraceWriter, load_traces,
    )

    out = tmp_path / "eval_traces.jsonl"
    with TraceWriter(out) as writer:
        for i in range(3):
            writer.append(EvidenceTrace(
                probe_id=f"p{i}",
                setup="adapter+rag",
                retrieval_method="bm25",
                prompt=f"What is concept {i}?",
                retrieved_chunks=[{"chunk_id": f"c{i}", "score": 0.9}],
                ground_truth_chunk_id=f"c{i}",
                retrieved_at_top_k=True,
                model_output=f"Answer [c{i}]",
                extracted_citations=[f"c{i}"],
                cited_correct_chunk=True,
                answer_correct=True,
                failure_mode="none",
            ))
    raw_lines = [
        line for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(raw_lines) == 3
    for line in raw_lines:
        row = json.loads(line)
        # All 12 documented fields land on every row (we count probe_id
        # under the "11 fields per row" header from the brief plus
        # probe_id; brief itemises the trace as 11 + probe_id).
        # Per the brief's docstring the row carries 11 fields - verify
        # the canonical set is present.
        assert _EXPECTED_FIELDS.issubset(row.keys())
    # round-trip read
    rows = load_traces(out)
    assert len(rows) == 3
    assert rows[0].probe_id == "p0"


def test_extract_citations_pulls_bracketed_ids():
    from Trainforge.eval.evidence_trace import extract_citations

    out = extract_citations("answer with [chunk-1] and [chunk_2] cites.")
    assert out == ["chunk-1", "chunk_2"]
    # Pure-numeric bracketed footnote refs should NOT be captured (we
    # require at least one alphabetic char).
    assert extract_citations("see [42]") == []
    # Empty / malformed
    assert extract_citations("") == []
    assert extract_citations("no brackets here") == []


def test_classify_failure_mode_clean_pass():
    from Trainforge.eval.evidence_trace import classify_failure_mode

    label = classify_failure_mode(
        retrieved_at_top_k=True,
        cited_correct_chunk=True,
        answer_correct=True,
        model_used_context=True,
    )
    assert label == "none"


def test_classify_failure_mode_retrieval_miss():
    from Trainforge.eval.evidence_trace import classify_failure_mode

    label = classify_failure_mode(
        retrieved_at_top_k=False,
        cited_correct_chunk=False,
        answer_correct=False,
        model_used_context=False,
    )
    assert label == "retrieval_miss"


def test_classify_failure_mode_retrieval_hit_no_cite():
    from Trainforge.eval.evidence_trace import classify_failure_mode

    label = classify_failure_mode(
        retrieved_at_top_k=True,
        cited_correct_chunk=False,
        answer_correct=False,
        model_used_context=True,  # used context (cited some chunk) but not GT
    )
    assert label == "retrieval_hit_no_cite"


def test_classify_failure_mode_model_ignored_context():
    from Trainforge.eval.evidence_trace import classify_failure_mode

    label = classify_failure_mode(
        retrieved_at_top_k=True,
        cited_correct_chunk=False,
        answer_correct=False,
        model_used_context=False,  # no citation, no chunk references at all
    )
    assert label == "model_ignored_context"


def test_trace_writer_close_is_idempotent(tmp_path):
    from Trainforge.eval.evidence_trace import EvidenceTrace, TraceWriter

    out = tmp_path / "eval_traces.jsonl"
    writer = TraceWriter(out)
    writer.append(EvidenceTrace(
        probe_id="p", setup="base", retrieval_method=None,
        prompt="q", model_output="a",
    ))
    writer.close()
    writer.close()  # second close should not raise
    assert out.exists()
