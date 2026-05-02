"""Regression: incremental teaching-role checkpoint preserves progress on crash.

Pre-fix bug: a 10-hour curriculum-alignment run on a 295-chunk corpus
that crashed at chunk 290 lost ALL classifications because chunks
were mutated in-memory and only flushed by ``write_corpus`` at the
end. Operators had no on-disk visibility while the run was alive
either.

These tests pin the post-fix contract:

  1. Each chunk's classification is written to a JSONL sidecar with
     ``flush()`` immediately after it lands. ``tail -f`` is live.
  2. A subsequent run that loads the same sidecar skips already-
     classified chunks (no LLM dispatch fires for cached IDs).
  3. Both success ("llm") and failure ("llm_failed") outcomes
     persist, including the ``teaching_role_failure`` metadata.
  4. Malformed sidecar lines are tolerated (best-effort cache).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge import align_chunks  # noqa: E402


def _chunk(idx: int, text: str = "x" * 50) -> dict:
    return {
        "id": f"c{idx:05d}",
        "text": text,
        "_position": idx,
        "concept_tags": [],
        "prereq_concepts": [],
        "chunk_type": "content",
        "source": {"resource_type": "html"},
    }


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Incremental write
# ---------------------------------------------------------------------------


def test_checkpoint_appended_after_each_chunk(tmp_path):
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    provider = MagicMock()
    provider.classify_teaching_role.side_effect = [
        "introduce", "elaborate", "reinforce",
    ]
    cp = tmp_path / "corpus" / ".teaching_role_checkpoint.jsonl"

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider,
        verbose=False, checkpoint_path=cp,
    )

    assert cp.exists()
    rows = _read_jsonl(cp)
    assert [r["id"] for r in rows] == ["c00001", "c00002", "c00003"]
    assert [r["teaching_role"] for r in rows] == [
        "introduce", "elaborate", "reinforce",
    ]
    assert all(r["teaching_role_source"] == "llm" for r in rows)


def test_checkpoint_persists_failure_metadata(tmp_path):
    chunks = [_chunk(1), _chunk(2)]
    provider = MagicMock()
    provider.classify_teaching_role.side_effect = [
        "introduce", RuntimeError("boom"),
    ]
    cp = tmp_path / ".teaching_role_checkpoint.jsonl"

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider,
        verbose=False, checkpoint_path=cp,
    )

    rows = _read_jsonl(cp)
    assert rows[0]["teaching_role_source"] == "llm"
    assert "teaching_role_failure" not in rows[0]
    assert rows[1]["teaching_role_source"] == "llm_failed"
    assert rows[1]["teaching_role_failure"] == {
        "error_class": "RuntimeError",
        "error_message": "boom",
    }


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_skips_cached_chunks_and_does_not_call_provider(tmp_path):
    """Simulate a crash mid-run: sidecar has classifications for
    chunks 1 and 2, then the process restarts. The provider must NOT
    be called for chunks 1 or 2 — only chunk 3 (uncached) gets a
    fresh classification."""
    cp = tmp_path / ".teaching_role_checkpoint.jsonl"
    cp.write_text(
        json.dumps({"id": "c00001", "teaching_role": "introduce",
                    "teaching_role_source": "llm"}) + "\n"
        + json.dumps({"id": "c00002", "teaching_role": "elaborate",
                      "teaching_role_source": "llm"}) + "\n"
    )

    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    provider = MagicMock()
    provider.classify_teaching_role.return_value = "reinforce"

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider,
        verbose=False, checkpoint_path=cp,
    )

    # Provider was called exactly once — for c00003 only.
    assert provider.classify_teaching_role.call_count == 1
    called_kwargs = provider.classify_teaching_role.call_args.kwargs
    assert called_kwargs["chunk_id"] == "c00003"

    # Cached values applied to chunks 1 and 2.
    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[0]["teaching_role_source"] == "llm"
    assert chunks[1]["teaching_role"] == "elaborate"
    assert chunks[1]["teaching_role_source"] == "llm"
    # Chunk 3 got the live classification.
    assert chunks[2]["teaching_role"] == "reinforce"


def test_resume_replays_failure_metadata_from_checkpoint(tmp_path):
    """A chunk that failed on a prior run should retain its llm_failed
    annotation on resume — we don't re-dispatch failures hoping for
    different luck this time. Operator can clear the sidecar manually
    to force a retry."""
    cp = tmp_path / ".teaching_role_checkpoint.jsonl"
    cp.write_text(
        json.dumps({
            "id": "c00001",
            "teaching_role": "elaborate",
            "teaching_role_source": "llm_failed",
            "teaching_role_failure": {
                "error_class": "TimeoutError", "error_message": "504",
            },
        }) + "\n"
    )

    chunks = [_chunk(1)]
    provider = MagicMock()

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider,
        verbose=False, checkpoint_path=cp,
    )

    provider.classify_teaching_role.assert_not_called()
    assert chunks[0]["teaching_role"] == "elaborate"
    assert chunks[0]["teaching_role_source"] == "llm_failed"
    assert chunks[0]["teaching_role_failure"]["error_class"] == "TimeoutError"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_malformed_checkpoint_lines_are_tolerated(tmp_path):
    cp = tmp_path / ".teaching_role_checkpoint.jsonl"
    cp.write_text(
        "this is not valid json\n"
        + json.dumps({"id": "c00001", "teaching_role": "introduce",
                      "teaching_role_source": "llm"}) + "\n"
        + "{ broken json\n"
    )

    chunks = [_chunk(1), _chunk(2)]
    provider = MagicMock()
    provider.classify_teaching_role.return_value = "elaborate"

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider,
        verbose=False, checkpoint_path=cp,
    )

    # c00001 was salvaged from the checkpoint; c00002 was classified live.
    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[1]["teaching_role"] == "elaborate"
    assert provider.classify_teaching_role.call_count == 1


def test_no_checkpoint_path_is_a_noop(tmp_path):
    """Backward-compat: callers that don't pass checkpoint_path get
    the legacy behavior — no sidecar file created."""
    chunks = [_chunk(1)]
    provider = MagicMock()
    provider.classify_teaching_role.return_value = "introduce"

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider, verbose=False,
    )

    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[0]["teaching_role_source"] == "llm"
    assert not list(tmp_path.iterdir())


def test_checkpoint_creates_parent_directories(tmp_path):
    """When the run starts before chunks.jsonl is written (e.g. fresh
    course directory tree), the corpus/ subdir may not exist yet. The
    helper must mkdir parents so the sidecar write doesn't fail."""
    cp = tmp_path / "deeply" / "nested" / "corpus" / ".checkpoint.jsonl"
    chunks = [_chunk(1)]
    provider = MagicMock()
    provider.classify_teaching_role.return_value = "introduce"

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider,
        verbose=False, checkpoint_path=cp,
    )

    assert cp.exists()
