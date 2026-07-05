"""concept_extraction Stage-3 per-window resume-checkpoint sidecar invariants.

A killed / timed-out ``concept_extraction`` phase must not lose the expensive
per-window ``synthesize_concepts`` 7B calls on resume. Regression driver: on
2026-07-05 the phase hit its batch-timeout at ~98% (≈4,060 of ≈4,100 local-7B
calls) and lost ALL work because it had no resume sidecar. These tests exercise
the sidecar wired into ``_run_concept_extraction``'s Stage-3 dispatch loop plus
its module-level helpers, modeled on
``test_pipeline_tools_stage2_checkpoint.py``.

Hermetic: a fake ``TextbookSynthesisProvider`` (no GPU / ollama / network) is
monkeypatched into the provider module; the phase is driven with an empty
staging dir (no chunks → one chapter_text window per chapter) so the concept
call is a deterministic function of the (fake) provider. No hardcoded course
slug — the course name is a test constant and the slug is derived from it.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402


# ---------------------------------------------------------------------------
# Fake provider (constructed internally by the handler as
# ``TextbookSynthesisProvider(capture=capture)``, so per-run knobs + the
# call log live at CLASS level and are reset by the autouse fixture).
# ---------------------------------------------------------------------------
class _FakeConceptProvider:
    model: str = "test-model-v1"
    batch_size: int = 10
    calls: List[str] = []
    #: When set, each ``synthesize_concepts`` records how many checkpoint
    #: records were ALREADY on disk when it fired (the batch-size-1 observer).
    sidecar_path: Optional[Path] = None
    records_seen: List[int] = []

    def __init__(self, *, capture: Any = None) -> None:
        self._model = type(self).model
        self._provider = "local"
        self._max_tokens = 4096

    def batch_chapters(self, items, batch_size: int = 10):
        bs = type(self).batch_size
        return [items[i:i + bs] for i in range(0, len(items), bs)]

    def synthesize_concepts(self, spec, *, course_name):
        cls = type(self)
        cid = str(spec.get("id") or "")
        widx = spec.get("window_index")
        if cls.sidecar_path is not None:
            cls.records_seen.append(
                len(pt._load_concept_windows_checkpoint(cls.sidecar_path))
            )
        cls.calls.append(f"{cid}#w{widx}")
        return {
            "concepts": [
                {
                    "canonical": f"Concept {cid} window {widx}",
                    "aliases": [f"alias {cid}"],
                    "chapter_ids": [cid],
                    "definition_hint": f"hint for {cid}",
                }
            ]
        }


@pytest.fixture(autouse=True)
def _reset_fake_provider():
    _FakeConceptProvider.model = "test-model-v1"
    _FakeConceptProvider.batch_size = 10
    _FakeConceptProvider.calls = []
    _FakeConceptProvider.sidecar_path = None
    _FakeConceptProvider.records_seen = []
    yield


def _sidecar_for(tmp_path: Path, course_name: str) -> Path:
    slug = course_name.lower().replace("_", "-").replace(" ", "-")
    return (
        tmp_path / "libv2" / "courses" / slug / "concept_graph"
        / pt._CONCEPT_EXTRACTION_CHECKPOINT_NAME
    )


def _run_phase(
    tmp_path: Path,
    monkeypatch,
    *,
    mutate_text: bool = False,
    course_name: str = "CONCEPT_CP",
    model: str = "test-model-v1",
    batch_size: int = 10,
    observe: bool = False,
) -> Dict[str, Any]:
    """Drive ``run_concept_extraction`` through Stage-3, hermetically."""
    import Courseforge.generators._textbook_synthesis_provider as _tsp

    monkeypatch.setattr(_tsp, "TextbookSynthesisProvider", _FakeConceptProvider)
    # Stage-3 only runs when TEXTBOOK_SYNTHESIS_PROVIDER is set.
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", "local")

    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)  # empty → no chunks → chapter_text windows

    suffix = " EDITED" if mutate_text else ""
    ts_path = tmp_path / "textbook_structure.json"
    ts_path.write_text(
        json.dumps({
            "chapters": [
                {"id": "ch1", "chapter_text": "Alpha chapter body." + suffix},
                {"id": "ch2", "chapter_text": "Bravo chapter body." + suffix},
            ]
        }),
        encoding="utf-8",
    )

    libv2 = tmp_path / "libv2"
    _FakeConceptProvider.model = model
    _FakeConceptProvider.batch_size = batch_size
    if observe:
        _FakeConceptProvider.sidecar_path = _sidecar_for(tmp_path, course_name)

    registry = pt._build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(tool(
        project_id="",
        course_name=course_name,
        staging_dir=str(staging),
        textbook_structure_path=str(ts_path),
        libv2_root=str(libv2),
    ))
    return json.loads(result)


# ---------------------------------------------------------------------------
# (a) sidecar written incrementally DURING the loop (batch-size-1 observer)
# ---------------------------------------------------------------------------
def test_sidecar_written_incrementally_per_batch(tmp_path, monkeypatch):
    """Records land AS EACH BATCH COMPLETES, not after the full dispatch loop.

    batch_size=1 → 2 sequential batches; each call records how many
    checkpoint records were ALREADY on disk when it started. Call N must see
    the N-1 previously-completed windows already persisted.
    """
    payload = _run_phase(
        tmp_path, monkeypatch, batch_size=1, observe=True,
    )
    assert payload["success"] is True, payload
    assert _FakeConceptProvider.calls == ["ch1#w0", "ch2#w0"]
    assert _FakeConceptProvider.records_seen == [0, 1]

    sidecar = _sidecar_for(tmp_path, "CONCEPT_CP")
    assert sidecar.exists()
    records = pt._load_concept_windows_checkpoint(sidecar)
    assert len(records) == 2
    for rec in records.values():
        assert rec["schema_version"] == pt._STAGE2_CHECKPOINT_SCHEMA_VERSION
        assert isinstance(rec["fingerprint"], str) and rec["fingerprint"]
        concepts = rec["concepts"]
        assert concepts and isinstance(concepts, list)
        assert concepts[0]["canonical"].startswith("Concept ")
    assert payload["concept_windows_reused_from_checkpoint"] == 0


# ---------------------------------------------------------------------------
# (b) resume skips matching windows — zero re-dispatch, byte-equivalent output
# ---------------------------------------------------------------------------
def test_resume_skips_matching_windows_byte_equivalent(tmp_path, monkeypatch):
    fresh = _run_phase(tmp_path, monkeypatch)
    assert _FakeConceptProvider.calls == ["ch1#w0", "ch2#w0"]
    assert fresh["concept_windows_reused_from_checkpoint"] == 0

    # Capture the persisted Stage-3 vocabulary artifact BEFORE the resume
    # overwrites it. The vocabulary carries no run-scoped fields (unlike the
    # graph, which embeds a per-run run_id in node/edge provenance and so has
    # a run-varying sha256), so it is the clean byte-equivalence surface.
    vocab_path = Path(fresh["domain_concept_vocabulary_path"])
    fresh_vocab_bytes = vocab_path.read_bytes()

    # Resume against the populated sidecar: every window fingerprint matches,
    # so ZERO provider dispatches — and the concept output is byte-equivalent.
    _FakeConceptProvider.calls = []
    resume = _run_phase(tmp_path, monkeypatch)
    assert _FakeConceptProvider.calls == []  # nothing re-dispatched
    assert resume["concept_windows_reused_from_checkpoint"] == 2

    # The synthesized concept output is byte-equivalent to the uninterrupted
    # run — same vocabulary bytes, same concept count, same graph shape.
    assert vocab_path.read_bytes() == fresh_vocab_bytes
    assert resume["domain_concept_count"] == fresh["domain_concept_count"]
    assert resume["domain_concept_count"] == 2
    assert resume["node_count"] == fresh["node_count"]
    assert resume["edge_count"] == fresh["edge_count"]


def test_partial_resume_dispatches_only_missing(tmp_path, monkeypatch):
    _run_phase(tmp_path, monkeypatch)
    assert _FakeConceptProvider.calls == ["ch1#w0", "ch2#w0"]

    # Simulate a partial run: drop one window record from the sidecar.
    sidecar = _sidecar_for(tmp_path, "CONCEPT_CP")
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    sidecar.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    _FakeConceptProvider.calls = []
    resume = _run_phase(tmp_path, monkeypatch)
    assert len(_FakeConceptProvider.calls) == 1  # only the dropped window re-ran
    assert resume["concept_windows_reused_from_checkpoint"] == 1


# ---------------------------------------------------------------------------
# (c) fingerprint mismatch re-runs
# ---------------------------------------------------------------------------
def test_fingerprint_mismatch_model_reruns(tmp_path, monkeypatch):
    _run_phase(tmp_path, monkeypatch, model="test-model-v1")

    # A different model id changes every fingerprint → full re-dispatch.
    _FakeConceptProvider.calls = []
    resume = _run_phase(tmp_path, monkeypatch, model="test-model-v2")
    assert _FakeConceptProvider.calls == ["ch1#w0", "ch2#w0"]
    assert resume["concept_windows_reused_from_checkpoint"] == 0


def test_fingerprint_mismatch_changed_chapter_text_reruns(tmp_path, monkeypatch):
    _run_phase(tmp_path, monkeypatch)

    # Same windows/keys, but each chapter's text changed → fingerprints
    # mismatch → full re-dispatch.
    _FakeConceptProvider.calls = []
    resume = _run_phase(tmp_path, monkeypatch, mutate_text=True)
    assert _FakeConceptProvider.calls == ["ch1#w0", "ch2#w0"]
    assert resume["concept_windows_reused_from_checkpoint"] == 0


# ---------------------------------------------------------------------------
# (d) flag opt-out disables both write and reuse
# ---------------------------------------------------------------------------
def test_opt_out_disables_write(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", "0")
    payload = _run_phase(tmp_path, monkeypatch)
    assert _FakeConceptProvider.calls == ["ch1#w0", "ch2#w0"]
    sidecar = _sidecar_for(tmp_path, "CONCEPT_CP")
    assert not sidecar.exists()  # no sidecar written when disabled
    assert payload["concept_windows_reused_from_checkpoint"] == 0


def test_opt_out_disables_reuse(tmp_path, monkeypatch):
    # First: a normal enabled run populates the sidecar.
    _run_phase(tmp_path, monkeypatch)
    sidecar = _sidecar_for(tmp_path, "CONCEPT_CP")
    assert sidecar.exists()

    # Then: with the flag off, a resume ignores the sidecar and re-dispatches.
    monkeypatch.setenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", "0")
    _FakeConceptProvider.calls = []
    resume = _run_phase(tmp_path, monkeypatch)
    assert _FakeConceptProvider.calls == ["ch1#w0", "ch2#w0"]
    assert resume["concept_windows_reused_from_checkpoint"] == 0


# ---------------------------------------------------------------------------
# (e) torn trailing line tolerated
# ---------------------------------------------------------------------------
def test_torn_trailing_line_tolerated(tmp_path, monkeypatch):
    _run_phase(tmp_path, monkeypatch)
    sidecar = _sidecar_for(tmp_path, "CONCEPT_CP")

    # Simulate a crash mid-append: a half-written final JSON line.
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "chapter_id": "ch2", "window_ind')

    # Loader skips the torn line without raising; the 2 valid records survive.
    records = pt._load_concept_windows_checkpoint(sidecar)
    assert len(records) == 2

    # A resume still fully reuses (torn line ignored, no re-dispatch).
    _FakeConceptProvider.calls = []
    resume = _run_phase(tmp_path, monkeypatch)
    assert _FakeConceptProvider.calls == []
    assert resume["concept_windows_reused_from_checkpoint"] == 2


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------
def test_fingerprint_is_deterministic_and_sensitive():
    base = dict(
        chapter_id="ch1",
        window_index=0,
        chunk_ids=["c1", "c2"],
        chapter_text="alpha body",
        course_name="MATH_101",
        model="m1",
        num_ctx=4096,
        max_tokens=4096,
        system_prompt="sys",
    )
    fp = pt._concept_window_fingerprint(**base)
    assert fp == pt._concept_window_fingerprint(**base)  # deterministic
    # chunk_ids order does not matter (sorted), but membership does.
    reordered = dict(base, chunk_ids=["c2", "c1"])
    assert pt._concept_window_fingerprint(**reordered) == fp
    # Each input axis flips the fingerprint.
    for key, val in [
        ("chapter_id", "ch2"),
        ("window_index", 1),
        ("chunk_ids", ["c1", "c3"]),
        ("chapter_text", "alpha body EDITED"),
        ("course_name", "MATH_102"),
        ("model", "m2"),
        ("num_ctx", 8192),
        ("max_tokens", 2048),
        ("system_prompt", "sys2"),
    ]:
        assert pt._concept_window_fingerprint(**dict(base, **{key: val})) != fp


def test_checkpoint_enabled_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    assert pt._concept_extraction_checkpoint_enabled() is True  # default ON
    for tok in ("garbage", "1", "true", "yes", "on"):
        monkeypatch.setenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", tok)
        assert pt._concept_extraction_checkpoint_enabled() is True
    for tok in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", tok)
        assert pt._concept_extraction_checkpoint_enabled() is False


def test_site_flag_beats_family_flag(monkeypatch):
    # Family off, site on → on (site flag, when set, is authoritative).
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    monkeypatch.setenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", "1")
    assert pt._concept_extraction_checkpoint_enabled() is True
    # Family on, site off → off.
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "1")
    monkeypatch.setenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", "0")
    assert pt._concept_extraction_checkpoint_enabled() is False
    # Site unset → falls through to family.
    monkeypatch.delenv("ED4ALL_CONCEPT_EXTRACTION_CHECKPOINT", raising=False)
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    assert pt._concept_extraction_checkpoint_enabled() is False


def test_loader_missing_file_returns_empty(tmp_path):
    assert pt._load_concept_windows_checkpoint(None) == {}
    assert pt._load_concept_windows_checkpoint(tmp_path / "nope.jsonl") == {}
