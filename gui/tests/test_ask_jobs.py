"""Ask-job store + single-lane runner lifecycle tests (``gui.services.ask_jobs``).

The answer backend is stubbed everywhere (no model, no LibV2 read): the worker's
answer callable is monkeypatched to a deterministic stub via the module-level
``_answer_fn`` seam, mirroring how the router/sync-path tests stub
``answer_service.ask``. State is isolated into ``tmp_path`` by the shared
``state_dir`` fixture (``ED4ALL_STATE_RUNS_DIR`` redirect).

Covers: submit→pending→done, error path, queue position, persistence across a
"service re-instantiation" (fresh read off disk after the worker finishes), and
the TTL sweep.
"""

from __future__ import annotations

import time

import pytest

from gui.services import ask_jobs


def _grounded(status="answered", answer_text="Velocity is a vector.", slug="phys-101"):
    return {
        "status": status,
        "query": "what is velocity?",
        "course_slug": slug,
        "engine": "lexical",
        "answer_text": answer_text,
        "citations": [],
        "refusal": None,
        "confidence": {},
        "groundedness": None,
        "warnings": [],
        "model_id": "qwen2.5:7b",
        "prompt_version": "v1",
        "generated_at": "2026-06-10T00:00:00Z",
        "latency_ms": 12.0,
    }


@pytest.fixture(autouse=True)
def _reset_answer_fn():
    """Reset the answer-fn seam after each test so stubs never leak."""
    original = ask_jobs._answer_fn
    try:
        yield
    finally:
        ask_jobs._answer_fn = original


def _await_status(ask_id, want, timeout=5.0):
    """Poll the on-disk job until it reaches ``want`` (or a terminal state)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = ask_jobs.read_job(ask_id)
        if rec and rec.get("status") == want:
            return rec
        if rec and rec.get("status") in (ask_jobs.STATUS_DONE, ask_jobs.STATUS_ERROR):
            return rec
        time.sleep(0.02)
    return ask_jobs.read_job(ask_id)


# ------------------------------------------------------------------- submit → done


def test_submit_returns_pending_then_runs_to_done(state_dir, monkeypatch):
    monkeypatch.setattr(ask_jobs, "_answer_fn", lambda slug, q, e: _grounded())
    rec = ask_jobs.submit("phys-101", "what is velocity?")
    assert rec["status"] == ask_jobs.STATUS_PENDING
    assert rec["ask_id"].startswith("ASK-")
    assert "queue_position" in rec

    done = _await_status(rec["ask_id"], ask_jobs.STATUS_DONE)
    assert done["status"] == ask_jobs.STATUS_DONE
    assert done["answer"]["status"] == "answered"
    assert done["answer"]["answer_text"] == "Velocity is a vector."
    assert "elapsed_ms" in done and done["elapsed_ms"] >= 0


def test_engine_is_threaded_to_answer_fn(state_dir, monkeypatch):
    seen = {}

    def stub(slug, query, engine):
        seen["engine"] = engine
        return _grounded()

    monkeypatch.setattr(ask_jobs, "_answer_fn", stub)
    rec = ask_jobs.submit("phys-101", "q", engine="semantic")
    _await_status(rec["ask_id"], ask_jobs.STATUS_DONE)
    assert seen["engine"] == "semantic"


# ------------------------------------------------------------------- error path


def test_error_path_persists_typed_error(state_dir, monkeypatch):
    class AnswerBackendUnavailable(Exception):
        pass

    def boom(slug, query, engine):
        raise AnswerBackendUnavailable("ollama down")

    monkeypatch.setattr(ask_jobs, "_answer_fn", boom)
    rec = ask_jobs.submit("phys-101", "q")
    err = _await_status(rec["ask_id"], ask_jobs.STATUS_ERROR)
    assert err["status"] == ask_jobs.STATUS_ERROR
    assert err["error"] == "AnswerBackendUnavailable"
    assert "ollama down" in err["detail"]
    assert "answer" not in err  # no fabricated answer on the error arm


# ------------------------------------------------------------------- queue position


def test_queue_position_reflects_pending_order(state_dir, monkeypatch):
    # A blocking answer fn so jobs pile up in the queue while we inspect order.
    release = {"go": False}

    def slow(slug, query, engine):
        while not release["go"]:
            time.sleep(0.01)
        return _grounded()

    monkeypatch.setattr(ask_jobs, "_answer_fn", slow)
    r1 = ask_jobs.submit("phys-101", "q1")
    r2 = ask_jobs.submit("phys-101", "q2")
    r3 = ask_jobs.submit("phys-101", "q3")

    # r1 may be running (pos 0); r2/r3 are queued behind it in submit order.
    pos2 = ask_jobs.status(r2["ask_id"])["queue_position"]
    pos3 = ask_jobs.status(r3["ask_id"])["queue_position"]
    assert pos2 < pos3, (pos2, pos3)
    assert ask_jobs.status(r1["ask_id"])["queue_position"] == 0

    release["go"] = True
    for r in (r1, r2, r3):
        _await_status(r["ask_id"], ask_jobs.STATUS_DONE)


# ----------------------------------------------------- persistence / re-instantiation


def test_finished_answer_survives_fresh_read(state_dir, monkeypatch):
    """A finished job is served straight off disk (no in-memory cache).

    Simulates a uvicorn restart: after the worker persists ``done``, a brand-new
    read (the only source of truth) still returns the answer.
    """
    monkeypatch.setattr(ask_jobs, "_answer_fn", lambda slug, q, e: _grounded())
    rec = ask_jobs.submit("phys-101", "q")
    _await_status(rec["ask_id"], ask_jobs.STATUS_DONE)

    # Fresh read — no worker, no queue, just the persisted file.
    reread = ask_jobs.read_job(rec["ask_id"])
    assert reread["status"] == ask_jobs.STATUS_DONE
    assert reread["answer"]["answer_text"] == "Velocity is a vector."
    # The job file physically exists under state/gui/ask_jobs/.
    assert (ask_jobs.ask_jobs_dir() / f"{rec['ask_id']}.json").is_file()


def test_status_unknown_id_is_none(state_dir):
    assert ask_jobs.status("ASK-nope-nope") is None


# ------------------------------------------------------------------- TTL sweep


def test_sweep_removes_expired_jobs(state_dir, monkeypatch):
    monkeypatch.setattr(ask_jobs, "_answer_fn", lambda slug, q, e: _grounded())
    rec = ask_jobs.submit("phys-101", "q")
    _await_status(rec["ask_id"], ask_jobs.STATUS_DONE)
    path = ask_jobs.ask_jobs_dir() / f"{rec['ask_id']}.json"
    assert path.is_file()

    # Sweep with a far-future "now" so the (fresh) file is older than the TTL.
    removed = ask_jobs.sweep(ttl_seconds=1, now=time.time() + 10_000)
    assert removed == 1
    assert not path.exists()


def test_sweep_keeps_fresh_jobs(state_dir, monkeypatch):
    monkeypatch.setattr(ask_jobs, "_answer_fn", lambda slug, q, e: _grounded())
    rec = ask_jobs.submit("phys-101", "q")
    _await_status(rec["ask_id"], ask_jobs.STATUS_DONE)
    removed = ask_jobs.sweep(ttl_seconds=ask_jobs.DEFAULT_TTL_SECONDS)
    assert removed == 0
    assert (ask_jobs.ask_jobs_dir() / f"{rec['ask_id']}.json").is_file()
