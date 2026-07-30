"""Tests for ``gui.shared_state`` — atomic writes, secret perms, event reads.

No fastapi needed (shared_state is web-free). State is isolated via the
``state_dir`` fixture (``ED4ALL_STATE_RUNS_DIR`` -> tmp_path) so the real
``runtime/state/gui/`` is never written.
"""

from __future__ import annotations

import json
import stat

from gui import shared_state


# --------------------------------------------------------------- secret perms


def test_atomic_write_text_is_owner_only_0600(state_dir):
    """Plaintext-secret files are chmod'd owner-only (no group/other bits)."""
    target = shared_state.gui_state_dir() / "secret.txt"
    shared_state._atomic_write_text(target, "sk-ant-very-secret")
    mode = target.stat().st_mode
    # No group/other permission bits at all.
    assert mode & 0o077 == 0
    assert stat.S_IMODE(mode) == 0o600
    assert target.read_text() == "sk-ant-very-secret"


def test_atomic_write_json_is_owner_only_0600(state_dir):
    target = shared_state.gui_state_dir() / "secret.json"
    shared_state._atomic_write_json(target, {"key": "value"})
    assert target.stat().st_mode & 0o077 == 0
    assert json.loads(target.read_text()) == {"key": "value"}


def test_register_run_record_is_owner_only_0600(state_dir):
    """The run registry (written via the atomic helper) is owner-only too."""
    run_id = shared_state.new_run_id("GUI")
    path = shared_state.register_run({"run_id": run_id, "status": "queued"})
    assert path.stat().st_mode & 0o077 == 0


def test_atomic_write_text_honors_explicit_mode(state_dir):
    """An explicit ``mode`` overrides the 0600 default."""
    target = shared_state.gui_state_dir() / "public.txt"
    shared_state._atomic_write_text(target, "not a secret", mode=0o644)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


# ------------------------------------------------------------------ read_events


def test_read_events_limit_returns_most_recent(state_dir):
    """``limit`` returns only the most-recent N events (after the since filter)."""
    for i in range(5):
        shared_state.append_event("gui", "message", {"i": i})

    limited = shared_state.read_events(limit=2)
    assert len(limited) == 2
    # Most-recent two events (append order, last wins on the tail slice).
    assert [e["payload"]["i"] for e in limited] == [3, 4]


def test_read_events_limit_zero_returns_empty(state_dir):
    shared_state.append_event("gui", "message", {"i": 0})
    assert shared_state.read_events(limit=0) == []


def test_read_events_since_then_limit(state_dir):
    """``since`` filter and ``limit`` slice compose."""
    for i in range(5):
        shared_state.append_event("gui", "message", {"i": i})
    # seq>=2 -> {2,3,4}; limit=2 -> tail {3,4}.
    out = shared_state.read_events(since=2, limit=2)
    assert [e["payload"]["i"] for e in out] == [3, 4]


def test_read_events_dedup_by_seq_last_wins(state_dir):
    """A reused seq (concurrent-writer race) is deduped, keeping the last write."""
    events = shared_state.events_path()
    events.parent.mkdir(parents=True, exist_ok=True)
    # Two records share seq=0 (simulating the documented seq-reuse race);
    # the later line is the authoritative write for that slot.
    lines = [
        {"seq": 0, "source": "gui", "kind": "message", "payload": {"v": "first"}},
        {"seq": 1, "source": "gui", "kind": "message", "payload": {"v": "second"}},
        {"seq": 0, "source": "gui", "kind": "message", "payload": {"v": "first-rewritten"}},
    ]
    events.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    out = shared_state.read_events()
    # One record per distinct seq; last occurrence of seq=0 wins.
    by_seq = {e["seq"]: e["payload"]["v"] for e in out}
    assert by_seq == {0: "first-rewritten", 1: "second"}
    assert len(out) == 2


def test_read_events_skips_malformed_lines(state_dir):
    """A malformed JSONL line never blocks the rest of the feed."""
    events = shared_state.events_path()
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps({"seq": 0, "source": "gui", "kind": "message", "payload": {}})
        + "\nnot-json-garbage\n"
        + json.dumps({"seq": 1, "source": "gui", "kind": "message", "payload": {}})
        + "\n",
        encoding="utf-8",
    )
    out = shared_state.read_events()
    assert [e["seq"] for e in out] == [0, 1]
