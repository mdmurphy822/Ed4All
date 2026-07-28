from __future__ import annotations

from MCP.core.workflow_runner import WorkflowRunner


def test_carried_required_seat_is_deep_probed(monkeypatch, tmp_path) -> None:
    from lib import vllm_container_lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle, "all_registered_seat_names", lambda: {"spark-super"}
    )
    monkeypatch.setattr(
        lifecycle, "resolve_seat_base_url",
        lambda seat: "http://localhost:8123",
    )
    monkeypatch.setattr(lifecycle, "stop_seat", lambda seat: True)
    seen = []

    class Result:
        ok = True
        reason = "already_serving"
        recreated = False
        load_seconds = 0.0

    monkeypatch.setattr(
        lifecycle,
        "start_seat_coherent",
        lambda seat, **kwargs: seen.append(seat) or Result(),
    )

    result = WorkflowRunner._apply_seat_schedule_blocking(
        "training_synthesis",
        {"spark-super"},
        {"spark-super"},
        tmp_path,
    )

    assert result == {"spark-super"}
    assert seen == ["spark-super"]
