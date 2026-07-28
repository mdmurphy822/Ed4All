from __future__ import annotations

import threading
import time
from pathlib import Path

from lib import vllm_container_lifecycle as lifecycle


def test_recovery_requires_two_post_recreate_generations(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        lifecycle, "parse_seat_registry",
        lambda value=None: {"spark-super": "http://localhost:8123"},
    )
    monkeypatch.setattr(lifecycle, "_engine_failure_signals", lambda seat: ("Hang detected",))
    answers = iter([False, True, True])
    monkeypatch.setattr(
        lifecycle, "coherence_probe",
        lambda *a, **k: next(answers),
    )
    recreates = []
    monkeypatch.setattr(
        lifecycle, "recreate_seat",
        lambda seat, **kwargs: recreates.append(seat) or 2.5,
    )
    monkeypatch.setattr(lifecycle, "STATE_PATH", tmp_path)

    result = lifecycle.recover_seat_for_base_url(
        "http://localhost:8123/v1",
        run_dir=tmp_path / "run",
        reason="test",
    )

    assert result.ok is True
    assert result.recreated is True
    assert result.engine_signals == ("Hang detected",)
    assert recreates == ["spark-super"]


def test_recovery_is_singleflight_across_threads(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        lifecycle, "parse_seat_registry",
        lambda value=None: {"spark-super": "http://localhost:8123"},
    )
    monkeypatch.setattr(lifecycle, "_engine_failure_signals", lambda seat: ())
    healthy = threading.Event()
    monkeypatch.setattr(
        lifecycle, "coherence_probe",
        lambda *a, **k: healthy.is_set(),
    )
    recreates = 0

    def recreate(*args, **kwargs):
        nonlocal recreates
        recreates += 1
        time.sleep(0.05)
        healthy.set()
        return 1.0

    monkeypatch.setattr(lifecycle, "recreate_seat", recreate)
    monkeypatch.setattr(lifecycle, "STATE_PATH", tmp_path)
    results = []

    def recover():
        results.append(lifecycle.recover_seat_for_base_url(
            "http://localhost:8123/v1", reason="test"
        ))

    threads = [threading.Thread(target=recover) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert recreates == 1
    assert all(result.ok for result in results)
    assert {result.reason for result in results} == {
        "recovered", "already_recovered"
    }
