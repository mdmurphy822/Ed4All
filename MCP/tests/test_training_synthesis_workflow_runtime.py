"""Runtime contract for long local-teacher training synthesis."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _training_synthesis_phase() -> dict:
    config = yaml.safe_load(
        (PROJECT_ROOT / "config" / "workflows.yaml").read_text(
            encoding="utf-8",
        )
    )
    phases = config["workflows"]["textbook_to_course"]["phases"]
    return next(phase for phase in phases if phase["name"] == "training_synthesis")


def test_training_synthesis_allows_multiday_local_workload() -> None:
    phase = _training_synthesis_phase()

    assert phase["timeout_minutes"] >= 7 * 24 * 60
    assert phase["batch_timeout_minutes"] == phase["timeout_minutes"]
    assert phase["max_concurrent"] == 1
    assert phase["parallel"] is False


def test_training_synthesis_retains_stop_safe_single_writer_shape() -> None:
    phase = _training_synthesis_phase()

    assert phase["depends_on"] == [
        "trainforge_assessment",
        "imscc_chunking",
    ]
    assert phase["optional"] is True
