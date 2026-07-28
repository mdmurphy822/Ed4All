"""Fail-before-dispatch coverage for staged local model identity."""

from pathlib import Path

import pytest

import Trainforge.synthesize_training as synthesis


SUPER_SNAPSHOT = "4f0cf9f6c9f84c0daac28fb98f1a5730"


def test_exact_model_identity_is_accepted() -> None:
    requested_urls = []

    def get_json(url: str):
        requested_urls.append(url)
        return {"object": "list", "data": [{"id": SUPER_SNAPSHOT}]}

    assert synthesis._preflight_local_staged_model_identity(
        base_url="http://127.0.0.1:8000/v1/",
        local_model=SUPER_SNAPSHOT,
        get_json=get_json,
    ) == SUPER_SNAPSHOT
    assert requested_urls == ["http://127.0.0.1:8000/v1/models"]


def test_missing_canonical_model_fails_before_models_request() -> None:
    called = False

    def get_json(_url: str):
        nonlocal called
        called = True
        return {"data": []}

    with pytest.raises(RuntimeError, match="LOCAL_SYNTHESIS_MODEL must be set"):
        synthesis._preflight_local_staged_model_identity(
            base_url="http://127.0.0.1:8000/v1",
            local_model=" ",
            generic_model=SUPER_SNAPSHOT,
            get_json=get_json,
        )
    assert called is False


def test_legacy_generic_model_cannot_conflict_with_canonical_model() -> None:
    with pytest.raises(RuntimeError, match="conflicting synthesis model selectors"):
        synthesis._preflight_local_staged_model_identity(
            base_url="http://127.0.0.1:8000/v1",
            local_model=SUPER_SNAPSHOT,
            generic_model="nemotron-3-nano-30b-a3b",
            get_json=lambda _url: pytest.fail("must fail before HTTP"),
        )


def test_wrong_served_model_is_fatal() -> None:
    with pytest.raises(RuntimeError, match="is not served"):
        synthesis._preflight_local_staged_model_identity(
            base_url="http://127.0.0.1:8000/v1",
            local_model=SUPER_SNAPSHOT,
            get_json=lambda _url: {
                "data": [{"id": "nemotron-3-nano-30b-a3b"}]
            },
        )


@pytest.mark.parametrize(
    "payload", [{}, {"data": None}, {"data": "not-a-list"}]
)
def test_malformed_models_response_is_fatal(payload) -> None:
    with pytest.raises(RuntimeError, match=r"must contain data\[\]"):
        synthesis._preflight_local_staged_model_identity(
            base_url="http://127.0.0.1:8000/v1",
            local_model=SUPER_SNAPSHOT,
            get_json=lambda _url: payload,
        )


def test_non_object_models_response_is_fatal() -> None:
    with pytest.raises(RuntimeError, match="is not an object"):
        synthesis._preflight_local_staged_model_identity(
            base_url="http://127.0.0.1:8000/v1",
            local_model=SUPER_SNAPSHOT,
            get_json=lambda _url: [],
        )


def test_run_preflight_failure_creates_no_output_or_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAINFORGE_STAGED_SYNTHESIS_V4", "true")
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", SUPER_SNAPSHOT)
    output_dir = tmp_path / "must-not-exist"

    def fail_preflight(**_kwargs):
        raise RuntimeError("identity gate rejected")

    monkeypatch.setattr(
        synthesis, "_preflight_local_staged_model_identity", fail_preflight
    )

    with pytest.raises(RuntimeError, match="identity gate rejected"):
        synthesis.run_synthesis(
            corpus_dir=tmp_path / "corpus-root",
            course_code="IDENTITY_TEST",
            provider="local",
            output_dir=output_dir,
        )
    assert not output_dir.exists()
