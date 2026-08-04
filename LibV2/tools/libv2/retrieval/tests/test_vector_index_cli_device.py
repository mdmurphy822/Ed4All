"""Verify device and batch-size plumbing on the vector-index CLI surface.

The tests enforce three CLI invariants in ``LibV2/tools/libv2/cli.py``:

1. ``vector-index build --device`` must accept ``cuda:N`` — a token the
   provider resolver accepts and the
   index manifest records verbatim. The option validates through the one
   canonical resolver, so the CLI and the build path can never disagree about
   what a device token is, and ``auto`` is rejected with the project's own
   "auto-detection is silent degradation" message.
2. Per-call environment overrides are scoped and restored so one course build
   cannot alter the device selected by a subsequent build in the same process.
3. ``retrieval-benchmark --build-index`` exposes the same controls so its
   manifest records the device that produced the measured index.

Hermetic: the deterministic ``fake`` embedding provider (no weights, no
network), CPU-only. The fake provider's resolved device is what the manifest
records, so the device assertions here are about PLUMBING, not about running
anything on a GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibV2.tools.libv2.cli import (  # noqa: E402
    _embedding_device_option,
    _scoped_embedding_env,
    main as libv2_main,
)

_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"
_SLUG = "mini-retrieval-101"


@pytest.fixture(autouse=True)
def _require_fixture():
    if not (_FIXTURE / "semantik_chunks" / "chunks.jsonl").exists():
        pytest.skip("mini_course fixture not present")


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")


def _materialize(repo_root: Path) -> Path:
    cdir = repo_root / "courses" / _SLUG
    (cdir / "semantik_chunks").mkdir(parents=True)
    shutil.copy(
        _FIXTURE / "semantik_chunks" / "chunks.jsonl",
        cdir / "semantik_chunks" / "chunks.jsonl",
    )
    (cdir / "manifest.json").write_text(
        json.dumps({"classification": {"primary_domain": "rag"}})
    )
    return cdir


def _build(repo_root: Path, *args: str):
    return CliRunner().invoke(
        libv2_main,
        ["--repo", str(repo_root), "vector-index", "build", "--course", _SLUG,
         *args],
    )


# --------------------------------------------------------------------------- #
# --device token validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [("cpu", "cpu"), ("CUDA", "cuda"), (" cuda:1 ", "cuda:1"), (None, None)],
)
def test_device_callback_normalizes(raw, expected):
    assert _embedding_device_option(None, None, raw) == expected


@pytest.mark.parametrize("raw", ["auto", "gpu", "cuda:x", "tpu"])
def test_device_callback_rejects_unknown_tokens(raw):
    import click

    with pytest.raises(click.BadParameter) as excinfo:
        _embedding_device_option(None, None, raw)
    assert "cpu" in str(excinfo.value)


def test_cuda_index_token_is_accepted_by_the_option(tmp_path):
    """``cuda:0`` reaches the resolver instead of being rejected by the parser.

    The fake provider never touches a device, so this only asserts the option
    stopped narrowing the token set — no CUDA work is performed.
    """
    _materialize(tmp_path)
    res = _build(tmp_path, "--device", "cuda:0")
    assert "is not one of" not in res.output
    assert "not a recognized device token" not in res.output


def test_auto_is_rejected_at_the_cli(tmp_path):
    _materialize(tmp_path)
    res = _build(tmp_path, "--device", "auto")
    assert res.exit_code != 0
    assert "auto" in res.output.lower()


# --------------------------------------------------------------------------- #
# Scoped env override
# --------------------------------------------------------------------------- #
def test_scoped_env_restores_prior_values(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    monkeypatch.delenv("ED4ALL_EMBEDDING_BATCH_SIZE", raising=False)

    with _scoped_embedding_env("cuda:2", 64):
        assert os.environ["ED4ALL_EMBEDDING_DEVICE"] == "cuda:2"
        assert os.environ["ED4ALL_EMBEDDING_BATCH_SIZE"] == "64"

    assert os.environ["ED4ALL_EMBEDDING_DEVICE"] == "cpu"
    assert "ED4ALL_EMBEDDING_BATCH_SIZE" not in os.environ


def test_scoped_env_is_a_noop_when_nothing_is_overridden(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    with _scoped_embedding_env(None, None):
        assert os.environ["ED4ALL_EMBEDDING_DEVICE"] == "cpu"
    assert os.environ["ED4ALL_EMBEDDING_DEVICE"] == "cpu"


def test_scoped_env_restores_even_when_the_body_raises(monkeypatch):
    monkeypatch.delenv("ED4ALL_EMBEDDING_DEVICE", raising=False)
    with pytest.raises(RuntimeError):
        with _scoped_embedding_env("cpu", 8):
            raise RuntimeError("build blew up")
    assert "ED4ALL_EMBEDDING_DEVICE" not in os.environ


def test_build_does_not_leak_the_override_into_the_process(tmp_path, monkeypatch):
    """The bug: a per-course ``--device`` used to pin every later course.

    Asserted end-to-end through the real command so the fix cannot be undone
    by moving the override back out of the scope.
    """
    pytest.importorskip("numpy")
    monkeypatch.delenv("ED4ALL_EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("ED4ALL_EMBEDDING_BATCH_SIZE", raising=False)
    _materialize(tmp_path)

    res = _build(tmp_path, "--device", "cpu", "--batch-size", "4")
    assert res.exit_code == 0, res.output
    assert "ED4ALL_EMBEDDING_DEVICE" not in os.environ
    assert "ED4ALL_EMBEDDING_BATCH_SIZE" not in os.environ


# --------------------------------------------------------------------------- #
# Resolved provenance is echoed, not assumed
# --------------------------------------------------------------------------- #
def test_build_echoes_the_recorded_provenance_triple(tmp_path):
    pytest.importorskip("numpy")
    cdir = _materialize(tmp_path)
    res = _build(tmp_path, "--batch-size", "4")
    assert res.exit_code == 0, res.output

    manifest = json.loads(
        (cdir / "vector_index" / "manifest.json").read_text("utf-8")
    )
    # The terminal must report what the manifest actually recorded — the whole
    # point is that an operator does not have to trust the ambient env.
    assert f"batch_size: {manifest['batch_size']}" in res.output
    assert f"device: {manifest['device']}" in res.output
    assert "dtype:" in res.output


def _spy_resolution(monkeypatch) -> list:
    """Record the env every provider resolution sees, in call order.

    A list, not a dict: the benchmark command resolves a SECOND client for the
    query arm after the build, deliberately outside the override scope, and a
    single-slot recorder would silently report that one instead.

    Asserting the RESOLVED batch size instead would test the provider registry,
    not this wiring: the ``fake`` provider entry declares ``batch_size_default``
    but no ``batch_size_env``, so it ignores the variable by construction (see
    the cross-lane note in the lane report). What belongs to the CLI is that
    the value is present in the environment for the duration of the build and
    gone afterwards.
    """
    import lib.embedding.providers as providers

    seen: list = []
    real = providers.build_embedding_client

    def _spy(*args, **kwargs):
        seen.append({
            "device": os.environ.get("ED4ALL_EMBEDDING_DEVICE"),
            "batch_size": os.environ.get("ED4ALL_EMBEDDING_BATCH_SIZE"),
        })
        return real(*args, **kwargs)

    monkeypatch.setattr(providers, "build_embedding_client", _spy)
    return seen


def test_overrides_are_visible_to_the_resolver_during_the_build(
    tmp_path, monkeypatch
):
    pytest.importorskip("numpy")
    monkeypatch.delenv("ED4ALL_EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("ED4ALL_EMBEDDING_BATCH_SIZE", raising=False)
    seen = _spy_resolution(monkeypatch)
    _materialize(tmp_path)

    res = _build(tmp_path, "--device", "cpu", "--batch-size", "4")
    assert res.exit_code == 0, res.output
    assert seen and seen[0] == {"device": "cpu", "batch_size": "4"}
    # ...and gone once the build returns.
    assert "ED4ALL_EMBEDDING_DEVICE" not in os.environ
    assert "ED4ALL_EMBEDDING_BATCH_SIZE" not in os.environ


def test_status_reports_the_full_reproducibility_triple(tmp_path):
    pytest.importorskip("numpy")
    _materialize(tmp_path)
    assert _build(tmp_path, "--batch-size", "4").exit_code == 0

    res = CliRunner().invoke(
        libv2_main,
        ["--repo", str(tmp_path), "vector-index", "status", "--course", _SLUG],
    )
    assert res.exit_code == 0, res.output
    assert "device:" in res.output
    assert "batch_size:" in res.output
    # ``dtype`` is optional on the manifest; absent must read as "unrecorded",
    # never as an assumed fp32.
    assert "dtype:" in res.output


# --------------------------------------------------------------------------- #
# retrieval-benchmark --build-index gained the same two options
# --------------------------------------------------------------------------- #
def test_retrieval_benchmark_accepts_device_and_batch_size(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.delenv("ED4ALL_EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("ED4ALL_EMBEDDING_BATCH_SIZE", raising=False)
    seen = _spy_resolution(monkeypatch)
    cdir = _materialize(tmp_path)
    (cdir / "retrieval_eval").mkdir(parents=True)
    shutil.copy(
        _FIXTURE / "retrieval_eval" / "gold_set.json",
        cdir / "retrieval_eval" / "gold_set.json",
    )

    res = CliRunner().invoke(
        libv2_main,
        ["--repo", str(tmp_path), "retrieval-benchmark", "--course", _SLUG,
         "--engines", "bm25,semantic", "--build-index",
         "--device", "cpu", "--batch-size", "4"],
    )
    assert res.exit_code == 0, res.output
    assert seen and seen[0] == {"device": "cpu", "batch_size": "4"}, (
        f"--device / --batch-size did not reach the inline build: {seen}"
    )

    manifest = json.loads(
        (cdir / "vector_index" / "manifest.json").read_text("utf-8")
    )
    assert f"batch_size: {manifest['batch_size']}" in res.output
    # The benchmark keeps querying in this process after the build; a leaked
    # pin would silently become the QUERY encoder's device too.
    assert "ED4ALL_EMBEDDING_DEVICE" not in os.environ


def test_retrieval_benchmark_rejects_an_unknown_device(tmp_path):
    _materialize(tmp_path)
    res = CliRunner().invoke(
        libv2_main,
        ["--repo", str(tmp_path), "retrieval-benchmark", "--course", _SLUG,
         "--build-index", "--device", "auto"],
    )
    assert res.exit_code != 0
    assert "auto" in res.output.lower()
