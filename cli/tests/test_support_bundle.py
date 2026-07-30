"""Tests for ``ed4all support-bundle`` (OP1).

Verifies the bundle (1) excludes secret-shaped files + masks secret-shaped JSON
keys, (2) never walks course content, (3) always carries a manifest listing
every included member, and (4) gates decision captures behind
``--include-captures``.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from click.testing import CliRunner

from cli.commands.support_bundle import (
    build_support_bundle_files,
    redact_file,
    support_bundle_command,
    write_bundle,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


def _make_state(tmp_path: Path) -> Path:
    """Build a tmp runtime/state/ root with a run dir, gui logs, and a secret file."""
    state = tmp_path / "state"
    run_dir = state / "runs" / "WF-TEST-run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "vram_trajectory.jsonl").write_text('{"free_mib": 1024}\n')
    (run_dir / "llm_usage.jsonl").write_text('{"tokens": 42}\n')
    (run_dir / "checkpoints" / "phase1.json").write_text('{"phase": "done"}')

    # A secret-only file inside the run dir MUST be dropped.
    (run_dir / "secrets.json").write_text('{"LOCAL_SYNTHESIS_API_KEY": "sk-REAL"}')
    # An env-shaped JSON with secret keys MUST be masked, not dropped.
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "provider": "local",
                "LOCAL_SYNTHESIS_API_KEY": "sk-LEAK-SHOULD-NOT-APPEAR",
                "TOGETHER_API_KEY": "tok-LEAK",
                "num_ctx": 4096,
            }
        )
    )

    logs = state / "gui" / "logs"
    logs.mkdir(parents=True)
    (logs / "WF-TEST-run.log").write_text("run started\nphase complete\n")
    return state


def _make_captures(tmp_path: Path) -> Path:
    root = tmp_path / "runtime/training-captures"
    d = root / "tool" / "COURSE" / "phase_1"
    d.mkdir(parents=True)
    (d / "decisions_1.jsonl").write_text('{"rationale": "quoted source text here"}\n')
    return root


# ---------------------------------------------------------------------- #
# redact_file
# ---------------------------------------------------------------------- #


def test_redact_drops_secret_only_files(tmp_path: Path) -> None:
    assert redact_file(tmp_path / "secrets.json", b"{}") is None
    assert redact_file(tmp_path / ".env.rendered", b"X=1") is None
    assert redact_file(tmp_path / "server.key", b"pem") is None


def test_redact_masks_secret_json_keys(tmp_path: Path) -> None:
    raw = json.dumps({"LOCAL_SYNTHESIS_API_KEY": "sk-REAL", "num_ctx": 4096}).encode()
    out = redact_file(tmp_path / "config.json", raw)
    assert out is not None
    doc = json.loads(out)
    assert doc["LOCAL_SYNTHESIS_API_KEY"] == "***REDACTED***"
    assert doc["num_ctx"] == 4096
    assert b"sk-REAL" not in out


def test_redact_passes_through_plain_text(tmp_path: Path) -> None:
    assert redact_file(tmp_path / "run.log", b"hello") == b"hello"


# ---------------------------------------------------------------------- #
# build_support_bundle_files
# ---------------------------------------------------------------------- #


def test_bundle_excludes_secrets_and_masks_config(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    built = build_support_bundle_files(
        run_id="WF-TEST-run",
        state_root=state,
        doctor_payload={"summary": "OK"},
    )
    arcnames = {f.arcname for f in built.files}

    # secrets.json is dropped entirely.
    assert not any(a.endswith("/secrets.json") for a in arcnames)
    # The real key value never appears in ANY bundled member.
    for f in built.files:
        assert b"sk-REAL" not in f.data
        assert b"sk-LEAK-SHOULD-NOT-APPEAR" not in f.data
        assert b"tok-LEAK" not in f.data

    # The config snapshot IS bundled (masked), the run jsonls + gui log too.
    assert any(a.endswith("/config_snapshot.json") for a in arcnames)
    assert any(a.endswith("/vram_trajectory.jsonl") for a in arcnames)
    assert any(a.endswith("/llm_usage.jsonl") for a in arcnames)
    assert any(a.startswith("gui-logs/") for a in arcnames)
    assert "doctor.json" in arcnames


def test_bundle_manifest_lists_every_file(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    built = build_support_bundle_files(
        run_id="WF-TEST-run", state_root=state, doctor_payload={"summary": "OK"}
    )
    manifest_file = next(f for f in built.files if f.arcname == "manifest.json")
    manifest = json.loads(manifest_file.data)
    listed = {e["arcname"] for e in manifest["files"]}
    # Every non-manifest bundled file is listed with size + sha256.
    for f in built.files:
        if f.arcname == "manifest.json":
            continue
        assert f.arcname in listed
    for entry in manifest["files"]:
        assert "size" in entry and "sha256" in entry
    # The excluded secret file is recorded as a warning.
    assert any("excluded secret-shaped file" in w for w in manifest["warnings"])


def test_bundle_excludes_course_content(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    # Even if a course-content-looking dir exists elsewhere it is never walked.
    (tmp_path / "LibV2" / "courses" / "SLUG").mkdir(parents=True)
    (tmp_path / "LibV2" / "courses" / "SLUG" / "content.html").write_text("<p>x</p>")
    built = build_support_bundle_files(
        run_id="WF-TEST-run", state_root=state, doctor_payload={"summary": "OK"}
    )
    arcnames = {f.arcname for f in built.files}
    assert not any("courses" in a for a in arcnames)
    assert not any("content.html" in a for a in arcnames)


def test_captures_gated_behind_flag(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    captures = _make_captures(tmp_path)

    without = build_support_bundle_files(
        run_id="WF-TEST-run",
        state_root=state,
        captures_root=captures,
        include_captures=False,
        doctor_payload={"summary": "OK"},
    )
    assert not any(f.arcname.startswith("captures/") for f in without.files)

    with_caps = build_support_bundle_files(
        run_id="WF-TEST-run",
        state_root=state,
        captures_root=captures,
        include_captures=True,
        doctor_payload={"summary": "OK"},
    )
    assert any(f.arcname.startswith("captures/") for f in with_caps.files)
    assert any("WARNING" in w for w in with_caps.warnings)


def test_newest_run_selected_when_no_run_id(tmp_path: Path) -> None:
    import os
    import time as _time

    state = _make_state(tmp_path)
    older = state / "runs" / "WF-OLD"
    older.mkdir(parents=True)
    (older / "a.json").write_text("{}")
    # Make the pre-existing WF-TEST-run newer.
    newer_run = state / "runs" / "WF-TEST-run"
    now = _time.time()
    os.utime(older, (now - 1000, now - 1000))
    os.utime(newer_run, (now, now))

    built = build_support_bundle_files(
        state_root=state, doctor_payload={"summary": "OK"}
    )
    manifest = json.loads(
        next(f for f in built.files if f.arcname == "manifest.json").data
    )
    assert manifest["run_id"] == "WF-TEST-run"


# ---------------------------------------------------------------------- #
# CLI wiring
# ---------------------------------------------------------------------- #


def test_cli_writes_bundle(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path)
    out = tmp_path / "bundle.tar.gz"

    # Stub the live diagnostics so the test never touches a GPU/ollama.
    monkeypatch.setattr(
        "cli.commands.support_bundle.collect_doctor_json",
        lambda run_id: {"summary": "OK", "results": []},
    )

    runner = CliRunner()
    res = runner.invoke(
        support_bundle_command,
        ["--run-id", "WF-TEST-run", "--output", str(out), "--state-root", str(state)],
    )
    assert res.exit_code == 0, res.output
    assert out.exists()
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    assert "manifest.json" in names
    assert "doctor.json" in names
    assert not any(n.endswith("/secrets.json") for n in names)


def test_write_bundle_roundtrip(tmp_path: Path) -> None:
    built = build_support_bundle_files(
        run_id="missing", state_root=tmp_path / "nope", doctor_payload={"summary": "OK"}
    )
    out = tmp_path / "b.tar.gz"
    size = write_bundle(built.files, out)
    assert size > 0
    with tarfile.open(out, "r:gz") as tar:
        assert "manifest.json" in tar.getnames()
