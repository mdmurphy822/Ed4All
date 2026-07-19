"""heading_judge phase — permanent pipeline wiring regression net.

Covers the SEMANTIK_HEADING_JUDGE textbook_to_course phase (between
``semantik_conversion`` and ``staging``; validated live 2026-07-18):

* (i)   flag-off skip-with-pass, zero side effects;
* (ii)  flag-on but no ``*.glmocr_layout.json`` sidecars -> skip-with-pass;
* (iii) mocked-subprocess success -> COPY-BACK happened with ``.prejudge.bak``
        / ``.bak``, the layout sidecar is untouched, the DecisionCapture fired
        with a dynamic rationale + the ``heading_level_judge`` discriminator,
        and the phase-output counts are correct;
* (iv)  mocked nonzero exit -> fail-open warning, the phase still passes and
        the conversion HTML is untouched;
* (v)   ``_PHASE_TOOL_MAPPING`` routes ``heading_judge`` ->
        ``run_heading_judge`` and the registry carries the handler;
* (vi)  workflows.yaml wiring: phase order + ``staging.depends_on`` flip +
        the validator-only ``agents: []`` shape.

No GPU, no network, no real seat — the judge subprocess is mocked.
No course slugs / device paths are hardcoded; everything lives in tmp_path.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.executor import _PHASE_TOOL_MAPPING  # noqa: E402
from MCP.tools import pipeline_tools  # noqa: E402

WORKFLOWS_YAML = PROJECT_ROOT / "config" / "workflows.yaml"

LAYOUT_BYTES = json.dumps(
    {"pages": [{"page_no": 1, "regions": [{"label": "text"}]}]}
).encode("utf-8")

OLD_ESCALATIONS = (
    json.dumps({"reason": "heading_level_pending", "region_index": 4}) + "\n"
    + json.dumps({"reason": "heading_level_pending", "region_index": 9}) + "\n"
)

CORRECTED_ESCALATIONS = (
    json.dumps({"reason": "heading_level_judged", "region_index": 4}) + "\n"
    + json.dumps({"reason": "heading_level_pending", "region_index": 9}) + "\n"
)

REPORT = {
    "applied": 5,
    "kept": 2,
    "clamped": 1,
    "dropped": 0,
    "windows": 3,
    "n_pending": 8,
    "n_headings": 40,
    "model": "nemotron-3-super",
}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """A scan corpus dir with one chapter's GLM-OCR sidecars + a conversion
    output dir (via ED4ALL_HOME so ``semantik_output_dir()`` resolves into
    tmp_path)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "ch01.glmocr_layout.json").write_bytes(LAYOUT_BYTES)
    (corpus_dir / "ch01.glmocr_escalations.jsonl").write_text(
        OLD_ESCALATIONS, encoding="utf-8"
    )

    home = tmp_path / "home"
    out_dir = home / "semantik-output"
    out_dir.mkdir(parents=True)
    (out_dir / "ch01_accessible.html").write_text(
        "<html>OLD PRE-JUDGE</html>", encoding="utf-8"
    )
    monkeypatch.setenv("ED4ALL_HOME", str(home))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("SEMANTIK_GLMOCR_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("SEMANTIK_PYTHON", raising=False)
    monkeypatch.delenv("SEMANTIK_RUNTIME_DIR", raising=False)
    return SimpleNamespace(corpus_dir=corpus_dir, semantik_out=out_dir)


class _RecordingCapture:
    """Fake DecisionCapture recording every log_decision call (no disk)."""

    calls = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def log_decision(self, **kwargs):
        _RecordingCapture.calls.append(
            {"init": self.init_kwargs, **kwargs}
        )


@pytest.fixture()
def recording_capture(monkeypatch):
    _RecordingCapture.calls = []
    import lib.decision_capture as dc

    monkeypatch.setattr(dc, "DecisionCapture", _RecordingCapture)
    return _RecordingCapture


def _fake_judge_run(record_env=None):
    """Build a fake ``subprocess.run`` that mimics a successful --apply
    standalone judge invocation: writes report + corrected layout +
    corrected escalations + re-rendered HTML into --out."""

    def fake_run(cmd, **kwargs):
        if record_env is not None:
            record_env.append(
                {"cmd": list(cmd), **{k: kwargs.get(k) for k in
                                      ("cwd", "timeout", "env")}}
            )
        layout = Path(cmd[3])
        out_dir = Path(cmd[cmd.index("--out") + 1])
        stem = layout.name[: -len(".glmocr_layout.json")]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.heading_judgments.json").write_text(
            json.dumps(REPORT), encoding="utf-8"
        )
        (out_dir / f"{stem}.corrected_layout.json").write_text(
            json.dumps({"region_provenance": [], "heading_tree": []}),
            encoding="utf-8",
        )
        (out_dir / f"{stem}.glmocr_escalations.jsonl").write_text(
            CORRECTED_ESCALATIONS, encoding="utf-8"
        )
        (out_dir / f"{stem}_accessible.html").write_text(
            "<html>JUDGED</html>", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    return fake_run


# ---------------------------------------------------------------------------
# (i) flag-off skip-with-pass
# ---------------------------------------------------------------------------


def test_flag_off_skips_with_pass_zero_side_effects(corpus, monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE", raising=False)

    def _boom(*a, **k):  # the judge subprocess must never launch
        raise AssertionError("subprocess.run called with flag off")

    monkeypatch.setattr(subprocess, "run", _boom)
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    assert result["skipped"] is True
    assert result["reason"] == "flag_off"
    # zero side effects: conversion HTML + escalations byte-identical
    assert (
        corpus.semantik_out / "ch01_accessible.html"
    ).read_text(encoding="utf-8") == "<html>OLD PRE-JUDGE</html>"
    assert (
        corpus.corpus_dir / "ch01.glmocr_escalations.jsonl"
    ).read_text(encoding="utf-8") == OLD_ESCALATIONS
    assert not list(corpus.corpus_dir.glob("*.bak"))
    assert not (corpus.corpus_dir / "heading_judge_out").exists()


@pytest.mark.parametrize("garbage", ["0", "false", "off", "banana", ""])
def test_flag_garbage_or_falsey_is_off(garbage, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", garbage)
    assert pipeline_tools._heading_judge_enabled() is False


@pytest.mark.parametrize("truthy", ["1", "true", "YES", "On"])
def test_flag_truthy_set(truthy, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", truthy)
    assert pipeline_tools._heading_judge_enabled() is True


# ---------------------------------------------------------------------------
# (ii) no-sidecars skip
# ---------------------------------------------------------------------------


def test_no_sidecars_skips_with_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.delenv("SEMANTIK_GLMOCR_OUTPUT_DIR", raising=False)
    born_digital = tmp_path / "born-digital"
    born_digital.mkdir()
    (born_digital / "book.pdf").write_bytes(b"%PDF-1.4")

    def _boom(*a, **k):
        raise AssertionError("subprocess.run called with no sidecars")

    monkeypatch.setattr(subprocess, "run", _boom)
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(born_digital / "book.pdf"), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    assert result["skipped"] is True
    assert result["reason"] == "no_sidecars"
    assert result["chapters_judged"] == 0


# ---------------------------------------------------------------------------
# (iii) mocked-subprocess success: copy-back + capture + counts
# ---------------------------------------------------------------------------


def test_success_copy_back_capture_and_counts(
    corpus, monkeypatch, recording_capture, tmp_path
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    # run-scoped usage tap + out dir
    monkeypatch.setenv("ED4ALL_RUN_ID", "WF-TEST-hj")
    state_runs = tmp_path / "state_runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(state_runs))

    seen = []
    monkeypatch.setattr(subprocess, "run", _fake_judge_run(record_env=seen))

    layout_before = (
        corpus.corpus_dir / "ch01.glmocr_layout.json"
    ).read_bytes()

    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))

    # --- phase output counts -------------------------------------------
    assert result["success"] is True
    assert result.get("skipped") is not True
    assert result["chapters_judged"] == 1
    assert result["chapters_skipped"] == 0
    assert result["chapters_failed"] == 0
    assert result["total_applied"] == REPORT["applied"]
    assert result["total_kept"] == REPORT["kept"]
    # one heading_level_pending row survives in the corrected escalations
    assert result["residual_pending"] == 1

    # --- subprocess invocation contract --------------------------------
    assert len(seen) == 1
    call = seen[0]
    assert call["cmd"][1:3] == [
        "-m", "semantik_structure.glmocr.heading_judge_standalone"
    ]
    assert "--apply" in call["cmd"]
    # cwd defaults to <repo>/SemantiK when SEMANTIK_RUNTIME_DIR is unset
    assert Path(call["cwd"]).name == "SemantiK"
    assert call["timeout"] == pytest.approx(5400.0)
    env = call["env"]
    assert env["SEMANTIK_HEADING_JUDGE"] == "1"
    assert env["SEMANTIK_LLM_USAGE_PHASE"] == "heading_judge"
    assert env["SEMANTIK_LLM_USAGE_PATH"] == str(
        state_runs / "WF-TEST-hj" / "llm_usage.jsonl"
    )
    # judged outputs land run-scoped
    assert result["judged_dir"] == str(
        state_runs / "WF-TEST-hj" / "heading_judge"
    )

    # --- COPY-BACK contract --------------------------------------------
    html_dest = corpus.semantik_out / "ch01_accessible.html"
    assert html_dest.read_text(encoding="utf-8") == "<html>JUDGED</html>"
    bak = corpus.semantik_out / "ch01_accessible.html.prejudge.bak"
    assert bak.read_text(encoding="utf-8") == "<html>OLD PRE-JUDGE</html>"

    esc_dest = corpus.corpus_dir / "ch01.glmocr_escalations.jsonl"
    assert esc_dest.read_text(encoding="utf-8") == CORRECTED_ESCALATIONS
    esc_bak = corpus.corpus_dir / "ch01.glmocr_escalations.jsonl.bak"
    assert esc_bak.read_text(encoding="utf-8") == OLD_ESCALATIONS

    # the layout sidecar is NEVER overwritten (corrected_layout.json has a
    # different shape — region_provenance+heading_tree, not pages)
    assert (
        corpus.corpus_dir / "ch01.glmocr_layout.json"
    ).read_bytes() == layout_before

    # --- DecisionCapture -----------------------------------------------
    assert len(recording_capture.calls) == 1
    cap = recording_capture.calls[0]
    assert cap["init"]["course_code"] == "TESTCOURSE"
    assert cap["init"]["phase"] == "semantik_conversion"
    assert cap["init"]["tool"] == "semantik"
    assert cap["decision_type"] == "structure_review"
    assert cap["heading_level_judge"] is True
    # dynamic rationale interpolates the chapter's own tallies + model
    rationale = cap["rationale"]
    assert len(rationale) >= 20
    assert "nemotron-3-super" in rationale
    assert "5 level correction(s) applied" in rationale
    assert "8 pending" in rationale
    assert "3 skeleton window(s)" in rationale
    assert "ch01" in rationale


def test_capture_failure_never_breaks_phase(corpus, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(subprocess, "run", _fake_judge_run())

    import lib.decision_capture as dc

    class _Boom:
        def __init__(self, **kwargs):
            raise RuntimeError("capture backend down")

    monkeypatch.setattr(dc, "DecisionCapture", _Boom)
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    assert result["chapters_judged"] == 1


# ---------------------------------------------------------------------------
# (iv) fail-open on nonzero exit
# ---------------------------------------------------------------------------


def test_nonzero_exit_fail_open_phase_still_passes(
    corpus, monkeypatch, recording_capture
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")

    def fake_fail(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="seat unreachable: boom"
        )

    monkeypatch.setattr(subprocess, "run", fake_fail)
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True, (
        "the heading judge must NEVER block the build (fail-open)"
    )
    assert result["chapters_failed"] == 1
    assert result["chapters_judged"] == 0
    assert any("ch01" in w and "exited 1" in w for w in result["warnings"])
    # the corpus keeps the chapter's pre-judge HTML + escalations
    assert (
        corpus.semantik_out / "ch01_accessible.html"
    ).read_text(encoding="utf-8") == "<html>OLD PRE-JUDGE</html>"
    assert (
        corpus.corpus_dir / "ch01.glmocr_escalations.jsonl"
    ).read_text(encoding="utf-8") == OLD_ESCALATIONS
    # no capture fires for a failed chapter
    assert recording_capture.calls == []


def test_timeout_fail_open(corpus, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_TIMEOUT", "7")

    def fake_timeout(cmd, **kwargs):
        assert kwargs.get("timeout") == pytest.approx(7.0)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=7)

    monkeypatch.setattr(subprocess, "run", fake_timeout)
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    assert result["chapters_failed"] == 1
    assert any("timed out" in w for w in result["warnings"])


@pytest.mark.parametrize("raw,expected", [
    ("", 5400.0),
    ("garbage", 5400.0),
    ("-3", 5400.0),
    ("0", 5400.0),
    ("120", 120.0),
    ("120.5", 120.5),
])
def test_chapter_timeout_parse_with_fallback(raw, expected, monkeypatch):
    if raw:
        monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_TIMEOUT", raw)
    else:
        monkeypatch.delenv(
            "SEMANTIK_HEADING_JUDGE_CHAPTER_TIMEOUT", raising=False
        )
    assert pipeline_tools._resolve_heading_judge_chapter_timeout() == (
        pytest.approx(expected)
    )


# ---------------------------------------------------------------------------
# (v) dispatch mapping + registry
# ---------------------------------------------------------------------------


def test_phase_tool_mapping_routes_heading_judge():
    assert _PHASE_TOOL_MAPPING.get("heading_judge") == "run_heading_judge"


def test_registry_carries_run_heading_judge():
    registry = pipeline_tools._build_tool_registry()
    assert registry.get("run_heading_judge") is (
        pipeline_tools._run_heading_judge
    )


# ---------------------------------------------------------------------------
# (vi) workflows.yaml wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflows_data():
    return yaml.safe_load(WORKFLOWS_YAML.read_text(encoding="utf-8"))


def _t2c_phases(workflows_data):
    return workflows_data["workflows"]["textbook_to_course"]["phases"]


def test_phase_sits_between_conversion_and_staging(workflows_data):
    names = [p["name"] for p in _t2c_phases(workflows_data)]
    assert "heading_judge" in names
    assert (
        names.index("semantik_conversion")
        < names.index("heading_judge")
        < names.index("staging")
    )


def test_phase_shape_validator_only(workflows_data):
    phase = next(
        p for p in _t2c_phases(workflows_data)
        if p["name"] == "heading_judge"
    )
    assert phase["agents"] == [], (
        "heading_judge must declare agents: [] (phase-name dispatch)"
    )
    assert phase["depends_on"] == ["semantik_conversion"]
    assert not phase.get("validation_gates"), "no new validation gates"


def test_staging_depends_on_heading_judge(workflows_data):
    staging = next(
        p for p in _t2c_phases(workflows_data) if p["name"] == "staging"
    )
    assert staging["depends_on"] == ["heading_judge"]
