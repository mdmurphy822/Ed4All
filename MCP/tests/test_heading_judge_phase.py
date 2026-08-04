"""heading_judge phase — permanent pipeline wiring regression net.

Covers the SEMANTIK_HEADING_JUDGE textbook_to_course phase (between
``semantik_conversion`` and ``staging``; validated live 2026-07-18):

* (i)   explicit falsey opt-out (default ON) skip-with-pass, zero side
        effects;
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


_AUDIT_MODULE = "semantik_structure.glmocr.heading_judge_audit_standalone"


def _is_audit_cmd(cmd):
    return _AUDIT_MODULE in list(cmd)


def _run_real_audit(cmd, flagged=None):
    """Stand in for the deterministic audit standalone subprocess (the SemantiK
    package is not importable from the MCP test venv). Writes a minimal-but-
    valid ``heading_judge_audit.json`` into --out and returns a subprocess-like
    success result; NEVER recorded as a judge invocation."""
    out_dir = Path(cmd[cmd.index("--out") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged = list(flagged or [])
    report = {
        "audit_schema_version": 1,
        "thresholds": {"collapse_share": 0.95, "min_headings": 4},
        "n_chapters": 0,
        "chapters": [],
        "book": {
            "level_distribution": {},
            "incomplete_chapters": [],
            "collapsed_chapters": [{"stem": s, "reasons": ["level_collapse_single"]}
                                   for s in flagged],
            "inconsistent_signatures": [],
        },
        "flagged_chapters": flagged,
    }
    (out_dir / "heading_judge_audit.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return SimpleNamespace(returncode=0, stdout="{}", stderr="")


def _fake_judge_run(record_env=None, report=None, html="<html>JUDGED</html>"):
    """Build a fake ``subprocess.run`` that mimics a successful --apply
    standalone judge invocation: writes report + corrected layout +
    corrected escalations + re-rendered HTML into --out.

    ``html`` is the re-rendered body; pass the corpus fixture's PRE-JUDGE HTML
    to simulate an IDEMPOTENT re-judge (a judge that re-confirms an
    already-judged layout emits byte-identical output).
    """
    report_doc = REPORT if report is None else report

    def fake_run(cmd, **kwargs):
        # The default-ON deterministic AUDIT shells out to its own standalone;
        # run it for real (over the judged sidecars) but never count it as a
        # judge invocation.
        if _is_audit_cmd(cmd):
            return _run_real_audit(cmd)
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
            json.dumps(report_doc), encoding="utf-8"
        )
        (out_dir / f"{stem}.corrected_layout.json").write_text(
            json.dumps({"region_provenance": [], "heading_tree": []}),
            encoding="utf-8",
        )
        (out_dir / f"{stem}.glmocr_escalations.jsonl").write_text(
            CORRECTED_ESCALATIONS, encoding="utf-8"
        )
        (out_dir / f"{stem}_accessible.html").write_text(
            html, encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    return fake_run


# ---------------------------------------------------------------------------
# (i) explicit falsey opt-out skip-with-pass (flag is default ON)
# ---------------------------------------------------------------------------


def test_flag_explicit_falsey_skips_with_pass_zero_side_effects(
    corpus, monkeypatch
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "0")

    def _boom(*a, **k):  # the judge subprocess must never launch
        raise AssertionError("subprocess.run called with flag opted out")

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


@pytest.mark.parametrize("falsey", ["0", "false", "no", "off"])
def test_flag_explicit_falsey_is_off(falsey, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", falsey)
    assert pipeline_tools._heading_judge_enabled() is False


@pytest.mark.parametrize("garbage", ["banana", ""])
def test_flag_garbage_or_blank_is_on(garbage, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", garbage)
    assert pipeline_tools._heading_judge_enabled() is True


def test_flag_unset_defaults_on(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE", raising=False)
    assert pipeline_tools._heading_judge_enabled() is True


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
    assert "5 verdict(s) applied" in rationale
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
        # The default-ON audit subprocess is not the judge — let it succeed.
        if _is_audit_cmd(cmd):
            return _run_real_audit(cmd)
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


# ---------------------------------------------------------------------------
# (vii) figure-enrichment preservation across the judge copy-back
#       (whole-book regression: the judge re-render rebuilds from the
#       layout sidecar, which never carried the VLM alt-text enrichment, so
#       VLM-captioned figures degraded to the sr-only "Figure." placeholder
#       and the copy-back shipped the degraded bytes)
# ---------------------------------------------------------------------------

ENRICHED_PRIOR_HTML = (
    "<html><h3>Pending Level</h3>"
    '<section data-semantik-block-id="s2" data-semantik-block-role="figure">'
    "<figure><figcaption>A network diagram showing interconnected nodes "
    "and links, illustrating a complex system structure."
    "</figcaption></figure></section>"
    '<section data-semantik-block-id="s5">'
    "<figure><figcaption>Figure 1.1: The system throughput."
    "</figcaption></figure></section></html>"
)

DEGRADED_JUDGED_HTML = (
    "<html><h2>Pending Level</h2>"  # the judge's level correction
    '<section data-semantik-block-id="s2" data-semantik-block-role="figure">'
    '<figure><figcaption><span class="sr-only">Figure.</span>'
    "</figcaption></figure></section>"
    '<section data-semantik-block-id="s5">'
    "<figure><figcaption>Figure 1.1: The system throughput."
    "</figcaption></figure></section></html>"
)


def _fake_judge_run_degraded(record_env=None):
    """A fake judge subprocess whose re-render DEGRADED the VLM figure
    caption (the layout sidecar carries no alt-text enrichment)."""

    def fake_run(cmd, **kwargs):
        if _is_audit_cmd(cmd):
            return _run_real_audit(cmd)
        if record_env is not None:
            record_env.append({"cmd": list(cmd)})
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
            DEGRADED_JUDGED_HTML, encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    return fake_run


def test_copy_back_preserves_vlm_figure_captions(corpus, monkeypatch):
    """The shipped HTML must carry BOTH the judged heading levels AND the
    VLM figure enrichment from the pre-judge render."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    (corpus.semantik_out / "ch01_accessible.html").write_text(
        ENRICHED_PRIOR_HTML, encoding="utf-8"
    )
    seen = []
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run_degraded(record_env=seen)
    )

    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    assert result["chapters_judged"] == 1

    # the subprocess is handed the prior enriched HTML to merge from
    cmd = seen[0]["cmd"]
    assert "--source-html" in cmd
    assert cmd[cmd.index("--source-html") + 1] == str(
        corpus.semantik_out / "ch01_accessible.html"
    )

    shipped = (
        corpus.semantik_out / "ch01_accessible.html"
    ).read_text(encoding="utf-8")
    # judged heading level survives (the point of the re-render)
    assert "<h2>Pending Level</h2>" in shipped
    # VLM caption restored — the accessibility regression this pins
    assert "A network diagram showing interconnected nodes" in shipped
    assert 'sr-only">Figure.' not in shipped
    # the extracted caption stays exactly as the judge rendered it
    assert "Figure 1.1: The system throughput." in shipped
    # .prejudge.bak still holds the enriched pre-judge bytes
    bak = corpus.semantik_out / "ch01_accessible.html.prejudge.bak"
    assert bak.read_text(encoding="utf-8") == ENRICHED_PRIOR_HTML


def test_legacy_runtime_without_source_html_retries_and_merge_still_wins(
    corpus, monkeypatch
):
    """A legacy SemantiK runtime that argparse-rejects --source-html gets a
    single retry without the flag; the copy-back merge still restores the
    figure enrichment Ed4All-side."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    (corpus.semantik_out / "ch01_accessible.html").write_text(
        ENRICHED_PRIOR_HTML, encoding="utf-8"
    )
    seen = []
    inner = _fake_judge_run_degraded(record_env=None)

    def fake_run(cmd, **kwargs):
        if _is_audit_cmd(cmd):
            return _run_real_audit(cmd)
        seen.append(list(cmd))
        if "--source-html" in cmd:
            return SimpleNamespace(
                returncode=2,
                stdout="",
                stderr="error: unrecognized arguments: --source-html",
            )
        return inner(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    assert result["chapters_failed"] == 0
    assert result["chapters_judged"] == 1
    assert len(seen) == 2
    assert "--source-html" in seen[0]
    assert "--source-html" not in seen[1]

    shipped = (
        corpus.semantik_out / "ch01_accessible.html"
    ).read_text(encoding="utf-8")
    assert "A network diagram showing interconnected nodes" in shipped
    assert 'sr-only">Figure.' not in shipped
    assert "<h2>Pending Level</h2>" in shipped


def test_copy_back_merge_failure_fails_open_to_plain_copy(
    corpus, monkeypatch
):
    """A merge failure must never block the copy-back — the judged HTML
    ships unmerged with a warning (the phase's fail-open posture)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    (corpus.semantik_out / "ch01_accessible.html").write_text(
        ENRICHED_PRIOR_HTML, encoding="utf-8"
    )
    monkeypatch.setattr(subprocess, "run", _fake_judge_run_degraded())

    import lib.semantik.figure_enrich_merge as fem

    def _boom(prior, judged):
        raise RuntimeError("merge exploded")

    monkeypatch.setattr(fem, "merge_figure_enrichment", _boom)

    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    shipped = (
        corpus.semantik_out / "ch01_accessible.html"
    ).read_text(encoding="utf-8")
    assert shipped == DEGRADED_JUDGED_HTML
    assert any(
        "figure-enrichment merge failed" in w for w in result["warnings"]
    )


# ---------------------------------------------------------------------------
# (vii) judged-kept vs UNJUDGED accounting (Fix 2 — the truncation hole must
#       never be silently folded into "kept")
# ---------------------------------------------------------------------------

REPORT_UNJUDGED = {
    "applied": 5,
    "kept": 21,
    "kept_judged": 2,
    "unjudged": 19,
    "clamped": 1,
    "dropped": 0,
    "windows": 2,
    "n_pending": 26,
    "n_headings": 60,
    "model": "nemotron-3-super",
    "meta": {"unjudged_ids": [101, 102, 103]},
}


def test_unjudged_accounting_distinguished_and_loud(
    corpus, monkeypatch, recording_capture
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_UNJUDGED)
    )
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["success"] is True
    # kept is still reported, but judged-kept and UNJUDGED are distinguished
    assert result["total_kept"] == 21
    assert result["total_kept_judged"] == 2
    assert result["total_unjudged"] == 19
    # a LOUD warning names the truncation hole (and the explicit id set)
    unjudged_warnings = [w for w in result["warnings"] if "UNJUDGED" in w]
    assert len(unjudged_warnings) == 1
    assert "ch01" in unjudged_warnings[0]
    assert "19 of 26" in unjudged_warnings[0]
    assert "[101, 102, 103]" in unjudged_warnings[0]
    # the DecisionCapture rationale carries the distinction
    assert len(recording_capture.calls) == 1
    rationale = recording_capture.calls[0]["rationale"]
    assert "2 judged-kept" in rationale
    assert "19 UNJUDGED" in rationale


def test_legacy_report_without_unjudged_keys_counts_kept_as_judged(
    corpus, monkeypatch, recording_capture
):
    """A pre-fix report (no kept_judged/unjudged keys) must not fabricate a
    truncation hole: unjudged reads 0 and every kept counts judged-kept."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(subprocess, "run", _fake_judge_run())  # legacy REPORT
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["total_kept"] == REPORT["kept"]
    assert result["total_kept_judged"] == REPORT["kept"]
    assert result["total_unjudged"] == 0
    assert not any("UNJUDGED" in w for w in result["warnings"])


def test_skip_paths_carry_the_new_count_keys(corpus, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "0")
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["total_kept_judged"] == 0
    assert result["total_unjudged"] == 0


# ---------------------------------------------------------------------------
# (vii) Shared-directory isolation and all-unjudged failure reporting.
#
# A file-scoped run must ignore unrelated sidecars in the same directory and
# must report a chapter with no judge verdicts as unjudged rather than passed.
# ---------------------------------------------------------------------------


@pytest.fixture()
def shared_corpus(tmp_path, monkeypatch):
    """A shared corpus dir containing target and unrelated sidecars."""
    corpus_dir = tmp_path / "shared-corpus"
    corpus_dir.mkdir()
    # this run's own book
    (corpus_dir / "MyBook.pdf").write_bytes(b"%PDF-1.4")
    (corpus_dir / "MyBook.glmocr_layout.json").write_bytes(LAYOUT_BYTES)
    (corpus_dir / "MyBook.glmocr_escalations.jsonl").write_text(
        OLD_ESCALATIONS, encoding="utf-8"
    )
    # a DIFFERENT book's drained-run leftovers
    (corpus_dir / "OtherBook.pdf").write_bytes(b"%PDF-1.4")
    (corpus_dir / "OtherBook.glmocr_layout.json").write_bytes(LAYOUT_BYTES)
    (corpus_dir / "OtherBook.glmocr_escalations.jsonl").write_text(
        OLD_ESCALATIONS, encoding="utf-8"
    )

    home = tmp_path / "home"
    out_dir = home / "semantik-output"
    out_dir.mkdir(parents=True)
    (out_dir / "MyBook_accessible.html").write_text(
        "<html>MINE PRE-JUDGE</html>", encoding="utf-8"
    )
    (out_dir / "OtherBook_accessible.html").write_text(
        "<html>FOREIGN — DO NOT TOUCH</html>", encoding="utf-8"
    )
    monkeypatch.setenv("ED4ALL_HOME", str(home))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("SEMANTIK_GLMOCR_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("SEMANTIK_PYTHON", raising=False)
    monkeypatch.delenv("SEMANTIK_RUNTIME_DIR", raising=False)
    return SimpleNamespace(corpus_dir=corpus_dir, semantik_out=out_dir)


def test_foreign_sidecar_never_judged_copied_back_or_counted(
    shared_corpus, monkeypatch, recording_capture
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    seen = []
    monkeypatch.setattr(subprocess, "run", _fake_judge_run(record_env=seen))

    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(shared_corpus.corpus_dir / "MyBook.pdf"),
        course_name="TESTCOURSE",
    )))

    # ONE judge subprocess, on THIS run's book only
    assert len(seen) == 1
    assert Path(seen[0]["cmd"][3]).name == "MyBook.glmocr_layout.json"

    # counts exclude the foreign book entirely
    assert result["chapters_judged"] == 1
    assert result["chapters_foreign_skipped"] == 1
    assert result["foreign_stems"] == ["OtherBook"]
    assert result["total_applied"] == REPORT["applied"]

    # the foreign book's conversion output + escalations are BYTE-IDENTICAL
    assert (
        shared_corpus.semantik_out / "OtherBook_accessible.html"
    ).read_text(encoding="utf-8") == "<html>FOREIGN — DO NOT TOUCH</html>"
    assert (
        shared_corpus.corpus_dir / "OtherBook.glmocr_escalations.jsonl"
    ).read_text(encoding="utf-8") == OLD_ESCALATIONS
    assert not (
        shared_corpus.corpus_dir / "OtherBook.glmocr_escalations.jsonl.bak"
    ).exists()
    assert not (
        shared_corpus.semantik_out / "OtherBook_accessible.html.prejudge.bak"
    ).exists()

    # ...and this run's own book WAS judged + copied back
    assert (
        shared_corpus.semantik_out / "MyBook_accessible.html"
    ).read_text(encoding="utf-8") == "<html>JUDGED</html>"

    # exactly one DecisionCapture — the foreign chapter never emits one
    assert len(recording_capture.calls) == 1
    assert "MyBook" in recording_capture.calls[0]["rationale"]


def test_directory_corpus_token_scopes_to_the_books_it_holds(
    shared_corpus, monkeypatch
):
    """A ``--corpus <dir>/`` run legitimately owns every book in that dir."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    seen = []
    monkeypatch.setattr(subprocess, "run", _fake_judge_run(record_env=seen))
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(shared_corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["chapters_foreign_skipped"] == 0
    assert len(seen) == 2
    assert sorted(Path(c["cmd"][3]).name for c in seen) == [
        "MyBook.glmocr_layout.json", "OtherBook.glmocr_layout.json"
    ]


def test_comma_separated_file_tokens_scope_to_their_own_stems(
    shared_corpus, monkeypatch
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    seen = []
    monkeypatch.setattr(subprocess, "run", _fake_judge_run(record_env=seen))
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=f"{shared_corpus.corpus_dir / 'MyBook.pdf'}",
        course_name="TESTCOURSE",
    )))
    assert result["chapters_foreign_skipped"] == 1
    assert len(seen) == 1


def test_unresolvable_stems_keep_legacy_unscoped_behavior_loudly(
    corpus, monkeypatch, caplog
):
    """The ``corpus`` fixture's dir holds sidecars but NO corpus input, so the
    stem set is unresolvable — judge everything (legacy) but say so."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(subprocess, "run", _fake_judge_run())
    with caplog.at_level("WARNING", logger=pipeline_tools.logger.name):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))
    assert result["chapters_judged"] == 1
    assert result["chapters_foreign_skipped"] == 0
    assert any(
        "could not resolve this run's corpus stems" in r.getMessage()
        for r in caplog.records
    )


def test_corpus_stems_helper_shapes():
    stems = pipeline_tools._heading_judge_corpus_stems
    assert stems("") == set()
    assert stems(None) == set()
    assert stems("/nowhere/Book One.pdf") == {"Book One"}
    assert stems(["/a/x.pdf", "/b/y.html"]) == {"x", "y"}
    assert stems("/a/x.pdf,/b/y.pdf") == {"x", "y"}


# --- a 100%-unjudged chapter is a REAL failure, loudly ----------------------

REPORT_TOTAL_HOLE = {
    "applied": 0,
    "kept": 58,
    "kept_judged": 0,
    "unjudged": 58,
    "clamped": 0,
    "dropped": 0,
    "windows": 2,
    "n_pending": 58,
    "n_headings": 199,
    "model": "nemotron-3-super",
    "failure_modes": ["seat_aborted"],
    "unjudged_reason": (
        "the judge seat ABORTED the in-flight request (finish_reason=abort) "
        "— the seat container was stopped/killed or its engine shut down "
        "mid-generation"
    ),
    "meta": {"unjudged_ids": [4, 7, 11]},
}


def test_total_hole_chapter_is_loud_and_states_the_mechanism(
    corpus, monkeypatch, caplog, recording_capture
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_TOTAL_HOLE)
    )
    with caplog.at_level("WARNING", logger=pipeline_tools.logger.name):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))

    # the phase still fail-opens (never blocks the build) ...
    assert result["success"] is True
    # ... but the outcome is COUNTED as a failure, not folded into "judged"
    assert result["chapters_unjudged"] == 1
    assert result["total_unjudged"] == 58

    warning = next(w for w in result["warnings"] if "UNJUDGED" in w)
    assert "58 of 58" in warning
    assert "REAL FAILURE" in warning
    assert "MECHANISM:" in warning
    assert "ABORTED" in warning
    assert "seat_aborted" in warning

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, "a 100%-unjudged chapter must log at ERROR"
    assert any("REAL FAILURE" in r.getMessage() for r in errors)
    # the summary line, too — it must never read as a clean "0 failed"
    assert any(
        "FULLY-UNJUDGED" in r.getMessage() for r in errors
    )


def test_partial_hole_stays_a_warning_but_still_names_the_mechanism(
    corpus, monkeypatch, caplog
):
    report = dict(REPORT_UNJUDGED)
    report["failure_modes"] = ["length_exhausted"]
    report["unjudged_reason"] = (
        "the judgment exhausted its completion budget (finish_reason=length)"
    )
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(subprocess, "run", _fake_judge_run(report=report))
    with caplog.at_level("WARNING", logger=pipeline_tools.logger.name):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))
    assert result["chapters_unjudged"] == 0
    warning = next(w for w in result["warnings"] if "UNJUDGED" in w)
    assert "REAL FAILURE" not in warning
    assert "MECHANISM:" in warning
    assert "completion budget" in warning
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_mechanism_absent_report_still_names_an_explicit_unknown(
    corpus, monkeypatch
):
    """A legacy judge runtime emits no mechanism keys — the warning must say
    UNKNOWN explicitly rather than silently omit the cause."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_UNJUDGED)
    )
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    warning = next(w for w in result["warnings"] if "UNJUDGED" in w)
    assert "MECHANISM: UNKNOWN" in warning


# ---------------------------------------------------------------------------
# (vii) VERDICT RECORDED vs LEVEL CHANGED + the idempotent-re-judge signal.
#
# 2026-07-22: the phase logged "applied=58 ... kept=0"; a reviewer diffed the
# judged HTML against its ``.prejudge.bak``, saw IDENTICAL bytes, and concluded
# the 58 re-levelings were being dropped before render. In fact only 28 of the
# 58 verdicts moved a level, and the layout had ALREADY been judged in-lane
# during ``semantik_conversion`` — identical bytes were CORRECT. The summary
# must make both facts impossible to misread.
# ---------------------------------------------------------------------------

PRE_JUDGE_HTML = "<html>OLD PRE-JUDGE</html>"

REPORT_MIXED = dict(
    REPORT,
    applied=5,
    changed=3,
    agreed=2,
    transitions={"3->2": 3},
)

#: applied > 0 but NOTHING moved — the exact shape that misled the reviewer.
REPORT_ALL_AGREED = dict(
    REPORT, applied=5, changed=0, agreed=5, transitions={}, kept=0
)

#: a pre-``changed``-key judge runtime: only the per-heading judgments map.
REPORT_LEGACY = {
    "applied": 2,
    "kept": 0,
    "clamped": 0,
    "dropped": 0,
    "windows": 1,
    "n_pending": 2,
    "model": "nemotron-3-super",
    "judgments": {
        "4": {"from": 3, "to": 2, "clamped": False},
        "9": {"from": 3, "to": 3, "clamped": False},
    },
}


def _summary_line(caplog):
    return next(
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("run_heading_judge: ")
        and "judged /" in r.getMessage()
    )


def test_summary_and_output_split_verdicts_from_level_changes(
    corpus, monkeypatch, caplog
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_MIXED)
    )
    with caplog.at_level("INFO", logger="MCP.tools.pipeline_tools"):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))

    # phase output: the counts are ADDITIVE, the legacy key is untouched
    assert result["total_applied"] == 5
    assert result["total_changed"] == 3
    assert result["total_agreed"] == 2
    assert result["total_applied"] == (
        result["total_changed"] + result["total_agreed"]
    )
    assert result["level_transitions"] == {"3->2": 3}

    # summary log: "applied=5" can no longer be read as "5 headings moved"
    line = _summary_line(caplog)
    assert (
        "applied=5 verdicts (3 changed level, 2 agreed with the "
        "pre-existing level) [3->2 x3]" in line
    )


def test_applied_with_zero_level_changes_says_so_in_the_summary(
    corpus, monkeypatch, caplog
):
    """applied > 0 AND changed == 0 — the reviewer's exact confusion."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run",
        _fake_judge_run(report=REPORT_ALL_AGREED, html=PRE_JUDGE_HTML),
    )
    with caplog.at_level("INFO", logger="MCP.tools.pipeline_tools"):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))

    assert result["total_applied"] == 5
    assert result["total_changed"] == 0
    assert result["total_agreed"] == 5
    assert result["level_transitions"] == {}

    line = _summary_line(caplog)
    assert (
        "applied=5 verdicts (0 changed level, 5 agreed with the "
        "pre-existing level)" in line
    )
    assert "[3->2" not in line, "no transition histogram when nothing moved"


def test_idempotent_re_judge_is_named_explicitly(corpus, monkeypatch, caplog):
    """The judged render equals the bytes it replaced -> say so plainly."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run",
        _fake_judge_run(report=REPORT_ALL_AGREED, html=PRE_JUDGE_HTML),
    )
    with caplog.at_level("INFO", logger="MCP.tools.pipeline_tools"):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))

    assert result["chapters_render_compared"] == 1
    assert result["chapters_render_unchanged"] == 1
    line = _summary_line(caplog)
    assert "IDEMPOTENT RE-JUDGE" in line
    assert "already applied these levels during semantik_conversion" in line
    assert "changed nothing on disk" in line
    # and the copy-back still happened (the backup is the PRE-COPY-BACK file)
    bak = corpus.semantik_out / "ch01_accessible.html.prejudge.bak"
    assert bak.read_text(encoding="utf-8") == PRE_JUDGE_HTML


def test_idempotent_signal_never_fires_when_the_render_changed(
    corpus, monkeypatch, caplog
):
    """The signal is MEASURED, not inferred — a real re-level must not claim
    idempotency even when the tallies look similar."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_MIXED)
    )  # default html differs from the fixture's pre-judge bytes
    with caplog.at_level("INFO", logger="MCP.tools.pipeline_tools"):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))

    assert result["chapters_render_compared"] == 1
    assert result["chapters_render_unchanged"] == 0
    line = _summary_line(caplog)
    assert "IDEMPOTENT" not in line
    assert "render CHANGED on all 1 chapter(s)" in line


def test_no_prior_render_reports_the_comparison_as_unavailable(
    corpus, monkeypatch, caplog
):
    """No pre-judge file to compare against -> never claim idempotency."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    (corpus.semantik_out / "ch01_accessible.html").unlink()
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_ALL_AGREED)
    )
    with caplog.at_level("INFO", logger="MCP.tools.pipeline_tools"):
        result = json.loads(_run(pipeline_tools._run_heading_judge(
            pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
        )))

    assert result["chapters_render_compared"] == 0
    assert result["chapters_render_unchanged"] == 0
    line = _summary_line(caplog)
    assert "IDEMPOTENT" not in line
    # the only honest statement available, verbatim
    assert (
        "0 level changes (layout already judged or judge agreed) — "
        "render-vs-pre-judge comparison unavailable" in line
    )


def test_legacy_report_changes_reconstructed_from_judgments(
    corpus, monkeypatch
):
    """A judge runtime predating the ``changed`` key must not be counted as
    all-changed — reconstruct from its own per-heading from/to map."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_LEGACY)
    )
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    )))
    assert result["total_applied"] == 2
    assert result["total_changed"] == 1
    assert result["total_agreed"] == 1
    assert result["level_transitions"] == {"3->2": 1}


def test_capture_rationale_separates_verdicts_from_level_changes(
    corpus, monkeypatch, recording_capture
):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    monkeypatch.setattr(
        subprocess, "run", _fake_judge_run(report=REPORT_MIXED)
    )
    _run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE"
    ))
    cap = recording_capture.calls[0]
    rationale = cap["rationale"]
    assert "5 verdict(s) applied" in rationale
    assert "3 CHANGED a level (3->2 x3)" in rationale
    assert "2 AGREED with the pre-existing level" in rationale
    assert cap["hj_applied"] == 5
    assert cap["hj_changed"] == 3
    assert cap["hj_agreed"] == 2
    assert "changed=3, agreed=2" in cap["decision"]


def test_change_counts_helper_handles_a_report_with_no_signal():
    """Neither the keys nor a judgments map -> report the honest unknown as
    zero CHANGED rather than inventing re-levelings."""
    changed, agreed, transitions = pipeline_tools._heading_judge_change_counts(
        {"applied": 7}
    )
    assert (changed, agreed, transitions) == (0, 7, {})
