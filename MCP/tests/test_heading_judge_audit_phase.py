"""heading_judge phase — post-judge AUDIT fold + targeted RE-JUDGE wiring.

Covers the ``SEMANTIK_HEADING_JUDGE_AUDIT`` (default ON) audit fold into the
phase output and the ``SEMANTIK_HEADING_JUDGE_REJUDGE`` (default OFF) targeted
re-judge:

* audit default-ON folds ``audit_report_path`` / ``chapters_flagged`` /
  ``chapters_collapsed`` / ``inconsistent_signature_count`` into the output;
* re-judge OFF → NO re-run subprocess (byte-identical judge invocation set);
* re-judge ON with a mocked judge → re-invoked on ONLY the flagged chapter,
  bounded to one attempt, with the reduced-window env, and a re-audit runs.

No GPU / no network / no real seat / no SemantiK import — every subprocess is
mocked. No course slugs / book references — everything lives in tmp_path.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402

_AUDIT_MODULE = "semantik_structure.glmocr.heading_judge_audit_standalone"
_JUDGE_MODULE = "semantik_structure.glmocr.heading_judge_standalone"

REPORT = {"applied": 3, "changed": 1, "agreed": 2, "kept": 0, "clamped": 0,
          "dropped": 0, "windows": 1, "n_pending": 3, "n_headings": 6,
          "unjudged": 0, "kept_judged": 0, "transitions": {"3->2": 1}}


def _run(coro):
    return asyncio.run(coro)


def _stem(cmd):
    layout = Path(cmd[3])
    return layout.name[: -len(".glmocr_layout.json")]


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for stem in ("ch01", "ch02"):
        (corpus_dir / f"{stem}.glmocr_layout.json").write_bytes(
            json.dumps({"pages": [{"page_no": 1, "regions": []}]}).encode()
        )
        (corpus_dir / f"{stem}.glmocr_escalations.jsonl").write_text(
            json.dumps({"reason": "heading_level_pending", "region_index": 1})
            + "\n", encoding="utf-8",
        )
    home = tmp_path / "home"
    out = home / "semantik-output"
    out.mkdir(parents=True)
    for stem in ("ch01", "ch02"):
        (out / f"{stem}_accessible.html").write_text("<html>OLD</html>",
                                                     encoding="utf-8")
    monkeypatch.setenv("ED4ALL_HOME", str(home))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("SEMANTIK_GLMOCR_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("SEMANTIK_PYTHON", raising=False)
    monkeypatch.delenv("SEMANTIK_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    return SimpleNamespace(corpus_dir=corpus_dir, semantik_out=out)


def _make_fake(judge_calls, audit_calls, *, first_flagged=()):
    """A subprocess.run fake: judge cmds write judged artifacts + record their
    stem + env; audit cmds write a controllable report (first audit → flagged,
    later → empty) so a re-judge can be observed to CLEAR the flag."""

    def fake_run(cmd, **kwargs):
        cmd = list(cmd)
        if _AUDIT_MODULE in cmd:
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            flagged = list(first_flagged) if not audit_calls else []
            audit_calls.append({"flagged": flagged})
            report = {
                "audit_schema_version": 1,
                "n_chapters": 2, "chapters": [],
                "book": {"level_distribution": {},
                         "incomplete_chapters": [],
                         "collapsed_chapters": [
                             {"stem": s, "reasons": ["level_collapse_single"]}
                             for s in flagged],
                         "inconsistent_signatures": [
                             {"signature": "summary", "occurrences": []}]},
                "flagged_chapters": flagged,
            }
            (out_dir / "heading_judge_audit.json").write_text(
                json.dumps(report), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")
        # judge cmd
        out_dir = Path(cmd[cmd.index("--out") + 1])
        stem = _stem(cmd)
        judge_calls.append({"stem": stem, "env": kwargs.get("env") or {}})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.heading_judgments.json").write_text(
            json.dumps(REPORT), encoding="utf-8")
        (out_dir / f"{stem}.corrected_layout.json").write_text(
            json.dumps({"region_provenance": [], "heading_tree": []}),
            encoding="utf-8")
        (out_dir / f"{stem}.glmocr_escalations.jsonl").write_text(
            json.dumps({"reason": "heading_level_judged", "region_index": 1})
            + "\n", encoding="utf-8")
        (out_dir / f"{stem}_accessible.html").write_text(
            "<html>JUDGED</html>", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    return fake_run


def test_audit_default_on_folds_fields(corpus, monkeypatch):
    judge_calls, audit_calls = [], []
    monkeypatch.setattr(subprocess, "run",
                        _make_fake(judge_calls, audit_calls,
                                   first_flagged=["ch01"]))
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE")))
    assert result["success"] is True
    assert len(audit_calls) == 1                     # audit ran once
    assert result["audit_report_path"] is not None
    assert result["chapters_flagged"] == 1           # ch01 flagged
    assert result["chapters_collapsed"] == 1
    assert result["inconsistent_signature_count"] == 1
    # re-judge OFF (default) → exactly the two initial judge calls, no re-run.
    assert sorted(c["stem"] for c in judge_calls) == ["ch01", "ch02"]


def test_audit_opt_out_byte_identical(corpus, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_AUDIT", "0")
    judge_calls, audit_calls = [], []
    monkeypatch.setattr(subprocess, "run",
                        _make_fake(judge_calls, audit_calls,
                                   first_flagged=["ch01"]))
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE")))
    assert result["success"] is True
    assert audit_calls == []                          # audit never ran
    assert result["audit_report_path"] is None
    assert result["chapters_flagged"] == 0
    assert sorted(c["stem"] for c in judge_calls) == ["ch01", "ch02"]


def test_rejudge_opt_in_runs_only_flagged_chapter(corpus, monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_REJUDGE", "1")
    judge_calls, audit_calls = [], []
    monkeypatch.setattr(subprocess, "run",
                        _make_fake(judge_calls, audit_calls,
                                   first_flagged=["ch01"]))
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE")))
    assert result["success"] is True
    # two audit runs: the initial audit + one re-audit after the re-run.
    assert len(audit_calls) == 2
    assert audit_calls[0]["flagged"] == ["ch01"]
    # judge invoked: ch01 + ch02 (initial) + ch01 ONLY (re-judge, bounded to 1).
    stems = [c["stem"] for c in judge_calls]
    assert sorted(stems) == ["ch01", "ch01", "ch02"]
    rejudge_calls = [c for c in judge_calls if c["stem"] == "ch01"]
    assert len(rejudge_calls) == 2                    # initial + one re-run
    # the re-run carries the reduced-window env (smaller windows + fresh key).
    rerun_env = rejudge_calls[-1]["env"]
    assert int(rerun_env["SEMANTIK_HEADING_JUDGE_MAX_TOKENS"]) < 30000
    assert int(rerun_env["SEMANTIK_HEADING_JUDGE_CTX_BUDGET"]) < 31500
    # the re-audit cleared the flag → delta reflected in the folded output.
    assert result["chapters_flagged"] == 0


def test_rejudge_off_never_re_runs(corpus, monkeypatch):
    # default OFF: even with a flagged chapter, no re-run subprocess fires.
    judge_calls, audit_calls = [], []
    monkeypatch.setattr(subprocess, "run",
                        _make_fake(judge_calls, audit_calls,
                                   first_flagged=["ch01", "ch02"]))
    result = json.loads(_run(pipeline_tools._run_heading_judge(
        pdf_paths=str(corpus.corpus_dir), course_name="TESTCOURSE")))
    assert result["success"] is True
    assert len(audit_calls) == 1                      # audit only, no re-audit
    assert sorted(c["stem"] for c in judge_calls) == ["ch01", "ch02"]
    assert result["chapters_flagged"] == 2            # left flagged, reported
