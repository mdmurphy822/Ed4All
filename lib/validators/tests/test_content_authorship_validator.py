"""Tests for the anti-silent-template guard (ContentAuthorshipValidator)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.validators.content_authorship import (
    ContentAuthorshipValidator,
    _build_content_authorship,
)


def _prov(tmp_path: Path, **over) -> str:
    doc = {
        "schema_version": "1.0",
        "generator_mode": "template_deterministic",
        "template_fallback_fired": True,
        "courseforge_provider": "",
        "two_pass_enabled": False,
        "agent_dispatch_enabled": False,
        "llm_authoring_intended": False,
        "allow_template_emitter": False,
        "weeks_prepared": 12,
        "page_count": 65,
        "course_code": "DEMO_101",
    }
    doc.update(over)
    p = tmp_path / "content_generation_provenance.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def test_template_fired_with_llm_intent_blocks(tmp_path, monkeypatch):
    monkeypatch.delenv("COURSEFORGE_ALLOW_TEMPLATE_EMITTER", raising=False)
    path = _prov(tmp_path, llm_authoring_intended=True)
    res = ContentAuthorshipValidator().validate({"provenance_path": path})
    assert res.passed is False
    assert any(i.code == "TEMPLATE_EMITTER_IN_REAL_RUN" for i in res.issues)
    assert res.issues[0].severity == "critical"


def test_template_fired_no_intent_warns_but_passes(tmp_path, monkeypatch):
    for v in ("ED4ALL_AGENT_DISPATCH", "COURSEFORGE_TWO_PASS",
              "COURSEFORGE_ALLOW_TEMPLATE_EMITTER"):
        monkeypatch.delenv(v, raising=False)
    path = _prov(tmp_path, llm_authoring_intended=False)
    res = ContentAuthorshipValidator().validate({"provenance_path": path})
    assert res.passed is True
    assert any(i.code == "TEMPLATE_EMITTER_NO_LLM_INTENT" for i in res.issues)
    assert res.issues[0].severity == "warning"


def test_allow_flag_downgrades_to_warning(tmp_path):
    path = _prov(tmp_path, llm_authoring_intended=True,
                 allow_template_emitter=True)
    res = ContentAuthorshipValidator().validate({"provenance_path": path})
    assert res.passed is True
    assert any(i.code == "TEMPLATE_EMITTER_ALLOWED" for i in res.issues)


def test_live_env_reasserts_intent(tmp_path, monkeypatch):
    # Stale provenance says no intent, but the live run has dispatch on.
    monkeypatch.setenv("ED4ALL_AGENT_DISPATCH", "true")
    monkeypatch.delenv("COURSEFORGE_ALLOW_TEMPLATE_EMITTER", raising=False)
    path = _prov(tmp_path, llm_authoring_intended=False)
    res = ContentAuthorshipValidator().validate({"provenance_path": path})
    assert res.passed is False


def test_llm_provider_passes(tmp_path):
    path = _prov(tmp_path, generator_mode="llm_provider",
                 template_fallback_fired=False, llm_authoring_intended=True)
    res = ContentAuthorshipValidator().validate({"provenance_path": path})
    assert res.passed is True
    assert res.issues == []


def test_missing_provenance_passes():
    res = ContentAuthorshipValidator().validate({})
    assert res.passed is True
    assert res.issues == []


def test_inline_fields_block(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_ALLOW_TEMPLATE_EMITTER", raising=False)
    res = ContentAuthorshipValidator().validate({
        "generator_mode": "template_deterministic",
        "template_fallback_fired": True,
        "llm_authoring_intended": True,
    })
    assert res.passed is False


def test_builder_reads_explicit_path(tmp_path):
    path = _prov(tmp_path, llm_authoring_intended=True)
    inputs, missing = _build_content_authorship(
        {"content_generation": {
            "content_generation_provenance_path": path,
            "template_fallback_fired": True,
            "generator_mode": "template_deterministic",
        }},
        {},
    )
    assert missing == []
    assert inputs.get("provenance_path") == path
    res = ContentAuthorshipValidator().validate(inputs)
    assert res.passed is False


def test_full_runtime_degradation_blocks(tmp_path, monkeypatch):
    # Provider was wired (llm intent) but authored zero pages -> the writer
    # flips generator_mode to template_deterministic + template_fallback_fired.
    monkeypatch.delenv("COURSEFORGE_ALLOW_TEMPLATE_EMITTER", raising=False)
    path = _prov(
        tmp_path,
        generator_mode="template_deterministic",
        template_fallback_fired=True,
        llm_authoring_intended=True,
        courseforge_provider="anthropic",
        llm_authored_pages=0,
        template_fallback_pages=8,
        llm_authoring_degraded=True,
    )
    res = ContentAuthorshipValidator().validate({"provenance_path": path})
    assert res.passed is False
    assert any(i.code == "TEMPLATE_EMITTER_IN_REAL_RUN" for i in res.issues)


def test_partial_runtime_degradation_warns(tmp_path):
    # Some pages LLM-authored, some fell back -> warn, don't block.
    path = _prov(
        tmp_path,
        generator_mode="llm_provider",
        template_fallback_fired=False,
        llm_authoring_intended=True,
        courseforge_provider="anthropic",
        llm_authored_pages=5,
        template_fallback_pages=3,
        llm_authoring_degraded=True,
    )
    res = ContentAuthorshipValidator().validate({"provenance_path": path})
    assert res.passed is True
    assert any(i.code == "PARTIAL_TEMPLATE_FALLBACK" for i in res.issues)


def test_builder_globs_export_dir(tmp_path):
    export = tmp_path / "project-theta"
    content_dir = export / "03_content_development"
    content_dir.mkdir(parents=True)
    _prov(export, llm_authoring_intended=True)  # writes into export/
    inputs, missing = _build_content_authorship(
        {"content_generation": {"content_dir": str(content_dir)}}, {},
    )
    assert "provenance_path" in inputs
    assert ContentAuthorshipValidator().validate(inputs).passed is False
