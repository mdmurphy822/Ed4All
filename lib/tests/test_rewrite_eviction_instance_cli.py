"""I4 stage 2 — CLI + routing wiring for --block-ids / --pages eviction.

Stage 1 (``--blocks``) evicts every rewrite-cached block of a TYPE. Stage 2
adds two ADDITIVE scopes:

* ``--block-ids`` — exact block-instance IDs → ``target_block_instance_ids``.
* ``--pages`` — page/module identifiers → ``target_page_ids``.

This file pins the CLI parse + build-params plumbing, the dry-run plan
annotation, the ``create_textbook_pipeline`` forward, and the
routing-parity contract (the legacy in-memory dict and the YAML
``inputs_from`` block must agree on the two new params, and both flow through
``_get_phase_param_routing``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.commands.run import (  # noqa: E402
    _build_workflow_params,
    _parse_csv_tokens,
)
from cli.main import cli  # noqa: E402


# --------------------------------------------------------------------------- #
# _parse_csv_tokens — no static enum; runtime artifacts validated downstream.
# --------------------------------------------------------------------------- #


class TestParseCsvTokens:
    def test_returns_list(self):
        assert _parse_csv_tokens("p1#a_0,p2#b_1") == ["p1#a_0", "p2#b_1"]

    def test_strips_and_dedupes(self):
        assert _parse_csv_tokens(" week_01 , week_02 , week_01 ,") == [
            "week_01",
            "week_02",
        ]

    def test_empty_returns_none(self):
        assert _parse_csv_tokens(None) is None
        assert _parse_csv_tokens("") is None
        assert _parse_csv_tokens(",,") is None

    def test_no_enum_validation(self):
        """Arbitrary tokens are accepted at parse time (validated in the
        rewrite handler against the real outline block set)."""
        assert _parse_csv_tokens("anything#goes_9") == ["anything#goes_9"]


# --------------------------------------------------------------------------- #
# _build_workflow_params
# --------------------------------------------------------------------------- #


def _base_params(**over):
    kw = dict(
        corpus="x.pdf",
        course_name="X_101",
        weeks=None,
        no_assessments=False,
        assessment_count=50,
        bloom_levels="remember,understand,apply,analyze",
        priority="normal",
        objectives_path=None,
    )
    kw.update(over)
    return _build_workflow_params("textbook_to_course", **kw)


class TestBuildWorkflowParams:
    def test_threads_instance_and_page_ids(self):
        params = _base_params(
            target_block_instance_ids=["p1#a_0"],
            target_page_ids=["week_01"],
        )
        assert params["target_block_instance_ids"] == ["p1#a_0"]
        assert params["target_page_ids"] == ["week_01"]

    def test_absent_when_unset(self):
        params = _base_params()
        assert "target_block_instance_ids" not in params
        assert "target_page_ids" not in params


# --------------------------------------------------------------------------- #
# CLI dry-run end-to-end plumbing.
# --------------------------------------------------------------------------- #


class TestCliDryRun:
    def _invoke(self, *extra):
        runner = CliRunner()
        return runner.invoke(
            cli,
            [
                "run",
                "textbook-to-course",
                "--corpus",
                "inputs/pdfs/fake.pdf",
                "--course-name",
                "X_101",
                *extra,
                "--dry-run",
                "--json",
            ],
        )

    def test_block_ids_and_pages_land_in_params(self):
        result = self._invoke(
            "--block-ids", "week_01_content_01#concept_a_0",
            "--pages", "week_02",
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["params"]["target_block_instance_ids"] == [
            "week_01_content_01#concept_a_0"
        ]
        assert payload["params"]["target_page_ids"] == ["week_02"]
        # Top-level summary fields.
        assert payload.get("block_ids_filter") == [
            "week_01_content_01#concept_a_0"
        ]
        assert payload.get("pages_filter") == ["week_02"]

    def test_rewrite_phase_annotated_filtered(self):
        result = self._invoke("--pages", "week_03")
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        rewrite = next(
            p for p in payload["phases"]
            if p["name"] == "content_generation_rewrite"
        )
        assert rewrite["status"] == "FILTERED"
        assert rewrite["pages_filter"] == ["week_03"]

    def test_default_no_filters_no_summary_keys(self):
        result = self._invoke()
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        # Byte-identical default: no eviction filters on params/summary.
        assert "target_block_instance_ids" not in payload["params"]
        assert "target_page_ids" not in payload["params"]
        assert "block_ids_filter" not in payload
        assert "pages_filter" not in payload

    def test_flags_appear_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--block-ids" in result.output
        assert "--pages" in result.output


# --------------------------------------------------------------------------- #
# create_textbook_pipeline forwards the new params onto persisted workflow
# params (the runtime seam, not just dry-run).
# --------------------------------------------------------------------------- #


class TestCreateTextbookPipelineForward:
    @pytest.mark.asyncio
    async def test_forwards_instance_and_page_ids(self, tmp_path, monkeypatch):
        from MCP.tools import pipeline_tools
        from MCP.tools.pipeline_tools import create_textbook_pipeline

        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)

        captured: dict = {}

        async def _fake_create_workflow_impl(**kwargs):
            captured.update(kwargs)
            return json.dumps({"success": True, "workflow_id": "WF-I4S2"})

        with patch(
            "MCP.tools.orchestrator_tools.create_workflow_impl",
            new=_fake_create_workflow_impl,
        ):
            raw = await create_textbook_pipeline(
                pdf_paths=str(pdf),
                course_name="I4S2_101",
                target_block_instance_ids=["p1#a_0"],
                target_page_ids=["week_01"],
            )

        assert json.loads(raw).get("success") is True
        forwarded = json.loads(captured["params"])
        assert forwarded.get("target_block_instance_ids") == ["p1#a_0"]
        assert forwarded.get("target_page_ids") == ["week_01"]

    @pytest.mark.asyncio
    async def test_absent_when_unset(self, tmp_path, monkeypatch):
        from MCP.tools import pipeline_tools
        from MCP.tools.pipeline_tools import create_textbook_pipeline

        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")
        monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)

        captured: dict = {}

        async def _fake_create_workflow_impl(**kwargs):
            captured.update(kwargs)
            return json.dumps({"success": True, "workflow_id": "WF-NO"})

        with patch(
            "MCP.tools.orchestrator_tools.create_workflow_impl",
            new=_fake_create_workflow_impl,
        ):
            raw = await create_textbook_pipeline(
                pdf_paths=str(pdf), course_name="NO_101",
            )

        assert json.loads(raw).get("success") is True
        forwarded = json.loads(captured["params"])
        assert "target_block_instance_ids" not in forwarded
        assert "target_page_ids" not in forwarded


# --------------------------------------------------------------------------- #
# Routing parity — legacy dict AND YAML inputs_from must carry the new params.
# --------------------------------------------------------------------------- #


class TestRoutingParity:
    def test_legacy_dict_carries_new_params(self):
        from MCP.core.workflow_runner import _LEGACY_PHASE_PARAM_ROUTING

        routing = _LEGACY_PHASE_PARAM_ROUTING["content_generation_rewrite"]
        assert routing["target_block_instance_ids"] == (
            "workflow_params", "target_block_instance_ids",
        )
        assert routing["target_page_ids"] == (
            "workflow_params", "target_page_ids",
        )

    def test_yaml_routing_matches_legacy(self):
        """The YAML-sourced routing must equal the legacy dict for the
        rewrite phase (the same contract the meta-schema parity test pins,
        scoped here to the new params)."""
        from MCP.core.workflow_runner import (
            _LEGACY_PHASE_PARAM_ROUTING,
            _get_phase_param_routing,
        )

        yaml_routing = _get_phase_param_routing("content_generation_rewrite")
        legacy = _LEGACY_PHASE_PARAM_ROUTING["content_generation_rewrite"]
        assert yaml_routing == legacy
        assert yaml_routing["target_block_instance_ids"] == (
            "workflow_params", "target_block_instance_ids",
        )
        assert yaml_routing["target_page_ids"] == (
            "workflow_params", "target_page_ids",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
