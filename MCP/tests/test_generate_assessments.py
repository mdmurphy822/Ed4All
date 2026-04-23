"""Worker β — ``_generate_assessments`` unit tests.

Verifies the Trainforge-execution tool runs the full corpus pipeline
(chunks + typed-edge graph + misconceptions + assessments) against a
packaged IMSCC, per the Wave Pipeline contract
(``plans/pipeline-execution-fixes/contracts.md`` §2).

Uses a minimal hand-built IMSCC fixture with enough content to trigger
multi-chunk chunking + typed-edge extraction. No network, no subprocess.
The tool is registered via a closure inside ``register_pipeline_tools``;
we reach it by passing a capturing MCP stand-in and invoking the
coroutine directly.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

COURSE_CODE = "TESTBETA_101"

# Rich fixture HTML: multiple sections with <strong> key terms and an
# explicit misconception paragraph. Big enough to produce >= 5 chunks
# post-chunking + >= 3 typed edges across >= 2 rule types.
_PAGE_OVERVIEW = """<!DOCTYPE html>
<html lang="en"><head><title>Photosynthesis: Overview</title></head>
<body><main id="main-content" role="main">
<h1>Photosynthesis Overview</h1>
<section><h2>What is Photosynthesis?</h2>
<p><strong>Photosynthesis</strong> is the biological process by which plants,
algae, and some bacteria convert light energy into chemical energy stored as
<strong>glucose</strong>. This fundamental process sustains nearly all life on
Earth by producing the oxygen we breathe and forming the base of most food
webs. A common misconception is that plants get their food from the soil. In
reality, plants produce their own food through photosynthesis; soil only
provides water and minerals. Students often think plants eat dirt, but that is
not how plant nutrition works.</p>
</section>
<section><h2>Chlorophyll and Light Capture</h2>
<p><strong>Chlorophyll</strong> is a green pigment found in chloroplasts that
absorbs light energy most effectively in the red and blue portions of the
visible spectrum. Chlorophyll is a pigment that enables light capture. Without
chlorophyll, photosynthesis could not occur. The green color of plants comes
directly from chlorophyll reflecting green wavelengths of light rather than
absorbing them.</p>
</section>
<section><h2>Why It Matters</h2>
<p>Photosynthesis is responsible for the oxygen in our atmosphere and forms
the base of nearly every food chain on Earth. Without photosynthesis, aerobic
life as we know it would not exist. Every breath you take depends on
photosynthesis, as does every meal you eat. The glucose produced stores
chemical energy that drives cellular respiration in nearly all organisms.</p>
</section>
</main></body></html>"""

_PAGE_STAGES = """<!DOCTYPE html>
<html lang="en"><head><title>Photosynthesis: The Two Stages</title></head>
<body><main id="main-content" role="main">
<h1>The Two Stages of Photosynthesis</h1>
<section><h2>Light-Dependent Reactions</h2>
<p>The <strong>light-dependent reactions</strong> occur in the thylakoid
membranes of the <strong>chloroplast</strong>. During this stage, chlorophyll
absorbs photons of light and uses that energy to split water molecules. The
splitting of water releases oxygen as a byproduct and generates
<strong>ATP</strong> and <strong>NADPH</strong>, which carry chemical energy
to the next stage of photosynthesis. Photosystem II initiates electron
transport while photosystem I regenerates NADPH. The thylakoid membrane is a
membrane that houses the photosystems.</p>
</section>
<section><h2>The Calvin Cycle</h2>
<p>The <strong>Calvin cycle</strong>, also known as the light-independent
reactions, takes place in the stroma of the chloroplast. The Calvin cycle uses
the ATP and NADPH produced in the light-dependent reactions to fix
atmospheric carbon dioxide into organic glucose molecules. The Calvin cycle
consists of three phases: carbon fixation, reduction, and regeneration of the
starting molecule ribulose-1,5-bisphosphate. Carbon fixation catalyzed by
RuBisCO is the most important step.</p>
</section>
<section><h2>Chloroplast Structure</h2>
<p>The <strong>chloroplast</strong> is a specialized organelle with a double
membrane surrounding an inner fluid called the stroma. Embedded within the
stroma are stacks of thylakoid membranes called grana. The thylakoid
membranes house chlorophyll and the protein complexes that carry out the
light-dependent reactions. The chloroplast is an organelle where
photosynthesis occurs. Chloroplasts originated from ancient cyanobacteria
through endosymbiosis.</p>
</section>
</main></body></html>"""


_MANIFEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="TESTBETA_101_manifest"
  xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
  xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.2.0</schemaversion>
    <lomimscc:lom><lomimscc:general><lomimscc:title><lomimscc:string
      language="en">TESTBETA 101: Photosynthesis Basics</lomimscc:string></lomimscc:title></lomimscc:general></lomimscc:lom>
  </metadata>
  <organizations>
    <organization identifier="ORG_1" structure="rooted-hierarchy">
      <item identifier="ROOT"><title>TESTBETA 101</title>
        <item identifier="ITEM_001" identifierref="RES_001"><title>Week 1 Overview</title></item>
        <item identifier="ITEM_002" identifierref="RES_002"><title>Week 1 Content</title></item>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="RES_001" type="webcontent" href="week_01/week_01_overview.html"><file href="week_01/week_01_overview.html"/></resource>
    <resource identifier="RES_002" type="webcontent" href="week_01/week_01_content_01_stages.html"><file href="week_01/week_01_content_01_stages.html"/></resource>
  </resources>
</manifest>"""


@pytest.fixture
def pipeline_registry(monkeypatch, tmp_path):
    """Build the internal tool registry against a tmp project root.

    Redirects ``_PROJECT_ROOT`` used by ``_generate_assessments`` so the
    tool writes to tmp paths (no pollution of the real exports/).
    """
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)

    tools: Dict[str, Callable] = _build_tool_registry()
    return tools, tmp_path


def _build_imscc(tmp_path: Path, project_id: str) -> Path:
    """Build a Courseforge-shaped project + IMSCC package at
    ``tmp_path/Courseforge/exports/{project_id}/05_final_package/*.imscc``
    so ``_generate_assessments`` can derive the project workspace from
    the IMSCC path.
    """
    project_dir = tmp_path / "Courseforge" / "exports" / project_id
    final_dir = project_dir / "05_final_package"
    final_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_config.json").write_text(
        json.dumps({
            "project_id": project_id,
            "course_name": COURSE_CODE,
            "duration_weeks": 1,
        }),
        encoding="utf-8",
    )
    imscc_path = final_dir / f"{COURSE_CODE}.imscc"
    with zipfile.ZipFile(imscc_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", _MANIFEST_XML)
        zf.writestr("week_01/week_01_overview.html", _PAGE_OVERVIEW)
        zf.writestr("week_01/week_01_content_01_stages.html", _PAGE_STAGES)
    return imscc_path


# ---------------------------------------------------------------------- #
# Core contract tests
# ---------------------------------------------------------------------- #


class TestGenerateAssessmentsContract:
    def test_produces_chunks_graph_misconceptions_assessments(
        self, pipeline_registry,
    ):
        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-001"
        imscc_path = _build_imscc(tmp_path, project_id)

        result = asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(imscc_path),
            question_count=6,
            bloom_levels="remember,understand,apply",
            objective_ids=f"{COURSE_CODE}_OBJ_1,{COURSE_CODE}_OBJ_2",
            project_id=project_id,
            domain="general",
            division="STEM",
        ))
        payload = json.loads(result)

        assert payload.get("success") is True, payload

        trainforge_dir = tmp_path / "Courseforge" / "exports" / project_id / "trainforge"
        assert trainforge_dir.exists(), f"trainforge/ not created in project workspace. payload={payload}"

        # 1. chunks.jsonl exists and is JSONL (one JSON object per line).
        chunks_path = Path(payload["chunks_path"])
        assert chunks_path.exists(), "chunks.jsonl missing"
        lines = [ln for ln in chunks_path.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1, "no chunk lines emitted"
        # Every line parses as JSON.
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert "id" in obj, f"chunk {i} missing id"
            assert "schema_version" in obj
            assert "source" in obj

        # 2. concept_graph_semantic.json exists and parses.
        semantic_path = Path(payload["concept_graph_path"])
        assert semantic_path.exists(), "concept_graph_semantic.json missing"
        graph = json.loads(semantic_path.read_text())
        assert graph.get("kind") == "concept_semantic"
        assert isinstance(graph.get("nodes"), list)
        assert isinstance(graph.get("edges"), list)

        # 3. misconceptions.json well-formed.
        mc_path = trainforge_dir / "graph" / "misconceptions.json"
        assert mc_path.exists(), "misconceptions.json missing"
        mc_doc = json.loads(mc_path.read_text())
        assert "misconceptions" in mc_doc
        assert isinstance(mc_doc["misconceptions"], list)
        # Every entity has the mc_<16 hex char> ID pattern.
        mc_id_re = re.compile(r"^mc_[0-9a-f]{16}$")
        for entity in mc_doc["misconceptions"]:
            assert mc_id_re.match(entity.get("id", "")), (
                f"misconception id doesn't match pattern: {entity.get('id')}"
            )
            assert entity.get("misconception")
            assert entity.get("correction")

        # 4. assessments.json is a single well-formed JSON document
        #    (NOT jsonl, NOT concatenated — the "Extra data" fix).
        assessments_path = Path(payload["assessments_path"])
        assert assessments_path.exists(), "assessments.json missing"
        # json.load must succeed on the whole file in one pass.
        with open(assessments_path) as f:
            assessment = json.load(f)
        assert assessment.get("assessment_id"), "no assessment_id"
        assert isinstance(assessment.get("questions"), list)
        assert assessment["question_count"] >= 1

    def test_assessments_json_is_well_formed_single_document(
        self, pipeline_registry,
    ):
        """Regression: the prior stub's assessments.json produced
        'Extra data' errors because it wrote metadata AFTER the main
        dump. Verify that json.load reads the whole file and there is
        no trailing non-whitespace content.
        """
        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-002"
        imscc_path = _build_imscc(tmp_path, project_id)

        asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(imscc_path),
            question_count=3,
            bloom_levels="understand",
            objective_ids=f"{COURSE_CODE}_OBJ_1",
            project_id=project_id,
        ))

        assessments_path = (
            tmp_path / "Courseforge" / "exports" / project_id
            / "trainforge" / "assessments.json"
        )
        raw = assessments_path.read_text()
        doc = json.loads(raw)
        assert isinstance(doc, dict)
        # Sanity: re-serialize and confirm the original parsed to a dict
        # (nothing after the closing brace). strict=False via json.loads
        # on the string — any trailing data raises.
        re_parsed = json.loads(raw)
        assert re_parsed == doc

    def test_honors_question_count_and_bloom_levels(self, pipeline_registry):
        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-003"
        imscc_path = _build_imscc(tmp_path, project_id)

        result = asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(imscc_path),
            question_count=4,
            bloom_levels="remember,apply",
            objective_ids=f"{COURSE_CODE}_OBJ_1,{COURSE_CODE}_OBJ_2",
            project_id=project_id,
        ))
        payload = json.loads(result)
        assert payload.get("success") is True

        assessments_path = Path(payload["assessments_path"])
        assessment = json.loads(assessments_path.read_text())
        questions = assessment["questions"]
        # AssessmentGenerator may drop leak-flagged questions, so allow
        # the count to dip below 4 but cap it at question_count (it must
        # never over-produce).
        assert len(questions) <= 4, f"over-produced: {len(questions)}"
        # Every question bloom_level is in the requested set (leak-drop
        # may remove questions; whichever remain must match params).
        allowed = {"remember", "apply"}
        for q in questions:
            assert q.get("bloom_level") in allowed, q.get("bloom_level")

    def test_no_imscc_returns_error(self, pipeline_registry):
        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-004"
        # Create the project dir but no IMSCC.
        (tmp_path / "Courseforge" / "exports" / project_id).mkdir(
            parents=True, exist_ok=True,
        )

        result = asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(tmp_path / "nonexistent.imscc"),
            question_count=3,
            bloom_levels="understand",
            objective_ids=f"{COURSE_CODE}_OBJ_1",
            project_id=project_id,
        ))
        payload = json.loads(result)
        assert "error" in payload
        assert "IMSCC" in payload["error"] or "not found" in payload["error"].lower()

    def test_output_under_courseforge_project_workspace(self, pipeline_registry):
        """Contract: output lands at ``{project_workspace}/trainforge/``.

        Verifies my decision to colocate the trainforge corpus with the
        Courseforge export dir (so Worker γ can locate + byte-copy it
        into LibV2 without a cross-tree lookup).
        """
        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-005"
        imscc_path = _build_imscc(tmp_path, project_id)

        result = asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(imscc_path),
            question_count=2,
            bloom_levels="understand",
            objective_ids=f"{COURSE_CODE}_OBJ_1",
            project_id=project_id,
        ))
        payload = json.loads(result)
        assert payload.get("success") is True

        trainforge_dir = Path(payload["trainforge_dir"])
        assert trainforge_dir.name == "trainforge"
        # It must live directly under the project dir.
        assert trainforge_dir.parent.name == project_id
        # Corpus + graph subdirs from CourseProcessor.
        assert (trainforge_dir / "corpus" / "chunks.jsonl").exists()
        assert (trainforge_dir / "graph" / "concept_graph_semantic.json").exists()
        assert (trainforge_dir / "graph" / "misconceptions.json").exists()
        assert (trainforge_dir / "assessments.json").exists()
        assert (trainforge_dir / "manifest.json").exists()

    def test_derives_workspace_from_imscc_path(self, pipeline_registry):
        """When no project_id kwarg is passed, the tool derives the
        workspace from imscc_path.parent.parent."""
        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-006"
        imscc_path = _build_imscc(tmp_path, project_id)

        result = asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(imscc_path),
            question_count=2,
            bloom_levels="understand",
            objective_ids=f"{COURSE_CODE}_OBJ_1",
            # project_id deliberately omitted.
        ))
        payload = json.loads(result)
        assert payload.get("success") is True
        trainforge_dir = Path(payload["trainforge_dir"])
        assert trainforge_dir.parent.name == project_id

    def test_runs_under_integration_test_strict_flags(
        self, pipeline_registry, monkeypatch,
    ):
        """Integration test sets the full strict-opt-in matrix. Verify
        that _generate_assessments still succeeds against a reference
        fixture under those flags (i.e., the strict-chunk-validation +
        strict-decision-validation paths don't break the handler).

        Consumes the committed ``reference_week_01`` HTML fixtures,
        which carry the JSON-LD shape Courseforge emits post-Worker-α.
        """
        for key in (
            "TRAINFORGE_CONTENT_HASH_IDS",
            "TRAINFORGE_SCOPE_CONCEPT_IDS",
            "TRAINFORGE_PRESERVE_LO_CASE",
            "TRAINFORGE_VALIDATE_CHUNKS",
            "TRAINFORGE_ENFORCE_CONTENT_TYPE",
            "TRAINFORGE_STRICT_EVIDENCE",
            "TRAINFORGE_SOURCE_PROVENANCE",
            "DECISION_VALIDATION_STRICT",
        ):
            monkeypatch.setenv(key, "true")

        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-strict"
        project_dir = tmp_path / "Courseforge" / "exports" / project_id
        final_dir = project_dir / "05_final_package"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Build the IMSCC from the committed reference_week_01 HTMLs.
        ref_week = PROJECT_ROOT / "tests" / "fixtures" / "pipeline" / "reference_week_01"
        assert ref_week.exists(), ref_week
        manifest_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<manifest identifier="TESTBETA_manifest" '
            'xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1">',
            '<resources>',
        ]
        html_files = sorted(ref_week.glob("*.html"))
        for i, h in enumerate(html_files, 1):
            rel = f"week_01/{h.name}"
            manifest_parts.append(
                f'<resource identifier="RES_{i:03d}" type="webcontent" href="{rel}">'
                f'<file href="{rel}"/></resource>'
            )
        manifest_parts.append("</resources></manifest>")
        imscc_path = final_dir / f"{COURSE_CODE}.imscc"
        with zipfile.ZipFile(imscc_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("imsmanifest.xml", "\n".join(manifest_parts))
            for h in html_files:
                zf.writestr(f"week_01/{h.name}", h.read_text(encoding="utf-8"))

        result = asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(imscc_path),
            question_count=6,
            bloom_levels="remember,understand,apply",
            objective_ids="CO-01,CO-02",
            project_id=project_id,
            domain="biology",
        ))
        payload = json.loads(result)
        assert payload.get("success") is True, payload

        # Chunk count + graph-edge count + misconception ID shape —
        # mirrors the integration test's Worker-β assertions.
        assert payload["chunks_count"] >= 3, payload

        graph = json.loads(Path(payload["concept_graph_path"]).read_text())
        assert len(graph.get("edges", [])) >= 3
        edge_types = {e["type"] for e in graph["edges"]}
        assert len(edge_types) >= 2, edge_types

        mc_path = Path(payload["misconceptions_path"])
        mc_doc = json.loads(mc_path.read_text())
        assert mc_doc["misconceptions"]
        mc_id_re = re.compile(r"^mc_[0-9a-f]{16}$")
        assert any(mc_id_re.match(m["id"]) for m in mc_doc["misconceptions"])

    def test_return_payload_shape(self, pipeline_registry):
        """Contract: returned JSON carries the paths the pipeline runner
        expects to thread downstream to LibV2 archival."""
        tools, tmp_path = pipeline_registry
        project_id = f"PROJ-{COURSE_CODE}-007"
        imscc_path = _build_imscc(tmp_path, project_id)

        result = asyncio.run(tools["generate_assessments"](
            course_id=COURSE_CODE,
            imscc_path=str(imscc_path),
            question_count=2,
            bloom_levels="understand",
            objective_ids=f"{COURSE_CODE}_OBJ_1",
            project_id=project_id,
        ))
        payload = json.loads(result)

        # Required keys for the return payload.
        for key in (
            "success", "assessment_id", "question_count",
            "chunks_path", "concept_graph_path", "misconceptions_path",
            "assessments_path", "trainforge_dir",
        ):
            assert key in payload, f"missing key {key!r}: {payload}"
        assert payload["success"] is True
        # Numeric counts are integers.
        assert isinstance(payload["question_count"], int)
        assert isinstance(payload.get("chunks_count", 0), int)
