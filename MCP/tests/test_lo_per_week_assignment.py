"""Verify LO-per-week scoping in ``_generate_course_content``.

Investigation Issue 12: each week previously got the **full** terminal
objectives list prepended (``list(terminal_objectives) + week_chapter_cos``
at pipeline_tools.py:1360). Downstream consequence: every chunk parsed
from the emitted pages carried ``learning_outcome_refs`` pointing at
every objective, which inflated the ``derived-from-objective`` edge
count from ~60 (the natural floor: chunks × ~1 LO each) to 896
(64 chunks × ~14 LOs) on the OLSR_201 corpus.

Post-remediation contract:

  * Each week's emitted pages carry at most ``ceil(N/D)`` terminals +
    up-to-2 chapter objectives per week, NOT the full terminal list.
  * For a 4-week course with 8 terminals, week 1 must NOT see all 8.
  * Distinct weeks must see distinct (overlap-allowed-but-bounded) LO
    sets so the prerequisite / over-assignment signal recovers.

Test strategy: emit a course with a synthetic objectives file carrying
8 terminals + 4 chapter objectives across 4 weeks. Parse the emitted
HTML for ``data-cf-objective-id`` attributes and assert the per-week
distribution is scoped, not bulk-prepended.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


COURSE_CODE = "LOTEST_101"


# Minimal DART HTML so parse_dart_html_files produces topics for each
# week. The 4-week allocator splits topics round-robin; we supply 4
# distinct section blocks so every week can bind to its own topic.
_DART_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>Sample Textbook</title></head>
<body>
<main id="main-content" role="main">
<section id="s1" aria-labelledby="s1h">
  <h2 id="s1h">Cellular Respiration Overview</h2>
  <p>Cellular respiration is the metabolic process by which cells convert
  biochemical energy from nutrients into adenosine triphosphate. This
  process takes place in the mitochondria of eukaryotic cells and the
  cytoplasm of prokaryotes.</p>
  <p>The overall reaction involves glucose and oxygen as inputs and
  produces carbon dioxide, water, and ATP energy molecules as outputs.</p>
</section>
<section id="s2" aria-labelledby="s2h">
  <h2 id="s2h">Glycolysis Pathway</h2>
  <p>Glycolysis is the initial ten-step pathway that breaks down one
  glucose molecule into two pyruvate molecules, producing a net gain of
  two ATP and two NADH in the process. Glycolysis occurs in the cytoplasm
  and does not require oxygen directly.</p>
  <p>Each step of glycolysis is catalyzed by a specific enzyme and
  regulated by feedback inhibition of phosphofructokinase.</p>
</section>
<section id="s3" aria-labelledby="s3h">
  <h2 id="s3h">Citric Acid Cycle</h2>
  <p>The citric acid cycle, also known as the Krebs cycle, oxidizes acetyl
  coenzyme A to carbon dioxide in a series of eight enzymatic steps
  within the mitochondrial matrix. The cycle generates NADH, FADH2, and
  a small amount of GTP.</p>
  <p>Intermediates from this cycle feed into amino acid synthesis and
  other biosynthetic pathways.</p>
</section>
<section id="s4" aria-labelledby="s4h">
  <h2 id="s4h">Electron Transport Chain</h2>
  <p>The electron transport chain uses a series of protein complexes
  embedded in the inner mitochondrial membrane to pump protons across
  the membrane and establish a proton gradient that drives ATP synthesis
  through chemiosmosis.</p>
  <p>Oxygen serves as the final electron acceptor, combining with
  electrons and protons to form water.</p>
</section>
</main>
</body>
</html>
"""


_DATA_CF_OBJECTIVE_RE = re.compile(
    r'data-cf-objective-id="([^"]+)"'
)


def _write_objectives_file(path: Path) -> None:
    """8 terminals + 4 chapter objectives for a 4-week course."""
    doc = {
        "terminal_objectives": [
            {
                "id": f"TO-{i:02d}",
                "statement": f"Apply concept {i} to solve related problems.",
                "bloom_level": "apply",
                "bloom_verb": "apply",
                "cognitive_domain": "procedural",
            }
            for i in range(1, 9)
        ],
        "chapter_objectives": [
            {
                "id": f"CO-{i:02d}",
                "statement": f"Describe topic {i} and related concepts.",
                "bloom_level": "understand",
                "bloom_verb": "describe",
                "cognitive_domain": "conceptual",
            }
            for i in range(1, 5)
        ],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


@pytest.fixture
def pipeline_registry(monkeypatch, tmp_path):
    staging_root = tmp_path / "cf_inputs"
    staging_root.mkdir()
    monkeypatch.setattr(pipeline_tools, "COURSEFORGE_INPUTS", staging_root)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)

    project_id = "PROJ-LOTEST-01"
    project_path = tmp_path / "Courseforge" / "exports" / project_id
    (project_path / "03_content_development").mkdir(parents=True)
    (project_path / "01_learning_objectives").mkdir()

    objectives_path = project_path / "01_learning_objectives" / "course_objectives.json"
    _write_objectives_file(objectives_path)

    config = {
        "project_id": project_id,
        "course_name": COURSE_CODE,
        "duration_weeks": 4,
        "objectives_path": str(objectives_path),
    }
    (project_path / "project_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    staging_dir = staging_root / "WF-LOTEST-01"
    staging_dir.mkdir()
    (staging_dir / "textbook.html").write_text(_DART_HTML, encoding="utf-8")

    return {
        "tools": _build_tool_registry(),
        "project_id": project_id,
        "project_path": project_path,
        "staging_dir": staging_dir,
    }


def _collect_per_week_objective_ids(project_path: Path) -> dict:
    """Return ``{week_num: set(objective_ids_seen_on_any_page)}``."""
    per_week: dict = defaultdict(set)
    content_root = project_path / "03_content_development"
    for week_dir in sorted(content_root.glob("week_*")):
        week_num = int(week_dir.name.split("_")[1])
        for html_file in week_dir.glob("*.html"):
            body = html_file.read_text(encoding="utf-8")
            for match in _DATA_CF_OBJECTIVE_RE.findall(body):
                per_week[week_num].add(match)
    return per_week


class TestPerWeekLoScoping:
    def test_no_week_carries_all_eight_terminals(self, pipeline_registry):
        fx = pipeline_registry
        asyncio.run(fx["tools"]["generate_course_content"](
            project_id=fx["project_id"],
            staging_dir=str(fx["staging_dir"]),
        ))
        per_week = _collect_per_week_objective_ids(fx["project_path"])
        all_terminals = {f"TO-{i:02d}" for i in range(1, 9)}

        assert per_week, "No pages parsed — generation may have failed"
        for week_num, ids in per_week.items():
            terminals_seen = ids & all_terminals
            assert terminals_seen != all_terminals, (
                f"Week {week_num} carries ALL {len(all_terminals)} "
                f"terminal objectives — over-assignment not fixed. "
                f"Seen: {sorted(terminals_seen)}"
            )

    def test_each_week_has_at_least_one_terminal(self, pipeline_registry):
        fx = pipeline_registry
        asyncio.run(fx["tools"]["generate_course_content"](
            project_id=fx["project_id"],
            staging_dir=str(fx["staging_dir"]),
        ))
        per_week = _collect_per_week_objective_ids(fx["project_path"])
        all_terminals = {f"TO-{i:02d}" for i in range(1, 9)}

        for week_num, ids in per_week.items():
            assert ids & all_terminals, (
                f"Week {week_num} has no terminal objective — page "
                f"objective gate will fail."
            )

    def test_weeks_scoped_to_at_most_four_terminals(self, pipeline_registry):
        """With 8 terminals / 4 weeks, each week should hold ≤ ceil(8/4)+1
        terminals (i.e., ≤ 3). Allow +1 slack for boundary rounding."""
        fx = pipeline_registry
        asyncio.run(fx["tools"]["generate_course_content"](
            project_id=fx["project_id"],
            staging_dir=str(fx["staging_dir"]),
        ))
        per_week = _collect_per_week_objective_ids(fx["project_path"])
        all_terminals = {f"TO-{i:02d}" for i in range(1, 9)}

        for week_num, ids in per_week.items():
            n_terms = len(ids & all_terminals)
            # ceil(8/4) = 2, plus 1 week's chapter fallback slack => 3
            assert n_terms <= 3, (
                f"Week {week_num} holds {n_terms} terminal objectives — "
                f"expected ≤ 3 given 8 terminals across 4 weeks."
            )

    def test_weeks_cover_distinct_terminal_slices(self, pipeline_registry):
        """At least two weeks must claim distinct terminal sets so the
        prerequisite signal has any chance of firing downstream."""
        fx = pipeline_registry
        asyncio.run(fx["tools"]["generate_course_content"](
            project_id=fx["project_id"],
            staging_dir=str(fx["staging_dir"]),
        ))
        per_week = _collect_per_week_objective_ids(fx["project_path"])
        all_terminals = {f"TO-{i:02d}" for i in range(1, 9)}

        week_sets: list = [
            tuple(sorted(ids & all_terminals))
            for ids in per_week.values()
        ]
        assert len(set(week_sets)) > 1, (
            "All weeks claim identical terminal sets — LO distribution "
            "has collapsed to a single bucket."
        )

    def test_total_terminals_referenced_across_weeks(self, pipeline_registry):
        """The union of all weeks' terminal references should cover the
        majority of the 8 supplied terminals. Allows a little under-
        coverage (some terminals may not map to any available topic),
        but requires at least 4 of 8."""
        fx = pipeline_registry
        asyncio.run(fx["tools"]["generate_course_content"](
            project_id=fx["project_id"],
            staging_dir=str(fx["staging_dir"]),
        ))
        per_week = _collect_per_week_objective_ids(fx["project_path"])
        all_terminals = {f"TO-{i:02d}" for i in range(1, 9)}

        covered: set = set()
        for ids in per_week.values():
            covered.update(ids & all_terminals)
        assert len(covered) >= 4, (
            f"Only {len(covered)}/8 terminals referenced anywhere — "
            f"LO scoping over-pruned."
        )


class TestDerivedFromObjectiveEdgeFloor:
    """The product of (chunks_per_page) × (pages_per_week × weeks) × (LOs
    per chunk) should land in a natural range, not balloon.

    Proxy for the concept graph's derived-from-objective edge count:
    multiplying the number of ``data-cf-objective-id`` attribute
    instances across pages gives an upper bound on the edges that
    downstream Trainforge will emit.
    """

    def test_total_objective_attributes_bounded(self, pipeline_registry):
        fx = pipeline_registry
        asyncio.run(fx["tools"]["generate_course_content"](
            project_id=fx["project_id"],
            staging_dir=str(fx["staging_dir"]),
        ))
        # With 4 weeks × 5 pages × ~3 LOs/page = ~60 attr instances max.
        # The old bulk-prepend path would produce 4 × 5 × ~10 = 200.
        content_root = fx["project_path"] / "03_content_development"
        total_attrs = 0
        for html_file in content_root.rglob("*.html"):
            body = html_file.read_text(encoding="utf-8")
            total_attrs += len(_DATA_CF_OBJECTIVE_RE.findall(body))
        # Natural ceiling: 4 weeks × 5 pages × (2 TOs + 2 COs) = 80,
        # plus some per-activity / self-check refs. Bound at 120 to
        # detect regressions; the old bulk behavior would exceed 200.
        assert total_attrs <= 120, (
            f"Observed {total_attrs} data-cf-objective-id attributes — "
            f"expected ≤ 120 with per-week scoping. Regression to "
            f"bulk-prepend behavior?"
        )
        assert total_attrs >= 8, (
            f"Observed {total_attrs} data-cf-objective-id attributes — "
            f"too few for a 4-week course."
        )
