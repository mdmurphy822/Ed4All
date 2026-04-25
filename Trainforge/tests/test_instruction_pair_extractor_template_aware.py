"""Wave 81: template-aware instruction-pair extractors.

Wave 79 C added four Courseforge content-generator templates (procedure,
real_world_scenario, common_pitfall, problem_solution) which Wave 81
propagates through the chunker via ``data-cf-template-type``. This module
tests that the four new extractors fire on chunks of those types and emit
the expected pair shapes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.instruction_pair_extractor import (  # noqa: E402
    METHOD_COMMON_PITFALL,
    METHOD_COMMON_PITFALL_MULTI_ARM,
    METHOD_PROBLEM_SOLUTION,
    METHOD_PROBLEM_SOLUTION_DPO,
    METHOD_PROCEDURE,
    METHOD_REAL_WORLD_SCENARIO,
    extract_from_common_pitfall,
    extract_from_problem_solution,
    extract_from_procedure,
    extract_from_real_world_scenario,
    run_extraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_chunk(**overrides):
    base = {
        "id": "chunk_001",
        "chunk_type": "explanation",
        "text": "Some text here.",
        "learning_outcome_refs": ["co-01"],
        "bloom_level": "understand",
        "difficulty": "foundational",
        "concept_tags": ["sample_topic"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# procedure
# ---------------------------------------------------------------------------


def test_procedure_chunk_emits_procedure_pair():
    chunk = _stub_chunk(
        id="proc_001",
        chunk_type="procedure",
        text=(
            "Procedure: Validate a graph against an SHACL shapes graph. "
            "When to use this procedure: pre-promotion check.\n\n"
            "Inputs: a data graph in any RDF serialization, a shapes "
            "graph, a SHACL processor.\n\n"
            "Steps: 1. Parse the shapes graph. 2. Parse the data graph. "
            "3. Invoke validation. 4. Inspect sh:conforms.\n\n"
            "Output: a SHACL validation report graph with sh:conforms.\n\n"
            "Worked Example: pyshacl -s shapes.ttl --advanced data.ttl "
            "produces Conforms: False with one Result entry."
        ),
    )
    pairs = extract_from_procedure(chunk)
    assert len(pairs) == 1
    p = pairs[0]
    md = p["metadata"]
    assert md["extraction_method"] == METHOD_PROCEDURE
    assert md["quality_score"] == 1.0
    assert md["has_inputs"] is True
    assert md["has_output_section"] is True
    assert md["has_worked_example"] is True
    # Instruction must reference inputs and output framing.
    assert "Given" in p["instruction"]
    # Output must contain the steps body.
    assert "Parse the shapes graph" in p["output"]


def test_procedure_chunk_without_steps_emits_nothing():
    chunk = _stub_chunk(
        chunk_type="procedure",
        text="Inputs: foo. Output: bar. (no procedure body)",
    )
    pairs = extract_from_procedure(chunk)
    assert pairs == []


# ---------------------------------------------------------------------------
# real_world_scenario
# ---------------------------------------------------------------------------


def test_real_world_scenario_chunk_emits_scenario_pair():
    chunk = _stub_chunk(
        id="rws_001",
        chunk_type="real_world_scenario",
        text=(
            "Scenario: At MetroHealth Network the clinical team needs "
            "patient referrals validated against shape graphs.\n\n"
            "Your Task: Author a SHACL-AF shape graph that materializes "
            "ex:effectivePCP and validates referral integrity.\n\n"
            "Approach: Use sh:TripleRule for the materialization. Use "
            "sh:condition to guard against missing primary-care-providers.\n\n"
            "Success Criteria: The rule materializes effectivePCP on every "
            "qualifying referral. Validation reports name the referral IRI."
        ),
    )
    pairs = extract_from_real_world_scenario(chunk)
    assert len(pairs) == 1
    p = pairs[0]
    md = p["metadata"]
    assert md["extraction_method"] == METHOD_REAL_WORLD_SCENARIO
    assert md["quality_score"] == 0.95
    assert md["has_success_criteria"] is True
    # Instruction must include both scenario context and Task framing.
    assert "MetroHealth" in p["instruction"]
    assert "Task:" in p["instruction"]
    # Output combines Approach + Success Criteria.
    assert "Use sh:TripleRule" in p["output"]
    assert "Success criteria:" in p["output"]


def test_real_world_scenario_without_task_emits_nothing():
    chunk = _stub_chunk(
        chunk_type="real_world_scenario",
        text="Scenario: a context. Approach: a plan. (no task section)",
    )
    assert extract_from_real_world_scenario(chunk) == []


# ---------------------------------------------------------------------------
# common_pitfall
# ---------------------------------------------------------------------------


def test_common_pitfall_emits_distinguish_and_multi_arm_pairs():
    chunk = _stub_chunk(
        id="cp_001",
        chunk_type="common_pitfall",
        section_heading="Negation in SHACL",
        text=(
            "Common Pitfall: treating sh:not as 'value not in list'. "
            "Intro paragraph.\n\n"
            "What looks like the right answer: I'll write sh:not [ "
            "sh:path :status ; sh:in (:Suspended :Closed :Frozen) ] "
            "and it should produce a clean negation.\n\n"
            "Why it's wrong: sh:not is shape-level, not value-level, so "
            "the report names the focus node but not the offending value, "
            "breaking diagnostic chains.\n\n"
            "The right approach: use sh:in with the allowed values, or "
            "sh:disjoint, or a SPARQL constraint. The validation report "
            "will name the offending value.\n\n"
            "Quick test: if you want the offender named, use a value-level "
            "constraint."
        ),
    )
    pairs = extract_from_common_pitfall(chunk)
    methods = [p["metadata"]["extraction_method"] for p in pairs]
    assert METHOD_COMMON_PITFALL in methods
    assert METHOD_COMMON_PITFALL_MULTI_ARM in methods
    # distinguish-style pair restates the misconception.
    distinguish = next(
        p
        for p in pairs
        if p["metadata"]["extraction_method"] == METHOD_COMMON_PITFALL
    )
    assert "Is this the right approach" in distinguish["instruction"]
    assert "use sh:in with the allowed values" in distinguish["output"]
    # multi-arm pair frames a situation + common mistake → asks for correct.
    multi = next(
        p
        for p in pairs
        if p["metadata"]["extraction_method"] == METHOD_COMMON_PITFALL_MULTI_ARM
    )
    assert "common mistake" in multi["instruction"]
    assert multi["metadata"]["situation"] == "Negation in SHACL"
    assert "sh:in" in multi["output"]


def test_common_pitfall_without_misconception_emits_nothing():
    chunk = _stub_chunk(
        chunk_type="common_pitfall",
        text=(
            "Why it's wrong: explanation goes here. The right approach: "
            "do this instead. (no misconception subsection)"
        ),
    )
    assert extract_from_common_pitfall(chunk) == []


# ---------------------------------------------------------------------------
# problem_solution
# ---------------------------------------------------------------------------


def test_problem_solution_emits_main_and_dpo_pairs():
    chunk = _stub_chunk(
        id="ps_001",
        chunk_type="problem_solution",
        text=(
            "Problem: Every paper must reference at least one author with "
            "the corresponding-author role. Authors live on author "
            "entities, not on the paper.\n\n"
            "Walkthrough: Identify the constraint scope is per-paper. "
            "Plan: use sh:targetClass with sh:sparql + FILTER NOT EXISTS. "
            "Execute: author the shape. Verify: validate against fixtures.\n\n"
            "Common Incorrect Approach: Use Core sh:class + sh:minCount on "
            "ex:author. This produces 'wrong-role' messages spread across "
            "every paper instead of per-paper 'no corresponding author'.\n\n"
            "Verification discipline: validate against three fixtures with "
            "expected report shapes."
        ),
    )
    pairs = extract_from_problem_solution(chunk)
    methods = [p["metadata"]["extraction_method"] for p in pairs]
    assert METHOD_PROBLEM_SOLUTION in methods
    assert METHOD_PROBLEM_SOLUTION_DPO in methods
    main = next(
        p
        for p in pairs
        if p["metadata"]["extraction_method"] == METHOD_PROBLEM_SOLUTION
    )
    assert "Every paper must reference" in main["instruction"]
    assert "sh:targetClass" in main["output"]
    assert main["metadata"]["has_verification"] is True
    # DPO pair carries chosen + rejected siblings.
    dpo = next(
        p
        for p in pairs
        if p["metadata"]["extraction_method"] == METHOD_PROBLEM_SOLUTION_DPO
    )
    assert "chosen" in dpo and "rejected" in dpo
    assert "FILTER NOT EXISTS" in dpo["chosen"]
    assert "Core sh:class" in dpo["rejected"]
    # output mirrors chosen for legacy SFT consumers.
    assert dpo["output"] == dpo["chosen"]


def test_problem_solution_without_walkthrough_emits_nothing():
    # No "Walkthrough" header anywhere in the text → nothing to lift.
    chunk = _stub_chunk(
        chunk_type="problem_solution",
        text="Problem: a problem statement only.",
    )
    assert extract_from_problem_solution(chunk) == []


def test_problem_solution_without_counter_example_omits_dpo_pair():
    chunk = _stub_chunk(
        chunk_type="problem_solution",
        text=(
            "Problem: a problem statement.\n\n"
            "Walkthrough: a walkthrough."
        ),
    )
    pairs = extract_from_problem_solution(chunk)
    methods = [p["metadata"]["extraction_method"] for p in pairs]
    assert METHOD_PROBLEM_SOLUTION in methods
    assert METHOD_PROBLEM_SOLUTION_DPO not in methods


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def test_run_extraction_dispatches_template_aware_methods(tmp_path):
    chunks = [
        _stub_chunk(
            id="proc_001",
            chunk_type="procedure",
            text=(
                "Procedure title.\n\n"
                "Inputs: an RDF graph and a SHACL shapes graph.\n\n"
                "Steps: 1. Parse. 2. Validate. 3. Inspect.\n\n"
                "Output: a validation report."
            ),
        ),
        _stub_chunk(
            id="rws_001",
            chunk_type="real_world_scenario",
            text=(
                "Scenario: Hospital triage requires referral integrity.\n\n"
                "Your Task: Author a shape graph.\n\n"
                "Approach: Use SHACL-AF rules to materialize derived "
                "attributes, then validate.\n\n"
                "Success Criteria: every referral has an effectivePCP."
            ),
        ),
        _stub_chunk(
            id="cp_001",
            chunk_type="common_pitfall",
            text=(
                "Pitfall intro.\n\n"
                "What looks like the right answer: misconception body.\n\n"
                "Why it's wrong: explanation.\n\n"
                "The right approach: correct path body."
            ),
        ),
        _stub_chunk(
            id="ps_001",
            chunk_type="problem_solution",
            text=(
                "Problem: a problem statement that goes here.\n\n"
                "Walkthrough: a multi-step walkthrough explaining how.\n\n"
                "Common Incorrect Approach: a counter-example explanation."
            ),
        ),
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    stats, pairs = run_extraction(
        chunks_path=chunks_path,
        output_dir=out_dir,
        course_code="TEST_001",
        capture=None,
    )
    methods = Counter(p["metadata"]["extraction_method"] for p in pairs)
    # Must see all four new methods plus the multi-arm + DPO siblings.
    expected = {
        METHOD_PROCEDURE,
        METHOD_REAL_WORLD_SCENARIO,
        METHOD_COMMON_PITFALL,
        METHOD_COMMON_PITFALL_MULTI_ARM,
        METHOD_PROBLEM_SOLUTION,
        METHOD_PROBLEM_SOLUTION_DPO,
    }
    assert expected <= set(methods.keys()), (
        f"missing methods: {expected - set(methods.keys())}; got {methods}"
    )
    # JSONL output exists.
    assert (out_dir / "instruction_pairs.jsonl").exists()


# ---------------------------------------------------------------------------
# rdf-shacl-551-2 corpus integration (after Wave 81 reclassification)
# ---------------------------------------------------------------------------


_CORPUS_PATH = (
    PROJECT_ROOT / "LibV2" / "courses" / "rdf-shacl-551-2" / "corpus"
    / "chunks.jsonl"
)


@pytest.mark.skipif(
    not _CORPUS_PATH.exists(),
    reason="rdf-shacl-551-2 corpus not present in this checkout",
)
def test_rdf_shacl_551_2_archive_yields_template_aware_methods(tmp_path):
    """After Wave 81 reclassification the rdf-shacl-551-2 archive should
    contain chunks of every Wave 79 C template type, and the extractor
    should fire at least four unique template-aware extraction methods on
    them (i.e. beyond the original 6 legacy methods).
    """
    out_dir = tmp_path / "out"
    stats, pairs = run_extraction(
        chunks_path=_CORPUS_PATH,
        output_dir=out_dir,
        course_code="RDFSHACL_551",
        capture=None,
    )
    template_methods = {
        METHOD_PROCEDURE,
        METHOD_REAL_WORLD_SCENARIO,
        METHOD_COMMON_PITFALL,
        METHOD_COMMON_PITFALL_MULTI_ARM,
        METHOD_PROBLEM_SOLUTION,
        METHOD_PROBLEM_SOLUTION_DPO,
    }
    fired = set(stats.pairs_by_method.keys()) & template_methods
    # The corpus must reclassify chunks; if reclassification hasn't run yet
    # we expect zero firings — the Wave 81 retroactive script populates this.
    assert len(fired) >= 4, (
        f"Wave 81 expected ≥4 unique template-aware extraction methods on "
        f"the rdf-shacl-551-2 archive; got {fired}. Run "
        f"scripts/wave81_reclassify_chunks.py first."
    )
