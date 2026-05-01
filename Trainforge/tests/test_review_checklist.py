"""Wave 137d-1: tests for the review checklist helper + drafting CLI integration.

Eight tests pin the contract:

1. ``test_seed_is_deterministic_for_same_curie_and_content`` —
   identical inputs produce identical seeds.
2. ``test_seed_changes_when_content_changes`` — any content edit
   reshuffles.
3. ``test_always_two_anchors_present`` — definitions[0] +
   usage_examples[0][1] always emitted in the rendered checklist.
4. ``test_sample_three_picked_from_remaining_pool`` — exactly 3
   candidates beyond the anchors when pool >= 3.
5. ``test_truncation_at_80_chars_with_ellipsis`` — long sentences
   are truncated to 80 chars with the ``...`` suffix.
6. ``test_padding_when_pool_under_three`` — emits placeholder lines
   when the entry has only definitions[0].
7. ``test_checklist_appended_to_drafting_stdout`` — the drafting
   CLI's stdout contains the REVIEW CHECKLIST header (when the
   validator passes).
8. ``test_backfill_yaml_slicer_skips_checklist`` —
   ``_extract_yaml_payload_from_drafting_stdout`` parses cleanly
   when the checklist is interposed between YAML and next-steps.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    SurfaceFormData,
)
from Trainforge.scripts._review_checklist import (  # noqa: E402
    _pick_sample,
    _seed_for,
    _truncate,
    build_review_checklist,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_entry(
    *,
    curie: str = "test:Foo",
    definitions: List[str] = None,
    usage_examples: List = None,
    reasoning_scenarios: List = None,
) -> SurfaceFormData:
    return SurfaceFormData(
        curie=curie,
        short_name=curie.split(":")[-1] if ":" in curie else curie,
        definitions=definitions if definitions is not None else [
            f"{curie} defines a baseline canonical concept for fixture-zero use.",
            f"{curie} formalises the second angle for fixture-one usage.",
            f"{curie} encodes the cardinality bound for fixture-two scenarios.",
            f"{curie} validates lexical pattern strings in fixture-three scope.",
            f"{curie} specifies hierarchy chains for fixture-four pages.",
        ],
        usage_examples=usage_examples if usage_examples is not None else [
            (f"How is {curie} used in fixture-zero context?",
             f"{curie} applies in fixture-zero like this canonical answer."),
            (f"What about {curie} in fixture-one context?",
             f"{curie} works in fixture-one as the alternate answer."),
            (f"Is {curie} viable in fixture-two scenario?",
             f"{curie} validates fixture-two as the third answer."),
        ],
        reasoning_scenarios=reasoning_scenarios if reasoning_scenarios is not None else [
            (f"Suppose fixture-A holds — what does {curie} imply?",
             f"{curie} implies fixture-A's reasoning conclusion."),
            (f"Suppose fixture-B holds — what does {curie} imply?",
             f"{curie} implies fixture-B's reasoning conclusion."),
        ],
        anchored_status="complete",
    )


# ----------------------------------------------------------------------
# 1. Determinism: same CURIE + content => same seed.
# ----------------------------------------------------------------------


def test_seed_is_deterministic_for_same_curie_and_content():
    entry_a = _make_entry()
    entry_b = _make_entry()
    assert _seed_for("test:Foo", entry_a) == _seed_for("test:Foo", entry_b)


# ----------------------------------------------------------------------
# 2. Determinism: content change reshuffles.
# ----------------------------------------------------------------------


def test_seed_changes_when_content_changes():
    entry_a = _make_entry()
    different_defs = list(entry_a.definitions)
    different_defs[0] = different_defs[0] + " (edited)"
    entry_b = _make_entry(definitions=different_defs)
    assert _seed_for("test:Foo", entry_a) != _seed_for("test:Foo", entry_b)


# ----------------------------------------------------------------------
# 3. Always-2 anchors present.
# ----------------------------------------------------------------------


def test_always_two_anchors_present():
    entry = _make_entry()
    rendered = build_review_checklist("test:Foo", entry)
    # definitions[0] anchor line.
    assert "definitions[0]:" in rendered
    # usage_examples[0][1] anchor line — the answer slot.
    assert "usage_examples[0][1] (answer):" in rendered
    # First definition content surfaces in the rendered checklist.
    assert "fixture-zero" in rendered
    # First usage_example answer surfaces in the rendered checklist.
    assert (
        "applies in fixture-zero" in rendered
        or "fixture-zero like" in rendered
    )


# ----------------------------------------------------------------------
# 4. Sample-3 picked from remaining pool.
# ----------------------------------------------------------------------


def test_sample_three_picked_from_remaining_pool():
    entry = _make_entry()
    seed = _seed_for("test:Foo", entry)
    sample = _pick_sample(entry, seed, k=3)
    # Pool is definitions[1:] (4) + usage_examples[1:] (2)
    # + reasoning_scenarios[*] (2) = 8 candidates; sample of 3.
    assert len(sample) == 3
    # Each tuple is (category, index, sentence).
    for category, idx, sentence in sample:
        assert category in (
            "definitions",
            "usage_examples",
            "reasoning_scenarios",
        )
        assert isinstance(idx, int)
        assert isinstance(sentence, str)
        # Anchors are excluded — definitions index 0 + usage_examples
        # index 0 must NOT appear.
        if category == "definitions":
            assert idx >= 1
        if category == "usage_examples":
            assert idx >= 1


# ----------------------------------------------------------------------
# 5. Truncation at 80 chars.
# ----------------------------------------------------------------------


def test_truncation_at_80_chars_with_ellipsis():
    short = "short string"
    assert _truncate(short) == short
    long_text = "a" * 200
    truncated = _truncate(long_text)
    assert truncated.endswith("...")
    # 80 chars + "..." = 83 total.
    assert len(truncated) == 83
    # Boundary: a string of exactly 80 chars passes through unchanged.
    eighty = "a" * 80
    assert _truncate(eighty) == eighty


# ----------------------------------------------------------------------
# 6. Padding when pool < 3.
# ----------------------------------------------------------------------


def test_padding_when_pool_under_three():
    """Entry with only definitions[0] + usage_examples[0] (no extras)
    => sample pool is empty; checklist must pad with placeholder
    lines so the operator still sees the always-2 anchors and can
    decide y/n/e/q."""
    entry = SurfaceFormData(
        curie="test:Bar",
        short_name="bar",
        definitions=[
            "test:Bar defines a single canonical concept anchor only."
        ],
        usage_examples=[
            (
                "How is test:Bar used in the fixture context?",
                "test:Bar applies as the only canonical anchored answer.",
            )
        ],
        anchored_status="complete",
    )
    rendered = build_review_checklist("test:Bar", entry)
    # Three placeholder lines emitted.
    placeholder_count = rendered.count("(no candidate available)")
    assert placeholder_count == 3
    # Always-2 anchors still rendered.
    assert "definitions[0]:" in rendered
    assert "usage_examples[0][1] (answer):" in rendered


# ----------------------------------------------------------------------
# 7. Drafting CLI: stdout contains REVIEW CHECKLIST header.
# ----------------------------------------------------------------------


class _FakeProvider:
    """Provider stub mirroring the drafting CLI's expected surface."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: List[Dict[str, Any]] = []
        self._oa_client = self

    def chat_completion(
        self,
        messages,
        *,
        max_tokens: int = 800,
        temperature: float = 0.4,
        decision_metadata=None,
        extra_payload=None,
    ) -> str:
        self.calls.append({"messages": messages})
        return self._response_text


def test_checklist_appended_to_drafting_stdout():
    """When the validator passes (post-Wave 137a-3 means PROVENANCE
    must be authored to a non-PENDING_REVIEW value to avoid Rule 4
    blocking emit), the drafting CLI must include REVIEW CHECKLIST
    in stdout. We stub the validator to bypass Rule 4 so we can
    measure the actual checklist-emit path."""
    from Trainforge.scripts import draft_form_data_entry as cli

    target_curie = "sh:datatype"
    payload = {
        "short_name": "datatype",
        "definitions": [
            f"{target_curie} defines literal datatype constraints alpha bravo charlie "
            f"delta echo foxtrot for property shapes here.",
            f"{target_curie} describes node membership predicates golf hotel india "
            f"juliet kilo lima per fixture spec text.",
            f"{target_curie} constrains cardinality bounds mike november oscar "
            f"papa quebec romeo across model arrows here.",
            f"{target_curie} validates lexical pattern strings sierra tango uniform "
            f"victor whiskey xray pinning syntax checks here.",
            f"{target_curie} specifies hierarchy chains yankee zulu morpheme nibble "
            f"wombat zebra over inheritance branches here.",
            f"{target_curie} requires IRI mapping yields apricot banana cucumber "
            f"durian elderberry fig keyed lookups here.",
            f"{target_curie} applies value-space conformance grouse heron iguana "
            f"jaguar koala lion within typed scopes here.",
        ],
        "usage_examples": [
            [
                f"How does {target_curie} relate to fixture facet {i}? Show the "
                f"surface-form pattern in a concrete SHACL fixture body.",
                f"In a property shape with sh:path ex:bar_{i}, {target_curie} "
                f"applies the fixture-{i} surface form pattern thoroughly.",
            ]
            for i in range(7)
        ],
        "comparison_targets": [],
        "reasoning_scenarios": [],
        "pitfalls": [],
        "combinations": [],
    }
    fake_provider = _FakeProvider(json.dumps(payload))

    # Stub validator so we exercise the checklist-emit path directly
    # (Rule 4 / PENDING_REVIEW would otherwise short-circuit at exit 3).
    def _passing_validator(*args, **kwargs):
        return {"passed": True, "missing_curies": [], "incomplete_curies": [],
                "invalid_status_curies": [], "content_violations": []}

    out = io.StringIO()
    err = io.StringIO()
    with patch.object(cli, "_build_provider", return_value=fake_provider), \
         patch.object(cli, "validate_form_data_contract", _passing_validator), \
         redirect_stdout(out), redirect_stderr(err):
        rc = cli.main([
            "--curie", target_curie,
            "--course-code", "rdf-shacl-551-2",
            "--provider", "local",
            "--force-overwrite",
        ])

    assert rc == 0, f"expected rc=0 with stubbed validator; got {rc}"
    stdout_text = out.getvalue()
    assert "REVIEW CHECKLIST" in stdout_text
    assert f"REVIEW CHECKLIST for {target_curie}" in stdout_text
    # Checklist sits between YAML and NEXT STEPS — both bookends must
    # exist in stdout.
    assert "forms:" in stdout_text
    assert "NEXT STEPS" in stdout_text
    yaml_pos = stdout_text.index("forms:")
    checklist_pos = stdout_text.index("REVIEW CHECKLIST")
    next_steps_pos = stdout_text.index("NEXT STEPS")
    assert yaml_pos < checklist_pos < next_steps_pos


# ----------------------------------------------------------------------
# 8. Backfill slicer compatibility — checklist between YAML and next-steps.
# ----------------------------------------------------------------------


def test_backfill_yaml_slicer_skips_checklist():
    """Wave 137d-1 update to the slicer: when a REVIEW CHECKLIST block
    sits between the YAML and the NEXT STEPS comment, the YAML still
    parses cleanly — the slicer cuts at the FIRST of either header so
    yaml.safe_load only sees the well-formed YAML head."""
    from Trainforge.scripts.backfill_form_data import (
        _extract_yaml_payload_from_drafting_stdout,
    )

    yaml_block = (
        "family: rdf_shacl\n"
        "forms:\n"
        "  sh:datatype:\n"
        "    short_name: datatype\n"
        "    anchored_status: complete\n"
        "    definitions:\n"
        "      - 'sh:datatype defines a canonical literal datatype constraint here.'\n"
        "    usage_examples:\n"
        "      - - 'How is sh:datatype used in property shapes?'\n"
        "        - 'sh:datatype applies as a literal datatype anchor here.'\n"
    )
    checklist_block = (
        "============================================================\n"
        "REVIEW CHECKLIST for sh:datatype\n"
        "============================================================\n"
        "\n"
        "Always review (load-bearing):\n"
        "  [ ] definitions[0]: ...\n"
        "  [ ] usage_examples[0][1] (answer): ...\n"
        "============================================================\n"
    )
    next_steps = "# NEXT STEPS\n# 0. Replace PENDING_REVIEW.\n"
    full_stdout = f"{yaml_block}\n{checklist_block}\n{next_steps}"

    payload = _extract_yaml_payload_from_drafting_stdout(full_stdout)
    assert isinstance(payload, dict)
    assert payload.get("family") == "rdf_shacl"
    assert "sh:datatype" in payload.get("forms", {})

    # Same slicer should still work when the checklist is absent
    # (legacy Wave 136d output shape).
    legacy_stdout = f"{yaml_block}\n{next_steps}"
    legacy_payload = _extract_yaml_payload_from_drafting_stdout(legacy_stdout)
    assert legacy_payload.get("family") == "rdf_shacl"
