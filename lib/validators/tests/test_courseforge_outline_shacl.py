"""Phase 4 Wave N1 Subtask 13 — tests for CourseforgeOutlineShaclValidator.

Verifies the validator's input handling, SHACL dispatch against
schemas/context/courseforge_v1.shacl.ttl, and the
GateResult.action mapping (block / regenerate / pass) per
sub-plan §A. Fixtures construct Block-derived JSON-LD payloads
matching ``Block.to_jsonld_entry()``'s minimal shape (Phase 2
Subtask 13) so the SHACL ``cfshapes:BlockShape`` (sh:targetClass
ed4all:Block) fires deterministically.

Skips the entire module when SHACL extras aren't installed; the
validator's deps-missing path is exercised in
``test_handles_missing_shacl_deps_via_runtime_path`` separately by
mocking ``_ensure_deps`` to raise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add repo root for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Skip the entire module when SHACL extras aren't installed; the
# validator's runtime-deps-missing path is exercised separately by
# monkey-patching ``_ensure_deps`` to raise (so the deps-missing
# behavior is covered even when extras ARE installed).
pytest.importorskip(
    "pyld",
    reason="pyld required for SHACL tests; install via the `shacl` extras.",
)
pytest.importorskip(
    "pyshacl",
    reason="pyshacl required for SHACL tests; install via the `shacl` extras.",
)
pytest.importorskip("rdflib", reason="rdflib comes with pyshacl.")

from lib.validators.courseforge_outline_shacl import (  # noqa: E402
    CourseforgeOutlineShaclValidator,
    DEFAULT_SHAPES_PATH,
)


# --------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------- #


def _make_block_payload(
    *,
    block_id: str = "week_01_overview#concept_intro_0",
    block_type: str = "concept",
    sequence: int = 0,
    content_hash: str | None = None,
) -> dict:
    """Build a Block JSON-LD entry per ``Block._minimal_block_jsonld``.

    Carries the audit fields (``blockId`` / ``blockType`` / ``sequence``)
    that ``cfshapes:BlockShape`` requires. The validator auto-injects
    ``@type: "Block"`` when a payload carries ``blockId`` so the
    sh:targetClass fires.
    """
    payload: dict = {
        "blockId": block_id,
        "blockType": block_type,
        "sequence": sequence,
    }
    if content_hash is not None:
        payload["contentHash"] = content_hash
    return payload


def _make_well_formed_blocks() -> list[dict]:
    """Return a small list of canonical, well-formed Block payloads."""
    return [
        _make_block_payload(
            block_id="week_01_overview#concept_intro_0",
            block_type="concept",
            sequence=0,
        ),
        _make_block_payload(
            block_id="week_01_overview#example_demo_1",
            block_type="example",
            sequence=1,
        ),
        _make_block_payload(
            block_id="week_01_overview#assessment_item_quiz_2",
            block_type="assessment_item",
            sequence=2,
        ),
    ]


def _wrap_block_in_html(payload: dict) -> str:
    """Embed a payload in a JSON-LD <script> tag (rewrite-tier shape).

    The validator's _coerce_block_payloads passes string entries
    through ``_extract_jsonld_blocks`` (the same helper
    ``PageObjectivesShaclValidator`` uses to scrape JSON-LD blocks
    out of generated HTML pages).
    """
    body = json.dumps(payload)
    return (
        "<!DOCTYPE html><html><head>"
        f'<script type="application/ld+json">{body}</script>'
        "</head><body><h1>Test page</h1></body></html>"
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_passes_well_formed_outline_blocks():
    """A list of well-formed Block payloads conforms — no violations,
    ``passed=True``, ``action=None``, ``score=1.0``."""
    v = CourseforgeOutlineShaclValidator()
    blocks = _make_well_formed_blocks()

    result = v.validate({"blocks": blocks})

    assert result.passed, f"Well-formed blocks rejected; issues: {result.issues}"
    assert result.action is None, (
        f"No violations should yield action=None; got {result.action!r}"
    )
    assert result.issues == []
    assert result.score == 1.0
    assert result.gate_id == "courseforge_outline_shacl"


def test_critical_violation_returns_action_block():
    """A SHACL critical violation (sh:Violation severity) maps to
    ``action="block"`` per Phase 3 §A: structural misses cannot be
    fixed by re-rolling the outline tier.

    Trigger: blockType outside the canonical 16-value enum fires
    ``cfshapes:BlockShape`` ``sh:in`` constraint with default severity
    sh:Violation.
    """
    v = CourseforgeOutlineShaclValidator()
    bad = _make_block_payload(
        block_id="week_01_overview#bogus_0",
        block_type="NOT_A_REAL_TYPE",
        sequence=0,
    )

    result = v.validate({"blocks": [bad]})

    assert not result.passed, "Bogus blockType should fail validation."
    assert result.action == "block", (
        f"sh:Violation must map to action='block'; got {result.action!r}"
    )
    assert any(i.severity == "critical" for i in result.issues), (
        f"At least one critical issue expected; got "
        f"{[(i.severity, i.code) for i in result.issues]}"
    )


def test_warning_violation_returns_action_regenerate(monkeypatch):
    """A SHACL warning-only run maps to ``action="regenerate"`` so the
    router re-rolls the outline tier.

    Synthesised via a payload that triggers the validator's
    ``_decide_action(critical=0, warning>0)`` branch directly. Since the
    canonical Phase-2 BlockShape uses default severity (Violation /
    critical) for every constraint, we patch the ShaclViolation severity
    routing to surface a warning-only result without forking the shape
    file. Mirrors the public-API test at
    ``test_shacl_runner.test_shacl_violation_to_gate_issue_carries_code_prefix``
    by exercising the action-mapping branch in isolation.
    """
    from lib.validators import courseforge_outline_shacl as mod

    # Patch run_shacl to return a synthetic warning-only result so we
    # can pin the action="regenerate" branch without minting a
    # warning-severity SHACL shape (which would be a separate plan
    # item — sub-plan §A reserves warning severity for shapes that
    # haven't reached graduation yet).
    from lib.validators.shacl_runner import ShaclViolation

    def _fake_run_shacl(shapes_path, graph):
        return False, [
            ShaclViolation(
                focus_node="https://ed4all.dev/data/block/test#0",
                path="https://ed4all.dev/ns/courseforge/v1#contentHash",
                severity="warning",
                message="WB-001: contentHash recommended for cache-keying.",
                source_shape="https://ed4all.dev/shapes/cf#BlockShape",
                source_constraint_component=(
                    "http://www.w3.org/ns/shacl#PatternConstraintComponent"
                ),
            )
        ]

    monkeypatch.setattr(mod, "run_shacl", _fake_run_shacl)

    v = CourseforgeOutlineShaclValidator()
    blocks = _make_well_formed_blocks()
    result = v.validate({"blocks": blocks})

    # Warning-only run still passes (no critical issues), but the
    # action signals the router to re-roll.
    assert result.passed, "Warning-only run should still pass (no critical)."
    assert result.action == "regenerate", (
        f"Warning-only run must map to action='regenerate'; "
        f"got {result.action!r}"
    )
    assert any(i.severity == "warning" for i in result.issues), (
        f"At least one warning issue expected; got "
        f"{[(i.severity, i.code) for i in result.issues]}"
    )


def test_handles_str_content_via_html_extraction():
    """HTML strings inside ``inputs['blocks']`` are scraped via the
    existing _extract_jsonld_blocks helper (the rewrite-tier
    Block.content shape — see Phase 3.5 inter_tier_gates shape-dispatch).

    A well-formed Block embedded in a JSON-LD <script> block inside an
    HTML envelope MUST validate just like the bare-dict path.
    """
    v = CourseforgeOutlineShaclValidator()
    block = _make_block_payload(
        block_id="week_01_overview#concept_intro_0",
        block_type="concept",
        sequence=0,
    )
    html = _wrap_block_in_html(block)

    result = v.validate({"blocks": [html]})

    assert result.passed, (
        f"Well-formed HTML-wrapped block rejected; issues: {result.issues}"
    )
    assert result.action is None
    assert result.issues == []


def test_handles_dict_content_directly():
    """Bare-dict entries in ``inputs['blocks']`` are validated as-is.

    Mirrors the canonical workflow-runner-emit shape: the Phase 3.5
    ``_run_inter_tier_validation`` and ``_run_post_rewrite_validation``
    helpers project Block instances onto dict entries via
    ``_block_to_snake_case_entry`` (or ``Block.to_jsonld_entry``) and
    pass the resulting list directly through the validator's
    ``inputs['blocks']`` channel.
    """
    v = CourseforgeOutlineShaclValidator()
    blocks = _make_well_formed_blocks()

    result = v.validate({"blocks": blocks})

    assert result.passed
    assert result.action is None
    assert all(isinstance(b, dict) for b in blocks)


def test_no_violations_returns_pass_action():
    """Empty payload list and well-formed payloads BOTH yield
    ``action=None`` — the validator only sets a router-consumable
    action when there's something to act on.

    Two cases:
      1. Empty list -> no SHACL run, ``passed=True``, no action.
      2. Well-formed payloads -> SHACL conforms, ``passed=True``, no
         action.
    """
    v = CourseforgeOutlineShaclValidator()

    # Case 1: empty list.
    r_empty = v.validate({"blocks": []})
    assert r_empty.passed
    assert r_empty.action is None
    assert r_empty.issues == []
    assert r_empty.score == 1.0

    # Case 2: well-formed payloads.
    r_ok = v.validate({"blocks": _make_well_formed_blocks()})
    assert r_ok.passed
    assert r_ok.action is None
    assert r_ok.issues == []


# --------------------------------------------------------------------- #
# Supplementary coverage — input-shape edge cases.
# --------------------------------------------------------------------- #


def test_missing_blocks_input_returns_block_action():
    """No ``blocks`` and no ``blocks_path`` -> ``action="block"``.

    The validator can't make a no-action decision when it has nothing
    to validate; it has to surface a ``passed=False`` outcome so the
    workflow runner sees the input-shape regression.
    """
    v = CourseforgeOutlineShaclValidator()
    result = v.validate({})

    assert not result.passed
    assert result.action == "block"
    assert len(result.issues) == 1
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"


def test_blocks_path_jsonl_input_loads_payloads(tmp_path):
    """``blocks_path`` pointing at a JSONL file deserialises one
    payload per non-empty line, then runs the same SHACL pipeline.
    """
    blocks_path = tmp_path / "blocks.jsonl"
    payloads = _make_well_formed_blocks()
    blocks_path.write_text(
        "\n".join(json.dumps(p) for p in payloads) + "\n",
        encoding="utf-8",
    )

    v = CourseforgeOutlineShaclValidator()
    result = v.validate({"blocks_path": str(blocks_path)})

    assert result.passed, f"JSONL load rejected; issues: {result.issues}"
    assert result.issues == []


def test_shapes_path_default_resolves_to_canonical_file():
    """The validator's default shapes_path points at the canonical
    multi-shape SHACL file (BlockShape + TouchShape + cohort)."""
    assert DEFAULT_SHAPES_PATH.exists(), (
        f"Canonical SHACL file missing at {DEFAULT_SHAPES_PATH}."
    )
    assert DEFAULT_SHAPES_PATH.name == "courseforge_v1.shacl.ttl"


def test_shacl_deps_missing_emits_warning_passes(monkeypatch):
    """When pyld/pyshacl/rdflib are unavailable, the validator emits a
    single warning issue with ``passed=True`` and no action — matches
    Phase 4 Subtask 8's embedding-extras opt-out pattern.
    """
    from lib.validators import courseforge_outline_shacl as mod
    from lib.validators.shacl_runner import ShaclDepsMissing

    def _raise(*_args, **_kwargs):
        raise ShaclDepsMissing("pyshacl not importable in this env")

    monkeypatch.setattr(mod, "_ensure_deps", _raise)

    v = CourseforgeOutlineShaclValidator()
    result = v.validate({"blocks": _make_well_formed_blocks()})

    assert result.passed
    assert result.action is None
    assert len(result.issues) == 1
    assert result.issues[0].severity == "warning"
    assert result.issues[0].code == "SHACL_DEPS_MISSING"
