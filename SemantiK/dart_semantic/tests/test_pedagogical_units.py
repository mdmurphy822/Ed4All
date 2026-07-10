"""Phase 1 — neutral pedagogical-unit boundary SoT (hoist, no behavior change).

The unit-boundary heuristic + the two pedagogy frozensets + ``ABSORB_MAX_RUN``
+ the ``resolve_unit_regroup_mode`` resolver were hoisted to
``dart_semantic.pedagogical_units`` so BOTH the assembler absorb and the
Stage-5e regroup import ONE contract cycle-free. CPU-only, no model load.
"""

from __future__ import annotations

from dart_semantic.pedagogical_units import (
    ABSORB_MAX_RUN,
    BODY_BEARING_COMPONENT_CLASSES,
    UNIT_START_PEDAGOGY_CLASSES,
    is_unit_boundary,
    resolve_unit_regroup_mode,
)
from dart_semantic.structure_graph import Region


def _region(kind, *, text=None, css_class=None, semantic_class=None) -> Region:
    payload = {}
    if text is not None:
        payload["text"] = text
    if css_class is not None:
        payload["css_class"] = css_class
    if semantic_class is not None:
        payload["semantic_class"] = semantic_class
    return Region(kind=kind, feature_block_indices=(0,), payload=payload)


def test_no_import_cycle():
    """The neutral module breaks the assembler <-> qwen_specialists cycle: all
    three importers load cleanly."""
    import importlib

    importlib.import_module("dart_semantic.qwen_specialists.block_resegment")
    importlib.import_module("dart_semantic.assembler.gold_shell_markup")
    importlib.import_module("dart_semantic.assembler.shell")

    # The resolver is re-exported from block_resegment for back-compat.
    from dart_semantic.qwen_specialists.block_resegment import (
        resolve_unit_regroup_mode as br_resolver,
    )

    assert br_resolver is resolve_unit_regroup_mode


def test_frozensets_match_deterministic_structure():
    """UNIT_START_PEDAGOGY_CLASSES is a SUBSET of the deterministic
    _PEDAGOGICAL_LABEL_CLASSES css-class values; the body labels
    (pedagogy-solution / pedagogy-step) are ABSENT (they are absorbable body)."""
    from dart_semantic.qwen_specialists.deterministic_structure import (
        _PEDAGOGICAL_LABEL_CLASSES,
    )

    all_label_css = {css for _pat, css in _PEDAGOGICAL_LABEL_CLASSES}
    assert UNIT_START_PEDAGOGY_CLASSES <= all_label_css
    assert "pedagogy-solution" not in UNIT_START_PEDAGOGY_CLASSES
    assert "pedagogy-step" not in UNIT_START_PEDAGOGY_CLASSES
    # The body-bearing component set + the cap are the hoisted SoT.
    assert "worked_example" in BODY_BEARING_COMPONENT_CLASSES
    assert ABSORB_MAX_RUN == 8


def test_is_unit_boundary_html_optional():
    """html=None (Stage-5e) substitutes payload['text'] emptiness; html supplied
    (assembler) keeps the byte-emptiness check; the positional call resolves."""
    heading = _region("heading", text="Chapter 1")
    unit_start = _region("paragraph", text="EXAMPLE 2", css_class="pedagogy-example")
    body_bearing = _region("paragraph", text="def", semantic_class="worked_example")
    empty_payload = _region("paragraph", text="   ")
    plain = _region("paragraph", text="just prose")

    # html=None arm.
    assert is_unit_boundary(heading, html=None) is True
    assert is_unit_boundary(unit_start, html=None) is True
    assert is_unit_boundary(body_bearing, html=None) is True
    assert is_unit_boundary(empty_payload, html=None) is True
    assert is_unit_boundary(plain, html=None) is False

    # The existing POSITIONAL (region, html) call still resolves — html supplied
    # uses byte-emptiness, so a non-empty plain paragraph is not a boundary even
    # if its payload text is blank.
    assert is_unit_boundary(plain, "<p>just prose</p>") is False
    assert is_unit_boundary(_region("paragraph", text="x"), "   ") is True
    assert is_unit_boundary(heading, "<h2>Chapter 1</h2>") is True


def test_is_unit_boundary_text_pattern():
    """html=None: a region whose payload['text'] text-STARTS a unit-opening
    label is a boundary even with NO css_class (the over-merge fix), while
    body labels ('Solution'/'Step N') + plain prose are NOT boundaries."""
    # Unit-start labels (no css_class) -> boundary on the html=None path.
    ex = _region("paragraph", text="EXAMPLE 1.114 Simplify the expression")
    tryit = _region("paragraph", text="TRY IT : : 1.2 Factor")
    howto = _region("paragraph", text="HOW TO Solve a linear equation")
    beprep = _region("paragraph", text="BE PREPARED 1.5 Evaluate")
    assert is_unit_boundary(ex, html=None) is True
    assert is_unit_boundary(tryit, html=None) is True
    assert is_unit_boundary(howto, html=None) is True
    assert is_unit_boundary(beprep, html=None) is True

    # Body labels + plain prose -> NOT boundaries (absorbable body).
    solution = _region("paragraph", text="Solution")
    step = _region("paragraph", text="Step 1. Add 3 to both sides.")
    midtext = _region("paragraph", text="This holds, for example, when x>0.")
    assert is_unit_boundary(solution, html=None) is False
    assert is_unit_boundary(step, html=None) is False
    assert is_unit_boundary(midtext, html=None) is False


def test_text_pattern_arm_html_path_unchanged():
    """The text-pattern arm is gated to html is None, so the assembler-absorb
    (html supplied) path is byte-identical: an EXAMPLE-text region with NO
    css_class is NOT a boundary when html is supplied (the absorb keeps keying
    off css_class / semantic_class only)."""
    ex_no_css = _region("paragraph", text="EXAMPLE 1.5 Simplify")
    # html supplied + non-empty + no css_class/semantic_class -> not a boundary.
    assert is_unit_boundary(ex_no_css, "<p>EXAMPLE 1.5 Simplify</p>") is False
    # html=None on the SAME region DOES trip (proves the gate is the only diff).
    assert is_unit_boundary(ex_no_css, html=None) is True


def test_resolve_unit_regroup_mode_default_on(monkeypatch):
    # ITEM1: default-ON falsey-set. Unset / blank / truthy / garbage -> on;
    # only an EXPLICIT falsey value reverts (byte-identical legacy lever).
    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP", raising=False)
    assert resolve_unit_regroup_mode() is True
    for v in ("1", "true", "yes", "on", "banana", "", "  "):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is True
    for v in ("0", "false", "no", "off"):
        monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", v)
        assert resolve_unit_regroup_mode() is False
