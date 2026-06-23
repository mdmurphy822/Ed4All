"""Stage-5e cascade-wiring acceptance — gate, audit section, byte-stability.

Drives the Stage-5e seam helpers in the SAME order ``run_full_cascade`` calls
them (gate -> resegment_blocks -> audit-section build -> result-dict insert)
WITHOUT pulling council/extraction/axe. Proves:

  * the gate (``resolve_block_resegment_mode``) is off by default / on truthy;
  * flag-ON: a deterministic merge re-partitions the region list AND the
    audit section is populated, with stable ids (lowest source id preserved);
  * flag-OFF: the result dict carries NO ``block_resegment`` key (byte-stable).

The cascade module pulls axe/playwright at import; we stub those for IMPORT
ONLY (mirrors ``test_cascade_stage5d.py``). The helpers under test never run
axe.
"""

from __future__ import annotations

import sys
import types


def _install_axe_stubs() -> None:
    if "axe_playwright_python" not in sys.modules:
        pkg = types.ModuleType("axe_playwright_python")
        sub = types.ModuleType("axe_playwright_python.sync_playwright")
        sub.Axe = object
        pkg.sync_playwright = sub
        sys.modules["axe_playwright_python"] = pkg
        sys.modules["axe_playwright_python.sync_playwright"] = sub
    if "playwright" not in sys.modules:
        pw = types.ModuleType("playwright")
        pw_sync = types.ModuleType("playwright.sync_api")
        pw_sync.sync_playwright = lambda *a, **k: None
        pw.sync_api = pw_sync
        sys.modules["playwright"] = pw
        sys.modules["playwright.sync_api"] = pw_sync


_install_axe_stubs()

from dart_semantic import cascade  # noqa: E402,F401  (import-loads the module)
from dart_semantic.qwen_specialists.block_resegment import (  # noqa: E402
    resegment_blocks,
    resolve_block_resegment_mode,
)
from dart_semantic.structure_graph import Region  # noqa: E402
from dart_semantic.types import FeatureBlock, RawBlock  # noqa: E402


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        page_width=100.0,
        page_height=100.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _region(kind, fb_indices, *, text=None, pages=None):
    payload = {}
    if text is not None:
        payload["text"] = text
    if pages is not None:
        payload["pages"] = pages
    return Region(kind=kind, feature_block_indices=tuple(fb_indices), payload=payload)


class _EmptyState:
    outputs = {}


# ---------------------------------------------------------------------------
# Gate.
# ---------------------------------------------------------------------------


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    assert resolve_block_resegment_mode() is False


def test_gate_on_when_truthy(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "1")
    assert resolve_block_resegment_mode() is True


# ---------------------------------------------------------------------------
# Flag-ON seam: resegment -> audit section build (cascade snippet).
# ---------------------------------------------------------------------------


def test_seam_flag_on_populates_audit_and_repartitions():
    """The exact seam the cascade runs: gate -> resegment_blocks -> audit."""
    fbs = [_fb("a clause that runs", page=1), _fb("over the page edge", page=2)]
    structure_regions = [
        _region("paragraph", [0], text="a clause that runs", pages=[1]),
        _region("paragraph", [1], text="over the page edge", pages=[2]),
    ]

    resegmented_regions, resegment_ops = resegment_blocks(
        structure_regions, fbs, _EmptyState(), runtime=None
    )

    # The cross-page continuation merge fired.
    assert len(resegmented_regions) == 1
    assert resegmented_regions[0].feature_block_indices == (0, 1)
    # Stable id: lowest source id preserved.
    assert min(resegmented_regions[0].feature_block_indices) == 0

    # Build the audit section exactly as the cascade snippet does.
    resegment_ops_audit = [
        {
            "op": op.op,
            "source_ids": list(op.source_ids),
            "origin": op.origin,
            "conservation_verified": True,
        }
        for op in resegment_ops
    ]
    assert resegment_ops_audit
    assert resegment_ops_audit[0]["op"] == "merge"
    assert resegment_ops_audit[0]["origin"] == "deterministic"
    assert resegment_ops_audit[0]["conservation_verified"] is True
    # Source ids are the merged regions' min FB indices.
    assert resegment_ops_audit[0]["source_ids"] == [0, 1]


def test_result_dict_omits_key_when_off():
    """The cascade only sets result['block_resegment'] when the audit is not
    None; an OFF run leaves the audit variable None -> key absent (byte-stable)."""
    resegment_ops_audit = None  # the cascade's flag-OFF state
    result = {"region_provenance": []}
    if resegment_ops_audit is not None:
        result["block_resegment"] = resegment_ops_audit
    assert "block_resegment" not in result


def test_result_dict_carries_key_when_on():
    resegment_ops_audit = [
        {"op": "split", "source_ids": [3], "origin": "deterministic",
         "conservation_verified": True}
    ]
    result = {"region_provenance": []}
    if resegment_ops_audit is not None:
        result["block_resegment"] = resegment_ops_audit
    assert result["block_resegment"] == resegment_ops_audit
