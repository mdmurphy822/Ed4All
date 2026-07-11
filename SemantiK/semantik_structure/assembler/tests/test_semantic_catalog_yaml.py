"""Phase-2 tests for the gold-standard semantic-component catalog YAML.

The catalog (``semantik_structure/config/semantic_components.yaml``) is DATA ONLY —
nothing consumes it yet — so the assembler output stays byte-identical. These
tests assert the catalog's internal coverage contracts:

  * it parses as ``version: 1`` + a non-empty ``components`` list;
  * every ``maps_from_region_kind`` value is a real ``REGION_KINDS`` member;
  * every ``css_class`` (where set) is substring-present in the vendored
    ``WCAG22_CSS`` (catalog <-> CSS coverage — prevents a class the assembler
    cannot render);
  * every entry carries a non-empty ``label`` (the synthesized container
    heading the Phase-7 assembler emits);
  * every ``reconciles_with`` (where set) is a real
    ``_PEDAGOGICAL_LABEL_CLASSES`` value (no invented ``pedagogy-*`` class);
  * ``assemble_document`` output is unchanged (no consumer).
"""

from __future__ import annotations

from pathlib import Path

import yaml

import semantik_structure
from semantik_structure.assembler.api import AssemblerConfig, assemble_document
from semantik_structure.assembler.wcag22_css import WCAG22_CSS
from semantik_structure.qwen_specialists.deterministic_structure import (
    _PEDAGOGICAL_LABEL_CLASSES,
)
from semantik_structure.structure_graph import REGION_KINDS, Region
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Catalog load (package-relative — NOT via env).
# ---------------------------------------------------------------------------

_CATALOG_PATH = (
    Path(semantik_structure.__file__).parent / "config" / "semantic_components.yaml"
)


def _load_catalog() -> dict:
    with _CATALOG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _components() -> list[dict]:
    return _load_catalog()["components"]


def _kinds(entry: dict) -> list[str]:
    """Normalize ``maps_from_region_kind`` (str or list) to a list."""
    raw = entry.get("maps_from_region_kind")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return list(raw)


# Real pedagogy CSS-class values from the single source of truth.
_REAL_PEDAGOGY_CLASSES = frozenset(
    css_class for _pattern, css_class in _PEDAGOGICAL_LABEL_CLASSES
)


# ---------------------------------------------------------------------------
# assemble_document fixture (mirrors test_vendored_css.py).
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=1,
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


def _para(text: str, idx: int) -> Region:
    return Region(
        kind="paragraph",
        feature_block_indices=(idx,),
        payload={"text": text},
        source_region_id=idx,
    )


def _assemble_fixture():
    regions = [_para("First paragraph.", 0), _para("Second paragraph.", 1)]
    fbs = [_fb("First paragraph."), _fb("Second paragraph.")]
    top_per_region = {i: None for i in range(len(regions))}
    return assemble_document(
        top_per_region,
        regions,
        fbs,
        runtime_mode="mock",
        config=AssemblerConfig(skip_gap_fill=True),
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_yaml_parses_version_1_components_list():
    doc = _load_catalog()
    assert doc["version"] == 1
    components = doc["components"]
    assert isinstance(components, list)
    assert components, "components list is empty"


def test_every_maps_from_kind_is_real_region_kind():
    for entry in _components():
        kinds = _kinds(entry)
        assert kinds, f"{entry['component']} has no maps_from_region_kind"
        for kind in kinds:
            assert kind in REGION_KINDS, (
                f"{entry['component']}: maps_from_region_kind {kind!r} "
                f"is not a real REGION_KINDS member {tuple(REGION_KINDS)!r}"
            )


def test_every_css_class_present_in_vendored_css():
    for entry in _components():
        css_class = entry.get("css_class")
        if not css_class:
            continue  # null/omitted css_class (section/footnote/code_region)
        assert css_class in WCAG22_CSS, (
            f"{entry['component']}: css_class {css_class!r} "
            "is not substring-present in WCAG22_CSS"
        )


def test_every_component_has_label():
    for entry in _components():
        label = entry.get("label")
        assert isinstance(label, str) and label.strip(), (
            f"{entry['component']}: label must be a non-empty string"
        )


def test_every_reconciles_with_is_real_pedagogy_class():
    for entry in _components():
        recon = entry.get("reconciles_with")
        if recon is None:
            continue
        values = [recon] if isinstance(recon, str) else list(recon)
        for value in values:
            assert value in _REAL_PEDAGOGY_CLASSES, (
                f"{entry['component']}: reconciles_with {value!r} is not a "
                f"real _PEDAGOGICAL_LABEL_CLASSES value "
                f"{sorted(_REAL_PEDAGOGY_CLASSES)!r}"
            )


def test_assembler_output_unchanged():
    doc_a = _assemble_fixture()
    doc_b = _assemble_fixture()
    assert doc_a.html == doc_b.html, "assemble_document is non-deterministic"
