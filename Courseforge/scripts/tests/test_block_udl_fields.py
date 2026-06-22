"""IB4.4 — UDL multiple-means coverage fields on the Block dataclass.

Covers:
    - default-constructed Block has empty UDL defaults
    - compute_content_hash() is byte-identical with/without the new fields
      (the keystone byte-stability assertion)
    - _derive_udl_coverage counts distinct representation modes
    - _derive_udl_coverage maps block_type -> response_formats
    - flag-off emit (to_html_attrs) is byte-identical to the legacy emit
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block, _derive_udl_coverage  # noqa: E402


def _block(content, *, block_type="concept", **kw):
    return Block(
        block_id="page_01#concept_demo_0",
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content,
        **kw,
    )


def test_default_udl_fields_empty() -> None:
    b = _block("<p>x</p>")
    assert b.n_representations == 0
    assert b.response_formats == ()
    assert b.engagement_affordance is None


def test_content_hash_byte_identical_with_udl_fields() -> None:
    """The keystone — populating UDL fields must NOT drift the hash."""
    base = _block("<p>same content</p>")
    enriched = _block(
        "<p>same content</p>",
        n_representations=3,
        response_formats=("select",),
        engagement_affordance="real_world",
    )
    assert base.compute_content_hash() == enriched.compute_content_hash()


def test_derive_counts_distinct_representations() -> None:
    b = _block("<p>prose</p><table></table><img alt='y'>")
    n, rf, ea = _derive_udl_coverage(b)
    assert n == 3  # prose + table + image


def test_derive_response_format_from_block_type() -> None:
    sc = _block("<details><summary>q</summary></details>",
                block_type="self_check_question")
    _n, rf, _ea = _derive_udl_coverage(sc)
    assert rf == ("select",)

    act = _block("<p>do this</p>", block_type="activity")
    _n2, rf2, _ea2 = _derive_udl_coverage(act)
    assert rf2 == ("construct",)


def test_derive_engagement_affordance() -> None:
    refl = _block("<p>reflect</p>", block_type="reflection_prompt")
    _n, _rf, ea = _derive_udl_coverage(refl)
    assert ea == "reflection"

    scen = _block("<p>Imagine you run a shop...</p>", block_type="concept")
    _n2, _rf2, ea2 = _derive_udl_coverage(scen)
    assert ea2 == "real_world"


def test_flag_off_html_emit_byte_identical() -> None:
    os.environ.pop("ED4ALL_BLOCK_A11Y", None)
    legacy = _block("<p>x</p>")
    enriched = _block(
        "<p>x</p>",
        n_representations=4,
        response_formats=("select", "reflect"),
        engagement_affordance="choice",
    )
    assert legacy.to_html_attrs() == enriched.to_html_attrs()


def test_flag_on_html_emit_adds_udl_attrs() -> None:
    os.environ["ED4ALL_BLOCK_A11Y"] = "1"
    try:
        enriched = _block(
            "<p>x</p>",
            n_representations=4,
            response_formats=("select",),
            engagement_affordance="choice",
        )
        attrs = enriched.to_html_attrs()
        assert 'data-cf-udl-representations="4"' in attrs
        assert 'data-cf-udl-response-formats="select"' in attrs
        assert 'data-cf-udl-engagement="choice"' in attrs
    finally:
        os.environ.pop("ED4ALL_BLOCK_A11Y", None)
