"""Verify figure-dataset path resolution without local fetch scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))

from data.builders.build_figure_alt_dataset import (  # noqa: E402
    FIGURE_IMAGES,
    _arxiv_image_path,
    _resolve_image,
)


def test_builder_imports_without_operator_local_fetch_modules() -> None:
    """The tracked builder must expose its resolver on a clean checkout."""
    assert callable(_resolve_image)


def test_arxiv_asset_reference_preserves_nested_asset_path() -> None:
    result = _arxiv_image_path(
        "arxiv",
        "document:id",
        "/html/example/assets/figure set/panel-1.png",
    )

    assert result == (
        FIGURE_IMAGES
        / "arxiv"
        / "document_id"
        / "figure_set"
        / "panel-1.png"
    )


def test_arxiv_non_asset_reference_uses_filename() -> None:
    result = _arxiv_image_path(
        "arxiv",
        "document-id",
        "https://example.invalid/images/panel.svg",
    )

    assert result == FIGURE_IMAGES / "arxiv" / "document-id" / "panel.svg"
