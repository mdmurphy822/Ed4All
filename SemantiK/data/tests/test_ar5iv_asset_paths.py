"""Verify source-specific figure asset path resolution."""

from __future__ import annotations

import sys
from pathlib import Path

_SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))

from data.builders import build_figure_alt_dataset as builder  # noqa: E402
from data.sources.ar5iv_assets import resolve_ar5iv_asset_path  # noqa: E402


def test_nested_asset_reference_uses_requested_root(tmp_path: Path) -> None:
    result = resolve_ar5iv_asset_path(
        tmp_path,
        "arxiv",
        "document:id",
        "/html/example/assets/figure set/chapter/panel-1.png",
    )

    assert result == (
        tmp_path
        / "arxiv"
        / "document_id"
        / "figure_set"
        / "chapter"
        / "panel-1.png"
    )


def test_non_asset_url_uses_filename(tmp_path: Path) -> None:
    result = resolve_ar5iv_asset_path(
        tmp_path,
        "arxiv",
        "document-id",
        "https://example.invalid/images/panel.svg",
    )

    assert result == tmp_path / "arxiv" / "document-id" / "panel.svg"


def test_arxiv_builder_uses_source_adapter(monkeypatch, tmp_path: Path) -> None:
    expected = tmp_path / "resolved.png"
    calls: list[tuple[Path, str, str, str]] = []

    def fake_resolver(root: Path, source: str, doc_id: str, ref: str) -> Path:
        calls.append((root, source, doc_id, ref))
        return expected

    monkeypatch.setattr(builder, "resolve_ar5iv_asset_path", fake_resolver)

    record = {"source": "arxiv", "doc_id": "document-id", "image_ref": "asset.png"}
    assert builder._resolve_image(record, {}) == expected
    assert calls == [
        (builder.FIGURE_IMAGES, "arxiv", "document-id", "asset.png")
    ]


def test_manifest_sources_keep_manifest_resolution() -> None:
    manifest = {
        ("pmc", "document-a", "figure-a"): "private/pmc/figure-a.png",
        ("openstax", "document-b", "figure-b"): "private/open/figure-b.png",
    }

    assert builder._resolve_image(
        {"source": "pmc", "doc_id": "document-a", "image_ref": "figure-a"},
        manifest,
    ) == Path("private/pmc/figure-a.png")
    assert builder._resolve_image(
        {
            "source": "openstax",
            "doc_id": "document-b",
            "image_ref": "https://example.invalid/figure-b.jpg",
        },
        manifest,
    ) == Path("private/open/figure-b.png")


def test_missing_manifest_source_returns_none() -> None:
    record = {"source": "pmc", "doc_id": "document-a", "image_ref": "missing"}
    assert builder._resolve_image(record, {}) is None


def test_tracked_builder_has_no_fetch_script_dependency() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "data.sources.ar5iv_assets" in source
    assert "scripts.fetch_ar5iv_figure_assets" not in source
