"""T4 — generator provenance metadata on every generated page.

Every page's JSON-LD carries an unconditional schema.org-style
``generator`` object ``{name, version, dateCreated}`` (ISO-8601 UTC):

* The name identifies the pipeline GENERICALLY (no model / vendor
  identity — that lives in ``provenance.tiers[]``).
* The version is resolved from a single real source (the ``ed4all``
  package metadata, i.e. the pyproject version) and also replaces the
  formerly-hardcoded ``provenance.pipelineVersion`` string.
* ``dateCreated`` is the generation timestamp and is kept OUT of the
  content-addressed ``contentHash``.

These tests cover the emit shape, the version-source single-sourcing,
the contentHash invariance, and the additive schema change.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_course  # noqa: E402
from generate_course import (  # noqa: E402
    _GENERATOR_NAME,
    _build_page_metadata,
    _generator_metadata,
    _generator_version,
)

from blocks import Block  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# --------------------------------------------------------------------- #
# version single-sourcing
# --------------------------------------------------------------------- #


def _pyproject_version() -> str:
    for line in (_PROJECT_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    ).splitlines():
        match = re.match(r"""\s*version\s*=\s*["']([^"']+)["']""", line)
        if match:
            return match.group(1)
    raise AssertionError("no version line in pyproject.toml")


def test_generator_version_matches_pyproject() -> None:
    """The resolved generator version traces back to the pyproject version.

    Whether it comes from installed distribution metadata or the source-tree
    fallback, both paths read the SAME pyproject ``version`` so the stamped
    string can never drift from the released package.
    """
    assert _generator_version() == _pyproject_version()


def test_generator_version_is_nontrivial() -> None:
    assert _generator_version() not in ("", "0.0.0")


# --------------------------------------------------------------------- #
# generator object shape
# --------------------------------------------------------------------- #


def test_generator_metadata_shape() -> None:
    gen = _generator_metadata()
    assert set(gen) == {"name", "version", "dateCreated"}
    assert gen["name"] == _GENERATOR_NAME == "Ed4All Courseforge"
    assert gen["version"] == _generator_version()
    # ISO-8601 UTC, and actually parseable as a real instant.
    assert _ISO_UTC_RE.match(gen["dateCreated"]), gen["dateCreated"]
    datetime.strptime(gen["dateCreated"], "%Y-%m-%dT%H:%M:%SZ")


def test_generator_name_carries_no_vendor_or_model() -> None:
    """The generic name must not leak a vendor / model identity — per-tier
    model + provider attribution is the provenance envelope's job.
    """
    name = _generator_metadata()["name"].lower()
    for banned in ("claude", "anthropic", "gpt", "openai", "qwen", "llama",
                   "sonnet", "opus", "nvidia", "together"):
        assert banned not in name, f"generator name leaked '{banned}'"


# --------------------------------------------------------------------- #
# emitted on every page (unconditional / default path)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("emit_blocks", ["", "1"])
def test_generator_emitted_unconditionally(monkeypatch, emit_blocks) -> None:
    """The generator object is present regardless of COURSEFORGE_EMIT_BLOCKS."""
    if emit_blocks:
        monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", emit_blocks)
    else:
        monkeypatch.delenv("COURSEFORGE_EMIT_BLOCKS", raising=False)
    meta = _build_page_metadata("TST_913", 1, "content", "week_01_content_01")
    assert "generator" in meta
    gen = meta["generator"]
    assert gen["name"] == "Ed4All Courseforge"
    assert gen["version"] == _generator_version()
    assert _ISO_UTC_RE.match(gen["dateCreated"])


def test_generated_page_carries_generator_in_jsonld(monkeypatch) -> None:
    """A full page emit surfaces the generator object in the <script> JSON-LD."""
    monkeypatch.delenv("COURSEFORGE_EMIT_BLOCKS", raising=False)
    meta = _build_page_metadata("TST_913", 1, "content", "week_01_content_01")
    page = generate_course._wrap_page(
        "T4 Page", "TST_913", 1, "<p>body</p>", page_metadata=meta,
    )
    match = re.search(
        r'<script\s+type="application/ld\+json">(.*?)</script>', page, re.DOTALL,
    )
    assert match, "no JSON-LD script block on the page"
    payload = json.loads(match.group(1))
    assert payload["generator"]["name"] == "Ed4All Courseforge"
    assert payload["generator"]["version"] == _generator_version()


# --------------------------------------------------------------------- #
# provenance.pipelineVersion is the same single-sourced version
# --------------------------------------------------------------------- #


def test_pipeline_version_is_generator_version(monkeypatch) -> None:
    """The provenance envelope's pipelineVersion is the resolved package
    version — no longer the hardcoded 'phase2' literal.
    """
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "1")
    block = Block(
        block_id="week_01_content_01#concept_intro_0",
        block_type="concept",
        page_id="week_01_content_01",
        sequence=0,
        content={"heading": "Intro", "body": "text"},
    )
    meta = _build_page_metadata(
        "TST_913", 1, "content", "week_01_content_01", blocks=[block],
    )
    assert meta["provenance"]["pipelineVersion"] == _generator_version()
    assert meta["provenance"]["pipelineVersion"] != "phase2"


# --------------------------------------------------------------------- #
# contentHash invariance — the per-render timestamp must not drift it
# --------------------------------------------------------------------- #


def test_generator_timestamp_excluded_from_content_hash(monkeypatch) -> None:
    """Two builds of the same content yield an identical contentHash even
    though their generator.dateCreated timestamps differ.
    """
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "1")
    block = Block(
        block_id="week_01_content_01#concept_intro_0",
        block_type="concept",
        page_id="week_01_content_01",
        sequence=0,
        content={"heading": "Intro", "body": "text"},
    )

    real = generate_course._generator_metadata

    def _stamp(ts):
        return lambda: {"name": _GENERATOR_NAME, "version": _generator_version(),
                        "dateCreated": ts}

    monkeypatch.setattr(generate_course, "_generator_metadata",
                        _stamp("2026-01-01T00:00:00Z"))
    meta1 = _build_page_metadata(
        "TST_913", 1, "content", "week_01_content_01", blocks=[block],
    )
    monkeypatch.setattr(generate_course, "_generator_metadata",
                        _stamp("2030-12-31T23:59:59Z"))
    meta2 = _build_page_metadata(
        "TST_913", 1, "content", "week_01_content_01", blocks=[block],
    )
    monkeypatch.setattr(generate_course, "_generator_metadata", real)

    assert meta1["generator"]["dateCreated"] != meta2["generator"]["dateCreated"]
    assert meta1["contentHash"] == meta2["contentHash"], (
        "generator timestamp leaked into the content-addressed hash"
    )


# --------------------------------------------------------------------- #
# schema — additive change
# --------------------------------------------------------------------- #


def _minimally_valid_metadata() -> dict:
    return {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "TST_913",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01_intro",
    }


def test_schema_accepts_generator_object() -> None:
    validator = generate_course._get_jsonld_validator()
    assert validator is not None
    meta = _minimally_valid_metadata()
    meta["generator"] = _generator_metadata()
    assert not list(validator.iter_errors(meta))


def test_schema_generator_is_optional() -> None:
    """Legacy metadata WITHOUT a generator still validates (additive)."""
    validator = generate_course._get_jsonld_validator()
    assert validator is not None
    assert not list(validator.iter_errors(_minimally_valid_metadata()))


def test_schema_generator_requires_all_three_keys() -> None:
    validator = generate_course._get_jsonld_validator()
    assert validator is not None
    meta = _minimally_valid_metadata()
    meta["generator"] = {"name": "Ed4All Courseforge", "version": "0.3.0"}
    assert list(validator.iter_errors(meta)), (
        "generator missing dateCreated should fail validation"
    )


def test_schema_generator_rejects_extra_property() -> None:
    validator = generate_course._get_jsonld_validator()
    assert validator is not None
    meta = _minimally_valid_metadata()
    meta["generator"] = {
        "name": "Ed4All Courseforge",
        "version": "0.3.0",
        "dateCreated": "2026-01-01T00:00:00Z",
        "model": "some-model",  # not allowed — model identity lives in tiers[]
    }
    assert list(validator.iter_errors(meta)), (
        "generator with an extra property should fail (additionalProperties:false)"
    )


def test_full_generated_meta_validates(monkeypatch) -> None:
    """The real _build_page_metadata output (with generator) validates."""
    monkeypatch.delenv("COURSEFORGE_EMIT_BLOCKS", raising=False)
    validator = generate_course._get_jsonld_validator()
    assert validator is not None
    meta = _build_page_metadata("TST_913", 1, "content", "week_01_content_01")
    assert not list(validator.iter_errors(meta))
