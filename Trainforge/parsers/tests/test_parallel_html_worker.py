"""W4 — the parallel HTML parse worker.

Covers the four invariants that make the pooled parse safe to substitute for the
serial loop:

  * ``HTMLContentParser`` holds no instance state, so a fresh parser per call is
    byte-equivalent to the reused single instance;
  * a bad input file comes back as a typed error envelope and never as a raised
    exception (a raise inside a pool aborts the ordered ``map`` and discards
    every completed parse after that index);
  * the envelope carries no ``raw_html`` and round-trips through ``pickle``;
  * the staged path is never resolved, so a symlink-staged corpus keeps its
    staging-relative ``item_path``.

Synthetic HTML strings and ``tmp_path`` only — no corpus files, no GPU, no
network. The one pooled test uses a ``spawn`` context explicitly (``fork`` is
not an accepted start method for this path).
"""
from __future__ import annotations

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from Trainforge.parsers.html_content_parser import HTMLContentParser
from Trainforge.parsers.parallel_html import (
    PARENT_SUPPLIED_ITEM_KEYS,
    WORKER_ITEM_KEYS,
    is_error,
    parse_html_path,
)

_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>Week 1 Content 02</title></head>
<body>
<main>
<article role="doc-chapter">
<h2>Simplify Square Roots</h2>
<section class="semantik-section" data-semantik-block-id="s1"
         data-cf-content-type="explanation" data-cf-template-type="explanation">
  <h3>Radical notation</h3>
  <p>The result is <strong>79</strong>.</p>
  <ul><li>First point</li><li>Second point</li></ul>
</section>
<section class="semantik-section" data-semantik-block-id="s2"
         data-cf-content-type="example">
  <h3>Worked example</h3>
  <p>Apply the rule to a concrete case.</p>
</section>
</article>
</main>
</body>
</html>
"""

_SECOND_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>Week 1 Content 03</title></head>
<body>
<main>
<article role="doc-chapter">
<h2>Rational Exponents</h2>
<section class="semantik-section" data-semantik-block-id="s1">
  <h3>Definitions</h3>
  <p>A rational exponent denotes a radical.</p>
</section>
</article>
</main>
</body>
</html>
"""

# A font asset renamed to *.html: valid bytes, invalid UTF-8. This is the shape
# that escapes an ``except OSError`` guard, because the failure is a
# UnicodeDecodeError (a ValueError) raised on decode, not on read.
_BINARY_ASSET_BYTES = b"wOFF\x00\x01\x00\x00\x00\x00\xff\xfe\xfd\xfc\x80\x81"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The stateless-parser invariant the whole port rests on
# ---------------------------------------------------------------------------

def test_parser_holds_no_instance_state():
    """A reused parser accumulates nothing, so per-call construction is safe.

    If this ever fails, the pooled path is not equivalent to the serial loop and
    the serial loop itself is carrying cross-file bleed.
    """
    parser = HTMLContentParser()
    assert parser.__dict__ == {}

    for _ in range(3):
        parser.parse(_PAGE_HTML)
        parser.parse(_SECOND_PAGE_HTML)

    assert parser.__dict__ == {}


def test_repeated_parses_are_identical_across_instances():
    """A fresh parser per call yields the same projection as a reused one."""
    reused = HTMLContentParser()
    first = reused.parse(_PAGE_HTML)
    reused.parse(_SECOND_PAGE_HTML)
    second = reused.parse(_PAGE_HTML)
    fresh = HTMLContentParser().parse(_PAGE_HTML)

    assert first == second == fresh


# ---------------------------------------------------------------------------
# Success envelope shape
# ---------------------------------------------------------------------------

def test_success_envelope_shape(tmp_path):
    staging = tmp_path / "staging"
    page = _write(staging / "week_01" / "week_01_content_02.html", _PAGE_HTML)

    env = parse_html_path((str(page), str(staging)))

    assert not is_error(env)
    assert env["error"] is None
    assert env["html_path"] == str(page)
    assert set(env["item"]) == set(WORKER_ITEM_KEYS)


def test_envelope_never_carries_raw_html(tmp_path):
    """Constraint 4 — the source markup does not ride back through the pool."""
    page = _write(tmp_path / "page.html", _PAGE_HTML)

    env = parse_html_path((str(page), None))

    for key in PARENT_SUPPLIED_ITEM_KEYS:
        assert key not in env["item"]
    assert "raw_html" not in env


def test_worker_item_matches_inline_parse(tmp_path):
    """Every worker-populated field equals the serial loop's own projection."""
    staging = tmp_path / "staging"
    page = _write(staging / "week_01_content_02.html", _PAGE_HTML)

    parsed = HTMLContentParser().parse(page.read_text(encoding="utf-8"))
    slug = page.stem.lower().replace(" ", "-")
    expected = {
        "item_id": slug,
        "item_path": str(page.relative_to(staging)),
        "title": parsed.title or slug,
        "resource_type": "page",
        "module_id": slug,
        "module_title": parsed.title or slug,
        "week_num": 0,
        "word_count": parsed.word_count,
        "sections": parsed.sections,
        "learning_objectives": parsed.learning_objectives,
        "key_concepts": parsed.key_concepts,
        "interactive_components": parsed.interactive_components,
        "page_id": parsed.page_id,
        "misconceptions": parsed.misconceptions,
        "suggested_assessment_types": parsed.suggested_assessment_types,
        "courseforge_metadata": parsed.metadata.get("courseforge"),
        "objective_refs": parsed.objective_refs,
        "source_references": parsed.source_references,
    }

    assert parse_html_path((str(page), str(staging)))["item"] == expected


def test_slug_lowercases_and_hyphenates_the_stem(tmp_path):
    """``item_id`` / ``module_id`` come from the file stem, not from the title."""
    page = _write(tmp_path / "Week 02 Content.html", _PAGE_HTML)

    item = parse_html_path((str(page), None))["item"]

    assert item["item_id"] == "week-02-content"
    assert item["module_id"] == "week-02-content"
    assert item["title"] == item["module_title"] == "Week 1 Content 02"


def test_title_falls_back_to_slug_when_parser_yields_none(tmp_path):
    """A whitespace-only ``<title>`` strips to '' and the slug takes over."""
    page = _write(tmp_path / "Week 02 Content.html", "<html><head><title>   </title></head>"
                                                     "<body><p>x</p></body></html>")

    item = parse_html_path((str(page), None))["item"]

    assert item["title"] == "week-02-content"
    assert item["module_title"] == "week-02-content"


# ---------------------------------------------------------------------------
# item_path derivation — never resolve the staged path
# ---------------------------------------------------------------------------

def test_item_path_is_staging_relative(tmp_path):
    staging = tmp_path / "staging"
    page = _write(staging / "week_01" / "page.html", _PAGE_HTML)

    item = parse_html_path((str(page), str(staging)))["item"]

    assert item["item_path"] == "week_01/page.html"


def test_item_path_falls_back_to_name_outside_staging(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    page = _write(tmp_path / "elsewhere" / "page.html", _PAGE_HTML)

    item = parse_html_path((str(page), str(staging)))["item"]

    assert item["item_path"] == "page.html"


def test_item_path_falls_back_to_name_without_staging_root(tmp_path):
    page = _write(tmp_path / "nested" / "page.html", _PAGE_HTML)

    item = parse_html_path((str(page), None))["item"]

    assert item["item_path"] == "page.html"


def test_symlink_staged_path_keeps_staging_relative_item_path(tmp_path):
    """Constraint 5 — staging is symlink-mode, so resolving would break provenance.

    The staged entry is a symlink pointing outside the staging root. Under a
    ``.resolve()`` the containment test fails and ``item_path`` degrades to the
    bare filename on EVERY item; unresolved, it stays staging-relative.
    """
    real = _write(tmp_path / "converted" / "chapter_01.html", _PAGE_HTML)
    staging = tmp_path / "staging"
    (staging / "week_01").mkdir(parents=True)
    link = staging / "week_01" / "chapter_01.html"
    link.symlink_to(real)

    item = parse_html_path((str(link), str(staging)))["item"]

    assert item["item_path"] == "week_01/chapter_01.html"
    assert item["item_path"] != link.name


# ---------------------------------------------------------------------------
# Typed error envelopes — the worker never raises for a bad input file
# ---------------------------------------------------------------------------

def test_binary_asset_named_html_returns_error_envelope(tmp_path):
    """A UnicodeDecodeError is a ValueError, NOT an OSError."""
    asset = tmp_path / "iconfont.html"
    asset.write_bytes(_BINARY_ASSET_BYTES)

    env = parse_html_path((str(asset), None))

    assert is_error(env)
    assert env["item"] is None
    assert env["error"]["type"] == "UnicodeDecodeError"
    assert env["error"]["stage"] == "read"
    assert env["error"]["path"] == str(asset)
    assert env["error"]["message"]


def test_missing_path_returns_error_envelope(tmp_path):
    missing = tmp_path / "does_not_exist.html"

    env = parse_html_path((str(missing), str(tmp_path)))

    assert is_error(env)
    assert env["error"]["type"] == "FileNotFoundError"
    assert env["error"]["stage"] == "read"


def test_directory_path_returns_error_envelope(tmp_path):
    directory = tmp_path / "not_a_file.html"
    directory.mkdir()

    env = parse_html_path((str(directory), str(tmp_path)))

    assert is_error(env)
    assert env["error"]["type"] == "IsADirectoryError"
    assert env["error"]["stage"] == "read"


@pytest.mark.parametrize(
    "markup",
    [
        "",
        "<html><body><p>unclosed",
        "<<>><h2 <p>garbage</h9></body>",
        "<html><body><script type=\"application/ld+json\">{not json</script></body></html>",
    ],
    ids=["empty", "unclosed", "garbage", "broken-jsonld"],
)
def test_malformed_markup_does_not_raise(tmp_path, markup):
    page = _write(tmp_path / "malformed.html", markup)

    env = parse_html_path((str(page), None))

    assert not is_error(env)
    assert env["item"]["item_id"] == "malformed"


# ---------------------------------------------------------------------------
# Picklability — the pool transports both the callable and the envelope
# ---------------------------------------------------------------------------

def test_worker_is_picklable_by_qualified_name():
    """``spawn`` pickles the callable as (module, qualname), not as bytecode."""
    assert pickle.loads(pickle.dumps(parse_html_path)) is parse_html_path


def test_success_envelope_round_trips_through_pickle(tmp_path):
    staging = tmp_path / "staging"
    page = _write(staging / "page.html", _PAGE_HTML)

    env = parse_html_path((str(page), str(staging)))
    restored = pickle.loads(pickle.dumps(env))

    assert restored == env
    assert restored["item"]["sections"] == env["item"]["sections"]
    assert restored["item"]["sections"][0].__class__ is env["item"]["sections"][0].__class__


def test_error_envelope_round_trips_through_pickle(tmp_path):
    asset = tmp_path / "iconfont.html"
    asset.write_bytes(_BINARY_ASSET_BYTES)

    env = parse_html_path((str(asset), None))

    assert pickle.loads(pickle.dumps(env)) == env


def test_payload_is_plain_strings_only(tmp_path):
    """Constraint 1 — the payload pickles small and drags no parent state."""
    payload = (str(tmp_path / "page.html"), str(tmp_path))

    assert all(isinstance(part, str) or part is None for part in payload)
    assert pickle.loads(pickle.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# End-to-end through a real spawn pool (CPU only, two workers)
# ---------------------------------------------------------------------------

def test_spawn_pool_output_matches_serial(tmp_path):
    staging = tmp_path / "staging"
    pages = [
        _write(staging / "week_01_content_02.html", _PAGE_HTML),
        _write(staging / "week_01_content_03.html", _SECOND_PAGE_HTML),
    ]
    asset = staging / "iconfont.html"
    asset.write_bytes(_BINARY_ASSET_BYTES)

    payloads = [(str(p), str(staging)) for p in [*pages, asset]]
    serial = [parse_html_path(payload) for payload in payloads]

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
        pooled = list(executor.map(parse_html_path, payloads, chunksize=1))

    assert pooled == serial
    assert [is_error(env) for env in pooled] == [False, False, True]
