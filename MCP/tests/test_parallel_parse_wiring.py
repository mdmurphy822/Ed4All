"""W5/W7 — parallel staged-HTML parse wiring in the chunking phases.

The chunking phase's per-file parse moved from an inline serial loop to
``_parse_staged_html_files``, which dispatches the SAME per-file worker either
in-process or across a ``spawn`` ``ProcessPoolExecutor``. This module pins the
properties that make that safe to turn on:

* **Serial and pooled emit are identical** — same ``chunks.jsonl`` SHA-256 AND
  the same per-file outcome ledger, on a corpus that is SYMLINK-staged (the
  production default) and contains the four shapes that have historically
  diverged: a multi-section page, a page whose only heading carries inline
  markup, a binary asset named ``*.html``, and an uppercase ``PAGE.HTML``.
* **Every discovered file is accounted for.** A rejected / unreadable /
  unparseable file contributes to the source digest and ``source_html_count``
  but zero blocks to the chunker, so it is registered under a named
  ``source_coverage`` drop reason with an explicitly widened consumed/dropped
  pair.
* **A stop unwinds the pool** rather than being swallowed or waited out.
* **The dedup ledger is a reversible sidecar**, its count balances the coverage
  histogram, and every drop resolves to a chunk that is actually on disk.
* **``chunks.jsonl`` is published atomically** — no truncated chunkset can be
  left behind for the resume guard to preserve.

CPU-pinned and hermetic: no LLM dispatch, no embedding backend, no GPU. The
pooled arms run a real 2-worker spawn pool (fast — the worker imports only
``html.parser``-level code).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    _html_asset_signature,
    _html_parse_drop_reasons,
    _resolve_html_asset_reject,
    _resolve_html_parse_start_method,
    _resolve_html_parse_workers,
)

_RUN_ID = "PARALLEL_PARSE_TESTRUN"


# --------------------------------------------------------------------------- #
# Fixture corpus
# --------------------------------------------------------------------------- #
def _multi_section_html(title: str) -> str:
    return (
        "<!doctype html><html><head><title>" + title + "</title></head><body>"
        "<h1>" + title + "</h1>"
        "<section><h2>Managing Cognitive Load</h2>"
        "<p>Effective instructional design manages cognitive load so learners "
        "are not overwhelmed by extraneous information presented all at once. "
        "Scaffolding breaks a complex task into manageable steps.</p></section>"
        "<section><h2>Backward Design</h2>"
        "<p>Backward design starts from the desired learning outcomes and "
        "works back toward the activities that produce them. Formative "
        "assessment surfaces misconceptions early in the module.</p></section>"
        "</body></html>"
    )


def _inline_markup_heading_html() -> str:
    """A page whose ONLY heading carries inline markup.

    The heading text has to survive the extractor's whitespace-correct
    assembly: a strip-and-rejoin would render ``The Chain Rule`` as
    ``The Chain Rule`` but a fabricated separator at the inline boundary shows
    up as a different section heading, which changes the chunk bytes.
    """
    return (
        "<!doctype html><html><head><title>Derivatives</title></head><body>"
        "<h2>The <strong>Chain</strong> Rule</h2>"
        "<p>The chain rule differentiates a composition of functions by "
        "multiplying the derivative of the outer function by the derivative "
        "of the inner function evaluated at the inner argument.</p>"
        "</body></html>"
    )


def _build_symlink_staged_corpus(tmp_path: Path) -> Path:
    """Real files under ``source/``, symlinked into ``staging/``.

    Staging is symlink-mode by default in production. That is load-bearing for
    ``item_path``: it is derived from ``relative_to(staging_dir)`` on the
    UNRESOLVED path, so anything that resolves the path would collapse every
    item to its bare filename and silently rewrite ``source.item_path`` on
    every emitted chunk.
    """
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (staging / "nested").mkdir(parents=True)

    files = {
        "lesson_01.html": _multi_section_html("Lesson One").encode("utf-8"),
        "lesson_02.html": _inline_markup_heading_html().encode("utf-8"),
        # A binary asset that matches the discovery glob. PNG signature.
        "sprite.html": b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
        # Uppercase suffix: accepted by the conversion input detector (which
        # lowercases the suffix) but invisible to a case-sensitive rglob.
        "PAGE.HTML": _multi_section_html("Upper Page").encode("utf-8"),
    }
    for name, payload in files.items():
        (source / name).write_bytes(payload)

    for name in ("lesson_01.html", "sprite.html", "PAGE.HTML"):
        (staging / name).symlink_to(source / name)
    # One file staged under a subdirectory so the staging-relative item_path is
    # actually exercised (not just the bare-filename case).
    (staging / "nested" / "lesson_02.html").symlink_to(source / "lesson_02.html")
    return staging


@pytest.fixture
def chunking_tool(monkeypatch, tmp_path):
    """``run_dart_chunking`` rooted entirely inside ``tmp_path``."""
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    return _build_tool_registry()["run_dart_chunking"]


def _run(tool, staging: Path, libv2_root: Path, **kwargs) -> dict:
    return json.loads(asyncio.run(tool(
        course_name="PARALLEL_PARSE",
        staging_dir=str(staging),
        libv2_root=str(libv2_root),
        **kwargs,
    )))


def _read_chunks(path: Path) -> list:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# Resolvers
# --------------------------------------------------------------------------- #
def test_worker_count_defaults_and_clamps(monkeypatch):
    monkeypatch.delenv("ED4ALL_HTML_PARSE_WORKERS", raising=False)
    # Clamped to the file count, which is the binding term on a small corpus.
    assert _resolve_html_parse_workers(3) == 3
    assert _resolve_html_parse_workers(1000) == min(10, __import__("os").cpu_count() or 1)


@pytest.mark.parametrize("raw", ["", "  ", "nonsense", "3.5", "-4"])
def test_worker_count_parses_with_fallback(monkeypatch, raw):
    """Garbage and NEGATIVE values fall back; the resolver never raises."""
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", raw)
    assert _resolve_html_parse_workers(1000) >= 1


@pytest.mark.parametrize("raw,expected", [("0", 0), ("1", 1)])
def test_worker_count_honors_the_serial_opt_out(monkeypatch, raw, expected):
    """``0`` / ``1`` are the documented serial request, NOT "non-positive garbage".

    Promoting them to the default would take the byte-identical no-pool path
    away from an operator who deliberately asked for it.
    """
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", raw)
    assert _resolve_html_parse_workers(500) == expected


def test_start_method_rejects_fork_loudly(monkeypatch, caplog):
    monkeypatch.setenv("ED4ALL_HTML_PARSE_START_METHOD", "fork")
    with caplog.at_level("WARNING"):
        assert _resolve_html_parse_start_method() == "spawn"
    assert "fork is not an accepted value" in caplog.text


@pytest.mark.parametrize(
    "raw,expected",
    [("spawn", "spawn"), ("FORKSERVER", "forkserver"), (" serial ", "serial"),
     ("bogus", "spawn"), ("", "spawn")],
)
def test_start_method_resolution(monkeypatch, raw, expected):
    monkeypatch.setenv("ED4ALL_HTML_PARSE_START_METHOD", raw)
    assert _resolve_html_parse_start_method() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [(None, True), ("", True), ("1", True), ("true", True),
     ("0", False), ("false", False), ("no", False), ("OFF", False)],
)
def test_asset_reject_default_on(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("ED4ALL_HTML_ASSET_REJECT", raising=False)
    else:
        monkeypatch.setenv("ED4ALL_HTML_ASSET_REJECT", raw)
    assert _resolve_html_asset_reject() is expected


def test_asset_signature_detects_binary_and_passes_markup(tmp_path):
    png = tmp_path / "a.html"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    assert _html_asset_signature(png) == "png"

    markup = tmp_path / "b.html"
    markup.write_text("<!doctype html><html><body><p>hi</p></body></html>")
    assert _html_asset_signature(markup) is None

    # An unreadable path is the worker's typed ``read`` failure to report, not
    # a reject reason — otherwise the same file lands under two different drop
    # reasons depending on the flag.
    assert _html_asset_signature(tmp_path / "missing.html") is None


def test_drop_reason_histogram_excludes_parsed_rows():
    ledger = [
        {"path": "a", "outcome": "parsed", "reason": ""},
        {"path": "b", "outcome": "asset_rejected", "reason": "png"},
        {"path": "c", "outcome": "asset_rejected", "reason": "jpeg"},
        {"path": "d", "outcome": "html_parse_error", "reason": "ValueError"},
    ]
    assert _html_parse_drop_reasons(ledger) == {
        "asset_rejected": 2, "html_parse_error": 1,
    }


# --------------------------------------------------------------------------- #
# Serial vs pooled parity
# --------------------------------------------------------------------------- #
def test_serial_and_pooled_emit_are_identical(chunking_tool, tmp_path, monkeypatch):
    """Byte-identical chunkset SHA + identical outcome ledger across modes."""
    staging = _build_symlink_staged_corpus(tmp_path)

    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    serial_root = tmp_path / "LibV2-serial"
    serial = _run(chunking_tool, staging, serial_root)

    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "2")
    monkeypatch.setenv("ED4ALL_HTML_PARSE_START_METHOD", "spawn")
    pooled_root = tmp_path / "LibV2-pooled"
    pooled = _run(chunking_tool, staging, pooled_root)

    assert serial["success"] and pooled["success"]
    assert serial["chunks_count"] > 0, "fixture produced no chunks"
    assert serial["semantik_chunks_sha256"] == pooled["semantik_chunks_sha256"]
    assert serial["source_html_parse_outcomes"] == (
        pooled["source_html_parse_outcomes"]
    )
    # And the bytes on disk really are the same, not just the recorded digest.
    serial_bytes = Path(serial["semantik_chunks_path"]).read_bytes()
    pooled_bytes = Path(pooled["semantik_chunks_path"]).read_bytes()
    assert serial_bytes == pooled_bytes
    assert hashlib.sha256(serial_bytes).hexdigest() == (
        serial["semantik_chunks_sha256"]
    )


def test_pooled_arm_really_builds_a_pool(chunking_tool, tmp_path, monkeypatch):
    """Guard against the parity test passing because both arms ran serially.

    Wraps the pool factory (the documented test seam) so the assertion is on
    the real construction, and pins the constructed shape: ``spawn`` start
    method, worker count clamped to the parseable-file count, and no
    ``initializer`` (the worker-side knobs are inherited from the parent
    environment because the child reads them at interpreter startup, before any
    initializer could run).
    """
    import os as _os

    prior_arena = _os.environ.get("MALLOC_ARENA_MAX")
    calls = []
    real = pipeline_tools._make_html_parse_pool

    def _snapshot():
        return {
            "omp": _os.environ.get("OMP_NUM_THREADS"),
            "arena": _os.environ.get("MALLOC_ARENA_MAX"),
            "hashseed": _os.environ.get("PYTHONHASHSEED"),
        }

    class _EnvRecordingPool:
        """Proxy recording the env at the moment workers are actually started.

        ``ProcessPoolExecutor`` starts no worker until the first ``submit``,
        which ``map`` performs — so snapshotting only at construction would
        still pass if the override scope closed too early and no child ever
        inherited the pinned environment.
        """

        def __init__(self, inner):
            self._inner = inner

        def map(self, *args, **kwargs):
            calls[-1]["at_map"] = _snapshot()
            return self._inner.map(*args, **kwargs)

        def shutdown(self, *args, **kwargs):
            return self._inner.shutdown(*args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    def _spy(workers, start_method):
        record = {"workers": workers, "start_method": start_method}
        record.update({f"ctor_{k}": v for k, v in _snapshot().items()})
        calls.append(record)
        return _EnvRecordingPool(real(workers, start_method))

    monkeypatch.setattr(pipeline_tools, "_make_html_parse_pool", _spy)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "8")
    monkeypatch.setenv("ED4ALL_HTML_PARSE_START_METHOD", "spawn")
    staging = _build_symlink_staged_corpus(tmp_path)
    result = _run(chunking_tool, staging, tmp_path / "LibV2-pooled-spy")

    assert result["success"]
    assert len(calls) == 1, "the pooled arm did not construct a pool"
    # 4 discovered - 1 rejected asset = 3 parseable files; 8 clamps to 3.
    assert calls[0]["workers"] == 3
    assert calls[0]["start_method"] == "spawn"
    # In force at construction...
    assert calls[0]["ctor_omp"] == "1"
    assert calls[0]["ctor_arena"] == "2"
    assert calls[0]["ctor_hashseed"]
    # ...and STILL in force at the first submit, which is when a worker is
    # actually started and inherits it. PYTHONHASHSEED is pinned so every child
    # agrees with every other child and across runs — the property the chunkset
    # SHA depends on.
    assert calls[0]["at_map"] == {
        "omp": "1", "arena": "2",
        "hashseed": calls[0]["ctor_hashseed"],
    }
    # Every override is restored to its exact prior state once the parse ends.
    assert _os.environ.get("MALLOC_ARENA_MAX") == prior_arena


def test_uppercase_suffix_page_is_discovered(chunking_tool, tmp_path, monkeypatch):
    """``PAGE.HTML`` reaches the chunker (case-sensitive rglob hid it before)."""
    staging = _build_symlink_staged_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-upper")

    assert result["source_html_count"] == 4
    chunks = _read_chunks(result["semantik_chunks_path"])
    item_paths = {c["source"]["item_path"] for c in chunks}
    assert "PAGE.HTML" in item_paths


def test_symlink_staging_preserves_relative_item_path(
    chunking_tool, tmp_path, monkeypatch
):
    """``item_path`` stays staging-relative through a symlink, incl. a subdir."""
    staging = _build_symlink_staged_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "2")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-symlink")

    item_paths = {
        c["source"]["item_path"]
        for c in _read_chunks(result["semantik_chunks_path"])
    }
    assert "nested/lesson_02.html" in item_paths, (
        f"resolved path leaked into item_path: {sorted(item_paths)}"
    )


# --------------------------------------------------------------------------- #
# Per-file outcome ledger + coverage accounting
# --------------------------------------------------------------------------- #
def test_asset_is_rejected_and_registered_as_a_named_drop(
    chunking_tool, tmp_path, monkeypatch
):
    staging = _build_symlink_staged_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-asset")

    outcomes = result["source_html_parse_outcomes"]
    assert outcomes.get("asset_rejected") == 1
    # Every discovered file lands in exactly one bucket.
    assert sum(outcomes.values()) == result["source_html_count"] == 4

    manifest = json.loads(Path(result["manifest_path"]).read_text("utf-8"))
    coverage = manifest["source_coverage"]
    assert coverage["drop_reasons"].get("asset_rejected") == 1
    # The widened pair keeps the balance invariant exact — no
    # ``internal_drop_reason_missing`` bucket is synthesized.
    assert coverage["dropped_count"] == sum(coverage["drop_reasons"].values())
    assert "internal_drop_reason_missing" not in coverage["drop_reasons"]


def test_asset_reject_off_still_records_the_file(
    chunking_tool, tmp_path, monkeypatch
):
    """Flag off changes WHICH reason the asset lands under, never whether."""
    staging = _build_symlink_staged_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    monkeypatch.setenv("ED4ALL_HTML_ASSET_REJECT", "0")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-noreject")

    outcomes = result["source_html_parse_outcomes"]
    assert "asset_rejected" not in outcomes
    # The PNG is still not markup: it fails the worker's strict utf-8 decode.
    assert outcomes.get("html_read_error") == 1
    assert sum(outcomes.values()) == result["source_html_count"] == 4

    coverage = json.loads(
        Path(result["manifest_path"]).read_text("utf-8")
    )["source_coverage"]
    assert coverage["drop_reasons"].get("html_read_error") == 1
    assert coverage["dropped_count"] == sum(coverage["drop_reasons"].values())


# --------------------------------------------------------------------------- #
# Graceful stop
# --------------------------------------------------------------------------- #
@pytest.fixture
def armed_env(state_runs_isolated, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


def test_stop_during_pooled_parse_unwinds_promptly(
    chunking_tool, tmp_path, monkeypatch, armed_env
):
    """A pre-armed stop propagates out of the pooled parse, and fast.

    The bound matters: binding ``executor.map(...)`` to a local before the
    ``for`` keeps the map generator alive in the frame the propagating
    traceback retains, so its per-future ``cancel()`` never runs before
    ``__exit__``'s ``shutdown(wait=True)`` and the pool drains EVERY queued
    task first. Inline consumption plus the explicit
    ``shutdown(cancel_futures=True)`` is what keeps this bounded.
    """
    staging = _build_symlink_staged_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "2")
    monkeypatch.setenv("ED4ALL_HTML_PARSE_START_METHOD", "spawn")
    stop_control.request_stop(scope="run", reason="test", source="test")

    started = time.monotonic()
    with pytest.raises(GracefulStopRequested) as excinfo:
        _run(chunking_tool, staging, tmp_path / "LibV2-stop")
    elapsed = time.monotonic() - started

    assert excinfo.value.site_id == "dart_chunking"
    assert elapsed < 60.0, f"pooled stop took {elapsed:.1f}s to unwind"
    # Nothing was published: the stop fires before the chunkset write.
    assert not (
        tmp_path / "LibV2-stop" / "courses" / "parallel-parse"
        / "semantik_chunks" / "chunks.jsonl"
    ).exists()


def test_stop_during_serial_parse_unwinds(
    chunking_tool, tmp_path, monkeypatch, armed_env
):
    """Same stop boundary on the no-pool path — the position did not move."""
    staging = _build_symlink_staged_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested) as excinfo:
        _run(chunking_tool, staging, tmp_path / "LibV2-stop-serial")
    assert excinfo.value.site_id == "dart_chunking"


# --------------------------------------------------------------------------- #
# Atomic chunkset publish
# --------------------------------------------------------------------------- #
def test_chunks_jsonl_is_published_atomically(chunking_tool, tmp_path, monkeypatch):
    """No ``.tmp`` residue, and the recorded SHA matches the published bytes."""
    staging = _build_symlink_staged_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-atomic")

    chunks_path = Path(result["semantik_chunks_path"])
    leftovers = sorted(chunks_path.parent.glob("chunks.jsonl.*.tmp"))
    assert not leftovers, f"temp files left behind: {leftovers}"
    assert hashlib.sha256(chunks_path.read_bytes()).hexdigest() == (
        result["semantik_chunks_sha256"]
    )


# --------------------------------------------------------------------------- #
# W7 — dedup ledger sidecar + drop accounting
# --------------------------------------------------------------------------- #
def _duplicate_corpus(tmp_path: Path) -> Path:
    """Three staged pages, two of which carry byte-identical prose."""
    staging = tmp_path / "dupe-staging"
    staging.mkdir()
    body = _multi_section_html("Shared Chapter")
    (staging / "page_a.html").write_text(body, encoding="utf-8")
    # Whitespace + case variant of the same content: the exact-normalized hash
    # collapses it onto page_a, a raw byte hash would not.
    (staging / "page_b.html").write_text(
        body.replace("Effective instructional", "EFFECTIVE   instructional"),
        encoding="utf-8",
    )
    (staging / "page_c.html").write_text(
        _inline_markup_heading_html(), encoding="utf-8"
    )
    return staging


def test_dedup_off_writes_no_sidecar(chunking_tool, tmp_path, monkeypatch):
    staging = _duplicate_corpus(tmp_path)
    monkeypatch.delenv("ED4ALL_CHUNK_DEDUP", raising=False)
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-nodedup")

    assert "dedup_ledger_path" not in result
    sidecar = Path(result["semantik_chunks_path"]).parent / "dedup_ledger.jsonl"
    assert not sidecar.exists()
    coverage = json.loads(
        Path(result["manifest_path"]).read_text("utf-8")
    )["source_coverage"]
    assert "within_package_duplicate" not in coverage["drop_reasons"]


def test_dedup_on_persists_a_balanced_reversible_ledger(
    chunking_tool, tmp_path, monkeypatch
):
    """Ledger lines == registered drops; every drop resolves to a live chunk."""
    staging = _duplicate_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_CHUNK_DEDUP", "1")
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-dedup")

    assert result.get("dedup_dropped_count", 0) > 0, (
        "fixture did not exercise the dedup path"
    )
    ledger_path = Path(result["dedup_ledger_path"])
    assert ledger_path.name == "dedup_ledger.jsonl"
    assert ledger_path.parent == Path(result["semantik_chunks_path"]).parent

    rows = [
        json.loads(line)
        for line in ledger_path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == result["dedup_dropped_count"]

    coverage = json.loads(
        Path(result["manifest_path"]).read_text("utf-8")
    )["source_coverage"]
    assert coverage["drop_reasons"]["within_package_duplicate"] == len(rows)
    assert coverage["dropped_count"] == sum(coverage["drop_reasons"].values())
    assert "internal_drop_reason_missing" not in coverage["drop_reasons"]

    # Reversibility: every dropped unit names a kept chunk that is on disk.
    live_ids = {c["id"] for c in _read_chunks(result["semantik_chunks_path"])}
    for row in rows:
        assert row["kept_chunk_id"] in live_ids, (
            f"ledger row {row!r} points at a chunk that is not in chunks.jsonl"
        )
        assert row["normalized_hash"]
        assert row["source_item_path"]


def test_dedup_on_still_passes_the_chunkset_manifest_gate(
    chunking_tool, tmp_path, monkeypatch
):
    """The sidecar is a FILE, not a manifest key — the gate must stay green.

    ``chunkset_manifest.schema.json`` is ``additionalProperties: false`` at the
    top level, so recording the dedup detail as a manifest key would fail the
    gate closed.
    """
    from lib.validators.chunkset_manifest import ChunksetManifestValidator

    staging = _duplicate_corpus(tmp_path)
    monkeypatch.setenv("ED4ALL_CHUNK_DEDUP", "1")
    monkeypatch.setenv("ED4ALL_HTML_PARSE_WORKERS", "1")
    result = _run(chunking_tool, staging, tmp_path / "LibV2-dedup-gate")

    gate = ChunksetManifestValidator().validate({
        "chunkset_manifest_path": result["manifest_path"],
    })
    criticals = [
        i for i in gate.issues
        if getattr(i.severity, "value", i.severity) == "critical"
    ]
    assert not criticals, f"chunkset gate raised criticals: {criticals}"
