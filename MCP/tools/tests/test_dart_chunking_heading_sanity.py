"""Live-path integration test for the chunk heading-sanity filter.

Re-chunks a real staged accessible-HTML conversion output through the live
``MCP/tools/pipeline_tools.py::_run_dart_chunking`` path (via the
``run_dart_chunking`` tool-registry entry) and asserts the CORPUS-AGNOSTIC
structural invariants of the ``lib/chunk_heading_sanity.py`` contract:

* WITH ``TRAINFORGE_HEADING_SANITY_FILTER=1`` (mirroring the corpus-gen flag
  set the pipeline auto-applies), the total chunk count is UNCHANGED vs the
  flag-off run (the filter repairs/flags headings, never adds or drops chunks),
  every chunk still validates against the v4 schema, and EVERY emitted heading
  is either (a) not suspect, (b) repaired to a non-suspect ancestor (heading
  changed vs the off-run for that slot, and the result is no longer suspect), or
  (c) left as-is and stamped ``heading_suspect`` — never a suspect heading
  relayed verbatim without a flag.
* WITH the flag OFF, the emitted ``section_heading``s are deterministic and NO
  chunk carries the ``heading_suspect`` flag (back-compat — the filter never
  mutates a heading when the gate is off).

The input HTML lives under the gitignored ``inputs/`` tree (no course-data
references in tracked code), so the test DISCOVERS it dynamically and
``pytest.skip()``s when absent — CI on a clean checkout stays green.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from lib.paths import PROJECT_ROOT

def _resolve_input_html() -> Path:
    """Discover a real accessible-HTML conversion output under ``inputs/``.

    The corpus lives under the gitignored ``inputs/`` tree (no course-data
    references in tracked code), so the path is DISCOVERED at runtime — glob
    every ``*/dart_in/*_accessible.html`` conversion output and take the first
    (sorted for determinism). ``pytest.skip()`` when none is present so a clean
    checkout stays green rather than erroring.
    """
    candidates = sorted(
        (PROJECT_ROOT / "inputs").glob("*/dart_in/*_accessible.html")
    )
    if candidates:
        return candidates[0]
    pytest.skip(
        "no staged accessible-HTML conversion output present under "
        "inputs/*/dart_in/ (gitignored corpus) — skipping live-path "
        "heading-sanity test"
    )


def _build_registry():
    from MCP.tools.pipeline_tools import _build_tool_registry

    return _build_tool_registry()


def _stage_html(tmp_path: Path, src_html: Path) -> Path:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / src_html.name).write_text(
        src_html.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return staging


def _run_chunking(tmp_path: Path, staging: Path, *, flag_on: bool) -> list:
    """Run the live ``run_dart_chunking`` tool and return the emitted chunks."""
    registry = _build_registry()
    run_dart_chunking = registry["run_dart_chunking"]

    libv2_root = tmp_path / ("libv2_on" if flag_on else "libv2_off")
    libv2_root.mkdir(parents=True, exist_ok=True)

    prev = os.environ.get("TRAINFORGE_HEADING_SANITY_FILTER")
    if flag_on:
        os.environ["TRAINFORGE_HEADING_SANITY_FILTER"] = "1"
    else:
        os.environ.pop("TRAINFORGE_HEADING_SANITY_FILTER", None)
    try:
        result_json = asyncio.run(
            run_dart_chunking(
                course_name="ALG_HEADING_SANITY",
                staging_dir=str(staging),
                libv2_root=str(libv2_root),
            )
        )
    finally:
        if prev is None:
            os.environ.pop("TRAINFORGE_HEADING_SANITY_FILTER", None)
        else:
            os.environ["TRAINFORGE_HEADING_SANITY_FILTER"] = prev

    result = json.loads(result_json)
    chunks_path = Path(result["semantik_chunks_path"])
    assert chunks_path.exists(), f"chunks.jsonl not emitted: {result}"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert chunks, "no chunks emitted"
    return chunks


def _heading(chunk) -> str:
    return (chunk.get("source") or {}).get("section_heading") or ""


def test_heading_sanity_structural_invariants(tmp_path):
    """The filter holds its contract on ANY corpus (structural, slug-agnostic).

    Asserts only invariants that hold for any discovered conversion output —
    no corpus-specific noise/real-heading literals (the discovered file is
    arbitrary, so content-specific expectations are not sound).
    """
    from lib.chunk_heading_sanity import is_suspect_section_heading

    src = _resolve_input_html()
    staging = _stage_html(tmp_path, src)

    on_chunks = _run_chunking(tmp_path, staging, flag_on=True)
    off_chunks = _run_chunking(tmp_path, staging, flag_on=False)

    # (a) total chunk count unchanged by the filter (repairs/flags headings,
    # never adds or drops chunks).
    assert len(on_chunks) == len(off_chunks), (
        f"chunk count changed: on={len(on_chunks)} off={len(off_chunks)}"
    )

    on_headings = [_heading(c) for c in on_chunks]
    off_headings = [_heading(c) for c in off_chunks]

    # (b) EVERY heading in the ON run satisfies the contract: it is either
    # not-suspect, repaired to a non-suspect ancestor (heading changed vs the
    # off-run slot), or left as-is and stamped ``heading_suspect``. A suspect
    # heading is NEVER relayed verbatim without the flag.
    for i, c in enumerate(on_chunks):
        h = on_headings[i]
        suspect_flag = (c.get("source") or {}).get("heading_suspect") is True
        if not is_suspect_section_heading(h):
            # Clean heading: must not carry the suspect flag.
            assert not suspect_flag, (
                f"clean heading {h[:50]!r} wrongly stamped heading_suspect"
            )
            continue
        # Heading still reads as suspect → it must have been LEFT as-is (no
        # clean ancestor) and therefore flagged; a repair would have replaced
        # it with a non-suspect heading.
        assert suspect_flag, (
            f"suspect heading {h[:50]!r} relayed verbatim WITHOUT the "
            f"heading_suspect flag in the ON run"
        )

    # (c) any slot whose heading CHANGED between off/on was a genuine repair:
    # suspect before, clean after (no false-positive demotion of a real title).
    changed_slots = [
        (off_headings[i], on_headings[i])
        for i in range(len(on_headings))
        if off_headings[i] != on_headings[i]
    ]
    for before, after in changed_slots:
        assert is_suspect_section_heading(before), (
            f"a NON-suspect heading {before[:50]!r} was changed to "
            f"{after[:50]!r} — false-positive demotion!"
        )
        assert not is_suspect_section_heading(after), (
            f"heading {before[:50]!r} repaired to STILL-suspect {after[:50]!r}"
        )

    # (d) a non-suspect (real) heading present off-flag survives on-flag — the
    # filter only ever touches suspect slots.
    off_clean = {h for h in off_headings if h and not is_suspect_section_heading(h)}
    on_present = set(on_headings)
    for real in off_clean:
        assert real in on_present, (
            f"clean heading {real[:50]!r} present off-flag but MISSING on-flag "
            f"(wrongly demoted)"
        )


def test_flag_off_byte_identical_headings(tmp_path):
    """With the flag OFF the emitted headings match the ON run's ORIGINALS.

    i.e. the off run is byte-identical to legacy behavior — the filter never
    mutates a heading when the gate is off.
    """
    src = _resolve_input_html()
    staging = _stage_html(tmp_path, src)

    off_a = _run_chunking(tmp_path, staging, flag_on=False)
    off_b = _run_chunking(tmp_path, staging, flag_on=False)

    # Deterministic: two off-runs produce identical headings.
    assert [_heading(c) for c in off_a] == [_heading(c) for c in off_b]
    # And NONE carry the heading_suspect flag when the gate is off.
    assert all(
        (c.get("source") or {}).get("heading_suspect") is None for c in off_a
    ), "heading_suspect stamped on a flag-OFF run (back-compat violation)"


def _build_chunk_v4_validator():
    """Offline-ref-resolving Draft202012 validator for chunk_v4.

    Mirrors ``Trainforge/tests/test_chunk_strict_validation.py::_build_validator``
    — registers every ``$id`` under ``schemas/`` so cross-schema ``$ref``s
    (e.g. the content_type taxonomy ``ChunkType``) resolve without network.
    """
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schemas_dir = PROJECT_ROOT / "schemas"
    schema_path = schemas_dir / "knowledge" / "chunk_v4.schema.json"
    if not schema_path.exists():
        pytest.skip("chunk_v4.schema.json not found")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    id_to_schema = {}
    for p in schemas_dir.rglob("*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            id_to_schema[sid] = s
    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    return Draft202012Validator(schema, registry=registry)


def test_on_run_chunks_schema_valid(tmp_path):
    """Every chunk emitted with the filter on still validates against v4."""
    src = _resolve_input_html()
    staging = _stage_html(tmp_path, src)
    on_chunks = _run_chunking(tmp_path, staging, flag_on=True)

    validator = _build_chunk_v4_validator()
    for c in on_chunks:
        errors = sorted(validator.iter_errors(c), key=lambda e: str(e.path))
        assert not errors, (
            f"chunk {c.get('id')} failed v4 schema: "
            f"{[e.message for e in errors[:3]]}"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
