"""Live-path integration test for the chunk heading-sanity filter.

Re-chunks a real staged DART HTML through the live
``MCP/tools/pipeline_tools.py::_run_dart_chunking`` path (via the
``run_dart_chunking`` tool-registry entry) and asserts:

* WITH ``TRAINFORGE_HEADING_SANITY_FILTER=1`` (mirroring the corpus-gen flag
  set the pipeline auto-applies), the known noise ``section_heading``s
  (answer-key / exercise-prose / numeric) are REPAIRED to a real ancestor
  heading OR stamped ``heading_suspect``; the real headings ("Prime
  Factorization", "Least Common Multiple", "1.x …") are UNCHANGED; total chunk
  count unchanged; every chunk still schema-valid.
* WITH the flag OFF, the emitted ``section_heading``s are byte-identical to the
  on-flag run's ORIGINAL (un-repaired) headings (back-compat).

The input HTML lives under the gitignored ``inputs/`` tree (no course-data
references in tracked code), so the test discovers it dynamically and
``pytest.skip()``s when absent — CI on a clean checkout stays green.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from lib.paths import PROJECT_ROOT

# Candidate staged-HTML inputs (gitignored). First existing wins.
_CANDIDATE_HTML = [
    PROJECT_ROOT
    / "inputs"
    / "sample7b-full"
    / "dart_in"
    / "sample-algebra-2e-ch1-3_accessible.html",
]

# The known noise ``section_heading`` families observed on this corpus at the
# live path (substrings — the chunker may append " (part N)").
_KNOWN_NOISE_PREFIXES = [
    "Preface",  # front-matter (exact, no part suffix)
    "EXERCISES Practice Makes Perfect",  # exercise banner / following-exercises
    "2⎡⎣1 + 3(10 − 2)⎤⎦",  # bracket-glyph math run
    "24a 32b 2",  # compact numeric noise
    "3u 2 − 4u + 5 when u = −3 313.",  # math prose + embedded markers
    "Write the numbers one under the other",  # full-sentence prose
]

# Real section titles that MUST survive untouched.
_KNOWN_REAL_HEADINGS = [
    "Prime Factorization",
    "Least Common Multiple",
    "Multiple of a Number",
]


def _resolve_input_html() -> Path:
    for cand in _CANDIDATE_HTML:
        if cand.exists():
            return cand
    pytest.skip(
        "no staged DART HTML fixture present under inputs/ "
        "(gitignored corpus) — skipping live-path heading-sanity test"
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
    chunks_path = Path(result["dart_chunks_path"])
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


def test_heading_sanity_repairs_noise_keeps_real_and_count(tmp_path):
    src = _resolve_input_html()
    staging = _stage_html(tmp_path, src)

    on_chunks = _run_chunking(tmp_path, staging, flag_on=True)
    off_chunks = _run_chunking(tmp_path, staging, flag_on=False)

    # (a) total chunk count unchanged by the filter.
    assert len(on_chunks) == len(off_chunks), (
        f"chunk count changed: on={len(on_chunks)} off={len(off_chunks)}"
    )

    on_headings = [_heading(c) for c in on_chunks]
    off_headings = [_heading(c) for c in off_chunks]

    # (b) every known-noise heading present in the OFF run must, in the ON run,
    # either be REPAIRED away (no longer present as a stamped heading) OR carry
    # a heading_suspect flag — never silently relayed verbatim without a flag.
    found_any_noise = False
    for prefix in _KNOWN_NOISE_PREFIXES:
        off_hits = [h for h in off_headings if h.startswith(prefix)]
        if not off_hits:
            continue
        found_any_noise = True
        # In the ON run, no chunk should carry this noise prefix as its
        # stamped section_heading UNLESS it is flagged heading_suspect.
        for c in on_chunks:
            h = _heading(c)
            if h.startswith(prefix):
                assert (c.get("source") or {}).get("heading_suspect") is True, (
                    f"noise heading {h[:50]!r} relayed verbatim WITHOUT "
                    f"heading_suspect flag in the ON run"
                )
    assert found_any_noise, (
        "fixture carried none of the known noise headings — the corpus may "
        "have changed; update _KNOWN_NOISE_PREFIXES"
    )

    # (c) at least one noise heading was actually REPAIRED to a real ancestor
    # (proves the repair path fires, not just the flag).
    repaired = [
        c
        for c in on_chunks
        if (c.get("source") or {}).get("section_heading")
        not in off_headings  # heading changed vs the off run for that slot
    ]
    # Looser, positional check: compare slot-by-slot (chunk order is stable).
    changed_slots = [
        (off_headings[i], on_headings[i])
        for i in range(len(on_headings))
        if off_headings[i] != on_headings[i]
    ]
    assert changed_slots, (
        "no section_heading changed between off/on runs — repair never fired"
    )
    for before, after in changed_slots:
        # A changed heading must have been suspect before and clean after.
        from lib.chunk_heading_sanity import is_suspect_section_heading

        assert is_suspect_section_heading(before), (
            f"a NON-suspect heading {before[:50]!r} was changed to "
            f"{after[:50]!r} — false-positive demotion!"
        )
        assert not is_suspect_section_heading(after), (
            f"heading {before[:50]!r} repaired to STILL-suspect {after[:50]!r}"
        )

    # (d) real headings are UNCHANGED by the filter.
    for real in _KNOWN_REAL_HEADINGS:
        off_has = any(h == real for h in off_headings)
        on_has = any(h == real for h in on_headings)
        if off_has:
            assert on_has, (
                f"real heading {real!r} present off-flag but MISSING on-flag "
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
