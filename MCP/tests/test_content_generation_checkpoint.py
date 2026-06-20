"""Unit tests for the crash-resume checkpoint sidecars wired into the
Courseforge two-pass content-generation tiers.

These exercise the PURE checkpoint helpers in
``MCP.tools.pipeline_tools`` (read / write / dedupe / cleanup / flag) and a
small in-test re-enactment of the per-block resume loop the outline +
rewrite tiers run — WITHOUT spinning up a pipeline run or an LLM. The whole
point is to prove the resume contract (skip already-authored units, dedupe
against fresh, no duplicates, flag-off = no sidecar, success deletes the
sidecar) in isolation.

Run (from the repo root):
    source .venv/bin/activate && \
        ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
        python -m pytest MCP/tests/test_content_generation_checkpoint.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from Courseforge.scripts.blocks import Block
from MCP.tools.pipeline_tools import (
    _COURSEFORGE_OUTLINE_CHECKPOINT_ENV,
    _COURSEFORGE_REWRITE_CHECKPOINT_ENV,
    _OUTLINE_CHECKPOINT_NAME,
    _REWRITE_CHECKPOINT_NAME,
    _append_block_checkpoint,
    _block_from_checkpoint_entry,
    _block_to_snake_case_entry,
    _checkpoint_enabled,
    _load_block_checkpoint,
    _remove_block_checkpoint,
)


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _mk_block(block_id: str, content: str = "authored prose") -> Block:
    """Build a minimal but realistic outline/rewrite Block stub."""
    return Block(
        block_id=block_id,
        block_type="explanation",
        page_id=block_id.split("#", 1)[0],
        sequence=0,
        content=content,
        objective_ids=("CO-01",),
        key_terms=("slope",),
    )


def _four_stub_ids() -> list[str]:
    return [
        "week_01_overview#explanation_a_0",
        "week_01_content_01#explanation_b_0",
        "week_02_application#explanation_c_0",
        "week_02_summary#explanation_d_0",
    ]


# --------------------------------------------------------------------------- #
# Flag parsing (default ON, parse-with-fallback)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env_name", [
    _COURSEFORGE_OUTLINE_CHECKPOINT_ENV,
    _COURSEFORGE_REWRITE_CHECKPOINT_ENV,
])
def test_checkpoint_flag_default_on(monkeypatch, env_name):
    monkeypatch.delenv(env_name, raising=False)
    assert _checkpoint_enabled(env_name) is True


@pytest.mark.parametrize("falsey", ["0", "false", "FALSE", "no", "Off", "  off  "])
def test_checkpoint_flag_falsey_disables(monkeypatch, falsey):
    monkeypatch.setenv(_COURSEFORGE_OUTLINE_CHECKPOINT_ENV, falsey)
    assert _checkpoint_enabled(_COURSEFORGE_OUTLINE_CHECKPOINT_ENV) is False


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "garbage", ""])
def test_checkpoint_flag_non_falsey_stays_on(monkeypatch, truthy):
    monkeypatch.setenv(_COURSEFORGE_OUTLINE_CHECKPOINT_ENV, truthy)
    assert _checkpoint_enabled(_COURSEFORGE_OUTLINE_CHECKPOINT_ENV) is True


# --------------------------------------------------------------------------- #
# Pure read / write / dedupe / cleanup
# --------------------------------------------------------------------------- #
def test_append_then_load_roundtrips_by_block_id(tmp_path):
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    blocks = [_mk_block(bid) for bid in _four_stub_ids()]
    for blk in blocks:
        _append_block_checkpoint(sidecar, _block_to_snake_case_entry(blk))

    loaded = _load_block_checkpoint(sidecar)
    assert set(loaded.keys()) == set(_four_stub_ids())
    # Each loaded entry rehydrates to an equivalent Block.
    for bid in _four_stub_ids():
        b = _block_from_checkpoint_entry(loaded[bid])
        assert b is not None
        assert b.block_id == bid
        assert b.content == "authored prose"


def test_load_missing_sidecar_is_empty(tmp_path):
    assert _load_block_checkpoint(tmp_path / "nope.jsonl") == {}


def test_load_last_line_wins_per_block_id(tmp_path):
    """A block re-authored across two interrupted attempts resolves to the
    most-recent entry (last-line-wins)."""
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    bid = "week_01_overview#explanation_a_0"
    _append_block_checkpoint(sidecar, _block_to_snake_case_entry(
        _mk_block(bid, content="first attempt")))
    _append_block_checkpoint(sidecar, _block_to_snake_case_entry(
        _mk_block(bid, content="second attempt")))

    loaded = _load_block_checkpoint(sidecar)
    assert len(loaded) == 1
    assert loaded[bid]["content"] == "second attempt"


def test_load_skips_blank_and_malformed_lines(tmp_path):
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    good = _mk_block("week_01_overview#explanation_a_0")
    _append_block_checkpoint(sidecar, _block_to_snake_case_entry(good))
    # Inject blank + non-JSON + JSON-without-block_id lines.
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("not json at all\n")
        fh.write('{"no_block_id": true}\n')

    loaded = _load_block_checkpoint(sidecar)
    assert set(loaded.keys()) == {"week_01_overview#explanation_a_0"}


def test_remove_sidecar(tmp_path):
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    _append_block_checkpoint(sidecar, _block_to_snake_case_entry(
        _mk_block("week_01_overview#explanation_a_0")))
    assert sidecar.exists()
    _remove_block_checkpoint(sidecar)
    assert not sidecar.exists()
    # Idempotent: removing a missing sidecar is a no-op (no raise).
    _remove_block_checkpoint(sidecar)


# --------------------------------------------------------------------------- #
# Resume-loop re-enactment (mirrors the outline/rewrite per-block loop)
# --------------------------------------------------------------------------- #
def _resume_loop(stub_ids, sidecar: Path, *, checkpoint_on: bool):
    """In-test re-enactment of the tier's per-block resume loop.

    Returns (final_block_ids, n_resumed, n_authored_fresh). ``author`` here
    stands in for the expensive ``route_with_self_consistency`` /
    ``route_rewrite_with_remediation`` dispatch; we record which stub ids it
    was actually invoked for so a test can assert the resumed ones were
    SKIPPED.
    """
    resume_map = _load_block_checkpoint(sidecar) if checkpoint_on else {}
    final_blocks: list[Block] = []
    authored_ids: list[str] = []
    n_resumed = 0
    n_fresh = 0
    for bid in stub_ids:
        if checkpoint_on:
            cached = resume_map.get(bid)
            if cached is not None:
                blk = _block_from_checkpoint_entry(cached)
                if blk is not None:
                    final_blocks.append(blk)
                    n_resumed += 1
                    continue
        # "author" the block (the expensive dispatch we are skipping above).
        authored = _mk_block(bid, content=f"freshly authored {bid}")
        authored_ids.append(bid)
        final_blocks.append(authored)
        n_fresh += 1
        if checkpoint_on:
            _append_block_checkpoint(
                sidecar, _block_to_snake_case_entry(authored))
    return final_blocks, authored_ids, n_resumed, n_fresh


def test_resume_skips_checkpointed_unit(tmp_path):
    """A unit appended to the sidecar is SKIPPED (no re-author) on resume."""
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    ids = _four_stub_ids()
    # Pre-seed the first two as already-authored.
    for bid in ids[:2]:
        _append_block_checkpoint(sidecar, _block_to_snake_case_entry(
            _mk_block(bid, content="prior attempt")))

    final, authored, n_resumed, n_fresh = _resume_loop(
        ids, sidecar, checkpoint_on=True)

    assert n_resumed == 2
    assert n_fresh == 2
    # The two pre-seeded ids were NOT re-authored.
    assert authored == ids[2:]
    # Resumed blocks kept their checkpointed content (byte-stable).
    by_id = {b.block_id: b for b in final}
    assert by_id[ids[0]].content == "prior attempt"
    assert by_id[ids[1]].content == "prior attempt"


def test_resume_plus_fresh_dedupe_to_full_set_no_dups(tmp_path):
    """Resume + fresh authoring dedupe to exactly the clean-run set."""
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    ids = _four_stub_ids()
    for bid in ids[:3]:
        _append_block_checkpoint(sidecar, _block_to_snake_case_entry(
            _mk_block(bid)))

    final, _authored, _r, _f = _resume_loop(ids, sidecar, checkpoint_on=True)

    final_ids = [b.block_id for b in final]
    # Exactly the four ids, in order, no duplicates.
    assert final_ids == ids
    assert len(final_ids) == len(set(final_ids))


def test_flag_off_writes_no_sidecar(tmp_path):
    """Flag OFF → no sidecar written + resume is a no-op (byte-identical)."""
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    ids = _four_stub_ids()
    final, authored, n_resumed, n_fresh = _resume_loop(
        ids, sidecar, checkpoint_on=False)

    assert not sidecar.exists()
    assert n_resumed == 0
    assert n_fresh == len(ids)
    # Every block was authored fresh (no resume).
    assert authored == ids
    assert [b.block_id for b in final] == ids


def test_successful_completion_deletes_sidecar(tmp_path):
    """A completed tier (final write done) deletes the resume sidecar."""
    sidecar = tmp_path / _REWRITE_CHECKPOINT_NAME
    ids = _four_stub_ids()
    final, _a, _r, _f = _resume_loop(ids, sidecar, checkpoint_on=True)
    assert sidecar.exists()  # written during the loop

    # Simulate the tier's final write + cleanup.
    final_jsonl = tmp_path / "blocks_final.jsonl"
    final_jsonl.write_text(
        "\n".join("" for _ in final), encoding="utf-8")
    _remove_block_checkpoint(sidecar)
    assert not sidecar.exists()


def test_simulated_midrun_resumes_to_full_four_no_dups(tmp_path):
    """Write 2 of 4 units, 'restart', resume to the full 4 with no dups.

    This is the headline crash scenario: a kill after unit 2 leaves a
    2-line sidecar; the restarted loop resumes those two and authors the
    remaining two — final set == clean-run set, zero duplicates.
    """
    sidecar = tmp_path / _OUTLINE_CHECKPOINT_NAME
    ids = _four_stub_ids()

    # --- Attempt 1: author 2 of 4, then "crash" (stop appending). ---
    for bid in ids[:2]:
        _append_block_checkpoint(sidecar, _block_to_snake_case_entry(
            _mk_block(bid, content=f"attempt1 {bid}")))
    # (kill here — units 3 & 4 never authored, no final JSONL written)

    # --- Attempt 2 (restart): full resume loop over all four ids. ---
    final, authored, n_resumed, n_fresh = _resume_loop(
        ids, sidecar, checkpoint_on=True)

    final_ids = [b.block_id for b in final]
    assert final_ids == ids                      # full set
    assert len(final_ids) == len(set(final_ids))  # no dups
    assert n_resumed == 2                         # units 1 & 2 from sidecar
    assert n_fresh == 2                           # units 3 & 4 authored now
    assert authored == ids[2:]                    # only the missing two

    # Resumed units are byte-stable from the checkpoint (attempt-1 content).
    by_id = {b.block_id: b for b in final}
    assert by_id[ids[0]].content == "attempt1 " + ids[0]
    assert by_id[ids[1]].content == "attempt1 " + ids[1]
    # The sidecar now carries all four (the two new ones were appended).
    assert set(_load_block_checkpoint(sidecar).keys()) == set(ids)
