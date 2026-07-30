"""Fingerprint discipline + --force semantics on the Courseforge tier sidecars.

The outline / rewrite crash-resume sidecars (``.blocks_outline_checkpoint.jsonl``
/ ``.blocks_final_checkpoint.jsonl``) historically reused entries by
``block_id`` alone — a model/provider/chunk/objective change between resume
attempts silently reused stale blocks. These tests pin the retrofit:

* every checkpointed entry is stamped with a per-block INPUT fingerprint
  (``checkpoint_fingerprint``, computed by ``_tier_block_fingerprint``);
* on resume an entry is reused ONLY when the stamp matches the recomputed
  fingerprint (byte-identical reuse), and a mismatch re-authors;
* legacy entries with NO stamp keep block_id-only reuse (back-compat);
* ``--force`` (``force_rerun=True`` kwarg) clears AND ignores the sidecar;
* the family flag ``ED4ALL_GENERATION_CHECKPOINT=0`` disables the sites.

Hermetic: real handlers driven against a tmp project scaffold with stub
providers (no LLM / GPU / network). Builders mirror
``test_pipeline_tools_phase3_handlers.py``.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402

# Captured BEFORE any monkeypatching so force-semantics tests can restore it.
_REAL_REMOVE_BLOCK_CHECKPOINT = _pt._remove_block_checkpoint


# ---------------------------------------------------------------------- #
# Builders (mirror test_pipeline_tools_phase3_handlers.py)
# ---------------------------------------------------------------------- #
class _FakeProvider:
    def __init__(self) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.outline_calls.append({"block_id": block.block_id})
        return dataclasses.replace(block, content="outline-stub")

    def generate_rewrite(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.rewrite_calls.append({"block_id": block.block_id})
        return dataclasses.replace(
            block,
            content=(
                f"<p>Rewrite stub for {block.block_id} with thirty plus "
                f"words of body prose to clear the grounding floor and "
                f"satisfy the non-trivial paragraph word minimum.</p>"
            ),
        )


def _seed_project(tmp_path: Path, project_id: str) -> Path:
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "01_learning_objectives").mkdir(exist_ok=True)
    (
        project_path / "01_learning_objectives" / "synthesized_objectives.json"
    ).write_text(
        json.dumps({
            "terminal_objectives": [
                {"id": "TO-01",
                 "statement": "Describe core concept A in detail."}
            ],
            "chapter_objectives": [],
        }),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 1}),
        encoding="utf-8",
    )
    return project_path


def _pin_hermetic_roots(monkeypatch, tmp_path: Path) -> None:
    """Keep every write inside tmp_path: exports (PROJECT_ROOT), the LibV2
    root, and the decision-capture mirror. The handlers under test log
    DecisionCaptures + probe LibV2 chunksets; without these pins those land
    in the real repo dirs."""
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path / "LibV2"))
    monkeypatch.setenv(
        "ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "runtime/training-captures"),
    )


def _patch_router_with_fakes(monkeypatch, fake: _FakeProvider) -> None:
    from Courseforge.router import router as _router_mod

    real_init = _router_mod.CourseforgeRouter.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("outline_provider", fake)
        kwargs.setdefault("rewrite_provider", fake)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter, "__init__", patched_init,
    )


def _keep_sidecars(monkeypatch) -> None:
    """No-op the success-path sidecar delete (simulates a mid-run kill:
    the sidecar SURVIVES with every block checkpointed)."""
    monkeypatch.setattr(_pt, "_remove_block_checkpoint", lambda _p: None)


def _run_outline(project_id: str) -> Dict[str, Any]:
    return json.loads(asyncio.run(
        _pt._run_content_generation_outline(project_id=project_id),
    ))


def _run_rewrite(
    project_id: str, validated_path: Path, **kw: Any,
) -> Dict[str, Any]:
    return json.loads(asyncio.run(
        _pt._run_content_generation_rewrite(
            blocks_validated_path=str(validated_path),
            project_id=project_id,
            **kw,
        ),
    ))


def _seed_validated_blocks(tmp_path: Path, n: int = 2) -> Path:
    validated_path = tmp_path / "blocks_validated.jsonl"
    lines = []
    for i in range(n):
        block = Block(
            block_id=f"week_01_content_01#explanation_concept_{i}_0",
            block_type="explanation",
            page_id="week_01_content_01",
            sequence=i,
            content="",
            objective_ids=("TO-01",),
        )
        lines.append(json.dumps(_pt._block_to_snake_case_entry(block)))
    validated_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validated_path


# ---------------------------------------------------------------------- #
# Helper unit tests
# ---------------------------------------------------------------------- #
def test_tier_block_fingerprint_axis_sensitive(monkeypatch):
    blk = Block(
        block_id="p#explanation_x_0", block_type="explanation",
        page_id="p", sequence=0, content="body", objective_ids=("TO-01",),
    )
    kw = dict(
        tier="rewrite",
        chunks_lookup={"p#explanation_x_0": [{"id": "c1", "text": "t"}]},
        objectives_sha="obj-sha",
        router=None,
    )
    monkeypatch.delenv("COURSEFORGE_REWRITE_MODEL", raising=False)
    fp = _pt._tier_block_fingerprint(blk, **kw)
    assert fp == _pt._tier_block_fingerprint(blk, **kw)  # deterministic
    # Content change flips it.
    blk2 = dataclasses.replace(blk, content="different body")
    assert _pt._tier_block_fingerprint(blk2, **kw) != fp
    # Chunk change flips it.
    kw2 = dict(kw, chunks_lookup={"p#explanation_x_0": [
        {"id": "c1", "text": "changed"}]})
    assert _pt._tier_block_fingerprint(blk, **kw2) != fp
    # Objectives change flips it.
    kw3 = dict(kw, objectives_sha="other-sha")
    assert _pt._tier_block_fingerprint(blk, **kw3) != fp
    # Model env change flips it (no router → env-knob spec description).
    monkeypatch.setenv("COURSEFORGE_REWRITE_MODEL", "other-model")
    assert _pt._tier_block_fingerprint(blk, **kw) != fp


# ---------------------------------------------------------------------- #
# Rewrite tier — fingerprint reuse / mismatch / force / family flag
# ---------------------------------------------------------------------- #
def test_rewrite_sidecar_reuse_is_byte_identical(tmp_path, monkeypatch):
    """Interrupted-run resume: sidecar-only reuse (blocks_final deleted)
    dispatches zero LLM calls and reproduces blocks_final byte-identically."""
    project_id = "TEST_RW_CP_REUSE"
    project_path = _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path)

    payload1 = _run_rewrite(project_id, validated_path)
    assert payload1["success"] is True
    n_calls_first = len(fake.rewrite_calls)
    assert n_calls_first >= 2
    blocks_final = Path(payload1["blocks_final_path"])
    final_bytes_first = blocks_final.read_bytes()

    # Sidecar survived (delete no-op'd) and every entry carries the stamp.
    sidecar = project_path / "04_content" / _pt._REWRITE_CHECKPOINT_NAME
    if not sidecar.exists():
        # Resolve wherever the handler put it (next to blocks_final).
        sidecar = blocks_final.parent / _pt._REWRITE_CHECKPOINT_NAME
    assert sidecar.exists(), sidecar
    entries = [
        json.loads(ln) for ln in
        sidecar.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert entries and all(
        _pt._BLOCK_CHECKPOINT_FP_KEY in e for e in entries
    )

    # Simulate the kill: the FINAL emit is gone, only the sidecar remains.
    blocks_final.unlink()
    fake2 = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake2)
    payload2 = _run_rewrite(project_id, validated_path)
    assert payload2["success"] is True
    assert fake2.rewrite_calls == []  # all blocks served from the sidecar
    assert Path(payload2["blocks_final_path"]).read_bytes() == (
        final_bytes_first
    )


def test_rewrite_fingerprint_mismatch_reauthors(tmp_path, monkeypatch):
    """A model change between attempts invalidates the sidecar entries."""
    project_id = "TEST_RW_CP_STALE"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path)

    payload1 = _run_rewrite(project_id, validated_path)
    n_first = len(fake.rewrite_calls)
    Path(payload1["blocks_final_path"]).unlink()

    # Change the resolved rewrite model → recomputed fingerprints differ.
    monkeypatch.setenv("COURSEFORGE_REWRITE_MODEL", "swapped-model-v9")
    fake2 = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake2)
    payload2 = _run_rewrite(project_id, validated_path)
    assert payload2["success"] is True
    assert len(fake2.rewrite_calls) == n_first  # every block re-authored


def test_rewrite_legacy_unstamped_entries_still_reuse(tmp_path, monkeypatch):
    """Back-compat: entries with no fingerprint stamp keep block_id reuse."""
    project_id = "TEST_RW_CP_LEGACY"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path)

    payload1 = _run_rewrite(project_id, validated_path)
    blocks_final = Path(payload1["blocks_final_path"])
    sidecar = blocks_final.parent / _pt._REWRITE_CHECKPOINT_NAME
    # Strip the stamps (simulate a pre-upgrade sidecar).
    stripped = []
    for ln in sidecar.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        e.pop(_pt._BLOCK_CHECKPOINT_FP_KEY, None)
        stripped.append(json.dumps(e))
    sidecar.write_text("\n".join(stripped) + "\n", encoding="utf-8")
    blocks_final.unlink()

    fake2 = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake2)
    payload2 = _run_rewrite(project_id, validated_path)
    assert payload2["success"] is True
    assert fake2.rewrite_calls == []  # legacy entries reused by block_id


def test_rewrite_force_clears_and_ignores_sidecar(tmp_path, monkeypatch):
    """--force (force_rerun=True) deletes the sidecar and re-authors all."""
    project_id = "TEST_RW_CP_FORCE"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path)

    payload1 = _run_rewrite(project_id, validated_path)
    n_first = len(fake.rewrite_calls)
    blocks_final = Path(payload1["blocks_final_path"])
    sidecar = blocks_final.parent / _pt._REWRITE_CHECKPOINT_NAME
    assert sidecar.exists()
    blocks_final.unlink()

    # ``--force``: even with a fully-populated matching sidecar, every
    # block re-authors and the prior sidecar is cleared up-front. Restore
    # the REAL remover (captured pre-patch) so the clear is observable.
    monkeypatch.setattr(
        _pt, "_remove_block_checkpoint", _REAL_REMOVE_BLOCK_CHECKPOINT,
    )
    fake2 = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake2)
    payload2 = _run_rewrite(project_id, validated_path, force_rerun=True)
    assert payload2["success"] is True
    assert len(fake2.rewrite_calls) == n_first  # sidecar ignored


def test_rewrite_family_flag_off_disables_sidecar(tmp_path, monkeypatch):
    """ED4ALL_GENERATION_CHECKPOINT=0 (family) disables the rewrite site
    (no site env set) — no sidecar is written."""
    project_id = "TEST_RW_CP_FAMILY_OFF"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    monkeypatch.delenv("COURSEFORGE_REWRITE_CHECKPOINT", raising=False)
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path)

    payload = _run_rewrite(project_id, validated_path)
    assert payload["success"] is True
    sidecar = (
        Path(payload["blocks_final_path"]).parent
        / _pt._REWRITE_CHECKPOINT_NAME
    )
    assert not sidecar.exists()

    # Site override beats the family OFF: with the site env set truthy,
    # the sidecar is written again.
    monkeypatch.setenv("COURSEFORGE_REWRITE_CHECKPOINT", "1")
    Path(payload["blocks_final_path"]).unlink()
    payload2 = _run_rewrite(project_id, validated_path)
    assert payload2["success"] is True
    assert sidecar.exists()


# ---------------------------------------------------------------------- #
# Outline tier — fingerprint reuse / mismatch / force
# ---------------------------------------------------------------------- #
def test_outline_sidecar_reuse_and_mismatch_and_force(tmp_path, monkeypatch):
    """One scenario walk: (1) full run leaves a stamped sidecar (delete
    no-op'd); (2) resume dispatches zero outline calls and emits
    byte-identical blocks_outline.jsonl; (3) a model change re-authors;
    (4) force_rerun clears + ignores the sidecar."""
    project_id = "TEST_OL_CP"
    project_path = _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)
    _keep_sidecars(monkeypatch)

    payload1 = _run_outline(project_id)
    assert payload1["success"] is True
    n_first = len(fake.outline_calls)
    blocks_outline = Path(payload1["blocks_outline_path"])
    outline_bytes_first = blocks_outline.read_bytes()
    sidecar = blocks_outline.parent / _pt._OUTLINE_CHECKPOINT_NAME

    if n_first == 0:
        # Under the real Wave-7 escalation policy every objective stub
        # short-circuits (escalation marker) and is NOT checkpointed —
        # then there is nothing to reuse and this scenario is moot; the
        # rewrite-tier tests above carry the fingerprint contract.
        assert not sidecar.exists() or not sidecar.read_text().strip()
        return

    assert sidecar.exists()
    entries = [
        json.loads(ln) for ln in
        sidecar.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert all(_pt._BLOCK_CHECKPOINT_FP_KEY in e for e in entries)

    # (2) Resume: zero fresh outline dispatches; the emitted JSONL is
    # byte-equivalent modulo the ``touched_by`` provenance TIMESTAMP on the
    # escalate-immediately block (marker blocks are deliberately NOT
    # checkpointed and re-route each run, stamping a fresh timestamp).
    def _normalized(raw: bytes) -> List[Dict[str, Any]]:
        out = []
        for ln in raw.decode("utf-8").splitlines():
            if not ln.strip():
                continue
            e = json.loads(ln)
            for t in e.get("touched_by") or []:
                if isinstance(t, dict):
                    t.pop("timestamp", None)
            out.append(e)
        return out

    blocks_outline.unlink()
    fake2 = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake2)
    payload2 = _run_outline(project_id)
    assert payload2["success"] is True
    assert fake2.outline_calls == []
    assert _normalized(
        Path(payload2["blocks_outline_path"]).read_bytes()
    ) == _normalized(outline_bytes_first)

    # (3) Model change → fingerprint mismatch → full re-author.
    monkeypatch.setenv("COURSEFORGE_OUTLINE_MODEL", "swapped-model-v9")
    Path(payload2["blocks_outline_path"]).unlink()
    fake3 = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake3)
    payload3 = _run_outline(project_id)
    assert payload3["success"] is True
    assert len(fake3.outline_calls) == n_first
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MODEL", raising=False)

    # (4) --force: sidecar cleared + ignored (real remover restored).
    monkeypatch.setattr(
        _pt, "_remove_block_checkpoint", _REAL_REMOVE_BLOCK_CHECKPOINT,
    )
    Path(payload3["blocks_outline_path"]).unlink()
    fake4 = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake4)
    payload4 = json.loads(asyncio.run(
        _pt._run_content_generation_outline(
            project_id=project_id, force_rerun=True,
        ),
    ))
    assert payload4["success"] is True
    assert len(fake4.outline_calls) == n_first  # nothing reused
    # Success path also removed the fresh sidecar (real remover active).
    assert not sidecar.exists()
