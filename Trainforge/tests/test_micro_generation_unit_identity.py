"""Regression tests for micro-v1's per-unit resume-store identity.

``MicroStagedSynthesisProvider._resume_store`` keys the per-unit resume FILE
PATH on an identity whose only caller-supplied field is ``variant`` -- every
other field it re-derives from the chunk and draft it was already handed.  Two
consequences drive these tests:

* Production drafts carry no ``_micro_manifest_identity`` (only the pilot
  harness stamps one), so ``_resume_store`` hard-raised on every production
  call: micro-v1 could not complete a single unit outside its own pilot.
* ``_clean(None) -> ""`` meant a caller supplying a BLANK variant passed the
  equality check silently, collapsing every instruction variant of one chunk
  onto a single state file where variant 1 replays variant 0's stages.

The variant token comes from ``(kind, variant_index)`` -- the canonical
production unit key already used by the checkpoint cache and generation
journal -- so the micro resume store stays 1:1 with the outer journal's units.
"""
from __future__ import annotations

import pytest

from Trainforge.generators.providers._synthesis_common import SynthesisProviderError
from Trainforge.generators import staged_synthesis_micro as micro


class _Capture:
    """Minimal DecisionCapture stand-in: only ``output_dir`` is read here."""

    def __init__(self, output_dir):
        self.output_dir = output_dir


def _chunk(chunk_id="chunk-0001"):
    return {
        "id": chunk_id,
        "text": "A place value names the position of a digit in a numeral.",
        "learning_outcome_refs": ["CO-01"],
    }


def _draft():
    return {"provider": "local", "prompt": "p", "completion": "c"}


def _provider(tmp_path, monkeypatch=None):
    """Build the REAL micro provider through the production factory."""
    from Trainforge.generators.providers._synthesis_provider import (
        build_synthesis_provider,
    )

    import os

    previous = {
        name: os.environ.get(name)
        for name in (
            "TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1",
            "TRAINFORGE_STAGED_SYNTHESIS_V4",
        )
    }
    os.environ["TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1"] = "true"
    os.environ.pop("TRAINFORGE_STAGED_SYNTHESIS_V4", None)
    try:
        instance = build_synthesis_provider("local", synthesis_seed=1234)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert isinstance(instance, micro.MicroStagedSynthesisProvider)
    instance._capture = _Capture(tmp_path)
    return instance


def _store_path(provider, *, kind, chunk=None, draft=None):
    store = provider._resume_store(
        chunk=chunk or _chunk(), draft=draft or _draft(), kind=kind,
    )
    assert store is not None
    return store.path


# --------------------------------------------------------------------------
# The production blocker: an unstamped draft must resolve, not hard-raise.
# --------------------------------------------------------------------------
def test_unbound_production_draft_still_fails_loudly(tmp_path):
    """No manifest stamp AND no binding is a loud failure, never a default."""
    provider = _provider(tmp_path)
    with pytest.raises(SynthesisProviderError) as excinfo:
        provider._resume_store(
            chunk=_chunk(), draft=_draft(), kind="instruction",
        )
    assert excinfo.value.code == "staged_micro_draft_identity_missing"


def test_bound_generation_unit_resolves_an_unstamped_draft(tmp_path):
    provider = _provider(tmp_path)
    with micro.bind_micro_generation_unit(
        kind="instruction", variant_index=0,
    ):
        path = _store_path(provider, kind="instruction")
    assert path.parent.name == "micro_synthesis_state"
    assert path.suffix == ".jsonl"


# --------------------------------------------------------------------------
# The collision hazard: variant keys the file path.
# --------------------------------------------------------------------------
def test_two_instruction_variants_of_one_chunk_do_not_collide(tmp_path):
    provider = _provider(tmp_path)
    paths = []
    for variant_index in (0, 1):
        with micro.bind_micro_generation_unit(
            kind="instruction", variant_index=variant_index,
        ):
            paths.append(_store_path(provider, kind="instruction"))
    assert paths[0] != paths[1], (
        "instruction variants 0 and 1 shared one resume-store file; variant 1 "
        "would replay variant 0's terminal artifacts stage for stage"
    )


def test_instruction_and_preference_units_do_not_collide(tmp_path):
    provider = _provider(tmp_path)
    with micro.bind_micro_generation_unit(
        kind="instruction", variant_index=0,
    ):
        instruction_path = _store_path(provider, kind="instruction")
    with micro.bind_micro_generation_unit(
        kind="preference", variant_index=0,
    ):
        preference_path = _store_path(provider, kind="preference")
    assert instruction_path != preference_path


def test_the_same_unit_resolves_to_a_stable_path(tmp_path):
    """Resume depends on this: same unit, same file, across processes."""
    provider = _provider(tmp_path)
    seen = set()
    for _ in range(2):
        with micro.bind_micro_generation_unit(
            kind="instruction", variant_index=1,
        ):
            seen.add(_store_path(provider, kind="instruction"))
    assert len(seen) == 1


def test_distinct_chunks_do_not_collide(tmp_path):
    provider = _provider(tmp_path)
    paths = []
    for chunk_id in ("chunk-0001", "chunk-0002"):
        with micro.bind_micro_generation_unit(
            kind="instruction", variant_index=0,
        ):
            paths.append(
                _store_path(
                    provider, kind="instruction", chunk=_chunk(chunk_id),
                )
            )
    assert paths[0] != paths[1]


# --------------------------------------------------------------------------
# Hardening: a blank variant must not pass silently.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_variant_is_rejected(tmp_path, blank):
    """``_clean(None) -> ''`` used to make this pass the equality check."""
    provider = _provider(tmp_path)
    draft = dict(_draft())
    draft["_micro_manifest_identity"] = {
        "chunk_id": "chunk-0001",
        "chunk_sha256": "unused",
        "kind": "instruction",
        "variant": blank,
        "repetition": 0,
        "draft": {"provider": "local", "source_chunk_id": ""},
    }
    with pytest.raises(SynthesisProviderError) as excinfo:
        provider._resume_store(
            chunk=_chunk(), draft=draft, kind="instruction",
        )
    assert excinfo.value.code == "staged_micro_draft_identity_variant_blank"


def test_binding_kind_must_match_the_call(tmp_path):
    provider = _provider(tmp_path)
    with micro.bind_micro_generation_unit(
        kind="preference", variant_index=0,
    ):
        with pytest.raises(SynthesisProviderError) as excinfo:
            provider._resume_store(
                chunk=_chunk(), draft=_draft(), kind="instruction",
            )
    assert excinfo.value.code == "staged_micro_draft_identity_invalid"


# --------------------------------------------------------------------------
# The pilot's manifest binding stays authoritative and unchanged.
# --------------------------------------------------------------------------
def test_manifest_stamped_draft_wins_over_the_binding(tmp_path):
    provider = _provider(tmp_path)
    stamped = dict(_draft())
    stamped["_micro_manifest_identity"] = micro.build_micro_manifest_identity(
        _chunk(), kind="instruction", variant="A_pilot_cell", repetition=3,
        draft=_draft(),
    )
    with micro.bind_micro_generation_unit(
        kind="instruction", variant_index=0,
    ):
        stamped_path = _store_path(
            provider, kind="instruction", draft=stamped,
        )
        bound_path = _store_path(provider, kind="instruction")
    assert stamped_path != bound_path


def test_stamped_identity_that_disagrees_with_runtime_inputs_is_rejected(
    tmp_path,
):
    provider = _provider(tmp_path)
    stamped = dict(_draft())
    identity = micro.build_micro_manifest_identity(
        _chunk("chunk-0001"), kind="instruction", variant="A_pilot_cell",
        draft=_draft(),
    )
    identity["chunk_id"] = "chunk-9999"
    stamped["_micro_manifest_identity"] = identity
    with pytest.raises(SynthesisProviderError) as excinfo:
        provider._resume_store(
            chunk=_chunk("chunk-0001"), draft=stamped, kind="instruction",
        )
    assert excinfo.value.code == "staged_micro_draft_identity_invalid"


# --------------------------------------------------------------------------
# Thread safety of the binding.
# --------------------------------------------------------------------------
def test_binding_does_not_leak_into_other_threads(tmp_path):
    import threading

    observed = {}

    def _worker():
        observed["bound"] = micro.current_micro_generation_unit()

    with micro.bind_micro_generation_unit(
        kind="instruction", variant_index=0,
    ):
        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join()
    assert observed["bound"] is None
    assert micro.current_micro_generation_unit() is None


def test_micro_refuses_instruction_variants_greater_than_one(
    tmp_path, monkeypatch,
):
    """Driven through ``run_synthesis``, the production entry point.

    micro-v1 has no per-variant entropy: the focused chunk is identical across
    variants and every micro stage keys on the RUN-level synthesis seed, so
    variant 1 would emit a duplicate row for a second full ladder of model
    calls.  Refuse rather than duplicate; giving micro genuine per-variant
    entropy is a contract change, not a default.
    """
    import json

    from Trainforge import synthesize_training as st

    monkeypatch.setenv("TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1", "true")
    monkeypatch.delenv("TRAINFORGE_STAGED_SYNTHESIS_V4", raising=False)

    corpus = tmp_path / "course"
    (corpus / "corpus").mkdir(parents=True)
    (corpus / "corpus" / "chunks.jsonl").write_text(
        json.dumps(
            {
                "id": "chunk-0001",
                "text": "A place value names a digit's position in a numeral.",
                "chunk_type": "explanation",
                "learning_outcome_refs": ["CO-01"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        st.run_synthesis(
            corpus_dir=corpus,
            course_code="TEST",
            provider="mock",
            instruction_variants_per_chunk=2,
        )
    message = str(excinfo.value)
    assert "micro-v1" in message
    assert "instruction_variants_per_chunk=2" in message


def test_binding_is_released_on_exit(tmp_path):
    with micro.bind_micro_generation_unit(
        kind="instruction", variant_index=0,
    ):
        assert micro.current_micro_generation_unit() is not None
    assert micro.current_micro_generation_unit() is None
