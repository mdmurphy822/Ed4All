"""Marketable-v1 D4 — license-clean-by-default gate for training-pair synthesis.

``Trainforge/synthesize_training.py::run_synthesis`` fails closed when an
Anthropic-family provider (``anthropic`` / ``claude_session``) is selected for
TRAINING-PAIR synthesis without the explicit
``TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS`` acknowledgment env. The emitted
instruction / preference pairs ARE the SLM training corpus (the adapter is a
derivative work of them), so per ``docs/LICENSING.md`` an Anthropic-family
provider is not a clean default here — it must be an explicit, fail-loud opt-in.

These tests drive the gate at the front of ``run_synthesis`` against a scratch
corpus, so the raise (or pass-through to the next branch) is exercised without
any live LLM dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Trainforge.synthesize_training import (
    SynthesisLicensingError,
    _ANTHROPIC_FAMILY_SYNTHESIS_PROVIDERS,
    run_synthesis,
)

_GATE_ENV = "TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS"
_PROVIDER_ENV = "TRAINFORGE_SYNTHESIS_PROVIDER"


def _scrub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_GATE_ENV, raising=False)
    monkeypatch.delenv(_PROVIDER_ENV, raising=False)


def _corpus(tmp_path: Path) -> Path:
    """Minimal corpus dir with an empty chunks.jsonl so run_synthesis reaches
    its provider gate before it tries to do anything substantive."""
    corpus_dir = tmp_path / "course"
    (corpus_dir / "corpus").mkdir(parents=True)
    (corpus_dir / "corpus" / "chunks.jsonl").write_text("", encoding="utf-8")
    return corpus_dir


# --------------------------------------------------------------------------
# Anthropic-family providers fail closed without the ack env.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(_ANTHROPIC_FAMILY_SYNTHESIS_PROVIDERS))
def test_anthropic_family_without_ack_fails_closed(tmp_path, monkeypatch, provider):
    _scrub(monkeypatch)
    with pytest.raises(SynthesisLicensingError) as exc:
        run_synthesis(
            corpus_dir=_corpus(tmp_path),
            course_code="LIC_TEST",
            provider=provider,
        )
    msg = str(exc.value)
    # Error names the offending provider, the clean alternatives, the ack env,
    # and the canonical licensing doc.
    assert provider in msg
    assert "docs/LICENSING.md" in msg
    assert _GATE_ENV in msg
    assert "local" in msg


def test_gate_set_matches_documented_anthropic_family():
    """The gated set is exactly the two Anthropic-family providers — never the
    license-clean ones (mock / local / together)."""
    assert _ANTHROPIC_FAMILY_SYNTHESIS_PROVIDERS == frozenset(
        {"anthropic", "claude_session"}
    )
    for clean in ("mock", "local", "together"):
        assert clean not in _ANTHROPIC_FAMILY_SYNTHESIS_PROVIDERS


# --------------------------------------------------------------------------
# With the ack env truthy, the gate passes (proceeds past the licensing check).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("truthy", ["true", "1", "yes", "on", "TRUE"])
def test_anthropic_with_ack_proceeds_past_gate(tmp_path, monkeypatch, truthy):
    """With the ack env set, the licensing gate does NOT raise. We assert the
    gate is cleared by checking the run fails LATER for a non-licensing reason
    (claude_session needs a dispatcher) — not a SynthesisLicensingError."""
    _scrub(monkeypatch)
    monkeypatch.setenv(_GATE_ENV, truthy)
    with pytest.raises(RuntimeError) as exc:
        run_synthesis(
            corpus_dir=_corpus(tmp_path),
            course_code="LIC_TEST",
            provider="claude_session",
        )
    # The dispatcher-required guard fires (next check after the gate), proving
    # the licensing gate was cleared.
    assert not isinstance(exc.value, SynthesisLicensingError)
    assert "dispatcher" in str(exc.value).lower() or "claude_session" in str(exc.value)


def test_falsey_ack_value_still_fails_closed(tmp_path, monkeypatch):
    """A non-truthy ack value (e.g. ``false`` / ``0``) does NOT unlock the
    gate — only the canonical truthy tokens count."""
    _scrub(monkeypatch)
    monkeypatch.setenv(_GATE_ENV, "false")
    with pytest.raises(SynthesisLicensingError):
        run_synthesis(
            corpus_dir=_corpus(tmp_path),
            course_code="LIC_TEST",
            provider="anthropic",
        )


# --------------------------------------------------------------------------
# Env override precedence: TRAINFORGE_SYNTHESIS_PROVIDER drives the gate even
# when the kwarg is a clean provider (the env wins inside run_synthesis).
# --------------------------------------------------------------------------


def test_env_provider_override_triggers_gate(tmp_path, monkeypatch):
    """A clean ``provider=`` kwarg is overridden by an Anthropic-family env
    var; the gate must fire on the EFFECTIVE (env-resolved) provider."""
    _scrub(monkeypatch)
    monkeypatch.setenv(_PROVIDER_ENV, "anthropic")
    with pytest.raises(SynthesisLicensingError):
        run_synthesis(
            corpus_dir=_corpus(tmp_path),
            course_code="LIC_TEST",
            provider="local",  # clean kwarg, but env override wins
        )


# --------------------------------------------------------------------------
# License-clean providers are never gated.
# --------------------------------------------------------------------------


def test_textbook_synthesis_provider_does_not_gate_training_synthesis(
    tmp_path, monkeypatch
):
    """Composition with A5: TEXTBOOK_SYNTHESIS_PROVIDER (authoring-adjacent
    textbook-structure synthesis) is a DISTINCT surface from training-pair
    synthesis. Setting it to ``anthropic`` must NOT make the training-pair
    ``run_synthesis`` gate fire — only TRAINFORGE_SYNTHESIS_PROVIDER (or the
    kwarg) governs this gate. No double-gating of authoring on the training
    path."""
    _scrub(monkeypatch)
    monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", "anthropic")
    try:
        run_synthesis(
            corpus_dir=_corpus(tmp_path),
            course_code="LIC_TEST",
            provider="local",  # training-pair provider is clean
        )
    except SynthesisLicensingError:
        pytest.fail(
            "TEXTBOOK_SYNTHESIS_PROVIDER=anthropic must not gate the "
            "training-pair run_synthesis path (distinct surface)"
        )
    except Exception:
        pass  # any non-licensing failure is fine


def test_local_provider_not_gated(tmp_path, monkeypatch):
    """``local`` never trips the licensing gate (no SynthesisLicensingError).
    It fails later for an unrelated reason (no local server / empty corpus),
    which is fine — the point is the gate let it through."""
    _scrub(monkeypatch)
    try:
        run_synthesis(
            corpus_dir=_corpus(tmp_path),
            course_code="LIC_TEST",
            provider="local",
        )
    except SynthesisLicensingError:  # pragma: no cover - would be a regression
        pytest.fail("local provider must never trip the D4 licensing gate")
    except Exception:
        # Any other failure (empty corpus, no server) is acceptable here — the
        # licensing gate is what we're asserting, and it did not fire.
        pass
