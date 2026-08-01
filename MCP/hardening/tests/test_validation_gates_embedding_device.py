"""Regression: an unavailable embedding DEVICE is never a silent gate pass.

Bug (same class as the W2.1 CUDA-OOM one in
``test_validation_gates_oom.py``, one layer up from the validators): the
statistical-tier validators were fixed to let
:class:`~lib.embedding.sentence_embedder.EmbeddingModelUnavailable` propagate
FATALLY instead of swallowing it into a vacuous pass — but all thirteen wirings
of those eight validators carry ``on_error: warn`` in
``config/workflows.yaml``, so ``ValidationGateManager.run_gate``'s broad
``except`` rewrote the raise back to ``passed=True``. The validator-level
fail-closed never reached the pipeline.

Fix: a TYPED passthrough in the gate manager, mirroring the OOM shape.
``EmbeddingModelUnavailable`` fails the gate closed regardless of
``behavior_on_error`` (no env escape hatch — the documented opt-out is the
explicit ``ED4ALL_EMBEDDING_DEVICE=cpu``).

The two embedding errors are DISTINCT contracts and this must not conflate
them: ``EmbeddingDepsMissing`` (optional ``[embedding]`` extras absent) keeps
honouring ``on_error: warn`` with a warning-severity ``passed=True``, and
``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` still flips it closed.

No GPU is touched anywhere here — every validator is a stub that raises.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from MCP.hardening.validation_gates import (
    GateBehavior,
    GateConfig,
    GateResult,
    GateSeverity,
    ValidationGateManager,
)
from lib.embedding.sentence_embedder import (
    EmbeddingDepsMissing,
    EmbeddingModelUnavailable,
)


class _RaisingValidator:
    name = "embedding_raiser"
    version = "1.0.0"

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def validate(self, inputs):  # noqa: ANN001, ANN201
        raise self._exc


class _DegradingValidator:
    """Mirrors the DEFAULT missing-extras path: returns, never raises."""

    name = "embedding_degrader"
    version = "1.0.0"

    def validate(self, inputs):  # noqa: ANN001, ANN201
        from MCP.hardening.validation_gates import GateIssue

        return GateResult(
            gate_id="",
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            issues=[GateIssue(
                severity="warning",
                code="EMBEDDING_DEPS_MISSING",
                message="sentence-transformers extras not installed; tier skipped.",
            )],
        )


def _install(manager: ValidationGateManager, validator, path: str) -> None:
    manager._validators[path] = validator


def _gate(path: str, on_error: GateBehavior = GateBehavior.WARN) -> GateConfig:
    """The production shape under test: severity warning + ``on_error: warn``."""
    return GateConfig(
        gate_id="embedding_gate",
        validator_path=path,
        severity=GateSeverity.WARNING,
        behavior_on_fail=GateBehavior.WARN,
        behavior_on_error=on_error,
    )


def _issue_codes(result: GateResult):
    return {i.code for i in result.issues}


def _device_exc(msg: str = "failed to construct SentenceTransformer") -> BaseException:
    return EmbeddingModelUnavailable(
        f"{msg} 'all-MiniLM-L6-v2' on device 'cuda': Torch not compiled with "
        f"CUDA enabled. No automatic CUDA→CPU downgrade is performed — set "
        f"ED4ALL_EMBEDDING_DEVICE=cpu to run this on CPU."
    )


# --------------------------------------------------------------------------- #
# (a) EmbeddingModelUnavailable fails the gate closed under on_error=warn
# --------------------------------------------------------------------------- #

def test_device_unavailable_fails_closed_despite_on_error_warn(monkeypatch):
    """The crux: ``on_error: warn`` must NOT rewrite the device raise to a pass."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    capture = MagicMock(name="capture")
    manager = ValidationGateManager(capture=capture)
    path = "lib.validators.source_coverage.DeviceStubWarn"
    _install(manager, _RaisingValidator(_device_exc()), path)

    result = manager.run_gate(_gate(path), inputs={})

    assert result.passed is False
    assert "EMBEDDING_MODEL_UNAVAILABLE" in _issue_codes(result)
    assert "VALIDATOR_ERROR" not in _issue_codes(result)
    assert "EMBEDDING_DEPS_MISSING" not in _issue_codes(result)
    assert result.validator_version == "embedding_device_unavailable"
    issue = next(i for i in result.issues if i.code == "EMBEDDING_MODEL_UNAVAILABLE")
    assert issue.severity == "critical"
    # Operator-actionable: names the explicit opt-out knob.
    assert "ED4ALL_EMBEDDING_DEVICE=cpu" in issue.suggestion
    # DecisionCapture fired on the device branch.
    assert capture.log_decision.called
    kwargs = capture.log_decision.call_args.kwargs
    assert kwargs["decision_type"] == "validation_result"
    assert "EMBEDDING_MODEL_UNAVAILABLE" in kwargs["decision"]


def test_device_unavailable_fails_closed_on_every_behavior(monkeypatch):
    """behavior_on_error is not consulted at all — all three block."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    for behavior in (GateBehavior.WARN, GateBehavior.BLOCK, GateBehavior.FAIL_CLOSED):
        manager = ValidationGateManager()
        path = f"lib.validators.source_coverage.DeviceStub_{behavior.value}"
        _install(manager, _RaisingValidator(_device_exc()), path)

        result = manager.run_gate(_gate(path, on_error=behavior), inputs={})

        assert result.passed is False, behavior
        assert "EMBEDDING_MODEL_UNAVAILABLE" in _issue_codes(result), behavior


def test_device_unavailable_ignores_require_embeddings_flag(monkeypatch):
    """Fatal regardless of TRAINFORGE_REQUIRE_EMBEDDINGS — including when OFF.

    The extras flag governs the OTHER contract; it must never be a way to
    soften a device failure.
    """
    for token in ("", "false", "0", "true"):
        if token:
            monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", token)
        else:
            monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
        manager = ValidationGateManager()
        path = f"lib.validators.source_coverage.DeviceStubFlag_{token or 'unset'}"
        _install(manager, _RaisingValidator(_device_exc()), path)

        result = manager.run_gate(_gate(path), inputs={})

        assert result.passed is False, token
        assert "EMBEDDING_MODEL_UNAVAILABLE" in _issue_codes(result), token


def test_device_unavailable_detected_through_wrapped_cause(monkeypatch):
    """A validator that re-raises the typed error WRAPPED is still caught."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    try:
        raise _device_exc()
    except EmbeddingModelUnavailable as inner:
        try:
            raise RuntimeError("validator wrapped the device error") from inner
        except RuntimeError as wrapped:
            outer: BaseException = wrapped

    manager = ValidationGateManager()
    path = "lib.validators.source_coverage.DeviceStubWrapped"
    _install(manager, _RaisingValidator(outer), path)

    result = manager.run_gate(_gate(path), inputs={})

    assert result.passed is False
    assert "EMBEDDING_MODEL_UNAVAILABLE" in _issue_codes(result)


def test_device_unavailable_wins_over_oom_message_sniff(monkeypatch):
    """A device error whose message mentions CUDA OOM still fails CLOSED.

    ``SentenceEmbedder._ensure_model`` wraps ANY construction failure (an OOM
    included) in ``EmbeddingModelUnavailable``, and ``is_cuda_oom`` sniffs
    messages — so without the device branch being checked first, the OOM branch
    would hand this back to ``behavior_on_error=warn`` (the silent pass again).
    """
    monkeypatch.delenv("ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM", raising=False)
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    manager = ValidationGateManager()
    path = "lib.validators.source_coverage.DeviceStubOomMsg"
    _install(
        manager,
        _RaisingValidator(EmbeddingModelUnavailable(
            "failed to construct SentenceTransformer on device 'cuda': "
            "CUDA out of memory. Tried to allocate 2GiB"
        )),
        path,
    )

    result = manager.run_gate(_gate(path), inputs={})

    assert result.passed is False
    assert "EMBEDDING_MODEL_UNAVAILABLE" in _issue_codes(result)
    assert "VALIDATOR_OOM" not in _issue_codes(result)


# --------------------------------------------------------------------------- #
# (b) EmbeddingDepsMissing is UNAFFECTED — warning + passed=True under warn
# --------------------------------------------------------------------------- #

def test_deps_missing_types_are_unrelated():
    """The two contracts must stay on unrelated types — neither subclasses the
    other, or the device passthrough would silently capture the extras path."""
    assert not issubclass(EmbeddingModelUnavailable, EmbeddingDepsMissing)
    assert not issubclass(EmbeddingDepsMissing, EmbeddingModelUnavailable)


def test_deps_missing_default_path_warns_and_passes(monkeypatch):
    """The DEFAULT extras degrade never raises at all: the validator returns a
    warning-severity ``passed=True`` result and the gate manager leaves it be."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    manager = ValidationGateManager()
    path = "lib.validators.source_coverage.DepsDegrade"
    _install(manager, _DegradingValidator(), path)

    result = manager.run_gate(_gate(path), inputs={})

    assert result.passed is True
    assert _issue_codes(result) == {"EMBEDDING_DEPS_MISSING"}
    assert all(i.severity == "warning" for i in result.issues)


def test_deps_missing_raise_warns_and_passes_under_on_error_warn(monkeypatch):
    """A RAISED EmbeddingDepsMissing with the strict flag off still honours
    ``on_error: warn`` → warning severity, ``passed=True``. Unchanged contract."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    manager = ValidationGateManager()
    path = "lib.validators.source_coverage.DepsRaiseWarn"
    _install(
        manager,
        _RaisingValidator(EmbeddingDepsMissing("sentence-transformers not installed")),
        path,
    )

    result = manager.run_gate(_gate(path), inputs={})

    assert result.passed is True
    assert "EMBEDDING_DEPS_MISSING" in _issue_codes(result)
    assert "EMBEDDING_MODEL_UNAVAILABLE" not in _issue_codes(result)
    issue = next(i for i in result.issues if i.code == "EMBEDDING_DEPS_MISSING")
    assert issue.severity == "warning"


# --------------------------------------------------------------------------- #
# (c) TRAINFORGE_REQUIRE_EMBEDDINGS=true still flips (b) closed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("token", ["true", "1", "yes", "on"])
def test_deps_missing_strict_mode_fails_closed_despite_warn(monkeypatch, token):
    """Strict mode is an operator opt-in; ``on_error: warn`` must not undo it."""
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", token)
    capture = MagicMock(name="capture")
    manager = ValidationGateManager(capture=capture)
    path = f"lib.validators.source_coverage.DepsStrict_{token}"
    _install(
        manager,
        _RaisingValidator(EmbeddingDepsMissing("strict mode: extras required")),
        path,
    )

    result = manager.run_gate(_gate(path), inputs={})

    assert result.passed is False
    issue = next(i for i in result.issues if i.code == "EMBEDDING_DEPS_MISSING")
    assert issue.severity == "critical"
    assert result.validator_version == "embedding_deps_missing"
    assert capture.log_decision.called


def test_deps_missing_strict_garbage_token_stays_permissive(monkeypatch):
    """Parse-with-fallback: a garbage strict token is OFF, so warn is honoured."""
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "banana")
    manager = ValidationGateManager()
    path = "lib.validators.source_coverage.DepsStrictGarbage"
    _install(
        manager,
        _RaisingValidator(EmbeddingDepsMissing("extras absent")),
        path,
    )

    result = manager.run_gate(_gate(path), inputs={})

    assert result.passed is True
    assert "EMBEDDING_DEPS_MISSING" in _issue_codes(result)


# --------------------------------------------------------------------------- #
# The real wirings: every statistical-tier gate in config/workflows.yaml
# --------------------------------------------------------------------------- #

#: The EIGHT validators on the embedding-DEVICE contract — i.e. the ones that
#: load through ``try_load_embedder()`` and can therefore surface
#: ``EmbeddingModelUnavailable``. Keep this list in sync with
#: ``git grep -l EmbeddingModelUnavailable -- 'lib/validators/*.py'``; between
#: them they account for the 13 wirings in ``config/workflows.yaml`` asserted
#: by :func:`test_every_configured_statistical_tier_gate_fails_closed_on_device_error`.
#:
#: ``bloom.classifier_disagreement`` is deliberately EXCLUDED despite sharing
#: the ``[embedding]`` extras and appearing in the graceful-degrade list in
#: ``lib/CLAUDE.md``: it wraps ``lib.classifiers.bloom_bert_ensemble`` rather
#: than ``try_load_embedder()``, so it never raises this exception and its
#: device policy is the NLI/BERT one, not this contract.
_STATISTICAL_TIER_VALIDATORS = (
    "lib.validators.objective_assessment_similarity.",
    "lib.validators.concept_example_similarity.",
    "lib.validators.objective_roundtrip_similarity.",
    "lib.validators.co_terminal_alignment.",
    "lib.validators.source_coverage.",
    "lib.validators.rewrite_source_grounding.",
    "lib.validators.terminal_objective_source_grounding.",
    "lib.validators.distractor_misconception_alignment.",
)


def _statistical_tier_gate_dicts():
    """Yield ``(workflow, phase, gate_dict)`` for every statistical-tier gate."""
    import yaml

    from lib.paths import get_project_root

    config_path = get_project_root() / "config" / "workflows.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for wf_name, wf in (data.get("workflows") or {}).items():
        for phase in wf.get("phases") or []:
            for gate in phase.get("validation_gates") or []:
                validator = gate.get("validator", "")
                if any(validator.startswith(p) for p in _STATISTICAL_TIER_VALIDATORS):
                    yield wf_name, phase.get("name"), gate


def test_statistical_tier_gates_are_wired_on_error_warn():
    """Guard the premise: these gates really do carry ``on_error: warn``.

    If that ever changes the passthrough is still correct, but this test is
    what makes the rest of the file a REAL regression rather than a
    hypothetical one — ``on_error: warn`` is exactly the config that used to
    rewrite the device raise into ``passed=True``.
    """
    gates = list(_statistical_tier_gate_dicts())
    assert gates, "no statistical-tier gates found in config/workflows.yaml"
    warn_wired = [g for _, _, g in gates
                  if (g.get("behavior") or {}).get("on_error") == "warn"]
    assert warn_wired, "expected at least one on_error: warn wiring"


def test_every_configured_statistical_tier_gate_fails_closed_on_device_error(monkeypatch):
    """End-to-end over the REAL gate configs: no wiring can warn-pass a device
    failure. This is the check the adversarial probe failed before the fix."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    monkeypatch.delenv("ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM", raising=False)

    checked = 0
    for wf_name, phase_name, gate_dict in _statistical_tier_gate_dicts():
        config = GateConfig.from_dict(dict(gate_dict))
        manager = ValidationGateManager()
        # Stub the validator so nothing loads a model or touches a device.
        _install(manager, _RaisingValidator(_device_exc()), config.validator_path)

        result = manager.run_gate(config, inputs={})

        where = f"{wf_name}/{phase_name}/{config.gate_id}"
        assert result.passed is False, f"device failure warn-passed at {where}"
        assert "EMBEDDING_MODEL_UNAVAILABLE" in _issue_codes(result), where
        checked += 1

    assert checked >= 8, f"expected the full statistical-tier wiring set, saw {checked}"


# --------------------------------------------------------------------------- #
# (d) Neither branch hijacks an ordinary validator error
# --------------------------------------------------------------------------- #

def test_ordinary_validator_error_unchanged(monkeypatch):
    """A plain exception still routes to the generic VALIDATOR_ERROR path."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)
    manager = ValidationGateManager()
    path = "lib.validators.source_coverage.PlainErr"
    _install(manager, _RaisingValidator(ValueError("some ordinary validator bug")), path)

    result = manager.run_gate(_gate(path), inputs={})

    assert "VALIDATOR_ERROR" in _issue_codes(result)
    assert "EMBEDDING_MODEL_UNAVAILABLE" not in _issue_codes(result)
    assert "EMBEDDING_DEPS_MISSING" not in _issue_codes(result)
    assert result.validator_version == "error"
    assert result.passed is True  # on_error=warn, unchanged
