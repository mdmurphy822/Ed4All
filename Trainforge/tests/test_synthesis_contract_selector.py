"""Regression tests for the ``--synthesis-contract`` operator selector.

The flag was declared in argparse and never read: ``args.synthesis_contract``
appeared nowhere in ``main()``, so an operator following the documented
invocation silently got whatever ``TRAINFORGE_STAGED_SYNTHESIS_V4`` /
``TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1`` happened to say.

These tests drive the PRODUCTION entry points -- ``build_parser()`` +
``main()`` and ``build_synthesis_provider`` -- not the resolver in isolation,
because a resolver that is never called is exactly the defect under test.
"""
from __future__ import annotations

import pytest

from Trainforge import synthesize_training as st
from Trainforge.generators._synthesis_provider import build_synthesis_provider
from Trainforge.generators.staged_synthesis_micro import (
    MicroStagedSynthesisProvider,
    staged_synthesis_micro_v1_enabled,
)
from Trainforge.generators.staged_synthesis_provider import (
    StagedSynthesisProvider,
    staged_synthesis_v4_enabled,
)

_ENV_V4 = st.ENV_STAGED_SYNTHESIS_V4
_ENV_MICRO = st.ENV_STAGED_SYNTHESIS_MICRO_V1


@pytest.fixture(autouse=True)
def _clean_contract_env(monkeypatch):
    """Every test starts from an unset contract environment."""
    monkeypatch.delenv(_ENV_V4, raising=False)
    monkeypatch.delenv(_ENV_MICRO, raising=False)


def _parse(*argv):
    return st.build_parser().parse_args(
        ["--corpus", "/nonexistent-corpus", *argv]
    )


def test_parser_exposes_every_documented_contract_spelling():
    for selection in st.SYNTHESIS_CONTRACT_CHOICES:
        args = _parse("--synthesis-contract", selection)
        assert args.synthesis_contract == selection


@pytest.mark.parametrize(
    "selection,expect_v4,expect_micro",
    [
        ("legacy", False, False),
        ("staged-v4", True, False),
        ("micro-v1", False, True),
    ],
)
def test_main_resolves_the_flag_into_the_contract_switches(
    monkeypatch, selection, expect_v4, expect_micro,
):
    """The flag ALONE must select the contract, with no env var set.

    ``main()`` is stopped immediately after the selector so the assertion is
    about what the contract switches say when the run begins -- the same
    moment ``build_synthesis_provider`` reads them.
    """
    observed = {}

    def _stop(*_args, **_kwargs):
        observed["v4"] = staged_synthesis_v4_enabled()
        observed["micro"] = staged_synthesis_micro_v1_enabled()
        raise SystemExit(0)

    monkeypatch.setattr(st, "_parse_stratify_arg", _stop)
    with pytest.raises(SystemExit):
        st.main(_parse("--synthesis-contract", selection))

    assert observed == {"v4": expect_v4, "micro": expect_micro}


def test_omitting_the_flag_leaves_the_ambient_environment_untouched(
    monkeypatch,
):
    """No flag == historical env-driven path, byte-for-byte."""
    monkeypatch.setenv(_ENV_V4, "true")
    observed = {}

    def _stop(*_args, **_kwargs):
        observed["v4"] = staged_synthesis_v4_enabled()
        observed["micro"] = staged_synthesis_micro_v1_enabled()
        raise SystemExit(0)

    monkeypatch.setattr(st, "_parse_stratify_arg", _stop)
    with pytest.raises(SystemExit):
        st.main(_parse())

    assert observed == {"v4": True, "micro": False}


def test_ambient_conflict_fails_loudly_instead_of_overriding(monkeypatch):
    """An operator must never silently get a contract they did not ask for.

    Overriding a conflicting ambient switch would be just as wrong as ignoring
    the flag: the run would differ from what the environment declares.
    """
    monkeypatch.setenv(_ENV_V4, "true")

    def _unreached(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("synthesis started despite a contract conflict")

    monkeypatch.setattr(st, "_parse_stratify_arg", _unreached)
    with pytest.raises(SystemExit) as excinfo:
        st.main(_parse("--synthesis-contract", "micro-v1"))

    message = str(excinfo.value)
    assert "--synthesis-contract micro-v1" in message
    assert _ENV_V4 in message
    # The environment is left exactly as the operator set it.
    assert staged_synthesis_v4_enabled() is True


def test_agreeing_ambient_value_is_not_a_conflict(monkeypatch):
    monkeypatch.setenv(_ENV_MICRO, "1")
    observed = {}

    def _stop(*_args, **_kwargs):
        observed["micro"] = staged_synthesis_micro_v1_enabled()
        raise SystemExit(0)

    monkeypatch.setattr(st, "_parse_stratify_arg", _stop)
    with pytest.raises(SystemExit):
        st.main(_parse("--synthesis-contract", "micro-v1"))

    assert observed == {"micro": True}


def test_selector_reaches_the_provider_factory(monkeypatch):
    """End of the wire: the flag decides which provider class is built."""
    st.apply_synthesis_contract_selection("micro-v1")
    micro = build_synthesis_provider("local", synthesis_seed=7)
    assert isinstance(micro, MicroStagedSynthesisProvider)

    monkeypatch.delenv(_ENV_V4, raising=False)
    monkeypatch.delenv(_ENV_MICRO, raising=False)
    st.apply_synthesis_contract_selection("staged-v4")
    assert isinstance(
        build_synthesis_provider("local", synthesis_seed=7),
        StagedSynthesisProvider,
    )

    monkeypatch.delenv(_ENV_V4, raising=False)
    monkeypatch.delenv(_ENV_MICRO, raising=False)
    st.apply_synthesis_contract_selection("legacy")
    legacy = build_synthesis_provider("local", synthesis_seed=7)
    assert not isinstance(
        legacy, (StagedSynthesisProvider, MicroStagedSynthesisProvider)
    )


def test_unknown_spelling_is_rejected_by_the_resolver():
    with pytest.raises(ValueError):
        st.resolve_synthesis_contract_env("micro")
