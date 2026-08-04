"""Keep Trainforge tests independent of optional model integrations.

The autouse fixture disables the Bloom compatibility resolver, active NLI
loader, and embedding loader. This keeps synthetic unit fixtures focused on
promotion contracts: Bloom classification abstains, NLI follows its documented
unavailable-model path, and embedding criteria use their documented lexical
paths. Tests that intentionally exercise provisioned integrations opt out with
``@pytest.mark.real_models``.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register the ``real_models`` opt-out marker (``--strict-markers``)."""
    config.addinivalue_line(
        "markers",
        "real_models: opt out of the autouse model-loader stub; the test "
        "intentionally loads provisioned optional model integrations.",
    )



# The repository-level conftest owns local synthesis reachability for every
# synthesis test family.


@pytest.fixture(autouse=True)
def _stub_real_models(request, monkeypatch):
    """Force optional Bloom, NLI, and embedding loaders to abstain.

    Routes promotion and claim-support checks through their documented
    unavailable-model paths so unit tests never load model weights. Opt out with
    ``@pytest.mark.real_models``.
    """
    if request.node.get_closest_marker("real_models"):
        yield
        return

    # Make Bloom-dependent promotion checks exercise the unavailable-model
    # contract without loading optional weights.
    try:
        from lib.validators.pair.promotion import (
            TrainingPairPromotionValidator,
        )

        monkeypatch.setattr(
            TrainingPairPromotionValidator,
            "_resolve_bloom_classifier",
            lambda self: None,
        )
    except Exception:  # noqa: BLE001 — module may be absent in a slim slice
        pass

    # Make claim-support checks exercise their unavailable-model contract.
    try:
        from lib.classifiers.nli_classifier import NliClassifier

        monkeypatch.setattr(
            NliClassifier,
            "get_or_load",
            classmethod(lambda cls: None),
        )
    except Exception:  # noqa: BLE001
        pass

    # Keep embedding-dependent criteria on their documented dependency-missing
    # paths so unit tests do not load optional embedding weights.
    try:
        from lib.validators.pair.promotion import TrainingPairPromotionValidator
        monkeypatch.setattr(
            TrainingPairPromotionValidator,
            "_resolve_embedder",
            lambda self: (None, False),
        )
    except Exception:  # noqa: BLE001
        pass

    yield
