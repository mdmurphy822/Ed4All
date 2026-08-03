"""Keep Trainforge unit tests independent of optional model integrations.

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



# NOTE: the "absent local synthesis backend → skip" hookwrapper lives in the
# repo-root conftest.py (lib.testing.reachability.make_local_synthesis_skip_hook)
# so it covers every synthesis test family from one place. Not duplicated here.


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

    # Seam 1: bloom classifier resolver on the promotion validator.
    # Returning None → _classify_bloom returns (None, None) → criterion 6
    # (low_bloom_alignment) skips. Patched on the class so every instance
    # (including ones the call-site constructs internally) is covered.
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

    # Seam 2: NLI classifier singleton loader. Returning None routes
    # claim_support through its deps-missing pass arm (warning-severity,
    # passed=True) instead of loading the DeBERTa NLI model.
    try:
        from lib.classifiers.nli_classifier import NliClassifier

        monkeypatch.setattr(
            NliClassifier,
            "get_or_load",
            classmethod(lambda cls: None),
        )
    except Exception:  # noqa: BLE001
        pass

    # Seam 3: sentence-embedder resolver on the promotion validator.
    # Returning (None, False) routes criteria 2/3/4 through the designed
    # deps-missing fallback arms (answer-support floor -1.0, jaccard
    # distinctness, prompt-chunk floor deactivated) so unit tests never
    # load sentence-transformers weights. Mirrors seams 1-2.
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
