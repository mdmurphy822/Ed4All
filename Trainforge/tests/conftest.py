"""Trainforge test configuration — model-hermeticity fixtures.

Why this file exists (Family-A test-cluster fix, Subtask-25 follow-up):

Unit tests under ``Trainforge/tests/`` synthesize toy instruction /
preference pairs and run them through the real pair-promotion validator
(``lib.validators.pair.promotion.TrainingPairPromotionValidator``). That
validator, plus the claim-support path it shares, will lazy-load real
~700MB BERT/DeBERTa models when the optional ``[bert]`` / ``[embedding]``
extras are installed in the venv — which they ARE here since 2026-05-08.

Two consequences without this fixture:

1. The 3-model BERT ensemble + the DeBERTa NLI model load on first use,
   adding minutes per suite (the cluster's pre-fix runtime was 18:09 vs
   8:42).
2. Worse, real DeBERTa NLI *legitimately* rejects the toy fixtures these
   unit tests feed it (a one-sentence stub chunk genuinely doesn't entail
   a synthetic claim), so tests that assert a pair PASSES promotion fail
   for the "right" reason on real models but the wrong reason for a unit
   test. Unit tests must exercise the DESIGNED deps-missing degrade arms,
   not the heavyweight model arms.

So this autouse fixture monkeypatches the two model-loader seams to
return ``None``, routing both validators through their graceful-degrade
("extras absent") code paths — exactly the arms a slim-install CI worker
would hit. Integration tests that genuinely want the real models opt out
with ``@pytest.mark.real_models``.

The repo-root ``conftest.py`` already provides LibV2 / training-captures /
state-runs tmp isolation; this file is additive and scoped to model
loading only — it does NOT duplicate that isolation.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register the ``real_models`` opt-out marker (``--strict-markers``)."""
    config.addinivalue_line(
        "markers",
        "real_models: opt out of the autouse model-loader stub; the test "
        "intentionally loads the real BERT/DeBERTa models (integration).",
    )



# NOTE: the "absent local synthesis backend → skip" hookwrapper lives in the
# repo-root conftest.py (lib.testing.reachability.make_local_synthesis_skip_hook)
# so it covers every synthesis test family from one place. Not duplicated here.


@pytest.fixture(autouse=True)
def _stub_real_models(request, monkeypatch):
    """Force the BERT-ensemble + NLI model loaders to return ``None``.

    Routes ``TrainingPairPromotionValidator`` (criterion 6 bloom) and
    the claim-support NLI path through their designed deps-missing pass
    arms so unit tests never load real weights. Opt out with
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
