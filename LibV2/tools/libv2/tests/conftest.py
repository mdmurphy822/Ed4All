"""LibV2 tools test configuration — ``real_models`` marker registration.

``pytest.ini`` runs with ``--strict-markers``, and the ``real_models``
marker is currently registered only in ``MCP/tests/conftest.py`` +
``Trainforge/tests/conftest.py``. Tests under
``LibV2/tools/libv2/tests/`` that opt into real embedding weights (the
``@pytest.mark.real_models`` MiniLM smoke) need the marker registered for
this collection root or pytest fails collection with an unknown-marker
error.

This file registers the marker only. It deliberately does NOT install an
autouse model-loader stub: the WS2 index build/read path is duck-typed
against the ``EmbeddingClient`` protocol and never loads
sentence-transformers itself, so the unit tests here run on the
deterministic ``fake`` provider with zero model loading. The autouse ST
stub belongs to the semantic-retrieval query path (a separate executor's
fence) where a real ST load could otherwise be triggered.
"""

from __future__ import annotations


def pytest_configure(config):
    """Register the ``real_models`` opt-in marker (``--strict-markers``)."""
    config.addinivalue_line(
        "markers",
        "real_models: the test intentionally loads a real embedding model "
        "(integration; skips when the weights are not in the HF cache).",
    )
