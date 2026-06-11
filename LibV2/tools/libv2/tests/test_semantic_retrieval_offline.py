"""Offline assertion: the semantic retrieval query path does zero network I/O.

Companion to ``tests/offline/test_retrieval_offline.py`` (which proves the
*lexical* BM25 path is offline-clean). That suite never exercises the vector
index, so the on-device semantic engine — the only retrieval arm that touches
an embedding client — had no ``no_network`` coverage anywhere under ``LibV2/``.
This file closes that gap.

The whole point is that the query path needs nothing off-box: provisioning
(model download / index build) happens earlier, and a query is pure file I/O
(read ``embeddings.npy`` / ``id_map.json`` / ``manifest.json``) plus a
deterministic in-process ``fake`` embed of the query string. We prove it by
building the index BEFORE entering the guard, then running the engine="semantic"
roundtrip end-to-end *under* :func:`lib.testing.no_network.no_network` (loopback
blocked — the semantic path is not the localhost-model-server answer path).

Both anti-poisoning arms of the fake-index guard are exercised inside the
offline scope so neither failure mode hides a sneaky network reach:

* without ``ED4ALL_EMBEDDING_ALLOW_FAKE`` the load must raise the typed
  :class:`FakeIndexRefused` (and that refusal is itself offline);
* with it set, results return.

A final sanity assertion confirms the guard actually trips on a raw socket
connect attempt, so a silently no-op guard could never make this suite pass.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

pytest.importorskip(
    "numpy", reason="[embedding] extras absent — semantic-retrieval tests need numpy"
)

from lib.embedding.providers import build_embedding_client
from lib.testing.no_network import NetworkBlockedError, no_network

from LibV2.tools.libv2.retriever import retrieve_chunks
from LibV2.tools.libv2.semantic_retriever import semantic_retrieve_chunks
from LibV2.tools.libv2.vector_index import FakeIndexRefused, build_vector_index

_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "retrieval"
    / "mini_course"
)
_COURSE_SLUG = "mini-retrieval-101"


def _fake_client():
    """Deterministic test embedding client — sha256-seeded vectors, no I/O."""
    return build_embedding_client(provider_name="fake")


@pytest.fixture()
def fake_built_repo(tmp_path: Path) -> Path:
    """Materialize the shared mini-course fixture into a tmp LibV2 layout and
    build a ``fake``-provider vector index for it.

    The index build runs OUTSIDE the network guard (the build is the
    provision-time step where downloads would happen for a real provider; only
    the *query* path is contracted offline). Returns the LibV2 repo root
    (``.../LibV2``) whose ``courses/<slug>/`` is query-ready.
    """
    if not _FIXTURE_ROOT.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"mini-course fixture missing at {_FIXTURE_ROOT}")
    import shutil

    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / _COURSE_SLUG
    shutil.copytree(_FIXTURE_ROOT, course_dir)

    # Provision the index with the deterministic fake provider (no downloads,
    # no network) BEFORE any test enters the offline guard.
    manifest = build_vector_index(course_dir, client=_fake_client())
    assert manifest.embedding_provider == "fake"
    assert manifest.chunks_count == 7  # the fixture ships 7 v4 chunks
    return libv2_root


# A query string that self-retrieves chunk mini_alpha_chunk_001 (the fake
# embedder maps identical strings to identical vectors -> cosine ~1.0).
_TARGET_QUERY = (
    "Vector Stores and Embeddings\nA vector store is a database that indexes "
    "high-dimensional embedding vectors so that approximate nearest-neighbour "
    "search can retrieve semantically similar passages."
)


def test_semantic_retrieve_runs_offline_with_allow_fake(
    fake_built_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    """engine="semantic" roundtrip returns results under no_network() when the
    fake index is explicitly allowed — the whole query path is offline-clean."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")

    with no_network():  # loopback blocked: the semantic path needs nothing on-box
        results = semantic_retrieve_chunks(
            fake_built_repo,
            _TARGET_QUERY,
            course_slug=_COURSE_SLUG,
            limit=3,
            client=_fake_client(),
            include_rationale=True,
        )

    assert results, "semantic retrieve returned no results under no_network()"
    # Self-retrieval: the exact-text chunk ranks #1 at cosine ~1.0.
    assert results[0].chunk_id == "mini_alpha_chunk_001"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].rationale["engine"] == "semantic"


def test_semantic_via_retrieve_chunks_offline(
    fake_built_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    """The public ``retrieve_chunks(engine="semantic")`` dispatch path is also
    fully offline (client auto-built from the manifest, fake-allowed)."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")

    with no_network():
        results = retrieve_chunks(
            fake_built_repo,
            _TARGET_QUERY,
            course_slug=_COURSE_SLUG,
            engine="semantic",
            limit=3,
        )

    assert results
    assert results[0].chunk_id == "mini_alpha_chunk_001"


def test_fake_index_refused_offline_without_allow_flag(
    fake_built_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    """Anti-poisoning arm A: without ED4ALL_EMBEDDING_ALLOW_FAKE the query must
    raise the typed FakeIndexRefused — and that refusal is itself offline (no
    network reach precedes the fail-closed)."""
    monkeypatch.delenv("ED4ALL_EMBEDDING_ALLOW_FAKE", raising=False)

    with no_network():
        with pytest.raises(FakeIndexRefused):
            semantic_retrieve_chunks(
                fake_built_repo,
                _TARGET_QUERY,
                course_slug=_COURSE_SLUG,
                limit=3,
                client=_fake_client(),
                # allow_fake defaults to None -> resolves from the (unset) env.
            )


def test_no_network_guard_trips_on_socket_attempt():
    """Sanity: the guard is not a silent no-op — a raw INET connect under it
    raises NetworkBlockedError. (If this passed with a broken guard, the
    offline assertions above would be meaningless.)"""
    with no_network():
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("93.184.216.34", 80), timeout=1)
