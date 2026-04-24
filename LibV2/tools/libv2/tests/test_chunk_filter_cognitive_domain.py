"""Wave 70 — ``cognitive_domain`` filter on ChunkFilter + CLI.

Covers:

* Positive / negative ``_matches_filter`` behavior when the chunk
  carries ``cognitive_domain`` directly (Wave 60/69 emit).
* The filter is case-insensitive (corpora with mixed case still match).
* Chunks missing the field are rejected when the filter is active.
* CLI wiring: ``libv2 retrieve --cognitive-domain factual ...`` fires
  the filter through to the underlying retrieve_chunks call.

Note: this wave adds the filter plumbing. Production chunks may not
carry ``cognitive_domain`` yet — the feature is dependent on Wave 69
extending the chunk emit. Until then, using the flag will return zero
results (documented behavior).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest import mock

import pytest
from click.testing import CliRunner

from LibV2.tools.libv2.cli import main
from LibV2.tools.libv2.retriever import ChunkFilter, _matches_filter


# -------------------------------------------------------------------- #
# _matches_filter behavior
# -------------------------------------------------------------------- #


class TestCognitiveDomainFilter:
    def test_positive_match_factual(self):
        chunk = {"cognitive_domain": "factual", "source": {}}
        assert _matches_filter(chunk, ChunkFilter(cognitive_domain="factual"))

    def test_positive_match_conceptual(self):
        chunk = {"cognitive_domain": "conceptual", "source": {}}
        assert _matches_filter(chunk, ChunkFilter(cognitive_domain="conceptual"))

    def test_negative_mismatch(self):
        chunk = {"cognitive_domain": "factual", "source": {}}
        assert not _matches_filter(
            chunk, ChunkFilter(cognitive_domain="procedural")
        )

    def test_missing_field_is_rejected(self):
        """A chunk without ``cognitive_domain`` must be rejected when
        the filter is active — we're filtering for a concrete property,
        not 'anything goes if unknown'."""
        chunk = {"source": {}}  # no cognitive_domain key
        assert not _matches_filter(
            chunk, ChunkFilter(cognitive_domain="factual")
        )

    def test_case_insensitive_match(self):
        """Corpora may emit mixed case (Wave 60 vs 69 drift); match both."""
        chunk = {"cognitive_domain": "Factual", "source": {}}
        assert _matches_filter(
            chunk, ChunkFilter(cognitive_domain="factual")
        )
        assert _matches_filter(
            chunk, ChunkFilter(cognitive_domain="FACTUAL")
        )

    def test_none_filter_does_not_fire(self):
        """Filter=None (the default) must not touch chunks, including
        those missing the field."""
        chunk = {"source": {}}
        assert _matches_filter(chunk, ChunkFilter())  # no filter specified


# -------------------------------------------------------------------- #
# CLI wiring via CliRunner
# -------------------------------------------------------------------- #


def _make_min_repo(tmp_path: Path) -> Path:
    """Build a skeleton repo root with courses/ and catalog/ so the CLI
    root-detection logic sees it."""
    (tmp_path / "courses").mkdir()
    (tmp_path / "catalog").mkdir()
    return tmp_path


def test_cli_cognitive_domain_flag_threads_to_retrieve():
    """Invoke ``libv2 retrieve --cognitive-domain factual`` and assert
    the filter reached ``retrieve_chunks`` via kwarg inspection."""
    runner = CliRunner()
    captured: dict = {}

    def _fake_retrieve_chunks(**kwargs):
        # Capture the kwargs so the test can assert the filter flowed through.
        captured.update(kwargs)
        return []  # empty result is fine — we're asserting on the call.

    with runner.isolated_filesystem() as fs:
        repo = _make_min_repo(Path(fs))
        with mock.patch(
            "LibV2.tools.libv2.retriever.retrieve_chunks",
            side_effect=_fake_retrieve_chunks,
        ):
            result = runner.invoke(
                main,
                [
                    "--repo",
                    str(repo),
                    "retrieve",
                    "sample query",
                    "--cognitive-domain",
                    "factual",
                ],
            )

    assert result.exit_code == 0, result.output
    assert captured.get("cognitive_domain") == "factual", (
        f"cognitive_domain did not reach retrieve_chunks: {captured}"
    )


def test_cli_cognitive_domain_flag_is_optional():
    """Absence of the flag passes None (not empty string) — so the
    filter short-circuits on the is-Truthy check in ``_matches_filter``."""
    runner = CliRunner()
    captured: dict = {}

    def _fake_retrieve_chunks(**kwargs):
        captured.update(kwargs)
        return []

    with runner.isolated_filesystem() as fs:
        repo = _make_min_repo(Path(fs))
        with mock.patch(
            "LibV2.tools.libv2.retriever.retrieve_chunks",
            side_effect=_fake_retrieve_chunks,
        ):
            result = runner.invoke(
                main,
                ["--repo", str(repo), "retrieve", "sample query"],
            )

    assert result.exit_code == 0, result.output
    assert captured.get("cognitive_domain") is None
    assert captured.get("hierarchy_level") is None


def test_cli_multi_retrieve_cognitive_domain_flag():
    """Same check on the ``multi-retrieve`` command (both CLI verbs
    must expose the filter per Wave 70 scope)."""
    runner = CliRunner()
    captured: dict = {}

    # multi-retrieve calls MultiQueryRetriever.retrieve under the hood;
    # we patch that to capture the kwargs.
    class _FakeResults:
        results = []
        fusion_method = "rrf"
        deduplication_stats = {}
        coherence_metrics = {}
        result_count = 0
        intent_coverage = {}
        all_intents_covered = True

        def to_dict(self):
            return {}

    def _fake_retrieve(self, **kwargs):
        captured.update(kwargs)
        return _FakeResults()

    with runner.isolated_filesystem() as fs:
        repo = _make_min_repo(Path(fs))
        with mock.patch(
            "LibV2.tools.libv2.multi_retriever.MultiQueryRetriever.retrieve",
            autospec=True,
            side_effect=_fake_retrieve,
        ):
            result = runner.invoke(
                main,
                [
                    "--repo",
                    str(repo),
                    "multi-retrieve",
                    "sample query",
                    "--cognitive-domain",
                    "conceptual",
                    "--no-decompose",
                ],
            )

    assert result.exit_code == 0, result.output
    assert captured.get("cognitive_domain") == "conceptual", (
        f"cognitive_domain did not reach multi-retrieve: {captured}"
    )
