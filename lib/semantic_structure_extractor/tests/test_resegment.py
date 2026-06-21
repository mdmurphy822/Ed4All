"""WS6b — collapse re-segmentation tests.

Exercises ``resegment_collapsed_structure`` with deterministic injected
embedding vectors (3 gaussian blobs in document order) so the suite needs
neither the real bge model nor a network.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.semantic_structure_extractor import resegment as resegment_mod
from lib.semantic_structure_extractor.resegment import (
    resegment_collapsed_structure,
)


class _StubEmbedClient:
    """Encode a section's text deterministically by a topic marker.

    Each section's ``section_text`` is prefixed with a topic id (``T0``/
    ``T1``/``T2``). We map that to one of three near-orthogonal blob centers
    plus a tiny deterministic jitter, and L2-normalize. Contiguous topic runs
    in document order → contiguous Ward clusters.
    """

    def __init__(self, dim: int = 16, seed: int = 7) -> None:
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Three well-separated blob centers.
        self._centers = {
            "T0": rng.normal(0, 1, dim) + np.array([5.0] + [0.0] * (dim - 1)),
            "T1": rng.normal(0, 1, dim) + np.array([0.0, 5.0] + [0.0] * (dim - 2)),
            "T2": rng.normal(0, 1, dim) + np.array([0.0, 0.0, 5.0] + [0.0] * (dim - 3)),
        }

    def encode_batch(self, texts):
        rows = []
        for i, t in enumerate(texts):
            topic = "T0"
            for key in self._centers:
                # Match the topic marker anywhere in the embed text (the
                # production embed text is ``headingText + ". " + section_text``
                # so the marker is not necessarily at index 0).
                if key in t:
                    topic = key
                    break
            center = self._centers[topic]
            jitter = np.sin(np.arange(self.dim) + i) * 0.05
            v = center + jitter
            v = v / (np.linalg.norm(v) or 1.0)
            rows.append(v)
        return np.asarray(rows, dtype=np.float32)


def _section(idx: int, heading: str, text: str) -> dict:
    return {
        "id": f"orig_s{idx}",
        "headingLevel": 3,
        "headingText": heading,
        "headingId": f"hid-{idx}",
        "section_text": text,
        "contentBlocks": [],
        "subsections": [],
    }


def _collapsed_three_topics(n: int = 60) -> list:
    """One chapter / ``n`` sections in 3 contiguous embedding-distinct runs."""
    sections = []
    per = n // 3
    for i in range(n):
        if i < per:
            topic, heading = "T0", f"Counting topic section {i}"
        elif i < 2 * per:
            topic, heading = "T1", f"Algebra topic section {i}"
        else:
            topic, heading = "T2", f"Geometry topic section {i}"
        sections.append(
            _section(i, heading, f"{topic} prose body for section {i} " * 4)
        )
    return [
        {
            "id": "ch1",
            "headingLevel": 2,
            "headingText": "Mega Chapter",
            "headingId": "mega",
            "explicitObjectives": [],
            "contentBlocks": [],
            "source_file": "/staging/everything.html",
            "chapter_text": "whole-book prose",
            "sections": sections,
        }
    ]


def test_resegment_splits_collapsed_into_contiguous_chapters():
    chapters = _collapsed_three_topics(60)
    client = _StubEmbedClient()

    result = resegment_collapsed_structure(chapters, embed_client=client)

    # K = clamp(round(60/13), 6, 20) = round(4.6)=5 -> clamped up to 6.
    assert len(result) == 6
    assert result is not chapters

    # Partition covers ALL original sections IN ORDER, no drops/dupes.
    flat = [s["id"] for ch in result for s in ch["sections"]]
    expected = [f"orig_s{i}" for i in range(60)]
    assert flat == expected

    # Each pseudo-chapter is a contiguous slice (ids strictly increasing run).
    for ch in result:
        idxs = [int(s["id"].removeprefix("orig_s")) for s in ch["sections"]]
        assert idxs == list(range(idxs[0], idxs[-1] + 1))

    # Each carries non-empty chapter_text and a unique chapter_id; no
    # source_file (so chunks_for_chapter takes order_fallback).
    seen_ids = set()
    for ch in result:
        assert ch["chapter_text"].strip()
        assert ch["id"] not in seen_ids
        seen_ids.add(ch["id"])
        assert "source_file" not in ch
    assert seen_ids == {f"ch{i}" for i in range(1, 7)}


def test_resegment_mega_chapter_among_siblings():
    """The [3, 141] regression: a tiny front-matter chapter + a 60-section
    mega-chapter. The strict ``len(chapters)==1`` trigger silently passed this
    through, poisoning downstream phases. Now the mega-chapter MUST be
    re-segmented while the small sibling is preserved in place; ids re-mint in
    document order."""
    small = {
        "id": "ch1",
        "headingLevel": 2,
        "headingText": "Front Matter",
        "headingId": "fm",
        "source_file": "/staging/everything.html",
        "chapter_text": "preface prose",
        "sections": [_section(900 + i, f"front {i}", f"preface body {i}") for i in range(3)],
    }
    mega = _collapsed_three_topics(60)[0]  # one 60-section mega-chapter
    chapters = [small, mega]

    result = resegment_collapsed_structure(
        chapters, embed_client=_StubEmbedClient()
    )

    # Small sibling preserved as ch1 + 6 pseudo-chapters from the mega = 7.
    assert len(result) == 7
    assert result is not chapters

    # ch1 is the preserved front-matter chapter (its 3 sections intact).
    assert [s["id"] for s in result[0]["sections"]] == [
        f"orig_s{900 + i}" for i in range(3)
    ]

    # ids re-minted sequentially in document order across the rebuilt list.
    assert [ch["id"] for ch in result] == [f"ch{i}" for i in range(1, 8)]

    # The 60 mega sections are covered IN ORDER across pseudo-chapters 2..7,
    # no drops/dupes.
    mega_flat = [s["id"] for ch in result[1:] for s in ch["sections"]]
    assert mega_flat == [f"orig_s{i}" for i in range(60)]

    # Pseudo-chapters carry no source_file (order_fallback); the preserved
    # sibling keeps its own.
    assert "source_file" not in result[1]
    assert result[0].get("source_file") == "/staging/everything.html"


def test_resegment_passthrough_multichapter():
    chapters = [
        {"id": "ch1", "headingText": "A", "sections": [_section(0, "x", "y")]},
        {"id": "ch2", "headingText": "B", "sections": [_section(1, "x", "y")]},
        {"id": "ch3", "headingText": "C", "sections": [_section(2, "x", "y")]},
    ]
    result = resegment_collapsed_structure(
        chapters, embed_client=_StubEmbedClient()
    )
    assert result is chapters


def test_resegment_passthrough_below_threshold():
    # 1 chapter / 20 sections — below the >40 collapse threshold.
    chapters = [
        {
            "id": "ch1",
            "headingText": "Small",
            "sections": [_section(i, f"h{i}", f"t{i}") for i in range(20)],
        }
    ]
    result = resegment_collapsed_structure(
        chapters, embed_client=_StubEmbedClient()
    )
    assert result is chapters


def test_resegment_graceful_when_sklearn_absent(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sklearn.cluster" or name.startswith("sklearn"):
            raise ImportError("sklearn intentionally absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    chapters = _collapsed_three_topics(60)
    result = resegment_collapsed_structure(
        chapters, embed_client=_StubEmbedClient()
    )
    # No crash; pass-through unchanged.
    assert result is chapters


def test_resegment_titles_skip_noncontent():
    # First section of the first segment carries a non-content heading
    # ("78 41. 900 42." answer-key noise) — title must skip to the next
    # content heading or fall back to "Unit N".
    chapters = _collapsed_three_topics(60)
    # Poison the very first section's heading with answer-key numeric noise.
    chapters[0]["sections"][0]["headingText"] = "78 41. 900 42. 800 43."
    client = _StubEmbedClient()

    result = resegment_collapsed_structure(chapters, embed_client=client)

    first = result[0]
    # The non-content noise heading must NOT become the title.
    assert first["headingText"] != "78 41. 900 42. 800 43."
    # It either picked a real content heading from the slice, or fell back to
    # "Unit 1" (only if every heading in the slice were non-content).
    assert first["headingText"].startswith("Counting topic") or (
        first["headingText"] == "Unit 1"
    )


def test_resegment_all_noncontent_segment_falls_back_to_unit():
    # Every section's heading is genuine answer-key noise (numeric-fraction
    # ratio over the threshold → _is_noncontent_heading True), so each
    # pseudo-chapter title must fall back to "Unit N".
    sections = []
    for i in range(60):
        topic = f"T{i % 3}"
        sections.append(
            {
                "id": f"orig_s{i}",
                "headingLevel": 3,
                # Pure answer-key numeric run — classified non-content.
                "headingText": "78 41. 900 42. 800 43.",
                "headingId": None,
                # Keep the embed text topic-distinct so clusters still form.
                "section_text": f"{topic} body for section {i} " * 3,
                "contentBlocks": [],
                "subsections": [],
            }
        )
    chapters = [
        {
            "id": "ch1",
            "headingText": "Mega",
            "sections": sections,
        }
    ]
    result = resegment_collapsed_structure(
        chapters, embed_client=_StubEmbedClient()
    )
    assert all(ch["headingText"].startswith("Unit ") for ch in result)
