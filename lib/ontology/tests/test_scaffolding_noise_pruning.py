"""Change B: tests for scaffolding-noise concept pruning.

Two layers:

1. Flag-agnostic unit on the pure helper
   ``lib.ontology.concept_classifier.is_scaffolding_noise`` — the 12
   named noise terms classify as noise; a curated keep-set of real
   domain slugs does not.

2. Build-level parity on ``build_cooccurrence_graph``:
   * flag OFF (default) → noise nodes are KEPT (byte-identical legacy
     behaviour);
   * flag ON  → noise nodes are DROPPED while real concepts survive.

The prune flag ``TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS`` is read at the
call sites; ``is_scaffolding_noise`` / ``classify_concept`` stay pure.
"""

from __future__ import annotations

import pytest

from lib.ontology.concept_classifier import is_scaffolding_noise
from lib.ontology.cooccurrence_graph import build_cooccurrence_graph


# ---------------------------------------------------------------------------
# Layer 1: flag-agnostic unit on the pure helper.
# ---------------------------------------------------------------------------

# The 12 named noise terms from the Change B problem statement.
_NAMED_NOISE = [
    "pros",
    "cons",
    "creating",
    "mutating",
    "consuming",
    "branch",
    "harder",
    "chain",
    "pipe",
    "task",
    "advanced",
    "recall",  # "recall the zero shot" -> stem token "recall"
]

# Real-corpus audit: single-token generic abstractions / logistics
# words that survived as DomainConcept (N1 stoplist additions).
_AUDIT_SINGLE_TOKEN_NOISE = [
    "optional",
    "objective",
    "problem",
    "insight",
    "assumption",
    "recommendation",
    "question",
    "purpose",
    "method",
    "reason",
    "menu",
    "repeat",
    "seem",
    "gathered",
    "primer",
    "exercice",
    "finished",
]

# Real-corpus audit: multi-token course-logistics / instructional
# phrases (N4 phrase rule).
_AUDIT_MULTI_TOKEN_NOISE = [
    "assess-task",
    "once-finished",
    "final-exercice",
    "primer-for-next-notebook",
    "plan-of-action",
    "honest-take",
    "stand-alone",
    "prompt-passing",
    "key-modifications",
    "typical-use-inspection",
    "edge-case-inspection",
]

# Real domain vocabulary that must survive the prune.
_KEEP = [
    "semantic-guardrailing",
    "knowledge-bases",
    "knowledge-base",
    "pydantic",
    "vector-database",
    "rdf-graph",
    "co-occurrence",
    # Real-corpus audit keep-set: real multi/single-word concepts that
    # must NOT be caught by the new N1/N4 noise rules.
    "vector-store",
    "running-state",
    "faiss-vector-store",
    "prompt-engineering",
    "json-enabled-slot-filling",
    "runnableassign",
]


@pytest.mark.parametrize("term", _NAMED_NOISE)
def test_named_noise_terms_are_scaffolding(term):
    assert is_scaffolding_noise(term) is True


@pytest.mark.parametrize("term", _AUDIT_SINGLE_TOKEN_NOISE)
def test_audit_single_token_noise_is_scaffolding(term):
    assert is_scaffolding_noise(term) is True


@pytest.mark.parametrize("term", _AUDIT_MULTI_TOKEN_NOISE)
def test_audit_multi_token_noise_is_scaffolding(term):
    assert is_scaffolding_noise(term) is True


@pytest.mark.parametrize("term", _KEEP)
def test_keep_set_is_not_scaffolding(term):
    assert is_scaffolding_noise(term) is False


def test_gerund_de_gerund_with_e_restore():
    # N2: "creating" -> "creat" -> restore "e" -> "create" (action stem).
    assert is_scaffolding_noise("creating") is True
    # Non-action gerunds are not noise.
    assert is_scaffolding_noise("modeling") is False


def test_comparative_short_single_token():
    # N3: short -er/-est single tokens.
    assert is_scaffolding_noise("faster") is True
    assert is_scaffolding_noise("simpler") is True
    # Long / hyphenated -er words are kept (not single short tokens).
    assert is_scaffolding_noise("parameter") is False
    assert is_scaffolding_noise("vector-database") is False


def test_helper_is_pure_no_flag_dependence(monkeypatch):
    # The pure helper ignores the prune flag entirely.
    monkeypatch.setenv("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS", "true")
    assert is_scaffolding_noise("pros") is True
    assert is_scaffolding_noise("rdf-graph") is False
    monkeypatch.delenv("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS", raising=False)
    assert is_scaffolding_noise("pros") is True


# ---------------------------------------------------------------------------
# Layer 2: build-level parity on build_cooccurrence_graph.
# ---------------------------------------------------------------------------

# Inline chunk fixture: concept_tags mix scaffolding noise with real
# concepts. Each tag must appear in >= 2 chunks to clear the default
# min_freq=2 node gate.
_NOISE_TAGS = {"pros", "cons", "creating"}
_REAL_TAGS = {"rdf-graph", "knowledge-base", "pydantic"}


def _fixture_chunks():
    all_tags = sorted(_NOISE_TAGS | _REAL_TAGS)
    return [
        {"id": "c1", "concept_tags": list(all_tags)},
        {"id": "c2", "concept_tags": list(all_tags)},
    ]


def _node_ids(graph):
    return {n["id"] for n in graph["nodes"]}


def test_flag_off_keeps_noise_nodes(monkeypatch):
    monkeypatch.delenv("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS", raising=False)
    graph = build_cooccurrence_graph(_fixture_chunks())
    ids = _node_ids(graph)
    # Legacy parity: noise nodes are present.
    for tag in _NOISE_TAGS:
        assert tag in ids, f"expected legacy noise node {tag!r} when flag OFF"
    for tag in _REAL_TAGS:
        assert tag in ids


def test_flag_on_drops_noise_keeps_real(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS", "true")
    graph = build_cooccurrence_graph(_fixture_chunks())
    ids = _node_ids(graph)
    # Noise nodes are pruned.
    for tag in _NOISE_TAGS:
        assert tag not in ids, f"expected {tag!r} pruned when flag ON"
    # Real concepts survive.
    for tag in _REAL_TAGS:
        assert tag in ids
    # And no edge touches a pruned node.
    for edge in graph["edges"]:
        assert edge["source"] not in _NOISE_TAGS
        assert edge["target"] not in _NOISE_TAGS
