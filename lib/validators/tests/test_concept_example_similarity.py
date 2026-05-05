"""Phase 4 Wave N2 Subtask 20 — tests for ConceptExampleSimilarityValidator.

Verifies the validator's input handling, embedding-tier dispatch, and
the GateResult.action mapping (regenerate / pass) per Wave N2's
PoC contract. Fixtures construct outline-tier ``Block`` instances
matching ``Courseforge.scripts.blocks.Block`` for the example
block_type so the validator's per-block-type filter fires.

Tests use a stub embedder returning deterministic vectors so the
suite runs WITHOUT the sentence-transformers extras installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Repo root + scripts dir on path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.concept_example_similarity import (  # noqa: E402
    DEFAULT_THRESHOLD,
    ConceptExampleSimilarityValidator,
    _slug_to_natural_text,
)


# --------------------------------------------------------------------- #
# Stub embedder (mirror of Subtask 19 fixture).
# --------------------------------------------------------------------- #


class _StubEmbedder:
    """Deterministic embedding stub keyed on text-prefix lookups."""

    def __init__(self, vector_map: Dict[str, List[float]]) -> None:
        self.vector_map = vector_map
        self.calls: List[str] = []

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        self.calls.append(text)
        match: Tuple[int, str] = (-1, "")
        for key in self.vector_map:
            if text.startswith(key) and len(key) > match[0]:
                match = (len(key), key)
        if match[0] >= 0:
            return self.vector_map[match[1]]
        return [0.0, 0.0, 1.0]


def _make_example_block(
    *,
    block_id: str = "page_01#example_demo_0",
    body: str = "EXAMPLE: A user logs in to GitHub via Google OAuth.",
    concept_refs: Tuple[str, ...] = ("ed4all:FederatedIdentity",),
    curies: Tuple[str, ...] = (),
) -> Block:
    content: Dict = {"body": body}
    if concept_refs:
        content["concept_refs"] = list(concept_refs)
    if curies:
        content["curies"] = list(curies)
    return Block(
        block_id=block_id,
        block_type="example",
        page_id="page_01",
        sequence=0,
        content=content,
    )


def _make_concept_block() -> Block:
    return Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={"key_claims": ["Federation requires trust."]},
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_passes_when_example_aligns_with_concept() -> None:
    """High cosine similarity between example body and concept def
    yields ``passed=True`` and ``action=None``."""
    embedder = _StubEmbedder(
        vector_map={
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:FederatedIdentity ": [1.0, 0.0, 0.0],
        }
    )
    validator = ConceptExampleSimilarityValidator(embedder=embedder)

    block = _make_example_block(concept_refs=("ed4all:FederatedIdentity",))
    result = validator.validate(
        {
            "blocks": [block],
            "concept_definitions": {
                "ed4all:FederatedIdentity": (
                    "An identity model where authentication is delegated."
                )
            },
        }
    )

    assert result.passed
    assert result.action is None
    assert result.score == 1.0


def test_low_similarity_emits_action_regenerate() -> None:
    """Low cosine similarity emits a critical issue with action='regenerate'."""
    embedder = _StubEmbedder(
        vector_map={
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:DiskEncryption ": [0.0, 1.0, 0.0],
        }
    )
    validator = ConceptExampleSimilarityValidator(embedder=embedder)

    block = _make_example_block(
        body="EXAMPLE: A user logs in via OAuth provider.",
        concept_refs=("ed4all:DiskEncryption",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "concept_definitions": {
                "ed4all:DiskEncryption": (
                    "Data-at-rest cryptographic protection."
                )
            },
        }
    )

    assert not result.passed
    assert result.action == "regenerate"
    assert any(
        i.code == "EXAMPLE_CONCEPT_LOW_SIMILARITY" and i.severity == "critical"
        for i in result.issues
    )


def test_skips_non_example_blocks() -> None:
    """Blocks with block_type != 'example' are skipped entirely."""
    embedder = _StubEmbedder(vector_map={})
    validator = ConceptExampleSimilarityValidator(embedder=embedder)

    result = validator.validate({"blocks": [_make_concept_block()]})

    assert result.passed
    assert result.action is None
    assert result.issues == []
    assert embedder.calls == []


def test_curies_used_when_concept_refs_absent() -> None:
    """Falls back to ``content['curies']`` when concept_refs absent."""
    embedder = _StubEmbedder(
        vector_map={
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:Foo ": [1.0, 0.0, 0.0],
        }
    )
    validator = ConceptExampleSimilarityValidator(embedder=embedder)

    block = _make_example_block(
        concept_refs=(),
        curies=("ed4all:Foo",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "concept_definitions": {"ed4all:Foo": "Foo definition"},
        }
    )

    assert result.passed
    # Verify the embedder was called for the curie.
    assert any("ed4all:Foo" in c for c in embedder.calls)


def test_missing_blocks_input_returns_regenerate_action() -> None:
    """No 'blocks' key -> passed=False, action='regenerate'."""
    validator = ConceptExampleSimilarityValidator(
        embedder=_StubEmbedder(vector_map={})
    )
    result = validator.validate({})

    assert not result.passed
    assert result.action == "regenerate"
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"


def test_unresolved_concept_falls_back_to_slug_surface() -> None:
    """Concept slug without a definition uses the slug itself as surface."""
    # When the validator falls back to slug-only, its embedded surface
    # is e.g. "ed4all:FederatedIdentity ed4all FederatedIdentity Federated Identity"
    # — we'll match the prefix used by _slug_to_natural_text.
    embedder = _StubEmbedder(
        vector_map={
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:FederatedIdentity": [1.0, 0.0, 0.0],
        }
    )
    validator = ConceptExampleSimilarityValidator(embedder=embedder)

    block = _make_example_block(concept_refs=("ed4all:FederatedIdentity",))
    # No concept_definitions supplied — validator falls back to slug surface.
    result = validator.validate({"blocks": [block]})

    # Passes because the slug-derived surface still matches the
    # example body's cosine signal in the stubbed map.
    assert result.passed
    assert any(
        i.code == "CONCEPT_DEFINITION_UNRESOLVED" and i.severity == "warning"
        for i in result.issues
    )


def test_deps_missing_emits_warning_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding extras missing -> warning issue, passed=True, no action."""
    from lib.validators import concept_example_similarity as mod

    monkeypatch.setattr(mod, "try_load_embedder", lambda: None)

    validator = ConceptExampleSimilarityValidator()
    result = validator.validate(
        {
            "blocks": [_make_example_block()],
            "concept_definitions": {
                "ed4all:FederatedIdentity": "Definition.",
            },
        }
    )

    assert result.passed
    assert result.action is None
    assert len(result.issues) == 1
    assert result.issues[0].code == "EMBEDDING_DEPS_MISSING"
    assert result.issues[0].severity == "warning"


def test_slug_to_natural_text_helper() -> None:
    """``_slug_to_natural_text`` splits CamelCase / snake / kebab boundaries
    on the local-part of the CURIE; the full slug is preserved verbatim
    at the start so the namespace context survives."""
    assert _slug_to_natural_text("ed4all:FederatedIdentity") == (
        "ed4all:FederatedIdentity Federated Identity"
    )
    assert _slug_to_natural_text("ed4all:foo_bar-baz") == (
        "ed4all:foo_bar-baz foo bar baz"
    )
    assert _slug_to_natural_text("BareSlug") == "BareSlug Bare Slug"
    assert _slug_to_natural_text("") == ""


def test_default_threshold_is_lower_than_assessment() -> None:
    """Sanity check: example/concept threshold (0.50) < assessment (0.55)."""
    from lib.validators.objective_assessment_similarity import (
        DEFAULT_THRESHOLD as ASSESS_THRESHOLD,
    )

    assert DEFAULT_THRESHOLD < ASSESS_THRESHOLD
    assert DEFAULT_THRESHOLD == 0.50


# --------------------------------------------------------------------- #
# H3 Wave W2 — DecisionCapture wiring smoke test.
# --------------------------------------------------------------------- #


class _StubCapture:
    """Records every log_decision invocation."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, str]] = []

    def log_decision(self, decision_type, decision, rationale, **kwargs):
        self.calls.append((decision_type, decision, rationale))


def test_decision_capture_emits_one_event_per_validate_call() -> None:
    """A single audited example block yields exactly one
    ``concept_example_similarity_check`` decision capture event, with
    rationale interpolating cosine + threshold + above/below flag
    dynamic signals."""
    embedder = _StubEmbedder(
        vector_map={
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:FederatedIdentity ": [1.0, 0.0, 0.0],
        }
    )
    capture = _StubCapture()
    validator = ConceptExampleSimilarityValidator(embedder=embedder)
    validator.validate(
        {
            "blocks": [_make_example_block(
                concept_refs=("ed4all:FederatedIdentity",)
            )],
            "concept_definitions": {
                "ed4all:FederatedIdentity": "An identity model.",
            },
            "decision_capture": capture,
        }
    )

    assert len(capture.calls) == 1
    decision_type, decision, rationale = capture.calls[0]
    assert decision_type == "concept_example_similarity_check"
    assert decision == "passed"
    assert len(rationale) >= 20
    assert "min_pair_cosine=" in rationale
    assert "threshold=" in rationale
    assert "above_threshold=True" in rationale
    assert "ed4all:FederatedIdentity" in rationale


def test_decision_capture_emits_for_low_similarity_failure() -> None:
    """A failing example block yields exactly one capture with
    decision='failed:...' and below-threshold signal."""
    embedder = _StubEmbedder(
        vector_map={
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:DiskEncryption ": [0.0, 1.0, 0.0],
        }
    )
    capture = _StubCapture()
    validator = ConceptExampleSimilarityValidator(embedder=embedder)
    validator.validate(
        {
            "blocks": [_make_example_block(
                body="EXAMPLE: A user logs in via OAuth.",
                concept_refs=("ed4all:DiskEncryption",),
            )],
            "concept_definitions": {
                "ed4all:DiskEncryption": "Cryptographic protection.",
            },
            "decision_capture": capture,
        }
    )
    assert len(capture.calls) == 1
    _, decision, rationale = capture.calls[0]
    assert decision.startswith("failed:")
    assert "EXAMPLE_CONCEPT_LOW_SIMILARITY" in decision
    assert "above_threshold=False" in rationale
