"""Phase 4 Wave N2 Subtasks 22-23 — integration tests for the three
embedding validators.

Cross-validator regression suite asserting the three Phase 4 Wave N2
embedding validators (objective_assessment_similarity,
concept_example_similarity, objective_roundtrip_similarity) compose
cleanly on a single Block list. Each validator filters its own
block_type subset, runs independently against the shared embedder /
paraphrase backend, and emits a separate ``GateResult``. The workflow
runner dispatches all three in sequence at both the
``inter_tier_validation`` and ``post_rewrite_validation`` seams
(Subtasks 17 + 18); this test pins that the trio composes without
cross-validator interference.

Stub backends (deterministic embedder + paraphrase fn) so the suite
runs WITHOUT the sentence-transformers extras installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# Repo root + scripts dir on path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.concept_example_similarity import (  # noqa: E402
    ConceptExampleSimilarityValidator,
)
from lib.validators.objective_assessment_similarity import (  # noqa: E402
    ObjectiveAssessmentSimilarityValidator,
)
from lib.validators.objective_roundtrip_similarity import (  # noqa: E402
    ObjectiveRoundtripSimilarityValidator,
)


# --------------------------------------------------------------------- #
# Shared stub backends.
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


def _passing_paraphrase(text: str) -> Optional[str]:
    """Return a paraphrase-prefix string so the embedder maps it to the
    same vector as the ORIGINAL prefix used in test fixtures."""
    if not text:
        return None
    return f"PARAPHRASE: {text}"


# --------------------------------------------------------------------- #
# Mixed-block-list fixture
# --------------------------------------------------------------------- #


def _build_mixed_block_list() -> List[Block]:
    """Return a list of 3 Blocks (one per validated block_type) plus
    one ignored block_type, so each validator filters in exactly one
    block."""
    return [
        Block(
            block_id="page_01#objective_intro_0",
            block_type="objective",
            page_id="page_01",
            sequence=0,
            content={
                "statement": (
                    "ORIGINAL: Define federated identity in single sign-on."
                )
            },
        ),
        Block(
            block_id="page_01#concept_intro_1",
            block_type="concept",  # ignored by all 3 validators
            page_id="page_01",
            sequence=1,
            content={"key_claims": ["Federation requires trust."]},
        ),
        Block(
            block_id="page_01#example_demo_2",
            block_type="example",
            page_id="page_01",
            sequence=2,
            content={
                "body": "EXAMPLE: A user logs in to GitHub via Google OAuth.",
                "concept_refs": ["ed4all:FederatedIdentity"],
            },
        ),
        Block(
            block_id="page_01#assessment_item_quiz_3",
            block_type="assessment_item",
            page_id="page_01",
            sequence=3,
            content={
                "stem": "ASSESSMENT: What enables federated single sign-on?",
                "answer_key": (
                    "ASSESSMENT: A trust relationship between the IdP and SP."
                ),
            },
            objective_ids=("TO-01",),
        ),
    ]


@pytest.mark.integration
def test_all_three_validators_pass_on_well_formed_block_list() -> None:
    """All three Wave N2 validators emit ``passed=True`` and ``action=None``
    on a block list where each per-block-type surface aligns with its
    related surface."""
    # All key prefixes map to the same unit vector → cosine = 1.0.
    aligned_vector = [1.0, 0.0, 0.0]
    embedder = _StubEmbedder(
        vector_map={
            "ORIGINAL:": aligned_vector,
            "PARAPHRASE:": aligned_vector,
            "OBJECTIVE:": aligned_vector,
            "ASSESSMENT:": aligned_vector,
            "EXAMPLE:": aligned_vector,
            "ed4all:FederatedIdentity ": aligned_vector,
        }
    )

    blocks = _build_mixed_block_list()
    inputs = {
        "blocks": blocks,
        "objective_statements": {
            "TO-01": (
                "OBJECTIVE: Federated identity enables SSO across providers."
            )
        },
        "concept_definitions": {
            "ed4all:FederatedIdentity": "Identity model with delegated authn."
        },
    }

    v_assess = ObjectiveAssessmentSimilarityValidator(embedder=embedder)
    v_concept = ConceptExampleSimilarityValidator(embedder=embedder)
    v_round = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_passing_paraphrase,
    )

    r_assess = v_assess.validate(inputs)
    r_concept = v_concept.validate(inputs)
    r_round = v_round.validate(inputs)

    assert r_assess.passed and r_assess.action is None, (
        f"assessment validator failed: {r_assess.issues}"
    )
    assert r_concept.passed and r_concept.action is None, (
        f"concept validator failed: {r_concept.issues}"
    )
    assert r_round.passed and r_round.action is None, (
        f"roundtrip validator failed: {r_round.issues}"
    )

    # Each validator's gate_id is distinct so downstream decision-event
    # filtering can stratify the three signals.
    assert r_assess.gate_id == "objective_assessment_similarity"
    assert r_concept.gate_id == "concept_example_similarity"
    assert r_round.gate_id == "objective_roundtrip_similarity"


@pytest.mark.integration
def test_all_three_validators_fire_on_misaligned_block_list() -> None:
    """When all three per-block-type surfaces are misaligned, all three
    validators emit ``passed=False`` and ``action="regenerate"`` —
    cross-validator regression test asserting the trio composes cleanly
    on a uniformly-broken block list."""
    embedder = _StubEmbedder(
        vector_map={
            # Originals on x-axis, related surfaces on y-axis →
            # orthogonal → cosine = 0.0 across the board.
            "ORIGINAL:": [1.0, 0.0, 0.0],
            "PARAPHRASE:": [0.0, 1.0, 0.0],
            "OBJECTIVE:": [0.0, 1.0, 0.0],
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:FederatedIdentity ": [0.0, 1.0, 0.0],
        }
    )

    blocks = _build_mixed_block_list()
    inputs = {
        "blocks": blocks,
        "objective_statements": {
            "TO-01": "OBJECTIVE: Disk encryption protects data at rest."
        },
        "concept_definitions": {
            "ed4all:FederatedIdentity": "Disk-at-rest cryptography."
        },
    }

    v_assess = ObjectiveAssessmentSimilarityValidator(embedder=embedder)
    v_concept = ConceptExampleSimilarityValidator(embedder=embedder)
    v_round = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_passing_paraphrase,
    )

    r_assess = v_assess.validate(inputs)
    r_concept = v_concept.validate(inputs)
    r_round = v_round.validate(inputs)

    # All three fired the regenerate action.
    for label, result in (
        ("assessment", r_assess),
        ("concept", r_concept),
        ("roundtrip", r_round),
    ):
        assert not result.passed, f"{label} should fail; got: {result.issues}"
        assert result.action == "regenerate", (
            f"{label} should emit action='regenerate'; got: {result.action!r}"
        )

    # Each validator filtered to its own block_type subset.
    assert any(
        i.code == "ASSESSMENT_OBJECTIVE_LOW_SIMILARITY"
        for i in r_assess.issues
    )
    assert any(
        i.code == "EXAMPLE_CONCEPT_LOW_SIMILARITY"
        for i in r_concept.issues
    )
    assert any(
        i.code == "OBJECTIVE_ROUNDTRIP_LOW_SIMILARITY"
        for i in r_round.issues
    )


@pytest.mark.integration
def test_validators_register_in_workflow_yaml() -> None:
    """Wave N2 Subtask 17/18 wired six gates (3 validators × 2 seams)
    into config/workflows.yaml. Pin the validator class paths against
    the shipped YAML so a future refactor of validator import paths
    breaks this test loudly instead of silently desynchronising the
    workflow definition.
    """
    import yaml

    yaml_path = _REPO_ROOT / "config" / "workflows.yaml"
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    expected_validators = {
        "lib.validators.objective_assessment_similarity.ObjectiveAssessmentSimilarityValidator",
        "lib.validators.concept_example_similarity.ConceptExampleSimilarityValidator",
        "lib.validators.objective_roundtrip_similarity.ObjectiveRoundtripSimilarityValidator",
    }

    seen_at_outline_seam: set[str] = set()
    seen_at_rewrite_seam: set[str] = set()

    workflow = config["workflows"]["textbook_to_course"]
    for phase in workflow["phases"]:
        gates = phase.get("validation_gates") or []
        if phase["name"] == "inter_tier_validation":
            for gate in gates:
                seen_at_outline_seam.add(gate["validator"])
        elif phase["name"] == "post_rewrite_validation":
            for gate in gates:
                seen_at_rewrite_seam.add(gate["validator"])

    missing_outline = expected_validators - seen_at_outline_seam
    missing_rewrite = expected_validators - seen_at_rewrite_seam
    assert not missing_outline, (
        f"Missing embedding validators at inter_tier_validation seam: "
        f"{sorted(missing_outline)}"
    )
    assert not missing_rewrite, (
        f"Missing embedding validators at post_rewrite_validation seam: "
        f"{sorted(missing_rewrite)}"
    )


@pytest.mark.integration
def test_partial_misalignment_only_fires_relevant_validator() -> None:
    """When ONLY the assessment surface misaligns (concept + roundtrip
    pass), only the assessment validator emits action='regenerate'.

    Pins per-validator independence: a misalignment in one block_type
    doesn't bleed into the other validators' GateResults.
    """
    embedder = _StubEmbedder(
        vector_map={
            # Misalign assessment vs objective (orthogonal vectors).
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            "OBJECTIVE:": [0.0, 1.0, 0.0],
            # Concept + example align (same vector).
            "EXAMPLE:": [1.0, 0.0, 0.0],
            "ed4all:FederatedIdentity ": [1.0, 0.0, 0.0],
            # Objective + paraphrase align (same vector).
            "ORIGINAL:": [1.0, 0.0, 0.0],
            "PARAPHRASE:": [1.0, 0.0, 0.0],
        }
    )

    blocks = _build_mixed_block_list()
    inputs = {
        "blocks": blocks,
        "objective_statements": {
            "TO-01": "OBJECTIVE: An unrelated objective on disk encryption."
        },
        "concept_definitions": {
            "ed4all:FederatedIdentity": "Identity model with delegated authn."
        },
    }

    v_assess = ObjectiveAssessmentSimilarityValidator(embedder=embedder)
    v_concept = ConceptExampleSimilarityValidator(embedder=embedder)
    v_round = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_passing_paraphrase,
    )

    r_assess = v_assess.validate(inputs)
    r_concept = v_concept.validate(inputs)
    r_round = v_round.validate(inputs)

    # Assessment fires the regenerate action.
    assert not r_assess.passed
    assert r_assess.action == "regenerate"

    # Concept + roundtrip stay clean (no cross-validator interference).
    assert r_concept.passed
    assert r_concept.action is None
    assert r_round.passed
    assert r_round.action is None
