"""Phase 4 Wave N2 Subtask 21 — tests for ObjectiveRoundtripSimilarityValidator.

Verifies the validator's input handling, paraphrase-dispatch surface,
and the GateResult.action mapping (regenerate / pass) per Wave N2's
PoC contract.

Per the plan's §Blockers escalation, the tests use a stub paraphrase
function as the dispatch surface — the validator's responsibility ends
at the cosine-similarity check, and wiring the rewrite-tier
``CourseforgeRouter`` into a unit test would require more router
setup than reasonable to mock. The router-adapter helper
``_paraphrase_via_router`` is exercised in a dedicated test that
verifies the adapter shape against a stub router (no LLM calls).

Tests use a stub embedder + stub paraphrase_fn so the suite runs
WITHOUT the sentence-transformers extras installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

# Repo root + scripts dir on path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.objective_roundtrip_similarity import (  # noqa: E402
    DEFAULT_THRESHOLD,
    ObjectiveRoundtripSimilarityValidator,
    _paraphrase_via_router,
)


# --------------------------------------------------------------------- #
# Stub embedder + paraphrase fn fixtures.
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


def _make_stub_paraphrase_fn(
    output_prefix: str = "PARAPHRASE:",
) -> Callable[[str], Optional[str]]:
    """Return a paraphrase_fn that prepends ``output_prefix`` to input."""

    def _paraphrase(text: str) -> Optional[str]:
        if not text:
            return None
        return f"{output_prefix} {text}"

    return _paraphrase


def _make_objective_block(
    *,
    block_id: str = "page_01#objective_intro_0",
    statement: str = (
        "ORIGINAL: Define the role of federated identity in single sign-on."
    ),
) -> Block:
    return Block(
        block_id=block_id,
        block_type="objective",
        page_id="page_01",
        sequence=0,
        content={"statement": statement},
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


def test_passes_when_paraphrase_preserves_meaning() -> None:
    """High cosine between original and paraphrase yields passed=True."""
    embedder = _StubEmbedder(
        vector_map={
            "ORIGINAL:": [1.0, 0.0, 0.0],
            "PARAPHRASE:": [1.0, 0.0, 0.0],
        }
    )
    validator = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_make_stub_paraphrase_fn("PARAPHRASE:"),
    )

    result = validator.validate({"blocks": [_make_objective_block()]})

    assert result.passed
    assert result.action is None
    assert result.score == 1.0


def test_low_similarity_emits_action_regenerate() -> None:
    """Low cosine between original and paraphrase yields action='regenerate'."""
    embedder = _StubEmbedder(
        vector_map={
            "ORIGINAL:": [1.0, 0.0, 0.0],
            "PARAPHRASE:": [0.0, 1.0, 0.0],
        }
    )
    validator = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_make_stub_paraphrase_fn("PARAPHRASE:"),
    )

    result = validator.validate({"blocks": [_make_objective_block()]})

    assert not result.passed
    assert result.action == "regenerate"
    assert any(
        i.code == "OBJECTIVE_ROUNDTRIP_LOW_SIMILARITY" and i.severity == "critical"
        for i in result.issues
    )


def test_skips_non_objective_blocks() -> None:
    """Blocks with block_type != 'objective' are skipped."""
    embedder = _StubEmbedder(vector_map={})
    validator = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_make_stub_paraphrase_fn(),
    )

    result = validator.validate({"blocks": [_make_concept_block()]})

    assert result.passed
    assert result.action is None
    assert result.issues == []
    assert embedder.calls == []


def test_missing_blocks_input_returns_regenerate_action() -> None:
    """No 'blocks' key -> passed=False, action='regenerate'."""
    validator = ObjectiveRoundtripSimilarityValidator(
        embedder=_StubEmbedder(vector_map={}),
        paraphrase_fn=_make_stub_paraphrase_fn(),
    )
    result = validator.validate({})

    assert not result.passed
    assert result.action == "regenerate"
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"


def test_paraphrase_not_configured_emits_warning() -> None:
    """No paraphrase_fn AND no router -> warning, passed=True, no action."""
    embedder = _StubEmbedder(
        vector_map={"ORIGINAL:": [1.0, 0.0, 0.0]}
    )
    # Construct without paraphrase_fn or router.
    validator = ObjectiveRoundtripSimilarityValidator(embedder=embedder)
    result = validator.validate({"blocks": [_make_objective_block()]})

    assert result.passed
    assert result.action is None
    assert len(result.issues) == 1
    assert result.issues[0].code == "PARAPHRASE_NOT_CONFIGURED"
    assert result.issues[0].severity == "warning"


def test_paraphrase_dispatch_failure_emits_warning() -> None:
    """paraphrase_fn raising surfaces a warning issue, block is skipped."""

    def _broken(_text: str) -> Optional[str]:
        raise RuntimeError("simulated paraphrase backend failure")

    embedder = _StubEmbedder(vector_map={"ORIGINAL:": [1.0, 0.0, 0.0]})
    validator = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_broken,
    )

    result = validator.validate({"blocks": [_make_objective_block()]})

    # Single block, paraphrase failed → no critical issues, just a
    # warning. Validator passes overall (no low-similarity findings).
    assert result.passed
    assert result.action is None
    assert any(
        i.code == "PARAPHRASE_DISPATCH_FAILED" and i.severity == "warning"
        for i in result.issues
    )


def test_threshold_override_via_inputs() -> None:
    """Per-call threshold override via inputs['threshold'] takes precedence."""
    # Cosine between [1,0,0] and [0.7,0.7,0] = ~0.707.
    embedder = _StubEmbedder(
        vector_map={
            "ORIGINAL:": [1.0, 0.0, 0.0],
            "PARAPHRASE:": [0.7, 0.7, 0.0],
        }
    )
    validator = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_make_stub_paraphrase_fn("PARAPHRASE:"),
    )

    inputs = {"blocks": [_make_objective_block()]}

    # Default threshold is 0.70 — borderline pass at 0.707.
    r_default = validator.validate(inputs)
    assert r_default.passed, (
        f"Expected pass at threshold {DEFAULT_THRESHOLD}; got {r_default.issues}"
    )

    # Override to 0.9 — should fail.
    inputs_strict: Dict[str, Any] = dict(inputs)
    inputs_strict["threshold"] = 0.9
    r_strict = validator.validate(inputs_strict)
    assert not r_strict.passed
    assert r_strict.action == "regenerate"


def test_deps_missing_emits_warning_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding extras missing -> warning issue, passed=True."""
    from lib.validators import objective_roundtrip_similarity as mod

    monkeypatch.setattr(mod, "try_load_embedder", lambda: None)

    validator = ObjectiveRoundtripSimilarityValidator(
        paraphrase_fn=_make_stub_paraphrase_fn(),
    )
    result = validator.validate({"blocks": [_make_objective_block()]})

    assert result.passed
    assert result.action is None
    assert len(result.issues) == 1
    assert result.issues[0].code == "EMBEDDING_DEPS_MISSING"


def test_paraphrase_via_router_adapter() -> None:
    """``_paraphrase_via_router`` adapts a router into a paraphrase_fn.

    Stub router exposes a ``route(block, tier, overrides)`` surface
    matching the production CourseforgeRouter contract; the adapter
    constructs an ephemeral Block with the input text in
    ``content['statement']`` and pulls the paraphrased text back out
    of the returned Block.
    """

    class _StubRouter:
        def __init__(self) -> None:
            self.calls: List[Tuple[str, str, Dict[str, Any]]] = []

        def route(self, block: Block, *, tier: str, overrides: Dict[str, Any]) -> Block:
            assert tier == "rewrite"
            assert overrides.get("prompt_template") == "Custom paraphrase template"
            text = block.content.get("statement", "")
            self.calls.append((block.block_id, text, overrides))
            # Return a Block with a paraphrased string surface.
            return Block(
                block_id=block.block_id,
                block_type="objective",
                page_id=block.page_id,
                sequence=block.sequence,
                content=f"Paraphrased: {text}",
            )

    router = _StubRouter()
    paraphrase_fn = _paraphrase_via_router(
        router, prompt_template="Custom paraphrase template"
    )

    out = paraphrase_fn("Define federation in SSO.")
    assert out == "Paraphrased: Define federation in SSO."
    assert len(router.calls) == 1


def test_inputs_paraphrase_fn_overrides_constructor() -> None:
    """``inputs['paraphrase_fn']`` overrides the constructor-time wiring."""
    embedder = _StubEmbedder(
        vector_map={
            "ORIGINAL:": [1.0, 0.0, 0.0],
            "ALT:": [1.0, 0.0, 0.0],
        }
    )

    constructor_calls: List[str] = []

    def _ctor_fn(text: str) -> Optional[str]:
        constructor_calls.append(text)
        # If this was used, cosine would be 0 against ALT: (orthogonal default).
        return f"CTOR_PARAPHRASE: {text}"

    inputs_calls: List[str] = []

    def _inputs_fn(text: str) -> Optional[str]:
        inputs_calls.append(text)
        return f"ALT: {text}"

    validator = ObjectiveRoundtripSimilarityValidator(
        embedder=embedder,
        paraphrase_fn=_ctor_fn,
    )

    result = validator.validate(
        {
            "blocks": [_make_objective_block()],
            "paraphrase_fn": _inputs_fn,
        }
    )

    # The inputs-side paraphrase_fn was used, NOT the constructor one.
    assert constructor_calls == []
    assert len(inputs_calls) == 1
    assert result.passed
