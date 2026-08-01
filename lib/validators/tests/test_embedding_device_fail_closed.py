"""Statistical-tier gates must not vacuously PASS when the device is gone.

``ED4ALL_EMBEDDING_DEVICE`` defaults to ``cuda``. Before this suite, every
embedding-tier validator wrapped ``embedder.encode`` in a bare
``except Exception`` that degraded to a warning-severity
``EMBEDDING_ENCODE_ERROR`` with ``passed=True`` — so on a host where CUDA is
absent, the lazily-raised
:class:`~lib.embedding.sentence_embedder.EmbeddingModelUnavailable` was
swallowed and all seven gates reported a clean pass without scoring a single
pair. This module pins the two contracts apart, per validator:

(a) **Extras present, requested device absent** ->
    :class:`EmbeddingModelUnavailable` is FATAL. It propagates out of
    ``validate()`` and is never converted into a passing ``GateResult``.
    Covered at both guard layers: the eager ``preload()`` at the top of the
    validate path, and the per-encode re-raise that backstops it.
(b) **Missing ``[embedding]`` extras** -> unchanged. A warning-severity
    ``EMBEDDING_DEPS_MISSING`` GateIssue with ``passed=True``.
(c) **``TRAINFORGE_REQUIRE_EMBEDDINGS=true``** -> still flips (b) closed.

Plus the negative control: a genuinely *transient* encode error still
degrades exactly as it does today. It is specifically the TYPED
unavailability that must not be swallowed.

No GPU is touched — every embedder here is a stub. Run CPU-pinned:
``ED4ALL_EMBEDDING_DEVICE=cpu ED4ALL_NLI_DEVICE=cpu``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple

import pytest

# Repo root + scripts dir on path (mirrors the sibling validator suites).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.embedding.sentence_embedder import (  # noqa: E402
    EmbeddingDepsMissing,
    EmbeddingModelUnavailable,
)
from lib.validators import co_terminal_alignment as _cot_mod  # noqa: E402
from lib.validators import concept_example_similarity as _ces_mod  # noqa: E402
from lib.validators import (  # noqa: E402
    distractor_misconception_alignment as _dma_mod,
)
from lib.validators import (  # noqa: E402
    objective_assessment_similarity as _oas_mod,
)
from lib.validators import (  # noqa: E402
    objective_roundtrip_similarity as _ors_mod,
)
from lib.validators import rewrite_source_grounding as _rsg_mod  # noqa: E402
from lib.validators import source_coverage as _sc_mod  # noqa: E402

# Reuse each validator's own fixture builders so these cases exercise the
# same input shapes the happy-path suites do.
from lib.validators.tests.test_co_terminal_alignment import (  # noqa: E402
    _co as _cot_co,
)
from lib.validators.tests.test_co_terminal_alignment import (  # noqa: E402
    _objectives as _cot_objectives,
)
from lib.validators.tests.test_co_terminal_alignment import (  # noqa: E402
    _to as _cot_to,
)
from lib.validators.tests.test_concept_example_similarity import (  # noqa: E402
    _make_example_block,
)
from lib.validators.tests.test_distractor_misconception_alignment import (  # noqa: E402
    _make_assessment_block as _make_dma_block,
)
from lib.validators.tests.test_objective_assessment_similarity import (  # noqa: E402
    _make_assessment_block as _make_oas_block,
)
from lib.validators.tests.test_objective_roundtrip_similarity import (  # noqa: E402
    _make_objective_block,
    _make_stub_paraphrase_fn,
)
from lib.validators.tests.test_rewrite_source_grounding import (  # noqa: E402
    _GROUNDED_HTML,
    _GROUNDING_SOURCE,
)
from lib.validators.tests.test_rewrite_source_grounding import (  # noqa: E402
    _make_block as _make_rsg_block,
)
from lib.validators.tests.test_source_coverage import (  # noqa: E402
    _BODY,
    _section,
)
from lib.validators.tests.test_source_coverage import (  # noqa: E402
    _co as _sc_co,
)
from lib.validators.tests.test_source_coverage import (  # noqa: E402
    _objectives as _sc_objectives,
)
from lib.validators.tests.test_source_coverage import (  # noqa: E402
    _structure,
)
from lib.validators.tests.test_source_coverage import (  # noqa: E402
    _to as _sc_to,
)

_DEVICE_ERR = (
    "failed to construct SentenceTransformer 'BAAI/bge-large-en-v1.5' on "
    "device 'cuda': Torch not compiled with CUDA enabled."
)


# --------------------------------------------------------------------- #
# Embedder stubs. None of these load a model or touch a device.
# --------------------------------------------------------------------- #


class _PreloadUnavailableEmbedder:
    """Extras installed; the requested device is absent.

    Models the real :class:`SentenceEmbedder`, whose ``preload()`` forces the
    ``SentenceTransformer`` construction and raises
    :class:`EmbeddingModelUnavailable` when the device is not there.
    """

    def __init__(self) -> None:
        self.encode_calls: List[str] = []

    def preload(self) -> None:
        raise EmbeddingModelUnavailable(_DEVICE_ERR)

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        self.encode_calls.append(text)
        return [1.0, 0.0, 0.0]


class _EncodeUnavailableEmbedder:
    """Duck-typed embedder with NO ``preload()`` — the backstop case.

    Proves the per-encode guards re-raise the typed error even when the eager
    ``preload()`` probe is unavailable on the injected object.
    """

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        raise EmbeddingModelUnavailable(_DEVICE_ERR)


class _TransientEncodeErrorEmbedder:
    """Loads fine, then fails per-encode for a NON-device reason.

    The negative control: this must keep degrading exactly as it does today.
    """

    def preload(self) -> None:
        return None

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        raise RuntimeError("transient encode blip")


# --------------------------------------------------------------------- #
# Per-validator case table.
# --------------------------------------------------------------------- #


class _Case(NamedTuple):
    id: str
    make_validator: Callable[[Any], Any]
    inputs: Callable[[], Dict[str, Any]]
    #: Object carrying the ``try_load_embedder`` attribute this validator
    #: actually calls (the module itself, except for the one validator that
    #: goes through the embedder module for monkeypatch ergonomics).
    loader_holder: Any
    #: True when strict mode is expected to raise out of ``validate()``
    #: rather than be caught and turned into a critical GateResult.
    strict_raises: bool = True
    #: True when a TRANSIENT (non-device) encode error degrades to a passing
    #: gate. False only for the validator whose per-pair fallback scores
    #: Jaccard against the COSINE threshold, so the degrade can fire a
    #: critical. That is pre-existing behavior, unrelated to the device flip.
    transient_passes: bool = True


def _ces_inputs() -> Dict[str, Any]:
    return {
        "blocks": [_make_example_block()],
        "concept_definitions": {
            "ed4all:FederatedIdentity": (
                "An identity model where authentication is delegated."
            )
        },
    }


def _oas_inputs() -> Dict[str, Any]:
    return {
        "blocks": [_make_oas_block(objective_ids=("TO-01",))],
        "objective_statements": {
            "TO-01": "OBJECTIVE: Define the role of federated identity in SSO.",
        },
    }


def _ors_inputs() -> Dict[str, Any]:
    return {"blocks": [_make_objective_block()]}


def _cot_inputs() -> Dict[str, Any]:
    return {
        "synthesized_objectives": _cot_objectives(
            [_cot_to("TO-01", "TO terminal statement")],
            [_cot_co("CO-01", "CO aligned one", terminal_id="TO-01")],
        )
    }


def _sc_inputs() -> Dict[str, Any]:
    return {
        "synthesized_objectives": _sc_objectives(
            [_sc_to("TO-01", "OBJ terminal statement")],
            [_sc_co("CO-01", "OBJ chapter one")],
        ),
        "textbook_structure": _structure(
            [_section("SEC alpha", "SEC alpha" + _BODY)]
        ),
    }


def _rsg_inputs() -> Dict[str, Any]:
    return {
        "blocks": [_make_rsg_block(content=_GROUNDED_HTML)],
        "source_chunks": {"semantik:slug#blk_0": _GROUNDING_SOURCE},
    }


def _dma_inputs() -> Dict[str, Any]:
    return {
        "blocks": [
            _make_dma_block(
                distractors=[
                    {"text": "Correct answer"},
                    {
                        "text": "forces always cause acceleration in every case",
                        "misconception_ref": "TO-01#m1",
                    },
                ],
            )
        ],
        "misconceptions": {
            "TO-01#m1": (
                "forces always cause acceleration even at constant velocity"
            ),
        },
    }


_CASES: List[_Case] = [
    _Case(
        id="concept_example_similarity",
        make_validator=lambda emb: _ces_mod.ConceptExampleSimilarityValidator(
            embedder=emb
        ),
        inputs=_ces_inputs,
        loader_holder=_ces_mod,
    ),
    _Case(
        id="co_terminal_alignment",
        make_validator=lambda emb: _cot_mod.CoTerminalAlignmentValidator(
            embedder=emb
        ),
        inputs=_cot_inputs,
        loader_holder=_cot_mod,
    ),
    _Case(
        id="distractor_misconception_alignment",
        make_validator=(
            lambda emb: _dma_mod.DistractorMisconceptionAlignmentValidator(
                embedder=emb
            )
        ),
        inputs=_dma_inputs,
        # This validator reads ``try_load_embedder`` off the embedder module
        # at validate() time rather than binding it at import.
        loader_holder=_dma_mod._sentence_embedder_mod,
        # Documented deviation — see test_strict_mode_deviation_distractor.
        strict_raises=False,
        # Its per-pair encode-failure fallback scores Jaccard but compares
        # against ``min_cosine``, so a transient blip can fire a critical
        # MISALIGNED. Pre-existing; pinned here so it stays visible.
        transient_passes=False,
    ),
    _Case(
        id="objective_assessment_similarity",
        make_validator=(
            lambda emb: _oas_mod.ObjectiveAssessmentSimilarityValidator(
                embedder=emb
            )
        ),
        inputs=_oas_inputs,
        loader_holder=_oas_mod,
    ),
    _Case(
        id="objective_roundtrip_similarity",
        make_validator=(
            lambda emb: _ors_mod.ObjectiveRoundtripSimilarityValidator(
                embedder=emb,
                paraphrase_fn=_make_stub_paraphrase_fn("PARAPHRASE:"),
            )
        ),
        inputs=_ors_inputs,
        loader_holder=_ors_mod,
    ),
    _Case(
        id="rewrite_source_grounding",
        make_validator=lambda emb: _rsg_mod.RewriteSourceGroundingValidator(
            embedder=emb
        ),
        inputs=_rsg_inputs,
        loader_holder=_rsg_mod,
        # This one CATCHES EmbeddingDepsMissing and returns a critical
        # GateResult instead of propagating.
        strict_raises=False,
    ),
    _Case(
        id="source_coverage",
        make_validator=lambda emb: _sc_mod.SourceCoverageValidator(
            embedder=emb
        ),
        inputs=_sc_inputs,
        loader_holder=_sc_mod,
    ),
]

_ALL_IDS = [c.id for c in _CASES]


def _param(cases: List[_Case]):
    return pytest.mark.parametrize(
        "case", cases, ids=[c.id for c in cases]
    )


@pytest.fixture(autouse=True)
def _no_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test pins TRAINFORGE_REQUIRE_EMBEDDINGS explicitly."""
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)


# --------------------------------------------------------------------- #
# The crux: EmbeddingModelUnavailable is distinct from EmbeddingDepsMissing.
# --------------------------------------------------------------------- #


def test_device_error_is_not_a_deps_missing_subclass() -> None:
    """The two contracts hang off two unrelated types.

    If ``EmbeddingModelUnavailable`` ever became a subclass of
    ``EmbeddingDepsMissing``, every ``except EmbeddingDepsMissing`` degrade
    path in this tier would start swallowing device failures again.
    """
    assert not issubclass(EmbeddingModelUnavailable, EmbeddingDepsMissing)
    assert not issubclass(EmbeddingDepsMissing, EmbeddingModelUnavailable)


# --------------------------------------------------------------------- #
# (a) Device unavailable -> FATAL, never a passing gate.
# --------------------------------------------------------------------- #


@_param(_CASES)
def test_device_unavailable_at_preload_is_fatal(case: _Case) -> None:
    """The eager ``preload()`` surfaces the device failure before any encode.

    This is the regression guard for the CUDA-default flip: with no eager
    probe the error arrived lazily inside ``encode()`` and every call site's
    ``except Exception`` degraded it to a passing gate.
    """
    embedder = _PreloadUnavailableEmbedder()
    validator = case.make_validator(embedder)

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(case.inputs())

    # The failure must land BEFORE any scoring work is attempted.
    assert embedder.encode_calls == []


@_param(_CASES)
def test_device_unavailable_at_encode_is_fatal(case: _Case) -> None:
    """Backstop: the per-encode guards re-raise the typed error too.

    Exercised with an injected embedder that has no ``preload()`` at all, so
    only the narrowed ``except`` clauses can catch this.
    """
    validator = case.make_validator(_EncodeUnavailableEmbedder())

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(case.inputs())


@_param(_CASES)
def test_device_unavailable_from_loader_is_fatal(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device failure raised by ``try_load_embedder`` is fatal as well.

    No ``embedder=`` override here, so the validator goes through its real
    loader seam — the path a production run takes.
    """

    def _raise_device(*args: Any, **kwargs: Any) -> Any:
        raise EmbeddingModelUnavailable(_DEVICE_ERR)

    monkeypatch.setattr(
        case.loader_holder, "try_load_embedder", _raise_device
    )
    validator = case.make_validator(None)

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(case.inputs())


# --------------------------------------------------------------------- #
# Negative control: a transient encode error still degrades as today.
# --------------------------------------------------------------------- #


@_param(_CASES)
def test_transient_encode_error_still_degrades(case: _Case) -> None:
    """Only the TYPED unavailability is fatal.

    A non-device encode failure keeps its existing degrade — narrowing the
    ``except`` clauses must not have widened what propagates. The load-bearing
    assertion is that ``validate()`` RETURNS rather than raising.
    """
    validator = case.make_validator(_TransientEncodeErrorEmbedder())

    result = validator.validate(case.inputs())

    if case.transient_passes:
        assert result.passed is True
        assert all(
            i.severity != "critical" for i in result.issues
        ), [i.code for i in result.issues]
    else:
        # Still a degrade, not a propagate: the warning proves the encode
        # error was caught and handled locally.
        assert "EMBEDDING_ENCODE_ERROR" in [
            i.code for i in result.issues if i.severity == "warning"
        ]


# --------------------------------------------------------------------- #
# (b) Missing extras -> warning + passed=True. UNCHANGED.
# --------------------------------------------------------------------- #


@_param(_CASES)
def test_missing_extras_still_warns_and_passes(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional-extras escape hatch survives the fail-closed change.

    A fix that made the device path fail closed by also breaking this is a
    failed fix — the two contracts must stay distinct.
    """
    monkeypatch.setattr(
        case.loader_holder, "try_load_embedder", lambda *a, **kw: None
    )
    validator = case.make_validator(None)

    result = validator.validate(case.inputs())

    assert result.passed is True
    warnings = [
        i.code for i in result.issues if i.severity == "warning"
    ]
    assert "EMBEDDING_DEPS_MISSING" in warnings
    assert all(i.severity != "critical" for i in result.issues)


# --------------------------------------------------------------------- #
# (c) Strict mode still flips missing-extras closed.
# --------------------------------------------------------------------- #


@_param([c for c in _CASES if c.strict_raises])
def test_strict_mode_missing_extras_fails_closed(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` keeps failing the build closed.

    These validators let ``EmbeddingDepsMissing`` propagate; the gate manager's
    ``on_error: fail_closed`` turns that into a blocked phase.
    """
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    def _raise_deps(*args: Any, **kwargs: Any) -> Any:
        raise EmbeddingDepsMissing("sentence-transformers is not installed")

    monkeypatch.setattr(case.loader_holder, "try_load_embedder", _raise_deps)
    validator = case.make_validator(None)

    with pytest.raises(EmbeddingDepsMissing):
        validator.validate(case.inputs())


def test_strict_mode_rewrite_source_grounding_fails_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one validator that catches the strict raise still fails closed.

    ``rewrite_source_grounding`` converts ``EmbeddingDepsMissing`` into a
    CRITICAL GateIssue with ``passed=False`` rather than propagating — the
    outcome is the same (blocked), the shape differs.
    """
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    def _raise_deps(*args: Any, **kwargs: Any) -> Any:
        raise EmbeddingDepsMissing("sentence-transformers is not installed")

    monkeypatch.setattr(_rsg_mod, "try_load_embedder", _raise_deps)

    result = _rsg_mod.RewriteSourceGroundingValidator().validate(
        _rsg_inputs()
    )

    assert result.passed is False
    critical = [i.code for i in result.issues if i.severity == "critical"]
    assert "EMBEDDING_DEPS_MISSING" in critical


def test_strict_mode_deviation_distractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PINS a documented deviation, so a future change to it is deliberate.

    ``distractor_misconception_alignment`` is the one gate in this tier that
    is always-useful without embeddings (Jaccard token-overlap), so it
    swallows ``EmbeddingDepsMissing`` and keeps scoring even under
    ``TRAINFORGE_REQUIRE_EMBEDDINGS=true``. That predates the device flip and
    is left unchanged here; the DEVICE contract is enforced on it exactly like
    its siblings (see the parametrized fatal cases above).
    """
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    def _raise_deps(*args: Any, **kwargs: Any) -> Any:
        raise EmbeddingDepsMissing("sentence-transformers is not installed")

    monkeypatch.setattr(
        _dma_mod._sentence_embedder_mod, "try_load_embedder", _raise_deps
    )

    result = _dma_mod.DistractorMisconceptionAlignmentValidator().validate(
        _dma_inputs()
    )

    assert result.passed is True
    assert "EMBEDDING_DEPS_MISSING" in [i.code for i in result.issues]
