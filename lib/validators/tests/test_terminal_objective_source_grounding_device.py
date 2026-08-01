"""The TO<->source grounding gate must not vacuously pass on a dead device.

``TerminalObjectiveSourceGroundingValidator`` sits on the same
``try_load_embedder`` loader as the seven statistical-tier validators pinned by
``test_embedding_device_fail_closed.py``, but it was NOT narrowed in that round:
both of its ``embedder.encode`` call sites swallowed
:class:`~lib.embedding.sentence_embedder.EmbeddingModelUnavailable` into a bare
``except Exception``, set the vector to ``None``, and then crashed formatting
the resulting ``None`` cosine::

    TypeError: unsupported format string passed to NoneType.__format__

So on a host where ``ED4ALL_EMBEDDING_DEVICE=cuda`` cannot come up, the gate
neither scored nor failed cleanly — it raised an untyped ``TypeError`` from a
message-formatting f-string. This module pins the three contracts apart:

(a) **Extras present, requested device absent** ->
    :class:`EmbeddingModelUnavailable` is FATAL. It propagates out of
    ``validate()`` and is never converted into a passing ``GateResult``.
    Covered at all three guard layers: the loader, the eager ``preload()``, and
    each of the two per-encode re-raises that backstop it.
(b) **Missing ``[embedding]`` extras** -> unchanged. A warning-severity
    ``EMBEDDING_DEPS_MISSING`` GateIssue with ``passed=True``.
(c) **``TRAINFORGE_REQUIRE_EMBEDDINGS=true``** -> still flips (b) closed.

Plus the negative control: a genuinely *transient* encode error still degrades
to a passing gate — but as a typed ``EMBEDDING_ENCODE_ERROR`` warning, NOT as a
``TypeError``. A crash is not an acceptable substitute for a clean failure.

No GPU is touched — every embedder here is a stub. Run CPU-pinned:
``ED4ALL_EMBEDDING_DEVICE=cpu ED4ALL_NLI_DEVICE=cpu``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

import lib.validators.terminal_objective_source_grounding as mod
from lib.embedding.sentence_embedder import (
    EmbeddingDepsMissing,
    EmbeddingModelUnavailable,
)
from lib.validators.terminal_objective_source_grounding import (
    TerminalObjectiveSourceGroundingValidator,
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
        return [1.0, 0.0]


class _EncodeUnavailableEmbedder:
    """Duck-typed embedder with NO ``preload()`` — the backstop case.

    Proves the TO-statement encode guard re-raises the typed error even when
    the eager ``preload()`` probe is unavailable on the injected object.
    """

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        raise EmbeddingModelUnavailable(_DEVICE_ERR)


class _ChunkEncodeUnavailableEmbedder:
    """The TO statement encodes; the first CHUNK encode hits the dead device.

    The validator has TWO encode call sites and the chunk one is reached only
    after the statement one succeeds, so it needs its own stub to be covered.
    """

    def __init__(self, statement_prefix: str) -> None:
        self._statement_prefix = statement_prefix

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        if text.startswith(self._statement_prefix):
            return [1.0, 0.0]
        raise EmbeddingModelUnavailable(_DEVICE_ERR)


class _TransientEncodeErrorEmbedder:
    """Loads fine, then fails per-encode for a NON-device reason.

    The negative control: this must keep degrading, and must not crash.
    """

    def preload(self) -> None:
        return None

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        raise RuntimeError("transient encode blip")


class _HealthyEmbedder:
    """Deterministic unit vectors, so the happy path stays scoreable.

    Mirrors the stub in ``test_terminal_objective_source_grounding.py``; kept
    local so this suite pins the device contract without depending on another
    module's fixture shape.
    """

    def __init__(self, angle_map: Dict[str, float]) -> None:
        self.angle_map = angle_map
        self.preload_calls = 0

    def preload(self) -> None:
        self.preload_calls += 1

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        match_len = -1
        angle = 0.0
        for key, ang in self.angle_map.items():
            if text.startswith(key) and len(key) > match_len:
                match_len = len(key)
                angle = ang
        return [math.cos(angle), math.sin(angle)]


# --------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------- #

_TO_STATEMENT = "photosynthesis converts light to chemical energy"
_CHUNK_TEXT = "photosynthesis converts light energy into chemical bonds"


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is opt-in; without this every case is a no-op pass."""
    monkeypatch.setenv("ED4ALL_TO_SOURCE_GROUNDING", "1")
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)


def _inputs() -> Dict[str, Any]:
    """A single TO with one REAL cited chunk that resolves in the chunkset."""
    return {
        "synthesized_objectives": {
            "terminal_objectives": [
                {
                    "id": "TO-01",
                    "statement": _TO_STATEMENT,
                    "source_refs": [{"chunk_ids": ["c1"]}],
                }
            ],
            "chapter_objectives": [
                {
                    "id": "CO-01",
                    "statement": "explain photosynthesis",
                    "terminal_id": "TO-01",
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        "chunks_by_id": {"c1": _CHUNK_TEXT},
    }


def _codes(result: Any) -> List[str]:
    return [i.code for i in result.issues]


# --------------------------------------------------------------------- #
# (a) Device unavailable -> FATAL, never a passing gate, never a TypeError.
# --------------------------------------------------------------------- #


def test_device_unavailable_at_preload_is_fatal() -> None:
    """The eager ``preload()`` surfaces the device failure before any encode.

    This is the regression guard for the CUDA-default flip: with no eager probe
    the error arrived lazily inside ``encode()``, where the bare
    ``except Exception`` logged it away.
    """
    embedder = _PreloadUnavailableEmbedder()
    validator = TerminalObjectiveSourceGroundingValidator(embedder=embedder)

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(_inputs())

    # The failure must land BEFORE any scoring work is attempted.
    assert embedder.encode_calls == []


def test_device_unavailable_at_statement_encode_is_fatal() -> None:
    """Backstop: the TO-statement encode guard re-raises the typed error.

    Exercised with an injected embedder that has no ``preload()`` at all, so
    only the narrowed ``except`` clause can catch this.
    """
    validator = TerminalObjectiveSourceGroundingValidator(
        embedder=_EncodeUnavailableEmbedder()
    )

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(_inputs())


def test_device_unavailable_at_chunk_encode_is_fatal() -> None:
    """Backstop: the SECOND encode call site re-raises the typed error too.

    The chunk-text encode is inside the per-chunk memoization loop, one level
    deeper than the statement encode, and had its own bare ``except``.
    """
    validator = TerminalObjectiveSourceGroundingValidator(
        embedder=_ChunkEncodeUnavailableEmbedder(_TO_STATEMENT)
    )

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(_inputs())


def test_device_unavailable_from_loader_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device failure raised by ``try_load_embedder`` is fatal as well.

    No ``embedder=`` override here, so the validator goes through its real
    loader seam — the path a production run takes.
    """

    def _raise_device(*args: Any, **kwargs: Any) -> Any:
        raise EmbeddingModelUnavailable(_DEVICE_ERR)

    monkeypatch.setattr(mod, "try_load_embedder", _raise_device)
    validator = TerminalObjectiveSourceGroundingValidator()

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(_inputs())


def test_device_unavailable_is_not_swallowed_as_a_typeerror() -> None:
    """The exact pre-fix symptom, pinned so it cannot come back.

    Before the fix this raised ``TypeError: unsupported format string passed to
    NoneType.__format__`` from the ``TO_SOURCE_UNGROUNDED`` message f-string —
    a crash that masked the real device failure. The typed error must reach the
    caller intact.
    """
    validator = TerminalObjectiveSourceGroundingValidator(
        embedder=_EncodeUnavailableEmbedder()
    )

    with pytest.raises(EmbeddingModelUnavailable) as excinfo:
        validator.validate(_inputs())

    assert not isinstance(excinfo.value, TypeError)
    assert "device" in str(excinfo.value)


# --------------------------------------------------------------------- #
# Negative control: a transient encode error degrades — cleanly.
# --------------------------------------------------------------------- #


def test_transient_encode_error_degrades_without_crashing() -> None:
    """Only the TYPED unavailability is fatal.

    A non-device encode failure keeps its degrade, but it must land as a
    warning-severity ``EMBEDDING_ENCODE_ERROR`` rather than the ``TypeError``
    the ``None`` cosine used to produce — and it must not be counted as an
    ungrounded TO, since an infrastructure blip is not evidence about the
    source text.
    """
    validator = TerminalObjectiveSourceGroundingValidator(
        embedder=_TransientEncodeErrorEmbedder()
    )

    result = validator.validate(_inputs())

    assert result.passed is True
    assert "EMBEDDING_ENCODE_ERROR" in [
        i.code for i in result.issues if i.severity == "warning"
    ]
    assert "TO_SOURCE_UNGROUNDED" not in _codes(result)
    assert all(i.severity != "critical" for i in result.issues)


def test_unresolved_chunk_ids_still_report_ungrounded() -> None:
    """The other ``best is None`` path is untouched by the crash fix.

    A TO whose cited chunk ids resolve to NOTHING in the chunkset takes the
    ``not texts`` message branch, which never formatted a cosine — it must keep
    reporting ``TO_SOURCE_UNGROUNDED``, not the new encode-error code.
    """
    inputs = _inputs()
    inputs["chunks_by_id"] = {"some-other-chunk": _CHUNK_TEXT}
    validator = TerminalObjectiveSourceGroundingValidator(
        embedder=_HealthyEmbedder({_TO_STATEMENT: 0.0, _CHUNK_TEXT: 0.0})
    )

    result = validator.validate(inputs)

    assert result.passed is True
    assert "TO_SOURCE_UNGROUNDED" in _codes(result)
    assert "EMBEDDING_ENCODE_ERROR" not in _codes(result)


def test_healthy_embedder_is_preloaded_and_scores() -> None:
    """The eager probe does not disturb the happy path.

    ``preload()`` is called exactly once per ``validate()``, and a well-aligned
    TO/chunk pair still scores as grounded.
    """
    embedder = _HealthyEmbedder({_TO_STATEMENT: 0.0, _CHUNK_TEXT: 0.0})
    validator = TerminalObjectiveSourceGroundingValidator(embedder=embedder)

    result = validator.validate(_inputs())

    assert embedder.preload_calls == 1
    assert result.passed is True
    assert result.score == 1.0
    assert _codes(result) == []


# --------------------------------------------------------------------- #
# (b) Missing extras -> warning + passed=True. UNCHANGED.
# --------------------------------------------------------------------- #


def test_missing_extras_still_warns_and_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional-extras escape hatch survives the fail-closed change.

    A fix that made the device path fail closed by also breaking this is a
    failed fix — the two contracts must stay distinct.
    """
    monkeypatch.setattr(mod, "try_load_embedder", lambda *a, **kw: None)
    validator = TerminalObjectiveSourceGroundingValidator()

    result = validator.validate(_inputs())

    assert result.passed is True
    assert "EMBEDDING_DEPS_MISSING" in [
        i.code for i in result.issues if i.severity == "warning"
    ]
    assert all(i.severity != "critical" for i in result.issues)


# --------------------------------------------------------------------- #
# (c) Strict mode still flips missing-extras closed.
# --------------------------------------------------------------------- #


def test_strict_mode_missing_extras_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TRAINFORGE_REQUIRE_EMBEDDINGS=true`` keeps failing the build closed.

    The validator lets ``EmbeddingDepsMissing`` propagate; the gate manager's
    ``on_error: fail_closed`` turns that into a blocked phase.
    """
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    def _raise_deps(*args: Any, **kwargs: Any) -> Any:
        raise EmbeddingDepsMissing("sentence-transformers is not installed")

    monkeypatch.setattr(mod, "try_load_embedder", _raise_deps)
    validator = TerminalObjectiveSourceGroundingValidator()

    with pytest.raises(EmbeddingDepsMissing):
        validator.validate(_inputs())


def test_device_error_stays_fatal_under_non_strict_mode() -> None:
    """The device contract does not depend on the strict-mode flag.

    ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is explicitly unset by the autouse
    fixture here, and the typed error is still fatal — that independence is the
    whole point of the two exception types.
    """
    validator = TerminalObjectiveSourceGroundingValidator(
        embedder=_PreloadUnavailableEmbedder()
    )

    with pytest.raises(EmbeddingModelUnavailable):
        validator.validate(_inputs())
