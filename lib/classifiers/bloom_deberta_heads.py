"""Load optional DeBERTa Bloom-classifier artifacts from local storage.

The ``ED4ALL_BLOOM_TRIVOTE_HEADS`` validation path can use these heads as an
additional high-precision voter alongside the asserted level, zero-shot NLI
heuristic (:mod:`lib.classifiers.bloom_zero_shot`), and deterministic verb
ontology. The head path is staged, unproven, and unprovisioned: Ed4All ships no
weights, enabling the flag does not make a classifier available, and this
loader does not authorize training. Current validator status is documented in
``docs/validation/validators.md``; model and training-data licensing posture is
documented in ``docs/LICENSING.md``.

The loader supports either of two trained-artifact shapes (see
``lib.classifiers.training.train_bloom_deberta``'s ``--head`` flag):

* **multiclass** — one ``num_labels=6`` softmax head at
  ``<heads_dir>/multiclass/final``.
  :meth:`BloomDebertaHeads.classify_batch` softmaxes the six logits and
  takes the argmax + its probability.
* **one-vs-rest** — six separate purpose-fine-tuned DeBERTa heads, one per
  canonical Bloom level, each trained as an
  independent one-vs-rest binary classifier ("is this text at level X,
  yes/no"). :meth:`BloomDebertaHeads.classify_batch` argmaxes the six
  per-level sigmoid scores.

Either way, the result abstains (returns ``None``) when the winning
score is below a confidence floor, so a low-confidence text contributes
no vote rather than a noisy one. :meth:`get_or_load` auto-detects which
artifact is on disk (multiclass checked FIRST) and surfaces the resolved
mode as :attr:`BloomDebertaHeads.head_mode` (``"multiclass"`` or
``"one-vs-rest"``).

Mirrors :class:`lib.classifiers.nli_classifier.NliClassifier`'s
singleton-load contract almost exactly:

* Class-level ``_INSTANCE`` / ``_LOAD_FAILED`` / ``_LOAD_LOCK`` +
  :meth:`get_or_load` — lazy process-singleton, negative result cached,
  thread-safe first load.
* Device policy mirrors :data:`lib.classifiers.nli_classifier.ENV_DEVICE`
  (``ED4ALL_NLI_DEVICE``) exactly — reused directly rather than adding a
  sibling knob, so the six heads share the same cuda/cpu decision (and
  the same VRAM-contention posture) as the NLI classifier they sit
  beside in the statistical validation tier. Default ``"cpu"`` (no
  ``.to()`` / ``.half()``, byte-identical to the historical no-heads
  behavior); on CUDA the heads are cast to fp16; CUDA-requested-but-
  unavailable falls back to CPU with a one-time warning rather than
  crashing.

The one contract this module OWNS: :data:`ENV_HEADS_DIR`
(``ED4ALL_BLOOM_HEADS_DIR``, default ``models/bloom_classifiers``) — the
base directory under which either loadable artifact lives:
``<heads_dir>/multiclass/final`` (single head) or
``<heads_dir>/<level>/final`` per :data:`BLOOM_LEVELS` entry (six heads).
:meth:`get_or_load` checks for the multiclass artifact FIRST — if
``<heads_dir>/multiclass/final`` exists, it loads that single head and
never even probes the six per-level directories; only when the
multiclass artifact is absent does it fall back to assembling the
one-vs-rest ladder. Every checkpoint is loaded via
``from_pretrained(<local filesystem path>)`` ONLY — never a HuggingFace
hub repo id — so the loader is HF-offline BY CONSTRUCTION: there is no
code path that can reach the network, with or without
``HF_HUB_OFFLINE``. ``local_files_only=True`` is passed to every
``from_pretrained`` call as defense-in-depth (belt-and-suspenders on
top of the local-path-only contract).

Graceful degrade is the default for both shapes: no repo ships either
artifact, so on a
fresh checkout ``ED4ALL_BLOOM_HEADS_DIR`` resolves to a directory that
doesn't exist, :meth:`get_or_load` returns ``None`` on the first (cheap)
directory-existence checks, and NO heavy ``torch`` / ``transformers``
import is even attempted. For the one-vs-rest ladder specifically: if
any of the six per-level directories is present but a SIBLING is
missing (a partial ladder), the loader ALSO returns ``None`` — a
partial vote set would corrupt the trivote's "the fourth voter fires or
it doesn't" contract, so that shape is all-or-nothing by design, never
a silent partial
vote. Missing ``torch`` / ``transformers`` extras and any per-head load
failure degrade the same way. Callers MUST handle the ``None`` return
exactly like :meth:`NliClassifier.get_or_load` — treat it as "this
voter abstains", never as an error.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.classifiers.nli_classifier import ENV_DEVICE, resolve_nli_device

logger = logging.getLogger(__name__)


#: Canonical Bloom's-taxonomy levels, lowest to highest. Mirrors
#: ``lib.ontology.bloom.BLOOM_LEVELS`` but inlined (same rationale as
#: ``lib.classifiers.bloom_bert_ensemble._BLOOM_LEVELS``): keeps this
#: classifier import-light rather than pulling in the JSON-taxonomy
#: loader ``lib.ontology`` triggers at import time.
BLOOM_LEVELS: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)

#: Env var: base directory holding the six per-Bloom-level DeBERTa head
#: checkpoints, one subdirectory per :data:`BLOOM_LEVELS` entry
#: (``<dir>/<level>/final``). Default ``"models/bloom_classifiers"`` —
#: a relative repo path that does not exist in a fresh checkout, so the
#: default-OFF behavior (no shipped weights -> loader degrades to
#: ``None``) holds without any additional gating. Documented in root
#: docs/operations/behavior-flags.md (indexed in root CLAUDE.md).
ENV_HEADS_DIR = "ED4ALL_BLOOM_HEADS_DIR"

#: Default base directory (see :data:`ENV_HEADS_DIR`).
_DEFAULT_HEADS_DIR = "models/bloom_classifiers"

#: Per-level checkpoint subdirectory name under ``<heads_dir>/<level>/``.
_HEAD_CHECKPOINT_SUBDIR = "final"

#: Multiclass single-head subdirectory name under ``<heads_dir>/`` --
#: mirrors ``lib.classifiers.training.train_bloom_deberta.default_multiclass_output_dir``
#: verbatim (same "cheap constant kept in sync by convention" rationale as
#: :data:`_DEFAULT_HEADS_DIR` vs. the trainer's own copy). The loader checks
#: ``<heads_dir>/multiclass/final`` for this artifact BEFORE falling back to
#: the six-way one-vs-rest ladder (see module docstring).
_MULTICLASS_SUBDIR = "multiclass"

#: Maximum tokenized sequence length (mirrors
#: ``lib.classifiers.nli_classifier._MAX_SEQUENCE_LENGTH`` — DeBERTa-v3
#: is trained on 512-token sequences).
_MAX_SEQUENCE_LENGTH: int = 512

#: Default forward-pass batch size per head (mirrors
#: ``lib.classifiers.nli_classifier._DEFAULT_BATCH_SIZE``).
_DEFAULT_BATCH_SIZE: int = 8

#: Default one-vs-rest sigmoid confidence floor. A text whose ARGMAX
#: level scores below this abstains (``None``) rather than casting a
#: low-confidence vote — the trivote abstention contract. ``0.5`` is
#: the natural per-class decision boundary for an independently-
#: calibrated one-vs-rest sigmoid head (unlike a softmax vote, each
#: head's score is not forced to sum to 1 across levels, so "which
#: level won" and "was the winner actually confident" are separate
#: questions; the floor answers the second one).
_DEFAULT_ABSTAIN_FLOOR: float = 0.5


def resolve_bloom_heads_dir(value: Optional[str] = None) -> str:
    """Resolve the base directory for the six per-level head checkpoints.

    Resolution chain: explicit ``value`` arg -> :data:`ENV_HEADS_DIR` env
    -> default :data:`_DEFAULT_HEADS_DIR`. Parse-with-fallback: a blank /
    whitespace-only value at any stage falls through to the next stage
    (an explicit-but-blank arg is treated as "not provided" rather than
    resolving to an empty path).
    """
    if value is not None and str(value).strip():
        return str(value).strip()
    raw = os.environ.get(ENV_HEADS_DIR)
    if raw is None or not raw.strip():
        return _DEFAULT_HEADS_DIR
    return raw.strip()


def default_registry(heads_dir: Optional[str] = None) -> Dict[str, Path]:
    """Return ``{level: checkpoint_dir}`` for the six canonical levels.

    ``checkpoint_dir`` is ``<resolved heads_dir>/<level>/final`` for
    every entry in :data:`BLOOM_LEVELS`. ``heads_dir`` follows
    :func:`resolve_bloom_heads_dir`'s resolution chain (explicit arg ->
    env -> default), so passing ``None`` (the common case) reads the
    live env at call time.
    """
    base = Path(resolve_bloom_heads_dir(heads_dir))
    return {
        level: base / level / _HEAD_CHECKPOINT_SUBDIR for level in BLOOM_LEVELS
    }


def resolve_multiclass_head_dir(heads_dir: Optional[str] = None) -> Path:
    """``<resolved heads_dir>/multiclass/final`` -- the single 6-way head
    checkpoint :meth:`BloomDebertaHeads.get_or_load` checks for BEFORE
    falling back to :func:`default_registry`'s six-way one-vs-rest ladder.

    ``heads_dir`` follows :func:`resolve_bloom_heads_dir`'s resolution
    chain (explicit arg -> env -> default), mirroring
    :func:`default_registry`.
    """
    base = Path(resolve_bloom_heads_dir(heads_dir))
    return base / _MULTICLASS_SUBDIR / _HEAD_CHECKPOINT_SUBDIR


def _is_loadable_head_dir(path: Path) -> bool:
    """True iff ``path`` looks like a real local HF checkpoint directory.

    Checks for ``config.json`` specifically (rather than mere directory
    existence) so an empty/partially-materialized directory (e.g. a
    checkpoint export still in flight) is correctly treated as "not
    ready yet" rather than triggering a ``from_pretrained`` failure deep
    inside the loader.
    """
    return path.is_dir() and (path / "config.json").is_file()


class BloomDebertaHeads:
    """Process-singleton holder of the Bloom DeBERTa classifier head(s).

    Backs EITHER of two trained-artifact shapes, never both at once (see
    module docstring):

    * **one-vs-rest** — six independently fine-tuned binary classifiers
      (single-logit ``AutoModelForSequenceClassification`` checkpoints),
      each answering "is this text at Bloom level X".
    * **multiclass** — one ``num_labels=6`` softmax
      ``AutoModelForSequenceClassification`` checkpoint over all six
      levels at once.

    Both are loaded exclusively from local filesystem checkpoint
    directories — see the module docstring for the HF-offline-by-
    construction contract. :attr:`head_mode` (``"one-vs-rest"`` or
    ``"multiclass"``) records which shape a given instance was built
    from. Singleton-load via :meth:`get_or_load`; the in-memory head(s)
    are shared across every consumer of this surface.
    """

    # Module-level singleton state — same shape as
    # ``NliClassifier``'s: ``_INSTANCE`` caches the loaded holder,
    # ``_LOAD_FAILED`` caches the negative result (missing checkpoints /
    # missing extras / load error) so repeat ``get_or_load()`` calls are
    # O(1) instead of re-probing the filesystem + re-importing torch on
    # every call, and the lock prevents two threads from racing the load.
    _INSTANCE: Optional["BloomDebertaHeads"] = None
    _LOAD_FAILED: bool = False
    _LOAD_LOCK: threading.Lock = threading.Lock()

    def __init__(
        self,
        *,
        heads: Optional[Dict[str, Tuple[Any, Any]]] = None,
        multiclass_head: Optional[Tuple[Any, Any]] = None,
        torch_module: Any,
        device: Optional[str] = None,
    ) -> None:
        """Direct constructor — prefer :meth:`get_or_load` from consumer code.

        Pass exactly ONE of ``heads`` (the one-vs-rest ladder) or
        ``multiclass_head`` (the single 6-way head) — passing both or
        neither raises ``RuntimeError``, the same "fail loudly rather
        than silently pick one" posture as the partial-ladder check
        below.

        ``heads`` maps every :data:`BLOOM_LEVELS` entry to its own
        ``(model, tokenizer)`` pair; a caller passing a partial or
        mismatched key set gets a loud ``RuntimeError`` at construction
        (defends against a silently-partial ladder the same way
        ``NliClassifier`` defends against a silently-shifted
        ``id2label``).

        ``multiclass_head`` is a single ``(model, tokenizer)`` pair for
        the ``num_labels=6`` softmax checkpoint.

        ``device`` follows :func:`resolve_nli_device`'s resolution chain
        (explicit arg > ``ED4ALL_NLI_DEVICE`` > ``"cpu"``) — the SAME
        knob :class:`~lib.classifiers.nli_classifier.NliClassifier`
        reads, so the two classifier families never disagree about
        which device to use on a given box.
        """
        if heads is not None and multiclass_head is not None:
            raise RuntimeError(
                "BloomDebertaHeads accepts either `heads` (the one-vs-rest "
                "ladder) or `multiclass_head` (the single 6-way head), "
                "never both."
            )
        if heads is None and multiclass_head is None:
            raise RuntimeError(
                "BloomDebertaHeads requires exactly one of `heads` or "
                "`multiclass_head`."
            )

        if multiclass_head is not None:
            self.head_mode = "multiclass"
            self._heads: Optional[Dict[str, Tuple[Any, Any]]] = None
            self._multiclass: Optional[Tuple[Any, Any]] = multiclass_head
            placement_input: Dict[str, Tuple[Any, Any]] = {
                "multiclass": multiclass_head
            }
        else:
            assert heads is not None  # narrowed by the checks above
            missing = set(BLOOM_LEVELS) - set(heads.keys())
            extra = set(heads.keys()) - set(BLOOM_LEVELS)
            if missing or extra:
                raise RuntimeError(
                    f"BloomDebertaHeads requires exactly the six BLOOM_LEVELS "
                    f"as keys; missing={sorted(missing)!r} extra={sorted(extra)!r}. "
                    f"A partial ladder would corrupt the trivote abstention "
                    f"contract, so construction fails loudly instead of "
                    f"silently voting with fewer levels."
                )
            self.head_mode = "one-vs-rest"
            self._heads = dict(heads)
            self._multiclass = None
            placement_input = self._heads

        self._torch = torch_module
        self._device, self._dtype = self._place_heads_on_device(
            placement_input, torch_module, resolve_nli_device(device)
        )

    @staticmethod
    def _place_heads_on_device(
        heads: Dict[str, Tuple[Any, Any]],
        torch_module: Any,
        device: str,
    ) -> Tuple[str, str]:
        """Move + (on CUDA) fp16-cast every head. Return ``(device, dtype)``.

        Mirrors ``NliClassifier._place_model_on_device`` without the NLI
        module's VRAM-contention eviction machinery. This loader defaults to
        CPU, casts to fp16 on CUDA, and falls back to CPU when CUDA is
        requested but unavailable.

        * ``device == "cpu"`` — no-op, byte-identical to the historical
          (no-heads) behavior. Returns ``("cpu", "float32")``.
        * CUDA requested and available — every head's model is moved +
          ``.half()``-cast. Returns ``(device, "float16")``.
        * CUDA requested but unavailable, or any placement error — logs
          a warning and falls back to CPU rather than crashing. Returns
          ``("cpu", "float32")``.
        """
        if device == "cpu":
            return "cpu", "float32"

        try:
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 — torch probe is best-effort
            logger.warning(
                "BloomDebertaHeads could not probe torch.cuda.is_available() "
                "(%s); falling back to CPU (fp32).",
                exc,
            )
            return "cpu", "float32"

        if not cuda_available:
            logger.warning(
                "BloomDebertaHeads device %r requested but torch.cuda is "
                "not available; falling back to CPU (fp32). Set %s=cpu to "
                "silence this, or provision a CUDA device.",
                device, ENV_DEVICE,
            )
            return "cpu", "float32"

        try:
            for _level, (model, _tokenizer) in heads.items():
                # fp16 keeps each ~184M-param head's VRAM footprint small
                # (mirrors the NLI classifier's rationale for coexisting
                # with a resident local ollama LLM on a shared card).
                model.to(device)
                model.half()
        except Exception as exc:  # noqa: BLE001 — OOM, driver error, etc.
            logger.warning(
                "BloomDebertaHeads failed to move a head to %r / cast fp16 "
                "(%s); falling back to CPU (fp32).",
                device, exc,
            )
            return "cpu", "float32"

        logger.info(
            "BloomDebertaHeads scoring on %s (fp16) — GPU-accelerated "
            "per-Bloom-level heads.",
            device,
        )
        return device, "float16"

    @property
    def device(self) -> str:
        """Resolved torch device the heads score on (``"cpu"`` / ``"cuda*"``)."""
        return self._device

    @property
    def dtype(self) -> str:
        """Resolved head dtype (``"float32"`` on CPU, ``"float16"`` on CUDA)."""
        return self._dtype

    @classmethod
    def get_or_load(cls) -> Optional["BloomDebertaHeads"]:
        """Return the process-singleton instance; lazy-load on first call.

        Checks for the **multiclass** artifact FIRST
        (``<heads_dir>/multiclass/final`` -- see :func:`resolve_multiclass_head_dir`):
        if it exists, loads that single 6-way head and never even probes
        the six per-level directories. Only when the multiclass artifact
        is absent does it fall back to assembling the one-vs-rest ladder
        (:func:`default_registry`).

        Returns ``None`` permanently if any of the following holds:

        * neither the multiclass artifact nor all six per-level
          one-vs-rest checkpoints are present (the common case on a
          fresh checkout — no weights are shipped in-repo);
        * ``transformers`` / ``torch`` extras are not installed;
        * the multiclass head's (or any one-vs-rest head's)
          model/tokenizer load fails for any reason (corrupt checkpoint,
          permission error, ...).

        The negative result is cached so subsequent calls are O(1) and
        never re-probe the filesystem or re-attempt the (slow) import.
        Callers MUST handle the ``None`` return the same way they handle
        :meth:`NliClassifier.get_or_load` returning ``None`` — this
        voter abstains from the ladder, it never raises into a gate.
        """
        if cls._INSTANCE is not None:
            return cls._INSTANCE
        if cls._LOAD_FAILED:
            return None

        with cls._LOAD_LOCK:
            # Re-check inside the lock — another thread may have
            # finished the load while we were waiting.
            if cls._INSTANCE is not None:
                return cls._INSTANCE
            if cls._LOAD_FAILED:
                return None

            # Cheap filesystem check FIRST, before importing the heavy
            # torch/transformers stack: a fresh checkout ships neither
            # artifact, so this is the path taken every time the flag
            # family is off, and it should cost nothing beyond a handful
            # of stat() calls. Multiclass wins the preference order when
            # both artifacts happen to be present.
            multiclass_dir = resolve_multiclass_head_dir()
            if _is_loadable_head_dir(multiclass_dir):
                return cls._load_multiclass_locked(multiclass_dir)
            return cls._load_one_vs_rest_locked()

    @classmethod
    def _load_multiclass_locked(cls, multiclass_dir: Path) -> Optional["BloomDebertaHeads"]:
        """Load the single 6-way multiclass head. MUST be called with
        ``_LOAD_LOCK`` already held (see :meth:`get_or_load`)."""
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            logger.debug(
                "BloomDebertaHeads deps missing (%s); get_or_load "
                "returning None",
                exc,
            )
            cls._LOAD_FAILED = True
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BloomDebertaHeads import raised unexpected error: %s; "
                "get_or_load returning None.",
                exc,
            )
            cls._LOAD_FAILED = True
            return None

        local_dir = str(multiclass_dir)
        try:
            # ``local_files_only=True`` is defense-in-depth on top of the
            # local-path-only contract: ``local_dir`` is a real filesystem
            # path (never a hub repo id), so ``from_pretrained`` never
            # needs the network -- this kwarg additionally forbids it
            # from EVER trying.
            tokenizer = AutoTokenizer.from_pretrained(
                local_dir, local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                local_dir, local_files_only=True,
            )
            model.eval()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BloomDebertaHeads failed to load the multiclass head "
                "from %r: %s; get_or_load returning None.",
                multiclass_dir, exc,
            )
            cls._LOAD_FAILED = True
            return None

        try:
            instance = cls(multiclass_head=(model, tokenizer), torch_module=torch)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BloomDebertaHeads construction failed after loading the "
                "multiclass head: %s; get_or_load returning None.",
                exc,
            )
            cls._LOAD_FAILED = True
            return None

        cls._INSTANCE = instance
        return instance

    @classmethod
    def _load_one_vs_rest_locked(cls) -> Optional["BloomDebertaHeads"]:
        """Load the six-way one-vs-rest ladder. MUST be called with
        ``_LOAD_LOCK`` already held (see :meth:`get_or_load`)."""
        registry = default_registry()
        missing_levels = sorted(
            level for level, path in registry.items()
            if not _is_loadable_head_dir(path)
        )
        if missing_levels:
            logger.debug(
                "BloomDebertaHeads: %d/%d per-level checkpoint dirs "
                "missing under %r (missing=%s); get_or_load returning "
                "None (trivote abstention contract — a partial ladder "
                "never loads).",
                len(missing_levels), len(registry),
                resolve_bloom_heads_dir(), missing_levels,
            )
            cls._LOAD_FAILED = True
            return None

        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            logger.debug(
                "BloomDebertaHeads deps missing (%s); get_or_load "
                "returning None",
                exc,
            )
            cls._LOAD_FAILED = True
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BloomDebertaHeads import raised unexpected error: %s; "
                "get_or_load returning None.",
                exc,
            )
            cls._LOAD_FAILED = True
            return None

        heads: Dict[str, Tuple[Any, Any]] = {}
        try:
            for level, path in registry.items():
                local_dir = str(path)
                # ``local_files_only=True`` is defense-in-depth on top of
                # the local-path-only contract: ``local_dir`` is a real
                # filesystem path (never a hub repo id), so
                # ``from_pretrained`` never needs the network — this
                # kwarg additionally forbids it from EVER trying.
                tokenizer = AutoTokenizer.from_pretrained(
                    local_dir, local_files_only=True,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    local_dir, local_files_only=True,
                )
                model.eval()
                heads[level] = (model, tokenizer)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BloomDebertaHeads failed to load head %r from %r: %s; "
                "get_or_load returning None.",
                level, path, exc,
            )
            cls._LOAD_FAILED = True
            return None

        try:
            instance = cls(heads=heads, torch_module=torch)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BloomDebertaHeads construction failed after loading "
                "all six heads: %s; get_or_load returning None.",
                exc,
            )
            cls._LOAD_FAILED = True
            return None

        cls._INSTANCE = instance
        return instance

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Test-only seam: clear the singleton state so a test can
        re-exercise the load path with a different mock.

        Production code MUST NOT call this; the singleton is
        process-stable by design.
        """
        cls._INSTANCE = None
        cls._LOAD_FAILED = False

    def classify(
        self, text: str, *, floor: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        """Classify a single ``text``. See :meth:`classify_batch`."""
        return self.classify_batch([text], floor=floor)[0]

    def classify_batch(
        self,
        texts: List[str],
        *,
        floor: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> List[Optional[Tuple[str, float]]]:
        """Classify ``texts`` via the loaded head(s) + argmax.

        Dispatches on :attr:`head_mode`:

        * ``"one-vs-rest"`` — every one of the six :data:`BLOOM_LEVELS`
          heads emits an independent sigmoid confidence ("is this text at
          level X"); the winner is the level with the highest score.
        * ``"multiclass"`` — the single 6-way head emits a softmax
          distribution over all six levels; the winner is the argmax
          level and its probability.

        Either way, ties resolve to the lexicographically-first level
        name (deterministic — mirrors ``BloomBertEnsemble._aggregate``'s
        tie-break so a replayed classification is stable across runs),
        and abstention is identical: when the winner's score is below
        ``floor`` (default :data:`_DEFAULT_ABSTAIN_FLOOR`), the text's
        result is ``None`` instead of a low-confidence vote — the
        trivote abstention contract. Returns a list the same length as
        ``texts``, one ``Optional[(level, confidence)]`` per input, in
        input order.

        Empty ``texts`` is a no-op pass that returns ``[]`` without
        dispatching any head.
        """
        if not texts:
            return []
        if self.head_mode == "multiclass":
            return self._classify_batch_multiclass(
                texts, floor=floor, batch_size=batch_size
            )
        return self._classify_batch_one_vs_rest(
            texts, floor=floor, batch_size=batch_size
        )

    def _classify_batch_one_vs_rest(
        self,
        texts: List[str],
        *,
        floor: Optional[float],
        batch_size: Optional[int],
    ) -> List[Optional[Tuple[str, float]]]:
        """``head_mode == "one-vs-rest"`` path: six independent sigmoid
        heads + argmax over their scores. See :meth:`classify_batch`."""
        resolved_floor = (
            float(floor) if floor is not None else _DEFAULT_ABSTAIN_FLOOR
        )
        size = (
            int(batch_size) if batch_size and batch_size > 0
            else _DEFAULT_BATCH_SIZE
        )
        torch = self._torch
        str_texts = [str(t) for t in texts]
        heads = self._heads
        assert heads is not None  # guaranteed by head_mode == "one-vs-rest"

        # per_level_scores[level][i] = sigmoid confidence for texts[i].
        per_level_scores: Dict[str, List[float]] = {
            level: [0.0] * len(str_texts) for level in BLOOM_LEVELS
        }

        with torch.no_grad():
            for level in BLOOM_LEVELS:
                model, tokenizer = heads[level]
                for start in range(0, len(str_texts), size):
                    batch = str_texts[start : start + size]
                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=_MAX_SEQUENCE_LENGTH,
                        return_tensors="pt",
                    )
                    if self._device != "cpu":
                        encoded = {
                            k: v.to(self._device) for k, v in encoded.items()
                        }
                    outputs = model(**encoded)
                    logits = outputs.logits
                    if logits.shape[-1] == 1:
                        # Single-logit head: sigmoid IS the positive prob.
                        probs = torch.sigmoid(logits[..., 0])
                    else:
                        # The train_bloom_deberta one-vs-rest recipe saves
                        # num_labels=2 heads (index 0 = negative class, 1 =
                        # positive). Softmax and read the POSITIVE column —
                        # sigmoid(logits[..., 0]) would score inverted.
                        probs = torch.softmax(logits, dim=-1)[..., 1]
                    if self._device != "cpu":
                        probs = probs.cpu()
                    for i in range(len(batch)):
                        score = float(probs[i].item())
                        per_level_scores[level][start + i] = score

        results: List[Optional[Tuple[str, float]]] = []
        for i in range(len(str_texts)):
            candidates = [
                (level, per_level_scores[level][i]) for level in BLOOM_LEVELS
            ]
            # Highest score wins; ties resolve lexicographically (stable,
            # replayable — mirrors BloomBertEnsemble._aggregate).
            winner_level, winner_score = sorted(
                candidates, key=lambda kv: (-kv[1], kv[0])
            )[0]
            if winner_score < resolved_floor:
                results.append(None)
            else:
                results.append((winner_level, round(winner_score, 4)))
        return results

    def _classify_batch_multiclass(
        self,
        texts: List[str],
        *,
        floor: Optional[float],
        batch_size: Optional[int],
    ) -> List[Optional[Tuple[str, float]]]:
        """``head_mode == "multiclass"`` path: one softmax head, argmax +
        max-probability over its six-way distribution. See
        :meth:`classify_batch`."""
        resolved_floor = (
            float(floor) if floor is not None else _DEFAULT_ABSTAIN_FLOOR
        )
        size = (
            int(batch_size) if batch_size and batch_size > 0
            else _DEFAULT_BATCH_SIZE
        )
        torch = self._torch
        str_texts = [str(t) for t in texts]
        assert self._multiclass is not None  # guaranteed by head_mode
        model, tokenizer = self._multiclass

        # per_text_scores[i] = [score_remember, score_understand, ...]
        # in BLOOM_LEVELS order.
        per_text_scores: List[List[float]] = [None] * len(str_texts)  # type: ignore[list-item]

        with torch.no_grad():
            for start in range(0, len(str_texts), size):
                batch = str_texts[start : start + size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=_MAX_SEQUENCE_LENGTH,
                    return_tensors="pt",
                )
                if self._device != "cpu":
                    encoded = {
                        k: v.to(self._device) for k, v in encoded.items()
                    }
                outputs = model(**encoded)
                logits = outputs.logits  # shape: (batch, 6)
                probs = torch.softmax(logits, dim=-1)
                if self._device != "cpu":
                    probs = probs.cpu()
                probs_rows = probs.tolist()  # List[List[float]]
                for i, row in enumerate(probs_rows):
                    per_text_scores[start + i] = list(row)

        results: List[Optional[Tuple[str, float]]] = []
        for row in per_text_scores:
            candidates = list(zip(BLOOM_LEVELS, row))
            # Highest score wins; ties resolve lexicographically (stable,
            # replayable — mirrors the one-vs-rest tie-break).
            winner_level, winner_score = sorted(
                candidates, key=lambda kv: (-kv[1], kv[0])
            )[0]
            if winner_score < resolved_floor:
                results.append(None)
            else:
                results.append((winner_level, round(winner_score, 4)))
        return results


__all__ = [
    "BloomDebertaHeads",
    "BLOOM_LEVELS",
    "ENV_HEADS_DIR",
    "resolve_bloom_heads_dir",
    "default_registry",
    "resolve_multiclass_head_dir",
]
