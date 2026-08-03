"""Three-member BERT ensemble for Bloom-level classification.

``BloomBertEnsemble`` exposes a ``classify`` contract over three pinned
classifier identities:

1. ``cip29/bert-blooms-taxonomy-classifier`` — purpose-built Bloom's
   classifier (6-class BERT-base fine-tune, apache-2.0 license,
   ``generated_from_trainer`` provenance). The model emits generic
   ``LABEL_0`` ... ``LABEL_5`` labels (no semantic ``id2label``
   mapping in its ``config.json``); :data:`_CIP29_TO_BLOOM` maps
   them onto the canonical Bloom enum following the standard
   hierarchical ordering convention (LABEL_0=remember,
   LABEL_1=understand, ..., LABEL_5=create).
2. ``distilbert-base-uncased-finetuned-sst-2-english`` — sentiment
   model used as a generic confidence-anchor signal. Its raw labels
   (``POSITIVE`` / ``NEGATIVE``) get mapped onto Bloom levels via the
   :data:`_SST2_TO_BLOOM` heuristic table; the goal is dispersion
   contribution, not a high-confidence vote, so the table is
   intentionally low-resolution.
3. ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` — zero-shot NLI
   classifier; given a candidate text + the six Bloom-level labels as
   hypotheses, picks the highest-entailment level.

SHA pinning: each member's ``revision`` field carries a concrete
HuggingFace git commit SHA so the ensemble's classification is
reproducible across runs. Revisions are captured in the
``bert_ensemble_member_loaded``
decision event so the audit trail records exactly which revision
produced each classification. Each entry's revision MUST match the
40-hex-char regex ``^[0-9a-f]{40}$`` (enforced by the test suite).

Graceful degradation: missing ``transformers`` extras raise
:class:`BertEnsembleDepsMissing` only when strict mode is on (see
:func:`is_strict_mode`); default mode logs a warning and returns
``[]`` from :meth:`_load_members`. The validator that consumes the
ensemble surfaces the missing-deps state via a warning-severity
GateIssue with code ``BERT_ENSEMBLE_DEPS_MISSING``, mirroring the
embedding-tier graceful-degrade pattern in
``lib/embedding/sentence_embedder.py``.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level config: ensemble member registry + Bloom-level mappings.
# ---------------------------------------------------------------------------

#: Default ensemble members. Each entry pins ``name`` (HuggingFace repo
#: id) and ``revision`` (concrete 40-hex-char git commit SHA). Override the
#: registry by passing ``members=`` to :class:`BloomBertEnsemble.__init__`.
#:
#: Compatibility behavior:
#:    The first two members are bypassed under ``ED4ALL_BLOOM_TRIVOTE``.
#:    ``cip29/bert-blooms-taxonomy-classifier``
#:    (member 0) carries no stated license — see ``docs/LICENSING.md`` — and
#:    ``distilbert-base-uncased-finetuned-sst-2-english`` (registry index 1) is a
#:    sentiment model mapped onto Bloom by a low-resolution heuristic. When
#:    the trivote flag is ON these two are **NEVER loaded**: the
#:    ``bloom_classifier_disagreement`` gate re-founds on three interpretable
#:    voters (the generator's OWN asserted level, a zero-shot pass over the
#:    ALREADY-LICENSED DeBERTa NLI member below, and the deterministic
#:    verb-ontology level) — see :mod:`lib.classifiers.bloom_zero_shot`. This
#:    list is retained UNCHANGED only to keep the legacy (flag-OFF) path
#:    byte-identical AND because :meth:`lib.classifiers.nli_classifier.
#:    NliClassifier.get_or_load` pulls the DeBERTa name+revision by INDEX
#:    (``_DEFAULT_ENSEMBLE_MEMBERS[2]``) — do NOT reorder or shorten this
#:    list without updating that pull, or every NLI-backed gate breaks.
_DEFAULT_ENSEMBLE_MEMBERS: List[Dict[str, str]] = [
    {
        "name": "cip29/bert-blooms-taxonomy-classifier",
        "revision": "ae343e4f4710e3cb48847b7db0d977d878c4a2e8",
    },
    {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
    },
    {
        "name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "revision": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
    },
]


#: Canonical Bloom's-taxonomy levels (mirrors ``lib.ontology.bloom.BLOOM_LEVELS``
#: but inlined to keep the classifier import-light — it's the only
#: per-module dependency the ensemble would otherwise pull from
#: ``lib.ontology``, which itself loads JSON taxonomies on import).
_BLOOM_LEVELS: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)


#: Heuristic mapping for the SST-2 sentiment member (registry index 1).
#: Sentiment
#: doesn't carry direct Bloom signal, so this table is intentionally
#: low-resolution: positive sentiment biases toward higher cognitive
#: levels (``evaluate`` / ``create``), negative toward lower
#: (``remember`` / ``understand``). Its role in the ensemble is to
#: contribute to dispersion when the other members disagree, not to
#: deliver a high-confidence vote on its own.
_SST2_TO_BLOOM: Dict[str, str] = {
    "POSITIVE": "evaluate",
    "NEGATIVE": "remember",
}


#: Translation table for ``cip29/bert-blooms-taxonomy-classifier``
#: (registry index 0). The model emits generic ``LABEL_0`` ...
#: ``LABEL_5`` labels with no semantic ``id2label`` mapping in its
#: ``config.json`` (verified at SHA-pin time —
#: ``huggingface_hub.hf_hub_download("cip29/bert-blooms-taxonomy-classifier",
#: "config.json")`` returns ``id2label = {"0": "LABEL_0", ...,
#: "5": "LABEL_5"}``). Per the standard convention for fine-tuned
#: 6-class Bloom classifiers, the canonical hierarchical ordering is
#: assumed: LABEL_0 → remember (lowest), ..., LABEL_5 → create
#: (highest). This mirrors the canonical :data:`_BLOOM_LEVELS` tuple
#: order. The translation runs inside :meth:`_classify_with_member`
#: against the model's raw argmax output before the resulting Bloom
#: level is returned to the ensemble vote aggregator. Every value MUST
#: be a member of :data:`_BLOOM_LEVELS` (enforced by the test suite).
_CIP29_TO_BLOOM: Dict[str, str] = {
    "LABEL_0": "remember",
    "LABEL_1": "understand",
    "LABEL_2": "apply",
    "LABEL_3": "analyze",
    "LABEL_4": "evaluate",
    "LABEL_5": "create",
}


#: Strict-mode env var (parallel of ``TRAINFORGE_REQUIRE_EMBEDDINGS``).
#: When truthy, missing ``transformers`` extras raise
#: :class:`BertEnsembleDepsMissing` instead of degrading silently.
_STRICT_MODE_ENV_VAR = "TRAINFORGE_REQUIRE_BERT_ENSEMBLE"
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})

#: Module-level latch so the "per-member dispatch unimplemented" warning
#: fires exactly once per process regardless of how many ensembles are
#: constructed (the synthesis call-path constructs one per validator,
#: and a per-instance log would flood the operator console).
_UNIMPLEMENTED_DISPATCH_WARNED = False


#: Default on-disk cache for downloaded model weights. Mirrors the
#: ``~/.cache/ed4all/`` convention used elsewhere; ``transformers``
#: itself respects ``TRANSFORMERS_CACHE`` and ``HF_HOME`` when set, so
#: this path is only used when the operator hasn't pinned one already.
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ed4all" / "bert_ensemble"


def is_strict_mode() -> bool:
    """Return True when ``TRAINFORGE_REQUIRE_BERT_ENSEMBLE`` is truthy."""
    raw = os.environ.get(_STRICT_MODE_ENV_VAR, "").strip().lower()
    return raw in _TRUTHY_VALUES


class BertEnsembleDepsMissing(RuntimeError):
    """Raised in strict mode when ``transformers`` is unavailable.

    Strict mode is opt-in via ``TRAINFORGE_REQUIRE_BERT_ENSEMBLE=true``.
    Default mode logs a warning and returns an empty member list from
    :meth:`BloomBertEnsemble._load_members` so downstream validators
    degrade to a warning-severity GateIssue instead of failing closed.
    """


@dataclass
class BertClassifier:
    """One loaded ensemble member.

    Holds the model + tokenizer references plus the registry metadata
    so :meth:`BloomBertEnsemble._classify_with_member` can dispatch on
    member name (the SST-2 + zero-shot members have different scoring
    paths than the native Bloom classifier).
    """

    name: str
    revision: str
    model: Any  # transformers.PreTrainedModel
    tokenizer: Any  # transformers.PreTrainedTokenizerBase


class BloomBertEnsemble:
    """Three-member BERT ensemble that classifies text into Bloom's levels.

    Public contract:

    .. code-block:: python

        ensemble = BloomBertEnsemble()
        result = ensemble.classify("Identify the main themes of the passage.")
        # {
        #     "winner_level": "remember",
        #     "winner_score": 0.82,
        #     "dispersion": 0.31,
        #     "per_member": [
        #         ("remember", 0.92),
        #         ("remember", 0.71),
        #         ("understand", 0.55),
        #     ],
        # }

    The ``per_member`` list preserves the registry order from
    :data:`_DEFAULT_ENSEMBLE_MEMBERS`. When a member fails to load
    (missing extras, network error, repo deleted), it is silently
    omitted from the ensemble and the remaining members vote among
    themselves. An empty member list returns a sentinel result
    (``winner_level="unknown"``, ``winner_score=0.0``,
    ``dispersion=0.0``, ``per_member=[]``) so downstream callers can
    short-circuit cleanly.
    """

    def __init__(
        self,
        members: Optional[List[Dict[str, str]]] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.members: List[Dict[str, str]] = (
            list(members) if members is not None else list(_DEFAULT_ENSEMBLE_MEMBERS)
        )
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._loaded: Optional[List[BertClassifier]] = None
        self._capture: Any = None  # optional DecisionCapture wired by caller
        # Optional per-member softmax-temperature override. ``None``
        # leaves vote confidences unscaled. When set to a scalar, the same
        # temperature applies to every member; when set to a list,
        # the i-th value applies to the i-th per-member vote (in
        # registry order).
        self._temperature: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify ``text`` into a Bloom level via majority + dispersion.

        Returns a dict with four keys: ``winner_level`` (str),
        ``winner_score`` (float — the winner's per-member-confidence
        sum, normalised to ``[0, 1]``), ``dispersion`` (float —
        entropy of the normalised per-level scores; high dispersion
        signals an unstable consensus), ``per_member`` (list of
        ``(level, confidence)`` tuples in registry order).
        """
        loaded = self._load_members()
        if not loaded:
            return {
                "winner_level": "unknown",
                "winner_score": 0.0,
                "dispersion": 0.0,
                "per_member": [],
            }

        per_member: List[Tuple[str, float]] = []
        for member in loaded:
            try:
                vote = self._classify_with_member(member, text)
            except Exception as exc:  # noqa: BLE001 — omit failed member votes
                logger.warning(
                    "BloomBertEnsemble member %s failed to classify: %s",
                    member.name,
                    exc,
                )
                continue
            per_member.append(vote)

        if not per_member:
            return {
                "winner_level": "unknown",
                "winner_score": 0.0,
                "dispersion": 0.0,
                "per_member": [],
            }

        winner_level, winner_score, dispersion = self._aggregate(
            per_member, temperature=self._temperature
        )
        return {
            "winner_level": winner_level,
            "winner_score": winner_score,
            "dispersion": dispersion,
            "per_member": per_member,
        }

    # ------------------------------------------------------------------ #
    # Loader + per-member classification
    # ------------------------------------------------------------------ #

    def _load_members(self) -> List[BertClassifier]:
        """Lazy-load every ensemble member.

        Uses a one-shot import probe of ``transformers``: when the
        extras are absent and strict mode is off, returns ``[]`` and
        downstream callers degrade to a warning-severity GateIssue.
        When strict mode is on (``TRAINFORGE_REQUIRE_BERT_ENSEMBLE=true``),
        raises :class:`BertEnsembleDepsMissing` with an operator-actionable
        install hint.

        Per-member loads are SHA-pinned via ``revision=member["revision"]``
        on both the tokenizer and model ``from_pretrained`` calls. Cache
        directory defaults to :data:`_DEFAULT_CACHE_DIR`
        (``~/.cache/ed4all/bert_ensemble/``), overridable via the
        constructor's ``cache_dir`` kwarg. ``transformers`` itself
        respects ``TRANSFORMERS_CACHE`` / ``HF_HOME`` env vars when set,
        which take precedence over the per-instance ``cache_dir``.

        Each load attempt — success or failure — emits one
        ``bert_ensemble_member_loaded`` decision event when a
        :class:`DecisionCapture` instance is attached via
        :meth:`attach_capture`. Members that fail to load are silently
        omitted from the ensemble; the remaining members vote among
        themselves.
        """
        if self._loaded is not None:
            return self._loaded

        global _UNIMPLEMENTED_DISPATCH_WARNED

        loaded: List[BertClassifier] = []
        try:
            # Probe-import only — actual model construction happens
            # per-member in :meth:`_load_one_member`.
            import transformers  # type: ignore  # noqa: F401
        except ImportError as exc:
            if is_strict_mode():
                raise BertEnsembleDepsMissing(
                    f"transformers is not installed but {_STRICT_MODE_ENV_VAR} "
                    f"is set: install via `pip install -e .[bert]`. "
                    f"Underlying error: {exc}"
                ) from exc
            logger.debug(
                "transformers not installed (%s); BloomBertEnsemble degrading "
                "to empty member list",
                exc,
            )
            self._loaded = loaded
            return loaded

        # Model-specific inference dispatch is not implemented yet, so
        # loading these models would produce placeholder votes rather than
        # classifications. Return no members so ``classify()`` reports the
        # explicit ``unknown`` sentinel. Strict mode raises instead because
        # callers that require the ensemble must not receive unknown results.
        #
        # TODO: implement per-member dispatch in ``_classify_with_member`` —
        # cip29 argmax via
        # ``_CIP29_TO_BLOOM``, SST-2 mapping via ``_SST2_TO_BLOOM``, and
        # DeBERTa zero-shot NLI entailment over the six Bloom labels.
        # Validate the implementation with observe-only calibration (see the
        # ``_ScriptedEnsemble`` aggregation tests in
        # ``lib/classifiers/tests/test_bloom_bert_ensemble.py`` for the
        # ``_aggregate`` contract), then restore the per-member load loop.
        if is_strict_mode():
            raise BertEnsembleDepsMissing(
                "BloomBertEnsemble per-member dispatch is unimplemented "
                f"(see Subtask-25), but {_STRICT_MODE_ENV_VAR} is set: "
                "refusing to fabricate votes. Implement "
                "_classify_with_member or unset the strict flag."
            )
        if not _UNIMPLEMENTED_DISPATCH_WARNED:
            logger.warning(
                "bloom ensemble per-member dispatch unimplemented; "
                "ensemble degrading to unknown — see Subtask-25"
            )
            _UNIMPLEMENTED_DISPATCH_WARNED = True
        self._loaded = loaded
        return loaded

    def _load_one_member(
        self, member: Dict[str, str]
    ) -> Optional[BertClassifier]:
        """Load a single ensemble member, SHA-pinned via ``revision``.

        Returns ``None`` on any load failure (network, missing revision,
        deleted repo). The caller logs the failure via
        :meth:`_emit_member_loaded` and continues with the remaining
        members; the ensemble's contract is "best-effort over the
        configured registry", not "fail-closed when any member is
        unreachable".
        """
        try:
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(
                member["name"],
                revision=member.get("revision", "main"),
                cache_dir=str(self.cache_dir),
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                member["name"],
                revision=member.get("revision", "main"),
                cache_dir=str(self.cache_dir),
            )
            return BertClassifier(
                name=member["name"],
                revision=member.get("revision", "main"),
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as exc:  # noqa: BLE001 — silent-degrade per contract
            logger.warning(
                "Failed to load BERT ensemble member %s@%s: %s",
                member.get("name"),
                member.get("revision"),
                exc,
            )
            return None

    def _classify_with_member(
        self, member: BertClassifier, text: str
    ) -> Tuple[str, float]:
        """Run inference on ``text`` with one ensemble member.

        Returns ``(bloom_level, confidence)``. Dispatches on member
        name: the cip29 Bloom classifier returns a 6-class argmax over
        its generic ``LABEL_0`` ... ``LABEL_5`` output, which is
        translated to canonical Bloom levels via
        :data:`_CIP29_TO_BLOOM`; the SST-2 member maps its 2-class output via
        :data:`_SST2_TO_BLOOM`; the zero-shot NLI member runs the
        six Bloom labels as candidate hypotheses and picks the highest
        entailment.

        Model-specific dispatch (cip29 argmax and label translation,
        SST-2 mapping, and zero-shot NLI entailment) is not implemented.
        The production loader
        (:meth:`_load_members`) now degrades to ``[]`` BEFORE any member
        reaches this method, so the default-constructed ensemble never
        invokes it — this body is reached only by test subclasses (e.g.
        ``_ScriptedEnsemble``) that override ``_load_members`` to inject
        scripted votes, or by a subclass that wires model-specific scoring.
        The placeholder return exists only to keep those scripted-vote tests
        exercising the full ``classify -> _aggregate`` path; it is not a real
        classification.

        TODO: implement the three model-specific dispatch branches, then
        restore the per-member load loop in
        :meth:`_load_members`.
        """
        return ("remember", 0.5)

    # ------------------------------------------------------------------ #
    # Decision capture — emit per-member load events
    # ------------------------------------------------------------------ #

    def _emit_member_loaded(
        self, member: Dict[str, str], *, success: bool
    ) -> None:
        """Emit a ``bert_ensemble_member_loaded`` decision event.

        No-ops when no capture is attached — the ensemble is usable
        stand-alone (e.g. from notebook smoke tests) without forcing
        callers to wire a capture path.
        """
        if self._capture is None:
            return
        try:
            self._capture.log_decision(
                decision_type="bert_ensemble_member_loaded",
                decision=(
                    f"loaded {member.get('name')}@{member.get('revision')}"
                    if success
                    else f"failed to load {member.get('name')}@{member.get('revision')}"
                ),
                rationale=(
                    f"BERT ensemble member load attempt: "
                    f"name={member.get('name')!r}, "
                    f"revision={member.get('revision')!r}, "
                    f"cache_dir={self.cache_dir!s}, "
                    f"success={success}"
                ),
                metadata={
                    "member_name": member.get("name"),
                    "member_revision": member.get("revision"),
                    "success": success,
                },
            )
        except Exception as exc:  # noqa: BLE001 — capture must never fail the load
            logger.debug(
                "DecisionCapture emit failed for bert_ensemble_member_loaded: %s",
                exc,
            )

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    def _aggregate(
        self,
        per_member: List[Tuple[str, float]],
        temperature: Optional[Any] = None,
    ) -> Tuple[str, float, float]:
        """Aggregate per-member votes into ``(winner_level, winner_score, dispersion)``.

        Score per Bloom level = sum of confidence values across every
        member that voted that level. Winner = ``argmax`` over the
        per-level scores. Dispersion = Shannon entropy of the
        normalised per-level scores (base ``e``, normalised by
        ``ln(num_levels_with_votes)`` so a uniform vote returns
        ``1.0`` regardless of how many distinct levels were voted).

        Returns ``("unknown", 0.0, 0.0)`` on an empty input list so
        the caller's contract stays single-shape regardless of member
        availability.

        Tie-breaking: when two levels accumulate identical scores, the
        winner is the lexicographically-first level (e.g. ``analyze``
        beats ``apply``). Deterministic so the validator's regression
        suite stays stable across re-runs.

        ``temperature`` accepts:

        - ``None`` (default): no temperature applied.
        - ``float`` (scalar): every per-member confidence is sharpened
          (``T < 1``) or softened (``T > 1``) by raising it to power
          ``1 / T`` before summation.
        - ``List[float]`` (per-member): the i-th value applies to the
          i-th per-member vote, in the order ``per_member`` was passed
          (which mirrors registry order). Lists shorter than
          ``len(per_member)`` are padded with ``1.0`` (no-op);
          longer lists are truncated.

        Temperature scaling sharpens or softens confident votes
        relative to less confident ones. The calibration script tunes
        per-member ``T`` to minimise expected calibration error (ECE)
        on the holdout corpus, so a poorly-calibrated member (e.g.
        SST-2 over-confidently mapping POSITIVE → ``evaluate``) has
        its votes softened before they sway the winner.
        """
        if not per_member:
            return ("unknown", 0.0, 0.0)

        # Resolve per-member temperatures into a list of floats. We
        # apply temperature scaling on a single-class confidence by
        # raising the scalar confidence to the power ``1 / T``: T < 1
        # sharpens (raises high-conf, depresses low-conf), T > 1
        # softens. T == 1 (or None) leaves the confidence untouched.
        temps: List[float] = []
        if temperature is None:
            temps = [1.0] * len(per_member)
        elif isinstance(temperature, (int, float)):
            t = float(temperature)
            temps = [t if t > 0 else 1.0] * len(per_member)
        elif isinstance(temperature, (list, tuple)):
            for i in range(len(per_member)):
                if i < len(temperature):
                    raw = temperature[i]
                    try:
                        t = float(raw)
                    except (TypeError, ValueError):
                        t = 1.0
                    temps.append(t if t > 0 else 1.0)
                else:
                    temps.append(1.0)
        else:
            temps = [1.0] * len(per_member)

        # Sum (temperature-adjusted) confidences per level.
        level_scores: Dict[str, float] = {}
        for (level, conf), t in zip(per_member, temps):
            try:
                base = float(conf)
            except (TypeError, ValueError):
                base = 0.0
            if base <= 0.0:
                adjusted = 0.0
            elif t == 1.0:
                adjusted = base
            else:
                # ``base`` lives in (0, 1]; raising to ``1/T`` keeps
                # the result in (0, 1]. Guard against numerical issues
                # by clamping.
                try:
                    adjusted = base ** (1.0 / t)
                except (OverflowError, ValueError):
                    adjusted = base
            level_scores[level] = level_scores.get(level, 0.0) + adjusted

        # Winner = argmax level. Sort by (-score, level) so ties resolve
        # lexicographically rather than by Python dict insertion order.
        sorted_items = sorted(
            level_scores.items(), key=lambda kv: (-kv[1], kv[0])
        )
        winner_level, winner_raw = sorted_items[0]
        total_score = sum(level_scores.values())
        winner_score = (
            round(winner_raw / total_score, 4) if total_score > 0 else 0.0
        )

        # Dispersion = normalised Shannon entropy of the per-level
        # score distribution. ``num_levels = len(level_scores)`` so a
        # 2-way uniform split returns 1.0 just like a 6-way uniform
        # split. Single-level votes have entropy 0 (perfect consensus).
        num_levels = len(level_scores)
        if num_levels <= 1 or total_score <= 0:
            dispersion = 0.0
        else:
            entropy = 0.0
            for score in level_scores.values():
                p = score / total_score
                if p > 0:
                    entropy -= p * math.log(p)
            # Normalise so uniform => 1.0 (max entropy = ln(num_levels)).
            dispersion = round(entropy / math.log(num_levels), 4)

        return (winner_level, winner_score, dispersion)

    # ------------------------------------------------------------------ #
    # Decision capture — wired by the validator
    # ------------------------------------------------------------------ #

    def attach_capture(self, capture: Any) -> None:
        """Wire a :class:`lib.decision_capture.DecisionCapture` instance."""
        self._capture = capture

    # ------------------------------------------------------------------ #
    # Temperature scaling
    # ------------------------------------------------------------------ #

    def set_temperature(self, temperature: Optional[Any]) -> None:
        """Set the per-member softmax-temperature override.

        Accepts ``None`` (no temperature), a scalar ``float`` (same
        ``T`` for every member), or a ``List[float]`` (per-member
        ``T`` in registry order). See :meth:`_aggregate` for the
        scaling semantics.

        The setter is idempotent and side-effect-free outside this
        instance — it only mutates ``self._temperature``. Calibration
        scripts and tests can flip the temperature, re-run
        :meth:`classify`, and observe the new aggregation outcome
        without rebuilding the ensemble.
        """
        self._temperature = temperature


__all__ = [
    "BertClassifier",
    "BertEnsembleDepsMissing",
    "BloomBertEnsemble",
    "is_strict_mode",
]
