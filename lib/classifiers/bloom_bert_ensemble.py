"""Abstaining compatibility scaffold for Bloom-level classification.

No reliable Bloom classifier is currently provisioned for this surface, and
model-specific dispatch is not implemented. A default
:class:`BloomBertEnsemble` therefore loads no members and returns the explicit
``unknown`` sentinel rather than fabricating a classification. Strict mode
raises :class:`BertEnsembleDepsMissing` instead of accepting that abstention.

The three pinned registry entries and the cip29/SST-2 label mappings are
non-authoritative compatibility metadata. They are retained because other
code currently reads the DeBERTa NLI identity from registry index 2; they do
not describe a usable default ensemble. In particular, the cip29 checkpoint
was not reliable enough to serve as Ed4All's classifier, and the SST-2 mapping
is a sentiment heuristic rather than a Bloom model.

The separate opt-in trivote path uses a general-purpose DeBERTa NLI model as a
zero-shot heuristic. It is not a trained Ed4All Bloom classifier and does not
make this compatibility scaffold operational. The long-term MultiBERT training
path is staged but unproven; it becomes a usable classifier only after training,
evaluation, and explicit provisioning.
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

#: Historical compatibility registry. Each entry pins ``name`` (HuggingFace
#: repo id) and ``revision`` (concrete 40-hex-char git commit SHA). It is not a
#: provisioned or validated default classifier, and production dispatch does
#: not load these members. Override the metadata by passing ``members=`` to
#: :class:`BloomBertEnsemble.__init__`.
#:
#: Compatibility behavior:
#:    The first two entries are non-authoritative retired metadata.
#:    ``cip29/bert-blooms-taxonomy-classifier``
#:    (registry index 0) carries no stated license and proved unreliable — see
#:    ``docs/LICENSING.md`` — and
#:    ``distilbert-base-uncased-finetuned-sst-2-english`` (registry index 1) is a
#:    sentiment model whose Bloom mapping is only a low-resolution heuristic.
#:    Neither is loaded by the current scaffold. The opt-in trivote path is a
#:    separate asserted-label / NLI-heuristic / verb-ontology check; it does
#:    not activate this registry. See :mod:`lib.classifiers.bloom_zero_shot`.
#:    This list remains unchanged because :meth:`lib.classifiers.nli_classifier.
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


#: Non-authoritative compatibility mapping for the retired SST-2 registry
#: entry. Sentiment carries no direct Bloom signal, and current production
#: dispatch never consumes this table.
_SST2_TO_BLOOM: Dict[str, str] = {
    "POSITIVE": "evaluate",
    "NEGATIVE": "remember",
}


#: Non-authoritative compatibility mapping for the retired cip29 registry
#: entry. The checkpoint exposes generic ``LABEL_0`` ... ``LABEL_5`` names,
#: and this table records the former assumed ordering. That assumption was not
#: reliable enough for production classification; current dispatch never
#: consumes this table.
_CIP29_TO_BLOOM: Dict[str, str] = {
    "LABEL_0": "remember",
    "LABEL_1": "understand",
    "LABEL_2": "apply",
    "LABEL_3": "analyze",
    "LABEL_4": "evaluate",
    "LABEL_5": "create",
}


#: Strict-mode env var. When truthy, unavailable dependencies or the
#: unimplemented classifier dispatch raise :class:`BertEnsembleDepsMissing`
#: instead of returning the ``unknown`` sentinel.
_STRICT_MODE_ENV_VAR = "TRAINFORGE_REQUIRE_BERT_ENSEMBLE"
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})

#: Module-level latch so the unavailable-dispatch warning fires exactly once
#: per process regardless of how many compatibility wrappers are constructed.
_UNIMPLEMENTED_DISPATCH_WARNED = False


#: Compatibility cache path used only by the dormant per-member loader.
#: Current production dispatch does not download or load ensemble weights.
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ed4all" / "bert_ensemble"


def is_strict_mode() -> bool:
    """Return True when ``TRAINFORGE_REQUIRE_BERT_ENSEMBLE`` is truthy."""
    raw = os.environ.get(_STRICT_MODE_ENV_VAR, "").strip().lower()
    return raw in _TRUTHY_VALUES


class BertEnsembleDepsMissing(RuntimeError):
    """Raised when strict mode cannot provide a usable ensemble.

    Strict mode is opt-in via ``TRAINFORGE_REQUIRE_BERT_ENSEMBLE=true``.
    This includes missing dependencies and the current unimplemented
    model-specific dispatch. Default mode returns an empty member list so
    downstream consumers can abstain.
    """


@dataclass
class BertClassifier:
    """Container for a member used by custom or future dispatch code.

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
    """Compatibility wrapper that currently abstains from classification.

    Public contract:

    .. code-block:: python

        ensemble = BloomBertEnsemble()
        result = ensemble.classify("<task text>")
        # {
        #     "winner_level": "unknown",
        #     "winner_score": 0.0,
        #     "dispersion": 0.0,
        #     "per_member": [],
        # }

    No reliable classifier is provisioned and model-specific dispatch is not
    implemented, so the production loader returns no members. The empty list
    produces a sentinel result
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
        """Return a Bloom vote when a subclass supplies members, else abstain.

        The default scaffold returns ``unknown`` with an empty ``per_member``
        list. The aggregation shape remains available to injected test or
        future implementations. Returns four keys: ``winner_level`` (str),
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
        """Return no production members while dispatch remains unavailable.

        Uses a one-shot import probe of ``transformers``: when the
        extras are absent and strict mode is off, returns ``[]`` and
        downstream callers degrade to a warning-severity GateIssue.
        When strict mode is on (``TRAINFORGE_REQUIRE_BERT_ENSEMBLE=true``),
        raises :class:`BertEnsembleDepsMissing` with an operator-actionable
        install hint.

        Even when ``transformers`` is importable, the method returns ``[]``
        because no reliable Ed4All Bloom classifier is provisioned and the
        compatibility registry has no implemented model-specific dispatch.
        Strict mode raises instead. The dormant per-member loader below is not
        reached by this production path.
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
        # The staged MultiBERT path requires training, validation, and explicit
        # provisioning before it can implement this contract. The retired
        # cip29 and SST-2 mappings are not production substitutes. The separate
        # DeBERTa zero-shot heuristic remains outside this scaffold.
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
        """Dormant compatibility loader for a SHA-pinned registry member.

        The production :meth:`_load_members` path never calls this method.
        Custom callers receive ``None`` on any load failure.
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

        Model-specific dispatch is not implemented. The retired cip29 and
        SST-2 metadata is non-authoritative, and the separate NLI heuristic is
        not an ensemble member implementation.
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
