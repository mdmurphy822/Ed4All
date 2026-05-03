"""Phase 4 Category D — BERT ensemble Bloom's-level classifier.

Subtask 24 of ``plans/phase4_statistical_tier_detailed.md`` lands the
skeleton: package structure, registry of three pre-resolved ensemble
members, ``BloomBertEnsemble`` class with the public ``classify``
contract, and stubs for ``_load_members`` (Subtask 25) +
``_aggregate`` (Subtask 26).

Three members per the pre-resolved Phase 4 plan decision:

1. ``kabir5297/bloom_taxonomy_classifier`` — purpose-built Bloom's
   classifier (6-class output natively aligned with the canonical
   ``BLOOM_LEVELS`` enum).
2. ``distilbert-base-uncased-finetuned-sst-2-english`` — sentiment
   model used as a generic confidence-anchor signal. Its raw labels
   (``POSITIVE`` / ``NEGATIVE``) get mapped onto Bloom levels via the
   :data:`_SST2_TO_BLOOM` heuristic table; the goal is dispersion
   contribution, not a high-confidence vote, so the table is
   intentionally low-resolution.
3. ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` — zero-shot NLI
   classifier; given a candidate text + the six Bloom-level labels as
   hypotheses, picks the highest-entailment level.

SHA pinning: each member's ``revision`` field carries a HuggingFace
git SHA so the ensemble's classification is reproducible across runs.
**Phase 4 followup**: The placeholder ``"main"`` revision below
resolves to whatever HEAD points at the time of model download. A
later commit should replace these with concrete commit SHAs (resolved
via ``huggingface_hub.HfApi().model_info(repo_id).sha`` against a
trusted pin), captured in the ``bert_ensemble_member_loaded`` decision
event so the audit trail records exactly which revision produced each
classification.

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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level config: ensemble member registry + Bloom-level mappings.
# ---------------------------------------------------------------------------

#: Default ensemble members. Each entry pins ``name`` (HuggingFace repo
#: id) and ``revision`` (git SHA — placeholder ``"main"`` documented
#: as a Phase 4 followup; see module docstring). Override the registry
#: by passing ``members=`` to :class:`BloomBertEnsemble.__init__`.
_DEFAULT_ENSEMBLE_MEMBERS: List[Dict[str, str]] = [
    {
        "name": "kabir5297/bloom_taxonomy_classifier",
        "revision": "main",
    },
    {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "revision": "main",
    },
    {
        "name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "revision": "main",
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


#: Heuristic mapping for the SST-2 sentiment member (member 2). Sentiment
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


#: Strict-mode env var (parallel of ``TRAINFORGE_REQUIRE_EMBEDDINGS``).
#: When truthy, missing ``transformers`` extras raise
#: :class:`BertEnsembleDepsMissing` instead of degrading silently.
_STRICT_MODE_ENV_VAR = "TRAINFORGE_REQUIRE_BERT_ENSEMBLE"
_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})


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
            except Exception as exc:  # noqa: BLE001 — silent-degrade per Subtask 24
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

        winner_level, winner_score, dispersion = self._aggregate(per_member)
        return {
            "winner_level": winner_level,
            "winner_score": winner_score,
            "dispersion": dispersion,
            "per_member": per_member,
        }

    # ------------------------------------------------------------------ #
    # Loader + per-member classification — implemented in Subtask 25
    # ------------------------------------------------------------------ #

    def _load_members(self) -> List[BertClassifier]:
        """Subtask 24 stub — returns ``[]`` until Subtask 25 wires the loader."""
        if self._loaded is not None:
            return self._loaded
        self._loaded = []
        return self._loaded

    def _classify_with_member(
        self, member: BertClassifier, text: str
    ) -> Tuple[str, float]:
        """Subtask 24 stub — returns deterministic (``"remember"``, 0.5).

        Real per-member dispatch (native Bloom softmax, SST-2 mapping,
        zero-shot NLI entailment) lands in Subtask 25/26.
        """
        return ("remember", 0.5)

    # ------------------------------------------------------------------ #
    # Aggregation — implemented in Subtask 26
    # ------------------------------------------------------------------ #

    def _aggregate(
        self, per_member: List[Tuple[str, float]]
    ) -> Tuple[str, float, float]:
        """Subtask 24 stub — returns sentinel ``("unknown", 0.0, 0.0)``.

        Real aggregation (per-level confidence sum + entropy dispersion)
        lands in Subtask 26.
        """
        return ("unknown", 0.0, 0.0)

    # ------------------------------------------------------------------ #
    # Decision capture — wired by the validator (Subtask 27)
    # ------------------------------------------------------------------ #

    def attach_capture(self, capture: Any) -> None:
        """Wire a :class:`lib.decision_capture.DecisionCapture` instance."""
        self._capture = capture


__all__ = [
    "BertClassifier",
    "BertEnsembleDepsMissing",
    "BloomBertEnsemble",
    "is_strict_mode",
]
