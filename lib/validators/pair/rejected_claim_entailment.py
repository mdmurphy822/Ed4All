"""WI-16 — :class:`RejectedClaimEntailmentValidator`.

Closes a hole in the claim-support family of gates
(:class:`lib.validators.claim_support.ClaimSupportValidator`,
:class:`lib.validators.pair.claim_support.PairClaimSupportValidator`):
both score the *chosen* / completion surface against the cited source
text, and neither ever looks at a preference pair's ``rejected`` side.

For a ``source="misconception"`` preference pair, ``rejected`` is
supposed to be a WRONG claim — the authored misconception statement —
so a DPO run can teach the model to prefer ``chosen`` over it. But the
authoring path (:func:`Trainforge.generators.pairs.preference.
synthesize_preference_pair`) never re-checks the misconception
statement against the source chunk it was drawn from. When the
"wrong" framing turns out to be entailed by the source window anyway
(an authoring slip, a misconception whose claim is true in the
chunk's actual context, or a stale/edited chunk), the pair's DPO
signal silently points a trained model AWAY from a statement the
source itself asserts. Nothing upstream catches this — the claim
support gates never read ``rejected``, and
:class:`lib.validators.pair.promotion.
TrainingPairPromotionValidator` only checks that ``chosen`` /
``rejected`` are sufficiently distinct, never their individual
truth-value against the source.

Single surface — a gate-runner :meth:`validate`. There is no
per-pair pre-write filter counterpart (unlike the sibling W4.A / W4.B
/ W4.C pair validators): gate wiring into ``config/workflows.yaml`` +
``GateInputRouter`` lands separately.

Algorithm:

1. Walk ``training_specs/preference_pairs.jsonl`` (instruction pairs
   have no ``rejected`` side and are skipped by construction).
2. Filter to ``pair["source"] == "misconception"``.
3. Resolve each pair's "source window" — the cited chunk's ``text`` —
   from a ``chunk_id -> text`` map built once from
   ``imscc_chunks/chunks.jsonl`` (legacy ``chunks.jsonl`` tolerated via
   :func:`lib.libv2_storage.resolve_imscc_chunks_path`).
4. Score ``nli.score_pair(premise=chunk_text, hypothesis=rejected)``
   via the resident :class:`lib.classifiers.nli_classifier.
   NliClassifier` singleton (``get_or_load`` — no second model load;
   the same in-memory instance the claim-support gates already paid
   the load cost for).
5. ``entailment >= entailment_floor`` (default 0.70, the same bucket
   floor :class:`NliScore` documents for "entailed") flags
   :data:`_CODE_REJECTED_ENTAILED_BY_SOURCE` — a true statement shipped
   as the rejected completion.

Both premise and hypothesis are normalized (typography + math
notation) before scoring via the same helper the claim-support gates
use, so this validator can't be defeated by the exact quote/notation
mismatch that motivated that normalization there.

Warning-severity, always ``passed=True`` — day-1 calibration posture,
never blocks a build. Graceful-degrade contract mirrors the other NLI
validators on this contract (see ``lib/CLAUDE.md`` § "Graceful-degrade
contract"): missing ``transformers`` / ``torch`` extras (or a failed
model load) emits a single warning :data:`_CODE_NLI_DEPS_MISSING`
issue with ``passed=True``, unless ``TRAINFORGE_REQUIRE_EMBEDDINGS`` is
set, in which case the validator raises instead of degrading.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier
from lib.embedding.sentence_embedder import is_strict_mode
from lib.semantik.math_fold import normalize_math_notation

logger = logging.getLogger(__name__)


# Same bucket floor NliScore documents for "entailed" (0.70) — kept as
# a local constant rather than importing the claim-support thresholds
# module so this file has no coupling to that sibling's internals.
_DEFAULT_ENTAILMENT_FLOOR: float = 0.70

_CODE_NLI_DEPS_MISSING: str = "NLI_DEPS_MISSING"
_CODE_REJECTED_ENTAILED_BY_SOURCE: str = "REJECTED_ENTAILED_BY_SOURCE"
_CODE_NO_CHUNKS: str = "REJECTED_ENTAILMENT_NO_CHUNKS"
_CODE_INVALID_CHUNKS: str = "REJECTED_ENTAILMENT_INVALID_CHUNKS"
_CODE_READ_ERROR: str = "REJECTED_ENTAILMENT_READ_ERROR"

_ISSUE_CAP: int = 50


_NLI_TYPOGRAPHY_TRANSLATION = str.maketrans({
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    " ": " ",
})


def _normalize_nli_text(text: str) -> str:
    """Canonicalize typography and math notation before NLI scoring.

    Mirrors :func:`lib.validators.pair.claim_support._normalize_nli_text`
    verbatim (typography fold + :func:`normalize_math_notation`,
    applied to BOTH premise and hypothesis). Duplicated locally rather
    than imported so this validator has no coupling to a sibling
    module's internals; the fix it encodes (a source line and a
    generator's near-verbatim restatement differing only in quote
    style or ``x^2`` vs ``x²`` scoring on opposite sides of the
    entailment floor) applies here exactly as it does there — the
    rejected side is prose drawn from ``chunk.misconceptions`` /
    authored against the chunk, so it carries the same notation risk
    as the chosen side.
    """
    return normalize_math_notation(
        str(text).translate(_NLI_TYPOGRAPHY_TRANSLATION)
    )


class RejectedClaimEntailmentValidator:
    """Flags ``source="misconception"`` preference pairs whose
    ``rejected`` completion is entailed by its cited source chunk.

    Warning-severity, never blocks (``passed`` is always ``True``).
    """

    name = "rejected_claim_entailment"
    version = "1.0.0"

    def __init__(
        self,
        *,
        entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
        nli: Optional[Any] = None,
    ) -> None:
        self._entailment_floor = float(entailment_floor)
        self._nli_override = nli

    def _get_nli(self) -> Optional[Any]:
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    # ------------------------------------------------------------------ #
    # Gate-runner surface
    # ------------------------------------------------------------------ #

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Walk ``preference_pairs.jsonl``; NLI-score every
        ``source="misconception"`` pair's ``rejected`` side against its
        cited chunk's text.

        Inputs:

        - ``preference_pairs_path`` (str) — explicit path. Optional
          when ``course_dir`` / ``training_specs_dir`` is provided.
        - ``course_dir`` (str) — resolves
          ``<course_dir>/training_specs/preference_pairs.jsonl`` and
          (via :func:`lib.libv2_storage.resolve_imscc_chunks_path`)
          the chunks file.
        - ``training_specs_dir`` (str) — sibling alternative for the
          pairs path only.
        - ``chunks_path`` (str) — explicit override for the chunks
          file. Takes precedence over ``course_dir`` resolution.
        - ``gate_id`` (str) — optional gate-id override for the result.

        Outputs: always ``passed=True``. ``score`` is
        ``(scored - flagged) / scored`` over misconception pairs whose
        source chunk actually resolved (``1.0`` when none were
        scoreable). ``issues`` carries one warning
        :data:`_CODE_REJECTED_ENTAILED_BY_SOURCE` per flagged pair
        (capped at :data:`_ISSUE_CAP`), plus at most one graceful-
        degrade / infra-issue entry when chunks or NLI deps are
        unavailable.
        """
        gate_id = inputs.get("gate_id", self.name)

        pref_path = self._resolve_preference_pairs_path(inputs)
        if pref_path is None or not pref_path.exists():
            # Nothing emitted yet (or this course generates no
            # preference pairs) — no-op pass.
            return self._pass(gate_id, score=1.0)

        misconception_pairs, read_error = self._collect_misconception_pairs(
            pref_path
        )
        if read_error is not None:
            return self._pass(
                gate_id,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code=_CODE_READ_ERROR,
                    message=f"Failed to read {pref_path}: {read_error}",
                    location=str(pref_path),
                )],
            )

        if not misconception_pairs:
            # No source='misconception' pairs in this corpus — the
            # finding this gate exists to catch structurally cannot
            # occur (a rule_synthesized rejected side is a negation
            # transform of `chosen`, not an authored claim being
            # checked for truth).
            return self._pass(gate_id, score=1.0)

        chunks_path = self._resolve_chunks_path(inputs)
        if chunks_path is None or not chunks_path.exists():
            return self._pass(
                gate_id,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code=_CODE_NO_CHUNKS,
                    message=(
                        f"{len(misconception_pairs)} misconception-sourced "
                        "preference pair(s) found but no chunks.jsonl could "
                        "be resolved to recover their source window; "
                        "rejected-side entailment check skipped."
                    ),
                    location=str(pref_path),
                )],
            )

        chunk_text_map = self._collect_chunk_text_map(chunks_path)
        if chunk_text_map is None:
            return self._pass(
                gate_id,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code=_CODE_INVALID_CHUNKS,
                    message=f"Failed to parse chunks JSONL: {chunks_path}",
                    location=str(chunks_path),
                )],
            )

        nli = self._get_nli()
        if nli is None:
            if is_strict_mode():
                raise RuntimeError(
                    "RejectedClaimEntailmentValidator requires the NLI "
                    "classifier (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) "
                    "but TRAINFORGE_REQUIRE_EMBEDDINGS is set and "
                    "NliClassifier.get_or_load() returned None. Install "
                    "transformers + torch via `pip install -e .[embedding]` "
                    "or unset TRAINFORGE_REQUIRE_EMBEDDINGS for graceful "
                    "degradation."
                )
            return self._pass(
                gate_id,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code=_CODE_NLI_DEPS_MISSING,
                    message=(
                        "NliClassifier.get_or_load() returned None; "
                        "rejected-side entailment check skipped for "
                        f"{len(misconception_pairs)} misconception-sourced "
                        "pair(s). Install transformers + torch via "
                        "`pip install -e .[embedding]` to enable. Set "
                        "TRAINFORGE_REQUIRE_EMBEDDINGS=true to fail closed "
                        "instead."
                    ),
                    suggestion=(
                        "The graceful-degrade path is intentional for "
                        "CPU-only dev boxes; production runs should "
                        "install the embedding extras."
                    ),
                )],
            )

        issues: List[GateIssue] = []
        scored = 0
        flagged = 0
        for line_num, pair in misconception_pairs:
            chunk_id = str(pair.get("chunk_id") or "").strip()
            chunk_text = chunk_text_map.get(chunk_id, "")
            rejected_text = str(pair.get("rejected") or "").strip()
            if not chunk_text or not rejected_text:
                # Source window unresolvable, or malformed pair — other
                # gates (pair_lo_refs' chunk-not-found check, schema
                # validation) already own that failure mode; this gate
                # simply can't score it.
                continue
            scored += 1
            nli_score = nli.score_pair(
                premise=_normalize_nli_text(chunk_text),
                hypothesis=_normalize_nli_text(rejected_text),
            )
            if nli_score.entailment >= self._entailment_floor:
                flagged += 1
                if len(issues) < _ISSUE_CAP:
                    pair_id = str(
                        pair.get("pair_id")
                        or pair.get("id")
                        or f"{pref_path.name}:{line_num}"
                    )
                    issues.append(GateIssue(
                        severity="warning",
                        code=_CODE_REJECTED_ENTAILED_BY_SOURCE,
                        message=(
                            f"{pref_path.name}:{line_num} preference pair "
                            f"{pair_id!r} (misconception_id="
                            f"{pair.get('misconception_id')!r}) cites chunk "
                            f"{chunk_id!r}; its rejected completion is "
                            f"ENTAILED by that chunk's text "
                            f"(entailment={nli_score.entailment:.3f} >= "
                            f"{self._entailment_floor}, contradiction="
                            f"{nli_score.contradiction:.3f}). The "
                            "misconception-labeled 'wrong' answer reads as "
                            "a true statement per its own source."
                        ),
                        location=str(pref_path),
                        suggestion=(
                            "Review the cited misconception statement "
                            "against the source chunk; either the "
                            "authored misconception is contextually "
                            "accurate (drop or rewrite it) or the source "
                            "chunk has drifted since authoring."
                        ),
                    ))

        score_value = 1.0 if scored == 0 else round((scored - flagged) / scored, 4)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=score_value,
            issues=issues,
            action=None,
            metadata={
                "misconception_pairs": len(misconception_pairs),
                "scored": scored,
                "flagged_entailed_rejected": flagged,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _pass(
        self,
        gate_id: str,
        *,
        score: float,
        issues: Optional[List[GateIssue]] = None,
    ) -> GateResult:
        """Always-``passed=True`` result — this gate never blocks."""
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=score,
            issues=issues or [],
            action=None,
        )

    @staticmethod
    def _resolve_preference_pairs_path(
        inputs: Dict[str, Any],
    ) -> Optional[Path]:
        explicit = inputs.get("preference_pairs_path")
        if isinstance(explicit, str) and explicit:
            return Path(explicit)
        course_dir_raw = inputs.get("course_dir")
        if course_dir_raw:
            return Path(course_dir_raw) / "training_specs" / "preference_pairs.jsonl"
        training_specs_dir_raw = inputs.get("training_specs_dir")
        if training_specs_dir_raw:
            return Path(training_specs_dir_raw) / "preference_pairs.jsonl"
        return None

    @staticmethod
    def _resolve_chunks_path(inputs: Dict[str, Any]) -> Optional[Path]:
        explicit = inputs.get("chunks_path")
        if isinstance(explicit, str) and explicit:
            return Path(explicit)
        course_dir_raw = inputs.get("course_dir")
        if course_dir_raw:
            try:
                from lib.libv2_storage import resolve_imscc_chunks_path
                return resolve_imscc_chunks_path(
                    Path(course_dir_raw), "chunks.jsonl"
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "resolve_imscc_chunks_path raised: %s", exc
                )
        return None

    @staticmethod
    def _collect_misconception_pairs(
        pref_path: Path,
    ) -> "tuple[List[Any], Optional[str]]":
        """Return ``(list of (line_num, pair) for source='misconception'
        pairs, read_error_or_None)``."""
        out: List[Any] = []
        try:
            with pref_path.open("r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pair = json.loads(line)
                    except json.JSONDecodeError:
                        # Soft-skip malformed lines; upstream emit gates
                        # should already filter these.
                        continue
                    if not isinstance(pair, dict):
                        continue
                    if str(pair.get("source") or "") != "misconception":
                        continue
                    out.append((line_num, pair))
        except (OSError, UnicodeDecodeError) as exc:
            return [], str(exc)
        return out, None

    @staticmethod
    def _collect_chunk_text_map(
        chunks_path: Path,
    ) -> Optional[Dict[str, str]]:
        """Build ``chunk_id -> text`` map from ``chunks.jsonl``.

        Returns ``None`` on read/parse error. Missing per-chunk text
        maps to ``""`` (falls through the caller's "unresolvable"
        check) rather than dropping the chunk_id entirely.
        """
        out: Dict[str, str] = {}
        try:
            with chunks_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    chunk_id_raw = chunk.get("chunk_id") or chunk.get("id")
                    if not isinstance(chunk_id_raw, str):
                        continue
                    chunk_id = chunk_id_raw.strip()
                    if not chunk_id:
                        continue
                    out[chunk_id] = str(chunk.get("text") or "")
        except (OSError, UnicodeDecodeError):
            return None
        return out


__all__ = [
    "RejectedClaimEntailmentValidator",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_REJECTED_ENTAILED_BY_SOURCE",
    "_CODE_NO_CHUNKS",
    "_CODE_INVALID_CHUNKS",
    "_CODE_READ_ERROR",
]
