"""Worker W3a — distractor / misconception alignment validator.

Closes the §3.C distractor-quality gap GPT cited in the W2-W7 review:
the W7 ``BlockAssessmentItemPayloadValidator`` (sibling at
``lib/validators/assessment_item_payload.py``) only validates the
*shape* of a distractor's ``misconception_ref`` slot — that the value
matches ``^[A-Z]{2,}-\\d{2,}#m\\d+$``. It does not check that the
declared ref *resolves* against the course's misconception inventory,
nor that the distractor's text is *actually about* the referenced
misconception.

This validator gates both axes for every distractor whose
``misconception_ref`` is populated:

  1. **Resolution**: the ref must resolve against the supplied
     misconception inventory (``inputs["misconceptions"]`` map or
     ``inputs["chunks"]``-derived map). Unresolved → critical
     ``DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED``.
  2. **Alignment**: when an embedder is available, cosine similarity
     between the distractor text surface and the resolved misconception
     statement must be ≥ ``min_cosine`` (default 0.45). When the
     ``[embedding]`` extras are absent, the validator falls back to a
     CPU-only Jaccard token-overlap floor (default 0.05) so the gate
     remains useful on slim installs.

A distractor *without* a ``misconception_ref`` is out-of-scope here
(W3b's ``DistractorPlausibilityValidator`` covers the no-ref
plausibility axis).

GateIssue codes (all ``critical`` unless noted):

- ``DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED`` — ref doesn't appear in
  the supplied misconception inventory.
- ``DISTRACTOR_MISCONCEPTION_MISALIGNED`` — distractor text /
  misconception text similarity below threshold (cosine when embedder
  loaded, Jaccard when not).
- ``EMBEDDING_DEPS_MISSING`` (warning) — sentence-transformers extras
  unavailable; falls through to Jaccard. Mirrors the Phase 4
  statistical-tier convention but does NOT short-circuit the gate
  (Jaccard is always available).

Outline-tier failure ``action="regenerate"`` so the rewrite tier sees
a valid draft. Rewrite-tier and assessment-tier wirings should pass
``action_on_fail="block"`` via input override (per W3a's three-gate
plan); the validator defaults to ``regenerate`` to match the most
common (outline-tier) wiring path.

References:

- ``lib/validators/concept_example_similarity.py`` — embedding-gated
  fallback pattern this validator mirrors.
- ``lib/validators/assessment_item_payload.py`` — W7 sibling that
  validates the ``misconception_ref`` *shape* this validator
  *resolves*.
- ``schemas/knowledge/chunk_v4.schema.json::$defs.Misconception`` —
  the canonical misconception payload shape this validator consumes
  from chunk corpora.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

# Lazy import the embedder module so a slim install (no
# sentence-transformers extras) doesn't pay the import cost on every
# orchestrator boot. The validator references the module attributes
# at validate() time, which lets monkeypatch-based tests substitute
# ``try_load_embedder``.
from lib.embedding import sentence_embedder as _sentence_embedder_mod
from lib.embedding._math import cosine_similarity

# ``blocks.py`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# import bridge used by the sibling validators so ``Block`` resolves
# regardless of how this module is loaded.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover — import-bridge tested via the test suite
    from blocks import Block  # type: ignore[import-not-found]  # noqa: E402
except Exception:  # noqa: BLE001
    Block = None  # type: ignore[assignment,misc]


# Default cosine-similarity floor below which a (distractor, referenced
# misconception) pair is flagged as misaligned. 0.45 is calibrated
# against the all-MiniLM-L6-v2 model's intrinsic similarity floor:
# distractors *intentionally* paraphrase a misconception (they must
# *embody* it to be a realistic wrong answer), so well-aligned pairs
# cluster above 0.55 and outright mismatches cluster below 0.30. The
# 0.45 floor leaves headroom for legitimate paraphrase variation while
# catching distractors that drifted onto an unrelated topic.
DEFAULT_MIN_COSINE: float = 0.45

# Default Jaccard token-overlap floor for the embedder-missing
# fallback. Strictly looser than the cosine floor because Jaccard is
# bag-of-words (no synonym handling); a low floor admits paraphrases
# while still catching wholly unrelated text. Calibrated against the
# same misalignment fixtures the cosine floor was tuned on.
DEFAULT_MIN_JACCARD: float = 0.05

# Cap on issues per validate() invocation so a uniformly broken batch
# doesn't drown the gate report. Mirrors the cap in
# ``Courseforge/router/inter_tier_gates.py::_ISSUE_LIST_CAP``.
_ISSUE_LIST_CAP: int = 50

# Misconception CURIE pattern (mirrors the W7 sibling +
# ``$defs.AssessmentItem`` in
# ``schemas/knowledge/courseforge_jsonld_v1.schema.json``).
_MISCONCEPTION_REF_RE = re.compile(r"^[A-Z]{2,}-\d{2,}#m\d+$")

# Lightweight HTML strip used on rewrite-tier (str-content) blocks.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Bag-of-words tokeniser for the Jaccard fallback. Lowercase, alpha-
# numeric, length≥2 to drop punctuation noise. Intentionally cheap.
_JACCARD_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,}")


def _strip_html(html: str) -> str:
    """Strip tags + collapse whitespace (mirrors the W7 helper)."""
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _tokenise(text: str) -> Set[str]:
    """Lowercase token-set for Jaccard overlap. Stripped of punctuation."""
    if not text:
        return set()
    return set(_JACCARD_TOKEN_RE.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    """Jaccard token-overlap between two strings. Symmetric, in [0, 1]."""
    ta = _tokenise(a)
    tb = _tokenise(b)
    if not ta or not tb:
        return 0.0
    union = ta | tb
    inter = ta & tb
    return len(inter) / len(union)


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs['blocks']``.

    Mirrors the sibling-validator helper. Returns ``(blocks,
    error_issue)``; non-None ``error_issue`` is wrapped into a
    ``passed=False, action="regenerate"`` GateResult by the caller.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _build_misconception_index(
    inputs: Dict[str, Any],
) -> Dict[str, str]:
    """Return a ``{ref → misconception_text}`` lookup map.

    Resolution priority:
      1. ``inputs["misconceptions"]`` — a ``Dict[str, str]`` mapping
         the canonical CURIE ref (e.g. ``"TO-01#m1"``) to the
         misconception statement. The input builder
         ``_build_distractor_misconception_input`` (W3a's plan
         §coordination commit) populates this from the assembled
         course-misconception inventory.
      2. ``inputs["chunks"]`` — a list of chunk dicts conforming to
         ``schemas/knowledge/chunk_v4.schema.json``. The validator
         walks each chunk's ``misconceptions[]`` array and synthesises
         refs of the form ``{learning_outcome_ref}#m{N}`` where N is
         the 1-based index inside the chunk's misconceptions list,
         per the canonical encoding scheme used by the Wave 60
         Courseforge emitter. Plain-string entries (the Misconception
         oneOf branch) are tokenised but not indexed (no canonical
         ref).

    Both inputs may be supplied; the explicit map wins on conflict
    (single source of truth for assessment-tier inventories).
    """
    explicit = inputs.get("misconceptions") or {}
    if not isinstance(explicit, dict):
        explicit = {}
    # Defensive copy so we don't mutate caller state.
    index: Dict[str, str] = {
        str(k): str(v)
        for k, v in explicit.items()
        if isinstance(k, str) and isinstance(v, str) and v.strip()
    }

    chunks = inputs.get("chunks") or []
    if not isinstance(chunks, list):
        return index

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        misconceptions = chunk.get("misconceptions") or []
        if not isinstance(misconceptions, list):
            continue
        # Canonical Wave 60 encoding pairs chunk.misconceptions[i]
        # with the i+1-th misconception in the LO's ordered list.
        # We synthesise refs against every learning_outcome_ref the
        # chunk cites so a distractor referencing any of the chunk's
        # LOs resolves cleanly. This intentionally over-indexes (one
        # misconception → multiple refs) to keep the resolution path
        # forgiving; the canonical inventory map (priority 1) is the
        # single source of truth when precision matters.
        lo_refs = chunk.get("learning_outcome_refs") or []
        if not isinstance(lo_refs, list):
            lo_refs = []
        for idx, entry in enumerate(misconceptions, start=1):
            text: Optional[str] = None
            if isinstance(entry, dict):
                # Wave 74+ accepts "statement" as a synonym for
                # "misconception" per the chunk_v4 schema's anyOf.
                text = entry.get("misconception") or entry.get("statement")
            elif isinstance(entry, str):
                text = entry
            if not isinstance(text, str) or not text.strip():
                continue
            for lo_ref in lo_refs:
                if not isinstance(lo_ref, str) or not lo_ref:
                    continue
                synth_ref = f"{lo_ref}#m{idx}"
                index.setdefault(synth_ref, text.strip())
    return index


def _extract_distractors_outline(block: Any) -> List[Tuple[int, str, Optional[str]]]:
    """Return a list of ``(idx, text, misconception_ref)`` tuples for an
    outline-tier (dict-content) ``assessment_item`` block.

    A dict ``distractors[i]`` entry MUST carry a ``text`` field per
    the W7 sibling's contract; entries missing ``text`` are skipped
    (W7 will already have flagged them with
    ``ASSESSMENT_ITEM_DISTRACTOR_TEXT_MISSING``).
    """
    content = block.content
    if not isinstance(content, dict):
        return []
    distractors = content.get("distractors")
    if not isinstance(distractors, list):
        return []
    out: List[Tuple[int, str, Optional[str]]] = []
    for idx, entry in enumerate(distractors):
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        ref = entry.get("misconception_ref")
        if isinstance(ref, str) and ref.strip():
            out.append((idx, text.strip(), ref.strip()))
        else:
            out.append((idx, text.strip(), None))
    return out


def _emit_decision(
    capture: Any,
    *,
    block_id: str,
    audited_count: int,
    resolved_count: int,
    misaligned_count: int,
    unresolved_count: int,
    skipped_no_ref_count: int,
    threshold: float,
    metric: str,
    embedder_loaded: bool,
) -> None:
    """Emit one ``distractor_misconception_alignment_check`` decision per
    ``validate()`` invocation.

    Rationale interpolates dynamic signals so captures are replayable
    post-hoc: the audited / resolved / misaligned / unresolved /
    skipped counts, the active threshold, the similarity metric used
    (cosine vs. jaccard), and whether the embedder loaded (so silent-
    degrade events are distinguishable from real misalignment events).
    """
    if capture is None:
        return
    decision = "passed" if (misaligned_count == 0 and unresolved_count == 0) else "failed"
    rationale = (
        f"distractor_misconception_alignment on Block {block_id!r}: "
        f"audited_distractors_with_ref={audited_count}, "
        f"resolved={resolved_count}, "
        f"misaligned={misaligned_count}, "
        f"unresolved_refs={unresolved_count}, "
        f"skipped_no_ref={skipped_no_ref_count}, "
        f"metric={metric}, "
        f"threshold={threshold:.4f}, "
        f"embedder_loaded={embedder_loaded}."
    )
    try:
        capture.log_decision(
            decision_type="distractor_misconception_alignment_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — degrade quietly on emit errors.
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "distractor_misconception_alignment_check: %s",
            exc,
        )


class DistractorMisconceptionAlignmentValidator:
    """Resolve + semantically-align each distractor's ``misconception_ref``.

    Walks ``inputs['blocks']``, filters to ``block.block_type ==
    "assessment_item"``, then for each distractor with a populated
    ``misconception_ref``:

      1. Looks the ref up against the supplied misconception inventory
         (``inputs["misconceptions"]`` and/or chunk-derived). Unresolved
         refs emit ``DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED``.
      2. Computes alignment between the distractor text and the
         resolved misconception statement. When the ``[embedding]``
         extras are loaded, alignment is cosine similarity; otherwise
         the validator falls back to Jaccard token-overlap. Below the
         applicable threshold emits
         ``DISTRACTOR_MISCONCEPTION_MISALIGNED``.

    Distractors WITHOUT a ``misconception_ref`` are skipped — that's
    W3b's surface (``DistractorPlausibilityValidator``).

    Mirrors the validate() shape of the four sibling Block validators
    in ``Courseforge/router/inter_tier_gates.py`` so the existing
    ``_build_block_input_outline`` / ``_build_block_input_rewrite``
    builders in ``MCP/hardening/gate_input_routing.py`` (extended by
    the W3a plan's coordination commit) wire it cleanly.
    """

    name = "distractor_misconception_alignment"
    version = "1.0.0"

    DEFAULT_MIN_COSINE = DEFAULT_MIN_COSINE
    DEFAULT_MIN_JACCARD = DEFAULT_MIN_JACCARD

    def __init__(
        self,
        *,
        min_cosine: float = DEFAULT_MIN_COSINE,
        min_jaccard: float = DEFAULT_MIN_JACCARD,
        embedder: Any = None,
    ) -> None:
        self._min_cosine = float(min_cosine)
        self._min_jaccard = float(min_jaccard)
        # Lazy-load on validate() so test injections take precedence
        # and a slim install doesn't pay the import cost at construction.
        self._embedder_override = embedder

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        min_cosine = float(inputs.get("min_cosine", self._min_cosine))
        min_jaccard = float(inputs.get("min_jaccard", self._min_jaccard))
        # Outline-tier wirings expect "regenerate"; rewrite + assessment
        # tiers can override to "block". Default tracks the most common
        # (outline-tier) wiring path.
        action_on_fail = inputs.get("action_on_fail", "regenerate")

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        assessments = [
            b for b in blocks
            if getattr(b, "block_type", None) == "assessment_item"
        ]
        if not assessments:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        misconception_index = _build_misconception_index(inputs)

        # Probe for embedder availability. self._embedder_override
        # wins (test injection); otherwise call try_load_embedder()
        # via the module attribute so tests can monkeypatch.
        embedder = self._embedder_override
        if embedder is None:
            try:
                embedder = _sentence_embedder_mod.try_load_embedder()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "try_load_embedder raised; falling back to Jaccard. %s",
                    exc,
                )
                embedder = None

        embedder_loaded = embedder is not None
        metric = "cosine" if embedder_loaded else "jaccard"
        threshold = min_cosine if embedder_loaded else min_jaccard

        issues: List[GateIssue] = []
        audited_with_ref = 0
        resolved = 0
        misaligned = 0
        unresolved = 0
        skipped_no_ref = 0
        # Track per-block aggregates so the decision-capture rationale
        # is keyed at validate() granularity (one event per call).
        block_id_repr = "<batch>"
        if len(assessments) == 1:
            block_id_repr = getattr(assessments[0], "block_id", "<unknown>")

        # Surface a single warning when extras are missing so operators
        # see the silent-degrade signal in the gate report. The Jaccard
        # fallback continues running — this validator is intentionally
        # always-useful, unlike the Phase 4 statistical-tier gates that
        # short-circuit on missing extras.
        if not embedder_loaded:
            issues.append(GateIssue(
                severity="warning",
                code="EMBEDDING_DEPS_MISSING",
                message=(
                    "sentence-transformers extras not installed; "
                    "DistractorMisconceptionAlignmentValidator falling "
                    "back to Jaccard token-overlap (threshold "
                    f"{min_jaccard:.4f}). Install via "
                    "`pip install -e .[embedding]` for cosine "
                    "similarity scoring."
                ),
            ))

        for block in assessments:
            block_id = getattr(block, "block_id", "<unknown>")
            distractors = _extract_distractors_outline(block)
            if not distractors:
                # Rewrite-tier (str content) blocks don't carry the
                # misconception_ref slot in their HTML body; the
                # outline-tier dict shape is the only resolvable
                # surface. Skipping cleanly so the validator is a
                # no-op on rewrite-tier wirings (the W3a plan stages
                # the gate symmetrically; the rewrite-tier wiring
                # surfaces no issues, which is the correct behaviour
                # because the outline-tier gate already enforced
                # alignment before rewrite saw the block).
                continue

            distractor_vec_cache: Dict[int, Any] = {}

            for d_idx, d_text, ref in distractors:
                if ref is None:
                    skipped_no_ref += 1
                    continue
                audited_with_ref += 1
                resolved_text = misconception_index.get(ref)
                if not resolved_text:
                    unresolved += 1
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="critical",
                            code="DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED",
                            message=(
                                f"assessment_item Block {block_id!r} "
                                f"distractor[{d_idx}].misconception_ref="
                                f"{ref!r} does not resolve against the "
                                f"course misconception inventory "
                                f"(inventory size: {len(misconception_index)})."
                            ),
                            location=block_id,
                            suggestion=(
                                "Either correct the ref to a CURIE that "
                                "exists in the course misconception "
                                "inventory, or drop the ref (then W3b's "
                                "DistractorPlausibilityValidator covers "
                                "the no-ref plausibility axis)."
                            ),
                        ))
                    continue

                # Resolved — compute alignment.
                resolved += 1
                similarity: float
                if embedder_loaded:
                    try:
                        if d_idx not in distractor_vec_cache:
                            distractor_vec_cache[d_idx] = embedder.encode(d_text)
                        misc_vec = embedder.encode(resolved_text)
                        similarity = cosine_similarity(
                            distractor_vec_cache[d_idx], misc_vec,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "embedder.encode raised on block %s "
                            "distractor[%d]: %s",
                            block_id, d_idx, exc,
                        )
                        # Fall back to Jaccard for this pair so a single
                        # bad encode doesn't poison the whole batch.
                        similarity = _jaccard(d_text, resolved_text)
                        if len(issues) < _ISSUE_LIST_CAP:
                            issues.append(GateIssue(
                                severity="warning",
                                code="EMBEDDING_ENCODE_ERROR",
                                message=(
                                    f"Failed to encode distractor[{d_idx}] / "
                                    f"misconception for block {block_id!r}: "
                                    f"{exc}; falling back to Jaccard for "
                                    f"this pair."
                                ),
                                location=block_id,
                            ))
                else:
                    similarity = _jaccard(d_text, resolved_text)

                if similarity < threshold:
                    misaligned += 1
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="critical",
                            code="DISTRACTOR_MISCONCEPTION_MISALIGNED",
                            message=(
                                f"assessment_item Block {block_id!r} "
                                f"distractor[{d_idx}] has {metric} "
                                f"similarity {similarity:.4f} with its "
                                f"declared misconception {ref!r}, below "
                                f"threshold {threshold:.4f}. The "
                                f"distractor text may not actually embody "
                                f"the referenced misconception."
                            ),
                            location=block_id,
                            suggestion=(
                                "Re-roll the rewrite-tier provider with "
                                "the misconception statement injected "
                                "verbatim into the prompt, or revise the "
                                "distractor's misconception_ref to match "
                                "what the distractor actually embodies."
                            ),
                        ))

        # Single decision-capture event per validate() call (per the
        # CLAUDE.md call-site instrumentation contract).
        _emit_decision(
            capture,
            block_id=block_id_repr,
            audited_count=audited_with_ref,
            resolved_count=resolved,
            misaligned_count=misaligned,
            unresolved_count=unresolved,
            skipped_no_ref_count=skipped_no_ref,
            threshold=threshold,
            metric=metric,
            embedder_loaded=embedder_loaded,
        )

        # Aggregate score: passing distractors / audited distractors.
        # When no distractor carried a ref, score is vacuously 1.0
        # (the validator's no-op semantic on a corpus without
        # misconception-targeted distractors).
        if audited_with_ref == 0:
            score = 1.0
        else:
            passing = audited_with_ref - misaligned - unresolved
            score = round(max(0.0, passing) / audited_with_ref, 4)

        critical_issue_count = sum(
            1 for issue in issues if issue.severity == "critical"
        )
        passed = critical_issue_count == 0
        action: Optional[str] = None
        if not passed:
            action = action_on_fail

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )


__all__ = [
    "DistractorMisconceptionAlignmentValidator",
    "DEFAULT_MIN_COSINE",
    "DEFAULT_MIN_JACCARD",
]
