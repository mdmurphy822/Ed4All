"""Phase 4 Wave N2 — Category C embedding validator (Subtask 15).

Validates that every ``example`` Block aligns semantically with the
concept(s) it claims to illustrate. Embedding-tier sibling of the
Phase-3 ``BlockCurieAnchoringValidator`` (which only checks that a
declared CURIE *appears* in the textual surface, not whether the
surrounding text is *about* the concept). This validator embeds the
example body and the concept slug+definition surface, then computes
pair-wise cosine similarity. A drop below the threshold (default
0.50) emits ``action="regenerate"`` so the rewrite-tier router
re-rolls the example with sharper concept targeting.

Inputs (``inputs`` dict, mirroring the Phase 3.5 inter_tier_gates'
``BlockCurieAnchoringValidator`` shape):

    blocks: List[Block]
        Outline- or rewrite-tier ``Courseforge.scripts.blocks.Block``
        instances. Only ``block.block_type == "example"`` rows are
        audited; other block types are skipped.

    concept_definitions: Optional[Dict[str, str]]
        Mapping of canonical concept slug / CURIE -> human-readable
        definition (e.g. ``{"ed4all:Foo": "A foo is a..."}``). The
        router / workflow runner populates this from the project's
        knowledge-graph definitions before dispatch (mirrors the
        ``valid_objective_ids`` / ``objective_statements`` asymmetry
        on the sibling validators).

    threshold: Optional[float]
        Override the default cosine-similarity floor (0.50). Below this
        value the gate emits a critical issue with action="regenerate".
        Default is intentionally lower than the assessment/objective
        gate (0.55) — examples are inherently more concrete and
        narrative than the abstract concept they illustrate, so the
        cosine floor sits a notch lower.

    embedder: Optional[SentenceEmbedder]
        Test injection point. Defaults to ``try_load_embedder()`` per
        Wave N1's fallback contract.

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"concept_example_similarity"``).

Behavior contract (Wave N1 fallback policy mirror):
    - Missing ``blocks`` input -> ``passed=False``, single critical
      issue, ``action="regenerate"``.
    - No example blocks -> ``passed=True``, ``action=None``, no issues.
    - Embedding extras missing -> single warning issue
      (``EMBEDDING_DEPS_MISSING``), ``passed=True``, ``action=None``.
    - At least one example with ``min(per-pair cosine) < threshold``
      -> ``passed=False``, ``action="regenerate"`` with one critical
      issue per low-similarity pair (capped at 50 entries).

References:
    - ``lib/validators/objective_assessment_similarity.py`` —
      sibling validator built in Subtask 14.
    - ``Courseforge/router/inter_tier_gates.py::BlockCurieAnchoringValidator``
      — structural sibling that this embedding validator complements.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    SentenceEmbedder,
    try_load_embedder,
)

# Import bridge for Block (mirror of Subtask 14 + inter_tier_gates).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Default cosine-similarity floor for the (example, concept) pair
#: alignment check. Calibrated lower than Subtask 14's
#: ``DEFAULT_THRESHOLD=0.55`` because examples are intentionally
#: concrete narratives that illustrate an abstract concept; the
#: surface-text overlap is naturally lower than between an assessment
#: stem and an objective statement.
DEFAULT_THRESHOLD: float = 0.50

#: Cap on per-block issue list (matches Subtask 14 + inter_tier_gates).
_ISSUE_LIST_CAP: int = 50


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Block], Optional[GateIssue]]:
    """Pull a list of Block instances out of ``inputs["blocks"]``.

    Mirror of the helper in Subtask 14 and inter_tier_gates so the
    error-issue shape stays consistent across the embedding-tier and
    structural-tier validators.
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


def _extract_example_surface(block: Block) -> Optional[str]:
    """Return the textual surface of an ``example`` Block.

    Outline-tier dict shape: prefers ``content["body"]`` (the per-block-type
    field for example narratives), falls back to joining
    ``content["key_claims"]`` when ``body`` is empty.
    Rewrite-tier str shape: strips HTML tags + collapses whitespace
    via the same lightweight regex helper the inter_tier_gates
    adapters use.
    """
    content = block.content
    if isinstance(content, dict):
        body = content.get("body")
        if isinstance(body, str) and body.strip():
            return body.strip()
        # Fall back to key_claims joined with newlines so the
        # embedding has *some* surface to work with even if the
        # outline tier didn't emit a body field.
        claims = content.get("key_claims") or []
        joined = "\n".join(c for c in claims if isinstance(c, str) and c)
        return joined or None
    if isinstance(content, str):
        import re

        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    return None


def _resolve_concept_refs(block: Block) -> List[str]:
    """Return the concept slug / CURIE list a block declares.

    The outline-tier shape carries CURIEs in ``content["curies"]`` (the
    canonical Phase-3 anchoring field — see
    ``Courseforge/generators/_outline_provider.py:347-350``); some
    fixtures additionally use ``concept_refs`` for explicit concept
    targeting. Both surfaces are unioned, deduplicated while
    preserving discovery order.
    """
    content = block.content
    refs: List[str] = []
    seen = set()

    def _add(items):
        for item in items or []:
            if isinstance(item, str) and item and item not in seen:
                seen.add(item)
                refs.append(item)

    if isinstance(content, dict):
        _add(content.get("concept_refs"))
        _add(content.get("curies"))
    return refs


class ConceptExampleSimilarityValidator:
    """Phase 4 Wave N2 — embedding-tier concept/example alignment gate.

    Validator-protocol-compatible class designed to be wired into
    ``inter_tier_validation`` (Subtask 17) and ``post_rewrite_validation``
    (Subtask 18). Severity is intentionally ``warning`` per Wave N2's
    PoC contract — the structural Phase-3
    ``BlockCurieAnchoringValidator`` remains authoritative; this gate
    surfaces semantic-drift signals (an example that *mentions* the
    concept but doesn't *illustrate* it) the structural validator
    cannot detect.
    """

    name = "concept_example_similarity"
    version = "0.1.0"  # Phase 4 Wave N2 PoC

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        embedder: Optional[SentenceEmbedder] = None,
    ) -> None:
        self._threshold = threshold
        self._embedder_override = embedder

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        threshold = float(inputs.get("threshold", self._threshold))

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

        examples = [
            b for b in blocks if getattr(b, "block_type", None) == "example"
        ]
        if not examples:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        embedder = self._embedder_override or try_load_embedder()
        if embedder is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="EMBEDDING_DEPS_MISSING",
                        message=(
                            "sentence-transformers extras not installed; "
                            "Phase 4 PoC concept/example similarity gate "
                            "skipped. Install via "
                            "`pip install -e .[embedding]` to enable."
                        ),
                    )
                ],
            )

        concept_definitions: Dict[str, str] = inputs.get(
            "concept_definitions", {}
        ) or {}
        if not isinstance(concept_definitions, dict):
            concept_definitions = {}

        issues: List[GateIssue] = []
        passed_count = 0
        audited = 0
        low_similarity_count = 0

        for block in examples:
            audited += 1
            surface = _extract_example_surface(block)
            if not surface:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="EXAMPLE_SURFACE_EMPTY",
                            message=(
                                f"Block {block.block_id!r} declares "
                                f"block_type='example' but carries no "
                                f"body/key_claims text surface; embedding "
                                f"similarity skipped for this block."
                            ),
                            location=block.block_id,
                        )
                    )
                continue

            concept_refs = _resolve_concept_refs(block)
            if not concept_refs:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="EXAMPLE_NO_CONCEPT_REFS",
                            message=(
                                f"Block {block.block_id!r} declares no "
                                f"concept_refs / curies; embedding similarity "
                                f"requires at least one declared concept."
                            ),
                            location=block.block_id,
                        )
                    )
                continue

            try:
                example_vec = embedder.encode(surface)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedder.encode raised on block %s: %s",
                    block.block_id,
                    exc,
                )
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="EMBEDDING_ENCODE_ERROR",
                            message=(
                                f"Failed to encode example surface for block "
                                f"{block.block_id!r}: {exc}"
                            ),
                            location=block.block_id,
                        )
                    )
                continue

            min_cos: Optional[float] = None
            min_concept: Optional[str] = None
            unresolved: List[str] = []
            for concept_ref in concept_refs:
                definition = concept_definitions.get(concept_ref)
                if not definition or not isinstance(definition, str):
                    unresolved.append(concept_ref)
                    # Even without an explicit definition, the slug /
                    # CURIE itself still carries semantic content; the
                    # validator falls back to embedding the slug + the
                    # local-part as the concept surface so this isn't
                    # a hard skip when the concept_definitions map is
                    # unpopulated.
                    concept_surface = _slug_to_natural_text(concept_ref)
                else:
                    concept_surface = f"{concept_ref} {definition}"
                try:
                    concept_vec = embedder.encode(concept_surface)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "embedder.encode raised on concept %s: %s",
                        concept_ref,
                        exc,
                    )
                    continue
                cos = cosine_similarity(example_vec, concept_vec)
                if min_cos is None or cos < min_cos:
                    min_cos = cos
                    min_concept = concept_ref

            if unresolved and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="CONCEPT_DEFINITION_UNRESOLVED",
                        message=(
                            f"Block {block.block_id!r} references concept_refs "
                            f"{unresolved!r} that don't appear in the supplied "
                            f"concept_definitions map; falling back to "
                            f"slug-only embedding surface (lower signal)."
                        ),
                        location=block.block_id,
                    )
                )

            if min_cos is None:
                continue

            if min_cos < threshold:
                low_similarity_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="EXAMPLE_CONCEPT_LOW_SIMILARITY",
                            message=(
                                f"Block {block.block_id!r} (example) has "
                                f"cosine similarity {min_cos:.4f} with its "
                                f"weakest declared concept {min_concept!r}, "
                                f"below threshold {threshold:.4f}. The "
                                f"example body may not actually illustrate "
                                f"the declared concept."
                            ),
                            location=block.block_id,
                            suggestion=(
                                "Re-roll the rewrite-tier provider with the "
                                "concept definition injected verbatim into "
                                "the prompt, or revise the block's "
                                "concept_refs to match what the example "
                                "actually illustrates."
                            ),
                        )
                    )
            else:
                passed_count += 1

        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        passed = low_similarity_count == 0
        action: Optional[str] = None
        if low_similarity_count > 0:
            action = "regenerate"

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )


def _slug_to_natural_text(slug: str) -> str:
    """Convert a slug or CURIE to a natural-language surface.

    Used as a fallback embedding surface when a concept_definition isn't
    available. ``ed4all:FooBar`` -> ``"ed4all FooBar Foo Bar"``: keeps
    the original CURIE for namespace context and additionally splits
    CamelCase / snake_case boundaries so the embedding model has more
    surface tokens to work with than the bare slug provides.
    """
    if not slug:
        return ""
    import re

    # Split on prefix:local-part
    parts = slug.split(":", 1)
    local = parts[-1]
    # Split CamelCase + replace -/_ with spaces.
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", local)
    spaced = re.sub(r"[-_]+", " ", spaced)
    return f"{slug} {spaced}".strip()


__all__ = [
    "ConceptExampleSimilarityValidator",
    "DEFAULT_THRESHOLD",
]
