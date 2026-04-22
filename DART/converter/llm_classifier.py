"""Phase 2 (Wave 14): Claude-backed block classification.

Every block produced by :mod:`DART.converter.block_segmenter` is routed
through an :class:`~MCP.orchestrator.llm_backend.LLMBackend` (injected by
the caller) so that role assignment is a Claude reasoning step rather
than a regex heuristic. This lets the classifier disambiguate edge
cases the regex classifier cannot — e.g. "I. INTRODUCTION" (arxiv
section heading) vs "1. First bullet item" (ordinary list item).

The heuristic classifier (``heuristic_classifier.HeuristicClassifier``)
remains the offline fallback. Any block the LLM omits, mislabels with
an unknown role, or fails to return for is filled in by the heuristic
classifier specifically for those blocks — the pipeline never crashes
on a partial response. Blocks classified by the heuristic fallback
carry ``classifier_source="heuristic"``; blocks classified by Claude
carry ``classifier_source="llm"``.

Design notes
------------

* **Batching** — blocks are grouped in batches of ~20 (configurable via
  ``batch_size``) to keep each prompt small enough to fit comfortably
  inside the default ``max_tokens`` response budget while still
  reducing API overhead. 20 blocks × (~500-char excerpt + ~400 chars
  of neighbour context) fits well under typical model context limits.

* **Prompt shape** — a system message names every allowed
  :class:`BlockRole` value and instructs Claude to return a strict JSON
  array. The user message lists each block's ID, text (truncated to
  500 chars), and ``neighbors.prev`` / ``neighbors.next`` excerpts so
  the model can disambiguate by context. Temperature is pinned at
  ``0.0`` for deterministic tagging.

* **No direct SDK use** — all LLM traffic flows through the
  :class:`LLMBackend` protocol. Callers inject a backend; tests inject
  :class:`~MCP.orchestrator.llm_backend.MockBackend`. There is no
  ``import anthropic`` anywhere in this module.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.heuristic_classifier import HeuristicClassifier

logger = logging.getLogger(__name__)


# Maximum characters sent per block body. Keeps batched prompts inside
# reasonable token budgets while preserving enough signal for
# classification. 500 chars ≈ 100-125 tokens per block.
DEFAULT_TEXT_TRUNCATION = 500

# Neighbour context is typically shorter — the classifier only needs a
# few sentences of surrounding content to pick the right role.
DEFAULT_NEIGHBOR_TRUNCATION = 200

# Default batch size — 20 blocks per LLM call balances prompt length
# against round-trip overhead. Tune via the ``batch_size`` kwarg.
DEFAULT_BATCH_SIZE = 20

# Default response budget for an LLM call; 20 blocks × ~150-200
# response tokens ≈ 4000 tokens.
DEFAULT_MAX_TOKENS = 4096


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` truncated to ``limit`` characters with a marker."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _build_system_message() -> str:
    """Return the system message naming every allowed role.

    Listing the closed-set role values in the system message lets the
    model know exactly which strings are valid in its JSON response,
    and the post-parse step rejects any role value that does not map
    to a :class:`BlockRole`.
    """
    role_lines = [f"- {role.value}" for role in BlockRole]
    roles_block = "\n".join(role_lines)
    return (
        "You classify document blocks produced by a PDF-to-HTML converter "
        "into one of the following roles. Each block must receive exactly "
        "one role from this closed set:\n\n"
        f"{roles_block}\n\n"
        "Use the surrounding context (the prev / next block excerpts) to "
        "disambiguate. For example, \"I. INTRODUCTION\" at the top of an "
        "arxiv-style paper is section_heading or abstract, while "
        "\"1. First item\" inside a bulleted list is paragraph.\n\n"
        "Respond with a single JSON array. Each element must be an object "
        "of the form:\n"
        "  {\"block_id\": \"<id>\", \"role\": \"<snake_case_role>\", "
        "\"confidence\": 0.0-1.0, \"attributes\": { ... }}\n"
        "The \"attributes\" field is optional per block and is where you put "
        "role-specific extras (e.g. heading_text, figure_number, caption, "
        "severity for callouts, number for bibliography_entry / footnote).\n"
        "Return only the JSON array — no prose, no markdown fencing."
    )


def _build_user_message(batch: List[RawBlock]) -> str:
    """Return the user message carrying the batch's block contents.

    Each block is rendered as a numbered item so the model can refer
    to it; the ``block_id`` is echoed verbatim so the response can be
    matched back to the :class:`RawBlock`.
    """
    lines: List[str] = ["Classify the following blocks:"]
    for idx, block in enumerate(batch, start=1):
        text = _truncate(block.text.strip(), DEFAULT_TEXT_TRUNCATION)
        prev = _truncate(
            (block.neighbors or {}).get("prev", ""),
            DEFAULT_NEIGHBOR_TRUNCATION,
        )
        nxt = _truncate(
            (block.neighbors or {}).get("next", ""),
            DEFAULT_NEIGHBOR_TRUNCATION,
        )
        lines.append(
            f"\n--- Block {idx} ---\n"
            f"block_id: {block.block_id}\n"
            f"text: {text}\n"
            f"neighbors.prev: {prev}\n"
            f"neighbors.next: {nxt}"
        )
    return "\n".join(lines)


def _strip_fences(raw: str) -> str:
    """Strip ```json ... ``` fencing that some model replies add."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        # Remove opening fence (``` or ```json on its own line).
        stripped = re.sub(r"^```[a-zA-Z0-9]*\n?", "", stripped)
        # Remove closing fence.
        if stripped.endswith("```"):
            stripped = stripped[: -3]
        stripped = stripped.strip()
    return stripped


def _parse_response(
    raw: str,
    batch: List[RawBlock],
) -> Dict[str, Dict[str, Any]]:
    """Parse the model response into ``{block_id: entry}`` mapping.

    Returns an empty dict when parsing fails; the caller then drops
    the whole batch to the heuristic fallback. Entries referring to
    block IDs not in the current batch are silently ignored.
    """
    try:
        payload = json.loads(_strip_fences(raw))
    except (ValueError, TypeError) as exc:
        logger.warning(
            "LLMClassifier: response is not valid JSON (%s); falling back",
            exc,
        )
        return {}

    if not isinstance(payload, list):
        logger.warning(
            "LLMClassifier: response root is %s, expected list; falling back",
            type(payload).__name__,
        )
        return {}

    valid_ids = {b.block_id for b in batch}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        block_id = entry.get("block_id")
        if not isinstance(block_id, str) or block_id not in valid_ids:
            continue
        role_value = entry.get("role")
        if not isinstance(role_value, str):
            continue
        out[block_id] = entry
    return out


def _role_from_string(value: str) -> Optional[BlockRole]:
    """Map a snake_case role string back to the :class:`BlockRole` enum."""
    try:
        return BlockRole(value)
    except ValueError:
        return None


class LLMClassifier:
    """Claude-backed block classifier.

    Parameters
    ----------
    llm:
        The injected :class:`LLMBackend`. **Required** — instantiation
        without a backend raises :class:`ValueError` with a clear
        message telling the caller to inject one (tests should pass a
        :class:`MockBackend`).
    batch_size:
        How many blocks to send per LLM call. Defaults to 20.
    model:
        Optional model override passed straight through to
        :meth:`LLMBackend.complete`. ``None`` lets the backend pick.
    max_tokens:
        Response-budget ceiling per LLM call. Defaults to 4096.
    fallback:
        The heuristic classifier used when the LLM response is
        unusable (invalid JSON, missing block IDs, unknown role
        strings). Defaults to a fresh :class:`HeuristicClassifier`.
    capture:
        Optional :class:`lib.decision_capture.DecisionCapture`
        instance. When supplied, one ``decision_type="structure_detection"``
        record is logged per LLM batch carrying the batch's block-ID
        range, token + confidence stats, and fallback fraction so the
        Wave-22 DC1 audit requirement (zero captures at per-batch
        classification sites) is met. When ``None`` (the default)
        logging is silently skipped — existing tests that don't care
        about captures keep passing byte-for-byte.
    """

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        fallback: Optional[HeuristicClassifier] = None,
        capture: Optional[Any] = None,
    ):
        if llm is None:
            raise ValueError(
                "LLMClassifier requires an LLMBackend. Inject one via "
                "llm=<backend>; tests should pass MockBackend. See "
                "MCP/orchestrator/llm_backend.py for the protocol."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.llm = llm
        self.batch_size = batch_size
        self.model = model
        self.max_tokens = max_tokens
        self.fallback = fallback or HeuristicClassifier()
        self.capture = capture

    async def classify(self, blocks: List[RawBlock]) -> List[ClassifiedBlock]:
        """Classify every block via the injected backend.

        Blocks are batched in groups of :attr:`batch_size`. When a
        batch response is unparseable, every block in that batch is
        routed through the heuristic fallback. When a batch response
        is partial (missing IDs, unknown role strings), only the
        affected blocks fall back — the rest keep their LLM label.

        Wave 16: blocks carrying an ``extractor_hint`` (structured
        extraction from pdfplumber / PyMuPDF) skip the LLM entirely.
        They are classified as the hinted role at confidence 1.0 and
        re-interleaved into the output in original position so the
        prompt never sees them and the backend is never asked to
        classify e.g. a table's row text as prose.
        """
        if not blocks:
            return []

        # Partition: hinted blocks never reach the LLM.
        hinted: Dict[str, ClassifiedBlock] = {}
        unhinted: List[RawBlock] = []
        for block in blocks:
            if block.extractor_hint is not None:
                hinted[block.block_id] = ClassifiedBlock(
                    raw=block,
                    role=block.extractor_hint,
                    confidence=1.0,
                    attributes=dict(block.extra or {}),
                    classifier_source="extractor_hint",
                )
            else:
                unhinted.append(block)

        classified_unhinted: List[ClassifiedBlock] = []
        for start in range(0, len(unhinted), self.batch_size):
            batch = unhinted[start : start + self.batch_size]
            classified_unhinted.extend(await self._classify_batch(batch))

        # Merge hinted + LLM-classified back into original input order.
        unhinted_map = {cb.raw.block_id: cb for cb in classified_unhinted}
        results: List[ClassifiedBlock] = []
        for block in blocks:
            if block.block_id in hinted:
                results.append(hinted[block.block_id])
            else:
                results.append(unhinted_map[block.block_id])
        return results

    async def _classify_batch(
        self, batch: List[RawBlock]
    ) -> List[ClassifiedBlock]:
        system = _build_system_message()
        user = _build_user_message(batch)

        try:
            raw_response = await self.llm.complete(
                system=system,
                user=user,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.warning(
                "LLMClassifier: backend raised %s (%s); falling back for batch",
                type(exc).__name__,
                exc,
            )
            fallback_classified = self.fallback.classify_sync(batch)
            self._log_batch_decision(batch, fallback_classified)
            return fallback_classified

        # Defensive: some backends may hand back non-strings in surprising
        # cases; a streaming iterator (we never request stream=True) would
        # also fall into this branch. Either way, drop to heuristic.
        if not isinstance(raw_response, str):
            logger.warning(
                "LLMClassifier: backend returned %s, expected str; falling back",
                type(raw_response).__name__,
            )
            fallback_classified = self.fallback.classify_sync(batch)
            self._log_batch_decision(batch, fallback_classified)
            return fallback_classified

        parsed = _parse_response(raw_response, batch)
        if not parsed:
            # Whole-batch fallback — parsing failed or nothing survived.
            fallback_classified = self.fallback.classify_sync(batch)
            self._log_batch_decision(batch, fallback_classified)
            return fallback_classified

        # Per-block merge: use the LLM label when we have a valid entry,
        # else fall back to the heuristic for that block alone.
        results: List[ClassifiedBlock] = []
        missing_blocks: List[RawBlock] = []
        for block in batch:
            entry = parsed.get(block.block_id)
            if entry is None:
                missing_blocks.append(block)
                continue
            role = _role_from_string(entry["role"])
            if role is None:
                missing_blocks.append(block)
                continue

            confidence = entry.get("confidence")
            if not isinstance(confidence, (int, float)):
                confidence = 0.8  # Reasonable default when LLM omits it.
            # Clamp to the documented [0.0, 1.0] range.
            confidence = max(0.0, min(1.0, float(confidence)))

            attributes = entry.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}

            results.append(
                ClassifiedBlock(
                    raw=block,
                    role=role,
                    confidence=confidence,
                    attributes=attributes,
                    classifier_source="llm",
                )
            )

        if missing_blocks:
            logger.info(
                "LLMClassifier: heuristic fallback filling %d/%d blocks",
                len(missing_blocks),
                len(batch),
            )
            fallback_results = self.fallback.classify_sync(missing_blocks)
            fallback_map = {cb.raw.block_id: cb for cb in fallback_results}
            # Re-interleave in original batch order so callers see a
            # deterministic sequence (position-stable with the input).
            interleaved: List[ClassifiedBlock] = []
            llm_map = {cb.raw.block_id: cb for cb in results}
            for block in batch:
                if block.block_id in llm_map:
                    interleaved.append(llm_map[block.block_id])
                else:
                    interleaved.append(fallback_map[block.block_id])
            self._log_batch_decision(batch, interleaved)
            return interleaved

        self._log_batch_decision(batch, results)
        return results

    def _log_batch_decision(
        self,
        batch: List[RawBlock],
        classified: List[ClassifiedBlock],
    ) -> None:
        """Wave 22 DC1: one decision record per LLM classification batch.

        Emits a single ``structure_detection`` decision carrying a
        summary of the role distribution, the block-ID range covered,
        aggregate token counts for the batch prompts, the fraction of
        blocks that ended up at low confidence (< 0.7), and the
        heuristic-fallback fraction. Rationale is dynamic — every
        number is interpolated from the actual batch so no two
        captures are byte-identical.

        Silently no-ops when no capture was injected.
        """
        capture = getattr(self, "capture", None)
        if capture is None or not batch:
            return

        # Role distribution summary (top-3, with counts).
        from collections import Counter

        role_counter = Counter(cb.role.value for cb in classified)
        top_roles = role_counter.most_common(3)
        role_summary = ", ".join(
            f"{count} {role}" for role, count in top_roles
        )

        # Source distribution: LLM vs extractor_hint vs heuristic fallback.
        source_counter = Counter(
            cb.classifier_source for cb in classified
        )
        fallback_count = source_counter.get("heuristic", 0)
        llm_count = source_counter.get("llm", 0)
        fallback_fraction = (
            fallback_count / len(classified) if classified else 0.0
        )

        # Confidence distribution.
        confidences = [cb.confidence for cb in classified]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        low_conf = sum(1 for c in confidences if c < 0.7)
        low_conf_fraction = low_conf / len(confidences) if confidences else 0.0

        # Token-count proxy: sum of block text lengths (chars → tokens
        # ≈ chars / 4 for English prose; we surface chars directly so
        # the number is auditable).
        total_chars = sum(len(b.text or "") for b in batch)

        # Block-ID range for auditability.
        first_id = batch[0].block_id
        last_id = batch[-1].block_id
        batch_ids = [b.block_id for b in batch]

        decision = (
            f"Classified {len(batch)}-block batch — {role_summary}"
            if role_summary
            else f"Classified {len(batch)}-block batch"
        )
        rationale = (
            f"Block range {first_id}..{last_id}; "
            f"LLM={llm_count}, heuristic_fallback={fallback_count} "
            f"({fallback_fraction * 100:.0f}% fallback); "
            f"avg confidence {avg_conf:.2f}, "
            f"{low_conf} blocks (<0.7) = {low_conf_fraction * 100:.0f}% low-confidence; "
            f"~{total_chars} char prompt payload; "
            f"model={self.model or 'default'} max_tokens={self.max_tokens}"
        )

        try:
            capture.log_decision(
                decision_type="structure_detection",
                decision=decision,
                rationale=rationale,
                context=(
                    f"batch_size={len(batch)}, "
                    f"block_ids={batch_ids[:6]}{'...' if len(batch_ids) > 6 else ''}"
                ),
                inputs_ref=[
                    {"source_type": "agent_output", "path_or_id": bid}
                    for bid in batch_ids
                ],
                confidence=avg_conf,
            )
        except Exception as exc:  # noqa: BLE001 — capture is best-effort
            logger.debug(
                "LLMClassifier capture emit failed (%s); continuing", exc
            )


__all__ = ["LLMClassifier"]
