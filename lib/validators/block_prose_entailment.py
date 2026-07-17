"""W4 — ``BlockProseEntailmentValidator`` (per-block full-prose NLI entailment).

The text-trace counterpart to the (W4-demoted) cosine
``RewriteSourceGroundingValidator`` AND the broadened counterpart to
``ClaimSupportValidator`` — where ``claim_support`` scores only the curated
``key_claims[]``, this validator scores ALL substantive rewrite-tier block
prose against the block's cited source chunk(s) via NLI entailment.

Cosine 0.45 catches gross off-topic drift but PASSES plausible-but-fabricated
on-topic prose (a fabricated sentence that stays on-topic scores 0.5+ cosine).
NLI entailment is the stronger text-trace signal: a fabricated sentence that
is not entailed by any cited chunk fails NLI even when it passes cosine. This
is the keystone failure class W4 closes.

Algorithm (per ``plans/finegrain/w4-nli-grounding-gate.md`` §1.2):

1. **Skip set** — reuse ``claim_support._SKIPPED_BLOCK_TYPES``
   (``assessment_item`` / ``objective`` / ``chrome`` / ``recap``). DELIBERATE
   asymmetry vs ``rewrite_source_grounding._SKIPPED_CONTENT_TYPES`` which ALSO
   skips ``example`` / ``self_check_question``: NLI scores entailment, not
   surface cosine, so ``example`` (code / URI payload depresses cosine but not
   entailment of the framing prose) and ``self_check_question`` ARE scored
   here. Computational sentences inside examples are exempted downstream by
   ``groundedness._is_computational``.
2. **Strip prose to text** — ``lib.utils.strip_html_to_text`` (the W-D6
   canonical helper). Rewrite-tier ``block.content`` is a ``str`` (HTML) →
   strip; outline-tier ``block.content`` is a ``dict`` → skip silently
   (the rewrite-tier seam is where prose entailment is checked).
3. **Resolve cited chunk texts** — walk the same id surfaces
   ``rewrite_source_grounding._resolve_block_source_chunks`` consults, but
   preserve the chunk_id so the report's ``best_chunk_id`` is meaningful.
4. **Score** — ``groundedness.score_groundedness`` (windowed scorer-v2). All
   claim splitting + computational exemption + structural-artifact drop +
   windowed rescue is INSIDE ``score_groundedness`` — the validator does NOT
   re-split.

Pass/fail (per block, §1.3):

- ``BLOCK_PROSE_UNGROUNDED`` when ``scored_count > 0`` AND
  ``block_entailment_rate < min_block_entailment_rate``. ``action="regenerate"``.
- ``BLOCK_PROSE_CONTRADICTED`` when ``contradicted_rate > max_contradicted_rate``
  — fires standalone (a contradiction is fabrication that *conflicts* with
  source, the worst case). ``action="regenerate"``.
- ``scored_count == 0`` (only computational / artifact prose) → no-op pass.
- No resolved cited chunks → warning ``BLOCK_PROSE_NO_GROUNDING_SOURCE``,
  ``passed=True`` (the resolution gate ``rewrite_source_refs`` owns the
  missing-source structural failure).

Graceful-degrade (§1.4): mirrors ``claim_support`` / the EMBEDDING_DEPS_MISSING
contract. Pre-check ``NliClassifier.get_or_load()``; non-strict + missing deps
→ warning ``NLI_DEPS_MISSING``, ``passed=True, action=None``; strict
(``TRAINFORGE_REQUIRE_EMBEDDINGS=true``) → raise ``RuntimeError``.

**Shadow knob (§4.3):** ``inputs["shadow"]`` (merged from the gate
``config.shadow: true`` by the executor). When truthy, every emitted
``BLOCK_PROSE_*`` GateIssue is forced to ``severity="warning"`` and
``action`` is never set to ``"regenerate"`` — the gate COMPUTES + CAPTURES
the entailment distribution but does NOT gate. This makes the W4 day-1 shadow
state EXPLICIT in the validator and independent of the YAML ``severity`` field,
so the §4 calibration runs are unambiguous. The critical flip is
``config.shadow: false`` + YAML ``severity: critical`` (DEFERRED — see the
``# TODO(calibration)`` marker at the gate in ``config/workflows.yaml``).

Decision-capture (§1.5): one ``block_prose_entailment_check`` event per SCORED
block; rationale interpolates dynamic signals (≥20 chars). Skip-silent blocks
emit nothing (matches ``claim_support``).

Reuse contract (§8): NLI model singleton (``NliClassifier.get_or_load``),
sentence/claim split + artifact drop + computational exemption + windowed
rescue (``score_groundedness``), HTML→text (``strip_html_to_text``),
strict-mode flag (``is_strict_mode``), skip set
(``claim_support._SKIPPED_BLOCK_TYPES``), floors
(``_DEFAULT_ENTAILMENT_FLOOR`` / ``_DEFAULT_CONTRADICTION_FLOOR``).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier
from lib.embedding.sentence_embedder import is_strict_mode
from lib.retrieval.groundedness import score_groundedness
from lib.utils import strip_html_to_text as _strip_html_to_text  # W-D6 canonical
from lib.validators.claim_support import _SKIPPED_BLOCK_TYPES

# Block lives at Courseforge/scripts/blocks.py — same import bridge as the
# sibling validators.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Per-block aggregate entailment floor: at least this share of scored
#: (non-computational, non-artifact) prose sentences must be entailed by the
#: block's cited chunks. Day-1 0.60 (calibratable — §4 pins the value before
#: the critical flip). Mirrors ``claim_support``'s training-pair-calibrated
#: floors; NOT generation-calibrated, which is WHY the gate ships SHADOW.
_DEFAULT_MIN_BLOCK_ENTAILMENT_RATE: float = 0.60

#: Per-claim entailment floor (reuse claim_support default — passed through to
#: ``score_groundedness``).
_DEFAULT_ENTAILMENT_FLOOR: float = 0.70

#: Per-claim contradiction floor (reuse claim_support default).
_DEFAULT_CONTRADICTION_FLOOR: float = 0.50

#: Per-block contradicted-prose rate ceiling. Above this, the block fires
#: ``BLOCK_PROSE_CONTRADICTED`` regardless of total entailment rate — any
#: contradiction is a hard fabrication signal.
_DEFAULT_MAX_CONTRADICTED_RATE: float = 0.05

#: Cap per-block issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50

#: Env flag — gate-side provenance resolution. Default OFF → byte-identical
#: legacy behavior (a block's cited ``semantik:{slug}#{anchor}`` refs are
#: resolved ONLY through the ``source_chunks`` map; unresolved → NO_GROUNDING
#: warning). When truthy AND ``inputs["chunk_provenance_index"]`` is present,
#: a block whose directly-cited ids resolve to nothing (rewrite-tier blocks
#: carry provenance as ``semantik:...#anchor`` refs, not ``{course}_chunk_NNNNN``
#: chunk ids) is re-resolved by mapping each cited provenance ref → the chunk
#: ids that carry it (a section-level ref maps to ALL chunk ids from that
#: section) and looking those chunk ids up in ``source_chunks``. Anti-fabrication:
#: only EXISTING refs are mapped to EXISTING chunks; unmappable refs still emit
#: the NO_GROUNDING warning. Resolution only ADDs — directly-resolvable ids are
#: never removed.
_ENV_PROVENANCE_RESOLVE: str = "ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _provenance_resolve_enabled() -> bool:
    """Parse-with-fallback truthy check for ``ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE``."""
    return os.environ.get(_ENV_PROVENANCE_RESOLVE, "").strip().lower() in _TRUTHY


def load_chunk_provenance_index(chunks_jsonl_path: str) -> Dict[str, List[str]]:
    """Build a ``{provenance_ref -> [chunk_id, ...]}`` reverse index from a
    ``chunks.jsonl`` file.

    Each chunk record carries its source anchors on
    ``source.source_references[].sourceId`` (the ``semantik:{slug}#{anchor}``
    / legacy ``dart:{slug}#{block_id}`` Source-Provenance contract) and its own
    canonical chunk id on the top-level ``id`` field
    (``{course}_chunk_NNNNN``). This inverts that: for every ``sourceId`` a
    chunk declares, the chunk's id is appended to that ref's list. Because a
    section-level ref (e.g. ``semantik:slug#1-1-introduction-to-whole-numbers``)
    is shared by every chunk emitted from that section, the ref naturally maps
    to ALL of the section's chunk ids — the section-ref tolerance the gate needs
    for the ``data-cf-source-ids`` anchor form.

    Best-effort: an absent / unreadable path returns an empty index so the
    gate's graceful NO_GROUNDING path stays intact. Order-preserving,
    per-ref de-duplicated.
    """
    index: Dict[str, List[str]] = {}
    path = Path(chunks_jsonl_path)
    if not path.exists():
        return index
    try:
        import json as _json

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                chunk_id = chunk.get("id") or chunk.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    continue
                chunk_id = chunk_id.strip()
                source = chunk.get("source")
                if not isinstance(source, dict):
                    continue
                src_refs = source.get("source_references") or []
                if not isinstance(src_refs, list):
                    continue
                for ref in src_refs:
                    if not isinstance(ref, dict):
                        continue
                    sid = ref.get("sourceId")
                    if not isinstance(sid, str) or not sid.strip():
                        continue
                    bucket = index.setdefault(sid.strip(), [])
                    if chunk_id not in bucket:
                        bucket.append(chunk_id)
    except OSError as exc:
        logger.debug(
            "load_chunk_provenance_index: read of %s failed: %s",
            chunks_jsonl_path, exc,
        )
        return {}
    return index


# --------------------------------------------------------------------- #
# Canonical GateIssue codes
# --------------------------------------------------------------------- #

_CODE_BLOCK_PROSE_UNGROUNDED: str = "BLOCK_PROSE_UNGROUNDED"
_CODE_BLOCK_PROSE_CONTRADICTED: str = "BLOCK_PROSE_CONTRADICTED"
_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE: str = "BLOCK_PROSE_NO_GROUNDING_SOURCE"
_CODE_NLI_DEPS_MISSING: str = "NLI_DEPS_MISSING"

_DECISION_TYPE: str = "block_prose_entailment_check"


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs["blocks"]``.

    Mirrors ``claim_support._coerce_blocks`` /
    ``rewrite_source_grounding._coerce_blocks``.
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


def _block_should_skip(block: Block) -> bool:
    """True when the block's block_type / content_type is in the skip set."""
    if block.block_type in _SKIPPED_BLOCK_TYPES:
        return True
    if (
        block.content_type_label
        and block.content_type_label in _SKIPPED_BLOCK_TYPES
    ):
        return True
    return False


def _resolve_block_cited_passages(
    block: Block,
    source_chunks: Dict[str, str],
    provenance_index: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    """Resolve the block's cited chunks into ``score_groundedness`` passages.

    Walks ``block.source_references[].sourceId`` + ``block.source_ids`` (the
    same id surfaces ``_resolve_block_source_chunks`` consults), looks each up
    in ``source_chunks``, and returns ``[{"chunk_id": sid, "text": text}, ...]``
    preserving the chunk_id so the report's ``best_chunk_id`` is meaningful.
    Duplicate ids are de-duplicated (first occurrence wins).

    When ``provenance_index`` is supplied (opt-in
    ``ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE``), an ADD pass runs after direct
    resolution: rewrite-tier blocks carry provenance as ``semantik:{slug}#anchor``
    refs, not ``{course}_chunk_NNNNN`` chunk ids, so a block that resolves to
    NOTHING directly is re-resolved by mapping each cited ref through the index
    (``ref -> [chunk_id, ...]``, a section-level ref covering ALL of the
    section's chunk ids) and looking those chunk ids up in ``source_chunks``.
    Anti-fabrication: only EXISTING refs map to EXISTING chunks; directly
    resolved ids are never removed — the index can only ADD supporters.
    """
    seen: set = set()
    passages: List[Dict[str, str]] = []
    candidate_ids: List[str] = []
    for ref in block.source_references or ():
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid:
                candidate_ids.append(sid)
    for sid in block.source_ids or ():
        if isinstance(sid, str) and sid:
            candidate_ids.append(sid)
    for sid in candidate_ids:
        if sid in seen:
            continue
        seen.add(sid)
        text = source_chunks.get(sid)
        if isinstance(text, str) and text.strip():
            passages.append({"chunk_id": sid, "text": text})

    # Opt-in provenance resolution ADD pass. Fires when a block's directly-cited
    # ids resolved to nothing (the rewrite-tier semantik-ref → chunk-id gap);
    # never removes an already-resolved passage (ADD-only).
    if provenance_index and not passages:
        for sid in candidate_ids:
            for chunk_id in provenance_index.get(sid, ()):  # section-level fan-out
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                text = source_chunks.get(chunk_id)
                if isinstance(text, str) and text.strip():
                    passages.append({"chunk_id": chunk_id, "text": text})
    return passages


def _emit_decision(
    capture: Any,
    block: Block,
    *,
    passed: bool,
    failure_codes: List[str],
    scored_count: int,
    entailed_count: int,
    unsupported_count: int,
    contradicted_count: int,
    computational_count: int,
    filtered_count: int,
    windowed_count: int,
    block_entailment_rate: float,
    min_block_entailment_rate: float,
    entailment_floor: float,
    nli_revision: Optional[str],
    nli_device: Optional[str],
    shadow: bool,
) -> None:
    """Emit one ``block_prose_entailment_check`` decision per scored block.

    Rationale interpolates dynamic per-call signals (≥20 chars) so the
    capture is replayable post-hoc — see the decision-capture contract in
    root ``CLAUDE.md`` § "LLM call-site instrumentation". Swallow
    ``log_decision`` exceptions (capture must not break the gate).
    """
    if capture is None:
        return
    decision = (
        "passed" if passed else f"failed:{','.join(failure_codes) or 'unknown'}"
    )
    rationale = (
        f"Block-prose NLI entailment on Block {block.block_id!r}: "
        f"block_type={block.block_type!r}, "
        f"scored_count={scored_count}, "
        f"entailed_count={entailed_count}, "
        f"unsupported_count={unsupported_count}, "
        f"contradicted_count={contradicted_count}, "
        f"computational_count={computational_count}, "
        f"filtered_count={filtered_count}, "
        f"windowed_count={windowed_count}, "
        f"block_entailment_rate={block_entailment_rate:.4f}, "
        f"min_rate={min_block_entailment_rate:.2f}, "
        f"entailment_floor={entailment_floor:.2f}, "
        f"nli_revision={nli_revision}, nli_device={nli_device}, "
        f"shadow={shadow}, "
        f"failure_codes={','.join(failure_codes) or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on %s: %s",
            _DECISION_TYPE, exc,
        )


class BlockProseEntailmentValidator:
    """Per-block full-prose NLI entailment gate (W4).

    Iterates every audited Block (skip set per
    ``claim_support._SKIPPED_BLOCK_TYPES``); for each rewrite-tier
    (string-content) block, strips the HTML prose and scores it against the
    block's cited source chunk(s) via ``score_groundedness`` (windowed
    scorer-v2). Fires ``BLOCK_PROSE_UNGROUNDED`` when the per-block
    entailment rate falls below ``min_block_entailment_rate`` and
    ``BLOCK_PROSE_CONTRADICTED`` when the contradicted-prose rate crosses
    ``max_contradicted_rate``.

    Inputs:
        blocks: List[Block]
        source_chunks: Dict[str, str]
            Mapping of canonical sourceId (e.g. ``dart:slug#blk_0``) to the
            chunk's plain-text body. The gate-input builder populates this
            from the staging manifest + DART chunks.jsonl body fallback; tests
            inject it directly.
        shadow: bool
            §4.3 shadow knob. When truthy, emitted ``BLOCK_PROSE_*`` issues are
            forced warning-severity and ``action`` is never ``"regenerate"``
            (compute + capture only). Wired via gate ``config.shadow: true``.
        min_block_entailment_rate / entailment_floor / contradiction_floor /
        max_contradicted_rate: Optional[float]
            Thresholds (merged from gate ``config.thresholds`` by the executor).
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per scored block.
        nli: Optional[NliClassifier]
            Test seam — inject a fake. When None, calls
            ``NliClassifier.get_or_load`` and threads the result into
            ``score_groundedness(nli=...)``.

    Graceful-degrade: ``NliClassifier.get_or_load()`` returns None AND
    ``TRAINFORGE_REQUIRE_EMBEDDINGS`` unset → single warning ``NLI_DEPS_MISSING``
    GateIssue, ``passed=True, action=None``. Strict mode raises ``RuntimeError``.
    """

    name = "block_prose_entailment"
    version = "1.0.0"

    def __init__(
        self,
        *,
        min_block_entailment_rate: float = _DEFAULT_MIN_BLOCK_ENTAILMENT_RATE,
        entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
        contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
        max_contradicted_rate: float = _DEFAULT_MAX_CONTRADICTED_RATE,
        nli: Optional[NliClassifier] = None,
    ) -> None:
        self._min_block_rate = float(min_block_entailment_rate)
        self._entailment_floor = float(entailment_floor)
        self._contradiction_floor = float(contradiction_floor)
        self._max_contradicted = float(max_contradicted_rate)
        self._nli_override = nli

    def _get_nli(self) -> Optional[NliClassifier]:
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    def _resolve_thresholds(
        self, inputs: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        """Resolve thresholds, honoring the executor's gate.config merge.

        The executor merges ``gate.config.thresholds`` into the inputs dict
        before dispatch (per the pattern at ``executor.py:1442``):

        .. code-block:: yaml

            config:
              thresholds:
                min_block_entailment_rate: 0.60
                entailment_floor: 0.70
                contradiction_floor: 0.50
                max_contradicted_rate: 0.05
        """
        return (
            float(
                inputs.get("min_block_entailment_rate", self._min_block_rate)
            ),
            float(inputs.get("entailment_floor", self._entailment_floor)),
            float(
                inputs.get("contradiction_floor", self._contradiction_floor)
            ),
            float(
                inputs.get("max_contradicted_rate", self._max_contradicted)
            ),
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        shadow = bool(inputs.get("shadow", False))

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        (
            min_block_rate,
            entailment_floor,
            contradiction_floor,
            max_contradicted,
        ) = self._resolve_thresholds(inputs)

        nli = self._get_nli()
        # Graceful-degrade: pre-check the loader exactly like claim_support so
        # the degrade branch is identical across the two NLI block gates and
        # the test seam (nli=...) overrides cleanly.
        if nli is None:
            if is_strict_mode():
                raise RuntimeError(
                    "BlockProseEntailmentValidator requires the NLI "
                    "classifier (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) "
                    "but TRAINFORGE_REQUIRE_EMBEDDINGS is set and "
                    "NliClassifier.get_or_load() returned None. "
                    "Install transformers + torch via "
                    "`pip install -e .[embedding]` or unset "
                    "TRAINFORGE_REQUIRE_EMBEDDINGS for graceful degradation."
                )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code=_CODE_NLI_DEPS_MISSING,
                        message=(
                            "NliClassifier.get_or_load() returned None; "
                            "W4 per-block prose NLI entailment gate skipped. "
                            "Install transformers + torch via "
                            "`pip install -e .[embedding]` to enable. "
                            "Set TRAINFORGE_REQUIRE_EMBEDDINGS=true to fail "
                            "closed instead."
                        ),
                        suggestion=(
                            "The graceful-degrade path is intentional for "
                            "CPU-only dev boxes; production runs should "
                            "install the embedding extras."
                        ),
                    )
                ],
                action=None,
            )

        source_chunks_raw = inputs.get("source_chunks", {}) or {}
        if not isinstance(source_chunks_raw, dict):
            source_chunks_raw = {}
        source_chunks: Dict[str, str] = {
            str(k): v for k, v in source_chunks_raw.items()
            if isinstance(v, str)
        }

        # Opt-in gate-side provenance resolution (default OFF → byte-identical).
        # When enabled AND the runner threaded a chunk_provenance_index (a
        # {provenance_ref -> [chunk_id, ...]} reverse index), a block whose
        # directly-cited semantik refs resolve to nothing is re-resolved through
        # the index. Off / absent → provenance_index stays None → legacy path.
        provenance_index: Optional[Dict[str, List[str]]] = None
        if _provenance_resolve_enabled():
            raw_index = inputs.get("chunk_provenance_index")
            if isinstance(raw_index, dict):
                provenance_index = {
                    str(k): [str(c) for c in v]
                    for k, v in raw_index.items()
                    if isinstance(v, (list, tuple))
                }

        # Severity of the BLOCK_PROSE_* failures honors the shadow knob:
        # warning during shadow (compute + capture only), critical after the
        # flip (so the validator's own ``passed`` flips on a real failure,
        # mirroring rewrite_source_grounding's ``passed = len(critical) == 0``).
        fail_severity = "warning" if shadow else "critical"

        issues: List[GateIssue] = []
        any_regenerate = False
        audited_blocks = 0
        passed_blocks = 0

        for block in blocks:
            content = block.content
            if not isinstance(content, str):
                # Outline-tier dict content — skip silently (rewrite-tier seam
                # is where prose entailment is checked).
                continue
            if _block_should_skip(block):
                # Skip silently — no event, no issue (matches claim_support).
                continue

            prose_text = _strip_html_to_text(content)
            if not prose_text.strip():
                continue

            cited_passages = _resolve_block_cited_passages(
                block, source_chunks, provenance_index
            )

            if not cited_passages:
                # No grounding surface. The resolution gate (rewrite_source_refs)
                # owns the missing-source structural failure; this gate emits a
                # warning so the operator knows the entailment signal is
                # unavailable, but passes the block.
                audited_blocks += 1
                passed_blocks += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE,
                            message=(
                                f"Rewrite-tier Block {block.block_id!r} "
                                f"(block_type={block.block_type!r}) declares "
                                f"no resolvable cited source chunks; per-block "
                                f"prose entailment cannot be evaluated."
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block,
                    passed=True,
                    failure_codes=[_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE],
                    scored_count=0,
                    entailed_count=0,
                    unsupported_count=0,
                    contradicted_count=0,
                    computational_count=0,
                    filtered_count=0,
                    windowed_count=0,
                    block_entailment_rate=0.0,
                    min_block_entailment_rate=min_block_rate,
                    entailment_floor=entailment_floor,
                    nli_revision=None,
                    nli_device=None,
                    shadow=shadow,
                )
                continue

            audited_blocks += 1
            report = score_groundedness(
                prose_text,
                cited_passages,
                nli=nli,
                entailment_floor=entailment_floor,
                contradiction_floor=contradiction_floor,
            )

            if not report.available:
                # Defensive: the pre-check above already handled the missing
                # loader, but score_groundedness can independently degrade
                # (e.g. an injected stub that resolves to None). Honor the
                # same graceful-degrade contract.
                if is_strict_mode():
                    raise RuntimeError(
                        "BlockProseEntailmentValidator: score_groundedness "
                        "reported the NLI model unavailable while "
                        "TRAINFORGE_REQUIRE_EMBEDDINGS is set. Install the "
                        "embedding extras or unset the strict flag."
                    )
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=True,
                    score=1.0,
                    issues=[
                        GateIssue(
                            severity="warning",
                            code=_CODE_NLI_DEPS_MISSING,
                            message=(
                                "score_groundedness reported the NLI model "
                                "unavailable; W4 per-block prose NLI "
                                "entailment gate skipped. Install transformers "
                                "+ torch via `pip install -e .[embedding]` to "
                                "enable."
                            ),
                        )
                    ],
                    action=None,
                )

            scored_count = report.scored_count
            entailed_count = (
                scored_count
                - report.unsupported_count
                - report.contradicted_count
            )
            windowed_count = sum(1 for c in report.claims if c.windowed)
            block_entailment_rate = report.groundedness_rate
            contradicted_rate = (
                (report.contradicted_count / scored_count)
                if scored_count > 0
                else 0.0
            )

            failure_codes: List[str] = []

            # BLOCK_PROSE_CONTRADICTED fires standalone above its rate ceiling.
            if (
                scored_count > 0
                and contradicted_rate > max_contradicted
                and len(issues) < _ISSUE_LIST_CAP
            ):
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=_CODE_BLOCK_PROSE_CONTRADICTED,
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} "
                            f"(block_type={block.block_type!r}): "
                            f"{report.contradicted_count}/{scored_count} "
                            f"({contradicted_rate:.1%}) of scored prose "
                            f"sentences were CONTRADICTED by their cited "
                            f"source chunks (contradiction >= "
                            f"{contradiction_floor:.2f}); per-block ceiling "
                            f"is {max_contradicted:.1%}. A contradiction is "
                            f"fabrication that conflicts with source — a hard "
                            f"signal regardless of total entailment rate."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "The block prose contradicts its declared source. "
                            "Re-prompt with the source chunks pinned and "
                            "require strict paraphrase."
                        ),
                    )
                )
                failure_codes.append(_CODE_BLOCK_PROSE_CONTRADICTED)

            # BLOCK_PROSE_UNGROUNDED fires when the entailment rate is below
            # the per-block floor (only when something was scored — never
            # fabricate a failure from an empty denominator).
            if (
                scored_count > 0
                and block_entailment_rate < min_block_rate
                and len(issues) < _ISSUE_LIST_CAP
            ):
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=_CODE_BLOCK_PROSE_UNGROUNDED,
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} "
                            f"(block_type={block.block_type!r}): only "
                            f"{entailed_count}/{scored_count} "
                            f"({block_entailment_rate:.1%}) of scored prose "
                            f"sentences were ENTAILED by their cited source "
                            f"chunks (entailment >= {entailment_floor:.2f} "
                            f"via windowed scorer-v2); minimum required rate "
                            f"is {min_block_rate:.1%}. Cosine grounding can "
                            f"pass on-topic fabrication that NLI catches."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "The rewrite-tier prose is on-topic but not "
                            "entailed by its declared source (likely "
                            "fabrication). Re-prompt with the source chunks "
                            "pinned and require verbatim or near-verbatim "
                            "paraphrase."
                        ),
                    )
                )
                failure_codes.append(_CODE_BLOCK_PROSE_UNGROUNDED)

            block_passed = not failure_codes
            if block_passed:
                passed_blocks += 1
            elif not shadow:
                # Only gate (regenerate) outside shadow mode.
                any_regenerate = True

            _emit_decision(
                capture, block,
                passed=block_passed,
                failure_codes=failure_codes,
                scored_count=scored_count,
                entailed_count=entailed_count,
                unsupported_count=report.unsupported_count,
                contradicted_count=report.contradicted_count,
                computational_count=report.computational_count,
                filtered_count=report.filtered_count,
                windowed_count=windowed_count,
                block_entailment_rate=block_entailment_rate,
                min_block_entailment_rate=min_block_rate,
                entailment_floor=entailment_floor,
                nli_revision=report.nli_model_revision,
                nli_device=report.nli_device,
                shadow=shadow,
            )

        # ``passed`` honors only critical-severity issues — during shadow the
        # BLOCK_PROSE_* failures are warning-severity so ``passed`` stays True;
        # after the critical flip they are critical-severity and ``passed``
        # flips on a real failure (mirrors rewrite_source_grounding).
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = (
            1.0
            if audited_blocks == 0
            else round(passed_blocks / audited_blocks, 4)
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action="regenerate" if any_regenerate else None,
        )


__all__ = [
    "BlockProseEntailmentValidator",
    "load_chunk_provenance_index",
    "_DEFAULT_MIN_BLOCK_ENTAILMENT_RATE",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_MAX_CONTRADICTED_RATE",
    "_CODE_BLOCK_PROSE_UNGROUNDED",
    "_CODE_BLOCK_PROSE_CONTRADICTED",
    "_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE",
    "_CODE_NLI_DEPS_MISSING",
]
