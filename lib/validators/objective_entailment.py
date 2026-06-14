"""W4 — ``ObjectiveEntailmentValidator`` (per-LO NLI entailment at course_planning).

The GATE counterpart to W2's in-synthesis NLI-grounding filter. W2 filters
candidate objectives at GENERATION time (decompose → map → NLI-grounding-filter
→ dedup → reconcile) so ungrounded candidates never become objectives; this
validator GATES the synthesized LO set at ``course_planning`` AFTER it lands on
disk in ``synthesized_objectives.json``. Same NLI machinery
(``score_groundedness`` + the ``NliClassifier`` singleton), different stage.
They are complementary: W2 reduces the regen load W4 would carry; W4 catches
anything W2 let through AND the case where W2 isn't run at all (e.g.
``--reuse-objectives``).

Sits ALONGSIDE the resolution-only ``ObjectiveSourceRefValidator`` (which checks
that an LO's ``source_refs[]`` resolve against a real upstream universe). This
validator additionally checks that the LO *statement* is ENTAILED by the text
of its cited chunk(s) — a synthesized objective with a valid-resolving
``chunk_ids[]`` but a statement the chunk does not entail (a fabricated-on-topic
objective) passes the resolution gate but fails this entailment gate.

Algorithm (per ``plans/finegrain/w4-nli-grounding-gate.md`` §2.2):

1. Read structured ``source_refs[] = [{ref, chunk_ids[]}]``. Skip LOs with
   legacy ``List[str]`` source_refs or empty refs (the resolution gate
   ``objective_source_refs`` owns those — record ``attribution_shape`` and
   continue, no entailment verdict). A TO with empty refs honors
   ``require_to_entailment`` (default skip, mirroring ``require_to_attribution``).
2. Build the cited-chunk premise set: union of ``chunk_ids[]`` across the LO's
   ``source_refs[]``, resolved to text via the id→text map (``§2.2`` loader,
   keying on BOTH the chunk top-level ``id`` AND each
   ``source.source_references[].sourceId``). Wrap as passages.
3. Score the LO ``statement`` as a single hypothesis via
   ``score_groundedness`` (windowed rescue matters — an LO paraphrases across a
   section so the whole-chunk premise often misses).
4. Verdict:
   - ``OBJECTIVE_UNENTAILED`` when ``scored_count > 0`` AND
     ``groundedness_rate < objective_entailment_rate_floor`` (default 1.0 — an
     LO is one hypothesis, binary entailed/not) AND not contradicted.
   - ``OBJECTIVE_CONTRADICTED`` when ``contradicted_count > 0`` (hard signal).
   - LO with no resolvable cited chunk text → warning
     ``OBJECTIVE_NO_GROUNDING_SOURCE``, ``passed=True``.

Graceful-degrade (§2.3): identical to ``claim_support`` — pre-check
``NliClassifier.get_or_load()``; non-strict → warning ``NLI_DEPS_MISSING``,
``passed=True``; strict (``TRAINFORGE_REQUIRE_EMBEDDINGS``) → raise. Additionally
mirror ``objective_source_refs``'s ``OBJECTIVE_SOURCE_REFS_NO_UNIVERSE``: when
``dart_chunks_manifest_path`` is absent OR the id→text map is empty, emit a
warning, ``passed=True`` (no universe to entail against).

**Shadow knob (§4.3):** ``inputs["shadow"]`` (merged from gate ``config.shadow``).
When truthy, ``OBJECTIVE_*`` issues are forced warning-severity and ``action``
is never ``"regenerate"`` — compute + capture only. The critical flip is
DEFERRED (``# TODO(calibration)`` marker at the gate in ``config/workflows.yaml``).

Decision-capture (§2.4): one ``objective_entailment_check`` event per audited LO.

Reuse contract (§8): NLI singleton + ``score_groundedness``; LO flatten
(``abcd_objective._flatten_objectives``); chunks.jsonl walk pattern
(``objective_source_refs._load_dart_chunks_universe``, extended to an
id→text sibling here); strict-mode flag (``is_strict_mode``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier
from lib.embedding.sentence_embedder import is_strict_mode
from lib.retrieval.groundedness import score_groundedness
from lib.validators.abcd_objective import _flatten_objectives
from lib.validators.objective_source_refs import (
    _classify_attribution_shape,
    _coerce_objectives_path,
    _hierarchy_level,
    _lo_id,
)

logger = logging.getLogger(__name__)


#: Per-claim entailment / contradiction floors threaded into
#: ``score_groundedness`` (reuse claim_support defaults).
_DEFAULT_ENTAILMENT_FLOOR: float = 0.70
_DEFAULT_CONTRADICTION_FLOOR: float = 0.50

#: Per-LO entailment rate floor. An LO is one hypothesis so the rate is
#: effectively binary (0.0 or 1.0); a multi-sentence statement fails when its
#: ``groundedness_rate`` falls below this. Default 1.0 (calibratable — §4).
_DEFAULT_OBJECTIVE_ENTAILMENT_RATE_FLOOR: float = 1.0

#: Cap per-LO issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50


# --------------------------------------------------------------------- #
# Canonical GateIssue codes
# --------------------------------------------------------------------- #

_CODE_OBJECTIVE_UNENTAILED: str = "OBJECTIVE_UNENTAILED"
_CODE_OBJECTIVE_CONTRADICTED: str = "OBJECTIVE_CONTRADICTED"
_CODE_OBJECTIVE_NO_GROUNDING_SOURCE: str = "OBJECTIVE_NO_GROUNDING_SOURCE"
_CODE_NLI_DEPS_MISSING: str = "NLI_DEPS_MISSING"

_DECISION_TYPE: str = "objective_entailment_check"


def _iter_chunks_jsonl(chunks_jsonl_path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield each parsed chunk object from a sibling ``chunks.jsonl``.

    Shared JSONL-walk generator factored so both the resolution gate's
    set-only universe loader and this gate's id→text loader read the same
    surface. Malformed lines are skipped (defensive — a single bad chunk
    doesn't break the whole universe build).
    """
    if not chunks_jsonl_path.exists():
        return
    try:
        with chunks_jsonl_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    logger.debug(
                        "ObjectiveEntailmentValidator: malformed chunk on "
                        "line %d of %s; skipping.",
                        line_no, chunks_jsonl_path,
                    )
                    continue
                if isinstance(chunk, Mapping):
                    yield chunk
    except OSError as exc:
        logger.warning(
            "ObjectiveEntailmentValidator: failed to read chunks.jsonl at "
            "%s: %s",
            chunks_jsonl_path, exc,
        )
        return


def _chunk_text(chunk: Mapping[str, Any]) -> str:
    """Best-effort read of a chunk's plain-text body.

    DART / IMSCC chunks carry the body on ``text`` (canonical); older
    surfaces used ``body``. Returns ``""`` when neither is a non-empty str.
    """
    for key in ("text", "body"):
        val = chunk.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _load_dart_chunks_text_map(
    dart_chunks_manifest_path: Optional[Path],
) -> Dict[str, str]:
    """Build an id→TEXT map from the DART chunkset sibling ``chunks.jsonl``.

    Sibling of ``objective_source_refs._load_dart_chunks_universe`` (which
    returns ids only). Keys EACH chunk's text on BOTH:

    * the chunk's top-level ``"id"`` (e.g. ``demo_course_chunk_00001``), and
    * every ``source.source_references[].sourceId`` (the
      ``dart:{slug}#{block_id}`` shape),

    so an LO ``chunk_ids[]`` entry resolves whichever id form it uses. The
    set-only universe loader is left intact for the resolution gate — this is
    an additive text-returning sibling, not a fork.

    Returns an empty map when the manifest path is missing / unreadable OR the
    sibling chunks.jsonl is missing / empty.
    """
    if dart_chunks_manifest_path is None:
        return {}
    if not dart_chunks_manifest_path.exists():
        return {}
    chunks_jsonl_path = dart_chunks_manifest_path.parent / "chunks.jsonl"
    text_map: Dict[str, str] = {}
    for chunk in _iter_chunks_jsonl(chunks_jsonl_path):
        body = _chunk_text(chunk)
        if not body:
            continue
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id.strip():
            text_map.setdefault(chunk_id.strip(), body)
        source = chunk.get("source")
        if not isinstance(source, Mapping):
            continue
        src_refs = source.get("source_references") or []
        if not isinstance(src_refs, list):
            continue
        for ref in src_refs:
            if not isinstance(ref, Mapping):
                continue
            src_id = ref.get("sourceId")
            if isinstance(src_id, str) and src_id.strip():
                text_map.setdefault(src_id.strip(), body)
    return text_map


def _lo_statement(lo: Mapping[str, Any]) -> str:
    """Pull the LO's hypothesis text (``statement`` / ``text``)."""
    for key in ("statement", "text"):
        val = lo.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _resolve_lo_cited_passages(
    source_refs: Any,
    text_map: Dict[str, str],
) -> List[Dict[str, str]]:
    """Resolve the union of an LO's structured ``chunk_ids[]`` to passages.

    Walks every structured ``{ref, chunk_ids[]}`` entry, unions the
    ``chunk_ids[]``, resolves each to text via ``text_map``, and returns
    ``[{"chunk_id": cid, "text": text}, ...]`` (de-duplicated, first wins).
    """
    if not isinstance(source_refs, list):
        return []
    seen: set = set()
    passages: List[Dict[str, str]] = []
    for entry in source_refs:
        if not isinstance(entry, dict):
            continue
        chunk_ids = entry.get("chunk_ids") or []
        if not isinstance(chunk_ids, list):
            continue
        for cid in chunk_ids:
            if not isinstance(cid, str):
                continue
            cid_stripped = cid.strip()
            if not cid_stripped or cid_stripped in seen:
                continue
            seen.add(cid_stripped)
            text = text_map.get(cid_stripped)
            if isinstance(text, str) and text.strip():
                passages.append({"chunk_id": cid_stripped, "text": text})
    return passages


def _emit_decision(
    capture: Any,
    *,
    lo_id_value: str,
    hierarchy: str,
    attribution_shape: str,
    cited_chunk_count: int,
    scored_count: int,
    entailed: bool,
    contradicted: bool,
    windowed: bool,
    groundedness_rate: float,
    entailment_floor: float,
    nli_revision: Optional[str],
    nli_device: Optional[str],
    verdict_code: str,
    shadow: bool,
) -> None:
    """Emit one ``objective_entailment_check`` decision per audited LO.

    Rationale interpolates dynamic per-LO signals (≥20 chars). Swallow
    ``log_decision`` exceptions (capture must not break the gate).
    """
    if capture is None:
        return
    decision = "passed" if verdict_code == "passed" else f"failed:{verdict_code}"
    rationale = (
        f"LO-statement NLI entailment on {lo_id_value} ({hierarchy}-level): "
        f"attribution_shape={attribution_shape!r}, "
        f"cited_chunk_count={cited_chunk_count}, "
        f"scored_count={scored_count}, "
        f"entailed={entailed}, contradicted={contradicted}, "
        f"windowed={windowed}, "
        f"groundedness_rate={groundedness_rate:.4f}, "
        f"entailment_floor={entailment_floor:.2f}, "
        f"nli_revision={nli_revision}, nli_device={nli_device}, "
        f"shadow={shadow}, verdict_code={verdict_code}."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.warning(
            "ObjectiveEntailmentValidator: decision_capture.log_decision "
            "failed: %s",
            exc,
        )


class ObjectiveEntailmentValidator:
    """Per-LO NLI entailment gate at ``course_planning`` (W4).

    Walks the synthesized LO set; for each LO with structured
    ``source_refs[]`` resolving to cited chunk text, scores the LO statement
    as a single hypothesis via ``score_groundedness``. Fires
    ``OBJECTIVE_UNENTAILED`` when the statement is not entailed and
    ``OBJECTIVE_CONTRADICTED`` when it conflicts with its cited chunk.

    Inputs (mirror ``objective_source_refs`` so the existing
    ``_build_objective_source_refs`` builder feeds it):
        synthesized_objectives_path: str | Path (required; or ``objectives``
            for test ergonomics — flattened via ``_flatten_objectives``).
        dart_chunks_manifest_path: Optional[str | Path] — id→text map source.
        textbook_structure_path: Optional (accepted for builder-shape compat;
            unused for entailment).
        require_to_entailment: bool — opt-in (mirrors require_to_attribution).
        shadow: bool — §4.3 knob (warning-only + no regenerate when truthy).
        decision_capture, gate_id: standard.
        nli: Optional[NliClassifier] — TEST SEAM.

    Graceful-degrade: missing NLI deps → warning ``NLI_DEPS_MISSING``
    (passed=True) / strict → raise; missing chunk universe → warning
    ``OBJECTIVE_NO_GROUNDING_SOURCE`` (passed=True).
    """

    name = "objective_entailment"
    version = "1.0.0"

    def __init__(
        self,
        *,
        entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
        contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
        objective_entailment_rate_floor: float = (
            _DEFAULT_OBJECTIVE_ENTAILMENT_RATE_FLOOR
        ),
        require_to_entailment: bool = False,
        nli: Optional[NliClassifier] = None,
    ) -> None:
        self._entailment_floor = float(entailment_floor)
        self._contradiction_floor = float(contradiction_floor)
        self._rate_floor = float(objective_entailment_rate_floor)
        self._require_to = bool(require_to_entailment)
        self._nli_override = nli

    def _get_nli(self) -> Optional[NliClassifier]:
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    def _resolve_thresholds(
        self, inputs: Dict[str, Any]
    ) -> Tuple[float, float, float, bool]:
        """Resolve thresholds, honoring the executor's gate.config merge."""
        return (
            float(inputs.get("entailment_floor", self._entailment_floor)),
            float(
                inputs.get("contradiction_floor", self._contradiction_floor)
            ),
            float(
                inputs.get("objective_entailment_rate_floor", self._rate_floor)
            ),
            bool(inputs.get("require_to_entailment", self._require_to)),
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")
        shadow = bool(inputs.get("shadow", False))

        objectives, err = _coerce_objectives_path(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Empty LO set is a no-op pass (back-compat with empty-corpus path).
        if not objectives:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        (
            entailment_floor,
            contradiction_floor,
            rate_floor,
            require_to,
        ) = self._resolve_thresholds(inputs)

        nli = self._get_nli()
        if nli is None:
            if is_strict_mode():
                raise RuntimeError(
                    "ObjectiveEntailmentValidator requires the NLI classifier "
                    "(MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) but "
                    "TRAINFORGE_REQUIRE_EMBEDDINGS is set and "
                    "NliClassifier.get_or_load() returned None. Install "
                    "transformers + torch via `pip install -e .[embedding]` or "
                    "unset TRAINFORGE_REQUIRE_EMBEDDINGS for graceful "
                    "degradation."
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
                            "NliClassifier.get_or_load() returned None; W4 "
                            "per-LO NLI entailment gate skipped. Install "
                            "transformers + torch via "
                            "`pip install -e .[embedding]` to enable. Set "
                            "TRAINFORGE_REQUIRE_EMBEDDINGS=true to fail closed "
                            "instead."
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

        # Resolve the id→text universe. Absent / empty → graceful-degrade
        # warning (no universe to entail against; the resolution gate owns the
        # missing-chunk structural failure).
        chunks_path_raw = inputs.get("dart_chunks_manifest_path")
        chunks_path = Path(chunks_path_raw) if chunks_path_raw else None
        text_map = _load_dart_chunks_text_map(chunks_path)
        if not text_map:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code=_CODE_OBJECTIVE_NO_GROUNDING_SOURCE,
                        message=(
                            "ObjectiveEntailmentValidator could not resolve an "
                            "id→text chunk universe (no "
                            "``dart_chunks_manifest_path`` or empty sibling "
                            "chunks.jsonl). Gate passes (graceful-degrade — "
                            "the resolution gate objective_source_refs owns "
                            "the missing-chunk structural failure)."
                        ),
                        suggestion=(
                            "Wire chunking.dart_chunks_path into this gate's "
                            "input builder so the LO statements can be scored "
                            "against their cited chunk text."
                        ),
                    )
                ],
                action=None,
            )

        fail_severity = "warning" if shadow else "critical"

        issues: List[GateIssue] = []
        any_regenerate = False
        audited = 0
        passed_count = 0

        for lo in objectives:
            if not isinstance(lo, Mapping):
                continue
            lo_id_value = _lo_id(lo)
            hierarchy = _hierarchy_level(lo_id_value)
            source_refs = lo.get("source_refs")
            shape = _classify_attribution_shape(source_refs)

            # Skip legacy List[str] / empty refs — the resolution gate owns
            # those. A TO with empty refs honors require_to_entailment.
            if shape == "legacy_string_list":
                continue
            if shape == "empty":
                if hierarchy == "terminal" and not require_to:
                    continue
                # A chapter-level LO (or a TO under require_to_entailment) with
                # no structured refs has nothing to entail against — the
                # resolution gate flags the missing refs; we skip the
                # entailment verdict (no premise) without firing.
                continue

            statement = _lo_statement(lo)
            if not statement:
                continue

            cited_passages = _resolve_lo_cited_passages(source_refs, text_map)
            audited += 1

            if not cited_passages:
                # Structured refs present but no chunk_id resolved to text.
                # The resolution gate owns the missing-chunk failure; this gate
                # warns + passes the LO.
                passed_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=_CODE_OBJECTIVE_NO_GROUNDING_SOURCE,
                            message=(
                                f"LO {lo_id_value!r} ({hierarchy}-level) "
                                f"declares structured source_refs but none of "
                                f"its chunk_ids resolved to chunk text; "
                                f"LO-statement entailment cannot be evaluated."
                            ),
                            location=lo_id_value,
                        )
                    )
                _emit_decision(
                    capture,
                    lo_id_value=lo_id_value,
                    hierarchy=hierarchy,
                    attribution_shape=shape,
                    cited_chunk_count=0,
                    scored_count=0,
                    entailed=False,
                    contradicted=False,
                    windowed=False,
                    groundedness_rate=0.0,
                    entailment_floor=entailment_floor,
                    nli_revision=None,
                    nli_device=None,
                    verdict_code=_CODE_OBJECTIVE_NO_GROUNDING_SOURCE,
                    shadow=shadow,
                )
                continue

            report = score_groundedness(
                statement,
                cited_passages,
                nli=nli,
                entailment_floor=entailment_floor,
                contradiction_floor=contradiction_floor,
            )

            if not report.available:
                if is_strict_mode():
                    raise RuntimeError(
                        "ObjectiveEntailmentValidator: score_groundedness "
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
                                "unavailable; W4 per-LO NLI entailment gate "
                                "skipped."
                            ),
                        )
                    ],
                    action=None,
                )

            scored_count = report.scored_count
            contradicted = report.contradicted_count > 0
            windowed = any(c.windowed for c in report.claims)
            rate = report.groundedness_rate
            entailed = scored_count > 0 and rate >= rate_floor

            verdict_code = "passed"
            lo_passed = True

            if scored_count > 0 and contradicted and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=_CODE_OBJECTIVE_CONTRADICTED,
                        message=(
                            f"LO {lo_id_value!r} ({hierarchy}-level) statement "
                            f"is CONTRADICTED by its cited chunk(s) "
                            f"(contradiction >= {contradiction_floor:.2f}); "
                            f"the synthesized objective conflicts with the "
                            f"source it cites — a hard fabrication signal."
                        ),
                        location=lo_id_value,
                        suggestion=(
                            "Re-run the LO synthesizer with the cited chunk "
                            "text pinned; the objective must not assert "
                            "anything its source contradicts."
                        ),
                    )
                )
                verdict_code = _CODE_OBJECTIVE_CONTRADICTED
                lo_passed = False
            elif (
                scored_count > 0
                and rate < rate_floor
                and len(issues) < _ISSUE_LIST_CAP
            ):
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=_CODE_OBJECTIVE_UNENTAILED,
                        message=(
                            f"LO {lo_id_value!r} ({hierarchy}-level) statement "
                            f"is NOT entailed by its cited chunk(s) "
                            f"(entailment rate {rate:.1%} < required "
                            f"{rate_floor:.1%} at floor "
                            f"{entailment_floor:.2f}, windowed scorer-v2). "
                            f"A valid-resolving chunk_id on a fabricated "
                            f"objective passes the resolution gate but fails "
                            f"this entailment gate."
                        ),
                        location=lo_id_value,
                        suggestion=(
                            "Re-run the LO synthesizer with the cited chunk "
                            "text pinned and require the objective statement "
                            "to be supported by the chunk it cites."
                        ),
                    )
                )
                verdict_code = _CODE_OBJECTIVE_UNENTAILED
                lo_passed = False

            if lo_passed:
                passed_count += 1
            elif not shadow:
                any_regenerate = True

            _emit_decision(
                capture,
                lo_id_value=lo_id_value,
                hierarchy=hierarchy,
                attribution_shape=shape,
                cited_chunk_count=len(cited_passages),
                scored_count=scored_count,
                entailed=entailed,
                contradicted=contradicted,
                windowed=windowed,
                groundedness_rate=rate,
                entailment_floor=entailment_floor,
                nli_revision=report.nli_model_revision,
                nli_device=report.nli_device,
                verdict_code=verdict_code,
                shadow=shadow,
            )

        # ``passed`` honors only critical-severity issues (warning during
        # shadow; critical after the flip).
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
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
    "ObjectiveEntailmentValidator",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_OBJECTIVE_ENTAILMENT_RATE_FLOOR",
    "_CODE_OBJECTIVE_UNENTAILED",
    "_CODE_OBJECTIVE_CONTRADICTED",
    "_CODE_OBJECTIVE_NO_GROUNDING_SOURCE",
    "_CODE_NLI_DEPS_MISSING",
    "_load_dart_chunks_text_map",
]
