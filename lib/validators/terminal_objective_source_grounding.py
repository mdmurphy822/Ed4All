"""W7.5 (M-TO) — TerminalObjectiveSourceGroundingValidator (course_planning).

No existing gate scores a TERMINAL objective against its SOURCE text. The
``terminal_objective_coverage`` gate is a structural roll-up ("does every TO have
>=1 CO pointing at it / does its chapter resolve?"); ``co_terminal_alignment``
scores each CHAPTER objective against its assigned TO STATEMENT (a statement<->
statement check, not statement<->source). So a coherent-but-HALLUCINATED terminal
objective — the "insurance TO-15" class where a semantic-outlier CO seeds a
wholesale-invented course-wide terminal competency (see
``ED4ALL_TO_CLUSTER_GUARDS`` in root ``CLAUDE.md``) — passes every gate: it is
statement-coherent AND its child COs carry ``terminal_id`` backlinks, so nothing
ever asks "is this TO actually grounded in the source chunks its cluster cites?".

This validator closes that gap. For each terminal objective it gathers the
SOURCE CHUNKS assigned to its cluster (the union of its child COs' cited chunks —
the REAL cluster->chunk assignment, read from the persisted objectives doc's
``source_refs[].chunk_ids`` / child-CO ``source_chunk_ids``, OR from the
planning-side ``cluster_signals`` surfaced by
``_derive_terminals_bottom_up``), embeds the TO statement + each chunk's text,
and takes the MAX cosine. A TO whose best supporting chunk is below a floor (or
that resolves to NO supporting chunk) is flagged ``TO_SOURCE_UNGROUNDED`` — it
has no adequately-grounded supporting chunk, i.e. the source does not establish
the competency the TO claims.

**Anti-fabrication:** the validator reads only REAL cluster->chunk assignments
(the child COs' actually-cited chunk ids) and REAL chunk text; it never invents a
chunk or a grounding link.

Gating: opt-in via ``ED4ALL_TO_SOURCE_GROUNDING`` (default OFF → byte-identical
no-op pass with a single ``TO_SOURCE_GROUNDING_DISABLED`` info issue,
parse-with-fallback). When ON, ALL issues are **warning-severity** day-1
(``passed`` always True) with a ``# TODO(calibration)`` deferred critical-flip
after a >=2-corpus FP measurement (WS3/W4 deferred-flip pattern; NOT IB3), since
on the current collapsed-structure corpus the weak-grounding rate is high.

Inputs (``inputs`` dict; mirrors ``co_terminal_alignment`` IO):

    synthesized_objectives_path (str) OR synthesized_objectives (dict)
        The course's synthesized objectives doc.
    chunks_by_id (dict) OR chunks (list of chunk dicts) OR dart_chunks_path (str)
        The source-chunk universe (id -> text). Absent → graceful skip
        (TO_SOURCE_NO_CHUNKSET warning, passed=True).
    cluster_signals (dict, optional)
        Planning-side signals from ``_derive_terminals_bottom_up``; when it
        carries ``to_source_chunk_assignments`` (``{to_id: [chunk_ids]}``) that
        map is the AUTHORITATIVE cluster->chunk assignment (falls back to the
        objectives-doc reconstruction otherwise).
    threshold (float, TOP-LEVEL config key)
        Per-TO max-cosine grounding floor (default 0.35).
    embedder (Optional[SentenceEmbedder])
        Test seam. Defaults to ``try_load_embedder()``.
    decision_capture / capture
        Injected by the executor. One ``validation_result`` decision per
        audited TO.

References:
    - ``lib/validators/co_terminal_alignment.py`` — embedding-validator class
      template + objectives IO.
    - ``lib/validators/assessment_retrieval_grounding.py`` — the answer-side
      source-grounding gate this mirrors for the TO<->source axis.
    - ``MCP/tools/pipeline_tools.py::_derive_terminals_bottom_up`` — surfaces
      the ``cluster_signals`` (incl. ``to_source_chunk_assignments``) consumed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    SentenceEmbedder,
    is_strict_mode,
    try_load_embedder,
)
from lib.ontology.terminal_coverage import (
    _PARENT_TERMINAL_KEYS,
    flatten_chapter_objectives,
)
from lib.validators.terminal_objective_coverage import _coerce_dict

logger = logging.getLogger(__name__)

#: Per-TO max-cosine grounding floor. A TO whose best supporting chunk is below
#: this is flagged ungrounded. Lower than the 0.45 CO<->TO statement floor
#: because this compares a short TO statement to raw (longer, noisier) chunk
#: text — the answer-side grounding floors sit near 0.30 for the same reason.
DEFAULT_THRESHOLD: float = 0.35

#: Cap on the emitted issue list (mirrors co_terminal_alignment).
_ISSUE_LIST_CAP: int = 50

_DECISION_TYPE: str = "validation_result"

#: Opt-in gate flag. Default OFF → byte-identical no-op pass.
_ENV_FLAG: str = "ED4ALL_TO_SOURCE_GROUNDING"
_TRUTHY: frozenset = frozenset({"1", "true", "yes", "on"})


def _to_source_grounding_enabled() -> bool:
    """Resolve ``ED4ALL_TO_SOURCE_GROUNDING`` (parse-with-fallback → off)."""
    raw = os.environ.get(_ENV_FLAG, "")
    return str(raw).strip().lower() in _TRUTHY


def _statement(obj: Dict[str, Any]) -> str:
    """Return an objective's statement text, tolerating common field names."""
    for key in ("statement", "text", "objective", "description"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _resolve_parent_terminal(co: Dict[str, Any]) -> Optional[str]:
    """Return the lower-cased terminal id a CO is assigned to, or None."""
    for key in _PARENT_TERMINAL_KEYS:
        val = co.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return None


def _build_chunks_lookup(inputs: Dict[str, Any]) -> Dict[str, str]:
    """Resolve an id -> text chunk lookup from the (dict | list | path) inputs.

    Accepts ``chunks_by_id`` (canonical), ``chunks`` (list of chunk dicts), or
    ``dart_chunks_path`` (a chunks.jsonl). Returns ``{}`` when none resolve.
    """
    raw = inputs.get("chunks_by_id")
    if isinstance(raw, dict) and raw:
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(v, str):
                out[str(k)] = v
            elif isinstance(v, dict):
                txt = v.get("text") or v.get("content")
                if isinstance(txt, str):
                    out[str(k)] = txt
        if out:
            return out

    def _from_chunk_list(chunks: List[Any]) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for c in chunks:
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or c.get("chunk_id")
            txt = c.get("text") or c.get("content")
            if isinstance(cid, str) and cid.strip() and isinstance(txt, str):
                lookup[cid.strip()] = txt
        return lookup

    raw_list = inputs.get("chunks")
    if isinstance(raw_list, list) and raw_list:
        lookup = _from_chunk_list(raw_list)
        if lookup:
            return lookup

    path = inputs.get("dart_chunks_path")
    if path:
        p = Path(path)
        if p.exists():
            chunks: List[Any] = []
            try:
                with p.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunks.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                chunks = []
            return _from_chunk_list(chunks)
    return {}


class TerminalObjectiveSourceGroundingValidator:
    """W7.5 (M-TO) — TO<->source-chunk grounding gate at ``course_planning``.

    Opt-in via ``ED4ALL_TO_SOURCE_GROUNDING`` (default OFF → no-op pass). When
    ON, embeds each TO statement + the source chunks its cluster cites, takes
    the max cosine, and flags ``TO_SOURCE_UNGROUNDED`` when a TO has no
    adequately-grounded supporting chunk. Warning-severity day-1; critical-flip
    deferred (``# TODO(calibration)``).
    """

    name = "terminal_objective_source_grounding"
    version = "0.1.0"  # W7.5

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        embedder: Optional[SentenceEmbedder] = None,
    ) -> None:
        self._threshold = threshold
        self._embedder_override = embedder

    # --------------------------------------------------------------- validate

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")

        # Opt-in gate: default OFF → byte-identical no-op pass.
        if not _to_source_grounding_enabled():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code="TO_SOURCE_GROUNDING_DISABLED",
                        message=(
                            "ED4ALL_TO_SOURCE_GROUNDING is off; terminal-"
                            "objective source-grounding audit skipped."
                        ),
                    )
                ],
            )

        threshold = float(inputs.get("threshold", self._threshold))

        objectives, err = _coerce_dict(
            inputs,
            explicit_key="synthesized_objectives",
            path_key="synthesized_objectives_path",
            code_prefix="TO_SOURCE_GROUNDING_OBJECTIVES",
        )
        if err is not None or objectives is None:
            # Unresolvable objectives → graceful no-op pass (never block).
            return self._pass(gate_id, [])

        terminals = [
            t
            for t in (
                objectives.get("terminal_objectives")
                or objectives.get("terminal_outcomes")
                or []
            )
            if isinstance(t, dict)
        ]
        chapter_objectives = (
            objectives.get("chapter_objectives")
            or objectives.get("component_objectives")
            or []
        )
        cos = flatten_chapter_objectives(chapter_objectives)
        if not terminals:
            return self._pass(gate_id, [])

        # Build the cluster->chunk assignment map (REAL assignments only).
        assignments = self._resolve_assignments(inputs, terminals, cos)

        chunks_lookup = _build_chunks_lookup(inputs)
        if not chunks_lookup:
            # No chunkset → graceful skip (can't score grounding).
            return self._pass(
                gate_id,
                [
                    GateIssue(
                        severity="warning",
                        code="TO_SOURCE_NO_CHUNKSET",
                        message=(
                            "No source chunkset resolved "
                            "(chunks_by_id / chunks / dart_chunks_path); "
                            "TO source-grounding audit skipped."
                        ),
                    )
                ],
            )

        embedder_strict = is_strict_mode()
        embedder = self._embedder_override or try_load_embedder()
        if embedder is None:
            self._emit_decision(
                capture,
                to_id="*",
                passed=True,
                code="EMBEDDING_DEPS_MISSING",
                cosine=None,
                threshold=threshold,
                n_chunks=0,
                embedder_strict=embedder_strict,
            )
            return self._pass(
                gate_id,
                [
                    GateIssue(
                        severity="warning",
                        code="EMBEDDING_DEPS_MISSING",
                        message=(
                            "sentence-transformers extras not installed; TO "
                            "source-grounding gate skipped. Install via "
                            "`pip install -e .[embedding]` to enable."
                        ),
                    )
                ],
            )

        issues: List[GateIssue] = []
        # Memoize chunk-text encodes across TOs (clusters share few chunks but
        # a chunk may back >1 TO in degenerate assignments).
        chunk_vec_cache: Dict[str, Any] = {}
        audited = 0
        ungrounded = 0
        passed_count = 0

        for to in terminals:
            tid_raw = to.get("id")
            if not isinstance(tid_raw, str) or not tid_raw.strip():
                continue
            tid = tid_raw.strip()
            to_stmt = _statement(to)
            if not to_stmt:
                continue
            audited += 1
            chunk_ids = assignments.get(tid.lower(), [])
            # Resolve the chunk texts (real assignments; missing ids skipped).
            texts = [
                chunks_lookup[c] for c in chunk_ids if c in chunks_lookup
            ]

            best: Optional[float] = None
            if texts:
                try:
                    to_vec = embedder.encode(to_stmt)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("embedder.encode raised on TO %s: %s", tid, exc)
                    to_vec = None
                if to_vec is not None:
                    for cid, txt in zip(
                        [c for c in chunk_ids if c in chunks_lookup], texts
                    ):
                        vec = chunk_vec_cache.get(cid)
                        if vec is None:
                            try:
                                vec = embedder.encode(txt)
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "embedder.encode raised on chunk %s: %s",
                                    cid, exc,
                                )
                                vec = None
                            chunk_vec_cache[cid] = vec
                        if vec is None:
                            continue
                        c_sim = cosine_similarity(to_vec, vec)
                        if best is None or c_sim > best:
                            best = c_sim

            grounded = best is not None and best >= threshold
            if grounded:
                passed_count += 1
                self._emit_decision(
                    capture,
                    to_id=tid,
                    passed=True,
                    code=None,
                    cosine=best,
                    threshold=threshold,
                    n_chunks=len(texts),
                    embedder_strict=embedder_strict,
                )
                continue

            ungrounded += 1
            if len(issues) < _ISSUE_LIST_CAP:
                if not texts:
                    detail = (
                        "resolves to NO supporting source chunk "
                        f"(cluster cited {len(chunk_ids)} chunk id(s), none in "
                        f"the chunkset)"
                    )
                else:
                    detail = (
                        f"best supporting chunk cosine "
                        f"{best:.4f} < floor {threshold:.4f} "
                        f"(over {len(texts)} cited chunk(s))"
                    )
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="TO_SOURCE_UNGROUNDED",
                        message=(
                            f"Terminal objective {tid!r} {detail}. The source "
                            f"may not establish the competency this TO claims "
                            f"(coherent-but-hallucinated-TO risk)."
                        ),
                        location=f"terminal_objectives[id={tid}]",
                        suggestion=(
                            "Re-derive TOs bottom-up from grounded CO clusters "
                            "(ED4ALL_TO_CLUSTER_GUARDS absorbs semantic-outlier "
                            "singletons) or drop the ungrounded TO."
                        ),
                    )
                )
            self._emit_decision(
                capture,
                to_id=tid,
                passed=False,
                code="TO_SOURCE_UNGROUNDED",
                cosine=best,
                threshold=threshold,
                n_chunks=len(texts),
                embedder_strict=embedder_strict,
            )

        ungrounded_rate = (ungrounded / audited) if audited else 0.0
        if audited and ungrounded_rate > 0.30:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="TO_SOURCE_UNGROUNDED_RATE_HIGH",
                    message=(
                        f"{ungrounded}/{audited} terminal objective(s) "
                        f"({ungrounded_rate:.2%}) have no adequately-grounded "
                        f"supporting chunk (max cosine < {threshold:.4f}). "
                        f"Bottom-up TO derivation from grounded clusters is the "
                        f"cure; this gate is detection only."
                    ),
                    location="terminal_objectives",
                )
            )

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0  # warning-day-1 → always True
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _resolve_assignments(
        inputs: Dict[str, Any],
        terminals: List[Dict[str, Any]],
        cos: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Build ``{to_id_lower: [chunk_ids]}`` from REAL cluster assignments.

        Precedence:
          1. ``cluster_signals["to_source_chunk_assignments"]`` — the
             planning-side map surfaced by ``_derive_terminals_bottom_up``.
          2. Each TO's ``source_refs[].chunk_ids`` (populated by the
             child-ref annotation pass — union of child COs' cited chunks).
          3. Reconstruct from the child COs' ``source_chunk_ids`` keyed by the
             CO's parent-terminal backlink.

        Anti-fabrication: only chunk ids the objectives already cite are used.
        """
        # (1) planning-side authoritative map.
        signals = inputs.get("cluster_signals")
        if isinstance(signals, dict):
            planned = signals.get("to_source_chunk_assignments")
            if isinstance(planned, dict) and planned:
                out: Dict[str, List[str]] = {}
                for tid, ids in planned.items():
                    if isinstance(ids, list):
                        out[str(tid).strip().lower()] = [
                            str(c) for c in ids if str(c).strip()
                        ]
                if out:
                    return out

        assignments: Dict[str, List[str]] = {}
        seen: Dict[str, set] = {}

        # (2) TO source_refs[].chunk_ids.
        for to in terminals:
            tid = to.get("id")
            if not isinstance(tid, str) or not tid.strip():
                continue
            key = tid.strip().lower()
            assignments.setdefault(key, [])
            seen.setdefault(key, set())
            for ref in to.get("source_refs") or []:
                if not isinstance(ref, dict):
                    continue
                for c in ref.get("chunk_ids") or []:
                    cs = str(c).strip()
                    if cs and cs not in seen[key]:
                        seen[key].add(cs)
                        assignments[key].append(cs)

        # (3) reconstruct from child COs where a TO carried none.
        for co in cos:
            parent = _resolve_parent_terminal(co)
            if not parent or parent not in assignments:
                continue
            if assignments[parent]:
                # TO already had source_refs chunk ids; keep them authoritative.
                continue
            raw = co.get("source_chunk_ids")
            if isinstance(raw, list):
                bucket = seen.setdefault(parent, set())
                for c in raw:
                    cs = str(c).strip()
                    if cs and cs not in bucket:
                        bucket.add(cs)
                        assignments[parent].append(cs)
        return assignments

    def _pass(
        self, gate_id: str, issues: List[GateIssue]
    ) -> GateResult:
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=1.0,
            issues=issues,
        )

    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        to_id: str,
        passed: bool,
        code: Optional[str],
        cosine: Optional[float],
        threshold: float,
        n_chunks: int,
        embedder_strict: bool,
    ) -> None:
        if capture is None:
            return
        cos_str = f"{cosine:.4f}" if cosine is not None else "n/a"
        rationale = (
            f"TO source-grounding check on terminal {to_id!r}: "
            f"max_chunk_cosine={cos_str}, threshold={threshold:.4f}, "
            f"n_supporting_chunks={n_chunks}, "
            f"embedder_strict_mode={embedder_strict}, "
            f"failure_code={code or 'none'}."
        )
        try:
            capture.log_decision(
                decision_type=_DECISION_TYPE,
                decision="passed" if passed else f"failed:{code or 'unknown'}",
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 — capture must not break the gate
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "terminal_objective_source_grounding: %s",
                exc,
            )


__all__ = [
    "TerminalObjectiveSourceGroundingValidator",
    "DEFAULT_THRESHOLD",
]
