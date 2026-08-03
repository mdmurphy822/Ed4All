"""Feature 2 / PRONG C — Bloom-profile COMPLEMENT (LLM-assisted, grounded).

Prongs A/B repair distinct-SKILL loss in the dedup pass; neither touches the
course's COGNITIVE profile. A real 7B objective run skews hard toward
apply/understand — the canonical CO set can carry ZERO analyze/evaluate
objectives, so the course never asks the learner to reason above the
apply ceiling. This pass is the cure: it MEASURES the higher-order share and, if
too thin, SYNTHESIZES a bounded number of grounded analyze/evaluate complement
objectives from the already-cited instructional chunks.

SCOPE (corpus, mirroring PRONG B). Grouping COs into terminal objectives happens
LATER (bottom-up TO derivation), so there is no TO scope available here. PRONG B
(:func:`lib.objectives.source_backfill.backfill_uncovered_chunks`) is
CORPUS-scoped — it operates over the whole canonical set + the whole chunk
universe in one call. PRONG C mirrors that: it measures the bloom distribution
of the WHOLE candidate set and tops up the analyze+evaluate share.

WHAT IT ADDS. When the share of analyze+evaluate+create COs is below
``ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE`` (default 0.15), it synthesizes
complement candidates via the SAME provider path stage-2 synthesis uses
(``TEXTBOOK_SYNTHESIS_PROVIDER`` seat) until the share is met or
``ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX`` (default 8) additions are made. Note the
asymmetry: the SHARE numerator counts analyze+evaluate+create (all higher-order),
but every ADDED objective targets analyze/evaluate ONLY — never ``create`` (an
open-ended "design/compose" objective is unassessable for this pipeline today).

HARD CONTRACTS (per added complement CO):
  (a) targets analyze OR evaluate, with the main verb drawn from the canonical
      verb table (``lib/ontology/bloom.py``) for that level;
  (b) is GROUNDED — the prompt offers a sampled set of ALREADY-CITED
      INSTRUCTIONAL chunks (exercise-like chunks demoted via the
      ``citation_reselect`` heuristic), and the CO must carry ``source_chunk_ids``
      drawn ONLY from that offered set. Anti-fabrication: a candidate citing ANY
      id not offered is DROPPED entirely (never silently pruned to the subset);
  (c) survives the existing dedup threshold vs the canonical set — a complement
      whose statement near-duplicates an existing CO (cosine ≥ the dedup
      threshold) is dropped;
  (d) passes the objective-echo test in reverse — its NAMED skill signature must
      be DISTINCT from every existing CO (reusing PRONG A's signature helpers),
      so it can never be a restatement of an existing objective.

DECISION CAPTURE. The LLM call site wires ``DecisionCapture`` per the project
contract: one ``objective_bloom_complement`` event PER CALL, dynamic rationale
(≥20 chars) interpolating target level, offered-chunk count, provider/model, raw
+ accepted counts, and the running share.

CHECKPOINT. PRONG B ships NO resume sidecar, so — per the
``ED4ALL_GENERATION_CHECKPOINT`` family convention (mirror the sibling prong) —
PRONG C ships none either.

Default OFF (opt-in via ``ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT``) — default-off is
byte-identical to today (no LLM call, canonical set untouched, no capture).
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.generation.llm_checkpoint import CheckpointStore, compute_fingerprint
from lib.generation.stop_control import check_stop

logger = logging.getLogger(__name__)

#: Graceful-stop + resume-sidecar wiring (P4 item 6, Resolved Decision D5 —
#: full sidecars even for the bounded ≤ ``max_additions`` loop). One record per
#: complement LLM call; the sidecar lives beside ``synthesized_objectives.json``
#: (``01_learning_objectives/``). ``checkpoint_path`` unwired → no sidecar
#: (byte-identical default), but the per-call ``check_stop`` stays live via
#: ``ED4ALL_RUN_ID``. Family flag ``ED4ALL_GENERATION_CHECKPOINT`` (site_env=None).
_BLOOM_COMPLEMENT_SITE_ID = "bloom_complement"
#: Canonical sidecar basename (sibling of ``synthesized_objectives.json``).
BLOOM_COMPLEMENT_CHECKPOINT_NAME = ".bloom_complement_checkpoint.jsonl"


def _bloom_complement_store(path: Optional[Path]) -> CheckpointStore:
    """Fingerprinted per-LLM-call resume sidecar for the bloom-complement pass."""
    return CheckpointStore(
        path, site_id=_BLOOM_COMPLEMENT_SITE_ID, site_env=None,
    )

#: PRONG C master gate. Default OFF (opt-in).
_DEFAULT_BLOOM_COMPLEMENT = False
ENV_BLOOM_COMPLEMENT = "ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT"

#: Minimum share of higher-order (analyze+evaluate+create) COs before the pass
#: stops topping up. Default 0.15. Clamped to [0, 1]; garbage → default.
_DEFAULT_MIN_SHARE = 0.15
ENV_MIN_SHARE = "ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MIN_SHARE"

#: Hard cap on the number of complement COs added in one pass. Default 8.
#: Garbage / negative → default; 0 = measure-only (add nothing).
_DEFAULT_MAX_ADDITIONS = 8
ENV_MAX_ADDITIONS = "ED4ALL_OBJECTIVE_BLOOM_COMPLEMENT_MAX"

#: SHARE numerator — the higher-order levels counted toward the measured share.
_HIGHER_ORDER_LEVELS = frozenset({"analyze", "evaluate", "create"})

#: GENERATION targets — the levels a complement may target (NOT ``create``).
_COMPLEMENT_TARGET_LEVELS: Tuple[str, ...] = ("analyze", "evaluate")

#: Cap on offered instructional chunks enumerated into one prompt (keeps the
#: prompt bounded; deterministic id-sorted sample).
_MAX_OFFERED_CHUNKS = 12

#: Cap on existing statements listed in the "do NOT restate" avoid block.
_MAX_AVOID_STATEMENTS = 20

#: Per-chunk body char cap in the prompt (mirrors the window prompt's terse
#: enumerated-body discipline; keeps the offered block small).
_CHUNK_BODY_CHARS = 600


@dataclass
class ComplementResult:
    """Outcome of the PRONG-C bloom-complement pass."""

    objectives: List[Dict[str, Any]] = field(default_factory=list)
    share_before: float = 0.0
    share_after: float = 0.0
    cos_added: int = 0
    added_statements: List[str] = field(default_factory=list)
    offered_chunk_count: int = 0
    llm_calls: int = 0
    #: candidates the model returned that FAILED a contract (⊆ / target / dedup
    #: / echo) — the anti-fabrication rejection count.
    candidates_rejected: int = 0
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "share_before": round(float(self.share_before), 6),
            "share_after": round(float(self.share_after), 6),
            "cos_added": self.cos_added,
            "added_statements": list(self.added_statements),
            "offered_chunk_count": self.offered_chunk_count,
            "llm_calls": self.llm_calls,
            "candidates_rejected": self.candidates_rejected,
            "available": self.available,
        }


# ---------------------------------------------------------------------------
# Resolvers (parse-with-fallback, mirroring the sibling prongs)
# ---------------------------------------------------------------------------


def resolve_bloom_complement(enabled: Optional[bool] = None) -> bool:
    """Resolve the PRONG-C master gate: arg → env → default (OFF)."""
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_BLOOM_COMPLEMENT, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_BLOOM_COMPLEMENT


def resolve_min_share(share: Optional[float] = None) -> float:
    """Resolve the higher-order share floor: arg → env → default. Clamped [0,1]."""
    if share is not None:
        try:
            return max(0.0, min(1.0, float(share)))
        except (TypeError, ValueError):
            return _DEFAULT_MIN_SHARE
    raw = os.environ.get(ENV_MIN_SHARE)
    if raw:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return _DEFAULT_MIN_SHARE
        if 0.0 <= val <= 1.0:
            return val
    return _DEFAULT_MIN_SHARE


def resolve_max_additions(value: Optional[int] = None) -> int:
    """Resolve the additions cap: arg → env → default. Garbage / negative →
    default; 0 is a legitimate measure-only value."""
    if value is not None:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_ADDITIONS
    raw = os.environ.get(ENV_MAX_ADDITIONS)
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_MAX_ADDITIONS
        if val >= 0:
            return val
    return _DEFAULT_MAX_ADDITIONS


# ---------------------------------------------------------------------------
# Bloom-level helpers (reuse lib/ontology/bloom.py — READ-ONLY)
# ---------------------------------------------------------------------------


def _effective_level(objective: Dict[str, Any]) -> Optional[str]:
    """Effective Bloom level of an objective: declared (if valid) else detected
    from the statement's main verb via the canonical detector."""
    try:
        from lib.ontology.bloom import BLOOM_LEVELS, detect_bloom_level
    except Exception:  # noqa: BLE001
        return None
    declared = str(objective.get("bloom_level") or "").strip().lower()
    if declared in set(BLOOM_LEVELS):
        return declared
    level, _verb = detect_bloom_level(_statement(objective))
    return level


def higher_order_share(objectives: Sequence[Dict[str, Any]]) -> float:
    """Fraction of objectives whose effective level is analyze/evaluate/create.

    Empty input → 0.0 (an empty course has no higher-order coverage to speak of).
    """
    total = 0
    higher = 0
    for obj in objectives:
        if not isinstance(obj, dict):
            continue
        total += 1
        if _effective_level(obj) in _HIGHER_ORDER_LEVELS:
            higher += 1
    return (higher / total) if total else 0.0


def _target_verbs(level: str, *, limit: int = 6) -> List[str]:
    """A deterministic sample of the canonical action verbs for ``level``."""
    try:
        from lib.ontology.bloom import get_verbs
    except Exception:  # noqa: BLE001
        return []
    return sorted(get_verbs().get(level, set()))[:limit]


# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------


def _statement(objective: Dict[str, Any]) -> str:
    return str(objective.get("statement") or objective.get("text") or "").strip()


def _source_chunk_ids(objective: Dict[str, Any]) -> List[str]:
    raw = objective.get("source_chunk_ids")
    if isinstance(raw, list):
        return [str(c) for c in raw if c]
    return []


def _chunk_text(rec: Any) -> str:
    if not isinstance(rec, dict):
        return ""
    return str(rec.get("text") or rec.get("body") or "").strip()


def _is_exercise_like(text: str) -> bool:
    """Defensive import of the citation_reselect exercise heuristic.

    When importable, prefers instructional prose over end-of-chapter
    exercise/answer-list chunks. Import failure → treat nothing as exercise-like
    (no starvation)."""
    try:
        from lib.objectives.citation_reselect import _is_exercise_like as _ex

        return bool(_ex(text))
    except Exception:  # noqa: BLE001
        return False


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Lenient JSON-object extractor (clean → fenced → outermost braces).

    Local (not the provider's classmethod) so the pass stays testable with a
    duck-typed provider stub. Mirrors the provider's drift-tolerance."""
    if not text:
        return None

    def _try(blob: str) -> Optional[Dict[str, Any]]:
        for cand in (blob, re.sub(r",(\s*[}\]])", r"\1", blob)):
            try:
                obj = json.loads(cand)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
        return None

    candidate = text.strip()
    obj = _try(candidate)
    if obj is not None:
        return obj
    fence = re.search(
        r"(?:^|\n)```(?:json)?[ \t]*\n?(.*?)\n?```", candidate, re.DOTALL
    )
    if fence:
        obj = _try(fence.group(1).strip())
        if obj is not None:
            return obj
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = _try(candidate[start:end + 1])
        if obj is not None:
            return obj
    return None


# ---------------------------------------------------------------------------
# Offered-chunk assembly + candidate validation
# ---------------------------------------------------------------------------


def _build_offered_chunks(
    canonical: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Sampled ALREADY-CITED instructional chunks to ground complements against.

    Universe = every chunk id cited by the canonical set that resolves in
    ``chunks_by_id`` and carries text. Exercise-like chunks are DEMOTED (kept
    only if instructional chunks don't fill the cap). Deterministic id-sorted;
    capped at :data:`_MAX_OFFERED_CHUNKS`.
    """
    cited: List[str] = []
    seen: set = set()
    for co in canonical:
        for cid in _source_chunk_ids(co):
            if cid not in seen:
                seen.add(cid)
                cited.append(cid)

    instructional: List[Dict[str, Any]] = []
    exercise: List[Dict[str, Any]] = []
    for cid in sorted(cited):
        rec = chunks_by_id.get(cid)
        body = _chunk_text(rec)
        if not body:
            continue
        entry = {"id": cid, "text": body, "rec": rec}
        if _is_exercise_like(body):
            exercise.append(entry)
        else:
            instructional.append(entry)

    offered = instructional[:_MAX_OFFERED_CHUNKS]
    if len(offered) < _MAX_OFFERED_CHUNKS:
        offered = offered + exercise[: _MAX_OFFERED_CHUNKS - len(offered)]
    return offered


def _render_complement_prompt(
    *,
    course_name: str,
    target_level: str,
    offered: List[Dict[str, Any]],
    avoid_statements: List[str],
) -> str:
    """Bespoke higher-order complement prompt (analyze/evaluate + ⊆ citation)."""
    verbs = _target_verbs(target_level)
    verb_hint = ", ".join(verbs) if verbs else target_level
    chunk_lines = [
        f"  - [{c['id']}] {str(c['text'])[:_CHUNK_BODY_CHARS]}"
        for c in offered
    ]
    chunks_block = "\n".join(chunk_lines) if chunk_lines else "  (none)"
    avoid_lines = [f"  - {s}" for s in avoid_statements[:_MAX_AVOID_STATEMENTS]]
    avoid_block = "\n".join(avoid_lines) if avoid_lines else "  (none)"
    return (
        f"Course: {course_name}\n\n"
        "This course's learning objectives are too shallow — they lack "
        f"higher-order ({target_level}) objectives. Author NEW "
        f"{target_level.upper()}-level objectives GROUNDED in the source "
        "chunks below.\n\n"
        "Source chunks (cite ONLY these ids in source_chunk_ids):\n"
        f"{chunks_block}\n\n"
        "Existing objectives — do NOT restate or paraphrase any of these; "
        "author a genuinely DISTINCT higher-order skill:\n"
        f"{avoid_block}\n\n"
        f"Author 1-3 {target_level} learning objectives the source chunks "
        "above genuinely support. Each objective's main action verb MUST be a "
        f"real {target_level} verb from this list: {verb_hint}. Do NOT author "
        "remember/understand/apply objectives, and do NOT author open-ended "
        "create/design objectives.\n"
        "RESPOND ONLY WITH A JSON OBJECT with top-level key "
        '"complement_objectives": [{"statement", "bloom_level" ("' +
        target_level + '"), "bloom_verb", "source_chunk_ids": ["<id>", ...]}].\n'
        "For each objective, source_chunk_ids MUST be a NON-EMPTY subset of the "
        "chunk ids listed under \"Source chunks\". NEVER cite an id not listed "
        "above. NEVER invent a chunk id. NEVER emit source_chunk_ids: [].\n"
        "No preamble, no markdown fences, no commentary."
    )


def _validate_candidate(
    raw: Dict[str, Any],
    *,
    offered_ids: set,
    chunks_by_id: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Enforce contracts (a) + (b) on one raw model candidate.

    Returns the normalized complement CO, or ``None`` when it violates:
      - empty statement,
      - target-level miss (statement's main verb is not analyze/evaluate),
      - citation miss (empty, or ANY id outside the offered set → whole
        candidate dropped, never silently pruned).
    """
    if not isinstance(raw, dict):
        return None
    statement = str(raw.get("statement") or raw.get("text") or "").strip()
    if not statement:
        return None

    # (a) target level — derive HONESTLY from the statement's main verb so a
    # mislabelled bloom_level can't smuggle a lower-order objective through.
    try:
        from lib.ontology.bloom import detect_bloom_level
    except Exception:  # noqa: BLE001
        return None
    level, verb = detect_bloom_level(statement)
    if level not in _COMPLEMENT_TARGET_LEVELS:
        return None

    # (b) citation ⊆ offered (STRICT: any foreign id → drop the whole candidate).
    cited = [str(c).strip() for c in (raw.get("source_chunk_ids") or []) if str(c).strip()]
    if not cited:
        return None
    if any(cid not in offered_ids for cid in cited):
        return None
    # Stable de-dupe preserving order.
    kept: List[str] = []
    for cid in cited:
        if cid not in kept:
            kept.append(cid)

    chapter_id = ""
    first = chunks_by_id.get(kept[0])
    if isinstance(first, dict):
        chapter_id = str(first.get("chapter_id") or "")

    co: Dict[str, Any] = {
        "statement": statement,
        "bloom_level": level,
        "source_chunk_ids": list(kept),
        "source_refs": [{"ref": chapter_id, "chunk_ids": list(kept)}],
        "chapter_id": chapter_id,
        "bloom_complement": True,
    }
    if verb:
        co["bloom_verb"] = str(verb).strip().lower()
    return co


# ---------------------------------------------------------------------------
# LLM call (the SAME provider path stage-2 synthesis uses)
# ---------------------------------------------------------------------------


def _dispatch_complement_call(
    provider: Any,
    prompt: str,
) -> Optional[str]:
    """Route one complement prompt through the provider's dispatch (the
    ``TEXTBOOK_SYNTHESIS_PROVIDER`` backend). Returns raw text or ``None``.

    Uses the provider's ``_dispatch_call`` (the same seam every stage-2 method
    routes through — it applies the local ``num_ctx`` pin + backend routing).
    Any failure (missing method / provider raise) degrades to ``None`` so the
    pass never sinks the run."""
    dispatch = getattr(provider, "_dispatch_call", None)
    if not callable(dispatch):
        return None
    try:
        result = dispatch(prompt)
    except Exception as exc:  # noqa: BLE001 — provider raise degrades to no-op
        logger.warning(
            "bloom_complement: provider dispatch raised (%s); skipping this "
            "complement call.", exc,
        )
        return None
    # ``_dispatch_call`` returns ``(text, retry_count)``; tolerate a bare str.
    if isinstance(result, tuple):
        return str(result[0] or "")
    return str(result or "")


def _synthesize_complements(
    provider: Any,
    *,
    course_name: str,
    target_level: str,
    offered: List[Dict[str, Any]],
    avoid_statements: List[str],
    chunks_by_id: Dict[str, Any],
    capture: Optional[Any],
    share_before: float,
    min_share: float,
) -> List[Dict[str, Any]]:
    """One grounded complement LLM call → validated complement candidates.

    Wires ``DecisionCapture`` per the project contract: exactly one
    ``objective_bloom_complement`` event for THIS call, dynamic rationale."""
    prompt = _render_complement_prompt(
        course_name=course_name,
        target_level=target_level,
        offered=offered,
        avoid_statements=avoid_statements,
    )
    raw_text = _dispatch_complement_call(provider, prompt)
    parsed = _extract_json(raw_text or "")
    offered_ids = {str(c["id"]) for c in offered}

    validated: List[Dict[str, Any]] = []
    raw_count = 0
    if isinstance(parsed, dict):
        raw_list = (
            parsed.get("complement_objectives")
            or parsed.get("candidate_objectives")
            or parsed.get("objectives")
            or []
        )
        if isinstance(raw_list, list):
            for raw in raw_list:
                raw_count += 1
                co = _validate_candidate(
                    raw, offered_ids=offered_ids, chunks_by_id=chunks_by_id
                )
                if co is not None:
                    validated.append(co)

    _emit_complement_capture(
        capture,
        provider=provider,
        target_level=target_level,
        offered_count=len(offered),
        raw_count=raw_count,
        validated_count=len(validated),
        share_before=share_before,
        min_share=min_share,
    )
    return validated


def _emit_complement_capture(
    capture: Any,
    *,
    provider: Any,
    target_level: str,
    offered_count: int,
    raw_count: int,
    validated_count: int,
    share_before: float,
    min_share: float,
) -> None:
    """Best-effort ``objective_bloom_complement`` capture (never raises).

    New LLM call site → mandatory decision-capture. Rationale interpolates ≥5
    dynamic signals so the capture is replayable post-hoc."""
    if capture is None:
        return
    prov = str(getattr(provider, "_provider", "") or "")
    model = str(getattr(provider, "_model", "") or "")
    try:
        capture.log_decision(
            decision_type="objective_bloom_complement",
            decision=(
                f"synthesized {validated_count}/{raw_count} valid {target_level} "
                f"complement objective(s)"
            ),
            rationale=(
                f"PRONG C (bloom-profile complement): higher-order share was "
                f"{share_before:.2f} (< floor); asked provider={prov} "
                f"model={model} for {target_level} objectives grounded in "
                f"{offered_count} already-cited instructional chunk(s). Model "
                f"returned {raw_count} candidate(s); {validated_count} passed "
                f"the ⊆-citation + {target_level}-verb contracts (rest dropped "
                f"as ungrounded / off-level). Anti-fabrication: every accepted "
                f"CO cites ONLY offered chunk ids; a candidate citing any "
                f"unoffered id is dropped whole."
            ),
            alternatives_considered=[
                {
                    "option": "Leave the objective set without complementation",
                    "score": share_before,
                    "reason_rejected": (
                        f"Rejected because higher-order share {share_before:.3f} "
                        f"was below the configured floor {min_share:.3f}."
                    ),
                },
                {
                    "option": "Accept higher-order objectives without source grounding",
                    "score": float(validated_count),
                    "reason_rejected": (
                        f"Rejected because only {validated_count} of {raw_count} "
                        f"candidates cited the {offered_count} offered chunks and "
                        f"satisfied the {target_level} verb contract."
                    ),
                },
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "objective_bloom_complement capture failed (%s); continuing", exc
        )


# ---------------------------------------------------------------------------
# Dedup + echo rejection (reuse PRONG A helpers)
# ---------------------------------------------------------------------------


def _near_duplicate(
    statement: str,
    existing_vecs: List[List[float]],
    embed: Any,
) -> bool:
    """Contract (c): cosine(statement, existing CO) ≥ dedup threshold → drop.

    Embed absent → ``False`` (the echo signature check is the fallback guard)."""
    if embed is None or not existing_vecs:
        return False
    try:
        from lib.embedding._math import cosine_similarity
        from lib.objectives.objective_dedup import resolve_dedup_threshold

        vec = embed.encode_batch([statement or " "])[0]
    except Exception:  # noqa: BLE001 — embed failure degrades to no-dedup
        return False
    threshold = resolve_dedup_threshold()
    for other in existing_vecs:
        try:
            if cosine_similarity(vec, other) >= threshold:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _echoes_existing(
    candidate: Dict[str, Any],
    existing_sigs: List[frozenset],
) -> bool:
    """Contract (d): the candidate's NAMED skill is NOT distinct from an
    existing CO → drop (reverse objective-echo). Reuses PRONG A's signature
    helpers so "distinct skill" means the same thing everywhere."""
    try:
        from lib.objectives.objective_dedup import (
            _signatures_distinct,
            _skill_signature,
        )
    except Exception:  # noqa: BLE001
        return False
    sig = _skill_signature(candidate)
    if not sig:
        # No nameable skill → cannot assert distinctness; treat as an echo so we
        # never add a vacuous objective.
        return True
    for existing in existing_sigs:
        if not _signatures_distinct(sig, existing):
            return True
    return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def complement_bloom_profile(
    *,
    canonical: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    provider: Any,
    course_name: str,
    capture: Optional[Any] = None,
    embed: Optional[Any] = None,
    enabled: Optional[bool] = None,
    min_share: Optional[float] = None,
    max_additions: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> ComplementResult:
    """Top up the canonical CO set with grounded analyze/evaluate complements.

    Mutates ``canonical`` by APPENDING accepted complement COs (id-less — the
    caller mints CO-NN after this pass). Default-off / empty inputs / provider
    absent → no-op (``available=False``, ``canonical`` untouched).

    Graceful stop (P4 item 6 / D5): a pending stop sentinel raises
    ``GracefulStopRequested`` at the per-LLM-call boundary (checked at the loop
    top, BEFORE dispatching that call). Every already-completed call's validated
    candidates are on the resume sidecar (``checkpoint_path``, when supplied); a
    resume replays them through the SAME dedup/echo acceptance without
    re-dispatching. The sidecar is removed once the loop finishes normally.

    Args (stop/resume additions):
        checkpoint_path: per-LLM-call resume sidecar path (natural sibling of
            ``synthesized_objectives.json``). ``None`` disables the sidecar
            (byte-identical default); the stop check stays live regardless.
        run_id: explicit run id for the stop-sentinel probe (else
            ``ED4ALL_RUN_ID``).
    """
    if not resolve_bloom_complement(enabled):
        return ComplementResult(objectives=canonical, available=False)
    if not canonical or provider is None or not chunks_by_id:
        return ComplementResult(objectives=canonical, available=False)

    target = resolve_min_share(min_share)
    max_add = resolve_max_additions(max_additions)
    share_before = higher_order_share(canonical)

    result = ComplementResult(
        objectives=canonical,
        share_before=share_before,
        share_after=share_before,
        available=True,
    )
    if share_before >= target or max_add <= 0:
        return result

    offered = _build_offered_chunks(canonical, chunks_by_id)
    result.offered_chunk_count = len(offered)
    if not offered:
        logger.info(
            "bloom_complement: no already-cited instructional chunks to ground "
            "complements against; skipping (share=%.2f).", share_before,
        )
        return result

    # Existing skill signatures (echo guard) + statement vecs (dedup guard).
    existing_sigs: List[frozenset] = []
    try:
        from lib.objectives.objective_dedup import _skill_signature

        for co in canonical:
            sig = _skill_signature(co)
            if sig:
                existing_sigs.append(sig)
    except Exception:  # noqa: BLE001
        existing_sigs = []

    existing_vecs: List[List[float]] = []
    if embed is not None:
        try:
            existing_vecs = list(
                embed.encode_batch([_statement(co) or " " for co in canonical])
            )
        except Exception:  # noqa: BLE001 — dedup then relies on the echo guard
            existing_vecs = []

    store = _bloom_complement_store(checkpoint_path)
    cache = store.load()
    added = 0
    calls = 0
    unproductive = 0
    max_calls = max(1, max_add)
    while (
        higher_order_share(canonical) < target
        and added < max_add
        and calls < max_calls
    ):
        # Stop at the per-LLM-call unit boundary — BEFORE dispatching this call.
        # Sequential loop → a plain raise is correct (no gathered coroutine to
        # lose); every completed call's candidates are already on the sidecar.
        check_stop(_BLOOM_COMPLEMENT_SITE_ID, calls, run_id=run_id)
        target_level = _COMPLEMENT_TARGET_LEVELS[added % len(_COMPLEMENT_TARGET_LEVELS)]
        avoid = [_statement(co) for co in canonical if _statement(co)]
        # Fingerprint the call's full input surface: the CURRENT canonical
        # statements (they grow as complements are accepted, and feed the avoid
        # block), the offered chunk-id menu, the target level, provider/model,
        # and the loop knobs — so any change re-runs the call on resume.
        unit_id = f"call#{calls}"
        fingerprint = compute_fingerprint({
            "canonical": sorted(_statement(co) for co in canonical if _statement(co)),
            "offered_ids": sorted(str(c["id"]) for c in offered),
            "target_level": target_level,
            "provider": str(getattr(provider, "_provider", "") or ""),
            "model": str(getattr(provider, "_model", "") or ""),
            "min_share": target,
            "max_add": max_add,
        })
        reused = store.reuse(unit_id, fingerprint, cache=cache)
        if reused is not None:
            candidates = [
                c for c in (reused.get("candidates") or []) if isinstance(c, dict)
            ]
        else:
            candidates = _synthesize_complements(
                provider,
                course_name=course_name,
                target_level=target_level,
                offered=offered,
                avoid_statements=avoid,
                chunks_by_id=chunks_by_id,
                capture=capture,
                share_before=higher_order_share(canonical),
                min_share=target,
            )
            # Checkpoint the call's validated candidates BEFORE the acceptance
            # pass so a stop at the next loop-top leaves this call recoverable.
            store.append(unit_id, fingerprint, {"candidates": candidates})
        calls += 1
        accepted_this = 0
        for co in candidates:
            if added >= max_add:
                break
            statement = _statement(co)
            # (c) dedup vs existing + (d) reverse echo.
            if _near_duplicate(statement, existing_vecs, embed):
                result.candidates_rejected += 1
                continue
            if _echoes_existing(co, existing_sigs):
                result.candidates_rejected += 1
                continue
            canonical.append(co)
            added += 1
            accepted_this += 1
            result.added_statements.append(statement[:80])
            # Register the new CO so later candidates dedup/echo against it too.
            new_sig = None
            try:
                from lib.objectives.objective_dedup import _skill_signature

                new_sig = _skill_signature(co)
            except Exception:  # noqa: BLE001
                new_sig = None
            if new_sig:
                existing_sigs.append(new_sig)
            if embed is not None:
                try:
                    existing_vecs.append(embed.encode_batch([statement or " "])[0])
                except Exception:  # noqa: BLE001
                    pass
        if accepted_this == 0:
            unproductive += 1
            if unproductive >= 2:
                break
        else:
            unproductive = 0

    # Loop finished normally (no stop) → drop the resume sidecar; the appended
    # complements are about to be minted + persisted by the caller. A stop
    # raise never reaches here, so a paused run keeps its sidecar for resume.
    store.remove()

    result.cos_added = added
    result.llm_calls = calls
    result.share_after = higher_order_share(canonical)
    if added:
        logger.info(
            "bloom_complement: added %d analyze/evaluate complement CO(s); "
            "higher-order share %.2f -> %.2f (%d LLM call(s), %d rejected).",
            added, share_before, result.share_after, calls,
            result.candidates_rejected,
        )
    return result


__all__ = [
    "ComplementResult",
    "complement_bloom_profile",
    "higher_order_share",
    "resolve_bloom_complement",
    "resolve_min_share",
    "resolve_max_additions",
    "_DEFAULT_BLOOM_COMPLEMENT",
    "_DEFAULT_MIN_SHARE",
    "_DEFAULT_MAX_ADDITIONS",
    "ENV_BLOOM_COMPLEMENT",
    "ENV_MIN_SHARE",
    "ENV_MAX_ADDITIONS",
]
