"""W2 §4.3 — deterministic CO→TO backlink (Pass E).

After reconciliation produces the canonical terminal objectives (TO-NN) and Pass
D produces the canonical chapter objectives (CO-NN), every CO needs a parent TO
so ``TerminalObjectiveCoverageValidator``'s reachability passes. This helper
mutates each CO in place setting ``terminal_id`` to its parent TO:

1. If the CO already carries a reconcile-provided CO→TO hint
   (``terminal_id`` / ``parent_terminal_id``) that resolves to an emitted TO,
   honor it.
2. Else nearest-TO by embedding cosine (reuse the Pass-D embed client) between
   the CO statement and each TO statement; assign argmax.
3. Fallback (embeddings absent): nearest TO by content-token overlap, or
   ``TO-01`` / the single TO when one TO exists. NEVER leave ``terminal_id``
   unset.

Sibling of ``lib/ontology/learning_objectives.py`` (LO identity helpers).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lib.embedding._math import cosine_similarity

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{1,}")

#: WS2 — cosine floor below which a CO→TO argmax link is "weak" (semantically
#: suspect). PURE MEASUREMENT: the link is STILL assigned (never-unset contract);
#: this only stamps weak_terminal_link=True so the mismatch is countable. On the
#: current collapsed-structure corpus ~0.87 of links fall below this; WS1
#: (bottom-up TO derivation) should drop the rate toward ~0. Env-overridable.
_DEFAULT_TO_BACKLINK_COSINE_FLOOR = 0.45

#: WS2 — token-overlap (Jaccard) analog, used on the embeddings-absent fallback.
#: Jaccard runs far lower than cosine for related text, so the floor is lower.
_DEFAULT_TO_BACKLINK_TOKEN_FLOOR = 0.10

#: Env override (cross-cutting ED4ALL_*, root-owned). The resolver returns the
#: (cosine_floor, token_floor) pair. A bare float overrides the cosine floor
#: only; a "cosine,token" pair overrides both.
ENV_TO_BACKLINK_FLOOR = "ED4ALL_TO_BACKLINK_FLOOR"

#: M5 Fix A — second-pass anti-junk-drawer reassignment + validator-parity
#: scoring. Default OFF (byte-stable to the historical PURE-MEASUREMENT pass).
#: When truthy, the backlink:
#:   1. RE-SCORES every CO against ALL TOs — including reconcile-hint-honored
#:      ones — so ``scored_count`` matches what the ``co_terminal_alignment``
#:      validator audits (it recomputes ``cosine(co, assigned_to)`` for EVERY
#:      CO with a resolvable ``terminal_id``, hints included). This reconciles
#:      the backlink's ``weak_to_link_rate`` with the validator's weak count.
#:   2. REASSIGNS a CO whose ASSIGNED-TO cosine is below the floor to its
#:      argmax (best-fitting) TO whenever that argmax scores strictly higher —
#:      pulling a CO out of a junk-drawer cluster into a terminal it actually
#:      fits. NEVER fabricates a TO; only re-points among the EXISTING TO set,
#:      preserving the never-leave-a-CO-ungrouped intent.
#:   3. Stamps ``weak_terminal_link`` on the FINAL (post-reassign) link when its
#:      cosine is still below the floor — an honest count of irreducibly-poor
#:      assignments (cosine-0.20 links that no existing TO improves).
ENV_TO_BACKLINK_REASSIGN = "ED4ALL_TO_BACKLINK_REASSIGN"

_TRUTHY = {"1", "true", "yes", "on"}


def resolve_reassign_mode(reassign: Optional[bool] = None) -> bool:
    """Resolve the M5 Fix-A reassign/validator-parity mode: explicit → env → off.

    Parse-with-fallback: any non-truthy / unset env value → ``False`` (the
    byte-stable PURE-MEASUREMENT pass). Mirrors the resolver posture of the
    other ``resolve_*`` helpers in this module.
    """
    if reassign is not None:
        return bool(reassign)
    raw = os.environ.get(ENV_TO_BACKLINK_REASSIGN)
    if raw is None:
        return False
    return str(raw).strip().lower() in _TRUTHY


def _statement(obj: Dict[str, Any]) -> str:
    return str(obj.get("statement") or obj.get("text") or "").strip()


def _content_tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def resolve_to_backlink_floor(
    cosine_floor: Optional[float] = None,
    token_floor: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve the dual CO→TO weak-link floor: explicit arg → env → default.

    Returns ``(cosine_floor, token_floor)``. Parse-with-fallback on garbage,
    mirroring ``objective_dedup.resolve_chunk_relevance_floor``:
      - explicit args win (each clamped to ``[0.0, 1.0]``; out-of-range → default);
      - else ``ED4ALL_TO_BACKLINK_FLOOR`` — a bare float overrides the COSINE
        floor only; a ``"<cos>,<tok>"`` pair overrides both; any unparseable /
        out-of-range component falls back to its default;
      - else the module defaults (0.45 / 0.10).
    A floor of 0.0 means "nothing is weak" (instrumentation off); 1.0 means
    "everything below perfect is weak".
    """
    cos_floor = _DEFAULT_TO_BACKLINK_COSINE_FLOOR
    tok_floor = _DEFAULT_TO_BACKLINK_TOKEN_FLOOR

    env_cos: Optional[float] = None
    env_tok: Optional[float] = None
    raw = os.environ.get(ENV_TO_BACKLINK_FLOOR)
    if raw:
        parts = [p.strip() for p in raw.split(",")]
        if parts and parts[0]:
            try:
                val = float(parts[0])
            except (TypeError, ValueError):
                val = None
            if val is not None and 0.0 <= val <= 1.0:
                env_cos = val
        if len(parts) > 1 and parts[1]:
            try:
                val = float(parts[1])
            except (TypeError, ValueError):
                val = None
            if val is not None and 0.0 <= val <= 1.0:
                env_tok = val

    # Cosine floor: explicit arg → env → default.
    if cosine_floor is not None:
        try:
            val = float(cosine_floor)
        except (TypeError, ValueError):
            val = None
        if val is not None and 0.0 <= val <= 1.0:
            cos_floor = val
    elif env_cos is not None:
        cos_floor = env_cos

    # Token floor: explicit arg → env → default.
    if token_floor is not None:
        try:
            val = float(token_floor)
        except (TypeError, ValueError):
            val = None
        if val is not None and 0.0 <= val <= 1.0:
            tok_floor = val
    elif env_tok is not None:
        tok_floor = env_tok

    return cos_floor, tok_floor


@dataclass(frozen=True)
class BacklinkResult:
    """WS2 — measurement rollup from a CO→TO backlink pass. PURE MEASUREMENT."""

    weak_link_count: int = 0          # COs below floor (weak_terminal_link=True)
    scored_count: int = 0             # COs through the SCORED path (rate denom).
                                      # PURE-MEASUREMENT mode EXCLUDES hint-honored
                                      # + early-return COs; M5 reassign mode
                                      # INCLUDES hint-honored COs (validator parity).
    used_embeddings: bool = False     # cosine (True) vs token-overlap (False)
    cosine_floor: float = 0.0
    token_floor: float = 0.0
    min_best_score: float = 0.0       # distribution probes (SCORED best_scores)
    mean_best_score: float = 0.0
    reassigned_count: int = 0         # M5 Fix A — COs re-pointed to a better TO
                                      # (0 in PURE-MEASUREMENT mode)

    @property
    def weak_link_rate(self) -> float:
        return (
            (self.weak_link_count / self.scored_count)
            if self.scored_count
            else 0.0
        )


def backlink_cos_to_tos(
    terminals: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    *,
    embed: Optional[Any] = None,
    cosine_floor: Optional[float] = None,   # WS2 — None → resolve_to_backlink_floor
    token_floor: Optional[float] = None,    # WS2 — None → resolve_to_backlink_floor
    reassign: Optional[bool] = None,        # M5 — None → resolve_reassign_mode
) -> "BacklinkResult":                       # WS2 — was None
    """Set ``terminal_id`` on every CO in ``chapters`` to its parent TO (in place).

    Args:
        terminals: the canonical TO-NN list (each with ``id`` + ``statement``).
        chapters: the canonical CO-NN list (mutated in place).
        embed: optional embedding client (reuse the Pass-D one). ``None`` →
            token-overlap fallback (no model load attempted here; the caller owns
            embed-client lifecycle).
        cosine_floor: WS2 weak-link cosine floor; ``None`` resolves from env.
        token_floor: WS2 weak-link token-overlap floor; ``None`` resolves from env.
        reassign: M5 Fix A reassign + validator-parity mode; ``None`` resolves
            from ``ED4ALL_TO_BACKLINK_REASSIGN``. See that env var's docstring.

    Returns:
        A :class:`BacklinkResult` measurement rollup.

        - ``reassign=False`` (default): PURE MEASUREMENT — assignment behavior is
          byte-identical to the WS2 pass; hint-honored COs are excluded from
          ``scored_count`` and never floor-eligible.
        - ``reassign=True`` (M5): hint-honored COs are RE-SCORED against all TOs
          (validator parity → ``scored_count`` matches what
          ``co_terminal_alignment`` audits), and any CO whose assigned-TO cosine
          is below the floor is re-pointed to its strictly-better argmax TO.
          ``weak_terminal_link`` is stamped on the FINAL link.
    """
    cos_floor, tok_floor = resolve_to_backlink_floor(cosine_floor, token_floor)
    reassign_mode = resolve_reassign_mode(reassign)

    if not chapters:
        return BacklinkResult(cosine_floor=cos_floor, token_floor=tok_floor)
    to_ids = [str(t.get("id") or "") for t in terminals]
    to_statements = [_statement(t) for t in terminals]
    valid_to_ids = {tid for tid in to_ids if tid}

    if not valid_to_ids:
        # No terminal to link to — leave terminal_id unset is not allowed, but
        # there is genuinely no parent. Stamp empty-string sentinel so a reader
        # can tell "no TO existed" apart from "unprocessed".
        for co in chapters:
            co.setdefault("terminal_id", "")
        return BacklinkResult(cosine_floor=cos_floor, token_floor=tok_floor)

    # Single-TO shortcut: every CO rolls up to the one terminal.
    if len(valid_to_ids) == 1:
        only = next(iter(valid_to_ids))
        for co in chapters:
            co["terminal_id"] = only
        return BacklinkResult(cosine_floor=cos_floor, token_floor=tok_floor)

    # Try the embedding path; fall back to token overlap on any failure.
    to_vecs = None
    co_vecs = None
    if embed is not None:
        try:
            to_vecs = embed.encode_batch(
                [s or " " for s in to_statements]
            )
            co_vecs = embed.encode_batch(
                [_statement(co) or " " for co in chapters]
            )
        except Exception:  # noqa: BLE001 — degrade to token overlap
            to_vecs = None
            co_vecs = None

    weak_count = 0
    scored_count = 0
    reassigned_count = 0
    used_embeddings = to_vecs is not None and co_vecs is not None
    floor = cos_floor if used_embeddings else tok_floor
    scores: List[float] = []

    # Index of each valid TO id (for scoring a hint-honored assignment in M5).
    to_idx_by_id = {tid: i for i, tid in enumerate(to_ids) if tid}

    def _score(co_idx: int, t_idx: int) -> float:
        if to_vecs is not None and co_vecs is not None:
            return cosine_similarity(co_vecs[co_idx], to_vecs[t_idx])
        return _token_overlap(_statement(chapters[co_idx]), to_statements[t_idx])

    def _argmax(co_idx: int) -> Tuple[int, float]:
        best_i, best_s = 0, -2.0
        for t_idx in range(len(terminals)):
            if not to_ids[t_idx]:
                continue
            s = _score(co_idx, t_idx)
            if s > best_s:
                best_s, best_i = s, t_idx
        return best_i, best_s

    for co_idx, co in enumerate(chapters):
        # (1) Honor a resolvable reconcile hint.
        hint = str(
            co.get("terminal_id") or co.get("parent_terminal_id") or ""
        ).strip()
        if hint in valid_to_ids:
            co["terminal_id"] = hint
            if not reassign_mode:
                # PURE-MEASUREMENT (§3.4): a hint is an explicit upstream
                # assertion, not a nearest-neighbor guess → NOT scored.
                continue
            # M5 Fix A — validator parity: SCORE the hint-honored link against
            # its ASSIGNED TO (exactly what co_terminal_alignment recomputes),
            # then reassign to a strictly-better argmax TO if the assigned link
            # is below the floor (anti-junk-drawer).
            assigned_idx = to_idx_by_id[hint]
            assigned_score = _score(co_idx, assigned_idx)
            best_idx, best_score = assigned_idx, assigned_score
            if assigned_score < floor:
                arg_idx, arg_score = _argmax(co_idx)
                if arg_score > assigned_score and to_ids[arg_idx]:
                    best_idx, best_score = arg_idx, arg_score
                    co["terminal_id"] = to_ids[arg_idx]
                    reassigned_count += 1
            scored_count += 1
            scores.append(float(best_score))
            if best_score < floor:
                co["weak_terminal_link"] = True
                co["weak_terminal_link_score"] = round(float(best_score), 6)
                weak_count += 1
            continue

        best_idx, best_score = _argmax(co_idx)
        # STILL assign argmax — never-unset contract preserved.
        co["terminal_id"] = to_ids[best_idx] if to_ids[best_idx] else next(
            iter(valid_to_ids)
        )

        # WS2 — measurement. Below-floor links are stamped, not un-set. In M5
        # reassign mode this is already the argmax (no better TO to move to),
        # so the stamp is honest about an irreducibly-poor fit.
        scored_count += 1
        scores.append(float(best_score))
        if best_score < floor:
            co["weak_terminal_link"] = True
            co["weak_terminal_link_score"] = round(float(best_score), 6)
            weak_count += 1

    return BacklinkResult(
        weak_link_count=weak_count,
        scored_count=scored_count,
        used_embeddings=used_embeddings,
        cosine_floor=cos_floor,
        token_floor=tok_floor,
        min_best_score=round(min(scores), 6) if scores else 0.0,
        mean_best_score=round(sum(scores) / len(scores), 6) if scores else 0.0,
        reassigned_count=reassigned_count,
    )


__all__ = [
    "backlink_cos_to_tos",
    "resolve_to_backlink_floor",
    "resolve_reassign_mode",
    "BacklinkResult",
    "ENV_TO_BACKLINK_FLOOR",
    "ENV_TO_BACKLINK_REASSIGN",
]
