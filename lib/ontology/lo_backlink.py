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

import re
from typing import Any, Dict, List, Optional

from lib.embedding._math import cosine_similarity

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{1,}")


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


def backlink_cos_to_tos(
    terminals: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    *,
    embed: Optional[Any] = None,
) -> None:
    """Set ``terminal_id`` on every CO in ``chapters`` to its parent TO (in place).

    Args:
        terminals: the canonical TO-NN list (each with ``id`` + ``statement``).
        chapters: the canonical CO-NN list (mutated in place).
        embed: optional embedding client (reuse the Pass-D one). ``None`` →
            token-overlap fallback (no model load attempted here; the caller owns
            embed-client lifecycle).
    """
    if not chapters:
        return
    to_ids = [str(t.get("id") or "") for t in terminals]
    to_statements = [_statement(t) for t in terminals]
    valid_to_ids = {tid for tid in to_ids if tid}

    if not valid_to_ids:
        # No terminal to link to — leave terminal_id unset is not allowed, but
        # there is genuinely no parent. Stamp empty-string sentinel so a reader
        # can tell "no TO existed" apart from "unprocessed".
        for co in chapters:
            co.setdefault("terminal_id", "")
        return

    # Single-TO shortcut: every CO rolls up to the one terminal.
    if len(valid_to_ids) == 1:
        only = next(iter(valid_to_ids))
        for co in chapters:
            co["terminal_id"] = only
        return

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

    for co_idx, co in enumerate(chapters):
        # (1) Honor a resolvable reconcile hint.
        hint = str(
            co.get("terminal_id") or co.get("parent_terminal_id") or ""
        ).strip()
        if hint in valid_to_ids:
            co["terminal_id"] = hint
            continue

        best_idx = 0
        if to_vecs is not None and co_vecs is not None:
            best_score = -2.0
            for t_idx in range(len(terminals)):
                if not to_ids[t_idx]:
                    continue
                score = cosine_similarity(co_vecs[co_idx], to_vecs[t_idx])
                if score > best_score:
                    best_score = score
                    best_idx = t_idx
        else:
            # Token-overlap fallback.
            co_stmt = _statement(co)
            best_score = -1.0
            for t_idx in range(len(terminals)):
                if not to_ids[t_idx]:
                    continue
                score = _token_overlap(co_stmt, to_statements[t_idx])
                if score > best_score:
                    best_score = score
                    best_idx = t_idx
        co["terminal_id"] = to_ids[best_idx] if to_ids[best_idx] else next(
            iter(valid_to_ids)
        )


__all__ = ["backlink_cos_to_tos"]
