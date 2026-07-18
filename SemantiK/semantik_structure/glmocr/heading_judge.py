"""Super heading-level JUDGE pass for the GLM-OCR lane (``SEMANTIK_HEADING_JUDGE``).

The deterministic transform (:mod:`.transform`) defaults every non-numbered
title to ``level=3`` and flags it ``heading_level_pending`` + a matching
``heading_level_pending`` escalation row. That is honest (the sidecar contract
records the uncertainty) but leaves the tree flat: a section's body subsection,
its objectives-list restatement, and its end-of-chapter review restatement all
sit at the same level. This pass hands the WHOLE chapter's ordered heading
skeleton to a reasoning model (Super/Nemotron-3), which assigns the correct
level (2-6) to EVERY pending heading so the tree is hierarchy-consistent, then
DETERMINISTICALLY re-stamps the pending headings (level only, never text) under
strict clamp rules that mirror the ``_anchor_declared_sections`` re-stamp +
escalation-hygiene precedent.

Design rule (the SemantiK house rule): the model PROPOSES levels; deterministic
code DECIDES. Every clamp is auditable, the fixed N.M spine / synthesized
chapter opener are immovable anchors, and any transport/parse failure fails
OPEN — the chapter keeps ALL current levels byte-identically and the pending
flags + escalations are RETAINED (no silent resolution).

Off (default) — ``SEMANTIK_HEADING_JUDGE`` unset — this module is never imported
by the lane, so ``region_provenance`` / ``heading_tree`` / escalations are
byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import (
    resolve_heading_judge_base_url,
    resolve_heading_judge_checkpoint,
    resolve_heading_judge_model,
    resolve_heading_judge_timeout,
)

logger = logging.getLogger(__name__)

# Prompt-contract version — folded into the cache key so a prompt change moves
# every sidecar key (a stale verdict is never served for a changed prompt).
JUDGE_PROMPT_VERSION = 1

# Token-budget target for one chapter digest (the ~24k whole-book ceiling; a
# 650-heading chapter ≈ 11k with pending-only anchors). A digest over this
# drops content anchors, then splits into N.M-anchored windows.
_DIGEST_BUDGET_TOKENS = 24000

_HEADING_TEXT_TRUNCATE = 90
_CONTEXT_TEXT_TRUNCATE = 40
_ANCHOR_TRUNCATE = 80

_TRANSIENT_RETRIES = 2  # extra attempts after the first, linear backoff
_MAX_TOKENS_CEILING = 30000
_MAX_TOKENS_FLOOR = 4096
_MAX_TOKENS_PER_PENDING = 24

_CACHE_BASENAME = "heading_judge_cache"

# The thinking-directive tuple folded into the cache key — this pass is
# thinking-ON by design (genuine relational reasoning), effort high.
_THINKING_DIRECTIVE = ("detailed thinking on", "reasoning effort: high")

_JUDGE_INSTRUCTIONS = (
    "You are a document-structure judge. You are given the complete ordered "
    "heading skeleton of one textbook chapter. Lines marked \"*\" are PENDING "
    "headings whose level was defaulted; lines marked \".\" are FIXED anchors "
    "(the chapter title and the numbered N.M section spine) that are correct "
    "and immutable. Your task: assign the correct heading level (an integer "
    "2-6) to EVERY pending heading so the tree is hierarchy-consistent — peers "
    "share a level, a child is exactly one level deeper than its parent, and "
    "every pending heading hangs under the nearest preceding fixed anchor or "
    "corrected pending parent. Typical patterns: a chapter's learning-objective "
    "topic titles repeat three times (as objectives list context, as the "
    "section-body subsection, and again inside end-of-chapter review/exercises) "
    "— the section-body occurrence is a child of its N.M section (level 3); "
    "review restatements sit under the review apparatus (usually level 4); "
    "apparatus titles like \"Learning Objectives\", \"Practice Makes Perfect\", "
    "\"Chapter Outline\" are children of their enclosing section. Never invent, "
    "drop, reorder, or rewrite headings. Never assign level 1. Judge only the "
    "ids marked \"*\".\n"
    "Respond with ONLY a JSON object, no prose, in exactly this shape:\n"
    "{\"levels\": {\"<region_id>\": <level int>, ...}}\n"
    "with one entry per PENDING region id (the integer after \"R\", quoted as "
    "a string key)."
)


# ── Skeleton model. ─────────────────────────────────────────────────────────
@dataclass
class SkeletonEntry:
    region_index: int
    level: int
    text: str
    source_page: Any
    pending: bool
    anchor: Optional[str] = None


@dataclass
class SkeletonPlan:
    entries: List[SkeletonEntry]
    digest: str  # the primary single-window digest (report + cache-key stable)
    pending_ids: List[int]
    windows: List[Tuple[str, List[int]]]  # (digest, [pending region ids]) per POST


@dataclass
class ApplyResult:
    applied: int = 0
    clamped: int = 0
    dropped: int = 0
    kept: int = 0
    corrections: Dict[int, Tuple[int, int, bool]] = field(default_factory=dict)


class _JudgeTransportError(Exception):
    """A POST transport error. ``transient`` decides whether it is retried."""

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


# ── Skeleton construction. ──────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_SENT_RE = re.compile(r"^(.+?[.!?])(\s|$)", re.S)


def _first_sentence(text: str, *, limit: int = _ANCHOR_TRUNCATE) -> str:
    t = " ".join((text or "").split())
    m = _SENT_RE.match(t)
    s = m.group(1) if m else t
    return s[:limit]


def _next_prose_sentence(prov: Sequence[Dict[str, Any]], pos: int) -> Optional[str]:
    """First sentence of the next plain-prose region after ``pos`` (≤80 chars)."""
    for r in prov[pos + 1:]:
        if r.get("region_kind") == "paragraph":
            body = str(r.get("raw_text") or "").strip()
            if body:
                return _first_sentence(body) or None
    return None


def _render_line(e: SkeletonEntry) -> str:
    mark = "*" if e.pending else "."
    page = e.source_page if e.source_page is not None else "?"
    return f"R{e.region_index} p{page} h{e.level}{mark} {e.text[:_HEADING_TEXT_TRUNCATE]}"


def _render_digest(entries: Sequence[SkeletonEntry], *, anchors: bool) -> str:
    lines: List[str] = []
    for e in entries:
        lines.append(_render_line(e))
        if anchors and e.pending and e.anchor:
            lines.append(f"    > {e.anchor}")
    return "\n".join(lines)


def _split_windows(entries: Sequence[SkeletonEntry]) -> List[Tuple[str, List[int]]]:
    """Overflow ladder step 2: split at fixed level-2 N.M anchors into windows,
    each carrying the chapter's full FIXED-anchor outline (marks + 40-char
    texts) as immutable context."""
    ctx_lines = [
        f"R{e.region_index} p{e.source_page} h{e.level}. {e.text[:_CONTEXT_TEXT_TRUNCATE]}"
        for e in entries if not e.pending
    ]
    ctx = "\n".join(ctx_lines)
    segments: List[List[SkeletonEntry]] = []
    cur: List[SkeletonEntry] = []
    for e in entries:
        if (not e.pending) and e.level == 2 and cur:
            segments.append(cur)
            cur = []
        cur.append(e)
    if cur:
        segments.append(cur)
    if len(segments) <= 1:
        # No level-2 spine to split on — fall back to one no-anchor window.
        d = _render_digest(entries, anchors=False)
        return [(d, [e.region_index for e in entries if e.pending])]
    windows: List[Tuple[str, List[int]]] = []
    for seg in segments:
        body = _render_digest(seg, anchors=False)
        digest = (
            "FIXED-ANCHOR OUTLINE (immutable context — the chapter spine):\n"
            f"{ctx}\n\nWINDOW HEADINGS:\n{body}"
        )
        windows.append((digest, [e.region_index for e in seg if e.pending]))
    return windows


def build_heading_skeleton(
    region_provenance: Sequence[Dict[str, Any]],
) -> SkeletonPlan:
    """Build the ordered heading skeleton + compact digest + window plan.

    HEADING regions only, in document order. One line per heading; pending nodes
    optionally carry a content anchor (first sentence of the next prose region).
    """
    prov = list(region_provenance)
    entries: List[SkeletonEntry] = []
    for pos, r in enumerate(prov):
        if r.get("region_kind") != "heading":
            continue
        ridx = r.get("first_raw_block_index")
        if ridx is None:
            continue
        text = str(r.get("heading_text") or "")
        try:
            level = int(r.get("level", 3) or 3)
        except (TypeError, ValueError):
            level = 3
        page = r.get("source_page")
        if page is None:
            pages = r.get("pages") or []
            page = pages[0] if pages else None
        pending = bool(r.get("heading_level_pending"))
        anchor = _next_prose_sentence(prov, pos) if pending else None
        entries.append(SkeletonEntry(int(ridx), level, text, page, pending, anchor))

    pending_ids = [e.region_index for e in entries if e.pending]

    digest_with = _render_digest(entries, anchors=True)
    if _estimate_tokens(digest_with) <= _DIGEST_BUDGET_TOKENS:
        return SkeletonPlan(entries, digest_with, pending_ids,
                            [(digest_with, pending_ids)])
    # ladder 1: drop content anchors.
    digest_without = _render_digest(entries, anchors=False)
    if _estimate_tokens(digest_without) <= _DIGEST_BUDGET_TOKENS:
        return SkeletonPlan(entries, digest_without, pending_ids,
                            [(digest_without, pending_ids)])
    # ladder 2: split into N.M-anchored windows.
    windows = _split_windows(entries)
    return SkeletonPlan(entries, digest_without, pending_ids, windows)


# ── Prompt + POST. ──────────────────────────────────────────────────────────
def build_judge_messages(
    digest: str, n_headings: int, n_pending: int
) -> List[Dict[str, str]]:
    system = "detailed thinking on\n" + _JUDGE_INSTRUCTIONS
    user = (
        "reasoning effort: high\n\n"
        f"Chapter heading skeleton ({n_headings} headings, {n_pending} pending):\n\n"
        f"{digest}\n\n"
        "Return the JSON now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _max_tokens_for(n_pending: int) -> int:
    return max(
        _MAX_TOKENS_FLOOR,
        min(_MAX_TOKENS_CEILING, _MAX_TOKENS_FLOOR + _MAX_TOKENS_PER_PENDING * n_pending),
    )


@dataclass
class HeadingJudgeSeat:
    base_url: str
    api_key: Optional[str]
    model: str
    timeout: float


def resolve_heading_judge_seat() -> HeadingJudgeSeat:
    api_key = os.environ.get("SEMANTIK_HEADING_JUDGE_API_KEY")
    api_key = api_key.strip() if api_key and api_key.strip() else None
    return HeadingJudgeSeat(
        base_url=resolve_heading_judge_base_url(),
        api_key=api_key,
        model=resolve_heading_judge_model(),
        timeout=resolve_heading_judge_timeout(),
    )


def _post_judge_completion(
    *,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: Sequence[Dict[str, str]],
    max_tokens: int,
    timeout: float,
    requests_module,
) -> Tuple[Optional[str], Optional[str]]:
    """One thinking-ON judge POST → ``(content, finish_reason)``.

    thinking is controlled ONLY by the system-prompt "detailed thinking on"
    directive (the ``reasoning_budget`` / ``chat_template_kwargs`` knobs are DEAD
    on this seat), so no thinking kwarg is emitted. The seat runs
    ``reasoning_parser nemotron_v3`` — reasoning may land in a separate
    ``reasoning_content`` channel; the ANSWER is ``choices[0].message.content``,
    which is what we parse.
    """
    from ..vlm_extract import _chat_completions_url

    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = _chat_completions_url(base_url)
    try:
        resp = requests_module.post(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — timeout / conn → transient
        raise _JudgeTransportError(
            f"heading-judge request failed ({type(exc).__name__}): {exc}",
            transient=True,
        ) from exc
    status = int(getattr(resp, "status_code", 200))
    if status != 200:
        transient = status >= 500 or status == 429
        head = (getattr(resp, "text", "") or "")[:300]
        raise _JudgeTransportError(
            f"heading-judge endpoint HTTP {status} (model={model}): {head!r}",
            transient=transient,
        )
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise _JudgeTransportError(
            "heading-judge endpoint returned a non-JSON body", transient=False
        ) from exc
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content")
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        keys = list(data) if isinstance(data, dict) else type(data)
        raise _JudgeTransportError(
            f"heading-judge malformed response (no choices[0].message): keys={keys}",
            transient=False,
        ) from exc
    return (None if content is None else str(content)), finish


def _make_default_post_fn(
    seat: HeadingJudgeSeat, *, requests_module=None
) -> Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]:
    """A ``post_fn(messages, max_tokens) -> (content, finish)`` wrapping the
    real seat with the bounded transient-retry ladder."""
    if requests_module is None:
        from ..vlm_extract import _lazy_requests

        requests_module = _lazy_requests()

    def _fn(
        messages: Sequence[Dict[str, str]], max_tokens: int
    ) -> Tuple[Optional[str], Optional[str]]:
        import time as _time

        attempts = _TRANSIENT_RETRIES + 1
        for i in range(attempts):
            try:
                return _post_judge_completion(
                    base_url=seat.base_url,
                    api_key=seat.api_key,
                    model=seat.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=seat.timeout,
                    requests_module=requests_module,
                )
            except _JudgeTransportError as exc:
                if not exc.transient or i == attempts - 1:
                    raise
                _time.sleep(0.5 * (i + 1))
        raise _JudgeTransportError("heading-judge retry ladder exhausted", transient=True)

    return _fn


# ── Response parse. ─────────────────────────────────────────────────────────
def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # drop opening fence line and trailing fence
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_balanced_json(s: str) -> Optional[str]:
    """Return the first balanced ``{...}`` object substring (string-aware)."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_judge_response(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Strict parse → ``{"levels": {...}}`` or ``None`` on any failure."""
    if raw is None:
        return None
    obj = _extract_balanced_json(_strip_fences(raw))
    if obj is None:
        return None
    try:
        data = json.loads(obj)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    levels = data.get("levels")
    if not isinstance(levels, dict):
        return None
    return {"levels": levels}


# ── Cache (content-addressed sidecar; mirrors reasoning_qc_cache). ───────────
def _judge_cache_root() -> Path:
    from . import paths as _semantik_paths

    return _semantik_paths.resolve_cache(_CACHE_BASENAME)


def _window_cache_key(digest: str, model: str, max_tokens: int) -> str:
    raw = (
        f"{hashlib.sha256(digest.encode('utf-8')).hexdigest()}|{model}|"
        f"{JUDGE_PROMPT_VERSION}|{max_tokens}|0.0|{'/'.join(_THINKING_DIRECTIVE)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str, root: Path) -> Path:
    return root / key[:2] / f"{key}.json"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    try:
        path = _cache_path(key, _judge_cache_root())
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception as exc:  # noqa: BLE001 — a corrupt sidecar → miss
        logger.debug("heading-judge cache read failed (non-fatal): %s", exc)
        return None


def _cache_put(key: str, verdict: Dict[str, Any]) -> None:
    try:
        path = _cache_path(key, _judge_cache_root())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(verdict, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 — a cache write must never break judging
        logger.debug("heading-judge cache write failed (non-fatal): %s", exc)


# ── Judge orchestration. ─────────────────────────────────────────────────────
def _judge_one_window(
    digest: str,
    n_headings: int,
    n_pending: int,
    max_tokens: int,
    *,
    post_fn: Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """POST one window (retry once on ``finish_reason=length`` with doubled
    ``max_tokens``); returns ``(parsed_or_None, wmeta)``."""
    messages = build_judge_messages(digest, n_headings, n_pending)
    try:
        content, finish = post_fn(messages, max_tokens)
    except _JudgeTransportError:
        return None, {"transport_failure": True, "finish": None}
    if finish == "length":
        new_max = min(_MAX_TOKENS_CEILING, max_tokens * 2)
        try:
            content, finish = post_fn(messages, new_max)
        except _JudgeTransportError:
            return None, {"transport_failure": True, "finish": "length"}
        if finish == "length":
            return None, {"length_exhausted": True, "finish": "length"}
    parsed = parse_judge_response(content)
    if parsed is None:
        return None, {"parse_failure": True, "finish": finish}
    return parsed, {"finish": finish}


def judge_heading_levels(
    plan: SkeletonPlan,
    *,
    seat: Optional[HeadingJudgeSeat] = None,
    post_fn: Optional[Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]] = None,
    use_cache: Optional[bool] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """One seat call per window → ``{region_index: proposed_level}`` (unfiltered
    levels; :func:`apply_judged_levels` owns range + clamp validation).

    Fail-soft: a window whose POST/parse fails contributes nothing (its pendings
    keep their current level). Never invents a level.
    """
    model = model or (seat.model if seat else resolve_heading_judge_model())
    if use_cache is None:
        use_cache = resolve_heading_judge_checkpoint()
    if post_fn is None:
        post_fn = _make_default_post_fn(seat or resolve_heading_judge_seat())

    verdict_map: Dict[int, int] = {}
    meta: Dict[str, Any] = {
        "windows": len(plan.windows), "posts": 0, "cache_hits": 0,
        "cache_misses": 0, "length_exhausted": 0, "parse_failures": 0,
        "transport_failures": 0, "finish": None, "cache_hit": False,
    }
    n_headings = len(plan.entries)
    for digest, win_pending in plan.windows:
        n_pending = len(win_pending)
        if n_pending == 0:
            continue
        max_tokens = _max_tokens_for(n_pending)
        key = _window_cache_key(digest, model, max_tokens) if use_cache else None
        parsed: Optional[Dict[str, Any]] = None
        if key is not None:
            parsed = _cache_get(key)
        if parsed is not None:
            meta["cache_hits"] += 1
            meta["cache_hit"] = True
        else:
            meta["posts"] += 1
            parsed, wmeta = _judge_one_window(
                digest, n_headings, n_pending, max_tokens, post_fn=post_fn)
            meta["finish"] = wmeta.get("finish")
            if wmeta.get("length_exhausted"):
                meta["length_exhausted"] += 1
            if wmeta.get("parse_failure"):
                meta["parse_failures"] += 1
            if wmeta.get("transport_failure"):
                meta["transport_failures"] += 1
            # Cache only a GENUINE verdict (≥1 level); a fail-open / empty result
            # is never cached (rule 7).
            if key is not None and parsed is not None and parsed.get("levels"):
                _cache_put(key, parsed)
            meta["cache_misses"] += 1
        if parsed:
            for k, v in (parsed.get("levels") or {}).items():
                try:
                    ik = int(str(k).strip())
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                verdict_map[ik] = iv
    meta["max_tokens"] = _max_tokens_for(len(plan.pending_ids))
    return verdict_map, meta


# ── Deterministic re-stamp (clamp rules). ────────────────────────────────────
def apply_judged_levels(
    region_provenance: List[Dict[str, Any]],
    heading_tree: List[Any],
    escalations: List[Dict[str, Any]],
    verdict_map: Dict[int, int],
) -> ApplyResult:
    """Re-stamp PENDING headings only, under the clamp rules. Mirrors
    ``transform._anchor_declared_sections``: level only (never text), then the
    heading_tree is rebuilt from region_provenance.

    Rule 5 fail-open: an empty ``verdict_map`` → NO mutation (byte-identical;
    pending flags + escalations retained)."""
    result = ApplyResult()
    if not verdict_map:
        return result

    prov = region_provenance
    # Index the writable (pending) heading regions by region id.
    pending_by_id: Dict[int, Dict[str, Any]] = {}
    for r in prov:
        if r.get("region_kind") == "heading" and bool(r.get("heading_level_pending")):
            rid = r.get("first_raw_block_index")
            if rid is not None:
                pending_by_id[int(rid)] = r

    # Rule 1 + 2 + per-entry validation: drop unknown / non-pending / out-of-range.
    accepted: Dict[int, int] = {}
    for rid, lvl in verdict_map.items():
        if rid not in pending_by_id:
            result.dropped += 1  # unknown id or a fixed-anchor target
            continue
        try:
            ilvl = int(lvl)
        except (TypeError, ValueError):
            result.dropped += 1
            continue
        if ilvl < 2 or ilvl > 6:  # rule 2: level 1 never assignable; range [2,6]
            result.dropped += 1
            continue
        accepted[rid] = ilvl

    if not accepted:
        # Everything dropped → nothing to apply (a fail-open keep-current).
        return result

    # Effective-level walk in document order (rules 3 + 4).
    headings = [r for r in prov if r.get("region_kind") == "heading"]
    prev_effective = 1
    enclosing_anchor = 1
    corrections: Dict[int, Tuple[int, int, bool]] = {}
    for h in headings:
        rid = h.get("first_raw_block_index")
        is_pending = bool(h.get("heading_level_pending"))
        try:
            cur_level = int(h.get("level", 3) or 3)
        except (TypeError, ValueError):
            cur_level = 3
        if is_pending and rid is not None and int(rid) in accepted:
            proposed = accepted[int(rid)]
            eff = proposed
            clamped = False
            # Rule 4: NO ORPHANING — never above the enclosing fixed anchor + 1.
            lo = enclosing_anchor + 1
            if eff < lo:
                eff, clamped = lo, True
            # Rule 3: PARENT+1 MAX JUMP — never deeper than prev_effective + 1.
            hi = prev_effective + 1
            if eff > hi:
                eff, clamped = hi, True
            # Rule 2 re-clamp (lo/hi could push out of range on a pathological tree).
            bounded = max(2, min(6, eff))
            if bounded != eff:
                clamped = True
            eff = bounded
            corrections[int(rid)] = (cur_level, eff, clamped or eff != proposed)
            prev_effective = eff
            # A corrected pending node can PARENT a following pending node
            # (via prev_effective) but is NOT a fixed anchor for orphaning.
        else:
            eff = cur_level
            prev_effective = eff
            if not is_pending:
                enclosing_anchor = eff  # only fixed anchors set the orphan floor

    # Apply corrections deterministically + escalation hygiene (rule 6).
    for rid, (from_lvl, to_lvl, clamped) in corrections.items():
        region = pending_by_id[rid]
        region["level"] = to_lvl
        region.pop("heading_level_pending", None)
        region["heading_level_judged"] = {
            "from": from_lvl, "to": to_lvl, "clamped": bool(clamped)
        }
        # Remove the matching heading_level_pending escalation row.
        escalations[:] = [
            e for e in escalations
            if not (e.get("reason") == "heading_level_pending"
                    and e.get("region_index") == rid)
        ]
        escalations.append({
            "region_index": rid,
            "source_page": region.get("source_page"),
            "native_label": region.get("native_label", ""),
            "reason": "heading_level_judged",
            "detail": (f"pending heading level judged {from_lvl}->{to_lvl}"
                       f"{' (clamped)' if clamped else ''}"),
            "text": str(region.get("heading_text") or "")[:120],
        })
        result.applied += 1
        if clamped:
            result.clamped += 1
    result.corrections = corrections
    result.kept = len(pending_by_id) - result.applied

    # Rebuild heading_tree from the final provenance (identical to
    # _anchor_declared_sections' rebuild).
    heading_tree[:] = [
        (int(p.get("level", 3) or 3), str(p.get("heading_text", "")))
        for p in prov if p.get("region_kind") == "heading"
    ]
    return result


# ── DecisionCapture (NEW LLM call site). ─────────────────────────────────────
def _detect_course_code() -> str:
    raw = (
        os.environ.get("SEMANTIK_COURSE_CODE")
        or os.environ.get("ED4ALL_COURSE_CODE")
        or ""
    ).strip()
    return raw or "SEMANTIK"


def _emit_judge_capture(
    *,
    model: str,
    n_pending: int,
    result: ApplyResult,
    meta: Dict[str, Any],
    course_code: Optional[str] = None,
) -> None:
    """One ``structure_review`` DecisionCapture per chapter (best-effort).

    Rides the existing ``structure_review`` enum with a ``heading_level_judge``
    metadata discriminator (no ``decision_event`` schema change). Rationale is
    dynamic + replayable."""
    try:
        from lib.decision_capture import DecisionCapture
    except Exception as exc:  # noqa: BLE001 — the cross-venv bridge has no lib/
        logger.debug("heading-judge: DecisionCapture unavailable (non-fatal): %s", exc)
        return
    try:
        cap = DecisionCapture(
            course_code=course_code or _detect_course_code(),
            phase="semantik_conversion",
            tool="semantik",
        )
        cap.log_decision(
            decision_type="structure_review",
            decision=(
                f"heading_level_judge: applied {result.applied} of {n_pending} "
                f"pending heading level(s)"
            ),
            rationale=(
                f"Super heading-level judge (model={model}) judged {n_pending} "
                f"PENDING heading(s) across {meta.get('windows', 1)} window(s); the "
                f"deterministic clamp DECIDED — applied={result.applied}, "
                f"clamped={result.clamped}, dropped={result.dropped} "
                f"(unknown/non-pending/out-of-range), kept={result.kept} "
                f"(absent from the verdict → level unchanged); "
                f"max_tokens={meta.get('max_tokens')}, finish={meta.get('finish')}, "
                f"cache={'hit' if meta.get('cache_hit') else 'miss'}, "
                f"posts={meta.get('posts', 0)}, "
                f"length_exhausted={meta.get('length_exhausted', 0)}, "
                f"parse_failures={meta.get('parse_failures', 0)}, "
                f"transport_failures={meta.get('transport_failures', 0)}."
            ),
            alternatives_considered=[
                "keep every pending heading at the defaulted level 3 (flat tree)",
                "trust the model's proposed level unclamped (rejected: could skip "
                "a tier or orphan a heading above its section)",
            ],
            heading_level_judge=True,
            hj_applied=result.applied,
            hj_clamped=result.clamped,
            hj_dropped=result.dropped,
            hj_kept=result.kept,
            hj_model=model,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("heading-judge: decision capture failed (non-fatal): %s", exc)


# ── The one public seam the lane calls. ──────────────────────────────────────
def run_heading_judge(
    region_provenance: List[Dict[str, Any]],
    heading_tree: List[Any],
    escalations: List[Dict[str, Any]],
    *,
    seat: Optional[HeadingJudgeSeat] = None,
    post_fn: Optional[Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]] = None,
    use_cache: Optional[bool] = None,
    course_code: Optional[str] = None,
    emit_capture: bool = True,
) -> Dict[str, Any]:
    """Build the skeleton, judge pending levels, apply the clamp rules IN PLACE.

    Mutates ``region_provenance`` / ``heading_tree`` / ``escalations``. Returns a
    report dict. A no-pending chapter is a natural no-op (no POST)."""
    plan = build_heading_skeleton(region_provenance)
    n_pending = len(plan.pending_ids)
    if n_pending == 0:
        return {"n_pending": 0, "applied": 0, "clamped": 0, "dropped": 0,
                "kept": 0, "windows": len(plan.windows), "digest": plan.digest}

    verdict_map, meta = judge_heading_levels(
        plan, seat=seat, post_fn=post_fn, use_cache=use_cache)
    result = apply_judged_levels(
        region_provenance, heading_tree, escalations, verdict_map)

    if emit_capture:
        _emit_judge_capture(
            model=(seat.model if seat else resolve_heading_judge_model()),
            n_pending=n_pending, result=result, meta=meta, course_code=course_code)

    return {
        "n_pending": n_pending,
        "applied": result.applied,
        "clamped": result.clamped,
        "dropped": result.dropped,
        "kept": result.kept,
        "windows": len(plan.windows),
        "corrections": {str(k): v for k, v in result.corrections.items()},
        "meta": meta,
        "digest": plan.digest,
    }


__all__ = [
    "JUDGE_PROMPT_VERSION",
    "SkeletonEntry",
    "SkeletonPlan",
    "ApplyResult",
    "HeadingJudgeSeat",
    "build_heading_skeleton",
    "build_judge_messages",
    "resolve_heading_judge_seat",
    "parse_judge_response",
    "judge_heading_levels",
    "apply_judged_levels",
    "run_heading_judge",
]


if __name__ == "__main__":  # pragma: no cover — delegate to the standalone runner
    from .heading_judge_standalone import main

    raise SystemExit(main())
