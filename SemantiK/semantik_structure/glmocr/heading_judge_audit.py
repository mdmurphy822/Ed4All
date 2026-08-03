"""Deterministic post-judge AUDIT of the GLM-OCR heading-judge output.

The Super heading-level judge (:mod:`.heading_judge`) re-levels every
``heading_level_pending`` heading, but it is FAIL-OPEN by contract: a chapter
whose seat POST(s) failed / over-deliberated / mode-collapsed keeps its
pre-judge levels, and past a per-chapter counter that degradation is currently
SILENT. This module is the deterministic, pure-function AUDIT over the JUDGED
sidecars (the ``region_provenance`` a judged ``{stem}.corrected_layout.json``
carries — heading regions stamped ``level`` / ``heading_level_pending`` /
``heading_level_judged``) that makes those failures ACTIONABLE.

Three arms:

* **Arm A — completeness / loud-failure (ACTIONABLE).** Chapters that did NOT
  complete judging — residual ``heading_level_pending`` headings after the
  judge ran (the ``chapters_unjudged`` fail-open class). Returns the chapter
  stems + residual-pending counts.
* **Arm B — bucket-collapse / degeneracy (ACTIONABLE).** Per-chapter judged
  level distribution; flags a chapter whose judged output is degenerate (all
  headings at ONE level, OR 0 verdicts applied while ≥N pending existed, OR
  >X% of headings sit at a single level). This is the classifier
  bucket-collapse audit pattern — pure detection, no LLM.
* **Arm C — cross-chapter consistency (REPORT-ONLY, never mutates).** A map of
  normalized recurring section-title signature → the {chapter, level, parent}
  it occurs with across ALL chapters; signatures occurring with >1 distinct
  level are reported. Each occurrence is ANNOTATED with its parent-heading
  context, because the same title at different tree DEPTHS is legitimately
  different (a chapter-end "Summary" vs an in-section "Summary"). Arm C is
  ADVISORY for human/owner review and NEVER normalizes or mutates — and it
  NEVER contributes to ``flagged_chapters``.

**DETERMINISTIC → NO LLM call site → NO DecisionCapture**: a pure function over
persisted artifacts emits no capture. **Cross-venv clean:** ZERO Ed4All ``lib/`` imports (stdlib
only), mirroring the ``stop_seam`` / ``heading_judge`` twin posture, so it runs
in the SemantiK venv beside the cascade. **Best-effort / fail-open:** the audit
is a pure READ pass — it never mutates a sidecar or HTML, and the
``heading_judge`` phase's fail-open contract is unaffected (the audit failing
must NEVER fail the phase).

All env resolvers are parse-with-fallback (unset / blank / garbage →
the documented default), the root ``ED4ALL_*`` / SemantiK ``SEMANTIK_*``
posture.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

#: Report-shape version — bumped when the report dict shape changes so a
#: downstream reader can pin the shape it understands.
AUDIT_SCHEMA_VERSION = 1

# ── Calibration thresholds (env-overridable, parse-with-fallback). ───────────
# TODO(calibration): the collapse share + min-headings floor are initial
# domain-agnostic guesses ("a judged chapter should not land ~everything on one
# level"), NOT corpus targets — measure the false-positive rate on real judged
# corpora before treating them as load-bearing.
_DEFAULT_COLLAPSE_SHARE = 0.95
_DEFAULT_MIN_HEADINGS = 4

_COLLAPSE_SHARE_ENV = "SEMANTIK_HEADING_JUDGE_AUDIT_COLLAPSE_SHARE"
_MIN_HEADINGS_ENV = "SEMANTIK_HEADING_JUDGE_AUDIT_MIN_HEADINGS"

# ── Per-chapter flag codes. ──────────────────────────────────────────────────
FLAG_INCOMPLETE = "incomplete_judging"           # Arm A
FLAG_COLLAPSE_SINGLE_LEVEL = "level_collapse_single"  # Arm B
FLAG_COLLAPSE_SHARE = "level_collapse_share"          # Arm B
FLAG_ZERO_APPLIED = "zero_applied_with_pending"       # Arm B

#: The flag codes whose presence puts a chapter on ``flagged_chapters`` (the
#: re-judge target set) — Arm A + Arm B only; Arm C never contributes.
_ACTIONABLE_FLAGS = frozenset({
    FLAG_INCOMPLETE,
    FLAG_COLLAPSE_SINGLE_LEVEL,
    FLAG_COLLAPSE_SHARE,
    FLAG_ZERO_APPLIED,
})
_COLLAPSE_FLAGS = frozenset({
    FLAG_COLLAPSE_SINGLE_LEVEL,
    FLAG_COLLAPSE_SHARE,
    FLAG_ZERO_APPLIED,
})


def resolve_audit_collapse_share() -> float:
    """Arm-B single-level share floor at/above which a chapter is collapsed.

    Float parse-with-fallback in ``[0, 1]``: blank / non-float / NaN / ±Inf /
    out-of-range → the default ``0.95``.
    """
    raw = (os.environ.get(_COLLAPSE_SHARE_ENV) or "").strip()
    if not raw:
        return _DEFAULT_COLLAPSE_SHARE
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_COLLAPSE_SHARE
    if val != val or val in (float("inf"), float("-inf")):
        return _DEFAULT_COLLAPSE_SHARE
    if val < 0.0 or val > 1.0:
        return _DEFAULT_COLLAPSE_SHARE
    return val


def resolve_audit_min_headings() -> int:
    """Minimum heading count below which Arm-B never flags a chapter collapsed
    (so a 2-heading chapter is never "collapsed"). Positive-int
    parse-with-fallback → the default ``4``.
    """
    raw = (os.environ.get(_MIN_HEADINGS_ENV) or "").strip()
    if not raw:
        return _DEFAULT_MIN_HEADINGS
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_MIN_HEADINGS
    return val if val >= 1 else _DEFAULT_MIN_HEADINGS


# ── Heading accessors (over a judged region_provenance list). ────────────────
def _headings(region_provenance: Sequence[Any]) -> List[Dict[str, Any]]:
    """Heading regions in document order (defensive over a malformed list)."""
    out: List[Dict[str, Any]] = []
    for r in region_provenance or []:
        if isinstance(r, dict) and r.get("region_kind") == "heading":
            out.append(r)
    return out


def _heading_level(region: Dict[str, Any]) -> int:
    try:
        return int(region.get("level", 3) or 3)
    except (TypeError, ValueError):
        return 3


_LEADING_NUM_RE = re.compile(r"^\s*\d+(?:\.\d+)*[.):]?\s+")


def normalize_signature(text: Any) -> str:
    """Casefold + strip a leading ``N`` / ``N.M`` / ``N.M.K`` numbering token +
    collapse whitespace → the recurring-title signature for Arm C.

    A heading that is ONLY a number keeps its original text (never empties).
    """
    collapsed = " ".join(str(text or "").split())
    stripped = _LEADING_NUM_RE.sub("", collapsed)
    return (stripped or collapsed).casefold().strip()


def _parent_texts(headings: Sequence[Dict[str, Any]]) -> Dict[int, Optional[str]]:
    """Nearest preceding STRICTLY-shallower heading text per heading index.

    This is the parent-heading CONTEXT Arm C annotates each occurrence with —
    the same title at different tree DEPTHS is legitimately different, so the
    parent lets a human tell a chapter-end "Summary" from an in-section one.
    """
    parents: Dict[int, Optional[str]] = {}
    stack: List[tuple] = []  # (level, text)
    for i, h in enumerate(headings):
        lvl = _heading_level(h)
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        parents[i] = stack[-1][1] if stack else None
        stack.append((lvl, str(h.get("heading_text") or "")))
    return parents


# ── Arm A + Arm B: per-chapter. ──────────────────────────────────────────────
def build_chapter_audit(
    stem: str, region_provenance: Sequence[Any]
) -> Dict[str, Any]:
    """One chapter's audit row: level distribution + Arm-A + Arm-B flags.

    Pure over the judged ``region_provenance`` — reads only, mutates nothing.
    """
    headings = _headings(region_provenance)
    n = len(headings)
    dist: Dict[int, int] = {}
    n_residual = 0
    n_judged = 0
    for h in headings:
        lvl = _heading_level(h)
        dist[lvl] = dist.get(lvl, 0) + 1
        if h.get("heading_level_pending"):
            n_residual += 1
        if h.get("heading_level_judged"):
            n_judged += 1

    flags: List[Dict[str, Any]] = []

    # Arm A — completeness / loud-failure: residual pending headings mean the
    # judge did NOT complete judging this chapter (fail-open / unjudged class).
    if n_residual > 0:
        flags.append({
            "code": FLAG_INCOMPLETE,
            "detail": (
                f"{n_residual} residual heading_level_pending heading(s) after "
                f"the judge ran (fail-open; judging did not complete)"
            ),
        })

    # Arm B — bucket-collapse / degeneracy (guarded by the min-headings floor so
    # a tiny chapter is never called "collapsed").
    min_headings = resolve_audit_min_headings()
    share = resolve_audit_collapse_share()
    if n >= min_headings:
        distinct_levels = len(dist)
        if distinct_levels <= 1:
            flags.append({
                "code": FLAG_COLLAPSE_SINGLE_LEVEL,
                "detail": (
                    f"all {n} heading(s) sit at a single level "
                    f"{sorted(dist)} — degenerate hierarchy"
                ),
            })
        else:
            top = max(dist.values())
            if n and (top / n) >= share:
                flags.append({
                    "code": FLAG_COLLAPSE_SHARE,
                    "detail": (
                        f"{top}/{n} heading(s) ({top / n:.2f}) sit at one level "
                        f">= collapse_share {share:.2f}"
                    ),
                })
        if n_judged == 0 and n_residual >= min_headings:
            flags.append({
                "code": FLAG_ZERO_APPLIED,
                "detail": (
                    f"0 judged verdict(s) applied while {n_residual} heading(s) "
                    f"were pending — the judge changed nothing"
                ),
            })

    return {
        "stem": stem,
        "n_headings": n,
        "n_residual_pending": n_residual,
        "n_judged": n_judged,
        "level_distribution": {str(k): dist[k] for k in sorted(dist)},
        "flags": flags,
    }


# ── Arm C: cross-chapter consistency (REPORT-ONLY, never mutates). ───────────
def cross_chapter_signatures(
    chapters: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Recurring normalized section-title signatures that occur with >1 level.

    ADVISORY only: each occurrence carries its chapter, assigned level, parent
    heading text (the DEPTH-context caveat), and verbatim text. This arm NEVER
    normalizes or mutates a sidecar — the same title at different tree depths is
    legitimately different, so it is surfaced for human/owner review and never
    feeds ``flagged_chapters``.
    """
    sig_occ: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ch in chapters:
        stem = ch.get("stem")
        headings = _headings(ch.get("region_provenance") or [])
        parents = _parent_texts(headings)
        for i, h in enumerate(headings):
            sig = normalize_signature(h.get("heading_text"))
            if not sig:
                continue
            sig_occ[sig].append({
                "stem": stem,
                "level": _heading_level(h),
                "parent": parents.get(i),
                "text": str(h.get("heading_text") or ""),
            })
    inconsistent: List[Dict[str, Any]] = []
    for sig in sorted(sig_occ):
        occ = sig_occ[sig]
        levels = sorted({o["level"] for o in occ})
        if len(levels) > 1:
            inconsistent.append({
                "signature": sig,
                "levels": levels,
                "occurrences": occ,
            })
    return inconsistent


# ── The report. ──────────────────────────────────────────────────────────────
def build_audit_report(chapters: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the versioned audit report over judged chapters.

    ``chapters``: a sequence of ``{"stem": str, "region_provenance": [...]}``
    dicts (the judged region list per chapter). Returns the report dict:

    * ``chapters``: per-chapter ``{stem, n_headings, n_residual_pending,
      n_judged, level_distribution, flags:[]}`` (Arm A + Arm B).
    * ``book``: ``{level_distribution, incomplete_chapters:[],
      collapsed_chapters:[], inconsistent_signatures:[{signature,
      occurrences:[...]}]}``.
    * ``flagged_chapters``: the union of Arm-A + Arm-B stems (the re-judge
      target set). Arm C NEVER contributes to it.
    """
    chapter_reports: List[Dict[str, Any]] = []
    book_dist: Dict[str, int] = {}
    incomplete: List[Dict[str, Any]] = []
    collapsed: List[Dict[str, Any]] = []
    flagged: List[str] = []

    for ch in chapters:
        cr = build_chapter_audit(
            ch.get("stem"), ch.get("region_provenance") or []
        )
        chapter_reports.append(cr)
        for k, v in cr["level_distribution"].items():
            book_dist[k] = book_dist.get(k, 0) + v
        codes = {f["code"] for f in cr["flags"]}
        if FLAG_INCOMPLETE in codes:
            incomplete.append({
                "stem": cr["stem"],
                "n_residual_pending": cr["n_residual_pending"],
            })
        collapse_codes = sorted(codes & _COLLAPSE_FLAGS)
        if collapse_codes:
            collapsed.append({"stem": cr["stem"], "reasons": collapse_codes})
        if codes & _ACTIONABLE_FLAGS and cr["stem"] is not None:
            flagged.append(cr["stem"])

    inconsistent = cross_chapter_signatures(chapters)

    def _level_key(s: str) -> Any:
        try:
            return (0, int(s))
        except (TypeError, ValueError):
            return (1, s)

    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "thresholds": {
            "collapse_share": resolve_audit_collapse_share(),
            "min_headings": resolve_audit_min_headings(),
        },
        "n_chapters": len(chapter_reports),
        "chapters": chapter_reports,
        "book": {
            "level_distribution": {
                k: book_dist[k] for k in sorted(book_dist, key=_level_key)
            },
            "incomplete_chapters": incomplete,
            "collapsed_chapters": collapsed,
            "inconsistent_signatures": inconsistent,
        },
        "flagged_chapters": sorted(set(flagged)),
    }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "FLAG_INCOMPLETE",
    "FLAG_COLLAPSE_SINGLE_LEVEL",
    "FLAG_COLLAPSE_SHARE",
    "FLAG_ZERO_APPLIED",
    "resolve_audit_collapse_share",
    "resolve_audit_min_headings",
    "normalize_signature",
    "build_chapter_audit",
    "cross_chapter_signatures",
    "build_audit_report",
]
