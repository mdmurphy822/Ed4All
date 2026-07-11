"""Gold-set re-pin / migration (retrieval-answer-eval-set §3).

Deterministic, read-only-on-input, stdlib + repo-internal deps only (no LLM,
no network). Re-anchors a gold set's passage chunk_ids to a NEW pinned
chunkset, surviving the two id-fragility modes the plan calls out:

  * **Pure renumber** (the union-corpus rebuild): the same chunk_id names
    DIFFERENT content across chunksets. Resolved by content_sha256 exact
    match (when present) or text_quote normalized containment.
  * **Re-split**: a chunk's text was re-segmented. The text_quote (a verbatim
    >= 40-char substring) still resolves into whichever new chunk contains it.

**Principle: eval stays fail-closed; re-anchoring is an explicit, audited
operator command — never a silent load-time fixup.** A frozen set is refused
unless ``unfreeze_for_repin=True``.

§3 resolution ladder (per passage, first match wins):

  1. ``content_sha256`` exact match against the new chunkset
     (``method="content_sha256"``) — survives a pure renumber incl. the
     union case.
  2. ``text_quote`` normalized containment, UNIQUE across the new chunkset
     (``method="text_quote"``) — survives a re-split.
  3. ``text_quote`` containment scoped to chunks whose ``item_path`` (and,
     when present, ``section_heading``) match the anchor — disambiguates an
     otherwise-ambiguous quote (``method="text_quote_scoped"``).
  4. unresolved → the passage is left as-is, the question is parked
     ``status: draft``, and the orphan is reported
     (``method="unresolved"``).

On a successful per-passage resolution the chunk_id is rewritten and
``anchor.content_sha256`` is backfilled (so the next re-pin is O(1) by hash).
The chunkset pin (``chunkset.{kind,chunks_path,chunks_sha256}``) is rewritten
to the new chunkset. A repin report JSON records per-question resolution
methods + the orphan list and is written under ``retrieval_eval/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.retrieval._text import normalize_ws
from lib.retrieval.gold_set import (
    GOLD_SET_FILENAME,
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    chunk_content_sha256,
)
from lib.utils import sha256_file


class GoldRepinError(Exception):
    """Raised on a fail-closed repin refusal (frozen set, missing chunkset)."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


# Chunkset kind -> default course-dir-relative chunks path.
# DART->semantik purge Stage 1 (dual-READ): accept the ratified kind + dir.
_KIND_DEFAULT_PATH = {
    "semantik": "semantik_chunks/chunks.jsonl",
    "dart": "dart_chunks/chunks.jsonl",
    "imscc": "imscc_chunks/chunks.jsonl",
    "corpus": "corpus/chunks.jsonl",
}


@dataclass
class PassageResolution:
    """Per-passage resolution outcome."""

    question_id: str
    old_chunk_id: Optional[str]
    new_chunk_id: Optional[str]
    method: str  # content_sha256 | text_quote | text_quote_scoped | unresolved
    text_quote: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "old_chunk_id": self.old_chunk_id,
            "new_chunk_id": self.new_chunk_id,
            "method": self.method,
            "text_quote": self.text_quote[:80],
        }


@dataclass
class RepinReport:
    """Aggregate repin outcome for one gold set."""

    course_slug: str
    kind: str
    chunks_path: str
    old_sha256: Optional[str]
    new_sha256: str
    generated_at: str = ""
    resolutions: List[PassageResolution] = field(default_factory=list)
    parked_question_ids: List[str] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {
            "content_sha256": 0,
            "text_quote": 0,
            "text_quote_scoped": 0,
            "unresolved": 0,
        }
        for r in self.resolutions:
            c[r.method] = c.get(r.method, 0) + 1
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_slug": self.course_slug,
            "generated_at": self.generated_at,
            "chunkset": {
                "kind": self.kind,
                "chunks_path": self.chunks_path,
                "old_chunks_sha256": self.old_sha256,
                "new_chunks_sha256": self.new_sha256,
            },
            "counts": self.counts,
            "parked_question_ids": self.parked_question_ids,
            "resolutions": [r.to_dict() for r in self.resolutions],
            "orphans": [r.to_dict() for r in self.resolutions if r.method == "unresolved"],
        }


# ---------------------------------------------------------------- ladder


def _resolve_passage(
    anchor: Dict[str, Any],
    old_chunk_id: Optional[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
    content_sha_index: Dict[str, str],
    *,
    question_id: str,
) -> PassageResolution:
    """Run the §3 ladder for one passage; return its resolution."""
    text_quote = anchor.get("text_quote") or ""
    declared_csha = anchor.get("content_sha256")

    # (1) content_sha256 exact match.
    if isinstance(declared_csha, str) and declared_csha in content_sha_index:
        return PassageResolution(
            question_id=question_id,
            old_chunk_id=old_chunk_id,
            new_chunk_id=content_sha_index[declared_csha],
            method="content_sha256",
            text_quote=text_quote,
        )

    needle = normalize_ws(text_quote).lower()
    if needle:
        # (2) unique text_quote containment.
        matches = [
            cid
            for cid, c in chunks_by_id.items()
            if needle in normalize_ws(c.get("text", "")).lower()
        ]
        if len(matches) == 1:
            return PassageResolution(
                question_id=question_id,
                old_chunk_id=old_chunk_id,
                new_chunk_id=matches[0],
                method="text_quote",
                text_quote=text_quote,
            )
        if len(matches) > 1:
            # (3) scope by item_path (+ section_heading) to disambiguate.
            item_path = (anchor.get("item_path") or "").strip()
            heading = (anchor.get("section_heading") or "").strip().lower()
            scoped = []
            for cid in matches:
                src = chunks_by_id[cid].get("source") or {}
                if item_path and (src.get("item_path") or "") != item_path:
                    continue
                if heading and normalize_ws(src.get("section_heading", "")).lower() != heading:
                    continue
                scoped.append(cid)
            if len(scoped) == 1:
                return PassageResolution(
                    question_id=question_id,
                    old_chunk_id=old_chunk_id,
                    new_chunk_id=scoped[0],
                    method="text_quote_scoped",
                    text_quote=text_quote,
                )

    # (4) unresolved.
    return PassageResolution(
        question_id=question_id,
        old_chunk_id=old_chunk_id,
        new_chunk_id=None,
        method="unresolved",
        text_quote=text_quote,
    )


def repin_gold_doc(
    gold: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    kind: str,
    chunks_path: str,
    new_sha256: str,
    drop_orphans: bool = False,
) -> Tuple[Dict[str, Any], RepinReport]:
    """Re-anchor every passage in a loaded gold doc against ``chunks_by_id``.

    Pure function (no I/O) — mutates and returns a deep copy of ``gold`` plus a
    :class:`RepinReport`. Resolved passages get rewritten chunk_ids +
    backfilled content_sha256; questions with any unresolved passage are parked
    ``status: draft`` (their stale anchor is left intact so an operator can
    re-author). The chunkset pin is rewritten to the new chunkset.

    When ``drop_orphans=True`` the parked (unresolvable) questions are REMOVED
    from the doc instead of left in place — so the re-pinned set loads clean
    against the new chunkset (eval-ready). Nothing is silent: every dropped
    question is recorded in the repin report's orphan list + parked ids.
    """
    gold = json.loads(json.dumps(gold))  # deep copy — never mutate the input
    old_sha = (gold.get("chunkset") or {}).get("chunks_sha256")

    # content-sha index of the NEW chunkset (last write wins on a dup hash).
    content_sha_index: Dict[str, str] = {}
    for cid, c in chunks_by_id.items():
        content_sha_index[chunk_content_sha256(c)] = cid

    report = RepinReport(
        course_slug=gold.get("course_slug", ""),
        kind=kind,
        chunks_path=chunks_path,
        old_sha256=old_sha,
        new_sha256=new_sha256,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    parked: List[str] = []
    backfilled_v1_1_field = False
    for q in gold.get("questions") or []:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id", "")
        question_unresolved = False
        for p in q.get("relevant_passages") or []:
            if not isinstance(p, dict):
                continue
            anchor = p.get("anchor") or {}
            res = _resolve_passage(
                anchor,
                p.get("chunk_id"),
                chunks_by_id,
                content_sha_index,
                question_id=qid,
            )
            report.resolutions.append(res)
            if res.new_chunk_id is not None:
                p["chunk_id"] = res.new_chunk_id
                # backfill content_sha256 for an O(1) next re-pin.
                anchor["content_sha256"] = chunk_content_sha256(
                    chunks_by_id[res.new_chunk_id]
                )
                p["anchor"] = anchor
                backfilled_v1_1_field = True
            else:
                question_unresolved = True
        if question_unresolved:
            # Park the question as a draft (loud, audited — not deleted). The
            # stale chunk_id + anchor are left intact so an operator can
            # re-author against the original quote; the question is reported as
            # an orphan in the repin report.
            auth = q.get("authoring")
            if isinstance(auth, dict):
                auth["status"] = "draft"
            parked.append(qid)

    report.parked_question_ids = parked

    if drop_orphans and parked:
        parked_set = set(parked)
        gold["questions"] = [
            q
            for q in gold.get("questions") or []
            if not (isinstance(q, dict) and q.get("question_id") in parked_set)
        ]

    # Rewrite the chunkset pin.
    gold["chunkset"] = {
        "kind": kind,
        "chunks_path": chunks_path,
        "chunks_sha256": new_sha256,
    }
    # Backfilling anchor.content_sha256 promotes the doc to the v1.1 shape
    # (content_sha256 is a v1.1-only anchor field); bump the version so the
    # loader validates it against the v1.1 schema rather than rejecting the
    # new field under the const-pinned v1.0 schema.
    if backfilled_v1_1_field and gold.get("schema_version") == "1.0":
        gold["schema_version"] = "1.1"
    return gold, report


# ---------------------------------------------------------------- I/O entry


def repin_gold_set(
    course_dir: Path,
    *,
    kind: str,
    chunks_path: Optional[str] = None,
    unfreeze_for_repin: bool = False,
    dry_run: bool = False,
    drop_orphans: bool = False,
    filename: str = GOLD_SET_FILENAME,
) -> Tuple[RepinReport, Optional[Path]]:
    """Re-pin ``<course_dir>/retrieval_eval/<filename>`` to a new chunkset.

    Args:
        course_dir: LibV2 course directory.
        kind: target chunkset kind (``dart`` / ``imscc`` / ``corpus``).
        chunks_path: course-dir-relative path to the target chunks.jsonl;
            defaults to the kind's conventional path.
        unfreeze_for_repin: required to re-pin a ``frozen: true`` set.
        dry_run: build the report + rewritten doc but do not write to disk.
        drop_orphans: remove unresolvable (parked) questions so the re-pinned
            set loads clean / eval-ready; orphans stay recorded in the report.
        filename: gold-set filename (overridable for the repin-report sidecar).

    Returns ``(report, written_gold_path_or_None)``. Raises
    :class:`GoldRepinError` on a fail-closed refusal.
    """
    course_dir = Path(course_dir)
    gold_path = course_dir / RETRIEVAL_EVAL_SUBDIR / filename
    if not gold_path.is_file():
        raise GoldRepinError("GOLD_SET_NOT_FOUND", f"no gold set at {gold_path}")

    try:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldRepinError("GOLD_SET_INVALID_JSON", str(exc)) from exc

    if gold.get("frozen") and not unfreeze_for_repin:
        raise GoldRepinError(
            "GOLD_SET_FROZEN",
            "gold set is frozen; pass unfreeze_for_repin=True "
            "(`--unfreeze-for-repin`) to re-pin it.",
        )

    rel = chunks_path or _KIND_DEFAULT_PATH.get(kind)
    if not rel:
        raise GoldRepinError(
            "GOLD_SET_UNKNOWN_KIND",
            f"unknown chunkset kind {kind!r}; pass --chunks-path explicitly.",
        )
    new_chunks_path = course_dir / rel
    if not new_chunks_path.is_file():
        raise GoldRepinError(
            "GOLD_SET_CHUNKSET_NOT_FOUND",
            f"target chunkset not found at {new_chunks_path}.",
        )

    new_sha = sha256_file(new_chunks_path)
    chunks_by_id = _load_chunks_by_id(new_chunks_path)

    new_gold, report = repin_gold_doc(
        gold, chunks_by_id, kind=kind, chunks_path=rel, new_sha256=new_sha,
        drop_orphans=drop_orphans,
    )

    if dry_run:
        return report, None

    # Write the re-pinned gold set + the repin report sidecar.
    gold_path.write_text(json.dumps(new_gold, indent=2) + "\n", encoding="utf-8")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = course_dir / RETRIEVAL_EVAL_SUBDIR / f"gold_repin_report_{ts}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report, gold_path


__all__ = [
    "GoldRepinError",
    "PassageResolution",
    "RepinReport",
    "repin_gold_doc",
    "repin_gold_set",
]
