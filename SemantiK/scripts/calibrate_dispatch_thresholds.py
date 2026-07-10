#!/usr/bin/env python3
"""Offline sweep of the SEMANTIK_BERT_AUTHORITATIVE tau/delta dispatch gate
against a prior run's persisted role distributions (ITEM6).

Dev-script — stdlib only (json + argparse), CPU-only, NO model / NO GPU / NO
network. It reads the persisted ``region_provenance[].role_top_k`` distributions
(the ITEM6 stamp) + a paired ``structure_review`` verdict list and sweeps the
tau/delta pair that gates ``reviewer._authoritative_dispatch_gate``, reporting
for each grid point: what fraction of regions would dispatch to the reviewer,
and how well that dispatch set covers (recall) / concentrates on (precision) the
regions the reviewer actually corrected.

A region "dispatches" iff  ``top1 < tau``  OR  ``(top1 - top2) < delta``  over
its ``role_top_k`` (the same margin test as the live gate). A region is a
POSITIVE iff its ``structure_review`` verdict is ``corrected`` (kind changed).

Inputs are CLI paths (never hardcoded corpus paths — wide-net directive):
  --provenance   a bridge / audit JSON carrying ``region_provenance`` (or a raw
                 provenance list). Repeatable.
  --review-audit a conformance_audit.json (or a bridge JSON) carrying
                 ``structure_review`` (or a raw verdict list). Repeatable,
                 paired position-for-position with --provenance.

Output: a TSV to stdout —
    tau  delta  dispatch_frac  corrected_recall  corrected_precision

Exit non-zero (with guidance) when NO input region carries ``role_top_k`` — that
means the artifacts predate ITEM6 and cannot be calibrated. Point the script at
the most recent conversion artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Sweep grid — tau in [0.30, 0.90] and delta in [0.05, 0.40], step 0.05.
_TAUS = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
_DELTAS = [round(0.05 + 0.05 * i, 2) for i in range(8)]  # 0.05 .. 0.40


def _load_provenance(path: str) -> list[dict[str, Any]]:
    """Return the region_provenance list from a bridge/audit JSON or a raw list."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if isinstance(doc, list):
        return [e for e in doc if isinstance(e, dict)]
    if isinstance(doc, dict):
        rp = doc.get("region_provenance")
        if isinstance(rp, list):
            return [e for e in rp if isinstance(e, dict)]
    return []


def _load_verdicts(path: str) -> list[dict[str, Any]]:
    """Return the structure_review verdict list from a conformance/bridge JSON."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if isinstance(doc, list):
        return [e for e in doc if isinstance(e, dict)]
    if isinstance(doc, dict):
        sr = doc.get("structure_review")
        if isinstance(sr, list):
            return [e for e in sr if isinstance(e, dict)]
    return []


def _top1_top2(role_top_k: Any) -> tuple[float, float] | None:
    """(top1, top2) from a role_top_k list; None if unusable. top2=0.0 when the
    distribution has a single entry (a maximally-wide margin)."""
    if not isinstance(role_top_k, (list, tuple)) or not role_top_k:
        return None
    try:
        top1 = float(role_top_k[0][1])
    except (IndexError, TypeError, ValueError):
        return None
    top2 = 0.0
    if len(role_top_k) >= 2:
        try:
            top2 = float(role_top_k[1][1])
        except (IndexError, TypeError, ValueError):
            top2 = 0.0
    return (top1, top2)


def _corrected_indices(verdicts: list[dict[str, Any]]) -> set[int]:
    """Region indices (block_id) the reviewer CORRECTED (kind changed)."""
    out: set[int] = set()
    for row in verdicts:
        bid = row.get("block_id")
        if not isinstance(bid, int):
            continue
        verdict = str(row.get("verdict") or "").strip().lower()
        kb = row.get("kind_before")
        ka = row.get("kind_after")
        if verdict == "corrected" or (kb is not None and ka is not None and kb != ka):
            out.add(bid)
    return out


def _collect(
    provenance_paths: list[str], review_paths: list[str]
) -> list[tuple[float, float, bool]]:
    """Join each (provenance, review) pair on region_index == block_id and
    return per-region (top1, top2, is_corrected) triples across all pairs."""
    rows: list[tuple[float, float, bool]] = []
    for prov_path, rev_path in zip(provenance_paths, review_paths):
        prov = _load_provenance(prov_path)
        corrected = _corrected_indices(_load_verdicts(rev_path))
        for entry in prov:
            ridx = entry.get("region_index")
            if not isinstance(ridx, int):
                continue
            tt = _top1_top2(entry.get("role_top_k"))
            if tt is None:
                continue
            rows.append((tt[0], tt[1], ridx in corrected))
    return rows


def _sweep(rows: list[tuple[float, float, bool]]) -> list[tuple[float, float, float, float, float]]:
    total = len(rows)
    n_pos = sum(1 for _, _, c in rows if c)
    out: list[tuple[float, float, float, float, float]] = []
    for tau in _TAUS:
        for delta in _DELTAS:
            dispatched = 0
            hit = 0
            for top1, top2, is_pos in rows:
                if top1 < tau or (top1 - top2) < delta:
                    dispatched += 1
                    if is_pos:
                        hit += 1
            dispatch_frac = dispatched / total if total else 0.0
            recall = hit / n_pos if n_pos else 0.0
            precision = hit / dispatched if dispatched else 0.0
            out.append((tau, delta, dispatch_frac, recall, precision))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", action="append", default=[], required=True,
                    help="bridge/audit JSON carrying region_provenance (repeatable)")
    ap.add_argument("--review-audit", action="append", default=[], required=True,
                    help="conformance_audit.json carrying structure_review (repeatable, paired)")
    args = ap.parse_args(argv)

    if len(args.provenance) != len(args.review_audit):
        print(
            "error: --provenance and --review-audit must be paired "
            f"({len(args.provenance)} vs {len(args.review_audit)})",
            file=sys.stderr,
        )
        return 2

    rows = _collect(args.provenance, args.review_audit)
    if not rows:
        print(
            "error: no input region carries 'role_top_k' — these artifacts "
            "predate ITEM6 (the role-distribution stamp). Re-convert with the "
            "current cascade and point --provenance at the fresh region_provenance.",
            file=sys.stderr,
        )
        return 1

    print("tau\tdelta\tdispatch_frac\tcorrected_recall\tcorrected_precision")
    for tau, delta, frac, recall, precision in _sweep(rows):
        print(f"{tau:.2f}\t{delta:.2f}\t{frac:.4f}\t{recall:.4f}\t{precision:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
