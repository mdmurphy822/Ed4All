#!/usr/bin/env python3
"""Offline CPU-only calibration harness for the Stage-9b reasoning-QC pass
(``SEMANTIK_REASONING_QC``).

Mirrors ``SemantiK/scripts/calibration/regroup_calibration_ab.py``: a measured-experiment
arbiter that feeds the owner-gated shadow→on default-flip bar. It replays the
SHADOW-mode ``reasoning_qc`` audit (``result['reasoning_qc']``, or the bridge
JSON's forwarded arm) over one or more operator corpora and reports applied-op
PRECISION per reconcile channel, so the flip lands only on evidence
(>=95% precision over >=2 corpora).

The harness itself runs NO LLM and NO GPU — it consumes qc_audit JSON that a
prior shadow-mode conversion already produced. Precision is scored against an
OPTIONAL per-doc gold file (``--gold``) that marks which proposed flags are
correct (a human/oracle label); absent a gold file the harness reports the
flag DISTRIBUTION only (a coverage/volume view, not a precision gate).

qc_audit shape consumed (see ``reasoning_qc.run_reasoning_qc``)::

    {
      "schema": "reasoning-qc/1.0", "mode": "shadow", "ran": true,
      "flagged": [ {"region_index", "failure_mode", "fixable",
                    "proposed_regroup_run", "applied"}, ... ],
      "windows": [ {"page", "regions", "order_divergence", "n_flagged"}, ... ],
      "toc": {"declared_ordinals", "heading_ordinals", "declared_missing"},
      ...
    }

Gold shape (optional, per doc), keyed by the same region_index/failure_mode::

    {"correct": [[region_index, failure_mode], ...],
     "incorrect": [[region_index, failure_mode], ...]}

Report JSON (per doc + a corpus rollup)::

    {
      "schema": "reasoning-qc-calibration-ab/1.0",
      "rollup": {"n_docs", "n_flags", "by_mode", "precision", "gate_pass"},
      "per_doc": [ ... ]
    }
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Owner-gated flip bar (root task): >=95% applied-op precision over >=2 corpora.
PRECISION_FLOOR = 0.95
MIN_CORPORA = 2

# The reconcile channels a flag routes to (audit-only labels).
_RETYPE_MODES = frozenset({"example_as_heading", "mistyped_component", "wrong_semantic_class"})
_MERGE_MOVE_MODES = frozenset({"example_misordered_from_body", "section_no_body"})


def _load_qc_audit(path: Path) -> dict[str, Any]:
    """Load a qc_audit dict from either a bare audit JSON or a bridge/result
    JSON that carries it under a ``reasoning_qc`` key."""
    doc = json.loads(path.read_text())
    if isinstance(doc, dict) and doc.get("schema") == "reasoning-qc/1.0":
        return doc
    if isinstance(doc, dict) and isinstance(doc.get("reasoning_qc"), dict):
        return doc["reasoning_qc"]
    return doc if isinstance(doc, dict) else {}


def _gold_sets(gold: dict[str, Any] | None) -> tuple[set, set]:
    if not gold:
        return set(), set()
    correct = {(int(r), str(m)) for r, m in gold.get("correct", [])}
    incorrect = {(int(r), str(m)) for r, m in gold.get("incorrect", [])}
    return correct, incorrect


def _analyse_doc(name: str, audit: dict[str, Any], gold: dict[str, Any] | None) -> dict[str, Any]:
    flagged = audit.get("flagged") or []
    by_mode = Counter(str(f.get("failure_mode")) for f in flagged)
    n_retype = sum(v for m, v in by_mode.items() if m in _RETYPE_MODES)
    n_merge_move = sum(v for m, v in by_mode.items() if m in _MERGE_MOVE_MODES)

    correct, incorrect = _gold_sets(gold)
    tp = fp = scored = 0
    if correct or incorrect:
        for f in flagged:
            key = (int(f.get("region_index", -1)), str(f.get("failure_mode")))
            if key in correct:
                tp += 1
                scored += 1
            elif key in incorrect:
                fp += 1
                scored += 1
    precision = (tp / scored) if scored else None

    return {
        "doc": name,
        "n_flags": len(flagged),
        "by_mode": dict(by_mode),
        "n_retype": n_retype,
        "n_merge_move": n_merge_move,
        "toc_declared_missing": len(audit.get("toc", {}).get("declared_missing", []) or []),
        "gold_scored": scored,
        "true_positive": tp,
        "false_positive": fp,
        "precision": precision,
    }


def _rollup(per_doc: list[dict[str, Any]]) -> dict[str, Any]:
    n_flags = sum(d["n_flags"] for d in per_doc)
    by_mode: Counter = Counter()
    for d in per_doc:
        by_mode.update(d["by_mode"])
    tp = sum(d["true_positive"] for d in per_doc)
    scored = sum(d["gold_scored"] for d in per_doc)
    precision = (tp / scored) if scored else None
    n_corpora = len(per_doc)
    gate_pass = (
        precision is not None
        and precision >= PRECISION_FLOOR
        and n_corpora >= MIN_CORPORA
        and scored > 0
    )
    return {
        "n_docs": n_corpora,
        "n_flags": n_flags,
        "by_mode": dict(by_mode),
        "gold_scored": scored,
        "precision": precision,
        "precision_floor": PRECISION_FLOOR,
        "min_corpora": MIN_CORPORA,
        "gate_pass": bool(gate_pass),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="append", default=[], help="qc_audit / bridge JSON (repeatable)")
    ap.add_argument("--audit-dir", default=None, help="dir of *.json qc_audit files")
    ap.add_argument("--gold", action="append", default=[], help="per-doc gold JSON (positional-parallel to --audit)")
    ap.add_argument("--out", required=True, help="report JSON path")
    args = ap.parse_args(argv)

    audits = list(args.audit)
    if args.audit_dir:
        audits += sorted(glob.glob(str(Path(args.audit_dir) / "*.json")))
    if not audits:
        ap.error("need >=1 --audit or --audit-dir")

    golds = list(args.gold)
    per_doc: list[dict[str, Any]] = []
    for i, ap_path in enumerate(audits):
        audit = _load_qc_audit(Path(ap_path))
        gold = None
        if i < len(golds):
            gold = json.loads(Path(golds[i]).read_text())
        per_doc.append(_analyse_doc(Path(ap_path).name, audit, gold))

    report = {
        "schema": "reasoning-qc-calibration-ab/1.0",
        "rollup": _rollup(per_doc),
        "per_doc": per_doc,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["rollup"], indent=2, sort_keys=True))
    return 0 if report["rollup"]["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
