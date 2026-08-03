#!/usr/bin/env python3
"""Tier-3 composite-unit subclassing pilot + corpus-apply driver (Build #23).

Drives the model-assisted subclassifier (:mod:`lib.semantik.subclassifier`) over
one or more SemantiK chapters WITHOUT re-running the cascade — it re-renders each
chapter from its persisted ``{stem}_accessible.cascade_ir.json`` sidecar (same
lossless path as ``scripts/ops/semantik_rerender.py``) and runs the subclass pass
WITHIN the render seam, so the output is the fully-rendered chapter PLUS the
payload-only ``data-semantik-subclass`` / ``semantik-sub-<label>`` attributes.

Two modes
---------
* ``--dry-run`` (default; NO GPU / network): a deterministic MOCK client returns
  a seed label per unit type. Produces the audit report only (per-unit
  assignments, per-type label distribution, fold events, new-label log, skip /
  reject counts, bucket-collapse flags) — no HTML is written unless ``--apply``.
* ``--apply``: write the subclass-annotated HTML for every chapter to
  ``--output-dir``. Combined with ``--dry-run`` the mock client is used (still no
  GPU); WITHOUT ``--dry-run`` the live license-clean local reviewer seat
  (``build_default_subclass_client`` → qwen2.5-7b via ``LOCAL_SYNTHESIS_*``) is
  used — the orchestrator runs this live AFTER the 7B seat frees up.

Chapter selection (mirrors ``semantik_rerender`` conventions)
-------------------------------------------------------------
* ``--ir <path>`` (repeatable): explicit cascade-IR sidecars.
* ``--input-dir <dir>`` + ``--chapters ch06,ch09``: discover
  ``*_accessible.cascade_ir.json`` under the dir and keep those whose stem
  carries one of the ``chNN`` tokens (all discovered when ``--chapters`` unset).
* ``--title-map <json>``: ``{stem_or_chNN: title}`` document-title overrides.

Example
-------
    # dry-run pilot on two chapters (report only, mocked, CPU)
    python Trainforge/scripts/harness/subclass_pilot.py --input-dir <INPUT_DIR> \
        --chapters ch06,ch09 --title-map titles.json \
        --dry-run --json-out /tmp/subclass_pilot.json

    # corpus-wide apply of all 10 chapters (LIVE local 7B seat)
    python Trainforge/scripts/harness/subclass_pilot.py --input-dir <INPUT_DIR> \
        --apply --output-dir /tmp/subclass_apply \
        --title-map titles.json --json-out /tmp/subclass_apply.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.semantik.adapter import normalize_cascade_to_ed4all  # noqa: E402
from lib.semantik.cascade_ir import build_chapters_ir  # noqa: E402
from lib.semantik.subclassifier import (  # noqa: E402
    detect_bucket_collapse,
    load_seed_vocab,
)

_CH_TOKEN_RE = re.compile(r"(ch\d{1,3})", re.IGNORECASE)


class _RerenderResult:
    """Minimal duck-typed cascade result the adapter reads (mirrors rerender)."""

    def __init__(self, *, chapters: List[Any], **signals: Any):
        self.chapters = chapters
        self.exit_action = signals.get("exit_action") or "ship_with_confidence"
        self.wcag_status = signals.get("wcag_status") or "passed"
        self.theta_score = signals.get("theta_score")
        self.flags = list(signals.get("flags") or [])
        self.lane_used = signals.get("lane_used")
        self.lang = signals.get("lang") or "en"


def _chapters_from_ir(ir_doc: Dict[str, Any]) -> _RerenderResult:
    return _RerenderResult(
        chapters=build_chapters_ir(ir_doc),
        exit_action=ir_doc.get("exit_action"),
        wcag_status=ir_doc.get("wcag_status"),
        theta_score=ir_doc.get("theta_score"),
        flags=ir_doc.get("flags"),
        lane_used=ir_doc.get("lane_used"),
        lang=ir_doc.get("lang"),
    )


def _resolve_title_override(
    pdf_stem: str, title_map: Optional[Dict[str, str]]
) -> Optional[str]:
    if not title_map:
        return None
    if pdf_stem in title_map:
        return title_map[pdf_stem]
    lower = {str(k).lower(): v for k, v in title_map.items()}
    if pdf_stem.lower() in lower:
        return lower[pdf_stem.lower()]
    m = _CH_TOKEN_RE.search(pdf_stem)
    if m and m.group(1).lower() in lower:
        return lower[m.group(1).lower()]
    return None


def make_mock_client(
    seeds_by_type: Dict[str, Any],
) -> Callable[..., str]:
    """Deterministic round-robin mock: cycle each unit type's seed vocabulary.

    Spreads labels across a type's seeds so the dry-run exercises the distribution
    audit realistically (not a single collapsed bucket). Falls back to a stable
    proposed label when a type has no seeds. Encodes the unit type into the prompt
    read-back so successive calls to the same type advance the cursor.
    """
    counters: Dict[str, int] = {}

    def _client(prompt: str, *, max_tokens: int = 64) -> str:
        m = re.search(r"Unit type:\s*(\S+)", prompt)
        unit_type = m.group(1) if m else "prose"
        seeds = list(seeds_by_type.get(unit_type, ()))
        idx = counters.get(unit_type, 0)
        counters[unit_type] = idx + 1
        if seeds:
            label = seeds[idx % len(seeds)]
        else:
            label = "mock-proposed"
        return json.dumps({"subclass": label, "confidence": 0.75})

    return _client


def _merge_distribution(
    corpus: Dict[str, Counter], per_chapter: Dict[str, Dict[str, int]]
) -> None:
    for unit_type, labels in per_chapter.items():
        bucket = corpus.setdefault(unit_type, Counter())
        for label, n in labels.items():
            bucket[label] += n


def _discover_irs(args: argparse.Namespace) -> List[Path]:
    irs: List[Path] = [Path(p) for p in (args.ir or [])]
    if args.input_dir:
        d = Path(args.input_dir)
        wanted = None
        if args.chapters:
            wanted = {c.strip().lower() for c in args.chapters.split(",") if c.strip()}
        for cand in sorted(d.glob("*_accessible.cascade_ir.json")):
            if wanted is not None:
                m = _CH_TOKEN_RE.search(cand.stem)
                if not (m and m.group(1).lower() in wanted):
                    continue
            irs.append(cand)
    # De-dup preserving order.
    seen: set = set()
    out: List[Path] = []
    for p in irs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ir", action="append", help="Cascade-IR sidecar path (repeatable)."
    )
    ap.add_argument("--input-dir", help="Dir of *_accessible.cascade_ir.json.")
    ap.add_argument(
        "--chapters", help="Comma list of chNN tokens to select (with --input-dir)."
    )
    ap.add_argument("--title-map", dest="title_map", help="JSON {stem_or_chNN: title}.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the deterministic mock client (NO GPU / network).",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write subclass-annotated HTML per chapter to --output-dir.",
    )
    ap.add_argument("--output-dir", help="Destination dir for --apply HTML.")
    ap.add_argument("--json-out", dest="json_out", help="Write the JSON report here.")
    ap.add_argument(
        "--review-sidecar",
        dest="review_sidecar",
        help="JSONL sidecar for genuinely-new (non-vocab) labels.",
    )
    args = ap.parse_args(argv)

    irs = _discover_irs(args)
    if not irs:
        print("no cascade-IR sidecars resolved (use --ir or --input-dir)", file=sys.stderr)
        return 2
    if args.apply and not args.output_dir:
        print("--apply requires --output-dir", file=sys.stderr)
        return 2

    title_map: Optional[Dict[str, str]] = None
    if args.title_map:
        raw = json.loads(Path(args.title_map).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            title_map = raw

    seeds_by_type = load_seed_vocab()
    if args.dry_run:
        client: Callable[..., str] = make_mock_client(seeds_by_type)
    else:
        from lib.semantik.subclassifier import build_default_subclass_client

        client = build_default_subclass_client()

    corpus_dist: Dict[str, Counter] = {}
    per_chapter_reports: List[Dict[str, Any]] = []
    total_new: List[Dict[str, Any]] = []
    total_folds: List[Dict[str, Any]] = []
    status_totals: Counter = Counter()

    for ir_path in irs:
        ir_doc = json.loads(ir_path.read_text(encoding="utf-8"))
        result = _chapters_from_ir(ir_doc)
        pdf_stem = ir_doc.get("pdf") or ir_path.stem.replace("_accessible.cascade_ir", "")
        title_override = _resolve_title_override(pdf_stem, title_map)

        adapter_out = normalize_cascade_to_ed4all(
            result,
            pdf_stem=pdf_stem,
            title_override=title_override,
            subclass_client=client,
            subclass_review_sidecar=args.review_sidecar,
        )
        report = adapter_out.get("subclass_report") or {}
        report["chapter_stem"] = pdf_stem
        per_chapter_reports.append(report)

        _merge_distribution(corpus_dist, report.get("label_distribution", {}))
        total_new.extend(report.get("new_labels", []))
        total_folds.extend(report.get("fold_events", []))
        for status, n in (report.get("status_counts") or {}).items():
            status_totals[status] += n

        if args.apply:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{pdf_stem}_accessible.html"
            out_path.write_text(adapter_out["html"], encoding="utf-8")
            print(f"wrote {out_path} ({adapter_out['html_length']} bytes)")

    corpus_distribution = {ut: dict(c) for ut, c in corpus_dist.items()}
    summary = {
        "mode": "apply" if args.apply else "dry-run-report",
        "dry_run": bool(args.dry_run),
        "chapters_processed": len(irs),
        "status_totals": dict(status_totals),
        "corpus_label_distribution": corpus_distribution,
        "corpus_bucket_collapse": detect_bucket_collapse(corpus_distribution),
        "new_labels": total_new,
        "fold_events": total_folds,
        "per_chapter": per_chapter_reports,
    }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote report {args.json_out}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["corpus_bucket_collapse"]:
        print(
            f"WARNING: bucket-collapse detected in "
            f"{len(summary['corpus_bucket_collapse'])} unit type(s)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
