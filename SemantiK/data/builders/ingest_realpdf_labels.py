"""Ingest CORRECTED real-PDF label tasks into the eval row schema.

Reads the per-doc ``task.json`` files emitted by ``build_realpdf_label_tasks.py``
AFTER an annotator has corrected the candidate labels, and emits
``data/structure_dataset_realpdf_handlabeled/{shippable,internal}/test.jsonl`` in
the EXACT row schema ``eval/measure_realpdf_structure.py`` consumes — so the
existing eval grades the head against human-grade, real-PDF-ANCHORED gold with
ZERO eval changes.

The whole point: the ``text`` / ``layout`` / ``bbox`` come from the real-PDF
candidate block (real-PDF-anchored); only the LABELS are the corrected annotation.

Corrected-label file format
----------------------------
The annotator edits each doc's ``task.json`` IN PLACE. For every block they set a
``"label"`` object (the corrected head labels) and flip ``"corrected": true``::

    {
      "block_id": "p00b003",
      "text": "...",                 # may be edited (e.g. after a merge)
      "layout": [...20 floats...],     # leave untouched (real-PDF-anchored)
      "bbox": [...], "page": 0, "pdf_page_num": 12,
      "candidate_label": { ... },      # the seed (kept for provenance/diff)
      "corrected": true,
      "label": {                       # <-- the corrected annotation
        "structural_role": 1,          # index into label_space.structural_role
        "is_heading": 1,               # 0/1
        "table_region": 0,             # 0/1
        "is_image_block": 0,           # 0/1
        "list_nesting": 0,             # 0..3
        "pedagogical_role": 0          # index into label_space.pedagogical_role
      }
    }

  * ACCEPT a candidate: copy ``candidate_label`` into ``label`` (or set
    ``"accept_candidate": true`` and omit ``label`` — the ingest copies the
    candidate for you). Either way set ``corrected: true`` so the block counts as
    reviewed (use ``--allow-unreviewed`` to ingest blocks still at the seed).
  * DROP furniture (running headers/footers, page numbers, de-duped boilerplate):
    set ``"drop": true``. Dropped blocks emit NO row.
  * MERGE two blocks: ``drop: true`` the absorbed block(s) and edit the surviving
    block's ``text`` to the concatenation (its bbox/layout stay the survivor's).
  * SPLIT one block into N: append new block entries with fresh ``block_id``s, each
    with its own ``text`` + ``label`` (copy the parent ``bbox``/``layout``/``page``;
    the layout vector is shared geometry, which is acceptable for a split), and
    ``drop: true`` the original. Label-space accepts both index ints and the role
    NAME string (resolved against ``label_space``).

Usage::

    python -m data.builders.ingest_realpdf_labels \
        --tasks-dir data/realpdf_label_tasks \
        --labeled-date 2026-06-29 \
        --label-method "claude-opus multimodal page review"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.builders.build_structure_data import (  # noqa: E402
    LAYOUT_FEATURE_DIM,
    PEDAGOGICAL_ROLE_NAMES,
    ROLE_NAMES,
)
from data.alignment.structure_align import SCHEMA_VERSION as ROW_SCHEMA_VERSION  # noqa: E402

LIST_NESTING_BUCKETS = (0, 1, 2, 3)
_BINARY_KEYS = ("is_heading", "table_region", "is_image_block")


class _IngestError(Exception):
    pass


def _resolve_role_id(value, names: tuple[str, ...], field: str, ctx: str) -> int:
    """Accept either an int index OR a role-NAME string; validate range."""
    if isinstance(value, str):
        if value not in names:
            raise _IngestError(f"{ctx}: {field}={value!r} not a valid name {names}")
        return names.index(value)
    try:
        i = int(value)
    except (TypeError, ValueError):
        raise _IngestError(f"{ctx}: {field}={value!r} is neither an int index nor a name")
    if not (0 <= i < len(names)):
        raise _IngestError(f"{ctx}: {field}={i} out of range [0,{len(names)})")
    return i


def _resolve_label(block: dict, ctx: str) -> dict:
    """Build the validated eval-row ``labels`` dict from a corrected block."""
    if block.get("accept_candidate") and "label" not in block:
        raw = dict(block.get("candidate_label") or {})
    else:
        raw = block.get("label")
        if raw is None:
            raise _IngestError(f"{ctx}: no 'label' (and no accept_candidate); not corrected?")
    labels: dict[str, int] = {}
    labels["structural_role"] = _resolve_role_id(
        raw.get("structural_role"), ROLE_NAMES, "structural_role", ctx
    )
    labels["pedagogical_role"] = _resolve_role_id(
        raw.get("pedagogical_role", "none"), PEDAGOGICAL_ROLE_NAMES, "pedagogical_role", ctx
    )
    for k in _BINARY_KEYS:
        v = int(raw.get(k, 0) or 0)
        if v not in (0, 1):
            raise _IngestError(f"{ctx}: {k}={v} must be 0/1")
        labels[k] = v
    ln = int(raw.get("list_nesting", 0) or 0)
    if ln not in LIST_NESTING_BUCKETS:
        raise _IngestError(f"{ctx}: list_nesting={ln} not in {LIST_NESTING_BUCKETS}")
    labels["list_nesting"] = ln
    return labels


def _row_from_block(block: dict, task: dict, args, ctx: str) -> dict:
    """Emit one eval row in the EXACT schema measure_realpdf_structure consumes."""
    layout = block.get("layout")
    if not isinstance(layout, list) or len(layout) != LAYOUT_FEATURE_DIM:
        raise _IngestError(f"{ctx}: layout missing or not {LAYOUT_FEATURE_DIM}-dim")
    labels = _resolve_label(block, ctx)
    return {
        "text": (block.get("text") or "").strip(),
        "layout": [float(x) for x in layout],
        "labels": labels,
        "html_tag": "",  # real-PDF-anchored: no HTML tag (eval scores labels, not tags)
        "source": task["source"],
        "schema_version": ROW_SCHEMA_VERSION,
        "provenance": {
            "page": block.get("page"),
            "pdf_page_num": block.get("pdf_page_num"),
            "bbox": list(block.get("bbox") or []),
            "block_id": block.get("block_id"),
            "doc_id": task["doc_id"],
        },
        "align": {"kind": "handlabeled", "confidence": 1.0},
        # Fields measure_realpdf_structure reads for bucketing / coverage.
        "realpdf": True,
        "domain": task["domain"],
        "license": task["license_partition"],
        "doc_id": task["doc_id"],
        "holdout_clean": bool(task.get("holdout_clean", True)),
        "aligned_gold_fraction": 1.0,  # hand-labeled gold == perfectly anchored
        # Provenance of the annotation itself (date passed in; never now()).
        "labeled_by": args.labeled_by,
        "label_method": args.label_method,
        "labeled_date": args.labeled_date,
    }


def ingest(args) -> None:
    tasks_dir: Path = args.tasks_dir
    task_files = sorted(tasks_dir.rglob("task.json"))
    if not task_files:
        raise SystemExit(f"no task.json files under {tasks_dir}")

    rows_by_part: dict[str, list[dict]] = defaultdict(list)
    docs_by_part: dict[str, set] = defaultdict(set)
    per_source: dict[str, Counter] = defaultdict(Counter)
    per_role: dict[str, Counter] = defaultdict(Counter)
    skipped = Counter()
    errors: list[str] = []

    for tf in task_files:
        try:
            task = json.loads(tf.read_text())
        except Exception as exc:
            errors.append(f"{tf}: read error {exc}")
            continue
        part = task.get("license_partition", "internal")
        blocks = task.get("blocks", [])
        for block in blocks:
            ctx = f"{task.get('doc_id')}/{block.get('block_id')}"
            if block.get("drop"):
                skipped["dropped"] += 1
                continue
            if not block.get("corrected") and not args.allow_unreviewed:
                skipped["unreviewed"] += 1
                continue
            try:
                row = _row_from_block(block, task, args, ctx)
            except _IngestError as exc:
                errors.append(str(exc))
                if args.strict:
                    raise SystemExit(f"strict: {exc}")
                skipped["invalid"] += 1
                continue
            rows_by_part[part].append(row)
            docs_by_part[part].add(task["doc_id"])
            per_source[part][task["source"]] += 1
            per_role[part][ROLE_NAMES[row["labels"]["structural_role"]]] += 1

    if errors:
        print(f"[ingest] {len(errors)} label issue(s):", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for part in ("shippable", "internal"):
        rows = rows_by_part.get(part, [])
        if not rows:
            continue
        part_dir = args.out_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        with (part_dir / "test.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        coverage = {
            "schema_version": ROW_SCHEMA_VERSION,
            "build": "realpdf_handlabeled",
            "license_partition": part,
            "labeled_by": args.labeled_by,
            "label_method": args.label_method,
            "labeled_date": args.labeled_date,
            "n_docs": len(docs_by_part[part]),
            "n_rows": len(rows),
            "per_source": dict(per_source[part]),
            "per_role": dict(per_role[part]),
            "role_names": list(ROLE_NAMES),
            "layout_feature_dim": LAYOUT_FEATURE_DIM,
            "skipped": dict(skipped),
            "label_errors": errors,
        }
        (part_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
        total_rows += len(rows)
        print(
            f"[ingest][{part}] -> {part_dir}  docs={len(docs_by_part[part])} "
            f"rows={len(rows)}  per_role={dict(per_role[part])}",
            file=sys.stderr,
        )

    print(
        f"[INGEST] {total_rows} rows across "
        f"{sum(len(v) for v in docs_by_part.values())} docs -> {args.out_dir}  "
        f"skipped={dict(skipped)}",
        file=sys.stderr,
    )
    if total_rows == 0:
        print(
            "[ingest] WARNING: emitted 0 rows. Did the annotator set corrected=true "
            "+ a 'label' on the blocks? (use --allow-unreviewed to ingest seeds).",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=Path("data/realpdf_label_tasks"),
                    help="Dir of corrected per-doc task.json files (recursively globbed).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/structure_dataset_realpdf_handlabeled"))
    ap.add_argument("--labeled-date", required=True,
                    help="Annotation date (YYYY-MM-DD) — passed in, never now().")
    ap.add_argument("--label-method", default="claude-opus multimodal page review")
    ap.add_argument("--labeled-by", default="claude-opus")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="Ingest blocks still at the candidate seed (corrected=false).")
    ap.add_argument("--strict", action="store_true",
                    help="Abort on the first invalid label instead of skipping it.")
    args = ap.parse_args()
    ingest(args)


if __name__ == "__main__":
    main()
