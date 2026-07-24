"""Standalone deterministic AUDIT runner over judged GLM-OCR sidecars.

Reads the JUDGED heading sidecars of a directory (or an explicit list of
sidecar files), builds the deterministic :mod:`.heading_judge_audit` report,
writes ``heading_judge_audit.json`` into the output dir, and prints a LOUD
summary naming every incomplete + collapsed chapter.

**REPORT-ONLY — never mutates a sidecar or HTML.** The audit is a pure READ
pass (the ``heading_judge`` phase's fail-open contract is unaffected).

Chapter input resolution (per stem, preference order):

* ``{stem}.corrected_layout.json`` — the JUDGED layout the standalone judge
  ``--apply`` writes (``region_provenance`` + ``heading_tree``); its heading
  regions carry the final ``level`` + residual ``heading_level_pending`` +
  ``heading_level_judged`` markers the audit needs. PREFERRED.
* ``{stem}.glmocr_layout.json`` — the RAW GLM pages (pre-transform). The raw
  regions carry no ``heading_level_pending`` flag (the transform mints it), so
  this is loaded ONLY as a fallback: it is re-transformed via
  :func:`transform.transform_document` (deterministic, no seat) to recover the
  ``region_provenance`` — best-effort, skipped with a warning if the transform
  is unavailable.

CLI
---
``python3 -m semantik_structure.glmocr.heading_judge_audit_standalone \
    <dir-or-sidecars>... [--out DIR]``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .heading_judge_audit import build_audit_report

logger = logging.getLogger(__name__)

_CORRECTED_SUFFIX = ".corrected_layout.json"
_RAW_SUFFIX = ".glmocr_layout.json"


def _stem_of(path: Path, suffix: str) -> str:
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _read_corrected_region_provenance(path: Path) -> Optional[List[Any]]:
    """``region_provenance`` list from a judged ``corrected_layout.json``."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("audit: unreadable corrected layout %s: %s", path, exc)
        return None
    if isinstance(doc, dict):
        rp = doc.get("region_provenance")
        if isinstance(rp, list):
            return rp
    logger.warning("audit: %s has no region_provenance list", path)
    return None


def _read_raw_region_provenance(path: Path) -> Optional[List[Any]]:
    """Re-transform a RAW ``glmocr_layout.json`` to recover region_provenance.

    Best-effort: the raw layout has no ``heading_level_pending`` flags, so it is
    only a fallback for stems with no judged corrected layout. Any transform /
    import failure returns ``None`` (that stem is skipped with a warning).
    """
    try:
        from .heading_judge_standalone import _load_layout_pages
        from .transform import transform_document

        pages = _load_layout_pages(path)
        tr = transform_document(pages)
        rp = list(tr.region_provenance)
        return rp
    except Exception as exc:  # noqa: BLE001 — fallback, never fatal
        logger.warning(
            "audit: could not re-transform raw layout %s (skipping): %s",
            path, exc)
        return None


def _discover(paths: List[Path]) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Map stem → corrected-layout path and stem → raw-layout path.

    A directory contributes every ``*.corrected_layout.json`` +
    ``*.glmocr_layout.json`` it holds; a file token contributes itself. First
    occurrence per stem wins (order-preserving).
    """
    corrected: Dict[str, Path] = {}
    raw: Dict[str, Path] = {}

    def _add(f: Path) -> None:
        name = f.name
        if name.endswith(_CORRECTED_SUFFIX):
            corrected.setdefault(_stem_of(f, _CORRECTED_SUFFIX), f)
        elif name.endswith(_RAW_SUFFIX):
            raw.setdefault(_stem_of(f, _RAW_SUFFIX), f)

    for p in paths:
        if p.is_dir():
            for f in sorted(p.glob("*" + _CORRECTED_SUFFIX)):
                _add(f)
            for f in sorted(p.glob("*" + _RAW_SUFFIX)):
                _add(f)
        elif p.is_file():
            _add(p)
    return corrected, raw


def load_judged_chapters(paths: List[Path]) -> List[Dict[str, Any]]:
    """Resolve ``[{stem, region_provenance}]`` for every judged sidecar found.

    Corrected-layout wins per stem; a raw-only stem falls back to a
    deterministic re-transform. A stem whose provenance cannot be resolved is
    skipped (never a hard failure).
    """
    corrected, raw = _discover(paths)
    stems = sorted(set(corrected) | set(raw))
    chapters: List[Dict[str, Any]] = []
    for stem in stems:
        rp: Optional[List[Any]] = None
        if stem in corrected:
            rp = _read_corrected_region_provenance(corrected[stem])
        if rp is None and stem in raw:
            rp = _read_raw_region_provenance(raw[stem])
        if rp is None:
            continue
        chapters.append({"stem": stem, "region_provenance": rp})
    return chapters


def _print_summary(report: Dict[str, Any]) -> None:
    """LOUD stdout summary naming every incomplete + collapsed chapter."""
    book = report.get("book") or {}
    incomplete = book.get("incomplete_chapters") or []
    collapsed = book.get("collapsed_chapters") or []
    inconsistent = book.get("inconsistent_signatures") or []
    flagged = report.get("flagged_chapters") or []
    n_chapters = report.get("n_chapters", len(report.get("chapters") or []))

    print("=" * 72)
    print(f"HEADING-JUDGE AUDIT — {n_chapters} chapter(s) audited")
    print(f"  book level distribution: {book.get('level_distribution')}")
    if incomplete:
        print(f"  INCOMPLETE (residual heading_level_pending) — {len(incomplete)}:")
        for row in incomplete:
            print(f"    * {row.get('stem')}: "
                  f"{row.get('n_residual_pending')} residual pending")
    else:
        print("  INCOMPLETE: none")
    if collapsed:
        print(f"  COLLAPSED (degenerate level distribution) — {len(collapsed)}:")
        for row in collapsed:
            print(f"    * {row.get('stem')}: {', '.join(row.get('reasons') or [])}")
    else:
        print("  COLLAPSED: none")
    print(f"  cross-chapter inconsistent signatures (ADVISORY, report-only): "
          f"{len(inconsistent)}")
    print(f"  FLAGGED for re-judge (Arm A + Arm B union): "
          f"{len(flagged)} -> {flagged}")
    print("=" * 72)


def run_audit(paths: List[Path], *, out_dir: Path) -> Dict[str, Any]:
    """Build the audit report over ``paths`` and write it into ``out_dir``.

    Returns the report dict (with an added ``report_path``). Never mutates any
    input sidecar or HTML.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chapters = load_judged_chapters([Path(p) for p in paths])
    report = build_audit_report(chapters)
    report_path = out_dir / "heading_judge_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic REPORT-ONLY audit over judged GLM-OCR "
                    "heading sidecars (never mutates inputs).")
    parser.add_argument(
        "paths", nargs="+",
        help="directory of judged sidecars OR explicit "
             "{stem}.corrected_layout.json / {stem}.glmocr_layout.json files")
    parser.add_argument(
        "--out", default="heading_judge_audit_out",
        help="output directory for heading_judge_audit.json "
             "(default ./heading_judge_audit_out)")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.paths]
    for p in paths:
        if not p.exists():
            print(f"warning: audit path not found: {p}", file=sys.stderr)

    report = run_audit(paths, out_dir=Path(args.out))
    _print_summary(report)
    print(f"audit report written: {report.get('report_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
