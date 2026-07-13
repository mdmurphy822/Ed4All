"""Standalone Stage-9b reasoning-QC RUNNER (task #39).

Judge an ALREADY-EMITTED SemantiK accessible-HTML file with the exact Stage-9b
reasoning-QC machinery (:mod:`reasoning_qc`) — **without re-running conversion**.
Purpose: the omni-vs-Super QC-seat A/B — a reasoning TEXT model (e.g. Super-120B)
judges the same combined ``*_accessible.html`` a full pipeline run produced, so
two seats can be compared on the SAME document.

This runner is **REPORT-ONLY**: it PROPOSES nothing to disk, applies no reconcile
op, and never rewrites the input HTML — the A/B needs the seats' VERDICTS, not
mutations. It reuses the production judgment path verbatim (the document-level
window + junction-seam partition, the bounded fan-out, the reasoning-preserving
split ladder, the verdict stitch, and the FlaggedBlock conversion) via the shared
:func:`reasoning_qc._fan_out_page_verifies` entry — **no judgment logic is
duplicated here**, and :func:`reasoning_qc.run_reasoning_qc`'s cascade / apply
behavior (and its tests) are untouched.

Parse strategy
--------------
:func:`parse_accessible_html` reads the emitted HTML with BeautifulSoup (``lxml``
parser — the same stack ``lib/semantik/vendor_ingest.py`` + the SemantiK gates
use) and walks every ``<section data-semantik-block-id=...>`` block in document
order. Each block becomes the SAME text-only record shape
(``{type, role, level, page, text}``) the QC machinery consumes (mirrors
:func:`reasoning_qc._block_record`), plus its stable ``block_id`` kept alongside
for the report.

**Attribute contract (2026-07 rename):** the live adapter seam
(``lib/semantik/adapter.py``) emits the ``data-semantik-*`` block attributes
(``data-semantik-block-id`` / ``data-semantik-block-role`` / ``data-semantik-pages``);
the historical ``data-dart-*`` spelling is accepted as a READ fallback so a
legacy converter output still parses. The block id / role / pages are resolved
via :data:`_BLOCK_ID_ATTRS` / :data:`_BLOCK_ROLE_ATTRS` / :data:`_PAGES_ATTRS`
(current spelling first, legacy second).

Type inference (coarse, documented) — from the block's FIRST significant
descendant element (``lib/semantik/adapter``'s emitted markup):

  * ``h1``..``h6``  → ``heading`` (``level`` = N)
  * ``table``       → ``table``
  * ``figure`` / ``img`` → ``figure``
  * ``ul`` / ``ol`` / ``dl`` → ``list``
  * ``pre``         → ``code``
  * ``blockquote``  → ``blockquote``
  * anything else (``p`` / bare text) → ``paragraph``

``role`` is the emitted ``data-dart-block-role`` (falling back to the inferred
type); ``page`` is the first integer of ``data-dart-pages`` (``None`` when
absent) — matching the region-path's "first physical page" representative.

CLI
---
``python3 -m semantik_structure.reasoning_qc_standalone <html_or_dir> [--out-dir DIR]``
judges one ``*_accessible.html`` file (or every such file in a directory),
writing ``<out_dir>/<stem>.qc_report.json`` per file.

``python3 -m semantik_structure.reasoning_qc_standalone --compare A.qc_report.json
B.qc_report.json [--out-dir DIR]`` aligns two reports and writes ``qc_compare.json``
+ prints a human-readable summary.

Seat: the normal env-driven chain applies (``SEMANTIK_REASONING_QC_BASE_URL`` /
``_MODEL`` > specialist seat > legacy VLM seat), so pointing at Super vs omni is
purely an environment change. :func:`lib.vllm_container_lifecycle.ensure_serving`
is called for the resolved seat first (fail-soft, a no-op when the container
lifecycle flag is off) and its measured ``load_seconds`` is recorded in the report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import reasoning_qc
from .reasoning_qc import QCWindow

logger = logging.getLogger(__name__)

STANDALONE_QC_SCHEMA = "reasoning-qc-standalone/1.0"
QC_COMPARE_SCHEMA = "reasoning-qc-compare/1.0"

# The first significant descendant tag → coarse block type. Documented in the
# module docstring; a heading tag additionally yields its level.
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_SIGNIFICANT_TAGS = (
    *_HEADING_TAGS,
    "table",
    "figure",
    "img",
    "ul",
    "ol",
    "dl",
    "pre",
    "blockquote",
    "p",
)
_TAG_TO_TYPE = {
    "table": "table",
    "figure": "figure",
    "img": "figure",
    "ul": "list",
    "ol": "list",
    "dl": "list",
    "pre": "code",
    "blockquote": "blockquote",
    "p": "paragraph",
}

# Block-attribute spellings, current first then the legacy ``data-dart-*`` fallback
# (the 2026-07 adapter rename data-dart-* -> data-semantik-*). Each resolver walks
# these in order and returns the first present value.
_BLOCK_ID_ATTRS = ("data-semantik-block-id", "data-dart-block-id")
_BLOCK_ROLE_ATTRS = ("data-semantik-block-role", "data-dart-block-role")
_PAGES_ATTRS = ("data-semantik-pages", "data-dart-pages")


def _first_attr(section: Any, names: tuple[str, ...]) -> str | None:
    """First present attribute value among ``names`` (current spelling first)."""
    for name in names:
        val = section.get(name)
        if val is not None:
            return val
    return None


# ---------------------------------------------------------------------------
# Parse — emitted accessible HTML → ordered block records (+ block ids).
# ---------------------------------------------------------------------------
def _first_page(pages_attr: str | None) -> int | None:
    """First integer of a ``data-dart-pages`` value (``"1"`` / ``"3-5"``), or None."""
    if not pages_attr:
        return None
    head = str(pages_attr).strip().replace("–", "-").split("-", 1)[0].strip()
    try:
        return int(head)
    except (TypeError, ValueError):
        return None


def _infer_type_and_level(section: Any) -> tuple[str, int | None]:
    """Infer ``(type, level)`` from the section's first significant descendant.

    Coarse, documented mapping (see the module docstring). A leading heading tag
    → ``("heading", N)``; otherwise the first block-level element decides the
    type and ``level`` is ``None``. A section with no recognized child (bare
    text) falls through to ``paragraph``.
    """
    first = section.find(_SIGNIFICANT_TAGS)
    if first is None:
        return "paragraph", None
    name = first.name
    if name in _HEADING_TAGS:
        try:
            return "heading", int(name[1:])
        except (TypeError, ValueError):
            return "heading", None
    return _TAG_TO_TYPE.get(name, "paragraph"), None


def parse_accessible_html(path: str | Path) -> list[dict[str, Any]]:
    """Parse an emitted ``*_accessible.html`` into ordered block records.

    Returns a list (in document order) of dicts shaped
    ``{index, block_id, type, role, level, page, text}`` — the ``type`` / ``role``
    / ``level`` / ``page`` / ``text`` keys are the SAME text-only record the QC
    machinery consumes (mirrors :func:`reasoning_qc._block_record`); ``index`` is
    the absolute 0-based block index and ``block_id`` the stable
    ``data-semantik-block-id`` (legacy ``data-dart-block-id`` accepted). Every
    ``<section data-semantik-block-id=...>`` in the document contributes exactly
    one record, in emission (reading) order.
    """
    from bs4 import BeautifulSoup

    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")
    records: list[dict[str, Any]] = []
    # A section is a block iff it carries a block-id under EITHER spelling.
    def _is_block(tag: Any) -> bool:
        return tag.name == "section" and any(tag.get(a) is not None for a in _BLOCK_ID_ATTRS)

    for section in soup.find_all(_is_block):
        block_id = _first_attr(section, _BLOCK_ID_ATTRS)
        btype, level = _infer_type_and_level(section)
        role = _first_attr(section, _BLOCK_ROLE_ATTRS) or btype
        page = _first_page(_first_attr(section, _PAGES_ATTRS))
        text = " ".join(section.get_text(" ", strip=True).split())
        records.append(
            {
                "index": len(records),
                "block_id": block_id,
                "type": btype,
                "role": role,
                "level": level,
                "page": page,
                "text": text,
            }
        )
    return records


def _window_from_records(records: list[dict[str, Any]]) -> QCWindow:
    """Build the ONE document-level :class:`QCWindow` over the parsed records.

    ``region_indices`` == the absolute block indices (identity), so a verdict
    keyed by local block index maps straight back to the absolute index (and thus
    the ``block_id``); ``block_records`` are the ``{type, role, level, page,
    text}`` dicts the judgment renders. Mirrors :func:`reasoning_qc.build_qc_windows`
    but sources the records from parsed HTML instead of ``capped`` regions.
    """
    n = len(records)
    block_records = [
        {"type": r["type"], "role": r["role"], "level": r["level"], "page": r["page"], "text": r["text"]}
        for r in records
    ]
    pages = [r["page"] for r in records if r["page"] is not None]
    return QCWindow(
        page=pages[0] if pages else 0,
        region_indices=list(range(n)),
        block_records=block_records,
        emit_positions=list(range(n)),
    )


# ---------------------------------------------------------------------------
# Run — judge one emitted HTML, REPORT-ONLY (no reconcile, no rewrite).
# ---------------------------------------------------------------------------
def run_standalone_qc(
    html_path: str | Path,
    *,
    out_dir: str | Path,
    log: Callable[[str], None] = print,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Judge ``html_path`` with the Stage-9b QC machinery; write a report; return it.

    Parses the emitted accessible HTML → the document block sequence → runs the
    SAME window/seam/fan-out/split-ladder judgment path
    (:func:`reasoning_qc._fan_out_page_verifies`, the production entry) → writes
    ``<out_dir>/<stem>.qc_report.json``. REPORT-ONLY: no reconcile op is applied
    and the input HTML is never rewritten.

    Seat resolution is the normal env-driven chain
    (:func:`reasoning_qc._resolve_qc_seat`); :func:`lib.vllm_container_lifecycle.
    ensure_serving` is called for the resolved seat FIRST (fail-soft) and its
    ``load_seconds`` recorded. The seat is unloaded best-effort in a ``finally``.
    """
    html_path = Path(html_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = parse_accessible_html(html_path)
    log(f"[qc-standalone] parsed {len(records)} block(s) from {html_path.name}")
    window = _window_from_records(records)
    idx_to_block_id = {r["index"]: r["block_id"] for r in records}

    seat = reasoning_qc._resolve_qc_seat()  # fail-loud on a misconfigured hosted seat
    # The seat's model is the id actually POSTed (run_qc_judgment uses seat.model),
    # which already reflects the SEMANTIK_REASONING_QC_MODEL override chain.
    model = getattr(seat, "model", None)

    # Container-lifecycle ensure_serving FIRST (fail-soft; no-op when the flag is
    # off) — record load_seconds so the A/B can attribute a cold-start cost.
    load_seconds = _ensure_serving(seat, run_dir=run_dir, log=log)

    wall_start = time.monotonic()
    try:
        verdicts_by_index = reasoning_qc._fan_out_page_verifies(
            seat, str(html_path), [window], log=log
        )
    finally:
        reasoning_qc._unload_seat(seat)
    wall_clock_seconds = time.monotonic() - wall_start

    verdict, flagged, divergence = verdicts_by_index.get(0, ({}, [], 0))

    # Findings — absolute index + block id + failure mode (+ any proposed run).
    findings: list[dict[str, Any]] = []
    for f in flagged:
        ridx = getattr(f, "region_index", None)
        run = list(getattr(f, "proposed_regroup_run", ()) or ())
        findings.append(
            {
                "index": ridx,
                "block_id": idx_to_block_id.get(ridx),
                "finding_kind": getattr(f, "failure_mode", None),
                "fix_hint": getattr(f, "fix_hint", None),
                "fixable": bool(getattr(f, "fixable", False)),
                "proposed_run": run,
                "proposed_run_block_ids": [idx_to_block_id.get(i) for i in run],
            }
        )

    # Unit-plan stats (windows + seams) mirror the fan-out's own partition.
    subwins, seams = reasoning_qc._plan_page_units(len(window.block_records))
    incomplete_positions = reasoning_qc._qc_int_list(verdict.get("_qc_incomplete"))
    incomplete_block_ids = [
        idx_to_block_id.get(p) for p in incomplete_positions if p in idx_to_block_id
    ]

    report: dict[str, Any] = {
        "schema": STANDALONE_QC_SCHEMA,
        "html_path": str(html_path.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seat": {
            "provider": getattr(seat, "provider", None),
            "base_url": getattr(seat, "base_url", None),
            "model": model,
        },
        "units": {
            "n_blocks": len(window.block_records),
            "windows": len(subwins),
            "seams": len(seams),
            "judged": len(subwins) + len(seams),
            "qc_incomplete_count": len(incomplete_positions),
            "qc_incomplete_positions": incomplete_positions,
            "qc_incomplete_block_ids": incomplete_block_ids,
        },
        "verdict_confidence": verdict.get("confidence"),
        "order_divergence": divergence,
        "n_findings": len(findings),
        "findings": findings,
        "per_unit_latency_seconds": {
            # A single wall-clock is reported (the fan-out judges units
            # concurrently); per-unit timing would require instrumenting the
            # shared fan-out, which this REPORT-ONLY runner deliberately does not.
            "aggregate_wall_clock": round(wall_clock_seconds, 3),
        },
        "wall_clock_seconds": round(wall_clock_seconds, 3),
        "load_seconds": load_seconds,
    }

    out_path = out_dir / f"{_stem(html_path)}.qc_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(
        f"[qc-standalone] {html_path.name}: {len(findings)} finding(s), "
        f"divergence={divergence}, wall={report['wall_clock_seconds']}s → {out_path}"
    )
    return report


def _stem(html_path: Path) -> str:
    """The report stem: drop a trailing ``_accessible`` for a tidy report name."""
    stem = html_path.stem
    if stem.endswith("_accessible"):
        stem = stem[: -len("_accessible")]
    return stem


def _ensure_serving(
    seat: Any, *, run_dir: str | Path | None, log: Callable[[str], None]
) -> float | None:
    """Best-effort :func:`lib.vllm_container_lifecycle.ensure_serving` for the seat.

    Returns the measured ``load_seconds`` (``0.0`` when already serving, a float
    when a container was started, ``None`` when the lifecycle flag is off / the
    seat is unmapped / the import fails). NEVER raises — a lifecycle miss must not
    stop the A/B judgment."""
    base_url = getattr(seat, "base_url", None)
    if not base_url:
        return None
    try:
        from lib.vllm_container_lifecycle import ensure_serving

        load_seconds = ensure_serving(base_url, run_dir=run_dir)
        if load_seconds:
            log(f"[qc-standalone] ensure_serving({base_url}) → load_seconds={load_seconds:.1f}")
        return load_seconds
    except Exception as exc:  # noqa: BLE001 — lifecycle is best-effort
        logger.debug("qc-standalone ensure_serving failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Compare — align two reports by (block_id, finding_kind).
# ---------------------------------------------------------------------------
def _finding_keys(report: dict[str, Any]) -> set[tuple[Any, Any]]:
    return {
        (f.get("block_id"), f.get("finding_kind"))
        for f in report.get("findings") or ()
    }


def compare_reports(
    report_a_path: str | Path, report_b_path: str | Path
) -> dict[str, Any]:
    """Align two standalone QC reports by ``(block_id, finding_kind)``.

    Returns a comparison dict: both / only-A / only-B finding counts (+ the keys),
    the two seats, ``qc_incomplete`` rates, and latency / wall-clock deltas. Used
    for the omni-vs-Super seat A/B. Reads the reports from disk (JSON).
    """
    a = json.loads(Path(report_a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(report_b_path).read_text(encoding="utf-8"))

    keys_a = _finding_keys(a)
    keys_b = _finding_keys(b)
    both = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    def _incomplete_rate(rep: dict[str, Any]) -> float | None:
        units = rep.get("units") or {}
        n = units.get("n_blocks") or 0
        if not n:
            return None
        return round(float(units.get("qc_incomplete_count") or 0) / float(n), 4)

    def _keypairs(keys: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
        return [{"block_id": k[0], "finding_kind": k[1]} for k in keys]

    compare: dict[str, Any] = {
        "schema": QC_COMPARE_SCHEMA,
        "report_a": str(Path(report_a_path).resolve()),
        "report_b": str(Path(report_b_path).resolve()),
        "seat_a": a.get("seat"),
        "seat_b": b.get("seat"),
        "findings": {
            "both": len(both),
            "only_a": len(only_a),
            "only_b": len(only_b),
            "both_keys": _keypairs(both),
            "only_a_keys": _keypairs(only_a),
            "only_b_keys": _keypairs(only_b),
        },
        "n_findings": {"a": a.get("n_findings"), "b": b.get("n_findings")},
        "qc_incomplete": {
            "a_count": (a.get("units") or {}).get("qc_incomplete_count"),
            "b_count": (b.get("units") or {}).get("qc_incomplete_count"),
            "a_rate": _incomplete_rate(a),
            "b_rate": _incomplete_rate(b),
        },
        "order_divergence": {"a": a.get("order_divergence"), "b": b.get("order_divergence")},
        "wall_clock_seconds": {
            "a": a.get("wall_clock_seconds"),
            "b": b.get("wall_clock_seconds"),
            "delta_b_minus_a": _sub(b.get("wall_clock_seconds"), a.get("wall_clock_seconds")),
        },
        "load_seconds": {"a": a.get("load_seconds"), "b": b.get("load_seconds")},
    }
    return compare


def _sub(x: Any, y: Any) -> float | None:
    try:
        return round(float(x) - float(y), 3)
    except (TypeError, ValueError):
        return None


def _format_compare_summary(compare: dict[str, Any]) -> str:
    """A human-readable one-screen summary of a :func:`compare_reports` dict."""
    f = compare["findings"]
    inc = compare["qc_incomplete"]
    wall = compare["wall_clock_seconds"]
    seat_a = (compare.get("seat_a") or {}).get("model")
    seat_b = (compare.get("seat_b") or {}).get("model")
    lines = [
        "=== reasoning-QC seat A/B ===",
        f"  A: model={seat_a}  ({compare['report_a']})",
        f"  B: model={seat_b}  ({compare['report_b']})",
        f"  findings: both={f['both']}  only-A={f['only_a']}  only-B={f['only_b']}",
        f"  n_findings: A={compare['n_findings']['a']}  B={compare['n_findings']['b']}",
        f"  qc_incomplete: A={inc['a_count']} (rate {inc['a_rate']})  "
        f"B={inc['b_count']} (rate {inc['b_rate']})",
        f"  order_divergence: A={compare['order_divergence']['a']}  "
        f"B={compare['order_divergence']['b']}",
        f"  wall_clock: A={wall['a']}s  B={wall['b']}s  (B-A={wall['delta_b_minus_a']}s)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _iter_html_inputs(target: Path) -> list[Path]:
    """A single ``*_accessible.html`` file, or every one under a directory."""
    if target.is_dir():
        return sorted(target.glob("*_accessible.html"))
    return [target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m semantik_structure.reasoning_qc_standalone",
        description=(
            "Standalone Stage-9b reasoning-QC runner: judge an already-emitted "
            "SemantiK accessible-HTML file (or a directory of them) with the "
            "reasoning-QC machinery, report-only. Or --compare two reports."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="an *_accessible.html file or a directory of them; "
        "with --compare, exactly two .qc_report.json paths",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="directory for the emitted *.qc_report.json / qc_compare.json (default: cwd)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare two prior *.qc_report.json files (writes qc_compare.json)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        if len(args.inputs) != 2:
            parser.error("--compare requires exactly two report paths")
        compare = compare_reports(args.inputs[0], args.inputs[1])
        out_path = out_dir / "qc_compare.json"
        out_path.write_text(json.dumps(compare, indent=2) + "\n", encoding="utf-8")
        print(_format_compare_summary(compare))
        print(f"[qc-standalone] wrote {out_path}")
        return 0

    if not args.inputs:
        parser.error("at least one *_accessible.html input (or a directory) is required")

    rc = 0
    for target in args.inputs:
        for html_path in _iter_html_inputs(Path(target)):
            try:
                run_standalone_qc(html_path, out_dir=out_dir)
            except Exception as exc:  # noqa: BLE001 — one bad file must not abort the batch
                print(f"[qc-standalone] ERROR judging {html_path}: {exc}", file=sys.stderr)
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
