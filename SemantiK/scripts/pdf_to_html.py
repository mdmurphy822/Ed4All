"""scripts/pdf_to_html.py — one command: PDF -> accessible HTML file.

Thin wrapper over :func:`semantik_structure.cascade.run_full_cascade` with
``return_html=True``. Runs the full v2 cascade (Stage 1..13) on ONE PDF and
writes the assembled, WCAG-2.2-AA-gated HTML *document* to ``--out``.

Why this exists: ``scripts/eval/eval_full_cascade.py`` deliberately persists JSON
reports carrying ``html_length`` only (to keep per-PDF reports small) — it does
not write the HTML itself. This script persists the product: ``result["html"]``
(a complete ``<!DOCTYPE html>`` document built by ``assembler/shell.py``)
written verbatim.

This script also doubles as the per-PDF VRAM-isolation worker for
``scripts/eval/eval_full_cascade.py`` (``--isolate-per-pdf``): when ``--out`` is
omitted and ``--report`` is given, it runs the cascade with
``return_html=False`` (report-only mode) and, on exception, writes a JSON
report carrying ``error`` + ``traceback`` + ``wall_elapsed`` before exiting
nonzero — the contract ``eval_full_cascade._run_cascade_isolated`` parses.

No stubs, no drift:
  * Theta runs the real v8 cross-encoder. ``SEMANTIK_ALLOW_THETA_STUB`` is neither
    set nor needed (the strict-by-default loader picks up ``models/theta/
    semantic_preservation/v8``); if theta weights were missing the cascade
    raises loudly rather than silently scoring 0.7 (feedback_no_silent_fallbacks).
  * The WCAG verdict is read from the canonical ``wcag_status_under_mock`` key
    (cascade.py:457) — the SAME key ``run_pipeline_v2`` reads (cascade.py:582) —
    not the ``wcag_status`` name the docstring loosely implies.

Usage::

    # Real runtime (loads council BERTs + 4 Qwen GGUFs + theta v8; minutes/PDF
    # on the 8GB card, models load serially):
    .venv/bin/python scripts/pdf_to_html.py --pdf in.pdf --out out.html

    # GPU-free, Chromium-free smoke (mock Qwen gap-fill, skip the axe gate):
    .venv/bin/python scripts/pdf_to_html.py --pdf in.pdf --out out.html \\
        --runtime mock --no-validate

    # Force a fresh extraction (the mtime-keyed data/extract_cache can be stale):
    .venv/bin/python scripts/pdf_to_html.py --pdf in.pdf --out out.html \\
        --invalidate-extract-cache
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# run_full_cascade imports semantik_structure.validate, which imports
# axe_playwright_python at module load (validate.py:26). When axe is absent the
# import fails BEFORE any flag is read, so --no-validate cannot rescue it — the
# only honest handling is a clear remediation message (exit 3), not a raw
# traceback. (Present in .venv; absent from a bare system python.)
try:
    from semantik_structure.cascade import run_full_cascade
    from semantik_structure.validate import HtmlValidator

    _IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - env-dependent
    run_full_cascade = None  # type: ignore[assignment]
    HtmlValidator = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _enum_value(obj):
    """Mirror cascade._enum_value: enum member -> its .value, else unchanged.

    Kept local so the printed theta action matches what run_pipeline_v2 reports
    (cascade.py:504-506) without importing a private helper."""
    return obj.value if hasattr(obj, "value") else obj


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Turn ONE PDF into an accessible WCAG-2.2-AA HTML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pdf", type=Path, required=True, help="input PDF")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .html path. Omit for report-only mode (requires --report): "
        "the cascade runs with return_html=False and no HTML is written.",
    )
    ap.add_argument(
        "--runtime",
        choices=("mock", "real"),
        default="real",
        help="real = council BERTs + Qwen GGUFs + theta v8 (default); "
        "mock = GPU-free (mock Qwen gap-fill).",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=2,
        help="Stage 9b gap-fill generations per region (default 2).",
    )
    ap.add_argument(
        "--max-regions-per-kind",
        type=int,
        default=0,
        help="cap regions per kind (0 = no cap, default).",
    )
    ap.add_argument(
        "--enable-glm-ocr-stage",
        action="store_true",
        help="enable optional Stage 5b GLM-OCR table enrichment (forwarded to run_full_cascade).",
    )
    ap.add_argument(
        "--no-validate",
        action="store_true",
        help="pass validator=None — skip the axe WCAG gate and Chromium. "
        "The printed verdict is then '(axe skipped)', NOT a real pass.",
    )
    ap.add_argument(
        "--invalidate-extract-cache",
        action="store_true",
        help="wipe data/extract_cache before the run (it is mtime-keyed and "
        "can be stale after extraction-code changes).",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional: also dump the full result dict (minus the html blob) "
        "as JSON for telemetry/debugging.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if _IMPORT_ERROR is not None or run_full_cascade is None:
        print(
            "[pdf2html] FATAL: cannot import the cascade — its axe dependency is "
            f"missing ({_IMPORT_ERROR!r}).\n"
            "           Install it (`uv pip install axe-playwright-python` + "
            "`playwright install chromium`) or run via the project .venv.",
            file=sys.stderr,
            flush=True,
        )
        return 3

    if args.out is None and args.report is None:
        print(
            "[pdf2html] FATAL: omit --out only in report-only mode, which requires --report.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if not args.pdf.exists():
        print(f"[pdf2html] PDF not found: {args.pdf}", file=sys.stderr, flush=True)
        return 2

    if args.invalidate_extract_cache:
        cache_dir = REPO_ROOT / "data" / "extract_cache"
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[pdf2html] wiped extract cache: {cache_dir}", flush=True)

    print(f"[pdf2html] pdf={args.pdf}", flush=True)
    print(
        f"[pdf2html] runtime={args.runtime} k={args.k} "
        f"validate={'off' if args.no_validate else 'on'}",
        flush=True,
    )
    if args.runtime == "real":
        print(
            "[pdf2html] real runtime: loading council BERTs, 4 Qwen GGUFs and "
            "theta v8 serially on the 8GB card — this takes minutes; streaming "
            "cascade progress below.",
            flush=True,
        )

    # Report-only mode (no --out): mirror eval_full_cascade's isolation worker —
    # the cascade runs with return_html=False (report carries html_length only).
    report_only = args.out is None
    want_html = not report_only

    t0 = time.perf_counter()
    # --no-validate -> validator=None: every gate consumer accepts None and
    # skips its axe check (hard_region.py:746, hard_document.py:378). Otherwise
    # own one Chromium-backed HtmlValidator for the run.
    validator_cm = nullcontext(None) if args.no_validate else HtmlValidator()
    try:
        with validator_cm as validator:
            result = run_full_cascade(
                args.pdf,
                validator=validator,
                max_regions_per_kind=args.max_regions_per_kind,
                k=args.k,
                runtime_mode=args.runtime,
                return_html=want_html,
                enable_glm_ocr_stage=args.enable_glm_ocr_stage,
                log=lambda m: print(f"[cascade] {m}", flush=True),
            )
    except Exception as exc:  # noqa: BLE001 — surface the failure, never swallow
        elapsed = time.perf_counter() - t0
        print(f"[pdf2html] FAILED: {exc!r}", file=sys.stderr, flush=True)
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        # No silent fallback: still exit nonzero. But when a --report path was
        # given, persist the failure so eval_full_cascade._run_cascade_isolated
        # can parse it (error/traceback/wall_elapsed) instead of dying with no
        # parseable report.
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "pdf": str(args.pdf),
                        "error": repr(exc),
                        "traceback": tb,
                        "wall_elapsed": elapsed,
                    },
                    indent=1,
                    default=str,
                ),
                encoding="utf-8",
            )
            print(f"[pdf2html] error report -> {args.report}", flush=True)
        return 1
    elapsed = time.perf_counter() - t0

    if want_html:
        # return_html=True was requested, so "html" MUST be present. Treat its
        # absence as a hard error rather than writing an empty file (no silent
        # fallback) — it would mean the cascade contract changed.
        html = result.get("html")
        if html is None:
            print(
                "[pdf2html] FATAL: cascade returned no 'html' despite "
                "return_html=True — the result contract changed.",
                file=sys.stderr,
                flush=True,
            )
            return 1

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(html, encoding="utf-8")  # full <!DOCTYPE html> doc

        # The product never ships without its verifiability artifact
        # (Plan 12 A1). write_conformance_audit raises on any failure;
        # that propagates as a hard nonzero exit like every other error.
        from semantik_structure.conformance_audit import write_conformance_audit

        audit = result.get("conformance_audit")
        if audit is None:
            print(
                "[pdf2html] FATAL: cascade returned no 'conformance_audit' — "
                "the result contract changed.",
                file=sys.stderr,
                flush=True,
            )
            return 1
        audit_path = args.out.with_suffix(".conformance_audit.json")
        write_conformance_audit(audit, audit_path)
        print(f"[pdf2html] conformance audit -> {audit_path}", flush=True)

    # Canonical verdict key (cascade.py:457) — NOT "wcag_status".
    wcag = result.get("wcag_status_under_mock")
    theta = result.get("theta") or {}
    theta_score = theta.get("theta_score")
    action = _enum_value(theta.get("action"))
    lane = result.get("lane_used")

    verdict_suffix = " (axe skipped — verdict not a real pass)" if args.no_validate else ""
    print(f"[pdf2html] WCAG: {wcag}{verdict_suffix}", flush=True)
    print(
        f"[pdf2html] theta_score={theta_score} exit_action={action} lane={lane}",
        flush=True,
    )
    if want_html:
        print(
            f"[pdf2html] html -> {args.out} ({result.get('html_length')} bytes, {elapsed:.1f}s)",
            flush=True,
        )
    else:
        print(
            f"[pdf2html] report-only ({result.get('html_length')} bytes html, {elapsed:.1f}s)",
            flush=True,
        )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {k: v for k, v in result.items() if k != "html"}
        report["wall_elapsed"] = elapsed
        args.report.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"[pdf2html] report -> {args.report}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
