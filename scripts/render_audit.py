#!/usr/bin/env python3
"""GAP 3 — headless render audit for assembled end-user HTML pages.

Systematizes the manual browser check from the round-8 visual review: load
each assembled page in a real headless Chromium, wait for MathJax to
typeset, then assert the rendered DOM is clean. Catches the defects that
only surface AFTER the browser runs — un-typeset math, literal LaTeX
delimiters leaking as visible text, MathJax error nodes, duplicate ids,
broken landmark structure, and images missing alt text.

Usage (name=path pairs, like scripts/shoot_pages.py)::

    python scripts/render_audit.py ch02=state/qa/visual/ch02_conv.html \\
        ch09=state/qa/visual/ch09_conv.html [--json-out report.json]

Exit code is non-zero when any page has a failure-severity finding.

RAM GUARD: mirrors scripts/shoot_pages.py — Chromium typesetting thousands
of MathJax equations can spike multiple GB, so this refuses to launch under
6 GB MemAvailable and never runs alongside an ed4all synthesis.

The DOM inspection returns raw counts + the outside-MathJax text; all
pass/fail logic lives in pure functions (``scan_literal_delimiters`` /
``find_duplicate_ids`` / ``assess_page``) that are unit-tested without a
browser. A tiny fixture-page smoke test runs only when Chromium is
available (skipped otherwise).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------ #
# Pure functions (unit-tested, no browser).
# ------------------------------------------------------------------ #

#: Literal math delimiters that must NOT appear as visible text after
#: MathJax runs. Their presence means math failed to typeset (or was
#: authored outside a math context).
LITERAL_DELIMITERS: Tuple[str, ...] = ("\\(", "\\)", "\\[", "\\]", "$$")


def scan_literal_delimiters(text: str) -> Dict[str, int]:
    """Count each literal delimiter occurring in ``text``.

    ``text`` is the rendered innerText/textContent with MathJax containers
    already stripped out — so any delimiter here is un-typeset source or
    stray markup, i.e. a real defect. Returns only the delimiters that
    actually occur (count > 0).
    """
    counts: Dict[str, int] = {}
    if not text:
        return counts
    for delim in LITERAL_DELIMITERS:
        c = text.count(delim)
        if c:
            counts[delim] = c
    return counts


def find_duplicate_ids(ids: Sequence[str]) -> Dict[str, int]:
    """Return ``{id: count}`` for every id that appears more than once."""
    seen: Dict[str, int] = {}
    for i in ids:
        if not i:
            continue
        seen[i] = seen.get(i, 0) + 1
    return {i: c for i, c in seen.items() if c > 1}


@dataclass
class PageFinding:
    code: str
    severity: str  # "failure" | "warning"
    detail: str


@dataclass
class PageAudit:
    name: str
    path: str
    findings: List[PageFinding] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return any(f.severity == "failure" for f in self.findings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "failed": self.failed,
            "findings": [
                {"code": f.code, "severity": f.severity, "detail": f.detail}
                for f in self.findings
            ],
            "notes": self.notes,
            "counts": {
                "mjx_merror": self.raw.get("mjx_merror_count", 0),
                "main": self.raw.get("main_count", 0),
                "skip_link": self.raw.get("skip_link_count", 0),
                "img_missing_alt": self.raw.get("img_missing_alt", 0),
                "total_ids": len(self.raw.get("ids", [])),
            },
        }


def assess_page(name: str, path: str, raw: Dict[str, Any]) -> PageAudit:
    """Turn the raw DOM-inspection payload into a PageAudit with findings.

    Failure-severity findings:
      * MJX_MERROR         — MathJax error node(s) present.
      * LITERAL_DELIMITERS — un-typeset LaTeX delimiter text outside math.
      * DUPLICATE_IDS      — repeated element ids (breaks anchors + a11y).
      * MISSING_MAIN       — not exactly one <main> landmark.
      * MISSING_SKIP_LINK  — not exactly one skip link.
    Warning-severity:
      * IMG_MISSING_ALT    — image(s) without an alt attribute.
    """
    audit = PageAudit(name=name, path=path, raw=raw)

    merror = int(raw.get("mjx_merror_count", 0) or 0)
    if merror > 0:
        audit.findings.append(PageFinding(
            "MJX_MERROR", "failure",
            f"{merror} MathJax error node(s) (mjx-merror / [data-mjx-error]).",
        ))

    delim_counts = scan_literal_delimiters(str(raw.get("text_outside_mjx", "")))
    if delim_counts:
        pretty = ", ".join(f"{d!r}×{c}" for d, c in delim_counts.items())
        audit.findings.append(PageFinding(
            "LITERAL_DELIMITERS", "failure",
            f"Visible literal math delimiters outside MathJax: {pretty}.",
        ))

    dupes = find_duplicate_ids(raw.get("ids", []) or [])
    if dupes:
        sample = ", ".join(f"{i}×{c}" for i, c in list(dupes.items())[:8])
        audit.findings.append(PageFinding(
            "DUPLICATE_IDS", "failure",
            f"{len(dupes)} duplicated id(s): {sample}.",
        ))

    main_count = int(raw.get("main_count", 0) or 0)
    if main_count != 1:
        audit.findings.append(PageFinding(
            "MISSING_MAIN", "failure",
            f"Expected exactly 1 <main> landmark, found {main_count}.",
        ))

    skip_count = int(raw.get("skip_link_count", 0) or 0)
    if skip_count != 1:
        audit.findings.append(PageFinding(
            "MISSING_SKIP_LINK", "failure",
            f"Expected exactly 1 skip link, found {skip_count}.",
        ))

    missing_alt = int(raw.get("img_missing_alt", 0) or 0)
    if missing_alt > 0:
        audit.findings.append(PageFinding(
            "IMG_MISSING_ALT", "warning",
            f"{missing_alt} image(s) missing an alt attribute.",
        ))

    if raw.get("mathjax_timeout"):
        audit.notes.append(
            "MathJax typeset wait timed out — delimiter scan may include "
            "still-un-typeset source."
        )
    if raw.get("mathjax_absent"):
        audit.notes.append("No MathJax on page — delimiter scan is literal.")

    return audit


# ------------------------------------------------------------------ #
# Browser inspection (live; not unit-tested — exercised via fixture).
# ------------------------------------------------------------------ #

#: JS that returns the raw inspection payload for one loaded page.
_INSPECT_JS = r"""
() => {
  const ids = Array.from(document.querySelectorAll('[id]')).map(e => e.id);
  const mjxErrors = document.querySelectorAll('mjx-merror, [data-mjx-error]').length;
  const mains = document.querySelectorAll('main, [role="main"]').length;
  const skipLinks = Array.from(document.querySelectorAll('a[href^="#"]')).filter(
    a => /skip/i.test(a.textContent || '') || /skip/i.test(a.className || '')
  ).length;
  const imgMissingAlt = Array.from(document.querySelectorAll('img')).filter(
    i => !i.hasAttribute('alt')
  ).length;
  // Text with MathJax output stripped so only un-typeset source / stray
  // delimiters remain visible to the scanner.
  const clone = document.body.cloneNode(true);
  clone.querySelectorAll(
    'mjx-container, mjx-assistive-mml, script, style'
  ).forEach(n => n.remove());
  const textOutsideMjx = clone.textContent || '';
  return {
    ids,
    mjx_merror_count: mjxErrors,
    main_count: mains,
    skip_link_count: skipLinks,
    img_missing_alt: imgMissingAlt,
    text_outside_mjx: textOutsideMjx,
  };
}
"""


def _wait_for_mathjax(page, timeout_ms: int) -> Dict[str, bool]:
    """Wait for MathJax to finish typesetting. Returns status flags."""
    status = {"mathjax_absent": False, "mathjax_timeout": False}
    has_mathjax = page.evaluate("() => typeof window.MathJax !== 'undefined'")
    if not has_mathjax:
        status["mathjax_absent"] = True
        return status
    try:
        # Resolve once startup + any pending typeset settle.
        page.evaluate(
            """() => {
                if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
                    return MathJax.startup.promise.then(() =>
                        (MathJax.typesetPromise ? MathJax.typesetPromise() : null)
                    );
                }
                return null;
            }""",
        )
    except Exception:
        status["mathjax_timeout"] = True
    return status


def inspect_pages(
    pages: Sequence[Tuple[str, str]],
    *,
    load_timeout_ms: int = 20000,
    mathjax_settle_ms: int = 3500,
) -> List[PageAudit]:
    """Load each page in headless Chromium and return per-page audits."""
    from playwright.sync_api import sync_playwright  # lazy import

    audits: List[PageAudit] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 950})
        pg = ctx.new_page()
        for name, fpath in pages:
            resolved = Path(fpath).resolve()
            pg.goto(f"file://{resolved}")
            try:
                pg.wait_for_load_state("networkidle", timeout=load_timeout_ms)
            except Exception:
                pass
            mj_status = _wait_for_mathjax(pg, load_timeout_ms)
            pg.wait_for_timeout(mathjax_settle_ms)
            raw = pg.evaluate(_INSPECT_JS)
            raw.update(mj_status)
            audits.append(assess_page(name, str(resolved), raw))
        ctx.close()
        browser.close()
    return audits


# ------------------------------------------------------------------ #
# RAM guard + CLI.
# ------------------------------------------------------------------ #


def _ram_guard() -> Optional[str]:
    """Return an error string if it is unsafe to launch Chromium, else None."""
    try:
        with open("/proc/meminfo") as fh:
            avail_kb = int(
                next(l for l in fh if "MemAvailable" in l).split()[1]
            )
    except (OSError, StopIteration, ValueError):
        return None  # non-Linux / unknown — don't block
    if avail_kb < 6 * 1024 * 1024:
        return (
            f"RAM guard: {avail_kb // 1024} MB available (<6GB) — refusing "
            f"chromium launch"
        )
    import subprocess

    if subprocess.run(
        ["pgrep", "-f", "ed4all run textbook"], capture_output=True
    ).stdout.strip():
        return "RAM guard: ed4all synthesis running — render audit deferred"
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless render audit for assembled end-user HTML pages."
    )
    parser.add_argument(
        "pages", nargs="+", metavar="name=path.html",
        help="One or more name=path.html pairs (like shoot_pages.py).",
    )
    parser.add_argument("--json-out", default=None,
                        help="Write the JSON report to this path.")
    args = parser.parse_args(argv)

    try:
        pages = [tuple(a.split("=", 1)) for a in args.pages]
        for pair in pages:
            if len(pair) != 2:
                raise ValueError
    except ValueError:
        print("usage: render_audit.py name=path.html [...]", file=sys.stderr)
        return 2
    for _name, fpath in pages:
        if not Path(fpath).exists():
            print(f"ERROR: page not found: {fpath}", file=sys.stderr)
            return 2

    guard = _ram_guard()
    if guard:
        print(guard, file=sys.stderr)
        return 2

    audits = inspect_pages(pages)

    report = {
        "pages": [a.to_dict() for a in audits],
        "any_failed": any(a.failed for a in audits),
    }
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    # Human output.
    for a in audits:
        status = "FAIL" if a.failed else "OK"
        print(f"[{status}] {a.name} ({a.path})")
        for note in a.notes:
            print(f"    note: {note}")
        if not a.findings:
            print("    clean — no findings.")
        for f in a.findings:
            print(f"    {f.severity.upper()}: {f.code} — {f.detail}")

    return 1 if report["any_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
