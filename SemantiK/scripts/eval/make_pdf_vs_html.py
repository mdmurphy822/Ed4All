"""Run the full SemantiK pipeline on a PDF and build a side-by-side viewer.

Output folder under eval/side_by_side/<run>/ contains:
  - input.pdf          copy of the source PDF
  - output.html        the pipeline's generated HTML
  - result.json        full pipeline result (html, role counts, axe, escalation)
  - index.html         two-pane viewer: PDF on the left, HTML on the right

Open index.html in a browser to compare.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Make the repo root importable so `from scripts._html_structure_audit import ...`
# works when this file is run as a direct script (python scripts/eval/make_pdf_vs_html.py),
# not just as a module (python -m scripts.make_pdf_vs_html). eval_v7_family.sh and
# the nohup runners invoke it as a direct script, where scripts/ — not the repo
# root — is on sys.path, so the package import would otherwise ModuleNotFoundError.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SemantiK side-by-side — {name}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: system-ui, sans-serif; }}
  header {{ padding: .5rem 1rem; background: #222; color: #eee; display: flex; gap: 1rem; align-items: baseline; flex-wrap: wrap; }}
  header h1 {{ font-size: 1rem; margin: 0; font-weight: 600; }}
  header .meta {{ color: #aaa; font-size: .85rem; }}
  header .ok {{ color: #5cdb6f; }} header .bad {{ color: #ff7b72; }}
  .pane-row {{ display: flex; height: calc(100% - 2.3rem); }}
  .pane {{ flex: 1 1 34%; border-right: 1px solid #444; display: flex; flex-direction: column; min-width: 0; }}
  .pane:last-child {{ border-right: 0; }}
  .pane-title {{ padding: .25rem .75rem; background: #333; color: #ddd; font-size: .8rem; }}
  .pane iframe {{ border: 0; flex: 1; width: 100%; }}
  .struct {{ flex: 1; overflow: auto; padding: .5rem .75rem; font-size: .8rem; line-height: 1.4; }}
  .struct h3 {{ font-size: .8rem; margin: .8rem 0 .3rem; color: #444; text-transform: uppercase; letter-spacing: .03em; }}
  .outline div {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: ui-monospace, monospace; }}
  .outline .lh1 {{ font-weight: 700; }} .outline .lh2 {{ padding-left: 1.2em; }}
  .outline .lh3 {{ padding-left: 2.4em; color: #555; }} .outline .lh4 {{ padding-left: 3.6em; color: #777; }}
  .smell {{ color: #b35900; margin: .15em 0; }} .smell .sc {{ color: #888; font-family: monospace; }}
  .clean {{ color: #2a8a3a; }}
  table.census {{ border-collapse: collapse; }} table.census td {{ padding: 0 .6em 0 0; }}
  table.census td:last-child {{ font-family: monospace; text-align: right; }}
</style>
</head>
<body>
<header>
  <h1>{name}</h1>
  <span class="meta">{engine} · blocks: {n_blocks} · axe_ok: <b class="{axe_class}">{axe_ok}</b> · verdict: {verdict} · headings: {n_headings} · structure smells: <b class="{smell_class}">{n_smells}</b></span>
</header>
<div class="pane-row">
  <div class="pane">
    <div class="pane-title">original PDF (before)</div>
    <iframe src="input.pdf"></iframe>
  </div>
  <div class="pane">
    <div class="pane-title">generated accessible HTML (after)</div>
    <iframe src="output.html"></iframe>
  </div>
  <div class="pane">
    <div class="pane-title">semantic structure audit</div>
    <div class="struct">{structure_panel}</div>
  </div>
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument(
        "--engine",
        choices=("council", "v1"),
        default="council",
        help="council = the production v2 cascade (Stage 1..13: "
        "BERT council + Qwen GGUF specialists + theta v8, real "
        "runtime) via run_full_cascade. v1 = the legacy "
        "classify+reason pipeline (distilbert classifier + Qwen "
        "reasoner LoRA); does NOT exercise the council.",
    )
    ap.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/reasoner_v8/final"),
        help="[v1 engine only] Reasoner LoRA adapter. Default tracks "
        "the most recently trained reasoner (v8).",
    )
    ap.add_argument(
        "--classifier",
        type=Path,
        default=Path("models/classifier_v5/final"),
        help="[v1 engine only] Stage-3a classifier.",
    )
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument(
        "--k",
        type=int,
        default=2,
        help="[council engine] Stage 9b gap-fill generations per region (default 2).",
    )
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-root", type=Path, default=Path("eval/side_by_side"))
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"not found: {args.pdf}")

    run_name = args.run_name or (f"{args.pdf.stem}_{datetime.now():%Y%m%d-%H%M}")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    if args.engine == "council":
        html, res_json = _run_council(args)
        engine_tag = "council (v2 cascade, real runtime)"
    else:
        html, res_json = _run_v1(args)
        engine_tag = f"v1 (classify+reason, adapter {args.adapter})"

    # Persist the EXPENSIVE outputs first — the cascade just spent minutes
    # generating this HTML; never let a cheap post-processing bug (e.g. the
    # structure audit) discard it.
    shutil.copy(args.pdf, out_dir / "input.pdf")
    (out_dir / "output.html").write_text(html)

    # Semantic-structure fidelity — measured on the output HTML for BOTH
    # engines (axe pass != correct structure). This is the second eval axis
    # the WCAG/axe verdict can't see.
    from scripts._html_structure_audit import audit_structure

    n_pages = _pdf_pages(args.pdf)
    audit = audit_structure(html, n_pages=n_pages)
    res_json["n_pages"] = n_pages
    res_json["structure_audit"] = audit

    (out_dir / "result.json").write_text(json.dumps(res_json, indent=2))
    (out_dir / "structure_audit.txt").write_text(_structure_text(audit, n_pages))
    (out_dir / "index.html").write_text(
        INDEX_HTML.format(
            name=args.pdf.name,
            engine=engine_tag,
            n_blocks=res_json.get("n_classified_blocks", "?"),
            axe_ok=res_json.get("axe_ok"),
            axe_class=("ok" if res_json.get("axe_ok") is True else "bad"),
            verdict=res_json.get("escalation_verdict"),
            n_headings=audit["census"]["headings_total"],
            n_smells=len(audit["smells"]),
            smell_class=("ok" if not audit["smells"] else "bad"),
            structure_panel=_structure_panel_html(audit, res_json),
        )
    )

    print(
        f"[done] open {out_dir}/index.html  (axe_ok={res_json.get('axe_ok')} "
        f"verdict={res_json.get('escalation_verdict')} "
        f"headings={audit['census']['headings_total']} smells={len(audit['smells'])})"
    )


def _pdf_pages(pdf) -> int | None:
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf)) as p:
            return len(p.pages)
    except Exception:
        return None


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _structure_panel_html(audit: dict, res_json: dict) -> str:
    """Render the structure audit as an HTML fragment for the viewer's 3rd pane."""
    parts = []
    theta = res_json.get("theta_score")
    if theta is not None:
        parts.append(f"<div>theta semantic-preservation: <b>{theta}</b></div>")
    cov = audit["coverage"]
    parts.append(
        "<div>table-header coverage: <b>{thc}</b> · img-alt coverage: <b>{iac}</b> · "
        "main landmark: <b>{ml}</b> · lang: <b>{lg}</b></div>".format(
            thc=cov["table_header_coverage"],
            iac=cov["img_alt_coverage"],
            ml=cov["has_main_landmark"],
            lg=cov["has_lang"],
        )
    )
    # smells
    parts.append("<h3>structure smells (axe-blind)</h3>")
    if not audit["smells"]:
        parts.append('<div class="clean">none — structurally clean ✓</div>')
    else:
        for s in audit["smells"]:
            parts.append(
                f'<div class="smell"><span class="sc">[SC {_esc(s["sc"])}]</span> '
                f"{_esc(s['code'])}: {_esc(s['detail'])}</div>"
            )
    # heading outline
    parts.append(f"<h3>heading outline ({audit['census']['headings_total']})</h3>")
    parts.append('<div class="outline">')
    for h in audit["outline"][:120]:
        lvl = min(h["level"], 4)
        parts.append(f'<div class="lh{lvl}">h{h["level"]} {_esc(h["text"])}</div>')
    if not audit["outline"]:
        parts.append('<div class="smell">(no headings — flattened structure)</div>')
    parts.append("</div>")
    # census table
    parts.append("<h3>element census</h3><table class='census'>")
    for k, v in audit["census"].items():
        if k in ("headings_by_level", "landmarks"):
            v = ", ".join(f"{kk}={vv}" for kk, vv in v.items()) or "—"
        parts.append(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>")
    parts.append("</table>")
    return "\n".join(parts)


def _structure_text(audit: dict, n_pages) -> str:
    lines = [f"STRUCTURE AUDIT (pages={n_pages})", "", "OUTLINE:"]
    for h in audit["outline"]:
        lines.append(f"  {'  ' * (h['level'] - 1)}h{h['level']}: {h['text'][:90]}")
    if not audit["outline"]:
        lines.append("  (none)")
    lines += [
        "",
        "COVERAGE:",
        f"  {audit['coverage']}",
        "",
        "CENSUS:",
        f"  {audit['census']}",
        "",
        f"SMELLS ({len(audit['smells'])}):",
    ]
    for s in audit["smells"]:
        lines.append(f"  [SC {s['sc']}] {s['code']}: {s['detail']}")
    if not audit["smells"]:
        lines.append("  none — structurally clean")
    return "\n".join(lines) + "\n"


def _run_council(args) -> tuple[str, dict]:
    """Real v2 cascade (Stage 1..13): council BERTs + Qwen GGUFs + theta v8.

    Mirrors scripts/pdf_to_html.py — runs run_full_cascade with
    runtime_mode='real' and a Chromium-backed axe validator, so the
    verdict here is a genuine WCAG-2.2-AA pass, not a mock.
    """
    from semantik_structure.cascade import run_full_cascade
    from semantik_structure.validate import HtmlValidator

    print(
        f"[council] real runtime: loading council BERTs + 4 Qwen GGUFs + theta v8 "
        f"serially on the 8GB card — minutes/PDF. pdf={args.pdf.name}"
    )
    with HtmlValidator() as validator:
        result = run_full_cascade(
            args.pdf,
            validator=validator,
            k=args.k,
            runtime_mode="real",
            return_html=True,
            log=lambda m: print(f"[cascade] {m}", flush=True),
        )
    html = result.get("html") or "<!-- cascade produced no HTML -->"
    region_kinds = result.get("region_kinds") or {}
    wcag = result.get("wcag_status_under_mock")  # canonical key (cascade.py:457)
    axe = result.get("stage10_axe_under_mock") or {}
    theta = result.get("theta") or {}
    res_json = {
        "engine": "council",
        "pdf_path": str(args.pdf),
        "n_classified_blocks": sum(region_kinds.values()) if region_kinds else None,
        "region_kinds": dict(region_kinds),
        "axe_ok": (
            str(wcag).lower() in ("pass", "passed", "ok", "true") if wcag is not None else None
        ),
        "wcag_status": wcag,
        "axe_violations": axe.get("violations") if isinstance(axe, dict) else None,
        "escalation_verdict": wcag,
        "theta_score": theta.get("theta_score"),
        "theta_action": (
            theta.get("action").value
            if hasattr(theta.get("action"), "value")
            else theta.get("action")
        ),
        "lane_used": result.get("lane_used"),
        "html_chars": len(html),
    }
    return html, res_json


def _run_v1(args) -> tuple[str, dict]:
    """Legacy v1 pipeline (distilbert classify -> Qwen reason). NOT the council."""
    from semantik_structure.classify import load_classifier
    from semantik_structure.pipeline import run_pipeline
    from semantik_structure.reason import load_reasoner, unload_reasoner

    print(f"[v1] load classifier {args.classifier}")
    classifier = load_classifier(args.classifier)
    print(f"[v1] load reasoner {args.base_model} + adapter {args.adapter}")
    reasoner = load_reasoner(args.base_model, adapter_path=args.adapter)
    try:
        print(f"[v1] run {args.pdf.name}")
        result = run_pipeline(args.pdf, language="en", classifier=classifier, reasoner=reasoner)
    finally:
        unload_reasoner(reasoner)

    html = result.html or "<!-- pipeline produced no HTML -->"
    role_counts = Counter(cb.role for cb in result.classified_blocks)
    src_counts = Counter(cb.source for cb in result.classified_blocks)
    res_json = {
        "engine": "v1",
        "pdf_path": str(result.pdf_path),
        "adapter": str(args.adapter),
        "n_raw_blocks": len(result.raw_blocks),
        "n_classified_blocks": len(result.classified_blocks),
        "role_counts": dict(role_counts),
        "source_counts": dict(src_counts),
        "axe_ok": (result.validation.ok if result.validation else None),
        "axe_violations": (
            [str(v) for v in result.validation.violations] if result.validation else None
        ),
        "escalation_verdict": (result.escalation.verdict if result.escalation else None),
        "escalation_reasons": (result.escalation.reasons if result.escalation else []),
        "stage_errors": result.stage_errors,
        "html_chars": len(html),
    }
    return html, res_json


if __name__ == "__main__":
    main()
