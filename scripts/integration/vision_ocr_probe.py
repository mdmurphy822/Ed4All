#!/usr/bin/env python3
"""Vision-vs-OCR fidelity probe for the image-only scan corpus.

Measures how much a multimodal model (Nemotron-Omni on the local vLLM seat)
recovers on OCR-hostile scanned pages over the compatibility cascade's
Tesseract OCR—the quality floor for that route before its council,
``structure_graph``, and Stage-6 authoring steps.

It is deliberately split into two phases so the safe half can run WHILE the
baseline conversion is still holding the GPU:

  * ``cpu``  — render each page (pypdfium2, same scale as the pipeline) and run
               Tesseract. No GPU, no endpoint. Safe to run any time.
  * ``gpu``  — send each rendered page image to the Omni endpoint for a
               faithful transcription. Contends with a live conversion for the
               single :8000 seat, so it is GATED: if an ``ed4all run`` process
               is detected it refuses unless ``--force-gpu`` is passed.

``all`` runs cpu then gpu. Re-running ``gpu`` after a ``cpu`` pass reuses the
rendered PNGs + Tesseract text on disk (idempotent), so the usual flow is:

    # now, alongside the baseline (CPU only):
    python scripts/integration/vision_ocr_probe.py --phase cpu --auto-select 5
    # after the baseline finishes (fires the Omni calls):
    python scripts/integration/vision_ocr_probe.py --phase gpu

Output tree (default ``inputs/scan-corpus/vision_probe/``):
    <out>/<pdf_stem>/page_<NNN>/page.png          rendered page
                               /tesseract.txt      raw Tesseract text
                               /tesseract_conf.json  per-page conf stats
                               /omni.md            Omni transcription (gpu phase)
                               /metrics.json       side-by-side comparison
    <out>/probe_report.json                        roll-up across pages

This is a MEASUREMENT harness, not a pipeline stage — the real judge is a
human/Claude visual read of page.png vs tesseract.txt vs omni.md; the metrics
are cheap proxies to rank pages and quantify the delta.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants / heuristics
# ---------------------------------------------------------------------------

# Render scale in pypdfium2 units (1.0 == 72 DPI). 3.0 == 216 DPI, matching the
# pipeline's SEMANTIK_OCR_RENDER_SCALE=3.0 so the Tesseract path here sees the
# SAME pixels the real conversion does.
_DEFAULT_SCALE = float(os.environ.get("SEMANTIK_OCR_RENDER_SCALE", "3.0"))
_DEFAULT_ENDPOINT = os.environ.get("SEMANTIK_SPECIALIST_BASE_URL", "http://localhost:8000/v1")
_DEFAULT_MODEL = os.environ.get("SEMANTIK_SPECIALIST_MODEL", "nemotron-3-nano-omni")
_DEFAULT_API_KEY = os.environ.get("SEMANTIK_SPECIALIST_API_KEY", "local")
_DEFAULT_PDF = "inputs/scan-corpus/pdf_in/sample-scan-ch01.pdf"
_DEFAULT_OUT = "inputs/scan-corpus/vision_probe"

# Faithful-transcription directive: we want a VERBATIM structured copy, not a
# summary. Math -> LaTeX, tables -> Markdown, reading order preserved. This is
# the analog of how an Omni-extraction lane would be prompted, so the probe is
# a real proxy for the design (see docs/architecture/hybrid-vision-extraction.md).
_TRANSCRIBE_SYSTEM = (
    "You are a faithful document transcriber for accessibility remediation. "
    "Transcribe the page image to clean Markdown EXACTLY as printed. Rules: "
    "(1) preserve ALL text verbatim — never summarize, omit, or invent; "
    "(2) render mathematics as inline/block LaTeX ($...$ / $$...$$); "
    "(3) render tabular data as GitHub-flavored Markdown tables with the real "
    "row/column structure; (4) preserve reading order and heading hierarchy "
    "(# for chapter/section titles); (5) transcribe multi-column layouts in "
    "natural reading order, column by column; (6) for a figure, emit a short "
    "[FIGURE: ...] note, do not hallucinate its contents. Output ONLY the "
    "Markdown transcription."
)
_TRANSCRIBE_USER = "Transcribe this scanned textbook page faithfully to Markdown."

# Math-ish signal: LaTeX markers + common math unicode. Used to compare how much
# math each path retains (Tesseract usually shreds it into isolated glyphs).
_MATH_MARKERS = re.compile(r"[\\$^_{}]|\\frac|\\sqrt|\\times|[×÷≤≥≠±∞√πθαβ∑∫]|\^\d|_\d")
# Replacement / mojibake proxy.
_GARBAGE = re.compile(r"[�□▪]")
# Markdown table row proxy.
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# CPU half: render + Tesseract
# ---------------------------------------------------------------------------

def render_page(pdf_path: str, page_idx: int, scale: float):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_idx]
        bitmap = page.render(scale=scale)
        return bitmap.to_pil().convert("RGB")
    finally:
        pdf.close()


def tesseract_page(img) -> tuple[str, dict[str, Any]]:
    import pytesseract
    from pytesseract import Output

    text = pytesseract.image_to_string(img)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
    words = [w for w in data.get("text", []) if w and w.strip()]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    low_conf_frac = (sum(1 for c in confs if c < 60) / len(confs)) if confs else 1.0
    return text, {
        "n_words": len(words),
        "n_conf": len(confs),
        "mean_conf": round(mean_conf, 2),
        "low_conf_frac": round(low_conf_frac, 4),
    }


def hostility_score(text: str, conf: dict[str, Any]) -> float:
    """Higher == more OCR-hostile (a better probe target).

    Blends: low mean confidence, high low-confidence-word fraction, mojibake
    density, and a 'shredding' proxy (many single-char tokens => OCR broke
    math/notation into isolated glyphs)."""
    n = max(1, len(text))
    garbage = len(_GARBAGE.findall(text)) / n
    singles = len(re.findall(r"(?:^|\s)\S(?:\s|$)", text)) / max(1, text.count(" ") + 1)
    conf_pen = (100.0 - conf["mean_conf"]) / 100.0
    return round(
        0.45 * conf_pen
        + 0.30 * conf["low_conf_frac"]
        + 0.15 * min(1.0, garbage * 50)
        + 0.10 * min(1.0, singles * 3),
        4,
    )


def auto_select(pdf_path: str, k: int, stride: int, scale: float) -> list[int]:
    """Scan every ``stride``-th page, score hostility, return the top-k indices."""
    import pypdfium2 as pdfium

    n_pages = len(pdfium.PdfDocument(pdf_path))
    candidates = list(range(0, n_pages, max(1, stride)))
    scored: list[tuple[float, int]] = []
    print(f"[auto-select] scanning {len(candidates)} of {n_pages} pages "
          f"(stride={stride}) for OCR hostility...", flush=True)
    for pi in candidates:
        img = render_page(pdf_path, pi, scale)
        text, conf = tesseract_page(img)
        s = hostility_score(text, conf)
        scored.append((s, pi))
        print(f"  page {pi+1:>3}: hostility={s:.3f} mean_conf={conf['mean_conf']:.1f} "
              f"words={conf['n_words']}", flush=True)
    scored.sort(reverse=True)
    picked = sorted(pi for _, pi in scored[:k])
    print(f"[auto-select] picked (0-based): {picked}  (1-based: {[p+1 for p in picked]})", flush=True)
    return picked


# ---------------------------------------------------------------------------
# GPU half: Omni vision transcription
# ---------------------------------------------------------------------------

def omni_transcribe(png_bytes: bytes, *, endpoint: str, model: str, api_key: str,
                    max_tokens: int, thinking: bool, timeout: float) -> dict[str, Any]:
    import requests

    b64 = base64.b64encode(png_bytes).decode("ascii")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _TRANSCRIBE_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": _TRANSCRIBE_USER},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if not thinking:
        # Faithful transcription needs no chain-of-thought; suppress it so the
        # budget goes to the copy (nemotron_v3 honors enable_thinking:false).
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    t0 = time.time()
    r = requests.post(f"{endpoint}/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    ch = data["choices"][0]
    content = ch["message"].get("content")
    usage = data.get("usage", {})
    return {
        "text": content or "",
        "is_null": content is None,
        "finish_reason": ch.get("finish_reason"),
        "wall_s": round(dt, 1),
        "completion_tokens": usage.get("completion_tokens"),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(tess: str, omni: str) -> dict[str, Any]:
    tess_math = len(_MATH_MARKERS.findall(tess))
    omni_math = len(_MATH_MARKERS.findall(omni))
    return {
        "tess_chars": len(tess),
        "omni_chars": len(omni),
        "tess_math_markers": tess_math,
        "omni_math_markers": omni_math,
        "tess_garbage": len(_GARBAGE.findall(tess)),
        "tess_md_table_rows": len(_MD_TABLE_ROW.findall(tess)),
        "omni_md_table_rows": len(_MD_TABLE_ROW.findall(omni)),
    }


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def conversion_running() -> str | None:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "ed4all run textbook-to-course"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=_DEFAULT_PDF)
    ap.add_argument("--pages", default="", help="explicit 1-based page list, e.g. 12,40,77")
    ap.add_argument("--auto-select", type=int, default=5,
                    help="pick the K most OCR-hostile pages (ignored if --pages given)")
    ap.add_argument("--sample-stride", type=int, default=12,
                    help="auto-select: score every Nth page")
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--phase", choices=["cpu", "gpu", "all"], default="all")
    ap.add_argument("--scale", type=float, default=_DEFAULT_SCALE)
    ap.add_argument("--endpoint", default=_DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--api-key", default=_DEFAULT_API_KEY)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--thinking", action="store_true", help="allow CoT on the transcription (default off)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--force-gpu", action="store_true",
                    help="run the GPU phase even if an ed4all conversion is live")
    args = ap.parse_args()

    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"ERROR: pdf not found: {pdf_path}", file=sys.stderr)
        return 2
    stem = Path(pdf_path).stem
    out_root = Path(args.out) / stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Page selection (0-based internally).
    if args.pages.strip():
        pages = sorted({int(p) - 1 for p in args.pages.split(",") if p.strip()})
    elif (args.phase in ("cpu", "all")):
        pages = auto_select(pdf_path, args.auto_select, args.sample_stride, args.scale)
    else:
        # gpu-only phase: reuse whatever page_* dirs already exist on disk.
        pages = sorted(int(d.name.split("_")[1]) - 1
                       for d in out_root.glob("page_*") if d.is_dir())
        if not pages:
            print("ERROR: gpu phase found no rendered pages — run --phase cpu first.",
                  file=sys.stderr)
            return 2

    do_cpu = args.phase in ("cpu", "all")
    do_gpu = args.phase in ("gpu", "all")

    if do_gpu:
        live = conversion_running()
        if live and not args.force_gpu:
            print("\n[guard] an ed4all conversion is LIVE — skipping the GPU phase to "
                  "avoid contending for the :8000 seat.\n"
                  f"        {live.splitlines()[0]}\n"
                  "        Re-run with --phase gpu after it finishes (or pass --force-gpu).")
            do_gpu = False

    report: dict[str, Any] = {"pdf": pdf_path, "scale": args.scale,
                              "model": args.model, "pages": []}

    for pi in pages:
        pdir = out_root / f"page_{pi+1:03d}"
        pdir.mkdir(parents=True, exist_ok=True)
        png_path = pdir / "page.png"
        tess_path = pdir / "tesseract.txt"
        conf_path = pdir / "tesseract_conf.json"
        omni_path = pdir / "omni.md"
        entry: dict[str, Any] = {"page": pi + 1}

        # -- CPU --
        if do_cpu or not png_path.exists():
            img = render_page(pdf_path, pi, args.scale)
            img.save(png_path, "PNG")
            tess, conf = tesseract_page(img)
            tess_path.write_text(tess, encoding="utf-8")
            conf_path.write_text(json.dumps(conf, indent=2), encoding="utf-8")
            entry.update({"tesseract_conf": conf,
                          "hostility": hostility_score(tess, conf)})
            print(f"[cpu ] page {pi+1:>3}: mean_conf={conf['mean_conf']:.1f} "
                  f"words={conf['n_words']} -> {tess_path}", flush=True)
        elif conf_path.exists():
            entry["tesseract_conf"] = json.loads(conf_path.read_text())

        # -- GPU --
        if do_gpu:
            png_bytes = png_path.read_bytes()
            try:
                res = omni_transcribe(
                    png_bytes, endpoint=args.endpoint, model=args.model,
                    api_key=args.api_key, max_tokens=args.max_tokens,
                    thinking=args.thinking, timeout=args.timeout)
                omni_path.write_text(res["text"], encoding="utf-8")
                tess = tess_path.read_text(encoding="utf-8") if tess_path.exists() else ""
                entry["omni"] = {k: v for k, v in res.items() if k != "text"}
                entry["compare"] = compare(tess, res["text"])
                print(f"[gpu ] page {pi+1:>3}: finish={res['finish_reason']} "
                      f"null={res['is_null']} wall={res['wall_s']}s "
                      f"comp_tok={res['completion_tokens']} -> {omni_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                entry["omni_error"] = str(exc)
                print(f"[gpu ] page {pi+1:>3}: ERROR {exc}", flush=True)

        report["pages"].append(entry)

    report_path = Path(args.out) / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== probe report -> {report_path} ===")
    for e in report["pages"]:
        cmp = e.get("compare")
        if cmp:
            print(f"  page {e['page']:>3}: math OCR={cmp['tess_math_markers']:>4} "
                  f"vision={cmp['omni_math_markers']:>4} | table_rows OCR="
                  f"{cmp['tess_md_table_rows']} vision={cmp['omni_md_table_rows']} | "
                  f"chars OCR={cmp['tess_chars']} vision={cmp['omni_chars']}")
        else:
            h = e.get("hostility")
            print(f"  page {e['page']:>3}: hostility={h} (GPU phase pending)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
