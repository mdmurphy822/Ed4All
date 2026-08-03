"""Evaluate the Stage 6b SmolVLM2 figure-captioning contract.

The component under evaluation is
``semantik_structure.figures.captioner.caption_figure_regions`` /
``_run_smolvlm_caption`` (Stage 6b), feeding the assembler figure emitter
``semantik_structure.assembler.fallbacks.fallback_figure`` (which now emits
``alt`` + ``aria-describedby``). The eval data is
``data/figure_alt_dataset/test.jsonl`` (figure image path + ground-truth
``caption``, joined to the rendered PNG under ``data/figure_images/``).

This harness scores four acceptance gates. It is not a
similarity-to-gold harness (there is no "gold alt"; the caption is the
ground truth a good alt must derive from):

  1. **axe pass / regression.** Render each sampled figure's HTML fragment
     via the assembler figure emitter and run the per-region wcag22aa axe
     gate (``semantik_structure.validate.HtmlValidator`` — the SAME gate the
     table and gap_fill evals use, and the one behind
     ``tests/test_hard_region_gate.py``). ``axe_pass`` is the fragment's
     pass rate; ``axe_regression`` is "an EMPTY-alt baseline passed but the
     generated-alt fragment broke it" — a defect the captioner introduced.
     (An empty-alt figure with a ``<figcaption>`` already passes axe, so the
     baseline is meaningful; alt is an SC 1.1.1 quality lever, not the only
     axe trigger.)
  2. **decorative -> alt="".** Figures the routing flags decorative MUST get
     ``alt=""`` and NO generated caption. We gate on the ROUTING decision
     (``route_figure``), not the model — a decorative figure must never reach
     the captioner. ``decorative_correct`` is 1.0 only when every decorative
     row emitted empty alt and no model caption.
  3. **caption-first precision.** When a ground-truth ``caption`` exists, the
     emitted alt must DERIVE from it, not be overridden by a model
     hallucination. ``caption_first`` is true when the alt is caption-derived
     (the routing chose the caption, or the model output is a substring-/
     token-subset of the caption).
  4. **no-hallucination spot-check.** Rule-based: every NUMERIC token in the
     generated description must also appear in the caption / ground-truth
     (``surrounding_context`` counts as ground truth too). ``no_hallucination``
     is false for any row whose generated text invents a number absent from
     the source; such rows are listed in ``suspect_rows`` for manual review.

Stratification: by ``source`` (arxiv / pmc / openstax) and by ``subtype``
when the row carries one (the current dataset does not, so the subtype
breakdown collapses to a single ``unknown`` stratum — kept so a future
subtype-tagged build lights it up without a code change).

Model seam (so this runs WITHOUT a GPU):
  The captioner call is injected as ``caption_fn(image_bytes, *, kind) ->
  CaptionResult``. The default real-run seam loads SmolVLM2 and calls
  ``figures.captioner._run_smolvlm_caption``. ``--self-test`` swaps in a
  canned stub over synthetic rows so the gate logic is exercisable on CPU
  with no model and no network. A captioner FAILURE is surfaced as
  ``FigureCaptionError``, never silently skipped — a failed row is recorded
  with ``caption_error`` and fails its
  gates rather than vanishing.

The alt and extended token caps mirror the production Stage 6b values
(``_ALT_MAX_NEW_TOKENS`` = 64,
``_EXTENDED_MAX_NEW_TOKENS`` = 256). The dataset caption tail is p99=337,
max=741 tokens, so the 256-token extended cap will truncate the longest
captions — but the EXTENDED description is the model's own summary, not a
reproduction of the caption, so the cap bounds verbosity rather than
truncating required content. Quoted here so a future raise is deliberate.

Usage::

    # REAL run on the GPU (loads SmolVLM2-256M):
    .venv/bin/python -m scripts.eval.eval_figure_captioner --tag v1
    .venv/bin/python -m scripts.eval.eval_figure_captioner --tag v1 --n 300
    .venv/bin/python -m scripts.eval.eval_figure_captioner --tag v1 --no-axe

    # CPU self-test (stub captioner, synthetic rows, no GPU/model/network):
    .venv/bin/python -m scripts.eval.eval_figure_captioner --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Use the production numeric guard so evaluation measures emitted behavior.
from semantik_structure.figures.captioner import (  # noqa: E402
    alt_from_caption,
    guard_figure_alt,
    strip_numeric_hallucinations,
)

# Use the production caption trim so the evaluator and emitter cannot drift.
_alt_from_caption = alt_from_caption

DEFAULT_TEST = REPO_ROOT / "data/figure_alt_dataset/test.jsonl"
DEFAULT_IMAGE_ROOT = REPO_ROOT  # image_path in rows is repo-relative
DEFAULT_REPORT_DIR = REPO_ROOT / "data/eval_reports"

# Match the production Stage-6b output limits.
ALT_MAX_NEW_TOKENS = 64
EXTENDED_MAX_NEW_TOKENS = 256

# Require every acceptance threshold for a ship verdict.
SHIP_THRESHOLDS = {
    "axe_pass": 0.98,            # fragments must be accessible
    "axe_regression": 0.02,     # <= : captioner must not introduce defects
    "decorative_correct": 1.0,  # decorative routing must be exact
    "caption_first": 0.95,      # alt derives from caption when one exists
    "no_hallucination": 0.95,   # generated text invents no numbers
}

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class CaptionResult:
    """What a captioner returns for one figure (or what routing decided)."""
    alt: str
    extended: str = ""
    source: str = "model"   # "model" | "caption" | "decorative" | "error"
    error: str | None = None


# Model seams accept image bytes and a figure kind, then return both descriptions.
CaptionFn = Callable[..., CaptionResult]


def make_smolvlm_caption_fn(run_extended: bool = True) -> CaptionFn:
    """Real-run seam: wrap ``figures.captioner._run_smolvlm_caption`` (loads
    SmolVLM2-256M on first call). A generation failure raises
    ``FigureCaptionError``, which the caller records as a failed row rather
    than swallowing."""
    from semantik_structure.figures.captioner import (  # local import: avoids torch at import time
        _ALT_PROMPT,
        _EXTENDED_PROMPT,
        FigureCaptionError,
        _run_smolvlm_caption,
    )

    def _fn(image_bytes: bytes, *, kind: str = "figure") -> CaptionResult:
        try:
            alt = _run_smolvlm_caption(
                image_bytes, _ALT_PROMPT, max_new_tokens=ALT_MAX_NEW_TOKENS,
            )
            ext = (
                _run_smolvlm_caption(
                    image_bytes, _EXTENDED_PROMPT,
                    max_new_tokens=EXTENDED_MAX_NEW_TOKENS,
                )
                if run_extended else ""
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't swallow
            raise FigureCaptionError(
                f"smolvlm caption failed: {type(exc).__name__}: {exc}"
            ) from exc
        return CaptionResult(alt=alt, extended=ext, source="model")

    return _fn


def is_decorative(row: dict) -> bool:
    """Decorative-figure routing decision (gate on this, not the model).

    A figure is decorative when the row says so explicitly (``decorative``
    flag, present in a future subtype-tagged build) OR when there is no
    caption AND no meaningful ``alt_raw`` — a figure carrying no textual
    intent. Such a figure must receive ``alt=""`` and never be captioned.
    """
    if row.get("decorative") is True:
        return True
    cap = (row.get("caption") or "").strip()
    alt_raw = (row.get("alt_raw") or "").strip().lower()
    has_alt_raw = bool(alt_raw) and alt_raw != "[uncaptioned image]"
    return not cap and not has_alt_raw


def route_figure(row: dict, caption_fn: CaptionFn, image_bytes: bytes) -> CaptionResult:
    """Stage 6b routing as the harness exercises it.

    Routing rules (the decisions the gates score):
      * decorative      -> alt="" , no model call.
      * caption present -> alt derives from the caption (caption-first); the
        model produces only the EXTENDED description, never the alt.
      * no caption      -> the model produces the alt.

    The model seam is only invoked for the extended description (caption
    present) or the alt (no caption) — a decorative figure never reaches it.
    """
    if is_decorative(row):
        return CaptionResult(alt="", extended="", source="decorative")

    cap = (row.get("caption") or "").strip()
    if cap:
        # Use the source caption for alt text and guard the model's longer
        # description against unsupported numeric claims.
        res = caption_fn(image_bytes, kind="figure")
        return CaptionResult(
            alt=_alt_from_caption(cap),
            extended=strip_numeric_hallucinations(res.extended or "", cap),
            source="caption",
            error=res.error,
        )
    # Without a source caption, strip unverifiable numeric claims from both
    # model-generated descriptions.
    res = caption_fn(image_bytes, kind="figure")
    return CaptionResult(
        alt=guard_figure_alt(res.alt, cap),
        extended=strip_numeric_hallucinations(res.extended, cap),
        source="model",
        error=res.error,
    )


# ---------------------------------------------------------------------------
# Assembler figure emitter — render the HTML fragment we gate on
# ---------------------------------------------------------------------------
def emit_figure_html(result: CaptionResult, row: dict, region_idx: int) -> str:
    """Build a figure Region from the caption result + ground-truth caption
    and run it through the REAL assembler emitter (``fallback_figure``).

    Mirrors what the cascade produces: caption text comes from the row's
    ground-truth ``caption`` (the ``<figcaption>`` source), alt /
    extended_description from the captioner result.
    """
    from semantik_structure.assembler.fallbacks import fallback_figure
    from semantik_structure.structure_graph import Region
    from semantik_structure.types import FeatureBlock, RawBlock

    cap = (row.get("caption") or "").strip()
    feature_blocks: list[FeatureBlock] = []
    payload: dict[str, Any] = {
        "alt_text": result.alt,
        "extended_description": result.extended,
    }
    if cap:
        raw = RawBlock(
            text=cap, page=1, bbox=(0.0, 0.0, 1.0, 1.0),
            page_width=1.0, page_height=1.0,
        )
        fb = FeatureBlock(
            raw=raw,
            size_bucket="md",
            gap_above=None,
            is_top_of_page=False,
            is_centered=False,
            caps=None,
            indent_bucket=0,
            relative_font_ratio=1.0,
        )
        feature_blocks = [fb]
        payload["caption_fb_index"] = 0
    region = Region(
        kind="figure",
        feature_block_indices=(0,) if feature_blocks else (region_idx,),
        payload=payload,
    )
    return fallback_figure(region, feature_blocks)


# ---------------------------------------------------------------------------
# Scoring the four acceptance gates
# ---------------------------------------------------------------------------
def _numeric_tokens(text: str) -> set[str]:
    """Numeric tokens, comma/period normalized (1,000 == 1000; 3.14 kept)."""
    out: set[str] = set()
    for m in _NUM_RE.findall(text or ""):
        out.add(m.replace(",", ""))
    return out


def _word_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def check_no_hallucination(generated: str, ground_truth: str) -> tuple[bool, list[str]]:
    """Every numeric token in ``generated`` must appear in ``ground_truth``.

    Returns (ok, invented_numbers). A generated number absent from the
    caption/context is a numeric hallucination — flag the row for review.
    """
    gen_nums = _numeric_tokens(generated)
    if not gen_nums:
        return True, []
    src_nums = _numeric_tokens(ground_truth)
    invented = sorted(gen_nums - src_nums)
    return (not invented), invented


def check_caption_first(result: CaptionResult, caption: str) -> bool:
    """When a caption exists, the alt must derive from it.

    True when routing chose the caption source (``source == "caption"``) OR
    the alt's word tokens are a subset of the caption's (the model did not
    inject content absent from the caption). When no caption exists, this
    gate does not apply (returns True — scored conditionally upstream)."""
    if not caption.strip():
        return True
    if result.source in ("caption", "decorative"):
        return True
    alt_toks = _word_tokens(result.alt)
    cap_toks = _word_tokens(caption)
    if not alt_toks:
        return True
    # allow a small stopword slack: alt may add articles/punctuation only.
    extra = alt_toks - cap_toks - _STOPWORDS
    return not extra


_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with",
    "is", "are", "this", "that", "shows", "showing", "image", "figure",
}


def score_one(row: dict, result: CaptionResult) -> dict:
    cap = (row.get("caption") or "").strip()
    ctx = (row.get("surrounding_context") or "").strip()
    alt_raw = (row.get("alt_raw") or "").strip()
    ground_truth = " ".join(p for p in (cap, ctx, alt_raw) if p)

    decorative = is_decorative(row)
    has_caption = bool(cap)

    # gate 2: decorative -> empty alt, no model caption.
    decorative_correct: bool | None = None
    if decorative:
        decorative_correct = (
            result.source == "decorative"
            and result.alt == ""
            and result.extended == ""
        )

    # gate 3: caption-first (conditional on a caption existing).
    caption_first: bool | None = None
    if has_caption:
        caption_first = check_caption_first(result, cap)

    # gate 4: no numeric hallucination across alt + extended.
    generated = " ".join(p for p in (result.alt, result.extended) if p)
    no_halluc, invented = check_no_hallucination(generated, ground_truth)

    return {
        "decorative": decorative,
        "has_caption": has_caption,
        "caption_source": result.source,
        "alt": result.alt,
        "extended": result.extended,
        "caption_error": result.error,
        "alt_empty": result.alt == "",
        "decorative_correct": decorative_correct,
        "caption_first": caption_first,
        "no_hallucination": no_halluc,
        "invented_numbers": invented,
    }


def _cond_rate(per_row: list[dict], key: str) -> tuple[float | None, int]:
    """Rate of a tri-state gate (True/False/None) over rows where it applies."""
    rel = [r for r in per_row if r["scores"][key] is not None]
    if not rel:
        return None, 0
    return sum(1 for r in rel if r["scores"][key]) / len(rel), len(rel)


def aggregate(per_row: list[dict]) -> dict:
    n = len(per_row)
    if n == 0:
        return {}
    agg: dict[str, Any] = {"n": n}
    agg["no_hallucination"] = sum(
        1 for r in per_row if r["scores"]["no_hallucination"]) / n
    agg["n_decorative"] = sum(1 for r in per_row if r["scores"]["decorative"])
    agg["n_has_caption"] = sum(1 for r in per_row if r["scores"]["has_caption"])
    agg["n_caption_error"] = sum(
        1 for r in per_row if r["scores"]["caption_error"])
    for key in ("decorative_correct", "caption_first"):
        rate, denom = _cond_rate(per_row, key)
        agg[key] = rate
        agg[f"n_{key}"] = denom
    if per_row and "axe_pass" in per_row[0]["scores"]:
        agg["axe_pass"] = sum(
            1 for r in per_row if r["scores"]["axe_pass"]) / n
        agg["axe_regression"] = sum(
            1 for r in per_row if r["scores"]["axe_regression"]) / n
    return agg


def _breakdown(per_row: list[dict], key: str) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by[r[key]].append(r)
    return {
        k: aggregate(rows)
        for k, rows in sorted(by.items(), key=lambda kv: -len(kv[1]))
    }


# ---------------------------------------------------------------------------
# axe gate (reuse the production wcag22aa HtmlValidator — same as table eval)
# ---------------------------------------------------------------------------
def _wrap_doc(fragment: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>fragment</title></head><body>{fragment}</body></html>"
    )


def run_axe(fragments: list[str]) -> list[dict]:
    from semantik_structure.validate import HtmlValidator

    results: list[dict] = []
    with HtmlValidator() as v:
        for frag in fragments:
            res = v.check(_wrap_doc(frag))
            results.append({
                "ok": res.ok,
                "error": res.error,
                "blocking": sorted({
                    f"{vi.impact}:{vi.id}" for vi in res.violations
                    if vi.impact in {"serious", "critical"}
                }),
            })
    return results


def baseline_fragment(row: dict, region_idx: int) -> str:
    """Empty-alt baseline: the SAME figure with alt="" and no extended desc
    so ``axe_regression`` detects defects introduced by generated text."""
    return emit_figure_html(
        CaptionResult(alt="", extended="", source="decorative"),
        row, region_idx,
    )


def attach_axe(per_row: list[dict], gen_fragments: list[str],
               base_fragments: list[str]) -> dict:
    base_axe = run_axe(base_fragments)
    gen_axe = run_axe(gen_fragments)
    n = len(per_row)
    base_pass = sum(1 for r in base_axe if r["ok"])
    for row, ba, na in zip(per_row, base_axe, gen_axe):
        row["scores"]["axe_pass"] = na["ok"]
        row["scores"]["axe_regression"] = bool(ba["ok"] and not na["ok"])
        row["axe_blocking"] = na["blocking"]
        row["baseline_axe_blocking"] = ba["blocking"]
    return {
        "baseline_pass": base_pass / n if n else 0.0,
        "baseline_blocking_rules": dict(
            Counter(r for a in base_axe for r in a["blocking"]).most_common()),
        "gen_blocking_rules": dict(
            Counter(r for a in gen_axe for r in a["blocking"]).most_common()),
    }


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_id(r: dict) -> tuple:
    return (r.get("source") or "", str(r.get("doc_id") or ""),
            int(r.get("figure_index", 0)))


def _interleave_by_source(rows: list[dict]) -> list[dict]:
    from collections import deque
    buckets: dict[str, deque] = defaultdict(deque)
    for r in sorted(rows, key=_row_id):
        buckets[r.get("source") or ""].append(r)
    order = sorted(buckets)
    out: list[dict] = []
    while any(buckets[s] for s in order):
        for s in order:
            if buckets[s]:
                out.append(buckets[s].popleft())
    return out


def select_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """All rows when n<=0 or n>=len; else a per-source-balanced cut so the
    minority sources (openstax) don't vanish. Final order interleaves
    sources so a sequential run sees a representative mix early."""
    if n <= 0 or n >= len(rows):
        return _interleave_by_source(rows)
    import random
    rng = random.Random(seed)
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r.get("source") or ""].append(r)
    per = max(1, n // max(1, len(by_src)))
    picked: list[dict] = []
    for _, srows in by_src.items():
        srows = list(srows)
        rng.shuffle(srows)
        picked.extend(srows[:per])
    rng.shuffle(picked)
    picked = picked[:n] if n < len(picked) else picked
    return _interleave_by_source(picked)


def _subtype(row: dict) -> str:
    return row.get("subtype") or row.get("figure_type") or "unknown"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_eval(
    sample: list[dict],
    caption_fn: CaptionFn,
    *,
    image_root: Path,
    run_axe_pass: bool,
    load_image_bytes: Callable[[dict], bytes] | None = None,
    on_row: Callable[[int, dict, CaptionResult], None] | None = None,
) -> tuple[list[dict], list[str], list[str], dict | None]:
    """Route + caption every row, returning (per_row, gen_fragments,
    base_fragments, axe_summary). ``axe_summary`` is None when
    ``run_axe_pass`` is False. ``load_image_bytes`` is injectable for the
    self-test (synthetic rows carry inline bytes instead of a file path)."""
    from semantik_structure.figures.captioner import FigureCaptionError

    per_row: list[dict] = []
    gen_fragments: list[str] = []
    base_fragments: list[str] = []
    for i, row in enumerate(sample):
        decorative = is_decorative(row)
        # Only load image bytes if the model will actually be called.
        img_bytes = b""
        if not decorative:
            if load_image_bytes is not None:
                img_bytes = load_image_bytes(row)
            else:
                img_path = image_root / row["image_path"]
                if not img_path.exists():
                    raise FileNotFoundError(
                        f"row {i}: image not found: {img_path} "
                        f"(doc_id={row.get('doc_id')})"
                    )
                img_bytes = img_path.read_bytes()
        try:
            result = route_figure(row, caption_fn, img_bytes)
        except FigureCaptionError as exc:
            # Record the failure and fail the row's gates while keeping the row
            # visible in the evaluation report.
            result = CaptionResult(
                alt="", extended="", source="error", error=str(exc))
        scores = score_one(row, result)
        per_row.append({
            "source": row.get("source", "?"),
            "subtype": _subtype(row),
            "doc_id": row.get("doc_id"),
            "figure_index": row.get("figure_index"),
            "scores": scores,
        })
        gen_fragments.append(emit_figure_html(result, row, i))
        base_fragments.append(baseline_fragment(row, i))
        if on_row is not None:
            on_row(i, row, result)
    axe_summary = None
    if run_axe_pass:
        axe_summary = attach_axe(per_row, gen_fragments, base_fragments)
    return per_row, gen_fragments, base_fragments, axe_summary


def build_report(
    tag: str, per_row: list[dict], *, run_axe_pass: bool,
    extra: dict | None = None,
) -> dict:
    overall = aggregate(per_row)
    report = {
        "tag": tag,
        "overall": overall,
        "by_source": _breakdown(per_row, "source"),
        "by_subtype": _breakdown(per_row, "subtype"),
        "verdict": _verdict(overall, run_axe_pass),
        "suspect_rows": [
            {
                "doc_id": r["doc_id"],
                "figure_index": r["figure_index"],
                "source": r["source"],
                "invented_numbers": r["scores"]["invented_numbers"],
                "alt": r["scores"]["alt"][:200],
                "extended": (r["scores"]["extended"] or "")[:200],
            }
            for r in per_row if not r["scores"]["no_hallucination"]
        ],
        "error_rows": [
            {
                "doc_id": r["doc_id"],
                "figure_index": r["figure_index"],
                "caption_error": r["scores"]["caption_error"],
            }
            for r in per_row if r["scores"]["caption_error"]
        ],
    }
    if extra:
        report.update(extra)
    return report


def _verdict(overall: dict, run_axe_pass: bool) -> dict:
    """Top-line ship/iterate verdict: every applicable gate must clear its
    threshold. A None (gate didn't apply to any row) does not block."""
    checks: dict[str, dict] = {}
    ship = True
    for key, thresh in SHIP_THRESHOLDS.items():
        if key in ("axe_pass", "axe_regression") and not run_axe_pass:
            continue
        val = overall.get(key)
        if val is None:
            checks[key] = {"value": None, "threshold": thresh,
                           "pass": None, "applicable": False}
            continue
        # axe_regression is a max (lower is better); the rest are minimums.
        if key == "axe_regression":
            ok = val <= thresh
        else:
            ok = val >= thresh
        checks[key] = {"value": round(val, 4), "threshold": thresh,
                       "pass": ok, "applicable": True}
        ship = ship and ok
    return {"ship": ship, "decision": "ship" if ship else "iterate",
            "checks": checks}


def _tiny_png_bytes() -> bytes:
    """A 2x2 white PNG — valid bytes the stub never actually decodes."""
    import base64
    # 1x1 transparent PNG.
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        b"+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def _synthetic_rows() -> list[dict]:
    """Three synthetic figure rows covering each gate's positive + negative.

    Row A: has caption, model stays caption-derived (caption-first PASS,
           no-hallucination PASS).
    Row B: decorative (no caption, no alt_raw) -> must route to empty alt.
    Row C: no caption, model invents a number absent from context
           (no-hallucination FAIL -> suspect row).
    """
    return [
        {
            "source": "arxiv", "doc_id": "SYNTH-A", "figure_index": 1,
            "caption": "Figure 1: Energy spectrum of the detector at 5 GeV.",
            "alt_raw": "", "surrounding_context": "The peak is at 5 GeV.",
            "image_path": "_synthetic/A.png",
        },
        {
            "source": "pmc", "doc_id": "SYNTH-B", "figure_index": 2,
            "caption": "", "alt_raw": "",
            "surrounding_context": "",
            "image_path": "_synthetic/B.png",
        },
        {
            # No caption, but a meaningful alt_raw -> NOT decorative, so the
            # model owns the alt and can hallucinate a number.
            "source": "openstax", "doc_id": "SYNTH-C", "figure_index": 3,
            "caption": "", "alt_raw": "Survey responses bar chart",
            "surrounding_context": "A bar chart of survey responses.",
            "image_path": "_synthetic/C.png",
        },
    ]


def _stub_caption_fn() -> CaptionFn:
    """Canned captioner. Returns a different alt/extended per synthetic row
    keyed by image bytes length is not reliable, so we route on a sentinel
    in the bytes: the self-test passes distinct bytes per row."""
    def _fn(image_bytes: bytes, *, kind: str = "figure") -> CaptionResult:
        marker = image_bytes[:8]
        if marker == b"ROW_A___":
            # caption present -> only extended is used; keep it clean.
            return CaptionResult(
                alt="ignored — caption-first overrides",
                extended="The detector's energy spectrum peaks at 5 GeV.",
                source="model",
            )
        if marker == b"ROW_C___":
            # no caption -> model owns alt; invent a number not in context.
            return CaptionResult(
                alt="A bar chart showing 42 percent agreement.",
                extended="Responses totaled 1234 participants.",
                source="model",
            )
        return CaptionResult(alt="generic alt", extended="", source="model")
    return _fn


def _self_test_load_bytes(row: dict) -> bytes:
    """Map synthetic rows to marked bytes the stub routes on."""
    base = _tiny_png_bytes()
    doc = row["doc_id"]
    marker = {"SYNTH-A": b"ROW_A___", "SYNTH-C": b"ROW_C___"}.get(doc, b"ROW_X___")
    return marker + base


def self_test(run_axe_pass: bool = False) -> dict:
    """Run the full gate logic over synthetic rows with the stub captioner.
    Returns the report dict (no file written). Used by the pytest."""
    sample = _synthetic_rows()
    per_row, _, _, axe_summary = run_eval(
        sample, _stub_caption_fn(),
        image_root=DEFAULT_IMAGE_ROOT,
        run_axe_pass=run_axe_pass,
        load_image_bytes=_self_test_load_bytes,
    )
    extra: dict = {"n_sampled": len(sample), "self_test": True}
    if axe_summary is not None:
        extra["axe_baseline"] = axe_summary
    return build_report("self_test", per_row, run_axe_pass=run_axe_pass,
                        extra=extra)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


def _print_report(report: dict) -> None:
    o = report["overall"]
    v = report["verdict"]
    print("\n" + "=" * 70)
    print(f"  Figure captioner eval — {report['tag']}  (n={o['n']})")
    print("=" * 70)
    print(f"  decorative -> empty alt : {_fmt(o.get('decorative_correct'))}"
          f"  (n={o.get('n_decorative_correct')})  <- gate 2")
    print(f"  caption-first precision : {_fmt(o.get('caption_first'))}"
          f"  (n={o.get('n_caption_first')})  <- gate 3")
    print(f"  no numeric hallucination: {_fmt(o.get('no_hallucination'))}"
          f"  (suspects={len(report['suspect_rows'])})  <- gate 4")
    if "axe_pass" in o:
        bp = report.get("axe_baseline", {}).get("baseline_pass")
        print(f"  axe wcag22aa pass       : {_fmt(o['axe_pass'])}"
              f"  (empty-alt baseline {_fmt(bp)})  <- gate 1")
        print(f"  axe regression vs base  : {_fmt(o['axe_regression'])}"
              f"  <- a11y defects introduced")
    if report.get("error_rows"):
        print(f"  captioner errors        : {len(report['error_rows'])} row(s) "
              f"(surfaced; gates failed)")
    print(f"\n  VERDICT: {v['decision'].upper()}"
          f"  (ship={v['ship']})")
    for k, c in v["checks"].items():
        if not c.get("applicable", True):
            continue
        flag = "ok " if c["pass"] else "XX "
        print(f"    [{flag}] {k:18} {c['value']}  (thresh {c['threshold']})")
    print("  by source:")
    for k, agg in report["by_source"].items():
        print(f"    {k:10} n={agg['n']:<4} "
              f"dec={_fmt(agg.get('decorative_correct'))} "
              f"cap_first={_fmt(agg.get('caption_first'))} "
              f"no_halluc={_fmt(agg.get('no_hallucination'))} "
              f"axe={_fmt(agg.get('axe_pass'))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v1",
                    help="report tag -> figure_captioner_<tag>.json")
    ap.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT,
                    help="root for repo-relative image_path values")
    ap.add_argument("--n", type=int, default=0,
                    help="sample size; 0 = all test rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-extended", action="store_true",
                    help="skip the extended-description generation pass")
    ap.add_argument("--no-axe", action="store_true",
                    help="skip the (slow, Chromium-backed) axe-core pass")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--self-test", action="store_true",
                    help="run the stub captioner over synthetic rows (no GPU, "
                         "no model, no network) and print the report")
    args = ap.parse_args()

    if args.self_test:
        report = self_test(run_axe_pass=not args.no_axe)
        _print_report(report)
        print("\n[self-test] gate logic exercised with stub captioner.")
        return 0

    if not args.test_path.exists():
        sys.exit(f"[fatal] test split not found: {args.test_path}")

    rows = _load_jsonl(args.test_path)
    sample = select_sample(rows, args.n, args.seed)
    src_dist = Counter(r.get("source") for r in sample)
    print(f"[sample] {len(sample)} rows  by source={dict(src_dist)}")
    run_axe_pass = not args.no_axe

    caption_fn = make_smolvlm_caption_fn(run_extended=not args.no_extended)

    t0 = time.time()
    per_row, _, _, axe_baseline = run_eval(
        sample, caption_fn,
        image_root=args.image_root,
        run_axe_pass=run_axe_pass,
        on_row=lambda i, r, res: (
            print(f"[caption] {i + 1}/{len(sample)}", flush=True)
            if (i + 1) % 50 == 0 else None
        ),
    )
    wall = time.time() - t0

    report = build_report(
        args.tag, per_row, run_axe_pass=run_axe_pass,
        extra={
            "test_path": str(args.test_path),
            "n_sampled": len(sample),
            "seed": args.seed,
            "alt_max_new_tokens": ALT_MAX_NEW_TOKENS,
            "extended_max_new_tokens": (
                0 if args.no_extended else EXTENDED_MAX_NEW_TOKENS),
            "wall_seconds": round(wall, 1),
            "axe_enabled": run_axe_pass,
            "source_dist": dict(src_dist),
        },
    )
    if axe_baseline is not None:
        report["axe_baseline"] = axe_baseline

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", args.tag)
    out_path = args.out_dir / f"figure_captioner_{safe}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    samples_path = out_path.with_suffix(".samples.jsonl")
    with samples_path.open("w", encoding="utf-8") as f:
        for r in per_row:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _print_report(report)
    print(f"\n  report : {out_path}")
    print(f"  samples: {samples_path}")
    print(f"\n[done] wall {wall:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
