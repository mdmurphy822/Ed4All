"""Quantify the SEMANTIK_COLUMN_ORDER fix's reading-order credit.

Reuses ``data/build_structure_data.py``'s render + extract + align pipeline
per realistic-eval doc, then sorts the SAME extracted blocks TWO ways
(``raster`` = the legacy ``y0, x0`` key the flag-off build uses; ``column`` =
the committed column-major ``column_index, y0, x0`` key the flag-on cascade
uses), aligns EACH ordering to the gold HTML, and scores both with
``eval/reading_order_metric.py`` vs the gold DOM reading order.

Rendering each doc exactly ONCE and re-sorting the extracted blocks (rather
than re-rendering per ordering) keeps the two orderings comparing the IDENTICAL
extracted block set — the only thing that differs is the feed order, which is
the variable under test. Column attribution (which physical column a block sits
in, from its left-edge x0) is computed ONCE per page and shared by both
orderings, so a "cross-column inversion" means the same blocks-in-different-
columns pair fed in the wrong order.

By default it reads the EXACT docs/profiles/seeds of an existing realistic-eval
build (``--from-dataset data/structure_dataset_realistic_on``, its
``realistic_docs.jsonl``) so the measurement lands on the same 30-doc set the
realistic eval already uses — no re-derivation of which docs / which profiles.

Output: ``eval_reports/reading_order_credit.json`` — aggregate raster-vs-column
reading-order quality (Kendall tau-b / pairwise accuracy / inversions /
cross-column inversions), split by render-profile family (multi-column vs
single-column) so the fix's credit (a lift on multi-column docs, ~no change on
single-column) is isolated, plus per-doc rows.

CPU-only. Network: none (local pair files + Playwright PDF render, as the
build). Run from the SemantiK repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Ensure the repo root is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from semantik_structure.extract_shared import extract_shared  # noqa: E402
from semantik_structure.reading_order import column_ids_for_bboxes  # noqa: E402
from semantik_structure.validate import HtmlValidator  # noqa: E402
from data import structure_align  # noqa: E402
from data.build_structure_data import extract_html_blocks  # noqa: E402
from data.render_augment import augment_html  # noqa: E402
from data.structure_align import (  # noqa: E402
    FBView,
    GoldView,
    LedgerIncompleteError,
    align_blocks,
    sim,
)
from eval.reading_order_metric import (  # noqa: E402
    aggregate_results,
    reading_order_metric,
)

# Mirror the realistic build's frozen aligner knobs (coverage_report.json).
GLOBAL_SOFT_FLOOR = 0.30
GLOBAL_MAX_RUN = 6


def _is_multi_column(profile: str) -> bool:
    """A profile is multi-column iff it lays out >1 text column — the only
    family where a raster sort interleaves columns and the column-order fix can
    help. The realistic profiles are ``two_column`` / ``two_column+serif_embed``
    (multi) vs ``serif_embed`` / ``dense_layout`` / ``clean`` (single)."""
    return "two_column" in (profile or "")


def _page_items(shared: dict):
    """Yield ``(page_no, items)`` where items is a list of dicts carrying the
    text-bearing merged block, its bbox, and its per-page column index.

    Column attribution is computed ONCE per page over the text-bearing blocks
    (the property under both orderings); the fed ORDER is what the two sort
    keys vary. Empty-text blocks are skipped, mirroring the build's views."""
    for page_no, page in enumerate(shared.get("pages", [])):
        merged = page.get("merged", {}).get("text_blocks", []) or []
        page_w = float(page.get("width", 612.0))
        blocks = [b for b in merged if (b.get("text") or "").strip()]
        if not blocks:
            continue
        bboxes = [b.get("bbox") or (0.0, 0.0, 0.0, 0.0) for b in blocks]
        col_ids = column_ids_for_bboxes(bboxes, page_w)
        items = [
            {"block": b, "bbox": b.get("bbox") or (0.0, 0.0, 0.0, 0.0), "col": int(col_ids[i])}
            for i, b in enumerate(blocks)
        ]
        yield page_no, items


def _ordered_fbviews(shared: dict, *, mode: str):
    """Build the fed block sequence (FBViews + per-block global column id) for
    one ordering mode over the already-extracted ``shared``.

    ``raster`` -> sort each page by ``(y0, x0)`` (legacy / flag-off).
    ``column`` -> sort each page by ``(col, y0, x0)`` (the committed flag-on
    column-major key). Pages are concatenated in page order in BOTH modes
    (only WITHIN-page order varies), so the aligner's page-boundary
    segmentation is unaffected. Returns ``(views, col_global_per_fed_pos)``."""
    views: list[FBView] = []
    col_global: list[int] = []
    order_index = 0
    for page_no, items in _page_items(shared):
        if mode == "column":
            items_sorted = sorted(
                items, key=lambda it: (it["col"], it["bbox"][1], it["bbox"][0])
            )
        else:  # raster
            items_sorted = sorted(items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
        for it in items_sorted:
            views.append(
                FBView.from_merged_text_block(
                    it["block"], page=page_no, order_index=order_index, layout=None
                )
            )
            # Unique per (page, column) so cross-page pairs are not lumped into
            # the same column; the raster-vs-column DELTA isolates within-page
            # column reordering (pages stay ordered in both modes).
            col_global.append(page_no * 1000 + it["col"])
            order_index += 1
    return views, col_global


def _gold_ranks_for(result, n_pdf: int) -> list[int | None]:
    """Per-fed-position gold reading-order rank from an AlignResult.

    A PDF block's gold rank = ``min`` of the gold indices it aligned to (gold
    HTML is in DOM order, so a gold element's index IS its reading-order rank;
    a block aligning to several gold elements via a SPLIT belongs, in a correct
    reading order, at its EARLIEST gold position). A MERGE row folds k pdf
    positions onto one gold index — each gets that rank. A PDF_GAP position
    appears in no row -> ``None`` (dropped by the metric)."""
    ranks: list[int | None] = [None] * n_pdf
    for row in result.rows:
        gidx = [g for g in (row.align.get("gold_indices") or [])]
        if not gidx:
            continue
        rank = min(gidx)
        for pos in row.provenance.get("pdf_block_indices") or []:
            if 0 <= pos < n_pdf and ranks[pos] is None:
                ranks[pos] = rank
    return ranks


def _independent_gold_ranks(
    views: list[FBView], gold_views: list[GoldView], *, floor: float
) -> list[int | None]:
    """Order-INDEPENDENT per-block gold rank: each PDF block is matched to its
    best-similarity gold element (argmax ``sim`` above ``floor``), with NO
    monotonic / order constraint.

    This is the load-bearing reading-order instrument. The global
    :func:`align_blocks` aligner is a MONOTONIC DP — it preserves order by
    construction, so it can NEVER express a column crossing (it absorbs a
    raster-interleaved feed's disorder as gaps/ties, leaving Kendall tau ~1.0
    in BOTH orderings). Matching each block to its best gold element
    independently lets the fed-order gold-rank sequence ZIGZAG when columns are
    interleaved, so the metric sees the real inversions the column-order fix
    removes. A block with no gold element above ``floor`` -> ``None`` (dropped).
    """
    gold_texts = [g.text for g in gold_views]
    ranks: list[int | None] = []
    for fb in views:
        ft = fb.text
        best_s = floor
        best_i: int | None = None
        for gi, gt in enumerate(gold_texts):
            s = sim(ft, gt)
            if s > best_s:
                best_s, best_i = s, gi
        ranks.append(best_i)
    return ranks


def _score_ordering(shared: dict, gold_views: list[GoldView], *, mode: str):
    """Extract->order->score one ordering under BOTH gold-rank derivations.

    Returns ``(independent_metric, monotonic_metric, aligned_gold_fraction)``
    or ``(None, None, None)`` on failure. ``independent`` is the order-free
    per-block argmax (the reading-order instrument); ``monotonic`` is the
    global-aligner ``min(gold_indices)`` derivation (the task's literal spec,
    kept as a cross-check that documents WHY a monotonic aligner can't measure
    reading order)."""
    views, col_global = _ordered_fbviews(shared, mode=mode)
    if not views:
        return None, None, None

    # Independent (order-free) derivation — the primary reading-order signal.
    indep_ranks = _independent_gold_ranks(views, gold_views, floor=GLOBAL_SOFT_FLOOR)
    indep_metric = reading_order_metric(indep_ranks, column_ids=col_global)

    # Monotonic global-aligner derivation — the task's literal spec + cross-check.
    try:
        result = align_blocks(
            views, gold_views, max_run=GLOBAL_MAX_RUN, soft_floor=GLOBAL_SOFT_FLOOR
        )
        mono_ranks = _gold_ranks_for(result, len(views))
        mono_metric = reading_order_metric(mono_ranks, column_ids=col_global)
        n_gold = len(gold_views)
        gold_gap = int(result.ledger.counts.get("gold_gap", 0))
        aligned_frac = (n_gold - gold_gap) / max(1, n_gold)
    except (LedgerIncompleteError, Exception):
        mono_metric, aligned_frac = None, None
    finally:
        try:
            structure_align._sim_cached.cache_clear()
        except Exception:
            pass
    return indep_metric, mono_metric, aligned_frac


def _load_doc_plan(dataset_dir: Path) -> list[dict]:
    """Read a realistic build's ``realistic_docs.jsonl`` -> the KEPT docs
    (``dropped is None``) with their pair stem + render profile + seed, so we
    measure on the exact same doc/profile set the realistic eval uses."""
    docs_path = dataset_dir / "realistic_docs.jsonl"
    if not docs_path.exists():
        raise SystemExit(f"missing {docs_path} — pass --from-dataset a built realistic dir")
    plan = []
    for line in docs_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("dropped") is not None:
            continue
        plan.append(rec)
    return plan


def _pair_file_for(rec: dict) -> Path | None:
    source = rec.get("source") or "arxiv"
    stem = rec["pair"]
    pf = Path("data/pairs") / source / f"{stem}.json"
    return pf if pf.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-dataset",
        type=Path,
        default=Path("data/structure_dataset_realistic_on"),
        help="A built realistic-eval dir (reads its realistic_docs.jsonl for "
        "the exact docs/profiles/seeds).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_reports/reading_order_credit.json"),
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap docs (0 = all).")
    args = ap.parse_args()

    plan = _load_doc_plan(args.from_dataset)
    if args.limit:
        plan = plan[: args.limit]

    per_doc: list[dict] = []
    # derivation -> mode -> family -> list[ReadingOrderResult]
    results: dict[str, dict[str, dict[str, list]]] = {
        "independent": {"raster": defaultdict(list), "column": defaultdict(list)},
        "monotonic": {"raster": defaultdict(list), "column": defaultdict(list)},
    }
    drops: dict[str, int] = defaultdict(int)
    start = time.time()

    with HtmlValidator() as validator:
        for rec in plan:
            stem = rec["pair"]
            profile = rec.get("render_profile") or "clean"
            seed = int(rec.get("render_seed") or 0)
            family = "multi_column" if _is_multi_column(profile) else "single_column"

            pf = _pair_file_for(rec)
            if pf is None:
                drops["pair_missing"] += 1
                continue
            try:
                pair = json.loads(pf.read_text())
            except Exception:
                drops["read_error"] += 1
                continue
            output_html = pair.get("output_html")
            if not output_html:
                drops["no_output_html"] += 1
                continue
            gold_records = extract_html_blocks(output_html)
            if not gold_records:
                drops["no_gold_blocks"] += 1
                continue

            rendered = augment_html(output_html, profile, seed)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_pdf = Path(tmp) / "ro.pdf"
                try:
                    validator.render_pdf(rendered, tmp_pdf)
                except Exception:
                    drops["render_error"] += 1
                    continue
                try:
                    shared = extract_shared(tmp_pdf)
                except Exception:
                    drops["extract_error"] += 1
                    continue

            gold_views = [
                GoldView.from_html_block(r, i) for i, r in enumerate(gold_records)
            ]
            ind_r, mon_r, af_raster = _score_ordering(shared, gold_views, mode="raster")
            ind_c, mon_c, af_column = _score_ordering(shared, gold_views, mode="column")
            if ind_r is None or ind_c is None or mon_r is None or mon_c is None:
                drops["align_failed"] += 1
                continue

            results["independent"]["raster"][family].append(ind_r)
            results["independent"]["column"][family].append(ind_c)
            results["monotonic"]["raster"][family].append(mon_r)
            results["monotonic"]["column"][family].append(mon_c)

            def _delta(c, r, attr):
                cv, rv = getattr(c, attr), getattr(r, attr)
                return None if (cv is None or rv is None) else round(cv - rv, 6)

            per_doc.append(
                {
                    "pair": stem,
                    "doc_id": rec.get("doc_id"),
                    "render_profile": profile,
                    "family": family,
                    "n_blocks": ind_r.n_total,
                    "n_dropped_no_gold_independent": ind_r.n_dropped,
                    "aligned_gold_fraction_raster": round(af_raster, 6),
                    "aligned_gold_fraction_column": round(af_column, 6),
                    "independent": {
                        "raster": ind_r.to_dict(),
                        "column": ind_c.to_dict(),
                        "delta_kendall_tau_b": _delta(ind_c, ind_r, "kendall_tau_b"),
                        "delta_pairwise_accuracy": _delta(ind_c, ind_r, "pairwise_accuracy"),
                        "delta_cross_column_inversions_normalized": _delta(
                            ind_c, ind_r, "cross_column_inversions_normalized"
                        ),
                    },
                    "monotonic": {
                        "raster": mon_r.to_dict(),
                        "column": mon_c.to_dict(),
                        "delta_kendall_tau_b": _delta(mon_c, mon_r, "kendall_tau_b"),
                        "delta_pairwise_accuracy": _delta(mon_c, mon_r, "pairwise_accuracy"),
                    },
                }
            )
            print(
                f"[ro] {stem[:38]:38} {family:13} "
                f"INDEP tau r={_fmt(ind_r.kendall_tau_b)} c={_fmt(ind_c.kendall_tau_b)} "
                f"| pa r={_fmt(ind_r.pairwise_accuracy)} c={_fmt(ind_c.pairwise_accuracy)}",
                file=sys.stderr,
            )

    # Aggregate per derivation, per mode, per family + overall.
    families = ["multi_column", "single_column"]
    summary: dict = {}
    for deriv in ("independent", "monotonic"):
        summary[deriv] = {"by_mode": {}}
        for mode in ("raster", "column"):
            all_results = [r for fam in families for r in results[deriv][mode][fam]]
            summary[deriv]["by_mode"][mode] = {
                "overall": aggregate_results(all_results),
                "by_family": {
                    fam: aggregate_results(results[deriv][mode][fam]) for fam in families
                },
            }

    report = {
        "metric": "reading_order_credit",
        "definitions": {
            "derivation.independent": "PRIMARY reading-order instrument: each PDF "
            "block's gold rank = argmax-similarity gold element (>= soft_floor), "
            "with NO order constraint, so a column-interleaved feed's gold-rank "
            "sequence genuinely zigzags and the metric sees the inversions the "
            "column fix removes.",
            "derivation.monotonic": "CROSS-CHECK (the task's literal spec): gold "
            "rank = min(gold_indices) from the global align_blocks aligner. That "
            "aligner is a MONOTONIC DP — it preserves order by construction and "
            "therefore CANNOT express a column crossing (it absorbs disorder as "
            "gaps/ties), so its Kendall tau is ~identical in both orderings. "
            "Included to DEMONSTRATE why a monotonic aligner cannot measure "
            "reading order, not as a credit signal.",
            "kendall_tau_b": "tau-b between fed order and gold reading-order rank "
            "(gold DOM index = reading-order rank); +1 perfect, 0 random, -1 "
            "reversed; tau-b tolerates tied gold ranks (fragmented gold).",
            "pairwise_accuracy": "fraction of rank-distinct fed block pairs in the "
            "correct relative order (equal-gold-rank pairs excluded).",
            "inversions_normalized": "inversions / rank-distinct pairs (== 1 - "
            "pairwise_accuracy).",
            "cross_column_inversions_normalized": "of cross-(page,column) "
            "rank-distinct pairs, the share inverted; isolates the column-"
            "interleaving the column-order fix targets.",
            "dropped_blocks": "PDF blocks that matched no gold element above "
            "soft_floor (furniture/noise/fragments); excluded from the metric, "
            "counted as n_dropped.",
            "family": "multi_column = profile contains 'two_column' (raster "
            "interleaves columns); single_column = serif_embed/dense_layout/clean.",
        },
        "aligner": {"soft_floor": GLOBAL_SOFT_FLOOR, "max_run": GLOBAL_MAX_RUN},
        "source_dataset": str(args.from_dataset),
        "n_docs_planned": len(plan),
        "n_docs_scored": len(per_doc),
        "drop_ledger": dict(drops),
        "elapsed_seconds": round(time.time() - start, 1),
        "summary": summary,
        "per_doc": per_doc,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[ro] wrote {args.out}", file=sys.stderr)
    _print_headline(summary, drops, len(per_doc))


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:+.3f}"


def _print_headline(summary: dict, drops: dict, n_scored: int) -> None:
    print("\n==== READING-ORDER CREDIT (raster vs column) ====", file=sys.stderr)
    print(f"docs scored: {n_scored}   drops: {dict(drops)}", file=sys.stderr)
    for deriv in ("independent", "monotonic"):
        tag = "PRIMARY — order-free" if deriv == "independent" else "cross-check — monotonic aligner (order-blind)"
        print(f"\n######## derivation = {deriv} ({tag}) ########", file=sys.stderr)
        for fam in ("multi_column", "single_column", "overall"):
            print(f"\n  [{fam}]", file=sys.stderr)
            for mode in ("raster", "column"):
                node = summary[deriv]["by_mode"][mode]
                agg = node["overall"] if fam == "overall" else node["by_family"][fam]
                print(
                    f"    {mode:7} docs={agg['n_docs']:3} "
                    f"tau={_fmt(agg['macro_kendall_tau_b'])} "
                    f"pairwise={_fmt(agg['macro_pairwise_accuracy'])} "
                    f"inv={_fmt(agg['macro_inversions_normalized'])} "
                    f"xcol_inv={_fmt(agg['macro_cross_column_inversions_normalized'])}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
