"""Build BERT-MathSpecialist training data.

The MathSpecialist is a multi-head DeBERTa-v3-base + LoRA classifier
trained on ar5iv ground-truth math elements:

    * head ``math_type``      ∈ {inline, display, numbered, multiline, matrix}
    * head ``eq_num_assoc``   ∈ {none, attached_left, attached_right, separate}

Source pipeline:
    1. Walk ``data/pairs/arxiv/*.json`` for ``arxiv_id``s.
    2. Fetch ar5iv HTML for each unique id (cached on disk under
       ``data/ar5iv_html_cache/<id>.html``).
    3. For every ``<math>`` element, derive:
        - ``math_type`` from display attr + structural tells
          (mtable, equation-number sibling).
        - ``eq_num_assoc`` from neighbouring ``ltx_tag_equation`` /
          ``(N)`` tokens.
        - Input text = the LaTeX in ``alttext`` (compact, dense in
          mathematical signal). When alttext is missing, fall back to
          the math element's plain text.
        - Surrounding-context text (sentence before + after) is
          concatenated with separator tokens for the BERT input.
    4. Stratified 80/10/10 split by ``math_type``.
    5. Print class-balance report; emit ``coverage_report.json``.

Network usage: only the ar5iv fetch, which is the single existing
network entry point in this repository (``parse_ar5iv.fetch_ar5iv_html``).
Once cached, subsequent runs are offline.

Usage:
    python data/build_math_specialist_data.py --limit 200
    python data/build_math_specialist_data.py --offline   # cache-only
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Label vocabularies — Plans/01 DP-2.3
# ---------------------------------------------------------------------------

MATH_TYPES = ["inline", "display", "numbered", "multiline", "matrix"]
EQ_NUM_ASSOC = ["none", "attached_left", "attached_right", "separate"]

MATH_TYPE_TO_ID = {n: i for i, n in enumerate(MATH_TYPES)}
EQ_NUM_TO_ID = {n: i for i, n in enumerate(EQ_NUM_ASSOC)}


# ---------------------------------------------------------------------------
# ar5iv HTML cache
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path("data/ar5iv_html_cache")
DEFAULT_PAIRS_DIR = Path("data/pairs/arxiv")
DEFAULT_OUT_DIR = Path("data/math_specialist_dataset")


def _cache_path(cache_dir: Path, arxiv_id: str) -> Path:
    safe = arxiv_id.replace("/", "_")
    return cache_dir / f"{safe}.html"


def _load_or_fetch_ar5iv(
    arxiv_id: str,
    cache_dir: Path,
    *,
    offline: bool,
    timeout: int,
    sleep: float,
) -> str | None:
    """Return ar5iv HTML for ``arxiv_id`` from the on-disk cache,
    falling back to a live fetch when not cached and not offline.
    """
    p = _cache_path(cache_dir, arxiv_id)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    if offline:
        return None
    # Live fetch — single network call point.
    from dart_semantic.parse_ar5iv import Ar5ivParseError, fetch_ar5iv_html
    try:
        html = fetch_ar5iv_html(arxiv_id, timeout=timeout)
    except Ar5ivParseError as exc:
        sys.stderr.write(f"[ar5iv] fetch fail {arxiv_id}: {exc}\n")
        return None
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[ar5iv] error {arxiv_id}: {exc}\n")
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    if sleep > 0:
        time.sleep(sleep)
    return html


# ---------------------------------------------------------------------------
# Math-element label derivation
# ---------------------------------------------------------------------------

EQ_NUM_TRAILING_RE = re.compile(r"\(\s*\d+(?:\.\d+)?\s*\)\s*$")
EQ_NUM_LEADING_RE = re.compile(r"^\s*\(\s*\d+(?:\.\d+)?\s*\)")


def _has_eq_tag(parent: Tag) -> str | None:
    """Look for an LaTeXML equation-number sibling.

    LaTeXML emits ``<span class="ltx_tag ltx_tag_equation">(N)</span>``
    inside the equation's wrapper. Returns 'attached_left' /
    'attached_right' / 'separate' / None depending on placement
    relative to the math element.
    """
    if parent is None:
        return None
    tag = parent.find(class_="ltx_tag_equation")
    if tag is None:
        # Some LaTeXML revisions use ltx_tag generally inside td of
        # equation tables.
        for cand in parent.find_all(class_="ltx_tag"):
            classes = cand.get("class") or []
            if "ltx_tag_equation" in classes or "ltx_tag" in classes:
                tag = cand
                break
    if tag is None:
        return None
    # Find the math element within the same parent, infer position.
    math = parent.find("math")
    if math is None:
        return "separate"
    # If the eq-tag appears before the math node in document order, it's
    # attached to the left; after → attached_right; if it lives in a
    # different table cell or is structurally separate → 'separate'.
    parents_of_math = list(math.parents)
    parents_of_tag = list(tag.parents)
    common = next((p for p in parents_of_math if p in parents_of_tag), None)
    if common is None:
        return "separate"
    # Compare document positions inside `common`.
    desc = list(common.descendants)
    try:
        i_tag = desc.index(tag)
        i_math = desc.index(math)
    except ValueError:
        return "separate"
    if i_tag < i_math:
        return "attached_left"
    return "attached_right"


def _is_matrix_math(math_el: Tag) -> bool:
    """Matrix detection: presence of <mtable> wrapped by <mfenced> /
    paren-pair, or an mtable that looks matrix-shaped.
    """
    mtables = math_el.find_all("mtable")
    if not mtables:
        return False
    for mt in mtables:
        if mt.find_parent(["mfenced"]) is not None:
            return True
        # Check siblings of mtable for fences <mo>(</mo> or <mo>[</mo>.
        prev = mt.find_previous_sibling()
        nxt = mt.find_next_sibling()
        if prev is not None and prev.name == "mo" and (prev.text or "").strip() in ("(", "[", "|"):
            if nxt is not None and nxt.name == "mo" and (nxt.text or "").strip() in (")", "]", "|"):
                return True
    return False


def _is_multiline(math_el: Tag) -> bool:
    """Multiline detection: an <mtable> whose row count > 1 and which is
    not a matrix (caller checks matrix first).
    """
    mtables = math_el.find_all("mtable")
    if not mtables:
        return False
    return any(len(mt.find_all("mtr")) > 1 for mt in mtables)


def _ltx_alttext(math_el: Tag) -> str:
    return (math_el.get("alttext") or "").strip()


def _math_text(math_el: Tag) -> str:
    """Best-effort flat text for a math element; alttext preferred."""
    alt = _ltx_alttext(math_el)
    if alt:
        return alt
    return " ".join((math_el.get_text() or "").split()).strip()


def _ancestor_block(math_el: Tag) -> Tag | None:
    """Walk up to the LaTeXML block ancestor that scopes this math
    element AND contains any equation-number sibling.

    LaTeXML wraps numbered equations in:
        <table class="ltx_equation ltx_eqn_table">
          <tr class="ltx_equation ltx_eqn_row">
            <td class="ltx_eqn_cell"> <math display="block">...</math> </td>
            <td class="ltx_eqn_cell ltx_eqn_eqno">
              <span class="ltx_tag ltx_tag_equation">(N)</span>
            </td>
          </tr>
        </table>

    The math element's immediate parent is just one ``<td>`` (the
    equation-cell) — the eq-number lives in the *sibling* td. So we
    must return the row (or the table) so that ``_has_eq_tag`` can see
    both descendants.

    Resolution order:
        1. nearest ``ltx_equation`` / ``ltx_eqn_row`` / ``ltx_eqn_table``
           / ``ltx_equationgroup`` / ``ltx_para`` ancestor;
        2. otherwise the nearest ``<p>`` / ``<li>`` / ``<caption>`` block;
        3. fall back to ``<td>`` / ``<th>`` only when nothing above
           matched (so inline math inside a table cell still has scope).
    """
    eq_class_targets = (
        "ltx_equation", "ltx_eqn_row", "ltx_eqn_table",
        "ltx_equationgroup", "ltx_para", "ltx_p", "ltx_caption",
    )
    cell_fallback: Tag | None = None
    for p in math_el.parents:
        if not isinstance(p, Tag):
            continue
        cls = p.get("class") or []
        if any(c in cls for c in eq_class_targets):
            return p
        if p.name in ("p", "li"):
            return p
        if p.name in ("td", "th") and cell_fallback is None:
            cell_fallback = p
    return cell_fallback


def _surrounding_text(math_el: Tag, *, max_chars: int = 200) -> str:
    """Pull text immediately around the math element (same paragraph)."""
    block = _ancestor_block(math_el) or math_el.parent
    if block is None:
        return ""
    # Collect text excluding any other math children.
    chunks: list[str] = []
    for descendant in block.descendants:
        if isinstance(descendant, Tag) and descendant.name == "math":
            chunks.append(" [MATH] ")
            continue
        if isinstance(descendant, Tag):
            continue
        s = str(descendant)
        if s.strip():
            chunks.append(s)
    text = " ".join(chunks)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        # Keep symmetric window around the [MATH] marker if present.
        idx = text.find("[MATH]")
        if idx >= 0:
            half = max_chars // 2
            start = max(0, idx - half)
            end = min(len(text), idx + half + len("[MATH]"))
            text = text[start:end]
        else:
            text = text[:max_chars]
    return text


def derive_labels(math_el: Tag) -> tuple[str, str]:
    """Return ``(math_type, eq_num_assoc)`` for one ``<math>`` element."""
    display = (math_el.get("display") or "inline").strip().lower()

    is_matrix = _is_matrix_math(math_el)
    is_multiline = (not is_matrix) and _is_multiline(math_el)

    parent_block = _ancestor_block(math_el)
    eq_assoc = _has_eq_tag(parent_block) if parent_block is not None else None

    # If we did not find an explicit ltx_tag, look at adjacent text for
    # a (N) trailing/leading pattern.
    if eq_assoc is None and parent_block is not None:
        text = " ".join((parent_block.get_text() or "").split())
        # Look at the math element's surrounding siblings.
        prev_text = ""
        next_text = ""
        for sib in math_el.previous_siblings:
            if isinstance(sib, Tag):
                prev_text = (sib.get_text() or "").strip()
            else:
                prev_text = str(sib).strip()
            if prev_text:
                break
        for sib in math_el.next_siblings:
            if isinstance(sib, Tag):
                next_text = (sib.get_text() or "").strip()
            else:
                next_text = str(sib).strip()
            if next_text:
                break
        if EQ_NUM_TRAILING_RE.search(next_text):
            eq_assoc = "attached_right"
        elif EQ_NUM_LEADING_RE.search(prev_text):
            eq_assoc = "attached_left"
        elif EQ_NUM_TRAILING_RE.search(text) and display == "block":
            eq_assoc = "separate"
    if eq_assoc is None:
        eq_assoc = "none"

    if is_matrix:
        math_type = "matrix"
    elif is_multiline:
        math_type = "multiline"
    elif display == "block" and eq_assoc != "none":
        math_type = "numbered"
    elif display == "block":
        math_type = "display"
    else:
        math_type = "inline"

    return math_type, eq_assoc


# ---------------------------------------------------------------------------
# Per-paper extraction
# ---------------------------------------------------------------------------


def extract_rows_from_html(arxiv_id: str, html: str, *, max_rows: int) -> list[dict]:
    """Walk every <math> in ar5iv HTML and emit one labelled row each."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for math_el in soup.find_all("math"):
        # Skip math elements that are MathML-only stubs with no text /
        # alttext — they carry no useful signal for the head.
        text = _math_text(math_el)
        if not text:
            continue
        math_type, eq_assoc = derive_labels(math_el)
        ctx = _surrounding_text(math_el)
        rows.append({
            "document_id": f"arxiv:{arxiv_id}",
            "alttext": _ltx_alttext(math_el),
            "math_text": text,
            "context": ctx,
            "math_type": math_type,
            "eq_num_assoc": eq_assoc,
            "display": (math_el.get("display") or "inline"),
        })
        if max_rows and len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# Stratified split + oversample bookkeeping
# ---------------------------------------------------------------------------


def stratified_split(
    rows: list[dict],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["math_type"]].append(r)
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for cls, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(lst[:n_train])
        val.extend(lst[n_train:n_train + n_val])
        test.extend(lst[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many distinct arxiv_ids (0 = all).")
    ap.add_argument("--max-rows-per-paper", type=int, default=200)
    ap.add_argument("--max-total-rows", type=int, default=20000)
    ap.add_argument("--offline", action="store_true",
                    help="Only use cached HTML, never fetch.")
    ap.add_argument("--fetch-timeout", type=int, default=20)
    ap.add_argument("--fetch-sleep", type=float, default=0.6,
                    help="Delay between live fetches (be nice to ar5iv).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-cache-only",
        action="store_true",
        help="Also include arxiv_ids that exist in the ar5iv HTML cache "
             "but not in any pair file (e.g. from a category-targeted "
             "crawl run by scripts/crawl_ar5iv_by_category.py).",
    )
    args = ap.parse_args()

    pair_files = sorted(args.pairs_dir.glob("*.json"))
    if not pair_files:
        sys.exit(f"no pairs under {args.pairs_dir}")

    # arxiv_id -> first-seen pair file (we only need one per id).
    seen_ids: dict[str, Path] = {}
    for p in pair_files:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        aid = d.get("arxiv_id")
        if not aid or aid in seen_ids:
            continue
        seen_ids[aid] = p

    print(f"[pairs] {len(seen_ids)} distinct arxiv_ids across {len(pair_files)} pair files")

    if args.include_cache_only and args.cache_dir.exists():
        n_added = 0
        for p in sorted(args.cache_dir.glob("*.html")):
            if p.stat().st_size <= 0 or p.name.startswith("_"):
                continue
            # The cache filename is the arxiv_id with '/' -> '_'.
            stem = p.stem
            # Old-style IDs (e.g. ``hep-th/0501001``) have a single '/'
            # which we mapped to '_'. Modern IDs (e.g. ``2401.12345``)
            # have no slash. Try the safe form first; if that's already
            # a known pair-derived id, use it; otherwise, assume modern
            # form.
            candidate = stem
            if candidate in seen_ids:
                continue
            # Try the slash-restored form too.
            slash_form = stem.replace("_", "/", 1)
            if slash_form in seen_ids:
                continue
            seen_ids[candidate] = p  # fake "source path" — points at HTML
            n_added += 1
        print(f"[cache] +{n_added} cache-only arxiv_ids "
              f"(total now {len(seen_ids)})")

    if args.limit:
        ids = list(seen_ids)[: args.limit]
        seen_ids = {a: seen_ids[a] for a in ids}
        print(f"[pairs] truncated to {len(seen_ids)} ids (--limit)")

    rows: list[dict] = []
    n_papers_with_math = 0
    n_papers_skipped = 0
    n_papers_attempted = 0

    for i, aid in enumerate(seen_ids):
        n_papers_attempted += 1
        html = _load_or_fetch_ar5iv(
            aid, args.cache_dir,
            offline=args.offline,
            timeout=args.fetch_timeout,
            sleep=args.fetch_sleep,
        )
        if html is None:
            n_papers_skipped += 1
            continue
        paper_rows = extract_rows_from_html(
            aid, html, max_rows=args.max_rows_per_paper,
        )
        if paper_rows:
            n_papers_with_math += 1
        rows.extend(paper_rows)
        if (i + 1) % 25 == 0 or args.limit and i + 1 == args.limit:
            print(f"[paper {i+1}/{len(seen_ids)}] rows={len(rows)} "
                  f"with_math={n_papers_with_math} skipped={n_papers_skipped}")
        if args.max_total_rows and len(rows) >= args.max_total_rows:
            print(f"[stop] hit --max-total-rows={args.max_total_rows}")
            break

    if not rows:
        sys.exit("[fatal] no math rows extracted — check --offline / cache state")

    # Class balance report.
    type_counts = Counter(r["math_type"] for r in rows)
    eq_counts = Counter(r["eq_num_assoc"] for r in rows)
    print(f"\n[balance] math_type ({len(rows)} rows):")
    for cls in MATH_TYPES:
        c = type_counts.get(cls, 0)
        pct = (100.0 * c / len(rows)) if rows else 0.0
        print(f"  {cls:10} {c:7} ({pct:5.2f}%)")
    print("[balance] eq_num_assoc:")
    for cls in EQ_NUM_ASSOC:
        c = eq_counts.get(cls, 0)
        pct = (100.0 * c / len(rows)) if rows else 0.0
        print(f"  {cls:14} {c:7} ({pct:5.2f}%)")

    # Stratified split by math_type.
    train, val, test = stratified_split(rows, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, args.out_dir / "train.jsonl")
    write_jsonl(val, args.out_dir / "val.jsonl")
    write_jsonl(test, args.out_dir / "test.jsonl")
    print(f"\n[split] train={len(train)} val={len(val)} test={len(test)} "
          f"-> {args.out_dir}")

    coverage = {
        "n_papers_attempted": n_papers_attempted,
        "n_papers_with_math": n_papers_with_math,
        "n_papers_skipped": n_papers_skipped,
        "total_rows": len(rows),
        "math_type_counts": dict(type_counts),
        "eq_num_assoc_counts": dict(eq_counts),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "math_types": MATH_TYPES,
        "eq_num_assoc": EQ_NUM_ASSOC,
    }
    (args.out_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2)
    )
    print(f"[save] coverage report -> {args.out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
