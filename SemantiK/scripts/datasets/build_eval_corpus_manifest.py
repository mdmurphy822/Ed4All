"""Build the stratified real-runtime eval-corpus manifest (Plan 12 V3 — C1/C2 prep).

The R10 real-runtime eval ran on n=3 held-out arXiv PDFs. This builder
produces ``data/eval_corpus/manifest_v1.json``: a deterministic (seed=42),
stratified 30-50 PDF corpus that ``scripts/eval/eval_full_cascade.py --manifest``
consumes verbatim, so the GPU run is a single command once the card frees up.

Strata:

- ``arxiv:<subject>`` — real arXiv PDFs from the local Arxiiv Repo, subject
  derived from the repo's ``papers/<subject>/`` folder layout (the same path
  recorded in every pair JSON's ``local_pdf``). Held-out by arXiv id: a PDF
  is eligible only when NO trained pair stem starts with its normalized id.
- ``wikipedia_prerender`` / ``openstax_prerender`` — pairs whose stem is NOT
  in any train/val split AND whose ``output_html`` render is already in
  ``data/prerender_cache`` (these sources have no source PDF; the prerendered
  ground-truth render is the canonical input document, exactly as in data-gen).
- ``side_by_side`` — the 4 dev bench PDFs, included for continuity with prior
  reports; ``held_out=false`` (weakly-held-out, same caveat the eval prints).
- ``c2_*`` / ``scanned_ocr`` (C2, manifest_v2+) — eval-ONLY non-arXiv PDFs
  fetched by ``scripts/fetch_c2_eval_pdfs.py`` into
  ``data/eval_corpus/c2_pdfs/<source>/``. Auto-detected when that directory
  exists (``--no-c2`` to disable). These back NO training pairs
  (``held_out=true``) and every entry records license provenance
  (``"us-gov-pd"``) straight from the per-source ``_fetch_manifest.jsonl``.

Honest gaps land in ``missing_sources`` (gov forms, synthetic forms,
courtlistener, govinfo, scans, ...) with the reason — NO network fetches here.
Static gaps covered by a fetched C2 stratum are dropped from the list.

Usage::

    .venv/bin/python scripts/datasets/build_eval_corpus_manifest.py \\
        --out data/eval_corpus/manifest_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantik_structure.arxiv_license import extract_arxiv_id  # noqa: E402
from semantik_structure.prerender_cache import cache_path_for  # noqa: E402

# Reuse the eval driver's held-out machinery verbatim — same dataset files,
# same pair-id semantics — so the manifest and the eval can never disagree
# about what "trained on" means.
from scripts.eval_full_cascade import (  # noqa: E402
    _DATASET_FILES,
    _SIDE_BY_SIDE_DIRS,
    _build_trained_pair_id_set,
)

# 1.1: additive — c2_* strata entries carry `license`/`url`/`sha256` keys.
SCHEMA_VERSION = "eval_corpus_manifest/1.1"

# arXiv paper repo location is machine-specific; supply it via the
# SEMANTIK_ARXIV_REPO env var or the --arxiv-repo flag (no hardcoded absolute
# path in tracked code). Falls back to ``~/arxiv-repo/papers``.
DEFAULT_ARXIV_REPO = Path(
    os.environ.get("SEMANTIK_ARXIV_REPO", str(Path.home() / "arxiv-repo" / "papers"))
).expanduser()
DEFAULT_C2_DIR = Path("data/eval_corpus/c2_pdfs")
C2_LICENSE = "us-gov-pd"

# >=5 subject areas x >=4 PDFs (Plan 12 C1). Chosen for spread: hard STEM
# (math-heavy), quantum/physics (two-column + heavy notation), ML, education
# (prose/figures) and disability law (long-form legal prose — the product's
# home turf and a C2 on-ramp).
DEFAULT_SUBJECTS = (
    "physics",
    "quantum_foundations",
    "ai_ml",
    "math",
    "education",
    "ada_disability_law",
)
DEFAULT_PER_SUBJECT = 4
DEFAULT_N_WIKIPEDIA = 5
DEFAULT_N_OPENSTAX = 5
# Skip pathological inputs: at ~15-40 min/PDF on the 3070 a 100MB scan-like
# PDF would blow the wall-clock budget for the whole run.
DEFAULT_MAX_PDF_MB = 20

# Planned-but-absent sources (Plan 12 C2 / data-expansion plan). Recorded in
# the manifest so the gap is explicit; fetching them is OUT OF SCOPE here.
_STATIC_MISSING_SOURCES = [
    {
        "source": "gov_forms_real_pdf",
        "note": (
            "pair_from_gov_forms.py downloads IRS/GSA fillable PDFs to "
            "tempfiles and does not retain them; no real federal-form PDF "
            "exists on disk, and all data/pairs/forms pairs are trained-on."
        ),
    },
    {
        "source": "synthetic_forms",
        "note": (
            "gen_synthetic_forms.py renders its HTML->PDF via tempfile; "
            "data/synthetic/ does not exist, so there are zero synthetic-form "
            "PDFs on disk to evaluate."
        ),
    },
    {
        "source": "govinfo",
        "note": "planned C2 source; zero local PDFs (network fetch out of scope).",
    },
    {
        "source": "scanned_ocr",
        "note": (
            "planned C2 source (Internet Archive scans / Tesseract-quality "
            "floors); zero local scanned PDFs."
        ),
    },
]

# Pair sources audited for held-out + locally-evaluable inputs. wikipedia and
# openstax are handled as strata; everything else that comes up empty lands in
# missing_sources with the measured reason.
_AUDITED_PAIR_SOURCES = (
    "forms",
    "courtlistener",
    "gutenberg",
    "federal_register",
    "nces_digest",
    "pmc",
    "cfr",
    "synthetic_blockquote_code",
)


def _norm_arxiv_id(aid: str) -> str:
    """Normalize an arXiv id the way pair stems are written (0704.1551 -> 0704_1551)."""
    return aid.replace(".", "_").replace("/", "_")


def trained_arxiv_prefixes(trained: set[str]) -> set[str]:
    """Normalized arXiv ids of every trained pair stem ('0704_1551__00_x' -> '0704_1551')."""
    return {pid.split("__", 1)[0] for pid in trained if "__" in pid}


def collect_arxiv_candidates(
    arxiv_repo: Path,
    subjects: tuple[str, ...],
    trained_prefixes: set[str],
    max_bytes: int,
) -> dict[str, list[tuple[str, Path]]]:
    """Per subject: sorted [(norm_id, pdf_path)] of held-out PDFs, deduped by id.

    A paper filed under two subject folders counts once (first subject in the
    given order wins) so the corpus never evaluates the same PDF twice.
    """
    seen_ids: set[str] = set()
    out: dict[str, list[tuple[str, Path]]] = {}
    for subject in subjects:
        cands: list[tuple[str, Path]] = []
        sdir = arxiv_repo / subject
        if sdir.is_dir():
            for pdf in sorted(sdir.rglob("*.pdf")):
                aid = extract_arxiv_id(pdf.name)
                if aid is None:
                    continue
                norm = _norm_arxiv_id(aid)
                if norm in trained_prefixes or norm in seen_ids:
                    continue
                try:
                    size = pdf.stat().st_size
                except OSError:
                    continue
                if size > max_bytes:
                    continue
                seen_ids.add(norm)
                cands.append((norm, pdf))
        out[subject] = cands
    return out


def collect_prerendered_heldout(
    pairs_dir: Path, cache_dir: Path, trained: set[str]
) -> list[tuple[str, Path]]:
    """Sorted [(pair_stem, cached_pdf)] for held-out pairs whose render is cached."""
    out: list[tuple[str, Path]] = []
    if not pairs_dir.is_dir():
        return out
    for f in sorted(pairs_dir.glob("*.json")):
        if f.stem in trained:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        html = d.get("output_html")
        if not isinstance(html, str) or not html:
            continue
        pdf = cache_path_for(html, cache_dir=cache_dir)
        if pdf.exists():
            out.append((f.stem, pdf))
    return out


def _sample(rng: random.Random, items: list, k: int) -> list:
    """Deterministic sample of k items (all of them when fewer are available)."""
    if len(items) <= k:
        return list(items)
    return sorted(rng.sample(items, k))


def _arxiv_pair_stem_index(repo_root: Path) -> dict[str, str]:
    """norm arXiv id -> first pair stem on disk (traceability for paired ids)."""
    idx: dict[str, str] = {}
    pairs_dir = repo_root / "data/pairs/arxiv"
    if not pairs_dir.is_dir():
        return idx
    for f in sorted(pairs_dir.glob("*.json")):
        prefix = f.stem.split("__", 1)[0]
        idx.setdefault(prefix, f.stem)
    return idx


# C2 fetch-dir source -> the static missing_sources gap it closes.
_C2_CLOSES_STATIC_GAP = {
    "gov_forms": "gov_forms_real_pdf",
    "govinfo": "govinfo",
    "scanned_ocr": "scanned_ocr",
}


def collect_c2_pdfs(c2_dir: Path, max_bytes: int, log=print) -> list[tuple[str, dict]]:
    """Sorted [(source_subdir, fetch_record)] of kept C2 PDFs on disk.

    Reads each ``<c2_dir>/<source>/_fetch_manifest.jsonl`` written by
    ``scripts/fetch_c2_eval_pdfs.py``; keeps records whose fetch succeeded
    (``ok``/``skip_exists``) AND whose PDF exists under the size cap. Failure
    records stay in the fetch manifest — they are provenance, not entries.
    Deduped by filename (last manifest line wins — the manifest is append-mode).
    """
    out: list[tuple[str, dict]] = []
    for mf in sorted(c2_dir.glob("*/_fetch_manifest.jsonl")):
        sub = mf.parent.name
        by_name: dict[str, dict] = {}
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") not in ("ok", "skip_exists"):
                continue
            name = Path(rec.get("local_path", "")).name
            if name:
                by_name[name] = rec
        kept = 0
        for name in sorted(by_name):
            rec = by_name[name]
            pdf = mf.parent / name
            if not pdf.exists():
                continue
            if pdf.stat().st_size > max_bytes:
                log(f"[manifest] c2:{sub}: {name} exceeds {max_bytes}B cap — skipped")
                continue
            rec["_pdf_path"] = pdf
            out.append((sub, rec))
            kept += 1
        log(f"[manifest] c2_{sub}: {kept} eval-only PDFs")
    return out


def detect_missing_sources(
    repo_root: Path, trained: set[str], c2_counts: Counter | None = None
) -> list[dict]:
    """Audit pair sources that yield zero locally-evaluable held-out PDFs.

    ``c2_counts`` (source-subdir -> n entries) drops static gaps that a
    fetched C2 stratum now covers, and annotates the courtlistener note.
    """
    c2_counts = c2_counts or Counter()
    missing: list[dict] = []
    cache_dir = repo_root / "data/prerender_cache"
    for src in _AUDITED_PAIR_SOURCES:
        pairs_dir = repo_root / "data/pairs" / src
        stems = [p.stem for p in sorted(pairs_dir.glob("*.json"))] if pairs_dir.is_dir() else []
        n_heldout = sum(1 for s in stems if s not in trained)
        if not stems:
            note = "no pairs on disk."
        elif n_heldout == 0:
            note = f"all {len(stems)} pairs trained-on; no held-out eval inputs."
            if src == "courtlistener":
                n_cached = len(list((repo_root / "data/cache/courtlistener").glob("*.pdf")))
                note += (
                    f" ({n_cached} real opinion PDFs cached at data/cache/courtlistener,"
                    " but every one backs a trained pair — fetch FRESH opinions as"
                    " eval-only inputs in C2, not here.)"
                )
        else:
            n_render = len(collect_prerendered_heldout(pairs_dir, cache_dir, trained))
            if n_render > 0:
                continue  # actually evaluable — not missing (future stratum)
            note = (
                f"{n_heldout}/{len(stems)} pairs held-out but none has a local "
                "source PDF or prerendered render in data/prerender_cache."
            )
        if src == "courtlistener" and c2_counts.get("courtlistener"):
            note += (
                f" Now partially covered: c2_courtlistener supplies"
                f" {c2_counts['courtlistener']} fresh eval-only opinions."
            )
        missing.append({"source": src, "n_pairs": len(stems), "n_heldout": n_heldout, "note": note})
    covered = {gap for sub, gap in _C2_CLOSES_STATIC_GAP.items() if c2_counts.get(sub)}
    missing.extend(dict(m) for m in _STATIC_MISSING_SOURCES if m["source"] not in covered)
    return missing


def build_manifest(
    repo_root: Path,
    *,
    arxiv_repo: Path,
    subjects: tuple[str, ...],
    per_subject: int,
    n_wikipedia: int,
    n_openstax: int,
    seed: int,
    max_pdf_mb: int,
    include_side_by_side: bool = True,
    c2_dir: Path | None = None,
    log=print,
) -> dict:
    rng = random.Random(seed)
    max_bytes = max_pdf_mb * 1024 * 1024

    trained, sizes = _build_trained_pair_id_set(repo_root)
    prefixes = trained_arxiv_prefixes(trained)
    log(f"[manifest] trained pair-ids union: {len(trained)} ({len(prefixes)} arXiv-id prefixes)")

    entries: list[dict] = []
    shortfalls: dict[str, int] = {}

    def _entry(pdf: Path, source: str, stratum: str, pair_id: str | None, held_out: bool, **extra):
        e = {
            "pdf_path": str(pdf.resolve()),
            "source": source,
            "stratum": stratum,
            "pair_id": pair_id,
            "held_out": held_out,
            "size_bytes": pdf.stat().st_size,
        }
        e.update(extra)
        entries.append(e)

    # ---------- (a) arXiv, stratified by subject folder ----------
    stem_idx = _arxiv_pair_stem_index(repo_root)
    candidates = collect_arxiv_candidates(arxiv_repo, subjects, prefixes, max_bytes)
    for subject in subjects:
        cands = candidates[subject]
        picked = _sample(rng, cands, per_subject)
        if len(picked) < per_subject:
            shortfalls[f"arxiv:{subject}"] = per_subject - len(picked)
        log(f"[manifest] arxiv:{subject}: {len(cands)} held-out candidates -> {len(picked)}")
        for norm, pdf in picked:
            _entry(
                pdf,
                "arxiv",
                f"arxiv:{subject}",
                stem_idx.get(norm),  # most held-out PDFs were never paired
                True,
                arxiv_id=norm,
            )

    # ---------- (b) non-arXiv prerendered sources ----------
    cache_dir = repo_root / "data/prerender_cache"
    for src, want in (("wikipedia", n_wikipedia), ("openstax", n_openstax)):
        cands = collect_prerendered_heldout(repo_root / "data/pairs" / src, cache_dir, trained)
        picked = _sample(rng, cands, want)
        if len(picked) < want:
            shortfalls[f"{src}_prerender"] = want - len(picked)
        log(f"[manifest] {src}_prerender: {len(cands)} held-out candidates -> {len(picked)}")
        for stem, pdf in picked:
            _entry(
                pdf,
                src,
                f"{src}_prerender",
                stem,
                True,
                note="input is the prerendered ground-truth render (no source PDF exists)",
            )

    # ---------- side-by-side bench (continuity with prior reports) ----------
    if include_side_by_side:
        for rel in _SIDE_BY_SIDE_DIRS:
            p = repo_root / rel / "input.pdf"
            if p.exists():
                _entry(
                    p,
                    "side_by_side",
                    "side_by_side",
                    Path(rel).name,
                    False,
                    note="weakly-held-out: used during development, never formally trained on",
                )

    # ---------- (c) C2 eval-only strata (Plan 12 V3.2/C2) ----------
    # No rng draws here: C2 keeps everything fetched (it IS the modest set),
    # so adding/removing the c2 dir cannot reshuffle the strata above.
    c2_counts: Counter = Counter()
    if c2_dir is not None and c2_dir.is_dir():
        for sub, rec in collect_c2_pdfs(c2_dir, max_bytes, log=log):
            c2_counts[sub] += 1
            _entry(
                rec["_pdf_path"],
                f"c2_{sub}",
                rec.get("stratum") or f"c2_{sub}",
                None,  # backs no pair by construction (eval-only fetch)
                True,
                license=rec.get("license", C2_LICENSE),
                url=rec.get("url", ""),
                sha256=rec.get("sha256", ""),
                note=rec.get("title", ""),
            )

    # ---------- (d) honest gaps ----------
    missing_sources = detect_missing_sources(repo_root, trained, c2_counts)

    counts = Counter(e["stratum"] for e in entries)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/datasets/build_eval_corpus_manifest.py",
        "seed": seed,
        "config": {
            "arxiv_repo": str(arxiv_repo),
            "subjects": list(subjects),
            "per_subject": per_subject,
            "n_wikipedia": n_wikipedia,
            "n_openstax": n_openstax,
            "max_pdf_mb": max_pdf_mb,
            "include_side_by_side": include_side_by_side,
            "c2_dir": str(c2_dir) if c2_dir is not None else None,
        },
        "trained_pair_exclusion": {
            "method": (
                "union of `pair` fields across the train/val splits below "
                "(scripts/eval_full_cascade._build_trained_pair_id_set). arXiv "
                "PDFs excluded when ANY trained pair stem starts with the PDF's "
                "normalized arXiv id; prerendered sources excluded by pair stem."
            ),
            "dataset_files": dict(_DATASET_FILES),
            "per_dataset_sizes": sizes,
            "union_size": len(trained),
            "n_trained_arxiv_id_prefixes": len(prefixes),
        },
        "counts_per_stratum": dict(sorted(counts.items())),
        "stratum_shortfalls": shortfalls,
        "n_entries": len(entries),
        "entries": entries,
        "missing_sources": missing_sources,
    }
    return manifest


def format_stratum_table(manifest: dict) -> str:
    lines = ["stratum                          n   held_out"]
    by_stratum: dict[str, list[dict]] = {}
    for e in manifest["entries"]:
        by_stratum.setdefault(e["stratum"], []).append(e)
    for stratum in sorted(by_stratum):
        es = by_stratum[stratum]
        n_held = sum(1 for e in es if e["held_out"])
        lines.append(f"{stratum:30s} {len(es):3d}   {n_held}/{len(es)}")
    lines.append(f"{'TOTAL':30s} {manifest['n_entries']:3d}")
    lines.append("")
    lines.append("missing_sources:")
    for m in manifest["missing_sources"]:
        lines.append(f"  {m['source']:28s} {m['note']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    repo_root_default = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Default targets v2 so a casual rebuild can never clobber the committed v1.
    ap.add_argument("--out", type=Path, default=Path("data/eval_corpus/manifest_v2.json"))
    ap.add_argument("--repo-root", type=Path, default=repo_root_default)
    ap.add_argument("--arxiv-repo", type=Path, default=DEFAULT_ARXIV_REPO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subjects", type=str, default=",".join(DEFAULT_SUBJECTS))
    ap.add_argument("--per-subject", type=int, default=DEFAULT_PER_SUBJECT)
    ap.add_argument("--n-wikipedia", type=int, default=DEFAULT_N_WIKIPEDIA)
    ap.add_argument("--n-openstax", type=int, default=DEFAULT_N_OPENSTAX)
    ap.add_argument("--max-pdf-mb", type=int, default=DEFAULT_MAX_PDF_MB)
    ap.add_argument("--no-side-by-side", dest="side_by_side", action="store_false", default=True)
    ap.add_argument(
        "--c2-dir",
        type=Path,
        default=DEFAULT_C2_DIR,
        help="C2 eval-only fetch dir (scripts/fetch_c2_eval_pdfs.py output); "
        "auto-skipped when absent.",
    )
    ap.add_argument(
        "--no-c2",
        dest="include_c2",
        action="store_false",
        default=True,
        help="Exclude the c2_* strata even when --c2-dir exists (v1 parity).",
    )
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    if not args.arxiv_repo.is_dir():
        raise SystemExit(f"arXiv repo not found: {args.arxiv_repo}")

    c2_dir: Path | None = None
    if args.include_c2:
        c2_dir = args.c2_dir if args.c2_dir.is_absolute() else repo_root / args.c2_dir

    manifest = build_manifest(
        repo_root,
        arxiv_repo=args.arxiv_repo,
        subjects=tuple(s for s in args.subjects.split(",") if s),
        per_subject=args.per_subject,
        n_wikipedia=args.n_wikipedia,
        n_openstax=args.n_openstax,
        seed=args.seed,
        max_pdf_mb=args.max_pdf_mb,
        include_side_by_side=args.side_by_side,
        c2_dir=c2_dir,
    )

    out = args.out if args.out.is_absolute() else repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] wrote {out} ({manifest['n_entries']} entries)")
    print()
    print(format_stratum_table(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
