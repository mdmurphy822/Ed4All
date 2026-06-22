"""Per-source example capping shared across the dataset builders.

A freshly-scaled source (e.g. the OpenStax textbook corpus, which grew
578 -> 7,669 pairs and carries ~48 blocks/pair) otherwise swamps the
minor classes that only a few sources supply — see
feedback_balance_dominant_category, where over-volume regressed the minor
classes twice. Every builder that ingests data/pairs/openstax
(build_structure_data, build_semantic_data, build_table_specialist_data)
should apply the same cap so the imbalance can't sneak back in through a
builder we forgot.

The cap is applied at the EXAMPLE level, after merge and before any
expensive cascade pass, as a seeded uniform random subsample within each
source — reproducible, and each source keeps its own internal label mix.

ROLE-AWARE PROTECTION: a *uniform* per-source cap fixes source dominance
but re-thins the rare classes that live mostly in the now-capped sources
(citation in arXiv/PMC bibliographies, legal/footer in CFR/CourtListener,
author in arXiv front-matter). So when a ``label_key`` is given, classes
whose GLOBAL share is below ``protect_below_frac`` are protected: every
example carrying a rare label is kept, and only the abundant-label bulk
(body / data cells) is subsampled to hit the ceiling. That buys both
source balance AND minor-class counts. A source may slightly exceed its
cap if its protected (rare-label) examples alone exceed it — protecting
rare classes wins over the ceiling, by design.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict


class ProtectionNotAllowedError(ValueError):
    """Raised when rare-label protection (--cap-protect-frac > 0) is requested
    for a builder that has explicitly opted out (its trainer owns class
    balance). A typed error, not a silently-ignored flag — see
    feedback_no_silent_fallbacks."""


def rare_labels(examples: list[dict], label_key: str, protect_below_frac: float) -> set:
    """Labels whose global share of ``examples`` is below the threshold."""
    counts = Counter(r["labels"][label_key] for r in examples)
    total = sum(counts.values())
    if not total:
        return set()
    return {lab for lab, c in counts.items() if c / total < protect_below_frac}


def cap_examples_per_source(
    examples: list[dict],
    default_cap: int | None,
    overrides: dict[str, int],
    seed: int = 42,
    *,
    label_key: str | None = None,
    protect_below_frac: float = 0.05,
) -> tuple[list[dict], dict[str, dict]]:
    """Subsample so no source exceeds its cap, protecting rare labels.

    With ``label_key`` set, all examples carrying a globally-rare label
    (share < ``protect_below_frac``) survive; only abundant-label examples
    are subsampled to reach a source's cap. Without it, the subsample is
    uniform.

    Returns (kept_examples, report) where report[source] =
    {"before", "after", "cap", "protected_kept"}. ``overrides`` takes
    precedence over ``default_cap`` per source; a None cap means "no
    limit".
    """
    protected = rare_labels(examples, label_key, protect_below_frac) if label_key else set()
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in examples:
        by_source[r["source"]].append(r)
    rng = random.Random(seed)
    kept: list[dict] = []
    report: dict[str, dict] = {}
    for src, lst in by_source.items():
        cap = overrides.get(src, default_cap)
        before = len(lst)
        prot_kept = 0
        if cap is None or before <= cap:
            chosen = lst
        elif not protected:
            rng.shuffle(lst)
            chosen = lst[:cap]
        else:
            keep = [r for r in lst if r["labels"][label_key] in protected]
            drop = [r for r in lst if r["labels"][label_key] not in protected]
            prot_kept = len(keep)
            budget = max(0, cap - len(keep))
            rng.shuffle(drop)
            chosen = keep + drop[:budget]
        report[src] = {
            "before": before,
            "after": len(chosen),
            "cap": cap,
            "protected_kept": prot_kept,
        }
        kept.extend(chosen)
    return kept, report


def parse_source_caps(specs: list[str]) -> dict[str, int]:
    """Parse ['arxiv=20000', 'gutenberg=5000'] -> {'arxiv': 20000, ...}."""
    out: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--source-cap expects SOURCE=N, got {spec!r}")
        src, n = spec.split("=", 1)
        out[src.strip()] = int(n)
    return out


def add_cap_args(ap) -> None:
    """Register the per-source cap flags on an ArgumentParser."""
    ap.add_argument(
        "--max-examples-per-source",
        type=int,
        default=None,
        help="Ceiling on training examples per source (seeded random "
        "subsample, applied after merge / before cascade). Stops a "
        "dominant source from swamping minor classes. None = no cap.",
    )
    ap.add_argument(
        "--source-cap",
        action="append",
        default=[],
        metavar="SOURCE=N",
        help="Per-source cap override, e.g. --source-cap arxiv=20000 "
        "(repeatable; overrides --max-examples-per-source for that "
        "source).",
    )
    ap.add_argument(
        "--cap-protect-frac",
        type=float,
        default=0.0,
        help="Role-aware protection: keep ALL examples whose primary label "
        "has a global share below this fraction, subsampling only the "
        "abundant bulk. Default 0.0 = OFF (uniform per-source cap). Class "
        "balance is normally handled at train time by the trainers' "
        "class-weighted CE + WeightedRandomSampler, so leave this off "
        "unless a head's minor classes are genuinely scarce (set e.g. "
        "0.05). NOTE: do NOT enable for table_specialist — its trainer's "
        "weight_cap=15 is tuned for the natural span~3.5% distribution.",
    )


def apply_caps_and_report(
    examples: list[dict],
    args,
    label_key: str | None = None,
    *,
    allow_label_protection: bool = True,
) -> tuple[list[dict], dict[str, dict]]:
    """Apply the cap flags from ``args`` to ``examples``; print the report.

    ``label_key`` is the builder's primary multiclass label field (e.g.
    'doc_role'); pass it to enable rare-class protection. No-op (returns
    examples unchanged, empty report) when neither --max-examples-per-source
    nor --source-cap is set.

    ``allow_label_protection=False`` opts a builder out of rare-label
    protection (its trainer owns class balance — e.g. table_specialist's
    weight_cap=15 is tuned for the natural distribution). If protection is
    requested anyway (--cap-protect-frac > 0), this raises
    :class:`ProtectionNotAllowedError` rather than silently ignoring the flag
    (see feedback_no_silent_fallbacks).
    """
    overrides = parse_source_caps(getattr(args, "source_cap", []))
    default_cap = getattr(args, "max_examples_per_source", None)
    if default_cap is None and not overrides:
        return examples, {}
    seed = getattr(args, "seed", 42)
    frac = getattr(args, "cap_protect_frac", 0.0)
    protect = bool(label_key) and frac > 0
    if frac > 0 and not allow_label_protection:
        raise ProtectionNotAllowedError(
            f"--cap-protect-frac={frac} requested, but the {label_key!r} builder "
            "opts out of rare-label protection (its trainer handles class "
            "balance). Re-run with --cap-protect-frac 0."
        )
    capped, report = cap_examples_per_source(
        examples,
        default_cap,
        overrides,
        seed=seed,
        label_key=label_key if protect else None,
        protect_below_frac=frac,
    )
    if protect:
        prot = rare_labels(examples, label_key, frac)
        print(
            f"\n[cap] protecting rare {label_key} labels "
            f"(global share < {frac:.0%}): {sorted(prot)}",
            file=sys.stderr,
        )
    elif label_key:
        print(
            "\n[cap] rare-label protection OFF (--cap-protect-frac=0); "
            "uniform per-source subsample (class balance left to the trainer)",
            file=sys.stderr,
        )
    print("[cap] per-source example caps applied:", file=sys.stderr)
    for src, info in sorted(report.items(), key=lambda x: -x[1]["before"]):
        tag = "" if info["after"] == info["before"] else "  <-- capped"
        extra = f"  [{info['protected_kept']} protected kept]" if info.get("protected_kept") else ""
        print(
            f"  {src:24} {info['before']:8} -> {info['after']:8}  (cap={info['cap']}){tag}{extra}",
            file=sys.stderr,
        )
        # A source may exceed its cap when its protected (rare-label) examples
        # alone overflow it — by design, but surface it loudly (don't let a
        # bounded cap look enforced when it wasn't).
        if info["cap"] is not None and info["after"] > info["cap"]:
            print(
                f"  [WARN] {src}: kept {info['after']} EXCEEDS cap {info['cap']} — "
                f"{info['protected_kept']} protected rare-label examples overflow it",
                file=sys.stderr,
            )
    print(f"[cap] total after capping: {len(capped)}", file=sys.stderr)
    return capped, report
