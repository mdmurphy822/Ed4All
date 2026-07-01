#!/usr/bin/env python3
"""Calibration harness — the keystone measurement tool for the framework gate-family
critical-flips.

PURPOSE
-------
Across the Ed4All instruction-block (IB) roadmap, ~9 framework gate families ship at
``warning`` severity carrying a ``# TODO(calibration)`` marker in
``config/workflows.yaml``. Every one of those deferred critical-flips is blocked on the
SAME missing input: a per-gate FIRE-RATE measurement across >= 2 corpora (the precursor
to a manual false-positive audit). This tool produces that measurement from REAL run
artifacts. It NEVER mutates gates, flips severities, or fabricates data.

WHAT IT READS (real artifacts only, discovered dynamically — no hardcoded slugs)
--------------------------------------------------------------------------------
For every discovered corpus (a LibV2 course slug or a Courseforge project export):

1. ``<course>/block_quality_rollup_report.json`` — BlockQualityRollupAggregator output
   (schema schemas/aggregators/block_quality_rollup.schema.json). Only present when
   ED4ALL_BLOCK_QUALITY_RUBRIC was on. Source for the IB6 rubric/anatomy/feedback gates.
2. Courseforge ``02_validation_report/report.json`` — per-block GateResult summaries
   (``per_block[].gate_results[]``: gate_id / passed / issue_count / action). This is
   the richest, always-present source for outline/rewrite-tier gate fire-rates.
3. Decision-capture JSONL under ``training-captures/*/<COURSE>/`` carrying
   ``block_validation_action`` / ``statistical_validation_*`` events (gate-level rollups
   with ``ml_features.gate_id`` / ``passed`` / ``block_count`` / ``issues_count``).
4. ``<export>/courseforge_validation_report.json`` (the post-loop aggregator) —
   ``per_phase[].gates[]``. This is the ONLY on-disk artifact that surfaces gates wired
   at the PRE-Courseforge phases (``chunking`` / ``course_planning``), which own no
   standalone ``02_validation_report``. Read for the three aggregator-only families
   (chunk WCAG status / CO<->TO alignment / source->objective coverage); a full
   ``textbook_to_course`` run emits it, a content-only two-pass slice does not.

A corpus missing ALL four sources for a gate is SKIPPED for that gate with a logged
reason (no fabricated data).

MULTI-CORPUS DISCOVERY (W8.3)
-----------------------------
The whole point of the harness is the >=2-DISTINCT-corpora precondition every deferred
flip is blocked on. Discovery enumerates every LibV2 course + Courseforge export that
carries a readable artifact (reusing the W0.8 master catalog to seed the LibV2 slug set
where present, unioned with the filesystem scan so an un-cataloged course is never
missed), collapses run-variants of one textbook to a single ``corpus_key``, and is
DISCOVER-OR-SKIP: a corpus with no readable artifact is skipped with a logged reason,
and when fewer than 2 DISTINCT corpora contribute, ``sufficient_corpora`` is False and
every gate stays ``flip_ready=false`` with a clear message (a 2nd-corpus run unblocks
the flips). The operator can point the harness at ADDITIONAL local corpus roots via the
``ED4ALL_CALIB_EXTRA_CORPORA`` env var (os.pathsep/comma-separated dirs; the ``--corpora``
CLI flag wins when both are set) — each extra root's immediate children are treated as
corpora (export-shaped, mirroring ``--runs-dir``). ``--max-corpora`` caps the distinct
set (keeps the highest-coverage corpora). Both are opt-in, parse-with-fallback (a
missing/garbage path is skipped, never fatal). No hardcoded slug; no tracked bytes.

WHAT IT EMITS
-------------
``calibration_report.json`` (path overridable) + a readable stdout summary. Per gate:
  {fire_rate, corpora[], sample[], expected_band, corpora_count, across_corpora,
   flip_ready, flip_blocked_reason}

``across_corpora`` is the W8.3 per-gate ACROSS-CORPORA aggregate — {n_corpora,
max_fire_rate, min_fire_rate, mean_fire_rate, pooled_fire_rate, max_in_band,
worst_corpus}. The MAX per-corpus fire-rate (worst-corpus FP signal) is the number an
operator reads to decide a flip: a gate can look clean POOLED yet spike on ONE corpus,
which is exactly the per-corpus false-positive a flip must not miss.

``flip_ready`` is a HEURISTIC, not an automatic FP verdict — FP rate needs ground truth
the harness cannot synthesize. It is True only when (corpora_count >= 2 AND the pooled
fire_rate is in the gate's DOCUMENTED expected band AND the WORST-corpus fire_rate is
also in band). The harness surfaces the measurement + the criterion; a downstream
human/adjudication step decides the flip.

USAGE
-----
    python3 scripts/calibration_harness.py
    python3 scripts/calibration_harness.py --course sample-course-a
    python3 scripts/calibration_harness.py --runs-dir Courseforge/exports
    python3 scripts/calibration_harness.py --corpora /data/extra_lib --max-corpora 4
    python3 scripts/calibration_harness.py --out /tmp/calibration_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

LOG = logging.getLogger("calibration_harness")

# --------------------------------------------------------------------------------------
# Per-gate flip-criteria config table.
#
# Each entry is a framework gate FAMILY carrying a `# TODO(calibration)` deferred flip in
# config/workflows.yaml. ``gate_ids`` lists the concrete gate_ids that family appears
# under across the outline/rewrite tiers and both workflows (course_generation +
# textbook_to_course). ``expected_band`` is the documented fire-rate window a clean
# corpus should land in; ``band_source`` records WHERE the band was sourced so the table
# is auditable. The flip-criterion is intentionally NOT a hardcoded pass/fail — the
# harness reports the measurement against the band and lets a human adjudicate.
#
# Band semantics: ``fire_rate`` = fraction of evaluated blocks (or modules, for
# module-scoped gates) on which the gate FIRED (passed=False / issue present). A gate is
# "in band" when min <= fire_rate <= max. A LOW band (e.g. <= 0.05) means "a clean corpus
# should rarely fire this; persistent firing = real defect, FP audit needed before flip".
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateFamily:
    family: str
    gate_ids: tuple[str, ...]
    expected_band: tuple[float, float]  # (min_fire_rate, max_fire_rate) for "in band"
    band_source: str
    scope: str = "block"  # "block" or "module"
    notes: str = ""


_CURATED_GATE_FAMILIES: tuple[GateFamily, ...] = (
    GateFamily(
        family="IB3 verb-triple / anchored-rubric / triangle (constructive-alignment keystone)",
        gate_ids=(
            "outline_anchored_rubric",
            "rewrite_anchored_rubric",
            "outline_triangle_completeness",
            "rewrite_triangle_completeness",
        ),
        expected_band=(0.0, 0.05),
        band_source=(
            "config/workflows.yaml IB3 FAST-FLIP marker (lines ~160-200): the keystone "
            "validity rule flips critical after ONE >=2-corpus manual FP audit clears; a "
            "well-aligned corpus should rarely trip it (low band)."
        ),
        notes="ACCELERATED fast-flip: needs only ONE >=2-corpus FP measurement (not multi-wave).",
    ),
    GateFamily(
        family="IB4 UDL multiple-means coverage",
        gate_ids=("udl_coverage",),
        expected_band=(0.0, 0.10),
        band_source=(
            "config/workflows.yaml IB4.5 marker (~line 205): flip UDL_SINGLE_REPRESENTATION "
            "to critical after a >=2-corpus FP measurement (WS3/W4 deferred-flip pattern)."
        ),
    ),
    GateFamily(
        family="IB4 chunk WCAG status",
        gate_ids=("chunk_wcag_status",),
        expected_band=(0.0, 0.10),
        band_source=(
            "config/workflows.yaml IB4.2 marker (~line 1150): flip to critical once a "
            ">=2-corpus measurement shows the flagged-rate is a true defect signal, not "
            "SemantiK-noise."
        ),
    ),
    GateFamily(
        family="IB5 multimedia/diagram a11y contract (rides rewrite_html_shape)",
        gate_ids=("rewrite_html_shape",),
        expected_band=(0.0, 0.05),
        band_source=(
            "config/workflows.yaml IB5 marker (~line 397): flip the IB5 multimedia/diagram "
            "REWRITE_IB5_A11Y_CONTRACT sub-check to critical after a >=2-corpus FP "
            "measurement. NOTE: rewrite_html_shape is ALREADY critical for its parse-fail "
            "arm; the calibration is for the IB5 a11y SUB-check only — fire_rate here is an "
            "upper bound (whole-gate), interpret with care."
        ),
    ),
    GateFamily(
        family="IB6.4 block cognitive load (D2 body ceiling)",
        gate_ids=("block_cognitive_load",),
        expected_band=(0.0, 0.10),
        band_source=(
            "config/workflows.yaml IB6 marker (~line 218): hard-gate critical flip deferred "
            "until the anchored 0-3 scale is calibrated on >=2 corpora (mean floors would "
            "block early runs)."
        ),
    ),
    GateFamily(
        family="IB6.3 interaction-feedback presence",
        gate_ids=("interaction_feedback",),
        expected_band=(0.0, 0.10),
        band_source="config/workflows.yaml IB6 marker (~line 218): same deferred 0-3 calibration block.",
    ),
    GateFamily(
        family="IB6.1 eight-dimension block-quality rubric",
        gate_ids=("block_quality_rubric",),
        expected_band=(0.0, 0.10),
        band_source=(
            "config/workflows.yaml IB6 marker (~line 218) + schemas/aggregators/"
            "block_quality_rollup.schema.json: mean>=2.0 floors + Accessibility=0 block-fail."
        ),
    ),
    GateFamily(
        family="IB6.7 15-point QA checklist",
        gate_ids=("qa_checklist",),
        expected_band=(0.0, 0.10),
        band_source="config/workflows.yaml IB6 marker (~line 218): same deferred 0-3 calibration block.",
    ),
    GateFamily(
        family="IB7.5b retrieval presence (module-scoped)",
        gate_ids=("retrieval_presence",),
        expected_band=(0.0, 0.05),
        band_source=(
            "config/workflows.yaml IB7.5b marker (~line 277): flip MODULE_NO_RETRIEVAL / "
            "RETRIEVAL_UNSPACED to critical after a >=2-corpus FP measurement."
        ),
        scope="module",
    ),
    GateFamily(
        family="WS3 CO<->TO semantic alignment",
        gate_ids=("co_terminal_alignment",),
        expected_band=(0.0, 0.10),
        band_source=(
            "config/workflows.yaml WS3 marker (~line 1430): promote to critical after WS1 "
            "proves the recomputed weak-link rate <= 0.10 on >=2 corpora (documented "
            "numeric target)."
        ),
    ),
    GateFamily(
        family="WS6a/I3 source->objective coverage",
        gate_ids=("source_coverage", "objective_source_refs"),
        expected_band=(0.0, 0.05),
        band_source=(
            "config/workflows.yaml WS6a marker (~line 1460): de-risk proof shows 135/135 "
            "sections covered (clean pass); calibrate floor UP toward ~0.55 on >=2 corpora "
            "before any critical flip. Low band = clean corpus rarely fires."
        ),
    ),
)

# --------------------------------------------------------------------------------------
# W8.2 — reconcile GATE_FAMILIES with the LIVE deferred-flip gates in
# config/workflows.yaml so the curated table above can NEVER silently drift behind the
# config again. The config is the source of truth: every gate carrying a
# ``# TODO(calibration)`` deferred-flip marker MUST be measured. Curated families (with a
# documented ``expected_band``) are preferred; any deferred gate the curated table does
# NOT cover is auto-synthesized into a one-gate fallback family (uncalibrated default
# band) so it still accrues fire-rate data. This is measurement wiring only — it NEVER
# flips a gate (the harness has no flip authority); a human still adjudicates every flip.
# --------------------------------------------------------------------------------------

#: Deferred gates the curated table intentionally does NOT own as a standalone family but
#: that ARE covered as a sibling gate_id inside a curated family (so they are NOT
#: "missing" and must not be double-counted as an auto family). Kept explicit so the
#: reconcile test can assert curated coverage is deliberate, not accidental.
#: (Populated at import from the curated families' gate_ids — see below.)

#: Module/course-scoped auto-synthesized gates. Their GateResults surface at the
#: phase level (course/module rollups), NOT per-block, so the reader must treat them as
#: module-scoped or it will look for them in ``per_block[]`` and never observe them.
_AUTO_MODULE_SCOPED_GATES: frozenset[str] = frozenset(
    {
        "course_level_qa",
        "cross_week_spacing",
        "cumulative_assessment",
        "bloom_distribution",
        "prerequisite_sequencing",
        "block_quality_rollup",
        "terminal_objective_coverage",
        "terminal_objective_source_grounding",
        "chunkset_manifest",
        "manifest_completeness",
        "libv2_manifest",
        "page_objectives_shacl",
        "difficulty_provenance",
    }
)

# Regex reused by the config parser (module scope so it compiles once).
_GATE_ID_RE = re.compile(r"gate_id:\s*([A-Za-z0-9_\-]+)")
_CAL_MARKER = "TODO(calibration)"


def _workflows_config_path() -> Path:
    """Resolve config/workflows.yaml relative to the repo root.

    Computes the repo root inline (``scripts/`` is a direct child of the repo root)
    rather than calling ``_repo_root`` — this runs at import time, BEFORE that helper is
    defined further down the module.
    """
    return Path(__file__).resolve().parent.parent / "config" / "workflows.yaml"


def deferred_flip_gate_ids_from_config(
    config_path: Path | None = None,
) -> frozenset[str]:
    """Parse config/workflows.yaml for every gate carrying a deferred-flip marker.

    A gate is "deferred" when a ``# TODO(calibration)`` marker is associated with its
    entry — either in the contiguous comment block IMMEDIATELY above its ``- gate_id:``
    line, or inside its own (non-comment) ``description:`` field. The trailing comment
    block that precedes the NEXT gate is deliberately excluded (it belongs to that next
    gate), so a gate is never falsely tagged from its neighbour's marker.

    Text-based (no PyYAML dependency, no ordering assumptions beyond the file's own
    "comment-above-the-gate" convention). Fully graceful: any read/parse error returns an
    empty set so the harness degrades to the curated table rather than crashing.
    """
    path = config_path or _workflows_config_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # noqa: BLE001 — degrade, never crash the harness
        LOG.warning("could not read %s for deferred-gate reconciliation: %s", path, exc)
        return frozenset()

    gate_line_idx = [
        i for i, ln in enumerate(lines) if re.match(r"\s*- gate_id:", ln)
    ]
    deferred: set[str] = set()
    for k, gi in enumerate(gate_line_idx):
        m = _GATE_ID_RE.search(lines[gi])
        if not m:
            continue
        gid = m.group(1)
        end = gate_line_idx[k + 1] if k + 1 < len(gate_line_idx) else len(lines)
        # Contiguous preceding comment block (this gate's own comment).
        p = gi - 1
        pre: list[str] = []
        while p >= 0 and lines[p].strip().startswith("#"):
            pre.append(lines[p])
            p -= 1
        # Non-comment body of THIS entry (excludes trailing comments = next gate's).
        body_noncomment = [
            ln for ln in lines[gi:end] if not ln.strip().startswith("#")
        ]
        assoc = "\n".join(pre) + "\n" + "\n".join(body_noncomment)
        if _CAL_MARKER in assoc:
            deferred.add(gid)
    return frozenset(deferred)


def _synthesize_missing_families(
    curated: tuple[GateFamily, ...],
    config_deferred_ids: frozenset[str],
) -> tuple[GateFamily, ...]:
    """Auto-build a fallback family per config-deferred gate the curated table misses.

    Prevents GATE_FAMILIES from drifting behind config: every deferred gate is measured,
    even before someone curates a documented band for it. The auto family carries a
    conservative default band + a ``band_source`` that flags it as UNCALIBRATED so the
    report never implies a real band was established.
    """
    covered = {gid for fam in curated for gid in fam.gate_ids}
    out: list[GateFamily] = []
    for gid in sorted(config_deferred_ids - covered):
        out.append(
            GateFamily(
                family=f"(auto) {gid} — deferred critical-flip",
                gate_ids=(gid,),
                # UNCALIBRATED default — a documented band must be curated before a flip.
                expected_band=(0.0, 0.10),
                band_source=(
                    "AUTO-DERIVED from config/workflows.yaml # TODO(calibration) marker "
                    f"on gate {gid!r} (W8.2 drift-guard). expected_band is a placeholder "
                    "(0.0-0.10) — NOT a documented calibration target; curate a real band "
                    "in _CURATED_GATE_FAMILIES before any flip adjudication."
                ),
                scope="module" if gid in _AUTO_MODULE_SCOPED_GATES else "block",
                notes="auto-synthesized to keep GATE_FAMILIES reconciled with config (W8.2).",
            )
        )
    return tuple(out)


#: The LIVE deferred-flip gate set (empty if config is unreadable — degrade to curated).
CONFIG_DEFERRED_GATE_IDS: frozenset[str] = deferred_flip_gate_ids_from_config()

#: The final, drift-reconciled family table: curated (documented bands) + auto-synthesized
#: fallbacks for any config-deferred gate the curated table does not already cover.
GATE_FAMILIES: tuple[GateFamily, ...] = _CURATED_GATE_FAMILIES + _synthesize_missing_families(
    _CURATED_GATE_FAMILIES, CONFIG_DEFERRED_GATE_IDS
)

# Flat gate_id -> family lookup for the aggregation pass.
_GATE_TO_FAMILY: dict[str, GateFamily] = {}
for _fam in GATE_FAMILIES:
    for _gid in _fam.gate_ids:
        _GATE_TO_FAMILY[_gid] = _fam

ALL_CALIBRATION_GATE_IDS: frozenset[str] = frozenset(_GATE_TO_FAMILY)

MAX_SAMPLE_PER_FAMILY = 8  # bounded sample of fired blocks per gate family


# --------------------------------------------------------------------------------------
# Per-corpus / per-gate accumulators
# --------------------------------------------------------------------------------------
@dataclass
class GateObservation:
    """One gate family's tally within ONE corpus."""

    fired: int = 0
    evaluated: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)

    @property
    def fire_rate(self) -> float:
        return (self.fired / self.evaluated) if self.evaluated else 0.0


@dataclass
class CorpusResult:
    corpus_id: str
    origin: str  # "libv2" | "courseforge_export"
    path: str
    corpus_key: str = ""  # underlying-corpus identity (timestamp/run-variant stripped)
    sources_read: list[str] = field(default_factory=list)
    skip_reasons: list[str] = field(default_factory=list)
    # family name -> GateObservation
    observations: dict[str, GateObservation] = field(default_factory=dict)

    def obs(self, family_name: str) -> GateObservation:
        return self.observations.setdefault(family_name, GateObservation())


# --------------------------------------------------------------------------------------
# Path resolution (relocatable; honors lib.paths if available, else repo-relative)
# --------------------------------------------------------------------------------------
def _repo_root() -> Path:
    # scripts/ is a direct child of the repo root.
    return Path(__file__).resolve().parent.parent


def _libv2_courses_root() -> Path:
    try:
        from lib.paths import get_libv2_root  # type: ignore

        return Path(get_libv2_root()) / "courses"
    except Exception:
        return _repo_root() / "LibV2" / "courses"


def _courseforge_exports_root() -> Path:
    return _repo_root() / "Courseforge" / "exports"


def _training_captures_root() -> Path:
    try:
        from lib.paths import get_training_captures_dir  # type: ignore

        return Path(get_training_captures_dir())
    except Exception:
        return _repo_root() / "training-captures"


def _libv2_root() -> Path:
    try:
        from lib.paths import get_libv2_root  # type: ignore

        return Path(get_libv2_root())
    except Exception:
        return _repo_root() / "LibV2"


# --------------------------------------------------------------------------------------
# W8.3 — extra-corpora env knob + W0.8 master-catalog reuse (multi-corpus discovery)
# --------------------------------------------------------------------------------------
ENV_EXTRA_CORPORA = "ED4ALL_CALIB_EXTRA_CORPORA"
#: Separators accepted inside ED4ALL_CALIB_EXTRA_CORPORA (os.pathsep + comma + newline).
_EXTRA_CORPORA_SEP_RE = re.compile(r"[,\n" + re.escape(os.pathsep) + r"]")


def resolve_extra_corpora_roots(
    explicit: Iterable[str] | None = None,
) -> list[Path]:
    """Resolve extra corpus-discovery roots (opt-in, parse-with-fallback).

    Precedence (high -> low): the ``--corpora`` CLI list (``explicit``) > the
    ``ED4ALL_CALIB_EXTRA_CORPORA`` env var (os.pathsep/comma/newline-separated). Every
    entry is expanded + resolved; a path that is not an existing directory is skipped
    with a warning (never fatal — mirrors the other ED4ALL_* parse-with-fallback knobs).
    De-duplicates while preserving first-seen order. Returns ``[]`` when unset.
    """
    raw_items: list[str] = []
    if explicit:
        raw_items.extend(str(x) for x in explicit)
    else:
        env = os.environ.get(ENV_EXTRA_CORPORA)
        if env:
            raw_items.extend(_EXTRA_CORPORA_SEP_RE.split(env))

    roots: list[Path] = []
    seen: set[Path] = set()
    for item in raw_items:
        item = item.strip()
        if not item:
            continue
        try:
            rp = Path(item).expanduser().resolve()
        except Exception:  # noqa: BLE001 — a malformed path is skipped, never fatal
            LOG.warning("ignoring unparseable extra-corpora path: %r", item)
            continue
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_dir():
            roots.append(rp)
        else:
            LOG.warning(
                "%s path is not a directory, skipping: %s", ENV_EXTRA_CORPORA, rp
            )
    return roots


def _catalog_course_names(libv2_root: Path) -> list[str]:
    """W0.8 master-catalog course slugs (empty on absence/error). Graceful.

    Reuses ``LibV2.tools.libv2.catalog.load_master_catalog`` (the same loader
    ``lib/retrieval/library_wide._catalog_slugs`` consults) to SEED the LibV2 course set
    from the built catalog. Unioned with the filesystem scan by the caller, so a course
    present in the catalog but not yet on disk (or vice-versa) is still discovered. Any
    import/read error returns ``[]`` — the filesystem scan remains the source of truth.
    """
    try:
        from LibV2.tools.libv2.catalog import load_master_catalog  # type: ignore

        catalog = load_master_catalog(libv2_root)
    except Exception:  # noqa: BLE001 — catalog absence is expected; degrade to fs scan
        return []
    if catalog is None:
        return []
    names: list[str] = []
    for entry in getattr(catalog, "courses", []) or []:
        slug = getattr(entry, "slug", None)
        if slug:
            names.append(str(slug))
    return names


# --------------------------------------------------------------------------------------
# Readers — each returns observations attributed to the supplied CorpusResult
# --------------------------------------------------------------------------------------
def _rel_to_repo(path: Path) -> str:
    """Repo-relative string for bookkeeping; falls back to the absolute path for
    arbitrary (e.g. test tmp) locations outside the repo."""
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash a corpus
        LOG.warning("could not read %s: %s", path, exc)
        return None


def _record_fire(
    res: CorpusResult,
    family: GateFamily,
    *,
    fired: bool,
    evaluated: int,
    source: str,
    sample: dict[str, Any] | None = None,
) -> None:
    obs = res.obs(family.family)
    obs.evaluated += evaluated
    obs.sources.add(source)
    if fired:
        obs.fired += evaluated if family.scope == "module" else 1
        if sample is not None and len(obs.samples) < MAX_SAMPLE_PER_FAMILY:
            obs.samples.append(sample)


def read_validation_report(res: CorpusResult, report_path: Path) -> bool:
    """Read a Courseforge 02_validation_report/report.json GateResult list.

    Two sources, by gate scope (schema v2 — the per-block ``issue_count``
    attribution fix):

    * BLOCK-scoped families read ``per_block[].gate_results[]``. The
      per-gate ``passed`` flag in that array is PHASE-level (the same on
      every block), so it can NOT distinguish which block actually
      tripped a gate — under the schema-v1 bug it smeared one phase
      verdict across all blocks and inflated fire-rates toward 100%.
      Instead, a block "fires" a block-scoped gate iff its OWN
      ``issue_count`` (now correctly attributed to that block's id) is
      > 0. Each block is one ``evaluated`` unit.

    * OBJECTIVE / MODULE / summary (non-block-attributable) families read
      the schema-v2 ``phase_level_gate_results[]`` section, which carries
      per gate the total ``issue_count`` + ``unattributed_issue_count``
      (issues whose location matched no block_id). A module-scoped family
      fires once with the gate's phase-level ``passed`` verdict (the
      existing module-scope contract). Gracefully degrades when the
      section is absent (legacy v1 report) — logged, skipped.

    Returns True if any calibration gate was observed.
    """
    data = _read_json(report_path)
    if not isinstance(data, dict):
        return False
    per_block = data.get("per_block")
    if not isinstance(per_block, list):
        return False

    src = f"validation_report:{report_path.name}"
    saw_any = False

    # --- BLOCK-scoped: fire per block on that block's OWN issue_count. ---
    for block in per_block:
        if not isinstance(block, dict):
            continue
        block_id = block.get("block_id") or block.get("page") or "<unknown>"
        for gr in block.get("gate_results", []) or []:
            if not isinstance(gr, dict):
                continue
            gid = gr.get("gate_id")
            fam = _GATE_TO_FAMILY.get(str(gid))
            if fam is None:
                continue
            # Module/objective-scoped families are read from the
            # phase-level section below, not smeared per block.
            if fam.scope == "module":
                continue
            saw_any = True
            try:
                issue_count = int(gr.get("issue_count") or 0)
            except (TypeError, ValueError):
                issue_count = 0
            fired = issue_count > 0
            sample = None
            if fired:
                sample = {
                    "corpus": res.corpus_id,
                    "block_id": block_id,
                    "gate_id": gid,
                    "issue_count": issue_count,
                    "action": gr.get("action"),
                    "source": src,
                }
            _record_fire(
                res, fam, fired=fired, evaluated=1, source=src, sample=sample
            )

    # --- Structural / module-scoped: read the phase-level section. ---
    phase_level = data.get("phase_level_gate_results")
    if isinstance(phase_level, list):
        for gr in phase_level:
            if not isinstance(gr, dict):
                continue
            gid = gr.get("gate_id")
            fam = _GATE_TO_FAMILY.get(str(gid))
            if fam is None:
                continue
            # Only structural (module/objective-scoped) families come
            # from here; block-scoped gates were already attributed per
            # block above and must not be double-counted.
            if fam.scope != "module":
                continue
            saw_any = True
            try:
                total_issues = int(gr.get("issue_count") or 0)
            except (TypeError, ValueError):
                total_issues = 0
            passed = bool(gr.get("passed", True))
            # A module-scoped gate fires when the phase verdict is fail
            # OR it logged any (structural) issue. _record_fire counts a
            # module-scoped fire as the whole evaluated unit.
            fired = (not passed) or total_issues > 0
            sample = None
            if fired:
                sample = {
                    "corpus": res.corpus_id,
                    "gate_id": gid,
                    "issue_count": total_issues,
                    "unattributed_issue_count": gr.get(
                        "unattributed_issue_count"
                    ),
                    "action": gr.get("action"),
                    "source": f"{src}:phase_level",
                }
            _record_fire(
                res, fam, fired=fired, evaluated=1,
                source=f"{src}:phase_level", sample=sample,
            )
    else:
        # Legacy v1 report (no phase-level section). Module-scoped
        # structural families simply aren't observed from this report;
        # the decision-capture reader remains their fallback source.
        LOG.debug(
            "validation_report %s has no phase_level_gate_results "
            "(legacy v1); structural gates skipped here",
            report_path.name,
        )

    if saw_any:
        res.sources_read.append(_rel_to_repo(report_path))
    return saw_any


def read_block_quality_rollup(res: CorpusResult, rollup_path: Path) -> bool:
    """Read a block_quality_rollup_report.json (IB6.6 aggregator output).

    A block "fires" the rubric family when block_pass is False; the accessibility
    hard-gate maps onto accessibility_gate_fail; alignment orphan maps onto alignment==0.
    """
    data = _read_json(rollup_path)
    if not isinstance(data, dict):
        return False
    per_block = data.get("per_block")
    if not isinstance(per_block, list):
        return False

    src = "block_quality_rollup_report.json"
    rubric_fam = _GATE_TO_FAMILY.get("block_quality_rubric")
    if rubric_fam is None:
        return False
    saw_any = False
    for block in per_block:
        if not isinstance(block, dict):
            continue
        saw_any = True
        fired = not bool(block.get("block_pass", True))
        sample = None
        if fired:
            sample = {
                "corpus": res.corpus_id,
                "block_id": block.get("block_id"),
                "module": block.get("module"),
                "mean": block.get("mean"),
                "weakest_dim": block.get("weakest_dim"),
                "accessibility_gate_fail": block.get("accessibility_gate_fail"),
                "source": src,
            }
        _record_fire(
            res, rubric_fam, fired=fired, evaluated=1, source=src, sample=sample
        )
    if saw_any:
        res.sources_read.append(_rel_to_repo(rollup_path))
    return saw_any


# Calibration gate families whose gates run at PRE-Courseforge phases
# (``chunking`` / ``objective_extraction`` / ``course_planning`` /
# ``concept_extraction``) rather than the two-pass outline/rewrite tiers. Those
# phases own NO on-disk ``02_validation_report/report.json`` — their per-phase
# GateResults are surfaced ONLY in the post-loop ``courseforge_validation_report.json``
# aggregator (``per_phase[].gates[]``), which a full ``textbook_to_course`` run emits.
# The per-block / phase-level validation-report readers above therefore never see them.
# These three families are read from the aggregator instead, as ONE phase-level
# observation per corpus that fires iff the gate logged any issue (``issue_count > 0``);
# ``passed`` is NOT the signal because these gates ship warning-day-1 (passed stays True
# even when issues are present, mirroring the per-block reader's ``issue_count > 0``
# fire rule).
_AGGREGATOR_ONLY_GATE_IDS: frozenset[str] = frozenset(
    {
        "chunk_wcag_status",        # IB4 chunk WCAG status (chunking phase)
        "co_terminal_alignment",    # WS3 CO<->TO semantic alignment (course_planning)
        "source_coverage",          # WS6a/I3 source->objective coverage (course_planning)
        "objective_source_refs",    # WS6a/I3 source->objective coverage (course_planning)
    }
)


def read_courseforge_validation_report(res: CorpusResult, report_path: Path) -> bool:
    """Read the post-loop ``courseforge_validation_report.json`` aggregator.

    A full ``textbook_to_course`` run walks every per-phase report and folds it into
    ``per_phase[].gates[]``, where each entry carries ``gate_id`` / ``issue_count`` /
    ``passed`` / ``severity`` for a PHASE-level gate. This is the only on-disk artifact
    that surfaces the gates wired at the PRE-Courseforge phases (``chunking`` /
    ``course_planning``) — those phases own no standalone ``02_validation_report``.

    To avoid double-counting gates the richer per-block / rollup / phase-level readers
    already observed, this reader records ONLY families in
    ``_AGGREGATOR_ONLY_GATE_IDS`` and ONLY when the family has no observation yet for
    this corpus. Each such gate is one phase-level ``evaluated`` unit that fires iff its
    ``issue_count > 0`` (warning-day-1: ``passed`` stays True even with issues, so it is
    NOT the fire signal). Returns True if any calibration gate was observed.
    """
    data = _read_json(report_path)
    if not isinstance(data, dict):
        return False
    per_phase = data.get("per_phase")
    if not isinstance(per_phase, list):
        return False

    src = f"courseforge_validation_report:{report_path.name}"
    saw_any = False
    # Families a RICHER reader (per-block / rollup / phase-level) already recorded for
    # this corpus before the aggregator reader ran. The aggregator must not override
    # those. Two distinct aggregator-only gate_ids that map to the SAME family (e.g.
    # source_coverage + objective_source_refs) DO each contribute a phase-level
    # observation, so this snapshot is taken ONCE up front (not re-checked per gate).
    pre_observed = {
        name for name, o in res.observations.items() if o.evaluated > 0
    }
    for phase_entry in per_phase:
        if not isinstance(phase_entry, dict):
            continue
        phase_name = phase_entry.get("phase")
        for gr in phase_entry.get("gates", []) or []:
            if not isinstance(gr, dict):
                continue
            gid = gr.get("gate_id")
            if str(gid) not in _AGGREGATOR_ONLY_GATE_IDS:
                continue
            fam = _GATE_TO_FAMILY.get(str(gid))
            if fam is None:
                continue
            # Don't override a richer per-block/phase-level observation of the
            # SAME family already recorded for this corpus by another reader.
            if fam.family in pre_observed:
                continue
            saw_any = True
            try:
                issue_count = int(gr.get("issue_count") or 0)
            except (TypeError, ValueError):
                issue_count = 0
            fired = issue_count > 0
            sample = None
            if fired:
                top_issues = gr.get("top_issues") or []
                first_loc = None
                if isinstance(top_issues, list) and top_issues:
                    first = top_issues[0]
                    if isinstance(first, dict):
                        first_loc = first.get("location")
                sample = {
                    "corpus": res.corpus_id,
                    "phase": phase_name,
                    "gate_id": gid,
                    "issue_count": issue_count,
                    "action": gr.get("action"),
                    "first_issue_location": first_loc,
                    "source": src,
                }
            _record_fire(res, fam, fired=fired, evaluated=1, source=src, sample=sample)

    if saw_any:
        res.sources_read.append(_rel_to_repo(report_path))
    return saw_any


def read_decision_captures(res: CorpusResult, capture_files: Iterable[Path]) -> bool:
    """Read block_validation_action / statistical_validation_* JSONL events.

    These are gate-level rollups: one event per (phase, gate_id) carrying
    ml_features.{gate_id, passed, block_count}. We attribute them at MODULE granularity
    only for module-scoped families; for block-scoped families they are a secondary
    fire-signal (a gate that returned passed=False against N blocks) used ONLY when the
    validation report did not already cover the gate (avoids double counting).
    """
    saw_any = False
    # Track which families the richer report-based reader already covered so we don't
    # double-count from the coarser JSONL rollups.
    already_covered = {
        name for name, o in res.observations.items() if o.evaluated > 0
    }
    files_used: set[str] = set()
    for fpath in capture_files:
        try:
            with fpath.open(encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not read capture %s: %s", fpath, exc)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:  # noqa: BLE001 — skip malformed lines
                continue
            dtype = evt.get("decision_type")
            if dtype not in ("block_validation_action", "statistical_validation_action",
                             "statistical_validation"):
                continue
            mf = evt.get("ml_features") or {}
            gid = mf.get("gate_id")
            fam = _GATE_TO_FAMILY.get(str(gid))
            if fam is None:
                continue
            # Only use JSONL for families the structured report missed (e.g. IB6 gates
            # absent from per_block, or module-scoped retrieval_presence).
            if fam.family in already_covered:
                continue
            passed = mf.get("passed")
            if passed is None:
                continue  # advisory/measure-only event, not a pass/fail signal
            block_count = mf.get("block_count") or mf.get("blocks_evaluated") or 1
            try:
                block_count = int(block_count)
            except (TypeError, ValueError):
                block_count = 1
            fired = passed is False
            saw_any = True
            files_used.add(_rel_to_repo(fpath))
            sample = None
            if fired:
                rationale = evt.get("rationale") or ""
                sample = {
                    "corpus": res.corpus_id,
                    "gate_id": gid,
                    "phase": evt.get("phase"),
                    "issues_count": mf.get("issues_count"),
                    "block_count": block_count,
                    "rationale_excerpt": rationale[:240],
                    "source": "decision_capture",
                }
            _record_fire(
                res,
                fam,
                fired=fired,
                evaluated=block_count,
                source="decision_capture",
                sample=sample,
            )
    if files_used:
        res.sources_read.extend(sorted(files_used))
    return saw_any


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------
def _course_token(name: str) -> str:
    """Normalize a course/export name to a comparable token for --course filtering."""
    n = name.lower()
    for prefix in ("proj-",):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


# Run-variant suffixes that denote the SAME underlying source corpus generated a
# different way (model size, full vs. slice). The ">=2 corpora" calibration requirement
# means two DISTINCT source corpora — NOT two runs of one. corpus_key collapses these.
_RUN_VARIANT_SUFFIXES = ("-full7b", "-7b", "-full", "-sonnet", "-70b")
_TIMESTAMP_RE = re.compile(r"-?\d{8,}(?:_\d+)*$")


def _corpus_key(name: str) -> str:
    """Derive the underlying-corpus identity from a course slug / export dir name.

    Strips the ``PROJ-`` prefix, a trailing run timestamp, and known run-variant
    suffixes so that e.g. ``PROJ-sample-course-a-20260621094139`` and
    ``sample-course-a`` both key to ``sample-course-a``. This is what the flip heuristic
    counts: ten timestamped runs of one textbook are ONE corpus, not ten.
    """
    n = _course_token(name)
    # Normalize separators so OPENSTAX_ALG_9 and sample-course-a are ONE corpus.
    n = n.replace("_", "-")
    # Strip trailing timestamp(s).
    n = _TIMESTAMP_RE.sub("", n)
    # Strip run-variant suffixes (repeatedly, in case of stacking).
    changed = True
    while changed:
        changed = False
        for suf in _RUN_VARIANT_SUFFIXES:
            if n.endswith(suf):
                n = n[: -len(suf)]
                changed = True
    return n.strip("-") or _course_token(name)


# --------------------------------------------------------------------------------------
# Content-based corpus identity
# --------------------------------------------------------------------------------------
# The slug-based ``_corpus_key`` above collapses RE-RUNS of one named course, but it
# CANNOT see that two DIFFERENTLY-NAMED courses are the same underlying textbook. Three
# exports — ``course-a-cal2``, ``course-a-calib``, ``sample-course-a`` — are all the
# OpenStax ``sample-algebra-2e`` textbook, yet they key to three distinct slugs and
# fake the ">=2 distinct corpora" diversity precondition the flip heuristic depends on.
# The fix: when an export carries a resolvable SOURCE-DOCUMENT signal, key on the
# normalized source-document identity instead of the slug, so same-textbook runs collapse.

# A dart source-id is ``dart:<doc-slug>#<block-hash>``; we want the ``<doc-slug>``.
_DART_SOURCE_RE = re.compile(r"^dart:([^#\s]+)")
# Section/chapter-range suffixes appended to a textbook doc slug for a partial ingest
# (e.g. ``sample-algebra-2e-s11to13`` / ``sample-algebra-2e-ch1-3``). Stripping
# them collapses partial slices of one textbook to a single stable identity.
_DOC_RANGE_SUFFIX_RE = re.compile(
    r"-(?:s\d+(?:to\d+)?|ch\d+(?:[-to]+\d+)?|sec\d+(?:[-to]+\d+)?|p\d+(?:[-to]+\d+)?)$",
    re.IGNORECASE,
)


def _normalize_source_doc(raw: str) -> str:
    """Normalize a source-document slug/path/basename to a stable textbook identity.

    ``sample-algebra-2e-s11to13_accessible.html`` and
    ``sample-algebra-2e-ch1-3_accessible`` both normalize to
    ``sample-algebra-2e`` so partial slices of one textbook share an identity.
    Returns ``""`` when nothing usable remains.
    """
    s = str(raw).strip()
    if not s:
        return ""
    # Drop any path components -> basename only.
    s = s.replace("\\", "/").rsplit("/", 1)[-1]
    s = s.lower()
    # Drop a file extension (.html/.pdf/.xhtml...).
    if "." in s:
        s = s.rsplit(".", 1)[0]
    s = s.replace("_", "-")
    # Drop the SemantiK/DART ``-accessible`` marker.
    for marker in ("-accessible", "-converted", "-clean"):
        if s.endswith(marker):
            s = s[: -len(marker)]
    # Repeatedly strip a trailing section/chapter range suffix.
    changed = True
    while changed:
        changed = False
        new = _DOC_RANGE_SUFFIX_RE.sub("", s)
        if new != s:
            s, changed = new, True
    return s.strip("-")


def _dominant(counter: dict[str, int]) -> str:
    """Return the most-frequent key (ties broken alphabetically for determinism)."""
    if not counter:
        return ""
    return max(sorted(counter), key=lambda k: counter[k])


def _source_doc_from_blocks(export_dir: Path) -> str:
    """Most-common normalized source document across this export's blocks.

    Reads ``source_ids`` (``dart:<slug>#...``) from the richest available blocks file
    (rewrite-tier ``blocks_final`` preferred, then ``blocks_validated``). Returns ``""``
    when no blocks / no dart source-ids are present.
    """
    candidates = (
        "04_rewrite/blocks_final.jsonl",
        "04_rewrite/blocks_validated.jsonl",
        "01_outline/blocks_validated.jsonl",
    )
    counter: dict[str, int] = {}
    for rel in candidates:
        fpath = export_dir / rel
        if not fpath.is_file():
            continue
        try:
            with fpath.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    block = json.loads(line)
                    sids = block.get("source_ids") or []
                    if not isinstance(sids, list):
                        continue
                    for sid in sids:
                        m = _DART_SOURCE_RE.match(str(sid))
                        if not m:
                            continue
                        doc = _normalize_source_doc(m.group(1))
                        if doc:
                            counter[doc] = counter.get(doc, 0) + 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOG.debug("blocks source-id read failed for %s: %s", fpath, exc)
            continue
        if counter:
            # First file that yielded a signal is authoritative; do not double-count.
            break
    return _dominant(counter)


def _source_doc_from_structure(course_dir: Path) -> str:
    """Most-common normalized source document from a ``textbook_structure.json``.

    Prefers the top-level ``source_files`` list; falls back to per-chapter
    ``source_file``. Multi-document corpora (e.g. an 8-file RDF/SHACL bundle) resolve to
    their DOMINANT (most-frequently-cited) document. Returns ``""`` on no signal.
    """
    structs = list(course_dir.rglob("01_learning_objectives/textbook_structure.json"))
    structs += list(course_dir.rglob("textbook_structure.json"))
    seen: set[Path] = set()
    counter: dict[str, int] = {}
    for sp in structs:
        rp = sp.resolve()
        if rp in seen or not sp.is_file():
            continue
        seen.add(rp)
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOG.debug("textbook_structure read failed for %s: %s", sp, exc)
            continue
        sources = data.get("source_files")
        if isinstance(sources, list):
            for src in sources:
                doc = _normalize_source_doc(str(src))
                if doc:
                    counter[doc] = counter.get(doc, 0) + 1
        for ch in data.get("chapters") or []:
            if isinstance(ch, dict) and ch.get("source_file"):
                doc = _normalize_source_doc(str(ch["source_file"]))
                if doc:
                    counter[doc] = counter.get(doc, 0) + 1
    return _dominant(counter)


def _source_doc_from_chunks(course_dir: Path) -> str:
    """Most-common normalized source document from a libv2 course's chunk files.

    Reads ``source.module_id`` / ``source.lesson_id`` from the dart/imscc chunk JSONL.
    Returns ``""`` on no signal (e.g. empty smoke-test chunk files).
    """
    counter: dict[str, int] = {}
    chunk_files = list(course_dir.rglob("dart_chunks/chunks.jsonl"))
    chunk_files += list(course_dir.rglob("imscc_chunks/chunks.jsonl"))
    for cf in chunk_files:
        if not cf.is_file():
            continue
        try:
            with cf.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    block = json.loads(line)
                    src = block.get("source")
                    if not isinstance(src, dict):
                        continue
                    raw = src.get("module_id") or src.get("lesson_id")
                    if not raw:
                        continue
                    doc = _normalize_source_doc(str(raw))
                    if doc:
                        counter[doc] = counter.get(doc, 0) + 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOG.debug("chunk source read failed for %s: %s", cf, exc)
            continue
        if counter:
            break
    return _dominant(counter)


def _content_corpus_key(course_dir: Path, name: str) -> str:
    """Corpus identity keyed on the export's SOURCE DOCUMENT when resolvable.

    Tries the content signals in order of ROBUSTNESS. ``textbook_structure.json``'s
    ``source_files`` is the ingested INPUT-FILE path, which carries the textbook
    identity verbatim across every provenance convention, so it is tried FIRST. The
    blocks' ``dart:<slug>`` source-ids are a useful fallback BUT older exports stamped
    the COURSE slug (not the textbook slug) into the dart id — so blocks are consulted
    only when no structure signal is present. Libv2 chunk source is the last resort.
    When ONE resolves, the corpus key is ``src:<normalized-doc>`` so two differently-
    named courses built from the same textbook collapse to a single corpus. When NONE
    resolves (legacy exports with no content signal, or a read error), falls back to the
    slug-based ``_corpus_key`` — discovery NEVER crashes on a missing/unreadable artifact.
    """
    try:
        doc = (
            _source_doc_from_structure(course_dir)
            or _source_doc_from_blocks(course_dir)
            or _source_doc_from_chunks(course_dir)
        )
    except Exception as exc:  # never let a malformed artifact break discovery
        LOG.warning(
            "content corpus-key derivation failed for %s (%s); "
            "falling back to slug key",
            course_dir,
            exc,
        )
        doc = ""
    if doc:
        return f"src:{doc}"
    LOG.debug("no content source signal under %s; using slug corpus key", course_dir)
    return _corpus_key(name)


def _find_capture_files_for(course_name: str, captures_root: Path) -> list[Path]:
    """Find decision-capture JSONL whose <COURSE> path segment matches the course.

    Slug-free: matches on a normalized token contained in the capture dir path, not a
    hardcoded slug.
    """
    if not captures_root.is_dir():
        return []
    token = _course_token(course_name)
    out: list[Path] = []
    for tool_dir in captures_root.iterdir():
        if not tool_dir.is_dir():
            continue
        for course_dir in tool_dir.iterdir():
            if not course_dir.is_dir():
                continue
            if _course_token(course_dir.name) != token and token not in _course_token(
                course_dir.name
            ):
                continue
            out.extend(sorted(course_dir.rglob("decisions_*.jsonl")))
    return out


def discover_corpora(
    *,
    course_filter: str | None,
    runs_dir: Path | None,
    extra_roots: Iterable[Path] | None = None,
) -> list[CorpusResult]:
    """Discover corpora dynamically from LibV2 + Courseforge exports (+ optional overrides).

    ``extra_roots`` (from ``--corpora`` / ``ED4ALL_CALIB_EXTRA_CORPORA``) are additional
    operator-supplied corpus roots; each root's immediate children are treated as
    export-shaped corpora (mirroring ``--runs-dir``), unioned onto the default discovery.

    Returns CorpusResult shells (paths resolved, sources not yet read).
    """
    corpora: dict[str, CorpusResult] = {}
    captures_root = _training_captures_root()

    def _add_courseforge_export(export_dir: Path) -> None:
        if not export_dir.is_dir():
            return
        name = export_dir.name
        if course_filter and _course_token(course_filter) not in _course_token(name):
            return
        res = corpora.setdefault(
            name,
            CorpusResult(
                corpus_id=name,
                origin="courseforge_export",
                path=str(export_dir),
                corpus_key=_content_corpus_key(export_dir, name),
            ),
        )
        # Read every validation report under this export (outline tier + 04_rewrite tier).
        reports = sorted(export_dir.rglob("02_validation_report/report.json"))
        if not reports:
            res.skip_reasons.append("no 02_validation_report/report.json under export")
        for rp in reports:
            read_validation_report(res, rp)
        # IB6 rollup, if archival emitted it here.
        rollup = export_dir / "block_quality_rollup_report.json"
        if rollup.is_file():
            read_block_quality_rollup(res, rollup)
        # Post-loop aggregator — the ONLY on-disk source for the pre-Courseforge
        # phase gates (chunk_wcag_status / co_terminal_alignment / source_coverage /
        # objective_source_refs), emitted by a full textbook_to_course run.
        agg = export_dir / "courseforge_validation_report.json"
        if agg.is_file():
            read_courseforge_validation_report(res, agg)
        # Decision captures for this course.
        caps = _find_capture_files_for(name, captures_root)
        if caps:
            read_decision_captures(res, caps)

    def _add_libv2_course(course_dir: Path) -> None:
        if not course_dir.is_dir():
            return
        name = course_dir.name
        if course_filter and _course_token(course_filter) not in _course_token(name):
            return
        res = corpora.setdefault(
            name,
            CorpusResult(
                corpus_id=name,
                origin="libv2",
                path=str(course_dir),
                corpus_key=_content_corpus_key(course_dir, name),
            ),
        )
        rollup = course_dir / "block_quality_rollup_report.json"
        got = False
        if rollup.is_file():
            got = read_block_quality_rollup(res, rollup) or got
        # Some libv2 courses keep a courseforge validation report alongside.
        for rp in sorted(course_dir.rglob("02_validation_report/report.json")):
            got = read_validation_report(res, rp) or got
        # Archived post-loop aggregator (pre-Courseforge phase gates).
        for agg in sorted(course_dir.rglob("courseforge_validation_report.json")):
            got = read_courseforge_validation_report(res, agg) or got
        caps = _find_capture_files_for(name, captures_root)
        if caps:
            got = read_decision_captures(res, caps) or got
        if not got:
            res.skip_reasons.append(
                "no block_quality_rollup_report.json, validation report, or "
                "block_validation_action captures found"
            )

    if runs_dir is not None:
        # Explicit override: treat each immediate child as a corpus (export shape).
        if runs_dir.is_dir():
            for child in sorted(runs_dir.iterdir()):
                if child.is_dir():
                    _add_courseforge_export(child)
        else:
            LOG.warning("--runs-dir %s is not a directory", runs_dir)
    else:
        cf_root = _courseforge_exports_root()
        if cf_root.is_dir():
            for child in sorted(cf_root.iterdir()):
                _add_courseforge_export(child)
        lv_root = _libv2_courses_root()
        if lv_root.is_dir():
            # Union the filesystem course dirs with the W0.8 master-catalog slugs so a
            # cataloged course is never missed (and vice-versa). Both are course NAMES;
            # only names whose dir exists on disk are added.
            names: set[str] = {
                child.name for child in lv_root.iterdir() if child.is_dir()
            }
            for slug in _catalog_course_names(_libv2_root()):
                if (lv_root / slug).is_dir():
                    names.add(slug)
            for name in sorted(names):
                _add_libv2_course(lv_root / name)

    # W8.3 — extra operator-supplied corpus roots (opt-in; unioned onto discovery).
    # Each root's immediate children are treated as export-shaped corpora (like
    # --runs-dir). A root with no child dirs is itself treated as one export.
    for root in extra_roots or []:
        root = Path(root)
        if not root.is_dir():
            continue
        try:
            children = [c for c in sorted(root.iterdir()) if c.is_dir()]
        except OSError as exc:
            LOG.warning("could not enumerate extra-corpora root %s: %s", root, exc)
            continue
        if children:
            for child in children:
                _add_courseforge_export(child)
        else:
            _add_courseforge_export(root)

    return list(corpora.values())


# --------------------------------------------------------------------------------------
# Aggregation across corpora
# --------------------------------------------------------------------------------------
def aggregate(
    corpora: list[CorpusResult], *, max_corpora: int | None = None
) -> dict[str, Any]:
    """Roll per-corpus observations up into the per-gate-family calibration report.

    Corpus IDENTITY for the ">=2 corpora" requirement is keyed on the underlying source
    corpus (``corpus_key``), NOT the run/export dir — ten timestamped runs of one
    textbook count as ONE corpus, AND two differently-named courses built from the same
    source document (resolved from blocks/structure/chunk content via
    ``_content_corpus_key``) also count as ONE. When multiple runs share a key, the most-recent run
    (latest path name, lexicographically — names carry sortable timestamps) is the
    representative whose observations are used, so the measurement reflects current
    behavior rather than summing stale + fresh runs.
    """
    # Only corpora that actually produced >=1 observation count toward coverage.
    contributing = [
        c for c in corpora if any(o.evaluated > 0 for o in c.observations.values())
    ]

    # Collapse to one representative per underlying corpus_key. The
    # representative is the run with the MOST gate-family coverage, tie-broken by
    # the latest run TIMESTAMP. A flags-ON run exercises every IB gate; an older
    # flags-OFF run of the same textbook exercises fewer, so coverage must lead.
    # NOTE: sorting by the full corpus_id name is WRONG — differing PROJ- name
    # prefixes (course-a vs openstax) dominate a lexicographic sort over the
    # trailing timestamp, so an older "openstax-..." run would beat a newer
    # "course-a-..." run of the SAME underlying corpus. Sort by the extracted
    # timestamp instead.
    def _coverage(c: CorpusResult) -> int:
        return sum(1 for o in c.observations.values() if o.evaluated > 0)

    def _run_ts(corpus_id: str) -> str:
        m = _TIMESTAMP_RE.search(corpus_id)
        return m.group(0).lstrip("-") if m else ""

    by_key: dict[str, CorpusResult] = {}
    for c in sorted(
        contributing, key=lambda r: (_coverage(r), _run_ts(r.corpus_id), r.corpus_id)
    ):
        by_key[c.corpus_key] = c  # most-coverage, then latest-timestamp, overwrites
    representatives = list(by_key.values())

    # W8.3 --max-corpora cap: keep the highest-coverage distinct corpora (tie-broken by
    # corpus_key for determinism). Parse-with-fallback: a non-positive cap is ignored.
    if max_corpora is not None and max_corpora > 0 and len(representatives) > max_corpora:
        representatives = sorted(
            representatives, key=lambda c: (-_coverage(c), c.corpus_key)
        )[:max_corpora]
        by_key = {c.corpus_key: c for c in representatives}

    gates_out: dict[str, Any] = {}
    for fam in GATE_FAMILIES:
        per_corpus: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        total_fired = 0
        total_eval = 0
        contributing_corpora = 0
        for c in representatives:
            obs = c.observations.get(fam.family)
            if obs is None or obs.evaluated == 0:
                continue
            contributing_corpora += 1
            total_fired += obs.fired
            total_eval += obs.evaluated
            per_corpus.append(
                {
                    "corpus": c.corpus_id,
                    "corpus_key": c.corpus_key,
                    "origin": c.origin,
                    "fired": obs.fired,
                    "evaluated": obs.evaluated,
                    "fire_rate": round(obs.fire_rate, 6),
                    "sources": sorted(obs.sources),
                }
            )
            for s in obs.samples:
                if len(samples) < MAX_SAMPLE_PER_FAMILY:
                    samples.append(s)

        fire_rate = (total_fired / total_eval) if total_eval else 0.0
        band_lo, band_hi = fam.expected_band
        in_band = band_lo <= fire_rate <= band_hi

        # W8.3 ACROSS-CORPORA aggregate — the per-corpus fire-rate distribution. The MAX
        # (worst-corpus) rate is the operator's per-corpus false-positive signal: a gate
        # can look clean POOLED yet spike on ONE corpus. Computed from the per-corpus rows
        # already assembled above (each corpus's own fire_rate), so it needs no re-read.
        per_corpus_rates = [pc["fire_rate"] for pc in per_corpus]
        if per_corpus_rates:
            max_fire_rate = max(per_corpus_rates)
            min_fire_rate = min(per_corpus_rates)
            mean_fire_rate = sum(per_corpus_rates) / len(per_corpus_rates)
            worst = max(per_corpus, key=lambda pc: pc["fire_rate"])
            worst_corpus = worst["corpus"]
        else:
            max_fire_rate = min_fire_rate = mean_fire_rate = 0.0
            worst_corpus = None
        max_in_band = band_lo <= max_fire_rate <= band_hi
        across_corpora = {
            "n_corpora": contributing_corpora,
            "max_fire_rate": round(max_fire_rate, 6),
            "min_fire_rate": round(min_fire_rate, 6),
            "mean_fire_rate": round(mean_fire_rate, 6),
            "pooled_fire_rate": round(fire_rate, 6),
            "max_in_band": max_in_band,
            "worst_corpus": worst_corpus,
        }

        # flip_ready HEURISTIC — see module docstring. Never an automatic FP verdict.
        # Requires >=2 corpora AND the POOLED rate in band AND the WORST-corpus rate in
        # band (a per-corpus spike must not hide behind a clean pool).
        if contributing_corpora == 0:
            flip_ready = False
            blocked = "no corpus produced an observation for this gate family"
        elif contributing_corpora < 2:
            flip_ready = False
            blocked = (
                f"needs >=2 corpora (found {contributing_corpora}); FP rate cannot be "
                "established from a single corpus"
            )
        elif not in_band:
            flip_ready = False
            blocked = (
                f"fire_rate {fire_rate:.4f} outside documented expected band "
                f"[{band_lo}, {band_hi}] — manual FP audit required before flip"
            )
        elif not max_in_band:
            flip_ready = False
            blocked = (
                f"pooled fire_rate {fire_rate:.4f} in band but worst-corpus rate "
                f"{max_fire_rate:.4f} ({worst_corpus}) exceeds band "
                f"[{band_lo}, {band_hi}] — a per-corpus spike needs a manual FP audit "
                "before flip"
            )
        else:
            flip_ready = True
            blocked = None

        gates_out[fam.family] = {
            "gate_ids": list(fam.gate_ids),
            "scope": fam.scope,
            "fire_rate": round(fire_rate, 6),
            "total_fired": total_fired,
            "total_evaluated": total_eval,
            "corpora_count": contributing_corpora,
            "corpora": per_corpus,
            "across_corpora": across_corpora,
            "sample": samples,
            "expected_band": {"min": band_lo, "max": band_hi, "in_band": in_band},
            "band_source": fam.band_source,
            "flip_ready": flip_ready,
            "flip_blocked_reason": blocked,
            "notes": fam.notes,
        }

    return {
        "report_version": "1.0",
        "tool": "scripts/calibration_harness.py",
        "purpose": (
            "Per-gate-family fire-rate measurement across discovered corpora to inform the "
            "framework gate-family critical-flips (config/workflows.yaml # TODO(calibration) "
            "markers). flip_ready is a HEURISTIC (>=2 corpora AND fire_rate in documented "
            "band); a manual FP audit, not this tool, authorizes any flip."
        ),
        "corpora_discovered": [
            {
                "corpus_id": c.corpus_id,
                "corpus_key": c.corpus_key,
                "origin": c.origin,
                "path": c.path,
                "contributed": any(o.evaluated > 0 for o in c.observations.values()),
                "is_representative": c in representatives,
                "sources_read": c.sources_read,
                "skip_reasons": c.skip_reasons,
            }
            for c in corpora
        ],
        # Distinct underlying-corpus count — the number that gates the >=2 requirement.
        "distinct_corpus_count": len(representatives),
        # W8.3 discover-or-skip signal: True iff >=2 DISTINCT corpora contributed data
        # (the precondition for ANY flip). False => every gate stays flip_ready=false.
        "sufficient_corpora": len(representatives) >= 2,
        "distinct_corpus_keys": sorted(by_key),
        # Raw run count that contributed data (informational; NOT the flip gate).
        "contributing_run_count": len(contributing),
        # W8.2 — config<->table reconciliation audit. ``uncovered_deferred_gates``
        # MUST be empty: every config/workflows.yaml # TODO(calibration) gate is either
        # curated or auto-synthesized into a family here. A non-empty list means the
        # config parser could not read the config (degraded to curated-only).
        "gate_family_reconciliation": _reconciliation_summary(),
        "gate_families": gates_out,
    }


def _reconciliation_summary() -> dict[str, Any]:
    """Audit block proving GATE_FAMILIES is reconciled with config deferred-flip gates."""
    curated_ids = {gid for fam in _CURATED_GATE_FAMILIES for gid in fam.gate_ids}
    auto_families = [f for f in GATE_FAMILIES if f.family.startswith("(auto) ")]
    auto_ids = {gid for fam in auto_families for gid in fam.gate_ids}
    covered = {gid for fam in GATE_FAMILIES for gid in fam.gate_ids}
    uncovered = sorted(CONFIG_DEFERRED_GATE_IDS - covered)
    return {
        "config_source": str(_workflows_config_path()),
        "config_deferred_gate_count": len(CONFIG_DEFERRED_GATE_IDS),
        "config_readable": bool(CONFIG_DEFERRED_GATE_IDS),
        "curated_family_count": len(_CURATED_GATE_FAMILIES),
        "curated_gate_ids": sorted(curated_ids),
        "auto_synthesized_family_count": len(auto_families),
        "auto_synthesized_gate_ids": sorted(auto_ids),
        # Deferred gates covered as a SIBLING inside a curated family (not their own
        # family, and not auto-synthesized) — deliberate, documented coverage.
        "deferred_covered_by_curated": sorted(
            CONFIG_DEFERRED_GATE_IDS & curated_ids
        ),
        # INVARIANT: must be empty. Non-empty ⇒ config unreadable (degraded).
        "uncovered_deferred_gates": uncovered,
    }


# --------------------------------------------------------------------------------------
# Stdout summary
# --------------------------------------------------------------------------------------
def print_summary(report: dict[str, Any]) -> None:
    print("=" * 88)
    print("CALIBRATION HARNESS — per-gate-family fire-rate measurement")
    print("=" * 88)
    discovered = report["corpora_discovered"]
    distinct = report["distinct_corpus_count"]
    runs = report["contributing_run_count"]
    print(
        f"Corpora discovered: {len(discovered)} dirs | {runs} run(s) contributed data | "
        f"{distinct} DISTINCT underlying corpus/corpora"
    )
    print(f"Distinct corpus keys: {', '.join(report['distinct_corpus_keys']) or '(none)'}")
    for c in discovered:
        if not c["contributed"]:
            continue
        rep = "*representative*" if c["is_representative"] else "(duplicate run of key)"
        print(f"  [OK] {c['corpus_id']}  key={c['corpus_key']}  {rep}")
    if distinct < 2:
        print(
            "\n** Only %d distinct corpus available — EVERY gate family is "
            "flip_ready=false (needs >=2 DISTINCT corpora; runs of one corpus do not "
            "count). A 2nd-corpus run unblocks the flips. **" % distinct
        )
    print("-" * 88)
    print(
        f"{'GATE FAMILY':<46}{'pooled':>9}{'maxFR':>9}{'corpora':>9}{'flip?':>7}"
    )
    print("-" * 88)
    for name, g in report["gate_families"].items():
        ready = "YES" if g["flip_ready"] else "no"
        short = name if len(name) <= 44 else name[:43] + "…"
        max_fr = g.get("across_corpora", {}).get("max_fire_rate", g["fire_rate"])
        print(
            f"{short:<46}{g['fire_rate']:>9.4f}{max_fr:>9.4f}"
            f"{g['corpora_count']:>9}{ready:>7}"
        )
    print("-" * 88)
    print("Per-gate flip-blocked reasons:")
    for name, g in report["gate_families"].items():
        if g["flip_blocked_reason"]:
            print(f"  - {name}: {g['flip_blocked_reason']}")
    print("=" * 88)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--course",
        default=None,
        help="Restrict discovery to corpora whose slug/export name matches this token.",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Explicit directory whose immediate children are treated as corpora "
        "(Courseforge-export shape). Overrides default LibV2 + Courseforge discovery.",
    )
    parser.add_argument(
        "--corpora",
        action="append",
        default=None,
        metavar="DIR",
        help="Additional corpus root(s) to UNION onto discovery (repeatable). Each "
        "root's immediate children are treated as export-shaped corpora. Wins over the "
        f"{ENV_EXTRA_CORPORA} env var. Parse-with-fallback: a non-directory is skipped.",
    )
    parser.add_argument(
        "--max-corpora",
        type=int,
        default=None,
        help="Cap the number of DISTINCT corpora used (keeps the highest-coverage ones). "
        "Non-positive values are ignored (parse-with-fallback).",
    )
    parser.add_argument(
        "--out",
        default=str(_repo_root() / "calibration_report.json"),
        help="Output path for calibration_report.json.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the stdout summary."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else None
    extra_roots = resolve_extra_corpora_roots(args.corpora)
    corpora = discover_corpora(
        course_filter=args.course, runs_dir=runs_dir, extra_roots=extra_roots
    )
    report = aggregate(corpora, max_corpora=args.max_corpora)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)

    if not args.quiet:
        print_summary(report)
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
