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

A corpus missing ALL three sources for a gate is SKIPPED for that gate with a logged
reason (no fabricated data).

WHAT IT EMITS
-------------
``calibration_report.json`` (path overridable) + a readable stdout summary. Per gate:
  {fire_rate, corpora[], sample[], expected_band, corpora_count, flip_ready,
   flip_blocked_reason}

``flip_ready`` is a HEURISTIC, not an automatic FP verdict — FP rate needs ground truth
the harness cannot synthesize. It is True only when (corpora_count >= 2 AND the observed
fire_rate falls within the gate's DOCUMENTED expected band). The harness surfaces the
measurement + the criterion; a downstream human/adjudication step decides the flip.

USAGE
-----
    python3 scripts/calibration_harness.py
    python3 scripts/calibration_harness.py --course sample-course-a
    python3 scripts/calibration_harness.py --runs-dir Courseforge/exports
    python3 scripts/calibration_harness.py --out /tmp/calibration_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
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


GATE_FAMILIES: tuple[GateFamily, ...] = (
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
        family="IB6.2 anatomy slot presence",
        gate_ids=("anatomy_slot_presence",),
        expected_band=(0.0, 0.10),
        band_source="config/workflows.yaml IB6 marker (~line 218): same deferred 0-3 calibration block.",
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
        family="IB7.6c per-type Bloom-range ceiling",
        gate_ids=("bloom_type_range",),
        expected_band=(0.0, 0.05),
        band_source=(
            "config/workflows.yaml IB7.6c marker (~line 295): flip BLOCK_BLOOM_OVER_CEILING "
            "to critical after a >=2-corpus FP measurement."
        ),
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
    GateFamily(
        family="Numeric-literal grounding (math-fabrication control)",
        gate_ids=("numeric_literal_grounding",),
        expected_band=(0.0, 0.05),
        band_source=(
            "root CLAUDE.md numeric-literal-grounding landing note: measured CLEAN on a real "
            "algebra corpus (grounded blocks 0% source-absent, zero false positives); flip "
            "to critical after a >=2-corpus FP-rate measurement."
        ),
    ),
    GateFamily(
        family="W4 NLI grounding (block prose entailment / objective entailment / claim support)",
        gate_ids=("block_prose_entailment", "objective_entailment", "claim_support"),
        expected_band=(0.0, 0.10),
        band_source=(
            "config/workflows.yaml W4 SHADOW marker + plans/finegrain/w4-nli-grounding-gate.md "
            "§4: calibration-gated critical flip deferred; promotes claim_support + "
            "rewrite_source_grounding to critical."
        ),
    ),
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
) -> list[CorpusResult]:
    """Discover corpora dynamically from LibV2 + Courseforge exports (+ optional override).

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
                corpus_key=_corpus_key(name),
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
                corpus_key=_corpus_key(name),
            ),
        )
        rollup = course_dir / "block_quality_rollup_report.json"
        got = False
        if rollup.is_file():
            got = read_block_quality_rollup(res, rollup) or got
        # Some libv2 courses keep a courseforge validation report alongside.
        for rp in sorted(course_dir.rglob("02_validation_report/report.json")):
            got = read_validation_report(res, rp) or got
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
            for child in sorted(lv_root.iterdir()):
                _add_libv2_course(child)

    return list(corpora.values())


# --------------------------------------------------------------------------------------
# Aggregation across corpora
# --------------------------------------------------------------------------------------
def aggregate(corpora: list[CorpusResult]) -> dict[str, Any]:
    """Roll per-corpus observations up into the per-gate-family calibration report.

    Corpus IDENTITY for the ">=2 corpora" requirement is keyed on the underlying source
    corpus (``corpus_key``), NOT the run/export dir — ten timestamped runs of one
    textbook count as ONE corpus. When multiple runs share a key, the most-recent run
    (latest path name, lexicographically — names carry sortable timestamps) is the
    representative whose observations are used, so the measurement reflects current
    behavior rather than summing stale + fresh runs.
    """
    # Only corpora that actually produced >=1 observation count toward coverage.
    contributing = [
        c for c in corpora if any(o.evaluated > 0 for o in c.observations.values())
    ]

    # Collapse to one representative per underlying corpus_key (latest run wins).
    by_key: dict[str, CorpusResult] = {}
    for c in sorted(contributing, key=lambda r: r.corpus_id):
        by_key[c.corpus_key] = c  # later (lexicographically larger ts) overwrites
    representatives = list(by_key.values())

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

        # flip_ready HEURISTIC — see module docstring. Never an automatic FP verdict.
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
        "distinct_corpus_keys": sorted(by_key),
        # Raw run count that contributed data (informational; NOT the flip gate).
        "contributing_run_count": len(contributing),
        "gate_families": gates_out,
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
        f"{'GATE FAMILY':<52}{'fire_rate':>11}{'corpora':>9}{'flip?':>8}"
    )
    print("-" * 88)
    for name, g in report["gate_families"].items():
        ready = "YES" if g["flip_ready"] else "no"
        short = name if len(name) <= 50 else name[:49] + "…"
        print(
            f"{short:<52}{g['fire_rate']:>11.4f}{g['corpora_count']:>9}{ready:>8}"
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
    corpora = discover_corpora(course_filter=args.course, runs_dir=runs_dir)
    report = aggregate(corpora)

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
