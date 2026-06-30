"""Honest IRT difficulty-calibration scaffold (response-data-gated).

The chunk/assessment ``difficulty`` band is, today, purely heuristic
(``Trainforge/process_course.py::_determine_difficulty`` Bloom→difficulty
cascade + keyword fallback, mirrored at
``MCP/tools/pipeline_tools.py::_resolve_chunk_difficulty``). This module
adds an HONEST scaffold around that heuristic, behind
``TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD`` (default off, byte-identical when
off):

* a ``difficulty_provenance`` tag (``heuristic`` | ``calibrated``) mirroring
  the existing ``chunk_v4.bloom_level_source`` provenance-tag idiom;
* a response-ingestion SEAM (``resolve_response_data_path`` →
  ``<libv2>/courses/<slug>/responses/item_responses.jsonl`` only when it
  exists) — schematized at ``schemas/assessment/item_response.schema.json``
  but NO response data is shipped;
* a deterministic proportion-correct → difficulty-band calibrator
  (``load_calibrated_difficulty``) that fits NOTHING — it computes a band
  ONLY for items carrying ``>= min_responses`` real responses and returns
  ``{}`` otherwise, NEVER fabricating IRT parameters with no data;
* a difficulty-distribution descriptor (``describe_difficulty_distribution``).

ANTI-FABRICATION CONTRACT: the only IRT block ever written is for a chunk
whose id is present in the calibrated map with real backing responses. A
no-response run leaves every chunk ``difficulty_provenance='heuristic'``
with NO ``difficulty_irt`` block (always-keep-the-heuristic). The
``discrimination_a`` parameter is the FIXED Rasch (1PL) convention value
``1.0`` — it is explicitly NOT a fitted 2PL discrimination; the honest
``difficulty_b`` is the only calibrated quantity, derived from the
observed proportion-correct, and ``n_responses`` records the real sample.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Flag + constants
# --------------------------------------------------------------------------- #

_FLAG_ENV = "TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD"
_TRUTHY = {"1", "true", "yes", "on"}

# Difficulty band enum — mirrors schemas/knowledge/chunk_v4.schema.json.
DIFFICULTY_FOUNDATIONAL = "foundational"
DIFFICULTY_INTERMEDIATE = "intermediate"
DIFFICULTY_ADVANCED = "advanced"
ORDERED_BANDS: Tuple[str, str, str] = (
    DIFFICULTY_FOUNDATIONAL,
    DIFFICULTY_INTERMEDIATE,
    DIFFICULTY_ADVANCED,
)

# Provenance tags.
DIFFICULTY_PROVENANCE_HEURISTIC = "heuristic"
DIFFICULTY_PROVENANCE_CALIBRATED = "calibrated"

# Minimum REAL responses an item needs before a calibrated band is derived.
# Deliberately a psychometric floor — below this an empirical proportion is
# too noisy to override the heuristic. Tests reference this constant directly
# so they are robust to a future re-calibration of the value.
DEFAULT_MIN_RESPONSES = 30

# Fixed Rasch (1PL) discrimination. NOT a fitted 2PL value — this is the
# honest scaffold convention (no 2PL fit anywhere). Stored verbatim so a
# downstream replay can tell calibration ran a 1PL-style transform, not a
# fabricated 2PL fit.
RASCH_DISCRIMINATION_A = 1.0

# Proportion-correct band cut points. High proportion-correct = easy
# (foundational); low = hard (advanced).
_P_FOUNDATIONAL_FLOOR = 0.70  # p >= 0.70 → foundational
_P_ADVANCED_CEIL = 0.40       # p < 0.40 → advanced; else intermediate

# Clamp for the logit transform so p in {0, 1} doesn't blow up to +-inf.
_P_EPSILON = 1e-3

# Response-ingestion seam location (relative to <libv2>/courses/<slug>/).
_RESPONSES_SUBDIR = "responses"
_RESPONSES_FILENAME = "item_responses.jsonl"


def irt_scaffold_enabled() -> bool:
    """Parse-with-fallback resolver for ``TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD``.

    Only ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive) enable the
    scaffold. Garbage / empty / falsey → off (byte-identical legacy emit).
    """
    return os.environ.get(_FLAG_ENV, "").strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# Response-ingestion seam
# --------------------------------------------------------------------------- #

def resolve_response_data_path(
    course_slug: str,
    libv2_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the learner-response file for ``course_slug``.

    Returns the path to ``<libv2>/courses/<slug>/responses/item_responses.jsonl``
    ONLY when it exists on disk, else ``None``. ``course_slug`` and
    ``libv2_root`` are always parameters — no slug/path is ever hardcoded.
    When ``libv2_root`` is None the canonical, ``ED4ALL_LIBV2_ROOT``-aware
    root is resolved via ``lib.paths.libv2_path``.
    """
    if not course_slug:
        return None
    if libv2_root is None:
        try:
            from lib.paths import libv2_path
            root = libv2_path()
        except Exception:  # noqa: BLE001 — never crash the emit on a path resolve
            return None
    else:
        root = Path(libv2_root)
    candidate = root / "courses" / str(course_slug) / _RESPONSES_SUBDIR / _RESPONSES_FILENAME
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

class CalibratedDifficulty(Dict[str, Any]):
    """A thin dict alias documenting the calibrated record shape.

    Keys: ``difficulty`` (band str), ``difficulty_b`` (float),
    ``discrimination_a`` (float, the fixed Rasch value), ``n_responses``
    (int), ``proportion_correct`` (float). Subclassing ``dict`` keeps it
    JSON-serialisable and equality-comparable in tests.
    """


def _band_from_proportion(p: float) -> str:
    """Map an observed proportion-correct to an ordered difficulty band."""
    if p >= _P_FOUNDATIONAL_FLOOR:
        return DIFFICULTY_FOUNDATIONAL
    if p < _P_ADVANCED_CEIL:
        return DIFFICULTY_ADVANCED
    return DIFFICULTY_INTERMEDIATE


def _difficulty_b_from_proportion(p: float) -> float:
    """Rasch (1PL) difficulty from observed proportion-correct.

    ``b = ln((1 - p) / p)`` — high proportion-correct → negative b (easy),
    low → positive b (hard). ``p`` is clamped away from 0/1 so the transform
    stays finite. This is the ONLY calibrated quantity; it is a deterministic
    statistic over real responses, not a fitted model.
    """
    p_clamped = min(max(p, _P_EPSILON), 1.0 - _P_EPSILON)
    return math.log((1.0 - p_clamped) / p_clamped)


def _iter_response_records(response_path: Path) -> List[Dict[str, Any]]:
    """Read the JSONL seam file into a list of records (tolerant).

    A malformed line is skipped (logged at debug), never fatal — a corrupt
    response file degrades to fewer calibrated items, never a crash.
    """
    records: List[Dict[str, Any]] = []
    try:
        with response_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    logger.debug("Skipping malformed item_responses line in %s", response_path)
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
    except OSError as exc:
        logger.warning("Failed to read item_responses file %s: %s", response_path, exc)
    return records


def load_calibrated_difficulty(
    response_path: Optional[Path],
    min_responses: int = DEFAULT_MIN_RESPONSES,
) -> Dict[str, CalibratedDifficulty]:
    """Compute a calibrated difficulty band per item from real responses.

    Reads the JSONL seam file and computes a proportion-correct → band map
    ONLY for items with ``>= min_responses`` real responses. Returns ``{}``
    when ``response_path`` is None or no item clears the floor.

    ANTI-FABRICATION: an item below the floor gets NO entry (its chunk stays
    heuristic). No IRT parameter is ever invented for an item with no data.
    The keying id is read verbatim from each record's ``chunk_id`` (preferred)
    or ``item_id`` — no id is minted.
    """
    if response_path is None:
        return {}
    try:
        floor = int(min_responses)
    except (TypeError, ValueError):
        floor = DEFAULT_MIN_RESPONSES
    if floor < 1:
        floor = DEFAULT_MIN_RESPONSES

    # Aggregate (n, n_correct) per item id.
    agg: Dict[str, List[int]] = {}
    for rec in _iter_response_records(response_path):
        item_id = rec.get("chunk_id") or rec.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            continue
        correct = rec.get("correct")
        if not isinstance(correct, bool):
            # The seam contract requires a boolean ``correct``; non-bool is
            # dropped rather than coerced (never fabricate a grade).
            continue
        bucket = agg.setdefault(item_id, [0, 0])
        bucket[0] += 1
        if correct:
            bucket[1] += 1

    calibrated: Dict[str, CalibratedDifficulty] = {}
    for item_id, (n, n_correct) in agg.items():
        if n < floor:
            continue
        p = n_correct / n if n else 0.0
        rec = CalibratedDifficulty()
        rec["difficulty"] = _band_from_proportion(p)
        rec["difficulty_b"] = round(_difficulty_b_from_proportion(p), 6)
        rec["discrimination_a"] = RASCH_DISCRIMINATION_A
        rec["n_responses"] = int(n)
        rec["proportion_correct"] = round(p, 6)
        calibrated[item_id] = rec
    return calibrated


# --------------------------------------------------------------------------- #
# Chunk tagging
# --------------------------------------------------------------------------- #

def tag_chunk_difficulty_provenance(
    chunk: Dict[str, Any],
    calibrated_map: Optional[Dict[str, CalibratedDifficulty]] = None,
) -> Dict[str, Any]:
    """Stamp difficulty provenance (+ optional IRT block) on a chunk in place.

    No-op (byte-identical) when the scaffold flag is off — the chunk is
    returned untouched, with NO new keys. When on:

    * always stamps ``difficulty_provenance='heuristic'``;
    * ONLY when the chunk's id is present in ``calibrated_map`` with real
      backing responses, overrides ``difficulty`` to the calibrated band,
      writes a ``difficulty_irt`` block
      (``difficulty_b`` / ``discrimination_a`` / ``n_responses``), and
      re-stamps ``difficulty_provenance='calibrated'``.

    Returns the same dict for call-site convenience.
    """
    if not irt_scaffold_enabled():
        return chunk
    calibrated_map = calibrated_map or {}
    chunk_id = chunk.get("id") or chunk.get("chunk_id")
    rec = calibrated_map.get(chunk_id) if isinstance(chunk_id, str) else None
    if rec is not None:
        chunk["difficulty"] = rec["difficulty"]
        chunk["difficulty_provenance"] = DIFFICULTY_PROVENANCE_CALIBRATED
        chunk["difficulty_irt"] = {
            "difficulty_b": rec["difficulty_b"],
            "discrimination_a": rec["discrimination_a"],
            "n_responses": rec["n_responses"],
        }
    else:
        chunk["difficulty_provenance"] = DIFFICULTY_PROVENANCE_HEURISTIC
        # Anti-fabrication: NO difficulty_irt block without backing responses.
    return chunk


# --------------------------------------------------------------------------- #
# Distribution descriptor
# --------------------------------------------------------------------------- #

def describe_difficulty_distribution(
    counts: Dict[str, int],
    calibrated_fraction: float,
) -> Dict[str, Any]:
    """Deterministic descriptor over the 3 ordered difficulty bands.

    Returns ``{modal_level, ordinal_entropy, balance_ratio,
    calibrated_fraction, provenance}``:

    * ``modal_level`` — the band with the largest count (ties → the
      lower-ordinal band, the ``ORDERED_BANDS`` order);
    * ``ordinal_entropy`` — normalized Shannon entropy over the 3 bands
      (0.0 = all mass on one band, 1.0 = uniform);
    * ``balance_ratio`` — ``min_share / max_share`` over the populated bands
      (1.0 = perfectly balanced; 0.0 when a band is empty);
    * ``calibrated_fraction`` — clamped to ``[0, 1]``;
    * ``provenance`` — ``heuristic`` (fraction == 0), ``calibrated``
      (fraction == 1), else ``mixed``.
    """
    safe_counts = {band: int(counts.get(band, 0) or 0) for band in ORDERED_BANDS}
    total = sum(safe_counts.values())

    # Modal level (deterministic tie-break by ordered-band order).
    modal_level: Optional[str] = None
    if total > 0:
        best = -1
        for band in ORDERED_BANDS:
            if safe_counts[band] > best:
                best = safe_counts[band]
                modal_level = band

    # Normalized Shannon entropy.
    ordinal_entropy = 0.0
    if total > 0:
        shares = [safe_counts[b] / total for b in ORDERED_BANDS]
        ent = -sum(s * math.log(s) for s in shares if s > 0)
        ordinal_entropy = ent / math.log(len(ORDERED_BANDS))

    # Balance ratio (min/max share). 0.0 when any band is empty.
    balance_ratio = 0.0
    if total > 0:
        shares = [safe_counts[b] / total for b in ORDERED_BANDS]
        max_share = max(shares)
        min_share = min(shares)
        balance_ratio = (min_share / max_share) if max_share > 0 else 0.0

    try:
        frac = float(calibrated_fraction)
    except (TypeError, ValueError):
        frac = 0.0
    frac = min(max(frac, 0.0), 1.0)

    if frac <= 0.0:
        provenance = DIFFICULTY_PROVENANCE_HEURISTIC
    elif frac >= 1.0:
        provenance = DIFFICULTY_PROVENANCE_CALIBRATED
    else:
        provenance = "mixed"

    return {
        "modal_level": modal_level,
        "ordinal_entropy": round(ordinal_entropy, 6),
        "balance_ratio": round(balance_ratio, 6),
        "calibrated_fraction": round(frac, 6),
        "provenance": provenance,
    }
