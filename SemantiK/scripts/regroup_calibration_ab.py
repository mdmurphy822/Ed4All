#!/usr/bin/env python3
"""Shadow A/B calibration harness for the Stage-5e pedagogical-unit REGROUP
(``SEMANTIK_UNIT_REGROUP``).

Measured-experiment arbiter (mirrors ``scripts/ocr_recall_ab.py``) for ITEM1 of
the SemantiK restructuring campaign: it quantifies what flipping the regroup
default ON does — over-merge rate (must be 0), the reduction in split
label-only boxes, the per-doc box count delta, and the sourceId id-set shrink —
so the flip lands only on evidence. CPU-only, stdlib + ``semantik_structure`` imports,
no GPU, no LLM call.

Two modes:

* ``detector`` (default, primary) — replays the DETERMINISTIC detector
  ``block_resegment._detect_unit_merges`` over minimal ``Region`` / ``FeatureBlock``
  shims reconstructed from a prior conversion's ``region_provenance`` (the
  ``run_cascade_json.py`` bridge JSON, passed via ``--bridge-json``). It reuses
  the SAME boundary single-source-of-truth as production: each shim region's
  ``css_class`` is taken from the provenance ``pedagogy_class`` echo when present,
  else re-derived via ``deterministic_structure._pedagogical_class_for`` — the
  exact regex the clean pass stamps the ``css_class`` from — so the replay is
  faithful to the anchor/boundary logic even on a bridge JSON produced without
  the council/reviewer models resident.

* ``cascade`` (secondary) — shells out to ``run_cascade_json.py --runtime mock``
  twice per PDF (``SEMANTIK_UNIT_REGROUP=0`` vs ``=1``) and diffs the two bridge
  JSONs (region counts, ``block_resegment`` audit rows, provenance id sets).

Fidelity caveats (documented, not hidden):

* The shim's ``feature_block_indices`` are reconstructed as the contiguous FB
  range ``[first_raw_i, first_raw_{i+1} - 1]`` because provenance carries only
  ``first_raw_block_index`` (the region's MIN FB), not the full owned set. This
  makes the detector's FB-adjacency test (``cur_first == prev_last + 1``)
  trivially hold for reading-order-consecutive regions — the same
  reading-order-contiguous assumption the production regroup already requires
  (its ``SEMANTIK_READING_ORDER_FIX`` driver guard).
* A bridge JSON produced without the council models resident has no ``heading``
  region kind (headings fail-soft to ``paragraph``/``metadata_drop``), so a
  heading boundary does not stop a run in the replay. This can only make the
  replay MERGE MORE than production (never less), so a 0-over-merge result on
  the replay is a conservative (upper-bound) guarantee; the over-merge oracle is
  independent of the run-length and catches any wrong fuse regardless.

Report JSON schema (per doc; corpus rollup mirrors the numeric fields):

    {
      "doc": <basename>,
      "anchor_count": int,          # regions that OPEN a unit (_is_unit_anchor)
      "n_regions_off": int,         # flag-OFF box count (no merges)
      "n_regions_on": int,          # flag-ON box count (after regroup)
      "regroup_ops": int,           # number of merge runs the detector emitted
      "regions_folded_hist": {run_len: count},
      "id_set_delta": int,          # sourceIds folded away (set shrink)
      "cap_hits": {run_cap, token_cap, passthrough_stop,
                   boundary_stop, adjacency_stop},   # diagnostic (report-only)
      "over_merge_flags": int,      # dual-oracle wrong-fuse count (GATE: == 0)
      "over_merge_detail": [...],
      "under_merge_off": int,       # OFF residual label-only-box population
      "under_merge_on": int,        # ON residual label-only-box population
      "under_merge_reduction": float,   # 1 - on/off (GATE arm-1: >= 0.50)
      "spot_check": [ up to 10 merged-unit previews ]
    }
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_SEMANTIK = _REPO / "SemantiK"
for _p in (str(_SEMANTIK), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from semantik_structure.pedagogical_units import (  # noqa: E402
    ABSORB_MAX_RUN,
    BODY_BEARING_COMPONENT_CLASSES,
    UNIT_START_PEDAGOGY_CLASSES,
    is_passthrough_region,
    is_unit_boundary,
)
from semantik_structure.qwen_specialists import block_resegment as brs  # noqa: E402
from semantik_structure.qwen_specialists.deterministic_structure import (  # noqa: E402
    _pedagogical_class_for,
)

# Second, data-driven over-merge oracle: the lexicon-profile opener vocabulary
# (independent of the detector's own ``_pedagogical_class_for`` SoT). A member
# whose leading text classifies to one of these START-opener roles is a wrong
# fuse. Body/answer openers (``solution`` / ``objectives``) are NOT start roles.
try:
    from lib.semantik.opener_classifier import classify_opener_label  # noqa: E402

    _HAVE_OPENER_ORACLE = True
except Exception:  # pragma: no cover - lexicon optional
    _HAVE_OPENER_ORACLE = False

    def classify_opener_label(_text: Any):  # type: ignore
        return None


_START_OPENER_ROLES = frozenset(
    {"worked_example", "try_it", "how_to", "readiness_check", "practice"}
)

_TOKEN_BUDGET = brs._UNIT_REGROUP_TOKEN_BUDGET


# ---------------------------------------------------------------------------
# Region / FeatureBlock shims reconstructed from region_provenance.
# ---------------------------------------------------------------------------


def _mk_fb(text: str) -> Any:
    return SimpleNamespace(raw=SimpleNamespace(text=text))


def _reconstruct(prov: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """Rebuild reading-order ``Region`` + ``FeatureBlock`` shims from provenance.

    See the module docstring for the FB-range reconstruction contract.
    """
    entries = sorted(
        (e for e in prov if isinstance(e, dict)),
        key=lambda e: int(e.get("first_raw_block_index", 0) or 0),
    )
    n = len(entries)
    firsts = [int(e.get("first_raw_block_index", i) or 0) for i, e in enumerate(entries)]
    fb_text: dict[int, str] = {}
    regions: list[Any] = []
    for i, e in enumerate(entries):
        start = firsts[i]
        nxt = firsts[i + 1] if i + 1 < n else None
        end = (nxt - 1) if (nxt is not None and nxt > start) else start
        fbs = list(range(start, end + 1))
        raw = str(e.get("raw_text") or "")
        fb_text[start] = raw
        for k in fbs:
            fb_text.setdefault(k, "")
        kind = e.get("region_kind") or "paragraph"
        css = e.get("pedagogy_class") or _pedagogical_class_for(raw) or None
        payload = {"text": raw, "css_class": css, "semantic_class": None}
        # kind-based passthrough marker: table/math/figure carry a non-null
        # source_region_id in production, so mark them so is_passthrough_region
        # stops a run before them (v1 passthrough-boundary parity).
        src = i if kind in {"table", "math", "figure"} else None
        regions.append(
            SimpleNamespace(
                kind=kind,
                payload=payload,
                feature_block_indices=fbs,
                source_region_id=src,
            )
        )
    maxfb = max(fb_text) if fb_text else -1
    feature_blocks = [_mk_fb(fb_text.get(i, "")) for i in range(maxfb + 1)]
    return regions, feature_blocks


# ---------------------------------------------------------------------------
# Oracles + metrics.
# ---------------------------------------------------------------------------


def _leading_text(region: Any) -> str:
    return str((getattr(region, "payload", {}) or {}).get("text") or "")


def _is_anchor(region: Any) -> bool:
    return brs._is_unit_anchor(region)


def _over_merge_member(region: Any) -> str | None:
    """Return a wrong-fuse reason if a NON-anchor merged member text-opens a new
    unit (either oracle fires), else None."""
    text = _leading_text(region)
    cls = _pedagogical_class_for(text)
    if cls in UNIT_START_PEDAGOGY_CLASSES:
        return f"pedagogical_class:{cls}"
    label = classify_opener_label(text[:80]) if _HAVE_OPENER_ORACLE else None
    if label and label[1] in _START_OPENER_ROLES:
        return f"opener_role:{label[1]}"
    return None


def _adjacent_absorbable(regions: list[Any], i: int) -> bool:
    """Whether region ``i+1`` is an FB-adjacent, non-boundary, non-passthrough
    follower — i.e. anchor ``i`` COULD start a run (label-only-box candidate)."""
    j = i + 1
    if j >= len(regions):
        return False
    prev, cur = regions[i], regions[j]
    prev_last = max(prev.feature_block_indices or (-1,))
    cur_first = min(cur.feature_block_indices or (-1,))
    if cur_first != prev_last + 1:
        return False
    if is_passthrough_region(cur):
        return False
    if is_unit_boundary(cur, html=None):
        return False
    return True


def _label_only_population(regions: list[Any], consumed: set[int]) -> int:
    """Count unit-start anchors that remain a run-length-1 box while their
    immediate follower is absorbable — the residual split label-only boxes."""
    total = 0
    for i, region in enumerate(regions):
        if i in consumed:
            continue
        if is_passthrough_region(region) or not _is_anchor(region):
            continue
        if _adjacent_absorbable(regions, i):
            total += 1
    return total


def _classify_cap_hits(regions: list[Any], feature_blocks: list[Any]) -> dict[str, int]:
    """Diagnostic mirror-walk that tallies WHY each anchor run stopped
    (report-only; the acceptance metrics come from the real detector ops)."""
    hits = {
        "run_cap": 0,
        "token_cap": 0,
        "passthrough_stop": 0,
        "boundary_stop": 0,
        "adjacency_stop": 0,
    }
    n = len(regions)
    consumed: set[int] = set()
    i = 0
    while i < n:
        if i in consumed:
            i += 1
            continue
        anchor = regions[i]
        if is_passthrough_region(anchor) or not _is_anchor(anchor):
            i += 1
            continue
        run = [i]
        acc = brs._region_token_count(anchor, feature_blocks)
        j = i + 1
        stop = None
        while j < n:
            if (j - i) >= ABSORB_MAX_RUN:
                stop = "run_cap"
                break
            prev, cur = regions[run[-1]], regions[j]
            prev_last = max(prev.feature_block_indices or (-1,))
            cur_first = min(cur.feature_block_indices or (-1,))
            if cur_first != prev_last + 1:
                stop = "adjacency_stop"
                break
            if is_passthrough_region(cur):
                stop = "passthrough_stop"
                break
            if is_unit_boundary(cur, html=None):
                stop = "boundary_stop"
                break
            cur_tokens = brs._region_token_count(cur, feature_blocks)
            if acc + cur_tokens > _TOKEN_BUDGET:
                stop = "token_cap"
                break
            run.append(j)
            acc += cur_tokens
            j += 1
        if len(run) >= 2 and stop is not None:
            hits[stop] += 1
        if len(run) >= 2:
            consumed.update(run)
            i = run[-1] + 1
        else:
            i += 1
    return hits


def _analyse_detector(doc: str, prov: list[dict[str, Any]]) -> dict[str, Any]:
    os.environ["SEMANTIK_UNIT_REGROUP"] = "1"  # force the self-gated detector on
    regions, feature_blocks = _reconstruct(prov)
    ops = brs._detect_unit_merges(regions, feature_blocks, state=None)

    consumed: set[int] = set()
    folded_hist: dict[str, int] = {}
    id_set_delta = 0
    over_flags = 0
    over_detail: list[dict[str, Any]] = []
    spot_check: list[dict[str, Any]] = []
    anchor_count = sum(
        1
        for r in regions
        if not is_passthrough_region(r) and _is_anchor(r)
    )
    for op in ops:
        idxs = list(op.region_indices)
        consumed.update(idxs)
        run_len = len(idxs)
        folded_hist[str(run_len)] = folded_hist.get(str(run_len), 0) + 1
        id_set_delta += run_len - 1
        for member_idx in idxs[1:]:  # skip the anchor (index 0)
            reason = _over_merge_member(regions[member_idx])
            if reason is not None:
                over_flags += 1
                if len(over_detail) < 25:
                    over_detail.append(
                        {
                            "anchor_region": idxs[0],
                            "member_region": member_idx,
                            "reason": reason,
                            "member_head": _leading_text(regions[member_idx])[:60],
                        }
                    )
        if len(spot_check) < 10:
            spot_check.append(
                {
                    "anchor_region": idxs[0],
                    "run_len": run_len,
                    "semantic_class": getattr(op, "semantic_class", None),
                    "anchor_head": _leading_text(regions[idxs[0]])[:70],
                    "member_heads": [
                        _leading_text(regions[k])[:40] for k in idxs[1:]
                    ],
                }
            )

    under_off = _label_only_population(regions, set())
    under_on = _label_only_population(regions, consumed)
    reduction = 0.0 if under_off == 0 else 1.0 - (under_on / under_off)

    return {
        "doc": doc,
        "anchor_count": anchor_count,
        "n_regions_off": len(regions),
        "n_regions_on": len(regions) - id_set_delta,
        "regroup_ops": len(ops),
        "regions_folded_hist": dict(sorted(folded_hist.items())),
        "id_set_delta": id_set_delta,
        "cap_hits": _classify_cap_hits(regions, feature_blocks),
        "over_merge_flags": over_flags,
        "over_merge_detail": over_detail,
        "under_merge_off": under_off,
        "under_merge_on": under_on,
        "under_merge_reduction": round(reduction, 4),
        "spot_check": spot_check,
    }


# ---------------------------------------------------------------------------
# cascade mode (secondary): shell out mock cascade twice, diff bridge JSONs.
# ---------------------------------------------------------------------------


def _run_cascade(pdf: str, regroup: str, out: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["SEMANTIK_UNIT_REGROUP"] = regroup
    env.setdefault("DART_ALLOW_THETA_STUB", "1")
    script = str(_SEMANTIK / "scripts" / "run_cascade_json.py")
    subprocess.run(
        [sys.executable, script, "--pdf", pdf, "--runtime", "mock", "--out-json", out],
        cwd=str(_REPO),
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.load(open(out))


def _analyse_cascade(pdf: str, tmpdir: Path) -> dict[str, Any]:
    stem = Path(pdf).stem
    off = _run_cascade(pdf, "0", str(tmpdir / f"{stem}.off.json"))
    on = _run_cascade(pdf, "1", str(tmpdir / f"{stem}.on.json"))

    def _ids(doc: dict[str, Any]) -> set[int]:
        return {
            int(r["first_raw_block_index"])
            for r in (doc.get("region_provenance") or [])
            if "first_raw_block_index" in r
        }

    off_ids, on_ids = _ids(off), _ids(on)
    return {
        "doc": stem,
        "n_regions_off": len(off.get("region_provenance") or []),
        "n_regions_on": len(on.get("region_provenance") or []),
        "block_resegment_off": bool(off.get("block_resegment")),
        "block_resegment_on": bool(on.get("block_resegment")),
        "id_set_delta": len(off_ids) - len(on_ids),
        "off_only_ids": len(off_ids - on_ids),
        "on_only_ids": len(on_ids - off_ids),
    }


# ---------------------------------------------------------------------------
# Rollup + CLI.
# ---------------------------------------------------------------------------

# Acceptance thresholds (§5 of the ITEM1 spec). ``label_only_reduction`` is the
# arm-1 (anchored) gate; an anchor-free control doc trivially passes it (its
# OFF population is 0, so no reduction is required).
THRESHOLDS = {
    "max_over_merge_flags": 0,
    "min_label_only_reduction": 0.50,
}


def _rollup(per_doc: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    total_over = sum(d.get("over_merge_flags", 0) for d in per_doc)
    total_ops = sum(d.get("regroup_ops", 0) for d in per_doc)
    total_off = sum(d.get("under_merge_off", 0) for d in per_doc)
    total_on = sum(d.get("under_merge_on", 0) for d in per_doc)
    reduction = 0.0 if total_off == 0 else 1.0 - (total_on / total_off)
    anchored = [d for d in per_doc if d.get("under_merge_off", 0) > 0]
    over_ok = total_over <= THRESHOLDS["max_over_merge_flags"]
    # arm-1 reduction gate applies only where there IS a label-only population.
    reduction_ok = (not anchored) or (
        reduction >= THRESHOLDS["min_label_only_reduction"]
    )
    return {
        "mode": mode,
        "n_docs": len(per_doc),
        "total_regroup_ops": total_ops,
        "total_over_merge_flags": total_over,
        "label_only_off": total_off,
        "label_only_on": total_on,
        "label_only_reduction": round(reduction, 4),
        "thresholds": THRESHOLDS,
        "over_merge_gate_pass": over_ok,
        "label_only_reduction_gate_pass": reduction_ok,
        "all_gates_pass": bool(over_ok and reduction_ok),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge-json", action="append", default=[], help="prior conversion bridge JSON (detector mode; repeatable)")
    ap.add_argument("--pdf", action="append", default=[], help="source PDF (cascade mode; repeatable)")
    ap.add_argument("--corpus-dir", default=None, help="dir of *.pdf (cascade mode)")
    ap.add_argument("--out", required=True, help="report JSON path")
    ap.add_argument("--mode", choices=("detector", "cascade"), default="detector")
    args = ap.parse_args(argv)

    per_doc: list[dict[str, Any]] = []
    if args.mode == "detector":
        if not args.bridge_json:
            ap.error("detector mode requires >=1 --bridge-json")
        for bj in args.bridge_json:
            doc = json.load(open(bj))
            prov = doc.get("region_provenance") or []
            per_doc.append(_analyse_detector(Path(bj).name, prov))
    else:
        pdfs = list(args.pdf)
        if args.corpus_dir:
            pdfs += sorted(glob.glob(str(Path(args.corpus_dir) / "*.pdf")))
        if not pdfs:
            ap.error("cascade mode requires --pdf or --corpus-dir")
        tmpdir = Path(args.out).resolve().parent
        for pdf in pdfs:
            per_doc.append(_analyse_cascade(pdf, tmpdir))

    report = {
        "schema": "regroup-calibration-ab/1.0",
        "mode": args.mode,
        "rollup": _rollup(per_doc, args.mode),
        "per_doc": per_doc,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    roll = report["rollup"]
    print(json.dumps(roll, indent=2, sort_keys=True))
    return 0 if roll.get("all_gates_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
