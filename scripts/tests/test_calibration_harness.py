"""Unit tests for scripts/calibration_harness.py.

All fixtures are SYNTHESIZED at fixture-load time (a hand-built validation-report dict
and a hand-built block_quality_rollup dict) — NO real course slug / path is referenced,
honoring the tier-2 discipline (tracked test code never pins a corpus slug). The tests
exercise the aggregation + flip heuristic + corpus-identity collapsing logic in
isolation from disk discovery.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "calibration_harness.py"
_spec = importlib.util.spec_from_file_location("calibration_harness", _SCRIPT)
ch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ch
_spec.loader.exec_module(ch)


# --------------------------------------------------------------------------------------
# Synthesized fixtures (NOT a real course)
# --------------------------------------------------------------------------------------
def _make_corpus(corpus_id: str, *, fired_gates: dict[str, int], total_blocks: int):
    """Build a CorpusResult by feeding a synthesized validation-report dict through the
    real reader, so the test exercises the production parse path.

    ``fired_gates`` maps gate_id -> number of blocks on which it FAILS (passed=False).
    """
    per_block = []
    for i in range(total_blocks):
        gate_results = []
        for gid, n_fail in fired_gates.items():
            gate_results.append(
                {
                    "gate_id": gid,
                    "passed": i >= n_fail,  # first n_fail blocks fail this gate
                    "issue_count": 1 if i < n_fail else 0,
                    "action": "regenerate" if i < n_fail else None,
                }
            )
        per_block.append({"block_id": f"blk_{i:03d}", "gate_results": gate_results})
    report_dict = {"per_block": per_block}

    res = ch.CorpusResult(
        corpus_id=corpus_id,
        origin="courseforge_export",
        path=f"/synth/{corpus_id}",
        corpus_key=ch._corpus_key(corpus_id),
    )
    # Drive the REAL production reader against a synthesized on-disk report.
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "report.json"
        rp.write_text(json.dumps(report_dict), encoding="utf-8")
        ch.read_validation_report(res, rp)
    return res


# --------------------------------------------------------------------------------------
# corpus_key derivation
# --------------------------------------------------------------------------------------
def test_corpus_key_strips_timestamp_prefix_and_variant():
    assert ch._corpus_key("PROJ-sample-course-a-20260621094139") == "sample-course-a"
    assert ch._corpus_key("PROJ-OPENSTAX_ALG_9-20260513113631") == "sample-course-a"
    assert ch._corpus_key("sample-course-b") == "sample-course-a"
    assert ch._corpus_key("biology-201") == "biology-201"


def test_corpus_key_dash_underscore_equivalence():
    # The dash vs underscore variants of one corpus must collapse to one key (the bug
    # that originally inflated distinct count to 2).
    assert ch._corpus_key("sample-course-a") == ch._corpus_key("OPENSTAX_ALG_9")


# --------------------------------------------------------------------------------------
# content-based corpus identity (collapses same-textbook, differently-named runs)
# --------------------------------------------------------------------------------------
def test_normalize_source_doc_collapses_slices_and_markers():
    # Partial section/chapter slices of one textbook + the SemantiK -accessible marker
    # + file extension all normalize to ONE stable identity.
    n = ch._normalize_source_doc
    assert n("sample-algebra-2e-s11to13_accessible.html") == "sample-algebra-2e"
    assert n("sample-algebra-2e-ch1-3_accessible") == "sample-algebra-2e"
    assert n("/abs/path/sample-algebra-2e-ch1to3_accessible.html") == "sample-algebra-2e"
    # A different textbook stays distinct.
    assert n("demo-ch1_accessible.html") == "demo"
    assert n("") == ""


def _write_export_with_blocks(root: Path, name: str, source_doc: str):
    """Synthesize a courseforge-export dir whose blocks cite ``source_doc`` (no slug)."""
    import json

    d = root / name
    (d / "04_rewrite").mkdir(parents=True)
    block = {
        "block_id": "blk_000",
        "source_ids": [f"dart:{source_doc}#aa11bb22", f"dart:{source_doc}#cc33dd44"],
        "source_references": [],
    }
    (d / "04_rewrite" / "blocks_final.jsonl").write_text(
        json.dumps(block) + "\n", encoding="utf-8"
    )
    return d


def _write_export_with_structure(root: Path, name: str, source_files: list[str]):
    """Synthesize an export dir whose textbook_structure cites ``source_files``."""
    import json

    d = root / name
    (d / "01_learning_objectives").mkdir(parents=True)
    struct = {
        "source_files": source_files,
        "chapters": [{"source_file": source_files[0]}],
    }
    (d / "01_learning_objectives" / "textbook_structure.json").write_text(
        json.dumps(struct), encoding="utf-8"
    )
    return d


def test_content_key_collapses_two_named_runs_of_one_textbook(tmp_path):
    # Two DIFFERENTLY-NAMED exports built from the same source document must share a key.
    a = _write_export_with_blocks(
        tmp_path, "course-a-calib", "sample-algebra-2e-s11to13_accessible"
    )
    b = _write_export_with_blocks(
        tmp_path, "sample-course-a", "sample-algebra-2e-ch1-3_accessible"
    )
    ka = ch._content_corpus_key(a, "course-a-calib")
    kb = ch._content_corpus_key(b, "sample-course-a")
    # Slug keys would have been distinct; content keys collapse to one.
    assert ka == kb == "src:sample-algebra-2e"
    assert ch._corpus_key("course-a-calib") != ch._corpus_key("sample-course-a")


def test_content_key_keeps_distinct_textbooks_distinct(tmp_path):
    alg = _write_export_with_blocks(
        tmp_path, "course-a-cal2", "sample-algebra-2e-s11to13_accessible"
    )
    demo = _write_export_with_blocks(
        tmp_path, "demo-ch1-calib", "demo-ch1_accessible"
    )
    assert ch._content_corpus_key(alg, "course-a-cal2") == "src:sample-algebra-2e"
    assert ch._content_corpus_key(demo, "demo-ch1-calib") == "src:demo"
    assert ch._content_corpus_key(alg, "course-a-cal2") != ch._content_corpus_key(
        demo, "demo-ch1-calib"
    )


def test_content_key_from_textbook_structure_when_no_blocks(tmp_path):
    # No blocks present -> falls through to textbook_structure source_files.
    d = _write_export_with_structure(
        tmp_path,
        "alg-from-structure",
        ["/inp/textbooks/TTC_x/sample-algebra-2e-s11to13_accessible.html"],
    )
    assert ch._content_corpus_key(d, "alg-from-structure") == "src:sample-algebra-2e"


def test_content_key_multidoc_corpus_resolves_dominant_doc(tmp_path):
    # A multi-document corpus resolves to its DOMINANT (most-cited) document
    # deterministically (the doc appearing most across source_files + chapter refs).
    d = _write_export_with_structure(
        tmp_path,
        "rdf-bundle",
        [
            "rdf11_primer_accessible.html",
            "rdf11_primer_accessible.html",
            "shacl_spec_accessible.html",
        ],
    )
    # rdf11_primer cited twice in source_files + once as chapter[0].source_file = 3 total
    # vs shacl_spec once -> dominant is rdf11-primer.
    assert ch._content_corpus_key(d, "rdf-bundle") == "src:rdf11-primer"


def test_content_key_falls_back_to_slug_when_no_content_signal(tmp_path):
    # Legacy export: dir exists but carries NO blocks / structure / chunk signal.
    d = tmp_path / "legacy-export"
    (d / "00_template_analysis").mkdir(parents=True)
    key = ch._content_corpus_key(d, "PROJ-legacy-export-20260101")
    assert key == ch._corpus_key("PROJ-legacy-export-20260101") == "legacy-export"


def test_content_key_never_crashes_on_malformed_blocks(tmp_path):
    # A malformed blocks file degrades to the slug key, never raises.
    d = tmp_path / "broken-export"
    (d / "04_rewrite").mkdir(parents=True)
    (d / "04_rewrite" / "blocks_final.jsonl").write_text(
        "{ this is not valid json\n", encoding="utf-8"
    )
    key = ch._content_corpus_key(d, "broken-export")
    # Malformed JSON line is skipped; no content signal -> slug fallback.
    assert key == "broken-export"


def test_discovery_collapses_same_textbook_runs(tmp_path):
    # End-to-end discovery: three differently-named exports of one textbook collapse to
    # ONE corpus_key, while a distinct textbook keys separately. Uses the --runs-dir
    # override so each child dir is treated as a courseforge export (hermetic, no real
    # course slug pinned).
    import json

    def _full_export(name: str, source_doc: str):
        d = _write_export_with_blocks(tmp_path, name, source_doc)
        # Give it a validation report so it produces >=1 observation (contributing).
        vr = d / "04_rewrite" / "02_validation_report"
        vr.mkdir(parents=True)
        report = {
            "per_block": [
                {
                    "block_id": "blk_000",
                    "gate_results": [
                        {"gate_id": "udl_coverage", "passed": True,
                         "issue_count": 0, "action": None},
                    ],
                }
            ]
        }
        (vr / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return d

    _full_export("course-a-cal2", "sample-algebra-2e-s11to13_accessible")
    _full_export("course-a-calib", "sample-algebra-2e-s11to13_accessible")
    _full_export("sample-course-a", "sample-algebra-2e-ch1-3_accessible")
    _full_export("demo-ch1-calib", "demo-ch1_accessible")

    corpora = ch.discover_corpora(course_filter=None, runs_dir=tmp_path)
    keys = {c.corpus_key for c in corpora}
    assert "src:sample-algebra-2e" in keys
    assert "src:demo" in keys
    # The three algebra exports collapsed to a single key.
    alg_runs = [c for c in corpora if c.corpus_key == "src:sample-algebra-2e"]
    assert len(alg_runs) == 3  # all three discovered...
    # ...but aggregate collapses them to ONE representative.
    distinct = {c.corpus_key for c in corpora}
    assert len(distinct) == 2  # sample-algebra-2e + demo, NOT 4


# --------------------------------------------------------------------------------------
# aggregation + flip heuristic
# --------------------------------------------------------------------------------------
def test_single_corpus_is_never_flip_ready():
    # One distinct corpus, even with a perfectly in-band fire rate, can never flip.
    c = _make_corpus(
        "synthcorpus-A-20260101000000",
        fired_gates={"co_terminal_alignment": 0},  # 0 fires -> rate 0.0 (in band)
        total_blocks=20,
    )
    report = ch.aggregate([c])
    assert report["distinct_corpus_count"] == 1
    fam = "WS3 CO<->TO semantic alignment"
    g = report["gate_families"][fam]
    assert g["flip_ready"] is False
    assert "needs >=2 corpora" in g["flip_blocked_reason"]


def test_two_distinct_corpora_in_band_flip_ready():
    # Two DISTINCT corpora, both with fire_rate within the documented band -> flip_ready.
    a = _make_corpus(
        "alpha-corpus-20260101000000",
        fired_gates={"co_terminal_alignment": 0},
        total_blocks=20,
    )
    b = _make_corpus(
        "beta-corpus-20260202000000",
        fired_gates={"co_terminal_alignment": 0},
        total_blocks=20,
    )
    report = ch.aggregate([a, b])
    assert report["distinct_corpus_count"] == 2
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    assert g["fire_rate"] == 0.0
    assert g["expected_band"]["in_band"] is True
    assert g["flip_ready"] is True
    assert g["flip_blocked_reason"] is None


def test_two_distinct_corpora_out_of_band_blocked():
    # Two distinct corpora but a fire_rate ABOVE the band -> not flip_ready, FP-audit msg.
    a = _make_corpus(
        "alpha-corpus-20260101000000",
        fired_gates={"co_terminal_alignment": 18},  # 18/20 = 0.9, way over 0.10 band
        total_blocks=20,
    )
    b = _make_corpus(
        "beta-corpus-20260202000000",
        fired_gates={"co_terminal_alignment": 16},  # 16/20 = 0.8
        total_blocks=20,
    )
    report = ch.aggregate([a, b])
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    assert g["corpora_count"] == 2
    assert g["fire_rate"] > 0.10
    assert g["flip_ready"] is False
    assert "manual FP audit" in g["flip_blocked_reason"]


def test_same_corpus_multiple_runs_count_as_one():
    # Ten timestamped runs of ONE underlying corpus must NOT satisfy the >=2 requirement.
    runs = [
        _make_corpus(
            f"PROJ-mybook-9-full7b-2026010{i}000000",
            fired_gates={"co_terminal_alignment": 0},
            total_blocks=10,
        )
        for i in range(1, 6)
    ]
    report = ch.aggregate(runs)
    assert report["distinct_corpus_count"] == 1
    assert report["contributing_run_count"] == 5
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    assert g["flip_ready"] is False
    assert "needs >=2 corpora" in g["flip_blocked_reason"]


def test_latest_run_is_representative():
    # When runs share a key, the lexicographically-latest (newest timestamp) wins, and
    # the per-corpus row carries the underlying key.
    old = _make_corpus(
        "PROJ-mybook-9-20260101000000",
        fired_gates={"co_terminal_alignment": 9},  # old run: many fires
        total_blocks=10,
    )
    new = _make_corpus(
        "PROJ-mybook-9-20260301000000",
        fired_gates={"co_terminal_alignment": 0},  # new run: clean
        total_blocks=10,
    )
    report = ch.aggregate([old, new])
    assert report["distinct_corpus_count"] == 1
    g = report["gate_families"]["WS3 CO<->TO semantic alignment"]
    # Representative is the NEW (clean) run -> fire_rate 0, not the old run's 0.9.
    assert g["fire_rate"] == 0.0
    assert g["corpora"][0]["corpus"] == "PROJ-mybook-9-20260301000000"


def test_rollup_reader_block_pass_drives_rubric_family():
    # Synthesized block_quality_rollup_report.json -> IB6.1 rubric family fires on
    # block_pass=False rows.
    import json
    import tempfile

    rollup = {
        "schema_version": "1.0",
        "per_block": [
            {"block_id": "b0", "module": "w1", "block_pass": True, "mean": 2.5,
             "weakest_dim": None, "accessibility_gate_fail": False},
            {"block_id": "b1", "module": "w1", "block_pass": False, "mean": 1.2,
             "weakest_dim": "feedback", "accessibility_gate_fail": False},
        ],
    }
    res = ch.CorpusResult(
        corpus_id="synthrollup-A", origin="libv2", path="/synth",
        corpus_key="synthrollup-a",
    )
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "block_quality_rollup_report.json"
        rp.write_text(json.dumps(rollup), encoding="utf-8")
        ch.read_block_quality_rollup(res, rp)
    obs = res.observations["IB6.1 eight-dimension block-quality rubric"]
    assert obs.evaluated == 2
    assert obs.fired == 1  # only b1 failed
    assert abs(obs.fire_rate - 0.5) < 1e-9


def test_report_has_all_gate_families():
    # Every configured framework gate family appears in the report even with no data.
    report = ch.aggregate([])
    assert set(report["gate_families"]) == {f.family for f in ch.GATE_FAMILIES}
    for g in report["gate_families"].values():
        assert g["flip_ready"] is False  # no data -> never ready


# --------------------------------------------------------------------------------------
# W8.2 — GATE_FAMILIES reconciled with config/workflows.yaml deferred-flip gates.
# --------------------------------------------------------------------------------------
def test_config_deferred_parser_associates_marker_with_own_gate_only():
    # Synthesize a tiny two-gate config where the FIRST gate has NO marker and the
    # SECOND gate's preceding comment carries TODO(calibration). The trailing comment
    # of gate-1's region (which is gate-2's leading comment) must NOT tag gate-1.
    import tempfile
    cfg = (
        "validation_gates:\n"
        "          - gate_id: plain_gate\n"
        "            severity: critical\n"
        "          # TODO(calibration) — flip DEFERRED after a >=2-corpus measurement.\n"
        "          - gate_id: deferred_gate\n"
        "            severity: warning\n"
        "            description: warning day-1.\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "workflows.yaml"
        p.write_text(cfg, encoding="utf-8")
        ids = ch.deferred_flip_gate_ids_from_config(p)
    assert "deferred_gate" in ids
    assert "plain_gate" not in ids  # trailing comment must not bleed onto the prior gate


def test_config_deferred_parser_reads_description_marker():
    import tempfile
    cfg = (
        "          - gate_id: desc_marked\n"
        "            severity: warning\n"
        "            description: X. Warning day-1; TODO(calibration) deferred flip.\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "workflows.yaml"
        p.write_text(cfg, encoding="utf-8")
        assert "desc_marked" in ch.deferred_flip_gate_ids_from_config(p)


def test_config_deferred_parser_graceful_on_missing_file(tmp_path):
    # Unreadable config -> empty set (harness degrades to curated, never crashes).
    assert ch.deferred_flip_gate_ids_from_config(tmp_path / "nope.yaml") == frozenset()


def test_gate_families_reconciled_with_live_config_no_drift():
    # The keystone anti-drift invariant: EVERY deferred-flip gate in the live
    # config/workflows.yaml is covered by SOME family in GATE_FAMILIES (curated or
    # auto-synthesized). Zero uncovered.
    config_ids = ch.deferred_flip_gate_ids_from_config()
    assert config_ids, "live config must parse to a non-empty deferred-gate set"
    covered = {gid for fam in ch.GATE_FAMILIES for gid in fam.gate_ids}
    uncovered = sorted(config_ids - covered)
    assert uncovered == [], f"GATE_FAMILIES drifted behind config; uncovered: {uncovered}"


def test_reconciliation_summary_invariant_in_report():
    report = ch.aggregate([])
    rec = report["gate_family_reconciliation"]
    assert rec["config_readable"] is True
    assert rec["uncovered_deferred_gates"] == []
    # Auto + curated coverage together account for every deferred gate.
    assert rec["config_deferred_gate_count"] == len(
        ch.deferred_flip_gate_ids_from_config()
    )


def test_uncurated_deferred_gate_is_auto_synthesized():
    # A deferred gate NOT in the curated table gets a one-gate auto family so it is
    # still measured (drift-guard).
    auto = ch._synthesize_missing_families(
        ch._CURATED_GATE_FAMILIES, frozenset({"brand_new_deferred_gate"})
    )
    assert len(auto) == 1
    fam = auto[0]
    assert fam.gate_ids == ("brand_new_deferred_gate",)
    assert fam.family.startswith("(auto) ")
    assert "AUTO-DERIVED" in fam.band_source
    # A gate ALREADY covered by curated is NOT re-synthesized.
    curated_gid = ch._CURATED_GATE_FAMILIES[0].gate_ids[0]
    assert ch._synthesize_missing_families(
        ch._CURATED_GATE_FAMILIES, frozenset({curated_gid})
    ) == ()


# --------------------------------------------------------------------------------------
# schema-v2 attribution: per-block fire_rate from per-block issue_count, NOT the
# smeared phase-level ``passed`` flag (the bug this fix removes).
# --------------------------------------------------------------------------------------
def _read_report_dict(report_dict: dict) -> "ch.CorpusResult":
    import json
    import tempfile

    res = ch.CorpusResult(
        corpus_id="attrib-corpus-20260101000000",
        origin="courseforge_export",
        path="/synth/attrib",
        corpus_key="attrib-corpus",
    )
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "report.json"
        rp.write_text(json.dumps(report_dict), encoding="utf-8")
        ch.read_validation_report(res, rp)
    return res


def test_block_fire_rate_uses_per_block_issue_count_not_phase_passed():
    # A block-scoped gate (udl_coverage) failed for the PHASE (passed=False on
    # every block, the schema-v2 phase-level flag), but only ONE of four blocks
    # actually carries an issue (issue_count>0). Fire rate must be 1/4, not 4/4.
    per_block = []
    for i in range(4):
        per_block.append(
            {
                "block_id": f"blk_{i:03d}",
                "gate_results": [
                    {
                        "gate_id": "udl_coverage",
                        "passed": False,            # PHASE-level: smeared on all
                        "issue_count": 1 if i == 0 else 0,  # only blk_000 fired
                        "action": "regenerate",
                    }
                ],
            }
        )
    report_dict = {"per_block": per_block, "phase_level_gate_results": []}
    res = _read_report_dict(report_dict)
    obs = res.observations["IB4 UDL multiple-means coverage"]
    assert obs.evaluated == 4
    assert obs.fired == 1            # NOT 4 (the schema-v1 smear)
    assert abs(obs.fire_rate - 0.25) < 1e-9


def test_structural_gate_read_from_phase_level_section():
    # Module-scoped retrieval_presence is read from phase_level_gate_results,
    # not smeared across per_block. A block-scoped gate in the SAME phase-level
    # list must NOT be double-counted from there.
    report_dict = {
        "per_block": [
            {
                "block_id": "blk_000",
                "gate_results": [
                    {"gate_id": "udl_coverage", "passed": False,
                     "issue_count": 1, "action": "regenerate"},
                ],
            }
        ],
        "phase_level_gate_results": [
            {"gate_id": "retrieval_presence", "passed": False,
             "issue_count": 3, "unattributed_issue_count": 3,
             "action": "regenerate"},
            # A block-scoped gate also appears here (total view) — must be
            # ignored by the phase-level reader to avoid double counting.
            {"gate_id": "udl_coverage", "passed": False,
             "issue_count": 1, "unattributed_issue_count": 0,
             "action": "regenerate"},
        ],
    }
    res = _read_report_dict(report_dict)

    # retrieval_presence (module-scoped) fired once from the phase-level section.
    rp_obs = res.observations["IB7.5b retrieval presence (module-scoped)"]
    assert rp_obs.evaluated == 1
    assert rp_obs.fired == 1

    # udl_coverage counted ONCE (from per_block), not twice.
    udl_obs = res.observations["IB4 UDL multiple-means coverage"]
    assert udl_obs.evaluated == 1
    assert udl_obs.fired == 1


# --------------------------------------------------------------------------------------
# Aggregator (courseforge_validation_report.json) reader — the pre-Courseforge phase
# gates (chunk_wcag_status / co_terminal_alignment / source_coverage /
# objective_source_refs) are surfaced ONLY here, not in the two-pass report.json.
# --------------------------------------------------------------------------------------
def _read_aggregator_dict(agg_dict: dict, corpus_id: str = "agg-corpus-20260101000000"):
    import json
    import tempfile

    res = ch.CorpusResult(
        corpus_id=corpus_id,
        origin="courseforge_export",
        path=f"/synth/{corpus_id}",
        corpus_key=ch._corpus_key(corpus_id),
    )
    with tempfile.TemporaryDirectory() as td:
        ap = Path(td) / "courseforge_validation_report.json"
        ap.write_text(json.dumps(agg_dict), encoding="utf-8")
        ch.read_courseforge_validation_report(res, ap)
    return res


def test_aggregator_reader_records_pre_courseforge_phase_families():
    # A full-run aggregator carrying chunk_wcag_status (chunking phase),
    # co_terminal_alignment + source_coverage + objective_source_refs (course_planning)
    # -> each family records ONE phase-level observation that fires iff issue_count > 0.
    # ``passed`` stays True (warning-day-1) and must NOT suppress the fire.
    agg = {
        "per_phase": [
            {
                "phase": "chunking",
                "gates": [
                    {"gate_id": "chunk_wcag_status", "passed": True,
                     "issue_count": 0, "severity": "critical", "action": None},
                ],
            },
            {
                "phase": "course_planning",
                "gates": [
                    {"gate_id": "co_terminal_alignment", "passed": True,
                     "issue_count": 0, "severity": "critical", "action": None},
                    {"gate_id": "source_coverage", "passed": True,
                     "issue_count": 7, "severity": "critical", "action": "regenerate",
                     "top_issues": [{"code": "SOURCE_SECTION_UNCOVERED",
                                     "location": "ch1-sec3", "severity": "warning"}]},
                    {"gate_id": "objective_source_refs", "passed": True,
                     "issue_count": 3, "severity": "critical", "action": "regenerate"},
                ],
            },
        ]
    }
    res = _read_aggregator_dict(agg)

    wcag = res.observations["IB4 chunk WCAG status"]
    assert wcag.evaluated == 1 and wcag.fired == 0  # 0 issues -> clean

    align = res.observations["WS3 CO<->TO semantic alignment"]
    assert align.evaluated == 1 and align.fired == 0

    cov = res.observations["WS6a/I3 source->objective coverage"]
    # source_coverage(7) + objective_source_refs(3) BOTH map to this one family;
    # each is one phase-level evaluated unit, both fired.
    assert cov.evaluated == 2 and cov.fired == 2
    # sample carries the first top_issue location for auditability.
    locs = [s.get("first_issue_location") for s in cov.samples]
    assert "ch1-sec3" in locs


def test_aggregator_reader_does_not_double_count_already_observed_family():
    # If a richer reader already recorded a family for this corpus, the aggregator
    # reader must NOT override it (no double counting). udl_coverage is read per-block;
    # if it also showed in the aggregator we'd skip it — but udl_coverage isn't an
    # aggregator-only id, so here we assert the guard on a shared family via a manual
    # pre-existing observation on the source->objective coverage family.
    res = ch.CorpusResult(
        corpus_id="guard-corpus-20260101000000",
        origin="courseforge_export",
        path="/synth/guard",
        corpus_key="guard-corpus",
    )
    # Pre-seed an observation for the coverage family (as if a per-block reader ran).
    pre = res.obs("WS6a/I3 source->objective coverage")
    pre.evaluated = 5
    pre.fired = 1

    import json
    import tempfile

    agg = {
        "per_phase": [
            {
                "phase": "course_planning",
                "gates": [
                    {"gate_id": "source_coverage", "passed": True,
                     "issue_count": 99, "severity": "critical", "action": "regenerate"},
                ],
            }
        ]
    }
    with tempfile.TemporaryDirectory() as td:
        ap = Path(td) / "courseforge_validation_report.json"
        ap.write_text(json.dumps(agg), encoding="utf-8")
        ch.read_courseforge_validation_report(res, ap)

    cov = res.observations["WS6a/I3 source->objective coverage"]
    # Untouched: the pre-existing per-block observation wins; no aggregator override.
    assert cov.evaluated == 5 and cov.fired == 1


def test_aggregator_reader_ignores_non_aggregator_only_gate_ids():
    # Gates that ALSO appear in the aggregator per_phase but are read elsewhere
    # (e.g. udl_coverage, bloom_type_range) must be IGNORED by this reader so they
    # aren't double-counted against the per-block reader.
    agg = {
        "per_phase": [
            {
                "phase": "post_rewrite_validation",
                "gates": [
                    {"gate_id": "udl_coverage", "passed": False,
                     "issue_count": 4, "severity": "warning", "action": "regenerate"},
                    {"gate_id": "bloom_type_range", "passed": False,
                     "issue_count": 2, "severity": "warning", "action": "regenerate"},
                ],
            }
        ]
    }
    res = _read_aggregator_dict(agg)
    assert "IB4 UDL multiple-means coverage" not in res.observations
    assert "IB7.6c per-type Bloom-range ceiling" not in res.observations


def test_legacy_v1_report_without_phase_level_section_degrades_gracefully():
    # A legacy v1 report (no phase_level_gate_results) still reads block-scoped
    # gates; structural/module families are simply not observed from it.
    report_dict = {
        "per_block": [
            {
                "block_id": "blk_000",
                "gate_results": [
                    {"gate_id": "udl_coverage", "passed": False,
                     "issue_count": 1, "action": "regenerate"},
                ],
            }
        ],
        # no phase_level_gate_results key
    }
    res = _read_report_dict(report_dict)
    assert res.observations["IB4 UDL multiple-means coverage"].fired == 1
    # retrieval_presence never observed (no phase-level section).
    assert "IB7.5b retrieval presence (module-scoped)" not in res.observations
