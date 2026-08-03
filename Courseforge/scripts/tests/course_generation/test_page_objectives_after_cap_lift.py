"""WS5 §6 — emit↔validator parity after the cap-lift (the §1 latent hazard).

The §1 correction: the validator's per-week allowed set comes from the
persisted ``"Week N"`` group labels (1:1 label-map branch), NOT from the
positional slicer. Pre-WS5, ``_plan_course_structure`` persisted one group per
CO (``"Week {idx}"``), so allowed-week-N = ``{chapter[N-1]}`` while emit-week-N
= the stride-``step`` slice — they only agreed at step==1, leaving a latent
``LO_SPECIFICITY_VIOLATION`` at any ``duration_weeks < len(chapter)``.

WS5 §2.4(A) persists one group per WEEK whose objectives are the SAME
ceil-stride slice the emitter places. This test builds that persisted shape for
62 COs + 10 TOs at weeks=10, then asserts each week's emitted LO set ⊆ the
validator's allowed set (no ``LO_SPECIFICITY_VIOLATION``).
"""

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_course import (  # noqa: E402
    _slice_cos_for_week,
    load_canonical_objectives,
    resolve_week_objectives,
)


def _build_synthesized(n_cos, n_tos, weeks):
    """Build the §2.4(A) persisted synthesized_objectives.json shape."""
    cos = [
        {"id": f"CO-{i:02d}", "terminal_id": f"TO-{(i % n_tos) + 1:02d}",
         "statement": f"chapter objective {i}", "bloomLevel": "apply"}
        for i in range(1, n_cos + 1)
    ]
    tos = [
        {"id": f"TO-{i:02d}", "statement": f"terminal objective {i}",
         "bloomLevel": "evaluate"}
        for i in range(1, n_tos + 1)
    ]
    # §2.4(A): one group per WEEK, objectives = emitter's ceil-stride slice.
    groups = [
        {
            "chapter": f"Week {w}",
            "objectives": [dict(c) for c in _slice_cos_for_week(cos, weeks, w)],
        }
        for w in range(1, weeks + 1)
    ]
    return {
        "course_name": "WS5_PARITY",
        "duration_weeks": weeks,
        "terminal_objectives": tos,
        "chapter_objectives": groups,
    }, cos, tos


class TestLiftedCapEmitPassesPageObjectives:
    @pytest.fixture
    def built(self, tmp_path):
        synthesized, cos, tos = _build_synthesized(62, 10, 10)
        p = tmp_path / "synthesized_objectives.json"
        p.write_text(json.dumps(synthesized))
        canonical = load_canonical_objectives(p)
        return canonical, cos, tos

    def test_lifted_cap_emit_subset_of_allowed_per_week(self, built):
        canonical, cos, tos = built
        to_ids = {t["id"] for t in tos}
        for w in range(1, 11):
            emit_cos = {o["id"] for o in _slice_cos_for_week(cos, 10, w)}
            emit_ids = emit_cos | to_ids  # week_objectives = terminals + COs
            allowed = {o["id"] for o in resolve_week_objectives(w, canonical)}
            extraneous = emit_ids - allowed
            assert not extraneous, (
                f"week {w} LO_SPECIFICITY_VIOLATION: emit ids not allowed: "
                f"{sorted(extraneous)}"
            )

    def test_all_62_cos_covered_across_weeks(self, built):
        canonical, cos, tos = built
        all_co_ids = {o["id"] for o in cos}
        covered = set()
        for w in range(1, 11):
            covered |= {
                o["id"] for o in resolve_week_objectives(w, canonical)
                if o["id"].startswith("CO")
            }
        assert all_co_ids <= covered, (
            f"validator allowed-set dropped COs: {sorted(all_co_ids - covered)[:10]}"
        )
