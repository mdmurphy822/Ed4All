"""W2 §4.3 — deterministic CO→TO backlink invariants."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.lo_backlink import (  # noqa: E402
    BacklinkResult,
    backlink_cos_to_tos,
    resolve_reassign_mode,
    resolve_to_backlink_floor,
)
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402


def test_single_to_shortcut():
    terminals = [{"id": "TO-01", "statement": "Master arithmetic."}]
    chapters = [
        {"id": "CO-01", "statement": "Add numbers."},
        {"id": "CO-02", "statement": "Subtract numbers."},
    ]
    backlink_cos_to_tos(terminals, chapters)
    assert all(co["terminal_id"] == "TO-01" for co in chapters)


def test_nearest_to_by_embedding():
    terminals = [
        {"id": "TO-01", "statement": "Understand algebra equations variables."},
        {"id": "TO-02", "statement": "Understand geometry shapes triangles angles."},
    ]
    chapters = [
        {"id": "CO-01", "statement": "Solve algebra equations with variables."},
        {"id": "CO-02", "statement": "Measure triangle angles in geometry shapes."},
    ]
    backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert chapters[0]["terminal_id"] == "TO-01"
    assert chapters[1]["terminal_id"] == "TO-02"


def test_token_overlap_fallback_never_unset():
    """Embeddings absent → token-overlap fallback; terminal_id always set."""
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure"},
    ]
    chapters = [
        {"id": "CO-01", "statement": "solve linear algebra equations"},
    ]
    backlink_cos_to_tos(terminals, chapters, embed=None)
    assert chapters[0]["terminal_id"] == "TO-01"


def test_reconcile_hint_honored():
    terminals = [
        {"id": "TO-01", "statement": "alpha"},
        {"id": "TO-02", "statement": "beta"},
    ]
    chapters = [{"id": "CO-01", "statement": "gamma", "terminal_id": "TO-02"}]
    backlink_cos_to_tos(terminals, chapters, embed=None)
    assert chapters[0]["terminal_id"] == "TO-02"


# ---------------------------------------------------------------------------
# WS2 §5.2 — cosine floor + weak-link instrumentation (PURE MEASUREMENT).
# The four tests above assert ONLY terminal_id and stay green: the
# None→BacklinkResult signature change is invisible to them.
# ---------------------------------------------------------------------------


def test_weak_link_count_fires_below_floor():
    """A disjoint CO argmaxes (still assigned) but is stamped weak (score ≈ 0)."""
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure"},
    ]
    chapters = [
        {"id": "CO-01", "statement": "photosynthesis chlorophyll sunlight plants"},
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert isinstance(result, BacklinkResult)
    assert result.weak_link_count == 1
    assert result.scored_count == 1
    assert chapters[0]["weak_terminal_link"] is True
    assert chapters[0]["weak_terminal_link_score"] < 0.01
    # never-unset contract: terminal_id STILL assigned.
    assert chapters[0]["terminal_id"] in {"TO-01", "TO-02"}


def test_strong_link_not_stamped():
    """An overlapping CO scores above the floor → NOT stamped."""
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear systems"},
        {"id": "TO-02", "statement": "geometry triangle angle measure shapes"},
    ]
    chapters = [
        {
            "id": "CO-01",
            "statement": "algebra equations variables linear systems",
        },
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.weak_link_count == 0
    assert chapters[0].get("weak_terminal_link") is None
    assert chapters[0]["terminal_id"] == "TO-01"


def test_floor_is_read_from_env(monkeypatch):
    """ED4ALL_TO_BACKLINK_FLOOR overrides cosine; garbage → default."""
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear systems"},
        {"id": "TO-02", "statement": "geometry triangle angle measure shapes"},
    ]

    def _strong():
        # Overlapping but NOT identical to TO-01 → cosine in (0.45, 0.99): above
        # the default floor (strong) but below 0.99.
        return [
            {
                "id": "CO-01",
                "statement": "algebra equations variables",
            },
        ]

    # default floor (0.45) → the overlapping link is strong (not weak).
    monkeypatch.delenv("ED4ALL_TO_BACKLINK_FLOOR", raising=False)
    chapters = _strong()
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.weak_link_count == 0

    # 0.99 floor → the same (sub-1.0) link is now weak.
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "0.99")
    chapters = _strong()
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.cosine_floor == 0.99
    assert result.weak_link_count == 1

    # 0.0 floor → instrumentation off, nothing weak.
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "0.0")
    chapters = _strong()
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.cosine_floor == 0.0
    assert result.weak_link_count == 0

    # garbage → default 0.45.
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "not-a-float")
    chapters = _strong()
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.cosine_floor == 0.45


def test_weak_rate_aggregate():
    """3 weak + 1 strong → count 3, scored 4, rate 0.75."""
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear systems"},
        {"id": "TO-02", "statement": "geometry triangle angle measure shapes"},
    ]
    chapters = [
        {"id": "CO-01", "statement": "photosynthesis chlorophyll sunlight"},
        {"id": "CO-02", "statement": "mitochondria cellular respiration energy"},
        {"id": "CO-03", "statement": "tectonic plates earthquake seismic"},
        {
            "id": "CO-04",
            "statement": "algebra equations variables linear systems",
        },
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.weak_link_count == 3
    assert result.scored_count == 4
    assert result.weak_link_rate == 0.75


def test_token_fallback_uses_token_floor():
    """embed=None → token-overlap path; token_floor applied, low-overlap stamped."""
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure"},
    ]
    chapters = [
        {"id": "CO-01", "statement": "photosynthesis chlorophyll sunlight plants"},
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=None)
    assert result.used_embeddings is False
    assert result.token_floor == 0.10
    assert result.weak_link_count == 1
    assert chapters[0]["weak_terminal_link"] is True


def test_hint_honored_not_scored():
    """A resolvable hint is honored and EXCLUDED from scored_count (§3.4)."""
    terminals = [
        {"id": "TO-01", "statement": "alpha beta gamma"},
        {"id": "TO-02", "statement": "delta epsilon zeta"},
    ]
    chapters = [
        {"id": "CO-01", "statement": "unrelated words here", "terminal_id": "TO-02"},
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert chapters[0]["terminal_id"] == "TO-02"
    assert result.scored_count == 0
    assert result.weak_link_count == 0
    assert chapters[0].get("weak_terminal_link") is None


def test_single_to_no_weak():
    """Single-TO early return: nothing scored, nothing weak."""
    terminals = [{"id": "TO-01", "statement": "master arithmetic"}]
    chapters = [
        {"id": "CO-01", "statement": "photosynthesis chlorophyll"},
        {"id": "CO-02", "statement": "tectonic plates"},
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.weak_link_count == 0
    assert result.scored_count == 0
    assert all(co.get("weak_terminal_link") is None for co in chapters)
    assert all(co["terminal_id"] == "TO-01" for co in chapters)


def test_empty_set_no_weak():
    """No valid TO id → empty-set early return: nothing scored, nothing weak."""
    terminals = [{"id": "", "statement": "no id"}]
    chapters = [{"id": "CO-01", "statement": "photosynthesis chlorophyll"}]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.weak_link_count == 0
    assert result.scored_count == 0
    assert chapters[0].get("weak_terminal_link") is None
    assert chapters[0]["terminal_id"] == ""


def test_dual_floor_resolver(monkeypatch):
    """Direct unit test of resolve_to_backlink_floor."""
    monkeypatch.delenv("ED4ALL_TO_BACKLINK_FLOOR", raising=False)
    # defaults
    assert resolve_to_backlink_floor() == (0.45, 0.10)
    # explicit-arg-wins (each clamped to [0,1]; out-of-range → default)
    assert resolve_to_backlink_floor(0.7, 0.3) == (0.7, 0.3)
    assert resolve_to_backlink_floor(1.5, -0.2) == (0.45, 0.10)
    # bare float env → cosine only
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "0.6")
    assert resolve_to_backlink_floor() == (0.6, 0.10)
    # pair env → both
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "0.6,0.2")
    assert resolve_to_backlink_floor() == (0.6, 0.2)
    # garbage env → both defaults
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "garbage")
    assert resolve_to_backlink_floor() == (0.45, 0.10)
    # out-of-range pair components → per-component default
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "2.0,0.2")
    assert resolve_to_backlink_floor() == (0.45, 0.2)
    # explicit arg wins over env
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_FLOOR", "0.6,0.2")
    assert resolve_to_backlink_floor(0.3, 0.05) == (0.3, 0.05)


# ---------------------------------------------------------------------------
# M5 Fix A — anti-junk-drawer reassignment + validator-parity scoring.
# ED4ALL_TO_BACKLINK_REASSIGN (default off):
#   - re-scores hint-honored COs so scored_count matches what
#     co_terminal_alignment audits (validator parity → weak-rate reconciles);
#   - re-points a below-floor CO to its strictly-better argmax TO;
#   - stamps weak_terminal_link on the FINAL (post-reassign) link.
# ---------------------------------------------------------------------------


def test_reassign_mode_resolver(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_BACKLINK_REASSIGN", raising=False)
    assert resolve_reassign_mode() is False
    assert resolve_reassign_mode(True) is True
    assert resolve_reassign_mode(False) is False
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "true")
    assert resolve_reassign_mode() is True
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "on")
    assert resolve_reassign_mode() is True
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "garbage")
    assert resolve_reassign_mode() is False
    # explicit arg wins over env
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "true")
    assert resolve_reassign_mode(False) is False


def test_off_mode_excludes_hints_from_score(monkeypatch):
    """Default (off): a hint-honored CO is NOT scored → not floor-eligible.

    This is the UNDER-REPORTING behavior the validator disagrees with: a
    cosine-0.2 hinted link is silently honored and never counted weak.
    """
    monkeypatch.delenv("ED4ALL_TO_BACKLINK_REASSIGN", raising=False)
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure"},
    ]
    # CO's text is disjoint from TO-01 (cosine ≈ 0) but it carries a HINT to it.
    chapters = [
        {
            "id": "CO-01",
            "statement": "photosynthesis chlorophyll sunlight plants",
            "terminal_id": "TO-01",
        },
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.scored_count == 0  # hint excluded
    assert result.weak_link_count == 0
    assert chapters[0].get("weak_terminal_link") is None
    assert chapters[0]["terminal_id"] == "TO-01"  # honored as-is


def test_reassign_stamps_weak_when_no_better_to(monkeypatch):
    """A CO whose nearest TO is cosine ~0.2 is STAMPED weak, not silent.

    Validator parity: in reassign mode the hint-honored CO IS scored, so the
    weak count matches the co_terminal_alignment recompute (1 of 1 below floor).
    """
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "true")
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure"},
    ]
    # Disjoint from BOTH TOs → argmax stays where it is (no better fit), but
    # the link IS below floor → must be stamped weak.
    chapters = [
        {
            "id": "CO-01",
            "statement": "photosynthesis chlorophyll sunlight plants cells",
            "terminal_id": "TO-01",
        },
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert result.scored_count == 1  # hint NOW scored (validator parity)
    assert result.weak_link_count == 1
    assert result.weak_link_rate == 1.0
    assert chapters[0]["weak_terminal_link"] is True
    assert chapters[0]["weak_terminal_link_score"] < 0.45
    assert result.reassigned_count == 0  # no better TO to move to


def test_reassign_repoints_to_better_terminal(monkeypatch):
    """A below-floor CO wrongly hinted to TO-01 is re-pointed to its real TO-02."""
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "true")
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure shapes"},
    ]
    # Statement is clearly geometry (TO-02) but the upstream hint mis-assigned
    # it to TO-01. cosine(CO, TO-01) is below the floor; cosine(CO, TO-02) is
    # high → reassign to TO-02.
    chapters = [
        {
            "id": "CO-01",
            "statement": "geometry triangle angle measure shapes",
            "terminal_id": "TO-01",
        },
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=FakeEmbed())
    assert chapters[0]["terminal_id"] == "TO-02"  # re-pointed off the junk drawer
    assert result.reassigned_count == 1
    # The final (TO-02) link is strong → NOT stamped weak.
    assert chapters[0].get("weak_terminal_link") is None
    assert result.weak_link_count == 0


def test_reassign_matches_validator_weak_count(monkeypatch):
    """Backlink weak-rate reconciles with a validator-style recompute.

    Mimics co_terminal_alignment: for every CO with a resolvable terminal_id,
    recompute cosine(co, assigned_to) at the SAME 0.45 floor. The backlink's
    weak_link_count (reassign mode) must equal that independent recount.
    """
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "true")
    embed = FakeEmbed()
    terminals = [
        {"id": "TO-01", "statement": "integer factors multiples divisibility lcm"},
        {"id": "TO-02", "statement": "geometry triangle angle measure shapes"},
    ]
    # 2 strong (match a TO), 2 junk-drawer hints to TO-01 with no better fit.
    chapters = [
        {"id": "CO-01", "statement": "integer factors multiples divisibility",
         "terminal_id": "TO-01"},
        {"id": "CO-02", "statement": "geometry triangle angle measure",
         "terminal_id": "TO-02"},
        {"id": "CO-03", "statement": "square roots radicals irrational",
         "terminal_id": "TO-01"},
        {"id": "CO-04", "statement": "additive inverse opposite number",
         "terminal_id": "TO-01"},
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=embed)

    # Independent validator-style recompute over the FINAL terminal_id.
    from lib.embedding._math import cosine_similarity
    to_vecs = {
        t["id"]: embed.encode_batch([t["statement"]])[0] for t in terminals
    }
    validator_weak = 0
    for co in chapters:
        co_vec = embed.encode_batch([co["statement"]])[0]
        cos = cosine_similarity(co_vec, to_vecs[co["terminal_id"]])
        if cos < 0.45:
            validator_weak += 1

    assert result.scored_count == 4  # all 4 audited (hints included)
    assert result.weak_link_count == validator_weak


def test_reassign_token_fallback(monkeypatch):
    """Reassign mode works on the embeddings-absent token-overlap path too."""
    monkeypatch.setenv("ED4ALL_TO_BACKLINK_REASSIGN", "true")
    terminals = [
        {"id": "TO-01", "statement": "algebra equations variables linear"},
        {"id": "TO-02", "statement": "geometry triangle angle measure"},
    ]
    chapters = [
        {"id": "CO-01", "statement": "photosynthesis chlorophyll sunlight",
         "terminal_id": "TO-01"},
    ]
    result = backlink_cos_to_tos(terminals, chapters, embed=None)
    assert result.used_embeddings is False
    assert result.scored_count == 1  # hint scored under token floor
    assert result.weak_link_count == 1
    assert chapters[0]["weak_terminal_link"] is True
