"""Tests for the grounding-safe objective-review pass.

Covers (per the task gate):
  (a) a CO statement improved while staying aligned → adopted;
  (b) a CO statement that drifts below the alignment floor → REVERTED;
  (c) dropped / duplicated / renamed ids → ignored, original id set preserved;
  (d) a TO statement change → adopted after bloom check;
  (e) a bloom verb/level mismatch → reverted.
  + source_refs byte-identical pre/post for every CO.
  + the objective_review DecisionCapture fires with a dynamic rationale and
    the decision_type is in the schema enum.
  + default-OFF (flag unset) is a no-op: byte-identical, no provider, no net.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.objectives.objective_review import (
    _DECISION_TYPE,
    merge_reviewed_objectives,
    review_objectives,
    resolve_objective_review_provider,
)
from lib.objectives.objective_review import _GroundingScorer


# ---------------------------------------------------------------------------
# Local fakes (FakeEmbed in _fakes.py only exposes encode_batch; the scorer
# needs encode()).
# ---------------------------------------------------------------------------


class _Embed:
    """Deterministic single-text encoder over a fixed bag-of-words basis.

    Cosine ≈ token-overlap, so we can author statements whose alignment we
    control: a CO sharing the TO's tokens scores high; a disjoint rewrite
    scores ~0.
    """

    _DIM = 512

    @staticmethod
    def _bucket(token: str) -> int:
        import hashlib

        h = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % _Embed._DIM

    def encode(self, text: str) -> List[float]:
        toks = [w.lower().strip(".,;:!?()'\"") for w in str(text).split()]
        toks = [w for w in toks if len(w) > 2]
        vec = [0.0] * self._DIM
        for w in toks:
            vec[self._bucket(w)] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class _Capture:
    """Minimal DecisionCapture stand-in recording log_decision calls."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _FakeClient:
    """OpenAICompatibleClient stand-in returning a canned JSON string.

    The review module constructs the real OpenAICompatibleClient but accepts
    an injected ``client`` (httpx-shaped); rather than mock the HTTP layer we
    monkeypatch the module's chat dispatch. See the fixtures below.
    """


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_objectives() -> Dict[str, Any]:
    """Two TOs, three COs. COs carry source_refs (TOs do not, by contract).

    CO-01/CO-02 roll up to TO-01 (share its tokens); CO-03 to TO-02.
    """
    terminals = [
        {
            "id": "TO-01",
            "statement": "Analyze cellular respiration energy pathways",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "abcd": {
                "audience": "students",
                "behavior": {"verb": "analyze", "action_object": "pathways"},
                "condition": "given a diagram",
                "degree": "accurately",
            },
            "source_refs": [],
        },
        {
            "id": "TO-02",
            "statement": "Evaluate photosynthesis light reactions experiments",
            "bloom_level": "evaluate",
            "bloom_verb": "evaluate",
            "abcd": {
                "audience": "students",
                "behavior": {"verb": "evaluate", "action_object": "experiments"},
                "condition": "given data",
                "degree": "with justification",
            },
            "source_refs": [],
        },
    ]
    chapter_objectives = [
        {
            "chapter": "all",
            "objectives": [
                {
                    "id": "CO-01",
                    "statement": "Describe cellular respiration energy pathways",
                    "bloom_level": "understand",
                    "bloom_verb": "describe",
                    "abcd": {
                        "audience": "students",
                        "behavior": {
                            "verb": "describe",
                            "action_object": "pathways",
                        },
                        "condition": "given notes",
                        "degree": "completely",
                    },
                    "terminal_id": "TO-01",
                    "source_refs": [
                        {"ref": "ch1#s1", "chunk_ids": ["c_0001", "c_0002"]}
                    ],
                },
                {
                    "id": "CO-02",
                    "statement": "List cellular respiration energy stages",
                    "bloom_level": "remember",
                    "bloom_verb": "list",
                    "abcd": {
                        "audience": "students",
                        "behavior": {
                            "verb": "list",
                            "action_object": "stages",
                        },
                        "condition": "from memory",
                        "degree": "in order",
                    },
                    "terminal_id": "TO-01",
                    "source_refs": [
                        {"ref": "ch1#s2", "chunk_ids": ["c_0003"]}
                    ],
                },
                {
                    "id": "CO-03",
                    "statement": "Explain photosynthesis light reactions experiments",
                    "bloom_level": "understand",
                    "bloom_verb": "explain",
                    "abcd": {
                        "audience": "students",
                        "behavior": {
                            "verb": "explain",
                            "action_object": "experiments",
                        },
                        "condition": "given a figure",
                        "degree": "clearly",
                    },
                    "terminal_id": "TO-02",
                    "source_refs": [
                        {"ref": "ch2#s1", "chunk_ids": ["c_0010"]}
                    ],
                },
            ],
        }
    ]
    return {"terminals": terminals, "chapter_objectives": chapter_objectives}


def _source_refs_snapshot(chapter_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for group in chapter_objectives:
        for co in group.get("objectives", []):
            snap[co["id"]] = copy.deepcopy(co.get("source_refs"))
    return snap


# ---------------------------------------------------------------------------
# (a)-(e) + source_refs byte-identity — exercised via merge (no network)
# ---------------------------------------------------------------------------


def test_merge_co_statement_improved_aligned_adopted():
    """(a) A CO statement that stays aligned (shares TO tokens) → adopted."""
    obj = _make_objectives()
    refs_before = _source_refs_snapshot(obj["chapter_objectives"])
    scorer = _GroundingScorer(_Embed())
    adjusted = {
        # Adds "metabolic" but keeps the TO's anchoring tokens → still aligned.
        "CO-01": {
            "statement": "Describe cellular respiration metabolic energy pathways"
        }
    }
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    co1 = obj["chapter_objectives"][0]["objectives"][0]
    assert co1["statement"] == (
        "Describe cellular respiration metabolic energy pathways"
    )
    assert counters["statements_adjusted"] == 1
    assert counters["statements_reverted"] == 0
    # source_refs byte-identical pre/post for every CO.
    assert _source_refs_snapshot(obj["chapter_objectives"]) == refs_before


def test_merge_co_statement_drift_below_floor_reverted():
    """(b) A CO statement drifting to disjoint tokens → REVERTED."""
    obj = _make_objectives()
    refs_before = _source_refs_snapshot(obj["chapter_objectives"])
    original = obj["chapter_objectives"][0]["objectives"][0]["statement"]
    scorer = _GroundingScorer(_Embed())
    adjusted = {
        # Totally disjoint from TO-01 → cosine ~0 < 0.45 → revert.
        "CO-01": {"statement": "Recall medieval poetry rhyme schemes thoroughly"}
    }
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    co1 = obj["chapter_objectives"][0]["objectives"][0]
    assert co1["statement"] == original  # reverted
    assert counters["statements_adjusted"] == 0
    assert counters["statements_reverted"] == 1
    assert _source_refs_snapshot(obj["chapter_objectives"]) == refs_before


def test_merge_dropped_duplicated_renamed_ids_ignored():
    """(c) Unknown / renamed ids dropped; original id set preserved."""
    obj = _make_objectives()
    ids_before = {
        co["id"]
        for g in obj["chapter_objectives"]
        for co in g["objectives"]
    } | {t["id"] for t in obj["terminals"]}
    scorer = _GroundingScorer(_Embed())
    adjusted = {
        # Renamed / invented ids — must be dropped, never added.
        "CO-99": {"statement": "Invented objective that should be ignored"},
        "TOTALLY-FAKE": {"statement": "junk"},
        # A real one, aligned, to confirm the real path still works.
        "CO-02": {
            "statement": "List cellular respiration energy stages carefully"
        },
    }
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    ids_after = {
        co["id"]
        for g in obj["chapter_objectives"]
        for co in g["objectives"]
    } | {t["id"] for t in obj["terminals"]}
    assert ids_after == ids_before  # no add / remove / rename
    assert counters["unknown_ids_dropped"] == 2
    # No new objective dicts materialised.
    assert len(obj["chapter_objectives"][0]["objectives"]) == 3
    assert len(obj["terminals"]) == 2


def test_merge_to_statement_change_adopted_after_bloom_check():
    """(d) A TO statement change → adopted (TOs carry no source_refs)."""
    obj = _make_objectives()
    scorer = _GroundingScorer(_Embed())
    adjusted = {
        "TO-01": {
            "statement": (
                "Analyze the energy pathways of cellular respiration in depth"
            )
        }
    }
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    assert obj["terminals"][0]["statement"] == (
        "Analyze the energy pathways of cellular respiration in depth"
    )
    assert counters["statements_adjusted"] == 1


def test_merge_bloom_mismatch_reverted():
    """(e) A bloom level/verb edit whose abcd verb misaligns → reverted."""
    obj = _make_objectives()
    scorer = _GroundingScorer(_Embed())
    # Change CO-01 to bloom_level "create" but its abcd verb stays "describe"
    # which is NOT in BLOOMS_VERBS["create"] → the whole bloom/abcd edit is
    # rejected.
    adjusted = {
        "CO-01": {
            "bloom_level": "create",
            "bloom_verb": "describe",
            # abcd left absent → candidate abcd is the original (verb=describe).
        }
    }
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    co1 = obj["chapter_objectives"][0]["objectives"][0]
    assert co1["bloom_level"] == "understand"  # original, reverted
    assert co1["bloom_verb"] == "describe"
    assert counters["bloom_edits_applied"] == 0
    assert counters["bloom_edits_rejected"] == 1


def test_merge_bloom_change_aligned_applied():
    """Sanity: a bloom edit whose abcd verb matches the new level is applied."""
    obj = _make_objectives()
    scorer = _GroundingScorer(_Embed())
    adjusted = {
        "CO-02": {
            "bloom_level": "understand",
            "bloom_verb": "describe",
            "abcd": {
                "audience": "students",
                "behavior": {"verb": "describe", "action_object": "stages"},
                "condition": "from notes",
                "degree": "in order",
            },
        }
    }
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    co2 = obj["chapter_objectives"][0]["objectives"][1]
    assert co2["bloom_level"] == "understand"
    assert counters["bloom_edits_applied"] >= 1
    assert counters["bloom_edits_rejected"] == 0


def test_merge_no_embedder_keeps_bloom_guardrail_skips_cosine():
    """No embedder → cosine guardrail off (CO statement edits adopted), but
    the bloom guardrail still reverts a mismatch."""
    obj = _make_objectives()
    scorer = _GroundingScorer(None)
    assert scorer.available is False
    adjusted = {
        "CO-01": {"statement": "Anything at all, no cosine check applies here"},
        "CO-02": {"bloom_level": "create", "bloom_verb": "list"},
    }
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    co1 = obj["chapter_objectives"][0]["objectives"][0]
    assert co1["statement"] == "Anything at all, no cosine check applies here"
    assert counters["statements_adjusted"] == 1
    # CO-02 bloom edit: list ∉ BLOOMS_VERBS["create"] → rejected.
    co2 = obj["chapter_objectives"][0]["objectives"][1]
    assert co2["bloom_level"] == "remember"
    assert counters["bloom_edits_rejected"] == 1


# ---------------------------------------------------------------------------
# Wave-2 Part 2: CO->TO remap (the relaxed guardrail) — via merge (no network)
# ---------------------------------------------------------------------------


def test_remap_co_to_better_to_sticks():
    """(a) A CO clearly fitting TO-B but assigned TO-A → remapped to TO-B,
    cosine improves, and the remap STICKS."""
    obj = _make_objectives()
    refs_before = _source_refs_snapshot(obj["chapter_objectives"])
    # CO-03's statement shares TO-02's tokens (photosynthesis light reactions)
    # but we deliberately MIS-assign it to TO-01 (cellular respiration) so the
    # reviewer must re-point it back to TO-02 (the better fit).
    co3 = obj["chapter_objectives"][0]["objectives"][2]
    co3["terminal_id"] = "TO-01"  # wrong parent on purpose
    scorer = _GroundingScorer(_Embed())
    adjusted = {"CO-03": {"terminal_id": "TO-02"}}
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    assert co3["terminal_id"] == "TO-02"  # remap applied
    assert counters["remaps_applied"] == 1
    assert counters["remaps_reverted"] == 0
    assert counters["remaps_unknown_to_dropped"] == 0
    # source_refs + id set untouched.
    assert _source_refs_snapshot(obj["chapter_objectives"]) == refs_before


def test_remap_that_worsens_alignment_is_reverted():
    """(b) A remap to a WORSE-aligned TO → REVERTED (keeps original parent)."""
    obj = _make_objectives()
    # CO-01 fits TO-01 (cellular respiration). The reviewer tries to re-point
    # it to TO-02 (photosynthesis) — strictly worse cosine → revert.
    co1 = obj["chapter_objectives"][0]["objectives"][0]
    assert co1["terminal_id"] == "TO-01"
    scorer = _GroundingScorer(_Embed())
    adjusted = {"CO-01": {"terminal_id": "TO-02"}}
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    assert co1["terminal_id"] == "TO-01"  # reverted — never-unset preserved
    assert counters["remaps_applied"] == 0
    assert counters["remaps_reverted"] == 1


def test_remap_to_nonexistent_to_is_dropped():
    """(c) A returned terminal_id NOT in the TO set → dropped (keeps original)."""
    obj = _make_objectives()
    co1 = obj["chapter_objectives"][0]["objectives"][0]
    scorer = _GroundingScorer(_Embed())
    adjusted = {"CO-01": {"terminal_id": "TO-99"}}  # no such TO
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    assert co1["terminal_id"] == "TO-01"  # original mapping preserved
    assert counters["remaps_applied"] == 0
    assert counters["remaps_reverted"] == 0
    assert counters["remaps_unknown_to_dropped"] == 1


def test_remap_does_not_change_id_set():
    """(d) A remap never adds/removes a CO or TO id — only the mapping moves."""
    obj = _make_objectives()
    co_ids_before = {
        co["id"] for g in obj["chapter_objectives"] for co in g["objectives"]
    }
    to_ids_before = {t["id"] for t in obj["terminals"]}
    co3 = obj["chapter_objectives"][0]["objectives"][2]
    co3["terminal_id"] = "TO-01"
    scorer = _GroundingScorer(_Embed())
    adjusted = {"CO-03": {"terminal_id": "TO-02"}}
    merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted=adjusted,
        scorer=scorer,
        threshold=0.45,
    )
    co_ids_after = {
        co["id"] for g in obj["chapter_objectives"] for co in g["objectives"]
    }
    to_ids_after = {t["id"] for t in obj["terminals"]}
    assert co_ids_after == co_ids_before
    assert to_ids_after == to_ids_before
    # The CO dict count is unchanged.
    assert len(obj["chapter_objectives"][0]["objectives"]) == 3
    assert len(obj["terminals"]) == 2


def test_remap_lifts_below_floor_link():
    """A below-floor old link lifted to >= floor by the remap → accepted.

    Uses an embedder where CO-X's tokens overlap TO-02 above the floor but
    its (mis)assigned TO-01 is below it — the remap is the cure."""
    obj = _make_objectives()
    co3 = obj["chapter_objectives"][0]["objectives"][2]
    # CO-03 is "Explain photosynthesis light reactions experiments" — token
    # overlap with TO-02 ("Evaluate photosynthesis light reactions
    # experiments") is high, with TO-01 (cellular respiration) ~0 (below floor).
    co3["terminal_id"] = "TO-01"
    scorer = _GroundingScorer(_Embed())
    old_cos = scorer.cosine(co3["statement"], obj["terminals"][0]["statement"])
    new_cos = scorer.cosine(co3["statement"], obj["terminals"][1]["statement"])
    assert old_cos < 0.45 <= new_cos  # the lift-below-floor condition holds
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted={"CO-03": {"terminal_id": "TO-02"}},
        scorer=scorer,
        threshold=0.45,
    )
    assert co3["terminal_id"] == "TO-02"
    assert counters["remaps_applied"] == 1


def test_remap_no_embedder_is_reverted():
    """No embedder → cannot prove improvement → remap conservatively reverted."""
    obj = _make_objectives()
    co3 = obj["chapter_objectives"][0]["objectives"][2]
    co3["terminal_id"] = "TO-01"
    scorer = _GroundingScorer(None)
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted={"CO-03": {"terminal_id": "TO-02"}},
        scorer=scorer,
        threshold=0.45,
    )
    assert co3["terminal_id"] == "TO-01"  # never re-point blind
    assert counters["remaps_applied"] == 0
    assert counters["remaps_reverted"] == 1


def test_remap_to_same_to_is_noop():
    """Re-asserting the current parent is not counted as a remap."""
    obj = _make_objectives()
    scorer = _GroundingScorer(_Embed())
    counters = merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted={"CO-01": {"terminal_id": "TO-01"}},  # already TO-01
        scorer=scorer,
        threshold=0.45,
    )
    assert counters["remaps_applied"] == 0
    assert counters["remaps_reverted"] == 0


def test_remap_writes_all_backpointer_keys():
    """A CO carrying parent_to + terminal_id has BOTH re-pointed (no stale)."""
    obj = _make_objectives()
    co3 = obj["chapter_objectives"][0]["objectives"][2]
    co3["terminal_id"] = "TO-01"
    co3["parent_to"] = "TO-01"
    scorer = _GroundingScorer(_Embed())
    merge_reviewed_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        adjusted={"CO-03": {"terminal_id": "TO-02"}},
        scorer=scorer,
        threshold=0.45,
    )
    assert co3["terminal_id"] == "TO-02"
    assert co3["parent_to"] == "TO-02"  # no stale back-pointer survives


def test_nvidia_review_model_defaults_to_70b(monkeypatch):
    """The nvidia review seat resolves to the 70B when no override is set."""
    from lib.objectives.objective_review import (
        resolve_objective_review_model,
        _NVIDIA_REVIEW_DEFAULT_MODEL,
    )

    monkeypatch.delenv("ED4ALL_OBJECTIVE_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_LARGE_MODEL", raising=False)
    assert resolve_objective_review_model("nvidia") == _NVIDIA_REVIEW_DEFAULT_MODEL
    assert _NVIDIA_REVIEW_DEFAULT_MODEL == "meta/llama-3.3-70b-instruct"
    # Operator overrides still win.
    monkeypatch.setenv("NVIDIA_LARGE_MODEL", "nvidia/custom")
    assert resolve_objective_review_model("nvidia") == "nvidia/custom"
    assert resolve_objective_review_model("nvidia", "explicit/x") == "explicit/x"


def test_review_objectives_emits_remap_counters(monkeypatch):
    """The end-to-end pass surfaces remap counts in the decision rationale."""
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    obj = _make_objectives()
    obj["chapter_objectives"][0]["objectives"][2]["terminal_id"] = "TO-01"
    refs_before = _source_refs_snapshot(obj["chapter_objectives"])
    # Single group → one CO chunk + one TO chunk. The CO chunk remaps CO-03.
    response = json.dumps({"adjusted": {"CO-03": {"terminal_id": "TO-02"}}})
    _patch_dispatch(monkeypatch, response)
    capture = _Capture()
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="BIO_101",
        capture=capture,
        embedder=_Embed(),
        client=object(),
    )
    assert result["applied"] is True
    assert result["counters"]["remaps_applied"] == 1
    co3 = obj["chapter_objectives"][0]["objectives"][2]
    assert co3["terminal_id"] == "TO-02"
    review_events = [
        e for e in capture.events if e.get("decision_type") == _DECISION_TYPE
    ]
    assert len(review_events) == 1
    rationale = review_events[0]["rationale"]
    assert "co_to_remaps_applied=1" in rationale
    assert "co_to_remaps_reverted_for_grounding=" in rationale
    assert _source_refs_snapshot(obj["chapter_objectives"]) == refs_before


# ---------------------------------------------------------------------------
# Decision-capture: objective_review fires with a dynamic rationale, and the
# decision_type is in the schema enum.
# ---------------------------------------------------------------------------


def _patch_dispatch(monkeypatch, response_json: str) -> None:
    """Patch the OpenAICompatibleClient chat dispatch to return canned JSON."""

    def _fake_chat(self, messages, *, max_tokens=0, temperature=0.0, **kw):
        return response_json

    import Trainforge.generators._openai_compatible_client as _oac

    monkeypatch.setattr(
        _oac.OpenAICompatibleClient, "chat_completion", _fake_chat
    )


def test_review_objectives_emits_decision_capture(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    obj = _make_objectives()
    refs_before = _source_refs_snapshot(obj["chapter_objectives"])
    response = json.dumps(
        {
            "adjusted": {
                "CO-01": {
                    "statement": (
                        "Describe cellular respiration metabolic energy "
                        "pathways"
                    )
                }
            }
        }
    )
    _patch_dispatch(monkeypatch, response)
    capture = _Capture()
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="BIO_101",
        capture=capture,
        embedder=_Embed(),
        client=object(),  # non-None so client construction doesn't need httpx
    )
    assert result["enabled"] is True
    assert result["applied"] is True
    # Exactly one objective_review event fired.
    review_events = [
        e for e in capture.events if e.get("decision_type") == _DECISION_TYPE
    ]
    assert len(review_events) == 1
    rationale = review_events[0]["rationale"]
    assert len(rationale) >= 20
    # Dynamic signals interpolated.
    assert "statements_adjusted=" in rationale
    assert "statements_reverted_for_grounding=" in rationale
    assert "bloom_edits_applied=" in rationale
    assert "BIO_101" in rationale or "course-planning" in rationale
    assert "nvidia" in rationale
    # source_refs untouched.
    assert _source_refs_snapshot(obj["chapter_objectives"]) == refs_before


def _multi_group_objectives() -> Dict[str, Any]:
    """Two TOs, two chapter groups (2 COs + 1 CO) so chunking has >1 group.

    CO-01/CO-02 (group "Week 1") roll up to TO-01; CO-03 (group "Week 2")
    to TO-02.
    """
    base = _make_objectives()
    terminals = base["terminals"]
    # Split the single group into two groups keyed by week.
    objs = base["chapter_objectives"][0]["objectives"]
    chapter_objectives = [
        {"chapter": "Week 1", "objectives": [objs[0], objs[1]]},
        {"chapter": "Week 2", "objectives": [objs[2]]},
    ]
    return {"terminals": terminals, "chapter_objectives": chapter_objectives}


def _patch_dispatch_per_chunk(monkeypatch, route_fn) -> None:
    """Patch chat_completion to route per chunk via ``route_fn(review_ids)``.

    ``route_fn`` receives the set of ids that appear under the prompt's
    'review' / review payload and returns the canned JSON string for that
    chunk, OR raises to simulate a per-chunk failure. We parse the user
    message JSON to discover which ids this chunk is reviewing.
    """

    def _fake_chat(self, messages, *, max_tokens=0, temperature=0.0, **kw):
        user = next(m for m in messages if m["role"] == "user")
        payload = json.loads(user["content"].split("INPUT:\n", 1)[1])
        review = payload.get("review") or []
        review_ids = tuple(item.get("id") for item in review)
        return route_fn(review_ids)

    import Trainforge.generators._openai_compatible_client as _oac

    monkeypatch.setattr(
        _oac.OpenAICompatibleClient, "chat_completion", _fake_chat
    )


def test_review_objectives_chunked_reviews_all_cos_and_tos(monkeypatch):
    """All chapter groups' COs AND the TOs are reviewed; counters aggregate.

    Asserts the single-shot bug (only the 12 TOs reviewed, zero COs) is gone:
    every CO id across every group lands an adjusted statement, and the TOs
    too, with reviewed == total objectives.
    """
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    obj = _multi_group_objectives()
    refs_before = _source_refs_snapshot(obj["chapter_objectives"])

    # Per-chunk adjusted maps: each CO gets a benign aligned rewrite (shares
    # its TO's anchoring tokens so the cosine guardrail accepts it); each TO
    # gets a benign rewrite.
    aligned = {
        "CO-01": {
            "statement": "Describe cellular respiration metabolic energy pathways"
        },
        "CO-02": {
            "statement": "List cellular respiration energy stages carefully"
        },
        "CO-03": {
            "statement": (
                "Explain photosynthesis light reactions experiments thoroughly"
            )
        },
        "TO-01": {
            "statement": "Analyze cellular respiration energy pathways in depth"
        },
        "TO-02": {
            "statement": (
                "Evaluate photosynthesis light reactions experiments rigorously"
            )
        },
    }

    def _route(review_ids):
        return json.dumps(
            {"adjusted": {i: aligned[i] for i in review_ids if i in aligned}}
        )

    _patch_dispatch_per_chunk(monkeypatch, _route)
    capture = _Capture()
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="BIO_101",
        capture=capture,
        embedder=_Embed(),
        client=object(),
    )
    assert result["applied"] is True
    # 3 chunks: Week 1 (2 COs), Week 2 (1 CO), terminals (2 TOs).
    assert result["chunks_attempted"] == 3
    assert result["chunks_failed"] == 0
    # reviewed counts EVERY id across all chunks: 3 COs + 2 TOs = 5.
    assert result["counters"]["reviewed"] == 5
    # Every CO + TO statement actually changed (aggregated).
    assert result["counters"]["statements_adjusted"] == 5
    # Every CO got its new statement.
    cos = [c for g in obj["chapter_objectives"] for c in g["objectives"]]
    assert all(
        c["statement"] == aligned[c["id"]]["statement"] for c in cos
    )
    # Both TOs too.
    assert all(
        t["statement"] == aligned[t["id"]]["statement"]
        for t in obj["terminals"]
    )
    # source_refs byte-identical pre/post.
    assert _source_refs_snapshot(obj["chapter_objectives"]) == refs_before
    # Exactly one aggregated decision event.
    review_events = [
        e for e in capture.events if e.get("decision_type") == _DECISION_TYPE
    ]
    assert len(review_events) == 1
    rationale = review_events[0]["rationale"]
    assert "chunks_attempted=3" in rationale
    assert "chunks_failed=0" in rationale
    assert "reviewed=5" in rationale


def test_review_objectives_one_chunk_fails_others_apply(monkeypatch):
    """A single chunk that RAISES is fail-soft; the other chunks still apply."""
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    obj = _multi_group_objectives()
    week1_before = obj["chapter_objectives"][0]["objectives"][0]["statement"]

    aligned = {
        "CO-01": {
            "statement": "Describe cellular respiration metabolic energy pathways"
        },
        "CO-02": {
            "statement": "List cellular respiration energy stages carefully"
        },
        "CO-03": {
            "statement": (
                "Explain photosynthesis light reactions experiments thoroughly"
            )
        },
        "TO-01": {
            "statement": "Analyze cellular respiration energy pathways in depth"
        },
        "TO-02": {
            "statement": (
                "Evaluate photosynthesis light reactions experiments rigorously"
            )
        },
    }

    def _route(review_ids):
        # The Week 1 chunk (contains CO-01) blows up; everything else works.
        if "CO-01" in review_ids:
            raise RuntimeError("simulated chunk timeout")
        return json.dumps(
            {"adjusted": {i: aligned[i] for i in review_ids if i in aligned}}
        )

    _patch_dispatch_per_chunk(monkeypatch, _route)
    capture = _Capture()
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="BIO_101",
        capture=capture,
        embedder=_Embed(),
        client=object(),
    )
    # Overall applied (the surviving chunks landed edits).
    assert result["applied"] is True
    assert result["chunks_attempted"] == 3
    assert result["chunks_failed"] == 1
    # Week 1's COs kept their ORIGINAL statements (the failed chunk).
    week1 = obj["chapter_objectives"][0]["objectives"]
    assert week1[0]["statement"] == week1_before  # CO-01 unchanged
    # Week 2's CO + the TOs applied.
    co3 = obj["chapter_objectives"][1]["objectives"][0]
    assert co3["statement"] == aligned["CO-03"]["statement"]
    assert obj["terminals"][0]["statement"] == aligned["TO-01"]["statement"]
    assert obj["terminals"][1]["statement"] == aligned["TO-02"]["statement"]
    # reviewed counts only the ids in the chunks that LANDED: CO-03 + 2 TOs.
    assert result["counters"]["reviewed"] == 3
    # One aggregated event reporting the failure.
    review_events = [
        e for e in capture.events if e.get("decision_type") == _DECISION_TYPE
    ]
    assert len(review_events) == 1
    assert "chunks_failed=1" in review_events[0]["rationale"]


def test_decision_type_in_schema_enum():
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "schemas"
        / "events"
        / "decision_event.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enum = schema["properties"]["decision_type"]["enum"]
    assert _DECISION_TYPE in enum


# ---------------------------------------------------------------------------
# Default-OFF: flag unset → strict no-op (byte-identical, no provider, no net).
# ---------------------------------------------------------------------------


def test_default_off_is_noop(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", raising=False)
    obj = _make_objectives()
    before = copy.deepcopy(obj)

    # If any network path were taken, this would raise (no client allowed).
    def _explode(self, *a, **k):  # pragma: no cover — must never be called
        raise AssertionError("network dispatch must not happen when off")

    import Trainforge.generators._openai_compatible_client as _oac

    monkeypatch.setattr(_oac.OpenAICompatibleClient, "chat_completion", _explode)

    capture = _Capture()
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="BIO_101",
        capture=capture,
    )
    assert result == {"enabled": False, "applied": False, "counters": {}}
    # Objectives byte-identical.
    assert obj == before
    # No decision captured.
    assert capture.events == []


def test_resolve_provider_unknown_is_off(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "openai")
    assert resolve_objective_review_provider() is None
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "")
    assert resolve_objective_review_provider() is None
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    assert resolve_objective_review_provider() == "nvidia"


def test_unparseable_response_keeps_originals(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    obj = _make_objectives()
    before = copy.deepcopy(obj)
    _patch_dispatch(monkeypatch, "this is not JSON at all <<<")
    capture = _Capture()
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="BIO_101",
        capture=capture,
        embedder=_Embed(),
        client=object(),
    )
    assert result["enabled"] is True
    assert result["applied"] is False
    assert obj == before  # unchanged
    # A no-op decision event still fired (audit trail).
    review_events = [
        e for e in capture.events if e.get("decision_type") == _DECISION_TYPE
    ]
    assert len(review_events) == 1


# ---------------------------------------------------------------------------
# 2026-06-21 — review must sub-chunk a large pre-week-slicing 'all' group so it
# does not make one huge call that blows the read timeout, and must surface
# chunks_failed (the fail-closed signal the _plan_course_structure caller reads
# to refuse downstream when the review is enabled but incomplete).
# ---------------------------------------------------------------------------


def _single_all_group_objectives(n_cos: int) -> Dict[str, Any]:
    """``n_cos`` COs in ONE ``'all'`` group (the pre-week-slicing shape the
    caller actually passes) + one TO they roll up to."""
    terminals = [
        {
            "id": "TO-01",
            "statement": "Apply arithmetic operations to algebraic expressions",
            "bloom_level": "apply",
            "bloom_verb": "apply",
            "abcd": {
                "audience": "students",
                "behavior": {"verb": "apply", "action_object": "operations"},
                "condition": "given expressions",
                "degree": "correctly",
            },
            "source_refs": [],
        }
    ]
    cos = [
        {
            "id": f"CO-{i:02d}",
            "statement": f"Apply arithmetic operation number {i} to expressions",
            "bloom_level": "apply",
            "bloom_verb": "apply",
            "abcd": {
                "audience": "students",
                "behavior": {"verb": "apply", "action_object": f"operation {i}"},
                "condition": "given expressions",
                "degree": "correctly",
            },
            "source_refs": [{"ref": "ch1", "chunk_ids": [f"chunk_{i:03d}"]}],
            "terminal_id": "TO-01",
        }
        for i in range(1, n_cos + 1)
    ]
    return {
        "terminals": terminals,
        "chapter_objectives": [{"chapter": "all", "objectives": cos}],
    }


def test_review_subchunks_large_all_group(monkeypatch):
    """A 25-CO 'all' group is split into ceil(25/12)=3 CO chunks + 1 TO chunk;
    pre-fix it was ONE 25-CO call (the 97-CO read-timeout root cause)."""
    from lib.objectives.objective_review import _MAX_COS_PER_REVIEW_CHUNK

    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    obj = _single_all_group_objectives(25)

    seen_sizes = []

    def _route(review_ids):
        # Echo each reviewed id with a benign no-op edit so the chunk parses as
        # a successful review (an empty adjusted-map is treated as a failure).
        seen_sizes.append(len(review_ids))
        return json.dumps(
            {"adjusted": {i: {"bloom_level": "apply"} for i in review_ids}}
        )

    _patch_dispatch_per_chunk(monkeypatch, _route)
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="ALG_101",
        capture=_Capture(),
        embedder=_Embed(),
        client=object(),
    )
    import math
    expected_co_chunks = math.ceil(25 / _MAX_COS_PER_REVIEW_CHUNK)
    # Sub-chunking proof: 3 CO sub-chunks (was ONE 25-CO call pre-fix) + 1 TO.
    assert result["chunks_attempted"] == expected_co_chunks + 1
    # No CO sub-chunk exceeds the cap.
    assert seen_sizes and max(seen_sizes) <= _MAX_COS_PER_REVIEW_CHUNK


def test_review_surfaces_chunks_failed_for_fail_closed(monkeypatch):
    """A sub-chunk that fails (after retries) surfaces chunks_failed>0 — the
    signal the _plan_course_structure caller reads to fail closed instead of
    shipping a course on partially-reviewed objectives."""
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    obj = _single_all_group_objectives(20)

    def _route(review_ids):
        # Fail the chunk that contains CO-01 (simulating a read timeout);
        # other chunks return clean.
        if "CO-01" in review_ids:
            raise RuntimeError("simulated read timeout")
        return json.dumps({"adjusted": {}})

    _patch_dispatch_per_chunk(monkeypatch, _route)
    result = review_objectives(
        terminals=obj["terminals"],
        chapter_objectives=obj["chapter_objectives"],
        course_name="ALG_101",
        capture=_Capture(),
        embedder=_Embed(),
        client=object(),
    )
    assert result["enabled"] is True
    assert result["chunks_failed"] >= 1
