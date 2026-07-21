"""Tests for the concept / sub-objective layer derivation (finding 17).

Covers (per the task gate):
  (a) deterministic clustering populates sub_objectives on a CO with grounded
      chunks (0/N -> >=1), splitting topically-distinct chunks into separate
      sub-objectives;
  (b) ANTI-FABRICATION: every derived source_chunk_ids is a strict subset of
      the CO's cited chunk set — no chunk id is ever invented, in BOTH the
      deterministic and the LLM arm;
  (c) graceful-degrade: no embedder -> a single un-clustered (but grounded)
      sub-objective; no grounded chunks -> [] (CO keeps empty sub_objectives);
  (d) the optional LLM arm (behind ED4ALL_SUB_OBJECTIVE_PROVIDER) parses +
      filters the model output, and an errored / unparseable / empty LLM result
      degrades to the deterministic path;
  (e) the sub_objective_derivation DecisionCapture fires with a dynamic
      rationale and the decision_type is in the schema enum;
  (f) populate_sub_objectives mutates COs in place and isolates per-CO failure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from lib.objectives.sub_objectives import (
    _DECISION_TYPE,
    _dedupe_within_co,
    derive_sub_objectives_for_co,
    populate_sub_objectives,
    resolve_max_sub_objectives,
    resolve_sub_objective_provider,
    resolve_sub_objective_quality,
    ENV_SUB_OBJECTIVE_MAX,
    ENV_SUB_OBJECTIVE_PROVIDER,
    ENV_SUB_OBJECTIVE_QUALITY,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Embed:
    """Deterministic single-text encoder over a token-hash basis.

    Cosine ~= token overlap, so chunks that share words cluster and disjoint
    chunks don't — letting us author chunk text whose clustering we control.
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


def _patch_dispatch(monkeypatch, response: str, counter: Dict[str, int]):
    """Patch OpenAICompatibleClient.chat_completion to return canned JSON.

    Mirrors test_objective_review._patch_dispatch — the module constructs a
    REAL OpenAICompatibleClient via the endpoint registry, so we stub its chat
    dispatch rather than injecting an httpx transport.
    """

    def _fake_chat(self, messages, *, max_tokens=0, temperature=0.0, **kw):
        counter["calls"] = counter.get("calls", 0) + 1
        return response

    import Trainforge.generators._openai_compatible_client as _oac

    monkeypatch.setattr(
        _oac.OpenAICompatibleClient, "chat_completion", _fake_chat
    )


def _patch_dispatch_raises(monkeypatch):
    """Patch chat_completion to raise — exercises the degrade path."""

    def _boom(self, messages, *, max_tokens=0, temperature=0.0, **kw):
        raise RuntimeError("provider down")

    import Trainforge.generators._openai_compatible_client as _oac

    monkeypatch.setattr(
        _oac.OpenAICompatibleClient, "chat_completion", _boom
    )


# ---------------------------------------------------------------------------
# Fixtures: a CO grounded on two topically-distinct chunk groups.
# ---------------------------------------------------------------------------


def _co_two_topics() -> Dict[str, Any]:
    return {
        "id": "CO-01",
        "statement": "Factor whole numbers and apply divisibility rules.",
        "source_chunk_ids": ["chk_a1", "chk_a2", "chk_b1", "chk_b2"],
        "sub_objectives": [],
    }


def _chunks_two_topics() -> Dict[str, Any]:
    # Group A: prime factorization vocabulary. Group B: divisibility vocabulary.
    return {
        "chk_a1": {
            "text": "Prime factorization expresses a number as a product of "
            "prime factors using a factor tree."
        },
        "chk_a2": {
            "text": "A factor tree breaks a composite number into its prime "
            "factorization product step by step."
        },
        "chk_b1": {
            "text": "Divisibility rules determine whether a number is "
            "divisible by 2, 3, 5, or 10 without dividing."
        },
        "chk_b2": {
            "text": "A divisibility rule for 3 checks whether the digit sum "
            "is divisible by 3."
        },
    }


# ---------------------------------------------------------------------------
# (a) deterministic clustering populates + splits distinct topics
# ---------------------------------------------------------------------------


def test_deterministic_populates_and_clusters_distinct_topics():
    co = _co_two_topics()
    chunks = _chunks_two_topics()
    cap = _Capture()

    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed(), capture=cap
    )

    assert subs, "expected >=1 sub-objective (CO was 0/N)"
    # The two disjoint topic groups should not collapse into one cluster.
    assert len(subs) >= 2, f"expected the two topics to split, got {subs}"
    # Each sub-objective carries a non-empty statement + grounded ids.
    for so in subs:
        assert so["statement"].strip()
        assert so["source_chunk_ids"]
    # The union of sub-objective ids covers the CO's grounded set (no loss).
    covered = {cid for so in subs for cid in so["source_chunk_ids"]}
    assert covered == set(co["source_chunk_ids"])
    # Capture fired with the right type + a dynamic rationale.
    assert len(cap.events) == 1
    ev = cap.events[0]
    assert ev["decision_type"] == _DECISION_TYPE
    assert "CO-01" in ev["rationale"]
    assert "anti-fabrication" in ev["rationale"].lower()


# ---------------------------------------------------------------------------
# (b) anti-fabrication: ids are always a subset of the CO's cited set
# ---------------------------------------------------------------------------


def test_deterministic_anti_fabrication_subset():
    co = _co_two_topics()
    chunks = _chunks_two_topics()
    allowed = set(co["source_chunk_ids"])

    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed()
    )
    for so in subs:
        assert set(so["source_chunk_ids"]).issubset(allowed)


def test_reads_chunk_ids_from_source_refs_shape():
    # CO carries grounding ONLY under source_refs[].chunk_ids (no flat field).
    co = {
        "id": "CO-09",
        "statement": "Simplify fractions.",
        "source_refs": [
            {"ref": "ch3", "chunk_ids": ["chk_x", "chk_y"]},
        ],
        "sub_objectives": [],
    }
    chunks = {
        "chk_x": {"text": "Simplify a fraction by dividing numerator and "
                          "denominator by the greatest common factor."},
        "chk_y": {"text": "An equivalent fraction has the same value after "
                          "reducing by a common factor."},
    }
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed()
    )
    assert subs
    covered = {cid for so in subs for cid in so["source_chunk_ids"]}
    assert covered.issubset({"chk_x", "chk_y"})


# ---------------------------------------------------------------------------
# (c) graceful degrade
# ---------------------------------------------------------------------------


def test_no_embedder_single_grounded_sub_objective():
    co = _co_two_topics()
    chunks = _chunks_two_topics()

    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=None  # no embedder injected
    )
    # Degrades to a single un-clustered sub-objective over the whole set —
    # still 3-level, never blind-clustered, never fabricated.
    assert len(subs) == 1
    assert set(subs[0]["source_chunk_ids"]) == set(co["source_chunk_ids"])


def test_no_grounded_chunks_returns_empty():
    co = {"id": "CO-02", "statement": "Ungrounded objective.",
          "source_chunk_ids": [], "sub_objectives": []}
    cap = _Capture()
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id={}, embedder=_Embed(), capture=cap
    )
    assert subs == []
    assert cap.events[0]["decision"].endswith("none")


def test_missing_chunk_body_still_grounded_by_id():
    # chunk dict absent from chunks_by_id -> empty body, but id still grounded.
    co = {
        "id": "CO-03",
        "statement": "Objective with one missing-body chunk.",
        "source_chunk_ids": ["chk_present", "chk_absent"],
        "sub_objectives": [],
    }
    chunks = {"chk_present": {"text": "Some grounded content about exponents."}}
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed()
    )
    assert subs
    covered = {cid for so in subs for cid in so["source_chunk_ids"]}
    # The absent-body chunk id is NOT dropped (anti-loss).
    assert "chk_absent" in covered


# ---------------------------------------------------------------------------
# (d) optional LLM arm
# ---------------------------------------------------------------------------


def test_llm_arm_parses_and_filters(monkeypatch):
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_PROVIDER, "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    co = _co_two_topics()
    chunks = _chunks_two_topics()
    # Model returns one valid sub-objective + one with a FABRICATED id (dropped)
    # + one entirely fabricated (dropped → leaves 1 grounded).
    response = json.dumps(
        {
            "sub_objectives": [
                {"statement": "Prime factorization",
                 "source_chunk_ids": ["chk_a1", "chk_a2"]},
                {"statement": "Divisibility (partly fabricated id)",
                 "source_chunk_ids": ["chk_b1", "chk_NOPE"]},
                {"statement": "All fabricated",
                 "source_chunk_ids": ["chk_ghost"]},
            ]
        }
    )
    counter: Dict[str, int] = {}
    _patch_dispatch(monkeypatch, response, counter)
    cap = _Capture()

    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed(), capture=cap,
    )
    assert counter.get("calls") == 1
    # The fully-fabricated sub-objective is dropped; the partly-fabricated one
    # keeps only its valid id.
    statements = [so["statement"] for so in subs]
    assert "Prime factorization" in statements
    assert "All fabricated" not in statements
    allowed = set(co["source_chunk_ids"])
    for so in subs:
        assert set(so["source_chunk_ids"]).issubset(allowed)
        assert "chk_NOPE" not in so["source_chunk_ids"]
    assert cap.events[0]["decision_type"] == _DECISION_TYPE
    assert ":llm" in cap.events[0]["decision"]


def test_llm_arm_error_degrades_to_deterministic(monkeypatch):
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_PROVIDER, "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    co = _co_two_topics()
    chunks = _chunks_two_topics()
    cap = _Capture()
    _patch_dispatch_raises(monkeypatch)

    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed(), capture=cap,
    )
    # Still gets grounded sub-objectives via the deterministic fallback.
    assert subs
    covered = {cid for so in subs for cid in so["source_chunk_ids"]}
    assert covered.issubset(set(co["source_chunk_ids"]))
    assert ":deterministic" in cap.events[0]["decision"]


def test_llm_arm_empty_response_degrades(monkeypatch):
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_PROVIDER, "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    co = _co_two_topics()
    chunks = _chunks_two_topics()
    _patch_dispatch(monkeypatch, "not json at all", {})
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed(),
    )
    assert subs  # deterministic fallback still produced grounded output


def test_unknown_provider_disables_llm_arm(monkeypatch):
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_PROVIDER, "totally-bogus")
    assert resolve_sub_objective_provider() is None


# ---------------------------------------------------------------------------
# (e) cap + (f) populate_sub_objectives in place
# ---------------------------------------------------------------------------


def test_max_cap_env(monkeypatch):
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_MAX, "2")
    assert resolve_max_sub_objectives() == 2
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_MAX, "garbage")
    assert resolve_max_sub_objectives() == 5  # parse-with-fallback
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_MAX, "0")
    assert resolve_max_sub_objectives() == 5


def test_cap_never_drops_chunk_ids():
    # 3 distinct topic groups but cap=2 -> overflow merges, no chunk lost.
    co = {
        "id": "CO-07",
        "statement": "Three distinct topics.",
        "source_chunk_ids": ["t1", "t2", "t3"],
        "sub_objectives": [],
    }
    chunks = {
        "t1": {"text": "Exponents and powers of numbers raised to integers."},
        "t2": {"text": "Square roots and radical simplification of surds."},
        "t3": {"text": "Order of operations with parentheses and brackets."},
    }
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed(), max_sub_objectives=2
    )
    assert len(subs) <= 2
    covered = {cid for so in subs for cid in so["source_chunk_ids"]}
    assert covered == {"t1", "t2", "t3"}


def test_populate_in_place_group_shape():
    groups = [
        {"chapter": "ch1", "objectives": [_co_two_topics()]},
        {
            "chapter": "ch2",
            "objectives": [
                {"id": "CO-50", "statement": "No grounding here.",
                 "source_chunk_ids": [], "sub_objectives": []},
            ],
        },
    ]
    counters = populate_sub_objectives(
        chapter_objectives=groups,
        chunks_by_id=_chunks_two_topics(),
        embedder=_Embed(),
    )
    assert counters["cos_seen"] == 2
    assert counters["cos_populated"] == 1  # only the grounded CO
    assert counters["sub_objectives_total"] >= 2
    # In-place mutation: the grounded CO now carries sub_objectives.
    assert groups[0]["objectives"][0]["sub_objectives"]
    # The ungrounded CO keeps its empty list (byte-stable status quo).
    assert groups[1]["objectives"][0]["sub_objectives"] == []


def test_populate_flat_list_shape():
    flat = [_co_two_topics()]
    counters = populate_sub_objectives(
        chapter_objectives=flat,
        chunks_by_id=_chunks_two_topics(),
        embedder=_Embed(),
    )
    assert counters["cos_populated"] == 1
    assert flat[0]["sub_objectives"]


# ---------------------------------------------------------------------------
# decision_type is in the schema enum
# ---------------------------------------------------------------------------


def test_decision_type_in_schema_enum():
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "schemas" / "events" / "decision_event.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enum = schema["properties"]["decision_type"]["enum"]
    assert _DECISION_TYPE in enum


# ---------------------------------------------------------------------------
# leak rejection: raw scanned-algebra exercise / callout / math fragments must NOT
# become sub-objective statements, and the "Understand:" template prefix must
# never leak (regression for a real calibration course_planning review).
# ---------------------------------------------------------------------------


def test_raw_exercise_and_callout_text_never_leaks_into_statements():
    co = {
        "id": "CO-LEAK",
        "statement": "Apply place value concepts to round whole numbers.",
        # Every grounded chunk body is raw scanned-algebra chrome (callout labels,
        # exercise glyphs, bare math) — exactly the text that leaked verbatim
        # (incl. the hardcoded "Understand:" prefix) in the prior derivation.
        "source_chunk_ids": ["c1", "c2", "c3", "c4"],
        "sub_objectives": [],
    }
    chunks = {
        "c1": {"text": "BE PREPARED : : 1.1 A more thorough introduction to the topics."},
        "c2": {"text": "TRY IT : : 1.6 Write the number eleven billion in words."},
        "c3": {"text": "3+5=8 The sum of three and five is equal to eight."},
        "c4": {"text": "ⓐ After completing the exercises, use this checklist."},
    }

    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed()
    )

    assert subs, "still 3-level (grounded ids preserved), just no garbage text"
    for so in subs:
        stmt = so["statement"]
        assert stmt.strip()
        # No leaked template prefix and no raw-chrome markers.
        assert not stmt.startswith("Understand:")
        for marker in ("BE PREPARED", "TRY IT", "::", "ⓐ", "3+5=8"):
            assert marker not in stmt, f"leaked {marker!r}: {stmt!r}"
        # With no clean concept sentence available, each sub-objective falls
        # back to the (clean) CO statement — never source noise.
        assert stmt == co["statement"]
    # Anti-fabrication still holds.
    covered = {cid for so in subs for cid in so["source_chunk_ids"]}
    assert covered.issubset(set(co["source_chunk_ids"]))


def test_clean_concept_sentence_is_preferred_over_leading_junk():
    co = {
        "id": "CO-MIX",
        "statement": "FALLBACK should not be used here.",
        "source_chunk_ids": ["m1"],
        "sub_objectives": [],
    }
    chunks = {
        "m1": {
            "text": "TRY IT :: 1.1 do this exercise. The opposite of a number "
            "is the same distance from zero on the number line."
        }
    }
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed()
    )
    assert len(subs) == 1
    stmt = subs[0]["statement"]
    assert stmt.startswith("The opposite of a number")
    assert "TRY IT" not in stmt and stmt != co["statement"]


# ---------------------------------------------------------------------------
# Defect D — deterministic statement quality (ED4ALL_SUB_OBJECTIVE_QUALITY).
# ---------------------------------------------------------------------------


def _one_chunk_co(cid: str, body: str, statement: str) -> Any:
    co = {
        "id": "CO-Q",
        "statement": statement,
        "source_chunk_ids": [cid],
        "sub_objectives": [],
    }
    return co, {cid: {"text": body}}


def test_quality_resolver_parse_with_fallback(monkeypatch):
    monkeypatch.delenv(ENV_SUB_OBJECTIVE_QUALITY, raising=False)
    assert resolve_sub_objective_quality() is False
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_QUALITY, "1")
    assert resolve_sub_objective_quality() is True
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_QUALITY, "off")
    assert resolve_sub_objective_quality() is False
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_QUALITY, "garbage")
    assert resolve_sub_objective_quality() is False
    # Explicit arg wins over env.
    assert resolve_sub_objective_quality(True) is True


def test_quality_prefix_strip_recovers_tail():
    co, chunks = _one_chunk_co(
        "p1",
        "By the end of this section, you will be able to: "
        "Round whole numbers to a given place value.",
        "CO fallback statement.",
    )
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=None, quality=True
    )
    assert len(subs) == 1
    assert subs[0]["statement"] == "Round whole numbers to a given place value"


def test_quality_run_on_skipped_not_truncated():
    run_on = (
        "This extended passage rambles across many clauses and ideas without "
        "ever reaching a terminal sentence boundary while listing numerous "
        "related concepts and details"
    )
    assert len(run_on.split()) > 18 and not run_on.endswith(".")
    co_on, chunks = _one_chunk_co("r1", run_on, "Fallback CO statement here.")
    subs_on = derive_sub_objectives_for_co(
        co=co_on, chunks_by_id=chunks, embedder=None, quality=True
    )
    # Run-on has no in-budget boundary → SKIPPED → CO-statement fallback,
    # NEVER a mid-clause truncation.
    assert subs_on[0]["statement"] == co_on["statement"]

    # Flag OFF still head-truncates to 14 words (legacy behavior) — proves the
    # quality path diverges and the OFF path is unchanged.
    co_off, chunks2 = _one_chunk_co("r1", run_on, "Fallback.")
    subs_off = derive_sub_objectives_for_co(
        co=co_off, chunks_by_id=chunks2, embedder=None, quality=False
    )
    assert len(subs_off[0]["statement"].split()) == 14
    assert subs_off[0]["statement"] != co_off["statement"]


def test_quality_glossary_extraction():
    co, chunks = _one_chunk_co(
        "g1",
        "Prime number : a whole number greater than one with exactly two factors.",
        "Fallback.",
    )
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=None, quality=True
    )
    assert subs[0]["statement"] == "Define Prime number"


def test_quality_boilerplate_only_rejected():
    co, chunks = _one_chunk_co(
        "b1", "In the following exercises, solve each equation.", "Clean CO fallback."
    )
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=None, quality=True
    )
    # The whole body is apparatus boilerplate → falls back to the CO statement.
    assert subs[0]["statement"] == co["statement"]


def test_quality_within_co_dedupe_unit():
    subs = [
        {"statement": "Define slope", "source_chunk_ids": ["a", "b"]},
        {"statement": "define  SLOPE ", "source_chunk_ids": ["b", "c"]},
        {"statement": "Compute area", "source_chunk_ids": ["d"]},
    ]
    out = _dedupe_within_co(subs)
    assert len(out) == 2
    assert out[0]["statement"] == "Define slope"
    assert out[0]["source_chunk_ids"] == ["a", "b", "c"]  # merged, no dup, ordered
    assert out[1]["source_chunk_ids"] == ["d"]


def test_quality_dedupe_identical_fallbacks_in_co():
    # Two vocab-disjoint boilerplate-only chunks → 2 clusters, both fall back to
    # the CO statement → deduped to one; no chunk id lost.
    co = {
        "id": "CO-D",
        "statement": "Shared fallback statement.",
        "source_chunk_ids": ["d1", "d2"],
        "sub_objectives": [],
    }
    chunks = {
        "d1": {"text": "In the following exercises solve problems."},
        "d2": {"text": "Be prepared to begin the readiness quiz now."},
    }
    subs = derive_sub_objectives_for_co(
        co=co, chunks_by_id=chunks, embedder=_Embed(), quality=True
    )
    assert len(subs) == 1
    assert subs[0]["statement"] == co["statement"]
    covered = {cid for so in subs for cid in so["source_chunk_ids"]}
    assert covered == {"d1", "d2"}


#: Frozen snapshot of the flag-OFF derivation for the two-topic fixture (the
#: legacy 14-word head-truncation). Pins the byte-stable status quo so a future
#: edit to the quality path can't silently perturb the OFF behavior.
_OFF_SNAPSHOT = [
    {"statement": "Prime factorization expresses a number as a product of prime factors using a factor",
     "source_chunk_ids": ["chk_a1"]},
    {"statement": "A factor tree breaks a composite number into its prime factorization product step by",
     "source_chunk_ids": ["chk_a2"]},
    {"statement": "Divisibility rules determine whether a number is divisible by 2, 3, 5, or 10",
     "source_chunk_ids": ["chk_b1"]},
    {"statement": "A divisibility rule for 3 checks whether the digit sum is divisible by 3",
     "source_chunk_ids": ["chk_b2"]},
]


def test_quality_flag_off_byte_identical_snapshot(monkeypatch):
    monkeypatch.delenv(ENV_SUB_OBJECTIVE_QUALITY, raising=False)
    chunks = _chunks_two_topics()
    # (1) env-unset default == (2) explicit quality=False == (3) frozen snapshot.
    unset = derive_sub_objectives_for_co(
        co=_co_two_topics(), chunks_by_id=chunks, embedder=_Embed()
    )
    explicit_off = derive_sub_objectives_for_co(
        co=_co_two_topics(), chunks_by_id=chunks, embedder=_Embed(), quality=False
    )
    assert unset == explicit_off == _OFF_SNAPSHOT
    # Flag ON diverges (keeps full sentences, no head-truncation) but preserves
    # the same clustering / id partition.
    on = derive_sub_objectives_for_co(
        co=_co_two_topics(), chunks_by_id=chunks, embedder=_Embed(), quality=True
    )
    assert on != _OFF_SNAPSHOT
    assert [so["source_chunk_ids"] for so in on] == [
        so["source_chunk_ids"] for so in _OFF_SNAPSHOT
    ]


def test_populate_fingerprint_includes_quality(monkeypatch):
    import lib.objectives.sub_objectives as som

    captured: List[Dict[str, Any]] = []
    real = som.compute_fingerprint

    def _spy(payload):
        captured.append(payload)
        return real(payload)

    monkeypatch.setattr(som, "compute_fingerprint", _spy)
    monkeypatch.setenv(ENV_SUB_OBJECTIVE_QUALITY, "1")
    populate_sub_objectives(
        chapter_objectives=[_co_two_topics()],
        chunks_by_id=_chunks_two_topics(),
        embedder=_Embed(),
    )
    assert captured, "populate must compute a per-CO fingerprint"
    assert all("quality" in payload for payload in captured)
    assert captured[0]["quality"] is True

    # Same fingerprint payload with the flag OFF hashes differently → a flag
    # flip invalidates cached per-CO derivations.
    off = real({**captured[0], "quality": False})
    on = real(captured[0])
    assert off != on
