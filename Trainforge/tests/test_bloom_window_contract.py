"""Bloom-ladder initiative (WI-11) — ``TRAINFORGE_BLOOM_WINDOWS`` contract.

Covers the three additive surfaces gated behind the flag:

* ``_structured_misconception_pairs`` / ``resolve_chunk_misconceptions``
  recover a misconception CARD's own ``data-cf-bloom-level`` instead of
  collapsing every card in a chunk onto one chunk-wide value.
* ``build_evidence_window`` accepts an optional ``target_rung`` ceiling and
  partitions/annotates candidate blocks by their own ``data-cf-bloom-level``.
* ``render_evidence_window`` renders a fourth ``|bloom`` tag segment only
  when a block actually carries one.

Every surface must be byte-identical to the pre-ladder contract when the
flag is unset (default OFF) — that is the hard constraint under test
throughout, not just in the dedicated byte-identical tests at the bottom.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from Trainforge.generators.synthesis_window_contract import (
    BLOOM_WINDOWS_ENV,
    _structured_misconception_pairs,
    build_evidence_window,
    render_evidence_window,
    resolve_bloom_windows_enabled,
    resolve_chunk_misconceptions,
)

_OBJECTIVE_ID = "CO-01"

_OBJECTIVES = {
    "co-01": {
        "id": "CO-01",
        "statement": "Distinguish factors from multiples of a whole number.",
        "bloom_level": "understand",
    },
}


# ---------------------------------------------------------------------------
# resolve_bloom_windows_enabled — parse-with-fallback, default OFF
# ---------------------------------------------------------------------------


def test_unset_env_resolves_off():
    assert resolve_bloom_windows_enabled(env={}) is False


def test_truthy_tokens_resolve_on():
    for token in ("1", "true", "TRUE", "Yes", "on", "ON"):
        assert resolve_bloom_windows_enabled(env={BLOOM_WINDOWS_ENV: token}) is True, token


def test_falsey_and_garbage_tokens_resolve_off():
    for token in ("0", "false", "no", "off", "", "garbage", "  "):
        assert resolve_bloom_windows_enabled(env={BLOOM_WINDOWS_ENV: token}) is False, token


def test_env_var_default_reads_os_environ(monkeypatch):
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    assert resolve_bloom_windows_enabled() is False
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    assert resolve_bloom_windows_enabled() is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _card_html(
    block_id: str,
    claim: str,
    correction: str,
    *,
    bloom_level: str | None = None,
) -> str:
    bloom_attr = f' data-cf-bloom-level="{bloom_level}"' if bloom_level else ""
    return (
        f'<div class="misconception-card" data-cf-block-id="{block_id}" '
        f'data-cf-objective-id="{_OBJECTIVE_ID}"{bloom_attr}>'
        "<h2>Common Misconception</h2>"
        f'<p class="misconception-claim">{claim}</p>'
        f'<p class="misconception-correction">{correction}</p>'
        "</div>"
    )


def _card_tag(html: str, block_id: str):
    soup = BeautifulSoup(html, "html.parser")
    return soup.find(attrs={"data-cf-block-id": block_id})


_CLAIM_A = "Students often confuse a factor of a whole number with a multiple."
_CORRECTION_A = (
    "A factor divides the whole number evenly, while a multiple is the "
    "product of that whole number and a counting number."
)
_CLAIM_B = "Learners sometimes assume every multiple is also a factor."
_CORRECTION_B = (
    "Every whole number divides its own multiples, so the multiple is the "
    "larger product and the factor is the smaller divisor."
)


def _chunk(html_body: str, *, chunk_id: str = "chunk-0001", bloom_level: str = "understand") -> dict:
    return {
        "id": chunk_id,
        "html": html_body,
        "bloom_level": bloom_level,
    }


# ---------------------------------------------------------------------------
# (a) _structured_misconception_pairs — per-CARD bloom recovery
# ---------------------------------------------------------------------------


def test_structured_pairs_flag_off_never_stamps_bloom_level(monkeypatch):
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    html = _card_html("blk-1", _CLAIM_A, _CORRECTION_A, bloom_level="apply")
    tag = _card_tag(html, "blk-1")
    pairs = _structured_misconception_pairs(tag)
    assert pairs == [{"claim": _CLAIM_A, "correction": _CORRECTION_A}]
    assert "bloom_level" not in pairs[0]


def test_structured_pairs_flag_on_recovers_card_bloom_level(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    html = _card_html("blk-1", _CLAIM_A, _CORRECTION_A, bloom_level="apply")
    tag = _card_tag(html, "blk-1")
    pairs = _structured_misconception_pairs(tag)
    assert pairs[0]["bloom_level"] == "apply"


def test_structured_pairs_flag_on_lowercases_and_strips_attr(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    html = _card_html("blk-1", _CLAIM_A, _CORRECTION_A, bloom_level=" Apply ")
    tag = _card_tag(html, "blk-1")
    pairs = _structured_misconception_pairs(tag)
    assert pairs[0]["bloom_level"] == "apply"


def test_structured_pairs_flag_on_no_attr_stamps_nothing(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    html = _card_html("blk-1", _CLAIM_A, _CORRECTION_A)
    tag = _card_tag(html, "blk-1")
    pairs = _structured_misconception_pairs(tag)
    assert "bloom_level" not in pairs[0]


# ---------------------------------------------------------------------------
# (a) resolve_chunk_misconceptions — multi-rung collapse fix
# ---------------------------------------------------------------------------


def _two_rung_chunk(bloom_a: str | None, bloom_b: str | None) -> dict:
    body = _card_html("blk-1", _CLAIM_A, _CORRECTION_A, bloom_level=bloom_a)
    body += _card_html("blk-2", _CLAIM_B, _CORRECTION_B, bloom_level=bloom_b)
    return _chunk(body, bloom_level="understand")


def test_resolve_chunk_misconceptions_flag_off_collapses_to_chunk_level(monkeypatch):
    """Pre-ladder (legacy) behavior: every card gets the SAME chunk-wide value."""
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    recovered = resolve_chunk_misconceptions(_two_rung_chunk("apply", "create"))
    assert len(recovered) == 2
    assert {entry["bloom_level"] for entry in recovered} == {"understand"}


def test_resolve_chunk_misconceptions_flag_on_preserves_per_card_rung(monkeypatch):
    """The fix: two cards at distinct rungs keep distinct bloom_level values."""
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    recovered = resolve_chunk_misconceptions(_two_rung_chunk("apply", "create"))
    assert len(recovered) == 2
    by_claim = {entry["misconception"]: entry["bloom_level"] for entry in recovered}
    assert by_claim[_CLAIM_A] == "apply"
    assert by_claim[_CLAIM_B] == "create"


def test_resolve_chunk_misconceptions_flag_on_falls_back_to_chunk_level(monkeypatch):
    """A card with no rung of its own still gets the chunk-wide fallback."""
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    recovered = resolve_chunk_misconceptions(_two_rung_chunk(None, "create"))
    by_claim = {entry["misconception"]: entry["bloom_level"] for entry in recovered}
    assert by_claim[_CLAIM_A] == "understand"  # chunk-level fallback
    assert by_claim[_CLAIM_B] == "create"       # card's own rung wins


def test_resolve_chunk_misconceptions_flag_on_no_chunk_level_either(monkeypatch):
    """Neither card nor chunk carries a rung -> no bloom_level key at all."""
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    chunk = _chunk(
        _card_html("blk-1", _CLAIM_A, _CORRECTION_A), bloom_level="",
    )
    recovered = resolve_chunk_misconceptions(chunk)
    assert len(recovered) == 1
    assert "bloom_level" not in recovered[0]


# ---------------------------------------------------------------------------
# (b) build_evidence_window — target_rung partition/annotation
# ---------------------------------------------------------------------------


def _evidence_block(block_id: str, text: str, bloom_level: str | None) -> str:
    bloom_attr = f' data-cf-bloom-level="{bloom_level}"' if bloom_level else ""
    return (
        f'<div data-cf-block-id="{block_id}" data-cf-objective-id="{_OBJECTIVE_ID}"'
        f"{bloom_attr}><p>{text}</p></div>"
    )


_REMEMBER_TEXT = (
    "A factor of a whole number divides that number evenly with no "
    "remainder, distinguishing factors from multiples of a whole number."
)
_CREATE_TEXT = (
    "Design a proof distinguishing factors from multiples of a whole "
    "number for an arbitrary base, distinguishing factors from multiples "
    "of a whole number in every case."
)
_UNTAGGED_TEXT = (
    "Distinguishing factors from multiples of a whole number is a core "
    "arithmetic skill covered across this chunk's distinguishing factors "
    "from multiples of a whole number examples."
)


def _rung_chunk() -> dict:
    body = (
        _evidence_block("blk-remember", _REMEMBER_TEXT, "remember")
        + _evidence_block("blk-create", _CREATE_TEXT, "create")
        + _evidence_block("blk-untagged", _UNTAGGED_TEXT, None)
    )
    return {"id": "chunk-0002", "html": body}


def test_build_evidence_window_flag_off_ignores_target_rung(monkeypatch):
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    window = build_evidence_window(_rung_chunk(), _OBJECTIVES["co-01"], target_rung="remember")
    block_ids = {b["block_id"] for b in window["blocks"]}
    # Nothing excluded, nothing annotated -- old shape exactly.
    assert block_ids == {"blk-remember", "blk-create", "blk-untagged"}
    for block in window["blocks"]:
        assert set(block) == {"block_id", "role", "polarity", "text"}


def test_build_evidence_window_flag_on_no_target_rung_annotates_only(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    window = build_evidence_window(_rung_chunk(), _OBJECTIVES["co-01"])
    by_id = {b["block_id"]: b for b in window["blocks"]}
    assert set(by_id) == {"blk-remember", "blk-create", "blk-untagged"}
    assert by_id["blk-remember"]["bloom_level"] == "remember"
    assert by_id["blk-create"]["bloom_level"] == "create"
    assert "bloom_level" not in by_id["blk-untagged"]


def test_build_evidence_window_flag_on_target_rung_excludes_later_rung(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    window = build_evidence_window(
        _rung_chunk(), _OBJECTIVES["co-01"], target_rung="remember",
    )
    block_ids = {b["block_id"] for b in window["blocks"]}
    # "create" ranks above "remember" -> excluded. Untagged block is always
    # kept (no assumption about content authored before any rung existed).
    assert block_ids == {"blk-remember", "blk-untagged"}


def test_build_evidence_window_target_rung_is_case_insensitive(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    window = build_evidence_window(
        _rung_chunk(), _OBJECTIVES["co-01"], target_rung="REMEMBER",
    )
    block_ids = {b["block_id"] for b in window["blocks"]}
    assert block_ids == {"blk-remember", "blk-untagged"}


def test_build_evidence_window_garbage_target_rung_excludes_nothing(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    window = build_evidence_window(
        _rung_chunk(), _OBJECTIVES["co-01"], target_rung="not-a-bloom-level",
    )
    block_ids = {b["block_id"] for b in window["blocks"]}
    assert block_ids == {"blk-remember", "blk-create", "blk-untagged"}
    # Still annotated -- only the ceiling comparison itself is skipped.
    by_id = {b["block_id"]: b for b in window["blocks"]}
    assert by_id["blk-remember"]["bloom_level"] == "remember"
    assert by_id["blk-create"]["bloom_level"] == "create"


def test_build_evidence_window_at_rung_ceiling_is_kept(monkeypatch):
    """A block AT exactly the target rung is not excluded (ceiling is inclusive)."""
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    window = build_evidence_window(
        _rung_chunk(), _OBJECTIVES["co-01"], target_rung="create",
    )
    block_ids = {b["block_id"] for b in window["blocks"]}
    assert block_ids == {"blk-remember", "blk-create", "blk-untagged"}


# ---------------------------------------------------------------------------
# (c) render_evidence_window — additive |bloom tag segment
# ---------------------------------------------------------------------------


def test_render_evidence_window_without_bloom_key_is_unchanged():
    window = {
        "blocks": [
            {"block_id": "b1", "role": "evidence", "polarity": "factual", "text": "hello"},
        ],
    }
    assert render_evidence_window(window) == "[b1|evidence|factual] hello"


def test_render_evidence_window_with_bloom_key_appends_fourth_segment():
    window = {
        "blocks": [
            {
                "block_id": "b1", "role": "evidence", "polarity": "factual",
                "text": "hello", "bloom_level": "apply",
            },
        ],
    }
    assert render_evidence_window(window) == "[b1|evidence|factual|apply] hello"


def test_render_evidence_window_mixed_blocks_render_independently():
    window = {
        "blocks": [
            {"block_id": "b1", "role": "evidence", "polarity": "factual", "text": "x"},
            {
                "block_id": "b2", "role": "evidence", "polarity": "factual",
                "text": "y", "bloom_level": "create",
            },
        ],
    }
    rendered = render_evidence_window(window)
    assert rendered.splitlines() == [
        "[b1|evidence|factual] x",
        "[b2|evidence|factual|create] y",
    ]


def test_render_evidence_window_end_to_end_with_build(monkeypatch):
    monkeypatch.setenv(BLOOM_WINDOWS_ENV, "1")
    window = build_evidence_window(
        _rung_chunk(), _OBJECTIVES["co-01"], target_rung="create",
    )
    rendered = render_evidence_window(window)
    assert "|remember]" in rendered
    assert "|create]" in rendered
    # The untagged block keeps the old 3-segment tag.
    assert "[blk-untagged|evidence|factual]" in rendered


# ---------------------------------------------------------------------------
# Byte-identical-when-unset regression net
# ---------------------------------------------------------------------------


def test_build_evidence_window_byte_identical_flag_unset(monkeypatch):
    """Same inputs, flag unset: two-arg call == three-arg call w/ target_rung=None."""
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    chunk = _rung_chunk()
    focus = _OBJECTIVES["co-01"]
    legacy = build_evidence_window(chunk, focus)
    explicit_none = build_evidence_window(chunk, focus, target_rung=None)
    explicit_ignored = build_evidence_window(chunk, focus, target_rung="remember")
    assert legacy == explicit_none == explicit_ignored


def test_resolve_chunk_misconceptions_byte_identical_flag_unset(monkeypatch):
    monkeypatch.delenv(BLOOM_WINDOWS_ENV, raising=False)
    chunk = _two_rung_chunk("apply", "create")
    first = resolve_chunk_misconceptions(chunk)
    second = resolve_chunk_misconceptions(chunk)
    assert first == second
    assert all(entry["bloom_level"] == "understand" for entry in first)
