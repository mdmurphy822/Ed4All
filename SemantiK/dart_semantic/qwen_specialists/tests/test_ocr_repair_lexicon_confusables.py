"""Wave #22 quick-wins — the OCR-confusable repair pass MERGES the SemantiK
lexicon's label confusables (e.g. ``trvit`` -> ``TRY IT``) with its built-in
character-level map.

The lexicon confusables are WHOLE-TOKEN label normalizations, so they:
  * extend the deterministic garble DETECTOR (a matching token flags its region), and
  * add a SEPARATE deterministic accept arm in ``evaluate_edit`` (bypassing the
    char-level within-token / Levenshtein gates because the canonical form may be
    multi-word), still auditable + replay-conservation-checked.

The built-in char-confusable behavior is preserved; when the lexicon can't be
loaded (no schema / lib not importable) the merge is a no-op.
"""
from __future__ import annotations

from dart_semantic.qwen_specialists.ocr_repair import (
    CONFUSABLE_MAP,
    apply_accepted_edits,
    assert_repair_conservation,
    evaluate_edit,
    region_text_is_garbled,
    _lexicon_confusable_rules,
    _match_lexicon_confusable,
    _token_matches_lexicon_confusable,
)


def _has_lexicon() -> bool:
    return bool(_lexicon_confusable_rules())


# --------------------------------------------------------------------------
# Merge loads (JSON-by-path fallback when lib/ontology is not importable)
# --------------------------------------------------------------------------


def test_lexicon_rules_load_trvit():
    rules = _lexicon_confusable_rules()
    # The default profile (generic-academic+open-textbook) ships the
    # trvit->TRY IT confusable. If the lexicon JSON is unreachable the merge
    # is an honest no-op and this asserts nothing (graceful fallback).
    if not rules:
        return
    canon = {r.canonical for r in rules}
    assert "TRY IT" in canon


def test_json_fallback_honors_legacy_profile_alias(monkeypatch):
    # Pre-rename compat: an operator env still set to the LEGACY profile
    # spelling resolves the SAME confusables as the renamed default (the
    # fallback JSON reader folds each spec segment through the alias twin).
    from dart_semantic.qwen_specialists import ocr_repair as m

    monkeypatch.setenv("SEMANTIK_LEXICON_PROFILE", "generic-academic+openstax")
    legacy = m._lexicon_confusables_from_json()
    monkeypatch.setenv("SEMANTIK_LEXICON_PROFILE", "generic-academic+open-textbook")
    renamed = m._lexicon_confusables_from_json()
    assert legacy == renamed
    if legacy:  # graceful no-op when the schema is unreachable
        assert {e["pattern"]: e["canonical"] for e in legacy}.get("tr[vy]it") == "TRY IT"


def test_char_map_unchanged_by_merge():
    # The built-in character-level map is NOT mutated by the lexicon merge.
    assert len(CONFUSABLE_MAP) == 21
    assert all(hasattr(r, "src") and hasattr(r, "dst") for r in CONFUSABLE_MAP)


# --------------------------------------------------------------------------
# Detector extension
# --------------------------------------------------------------------------


def test_detector_flags_lexicon_confusable_token():
    if not _has_lexicon():
        return
    assert _token_matches_lexicon_confusable("trvit") is True
    assert region_text_is_garbled("Here is the trvit exercise below") is True


def test_detector_idempotent_on_canonical():
    # The canonical multi-word form ("TRY", "IT") is never re-flagged.
    assert _token_matches_lexicon_confusable("TRY") is False
    assert _token_matches_lexicon_confusable("IT") is False


# --------------------------------------------------------------------------
# Accept gate — lexicon arm
# --------------------------------------------------------------------------


def test_lexicon_edit_accepted_bypassing_char_gates():
    if not _has_lexicon():
        return
    region = "Here is the trvit exercise"
    decision = evaluate_edit(region, "trvit", "TRY IT")
    assert decision.accepted is True
    assert decision.reason.startswith("lexicon_confusable")
    assert decision.token_index is not None


def test_lexicon_edit_matcher_requires_exact_canonical():
    if not _has_lexicon():
        return
    # Right pattern, wrong canonical → no lexicon match (falls through to the
    # char gates, which reject the whitespace/multi-char edit).
    assert _match_lexicon_confusable("trvit", "TRY IT") is not None
    assert _match_lexicon_confusable("trvit", "try it") is None
    decision = evaluate_edit("the trvit here", "trvit", "TRY THIS")
    assert decision.accepted is False


def test_lexicon_edit_replay_conserved():
    if not _has_lexicon():
        return
    original = "Here is the trvit exercise"
    edits = [{"before": "trvit", "after": "TRY IT"}]
    repaired = apply_accepted_edits(original, edits)
    assert "TRY IT" in repaired
    assert "trvit" not in repaired
    # Strict replay check must pass (no drift).
    assert_repair_conservation(original, repaired, edits)


# --------------------------------------------------------------------------
# Existing char-confusable behavior preserved
# --------------------------------------------------------------------------


def test_char_confusable_still_accepted():
    decision = evaluate_edit("the hel1o world", "hel1o", "hello")
    assert decision.accepted is True
    assert decision.reason == "1_to_l"


def test_non_lexicon_whitespace_edit_still_rejected():
    # A whitespace-crossing edit that is NOT a lexicon confusable is still
    # rejected by the within-token gate (existing behavior).
    decision = evaluate_edit("foo bar baz", "foo", "foo bar")
    assert decision.accepted is False
    assert decision.reason == "not_within_token"
