"""Fix B — CID-blind text-quality gate.

``_text_layer_quality_ok`` previously counted only Unicode replacement chars
against ``REPLACEMENT_MAX_FRAC`` (0.10). A font with no ToUnicode CMap surfaces
every glyph as an undecoded ``(cid:NN)`` token rather than ``�``, so a fully
CID-corrupt text layer passed the gate and shipped garbage. Fix B adds the
chars spanned by ``(cid:N)`` tokens (reusing ``region_detection.CID_RE``) to
the corruption numerator. Threshold unchanged at 0.10.
"""
from __future__ import annotations

from semantik_structure.extract_shared import _text_layer_quality_ok


def test_clean_text_passes():
    text = (
        "The quantum Zeno effect arises when frequent measurement freezes "
        "the evolution of a system. In this contribution we examine the "
        "decoherent histories formulation of the same phenomenon."
    )
    assert _text_layer_quality_ok(text) is True


def test_cid_heavy_text_trips_gate():
    # Mostly undecoded glyph tokens -> well above the 0.10 corruption fraction.
    text = "(cid:12)(cid:5)(cid:99)(cid:3)(cid:71)(cid:72) " * 6 + "abc"
    assert _text_layer_quality_ok(text) is False


def test_lightly_corrupt_below_threshold_passes():
    # One short cid token in a long clean paragraph -> well under 0.10.
    clean = (
        "Decoherent histories provide a consistent framework for quantum "
        "mechanics without an external observer, assigning probabilities to "
        "sequences of events that satisfy a decoherence condition."
    )
    text = clean + " (cid:7)"
    frac = len("(cid:7)") / len(text)
    assert frac < 0.10  # guard the fixture
    assert _text_layer_quality_ok(text) is True


def test_replacement_and_cid_combine():
    # Neither alone trips 0.10, but combined corruption crosses it.
    body = "x" * 80  # 80 letters
    repl = "�" * 6  # 6 replacement chars
    cids = "(cid:1)" * 2  # 14 chars of cid tokens
    text = body + repl + cids  # len 100; corruption 20/100 = 0.20 > 0.10
    assert _text_layer_quality_ok(text) is False


def test_no_cid_path_unchanged():
    # A pure-replacement-char corrupt string still trips (legacy behaviour).
    text = "�" * 20 + "abcdefghij"
    assert _text_layer_quality_ok(text) is False
