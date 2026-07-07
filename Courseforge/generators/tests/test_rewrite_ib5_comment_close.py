"""Tests for the IB5 B04/B06 a11y producer fixes in ``RewriteProvider``.

A real 7B-output gap (calibration corpus): the two-pass rewrite tier
free-authors B04 ``multimedia`` / B06 ``diagram`` HTML and (1) emits MALFORMED
comment closes — ``--&gt;`` (HTML-entity-escaped) instead of a real ``-->`` —
which, per the WHATWG / ``html.parser`` tokenizer, leave the comment
UNTERMINATED and SWALLOW the downstream ``<track>`` / ``<details>`` /
``</table>`` siblings (captions / transcript / long-description genuinely
destroyed in a real browser), and (2) sometimes drop a renderer-guaranteed a11y
piece (e.g. the B04 audio-description note).

This suite exercises the two deterministic producer fixes:

- ``_fix_malformed_comment_closes`` — rewrites the entity-escaped comment close
  back to a real ``-->`` so the swallowed siblings survive; idempotent; leaves
  well-formed ``-->`` and legitimate escaped entities untouched.
- ``_inject_ib5_a11y_skeleton`` — re-injects ONLY the renderer-guaranteed
  structural a11y skeleton a shipping B04/B06 block is missing; anti-fabrication
  (no invented prose); idempotent (well-formed blocks byte-identical).

Each assertion is cross-checked against the actual IB5 a11y shape gate
(``lib.validators.rewrite_html_shape._check_ib5_a11y_shape``) — the producer fix
is correct iff the gate that fires on the broken HTML passes on the repaired
HTML.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# blocks.py is imported as a top-level module across the codebase.
SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from Courseforge.generators._rewrite_provider import (  # noqa: E402
    _fix_malformed_comment_closes,
    _inject_ib5_a11y_skeleton,
)
from lib.validators.rewrite_html_shape import (  # noqa: E402
    _ShapeParser,
    _check_ib5_a11y_shape,
)
from blocks import Block  # noqa: E402


def _gate(block_type: str, html: str):
    """Run the real IB5 a11y shape gate; return its ``[(code, reason), ...]``."""
    parser = _ShapeParser()
    parser.feed(html)
    parser.close()
    return _check_ib5_a11y_shape(block_type, parser)


def _gate_codes(block_type: str, html: str):
    return [code for code, _ in _gate(block_type, html)]


def _block(block_type: str, content, **kw) -> Block:
    return Block(
        block_id=f"week_01_content_01#{block_type}_t",
        block_type=block_type,
        page_id="week_01_content_01",
        sequence=0,
        content=content,
        **kw,
    )


# A B04 multimedia block whose comment close was entity-escaped, so the comment
# runs on and swallows the <track>, <details>, and audio-description note.
_MALFORMED_MULTIMEDIA = (
    '<figure class="multimedia">\n'
    '  <video controls>\n'
    '    <!-- placeholder URL --&gt;\n'
    '    <source src="x.mp4" type="video/mp4">\n'
    '  </video>\n'
    '  <track kind="captions" srclang="en" label="English">\n'
    '  <details data-cf-transcript><summary>Transcript</summary>'
    '<p>Real transcript text.</p></details>\n'
    '  <p class="audio-description">Audio description: a real note.</p>\n'
    '</figure>'
)

# A B06 diagram whose comment close was entity-escaped, swallowing the
# long-description <details>.
_MALFORMED_DIAGRAM = (
    '<figure class="diagram">\n'
    '  <img src="d.png" alt="A diagram">\n'
    '  <!-- more rows can be added --&gt;\n'
    '  <details class="diagram-longdesc"><summary>Long description</summary>'
    '<p>The nodes connect A to B.</p></details>\n'
    '  <table><caption>data</caption><tbody></tbody></table>\n'
    '</figure>'
)

# A fully well-formed B04 multimedia block (negative control — must be
# byte-identical through the producer fixes).
_WELLFORMED_MULTIMEDIA = (
    '<figure class="multimedia">\n'
    '  <video controls src="x.mp4">'
    '<track kind="captions" srclang="en" label="English captions"></video>\n'
    '  <details data-cf-transcript><summary>Transcript</summary>'
    '<p>Transcript here.</p></details>\n'
    '  <p class="audio-description">Audio description: a note.</p>\n'
    '</figure>'
)

# A fully well-formed B06 diagram (negative control).
_WELLFORMED_DIAGRAM = (
    '<figure class="diagram">\n'
    '  <img src="d.png" alt="A diagram">\n'
    '  <details class="diagram-longdesc"><summary>Long description</summary>'
    '<p>Nodes connect A to B.</p></details>\n'
    '  <table><caption>data</caption><tbody></tbody></table>\n'
    '</figure>'
)


# --------------------------------------------------------------------------- #
# 1. Malformed comment close swallow → repair restores the swallowed siblings.
# --------------------------------------------------------------------------- #

def test_malformed_comment_close_swallows_siblings_before_fix():
    """The raw broken HTML fires the IB5 gate because the comment ate the
    <track>/<details> (captions/transcript destroyed)."""
    codes = _gate_codes("multimedia", _MALFORMED_MULTIMEDIA)
    assert "MULTIMEDIA_CAPTIONS_MISSING" in codes
    assert "MULTIMEDIA_TRANSCRIPT_MISSING" in codes


def test_comment_close_repair_restores_track_and_transcript():
    """After the comment-close repair, the previously-swallowed <track> +
    <details> survive and the gate passes."""
    fixed = _fix_malformed_comment_closes(_MALFORMED_MULTIMEDIA)
    assert "--&gt;" not in fixed
    assert "-->" in fixed
    assert _gate_codes("multimedia", fixed) == []


def test_comment_close_repair_restores_diagram_longdesc():
    """B06: the entity-escaped comment swallowed the long-description <details>;
    repair restores it and the gate passes."""
    assert "REWRITE_IB5_A11Y_CONTRACT" in _gate_codes("diagram", _MALFORMED_DIAGRAM)
    fixed = _fix_malformed_comment_closes(_MALFORMED_DIAGRAM)
    assert _gate_codes("diagram", fixed) == []


def test_comment_close_repair_handles_double_escaped():
    """Double-/triple-escaped variants (``--&amp;gt;``) collapse to one real
    close."""
    s = '<video controls><!-- x --&amp;gt;<track kind="captions"></video>'
    fixed = _fix_malformed_comment_closes(s)
    assert "-->" in fixed and "&gt;" not in fixed
    s3 = '<video controls><!-- x --&amp;amp;gt;<track kind="captions"></video>'
    assert "-->" in _fix_malformed_comment_closes(s3)


def test_comment_close_repair_is_idempotent():
    fixed = _fix_malformed_comment_closes(_MALFORMED_MULTIMEDIA)
    assert _fix_malformed_comment_closes(fixed) == fixed


def test_comment_close_repair_leaves_wellformed_and_entities_untouched():
    """A well-formed ``-->`` and a legitimate escaped entity in prose are
    untouched — the repair only matches an escaped comment CLOSE."""
    s = (
        '<video controls><!-- real comment -->'
        '<track kind="captions"></video>'
        '<p>5 &gt; 3 and the arrow --&gt; is prose? no, it is a close test</p>'
    )
    # The ``--&gt;`` in prose IS a candidate close-shape; but a lone ``&gt;`` not
    # preceded by ``--`` must survive. Confirm the standalone entity survives.
    plain = '<p>5 &gt; 3</p>'
    assert _fix_malformed_comment_closes(plain) == plain
    # And a well-formed comment with no ``--&`` is byte-identical.
    clean = '<video controls><!-- ok --><track kind="captions"></video>'
    assert _fix_malformed_comment_closes(clean) == clean


# --------------------------------------------------------------------------- #
# 2. IB5 structural backstop re-injects a missing renderer-guaranteed piece.
# --------------------------------------------------------------------------- #

def test_backstop_reinjects_missing_audio_description():
    """A well-formed multimedia block that DROPPED the audio-description note
    (the (2) failure mode) gets the structural AD skeleton re-injected."""
    dropped_ad = (
        '<figure class="multimedia">\n'
        '  <video controls src="x.mp4">'
        '<track kind="captions" srclang="en" label="English"></video>\n'
        '  <details data-cf-transcript><summary>Transcript</summary>'
        '<p>T.</p></details>\n'
        '</figure>'
    )
    assert "MULTIMEDIA_AUDIO_DESC_MISSING" in _gate_codes("multimedia", dropped_ad)
    blk = _block("multimedia", {"audio_desc": "An authored AD note."})
    out = _inject_ib5_a11y_skeleton(dropped_ad, blk)
    assert _gate_codes("multimedia", out) == []
    # Anti-fabrication: it threaded the block's OWN field text, not invented prose.
    assert "An authored AD note." in out


def test_backstop_reinjects_missing_diagram_table():
    """A diagram missing the <table> data-equivalent gets the structural table
    skeleton re-injected."""
    no_table = (
        '<figure class="diagram">\n'
        '  <img src="d.png" alt="D">\n'
        '  <details class="diagram-longdesc"><summary>Long description</summary>'
        '<p>desc</p></details>\n'
        '</figure>'
    )
    assert "REWRITE_IB5_A11Y_CONTRACT" in _gate_codes("diagram", no_table)
    out = _inject_ib5_a11y_skeleton(no_table, _block("diagram", {}))
    assert _gate_codes("diagram", out) == []


def test_backstop_emits_empty_labelled_skeleton_when_no_source_text():
    """Anti-fabrication: with NO source text, the backstop emits the empty
    labelled 'pending' skeleton the renderer would — never invented narration."""
    bare = '<figure class="diagram"><img src="d.png" alt="D"></figure>'
    out = _inject_ib5_a11y_skeleton(bare, _block("diagram", {}))
    assert _gate_codes("diagram", out) == []
    assert "Long description pending." in out


def test_backstop_threads_long_description_field():
    """The backstop uses the block's ``long_description`` field text verbatim."""
    bare = '<figure class="diagram"><img src="d.png" alt="D"></figure>'
    blk = _block("diagram", {}, long_description="Authored long desc text.")
    out = _inject_ib5_a11y_skeleton(bare, blk)
    assert "Authored long desc text." in out
    assert "pending" not in out.lower().split("authored")[0][-40:]


# --------------------------------------------------------------------------- #
# 3. Well-formed blocks are byte-identical (idempotency / byte-stability).
# --------------------------------------------------------------------------- #

def test_wellformed_multimedia_byte_identical():
    assert _gate_codes("multimedia", _WELLFORMED_MULTIMEDIA) == []
    out = _fix_malformed_comment_closes(_WELLFORMED_MULTIMEDIA)
    out = _inject_ib5_a11y_skeleton(out, _block("multimedia", {}))
    assert out == _WELLFORMED_MULTIMEDIA


def test_wellformed_diagram_byte_identical():
    assert _gate_codes("diagram", _WELLFORMED_DIAGRAM) == []
    out = _fix_malformed_comment_closes(_WELLFORMED_DIAGRAM)
    out = _inject_ib5_a11y_skeleton(out, _block("diagram", {}))
    assert out == _WELLFORMED_DIAGRAM


def test_backstop_skips_non_ib5_block_types():
    """Only B04/B06 carry the structural a11y obligation; other types are
    untouched."""
    html = "<section><p>concept prose</p></section>"
    assert _inject_ib5_a11y_skeleton(html, _block("concept", "x")) == html


def test_backstop_idempotent_after_reinjection():
    no_table = (
        '<figure class="diagram"><img src="d.png" alt="D">'
        '<details><summary>Long description</summary><p>x</p></details></figure>'
    )
    once = _inject_ib5_a11y_skeleton(no_table, _block("diagram", {}))
    twice = _inject_ib5_a11y_skeleton(once, _block("diagram", {}))
    assert once == twice


# --------------------------------------------------------------------------- #
# 4. End-to-end on the order the provider applies them (fix → backstop).
# --------------------------------------------------------------------------- #

def test_combined_fix_then_backstop_passes_gate():
    """The malformed multimedia (comment swallow) flows through the EXACT
    producer order (comment-fix → backstop) and clears the gate."""
    blk = _block("multimedia", {"audio_desc": "AD note."})
    out = _fix_malformed_comment_closes(_MALFORMED_MULTIMEDIA)
    out = _inject_ib5_a11y_skeleton(out, blk)
    assert _gate_codes("multimedia", out) == []
