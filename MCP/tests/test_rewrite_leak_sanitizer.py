"""7B→Sonnet parity P0: CURIE / LO-id leak sanitizer (pipeline_tools).

Unit tests for the standalone, LLM-free sanitizer helpers in
``MCP/tools/pipeline_tools.py``:

  * ``_curie_visible_to_surface`` — visible minted-CURIE → surface term, with
    the hidden ``<span data-cf-curie>`` anchoring signal protected.
  * ``_resolve_objective_li_statements`` — objective ``<li>`` bare-id → the LO
    statement.
  * ``_objective_id_to_statement`` — both the Courseforge synthesized form and
    the LibV2 archive form.
  * ``sanitize_leaked_tokens_in_html`` — the post-hoc convenience wrapper.

No LLM dispatch, no pipeline run.
"""

import MCP.tools.pipeline_tools as pt


# ---------------------------------------------------------------------------
# P0 — visible CURIE leak sanitizer
# ---------------------------------------------------------------------------

def test_code_voice_minted_curie_replaced_with_surface_term():
    html = (
        "<p>The set of <code>democourse:integers</code> includes "
        "negative numbers.</p>"
    )
    out = pt._curie_visible_to_surface(
        html, "democourse:integers", "integers",
    )
    assert "<code>democourse:integers</code>" not in out
    assert "democourse:integers" not in out
    assert "The set of integers includes" in out


def test_data_cf_term_full_curie_reduced_to_localname():
    html = '<span data-cf-term="democourse:integers">integers</span>'
    out = pt._curie_visible_to_surface(
        html, "democourse:integers", "integers",
    )
    assert 'data-cf-term="integers"' in out
    assert 'data-cf-term="democourse:integers"' not in out


def test_bare_prose_token_replaced():
    html = "<p>We will study democourse:integers today.</p>"
    out = pt._curie_visible_to_surface(
        html, "democourse:integers", "integers",
    )
    assert "democourse:integers" not in out
    assert "study integers today" in out


def test_hidden_curie_span_token_preserved():
    """The hidden anchoring span MUST keep the literal CURIE token."""
    html = (
        "<p>About democourse:integers.</p>"
        '<span hidden data-cf-curie="democourse:integers">'
        "democourse:integers</span>"
    )
    out = pt._curie_visible_to_surface(
        html, "democourse:integers", "integers",
    )
    # Visible prose cleaned.
    visible = out.split("<span hidden")[0]
    assert "democourse:integers" not in visible
    assert "About integers." in visible
    # Hidden span token survives (curie-anchoring gate signal).
    assert (
        'data-cf-curie="democourse:integers">'
        "democourse:integers</span>" in out
    )


def test_curie_sanitizer_idempotent_on_clean_page():
    html = "<p>Just integers, no tokens.</p>"
    out = pt._curie_visible_to_surface(html, "alg:integers", "integers")
    assert out == html


def test_rdf_style_curie_localname_unchanged_signature():
    # A real CURIE wrapped in code is still replaced when called by the
    # sanitizer (the sanitizer is minted-only by caller contract), but the
    # function itself only ever rewrites the EXACT token it's handed.
    html = "<p>unrelated foo:bar token</p>"
    out = pt._curie_visible_to_surface(html, "alg:integers", "integers")
    assert "foo:bar" in out  # never touches a different CURIE


# ---------------------------------------------------------------------------
# P0 — objective <li> body resolution
# ---------------------------------------------------------------------------

def test_objective_li_bare_id_becomes_statement():
    html = '<ul><li data-cf-objective-id="TO-01">TO-01</li></ul>'
    out = pt._resolve_objective_li_statements(
        html, {"TO-01": "Explain prime numbers and provide examples."},
    )
    assert "<strong>TO-01:</strong>" in out
    assert "Explain prime numbers and provide examples." in out
    # id stays in the attribute.
    assert 'data-cf-objective-id="TO-01"' in out


def test_objective_li_empty_body_filled():
    html = '<li data-cf-objective-id="CO-02"></li>'
    out = pt._resolve_objective_li_statements(
        html, {"CO-02": "Apply divisibility tests."},
    )
    assert "Apply divisibility tests." in out


def test_objective_li_existing_prose_untouched():
    html = (
        '<li data-cf-objective-id="CO-03">'
        "<strong>CO-03:</strong> Already written statement.</li>"
    )
    out = pt._resolve_objective_li_statements(
        html, {"CO-03": "A DIFFERENT statement."},
    )
    assert "Already written statement." in out
    assert "A DIFFERENT statement." not in out


def test_objective_li_unresolved_id_left_as_is():
    html = '<li data-cf-objective-id="CO-99">CO-99</li>'
    out = pt._resolve_objective_li_statements(html, {"TO-01": "x"})
    assert out == html


# ---------------------------------------------------------------------------
# P0 — objective id → statement map (both doc forms)
# ---------------------------------------------------------------------------

def test_objective_map_courseforge_synthesized_form():
    doc = {
        "terminal_objectives": [{"id": "TO-01", "statement": "A"}],
        "chapter_objectives": [
            {"objectives": [{"id": "CO-01", "statement": "B"}]},
        ],
    }
    assert pt._objective_id_to_statement(doc) == {"TO-01": "A", "CO-01": "B"}


def test_objective_map_libv2_archive_form():
    doc = {
        "terminal_outcomes": [{"id": "TO-02", "statement": "C"}],
        "component_objectives": [{"id": "CO-03", "statement": "D"}],
    }
    assert pt._objective_id_to_statement(doc) == {"TO-02": "C", "CO-03": "D"}


def test_objective_map_defensive_on_garbage():
    assert pt._objective_id_to_statement(None) == {}
    assert pt._objective_id_to_statement({"terminal_objectives": "x"}) == {}


# ---------------------------------------------------------------------------
# P0 — post-hoc convenience wrapper
# ---------------------------------------------------------------------------

def test_post_hoc_wrapper_cleans_both_leak_classes():
    html = (
        '<ul><li data-cf-objective-id="TO-01">TO-01</li></ul>'
        "<p>Study <code>alg:integers</code> here.</p>"
    )
    out = pt.sanitize_leaked_tokens_in_html(
        html,
        minted_curie_map={"alg:integers": {"surface_forms": ["integers"]}},
        objective_statements={"TO-01": "Explain integers."},
    )
    assert "<code>alg:integers</code>" not in out
    assert "Study integers here." in out
    assert "Explain integers." in out


# ---------------------------------------------------------------------------
# P0 — VISIBLE provenance-id leak sanitizer (CLASS 1/2/3 identifier leakage)
# ---------------------------------------------------------------------------

_DART_ID = "dart:demo-textbook-ch1_accessible#af6e8d9fc7932ce6"
_DART_ID2 = "dart:demo-textbook-ch1_accessible#7bb8524612ed155d"


def test_class1_source_attribution_cite_stripped():
    html = (
        '<section data-cf-source-ids="%s,%s">'
        "<p>The product property holds.</p>"
        '<cite class="source-attribution">%s; %s</cite>'
        "</section>" % (_DART_ID, _DART_ID2, _DART_ID, _DART_ID2)
    )
    out = pt._strip_visible_provenance_ids(html)
    # Visible cite gone; ids gone from visible text.
    assert "source-attribution" not in out
    assert "#af6e8d9fc7932ce6</cite>" not in out
    # But the hidden attribute STILL carries the ids.
    assert 'data-cf-source-ids="%s,%s"' % (_DART_ID, _DART_ID2) in out
    # Prose preserved.
    assert "The product property holds." in out


def test_class1_bare_dart_id_run_in_prose_stripped():
    html = (
        "<p>This follows from %s; %s in the source.</p>" % (_DART_ID, _DART_ID2)
    )
    out = pt._strip_visible_provenance_ids(html)
    assert "dart:demo-textbook" not in out
    assert "This follows from" in out


def test_class1_dart_id_inside_attribute_preserved():
    html = '<section data-cf-source-ids="%s"><p>x.</p></section>' % _DART_ID
    out = pt._strip_visible_provenance_ids(html)
    assert 'data-cf-source-ids="%s"' % _DART_ID in out


def test_class2_according_to_source_chunk_cleaned():
    html = (
        "<p>According to the source, "
        "<cite>democourse_chunk_00013</cite>, the LCM of two "
        "numbers can be found by listing multiples.</p>"
    )
    out = pt._strip_visible_provenance_ids(html)
    assert "chunk_00013" not in out
    assert "According to the source" not in out
    assert "the LCM of two numbers can be found by listing multiples." in out


def test_class2_according_to_source_chunk_variant_cleaned():
    html = (
        "<p>According to the source chunk "
        "<cite>democourse_chunk_00016</cite>, the equal sign "
        "indicates equality.</p>"
    )
    out = pt._strip_visible_provenance_ids(html)
    assert "chunk_00016" not in out
    assert "According to the source" not in out
    assert "the equal sign indicates equality." in out


def test_class2_bare_chunk_cite_cleaned():
    html = "<p>The method works <cite>x_chunk_00010</cite> as shown.</p>"
    out = pt._strip_visible_provenance_ids(html)
    assert "chunk_00010" not in out
    assert "The method works" in out
    assert "as shown." in out


def test_class3_inline_parenthetical_stripped():
    html = "<p>We compute the LCM (CO-33) and the GCF here.</p>"
    out = pt._strip_visible_provenance_ids(html)
    assert "(CO-33)" not in out
    assert "We compute the LCM and the GCF here." in out


def test_class3_nonbreaking_hyphen_variant_stripped():
    # U+2011 non-breaking hyphen variant from the real export.
    html = "<p>Factor the expression (CO‑15) carefully.</p>"
    out = pt._strip_visible_provenance_ids(html)
    assert "CO‑" not in out
    assert "CO-15" not in out
    assert "Factor the expression carefully." in out


def test_class3_objective_li_prefix_preserved():
    # The objective <li> "CO-34: <statement>" is INTENDED — not a
    # parenthetical, so it must survive untouched.
    html = (
        '<li data-cf-objective-id="CO-34">'
        "<strong>CO-34:</strong> Solve linear equations.</li>"
    )
    out = pt._strip_visible_provenance_ids(html)
    assert "<strong>CO-34:</strong> Solve linear equations." in out
    assert 'data-cf-objective-id="CO-34"' in out


def test_hidden_curie_span_never_touched_by_provenance_strip():
    html = (
        "<p>About fractions.</p>"
        '<span hidden data-cf-curie="alg:fraction">alg:fraction</span>'
    )
    out = pt._strip_visible_provenance_ids(html)
    assert (
        'data-cf-curie="alg:fraction">alg:fraction</span>' in out
    )


def test_provenance_strip_idempotent():
    html = (
        '<section data-cf-source-ids="%s">'
        "<p>According to the source, "
        "<cite>x_chunk_00013</cite>, factor (CO-33) now.</p>"
        '<cite class="source-attribution">%s</cite>'
        "</section>" % (_DART_ID, _DART_ID)
    )
    once = pt._strip_visible_provenance_ids(html)
    twice = pt._strip_visible_provenance_ids(once)
    assert once == twice
    # attribute survives, visible leaks gone.
    assert 'data-cf-source-ids="%s"' % _DART_ID in once
    assert "source-attribution" not in once
    assert "chunk_00013" not in once
    assert "(CO-33)" not in once


def test_clean_page_unchanged_by_provenance_strip():
    html = "<p>Plain prose with no leaked identifiers at all.</p>"
    assert pt._strip_visible_provenance_ids(html) == html


def test_wrapper_runs_provenance_strip():
    html = (
        "<p>According to the source, <cite>x_chunk_00013</cite>, "
        "factor (CO-33).</p>"
    )
    out = pt.sanitize_leaked_tokens_in_html(html)
    assert "chunk_00013" not in out
    assert "(CO-33)" not in out


# ---------------------------------------------------------------------------
# P0 — sanitize_export_pages (in-place export cleaner)
# ---------------------------------------------------------------------------

def test_sanitize_export_pages_cleans_in_place(tmp_path):
    import json

    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    obj_dir = tmp_path / "01_learning_objectives"
    obj_dir.mkdir()
    (obj_dir / "synthesized_objectives.json").write_text(
        json.dumps(
            {"chapter_objectives": [
                {"objectives": [{"id": "CO-34",
                                 "statement": "Solve linear equations."}]},
            ]}
        ),
        encoding="utf-8",
    )
    dirty = content_dir / "week_01_content_01.html"
    dirty.write_text(
        '<section data-cf-source-ids="%s">'
        "<p>According to the source, <cite>x_chunk_00013</cite>, "
        "factor (CO-33).</p>"
        '<cite class="source-attribution">%s</cite></section>'
        '<ul><li data-cf-objective-id="CO-34">CO-34</li></ul>'
        % (_DART_ID, _DART_ID),
        encoding="utf-8",
    )
    clean = content_dir / "week_02_content_01.html"
    clean.write_text("<p>Plain clean page.</p>", encoding="utf-8")

    n = pt.sanitize_export_pages(str(tmp_path))
    assert n == 1  # only the dirty page rewritten

    out = dirty.read_text(encoding="utf-8")
    assert "chunk_00013" not in out
    assert "(CO-33)" not in out
    assert "source-attribution" not in out
    # attribute preserved.
    assert 'data-cf-source-ids="%s"' % _DART_ID in out
    # objective li resolved to its statement.
    assert "Solve linear equations." in out
    # clean page untouched.
    assert clean.read_text(encoding="utf-8") == "<p>Plain clean page.</p>"


def test_sanitize_export_pages_accepts_content_dir_directly(tmp_path):
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir()
    (content_dir / "p.html").write_text(
        "<p>According to the source, <cite>x_chunk_1</cite>, ok.</p>",
        encoding="utf-8",
    )
    n = pt.sanitize_export_pages(str(content_dir))
    assert n == 1
    assert "chunk_1" not in (content_dir / "p.html").read_text(encoding="utf-8")


def test_sanitize_export_pages_missing_dir_returns_zero(tmp_path):
    assert pt.sanitize_export_pages(str(tmp_path / "nope")) == 0
