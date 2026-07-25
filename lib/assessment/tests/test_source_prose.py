"""Contract net for the mining-time prose-only chunk view.

Covers ``lib/assessment/source_prose.py`` — the filter that keeps figure
alt-text, worked solutions, exercise banks, display-math and flattened tables
out of the assessment mining pool.

Three contracts are pinned:

1. **Default OFF is byte-identical** — with ``ED4ALL_ASSESSMENT_CLEAN_PROSE``
   unset the resolver is False and no caller filters anything.
2. **Source-agnostic** — every fixture below is synthetic and subject-neutral.
   The rule keys off ``data-semantik-block-role`` plus generic element kinds,
   so a fixture that never mentions a real corpus still exercises the whole
   decision path.
3. **Removal is bounded** — an unknown role is KEPT (a new converter role must
   not silently start deleting content) and prose adjacent to apparatus
   survives.
"""

import pytest

from lib.assessment.source_prose import (
    ENV_CLEAN_PROSE,
    MIN_SURVIVING_RUN,
    NON_PROSE_ROLES,
    ProseFilter,
    SHINGLE_CHARS,
    build_prose_filter,
    clean_chunks,
    resolve_clean_prose,
    resolve_source_html_paths,
)

pytest.importorskip("lxml")

# Long enough to shingle (>= SHINGLE_CHARS) and clearly non-prose.
ALT_TEXT = "A gray arrow inside a square, indicating the next action."
CAPTION = "Table 9.9 A summary of the values discussed in this section."
SOLUTION = "Substitute the value back into the original relation and simplify."
PROSE = "A closed shape is one whose boundary returns to its starting point."
PROSE_2 = "Two quantities are proportional when their ratio stays constant."

DOC = f"""<html><body><main>
  <section data-semantik-block-role="figure">
    <img alt="{ALT_TEXT}"/>
    <figcaption>{CAPTION}</figcaption>
  </section>
  <section><p>{PROSE}</p></section>
  <section data-semantik-block-role="solution"><p>{SOLUTION}</p></section>
  <section data-semantik-block-role="how_to"><p>{PROSE_2}</p></section>
  <table><tr><td>Step 1.</td><td>Locate the leading digit and mark it.</td></tr></table>
</main></body></html>"""


@pytest.fixture()
def doc(tmp_path):
    path = tmp_path / "chapter_one_accessible.html"
    path.write_text(DOC, encoding="utf-8")
    return path


# ---- flag resolution -------------------------------------------------------

def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(ENV_CLEAN_PROSE, raising=False)
    assert resolve_clean_prose() is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("", False), ("0", False), ("false", False), ("off", False),
    ],
)
def test_flag_parse_with_fallback(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_CLEAN_PROSE, raw)
    assert resolve_clean_prose() is expected


def test_garbage_flag_value_resolves_on_not_crash(monkeypatch):
    """Anything not explicitly falsey is truthy — never a raise."""
    monkeypatch.setenv(ENV_CLEAN_PROSE, "garbage")
    assert resolve_clean_prose() is True


# ---- fragment harvest ------------------------------------------------------

def test_harvest_covers_roles_and_structural_carriers(doc):
    filt = build_prose_filter([doc])
    assert filt
    assert filt.is_non_prose(ALT_TEXT)
    assert filt.is_non_prose(CAPTION)
    assert filt.is_non_prose(SOLUTION)


def test_kept_roles_are_not_harvested(doc):
    """``how_to`` carries instructional narrative and must survive."""
    filt = build_prose_filter([doc])
    assert not filt.is_non_prose(PROSE_2)


def test_unmarked_prose_survives(doc):
    filt = build_prose_filter([doc])
    assert not filt.is_non_prose(PROSE)


def test_unknown_role_is_kept(tmp_path):
    """A role the converter adds later must not start deleting content."""
    novel = "This narrative belongs to a role that did not exist before."
    path = tmp_path / "novel_accessible.html"
    path.write_text(
        f'<html><body><section data-semantik-block-role="brand_new_role">'
        f"<p>{novel}</p></section></body></html>",
        encoding="utf-8",
    )
    filt = build_prose_filter([path])
    assert "brand_new_role" not in NON_PROSE_ROLES
    assert not filt.is_non_prose(novel)


def test_missing_document_yields_none(tmp_path):
    assert build_prose_filter([]) is None


def test_unparseable_document_does_not_kill_the_phase(tmp_path):
    bad = tmp_path / "truncated_accessible.html"
    bad.write_bytes(b"")
    # An empty doc parses to nothing rather than raising; either way the
    # caller must get a value back, never an exception.
    build_prose_filter([bad])


# ---- excision --------------------------------------------------------------

def test_clean_removes_apparatus_and_keeps_neighbouring_prose(doc):
    filt = build_prose_filter([doc])
    mixed = f"{PROSE} {ALT_TEXT} {PROSE_2}"
    out = filt.clean(mixed)
    assert ALT_TEXT not in out
    assert PROSE.rstrip(".") in out
    assert PROSE_2.rstrip(".") in out


def test_clean_survives_shard_threshold(doc):
    """Runs left between two excised regions are dropped, not emitted."""
    filt = build_prose_filter([doc])
    out = filt.clean(f"{ALT_TEXT} tiny {CAPTION}")
    assert "tiny" not in out


def test_short_text_is_returned_unchanged(doc):
    filt = build_prose_filter([doc])
    short = "Too short to shingle."
    assert len(short) < SHINGLE_CHARS
    assert filt.clean(short) == short


def test_is_non_prose_ignores_sub_shingle_spans(doc):
    filt = build_prose_filter([doc])
    assert filt.is_non_prose("Solution") is False


def test_whitespace_differences_do_not_defeat_matching(doc):
    filt = build_prose_filter([doc])
    assert filt.is_non_prose(ALT_TEXT.replace(" ", "\n  "))


# ---- chunk plumbing --------------------------------------------------------

def test_clean_chunks_preserves_identity_fields(doc):
    filt = build_prose_filter([doc])
    chunks = [
        {"id": "c1", "text": f"{PROSE} {ALT_TEXT}", "word_count": 99,
         "learning_outcome_refs": ["CO-01"], "source": {"item_path": "a.html"}},
    ]
    cleaned, stats = clean_chunks(chunks, filt)
    assert cleaned[0]["id"] == "c1"
    assert cleaned[0]["learning_outcome_refs"] == ["CO-01"]
    assert cleaned[0]["source"] is chunks[0]["source"]
    assert ALT_TEXT not in cleaned[0]["text"]
    assert cleaned[0]["word_count"] == len(cleaned[0]["text"].split())
    assert stats["changed"] == 1


def test_clean_chunks_does_not_mutate_the_input(doc):
    filt = build_prose_filter([doc])
    original = f"{PROSE} {ALT_TEXT}"
    chunks = [{"id": "c1", "text": original}]
    clean_chunks(chunks, filt)
    assert chunks[0]["text"] == original


def test_clean_chunks_keeps_an_emptied_chunk_addressable(doc):
    """A fully-apparatus chunk stays in the list so id lookups never miss."""
    filt = build_prose_filter([doc])
    cleaned, stats = clean_chunks([{"id": "c1", "text": ALT_TEXT}], filt)
    assert len(cleaned) == 1
    assert cleaned[0]["id"] == "c1"
    assert cleaned[0]["text"] == ""
    assert stats["emptied"] == 1


def test_untouched_chunk_is_returned_by_identity(doc):
    filt = build_prose_filter([doc])
    chunk = {"id": "c1", "text": PROSE}
    cleaned, stats = clean_chunks([chunk], filt)
    assert cleaned[0] is chunk
    assert stats["changed"] == 0


# ---- source resolution -----------------------------------------------------

def test_resolve_source_html_paths_prefers_first_search_dir(tmp_path):
    preferred = tmp_path / "staged"
    fallback = tmp_path / "output"
    preferred.mkdir()
    fallback.mkdir()
    (preferred / "one_accessible.html").write_text("<html/>", encoding="utf-8")
    (fallback / "one_accessible.html").write_text("<html/>", encoding="utf-8")
    chunks = [{"source": {"item_path": "one_accessible.html"}}]
    paths = resolve_source_html_paths(chunks, [preferred, fallback])
    assert paths == [preferred / "one_accessible.html"]


def test_resolve_source_html_paths_falls_back(tmp_path):
    preferred = tmp_path / "staged"
    fallback = tmp_path / "output"
    preferred.mkdir()
    fallback.mkdir()
    (fallback / "two_accessible.html").write_text("<html/>", encoding="utf-8")
    chunks = [{"source": {"item_path": "sub/dir/two_accessible.html"}}]
    paths = resolve_source_html_paths(chunks, [preferred, fallback])
    assert paths == [fallback / "two_accessible.html"]


def test_resolve_source_html_paths_dedups_and_skips_missing(tmp_path):
    base = tmp_path / "staged"
    base.mkdir()
    (base / "here_accessible.html").write_text("<html/>", encoding="utf-8")
    chunks = [
        {"source": {"item_path": "here_accessible.html"}},
        {"source": {"item_path": "here_accessible.html"}},
        {"source": {"item_path": "gone_accessible.html"}},
        {"source": None},
        {},
    ]
    paths = resolve_source_html_paths(chunks, [base])
    assert paths == [base / "here_accessible.html"]


# ---- guardrails ------------------------------------------------------------

def test_empty_filter_is_falsey():
    assert not ProseFilter([])


def test_fragments_below_shingle_length_are_ignored():
    filt = ProseFilter(["short"])
    assert not filt
    assert filt.fragment_count == 0


def test_min_surviving_run_is_below_shingle_width():
    """A surviving run must be able to exist between two masked regions."""
    assert MIN_SURVIVING_RUN < SHINGLE_CHARS
