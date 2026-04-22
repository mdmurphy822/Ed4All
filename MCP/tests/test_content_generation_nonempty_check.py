"""Wave 32 Deliverable C — content_generation fails on empty output.

Pre-Wave-32, live re-sims reported
``content_generation: 12/12 complete, gates=pass`` even when the
content-gen dispatcher returned zero actual body content (every page
was a template skeleton: ``<h1>Week N</h1><h2>Overview</h2>`` with no
paragraphs ≥ 30 words). The phase counted dispatched tasks, not
produced artifacts.

Wave 32 adds a phase-level empty-content guard that runs inline at the
end of ``_generate_course_content``. When every emitted page has fewer
than :data:`NON_TRIVIAL_WORD_FLOOR` (= 30) words in its body tags
(``<p>`` / ``<li>`` / ``<blockquote>`` / ``<figcaption>`` inside
``<main>``), the phase fails loudly with ``CONTENT_GENERATION_EMPTY``
and an actionable error message that mentions
``LOCAL_DISPATCHER_ALLOW_STUB`` and the agent_tool wiring gap.

The check is independent of the ``ContentGroundingValidator`` gate —
gates need routing + inputs to fire; this phase-level check
guarantees that even when the gate skips (pre-Wave-32 behaviour) the
dispatcher cannot silently return zero real content.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from MCP.tools.pipeline_tools import (
    _CONTENT_NONTRIVIAL_WORD_FLOOR,
    _check_content_nonempty,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


_EMPTY_TEMPLATE = """<!DOCTYPE html><html lang="en"><head>
<title>Week {n}</title></head><body>
<main role="main"><h1>Week {n}</h1><h2>Overview</h2></main>
</body></html>"""


_NONTRIVIAL_PAGE = """<!DOCTYPE html><html lang="en"><head>
<title>Week {n}</title></head><body>
<main role="main">
  <h1>Week {n}: Introduction</h1>
  <h2>Overview</h2>
  <p>Knowledge graphs organise information as a collection of nodes
  connected by typed edges. Each node represents an entity — a concept,
  person, place, or event — and each edge carries a semantic relation
  such as ``is a`` or ``depends on``.</p>
  <p>This week covers the foundational ontology-engineering vocabulary,
  including classes, properties, and individuals, and the motivation
  for committing to a shared ontology before scaling the graph.</p>
</main></body></html>"""


def _write_empty_pages(tmp_path: Path, count: int) -> List[str]:
    """Write ``count`` template-skeleton pages and return their paths."""
    paths: List[str] = []
    for i in range(1, count + 1):
        p = tmp_path / f"week_{i:02d}.html"
        p.write_text(_EMPTY_TEMPLATE.format(n=i), encoding="utf-8")
        paths.append(str(p))
    return paths


def _write_nontrivial_pages(tmp_path: Path, count: int) -> List[str]:
    paths: List[str] = []
    for i in range(1, count + 1):
        p = tmp_path / f"week_{i:02d}.html"
        p.write_text(_NONTRIVIAL_PAGE.format(n=i), encoding="utf-8")
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------- #
# Deliverable C contract
# ---------------------------------------------------------------------- #


def test_twelve_empty_template_pages_fail(tmp_path: Path):
    """12 empty-template pages → CONTENT_GENERATION_EMPTY error string."""
    paths = _write_empty_pages(tmp_path, 12)
    error = _check_content_nonempty(paths)
    assert error is not None
    assert "CONTENT_GENERATION_EMPTY" in error


def test_twelve_real_content_pages_pass(tmp_path: Path):
    """12 pages with real body content → returns None (phase passes)."""
    paths = _write_nontrivial_pages(tmp_path, 12)
    error = _check_content_nonempty(paths)
    assert error is None


def test_mixed_pages_pass(tmp_path: Path):
    """Only all-empty fails; partial coverage is the gate's job, not the phase check.

    The phase-level check's contract is minimum-viability: the
    dispatcher returned at least ONE page with real content. Fine-
    grained grounding checks are left to ``ContentGroundingValidator``.
    """
    empty_paths = _write_empty_pages(tmp_path, 9)
    nt_dir = tmp_path / "nt"
    nt_dir.mkdir()
    nontrivial_paths = _write_nontrivial_pages(nt_dir, 3)
    mixed = empty_paths + nontrivial_paths
    assert len(mixed) == 12
    error = _check_content_nonempty(mixed)
    assert error is None, (
        "3/12 non-empty pages should be enough for the phase to pass; "
        "only the all-empty case fails CONTENT_GENERATION_EMPTY."
    )


def test_error_message_is_actionable(tmp_path: Path):
    """The error must mention LOCAL_DISPATCHER_ALLOW_STUB + the likely cause."""
    paths = _write_empty_pages(tmp_path, 12)
    error = _check_content_nonempty(paths)
    assert error is not None
    # Actionable triage hints — mentions the bypass flag + agent_tool
    # wiring so operators know what to inspect.
    assert "LOCAL_DISPATCHER_ALLOW_STUB" in error
    assert "agent_tool" in error
    # Includes the specific thresholds so the log is self-explaining.
    assert str(_CONTENT_NONTRIVIAL_WORD_FLOOR) in error


def test_empty_page_list_is_not_empty_content(tmp_path: Path):
    """Back-compat: empty page_paths list → returns None.

    When the dispatcher bails out early with zero pages the tool has
    already surfaced an upstream error; the phase-level empty-content
    guard is not responsible for flagging that condition (other
    phases' error paths already fire). Returning None here ensures
    the check doesn't double-fail clean error paths.
    """
    assert _check_content_nonempty([]) is None


def test_nontrivial_content_in_body_without_main_still_counts(tmp_path: Path):
    """Pages without <main> scope still check against body tags.

    The real Courseforge emitter uses ``role="main"`` and ``<main>``
    both; tests should not be brittle against pages that only use
    the ARIA role or neither. When no main wrapper is found the
    scope falls back to the document body, matching the
    ContentGroundingValidator's behaviour.
    """
    p = tmp_path / "week_01.html"
    # Pure body, no main tag or role attribute. Body paragraph must
    # clear the 30-word floor to be counted as non-trivial.
    p.write_text(
        """<!DOCTYPE html><html><body>
  <h1>Week 1</h1>
  <p>Knowledge graphs organise information as nodes and edges with
  typed semantic relations between entities. This forms the foundation
  of modern ontology engineering practice and enables downstream
  reasoning over the structured domain knowledge that the graph
  captures across multiple interconnected conceptual regions.</p>
</body></html>""",
        encoding="utf-8",
    )
    error = _check_content_nonempty([str(p)])
    assert error is None, (
        "A body-level <p> with ≥ 30 words should count as non-trivial "
        f"even when the page has no <main> wrapper. Got: {error}"
    )
