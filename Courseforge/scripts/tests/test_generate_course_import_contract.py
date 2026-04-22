"""Wave 51 — cross-module import contract lock-in.

Waves 48 and 50 were developed in parallel; Wave 50's PR branch
(``wave50/content-type-enum-validation``) picked up Wave 48's edits to
``Courseforge/scripts/generate_course.py`` (the new
``from lib.ontology.bloom import bloom_to_cognitive_domain as
_bloom_to_cognitive_domain`` import) WITHOUT the matching
``lib/ontology/bloom.py`` helper definition that Wave 48 added. At
commit ``ceaaaea`` (Wave 50 alone) the module raised ``ImportError``
at load time — any test importing ``generate_course`` would have
failed before generation logic ran. Only the subsequent merge of
Wave 48 into ``dev-v0.3.0`` papered it over at HEAD.

These guards catch the failure mode at test time so future parallel
workers can't reintroduce a transient broken-import window:

1. ``generate_course`` imports cleanly against the current tree.
2. Every symbol ``generate_course`` imports from
   ``lib.ontology.bloom`` is actually defined on that module — so a
   stale ``from lib.ontology.bloom import <x>`` statement without a
   matching definition on ``bloom.py`` trips immediately.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_generate_course_imports_cleanly():
    """Module-level import must not raise."""
    import Courseforge.scripts.generate_course  # noqa: F401


def test_bloom_to_cognitive_domain_wiring_present():
    """Wave 48 locked this symbol in; verify it's still callable."""
    from lib.ontology.bloom import bloom_to_cognitive_domain

    assert callable(bloom_to_cognitive_domain)
    # Wave 48 contract: every canonical Bloom level resolves to a
    # value in the cognitive_domain.json enum.
    for level in ("remember", "understand", "apply", "analyze",
                  "evaluate", "create"):
        assert isinstance(bloom_to_cognitive_domain(level), str)


def test_all_lib_ontology_bloom_imports_resolve():
    """Parse ``generate_course.py`` + ``_content_gen_helpers.py`` and
    verify every ``from lib.ontology.bloom import X`` symbol is
    defined on the bloom module.

    This catches the exact Wave 48/50 race condition: a script
    branch adds a new bloom import without the definition-side
    companion commit. The test scans via AST so it doesn't depend
    on module import order or on helper internals.
    """
    import lib.ontology.bloom as bloom_mod

    candidate_files = [
        PROJECT_ROOT / "Courseforge" / "scripts" / "generate_course.py",
        PROJECT_ROOT / "MCP" / "tools" / "_content_gen_helpers.py",
    ]

    missing: list[tuple[str, str]] = []
    for path in candidate_files:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "lib.ontology.bloom":
                continue
            for alias in node.names:
                symbol = alias.name
                if symbol == "*":
                    continue
                if not hasattr(bloom_mod, symbol):
                    missing.append((str(path.relative_to(PROJECT_ROOT)), symbol))

    assert not missing, (
        "generate_course.py / _content_gen_helpers.py import symbols "
        "from lib.ontology.bloom that are NOT defined on that module. "
        "Parallel workers added an import without the matching "
        "definition — the module will ImportError at load time. "
        f"Missing symbols: {missing}"
    )


def test_section_content_type_enum_wiring_present():
    """Wave 50 locked the content_type enum wiring; verify it's still
    in place (companion guard to the Wave 48 helper check above)."""
    from Courseforge.scripts import generate_course

    assert hasattr(generate_course, "SECTION_CONTENT_TYPE_ENUM")
    assert "explanation" in generate_course.SECTION_CONTENT_TYPE_ENUM


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
