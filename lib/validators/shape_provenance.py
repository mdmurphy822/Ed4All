"""Phase 6 — Shape-source provenance scanner.

Maps SHACL shape IRIs to ``(Turtle file, line number)`` by regex-scanning
the shape file for shape declarations. Zero new dependencies; lazy
per-file cache so multiple lookups against the same file share one
parse.

Why a regex scan instead of subclassing ``rdflib.plugins.parsers.notation3``
or pulling in a third-party Turtle AST tool: rdflib's parser strips
source positions before yielding triples, and subclassing it couples
to internals that aren't a stable API. A 30-line line-anchored scan
handles the canonical authoring style ``cfshapes:Foo a sh:NodeShape ;``
exactly, which is what every Phase 4+ shape file in the repo uses.

Documented limitations of the scanner:

* **Inline blank-node PropertyShapes.** A property shape declared inline
  inside ``sh:property [ ... ]`` has no IRI of its own — pyshacl reports
  ``sh:sourceShape`` as a blank node. The scanner never indexes blank
  nodes; callers fall back to ``shape_line=None`` for those. The Phase
  4 ``PageObjectivesMinCountShape`` is the only NodeShape in the canonical
  shape file; its inline PropertyShape inherits the parent's line at
  the consumer level if the consumer wants editor feedback.
* **Anonymous NodeShapes.** ``[ a sh:NodeShape ; sh:targetClass ... ]``
  isn't matched (no leading IRI). Phase 4 doesn't use this style.
* **Multi-line declarations.** A NodeShape whose IRI lives on one line
  and whose ``a sh:NodeShape`` lives on a continuation line isn't
  matched. Phase 4 doesn't use this style either.

Returns ``None`` from ``lookup`` rather than raising for any of the
above; the parent Phase 6 task constraint is that the enricher must
tolerate shape IRIs that aren't in the cache.

See ``plans/phase-6-shacl-result-enrichment.md`` § 2.1 for the
architectural decision and Q-new (``q_20260426_233557_671ecebc``) for
the corpus-grounded justification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

__all__ = [
    "ShapeSourceLocation",
    "ShapeSourceIndex",
]


# ``@prefix foo: <http://example.org/ns#> .``
_PREFIX_RE = re.compile(
    r"^\s*@prefix\s+([\w-]*):\s*<([^>]+)>\s*\.\s*$"
)


# Shape declaration: handles the canonical Phase 4 style where the shape
# identifier is on its own line and ``a sh:NodeShape`` follows on the
# next non-blank line:
#
#   cfshapes:PageObjectivesMinCountShape
#     a sh:NodeShape ;
#
# AND the inline single-line style:
#
#   cfshapes:Foo a sh:NodeShape ;
#
# The strategy is two-pass: scan for lines that begin with a shape
# identifier (CURIE or angle-bracketed IRI), then look ahead at the
# next non-blank/non-comment line for the ``a sh:NodeShape`` anchor.
# When the identifier and anchor are on the SAME line the look-ahead
# is a no-op.

# Line-leading shape identifier (no anchor — that gets matched on the
# next pass). The trailing assertion ``[^@\w]`` excludes ``@prefix`` /
# ``@base`` directives and bare keywords that happen to start with
# letters.
_SHAPE_IDENT_RE = re.compile(
    r"""
    ^\s*
    (                                    # group 1: shape identifier
        <[^>]+>                          #   <full IRI>
        |
        [\w-]+:[\w-]+                    #   prefix:localname CURIE
    )
    (?:\s|$)                              # whitespace or end-of-line
    """,
    re.VERBOSE,
)


# ``a sh:NodeShape`` / ``rdf:type sh:PropertyShape`` anchor; the
# whole line may be just the anchor (continuation line in multi-line
# style) or include the identifier (inline style). The leading-IRI
# group is optional so the same regex matches both.
_SHAPE_ANCHOR_RE = re.compile(
    r"""
    ^\s*
    (?:                                   # optional leading identifier
        (
            <[^>]+>                       #   <full IRI>
            |
            [\w-]+:[\w-]+                 #   CURIE
        )
        \s+
    )?
    (?:a|rdf:type)\s+
    ([\w-]+):(NodeShape|PropertyShape)
    \b
    """,
    re.VERBOSE,
)


# Lines we skip during the look-ahead scan (blank or comment-only).
_SKIP_LINE_RE = re.compile(r"^\s*(?:#.*)?$")


_SHACL_NS = "http://www.w3.org/ns/shacl#"


@dataclass(frozen=True)
class ShapeSourceLocation:
    """Where a SHACL shape was authored.

    ``file_path`` is whatever path the caller passed to ``build_for_file``
    — typically a project-relative ``Path`` (e.g.
    ``lib/validators/shacl/page_objectives_shacl.ttl``). ``line_number``
    is 1-indexed (matches editor + GitHub annotation conventions).
    """

    file_path: Path
    line_number: int


class ShapeSourceIndex:
    """Lazy per-file regex-built index of ``shape_iri -> location``.

    Usage:

    >>> idx = ShapeSourceIndex()
    >>> idx.build_for_file(Path("lib/validators/shacl/page_objectives_shacl.ttl"))
    >>> idx.lookup("https://ed4all.dev/ns/courseforge/v1/shapes#PageObjectivesMinCountShape")
    ShapeSourceLocation(file_path=PosixPath('...page_objectives_shacl.ttl'), line_number=41)
    """

    def __init__(self) -> None:
        # path -> {shape_iri: location}
        self._cache: Dict[Path, Dict[str, ShapeSourceLocation]] = {}

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #

    def build_for_file(self, ttl_path: Path) -> Dict[str, ShapeSourceLocation]:
        """Scan ``ttl_path`` and return a ``{shape_iri: location}`` mapping.

        Idempotent + cached: a second call with the same path returns the
        cached mapping. Tolerant of malformed Turtle — returns whatever
        shape declarations the regex matched and never raises on parse
        problems. A missing file raises ``FileNotFoundError`` (the
        caller almost certainly mistyped the path; failing loud is
        kinder than silently returning an empty mapping).
        """
        ttl_path = Path(ttl_path)
        if ttl_path in self._cache:
            return self._cache[ttl_path]

        text = ttl_path.read_text(encoding="utf-8")
        prefixes = self._collect_prefixes(text)
        shapes: Dict[str, ShapeSourceLocation] = {}
        lines = text.splitlines()

        # Two-pass scan: walk every line looking for the
        # ``a sh:NodeShape`` / ``a sh:PropertyShape`` anchor. When the
        # anchor matches, the shape identifier is either on the same
        # line (inline style) or on the most recent prior non-blank
        # non-comment line that begins with a shape identifier
        # (multi-line style — Phase 4's canonical pattern). The
        # latter is the load-bearing case.
        pending_ident: Optional[str] = None
        pending_ident_line: Optional[int] = None

        for line_no, raw in enumerate(lines, start=1):
            if _SKIP_LINE_RE.match(raw):
                # Blank or comment line — preserve any pending identifier;
                # the anchor may still appear on a later non-blank line.
                continue

            anchor_match = _SHAPE_ANCHOR_RE.match(raw)
            if anchor_match:
                inline_token = anchor_match.group(1)
                shape_prefix = anchor_match.group(2)
                # Confirm the shape-type prefix resolves to the SHACL ns.
                shape_ns = prefixes.get(shape_prefix)
                if shape_ns != _SHACL_NS:
                    # Heterodox prefix usage we don't recognise. Skip
                    # silently rather than mis-index.
                    pending_ident = None
                    pending_ident_line = None
                    continue

                if inline_token is not None:
                    shape_iri = self._resolve_token(inline_token, prefixes)
                    decl_line = line_no
                elif pending_ident is not None:
                    shape_iri = self._resolve_token(pending_ident, prefixes)
                    decl_line = pending_ident_line  # type: ignore[assignment]
                else:
                    # Anonymous NodeShape (``[ a sh:NodeShape ; ... ]``)
                    # or look-ahead failed — skip. Documented limitation.
                    pending_ident = None
                    pending_ident_line = None
                    continue

                if shape_iri is not None and decl_line is not None:
                    shapes[shape_iri] = ShapeSourceLocation(
                        file_path=ttl_path, line_number=decl_line
                    )
                pending_ident = None
                pending_ident_line = None
                continue

            ident_match = _SHAPE_IDENT_RE.match(raw)
            if ident_match:
                # Defer commitment — we don't know yet whether the next
                # non-blank line declares it as a NodeShape /
                # PropertyShape. Stash and let the anchor branch above
                # commit (or clear) on the next iteration.
                pending_ident = ident_match.group(1)
                pending_ident_line = line_no
            else:
                # Some other line (a property assertion mid-block, a
                # base directive, etc.). Pending identifier carries
                # over only across blank lines / comments — any other
                # content invalidates it.
                pending_ident = None
                pending_ident_line = None

        self._cache[ttl_path] = shapes
        return shapes

    def build_for_files(self, paths: Iterable[Path]) -> None:
        """Eagerly populate the cache for multiple files.

        Useful when the enricher knows the full set of shape files at
        startup and wants to amortise the scan cost.
        """
        for path in paths:
            self.build_for_file(Path(path))

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def lookup(
        self,
        shape_iri: str,
        *,
        hint_file: Optional[Path] = None,
    ) -> Optional[ShapeSourceLocation]:
        """Return the source location for ``shape_iri``, or ``None``.

        Search order:

        1. If ``hint_file`` is given and it's already in the cache, look
           there first (typical optimization when the caller knows
           which shape file produced the report).
        2. Otherwise iterate every cached file's mapping. (Linear in the
           number of shape files, but the cache is typically a single
           file in Phase 4 / Phase 6.)
        3. Return ``None`` if no cached file has the shape.

        Never raises: blank-node ``shape_iri`` strings, unknown IRIs,
        and shapes declared in unscanned files all return ``None``.
        """
        if not shape_iri:
            return None
        # Blank nodes serialised as strings start with ``_:`` — not an
        # IRI we can index, so we don't even bother probing the cache.
        if shape_iri.startswith("_:"):
            return None

        if hint_file is not None:
            hint = Path(hint_file)
            mapping = self._cache.get(hint)
            if mapping is not None:
                loc = mapping.get(shape_iri)
                if loc is not None:
                    return loc

        for mapping in self._cache.values():
            loc = mapping.get(shape_iri)
            if loc is not None:
                return loc
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_prefixes(text: str) -> Dict[str, str]:
        """Return a ``{prefix: namespace_uri}`` table from ``@prefix`` lines."""
        prefixes: Dict[str, str] = {}
        for raw in text.splitlines():
            m = _PREFIX_RE.match(raw)
            if m:
                prefixes[m.group(1)] = m.group(2)
        return prefixes

    @staticmethod
    def _resolve_token(token: str, prefixes: Dict[str, str]) -> Optional[str]:
        """Resolve a shape identifier token to a full IRI string.

        Token forms:

        * ``<http://example.org/Foo>`` -> ``http://example.org/Foo``
        * ``cfshapes:Foo`` + a known prefix table -> namespace + ``Foo``

        Returns ``None`` when the prefix isn't in the table (the
        regex matched but the file didn't declare the prefix —
        treat as not-indexable).
        """
        if token.startswith("<") and token.endswith(">"):
            return token[1:-1]
        if ":" in token:
            prefix, local = token.split(":", 1)
            ns = prefixes.get(prefix)
            if ns is None:
                return None
            return f"{ns}{local}"
        return None
