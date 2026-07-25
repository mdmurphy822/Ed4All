"""Mining-time PROSE-ONLY view of an assessment chunkset.

Why this exists
---------------
A chunk's ``text`` is a *flattened* rendering of a SemantiK region, and that
region routinely contains non-prose apparatus alongside the instructional
sentences: figure alt-text, ``<figcaption>`` bodies, worked-solution steps,
exercise banks, display-math dumps and OCR-flattened tables. Because the
flattening is lossy, an assessment generator mining sentences out of
``chunk["text"]`` cannot tell an image description from a definition — so
strings like *"A gray checkmark inside a circle, indicating correct"* and raw
``\\begin{array}`` LaTeX ship as quiz distractors AND as correct answers.

The discriminating information is not gone, it is simply upstream: SemantiK
already labels these regions on the accessible HTML via
``data-semantik-block-role``, and the remaining carriers are ordinary
structural elements (``<table>``, ``<figcaption>``, ``img[@alt]``).

This module re-derives the prose-only view **at mining time** by reading that
markup back out of the source HTML and dropping any chunk sentence that came
from a non-prose region. The chunkset on disk is never rewritten, so
``semantik_chunks_sha256`` stays stable and no upstream phase re-runs — only
``assessment_synthesis`` sees the cleaned text.

Source-agnostic by construction
-------------------------------
The rule keys off the converter's OWN structural taxonomy plus generic HTML
element kinds. There is no corpus phrase list, no subject vocabulary and no
per-book tuning: a different textbook, a different publisher or a different
subject exercises exactly the same code path.

Roles are *kept* when they carry instructional narrative (``how_to``,
``objectives``, ``worked_example``, ``list``) and *dropped* when they are
apparatus around the narrative. Dropping is deliberately biased toward
recall: the mining pool only needs to be clean, it does not need to be
complete, so a borderline region is discarded rather than risked.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

logger = logging.getLogger(__name__)

ENV_CLEAN_PROSE = "ED4ALL_ASSESSMENT_CLEAN_PROSE"

#: ``data-semantik-block-role`` values whose text is apparatus, not prose.
#: Everything not listed here (``how_to``, ``objectives``, ``worked_example``,
#: ``list``, and any role a future converter version introduces) is KEPT, so a
#: new role never silently starts deleting content.
NON_PROSE_ROLES = frozenset({
    "apparatus",
    "apparatus_heading",
    "caption",
    "exercise",
    "figure",
    "footnote",
    "math",          # role="math" is only ever a block-level display-math
                     # section; inline math inside a prose <p> is unmarked and
                     # therefore survives.
    "readiness_check",
    "solution",
    "try_it",
})

#: Structural carriers of non-prose text that are not role-marked. A flattened
#: table row and an image description read as sentences once the tags are gone.
NON_PROSE_XPATHS = ("//table", "//figcaption")

#: Window used to decide "this span came from that region". Long enough to be
#: specific (a 40-char verbatim collision between real prose and apparatus is
#: vanishingly rare), short enough to survive the whitespace and entity
#: differences between the chunker's flattening and our own text extraction.
SHINGLE_CHARS = 40

#: A surviving run shorter than this is a shard left behind between two excised
#: regions ("Step 2.", a stray operand) rather than a sentence. Dropping it
#: keeps the mining pool readable.
MIN_SURVIVING_RUN = 25

_FALSEY = {"", "0", "false", "no", "off"}


def resolve_clean_prose(env: Dict[str, str] | None = None) -> bool:
    """Resolve :data:`ENV_CLEAN_PROSE` (default OFF, parse-with-fallback)."""
    raw = (env if env is not None else os.environ).get(ENV_CLEAN_PROSE)
    if raw is None:
        return False
    return str(raw).strip().lower() not in _FALSEY


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class ProseFilter:
    """Excises spans of chunk text that came from a non-prose region.

    Matching is done on 40-character shingles rather than on sentences. The
    chunker's flattening does not agree with ours at sentence granularity — a
    flattened table becomes ``"Step 1. hundredths place 23,658 Step 2."``,
    whose "sentences" are sub-20-character shards — so a sentence-level filter
    keeps exactly the shards that carry the apparatus. Character masking has
    no such blind spot: any 40-character window that occurs verbatim inside a
    non-prose region is masked out, and whatever survives between the masked
    runs is the prose.
    """

    def __init__(self, fragments: Iterable[str]) -> None:
        self._shingles: set[str] = set()
        n = 0
        for frag in fragments:
            frag = _norm(frag)
            if len(frag) < SHINGLE_CHARS:
                continue
            n += 1
            for i in range(len(frag) - SHINGLE_CHARS + 1):
                self._shingles.add(frag[i:i + SHINGLE_CHARS])
        self.fragment_count = n

    def __bool__(self) -> bool:
        return bool(self._shingles)

    def is_non_prose(self, span: str) -> bool:
        """True when any window of ``span`` occurs inside a non-prose region."""
        span = _norm(span)
        if len(span) < SHINGLE_CHARS:
            return False
        return any(
            span[i:i + SHINGLE_CHARS] in self._shingles
            for i in range(len(span) - SHINGLE_CHARS + 1)
        )

    def clean(self, text: str) -> str:
        """Return ``text`` with every non-prose span masked out."""
        span = _norm(text)
        if len(span) < SHINGLE_CHARS:
            return span
        mask = bytearray(len(span))
        for i in range(len(span) - SHINGLE_CHARS + 1):
            if span[i:i + SHINGLE_CHARS] in self._shingles:
                mask[i:i + SHINGLE_CHARS] = b"\x01" * SHINGLE_CHARS
        runs: List[str] = []
        current: List[str] = []
        for char, masked in zip(span, mask):
            if masked:
                if current:
                    runs.append("".join(current))
                    current = []
            else:
                current.append(char)
        if current:
            runs.append("".join(current))
        return " ".join(
            run.strip() for run in runs if len(run.strip()) >= MIN_SURVIVING_RUN
        )


def build_prose_filter(html_paths: Sequence[Path]) -> ProseFilter | None:
    """Harvest non-prose fragments from the SemantiK accessible HTML.

    Returns ``None`` when lxml is unavailable or no document could be parsed —
    the caller then keeps the unfiltered chunkset rather than mining from a
    partially-harvested filter, which would clean some chapters and not others.
    """
    try:
        from lxml import html as lxml_html
    except ImportError:
        logger.warning(
            "assessment clean-prose: lxml is unavailable — mining from the "
            "raw flattened chunk text (apparatus may leak into options). "
            "Install the lxml dependency to enable prose-only mining."
        )
        return None

    fragments: List[str] = []
    parsed = 0
    for path in html_paths:
        try:
            # An empty / non-markup file parses to a tree whose root is None
            # rather than raising, so both outcomes need the same guard.
            root = lxml_html.parse(str(path)).getroot()
        except Exception as exc:  # noqa: BLE001 - a bad doc must not kill the phase
            root = None
            reason: object = exc
        else:
            reason = "no document element"
        if root is None:
            logger.warning(
                "assessment clean-prose: could not parse %s (%s); its "
                "apparatus will not be filtered.", path, reason,
            )
            continue
        parsed += 1

        def _emit(el) -> None:
            # Two flattening conventions, because the chunker's is not ours:
            # ``text_content()`` concatenates across tag boundaries with no
            # separator, while joining ``itertext()`` inserts one. A flattened
            # table matches under exactly one of them, so harvest both.
            fragments.append(el.text_content())
            fragments.append(" ".join(el.itertext()))

        for el in root.xpath("//*[@data-semantik-block-role]"):
            if el.get("data-semantik-block-role") in NON_PROSE_ROLES:
                _emit(el)
        for xpath in NON_PROSE_XPATHS:
            for el in root.xpath(xpath):
                _emit(el)
        for el in root.xpath("//img[@alt]"):
            fragments.append(el.get("alt") or "")

    if not parsed:
        logger.warning(
            "assessment clean-prose: no source HTML parsed from %d candidate "
            "path(s) — mining from the raw flattened chunk text.",
            len(html_paths),
        )
        return None

    filt = ProseFilter(fragments)
    logger.info(
        "assessment clean-prose: harvested %d non-prose fragments from %d "
        "source document(s).", filt.fragment_count, parsed,
    )
    return filt


def resolve_source_html_paths(
    chunks: Sequence[Dict[str, Any]],
    search_dirs: Sequence[Path],
) -> List[Path]:
    """Map each chunk's ``source.item_path`` onto a readable HTML file.

    ``search_dirs`` are tried in order, so a caller can prefer the run's own
    staging dir over the converter's output dir.
    """
    wanted: List[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        src = chunk.get("source")
        if not isinstance(src, dict):
            continue
        item = str(src.get("item_path") or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        wanted.append(item)

    resolved: List[Path] = []
    missing: List[str] = []
    for item in wanted:
        name = Path(item).name
        for base in search_dirs:
            candidate = base / name
            try:
                if candidate.is_file():
                    resolved.append(candidate)
                    break
            except OSError:
                continue
        else:
            missing.append(name)
    if missing:
        logger.warning(
            "assessment clean-prose: %d/%d source document(s) not found under "
            "%s (e.g. %s); their apparatus will not be filtered.",
            len(missing), len(wanted),
            ", ".join(str(d) for d in search_dirs) or "<no search dir>",
            ", ".join(missing[:3]),
        )
    return resolved


def clean_chunks(
    chunks: Sequence[Dict[str, Any]],
    filt: ProseFilter,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return copies of ``chunks`` whose ``text`` holds prose only.

    Chunks are copied shallowly — only ``text`` (and ``word_count``, kept
    consistent) is replaced, so every other field the generator relies on
    (ids, ``learning_outcome_refs``, ``source``) is preserved by identity.

    A chunk whose text filters away entirely keeps an empty string rather than
    being dropped, so downstream id-keyed lookups never miss.
    """
    cleaned: List[Dict[str, Any]] = []
    stats = {"chunks": 0, "changed": 0, "emptied": 0,
             "chars_in": 0, "chars_out": 0}
    for chunk in chunks:
        stats["chunks"] += 1
        original = chunk.get("text") or ""
        stats["chars_in"] += len(original)
        new_text = filt.clean(original) if original else original
        stats["chars_out"] += len(new_text)
        if new_text == original:
            cleaned.append(chunk)
            continue
        stats["changed"] += 1
        if not new_text:
            stats["emptied"] += 1
        copy = dict(chunk)
        copy["text"] = new_text
        if "word_count" in copy:
            copy["word_count"] = len(new_text.split())
        cleaned.append(copy)
    return cleaned, stats
