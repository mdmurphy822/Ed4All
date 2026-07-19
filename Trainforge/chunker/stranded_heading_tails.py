"""Deterministic post-pass — relocate stranded next-section heading tails.

Defect D4 (MAJOR, SemantiK GLM-OCR lane). In a scan-OCR'd chunkset the
opener of the FOLLOWING section can end up glued onto the TAIL of the
prior section's last chunk — e.g. a chunk that ends
``"...Figure. 1.1 EXERCISES"`` where ``"1.1 EXERCISES"`` is really the
``1.1 Exercises`` section heading that begins the NEXT chunk. Left in
place the stranded marker (a) pollutes the prior chunk's retrieval text
with a heading that belongs to a different section, and (b) strips the
following chunk of its own opening heading anchor.

This module is a deterministic post-pass over the FINAL emitted chunk
list, mirroring the ``apply_chunk_overlap`` precedent in ``chunker.py``:
it takes ``List[Dict]`` chunk dicts and mutates ``text`` +
``word_count`` + ``tokens_estimate`` ONLY. ``html`` is deliberately
UNTOUCHED (exactly like ``apply_chunk_overlap``) — the relocation is a
retrieval-text correctness fix, not an HTML re-render; the rendered
markup is owned upstream by the extractor.

The pass is CONSERVATIVE by construction (see ``_match_stranded_tail``):
it fires only when a standalone ``N.M <SECTION-MARKER>`` sits at the very
tail of a chunk (preceded by a newline OR by sentence-terminal
punctuation + whitespace), the all-caps section word is >= 3 letters,
there is an eligible SAME-FLOW following chunk to receive it, and moving
the marker never empties the source chunk. It is idempotent — after a
relocation the source tail no longer matches, so a second application is
a no-op.

Gated by ``TRAINFORGE_RELOCATE_STRANDED_HEADINGS`` (default OFF ->
byte-identical legacy emit). The resolver + pure function follow the
``apply_chunk_overlap`` pattern: the function itself is pure (it simply
finds nothing to do on a healthy corpus), and the CALL SITE gates on the
resolver so an off flag skips the pass entirely.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

__all__ = [
    "RELOCATE_STRANDED_HEADINGS_ENV",
    "relocate_stranded_heading_tails",
    "resolve_relocate_stranded_headings",
]

#: Env var gating the stranded-heading-tail relocation post-pass. Default
#: OFF (unset / falsey / garbage -> byte-identical legacy emit).
RELOCATE_STRANDED_HEADINGS_ENV: str = "TRAINFORGE_RELOCATE_STRANDED_HEADINGS"

#: A stranded next-section heading marker at a chunk's TAIL: an ``N.M``
#: section number followed by a single trailing SECTION word that is
#: either the literal ``EXERCISES`` or any >= 3-letter all-caps token
#: (REVIEW / SUMMARY / ...), with nothing but optional whitespace after
#: it (the ``$`` anchor guarantees no lowercase sentence continuation —
#: so the marker is never fired inside a real sentence).
_TAIL_MARKER_RE = re.compile(r"(\d+\.\d+\s+(?:EXERCISES|[A-Z]{3,}))\s*$")

#: Sentence-terminal punctuation that (with intervening whitespace) makes
#: a tail marker "standalone" even without a hard newline before it.
_SENTENCE_TERMINALS = (".", "!", "?")


def resolve_relocate_stranded_headings(
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Resolve ``TRAINFORGE_RELOCATE_STRANDED_HEADINGS`` (parse-with-fallback, OFF).

    Mirrors ``resolve_chunk_section_hard_break``'s parse style: truthy on
    ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive); everything else
    (unset / empty / garbage) -> ``False`` (feature off -> byte-identical
    legacy emit).
    """
    import os

    src = env if env is not None else os.environ
    return src.get(RELOCATE_STRANDED_HEADINGS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _match_stranded_tail(text: str) -> Optional[str]:
    """Return the stranded section-heading marker at ``text``'s tail, else None.

    Fires only when the marker is STANDALONE at the tail — i.e. it is
    preceded by a newline, OR by sentence-terminal punctuation + whitespace.
    A chunk that is JUST the marker (nothing before it) is excluded so the
    relocation never empties a chunk. Returns the exact marker substring
    (e.g. ``"1.1 EXERCISES"``) to move; ``None`` when no eligible marker.
    """
    if not text:
        return None
    stripped = text.rstrip()
    match = _TAIL_MARKER_RE.search(stripped)
    if match is None:
        return None
    prefix = stripped[: match.start(1)]
    # Whole chunk is just the marker (or leading whitespace only) -> never
    # empty a chunk.
    if not prefix.strip():
        return None
    # The whitespace run separating the prefix body from the marker.
    trailing_ws = prefix[len(prefix.rstrip()):]
    prefix_body = prefix.rstrip()
    # Standalone iff a newline sits between the body and the marker, OR the
    # body ends in sentence-terminal punctuation (with the intervening
    # whitespace the regex + this slice already require).
    if "\n" in trailing_ws:
        return match.group(1)
    if trailing_ws and prefix_body.endswith(_SENTENCE_TERMINALS):
        return match.group(1)
    return None


def _module_id(chunk: Dict[str, Any]) -> Any:
    source = chunk.get("source")
    if isinstance(source, dict):
        return source.get("module_id")
    return None


def _same_flow(cur: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    """Is ``nxt`` in the same document flow as ``cur`` (its immediate successor)?

    Accepts when ``nxt.follows_chunk == cur.id`` (the canonical linkage) OR
    when both chunks carry an equal ``source.module_id`` (``nxt`` is already
    the immediate list successor by construction). A cross-module boundary
    (differing / absent module ids AND no ``follows_chunk`` linkage) is
    rejected so a marker never leaks across a module seam.
    """
    cur_id = cur.get("id")
    follows = nxt.get("follows_chunk")
    if cur_id is not None and follows is not None and follows == cur_id:
        return True
    cur_mod = _module_id(cur)
    nxt_mod = _module_id(nxt)
    if cur_mod is not None and nxt_mod is not None and cur_mod == nxt_mod:
        return True
    return False


def _recount(chunk: Dict[str, Any]) -> None:
    """Recompute ``word_count`` / ``tokens_estimate`` from ``text`` in place.

    Matches ``_create_chunk``'s formula: ``tokens_estimate = int(word_count
    * 1.3)``. Only keys already present on the chunk are updated (parity with
    ``apply_chunk_overlap``).
    """
    new_word_count = len(str(chunk.get("text", "")).split())
    if "word_count" in chunk:
        chunk["word_count"] = new_word_count
    if "tokens_estimate" in chunk:
        chunk["tokens_estimate"] = int(new_word_count * 1.3)


def relocate_stranded_heading_tails(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Move stranded next-section heading markers from a chunk tail to the next.

    For each chunk whose right-stripped ``text`` ends with a standalone
    ``N.M <SECTION-MARKER>`` (per ``_match_stranded_tail``) AND which has an
    eligible SAME-FLOW following chunk (per ``_same_flow``): strip the marker
    from the source chunk's tail and PREPEND ``"<marker>\\n"`` to the
    following chunk's ``text`` — unless the following chunk already STARTS
    with that same marker, in which case the tail is stripped WITHOUT
    duplicating the marker.

    Text-only mutation (``text`` + recomputed ``word_count`` /
    ``tokens_estimate``); ``html``, ``id``, ``follows_chunk``, ``source``,
    and every other field are untouched — the documented ``apply_chunk_overlap``
    precedent. In place; the same list is returned.

    Idempotent: after relocation the source tail no longer matches, so a
    second call is a no-op. Fewer than two chunks -> no-op.
    """
    if len(chunks) < 2:
        return chunks

    for i in range(len(chunks) - 1):
        cur = chunks[i]
        nxt = chunks[i + 1]
        cur_text = str(cur.get("text", ""))
        marker = _match_stranded_tail(cur_text)
        if marker is None:
            continue
        if not _same_flow(cur, nxt):
            continue

        # Strip the marker (and its trailing whitespace) from the source tail.
        stripped = cur_text.rstrip()
        source_body = stripped[: _TAIL_MARKER_RE.search(stripped).start(1)]
        new_source_text = source_body.rstrip()
        # Guard: never empty the source chunk (belt-and-suspenders with the
        # _match_stranded_tail prefix check).
        if not new_source_text:
            continue
        cur["text"] = new_source_text
        _recount(cur)

        # Prepend to the following chunk unless it already opens with the
        # marker (avoid a duplicate heading).
        nxt_text = str(nxt.get("text", ""))
        if nxt_text.lstrip().startswith(marker):
            continue
        nxt["text"] = f"{marker}\n{nxt_text}" if nxt_text else marker
        _recount(nxt)

    return chunks
