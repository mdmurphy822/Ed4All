"""Deterministic HTML tag-balance repair for rewrite-tier block fragments.

The rewrite tier's ``rewrite_html_shape`` gate
(:mod:`lib.validators.rewrite_html_shape`) parses every emitted block HTML
fragment with a stdlib :class:`html.parser.HTMLParser` open-stack and fails the
block (critical ``REWRITE_HTML_PARSE_FAIL``) when

* an end tag closes NOTHING currently open (stray close),
* an end tag matches a tag DEEPER in the stack (the intermediates are popped
  and the block is marked unbalanced), or
* the open-stack is non-empty after the parse (missing closes).

Model-emitted fragments hit all three (observed live: an extra ``</h3>`` /
``</p>`` closing nothing, ``</div>`` closing over a still-open ``<span>``,
and unclosed wrapper ``<div>`` / ``<section>`` elements).

:func:`repair_html_balance` is the deterministic, single-pass, LLM-free
repair that makes such a fragment satisfy EXACTLY the gate's stack rules:

1. **Stray close tags are DROPPED** (removed byte-for-byte; nothing else on
   that line moves).
2. **Mid-stack closes get the missing intermediate closes INSERTED** directly
   before them, in proper LIFO order (``<div><span>x</div>`` →
   ``<div><span>x</span></div>``), because the gate marks a pop-over-
   intermediates unbalanced even when the final stack empties.
3. **Missing closes are APPENDED at fragment end** in LIFO order. When the
   fragment ends with the postmint's hidden ``<span data-cf-curie>`` tail
   (legal trailing fragment elements appended AFTER the block's closing
   wrapper), the appended closes are inserted BEFORE that tail so the curie
   spans stay top-level siblings.

Guarantees:

* Text nodes, attributes, comments, entity refs and tag spelling are NEVER
  edited or reordered — the only edits are close-tag removals and close-tag
  insertions.
* Void elements (:data:`VOID_TAGS`, kept in lockstep with the gate's
  ``_VOID_TAGS``) are never pushed, never closed, never touched — including
  the browser-tolerated ``</br>`` form the gate ignores.
* Self-closing ``<tag />`` forms are treated exactly as the gate treats them
  (not pushed; untouched).
* Deterministic and idempotent: ``repair(repair(x)) == repair(x)``; a
  fragment the gate already considers balanced round-trips byte-identical
  with zero ops.

Standalone runner (writes a REPAIRED copy of a rewrite ``blocks_final.jsonl``
plus a ``repair_report.json`` — never modifies the input in place)::

    python -m lib.utils.html_balance <blocks_final.jsonl> --out <dir>

Pipeline wire-in: ``MCP/tools/pipeline_tools.py`` applies this pass at
rewrite-emit time behind ``COURSEFORGE_REWRITE_HTML_REPAIR`` (default OFF →
byte-identical emit; see :func:`resolve_rewrite_html_repair`).
"""
from __future__ import annotations

import argparse
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "VOID_TAGS",
    "repair_html_balance",
    "repair_jsonl",
    "resolve_rewrite_html_repair",
    "summarize_ops",
]

# HTML void elements — MUST stay in lockstep with
# ``lib.validators.rewrite_html_shape._VOID_TAGS`` (the gate's set defines
# "balanced" for this repair). A lockstep regression test asserts equality
# (``lib/utils/tests/test_html_balance.py::test_void_tags_lockstep_with_gate``).
VOID_TAGS: frozenset = frozenset(
    {
        "br", "img", "hr", "input", "meta", "link", "area",
        "base", "col", "embed", "param", "source", "track", "wbr",
    }
)

#: Env flag gating the rewrite-emit wire-in. Default OFF (parse-with-fallback:
#: only the canonical truthy tokens enable), mirroring COURSEFORGE_TWO_PASS.
ENV_REWRITE_HTML_REPAIR = "COURSEFORGE_REWRITE_HTML_REPAIR"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_rewrite_html_repair() -> bool:
    """``COURSEFORGE_REWRITE_HTML_REPAIR`` truthy → apply repair at emit.

    Read each call so tests can toggle it (mirrors the two-pass resolver in
    ``MCP/tools/pipeline_tools.py``).
    """
    return (
        os.environ.get(ENV_REWRITE_HTML_REPAIR, "").strip().lower() in _TRUTHY
    )


class _RepairParser(HTMLParser):
    """Mirror of the gate's ``_ShapeParser`` stack discipline that records
    source-anchored repair edits instead of an unbalanced verdict.

    Feed the ENTIRE fragment in one ``feed()`` call (position bookkeeping
    relies on it). ``getpos()`` inside ``handle_endtag`` is the position of
    the ``<`` of that end tag — verified against CPython's ``goahead`` loop
    (``updatepos`` runs before the tag branch dispatches).
    """

    def __init__(self, src: str) -> None:
        super().__init__(convert_charrefs=True)
        self._src = src
        # Absolute offset of the start of each 1-based line.
        self._line_starts: List[int] = [0]
        for i, ch in enumerate(src):
            if ch == "\n":
                self._line_starts.append(i + 1)
        self.stack: List[str] = []
        # (start, end, replacement) — drops replace a span with "", inserts
        # replace an empty span with close-tag text.
        self.edits: List[Tuple[int, int, str]] = []
        self.ops: List[Dict[str, Any]] = []
        # Source spans of every consumed start-tag token. A MALFORMED start
        # tag (unterminated attribute quote) can swallow kilobytes of later
        # markup into one token; an insert position inside such a span would
        # be invisible to the parser, so append positions are validated
        # against these spans.
        self.token_spans: List[Tuple[int, int]] = []

    # -- position helpers -------------------------------------------------
    def _abs_pos(self) -> int:
        line, col = self.getpos()
        try:
            return self._line_starts[line - 1] + col
        except IndexError:  # pragma: no cover — defensive; cannot happen on a single feed
            return len(self._src)

    # -- gate-mirroring handlers ------------------------------------------
    def _record_token_span(self) -> None:
        raw = self.get_starttag_text()
        if raw:
            pos = self._abs_pos()
            self.token_spans.append((pos, pos + len(raw)))

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        self._record_token_span()
        if tag.lower() not in VOID_TAGS:
            self.stack.append(tag.lower())

    def handle_startendtag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        # Self-closing form ``<tag />`` — the gate never pushes it. Override
        # the stdlib default (which would call handle_starttag + handle_endtag)
        # so the repair never touches it either.
        self._record_token_span()
        return

    def safe_append_pos(self, candidate: int) -> int:
        """Clamp an append position that falls INSIDE a consumed start-tag
        token (malformed-quote swallow) to the true end of the source."""
        for start, end in self.token_spans:
            if start < candidate < end:
                return len(self._src)
        return candidate

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in VOID_TAGS:
            # Spec-violating end tag for a void element — the gate ignores it
            # (browser-tolerated), so the repair leaves it untouched.
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
            return
        pos = self._abs_pos()
        if tag in self.stack:
            # Mid-stack close: the gate pops the intermediates AND marks the
            # block unbalanced. Repair: close the intermediates (LIFO) right
            # before this close tag so every pop matches top-of-stack.
            closes: List[str] = []
            while self.stack and self.stack[-1] != tag:
                inner = self.stack.pop()
                closes.append(inner)
                self.ops.append(
                    {"op": "insert_close", "tag": inner, "pos": pos}
                )
            if self.stack:
                self.stack.pop()
            self.edits.append(
                (pos, pos, "".join(f"</{t}>" for t in closes))
            )
            return
        # Stray close (nothing matching open) — drop the tag token itself.
        # The gate's end-tag token runs from '<' to the next '>' inclusive
        # (stdlib ``endendtag`` regex), so mirror that span.
        if not self._src.startswith("</", pos):
            # Position bookkeeping failed to land on an end tag (should be
            # impossible on a single-feed parse). NEVER risk deleting
            # non-tag bytes — record the anomaly and leave the source alone.
            self.ops.append(
                {"op": "skip_unlocatable_stray", "tag": tag, "pos": pos}
            )
            return
        gt = self._src.find(">", pos)
        end = (gt + 1) if gt != -1 else len(self._src)
        self.edits.append((pos, end, ""))
        self.ops.append(
            {
                "op": "drop_stray_close",
                "tag": tag,
                "pos": pos,
                "raw": self._src[pos:end],
            }
        )


def _curie_tail_start(src: str) -> int:
    """Offset where the trailing hidden ``data-cf-curie`` span run begins.

    The CURIE postmint appends hidden ``<span data-cf-curie=...>...</span>``
    elements AFTER the block's closing wrapper — legal top-level fragment
    siblings. Missing closes are appended BEFORE that tail so the curie spans
    are never swallowed into the repaired wrapper. Returns ``len(src)`` when
    no such balanced tail exists (an UNCLOSED trailing curie span is not a
    tail — it is itself a missing-close defect and closes append at the very
    end).

    Deterministic backward scan (no regex backtracking): repeatedly peel a
    trailing ``<span ...data-cf-curie...>...</span>`` (with optional
    surrounding whitespace) off the end.
    """
    end = len(src)
    while True:
        seg_end = end
        while seg_end > 0 and src[seg_end - 1].isspace():
            seg_end -= 1
        if not src[:seg_end].endswith("</span>"):
            return end
        open_start = src.rfind("<span", 0, seg_end - len("</span>"))
        if open_start == -1:
            return end
        open_gt = src.find(">", open_start)
        if open_gt == -1 or open_gt >= seg_end - len("</span>"):
            return end
        open_tag = src[open_start : open_gt + 1]
        inner = src[open_gt + 1 : seg_end - len("</span>")]
        # The tail span must carry the curie marker and contain no nested
        # tags (postmint emits text-only spans) — anything else is regular
        # content, not the postmint tail.
        if "data-cf-curie" not in open_tag or "<" in inner:
            return end
        # Peel this span (and any whitespace between it and the previous one).
        end = open_start
        while end > 0 and src[end - 1].isspace():
            end -= 1


def _repair_once(fragment: str) -> Tuple[str, List[Dict[str, Any]]]:
    """One parse-and-edit pass of the balance repair (see the public API)."""
    parser = _RepairParser(fragment)
    try:
        parser.feed(fragment)
        parser.close()
    except Exception:  # noqa: BLE001 — mirror the gate: an unparseable
        # fragment fails REWRITE_HTML_PARSE_FAIL there; the repair must never
        # produce a partial edit set for it. Leave it byte-identical.
        return fragment, []

    edits = list(parser.edits)
    ops = list(parser.ops)

    if parser.stack:
        # Missing closes — append in LIFO order, before the curie tail.
        # ``safe_append_pos`` guards against a textual tail start that the
        # parser consumed inside a malformed start-tag token (the closes
        # would be swallowed and invisible → the repair could never
        # converge); such positions clamp to the absolute end.
        tail_pos = parser.safe_append_pos(_curie_tail_start(fragment))
        closes = list(reversed(parser.stack))
        edits.append(
            (tail_pos, tail_pos, "".join(f"</{t}>" for t in closes))
        )
        for t in closes:
            ops.append({"op": "append_close", "tag": t, "pos": tail_pos})

    if not edits:
        return fragment, ops

    # Apply edits ascending; spans never overlap (each drop covers one
    # end-tag token; each insert is a point between tokens).
    edits.sort(key=lambda e: (e[0], e[1]))
    out: List[str] = []
    cursor = 0
    for start, end, replacement in edits:
        if start < cursor:  # pragma: no cover — defensive; spans are disjoint
            continue
        out.append(fragment[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(fragment[cursor:])
    return "".join(out), ops


#: Fixed-point iteration cap. A well-formed-but-unbalanced fragment converges
#: in one pass; fragments carrying MALFORMED start tags (e.g. an unterminated
#: attribute quote like ``<div class="key-rule>``) make the stdlib parser
#: swallow later markup into the attribute value, so a pass's edits can expose
#: previously-swallowed tags and require another pass. Real-corpus worst case
#: observed: 2 passes.
_MAX_PASSES = 10


def repair_html_balance(
    fragment: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Repair tag balance of an HTML body fragment against the gate's rules.

    Returns ``(repaired, ops)``. ``ops`` is a list of dicts, each one of:

    * ``{"op": "drop_stray_close", "tag", "pos", "raw"}`` — a close tag that
      matched nothing open was removed (``raw`` is the removed token).
    * ``{"op": "insert_close", "tag", "pos"}`` — a missing intermediate close
      was inserted at ``pos`` (immediately before a mid-stack close).
    * ``{"op": "append_close", "tag", "pos"}`` — a missing close was appended
      at fragment end (before the trailing hidden curie-span tail, if any).
    * ``{"op": "skip_unlocatable_stray", ...}`` — defensive: a stray close
      whose source token could not be located was left in place.
    * ``{"op": "unconverged", "passes"}`` — the fragment did not reach a
      fixed point within :data:`_MAX_PASSES`; the ORIGINAL fragment is
      returned byte-identical (loud honest failure at the gate — never a
      partial repair).

    ``pos`` values in multi-pass repairs refer to the intermediate string of
    the pass that recorded the op (``pass`` key present from pass 2 on).

    Empty ``ops`` ⇒ ``repaired == fragment`` (byte-identical). Deterministic
    (pure function of the input) and idempotent: the returned string is a
    fixed point, so ``repair(repair(x)) == repair(x)``.
    """
    if not isinstance(fragment, str) or not fragment.strip():
        return fragment, []

    current = fragment
    all_ops: List[Dict[str, Any]] = []
    for n_pass in range(1, _MAX_PASSES + 1):
        repaired, ops = _repair_once(current)
        actionable = [o for o in ops if o["op"] != "skip_unlocatable_stray"]
        if not actionable:
            # Fixed point (skip-markers are informational; the source was
            # not modified for them).
            if n_pass == 1:
                all_ops.extend(ops)
            return current, all_ops
        if n_pass > 1:
            for o in ops:
                o["pass"] = n_pass
        all_ops.extend(ops)
        current = repaired
    # Did not converge — never ship a partial repair. Return the original
    # untouched so the gate fails it honestly (and a re-repair of that
    # original is byte-identical → idempotent).
    return fragment, [{"op": "unconverged", "passes": _MAX_PASSES}]


def summarize_ops(ops: List[Dict[str, Any]]) -> str:
    """Compact ``op=count`` summary (deterministic order) for audit fields."""
    counts: Dict[str, int] = {}
    for op in ops:
        counts[op["op"]] = counts.get(op["op"], 0) + 1
    return ";".join(f"{k}={v}" for k, v in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def repair_jsonl(
    blocks_path: Path, out_dir: Path
) -> Dict[str, Any]:
    """Repair every ``content`` field of a rewrite ``blocks_final.jsonl``.

    Writes ``<out_dir>/blocks_final.repaired.jsonl`` (same schema, only
    ``content`` changed) plus ``<out_dir>/repair_report.json``. The input
    file is NEVER modified. Returns the report dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = out_dir / "blocks_final.repaired.jsonl"
    report_path = out_dir / "repair_report.json"

    per_block: List[Dict[str, Any]] = []
    op_counts: Dict[str, int] = {}
    n_repaired = 0
    n_untouched = 0
    n_nonstring = 0
    rows: List[Dict[str, Any]] = []

    with blocks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    with repaired_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            content = row.get("content")
            if not isinstance(content, str):
                n_nonstring += 1
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")
                continue
            repaired, ops = repair_html_balance(content)
            if ops:
                n_repaired += 1
                for op in ops:
                    op_counts[op["op"]] = op_counts.get(op["op"], 0) + 1
                per_block.append(
                    {
                        "block_id": row.get("block_id"),
                        "block_type": row.get("block_type"),
                        "ops": ops,
                    }
                )
                row = dict(row)
                row["content"] = repaired
            else:
                n_untouched += 1
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    report: Dict[str, Any] = {
        "input": str(blocks_path),
        "output": str(repaired_path),
        "total_blocks": len(rows),
        "repaired": n_repaired,
        "untouched": n_untouched,
        "non_string_content": n_nonstring,
        "op_counts": op_counts,
        "blocks": per_block,
    }

    # Best-effort before/after gate probe with the ACTUAL rewrite_html_shape
    # validator (no decision capture → no side effects). Failure to import /
    # run the gate never fails the repair run.
    try:
        report["gate"] = _gate_probe(rows, repaired_path)
    except Exception as exc:  # noqa: BLE001
        report["gate"] = {"error": f"{type(exc).__name__}: {exc}"}

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _gate_probe(
    original_rows: List[Dict[str, Any]], repaired_path: Path
) -> Dict[str, Any]:
    """Run ``RewriteHtmlShapeValidator`` over original + repaired rows."""
    from types import SimpleNamespace

    from lib.validators.rewrite_html_shape import RewriteHtmlShapeValidator

    def _score(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        blocks = [SimpleNamespace(**row) for row in rows]
        result = RewriteHtmlShapeValidator().validate({"blocks": blocks})
        return {
            "passed": result.passed,
            "score": result.score,
            "critical_issues": sum(
                1 for i in result.issues if i.severity == "critical"
            ),
            "issue_codes": sorted(
                {i.code for i in result.issues if i.severity == "critical"}
            ),
        }

    repaired_rows = [
        json.loads(line)
        for line in repaired_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "validator": "rewrite_html_shape",
        "before": _score(original_rows),
        "after": _score(repaired_rows),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lib.utils.html_balance",
        description=(
            "Deterministically repair HTML tag balance in a rewrite-tier "
            "blocks_final.jsonl (writes a repaired COPY + repair_report.json; "
            "never modifies the input)."
        ),
    )
    parser.add_argument(
        "blocks_jsonl", type=Path,
        help="Path to the rewrite-tier blocks_final.jsonl to repair.",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help=(
            "Output DIRECTORY for blocks_final.repaired.jsonl + "
            "repair_report.json."
        ),
    )
    args = parser.parse_args(argv)

    if not args.blocks_jsonl.is_file():
        parser.error(f"input not found: {args.blocks_jsonl}")
    if args.out.resolve() == args.blocks_jsonl.parent.resolve():
        parser.error(
            "--out must not be the input file's own directory "
            "(the input is never modified in place; write the repaired "
            "copy elsewhere)."
        )

    report = repair_jsonl(args.blocks_jsonl, args.out)
    gate = report.get("gate", {})
    print(
        json.dumps(
            {
                "total_blocks": report["total_blocks"],
                "repaired": report["repaired"],
                "untouched": report["untouched"],
                "op_counts": report["op_counts"],
                "gate_before": gate.get("before"),
                "gate_after": gate.get("after"),
                "report": str(args.out / "repair_report.json"),
            },
            indent=2,
        )
    )
    print(f"repaired jsonl: {report['output']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
