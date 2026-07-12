"""Pre-synthesis chunk + structure HEALTH gate (``ChunkHealthValidator``).

Ports the standalone ``chunk_health_preflight.py`` prototype into a real
validation gate so no ``textbook_to_course`` run ever synthesizes objectives /
content from a *poisoned* chunkset. The gate samples the SemantiK-produced
``textbook_structure.json`` + the emitted ``chunks.jsonl`` the course is about to
be synthesized from, and flags defect classes that quietly wreck synthesis
quality (phantom chapters from a collapsed structure, an apparatus-dominated
chunkset with no instructional prose to ground objectives in, OCR mojibake,
empty chunks, ...). Wired AFTER ``chunking`` + ``objective_extraction`` and
BEFORE the expensive ``course_planning`` / ``content_generation`` synthesis so a
bad chunkset blocks the GPU-hours instead of poisoning them.

Domain-agnostic heuristics (no corpus-specific vocab / slug hardcoding — see the
``feedback_semantik_wide_net`` principle): every threshold is a ratio / rate /
count derivable from any textbook corpus, and every one is env-overridable via
an ``ED4ALL_CHUNK_HEALTH_*`` knob (parse-with-fallback — garbage falls back to
the documented default, never crashes).

**Opt-in gate (default OFF).** The gate is gated behind
``ED4ALL_CHUNK_HEALTH_GATE``; unset / falsey → the validator returns a no-op
``passed=True`` *before touching any input* (byte-identical to pre-gate runs,
preserving backward-compat with legacy corpora). Set truthy
(``1`` / ``true`` / ``yes`` / ``on``) to enforce.

Severity contract (per the gate spec):

* **Critical** (blocks synthesis) — the structural-collapse + starvation
  defects that make synthesis grounding impossible:
  ``CHUNK_HEALTH_CHAPTER_EXPLOSION`` (S1),
  ``CHUNK_HEALTH_ARBITRARY_RESEGMENT`` (S2),
  ``CHUNK_HEALTH_SECTION_EXPLOSION`` (S3),
  ``CHUNK_HEALTH_INSTRUCTIONAL_STARVED`` (C2),
  ``CHUNK_HEALTH_EMPTY_CHUNKS`` (C7),
  plus the input-integrity criticals ``CHUNK_HEALTH_CHUNKS_NOT_FOUND`` /
  ``CHUNK_HEALTH_CHUNKS_UNREADABLE`` (a missing / unparseable chunkset with the
  gate ON is itself a fail-closed condition — never synthesize from nothing).
* **Warning** (logged, non-blocking) — everything else (S1/S3 near-threshold
  bands, S4 apparatus headings, S5 furniture titles, C1 thin chunkset, C3
  degenerate/mega chunks, C4 mojibake, C5 running-header bleed, C6 provenance
  furniture, C8 answer-key contamination, C9 numeric dumps, C10 reading-order
  scramble).

Inputs contract:

* ``inputs["chunks_path"]`` — path to the ``chunks.jsonl`` (required when the
  gate is ON). The gate-input builder
  (``MCP/hardening/gate_input_routing.py::_build_chunk_health``) resolves it from
  ``phase_outputs.chunking.semantik_chunks_path`` (legacy
  ``dart_chunks_path``).
* ``inputs["textbook_structure_path"]`` / ``inputs["textbook_structure"]`` —
  the structure to audit (optional; the S-class structure checks skip when
  absent — a chunk-only import legitimately lacks it).
* ``inputs["decision_capture"]`` — optional ``DecisionCapture``; the validator
  emits exactly one ``content_structure_check`` event per run (an EXISTING enum
  member — validators do not mint bespoke decision_types, see
  ``docs/architecture/decision-capture.md``).
* ``inputs["gate_id"]`` — optional GateResult ID override.

Cross-references:

* ``lib/validators/chunkset_manifest.py`` — sibling chunkset gate (SHA / count /
  coverage integrity); this gate is the CONTENT-quality complement.
* ``lib/validators/textbook_structure.py`` — the ``objective_extraction`` gate
  this one sits beside; both audit ``textbook_structure.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Domain-agnostic lexical heuristics (mirror the prototype verbatim).
# --------------------------------------------------------------------------- #

#: Apparatus-heading prefix: a section/title that is drill scaffolding
#: ("Example 3", "Try It 1.4", "Be Prepared", "Exercises", "Solution", "Answers",
#: "Key") rather than instructional prose. Not subject-specific.
_APPARATUS = re.compile(
    r"^\s*(example|try it|be prepared|exercise|solution|answers?|key)\b", re.I
)

#: Running-header / page-number furniture in a title, e.g. "Chapter 1 Foundations 83".
_FURNITURE = re.compile(r"\b(chapter\s+\d+)\b.*\b\d{2,4}\b|\bpage\s+\d+\b", re.I)

#: OCR mojibake / PDF CID markers / U+FFFD replacement / NUL.
_MOJIBAKE = re.compile(r"\(cid:\d+\)|�|â€|Ã[©¨¤]|\x00")

#: An "N.M" section ordinal, e.g. "3.2".
_NM_SECTION = re.compile(r"^\s*\d+\.\d+\b")

#: chunk_type values that carry teachable prose (as opposed to example /
#: exercise / drill apparatus). Synthesis grounds objectives in prose.
_INSTRUCTIONAL_TYPES = frozenset(
    {"explanation", "overview", "concept", "definition", "objective", "introduction"}
)

#: chunk_type values that MAY be worked examples. A worked example (an example
#: chunk that carries a worked solution) TEACHES — it walks a learner through a
#: solved instance step by step — so it counts as instructional (C2) when it
#: contains a solution marker. A BARE example (label only, no solution) does
#: not. Worked-example-driven workbooks (e.g. algebra) are legitimately
#: example-heavy; treating every example as non-instructional false-blocks them.
_WORKED_EXAMPLE_TYPES = frozenset({"example", "worked_example"})

#: Generic (domain-agnostic) worked-solution / guided-practice marker: a
#: "Solution", "Try It", "Step N", "worked example", or "How To" cue. These are
#: pedagogical apparatus labels common to any subject's worked-example textbook,
#: NOT subject-specific vocabulary. Presence of one in an example chunk means it
#: walks through a solved instance (teaches) rather than posing a bare problem.
_WORKED_SOLUTION_MARKER = re.compile(
    r"(?:✓\s*)?\bsolution\b"
    r"|\btry\s+it\b"
    r"|\bstep\s+\d"
    r"|\bworked\s+example\b"
    r"|\bhow\s+to\b",
    re.I,
)

#: chunk_type values that are drill / navigation apparatus, whose internal N.M
#: numbering (exercise / example numbers, page/TOC references) is legitimately
#: non-monotonic and MUST NOT count toward the C10 reading-order scramble check
#: (which only means something for continuous instructional PROSE).
_APPARATUS_CHUNK_TYPES = frozenset(
    {
        "example",
        "worked_example",
        "exercise",
        "answer_key",
        "answer-key",
        "answerkey",
        "toc",
        "table_of_contents",
    }
)

#: Validator-capture decision_type — an EXISTING enum member (validators do not
#: mint bespoke types, see docs/architecture/decision-capture.md).
_DECISION_TYPE = "content_structure_check"

#: Cap emitted issues so a uniformly-broken corpus doesn't drown the report.
_ISSUE_LIST_CAP = 60

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off", ""})


# --------------------------------------------------------------------------- #
# Env resolvers (parse-with-fallback — garbage → documented default).
# --------------------------------------------------------------------------- #


def resolve_chunk_health_gate_enabled() -> bool:
    """Return True iff ``ED4ALL_CHUNK_HEALTH_GATE`` is truthy (default OFF).

    Read at call time (not import) so tests / operators can toggle per-run.
    Only the explicit truthy tokens enable it; unset / empty / garbage → OFF,
    making the whole gate a byte-identical no-op.
    """
    raw = os.environ.get("ED4ALL_CHUNK_HEALTH_GATE", "").strip().lower()
    return raw in _TRUTHY


def resolve_worked_example_instructional() -> bool:
    """Return True iff worked examples count as instructional for C2 (default ON).

    A worked example (an ``example`` / ``worked_example`` chunk carrying a
    solution marker) TEACHES, so by default it is counted toward the C2
    instructional share — this is what keeps a legitimately example-heavy
    worked-example workbook (e.g. an algebra textbook that is ~80% worked
    examples/exercises) from false-blocking on ``CHUNK_HEALTH_INSTRUCTIONAL_STARVED``.

    Read at call time (not import). Default ON; only an explicit falsey token
    (``0`` / ``false`` / ``no`` / ``off``) reverts to the pre-fix behaviour of
    treating every example as non-instructional apparatus.
    """
    raw = os.environ.get("ED4ALL_CHUNK_HEALTH_WORKED_EXAMPLE_INSTRUCTIONAL")
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY


def _has_worked_solution(row: Mapping[str, Any]) -> bool:
    """True when a chunk contains a generic worked-solution / guided-practice marker.

    Scans the chunk text + html for a domain-agnostic ``Solution`` / ``Try It``
    / ``Step N`` / ``worked example`` / ``How To`` cue (see
    ``_WORKED_SOLUTION_MARKER``). Used to distinguish a WORKED example (teaches
    — counts as instructional) from a BARE exercise/problem (does not).
    """
    blob = (row.get("text") or "") + " " + (row.get("html") or "")
    return bool(_WORKED_SOLUTION_MARKER.search(blob))


def _resolve_float(env_name: str, default: float) -> float:
    """Parse a float threshold env; non-float / unset → ``default``."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    if val <= 0.0:
        return default
    return val


def _resolve_int(env_name: str, default: int) -> int:
    """Parse an int threshold env; non-int / unset → ``default``."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    if val <= 0:
        return default
    return val


class _Thresholds:
    """Snapshot of every env-overridable threshold, resolved once per run."""

    def __init__(self) -> None:
        # S1 chapter/source ratios.
        self.chapter_ratio_fail = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_CHAPTER_RATIO_FAIL", 3.0
        )
        self.chapter_ratio_warn = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_CHAPTER_RATIO_WARN", 1.5
        )
        # S3 sections-per-source expectation (drives the explosion multiples).
        self.sections_per_chapter = _resolve_int(
            "ED4ALL_CHUNK_HEALTH_SECTIONS_PER_CHAPTER", 15
        )
        # S4 apparatus-heading rate bands.
        self.apparatus_fail = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_APPARATUS_FAIL", 0.30
        )
        self.apparatus_warn = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_APPARATUS_WARN", 0.10
        )
        # C1 thin chunkset floor.
        self.min_chunks = _resolve_int("ED4ALL_CHUNK_HEALTH_MIN_CHUNKS", 30)
        # C2 instructional-share bands (the KEY synthesis-quality signal).
        self.instructional_fail = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_INSTRUCTIONAL_FAIL", 0.20
        )
        self.instructional_warn = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_INSTRUCTIONAL_WARN", 0.40
        )
        # C3 size bounds.
        self.tiny_words = _resolve_int("ED4ALL_CHUNK_HEALTH_TINY_WORDS", 20)
        self.mega_words = _resolve_int("ED4ALL_CHUNK_HEALTH_MEGA_WORDS", 1500)
        # C4 mojibake rate.
        self.mojibake_rate = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_MOJIBAKE_RATE", 0.10
        )
        # C9 numeric-dump rate.
        self.numdump_rate = _resolve_float(
            "ED4ALL_CHUNK_HEALTH_NUMDUMP_RATE", 0.15
        )


# --------------------------------------------------------------------------- #
# Decision capture
# --------------------------------------------------------------------------- #


def _emit_decision(
    capture: Any,
    *,
    decision: str,
    rationale: str,
    context: Optional[str] = None,
) -> None:
    """Emit one DecisionCapture event, swallowing capture-side errors.

    Capture is observability, not control flow — a capture failure must not
    break the gate result.
    """
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.warning(
            "ChunkHealthValidator: decision_capture.log_decision failed: %s",
            exc,
        )


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _coerce_structure(
    inputs: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve the textbook_structure dict from ``inputs`` (optional).

    Priority: ``inputs['textbook_structure']`` (in-memory) >
    ``inputs['textbook_structure_path']`` (on-disk). Returns
    ``(structure, error_issue)``; a MISSING structure is NOT an error (the
    S-class checks skip) — only a present-but-unparseable one flags a warning.
    """
    explicit = inputs.get("textbook_structure")
    if isinstance(explicit, Mapping):
        return dict(explicit), None

    path_raw = inputs.get("textbook_structure_path")
    if not path_raw:
        return None, None
    p = Path(path_raw)
    if not p.exists():
        return None, GateIssue(
            severity="warning",
            code="CHUNK_HEALTH_STRUCTURE_NOT_FOUND",
            message=(
                f"textbook_structure_path {str(p)!r} does not exist; "
                f"skipping the structure (S-class) health checks."
            ),
            location=str(p),
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, GateIssue(
            severity="warning",
            code="CHUNK_HEALTH_STRUCTURE_UNREADABLE",
            message=(
                f"Failed to parse textbook_structure_path {str(p)!r}: "
                f"{exc.__class__.__name__}: {exc}; skipping structure checks."
            ),
            location=str(p),
        )
    if not isinstance(data, dict):
        return None, GateIssue(
            severity="warning",
            code="CHUNK_HEALTH_STRUCTURE_MALFORMED",
            message=(
                f"textbook_structure_path {str(p)!r} did not parse to a JSON "
                f"object (got {type(data).__name__}); skipping structure checks."
            ),
            location=str(p),
        )
    return data, None


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class ChunkHealthValidator:
    """Pre-synthesis chunk + structure health gate.

    Opt-in (``ED4ALL_CHUNK_HEALTH_GATE``, default OFF); skip-with-pass when the
    flag is unset so legacy corpora / default runs are byte-identical. When ON,
    blocks synthesis on the structural-collapse + instructional-starvation
    defect classes (critical) while surfacing the softer OCR / furniture /
    reading-order signals as warnings.
    """

    name = "chunk_health"
    version = "0.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")

        # ---- 0. Opt-in gate. OFF → byte-identical no-op pass. Read FIRST,
        # before any input is touched, so default-off runs never fail even if
        # inputs are absent.
        if not resolve_chunk_health_gate_enabled():
            _emit_decision(
                capture,
                decision="skip-with-pass: ED4ALL_CHUNK_HEALTH_GATE off",
                rationale=(
                    "ChunkHealthValidator is opt-in via ED4ALL_CHUNK_HEALTH_GATE "
                    "and the flag is unset/falsey; returning a no-op pass so "
                    "default runs and legacy corpora are byte-identical to the "
                    "pre-gate behaviour."
                ),
                context="gate_enabled=false",
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        th = _Thresholds()
        issues: List[GateIssue] = []

        # ---- 1. Chunks required (gate ON). Missing / unreadable → fail-closed.
        chunks_path_raw = inputs.get("chunks_path")
        if not chunks_path_raw:
            issue = GateIssue(
                severity="critical",
                code="CHUNK_HEALTH_CHUNKS_NOT_FOUND",
                message=(
                    "ChunkHealthValidator requires inputs['chunks_path'] when "
                    "the gate is enabled; no chunkset path was resolved — "
                    "refusing to synthesize from an absent chunkset."
                ),
            )
            return self._fail_closed(gate_id, [issue], capture, chunks_n=0)

        chunks_path = Path(chunks_path_raw)
        if not chunks_path.exists() or not chunks_path.is_file():
            issue = GateIssue(
                severity="critical",
                code="CHUNK_HEALTH_CHUNKS_NOT_FOUND",
                message=(
                    f"chunkset not found at {chunks_path}; refusing to "
                    f"synthesize from an absent chunkset."
                ),
                location=str(chunks_path),
            )
            return self._fail_closed(gate_id, [issue], capture, chunks_n=0)

        try:
            rows = _load_jsonl(chunks_path)
        except OSError as exc:
            issue = GateIssue(
                severity="critical",
                code="CHUNK_HEALTH_CHUNKS_UNREADABLE",
                message=(
                    f"failed to read chunkset {chunks_path}: "
                    f"{exc.__class__.__name__}: {exc}."
                ),
                location=str(chunks_path),
            )
            return self._fail_closed(gate_id, [issue], capture, chunks_n=0)

        # ---- 2. Structure (optional).
        structure, struct_err = _coerce_structure(inputs)
        if struct_err is not None:
            issues.append(struct_err)
        if structure is not None:
            issues.extend(self._audit_structure(structure, th))

        # ---- 3. Chunks.
        issues.extend(self._audit_chunks(rows, th))

        issues = issues[:_ISSUE_LIST_CAP]
        critical_count = sum(1 for i in issues if i.severity == "critical")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        passed = critical_count == 0
        action: Optional[str] = "block" if not passed else None
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        verdict = (
            "NOT SYNTHESIS-READY"
            if not passed
            else ("SYNTHESIS-READY (with warnings)" if warning_count else "SYNTHESIS-READY")
        )
        _emit_decision(
            capture,
            decision=(
                f"{verdict}: {len(rows)} chunk(s); "
                f"{critical_count} critical, {warning_count} warning issue(s)."
            ),
            rationale=(
                f"ChunkHealthValidator audited the pre-synthesis chunkset "
                f"({chunks_path.name}, {len(rows)} chunks) + textbook_structure "
                f"({'present' if structure is not None else 'absent'}) for "
                f"synthesis-poisoning defects: {critical_count} critical "
                f"(phantom-chapter / arbitrary-resegment / section-explosion / "
                f"instructional-starvation / empty-chunk) and {warning_count} "
                f"warning issue(s); verdict={verdict}, passed={passed}."
            ),
            context=(
                f"chunks={len(rows)}; critical={critical_count}; "
                f"warnings={warning_count}; passed={passed}"
            ),
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _fail_closed(
        self,
        gate_id: str,
        issues: List[GateIssue],
        capture: Any,
        *,
        chunks_n: int,
    ) -> GateResult:
        code = next((i.code for i in issues if i.severity == "critical"), None)
        _emit_decision(
            capture,
            decision=f"NOT SYNTHESIS-READY: {code or 'input-error'}",
            rationale=(
                f"ChunkHealthValidator failed closed before auditing: {code}. "
                f"With the gate enabled a missing/unreadable chunkset is itself "
                f"a blocking condition — never synthesize from nothing."
            ),
            context=f"chunks={chunks_n}; fail_closed=true; code={code}",
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=False,
            score=0.0,
            issues=issues,
            action="block",
        )

    # ---- Structure (S-class) ----------------------------------------- #

    def _audit_structure(
        self, d: Mapping[str, Any], th: _Thresholds
    ) -> List[GateIssue]:
        issues: List[GateIssue] = []
        chapters = d.get("chapters") or []
        if not isinstance(chapters, list):
            chapters = []
        srcs = d.get("source_files") or []
        nsrc = len(srcs) if isinstance(srcs, list) else 0
        if not nsrc:
            nsrc = len(
                {
                    str(c.get("source_file"))
                    for c in chapters
                    if isinstance(c, Mapping) and c.get("source_file")
                }
            )
        diag = d.get("structureDiagnostics") or {}
        if not isinstance(diag, Mapping):
            diag = {}

        n_ch = len(chapters)

        # S1 chapter/source ratio — CRITICAL on explosion, WARNING near band.
        if nsrc and n_ch > th.chapter_ratio_fail * nsrc:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNK_HEALTH_CHAPTER_EXPLOSION",
                    message=(
                        f"{n_ch} chapters from {nsrc} source file(s) "
                        f"(>{th.chapter_ratio_fail:g}x) — phantom chapters from "
                        f"a collapsed structure; objectives would anchor to "
                        f"fabricated chapter boundaries."
                    ),
                    location="chapters",
                    suggestion=(
                        "Inspect textbook_structure.json — a source that "
                        "over-segmented (scan/OCR) or resegmented arbitrarily "
                        "needs re-conversion, not synthesis."
                    ),
                )
            )
        elif nsrc and n_ch > th.chapter_ratio_warn * nsrc:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_CHAPTER_EXPLOSION",
                    message=(
                        f"{n_ch} chapters from {nsrc} source file(s) "
                        f"(>{th.chapter_ratio_warn:g}x) — elevated chapter count."
                    ),
                    location="chapters",
                )
            )

        # S2 arbitrary contiguous_ward resegment — CRITICAL.
        method = str(diag.get("method", ""))
        if diag.get("resegmented") and "ward" in method:
            k = diag.get("k")
            osc = diag.get("original_section_count")
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNK_HEALTH_ARBITRARY_RESEGMENT",
                    message=(
                        f"structure collapsed → resegmented into {k} arbitrary "
                        f"contiguous chapters (method={method!r}, from {osc} "
                        f"sections); chapter boundaries are geometric, not "
                        f"semantic — objectives synthesized against them are "
                        f"junk."
                    ),
                    location="structureDiagnostics",
                    suggestion=(
                        "The upstream structure extractor could not recover a "
                        "real chapter hierarchy; fix conversion before "
                        "synthesizing."
                    ),
                )
            )

        # S3 + S4 section titles.
        sec_titles: List[str] = []
        for c in chapters:
            if not isinstance(c, Mapping):
                continue
            for s in c.get("sections") or []:
                if not isinstance(s, Mapping):
                    continue
                sec_titles.append(
                    str(s.get("headingText") or s.get("title") or "").strip()
                )
        nsec = len(sec_titles)
        exp_sec = max(nsrc, 1) * th.sections_per_chapter

        # S3 section explosion — CRITICAL on >4x expected, WARNING on >2x.
        if nsec > 4 * exp_sec:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNK_HEALTH_SECTION_EXPLOSION",
                    message=(
                        f"{nsec} sections (~{nsec // max(nsrc, 1)}/source) — "
                        f"expected ~{th.sections_per_chapter}/chapter; a "
                        f"shattered structure fragments every objective's "
                        f"grounding window."
                    ),
                    location="chapters[].sections",
                )
            )
        elif nsec > 2 * exp_sec:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_SECTION_EXPLOSION",
                    message=(
                        f"{nsec} sections (~{nsec // max(nsrc, 1)}/source) — "
                        f"elevated vs expected ~{th.sections_per_chapter}/chapter."
                    ),
                    location="chapters[].sections",
                )
            )

        # S4 apparatus headings — WARNING (rate-graded message).
        if sec_titles:
            appa = sum(1 for t in sec_titles if _APPARATUS.match(t))
            furn = sum(1 for t in sec_titles if _FURNITURE.search(t))
            nm = sum(1 for t in sec_titles if _NM_SECTION.match(t))
            ap_rate = appa / len(sec_titles)
            if ap_rate > th.apparatus_warn:
                band = "high" if ap_rate > th.apparatus_fail else "elevated"
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="CHUNK_HEALTH_APPARATUS_HEADINGS",
                        message=(
                            f"{appa}/{len(sec_titles)} ({ap_rate:.0%}, {band}) "
                            f"section titles are apparatus (Example/Try It/…); "
                            f"{furn} furniture; {nm} real N.M — headings are "
                            f"drill scaffolding, not a section outline."
                        ),
                        location="chapters[].sections",
                    )
                )

        # S5 furniture in chapter titles — WARNING.
        if chapters:
            ch_furn = sum(
                1
                for c in chapters
                if isinstance(c, Mapping)
                and _FURNITURE.search(str(c.get("headingText") or ""))
            )
            if ch_furn / len(chapters) > 0.3:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="CHUNK_HEALTH_FURNITURE_TITLES",
                        message=(
                            f"{ch_furn}/{len(chapters)} chapter titles carry "
                            f"running-header / page-number furniture."
                        ),
                        location="chapters",
                    )
                )

        return issues

    # ---- Chunks (C-class) -------------------------------------------- #

    def _audit_chunks(
        self, rows: List[Dict[str, Any]], th: _Thresholds
    ) -> List[GateIssue]:
        issues: List[GateIssue] = []
        n = len(rows)
        types: Counter = Counter(r.get("chunk_type") for r in rows)

        # C1 thin chunkset — WARNING.
        if n < th.min_chunks:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_THIN_CHUNKSET",
                    message=(
                        f"only {n} chunk(s) (floor {th.min_chunks}) — may be too "
                        f"coarse to ground a multi-chapter corpus."
                    ),
                )
            )

        # C2 instructional-vs-apparatus share — CRITICAL below fail floor.
        # A chunk is instructional if (a) its type is teachable prose OR (b) it
        # is an example/worked_example carrying a worked-solution marker (a
        # WORKED example teaches). Only BARE exercises/problems, answer-dumps
        # and apparatus count as non-instructional. This keeps a legitimately
        # example-heavy worked-example workbook from false-blocking; opt out via
        # ED4ALL_CHUNK_HEALTH_WORKED_EXAMPLE_INSTRUCTIONAL=0.
        worked_examples_count = 0
        if resolve_worked_example_instructional():
            worked_examples_count = sum(
                1
                for r in rows
                if r.get("chunk_type") in _WORKED_EXAMPLE_TYPES
                and _has_worked_solution(r)
            )
        instr = (
            sum(v for k, v in types.items() if k in _INSTRUCTIONAL_TYPES)
            + worked_examples_count
        )
        share = instr / n if n else 0.0
        if share < th.instructional_fail:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNK_HEALTH_INSTRUCTIONAL_STARVED",
                    message=(
                        f"{instr}/{n} ({share:.0%}) chunks are instructional "
                        f"(explanation/overview/concept/… + worked examples with "
                        f"a solution); the rest is bare exercise / answer-dump "
                        f"apparatus. Synthesis grounds objectives in teaching "
                        f"content, not bare drill — below the "
                        f"{th.instructional_fail:.0%} floor the chunkset cannot "
                        f"ground objectives."
                    ),
                    suggestion=(
                        "The converter kept bare drill apparatus but dropped "
                        "instructional prose / worked-solution content; "
                        "re-convert with the apparatus demotion / "
                        "prose-preservation path before synthesizing."
                    ),
                )
            )
        elif share < th.instructional_warn:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_INSTRUCTIONAL_STARVED",
                    message=(
                        f"{instr}/{n} ({share:.0%}) chunks are instructional — "
                        f"below the {th.instructional_warn:.0%} comfort band; "
                        f"synthesis grounding may be thin."
                    ),
                )
            )

        # C3 size distribution — WARNING (tiny / mega).
        wcs = [
            r.get("word_count") or len((r.get("text") or "").split()) for r in rows
        ]
        tiny = [w for w in wcs if isinstance(w, int) and w < th.tiny_words]
        huge = [w for w in wcs if isinstance(w, int) and w > th.mega_words]
        if tiny:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_DEGENERATE_CHUNKS",
                    message=(
                        f"{len(tiny)} chunk(s) < {th.tiny_words} words "
                        f"(degenerate); min={min(w for w in wcs if isinstance(w, int))}."
                    ),
                )
            )
        if huge:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_MEGA_CHUNKS",
                    message=(
                        f"{len(huge)} chunk(s) > {th.mega_words} words — may "
                        f"fuse multiple topics into one grounding window."
                    ),
                )
            )

        # C4 OCR mojibake / CID — WARNING (rate-annotated).
        garb = sum(
            1
            for r in rows
            if _MOJIBAKE.search((r.get("text") or "") + (r.get("html") or ""))
        )
        if garb:
            rate = garb / n if n else 0.0
            band = "high" if rate > th.mojibake_rate else "present"
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_OCR_MOJIBAKE",
                    message=(
                        f"{garb}/{n} ({rate:.0%}, {band}) chunk(s) contain "
                        f"mojibake / CID / replacement chars — OCR corruption "
                        f"bleeds into synthesized text."
                    ),
                )
            )

        # C5 running-header duplication — WARNING.
        firstlines: Counter = Counter()
        for r in rows:
            for ln in (r.get("text") or "").splitlines():
                ln = ln.strip()
                if 8 < len(ln) < 80:
                    firstlines[ln] += 1
                    break
        rep = [(ln, c) for ln, c in firstlines.items() if c >= max(3, n // 20)]
        if rep:
            example, count = rep[0]
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_REPEATED_FURNITURE",
                    message=(
                        f"{len(rep)} line(s) repeat across many chunks "
                        f"(running-header bleed), e.g. {example[:50]!r}×{count}."
                    ),
                )
            )

        # C6 furniture in provenance module_title — WARNING.
        prov_furn = sum(
            1
            for r in rows
            if _FURNITURE.search(
                str((r.get("source") or {}).get("module_title") or "")
            )
        )
        if n and prov_furn / n > 0.3:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_PROVENANCE_FURNITURE",
                    message=(
                        f"{prov_furn}/{n} chunk(s) carry running-header furniture "
                        f"in source.module_title."
                    ),
                )
            )

        # C7 empty / whitespace chunks — CRITICAL.
        empt = sum(1 for r in rows if not (r.get("text") or "").strip())
        if empt:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CHUNK_HEALTH_EMPTY_CHUNKS",
                    message=(
                        f"{empt} empty-text chunk(s) — a chunk with no prose "
                        f"contributes zero grounding and pollutes retrieval."
                    ),
                    suggestion=(
                        "Empty chunks indicate a conversion drop; re-emit the "
                        "chunkset before synthesizing."
                    ),
                )
            )

        # C8 answer-key contamination in instructional chunks — WARNING.
        contam = sum(
            1
            for r in rows
            if r.get("chunk_type") in ("explanation", "overview")
            and re.search(
                r"\b(answer key|solution:|answers to)\b",
                (r.get("text") or ""),
                re.I,
            )
        )
        if contam:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_ANSWERKEY_CONTAMINATION",
                    message=(
                        f"{contam} instructional chunk(s) contain answer-key / "
                        f"solution markers — apparatus leaked into prose chunks."
                    ),
                )
            )

        # C9 numeric-dump chunks — WARNING.
        dumps = sum(1 for r in rows if _is_numeric_dump(r.get("text") or ""))
        if dumps:
            rate = dumps / n if n else 0.0
            band = "high" if rate > th.numdump_rate else "present"
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_NUMERIC_DUMPS",
                    message=(
                        f"{dumps}/{n} ({rate:.0%}, {band}) chunk(s) are >45% bare "
                        f"numbers (answer-key / number-list dumps, not content)."
                    ),
                )
            )

        # C10 reading-order scramble — WARNING. Only evaluate non-apparatus
        # PROSE chunks: an example/exercise/answer-key/TOC chunk carries drill
        # or navigation numbering (example numbers, exercise numbers, page refs)
        # that is legitimately non-monotonic and is NOT a reading-order defect.
        # Non-monotonic N.M ordinals only signal scrambled reading order in
        # continuous instructional prose.
        scr = sum(
            1
            for r in rows
            if r.get("chunk_type") not in _APPARATUS_CHUNK_TYPES
            and _is_reading_order_scrambled(r.get("text") or "")
        )
        if scr:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CHUNK_HEALTH_READING_ORDER",
                    message=(
                        f"{scr} non-apparatus prose chunk(s) have non-monotonic "
                        f"N.M sequences (reading-order scramble); apparatus "
                        f"(example/exercise/answer-key/TOC) numbering is excluded."
                    ),
                )
            )

        return issues


# --------------------------------------------------------------------------- #
# Module-level text heuristics (mirror the prototype).
# --------------------------------------------------------------------------- #


def _is_numeric_dump(txt: str) -> bool:
    """True when >45% of tokens are bare numbers/punctuation (answer-key dump)."""
    toks = re.findall(r"\S+", txt)
    if len(toks) < 15:
        return False
    nums = sum(1 for t in toks if re.fullmatch(r"[\d,.$%()a-e]+", t))
    return nums / len(toks) > 0.45


def _is_reading_order_scrambled(txt: str) -> bool:
    """True when a chunk's N.M ordinals are non-monotonic (order scramble)."""
    nms = [tuple(map(int, m.split("."))) for m in re.findall(r"\b(\d{1,2}\.\d{1,2})\b", txt)]
    if len(nms) < 4:
        return False
    inv = sum(1 for i in range(len(nms) - 1) if nms[i] > nms[i + 1])
    return inv >= 2


__all__ = [
    "ChunkHealthValidator",
    "resolve_chunk_health_gate_enabled",
    "resolve_worked_example_instructional",
]
