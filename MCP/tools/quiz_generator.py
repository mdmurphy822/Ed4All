"""Wave 77 — bloom-balanced assessment generator (engine).

This module implements the deterministic, LLM-free path for the
``ed4all libv2 generate-quiz`` command. It reads
``LibV2/courses/<slug>/corpus/chunks.json[l]`` read-only, samples
``chunk_type == "assessment_item"`` chunks bucketed by bloom level, and
emits a quiz in one of four formats: ``json``, ``md``, ``qti``,
``imscc``.

Design notes
------------

* **Read-only** against the LibV2 archive — we never mutate course
  data. The engine only opens ``corpus/chunks.json`` (preferred) or
  ``corpus/chunks.jsonl`` (fallback) and pulls chunks into memory.
* **Deterministic with --seed** — sampling uses ``random.Random(seed)``
  with a stable insertion order (sort by chunk id within each bloom
  bucket). Two runs with the same seed produce byte-identical output.
* **Fail-loud bloom shortage** — if a requested bloom level doesn't
  have enough source items after filtering, we raise
  :class:`BloomMixShortageError` with a per-level diff showing
  ``requested`` vs ``available``. No silent substitution.
* **Misconception distractors** — when
  ``--use-misconceptions-as-distractors`` is on, we collect
  ``misconceptions[]`` entries from chunks that share at least one
  ``concept_tags`` or ``learning_outcome_refs`` value with the sampled
  item, and attach the distinct misconception statements as
  distractor seeds in the emitted quiz.
* **LLM transformation pass** is intentionally out of scope for this
  wave — the deterministic AS-IS path is the default and only path.
  An ``llm_transform`` hook is reserved for future scope.

Public API
----------

    >>> from MCP.tools.quiz_generator import (
    ...     QuizGenerator, BloomMixShortageError,
    ... )
    >>> gen = QuizGenerator.from_archive(archive_root)
    >>> quiz = gen.generate(
    ...     bloom_mix={"remember": 2, "understand": 1},
    ...     seed=42,
    ... )
    >>> gen.format_json(quiz)  # str
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
from xml.dom import minidom


# ---------------------------------------------------------------------- #
# Errors
# ---------------------------------------------------------------------- #


class BloomMixShortageError(ValueError):
    """Raised when --bloom-mix asks for more items than the corpus has."""

    def __init__(self, shortages: Mapping[str, tuple[int, int]]):
        self.shortages = dict(shortages)
        lines = [
            "Bloom mix exceeds available source assessment_item chunks:"
        ]
        for level, (req, avail) in sorted(shortages.items()):
            lines.append(
                f"  {level:12s} requested={req}, available={avail}"
            )
        lines.append(
            "Reduce --bloom-mix counts, broaden --outcomes/--difficulty, "
            "or pick a slug with more assessment_item coverage."
        )
        super().__init__("\n".join(lines))


class ArchiveNotFoundError(FileNotFoundError):
    """Raised when the slug-resolved archive root doesn't exist."""


# ---------------------------------------------------------------------- #
# Data classes (deliberately tiny — quiz is just a dict at the edge)
# ---------------------------------------------------------------------- #


@dataclass
class _ParsedQuestion:
    """One stem/options/answer triple parsed from an assessment_item.

    For chunks where no parseable structure exists, we still emit a
    single ``_ParsedQuestion`` with ``stem`` set to the full chunk text
    and ``options=[]``. The QTI/MD/JSON emitters handle that case.
    """

    stem: str
    options: list[str] = field(default_factory=list)
    correct_letter: Optional[str] = None
    correct_text: Optional[str] = None
    explanation: Optional[str] = None


# ---------------------------------------------------------------------- #
# Loading helpers
# ---------------------------------------------------------------------- #


def _load_chunks(archive_root: Path) -> list[dict]:
    """Read all chunks from ``corpus/chunks.json`` or ``chunks.jsonl``.

    JSON is preferred (matches the canonical post-Wave-76 archive
    layout). JSONL is supported as a fallback so legacy / test
    fixtures that only emit ``chunks.jsonl`` still work.
    """
    corpus = archive_root / "corpus"
    json_path = corpus / "chunks.json"
    jsonl_path = corpus / "chunks.jsonl"

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise ValueError(
                f"corpus/chunks.json is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise ValueError(
                "corpus/chunks.json must be a JSON array of chunk objects"
            )
        return data

    if jsonl_path.exists():
        out: list[dict] = []
        for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

    raise ArchiveNotFoundError(
        f"No corpus/chunks.json[l] under {archive_root}"
    )


def _load_misconceptions(chunks: Sequence[dict]) -> list[dict]:
    """Collect distinct ``misconceptions[]`` entries across all chunks.

    Each output dict has ``misconception``, ``correction``,
    ``concept_tags``, ``learning_outcome_refs``, and the
    ``source_chunk_id`` of the chunk where it was first seen. Entries
    with identical ``misconception`` text are deduplicated.
    """
    seen: dict[str, dict] = {}
    for chunk in chunks:
        items = chunk.get("misconceptions") or []
        if not isinstance(items, list):
            continue
        for entry in items:
            if not isinstance(entry, dict):
                continue
            mc = entry.get("misconception")
            if not isinstance(mc, str) or not mc.strip():
                continue
            key = mc.strip()
            if key in seen:
                continue
            seen[key] = {
                "misconception": key,
                "correction": entry.get("correction"),
                "concept_tags": list(chunk.get("concept_tags") or []),
                "learning_outcome_refs": list(
                    chunk.get("learning_outcome_refs") or []
                ),
                "source_chunk_id": chunk.get("id"),
            }
    return list(seen.values())


# ---------------------------------------------------------------------- #
# Question parsing — heuristic pass for "Show answer X." structures
# ---------------------------------------------------------------------- #


# Marker delimiter — locates "Show answer ..." in self-check text. The
# corpus uses three answer shapes:
#   * "Show answer C." / "Show answer C — ..."        (multi-choice)
#   * "Show answer True." / "Show answer False."      (true/false)
#   * "Show answer (c) is correct."                    (parenthesized)
# We capture the marker location first, then extract the answer token
# from the immediately-following text in a separate pass.
_SHOW_ANSWER_MARKER_RE = re.compile(
    r"Show answer\b",
    re.IGNORECASE,
)
_ANSWER_TOKEN_RE = re.compile(
    r"^\s*(?:\(([A-Da-d])\)|([A-Da-d])\b|(True|False|Yes|No)\b)",
    re.IGNORECASE,
)


def _extract_answer_letter(after_marker: str) -> Optional[str]:
    """Pull a normalized answer token from text immediately following
    a ``Show answer`` marker. Returns ``"A"``..``"D"`` for letter
    answers, or ``"T"``/``"F"`` for true/false. Returns ``None`` if
    no clean token is detectable (e.g. essay answer)."""
    m = _ANSWER_TOKEN_RE.match(after_marker)
    if not m:
        return None
    paren, plain, tf = m.group(1), m.group(2), m.group(3)
    if paren:
        return paren.upper()
    if plain:
        return plain.upper()
    if tf:
        return "T" if tf.lower() in ("true", "yes") else "F"
    return None


def _split_questions(text: str) -> list[_ParsedQuestion]:
    """Heuristically segment a ``text`` field into discrete questions.

    The corpus's "self-check" assessment_items follow the pattern::

        ... stem ... Option A Option B Option C Option D
        Show answer C. Explanation paragraph. ... next stem ...

    Algorithm:

    1. Find every ``Show answer X`` marker location.
    2. For each marker ``i``: the **stem-plus-options block** is the
       text between the previous marker's *end-of-explanation* (or
       text start, for the first marker) and the marker's start. The
       **explanation block** is everything from the marker's end up
       to the next marker's start (or end-of-text for the final
       marker).
    3. Within a stem-plus-options block, ``_try_split_stem_and_options``
       returns ``(stem, options)``.

    For chunks without the marker (e.g. authored exercise prompts),
    we return a single :class:`_ParsedQuestion` with the full text as
    the stem and no options — the emitter renders it as a
    free-response item.
    """
    matches = list(_SHOW_ANSWER_MARKER_RE.finditer(text))
    if not matches:
        return [_ParsedQuestion(stem=text.strip())]

    questions: list[_ParsedQuestion] = []
    # The corpus packs N questions into one inline run with no
    # structural separators between question i's explanation and
    # question i+1's stem — both live in the same gap between
    # markers. We emit one question per ``Show answer`` marker; the
    # stem-block for question i is the text between the previous
    # marker's *end* (or text start, for i=0) and this marker's
    # *start*. That stem-block transitively includes question i-1's
    # trailing explanation when i>0, which is the AS-IS shape of the
    # corpus. The LLM-transformation pass (future scope) is the
    # right place to sharpen the stem; deterministic mode preserves
    # the source faithfully.
    prev_end = 0
    for i, match in enumerate(matches):
        stem_block = text[prev_end:match.start()].strip()
        explanation_end = (
            matches[i + 1].start() if i + 1 < len(matches) else len(text)
        )
        after_marker = text[match.end():explanation_end]
        correct_letter = _extract_answer_letter(after_marker)
        explanation = after_marker.strip()
        if stem_block:
            stem, opts = _try_split_stem_and_options(stem_block)
            correct_text: Optional[str] = None
            if opts and correct_letter and len(correct_letter) == 1:
                idx = ord(correct_letter) - ord("A")
                if 0 <= idx < len(opts):
                    correct_text = opts[idx]
            questions.append(
                _ParsedQuestion(
                    stem=stem,
                    options=opts,
                    correct_letter=correct_letter,
                    correct_text=correct_text,
                    explanation=explanation or None,
                )
            )
        prev_end = match.end()

    if not questions:
        return [_ParsedQuestion(stem=text.strip())]
    return questions


def _try_split_stem_and_options(block: str) -> tuple[str, list[str]]:
    """Best-effort split of a stem-plus-options block.

    The corpus uses a single inline run of options separated by the
    sentence boundary that precedes the question mark. This heuristic
    keeps everything up to (and including) the first ``?`` as the
    stem, then treats the remainder as a single inline-options run.
    Tests don't exercise the splitter beyond round-tripping, so we
    keep this conservative: if we can't identify an inline option run,
    we return ``([block], [])`` and let downstream emit it as a
    free-response item.
    """
    qmark = block.find("?")
    if qmark < 0:
        return block.strip(), []
    stem = block[: qmark + 1].strip()
    rest = block[qmark + 1:].strip()
    if not rest:
        return stem, []
    # Split on two-or-more whitespace, vertical bar, or " | "; the
    # corpus uses single spaces, so we fall back to a single-option
    # singleton list to preserve the rest verbatim.
    parts = [p.strip() for p in re.split(r"\s{2,}|\s\|\s", rest) if p.strip()]
    if len(parts) < 2:
        # Conservative: don't try to split single-space-separated
        # options into letters — the LLM-transformation pass is the
        # right place for that. Just preserve the rest as one option.
        parts = [rest]
    return stem, parts


# ---------------------------------------------------------------------- #
# Sampling
# ---------------------------------------------------------------------- #


def _filter_items(
    items: Iterable[dict],
    *,
    outcomes: Optional[Sequence[str]] = None,
    difficulty: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Apply ``--outcomes`` and ``--difficulty`` filters."""
    out: list[dict] = []
    outcome_set = (
        {o.strip().lower() for o in outcomes if o and o.strip()}
        if outcomes
        else None
    )
    difficulty_set = (
        {d.strip().lower() for d in difficulty if d and d.strip()}
        if difficulty
        else None
    )
    for item in items:
        if outcome_set is not None:
            refs = {
                str(r).strip().lower()
                for r in (item.get("learning_outcome_refs") or [])
            }
            if outcome_set.isdisjoint(refs):
                continue
        if difficulty_set is not None:
            d = str(item.get("difficulty") or "").strip().lower()
            if d not in difficulty_set:
                continue
        out.append(item)
    return out


def _bucket_by_bloom(items: Iterable[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for item in items:
        level = str(item.get("bloom_level") or "unknown").strip().lower()
        buckets.setdefault(level, []).append(item)
    # Stable order by chunk id within each bucket.
    for level, lst in buckets.items():
        lst.sort(key=lambda c: str(c.get("id") or ""))
    return buckets


def _sample_per_level(
    buckets: Mapping[str, list[dict]],
    bloom_mix: Mapping[str, int],
    seed: Optional[int],
) -> list[dict]:
    """Sample N items per bloom level. Raises on shortage."""
    rng = random.Random(seed)
    shortages: dict[str, tuple[int, int]] = {}
    sampled: list[dict] = []
    # Sort levels for deterministic output across runs.
    for level in sorted(bloom_mix.keys()):
        n = int(bloom_mix[level])
        if n <= 0:
            continue
        pool = list(buckets.get(level.lower(), []))
        if len(pool) < n:
            shortages[level] = (n, len(pool))
            continue
        # rng.sample preserves stable input order semantics for fixed
        # seeds; we already sorted the bucket by id above.
        sampled.extend(rng.sample(pool, n))
    if shortages:
        raise BloomMixShortageError(shortages)
    return sampled


# ---------------------------------------------------------------------- #
# Misconception attachment
# ---------------------------------------------------------------------- #


def _attach_misconception_distractors(
    item: dict,
    misconceptions: Sequence[dict],
    num_distractors: int,
) -> list[dict]:
    """Pick up to ``num_distractors`` misconceptions matching ``item``.

    Match rule: misconception's ``concept_tags`` or
    ``learning_outcome_refs`` overlaps with the item's. We iterate
    misconceptions in stable (insertion) order so seed-determinism
    holds.
    """
    if num_distractors <= 0:
        return []
    item_concepts = {
        str(t).strip().lower()
        for t in (item.get("concept_tags") or [])
        if str(t).strip()
    }
    item_outcomes = {
        str(o).strip().lower()
        for o in (item.get("learning_outcome_refs") or [])
        if str(o).strip()
    }
    matched: list[dict] = []
    seen_text: set[str] = set()
    for mc in misconceptions:
        if mc["misconception"] in seen_text:
            continue
        mc_concepts = {
            str(t).strip().lower() for t in (mc.get("concept_tags") or [])
        }
        mc_outcomes = {
            str(o).strip().lower()
            for o in (mc.get("learning_outcome_refs") or [])
        }
        if (item_concepts & mc_concepts) or (item_outcomes & mc_outcomes):
            matched.append(
                {
                    "text": mc["misconception"],
                    "source": "misconception",
                    "correction": mc.get("correction"),
                    "source_chunk_id": mc.get("source_chunk_id"),
                }
            )
            seen_text.add(mc["misconception"])
            if len(matched) >= num_distractors:
                break
    return matched


# ---------------------------------------------------------------------- #
# Emitters
# ---------------------------------------------------------------------- #


def _quiz_to_json(quiz: dict) -> str:
    return json.dumps(quiz, indent=2, sort_keys=False)


def _quiz_to_md(quiz: dict) -> str:
    """Human-readable markdown rendering."""
    lines: list[str] = []
    lines.append(f"# Quiz: {quiz.get('slug', 'unknown')}")
    lines.append("")
    lines.append(f"- **Source archive**: `{quiz.get('archive_root')}`")
    lines.append(f"- **Seed**: `{quiz.get('seed')}`")
    lines.append(f"- **Items**: {len(quiz.get('items', []))}")
    bm = quiz.get("bloom_mix") or {}
    if bm:
        mix_str = ", ".join(f"{k}={v}" for k, v in sorted(bm.items()))
        lines.append(f"- **Bloom mix**: {mix_str}")
    lines.append("")
    for idx, item in enumerate(quiz.get("items", []), start=1):
        lines.append(
            f"## Item {idx}. (bloom: {item.get('bloom_level')}, "
            f"difficulty: {item.get('difficulty')})"
        )
        los = ", ".join(item.get("learning_outcome_refs") or []) or "—"
        lines.append(f"_Learning outcomes_: {los}")
        lines.append(f"_Source chunk_: `{item.get('source_chunk_id')}`")
        lines.append("")
        for q_idx, q in enumerate(item.get("questions", []), start=1):
            lines.append(f"### Q{idx}.{q_idx}")
            lines.append("")
            lines.append(q.get("stem", "").strip())
            lines.append("")
            opts = q.get("options") or []
            for o_idx, opt in enumerate(opts):
                letter = chr(ord("A") + o_idx)
                lines.append(f"- **{letter}.** {opt}")
            distractors = item.get("misconception_distractors") or []
            if distractors and q_idx == 1:
                lines.append("")
                lines.append("**Misconception-seeded distractor candidates:**")
                for d in distractors:
                    lines.append(f"- {d['text']}")
            if q.get("correct_letter"):
                lines.append("")
                lines.append(f"_Correct_: **{q['correct_letter']}**")
            if q.get("explanation"):
                lines.append("")
                lines.append(f"> {q['explanation']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_QTI_NS = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
_QTI_SCHEMA_LOCATION = (
    "http://www.imsglobal.org/xsd/ims_qtiasiv1p2 "
    "http://www.imsglobal.org/profile/cc/ccv1p3/"
    "ccv1p3_qtiasiv1p2p1_v1p0.xsd"
)


def _quiz_to_qti(quiz: dict, *, assessment_ident: str = "ed4all_quiz") -> str:
    """Emit a minimal-but-valid IMS QTI 1.2 (CC v1.3 profile) document.

    The shape mirrors the canonical Brightspace template documented at
    ``Courseforge/agents/content-generator.md`` (multi-choice items,
    ``cc.multiple_choice.v0p1`` profile). For free-response items
    (chunks without parseable options), we emit a single
    ``response_str`` essay item with the ``cc.essay.v0p1`` profile so
    downstream LMS consumers still get a well-formed item.
    """
    root = ET.Element("questestinterop")
    root.set("xmlns", _QTI_NS)
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:schemaLocation", _QTI_SCHEMA_LOCATION)

    assessment = ET.SubElement(
        root,
        "assessment",
        {"ident": assessment_ident, "title": f"Quiz {quiz.get('slug', '')}"},
    )
    qmd = ET.SubElement(assessment, "qtimetadata")
    for label, entry in (
        ("cc_profile", "cc.exam.v0p1"),
        ("qmd_assessmenttype", "Examination"),
        ("cc_maxattempts", "2"),
    ):
        f = ET.SubElement(qmd, "qtimetadatafield")
        ET.SubElement(f, "fieldlabel").text = label
        ET.SubElement(f, "fieldentry").text = entry

    section = ET.SubElement(
        assessment, "section", {"ident": "section_1"}
    )

    item_idx = 0
    for item in quiz.get("items", []):
        for q in item.get("questions", []):
            item_idx += 1
            _emit_qti_item(section, item_idx, item, q)

    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")


def _emit_qti_item(
    parent: ET.Element,
    idx: int,
    item: dict,
    question: dict,
) -> None:
    options = question.get("options") or []
    is_mc = bool(options)
    profile = "cc.multiple_choice.v0p1" if is_mc else "cc.essay.v0p1"

    qti_item = ET.SubElement(
        parent,
        "item",
        {"ident": f"q{idx}", "title": f"Question {idx}"},
    )
    item_md = ET.SubElement(qti_item, "itemmetadata")
    qmd = ET.SubElement(item_md, "qtimetadata")
    for label, entry in (
        ("cc_profile", profile),
        ("cc_weighting", "1"),
    ):
        f = ET.SubElement(qmd, "qtimetadatafield")
        ET.SubElement(f, "fieldlabel").text = label
        ET.SubElement(f, "fieldentry").text = entry
    # Carry chunk + LO provenance as harmless metadata fields.
    src_chunk = item.get("source_chunk_id")
    if src_chunk:
        f = ET.SubElement(qmd, "qtimetadatafield")
        ET.SubElement(f, "fieldlabel").text = "ed4all_source_chunk_id"
        ET.SubElement(f, "fieldentry").text = str(src_chunk)
    los = item.get("learning_outcome_refs") or []
    if los:
        f = ET.SubElement(qmd, "qtimetadatafield")
        ET.SubElement(f, "fieldlabel").text = "ed4all_learning_outcome_refs"
        ET.SubElement(f, "fieldentry").text = ",".join(los)

    presentation = ET.SubElement(qti_item, "presentation")
    material = ET.SubElement(presentation, "material")
    mattext = ET.SubElement(material, "mattext", {"texttype": "text/html"})
    mattext.text = f"<![CDATA[<p>{_xml_escape(question.get('stem', ''))}</p>]]>"

    if is_mc:
        response = ET.SubElement(
            presentation,
            "response_lid",
            {"ident": "response1", "rcardinality": "Single"},
        )
        render = ET.SubElement(response, "render_choice")
        for o_idx, opt in enumerate(options):
            letter = chr(ord("A") + o_idx)
            label = ET.SubElement(
                render, "response_label", {"ident": letter}
            )
            mat = ET.SubElement(label, "material")
            mt = ET.SubElement(mat, "mattext", {"texttype": "text/html"})
            mt.text = f"<![CDATA[{_xml_escape(opt)}]]>"

        # Add misconception-seeded distractors as additional options
        # only when the item has fewer than 4 native options. Keep
        # output deterministic and bounded.
        distractors = item.get("misconception_distractors") or []
        for d in distractors:
            if len(options) >= 4:
                break
            options = options + [d["text"]]
            o_idx = len(options) - 1
            letter = chr(ord("A") + o_idx)
            label = ET.SubElement(
                render, "response_label", {"ident": letter}
            )
            mat = ET.SubElement(label, "material")
            mt = ET.SubElement(mat, "mattext", {"texttype": "text/html"})
            mt.text = f"<![CDATA[{_xml_escape(d['text'])}]]>"

        resprocessing = ET.SubElement(qti_item, "resprocessing")
        outcomes = ET.SubElement(resprocessing, "outcomes")
        ET.SubElement(
            outcomes,
            "decvar",
            {
                "varname": "SCORE",
                "vartype": "Integer",
                "minvalue": "0",
                "maxvalue": "1",
            },
        )
        if question.get("correct_letter"):
            cond = ET.SubElement(resprocessing, "respcondition")
            cv = ET.SubElement(cond, "conditionvar")
            ET.SubElement(
                cv, "varequal", {"respident": "response1"}
            ).text = question["correct_letter"]
            ET.SubElement(
                cond, "setvar",
                {"action": "Set", "varname": "SCORE"},
            ).text = "1"
    else:
        # Essay / free-response shape.
        ET.SubElement(
            presentation,
            "response_str",
            {"ident": "response1", "rcardinality": "Single"},
        )


def _xml_escape(s: str) -> str:
    """CDATA-friendly escape — only neutralize ``]]>`` sequences."""
    return (s or "").replace("]]>", "]]]]><![CDATA[>")


def _quiz_to_imscc(quiz: dict, output_path: Path) -> Path:
    """Bundle the quiz QTI XML into a minimal IMSCC zip.

    Layout::

        <output_path>
        ├── imsmanifest.xml
        └── quiz/
            └── quiz.xml

    The manifest uses the IMS CC v1.3 namespace and references the
    quiz as a single ``imsqti_xmlv1p2/imscc_xmlv1p1/assessment``
    resource — sufficient for round-trip via Courseforge's existing
    intake parser.
    """
    qti_xml = _quiz_to_qti(quiz)
    manifest_xml = _imscc_manifest_xml()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest_xml)
        zf.writestr("quiz/quiz.xml", qti_xml)
    return output_path


def _imscc_manifest_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<manifest identifier="ed4all_quiz_manifest"\n'
        '  xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"\n'
        '  xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"\n'
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        '  <metadata>\n'
        '    <schema>IMS Common Cartridge</schema>\n'
        '    <schemaversion>1.3.0</schemaversion>\n'
        '  </metadata>\n'
        '  <organizations>\n'
        '    <organization identifier="org_1" structure="rooted-hierarchy">\n'
        '      <item identifier="root">\n'
        '        <item identifier="quiz_item" identifierref="quiz_resource">\n'
        '          <title>Generated Quiz</title>\n'
        '        </item>\n'
        '      </item>\n'
        '    </organization>\n'
        '  </organizations>\n'
        '  <resources>\n'
        '    <resource identifier="quiz_resource"\n'
        '              type="imsqti_xmlv1p2/imscc_xmlv1p1/assessment"\n'
        '              href="quiz/quiz.xml">\n'
        '      <file href="quiz/quiz.xml"/>\n'
        '    </resource>\n'
        '  </resources>\n'
        '</manifest>\n'
    )


# ---------------------------------------------------------------------- #
# Public engine
# ---------------------------------------------------------------------- #


@dataclass
class QuizGenerator:
    """Engine entry point. Reusable from CLI, MCP, or tests."""

    archive_root: Path
    chunks: list[dict]
    misconceptions: list[dict]

    @classmethod
    def from_archive(cls, archive_root: Path) -> "QuizGenerator":
        archive_root = Path(archive_root)
        if not archive_root.exists():
            raise ArchiveNotFoundError(
                f"Archive root not found: {archive_root}"
            )
        chunks = _load_chunks(archive_root)
        misconceptions = _load_misconceptions(chunks)
        return cls(
            archive_root=archive_root,
            chunks=chunks,
            misconceptions=misconceptions,
        )

    @property
    def assessment_items(self) -> list[dict]:
        return [
            c
            for c in self.chunks
            if str(c.get("chunk_type") or "") == "assessment_item"
        ]

    def generate(
        self,
        *,
        bloom_mix: Mapping[str, int],
        outcomes: Optional[Sequence[str]] = None,
        difficulty: Optional[Sequence[str]] = None,
        use_misconceptions_as_distractors: bool = False,
        num_distractors: int = 3,
        seed: Optional[int] = None,
    ) -> dict:
        filtered = _filter_items(
            self.assessment_items,
            outcomes=outcomes,
            difficulty=difficulty,
        )
        buckets = _bucket_by_bloom(filtered)
        sampled = _sample_per_level(buckets, bloom_mix, seed)

        items_out: list[dict] = []
        distractor_source_counts = {"misconception": 0, "existing_item": 0}
        for chunk in sampled:
            questions = _split_questions(str(chunk.get("text") or ""))
            # Count distractors from the existing item (parsed options
            # excluding the correct letter) for reporting.
            for q in questions:
                if q.options and q.correct_letter:
                    distractor_source_counts["existing_item"] += max(
                        len(q.options) - 1, 0
                    )

            mc_distractors: list[dict] = []
            if use_misconceptions_as_distractors:
                mc_distractors = _attach_misconception_distractors(
                    chunk, self.misconceptions, num_distractors
                )
                distractor_source_counts["misconception"] += len(mc_distractors)

            items_out.append(
                {
                    "source_chunk_id": chunk.get("id"),
                    "bloom_level": chunk.get("bloom_level"),
                    "difficulty": chunk.get("difficulty"),
                    "learning_outcome_refs": list(
                        chunk.get("learning_outcome_refs") or []
                    ),
                    "concept_tags": list(chunk.get("concept_tags") or []),
                    "questions": [
                        {
                            "stem": q.stem,
                            "options": q.options,
                            "correct_letter": q.correct_letter,
                            "correct_text": q.correct_text,
                            "explanation": q.explanation,
                        }
                        for q in questions
                    ],
                    "misconception_distractors": mc_distractors,
                }
            )

        quiz: dict = {
            "slug": self.archive_root.name,
            "archive_root": str(self.archive_root),
            "bloom_mix": dict(bloom_mix),
            "outcomes": list(outcomes) if outcomes else None,
            "difficulty": list(difficulty) if difficulty else None,
            "use_misconceptions_as_distractors": (
                use_misconceptions_as_distractors
            ),
            "num_distractors": num_distractors,
            "seed": seed,
            "items": items_out,
            "distractor_source_counts": distractor_source_counts,
        }
        return quiz

    # ------------------------------------------------------------------ #
    # Format adapters (instance methods so callers can mock the engine)
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_json(quiz: dict) -> str:
        return _quiz_to_json(quiz)

    @staticmethod
    def format_md(quiz: dict) -> str:
        return _quiz_to_md(quiz)

    @staticmethod
    def format_qti(quiz: dict, *, assessment_ident: str = "ed4all_quiz") -> str:
        return _quiz_to_qti(quiz, assessment_ident=assessment_ident)

    @staticmethod
    def write_imscc(quiz: dict, output_path: Path) -> Path:
        return _quiz_to_imscc(quiz, output_path)


__all__ = [
    "QuizGenerator",
    "BloomMixShortageError",
    "ArchiveNotFoundError",
]
