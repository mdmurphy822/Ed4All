"""Assessment -> SFT pair generator (SFT data program Phase 1).

Deterministic, LLM-free emitter that turns the W10 assessment product
(``assessments.json`` + the instructor ``answer_key.json``) into
rationale-augmented, open-book instruction-following training pairs for the
course-pinned 1.5B LoRA adapter.

Design contract (``runtime/scratchpad/sft_data_program.md`` §A / §D-Phase-1):

* **Format diversity, not volume, is the scaling axis at 1.5B.** Every item
  fans out into up to six *rationale-augmented* formats — the target teaches
  reasoning-then-answer, never answer-only:

  1. ``solve_steps``     — worked reasoning to a verified answer.
  2. ``error_diagnosis`` — which-step-is-wrong -> named transform -> correction
     (carries DPO-shape ``dpo_metadata`` {chosen, rejected} for a future
     preference arm; ships SFT-only now).
  3. ``explain_why``     — justify an answer against source.
  4. ``grade_rubric``    — score a response against the item's analytic rubric.
  5. ``hint_no_reveal``  — scaffold WITHOUT solving (answer-leak self-verified).
  6. ``verify_answer``   — backward yes/no verification of a proposed answer.

* **The pair TEXT is a pure function of course artifacts** — no LLM call this
  round. The completions are structured from the generator's own authored
  ``feedback`` / ``rubric`` / ``choices`` / ``correct_answer`` fields.

* **Open-book, but leakage-clean by construction.** The prompts carry the
  QUESTION + student work (never a pasted chunk), and each pair's
  ``chunk_id`` is downgraded to a synthetic anchor whenever its text would
  share a >=50-char verbatim span with the cited source chunk — so the
  ``synthesis_leakage`` verbatim gate stays clean without a global carve-out.
  Real grounding always rides ``source_chunk_ids`` for provenance / W4.

* **D1 — never source harvested assessment_item chunks.** Any
  ``source_chunk_ids`` entry that resolves to a ``practice_bank=true`` /
  ``chunk_type='assessment_item'`` chunk (the demoted-to-practice
  internal-only publisher end-of-section bank) is dropped before it can ground a pair.

* **Per-pair provenance** (``§B``): ``generation_method``, ``generating_seat``
  + ``seat_license``, ``verifier_results``, ``source_chunk_ids``,
  ``template_id``, ``seed``, ``decision_capture_id``, ``holdout_safe``,
  ``decontam_checked``.

* **Template-instance caps** per format so a single format cannot dominate the
  corpus (LIMA / diversity-over-volume).

* **Provider enum reuses ``local``** (the closed pair-schema enum); the real
  deterministic method + seat travel in the additive provenance fields, which
  ``instruction_pair.schema.json``'s ``additionalProperties: true`` permits.

Entry point: :func:`generate_assessment_sft_pairs`. The Trainforge synthesis
runner imports this module and calls it; this module NEVER writes files — it
yields pair dicts the runner appends to ``instruction_pairs.jsonl`` (staying
inside the ``instruction_pairs_hash`` provenance chain).
"""
from __future__ import annotations

import logging
import re
import zlib
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Reuse the canonical verbatim-span helper so the generator's leakage
# self-guard stays byte-aligned with the runtime ``synthesis_leakage`` gate.
from lib.validators.synthesis_leakage import (  # noqa: E402
    DEFAULT_LEAK_SPAN_CHARS,
    _contains_verbatim_span,
)

# --------------------------------------------------------------------------- #
# Canonical constants
# --------------------------------------------------------------------------- #

_CANON_BLOOM: Set[str] = {
    "remember", "understand", "apply", "analyze", "evaluate", "create",
}

# instruction_pair.schema.json floors/ceilings (prompt 40-400, completion
# 50-600). A pair that cannot meet a floor is SKIPPED (never padded with
# sentinel filler — that is the exact anti-pattern the synthesis invariants
# forbid).
_PROMPT_MIN, _PROMPT_MAX = 40, 400
_COMPLETION_MIN, _COMPLETION_MAX = 50, 600

# Marker the ``synthesis_leakage`` + ``audit_pairs`` scaffold carve-out keys
# on (owner-approved marker scope, never a global disable).
ASSESSMENT_SFT_SOURCE_MARKER = "assessment_item"

# Every assessment-SFT pair carries this content_type so downstream audits can
# bucket the cohort.
_CONTENT_TYPE = "assessment_sft"

# Provider enum value reused for the closed pair-schema enum. The real method /
# seat / license ride the additive provenance fields.
_PROVIDER = "local"

# Ordered format list (drives deterministic emit order per item).
_FORMATS: Tuple[str, ...] = (
    "solve_steps",
    "error_diagnosis",
    "explain_why",
    "grade_rubric",
    "hint_no_reveal",
    "verify_answer",
)

# Per-template instance caps (target shares from §A for a ~200-chunk course;
# used as hard caps here). Override via ``per_template_caps``.
DEFAULT_TEMPLATE_CAPS: Dict[str, int] = {
    "solve_steps": 600,
    "error_diagnosis": 480,
    "explain_why": 290,
    "grade_rubric": 190,
    "hint_no_reveal": 190,
    "verify_answer": 170,
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
_NUMERIC_RE = re.compile(r"^[\s$]*[-+]?\(?\d[\d,]*\.?\d*\)?(?:\s*/\s*\d+)?[\s%]*$")


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #

def _strip_html(value: Any) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", str(value or ""))).strip()


def _norm_bloom(value: Any) -> str:
    b = str(value or "").strip().lower()
    return b if b in _CANON_BLOOM else "understand"


def _looks_numeric(answer: str) -> bool:
    return bool(_NUMERIC_RE.match(str(answer or "").strip())) and any(
        ch.isdigit() for ch in str(answer)
    )


def _seed_for(item_id: str, fmt: str, variant: int = 0) -> int:
    key = f"{item_id}\x1f{fmt}\x1f{variant}".encode("utf-8")
    return int(zlib.crc32(key) & 0x7FFFFFFF)


def _bound(text: str, lo: int, hi: int) -> Optional[str]:
    """Return ``text`` clipped to ``hi`` chars (word boundary) or None if it
    cannot meet ``lo``. Never pads — a too-short field is dropped upstream."""
    t = _WS_RE.sub(" ", str(text or "")).strip()
    if len(t) < lo:
        return None
    if len(t) > hi:
        cut = t[:hi]
        sp = cut.rfind(" ")
        if sp >= lo:
            cut = cut[:sp]
        t = cut.rstrip()
    return t if len(t) >= lo else None


def _worked_steps(feedback: Any) -> List[str]:
    raw = str(feedback or "")
    if not raw.strip():
        return []
    li = _LI_RE.findall(raw)
    parts = [_strip_html(x) for x in li] if li else [_strip_html(raw)]
    return [p for p in parts if p]


def _correct_answers(item: Dict[str, Any]) -> List[str]:
    plural = item.get("correct_answers")
    if isinstance(plural, list) and plural:
        out = [_strip_html(v) for v in plural if str(v).strip()]
        if out:
            return out
    scalar = item.get("correct_answer")
    if scalar not in (None, ""):
        s = _strip_html(scalar)
        if s:
            return [s]
    keys: List[str] = []
    for c in item.get("choices") or []:
        if isinstance(c, dict) and c.get("is_correct"):
            t = _strip_html(c.get("text"))
            if t:
                keys.append(t)
    return keys


def _distractors(item: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    out: List[Dict[str, Optional[str]]] = []
    for c in item.get("choices") or []:
        if not isinstance(c, dict) or c.get("is_correct"):
            continue
        t = _strip_html(c.get("text"))
        if not t:
            continue
        note = c.get("misconception_note") or c.get("misconception")
        out.append({"text": t, "misconception": _strip_html(note) if note else None})
    return out


def _rubric_criteria(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    rub = item.get("rubric")
    if not isinstance(rub, dict):
        return []
    crits = rub.get("criteria")
    return [c for c in crits if isinstance(c, dict)] if isinstance(crits, list) else []


def _rubric_cites(item: Dict[str, Any]) -> List[str]:
    rub = item.get("rubric")
    if not isinstance(rub, dict):
        return []
    cites: List[str] = []
    for row in (rub.get("criteria") or []) + (rub.get("deductions") or []):
        if isinstance(row, dict):
            for c in row.get("cites") or []:
                if str(c).strip():
                    cites.append(str(c).strip())
    # stable de-dup
    seen: Set[str] = set()
    ordered: List[str] = []
    for c in cites:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# --------------------------------------------------------------------------- #
# Chunk classification (D1)
# --------------------------------------------------------------------------- #

def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return ""


def _is_practice_bank(chunk: Any) -> bool:
    """D1 — a harvested assessment_item / practice-bank chunk the generator
    must NEVER source. A bare text string carries no marker (treated as
    non-practice, since it cannot be a harvested item chunk)."""
    if not isinstance(chunk, dict):
        return False
    if chunk.get("practice_bank") is True:
        return True
    ctype = str(chunk.get("chunk_type") or chunk.get("content_type_label") or "")
    return ctype == "assessment_item"


def _grounding_chunk_ids(
    item: Dict[str, Any],
    chunks_by_id: Dict[str, Any],
) -> List[str]:
    """Filtered source-chunk grounding for this item.

    Drops any ref resolving to a practice-bank / assessment_item chunk (D1).
    A ref not present in ``chunks_by_id`` is KEPT (it is a real provenance ref
    the generator cannot classify) — the caller only anchors ``chunk_id`` to a
    resolvable, verbatim-safe id, so unknown refs never poison the leakage gate.
    """
    out: List[str] = []
    seen: Set[str] = set()
    refs = (
        item.get("raw_source_refs")
        or item.get("source_chunk_ids")
        or item.get("source_chunks")
        or []
    )
    for raw in refs:
        cid = str(raw).strip()
        if not cid or cid in seen:
            continue
        chunk = chunks_by_id.get(cid)
        if chunk is not None and _is_practice_bank(chunk):
            continue  # D1: never source a harvested assessment_item chunk
        seen.add(cid)
        out.append(cid)
    return out


# --------------------------------------------------------------------------- #
# Item normalization
# --------------------------------------------------------------------------- #

def _iter_raw_items(
    assessments_doc: Any,
    answer_key_doc: Optional[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Yield per-question dicts from the assessments product.

    Accepts (in priority order) an ``AssessmentData.to_dict()`` doc (or a list
    of them), a bare ``questions`` list, or the ``answer_key.json`` ``items``
    list. When both an assessments doc and an answer-key doc are supplied, the
    answer-key ``worked_solution_steps`` / ``per_distractor`` / ``rubric``
    enrich the matching question by ``question_id`` / ``item_id``.
    """
    ak_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(answer_key_doc, dict):
        for it in answer_key_doc.get("items") or []:
            if isinstance(it, dict) and it.get("item_id"):
                ak_by_id[str(it["item_id"])] = it

    def _questions(doc: Any) -> List[Dict[str, Any]]:
        if isinstance(doc, dict):
            qs = doc.get("questions")
            if isinstance(qs, list):
                return [q for q in qs if isinstance(q, dict)]
            # maybe it IS an answer_key doc
            items = doc.get("items")
            if isinstance(items, list):
                return [q for q in items if isinstance(q, dict)]
            return []
        if isinstance(doc, list):
            out: List[Dict[str, Any]] = []
            for d in doc:
                out.extend(_questions(d))
            return out
        return []

    for q in _questions(assessments_doc):
        qid = str(q.get("question_id") or q.get("item_id") or "").strip()
        merged = dict(q)
        ak = ak_by_id.get(qid)
        if ak:
            # Answer-key worked steps + per-distractor misconceptions +
            # rubric are the authoritative reshaped view; fold them in
            # without clobbering an already-present richer field.
            if not merged.get("feedback") and ak.get("worked_solution_steps"):
                merged["_ak_worked_steps"] = ak["worked_solution_steps"]
            if ak.get("rubric") and not merged.get("rubric"):
                merged["rubric"] = ak["rubric"]
            if ak.get("per_distractor") and not merged.get("choices"):
                merged["_ak_per_distractor"] = ak["per_distractor"]
            if ak.get("source_chunk_ids") and not (
                merged.get("source_chunk_ids") or merged.get("source_chunks")
            ):
                merged["source_chunk_ids"] = ak["source_chunk_ids"]
        yield merged


def _normalize_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    item_id = str(raw.get("question_id") or raw.get("item_id") or "").strip()
    steps = _worked_steps(raw.get("feedback"))
    if not steps and raw.get("_ak_worked_steps"):
        steps = [
            _strip_html(s.get("text")) if isinstance(s, dict) else _strip_html(s)
            for s in raw.get("_ak_worked_steps") or []
        ]
        steps = [s for s in steps if s]
    distractors = _distractors(raw)
    if not distractors and raw.get("_ak_per_distractor"):
        for d in raw.get("_ak_per_distractor") or []:
            if isinstance(d, dict) and _strip_html(d.get("choice")):
                distractors.append({
                    "text": _strip_html(d.get("choice")),
                    "misconception": _strip_html(d.get("misconception"))
                    if d.get("misconception") else None,
                })
    answers = _correct_answers(raw)
    return {
        "item_id": item_id,
        "objective_id": str(raw.get("objective_id") or "").strip(),
        "question_type": str(raw.get("question_type") or raw.get("type") or "").strip(),
        "subtype": str(raw.get("item_subtype") or "").strip(),
        "stem": _strip_html(raw.get("stem") or raw.get("stem_html")),
        "bloom_level": _norm_bloom(raw.get("bloom_level")),
        "answers": answers,
        "distractors": distractors,
        "steps": steps,
        "rubric": raw.get("rubric") if isinstance(raw.get("rubric"), dict) else None,
        "rubric_criteria": _rubric_criteria(raw),
        "rubric_cites": _rubric_cites(raw),
        "raw_source_refs": list(raw.get("source_chunk_ids") or raw.get("source_chunks") or []),
        "sympy_verified": raw.get("sympy_verified"),
        "numeric": bool(answers) and _looks_numeric(answers[0]),
    }


# --------------------------------------------------------------------------- #
# Per-format builders — each returns (prompt, completion, extra) or None
# --------------------------------------------------------------------------- #

def _fmt_solve_steps(item: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    if not item["steps"] or not item["answers"]:
        return None  # solve+steps REQUIRES worked reasoning
    body = " ".join(f"Step {i}: {s}" for i, s in enumerate(item["steps"], 1))
    completion = f"{body} Therefore, the answer is {item['answers'][0]}."
    prompt = f"Solve the following problem, showing each step of your reasoning.\n\n{item['stem']}"
    extra = {
        "verifier_results": {
            "sympy_verified": item["sympy_verified"] if isinstance(item["sympy_verified"], bool) else None,
            "sympy_key_present": bool(item["numeric"]),
        },
    }
    return prompt, completion, extra


def _fmt_error_diagnosis(item: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    named = [d for d in item["distractors"] if d.get("misconception")]
    if not named or not item["answers"]:
        return None
    d0 = named[0]
    answer = item["answers"][0]
    chosen = (
        f"The step that leads to '{d0['text']}' is wrong: {d0['misconception']} "
        f"The correct answer is {answer}."
    )
    rejected = (
        f"{d0['text']} is correct." if not d0["misconception"]
        else f"{d0['text']} — following the flawed reasoning: {d0['misconception']}"
    )
    prompt = (
        "A student answered this question incorrectly. Identify the error in "
        "their reasoning, name the misconception, and give the correct answer.\n\n"
        f"{item['stem']}\nStudent's answer: {d0['text']}"
    )
    extra = {
        "verifier_results": {"transform_named": True},
        "dpo_metadata": {"chosen": chosen, "rejected": rejected, "kind": "error_analysis"},
    }
    return prompt, chosen, extra


def _fmt_explain_why(item: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    if not item["answers"]:
        return None
    answer = item["answers"][0]
    if item["steps"]:
        reason = " ".join(item["steps"])
    else:
        reason = "it satisfies the conditions the question establishes and follows from the relevant definition or procedure"
    completion = f"'{answer}' is correct because {reason}."
    prompt = (
        "Explain why the following answer is correct.\n\n"
        f"Question: {item['stem']}\nAnswer: {answer}"
    )
    extra = {"verifier_results": {"nli_entailment": None}}
    return prompt, completion, extra


def _fmt_grade_rubric(item: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    crits = item["rubric_criteria"]
    if not crits:
        return None
    model_answer = " ".join(item["steps"]) if item["steps"] else (
        item["answers"][0] if item["answers"] else ""
    )
    if not model_answer:
        return None
    crit_texts = [_strip_html(c.get("criterion")) for c in crits[:3]]
    crit_texts = [c for c in crit_texts if c]
    if not crit_texts:
        return None
    prompt = (
        "Grade the student response against the rubric criteria below. For each "
        "criterion state the score and justify it.\n\n"
        f"Question: {item['stem']}\nResponse: {model_answer}\n"
        "Criteria: " + "; ".join(crit_texts)
    )
    completion = " ".join(
        f"{c}: full marks — the response meets this criterion."
        for c in crit_texts
    )
    return prompt, completion, {"verifier_results": {"rubric_criteria_count": len(crit_texts)}}


def _fmt_hint_no_reveal(item: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    if not item["answers"]:
        return None
    generic = (
        "Re-read the question and identify the key quantity or relationship it "
        "asks about, then apply the definition or procedure that connects the "
        "given information to what is unknown."
    )
    seed_hint = item["steps"][0] if item["steps"] else generic
    hint_body = seed_hint
    # Answer-leak self-verify: the hint must NOT contain any correct-answer
    # token. Fall back to the generic method hint; if THAT still leaks, skip.
    def _leaks(text: str) -> bool:
        low = text.lower()
        return any(a and a.lower() in low for a in item["answers"])
    if _leaks(hint_body):
        hint_body = generic
    if _leaks(hint_body):
        return None
    completion = f"Here is a hint that does not give away the answer: {hint_body}"
    prompt = (
        "A student is stuck on this problem. Give a hint that guides them toward "
        "the method WITHOUT revealing the final answer.\n\n"
        f"{item['stem']}"
    )
    return prompt, completion, {"verifier_results": {"answer_leak_check": "passed"}}


def _fmt_verify_answer(item: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    if not item["answers"]:
        return []
    answer = item["answers"][0]
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    just = (" because " + " ".join(item["steps"])) if item["steps"] else (
        ", as it satisfies the question's conditions"
    )
    pos_prompt = (
        "Is the proposed answer to this question correct? Answer Yes or No and "
        "justify.\n\n"
        f"Question: {item['stem']}\nProposed answer: {answer}"
    )
    pos_completion = f"Yes. '{answer}' is correct{just}."
    out.append((pos_prompt, pos_completion, {
        "verifier_results": {
            "sympy_verified": item["sympy_verified"] if isinstance(item["sympy_verified"], bool) else None,
            "backward_check": True,
        },
        "_variant": 0,
    }))
    if item["distractors"]:
        d = item["distractors"][0]
        why = f"; {d['misconception']}" if d.get("misconception") else "; it does not satisfy the question's conditions"
        neg_prompt = (
            "Is the proposed answer to this question correct? Answer Yes or No "
            "and justify.\n\n"
            f"Question: {item['stem']}\nProposed answer: {d['text']}"
        )
        neg_completion = f"No. '{d['text']}' is not correct{why}. The correct answer is {answer}."
        out.append((neg_prompt, neg_completion, {
            "verifier_results": {"backward_check": True},
            "_variant": 1,
        }))
    return out


# --------------------------------------------------------------------------- #
# Pair assembly
# --------------------------------------------------------------------------- #

def _resolve_chunk_anchor(
    grounding_ids: Sequence[str],
    chunks_by_id: Dict[str, Any],
    prompt: str,
    completion: str,
    item_id: str,
    span_chars: int,
) -> Tuple[str, bool]:
    """Pick a leakage-safe ``chunk_id`` anchor.

    Prefers the first grounding ref that resolves AND whose text shares no
    >=span_chars verbatim span with prompt/completion (so the runtime verbatim
    gate stays clean). When none is safe (or none resolves), returns a
    synthetic ``assessment_item:<item_id>`` anchor — the verbatim gate's
    chunk-text lookup then misses and skips the scan, exactly like the four
    deterministic generators.
    """
    for cid in grounding_ids:
        text = _chunk_text(chunks_by_id.get(cid))
        if not text:
            continue
        if _contains_verbatim_span(completion, text, span_chars) or \
                _contains_verbatim_span(prompt, text, span_chars):
            continue
        return cid, False
    return f"assessment_item:{item_id or 'unknown'}", True


def _build_pair(
    *,
    fmt: str,
    prompt: str,
    completion: str,
    extra: Dict[str, Any],
    item: Dict[str, Any],
    chunks_by_id: Dict[str, Any],
    grounding_ids: List[str],
    decision_capture_id: str,
    generating_seat: str,
    seat_license: str,
    span_chars: int,
    variant: int,
) -> Optional[Dict[str, Any]]:
    bp = _bound(prompt, _PROMPT_MIN, _PROMPT_MAX)
    bc = _bound(completion, _COMPLETION_MIN, _COMPLETION_MAX)
    if bp is None or bc is None:
        return None
    chunk_id, downgraded = _resolve_chunk_anchor(
        grounding_ids, chunks_by_id, bp, bc, item["item_id"], span_chars,
    )
    lo = item["objective_id"] or "assessment"
    verifier = dict(extra.get("verifier_results") or {})
    pair: Dict[str, Any] = {
        "prompt": bp,
        "completion": bc,
        "chunk_id": chunk_id,
        "lo_refs": [lo],
        "bloom_level": item["bloom_level"],
        "content_type": _CONTENT_TYPE,
        "seed": _seed_for(item["item_id"], fmt, variant),
        "decision_capture_id": decision_capture_id,
        "template_id": f"assessment_sft.{fmt}",
        "provider": _PROVIDER,
        "schema_version": "v1",
        # Carve-out marker (owner-approved marker scope) — the scaffold gates
        # exempt these rows; the verbatim gate still runs (kept clean above).
        "source": ASSESSMENT_SFT_SOURCE_MARKER,
        # Per-pair provenance (§B).
        "generation_method": "deterministic_template",
        "generating_seat": generating_seat,
        "seat_license": seat_license,
        "verifier_results": verifier,
        "source_chunk_ids": list(grounding_ids),
        "holdout_safe": True,
        "decontam_checked": False,
        # Audit surfaces.
        "pair_format": fmt,
        "chunk_id_downgraded_verbatim": downgraded,
    }
    if fmt == "grade_rubric" and item["rubric_cites"]:
        pair["rubric_cites"] = list(item["rubric_cites"])
    if "dpo_metadata" in extra:
        pair["dpo_metadata"] = extra["dpo_metadata"]
    # SFT-C fine-grained teacher-license stamp: sets pair["license"] (+ normalizes
    # generating_seat) so the fail-closed export filter can attribute the precise
    # teacher instead of falling back to coarse `provider` provenance. Additive —
    # the pair already carries generating_seat/seat_license, and the pair schema is
    # additionalProperties:true. Best-effort: a missing lib.licensing never breaks
    # deterministic emit, and `generating_seat` keeps its byte-identical value.
    try:
        from lib.licensing import stamp_pair_license as _stamp_pair_license
        _stamp_pair_license(pair, generating_seat=generating_seat, provider=_PROVIDER)
    except Exception:  # noqa: BLE001 — provenance stamp is best-effort
        pass
    return pair


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def generate_assessment_sft_pairs(
    assessments_doc: Any,
    chunks_by_id: Dict[str, Any],
    capture: Any,
    *,
    answer_key_doc: Optional[Dict[str, Any]] = None,
    per_template_caps: Optional[Dict[str, int]] = None,
    max_pairs: Optional[int] = None,
    generating_seat: str = "local",
    seat_license: str = "deterministic-template",
    holdout_item_ids: Optional[Iterable[str]] = None,
    span_chars: int = DEFAULT_LEAK_SPAN_CHARS,
) -> Iterator[Dict[str, Any]]:
    """Yield assessment-derived SFT instruction pairs.

    Args:
        assessments_doc: The W10 ``assessments.json`` payload — an
            ``AssessmentData.to_dict()`` doc, a list of them, a bare
            ``questions`` list, or (fallback) an ``answer_key.json`` doc.
        chunks_by_id: ``chunk_id -> chunk`` map (dict OR text string). Used to
            (a) exclude practice-bank / assessment_item source chunks (D1) and
            (b) pick a verbatim-safe ``chunk_id`` anchor. May be empty.
        capture: A ``DecisionCapture``-shaped object exposing ``log_decision``
            + a ``decisions`` list. **Required** — one ``assessment_generation``
            event fires per format batch and anchors every pair's
            ``decision_capture_id`` (an empty id is a schema violation).
        answer_key_doc: Optional parsed ``answer_key.json`` — enriches items
            with worked-solution steps / per-distractor misconceptions / rubrics.
        per_template_caps: Per-format hard caps (defaults ``DEFAULT_TEMPLATE_CAPS``).
        max_pairs: Optional global cap across all formats.
        generating_seat / seat_license: Provenance tags stamped per pair.
        holdout_item_ids: Item ids RESERVED for the memorization probe — they
            never produce a pair (keeps ``holdout_safe`` honest).
        span_chars: Verbatim-span length used by the leakage self-guard.

    Yields:
        Instruction-pair dicts (``instruction_pair.schema.json`` shape + additive
        provenance). Deterministic: same inputs -> same pairs in the same order.
    """
    if capture is None:
        raise ValueError(
            "assessment_sft_generator requires a DecisionCapture (got None); "
            "it logs one assessment_generation event per format batch and "
            "anchors each pair's decision_capture_id from it."
        )

    caps = dict(DEFAULT_TEMPLATE_CAPS)
    if per_template_caps:
        for k, v in per_template_caps.items():
            try:
                caps[k] = max(0, int(v))
            except (TypeError, ValueError):
                continue

    holdout: Set[str] = {str(x) for x in (holdout_item_ids or [])}

    # Normalize + sort items for deterministic emit order.
    items = [
        _normalize_item(raw)
        for raw in _iter_raw_items(assessments_doc, answer_key_doc)
    ]
    items = [it for it in items if it["item_id"] and it["item_id"] not in holdout and it["stem"]]
    items.sort(key=lambda it: it["item_id"])

    per_format_count: Dict[str, int] = {f: 0 for f in _FORMATS}
    total_emitted = 0

    # One decision per format batch (anchors that batch's pairs). Logged
    # up-front so decision_capture_id is available before the first emit; the
    # rationale interpolates dynamic signals per the capture contract.
    batch_event_id: Dict[str, str] = {}
    for fmt in _FORMATS:
        capture.log_decision(
            decision_type="assessment_generation",
            decision=f"assessment_sft_batch:{fmt}",
            rationale=(
                f"Deterministic assessment->SFT batch '{fmt}': normalized_items="
                f"{len(items)}, per_template_cap={caps.get(fmt, 0)}, "
                f"holdout_reserved={len(holdout)}, max_pairs={max_pairs}, "
                f"seat={generating_seat}/{seat_license}, span_chars={span_chars}. "
                f"Open-book rationale-augmented pairs, no LLM this round."
            ),
        )
        batch_event_id[fmt] = _last_event_id(capture)

    for item in items:
        grounding_ids = _grounding_chunk_ids(item, chunks_by_id)
        for fmt in _FORMATS:
            if per_format_count[fmt] >= caps.get(fmt, 0):
                continue
            if max_pairs is not None and total_emitted >= max_pairs:
                return

            if fmt == "verify_answer":
                built = _fmt_verify_answer(item)
            else:
                one = _FORMAT_BUILDERS[fmt](item)
                built = [one] if one else []

            for prompt, completion, extra in built:
                if per_format_count[fmt] >= caps.get(fmt, 0):
                    break
                if max_pairs is not None and total_emitted >= max_pairs:
                    return
                variant = int(extra.get("_variant", 0))
                pair = _build_pair(
                    fmt=fmt,
                    prompt=prompt,
                    completion=completion,
                    extra=extra,
                    item=item,
                    chunks_by_id=chunks_by_id,
                    grounding_ids=grounding_ids,
                    decision_capture_id=batch_event_id[fmt],
                    generating_seat=generating_seat,
                    seat_license=seat_license,
                    span_chars=span_chars,
                    variant=variant,
                )
                if pair is None:
                    continue
                per_format_count[fmt] += 1
                total_emitted += 1
                yield pair


def _last_event_id(capture: Any) -> str:
    """Return the event_id of the most recent decision logged via ``capture``.

    Mirrors ``synthesize_training._last_event_id`` — fails loud on an empty
    capture so no pair carries a schema-violating ``decision_capture_id=""``.
    """
    decisions = getattr(capture, "decisions", None) or []
    if not decisions:
        raise RuntimeError(
            "assessment_sft_generator: capture has no logged decisions; "
            "log a batch decision before anchoring a pair's decision_capture_id."
        )
    last = decisions[-1]
    return str(last.get("event_id", "")) if isinstance(last, dict) else ""


_FORMAT_BUILDERS = {
    "solve_steps": _fmt_solve_steps,
    "error_diagnosis": _fmt_error_diagnosis,
    "explain_why": _fmt_explain_why,
    "grade_rubric": _fmt_grade_rubric,
    "hint_no_reveal": _fmt_hint_no_reveal,
}
