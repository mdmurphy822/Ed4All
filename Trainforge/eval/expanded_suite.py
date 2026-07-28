"""Deterministic, contamination-aware construction of final adapter eval suites.

This module is intentionally offline.  It consumes already-verified assessment,
objective, chunk, and answer-key artifacts; it never calls a model and never
touches workflow state.  A suite is publishable only when every requested split
is full, objective coverage is complete, and all enabled leakage layers pass.
Deficits are first-class output rather than an excuse to duplicate questions
between development and final-test splits.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SPLIT_TARGETS = {
    "checkpoint_dev": 110,
    "objective_heldout": 314,
    "grounding_stress": 75,
    "pedagogy_misconception": 75,
    "out_of_domain": 30,
}
EVAL_ARMS = (
    "base_no_rag",
    "base_rag",
    "sft_no_rag",
    "sft_rag",
    "dpo_no_rag",
    "dpo_rag",
)
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CF_BLOCK_ID_RE = re.compile(r"""data-cf-block-id=["']([^"']+)["']""")
_LOW_SIGNAL_OBJECTIVE_TOKENS = {
    "a", "an", "and", "by", "for", "given", "in", "of", "or", "student",
    "the", "to", "will", "with",
}


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_json(value)
    return hashlib.sha256(payload).hexdigest()


def _sorted_line_hash(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_text(value: Any) -> str:
    text = html.unescape(_TAG_RE.sub(" ", str(value or ""))).lower()
    return " ".join(_TOKEN_RE.findall(text))


def _objective_content_score(statement: str, text: str) -> float:
    """Return deterministic objective support in the content-bearing body."""
    body = text.split("Key Idea:", 1)[-1] if "Key Idea:" in text else text
    objective_tokens = set(_TOKEN_RE.findall(statement.lower()))
    objective_tokens -= _LOW_SIGNAL_OBJECTIVE_TOKENS
    body_tokens = set(_TOKEN_RE.findall(body.lower()))
    if not objective_tokens:
        return 0.0
    return len(objective_tokens & body_tokens) / len(objective_tokens)


def _plain(value: Any) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", str(value or ""))).split())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _objective_map(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(document.get("terminal_outcomes") or []) + list(
        document.get("component_objectives") or []
    )
    return {
        str(row.get("id") or "").upper(): dict(row)
        for row in rows
        if row.get("id")
    }


def _assessment_questions(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assessment in document.get("assessments") or []:
        for question in assessment.get("questions") or []:
            if isinstance(question, dict):
                rows.append(dict(question))
    return rows


def _answer_map(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(document.get("items") or []) + list(document.get("rubric_items") or [])
    return {
        str(row.get("item_id") or row.get("question_id") or ""): dict(row)
        for row in rows
        if row.get("item_id") or row.get("question_id")
    }


def _correct_answers(question: Mapping[str, Any], answer: Mapping[str, Any]) -> list[str]:
    values = [str(v) for v in answer.get("correct_answers") or [] if str(v).strip()]
    if not values and question.get("correct_answer"):
        values = [str(question["correct_answer"])]
    if not values:
        values = [
            str(choice.get("text") or "")
            for choice in question.get("choices") or []
            if isinstance(choice, dict) and choice.get("is_correct")
        ]
    # Constructed-response items carry their grounded model answer in feedback
    # and their analytic scoring contract in ``rubric`` rather than choices.
    if not values and question.get("feedback") and question.get("rubric"):
        values = [str(question["feedback"])]
    return [_plain(value) for value in values if _plain(value)]


def _keypoints(
    question: Mapping[str, Any],
    answer: Mapping[str, Any],
    answers: Sequence[str],
) -> list[str]:
    points: list[str] = []
    for step in answer.get("worked_solution_steps") or []:
        if isinstance(step, dict) and _plain(step.get("text")):
            points.append(_plain(step["text"]))
    rubric = question.get("rubric") or {}
    for criterion in rubric.get("criteria") or []:
        if isinstance(criterion, dict) and _plain(criterion.get("criterion")):
            points.append(_plain(criterion["criterion"]))
    return points or list(answers)


def _difficulty(question: Mapping[str, Any], bloom: str) -> str:
    explicit = str(question.get("difficulty") or "").lower()
    if explicit in {"easy", "medium", "hard"}:
        return explicit
    if bloom in {"analyze", "evaluate", "create"}:
        return "hard"
    if bloom in {"apply", "understand"}:
        return "medium"
    return "easy"


def _source_refs(
    chunk_ids: Sequence[str], chunks: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for chunk_id in sorted(set(map(str, chunk_ids))):
        chunk = chunks.get(chunk_id) or {}
        source = chunk.get("source") or {}
        refs.append(
            {
                "chunk_id": chunk_id,
                "item_path": source.get("item_path"),
                "section_heading": source.get("section_heading"),
                "char_span": source.get("char_span"),
                "source_ids": sorted(
                    {
                        str(ref.get("sourceId"))
                        for ref in source.get("source_references") or []
                        if isinstance(ref, dict) and ref.get("sourceId")
                    }
                ),
            }
        )
    return refs


def build_candidates(
    assessments: Mapping[str, Any],
    answer_key: Mapping[str, Any],
    objectives: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Normalize verified assessment items into evaluation candidates."""
    objective_by_id = _objective_map(objectives)
    answers_by_id = _answer_map(answer_key)
    chunks_by_id = {str(row.get("id")): row for row in chunks if row.get("id")}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for question in _assessment_questions(assessments):
        question_id = str(question.get("question_id") or "")
        objective_id = str(question.get("objective_id") or "").upper()
        objective = objective_by_id.get(objective_id)
        answer_row = answers_by_id.get(question_id, {})
        answers = _correct_answers(question, answer_row)
        chunk_ids = list(
            answer_row.get("source_chunk_ids") or question.get("source_chunks") or []
        )
        reasons = []
        if not question_id:
            reasons.append("missing_question_id")
        if not _plain(question.get("stem")):
            reasons.append("missing_stem")
        if objective is None:
            reasons.append("unknown_objective")
        if not answers:
            reasons.append("missing_answer")
        if not chunk_ids or any(str(cid) not in chunks_by_id for cid in chunk_ids):
            reasons.append("unresolved_source_chunk")
        if reasons:
            rejected.append({"question_id": question_id, "reason": ",".join(reasons)})
            continue
        bloom = str(question.get("bloom_level") or objective.get("bloom_level") or "")
        source_refs = _source_refs(chunk_ids, chunks_by_id)
        item = {
            "question_id": question_id,
            "question": _plain(question["stem"]),
            "canonical_objective_id": objective_id,
            "canonical_objective": str(objective.get("statement") or ""),
            "bloom_level": bloom,
            "answers": answers,
            "expected_keypoints": _keypoints(question, answer_row, answers),
            "source_refs": source_refs,
            "content_type": str(
                question.get("item_subtype")
                or question.get("question_type")
                or "assessment_item"
            ),
            "difficulty": _difficulty(question, bloom),
            "retrieval": {
                "required": True,
                "expected_chunk_ids": sorted(set(map(str, chunk_ids))),
                "arms": list(EVAL_ARMS),
            },
            "provenance": {
                "method": "verified_assessment_reuse",
                "assessment_id": question_id,
                "source_artifact": "assessment_items.json",
                "answer_artifact": "answer_key.json",
                "manual_review_status": "inherited_verified_source",
            },
        }
        item["fingerprint"] = _sha256(item)
        candidates.append(item)
    candidates.sort(key=lambda row: (row["canonical_objective_id"], row["fingerprint"]))
    return candidates, rejected


def _assign(
    candidates: Sequence[dict[str, Any]],
    *,
    objective_ids: Sequence[str],
    targets: Mapping[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Assign disjoint fingerprints; objective-heldout always gets priority."""
    by_objective: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_objective[row["canonical_objective_id"]].append(row)
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}
    used: set[str] = set()
    for objective_id in sorted(objective_ids):
        options = by_objective.get(objective_id, [])
        if options:
            row = dict(options[0])
            row["split"] = "objective_heldout"
            assigned["objective_heldout"].append(row)
            used.add(row["fingerprint"])
    remaining = [row for row in candidates if row["fingerprint"] not in used]
    for split in (
        "checkpoint_dev",
        "grounding_stress",
        "pedagogy_misconception",
        "out_of_domain",
    ):
        for original in remaining[: int(targets.get(split, 0))]:
            row = dict(original)
            row["split"] = split
            assigned[split].append(row)
            used.add(row["fingerprint"])
        remaining = [row for row in remaining if row["fingerprint"] not in used]
    deficits = {
        split: max(0, int(target) - len(assigned.get(split, [])))
        for split, target in targets.items()
    }
    return assigned, deficits


def _pair_texts(pairs: Iterable[Mapping[str, Any]]) -> Iterable[tuple[str, set[str]]]:
    for pair in pairs:
        text = " ".join(
            str(pair.get(key) or "")
            for key in ("prompt", "completion", "chosen", "rejected")
        )
        source_ids = {
            str(value)
            for key in ("source_chunk_id", "chunk_id")
            for value in ([pair.get(key)] if pair.get(key) else [])
        }
        source_ids.update(map(str, pair.get("source_chunk_ids") or []))
        yield text, source_ids


def leakage_findings(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    training_pairs: Sequence[Mapping[str, Any]],
    *,
    semantic_similarity: Callable[[str, str], float] | None = None,
    semantic_floor: float = 0.92,
    fuzzy_floor: float = 0.92,
) -> list[dict[str, Any]]:
    """Check exact/normalized/fuzzy/semantic/source/keypoint contamination."""
    findings: list[dict[str, Any]] = []
    train = list(_pair_texts(training_pairs))
    train_norm = [
        (text, normalize_text(text), sources) for text, sources in train
    ]
    for split, items in splits.items():
        if split == "checkpoint_dev":
            continue
        for item in items:
            fields = [item["question"], *item.get("answers", []), *item.get("expected_keypoints", [])]
            refs = {
                ref["chunk_id"] for ref in item.get("source_refs", []) if ref.get("chunk_id")
            }
            for index, (train_raw, train_text, train_sources) in enumerate(train_norm):
                if refs & train_sources:
                    findings.append(
                        {"split": split, "question_id": item["question_id"],
                         "layer": "source_id", "training_index": index}
                    )
                    continue
                for field in fields:
                    norm = normalize_text(field)
                    if not norm:
                        continue
                    if norm == train_text or norm in train_text or train_text in norm:
                        layer = "exact_or_normalized" if norm == train_text else "keypoint_containment"
                        findings.append(
                            {"split": split, "question_id": item["question_id"],
                             "layer": layer, "training_index": index}
                        )
                        break
                    if SequenceMatcher(None, norm, train_text).ratio() >= fuzzy_floor:
                        findings.append(
                            {"split": split, "question_id": item["question_id"],
                             "layer": "fuzzy", "training_index": index}
                        )
                        break
                    if (
                        semantic_similarity
                        and semantic_similarity(field, train_raw) >= semantic_floor
                    ):
                        findings.append(
                            {"split": split, "question_id": item["question_id"],
                             "layer": "semantic", "training_index": index}
                        )
                        break
    return findings


def arm_cases(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project identical immutable questions across every comparison arm."""
    cases: list[dict[str, Any]] = []
    for split, items in suite.get("splits", {}).items():
        if split == "checkpoint_dev":
            continue
        for item in items:
            for arm in EVAL_ARMS:
                cases.append(
                    {
                        "case_id": f"{item['fingerprint']}:{arm}",
                        "item_fingerprint": item["fingerprint"],
                        "question_id": item["question_id"],
                        "split": split,
                        "arm": arm,
                        "use_adapter": not arm.startswith("base_"),
                        "adapter_stage": (
                            "dpo" if arm.startswith("dpo_")
                            else "sft" if arm.startswith("sft_")
                            else None
                        ),
                        "use_retrieval": arm.endswith("_rag") and not arm.endswith("_no_rag"),
                    }
                )
    return cases


def _authoring_queue(
    chunks: Sequence[Mapping[str, Any]],
    objectives: Mapping[str, Any],
    dev_chunk_ids: set[str],
    final_chunk_ids: set[str],
    family_by_chunk: Mapping[str, str],
    deficits: Mapping[str, int],
) -> dict[str, Any]:
    """Plan drafting under the frozen dev/final source-family reuse policy."""
    objective_by_id = _objective_map(objectives)
    dev_pool: list[tuple[str, str, str, float]] = []
    final_pool: list[tuple[str, str, str, float]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id or chunk_id not in dev_chunk_ids | final_chunk_ids:
            continue
        refs = [
            str(value).upper()
            for value in chunk.get("learning_outcome_refs") or []
            if str(value).upper() in objective_by_id
        ]
        # Aggregate overview chunks label themselves with dozens or hundreds
        # of objectives and are not valid evidence for any one objective.
        if not refs or len(set(refs)) > 5:
            continue
        scored = sorted(
            (
                _objective_content_score(
                    str(objective_by_id[objective_id].get("statement") or ""),
                    str(chunk.get("text") or ""),
                ),
                objective_id,
            )
            for objective_id in set(refs)
        )
        qualified = [
            (score, objective_id)
            for score, objective_id in reversed(scored)
            if score >= 0.25
        ]
        family_id = family_by_chunk.get(chunk_id, chunk_id)
        destination = dev_pool if chunk_id in dev_chunk_ids else final_pool
        for score, objective_id in qualified[:2]:
            destination.append((family_id, chunk_id, objective_id, score))
    dev_pool.sort()
    final_pool.sort()
    queued: list[dict[str, Any]] = []
    intents = {
        "checkpoint_dev": "independent objective-aligned problem",
        "grounding_stress": "retrieval distractor/conflict/citation stress probe",
        "pedagogy_misconception": "Bloom-honest misconception diagnosis probe",
    }
    used_families: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"splits": set(), "objectives": set()}
    )

    def append(split: str, ordinal: int, opportunity: tuple[str, str, str, float]) -> None:
        family_id, chunk_id, objective_id, objective_score = opportunity
        used_families[family_id]["splits"].add(split)
        used_families[family_id]["objectives"].add(objective_id)
        queued.append(
            {
                "candidate_id": f"{split}-{ordinal + 1:04d}",
                "target_split": split,
                "source_chunk_id": chunk_id,
                "source_family_id": family_id,
                "canonical_objective_id": objective_id,
                "canonical_objective": objective_by_id[objective_id].get("statement"),
                "bloom_level": objective_by_id[objective_id].get("bloom_level"),
                "objective_content_score": objective_score,
                "intent": intents[split],
                "status": "awaiting_license_clean_authoring_and_manual_review",
            }
        )

    dev_seen: set[str] = set()
    dev_ordinal = 0
    for opportunity in dev_pool:
        family_id = opportunity[0]
        if family_id in dev_seen:
            continue
        append("checkpoint_dev", dev_ordinal, opportunity)
        dev_seen.add(family_id)
        dev_ordinal += 1
        if dev_ordinal == int(deficits.get("checkpoint_dev", 0)):
            break

    final_ordinals = {"grounding_stress": 0, "pedagogy_misconception": 0}
    final_cursor = 0
    while any(
        final_ordinals[split] < int(deficits.get(split, 0))
        for split in final_ordinals
    ):
        progressed = False
        for split in ("grounding_stress", "pedagogy_misconception"):
            if final_ordinals[split] >= int(deficits.get(split, 0)):
                continue
            selected_for_split = False
            for require_unused_family in (True, False):
                for index in range(final_cursor, len(final_pool)):
                    family_id, _, objective_id, _ = final_pool[index]
                    use = used_families[family_id]
                    if (
                        "checkpoint_dev" in use["splits"]
                        or len(use["splits"]) >= 2
                        or split in use["splits"]
                        or objective_id in use["objectives"]
                        or (require_unused_family and bool(use["splits"]))
                    ):
                        continue
                    append(split, final_ordinals[split], final_pool[index])
                    final_ordinals[split] += 1
                    final_pool.pop(index)
                    progressed = True
                    selected_for_split = True
                    break
                if selected_for_split:
                    break
        if not progressed:
            break
    requested_course = sum(
        int(deficits.get(name, 0))
        for name in ("checkpoint_dev", "grounding_stress", "pedagogy_misconception")
    )
    return {
        "provider_policy": "license_clean_local_or_together_only",
        "active_model_seat_must_not_be_shared": True,
        "course_candidate_count": len(queued),
        "course_candidate_deficit": max(0, requested_course - len(queued)),
        "source_reuse_policy": {
            "name": "dev_final_disjoint_family_max2_v1",
            "dev_final_family_disjoint": True,
            "training_disjoint": True,
            "final_max_items_per_family": 2,
            "second_item_requires_distinct_objective_task_and_failure_mode": True,
            "cross_split_normalized_fuzzy_semantic_dedup_required": True,
        },
        "out_of_domain_requested": int(deficits.get("out_of_domain", 0)),
        "out_of_domain_source_policy": (
            "real reviewed chunks from other licensed courses; never fabricated target-course facts"
        ),
        "required_validation": [
            "schema",
            "manual_review",
            "objective_and_bloom",
            "answer_and_keypoints",
            "citation_anchor",
            "exact_normalized_fuzzy_semantic_leakage",
            "source_id_and_keypoint_leakage",
        ],
        "candidates": queued,
    }


def _merge_authored_items(
    splits: dict[str, list[dict[str, Any]]],
    authored_items: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
) -> list[dict[str, str]]:
    """Promote only preassigned, reviewed authored items into remaining slots."""
    rejected: list[dict[str, str]] = []
    used = {
        str(row.get("fingerprint"))
        for rows in splits.values()
        for row in rows
        if row.get("fingerprint")
    }
    allowed = {"checkpoint_dev", "grounding_stress", "pedagogy_misconception"}
    for source in sorted(
        authored_items,
        key=lambda row: (str(row.get("split")), str(row.get("question_id"))),
    ):
        row = dict(source)
        split = str(row.get("split") or "")
        reason = ""
        if split not in allowed:
            reason = "invalid_authored_split"
        elif row.get("provenance", {}).get("review_status") != "QUALIFIED_AUTOMATIC_REVIEW":
            reason = "review_not_qualified"
        elif not row.get("question") or not row.get("answers") or not row.get("expected_keypoints"):
            reason = "incomplete_scoring_contract"
        elif not row.get("source_refs"):
            reason = "missing_source_refs"
        elif not row.get("fingerprint") or row["fingerprint"] in used:
            reason = "missing_or_duplicate_fingerprint"
        elif len(splits[split]) >= int(targets.get(split, 0)):
            reason = "split_already_full"
        if reason:
            rejected.append(
                {"question_id": str(row.get("question_id") or ""), "reason": reason}
            )
            continue
        splits[split].append(row)
        used.add(str(row["fingerprint"]))
    return rejected


def _reviewed_ood_items(
    document: Mapping[str, Any], target: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Convert reviewed foreign-course refusal probes into immutable OOD cases."""
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    foreign_slug = str(document.get("course_slug") or "foreign-course")
    for probe in document.get("probes") or []:
        authoring = probe.get("authoring") or {}
        dry_run = probe.get("dry_run") or {}
        probe_id = str(probe.get("probe_id") or "")
        if (
            authoring.get("reviewed_by") in {None, "", "PENDING_REVIEW"}
            or authoring.get("status") != "reviewed"
        ):
            rejected.append({"question_id": probe_id, "reason": "not_operator_reviewed"})
            continue
        if dry_run.get("top_passage_answers") is not False:
            rejected.append({"question_id": probe_id, "reason": "foreign_dry_run_not_verified"})
            continue
        question = _plain(probe.get("question_text"))
        rationale = _plain(probe.get("why_unanswerable"))
        if not question or not rationale:
            rejected.append({"question_id": probe_id, "reason": "missing_ood_contract"})
            continue
        item = {
            "question_id": probe_id,
            "question": question,
            "canonical_objective_id": "__OUT_OF_DOMAIN__",
            "canonical_objective": "Recognize when the supplied corpus cannot support an answer.",
            "bloom_level": "evaluate",
            "answers": [
                "State that the supplied course evidence does not support an answer."
            ],
            "expected_keypoints": [
                "Do not fabricate an answer.",
                "Identify that the question is outside the supplied course scope.",
                rationale,
            ],
            "source_refs": [
                {
                    "foreign_course": foreign_slug,
                    "chunk_id": str(dry_run.get("top_chunk_id") or ""),
                    "retrieval_engine": dry_run.get("engine"),
                }
            ],
            "content_type": "out_of_domain_refusal",
            "difficulty": "hard",
            "retrieval": {
                "required": True,
                "expected_chunk_ids": [],
                "expected_behavior": "abstain_without_fabrication",
                "arms": list(EVAL_ARMS),
            },
            "split": "out_of_domain",
            "provenance": {
                "method": "reviewed_foreign_course_refusal_probe",
                "foreign_course": foreign_slug,
                "reviewed_by": authoring.get("reviewed_by"),
                "foreign_probe_id": probe_id,
            },
        }
        item["fingerprint"] = _sha256(item)
        rows.append(item)
        if len(rows) == target:
            break
    return rows, rejected


def _source_family_closure(
    chunks: Sequence[Mapping[str, Any]], seed_ids: set[str]
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Close over explicit block IDs or proven contiguous split descendants.

    ``source_references`` is deliberately ignored: current IMSCC chunks carry
    aggregate citation lists, so transitive closure over those references
    incorrectly merges almost the entire course.
    """
    by_id = {str(row.get("id")): row for row in chunks if row.get("id")}
    block_groups: dict[str, list[str]] = defaultdict(list)
    for chunk_id, row in by_id.items():
        match = _CF_BLOCK_ID_RE.search(str(row.get("html") or ""))
        if match:
            block_groups[match.group(1)].append(chunk_id)
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_evidence: list[dict[str, Any]] = []
    for chunk_id, row in by_id.items():
        parent_id = str(row.get("follows_chunk") or "")
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        source = row.get("source") or {}
        parent_source = parent.get("source") or {}
        span = source.get("char_span")
        parent_span = parent_source.get("char_span")
        if (
            source.get("item_path") == parent_source.get("item_path")
            and isinstance(span, list)
            and len(span) == 2
            and isinstance(parent_span, list)
            and len(parent_span) == 2
            and span[0] in {parent_span[1], parent_span[1] + 1}
        ):
            adjacency[chunk_id].add(parent_id)
            adjacency[parent_id].add(chunk_id)
            edge_evidence.append(
                {
                    "parent_chunk_id": parent_id,
                    "child_chunk_id": chunk_id,
                    "item_path": source.get("item_path"),
                    "parent_char_span": parent_span,
                    "child_char_span": span,
                    "boundary_delta": span[0] - parent_span[1],
                    "reason": "direct_follows_same_item_path_contiguous_char_span",
                }
            )
    reserved = set(seed_ids)
    changed = True
    while changed:
        changed = False
        for ids in block_groups.values():
            if reserved.intersection(ids) and not set(ids).issubset(reserved):
                reserved.update(ids)
                changed = True
        for chunk_id in tuple(reserved):
            siblings = adjacency.get(chunk_id, set())
            if not siblings.issubset(reserved):
                reserved.update(siblings)
                changed = True
    derived = sorted(reserved - seed_ids)
    sizes = [len(ids) for ids in block_groups.values()]
    audit = {
        "contract": "cf_block_id_or_contiguous_follows_v1",
        "source_references_used": False,
        "block_id_family_count": len(block_groups),
        "block_id_multi_member_family_count": sum(size > 1 for size in sizes),
        "block_id_max_family_size": max(sizes, default=0),
        "contiguous_follows_edge_count": sum(map(len, adjacency.values())) // 2,
        "contiguous_follows_evidence": sorted(
            edge_evidence,
            key=lambda row: (row["parent_chunk_id"], row["child_chunk_id"]),
        ),
        "contiguous_follows_evidence_sha256": _sha256(
            sorted(
                edge_evidence,
                key=lambda row: (row["parent_chunk_id"], row["child_chunk_id"]),
            )
        ),
        "derived_count": len(derived),
        "status": "approved",
    }
    return sorted(reserved), derived, audit


def _source_family_map(
    chunks: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Map chunks to canonical family hashes under the approved contract."""
    by_id = {str(row.get("id")): row for row in chunks if row.get("id")}
    parent = {chunk_id: chunk_id for chunk_id in by_id}

    def find(chunk_id: str) -> str:
        while parent[chunk_id] != chunk_id:
            parent[chunk_id] = parent[parent[chunk_id]]
            chunk_id = parent[chunk_id]
        return chunk_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    block_owner: dict[str, str] = {}
    for chunk_id, row in by_id.items():
        match = _CF_BLOCK_ID_RE.search(str(row.get("html") or ""))
        if match:
            owner = block_owner.setdefault(match.group(1), chunk_id)
            union(owner, chunk_id)
        follows_id = str(row.get("follows_chunk") or "")
        follows = by_id.get(follows_id)
        if follows is None:
            continue
        source, follows_source = row.get("source") or {}, follows.get("source") or {}
        span, follows_span = source.get("char_span"), follows_source.get("char_span")
        if (
            source.get("item_path") == follows_source.get("item_path")
            and isinstance(span, list)
            and len(span) == 2
            and isinstance(follows_span, list)
            and len(follows_span) == 2
            and span[0] in {follows_span[1], follows_span[1] + 1}
        ):
            union(chunk_id, follows_id)
    components: dict[str, list[str]] = defaultdict(list)
    for chunk_id in by_id:
        components[find(chunk_id)].append(chunk_id)
    result: dict[str, str] = {}
    for members in components.values():
        family_id = f"family-{_sha256(sorted(members))[:16]}"
        result.update({chunk_id: family_id for chunk_id in members})
    return result


def _assessment_surface_chunk_ids(
    items: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]
) -> set[str]:
    surfaces = [
        (str(chunk["id"]), normalize_text(chunk.get("text")))
        for chunk in chunks
        if chunk.get("id") and chunk.get("chunk_type") == "assessment_item"
    ]
    return {
        chunk_id
        for item in items
        for chunk_id, text in surfaces
        if normalize_text(item.get("question"))
        and normalize_text(item.get("question")) in text
    }


def build_suite(
    *,
    assessments_path: Path,
    answer_key_path: Path,
    objectives_path: Path,
    chunks_path: Path,
    training_pair_paths: Sequence[Path] = (),
    authored_items_path: Path | None = None,
    ood_probes_path: Path | None = None,
    targets: Mapping[str, int] = SPLIT_TARGETS,
) -> dict[str, Any]:
    assessments = _read_json(assessments_path)
    answer_key = _read_json(answer_key_path)
    objectives = _read_json(objectives_path)
    chunks = _read_jsonl(chunks_path)
    candidates, rejected = build_candidates(assessments, answer_key, objectives, chunks)
    objective_ids = sorted(_objective_map(objectives))
    splits, deficits = _assign(candidates, objective_ids=objective_ids, targets=targets)
    authored_rejected: list[dict[str, str]] = []
    if authored_items_path is not None:
        authored_rejected = _merge_authored_items(
            splits, _read_jsonl(authored_items_path), targets
        )
    ood_rejected: list[dict[str, str]] = []
    if ood_probes_path is not None:
        ood_rows, ood_rejected = _reviewed_ood_items(
            _read_json(ood_probes_path), int(targets.get("out_of_domain", 0))
        )
        splits["out_of_domain"] = ood_rows
    deficits = {
        split: max(0, int(target) - len(splits.get(split, [])))
        for split, target in targets.items()
    }
    pairs = [row for path in training_pair_paths for row in _read_jsonl(path)]
    findings = leakage_findings(splits, pairs)
    source_paths = {
        "assessments": assessments_path,
        "answer_key": answer_key_path,
        "objectives": objectives_path,
        "chunks": chunks_path,
    }
    source_paths.update(
        {f"training_pairs_{index}": path for index, path in enumerate(training_pair_paths)}
    )
    if authored_items_path is not None:
        source_paths["authored_items"] = authored_items_path
    if ood_probes_path is not None:
        source_paths["ood_probes"] = ood_probes_path
    sources = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    assessment_chunk_ids = sorted(
        str(row["id"])
        for row in chunks
        if row.get("id") and row.get("chunk_type") == "assessment_item"
    )
    missing_objectives = sorted(
        set(objective_ids)
        - {row["canonical_objective_id"] for row in splits["objective_heldout"]}
    )
    target_chunk_ids = {str(row["id"]) for row in chunks if row.get("id")}
    final_reserved_chunk_ids = {
        str(ref["chunk_id"])
        for split, rows in splits.items()
        for row in rows
        for ref in row.get("source_refs", [])
        if (
            split != "checkpoint_dev"
            and ref.get("chunk_id")
            and str(ref["chunk_id"]) in target_chunk_ids
        )
    }
    family_closed_final_ids, derived_family_ids, family_audit = _source_family_closure(
        chunks, final_reserved_chunk_ids
    )
    training_exclusion_ids = sorted(
        set(assessment_chunk_ids) | set(family_closed_final_ids)
    )
    family_by_chunk = _source_family_map(chunks)
    dev_surface_ids = _assessment_surface_chunk_ids(
        splits["checkpoint_dev"], chunks
    )
    final_surface_ids = _assessment_surface_chunk_ids(
        splits["objective_heldout"], chunks
    )
    dev_authoring_ids = (
        dev_surface_ids - final_surface_ids
    ) & set(training_exclusion_ids)
    final_authoring_ids = (
        set(family_closed_final_ids) - set(assessment_chunk_ids)
    )
    dev_authoring_families = sorted(
        {family_by_chunk[chunk_id] for chunk_id in dev_authoring_ids}
    )
    final_authoring_families = sorted(
        {family_by_chunk[chunk_id] for chunk_id in final_authoring_ids}
    )
    source_reuse_policy = {
        "schema_version": "1.0",
        "name": "dev_final_disjoint_family_max2_v1",
        "frozen_exclusion_chunk_count": len(training_exclusion_ids),
        "frozen_exclusion_chunk_ids_sha256": _sha256(training_exclusion_ids),
        "dev_final_family_disjoint": not bool(
            set(dev_authoring_families) & set(final_authoring_families)
        ),
        "training_disjoint": True,
        "dev_authoring_chunk_ids": sorted(dev_authoring_ids),
        "dev_authoring_family_ids": dev_authoring_families,
        "final_authoring_chunk_ids": sorted(final_authoring_ids),
        "final_authoring_family_ids": final_authoring_families,
        "final_max_items_per_family": 2,
        "second_item_requires_distinct_objective_task_and_failure_mode": True,
        "cross_split_normalized_fuzzy_semantic_dedup_required": True,
    }
    source_reuse_policy["policy_fingerprint"] = _sha256(source_reuse_policy)
    split_fingerprints = {
        split: _sha256([row["fingerprint"] for row in rows])
        for split, rows in splits.items()
    }
    suite_gate_passed = (
        not any(deficits.values()) and not missing_objectives and not findings
    )
    exclusion_registry = {
        "schema_version": "1.0",
        "required": True,
        "ready": suite_gate_passed,
        "reason": (
            "assessment surfaces and final-evaluation source passages are "
            "reserved from synthesis"
        ),
        "assessment_chunk_ids": assessment_chunk_ids,
        "assessment_chunk_ids_sorted_newline_sha256": _sorted_line_hash(
            assessment_chunk_ids
        ),
        "final_source_chunk_ids": sorted(final_reserved_chunk_ids),
        "derived_source_family_chunk_ids": derived_family_ids,
        "family_closed_final_source_chunk_ids": family_closed_final_ids,
        "family_contract": family_audit,
        "source_reuse_policy": source_reuse_policy,
        "chunk_ids": training_exclusion_ids,
        "chunk_ids_sha256": _sha256(training_exclusion_ids),
        "split_fingerprints": split_fingerprints,
        "source_hashes": sources,
    }
    exclusion_registry["registry_fingerprint"] = _sha256(exclusion_registry)
    authoring_queue = _authoring_queue(
        chunks,
        objectives,
        dev_authoring_ids,
        final_authoring_ids,
        family_by_chunk,
        deficits,
    )
    suite: dict[str, Any] = {
        "schema_version": "1.0",
        "targets": dict(targets),
        "splits": splits,
        "evaluation_arms": list(EVAL_ARMS),
        "source_hashes": sources,
        "training_input_exclusion": exclusion_registry,
        "authoring_queue": authoring_queue,
        "candidate_count": len(candidates),
        "rejected_candidates": rejected,
        "rejected_authored_items": authored_rejected,
        "rejected_ood_probes": ood_rejected,
        "deficits": deficits,
        "missing_objective_ids": missing_objectives,
        "leakage": {
            "layers": [
                "exact",
                "normalized",
                "fuzzy",
                "semantic",
                "source_id",
                "keypoint",
            ],
            "semantic_status": "pending_external_embedder_validation",
            "findings": findings,
        },
    }
    suite["ready"] = suite_gate_passed
    suite["suite_fingerprint"] = _sha256(suite)
    suite["arm_case_count"] = len(arm_cases(suite))
    return suite


def write_suite(suite: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "expanded_eval_suite.json"
    path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "suite_file": path.name,
        "suite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "suite_fingerprint": suite["suite_fingerprint"],
        "ready": suite["ready"],
        "split_counts": {
            name: len(rows) for name, rows in suite.get("splits", {}).items()
        },
        "deficits": suite["deficits"],
        "source_hashes": suite["source_hashes"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--objectives", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--training-pairs", type=Path, action="append", default=[])
    parser.add_argument("--authored-items", type=Path)
    parser.add_argument("--ood-probes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    suite = build_suite(
        assessments_path=args.assessments,
        answer_key_path=args.answer_key,
        objectives_path=args.objectives,
        chunks_path=args.chunks,
        training_pair_paths=args.training_pairs,
        authored_items_path=args.authored_items,
        ood_probes_path=args.ood_probes,
    )
    path = write_suite(suite, args.output_dir)
    print(json.dumps({"path": str(path), "ready": suite["ready"], "deficits": suite["deficits"]}))
    return 0 if suite["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
