"""Defect F — deterministic ``ed4all objectives restructure`` core (NO LLM).

Restructures an already-synthesized objectives doc WITHOUT a 7B re-roll. It
composes the deterministic cores landed by waves W1-W4 — Defect E (lexical
near-restatement dedup), Defect B (vacuity annotate / drop), Defect A
(chapter-anchored TO re-derivation), Defect D (sub-objective quality) — over an
existing ``synthesized_objectives.json`` and emits a fresh doc that round-trips
the ``--reuse-objectives`` plumbing (``_validate_reuse_objectives_file`` +
``_normalize_to_courseforge_form``). The operator runs it in minutes, then feeds
the output back through ``ed4all run ... --reuse-objectives <out>`` instead of
paying for another nondeterministic course-planner dispatch.

**Zero LLM call sites.** Every stage is deterministic:

* **E merge** — statement-cosine clustering + complete-linkage lexical merge
  (``objective_dedup.cluster_by_cosine`` + ``merge_clusters_lexical``); one
  representative CO survives per near-restatement cluster, losers recorded in the
  report. CO ids are NEVER re-minted — a surviving CO keeps its original id and
  absorbs the losers' cited chunks (grounding preserved, no chunk invented).
* **B annotate / drop** — each CO gets a ``vacuous`` flag from the shared
  content-residual floor (``objective_specificity._content_residual``); with
  ``drop_vacuous`` the V1 hard-fails are removed (keep-≥1 guard).
* **A re-derivation** — COs are anchored to their SemantiK module by cited-chunk
  plurality (``chapter_anchor.assign_cos_to_modules``); one book-ordered terminal
  objective per module with a DETERMINISTIC module-titled statement (the 7B
  author is never called). TO ids are re-minted in book order; ``terminal_id`` is
  re-pointed on every CO; ``child_co_ids`` / ``source_refs`` are rebuilt via the
  lifted ``terminal_children._annotate_terminals_with_children``.
* **D** — sub-objective quality forced on, deterministic (no provider).
* week groups — TO-membership when ``duration_weeks == num_tos`` else ceil-stride
  over the chapter-sorted CO order (single-sourced via
  ``pipeline_tools._week_co_groups``).

**Stop handling.** Every per-CO loop calls ``stop_control.check_stop`` at its unit
boundary, so ``ed4all stop`` pauses a restructure the same way it pauses a live
run. Because the pass is LLM-FREE and pure, there is NO resume sidecar: a paused
restructure simply re-runs from the (immutable) input doc — the recompute is
lossless and cheap, so a checkpoint would be dead weight.

Public surface: :func:`restructure_objectives_doc`.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from lib.generation.stop_control import check_stop

logger = logging.getLogger(__name__)

__all__ = ["RestructureOptions", "restructure_objectives_doc", "RefuseFakeEmbedding"]

#: Stop-sentinel site id for the per-CO restructure loops.
_SITE_ID = "objectives_restructure"

_MINT_METHOD = "restructured_objectives"


class RefuseFakeEmbedding(RuntimeError):
    """Raised when the resolved embedding client is the poisoning ``fake`` kind.

    Defect F's E-merge depends on real statement vectors; a ``fake`` provider
    would silently collapse or scatter clusters, so we fail closed rather than
    emit a mis-deduped doc (mirrors the ``ED4ALL_EMBEDDING_ALLOW_FAKE`` anti-
    poisoning posture on the retrieval read path).
    """


@dataclass
class RestructureOptions:
    """Explicit operator intent — no behavior flags (the CLI IS the switch)."""

    course_name: str
    #: ``True`` drops V1-vacuous COs (keep-≥1); ``False`` only annotates them.
    drop_vacuous: bool = False
    #: Operator ``--weeks`` override; ``None`` → ``max(8, num_tos)``.
    weeks: Optional[int] = None
    #: Provenance label written to ``generated_from`` (usually the input path).
    generated_from: str = ""
    #: Inject an embedding client for tests; ``None`` builds one offline.
    embed: Optional[Any] = None
    #: Inject a DecisionCapture; ``None`` → best-effort internal one.
    capture: Optional[Any] = None
    #: Explicit run id for the stop-sentinel probe (else ``ED4ALL_RUN_ID``).
    run_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Flatten (accept both the group + flat CO shapes).
# --------------------------------------------------------------------------- #

def _flatten_cos(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten ``chapter_objectives`` to a CO list (group OR flat shape).

    Mirrors ``sub_objectives.populate_sub_objectives._iter_cos``: a
    ``[{chapter, objectives: [...]}]`` group list yields the nested objectives; a
    flat ``[CO, ...]`` list yields its ``id``-bearing dicts. COs are returned in
    document order (the chapter signal is recovered from chunk metadata in A, so
    input order does not matter for correctness).
    """
    cos: List[Dict[str, Any]] = []
    for entry in doc.get("chapter_objectives") or []:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("objectives"), list):
            for co in entry["objectives"]:
                if isinstance(co, dict):
                    cos.append(co)
        elif entry.get("id"):  # flat CO list
            cos.append(entry)
    return cos


# --------------------------------------------------------------------------- #
# Embedding client (refuse the fake provider).
# --------------------------------------------------------------------------- #

def _resolve_embed(injected: Optional[Any]) -> Any:
    """Return an embedding client, refusing the ``fake`` provider.

    ``injected`` (tests) is used verbatim; ``None`` builds an offline client via
    ``build_embedding_client(offline=True)``. Either way, a resolved provider
    whose ``kind == "fake"`` raises :class:`RefuseFakeEmbedding`.
    """
    client = injected
    if client is None:
        from lib.embedding.providers import build_embedding_client  # noqa: PLC0415

        client = build_embedding_client(offline=True)
    kind = getattr(getattr(client, "resolved", None), "kind", None)
    if kind == "fake":
        raise RefuseFakeEmbedding(
            "objectives restructure: the resolved embedding provider is the "
            "'fake' kind, which cannot ground the E-merge dedup. Configure a "
            "real ED4ALL_EMBEDDING_PROVIDER (st / local-openai)."
        )
    return client


def _co_statement(co: Dict[str, Any]) -> str:
    return str(co.get("statement") or co.get("text") or "").strip()


# --------------------------------------------------------------------------- #
# E merge — cosine cluster + complete-linkage lexical merge; keep one rep per
# cluster (CO ids never re-minted; losers recorded; grounding preserved).
# --------------------------------------------------------------------------- #

def _e_merge(
    cos: List[Dict[str, Any]],
    embed: Any,
    *,
    capture: Optional[Any],
    run_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collapse near-restatement COs to one representative each.

    Composes ``cluster_by_cosine`` (the 0.88 single-link dedup grouping) and
    ``merge_clusters_lexical`` (Defect E complete-linkage second pass). For each
    final multi-member cluster the best-grounded member survives (keeping its id);
    the losers are dropped and their cited chunks are UNIONED into the survivor so
    no grounding is lost. Partition-safe: never invents a CO.
    """
    import numpy as np

    from lib.objectives.objective_dedup import (  # noqa: PLC0415
        _best_grounded_member,
        cluster_by_cosine,
        merge_clusters_lexical,
        resolve_dedup_threshold,
    )

    report: Dict[str, Any] = {
        "input_cos": len(cos),
        "survivors": len(cos),
        "clusters_merged": 0,
        "cos_dropped": 0,
        "losers": [],
    }
    if len(cos) < 2:
        return list(cos), report

    vecs_np = embed.encode_batch([_co_statement(c) or " " for c in cos])
    vecs: List[List[float]] = np.asarray(vecs_np, dtype=float).tolist()

    clusters, _max_cos, _pairs = cluster_by_cosine(
        vecs, resolve_dedup_threshold(None)
    )
    merged_clusters, _ops, _absorbed, _counts = merge_clusters_lexical(
        clusters, cos, vecs
    )

    survivors: List[Dict[str, Any]] = []
    dropped_idxs: set = set()
    clusters_merged = 0
    for c_idx, cluster in enumerate(merged_clusters):
        # Stop at the cluster boundary — LLM-free, so a plain raise is lossless.
        check_stop(_SITE_ID, c_idx, run_id=run_id)
        if not cluster:
            continue
        if len(cluster) == 1:
            survivors.append(cos[cluster[0]])
            continue
        clusters_merged += 1
        rep_idx = _best_grounded_member(cluster, cos)
        rep = cos[rep_idx]
        rep_chunks = list(rep.get("source_chunk_ids") or [])
        seen = {str(x) for x in rep_chunks}
        for member in cluster:
            if member == rep_idx:
                continue
            dropped_idxs.add(member)
            loser = cos[member]
            # Union the loser's cited chunks into the survivor (grounding kept).
            for ch in loser.get("source_chunk_ids") or []:
                ch_s = str(ch)
                if ch_s and ch_s not in seen:
                    seen.add(ch_s)
                    rep_chunks.append(ch_s)
            report["losers"].append({
                "kept_co_id": str(rep.get("id") or ""),
                "dropped_co_id": str(loser.get("id") or ""),
                "dropped_statement": _co_statement(loser),
            })
        if len(rep_chunks) != len(rep.get("source_chunk_ids") or []):
            rep["source_chunk_ids"] = rep_chunks
        survivors.append(rep)

    report["survivors"] = len(survivors)
    report["clusters_merged"] = clusters_merged
    report["cos_dropped"] = len(dropped_idxs)

    if capture is not None:
        _safe_capture(
            capture,
            decision_type="content_selection",
            decision=(
                f"e_merge:{len(cos)}->{len(survivors)} CO(s) "
                f"({len(dropped_idxs)} near-restatement dropped)"
            ),
            rationale=(
                "Defect-E lexical near-restatement dedup over the existing doc: "
                f"cosine-clustered {len(cos)} CO(s), complete-linkage merged "
                f"{clusters_merged} cluster(s), dropped {len(dropped_idxs)} "
                "loser CO(s) and unioned their cited chunks into each surviving "
                "representative (ids preserved, grounding never lost)."
            ),
        )
    return survivors, report


# --------------------------------------------------------------------------- #
# B annotate / drop — vacuity flag via the shared content-residual floor.
# --------------------------------------------------------------------------- #

def _annotate_vacuity(
    cos: List[Dict[str, Any]],
    *,
    drop_vacuous: bool,
    capture: Optional[Any],
    run_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Stamp each CO ``vacuous`` (V1 content-residual floor); optionally drop.

    Reuses ``objective_specificity._content_residual`` +
    ``DEFAULT_MIN_CONTENT_RESIDUAL`` (the SAME V1 check the gate applies). With
    ``drop_vacuous`` the V1 hard-fails are removed, but a keep-≥1 guard never
    returns an empty CO list (mirrors the in-synthesis filter arm).
    """
    from lib.validators.objective_specificity import (  # noqa: PLC0415
        DEFAULT_MIN_CONTENT_RESIDUAL,
        _content_residual,
    )

    kept: List[Dict[str, Any]] = []
    vacuous_ids: List[str] = []
    for idx, co in enumerate(cos):
        check_stop(_SITE_ID, idx, run_id=run_id)
        residual = _content_residual(_co_statement(co))
        is_vacuous = len(residual) < DEFAULT_MIN_CONTENT_RESIDUAL
        co["vacuous"] = bool(is_vacuous)
        if is_vacuous:
            vacuous_ids.append(str(co.get("id") or ""))
        if drop_vacuous and is_vacuous:
            continue
        kept.append(co)

    dropped: List[str] = []
    if drop_vacuous:
        if not kept and cos:
            # keep-≥1 guard: never annihilate the CO set — keep the input as-is.
            kept = list(cos)
            logger.warning(
                "objectives restructure: --drop-vacuous would remove ALL %d "
                "CO(s); keeping them (keep-≥1 guard).", len(cos),
            )
        else:
            dropped = list(vacuous_ids)

    report: Dict[str, Any] = {
        "vacuous_count": len(vacuous_ids),
        "vacuous_ids": vacuous_ids,
        "dropped_ids": dropped,
        "mode": "drop_vacuous" if drop_vacuous else "annotate_only",
    }
    if capture is not None:
        _safe_capture(
            capture,
            decision_type="content_selection",
            decision=(
                f"vacuity:{len(vacuous_ids)}/{len(cos)} vacuous "
                f"({'dropped ' + str(len(dropped)) if drop_vacuous else 'annotated'})"
            ),
            rationale=(
                "Defect-B V1 content-residual floor over "
                f"{len(cos)} CO(s): flagged {len(vacuous_ids)} as vacuous "
                "(fewer than the min content tokens after Bloom-verb / stopword "
                "/ filler-lexicon subtraction); "
                + (
                    f"dropped {len(dropped)} under --drop-vacuous."
                    if drop_vacuous
                    else "annotate-only, none removed."
                )
            ),
        )
    return kept, report


# --------------------------------------------------------------------------- #
# A re-derivation — chapter-anchored, DETERMINISTIC TO statements only.
# --------------------------------------------------------------------------- #

def _most_common_bloom_verb(cos: List[Dict[str, Any]]) -> str:
    """Most-common member ``bloom_verb`` (capitalized) for a fallback TO stem.

    Falls back to detecting the main Bloom verb off each CO statement, then to a
    neutral ``"Master"`` when nothing resolves. Pure / deterministic (mirrors
    ``pipeline_tools._most_common_bloom_verb``).
    """
    verbs: List[str] = []
    for co in cos:
        if not isinstance(co, dict):
            continue
        verb = str(co.get("bloom_verb") or "").strip().lower()
        if not verb:
            try:
                from lib.ontology.bloom import detect_bloom_level  # noqa: PLC0415

                _lvl, detected = detect_bloom_level(_co_statement(co))
                verb = str(detected or "").strip().lower()
            except Exception:  # noqa: BLE001 — best-effort detection
                verb = ""
        if verb:
            verbs.append(verb)
    if not verbs:
        return "Master"
    return Counter(verbs).most_common(1)[0][0].capitalize()


def _most_common_bloom_level(cos: List[Dict[str, Any]]) -> str:
    levels = [
        str(c.get("bloom_level") or "").strip().lower()
        for c in cos
        if isinstance(c, dict) and str(c.get("bloom_level") or "").strip()
    ]
    if not levels:
        return "understand"
    return Counter(levels).most_common(1)[0][0]


def _derive_terminals_deterministic(
    cos: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    *,
    course_name: str,
    capture: Optional[Any],
    run_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Re-derive book-ordered terminal objectives, one per SemantiK module.

    Assigns every CO to its module by cited-chunk plurality
    (``chapter_anchor.assign_cos_to_modules``), then authors one DETERMINISTIC
    module-titled TO per non-empty module in book order — the 7B author is never
    consulted. TO ids are minted ``TO-NN`` in book order; ``terminal_id`` is set
    on every member CO in place. Anti-fabrication: TO statements only name the
    module + the members' most-common Bloom verb.
    """
    from lib.objectives.chapter_anchor import (  # noqa: PLC0415
        _build_groups,
        assign_cos_to_modules,
    )
    from lib.ontology.learning_objectives import mint_lo_id  # noqa: PLC0415

    result = assign_cos_to_modules(cos, chunks_by_id, all_chunks)

    if result.module_order:
        groups = result.groups
        degraded = None
    else:
        # No module signal (empty / metadata-free chunkset) — deterministic
        # single-terminal degrade so F still emits a valid doc.
        groups = [("__course__", list(range(len(cos))))] if cos else []
        degraded = "no_module_signal"

    terminals: List[Dict[str, Any]] = []
    for m_idx, (module_id, member_idxs) in enumerate(groups, start=1):
        check_stop(_SITE_ID, m_idx - 1, run_id=run_id)
        cluster_cos = [cos[i] for i in member_idxs]
        if module_id == "__course__" or not module_id:
            module_title = course_name or "the course"
        else:
            module_title = result.module_title_by_id.get(module_id) or module_id
        to_id = mint_lo_id("terminal", m_idx)
        statement = (
            f"{_most_common_bloom_verb(cluster_cos)} the concepts and skills of "
            f"{module_title}."
        )
        to: Dict[str, Any] = {
            "id": to_id,
            "statement": statement,
            "bloom_level": _most_common_bloom_level(cluster_cos),
            "source_refs": [],
            "to_synthesis": "restructure_chapter_anchor_deterministic",
        }
        if module_id and module_id != "__course__":
            to["anchor_module_id"] = module_id
            to["anchor_module_title"] = module_title
        terminals.append(to)
        for i in member_idxs:
            cos[i]["terminal_id"] = to_id

    report: Dict[str, Any] = {
        "terminals": len(terminals),
        "modules": len(groups),
        "ties": result.ties,
        "multi_module_cos": result.multi_module_cos,
        "unresolved_cos": len(result.unresolved_co_indices),
        "co_coverage": round(result.co_coverage, 4),
    }
    if degraded:
        report["degraded_reason"] = degraded

    if capture is not None:
        _safe_capture(
            capture,
            decision_type="content_selection",
            decision=(
                f"chapter_anchored_tos:{len(groups)} module(s) -> "
                f"{len(terminals)} deterministic TO(s) over {len(cos)} CO(s)"
            ),
            rationale=(
                "Defect-A deterministic TO re-derivation (no 7B author): "
                f"anchored {len(cos)} CO(s) to {report['modules']} SemantiK "
                f"module(s) by cited-chunk plurality (coverage="
                f"{result.co_coverage:.3f}, {result.ties} tie(s), "
                f"{result.multi_module_cos} multi-module, "
                f"{len(result.unresolved_co_indices)} zero-resolve); one "
                "book-ordered module-titled terminal objective per module."
            ),
        )
    return terminals, report


# --------------------------------------------------------------------------- #
# D — sub-objective quality (forced on, deterministic, no provider).
# --------------------------------------------------------------------------- #

@contextmanager
def _forced_quality() -> Iterator[None]:
    """Force ``ED4ALL_SUB_OBJECTIVE_QUALITY`` on for the populate call.

    Explicit CLI intent — F always runs the Defect-D quality path regardless of
    the operator's ambient env. Restores the prior value afterward.
    """
    from lib.objectives.sub_objectives import (  # noqa: PLC0415
        ENV_SUB_OBJECTIVE_QUALITY,
    )

    prior = os.environ.get(ENV_SUB_OBJECTIVE_QUALITY)
    os.environ[ENV_SUB_OBJECTIVE_QUALITY] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(ENV_SUB_OBJECTIVE_QUALITY, None)
        else:
            os.environ[ENV_SUB_OBJECTIVE_QUALITY] = prior


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #

def _clone_lo(src: Dict[str, Any], hierarchy: str) -> Dict[str, Any]:
    cloned = dict(src)
    cloned["hierarchy_level"] = hierarchy
    return cloned


def _safe_capture(capture: Any, **kwargs: Any) -> None:
    try:
        capture.log_decision(**kwargs)
    except Exception:  # noqa: BLE001 — capture must never break a restructure
        pass


def restructure_objectives_doc(
    doc: Dict[str, Any],
    chunks_by_id: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    *,
    options: RestructureOptions,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Deterministically restructure an objectives doc (Defect F).

    Pipeline (all deterministic; no LLM): flatten → E merge → B annotate/drop →
    A chapter-anchored TO re-derivation → D sub-objective quality → rebuild
    ``learning_outcomes`` + week groups → assemble a ``--reuse-objectives``-shaped
    output doc. Returns ``(new_doc, report)``.
    """
    from MCP.tools.pipeline_tools import _week_co_groups  # noqa: PLC0415
    from lib.objectives.sub_objectives import (  # noqa: PLC0415
        populate_sub_objectives,
    )
    from lib.objectives.terminal_children import (  # noqa: PLC0415
        _annotate_terminals_with_children,
    )

    capture = options.capture
    run_id = options.run_id

    # 1. Flatten (group + flat shapes).
    cos = [dict(c) for c in _flatten_cos(doc)]
    input_co_count = len(cos)
    input_to_count = len(doc.get("terminal_objectives") or [])

    # 2. E merge (embed; refuse fake).
    embed = _resolve_embed(options.embed)
    cos, e_report = _e_merge(cos, embed, capture=capture, run_id=run_id)

    # 3. B annotate (default) / drop-vacuous (V1 only).
    cos, b_report = _annotate_vacuity(
        cos, drop_vacuous=options.drop_vacuous, capture=capture, run_id=run_id
    )

    # 4. A re-derivation — deterministic chapter-anchored TOs; terminal_id set.
    terminals, a_report = _derive_terminals_deterministic(
        cos, chunks_by_id, all_chunks,
        course_name=options.course_name, capture=capture, run_id=run_id,
    )

    num_tos = len(terminals)
    duration_weeks = (
        int(options.weeks) if options.weeks is not None else max(8, num_tos)
    )

    # 5. Week groups — TO-membership when weeks==num_tos else ceil-stride over the
    #    chapter-sorted CO order (single-sourced via _week_co_groups, enabled=True
    #    so F picks TO-membership without the ambient ED4ALL_WEEK_TO_GROUPS flag).
    week_groups = _week_co_groups(cos, terminals, duration_weeks, enabled=True)
    week_mode = (
        "to_membership"
        if (duration_weeks == num_tos and num_tos > 0)
        else "ceil_stride"
    )
    chapter_objectives = [
        {"chapter": f"Week {w}", "objectives": [dict(c) for c in week_groups.get(w, [])]}
        for w in range(1, duration_weeks + 1)
    ]

    # 6. D — sub-objective quality, forced on + deterministic (no provider). The
    #    group-shaped chapter_objectives is mutated in place.
    with _forced_quality():
        d_counters = populate_sub_objectives(
            chapter_objectives=chapter_objectives,
            chunks_by_id=chunks_by_id,
            embedder=None,  # single un-clustered sub-objective; no extra model load
            provider=None,  # deterministic arm only
            capture=capture,
            checkpoint_path=None,  # LLM-free → no sidecar
            run_id=run_id,
        )

    # 7. TO child back-annotation (child_co_ids + source_refs union).
    flat_cos = [c for grp in chapter_objectives for c in grp["objectives"]]
    to_annot = _annotate_terminals_with_children(terminals, flat_cos)

    # 8. learning_outcomes flat list (terminals first, then COs in course order).
    lo_entries: List[Dict[str, Any]] = []
    for to in terminals:
        lo_entries.append(_clone_lo(to, "terminal"))
    for co in flat_cos:
        lo_entries.append(_clone_lo(co, "chapter"))

    new_doc: Dict[str, Any] = {
        "course_name": options.course_name,
        "generated_from": options.generated_from,
        "mint_method": _MINT_METHOD,
        "objectives_source": "operator",
        "duration_weeks": duration_weeks,
        "learning_outcomes": lo_entries,
        "terminal_objectives": [dict(t) for t in terminals],
        "chapter_objectives": chapter_objectives,
    }

    report: Dict[str, Any] = {
        "mint_method": _MINT_METHOD,
        "generated_from": options.generated_from,
        "input_co_count": input_co_count,
        "input_to_count": input_to_count,
        "surviving_co_count": len(flat_cos),
        "num_tos": num_tos,
        "duration_weeks": duration_weeks,
        "week_mode": week_mode,
        "e_merge": e_report,
        "b_vacuity": b_report,
        "a_derivation": a_report,
        "d_sub_objectives": dict(d_counters),
        "to_annotation": to_annot,
    }
    return new_doc, report
