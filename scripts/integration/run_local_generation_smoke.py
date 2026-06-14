#!/usr/bin/env python3
"""Local 7B course-generation integration harness (end-to-end, no-stubs).

Drives the FULLY-LOCAL course-generation pipeline end-to-end against a real DART
chunkset and VALIDATES every synthesized block. This is the success-condition
harness for the local-generation goal: it answers "does the 7B path produce
GROUNDED + PROVENANCED objectives and blocks?" with live measured numbers.

It reuses the real pipeline helpers — it does NOT reimplement synthesis:

  STAGE 1  Objective synthesis (W2 window synthesis)
           ->  MCP.tools.pipeline_tools registry["plan_course_structure"]
               (which calls _run_stage2_window_synthesis over the chunkset)
           ->  synthesized_objectives.json with grounded TO/CO + source_chunk_ids

  STAGE 2  Content generation two-pass (real CourseforgeRouter)
           ->  route_with_self_consistency  (outline tier, local 7B)
           ->  route_rewrite_with_remediation (rewrite tier, local 7B)
           ->  _emit_block_synthesis_manifest  (W3 provenance sidecar)
           feeding REAL dart chunks as source_chunks=[{id,text}] so the model
           cites resolvable chunk ids in key_claims[].source_chunk_ids[].

  STAGE 3  Validation (the success criterion)
           ->  ManifestCompletenessValidator   (W3 provenance/resolution)
           ->  run_generation_quality_eval      (W5 harness: block-prose +
               objective NLI entailment, provenance, bloom, dedup, coverage)

The course slug + chunkset path are CLI args (no hardcoded course slugs in
tracked code). All synthesis routes to the local Ollama 7B; the harness asserts
ZERO cloud (no ANTHROPIC_API_KEY) before running.

Usage:
    .venv/bin/python scripts/integration/run_local_generation_smoke.py \
        --chunks tests/fixtures/retrieval/mini_course/dart_chunks/chunks.jsonl \
        --course-slug mini-smoke \
        --max-chunks 10 \
        --base-url http://localhost:11434/v1 \
        --model qwen2.5-7b-8k
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root on sys.path (scripts/integration/<this>).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# All-local environment (set BEFORE any pipeline import reads os.environ).
# --------------------------------------------------------------------------- #

def _configure_local_env(
    base_url: str, model: str, *, num_ctx: int, block_routing: Optional[str] = None
) -> None:
    """Set the all-local authoring-route env. Fails closed on cloud key."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is set — refusing to run the 'all-local' smoke "
            "with a cloud key present (it would not prove the local path). "
            "Unset ANTHROPIC_API_KEY and re-run."
        )
    # Global routing.
    os.environ["LLM_PROVIDER"] = "local"
    os.environ["LLM_MODE"] = "local"
    # Authoring-route providers (explicit single-switch; W1 also honors these
    # via WorkflowRunner._apply_authoring_route_env when run through the runner).
    for var in (
        "COURSEFORGE_PROVIDER",
        "COURSEPLANNER_PROVIDER",
        "TEXTBOOK_SYNTHESIS_PROVIDER",
        "COURSEFORGE_OUTLINE_PROVIDER",
        "COURSEFORGE_REWRITE_PROVIDER",
    ):
        os.environ[var] = "local"
    # Local server connection (shared by every synthesis surface).
    os.environ["LOCAL_SYNTHESIS_BASE_URL"] = base_url
    os.environ["LOCAL_SYNTHESIS_MODEL"] = model
    os.environ.setdefault("LOCAL_SYNTHESIS_API_KEY", "local")
    # Per-tier model pins (so outline + rewrite tiers use the same 8k model).
    os.environ["COURSEFORGE_OUTLINE_MODEL"] = model
    os.environ["COURSEFORGE_REWRITE_MODEL"] = model
    os.environ["TEXTBOOK_SYNTHESIS_MODEL"] = model
    # Two-pass + block-emission gates.
    os.environ["COURSEFORGE_TWO_PASS"] = "true"
    os.environ["COURSEFORGE_EMIT_BLOCKS"] = "true"
    # W2 grammar mode: json_object is the safe default for Ollama legacy
    # `format: json` (full json_schema requires Ollama 0.5+ schema support).
    os.environ.setdefault("COURSEFORGE_OUTLINE_GRAMMAR_MODE", "json_object")
    os.environ.setdefault("TEXTBOOK_SYNTHESIS_GRAMMAR_MODE", "json_object")
    # Serving-window budget (8k model).
    os.environ["ED4ALL_ANSWER_NUM_CTX"] = str(num_ctx)
    os.environ["TEXTBOOK_SYNTHESIS_NUM_CTX"] = str(num_ctx)
    # NLI on GPU (validation speed). Honored when the caller exported it; set a
    # default here too.
    os.environ.setdefault("ED4ALL_NLI_DEVICE", "cuda")
    # Keep regen budgets modest so a smoke run terminates in reasonable time.
    os.environ.setdefault("COURSEFORGE_OUTLINE_REGEN_BUDGET", "3")
    os.environ.setdefault("COURSEFORGE_REWRITE_REGEN_BUDGET", "3")
    os.environ.setdefault("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")
    # Optional block-routing override so the authoring tiers use the configured
    # model (the shipped block_routing.yaml pins q4 models that WIN over the
    # COURSEFORGE_*_MODEL env vars per the documented resolution priority).
    if block_routing:
        os.environ["COURSEFORGE_BLOCK_ROUTING_PATH"] = str(Path(block_routing).resolve())


# --------------------------------------------------------------------------- #
# Chunkset loading + minimal textbook_structure construction.
# --------------------------------------------------------------------------- #

def _load_chunks(path: Path, max_chunks: Optional[int]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and rec.get("id"):
            chunks.append(rec)
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks


def _chunk_text(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("text") or "").strip()


def _group_chunks_into_chapters(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group chunks by source.module_id into a minimal textbook chapter list.

    Builds the chapter surface _plan_course_structure's Stage-2 consumes:
    each chapter carries id/title + a chapter_text (joined chunk text) + a
    sections[] list with the chunk ids so the window-synthesis pass can cite
    resolvable source_chunk_ids.
    """
    by_module: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for ch in chunks:
        src = ch.get("source") or {}
        mod = str(src.get("module_id") or src.get("lesson_id") or "module")
        title = str(src.get("module_title") or src.get("section_heading") or mod)
        if mod not in by_module:
            by_module[mod] = {
                "id": mod,
                "title": title,
                "chapter_text_parts": [],
                "sections": [],
                "chunk_ids": [],
            }
            order.append(mod)
        entry = by_module[mod]
        text = _chunk_text(ch)
        if text:
            entry["chapter_text_parts"].append(text)
        entry["chunk_ids"].append(ch["id"])
        entry["sections"].append({
            "id": ch["id"],
            "heading": str(src.get("section_heading") or title),
            "chunk_id": ch["id"],
            "text": text,
            "paragraphs": [text] if text else [],
        })
    chapters: List[Dict[str, Any]] = []
    for mod in order:
        e = by_module[mod]
        chapters.append({
            "id": e["id"],
            "title": e["title"],
            "chapter_text": "\n\n".join(e["chapter_text_parts"]),
            "sections": e["sections"],
            "chunk_ids": e["chunk_ids"],
        })
    return chapters


def _build_textbook_structure(
    course_name: str, chapters: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Minimal textbook_structure.json with chapters + draft_terminal_objectives."""
    draft_tos: List[Dict[str, Any]] = []
    # One coarse draft TO per chapter (reconciled by the Stage-2 reconcile call).
    for idx, ch in enumerate(chapters, start=1):
        draft_tos.append({
            "id": f"TO-{idx:02d}",
            "statement": f"Understand the core concepts of {ch['title']}.",
            "bloom_level": "understand",
        })
    return {
        "course_name": course_name,
        "chapters": chapters,
        "draft_terminal_objectives": draft_tos,
        "chapter_count": len(chapters),
        "generated_by": "integration_smoke_harness",
    }


# --------------------------------------------------------------------------- #
# Stage 1 — objective synthesis via the real plan_course_structure.
# --------------------------------------------------------------------------- #

async def _run_objective_synthesis(
    *,
    registry: Dict[str, Any],
    project_id: str,
    project_path: Path,
    course_name: str,
    libv2_root: Path,
    duration_weeks: int,
) -> Dict[str, Any]:
    result_raw = await registry["plan_course_structure"](
        project_id=project_id,
        course_name=course_name,
        libv2_root=str(libv2_root),
        duration_weeks=duration_weeks,
        duration_weeks_explicit=True,
    )
    return json.loads(result_raw)


# --------------------------------------------------------------------------- #
# Stage 2 — two-pass content generation via the real CourseforgeRouter.
# --------------------------------------------------------------------------- #

# Substantive block types we synthesize (in the W3 SUBSTANTIVE set, scored by
# W4/W5 for prose grounding). One concept + one explanation per chapter.
_BLOCK_PLAN = [("concept", 0), ("explanation", 1)]


def _build_block_stubs(
    chapters: List[Dict[str, Any]],
    objectives_by_chapter: Dict[str, List[str]],
):
    """Build substantive Block stubs (one per (chapter, block-type)).

    Returns (blocks, source_chunks_by_block_id). Each block's source_chunks is
    the chapter's real dart chunks as [{id, text}] so the outline model cites
    resolvable chunk ids; objective_ids point at the chapter's synthesized COs.
    """
    from Courseforge.scripts.blocks import Block
    from lib.ontology.slugs import canonical_slug as _slug

    blocks: List[Any] = []
    src_by_block: Dict[str, List[Dict[str, str]]] = {}
    for week_idx, ch in enumerate(chapters, start=1):
        page_id = f"week_{week_idx:02d}_content_01"
        chunk_payload = [
            {"id": s["chunk_id"], "text": s["text"]}
            for s in ch["sections"] if s.get("text")
        ]
        if not chunk_payload:
            continue
        oids = tuple(objectives_by_chapter.get(ch["id"], []))
        for btype, seq in _BLOCK_PLAN:
            slug_value = _slug(f"{ch['title']}-{btype}")
            block_id = Block.stable_id(
                page_id=page_id, block_type=btype, slug=slug_value, idx=seq
            )
            stub = Block(
                block_id=block_id,
                block_type=btype,
                page_id=page_id,
                sequence=seq,
                content="",
                objective_ids=oids,
                key_terms=tuple(ch.get("title", "").split()[:4]),
            )
            blocks.append(stub)
            src_by_block[block_id] = chunk_payload
    return blocks, src_by_block


def _run_two_pass_content(
    *,
    chapters: List[Dict[str, Any]],
    objectives_by_chapter: Dict[str, List[str]],
    objectives_payload: List[Dict[str, Any]],
    content_dir: Path,
    dart_chunks_path: Path,
) -> Tuple[List[Any], List[Any], Optional[Path]]:
    """Drive outline -> rewrite via the real router; emit the W3 manifest.

    Returns (outline_blocks, rewrite_blocks, manifest_path).
    """
    from Courseforge.router.router import CourseforgeRouter
    from Courseforge.router.policy import load_block_routing_policy
    from MCP.tools.pipeline_tools import _emit_block_synthesis_manifest
    from lib.decision_capture import DecisionCapture

    capture = DecisionCapture(
        course_code="integration-smoke",
        phase="courseforge-content-generator-outline",
        tool="courseforge",
        streaming=True,
    )
    router = CourseforgeRouter(capture=capture, policy=load_block_routing_policy())

    stubs, src_by_block = _build_block_stubs(chapters, objectives_by_chapter)

    # Graceful degrade: a single block that exhausts the outline-tier retry
    # budget (e.g. the 7B keeps emitting a malformed CURIE) raises
    # OutlineProviderError. Mirror the production router's degrade-and-continue
    # posture — log + skip that one block rather than aborting the whole
    # course. The W5 eval then scores the blocks that DID synthesize.
    from Courseforge.generators._outline_provider import OutlineProviderError

    outline_blocks: List[Any] = []
    skipped: List[str] = []
    for blk in stubs:
        t0 = time.time()
        try:
            outlined = router.route_with_self_consistency(
                blk,
                validators=[],  # smoke: skip in-loop gates; validate post-hoc.
                source_chunks=src_by_block.get(blk.block_id, []),
                objectives=objectives_payload,
            )
        except OutlineProviderError as exc:
            skipped.append(blk.block_id)
            print(
                f"  [outline] {blk.block_type:12s} {blk.block_id[:42]:42s} "
                f"SKIPPED (outline exhausted): {str(exc)[:80]} "
                f"{time.time() - t0:5.1f}s",
                flush=True,
            )
            continue
        print(
            f"  [outline] {blk.block_type:12s} {blk.block_id[:42]:42s} "
            f"marker={outlined.escalation_marker or '-':24s} "
            f"{time.time() - t0:5.1f}s",
            flush=True,
        )
        outline_blocks.append(outlined)
    if skipped:
        print(f"  [outline] degraded: {len(skipped)} block(s) skipped, "
              f"{len(outline_blocks)} synthesized", flush=True)

    rewrite_blocks: List[Any] = []
    citation_stats: List[Dict[str, Any]] = []
    for blk in outline_blocks:
        t0 = time.time()
        try:
            rewritten = router.route_rewrite_with_remediation(
                blk,
                validators=[],
                source_chunks=src_by_block.get(blk.block_id, []),
                objectives=objectives_payload,
            )
        except Exception as exc:  # noqa: BLE001 — degrade, don't abort the course
            print(
                f"  [rewrite] {blk.block_type:12s} {blk.block_id[:42]:42s} "
                f"SKIPPED (rewrite error): {str(exc)[:80]} "
                f"{time.time() - t0:5.1f}s",
                flush=True,
            )
            continue
        # The rewrite tier replaces content (dict -> HTML str), which drops the
        # outline's content["key_claims"][].source_chunk_ids that the W3
        # manifest projection reads. Propagate those chunk ids onto the rewrite
        # block's source_references tuple (the manifest's documented fallback).
        rewritten = _carry_grounding(
            blk, rewritten, source_chunks=src_by_block.get(blk.block_id, []),
            stats=citation_stats,
        )
        print(
            f"  [rewrite] {blk.block_type:12s} {blk.block_id[:42]:42s} "
            f"marker={rewritten.escalation_marker or '-':24s} "
            f"{time.time() - t0:5.1f}s",
            flush=True,
        )
        rewrite_blocks.append(rewritten)

    content_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _emit_block_synthesis_manifest(
        rewrite_blocks, content_dir, dart_chunks_path, capture
    )
    return outline_blocks, rewrite_blocks, manifest_path, citation_stats


def _summarize_citation_validity(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-block claim-citation validity (eval blind-spot surfacer).

    ``coarse_provenance_rate`` is the share of blocks whose manifest chunks
    came from the source-chunk FALLBACK because the 7B cited nothing
    resolvable — these pass provenance_completeness trivially yet the model's
    own per-claim citations were invalid. ``mean_claim_citation_valid_rate``
    is the mean fraction of model-cited ids that were real chunk ids.
    """
    n = len(stats)
    if not n:
        return {
            "blocks": 0, "coarse_provenance_blocks": 0, "coarse_provenance_rate": None,
            "mean_claim_citation_valid_rate": None,
        }
    coarse = sum(1 for s in stats if s.get("used_fallback"))
    rates = [s.get("claim_citation_valid_rate", 0.0) for s in stats]
    return {
        "blocks": n,
        "coarse_provenance_blocks": coarse,
        "coarse_provenance_rate": round(coarse / n, 4),
        "mean_claim_citation_valid_rate": round(sum(rates) / n, 4),
        "per_block": stats,
    }


def _carry_grounding(
    outline_blk: Any,
    rewrite_blk: Any,
    source_chunks: Optional[List[Dict[str, str]]] = None,
    stats: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Propagate VALID source chunk ids onto the rewrite block.

    The rewrite tier replaces ``content`` (dict -> HTML str), which drops the
    outline's ``content["key_claims"][].source_chunk_ids`` that the W3 manifest
    projection reads. We harvest those ids and stamp them onto the rewrite
    block's ``source_references`` tuple — the documented manifest fallback
    (``Block.to_synthesis_manifest`` reads ``self.source_references`` when
    ``content`` is not a grounding dict).

    The 7B outline tier is unreliable on the per-claim ``source_chunk_ids``
    field: it sometimes emits a PROSE STRING ("Source chunk: <sentence>")
    instead of a real chunk id, which then fails to resolve and leaves the
    manifest empty (block ungrounded). So we VALIDATE every harvested id
    against the block's actual ``source_chunks`` id set and keep only the
    real ones. When the model cited NO valid id (all garbage), we fall back
    to the block's full ``source_chunks`` ids — the chunks the block was
    genuinely synthesized from, so they are legitimate block-level provenance
    (NOT fabrication; the W5 NLI scorer then judges grounding against those
    real premises). With no ``source_chunks`` supplied, the legacy harvest
    (unfiltered) is preserved for back-compat.
    """
    valid_ids = {
        c["id"] for c in (source_chunks or []) if isinstance(c, dict) and c.get("id")
    }
    outline_content = outline_blk.content if isinstance(outline_blk.content, dict) else {}
    cited: List[str] = []
    seen = set()

    def _add(cid: Any) -> None:
        if isinstance(cid, str) and cid and cid not in seen:
            seen.add(cid)
            cited.append(cid)

    for claim in outline_content.get("key_claims", []) or []:
        if isinstance(claim, dict):
            for cid in claim.get("source_chunk_ids", []) or []:
                _add(cid)
    for ref in outline_content.get("source_refs", []) or []:
        if isinstance(ref, dict):
            _add(ref.get("sourceId"))

    n_valid = sum(1 for c in cited if c in valid_ids) if valid_ids else 0
    used_fallback = False
    if valid_ids:
        # Keep only ids that name a real chunk the block was given; drop the
        # 7B's prose-string hallucinations.
        chunk_ids = [c for c in cited if c in valid_ids]
        if not chunk_ids:
            # Model cited nothing resolvable → ground in the block's own
            # source chunks (legitimate block-level provenance).
            used_fallback = True
            chunk_ids = [c["id"] for c in source_chunks if isinstance(c, dict) and c.get("id")]
    else:
        # No source_chunks supplied (legacy/back-compat): keep the raw harvest.
        chunk_ids = cited

    # Record per-block citation validity so the eval can SURFACE the 7B's
    # actual citation quality (the model often emits a prose string instead
    # of a chunk id). ``used_fallback`` blocks have COARSE provenance — they
    # pass provenance_completeness trivially but the model cited nothing
    # resolvable, which provenance_completeness alone can't see.
    if stats is not None:
        stats.append({
            "block_id": getattr(rewrite_blk, "block_id", "?"),
            "n_cited": len(cited),
            "n_valid": n_valid,
            "used_fallback": used_fallback,
            "claim_citation_valid_rate": (n_valid / len(cited)) if cited else 0.0,
        })

    if not chunk_ids:
        return rewrite_blk
    new_refs = tuple(
        {"sourceId": cid, "role": "primary"} for cid in chunk_ids
    )
    existing = tuple(rewrite_blk.source_references or ())
    return dataclasses.replace(
        rewrite_blk,
        source_references=existing + new_refs,
        source_ids=tuple(chunk_ids),
    )


# --------------------------------------------------------------------------- #
# Stage 3 — validation.
# --------------------------------------------------------------------------- #

def _w3_manifest_to_w5(
    manifest_lines: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Adapt the W3 block_synthesis_manifest.jsonl shape to the W5 manifests dict.

    W3 emits per-line: source_chunk_ids, template_id, concept_tags,
    synthesis.tiers[]. W5's _manifest_complete expects, per block_id:
    source_chunk_ids, model, params, concept_tags, and (template_id+slots) OR
    free_scaffold. We derive model/params from the last synthesis tier (the
    authoritative author) and supply free_scaffold=True (these blocks are
    LLM-authored free prose, not slot-fill templates) — a faithful projection,
    no fabricated provenance.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for line in manifest_lines:
        bid = line.get("block_id")
        if not bid:
            continue
        tiers = (line.get("synthesis") or {}).get("tiers") or []
        last = tiers[-1] if tiers else {}
        out[bid] = {
            "source_chunk_ids": list(line.get("source_chunk_ids") or []),
            "model": last.get("model") or "local",
            "params": {"provider": last.get("provider") or "local"},
            "concept_tags": list(line.get("concept_tags") or []),
            "template_id": line.get("template_id"),
            "free_scaffold": True,
        }
    return out


def _run_manifest_completeness(
    manifest_path: Optional[Path],
    chunks_by_id: Dict[str, Any],
) -> Dict[str, Any]:
    from lib.validators.manifest_completeness import ManifestCompletenessValidator

    validator = ManifestCompletenessValidator()
    inputs = {
        "manifest_path": str(manifest_path) if manifest_path else None,
        "valid_source_ids": set(chunks_by_id.keys()),
        "require_concept_tags": False,
    }
    res = validator.validate(inputs)
    return {
        "passed": res.passed,
        "score": res.score,
        "action": res.action,
        "issues": [
            {"severity": i.severity, "code": i.code, "message": i.message}
            for i in res.issues
        ],
    }


def _block_to_w5_dict(blk: Any) -> Dict[str, Any]:
    """Project a rewrite Block into the dict shape W5's _block_* helpers read."""
    from MCP.tools.pipeline_tools import _block_to_snake_case_entry

    return _block_to_snake_case_entry(blk)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #

def _print_objectives_for_review(objectives_doc: dict, objectives_path: str) -> None:
    """Print the synthesized learning objectives in a review-friendly form."""
    import collections

    def _los(doc):
        if doc.get("learning_outcomes"):
            return list(doc["learning_outcomes"])
        return list(doc.get("terminal_objectives", [])) + list(
            doc.get("chapter_objectives", [])
        )

    los = [o for o in _los(objectives_doc) if isinstance(o, dict)]
    hist = collections.Counter((o.get("bloom_level") or "?") for o in los)
    distinct = sum(1 for v in hist.values() if v)
    total = len(los) or 1
    max_share = max(hist.values()) / total if los else 0.0

    print("\n" + "=" * 72, flush=True)
    print(f"SYNTHESIZED LEARNING OBJECTIVES (7B) — {len(los)} total", flush=True)
    print(f"  mint_method={objectives_doc.get('mint_method')} "
          f"duration_weeks={objectives_doc.get('duration_weeks')}", flush=True)
    print(f"  bloom: {dict(hist)} | distinct_levels={distinct} "
          f"max_share={max_share:.3f}", flush=True)
    print("=" * 72, flush=True)
    for o in los:
        ab = o.get("abcd") if isinstance(o.get("abcd"), dict) else {}
        verb = (ab.get("behavior") or {}).get("verb") if isinstance(
            ab.get("behavior"), dict) else o.get("bloom_verb")
        sc = o.get("source_chunk_ids") or [
            c for ref in (o.get("source_refs") or []) for c in ref.get("chunk_ids", [])
        ]
        tid = o.get("terminal_id")
        link = f" -> {tid}" if tid else ""
        print(f"  [{o.get('id')}] ({o.get('hierarchy_level')}{link}) "
              f"bloom={o.get('bloom_level')}/verb={verb} chunks={len(sc)}", flush=True)
        print(f"      {o.get('statement','')}", flush=True)
    print("=" * 72, flush=True)
    print(f"Written to: {objectives_path}", flush=True)


async def _amain(args: argparse.Namespace) -> int:
    from MCP.tools.pipeline_tools import (
        _build_tool_registry,
        courseforge_exports_dir,
    )

    chunks_path = Path(args.chunks).resolve()
    if not chunks_path.exists():
        raise SystemExit(f"chunkset not found: {chunks_path}")

    course_slug = args.course_slug
    course_name = course_slug  # _load_dart_chunkset_for_planning keys on slug.

    # --- Workspace: temp libv2 root + project export dir. ---
    out_root = Path(args.out_dir).resolve() if args.out_dir else Path(
        tempfile.mkdtemp(prefix="ed4all-gen-smoke-")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    libv2_root = out_root / "libv2"
    course_chunks_dir = libv2_root / "courses" / course_slug / "dart_chunks"
    course_chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = _load_chunks(chunks_path, args.max_chunks)
    if not chunks:
        raise SystemExit("no chunks loaded")
    dart_chunks_path = course_chunks_dir / "chunks.jsonl"
    dart_chunks_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )
    chunks_by_id = {c["id"]: c for c in chunks}
    print(
        f"Loaded {len(chunks)} chunks from {chunks_path} -> {dart_chunks_path}",
        flush=True,
    )

    chapters = _group_chunks_into_chapters(chunks)
    duration_weeks = max(1, len(chapters))
    print(f"Grouped into {len(chapters)} chapter(s); duration_weeks={duration_weeks}", flush=True)

    # Project export dir under the real Courseforge exports root.
    project_id = f"PROJ-{course_slug}-smoke"
    project_path = courseforge_exports_dir() / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    ts = _build_textbook_structure(course_name, chapters)
    (project_path / "01_learning_objectives" / "textbook_structure.json").write_text(
        json.dumps(ts, indent=2), encoding="utf-8"
    )
    (project_path / "project_config.json").write_text(
        json.dumps({
            "course_name": course_name,
            "project_id": project_id,
            "duration_weeks": duration_weeks,
        }, indent=2),
        encoding="utf-8",
    )

    registry = _build_tool_registry()

    # =================== STAGE 1: objective synthesis =================== #
    print("\n=== STAGE 1: objective synthesis (W2 window synthesis, local 7B) ===", flush=True)
    t0 = time.time()
    plan = await _run_objective_synthesis(
        registry=registry,
        project_id=project_id,
        project_path=project_path,
        course_name=course_name,
        libv2_root=libv2_root,
        duration_weeks=duration_weeks,
    )
    print(f"  plan_course_structure: {time.time()-t0:.1f}s; mint_method={plan.get('mint_method')}", flush=True)
    if plan.get("error"):
        print(f"  ERROR: {plan['error']}", flush=True)
        return 2
    objectives_path = plan.get("synthesized_objectives_path")
    objectives_doc = json.loads(Path(objectives_path).read_text(encoding="utf-8"))
    print(f"  objectives: {len(plan.get('objective_ids') or [])} ids; "
          f"terminal={plan.get('terminal_count')} chapter={plan.get('chapter_count')}", flush=True)

    if getattr(args, "objectives_only", False):
        _print_objectives_for_review(objectives_doc, objectives_path)
        return 0

    # Map chapter -> CO ids (the chapter_objectives carry a `chapter` key
    # matching the chapter title) for block->objective binding.
    objectives_by_chapter = _map_chapter_objectives(objectives_doc, chapters)
    objectives_payload = _flatten_objectives_payload(objectives_doc)

    # =================== STAGE 2: content generation =================== #
    print("\n=== STAGE 2: two-pass content generation (outline -> rewrite, local 7B) ===", flush=True)
    content_dir = project_path / "03_content_development"
    t0 = time.time()
    outline_blocks, rewrite_blocks, manifest_path, citation_stats = _run_two_pass_content(
        chapters=chapters,
        objectives_by_chapter=objectives_by_chapter,
        objectives_payload=objectives_payload,
        content_dir=content_dir,
        dart_chunks_path=dart_chunks_path,
    )
    citation_validity = _summarize_citation_validity(citation_stats)
    print(f"  claim-citation validity: coarse_provenance_rate="
          f"{citation_validity.get('coarse_provenance_rate')} "
          f"mean_valid_rate={citation_validity.get('mean_claim_citation_valid_rate')}",
          flush=True)
    print(f"  generated {len(rewrite_blocks)} blocks in {time.time()-t0:.1f}s; "
          f"manifest={manifest_path}", flush=True)

    manifest_lines: List[Dict[str, Any]] = []
    if manifest_path and Path(manifest_path).exists():
        for ln in Path(manifest_path).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                manifest_lines.append(json.loads(ln))

    # =================== STAGE 3: validation =================== #
    print("\n=== STAGE 3: validation ===", flush=True)
    manifest_res = _run_manifest_completeness(manifest_path, chunks_by_id)
    print(f"  ManifestCompleteness: passed={manifest_res['passed']} "
          f"score={manifest_res['score']:.3f} issues={len(manifest_res['issues'])}", flush=True)

    from lib.objectives.generation_quality_eval import (
        run_generation_quality_eval,
        PROPOSED_BLOCK_GROUNDING_FLOOR,
        PROPOSED_PROVENANCE_COMPLETENESS_FLOOR,
        PROPOSED_OBJECTIVE_GROUNDING_FLOOR,
    )

    block_dicts = [_block_to_w5_dict(b) for b in rewrite_blocks]
    w5_manifests = _w3_manifest_to_w5(manifest_lines)

    # Same resolver the W3 manifest projection used (chunk id ∪ dart:slug →
    # chunk record). The provenance cross-check resolves the block's cited
    # source ids to chunk-id space so a block that cited by DART slug (while
    # the manifest holds the resolved chunk id) is not a spurious mismatch.
    from MCP.tools.pipeline_tools import _build_manifest_chunk_resolver
    source_resolver = _build_manifest_chunk_resolver(dart_chunks_path)

    print("  loading NLI + embedder for W5 entailment scoring (cuda)...", flush=True)
    t0 = time.time()
    # The W5 harness writes to output_path directly (mkdir only on its own
    # default eval_dir, not a custom output_path parent) — create the parent.
    w5_out = content_dir.parent / "retrieval_eval" / "generation_quality_eval_LOCAL7B.json"
    w5_out.parent.mkdir(parents=True, exist_ok=True)
    report = run_generation_quality_eval(
        _REPO_ROOT,
        course_slug,
        objectives=objectives_doc,
        blocks=block_dicts,
        chunks_by_id=chunks_by_id,
        manifests=w5_manifests,
        technique_config="LOCAL7B",
        write=True,
        output_path=w5_out,
        source_resolver=source_resolver,
    )
    print(f"  run_generation_quality_eval: {time.time()-t0:.1f}s", flush=True)

    # =================== REPORT =================== #
    _print_report(
        report=report,
        manifest_res=manifest_res,
        manifest_lines=manifest_lines,
        objectives_doc=objectives_doc,
        rewrite_blocks=rewrite_blocks,
        floors={
            "block_prose_grounding_rate": PROPOSED_BLOCK_GROUNDING_FLOOR,
            "provenance_completeness_rate": PROPOSED_PROVENANCE_COMPLETENESS_FLOOR,
            "objective_grounding_rate": PROPOSED_OBJECTIVE_GROUNDING_FLOOR,
        },
        out_root=out_root,
    )

    # Persist a combined harness artifact.
    artifact_dir = content_dir.parent / "retrieval_eval"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "integration_smoke_summary.json").write_text(
        json.dumps({
            "course_slug": course_slug,
            "chunks": len(chunks),
            "chapters": len(chapters),
            "objectives": plan.get("objective_ids"),
            "manifest_completeness": manifest_res,
            "claim_citation_validity": citation_validity,
            "w5_headline": report.get("headline"),
            "w5_success_condition": report.get("success_condition"),
            "project_path": str(project_path),
            "libv2_root": str(libv2_root),
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nArtifacts under: {artifact_dir}", flush=True)

    return 0


def _map_chapter_objectives(
    objectives_doc: Dict[str, Any], chapters: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """Map chapter id -> [CO ids] using the chapter_objectives `chapter` key."""
    title_to_id = {ch["title"]: ch["id"] for ch in chapters}
    out: Dict[str, List[str]] = {}
    for grp in objectives_doc.get("chapter_objectives", []) or []:
        chap_label = grp.get("chapter")
        objs = grp.get("objectives", []) or []
        ids = [str(o.get("id")) for o in objs if o.get("id")]
        chap_id = title_to_id.get(chap_label)
        if chap_id:
            out.setdefault(chap_id, []).extend(ids)
    # Positional fallback when titles didn't match.
    if not out:
        groups = objectives_doc.get("chapter_objectives", []) or []
        for ch, grp in zip(chapters, groups):
            ids = [str(o.get("id")) for o in grp.get("objectives", []) if o.get("id")]
            out[ch["id"]] = ids
    return out


def _flatten_objectives_payload(objectives_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for o in objectives_doc.get("terminal_objectives", []) or []:
        payload.append({
            "id": o.get("id"),
            "statement": o.get("statement"),
            "bloom_level": o.get("bloom_level"),
        })
    for grp in objectives_doc.get("chapter_objectives", []) or []:
        for o in grp.get("objectives", []) or []:
            payload.append({
                "id": o.get("id"),
                "statement": o.get("statement"),
                "bloom_level": o.get("bloom_level"),
            })
    return payload


def _print_report(
    *,
    report: Dict[str, Any],
    manifest_res: Dict[str, Any],
    manifest_lines: List[Dict[str, Any]],
    objectives_doc: Dict[str, Any],
    rewrite_blocks: List[Any],
    floors: Dict[str, float],
    out_root: Path,
) -> None:
    headline = report.get("headline", {})
    sc = report.get("success_condition", {})

    def _fmt(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    print("\n" + "=" * 72)
    print("MEASURED vs THRESHOLD (W5 success condition)")
    print("=" * 72)
    rows = [
        ("block_prose_grounding_rate", headline.get("block_prose_grounding_rate"),
         floors["block_prose_grounding_rate"], ">="),
        ("provenance_completeness_rate", headline.get("provenance_completeness_rate"),
         floors["provenance_completeness_rate"], ">="),
        ("objective_grounding_rate", headline.get("objective_grounding_rate"),
         floors["objective_grounding_rate"], ">="),
    ]
    print(f"{'metric':38s} {'measured':>10s} {'thresh':>8s} {'verdict':>8s}")
    for name, val, thr, op in rows:
        if val is None:
            verdict = "N/A"
        else:
            verdict = "PASS" if float(val) >= thr else "FAIL"
        print(f"{name:38s} {_fmt(val):>10s} {op}{_fmt(thr):>7s} {verdict:>8s}")
    print(f"{'contradicted_block_count':38s} {_fmt(headline.get('contradicted_block_count')):>10s} "
          f"  ==0   {'PASS' if (headline.get('contradicted_block_count') or 0)==0 else 'FAIL':>8s}")
    print(f"{'objective_contradicted_count':38s} {_fmt(headline.get('objective_contradicted_count')):>10s} "
          f"  ==0   {'PASS' if (headline.get('objective_contradicted_count') or 0)==0 else 'FAIL':>8s}")
    print(f"{'manifest_completeness (W3)':38s} {_fmt(manifest_res['score']):>10s} "
          f"   pass  {'PASS' if manifest_res['passed'] else 'FAIL':>8s}")

    bloom = headline.get("bloom") or {}
    print(f"\nbloom: distinct={bloom.get('distinct_levels')} max_share={_fmt(bloom.get('max_share'))}")
    print(f"concept_coverage_rate={_fmt(headline.get('concept_coverage_rate'))} "
          f"chunk_coverage_rate={_fmt(headline.get('chunk_coverage_rate'))}")
    # Diagnostics (informational — not success-condition pillars): surface the
    # instructional-depth + claim-citation-quality gaps the headline grounding
    # metrics can mask.
    btd = headline.get("block_type_diversity") or {}
    print(f"instructional_completeness={_fmt(btd.get('instructional_completeness'))} "
          f"(explain={btd.get('has_explain')} example={btd.get('has_example')} "
          f"assessment={btd.get('has_assessment')}) block_types={btd.get('histogram')}")
    pa = report.get("provenance_audit", {})
    print(f"manifest_chunk_resolution_rate={_fmt(pa.get('manifest_chunk_resolution_rate'))} "
          f"manifest_matches_cited_rate={_fmt(pa.get('manifest_matches_cited_rate'))}")

    # --- Sample objectives with source_chunk_ids ---
    print("\n" + "-" * 72)
    print("SAMPLE OBJECTIVES (with cited source chunk ids)")
    print("-" * 72)
    flat = (objectives_doc.get("terminal_objectives", []) or [])[:2]
    for grp in (objectives_doc.get("chapter_objectives", []) or [])[:2]:
        flat.extend((grp.get("objectives", []) or [])[:2])
    for o in flat[:5]:
        refs = o.get("source_refs") or o.get("source_chunk_ids") or []
        cids: List[str] = []
        if isinstance(refs, list):
            for r in refs:
                if isinstance(r, dict):
                    cids.extend(r.get("chunk_ids", []) or [])
                elif isinstance(r, str):
                    cids.append(r)
        print(f"  [{o.get('id')}] {str(o.get('statement'))[:90]}")
        print(f"      source_chunk_ids: {cids[:4]}")

    # --- Sample blocks with manifest entries ---
    print("\n" + "-" * 72)
    print("SAMPLE BLOCKS (with W3 manifest entries)")
    print("-" * 72)
    by_id = {m.get("block_id"): m for m in manifest_lines}
    block_rows = {r["block_id"]: r for r in report.get("blocks", [])}
    from lib.utils import strip_html_to_text
    for blk in rewrite_blocks[:4]:
        m = by_id.get(blk.block_id, {})
        row = block_rows.get(blk.block_id, {})
        prose = blk.content if isinstance(blk.content, str) else json.dumps(blk.content)
        prose_txt = strip_html_to_text(prose) if isinstance(prose, str) else str(prose)
        print(f"  [{blk.block_type}] {blk.block_id}")
        print(f"      prose: {prose_txt[:120].strip()}")
        print(f"      manifest.source_chunk_ids: {m.get('source_chunk_ids')}")
        print(f"      manifest.template_id: {m.get('template_id')} "
              f"concept_tags: {m.get('concept_tags')}")
        print(f"      W5 entailment_rate: {_fmt(row.get('entailment_rate'))} "
              f"grounded={row.get('grounded')} contradicted={row.get('contradicted')}")

    # --- Verdict ---
    print("\n" + "=" * 72)
    verdict = "PASS" if sc.get("met") else "FAIL (modest 7B quality expected; see grounded/provenanced signals)"
    print(f"W5 success_condition.met = {sc.get('met')}  =>  HARNESS VERDICT: {verdict}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True, help="Path to dart_chunks/chunks.jsonl")
    ap.add_argument("--course-slug", required=True, help="Course slug (no hardcoded slug in tracked code)")
    ap.add_argument("--max-chunks", type=int, default=10, help="Cap chunks for a fast first run")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5-7b-8k")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--out-dir", default=None, help="Workspace dir (default: tempdir)")
    ap.add_argument(
        "--block-routing", default=None,
        help="Optional block_routing.yaml override (pins authoring tiers to --model)",
    )
    ap.add_argument(
        "--objectives-only", action="store_true",
        help="Run ONLY Stage 1 (objective synthesis), print the learning "
             "objectives for review, and exit before content/KG.",
    )
    args = ap.parse_args()

    _configure_local_env(
        args.base_url, args.model, num_ctx=args.num_ctx,
        block_routing=args.block_routing,
    )
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
