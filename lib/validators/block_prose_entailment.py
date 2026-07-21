"""W4 — ``BlockProseEntailmentValidator`` (per-block full-prose NLI entailment).

The text-trace counterpart to the (W4-demoted) cosine
``RewriteSourceGroundingValidator`` AND the broadened counterpart to
``ClaimSupportValidator`` — where ``claim_support`` scores only the curated
``key_claims[]``, this validator scores ALL substantive rewrite-tier block
prose against the block's cited source chunk(s) via NLI entailment.

Cosine 0.45 catches gross off-topic drift but PASSES plausible-but-fabricated
on-topic prose (a fabricated sentence that stays on-topic scores 0.5+ cosine).
NLI entailment is the stronger text-trace signal: a fabricated sentence that
is not entailed by any cited chunk fails NLI even when it passes cosine. This
is the keystone failure class W4 closes.

Algorithm (per ``plans/finegrain/w4-nli-grounding-gate.md`` §1.2):

1. **Skip set** — reuse ``claim_support._SKIPPED_BLOCK_TYPES``
   (``assessment_item`` / ``objective`` / ``chrome`` / ``recap``). DELIBERATE
   asymmetry vs ``rewrite_source_grounding._SKIPPED_CONTENT_TYPES`` which ALSO
   skips ``example`` / ``self_check_question``: NLI scores entailment, not
   surface cosine, so ``example`` (code / URI payload depresses cosine but not
   entailment of the framing prose) and ``self_check_question`` ARE scored
   here. Computational sentences inside examples are exempted downstream by
   ``groundedness._is_computational``.
2. **Strip prose to text** — ``lib.utils.strip_html_to_text`` (the W-D6
   canonical helper). Rewrite-tier ``block.content`` is a ``str`` (HTML) →
   strip; outline-tier ``block.content`` is a ``dict`` → skip silently
   (the rewrite-tier seam is where prose entailment is checked).
3. **Resolve cited chunk texts** — walk the same id surfaces
   ``rewrite_source_grounding._resolve_block_source_chunks`` consults, but
   preserve the chunk_id so the report's ``best_chunk_id`` is meaningful.
4. **Score** — ``groundedness.score_groundedness`` (windowed scorer-v2). All
   claim splitting + computational exemption + structural-artifact drop +
   windowed rescue is INSIDE ``score_groundedness`` — the validator does NOT
   re-split.

Pass/fail (per block, §1.3):

- ``BLOCK_PROSE_UNGROUNDED`` when ``scored_count > 0`` AND
  ``block_entailment_rate < min_block_entailment_rate``. ``action="regenerate"``.
- ``BLOCK_PROSE_CONTRADICTED`` when ``contradicted_rate > max_contradicted_rate``
  — fires standalone (a contradiction is fabrication that *conflicts* with
  source, the worst case). ``action="regenerate"``.
- ``scored_count == 0`` (only computational / artifact prose) → no-op pass.
- No resolved cited chunks → warning ``BLOCK_PROSE_NO_GROUNDING_SOURCE``,
  ``passed=True`` (the resolution gate ``rewrite_source_refs`` owns the
  missing-source structural failure).

Graceful-degrade (§1.4): mirrors ``claim_support`` / the EMBEDDING_DEPS_MISSING
contract. Pre-check ``NliClassifier.get_or_load()``; non-strict + missing deps
→ warning ``NLI_DEPS_MISSING``, ``passed=True, action=None``; strict
(``TRAINFORGE_REQUIRE_EMBEDDINGS=true``) → raise ``RuntimeError``.

**Shadow knob (§4.3):** ``inputs["shadow"]`` (merged from the gate
``config.shadow: true`` by the executor). When truthy, every emitted
``BLOCK_PROSE_*`` GateIssue is forced to ``severity="warning"`` and
``action`` is never set to ``"regenerate"`` — the gate COMPUTES + CAPTURES
the entailment distribution but does NOT gate. This makes the W4 day-1 shadow
state EXPLICIT in the validator and independent of the YAML ``severity`` field,
so the §4 calibration runs are unambiguous. The critical flip is
``config.shadow: false`` + YAML ``severity: critical`` (DEFERRED — see the
``# TODO(calibration)`` marker at the gate in ``config/workflows.yaml``).

Decision-capture (§1.5): one ``block_prose_entailment_check`` event per SCORED
block; rationale interpolates dynamic signals (≥20 chars). Skip-silent blocks
emit nothing (matches ``claim_support``).

Reuse contract (§8): NLI model singleton (``NliClassifier.get_or_load``),
sentence/claim split + artifact drop + computational exemption + windowed
rescue (``score_groundedness``), HTML→text (``strip_html_to_text``),
strict-mode flag (``is_strict_mode``), skip set
(``claim_support._SKIPPED_BLOCK_TYPES``), floors
(``_DEFAULT_ENTAILMENT_FLOOR`` / ``_DEFAULT_CONTRADICTION_FLOOR``).
"""
from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier
from lib.embedding.sentence_embedder import is_strict_mode
from lib.generation.llm_checkpoint import CheckpointStore, compute_fingerprint
from lib.generation.stop_control import (
    GracefulStopRequested,
    check_stop as _check_stop,
    stop_requested as _stop_requested,
)
from lib.retrieval.groundedness import (
    SCORER_VERSION,
    THRESHOLDS_PROVENANCE_DEFAULTS,
    ClaimVerdict,
    GroundednessReport,
    score_groundedness,
)
from lib.utils import strip_html_to_text as _strip_html_to_text  # W-D6 canonical
from lib.validators.claim_support import _SKIPPED_BLOCK_TYPES

# Block lives at Courseforge/scripts/blocks.py — same import bridge as the
# sibling validators.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Per-block aggregate entailment floor: at least this share of scored
#: (non-computational, non-artifact) prose sentences must be entailed by the
#: block's cited chunks. Day-1 0.60 (calibratable — §4 pins the value before
#: the critical flip). Mirrors ``claim_support``'s training-pair-calibrated
#: floors; NOT generation-calibrated, which is WHY the gate ships SHADOW.
_DEFAULT_MIN_BLOCK_ENTAILMENT_RATE: float = 0.60

#: Per-claim entailment floor (reuse claim_support default — passed through to
#: ``score_groundedness``).
_DEFAULT_ENTAILMENT_FLOOR: float = 0.70

#: Per-claim contradiction floor (reuse claim_support default).
_DEFAULT_CONTRADICTION_FLOOR: float = 0.50

#: Per-block contradicted-prose rate ceiling. Above this, the block fires
#: ``BLOCK_PROSE_CONTRADICTED`` regardless of total entailment rate — any
#: contradiction is a hard fabrication signal.
_DEFAULT_MAX_CONTRADICTED_RATE: float = 0.05

#: Cap per-block issue list (mirrors sibling validators).
_ISSUE_LIST_CAP: int = 50

#: Env flag — gate-side provenance resolution. Default OFF → byte-identical
#: legacy behavior (a block's cited ``semantik:{slug}#{anchor}`` refs are
#: resolved ONLY through the ``source_chunks`` map; unresolved → NO_GROUNDING
#: warning). When truthy AND ``inputs["chunk_provenance_index"]`` is present,
#: a block whose directly-cited ids resolve to nothing (rewrite-tier blocks
#: carry provenance as ``semantik:...#anchor`` refs, not ``{course}_chunk_NNNNN``
#: chunk ids) is re-resolved by mapping each cited provenance ref → the chunk
#: ids that carry it (a section-level ref maps to ALL chunk ids from that
#: section) and looking those chunk ids up in ``source_chunks``. Anti-fabrication:
#: only EXISTING refs are mapped to EXISTING chunks; unmappable refs still emit
#: the NO_GROUNDING warning. Resolution only ADDs — directly-resolvable ids are
#: never removed.
_ENV_PROVENANCE_RESOLVE: str = "ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _provenance_resolve_enabled() -> bool:
    """Parse-with-fallback truthy check for ``ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE``."""
    return os.environ.get(_ENV_PROVENANCE_RESOLVE, "").strip().lower() in _TRUTHY


#: Env flag (REPURPOSED) — shard THIS gate's per-block NLI entailment scoring
#: across a spawn ``ProcessPoolExecutor`` instead of the historical serial
#: per-block loop. Default OFF → the per-block ``score_groundedness`` calls run
#: sequentially on the raw NLI singleton (byte-identical to the legacy path).
#: When truthy, the SCORABLE blocks (exactly the ones the serial loop would
#: score, via :func:`_block_score_args`) are partitioned across a set of
#: persistent worker processes. Each worker loads its OWN NLI ONCE at init (a
#: picklable module-level factory — the MODEL is never pickled, only a dotted
#: factory string crosses the process boundary) and runs the UNCHANGED
#: ``score_groundedness`` per block end-to-end. Results are keyed by block index
#: and merged SERIALLY in the parent in original block order; the model reload
#: churn that dominated the resume-loop wall-clock is eliminated because each
#: worker holds one resident NLI for its whole shard.
#:
#: This REPLACES the earlier in-process coalescing thread-dispatcher design (the
#: thread pool could not defeat the single-GPU forward-pass serialization; a
#: process pool gives genuine parallel forward passes on distinct CUDA
#: contexts). It stays a DELIBERATELY separate opt-in from ``ED4ALL_NLI_MICROBATCH``
#: (the ``score_candidate`` best-of-N path): this gate is a load-bearing quality
#: gate AND this change introduces NEW cross-block parallelism, so it earns
#: independent, default-OFF control.
#:
#: Correctness: each worker computes an INDEPENDENT ``GroundednessReport`` per
#: block via the identical ``score_groundedness`` call the serial path makes, so
#: the report is verdict-identical (the only cross-process difference is the
#: low-order-bit GPU/CPU softmax non-associativity already documented on the NLI
#: device knob — well under the 0.70/0.50 verdict thresholds). Issue-list
#: assembly, the 50-issue cap, the audited/passed counters, and decision-capture
#: all still run SERIALLY in original block order in the parent, so the emitted
#: GateResult is verdict-identical and order-stable vs the serial path. Any
#: pool-start / worker failure degrades gracefully to the serial path.
_ENV_MICROBATCH_VALIDATORS: str = "ED4ALL_NLI_MICROBATCH_VALIDATORS"

#: Satellite of ``ED4ALL_NLI_MICROBATCH_VALIDATORS`` — the number of persistent
#: scoring worker processes (spawn ``ProcessPoolExecutor`` size). Default 4
#: (the investigation's memory-budget recommendation: 4 workers ≤ ~6 GB
#: worst-case at batch-8×512 DeBERTa activations, leaving ample reserve beside a
#: resident vLLM seat; a hard ceiling of 12-14 is the box's safe max). Garbage /
#: non-positive → the default. Always capped at the number of scorable blocks.
#: No-op when the master flag is off.
_ENV_VALIDATORS_PROCS: str = "ED4ALL_NLI_VALIDATORS_PROCS"
_DEFAULT_VALIDATORS_PROCS: int = 4

#: PLUMBING / test seam (NOT an operator flag) — a ``module:function`` dotted
#: path naming the picklable factory each pool worker calls ONCE at init to
#: build its resident NLI. Unset → :func:`default_nli_factory`
#: (``NliClassifier.get_or_load``, the production singleton loader). A test
#: points it at a deterministic-stub factory so the real spawn-pool path can be
#: exercised without a GPU. The dotted STRING (never the model) is what crosses
#: the process boundary.
_ENV_NLI_VALIDATORS_FACTORY: str = "ED4ALL_NLI_VALIDATORS_FACTORY"

#: Site flag (under the ``ED4ALL_GENERATION_CHECKPOINT`` family) — per-block
#: resume sidecar for this slow NLI gate. Default ON. Each scored block's
#: ``GroundednessReport`` is content-addressed (see :func:`_block_fingerprint`)
#: and persisted to a fingerprinted JSONL sidecar next to the rewrite export, so
#: a killed / ``ed4all stop``-ed / timed-out validation RESUMES nearly free
#: (already-scored blocks are cache HITS, skipping the DeBERTa forward pass). A
#: degraded / unavailable report is NEVER cached (it must re-run). Byte-identical
#: to the legacy path when no cache dir is resolvable OR the family/site flag is
#: falsey (``0``/``false``/``no``/``off``).
_ENV_VALIDATION_CHECKPOINT: str = "ED4ALL_VALIDATION_CHECKPOINT"

#: Site id for the resume sidecar records.
_CHECKPOINT_SITE_ID: str = "post_rewrite_prose_entailment"

#: Basename of the resume sidecar file inside the cache dir.
_CHECKPOINT_BASENAME: str = "prose_entailment.checkpoint.jsonl"

#: Cooperative-stop site id (surfaced on GracefulStopRequested).
_STOP_SITE_ID: str = "post_rewrite_prose_entailment"


def _microbatch_validators_enabled() -> bool:
    """Parse-with-fallback truthy check for ``ED4ALL_NLI_MICROBATCH_VALIDATORS``.

    (Kept name for back-compat imports; now gates the process-pool path.)
    """
    return (
        os.environ.get(_ENV_MICROBATCH_VALIDATORS, "").strip().lower() in _TRUTHY
    )


#: Alias with the intent-revealing name for the repurposed flag.
_pool_scoring_enabled = _microbatch_validators_enabled


def _crossblock_enabled() -> bool:
    """Truthy check for ``ED4ALL_NLI_CROSSBLOCK`` (single-process thread fan-out).

    The SUPERSEDING path for cross-block gate NLI: instead of the retired
    ``ED4ALL_NLI_MICROBATCH_VALIDATORS`` spawn-ProcessPool (which livelocked 4
    CUDA contexts on one card), the SCORABLE cache-miss blocks are fanned across
    a ``ThreadPoolExecutor`` whose workers submit their pair-batches to the
    single-process coalescing dispatcher (one scorer thread owns the GPU model).
    Default OFF → serial scoring, byte-identical. Independent of
    ``_pool_scoring_enabled`` — if both are set, the (dormant) process pool wins
    for back-compat and the thread path is only tried when the pool degrades.
    """
    from lib.classifiers.nli_microbatch import resolve_crossblock_enabled

    return resolve_crossblock_enabled()


def _resolve_validators_procs() -> int:
    """Resolve the worker-process count. Garbage / non-positive → the default."""
    raw = os.environ.get(_ENV_VALIDATORS_PROCS)
    if raw is None or not str(raw).strip():
        return _DEFAULT_VALIDATORS_PROCS
    try:
        iv = int(str(raw).strip())
        return iv if iv > 0 else _DEFAULT_VALIDATORS_PROCS
    except (TypeError, ValueError):
        return _DEFAULT_VALIDATORS_PROCS


# --------------------------------------------------------------------- #
# Process-pool machinery (module-level so the initializer + task fn are
# picklable by the spawn ProcessPoolExecutor; the MODEL is NEVER pickled).
# --------------------------------------------------------------------- #

#: Per-WORKER resident NLI, loaded once by :func:`_pool_worker_init`.
_WORKER_NLI: Any = None


def default_nli_factory() -> Any:
    """Picklable module-level factory → the production NLI singleton loader."""
    return NliClassifier.get_or_load()


def _resolve_factory_dotted() -> str:
    """The ``module:function`` dotted path each worker calls at init.

    Unset / blank → the sentinel empty string (worker uses
    :func:`default_nli_factory`). A test seam points it at a stub factory.
    """
    return os.environ.get(_ENV_NLI_VALIDATORS_FACTORY, "").strip()


def _import_factory(dotted: str) -> Any:
    """Import + return the ``module:function`` factory (empty → default)."""
    if not dotted:
        return default_nli_factory
    mod_name, _, fn_name = dotted.partition(":")
    module = importlib.import_module(mod_name)
    return getattr(module, fn_name)


def _pool_worker_init(factory_dotted: str) -> None:
    """Pool initializer — load this worker's OWN NLI exactly once.

    Runs once per worker process. Stores the resident model on the module
    global so every task the worker handles reuses it (no per-task reload).
    """
    global _WORKER_NLI
    _WORKER_NLI = _import_factory(factory_dotted)()


def _pool_score_task(
    task: Tuple[int, str, List[Dict[str, str]], float, float],
) -> Tuple[int, GroundednessReport]:
    """Worker task — score ONE block end-to-end via the UNCHANGED scorer.

    ``task`` = ``(idx, prose_text, cited_passages, entailment_floor,
    contradiction_floor)``. Returns ``(idx, GroundednessReport)`` (both
    picklable); ``idx`` keys the result back to the block in the parent.
    """
    idx, prose_text, cited_passages, ent_floor, con_floor = task
    report = score_groundedness(
        prose_text,
        cited_passages,
        nli=_WORKER_NLI,
        entailment_floor=ent_floor,
        contradiction_floor=con_floor,
    )
    return idx, report


def _make_scoring_pool(n_procs: int, factory_dotted: str) -> Any:
    """Build the spawn ``ProcessPoolExecutor`` (isolated as a test seam).

    Uses the ``spawn`` start method (a fresh interpreter per worker → no
    inherited CUDA context / fork-unsafe torch state) with the module-level
    initializer that pins one NLI per worker. Isolated so tests can inject a
    synchronous / in-process executor without spawning real subprocesses.
    """
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = _mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=max(1, n_procs),
        mp_context=ctx,
        initializer=_pool_worker_init,
        initargs=(factory_dotted,),
    )


# --------------------------------------------------------------------- #
# Per-block resume sidecar (content-addressed GroundednessReport cache).
# --------------------------------------------------------------------- #


def _block_fingerprint(
    prose_text: str,
    cited_passages: List[Dict[str, str]],
    entailment_floor: float,
    contradiction_floor: float,
    nli: Any,
) -> str:
    """Content-address one block's SCORING inputs → a sha256 hex key.

    Hashes exactly what determines the cached ``GroundednessReport``: the
    prose text, the SORTED cited (chunk_id, text) pairs, the entailment /
    contradiction floors, the scorer surface version, the NLI model revision,
    and the NLI DEVICE-CLASS (``cuda`` vs ``cpu`` — not the exact index, since
    a cuda:0 vs cuda:1 pin scores identically to within the documented
    softmax-non-associativity tolerance). Any change to any axis moves the key
    so a stale report is never served for a changed input.
    """
    dev = getattr(nli, "device", None)
    dev_class = "cuda" if isinstance(dev, str) and dev.startswith("cuda") else "cpu"
    payload = {
        "prose": hashlib.sha256(prose_text.encode("utf-8")).hexdigest(),
        "passages": [
            [p.get("chunk_id", ""), p.get("text", "")]
            for p in sorted(
                cited_passages, key=lambda p: (p.get("chunk_id", ""), p.get("text", ""))
            )
        ],
        "entailment_floor": entailment_floor,
        "contradiction_floor": contradiction_floor,
        "scorer_version": SCORER_VERSION,
        "nli_revision": getattr(nli, "_revision", None),
        "device_class": dev_class,
    }
    # Fold the opt-in top-K windowed stage-1 mode into the key ONLY when it is
    # active: a top-K-scored report must never be served to a legacy-grid run
    # (or vice versa), while legacy fingerprints stay byte-identical to every
    # sidecar written before the flag existed.
    from lib.retrieval.groundedness import (
        resolve_groundedness_frontier,
        resolve_groundedness_frontier_width,
        resolve_groundedness_s1_topk,
    )

    # FRONTIER supersedes top-K (mutually exclusive) — mirror that precedence in
    # the key so a frontier-scored report never crosses modes with a top-K /
    # legacy sidecar. Each mode folds its own key ONLY when active, so the
    # legacy fingerprint stays byte-identical to every pre-flag sidecar.
    if resolve_groundedness_frontier():
        payload["frontier"] = 1
        payload["frontier_width"] = resolve_groundedness_frontier_width()
    else:
        _s1k = resolve_groundedness_s1_topk()
        if _s1k > 0:
            payload["s1_topk"] = _s1k
    return compute_fingerprint(payload)


def _report_from_cache_dict(d: Dict[str, Any]) -> GroundednessReport:
    """Reconstruct a ``GroundednessReport`` from a cached :meth:`to_dict` blob.

    A true drop-in for a freshly-scored report — the validate loop reads
    ``scored_count`` / ``unsupported_count`` / ``contradicted_count`` /
    ``groundedness_rate`` / ``computational_count`` / ``filtered_count`` /
    ``nli_model_revision`` / ``nli_device`` and the per-claim ``windowed``
    flags, all restored here.
    """
    claims = [
        ClaimVerdict(
            claim_text=c.get("claim_text", ""),
            verdict=c.get("verdict", ""),
            entailment=float(c.get("entailment", 0.0)),
            contradiction=float(c.get("contradiction", 0.0)),
            best_chunk_id=c.get("best_chunk_id"),
            windowed=bool(c.get("windowed", False)),
            best_chunk_cited=c.get("best_chunk_cited"),
        )
        for c in d.get("claims", [])
        if isinstance(c, dict)
    ]
    return GroundednessReport(
        available=bool(d.get("available", False)),
        claims=claims,
        groundedness_rate=float(d.get("groundedness_rate", 0.0)),
        unsupported_count=int(d.get("unsupported_count", 0)),
        contradicted_count=int(d.get("contradicted_count", 0)),
        scored_count=int(d.get("scored_count", 0)),
        thresholds=dict(d.get("thresholds", {})),
        thresholds_provenance=d.get(
            "thresholds_provenance", THRESHOLDS_PROVENANCE_DEFAULTS
        ),
        nli_model_revision=d.get("nli_model_revision"),
        nli_device=d.get("nli_device"),
        reason=d.get("reason"),
        computational_count=int(d.get("computational_count", 0)),
        filtered_count=int(d.get("filtered_count", 0)),
        scorer_version=d.get("scorer_version", SCORER_VERSION),
    )


def _resolve_cache_store(inputs: Dict[str, Any]) -> Optional[CheckpointStore]:
    """Resolve the per-block resume sidecar store, or ``None`` (disabled).

    Cache dir precedence: an explicit ``inputs["prose_entailment_cache_dir"]``
    (tests / operator override) → the rewrite export dir inferred from
    ``inputs["blocks_final_path"]`` (``<export>/.prose_entailment_cache``,
    following the reasoning_qc_cache "next to the export" precedent). No
    resolvable dir → ``None`` (byte-identical, no cache activity). The
    returned store self-gates on ``ED4ALL_VALIDATION_CHECKPOINT`` (site) /
    ``ED4ALL_GENERATION_CHECKPOINT`` (family); a falsey flag → ``None`` here.
    """
    cache_dir = inputs.get("prose_entailment_cache_dir")
    if not (isinstance(cache_dir, (str, Path)) and str(cache_dir)):
        bfp = inputs.get("blocks_final_path")
        if isinstance(bfp, str) and bfp:
            cache_dir = Path(bfp).resolve().parent / ".prose_entailment_cache"
        else:
            cache_dir = None
    if not cache_dir:
        return None
    store = CheckpointStore(
        Path(cache_dir) / _CHECKPOINT_BASENAME,
        site_id=_CHECKPOINT_SITE_ID,
        site_env=_ENV_VALIDATION_CHECKPOINT,
    )
    return store if store.enabled else None


def load_chunk_provenance_index(chunks_jsonl_path: str) -> Dict[str, List[str]]:
    """Build a ``{provenance_ref -> [chunk_id, ...]}`` reverse index from a
    ``chunks.jsonl`` file.

    Each chunk record carries its source anchors on
    ``source.source_references[].sourceId`` (the ``semantik:{slug}#{anchor}``
    Source-Provenance contract) and its own
    canonical chunk id on the top-level ``id`` field
    (``{course}_chunk_NNNNN``). This inverts that: for every ``sourceId`` a
    chunk declares, the chunk's id is appended to that ref's list. Because a
    section-level ref (e.g. ``semantik:slug#1-1-introduction-to-whole-numbers``)
    is shared by every chunk emitted from that section, the ref naturally maps
    to ALL of the section's chunk ids — the section-ref tolerance the gate needs
    for the ``data-cf-source-ids`` anchor form.

    Best-effort: an absent / unreadable path returns an empty index so the
    gate's graceful NO_GROUNDING path stays intact. Order-preserving,
    per-ref de-duplicated.
    """
    index: Dict[str, List[str]] = {}
    path = Path(chunks_jsonl_path)
    if not path.exists():
        return index
    try:
        import json as _json

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                chunk_id = chunk.get("id") or chunk.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    continue
                chunk_id = chunk_id.strip()
                source = chunk.get("source")
                if not isinstance(source, dict):
                    continue
                src_refs = source.get("source_references") or []
                if not isinstance(src_refs, list):
                    continue
                for ref in src_refs:
                    if not isinstance(ref, dict):
                        continue
                    sid = ref.get("sourceId")
                    if not isinstance(sid, str) or not sid.strip():
                        continue
                    bucket = index.setdefault(sid.strip(), [])
                    if chunk_id not in bucket:
                        bucket.append(chunk_id)
    except OSError as exc:
        logger.debug(
            "load_chunk_provenance_index: read of %s failed: %s",
            chunks_jsonl_path, exc,
        )
        return {}
    return index


# --------------------------------------------------------------------- #
# Canonical GateIssue codes
# --------------------------------------------------------------------- #

_CODE_BLOCK_PROSE_UNGROUNDED: str = "BLOCK_PROSE_UNGROUNDED"
_CODE_BLOCK_PROSE_CONTRADICTED: str = "BLOCK_PROSE_CONTRADICTED"
_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE: str = "BLOCK_PROSE_NO_GROUNDING_SOURCE"
_CODE_NLI_DEPS_MISSING: str = "NLI_DEPS_MISSING"

_DECISION_TYPE: str = "block_prose_entailment_check"


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs["blocks"]``.

    Mirrors ``claim_support._coerce_blocks`` /
    ``rewrite_source_grounding._coerce_blocks``.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _block_should_skip(block: Block) -> bool:
    """True when the block's block_type / content_type is in the skip set."""
    if block.block_type in _SKIPPED_BLOCK_TYPES:
        return True
    if (
        block.content_type_label
        and block.content_type_label in _SKIPPED_BLOCK_TYPES
    ):
        return True
    return False


def _resolve_block_cited_passages(
    block: Block,
    source_chunks: Dict[str, str],
    provenance_index: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    """Resolve the block's cited chunks into ``score_groundedness`` passages.

    Walks ``block.source_references[].sourceId`` + ``block.source_ids`` (the
    same id surfaces ``_resolve_block_source_chunks`` consults), looks each up
    in ``source_chunks``, and returns ``[{"chunk_id": sid, "text": text}, ...]``
    preserving the chunk_id so the report's ``best_chunk_id`` is meaningful.
    Duplicate ids are de-duplicated (first occurrence wins).

    When ``provenance_index`` is supplied (opt-in
    ``ED4ALL_PROSE_GATE_PROVENANCE_RESOLVE``), an ADD pass runs after direct
    resolution: rewrite-tier blocks carry provenance as ``semantik:{slug}#anchor``
    refs, not ``{course}_chunk_NNNNN`` chunk ids, so a block that resolves to
    NOTHING directly is re-resolved by mapping each cited ref through the index
    (``ref -> [chunk_id, ...]``, a section-level ref covering ALL of the
    section's chunk ids) and looking those chunk ids up in ``source_chunks``.
    Anti-fabrication: only EXISTING refs map to EXISTING chunks; directly
    resolved ids are never removed — the index can only ADD supporters.
    """
    seen: set = set()
    passages: List[Dict[str, str]] = []
    candidate_ids: List[str] = []
    for ref in block.source_references or ():
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid:
                candidate_ids.append(sid)
    for sid in block.source_ids or ():
        if isinstance(sid, str) and sid:
            candidate_ids.append(sid)
    for sid in candidate_ids:
        if sid in seen:
            continue
        seen.add(sid)
        text = source_chunks.get(sid)
        if isinstance(text, str) and text.strip():
            passages.append({"chunk_id": sid, "text": text})

    # Opt-in provenance resolution ADD pass. Fires when a block's directly-cited
    # ids resolved to nothing (the rewrite-tier semantik-ref → chunk-id gap);
    # never removes an already-resolved passage (ADD-only).
    if provenance_index and not passages:
        for sid in candidate_ids:
            for chunk_id in provenance_index.get(sid, ()):  # section-level fan-out
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                text = source_chunks.get(chunk_id)
                if isinstance(text, str) and text.strip():
                    passages.append({"chunk_id": chunk_id, "text": text})
    return passages


def _block_score_args(
    block: Block,
    source_chunks: Dict[str, str],
    provenance_index: Optional[Dict[str, List[str]]] = None,
    cache: Any = None,
) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    """Return ``(prose_text, cited_passages)`` iff ``block`` reaches the
    ``score_groundedness`` call in :meth:`BlockProseEntailmentValidator.validate`,
    else ``None``.

    This is the SINGLE-SOURCE gate for "is this block scored?" — it replays the
    EXACT sequence of skip decisions the main validate loop makes before it
    scores a block (string content, not in the skip set, non-empty stripped
    prose, ≥1 resolvable cited passage). It exists so the opt-in micro-batch
    pre-pass can pre-compute reports for precisely the blocks the serial loop
    would score, guaranteeing the parallelized path is verdict-identical. All
    the branchy no-score handling (skip-silent / empty-prose / no-grounding
    warning) stays in the main loop; this helper only answers the yes/no + hands
    back the deterministic scoring inputs.
    """
    content = block.content
    if not isinstance(content, str):
        return None
    if _block_should_skip(block):
        return None
    prose_text = (
        cache.stripped_text(block)
        if cache is not None
        else _strip_html_to_text(content)
    )
    if not prose_text.strip():
        return None
    cited_passages = (
        cache.resolved_passages(block, source_chunks, provenance_index)
        if cache is not None
        else _resolve_block_cited_passages(block, source_chunks, provenance_index)
    )
    if not cited_passages:
        return None
    return prose_text, cited_passages


def _emit_decision(
    capture: Any,
    block: Block,
    *,
    passed: bool,
    failure_codes: List[str],
    scored_count: int,
    entailed_count: int,
    unsupported_count: int,
    contradicted_count: int,
    computational_count: int,
    filtered_count: int,
    windowed_count: int,
    block_entailment_rate: float,
    min_block_entailment_rate: float,
    entailment_floor: float,
    nli_revision: Optional[str],
    nli_device: Optional[str],
    shadow: bool,
) -> None:
    """Emit one ``block_prose_entailment_check`` decision per scored block.

    Rationale interpolates dynamic per-call signals (≥20 chars) so the
    capture is replayable post-hoc — see the decision-capture contract in
    root ``CLAUDE.md`` § "LLM call-site instrumentation". Swallow
    ``log_decision`` exceptions (capture must not break the gate).
    """
    if capture is None:
        return
    decision = (
        "passed" if passed else f"failed:{','.join(failure_codes) or 'unknown'}"
    )
    rationale = (
        f"Block-prose NLI entailment on Block {block.block_id!r}: "
        f"block_type={block.block_type!r}, "
        f"scored_count={scored_count}, "
        f"entailed_count={entailed_count}, "
        f"unsupported_count={unsupported_count}, "
        f"contradicted_count={contradicted_count}, "
        f"computational_count={computational_count}, "
        f"filtered_count={filtered_count}, "
        f"windowed_count={windowed_count}, "
        f"block_entailment_rate={block_entailment_rate:.4f}, "
        f"min_rate={min_block_entailment_rate:.2f}, "
        f"entailment_floor={entailment_floor:.2f}, "
        f"nli_revision={nli_revision}, nli_device={nli_device}, "
        f"shadow={shadow}, "
        f"failure_codes={','.join(failure_codes) or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on %s: %s",
            _DECISION_TYPE, exc,
        )


class BlockProseEntailmentValidator:
    """Per-block full-prose NLI entailment gate (W4).

    Iterates every audited Block (skip set per
    ``claim_support._SKIPPED_BLOCK_TYPES``); for each rewrite-tier
    (string-content) block, strips the HTML prose and scores it against the
    block's cited source chunk(s) via ``score_groundedness`` (windowed
    scorer-v2). Fires ``BLOCK_PROSE_UNGROUNDED`` when the per-block
    entailment rate falls below ``min_block_entailment_rate`` and
    ``BLOCK_PROSE_CONTRADICTED`` when the contradicted-prose rate crosses
    ``max_contradicted_rate``.

    Inputs:
        blocks: List[Block]
        source_chunks: Dict[str, str]
            Mapping of canonical sourceId (e.g. ``semantik:slug#blk_0``) to the
            chunk's plain-text body. The gate-input builder populates this
            from the staging manifest + SemantiK chunks.jsonl body fallback; tests
            inject it directly.
        shadow: bool
            §4.3 shadow knob. When truthy, emitted ``BLOCK_PROSE_*`` issues are
            forced warning-severity and ``action`` is never ``"regenerate"``
            (compute + capture only). Wired via gate ``config.shadow: true``.
        min_block_entailment_rate / entailment_floor / contradiction_floor /
        max_contradicted_rate: Optional[float]
            Thresholds (merged from gate ``config.thresholds`` by the executor).
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per scored block.
        nli: Optional[NliClassifier]
            Test seam — inject a fake. When None, calls
            ``NliClassifier.get_or_load`` and threads the result into
            ``score_groundedness(nli=...)``.

    Graceful-degrade: ``NliClassifier.get_or_load()`` returns None AND
    ``TRAINFORGE_REQUIRE_EMBEDDINGS`` unset → single warning ``NLI_DEPS_MISSING``
    GateIssue, ``passed=True, action=None``. Strict mode raises ``RuntimeError``.
    """

    name = "block_prose_entailment"
    version = "1.0.0"

    def __init__(
        self,
        *,
        min_block_entailment_rate: float = _DEFAULT_MIN_BLOCK_ENTAILMENT_RATE,
        entailment_floor: float = _DEFAULT_ENTAILMENT_FLOOR,
        contradiction_floor: float = _DEFAULT_CONTRADICTION_FLOOR,
        max_contradicted_rate: float = _DEFAULT_MAX_CONTRADICTED_RATE,
        nli: Optional[NliClassifier] = None,
    ) -> None:
        self._min_block_rate = float(min_block_entailment_rate)
        self._entailment_floor = float(entailment_floor)
        self._contradiction_floor = float(contradiction_floor)
        self._max_contradicted = float(max_contradicted_rate)
        self._nli_override = nli

    def _get_nli(self) -> Optional[NliClassifier]:
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    def _resolve_thresholds(
        self, inputs: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        """Resolve thresholds, honoring the executor's gate.config merge.

        The executor merges ``gate.config.thresholds`` into the inputs dict
        before dispatch (per the pattern at ``executor.py:1442``):

        .. code-block:: yaml

            config:
              thresholds:
                min_block_entailment_rate: 0.60
                entailment_floor: 0.70
                contradiction_floor: 0.50
                max_contradicted_rate: 0.05
        """
        return (
            float(
                inputs.get("min_block_entailment_rate", self._min_block_rate)
            ),
            float(inputs.get("entailment_floor", self._entailment_floor)),
            float(
                inputs.get("contradiction_floor", self._contradiction_floor)
            ),
            float(
                inputs.get("max_contradicted_rate", self._max_contradicted)
            ),
        )

    def _pool_score_misses(
        self,
        misses: List[Tuple[int, Any, str, List[Dict[str, str]]]],
        entailment_floor: float,
        contradiction_floor: float,
        run_id: Optional[str],
    ) -> Tuple[Optional[Dict[int, GroundednessReport]], bool]:
        """Score the cache-MISS blocks across the spawn process pool.

        Returns ``(reports_by_idx, stop_pending)``. ``reports_by_idx is None``
        signals a graceful degrade to the serial path (pool start / worker
        failure) — the caller re-scores every miss serially (idempotent, no
        double-append since nothing is persisted on the ``None`` path). A
        partial dict + ``stop_pending=True`` means an ``ed4all stop`` sentinel
        tripped BEFORE some tasks were submitted: the already-submitted tasks
        were drained (finish in-flight) and the caller persists them, then
        raises the cooperative stop.
        """
        from concurrent.futures import as_completed
        from concurrent.futures.process import BrokenProcessPool

        tasks = [
            (idx, prose, passages, entailment_floor, contradiction_floor)
            for (idx, _block, prose, passages) in misses
        ]
        n_procs = min(_resolve_validators_procs(), len(tasks))
        factory_dotted = _resolve_factory_dotted()
        try:
            pool = _make_scoring_pool(n_procs, factory_dotted)
        except Exception as exc:  # noqa: BLE001 — any pool-start failure → serial
            logger.warning(
                "block_prose_entailment: process pool start failed (%s); "
                "falling back to serial scoring.", exc,
            )
            return None, False

        reports: Dict[int, GroundednessReport] = {}
        stop_pending = False
        try:
            futures = {}
            for task in tasks:
                # STOP-SENTINEL POLL before submitting each shard task.
                if _stop_requested(run_id):
                    stop_pending = True
                    break
                futures[pool.submit(_pool_score_task, task)] = task
            for fut in as_completed(futures):
                idx, report = fut.result()
                reports[idx] = report
        except BrokenProcessPool as exc:
            logger.warning(
                "block_prose_entailment: process pool broke mid-scoring (%s); "
                "falling back to serial scoring.", exc,
            )
            pool.shutdown(wait=False)
            return None, False
        except GracefulStopRequested:
            raise
        except Exception as exc:  # noqa: BLE001 — any pool error → serial
            logger.warning(
                "block_prose_entailment: process pool error (%s); falling back "
                "to serial scoring.", exc,
            )
            pool.shutdown(wait=False)
            return None, False
        finally:
            pool.shutdown(wait=True)
        return reports, stop_pending

    def _crossblock_score_misses(
        self,
        misses: List[Tuple[int, Any, str, List[Dict[str, str]]]],
        nli: Any,
        entailment_floor: float,
        contradiction_floor: float,
        run_id: Optional[str],
        *,
        store: Optional[CheckpointStore] = None,
        fps: Optional[Dict[int, str]] = None,
    ) -> Tuple[Optional[Dict[int, GroundednessReport]], bool, set]:
        """Score the cache-MISS blocks with cross-block NLI thread fan-out.

        Single-process, threads + the existing coalescing dispatcher
        (``ED4ALL_NLI_CROSSBLOCK``): each worker thread runs the UNCHANGED
        ``score_groundedness(prose, passages, nli=dispatcher, ...)`` — its Python
        work is trivial pair-list assembly then a GIL-released ``event.wait``;
        tokenize + forward run on the dispatcher's single scorer thread, which
        coalesces pairs ACROSS blocks into full batches. Verdict-identical to
        serial (batched-vs-serial scores differ only in low-order softmax bits,
        the documented tolerance vs the 0.70/0.50 floors).

        Each block's report is PERSISTED to ``store`` as soon as its future
        completes (when ``store``/``fps`` are given) — a killed / stopped gate
        keeps every finished block, matching the serial path's incremental
        sidecar contract — and progress is logged every 10 completions so the
        gate is diagnosable live. The stop sentinel is polled both before each
        submit AND on every completion; a mid-gate stop cancels all queued
        futures (running ones finish and persist), so ``ed4all stop`` pauses
        within ~one block's scoring time instead of after the whole fan-out.

        Returns ``(reports_by_idx, stop_pending, persisted_idx_set)``; ``None``
        reports ⇒ graceful degrade to serial (no dispatcher, or any thread-path
        failure); the caller persists + merges everything not already in
        ``persisted_idx_set``.
        """
        from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

        from lib.classifiers.nli_microbatch import (
            get_dispatcher,
            resolve_crossblock_threads,
        )

        dispatcher = get_dispatcher(nli)
        if dispatcher is None:
            return None, False, set()
        n_threads = min(resolve_crossblock_threads(), len(misses))

        def _score_one(
            task: Tuple[int, Any, str, List[Dict[str, str]]],
        ) -> Tuple[int, GroundednessReport]:
            idx, _block, prose, passages = task
            rep = score_groundedness(
                prose,
                passages,
                nli=dispatcher,
                entailment_floor=entailment_floor,
                contradiction_floor=contradiction_floor,
            )
            return idx, rep

        block_by_idx = {task[0]: task[1] for task in misses}
        reports: Dict[int, GroundednessReport] = {}
        persisted: set = set()
        stop_pending = False
        done_count = 0
        ex = ThreadPoolExecutor(max_workers=max(1, n_threads))
        try:
            futures = []
            for task in misses:
                # STOP-SENTINEL POLL before submitting each block.
                if _stop_requested(run_id):
                    stop_pending = True
                    break
                futures.append(ex.submit(_score_one, task))
            for fut in as_completed(futures):
                try:
                    idx, rep = fut.result()
                except CancelledError:
                    continue
                reports[idx] = rep
                done_count += 1
                # Incremental persist: a killed gate keeps every finished block.
                if store is not None and fps is not None and rep.available:
                    try:
                        store.append(
                            block_by_idx[idx].block_id, fps[idx], rep.to_dict()
                        )
                        persisted.add(idx)
                    except Exception as p_exc:  # noqa: BLE001 — persist is best-effort
                        logger.warning(
                            "block_prose_entailment: incremental sidecar persist "
                            "failed for block %s (%s); caller will retry.",
                            getattr(block_by_idx[idx], "block_id", idx), p_exc,
                        )
                if done_count % 10 == 0 or done_count == len(futures):
                    logger.info(
                        "block_prose_entailment: crossblock %d/%d blocks scored",
                        done_count, len(futures),
                    )
                # STOP-SENTINEL POLL on completion: cancel queued futures so the
                # gate pauses within ~one block, not after the whole fan-out.
                if not stop_pending and _stop_requested(run_id):
                    stop_pending = True
                    for pending_fut in futures:
                        pending_fut.cancel()
        except GracefulStopRequested:
            raise
        except Exception as exc:  # noqa: BLE001 — any thread-path error → serial
            logger.warning(
                "block_prose_entailment: cross-block thread scoring failed (%s); "
                "falling back to serial scoring.", exc,
            )
            return None, False, persisted
        finally:
            ex.shutdown(wait=True)
        return reports, stop_pending, persisted

    def _score_scorable(
        self,
        blocks: List[Any],
        nli: Any,
        source_chunks: Dict[str, str],
        provenance_index: Optional[Dict[str, List[str]]],
        entailment_floor: float,
        contradiction_floor: float,
        store: Optional[CheckpointStore],
        cache: Dict[str, Dict[str, Any]],
        run_id: Optional[str],
        feature_cache: Any = None,
    ) -> Dict[int, GroundednessReport]:
        """Produce ``{block_index: GroundednessReport}`` for every SCORABLE block.

        Combines the three overhaul features (all default-off byte-identical):

        * **Resume sidecar** — a per-block content-addressed cache HIT restores
          the report with NO scoring; a MISS is scored then persisted (only when
          ``available``). A killed / stopped validation resumes nearly free.
        * **Process-pool scoring** — MISS blocks are sharded across the spawn
          pool when the flag is on (serial fallback otherwise).
        * **Stop-sentinel** — the serial miss loop probes the run-scoped stop
          sentinel before each block, and the pool loop before each submit; on a
          stop the completed reports are persisted and ``GracefulStopRequested``
          is raised (the executor maps it to ``paused``).

        The "scorable" set is exactly the blocks the ``validate`` loop reaches
        the scoring branch for (via :func:`_block_score_args`), so every returned
        idx is consumed and no report is fabricated for a skipped block.
        """
        scorable: List[Tuple[int, Any, str, List[Dict[str, str]]]] = []
        for idx, block in enumerate(blocks):
            args = _block_score_args(
                block, source_chunks, provenance_index, feature_cache
            )
            if args is not None:
                scorable.append((idx, block, args[0], args[1]))
        reports: Dict[int, GroundednessReport] = {}
        if not scorable:
            return reports

        # Phase 1 — resolve cache HITs; collect MISSes with their fingerprints.
        fps: Dict[int, str] = {}
        misses: List[Tuple[int, Any, str, List[Dict[str, str]]]] = []
        for idx, block, prose, passages in scorable:
            fp = _block_fingerprint(
                prose, passages, entailment_floor, contradiction_floor, nli
            )
            fps[idx] = fp
            hit = (
                store.reuse(block.block_id, fp, cache=cache)
                if store is not None
                else None
            )
            if hit is not None:
                reports[idx] = _report_from_cache_dict(hit)
            else:
                misses.append((idx, block, prose, passages))
        if not misses:
            return reports

        # Phase 2 — score the MISSes (pool / cross-block threads / serial),
        # persist + merge. The dormant spawn pool wins for back-compat when its
        # flag is on; the cross-block thread path is tried when the pool is off
        # OR degraded (returns None). Both feed the SAME merge/persist block.
        pool_reports: Optional[Dict[int, GroundednessReport]] = None
        stop_pending = False
        already_persisted: set = set()
        if _pool_scoring_enabled():
            pool_reports, stop_pending = self._pool_score_misses(
                misses, entailment_floor, contradiction_floor, run_id
            )
        if pool_reports is None and _crossblock_enabled():
            pool_reports, stop_pending, already_persisted = (
                self._crossblock_score_misses(
                    misses, nli, entailment_floor, contradiction_floor, run_id,
                    store=store, fps=fps,
                )
            )

        if pool_reports is not None:
            for idx, block, _prose, _passages in misses:
                rep = pool_reports.get(idx)
                if rep is None:
                    # Not scored (stop tripped before its submit); left for the
                    # validate-loop defensive inline fallback on a cleared stop.
                    continue
                reports[idx] = rep
                if store is not None and rep.available and idx not in already_persisted:
                    store.append(block.block_id, fps[idx], rep.to_dict())
            if stop_pending:
                # Persisted completed reports above; now raise the cooperative
                # stop (a no-op if the sentinel was cleared meanwhile — the
                # remaining misses then take the validate-loop inline fallback).
                _check_stop(_STOP_SITE_ID, len(reports), run_id)
            return reports

        # Serial fallback (pool disabled or degraded).
        for idx, block, prose, passages in misses:
            # STOP-SENTINEL POLL before scoring each block (completed blocks are
            # already persisted below, so worst-case loss is one in-flight call).
            _check_stop(_STOP_SITE_ID, len(reports), run_id)
            rep = score_groundedness(
                prose,
                passages,
                nli=nli,
                entailment_floor=entailment_floor,
                contradiction_floor=contradiction_floor,
            )
            reports[idx] = rep
            if store is not None and rep.available:
                store.append(block.block_id, fps[idx], rep.to_dict())
        return reports

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        shadow = bool(inputs.get("shadow", False))

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        (
            min_block_rate,
            entailment_floor,
            contradiction_floor,
            max_contradicted,
        ) = self._resolve_thresholds(inputs)

        nli = self._get_nli()
        # Graceful-degrade: pre-check the loader exactly like claim_support so
        # the degrade branch is identical across the two NLI block gates and
        # the test seam (nli=...) overrides cleanly.
        if nli is None:
            if is_strict_mode():
                raise RuntimeError(
                    "BlockProseEntailmentValidator requires the NLI "
                    "classifier (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) "
                    "but TRAINFORGE_REQUIRE_EMBEDDINGS is set and "
                    "NliClassifier.get_or_load() returned None. "
                    "Install transformers + torch via "
                    "`pip install -e .[embedding]` or unset "
                    "TRAINFORGE_REQUIRE_EMBEDDINGS for graceful degradation."
                )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code=_CODE_NLI_DEPS_MISSING,
                        message=(
                            "NliClassifier.get_or_load() returned None; "
                            "W4 per-block prose NLI entailment gate skipped. "
                            "Install transformers + torch via "
                            "`pip install -e .[embedding]` to enable. "
                            "Set TRAINFORGE_REQUIRE_EMBEDDINGS=true to fail "
                            "closed instead."
                        ),
                        suggestion=(
                            "The graceful-degrade path is intentional for "
                            "CPU-only dev boxes; production runs should "
                            "install the embedding extras."
                        ),
                    )
                ],
                action=None,
            )

        source_chunks_raw = inputs.get("source_chunks", {}) or {}
        if not isinstance(source_chunks_raw, dict):
            source_chunks_raw = {}
        source_chunks: Dict[str, str] = {
            str(k): v for k, v in source_chunks_raw.items()
            if isinstance(v, str)
        }

        # Opt-in gate-side provenance resolution (default OFF → byte-identical).
        # When enabled AND the runner threaded a chunk_provenance_index (a
        # {provenance_ref -> [chunk_id, ...]} reverse index), a block whose
        # directly-cited semantik refs resolve to nothing is re-resolved through
        # the index. Off / absent → provenance_index stays None → legacy path.
        provenance_index: Optional[Dict[str, List[str]]] = None
        if _provenance_resolve_enabled():
            raw_index = inputs.get("chunk_provenance_index")
            if isinstance(raw_index, dict):
                provenance_index = {
                    str(k): [str(c) for c in v]
                    for k, v in raw_index.items()
                    if isinstance(v, (list, tuple))
                }

        # Severity of the BLOCK_PROSE_* failures honors the shadow knob:
        # warning during shadow (compute + capture only), critical after the
        # flip (so the validator's own ``passed`` flips on a real failure,
        # mirroring rewrite_source_grounding's ``passed = len(critical) == 0``).
        fail_severity = "warning" if shadow else "critical"

        # Run-scoped id for the cooperative-stop sentinel (explicit input wins;
        # else stop_control reads ED4ALL_RUN_ID; a global STOP_ALL works without
        # either).
        run_id = inputs.get("run_id")
        if not (isinstance(run_id, str) and run_id):
            run_id = None

        # Per-block resume sidecar (ED4ALL_VALIDATION_CHECKPOINT, default ON;
        # disabled → None → byte-identical). Loaded ONCE per validate.
        store = _resolve_cache_store(inputs)
        cache = store.load() if store is not None else {}

        # Pre-score every SCORABLE block through the three overhaul features
        # (resume-cache HIT / spawn process-pool / serial fallback), all keyed by
        # block index. Feature-off → this is the serial inline scoring the loop
        # historically did, just hoisted here; ALL issue/counter/capture assembly
        # below still runs SERIALLY in original block order so the emitted
        # GateResult is verdict-identical + order-stable. A cooperative stop
        # (GracefulStopRequested) propagates from here after persisting the
        # completed blocks (executor maps it to ``paused``).
        # Opt-in shared per-block feature cache (ED4ALL_VALIDATION_FEATURE_CACHE).
        # Absent → None → every per-block strip/passage self-computes exactly as
        # before (byte-identical). The cache's stripped_text / resolved_passages
        # delegate to the SAME strip_html_to_text / _resolve_block_cited_passages,
        # so results are verdict-identical — it only removes recomputation shared
        # across the 52-gate suite.
        feature_cache = inputs.get("feature_cache")

        precomputed_reports = self._score_scorable(
            blocks, nli, source_chunks, provenance_index,
            entailment_floor, contradiction_floor, store, cache, run_id,
            feature_cache,
        )

        issues: List[GateIssue] = []
        any_regenerate = False
        audited_blocks = 0
        passed_blocks = 0

        for idx, block in enumerate(blocks):
            content = block.content
            if not isinstance(content, str):
                # Outline-tier dict content — skip silently (rewrite-tier seam
                # is where prose entailment is checked).
                continue
            if _block_should_skip(block):
                # Skip silently — no event, no issue (matches claim_support).
                continue

            prose_text = (
                feature_cache.stripped_text(block)
                if feature_cache is not None
                else _strip_html_to_text(content)
            )
            if not prose_text.strip():
                continue

            cited_passages = (
                feature_cache.resolved_passages(
                    block, source_chunks, provenance_index
                )
                if feature_cache is not None
                else _resolve_block_cited_passages(
                    block, source_chunks, provenance_index
                )
            )

            if not cited_passages:
                # No grounding surface. The resolution gate (rewrite_source_refs)
                # owns the missing-source structural failure; this gate emits a
                # warning so the operator knows the entailment signal is
                # unavailable, but passes the block.
                audited_blocks += 1
                passed_blocks += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE,
                            message=(
                                f"Rewrite-tier Block {block.block_id!r} "
                                f"(block_type={block.block_type!r}) declares "
                                f"no resolvable cited source chunks; per-block "
                                f"prose entailment cannot be evaluated."
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block,
                    passed=True,
                    failure_codes=[_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE],
                    scored_count=0,
                    entailed_count=0,
                    unsupported_count=0,
                    contradicted_count=0,
                    computational_count=0,
                    filtered_count=0,
                    windowed_count=0,
                    block_entailment_rate=0.0,
                    min_block_entailment_rate=min_block_rate,
                    entailment_floor=entailment_floor,
                    nli_revision=None,
                    nli_device=None,
                    shadow=shadow,
                )
                continue

            audited_blocks += 1
            report = precomputed_reports.get(idx)
            if report is None:
                # Defensive inline fallback — every scorable idx is pre-computed
                # by ``_score_scorable`` (cache HIT / pool / serial), so this is
                # only reached in the rare "pool stop tripped then the sentinel
                # was cleared" window, where the un-submitted misses land here.
                report = score_groundedness(
                    prose_text,
                    cited_passages,
                    nli=nli,
                    entailment_floor=entailment_floor,
                    contradiction_floor=contradiction_floor,
                )

            if not report.available:
                # Defensive: the pre-check above already handled the missing
                # loader, but score_groundedness can independently degrade
                # (e.g. an injected stub that resolves to None). Honor the
                # same graceful-degrade contract.
                if is_strict_mode():
                    raise RuntimeError(
                        "BlockProseEntailmentValidator: score_groundedness "
                        "reported the NLI model unavailable while "
                        "TRAINFORGE_REQUIRE_EMBEDDINGS is set. Install the "
                        "embedding extras or unset the strict flag."
                    )
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=True,
                    score=1.0,
                    issues=[
                        GateIssue(
                            severity="warning",
                            code=_CODE_NLI_DEPS_MISSING,
                            message=(
                                "score_groundedness reported the NLI model "
                                "unavailable; W4 per-block prose NLI "
                                "entailment gate skipped. Install transformers "
                                "+ torch via `pip install -e .[embedding]` to "
                                "enable."
                            ),
                        )
                    ],
                    action=None,
                )

            scored_count = report.scored_count
            entailed_count = (
                scored_count
                - report.unsupported_count
                - report.contradicted_count
            )
            windowed_count = sum(1 for c in report.claims if c.windowed)
            block_entailment_rate = report.groundedness_rate
            contradicted_rate = (
                (report.contradicted_count / scored_count)
                if scored_count > 0
                else 0.0
            )

            failure_codes: List[str] = []

            # BLOCK_PROSE_CONTRADICTED fires standalone above its rate ceiling.
            if (
                scored_count > 0
                and contradicted_rate > max_contradicted
                and len(issues) < _ISSUE_LIST_CAP
            ):
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=_CODE_BLOCK_PROSE_CONTRADICTED,
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} "
                            f"(block_type={block.block_type!r}): "
                            f"{report.contradicted_count}/{scored_count} "
                            f"({contradicted_rate:.1%}) of scored prose "
                            f"sentences were CONTRADICTED by their cited "
                            f"source chunks (contradiction >= "
                            f"{contradiction_floor:.2f}); per-block ceiling "
                            f"is {max_contradicted:.1%}. A contradiction is "
                            f"fabrication that conflicts with source — a hard "
                            f"signal regardless of total entailment rate."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "The block prose contradicts its declared source. "
                            "Re-prompt with the source chunks pinned and "
                            "require strict paraphrase."
                        ),
                    )
                )
                failure_codes.append(_CODE_BLOCK_PROSE_CONTRADICTED)

            # BLOCK_PROSE_UNGROUNDED fires when the entailment rate is below
            # the per-block floor (only when something was scored — never
            # fabricate a failure from an empty denominator).
            if (
                scored_count > 0
                and block_entailment_rate < min_block_rate
                and len(issues) < _ISSUE_LIST_CAP
            ):
                issues.append(
                    GateIssue(
                        severity=fail_severity,
                        code=_CODE_BLOCK_PROSE_UNGROUNDED,
                        message=(
                            f"Rewrite-tier Block {block.block_id!r} "
                            f"(block_type={block.block_type!r}): only "
                            f"{entailed_count}/{scored_count} "
                            f"({block_entailment_rate:.1%}) of scored prose "
                            f"sentences were ENTAILED by their cited source "
                            f"chunks (entailment >= {entailment_floor:.2f} "
                            f"via windowed scorer-v2); minimum required rate "
                            f"is {min_block_rate:.1%}. Cosine grounding can "
                            f"pass on-topic fabrication that NLI catches."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "The rewrite-tier prose is on-topic but not "
                            "entailed by its declared source (likely "
                            "fabrication). Re-prompt with the source chunks "
                            "pinned and require verbatim or near-verbatim "
                            "paraphrase."
                        ),
                    )
                )
                failure_codes.append(_CODE_BLOCK_PROSE_UNGROUNDED)

            block_passed = not failure_codes
            if block_passed:
                passed_blocks += 1
            elif not shadow:
                # Only gate (regenerate) outside shadow mode.
                any_regenerate = True

            _emit_decision(
                capture, block,
                passed=block_passed,
                failure_codes=failure_codes,
                scored_count=scored_count,
                entailed_count=entailed_count,
                unsupported_count=report.unsupported_count,
                contradicted_count=report.contradicted_count,
                computational_count=report.computational_count,
                filtered_count=report.filtered_count,
                windowed_count=windowed_count,
                block_entailment_rate=block_entailment_rate,
                min_block_entailment_rate=min_block_rate,
                entailment_floor=entailment_floor,
                nli_revision=report.nli_model_revision,
                nli_device=report.nli_device,
                shadow=shadow,
            )

        # ``passed`` honors only critical-severity issues — during shadow the
        # BLOCK_PROSE_* failures are warning-severity so ``passed`` stays True;
        # after the critical flip they are critical-severity and ``passed``
        # flips on a real failure (mirrors rewrite_source_grounding).
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = (
            1.0
            if audited_blocks == 0
            else round(passed_blocks / audited_blocks, 4)
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action="regenerate" if any_regenerate else None,
        )


__all__ = [
    "BlockProseEntailmentValidator",
    "load_chunk_provenance_index",
    "default_nli_factory",
    "_ENV_MICROBATCH_VALIDATORS",
    "_ENV_VALIDATORS_PROCS",
    "_ENV_NLI_VALIDATORS_FACTORY",
    "_ENV_VALIDATION_CHECKPOINT",
    "_microbatch_validators_enabled",
    "_pool_scoring_enabled",
    "_resolve_validators_procs",
    "_block_fingerprint",
    "_report_from_cache_dict",
    "_resolve_cache_store",
    "_DEFAULT_MIN_BLOCK_ENTAILMENT_RATE",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_DEFAULT_CONTRADICTION_FLOOR",
    "_DEFAULT_MAX_CONTRADICTED_RATE",
    "_CODE_BLOCK_PROSE_UNGROUNDED",
    "_CODE_BLOCK_PROSE_CONTRADICTED",
    "_CODE_BLOCK_PROSE_NO_GROUNDING_SOURCE",
    "_CODE_NLI_DEPS_MISSING",
]
