"""Near-duplicate anchor-example gate — within-week worked-example de-duplication.

Motivation (plan §6 item 11)
----------------------------
The two-pass content generator hands EVERY block on a page the same page-ranked
top-K chunk universe (see ``MCP/tools/pipeline_tools.py`` outline emit + the
``ED4ALL_CHUNK_ROLE_DIVERSIFY`` diversification switch). When every block leads
with the same source chunk, the authoring model draws its anchor worked example
from that chunk over and over — one worked problem recurred across 15/65 week-1
blocks and 19-20/53 week-10 blocks in a real 7B build. That is a monotony /
cognitive-load defect the semantic gates (NLI entailment / numeric grounding /
symbolic math) are all blind to: each recurrence is individually grounded and
individually correct — they just repeat the SAME example.

The detector
------------
For each block it derives a **numeric fingerprint** of the worked problem: the
ordered, normalized sequence of number literals in the block's visible text (a
"number-sequence signature", e.g. the recurring ``3.99 / 24 / 0.166…`` anchor).
Blocks are grouped by MODULE/WEEK (derived from the ``page_id`` module prefix,
e.g. ``week_01`` from ``week_01_content_02``). Within each module, a signature
shared by ``>= min_repeat`` blocks (default 3) is flagged as a recurring anchor
example.

High-precision by construction: a signature must carry ``>= _MIN_SIGNATURE_LEN``
number literals with ``>= _MIN_DISTINCT`` distinct values before it can match, so
trivial shared numbers ("Week 1", a lone ``2``) never collide. Signatures are
normalized (``3.990`` == ``3.99``, thousands separators stripped) so display
variance does not hide a reuse. Deterministic, no LLM, no embeddings.

Domain-agnostic (wide-net discipline): the detector keys only on number
sequences + the pipeline's own ``<word><number>`` module-id shape — no course
slug, no publisher vocabulary, no subject-specific token.

Wiring
------
Warning-severity day-1 on the ``inter_tier_validation`` phase of both
``course_generation`` and ``textbook_to_course`` (mirrors ``retrieval_presence`` /
``block_sequence_order`` — runs regardless of any flag). Consumes only
``inputs['blocks']`` (the outline-tier Block surface). The ``BlockCognitiveLoad``
gate already reads block ``content`` at the outline tier, so the body text this
detector fingerprints is present pre-render.

Decision-capture
----------------
One ``near_dup_example_check`` event per MODULE that carried >=1 signature-bearing
block; rationale interpolates dynamic per-call signals (>=20 chars): module id,
blocks scanned, signature-bearing blocks, distinct signatures, the max repeat
count, the active threshold, and whether the module flagged.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Block lives at Courseforge/scripts/blocks.py — same import bridge as the
# sibling block validators (worked_example_math / numeric_literal_grounding).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover — Block always present in this repo
    from blocks import Block  # type: ignore[import-not-found]  # noqa: E402,F401
except Exception:  # noqa: BLE001
    Block = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical GateIssue codes + decision type
# --------------------------------------------------------------------------- #
_CODE_NEAR_DUP: str = "NEAR_DUP_ANCHOR_EXAMPLE"
_DECISION_TYPE: str = "near_dup_example_check"

#: Default minimum number of blocks in a module that must share a signature for
#: it to be a recurring anchor example. Overridable via ``inputs['min_repeat']``
#: (merged from the gate ``config.min_repeat``).
_DEFAULT_MIN_REPEAT: int = 3

#: A signature needs at least this many number literals to be a worked-problem
#: fingerprint (below this, a shared number is coincidental, not a reused
#: example).
_MIN_SIGNATURE_LEN: int = 3

#: ...and at least this many DISTINCT values (so "1, 1, 1" is not a signature).
_MIN_DISTINCT: int = 2

#: Cap on the issue list + the block-id sample printed per issue.
_ISSUE_LIST_CAP: int = 50
_BLOCK_ID_SAMPLE: int = 6

#: Cap on the signature length that participates in a match (a long id-heavy
#: number stream is truncated to a stable head so incidental tails don't split
#: an otherwise-identical example).
_MAX_SIGNATURE_LEN: int = 24

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
#: Integer / decimal literal (with optional thousands separators, or a bare
#: leading-dot decimal). Currency / percent symbols sit OUTSIDE the capture so
#: ``$3.99`` and ``3.99`` normalize to the same token.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
#: Module-id prefix: a word run followed by a number (``week_01`` / ``module3`` /
#: ``unit-2``). Matches the pipeline's own ``--pages`` module-prefix convention.
_MODULE_RE = re.compile(r"^([A-Za-z]+[_-]?\d+)")


def _block_field(block: Any, field: str) -> Any:
    """Read ``field`` from a Block instance OR a plain dict (JSONL row)."""
    if isinstance(block, dict):
        return block.get(field)
    return getattr(block, field, None)


def _visible_text(html: str) -> str:
    """Strip tags + collapse whitespace (drops id-bearing attributes)."""
    if not html:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()


def _norm_number(token: str) -> Optional[str]:
    """Canonicalize one numeric literal (``3.990`` -> ``3.99``; strip commas).

    Returns ``None`` for an unparseable token (never raises).
    """
    cleaned = token.replace(",", "")
    try:
        dec = Decimal(cleaned).normalize()
    except (InvalidOperation, ValueError):
        return None
    # ``format(d, 'f')`` avoids exponent notation (Decimal('24').normalize()
    # is 2.4E+1 → '24'), so equal magnitudes share one canonical string.
    return format(dec, "f")


def number_signature(text: str) -> Optional[Tuple[str, ...]]:
    """Ordered, normalized number-sequence signature of a block's visible text.

    Returns ``None`` when the block has too few numbers / too few distinct
    values to be a worked-problem fingerprint (see ``_MIN_SIGNATURE_LEN`` /
    ``_MIN_DISTINCT``). Deterministic; the ordered sequence (capped to
    ``_MAX_SIGNATURE_LEN``) is the high-precision match key.
    """
    visible = _visible_text(text)
    if not visible:
        return None
    nums: List[str] = []
    for raw in _NUM_RE.findall(visible):
        norm = _norm_number(raw)
        if norm is not None:
            nums.append(norm)
    if len(nums) < _MIN_SIGNATURE_LEN:
        return None
    if len(set(nums)) < _MIN_DISTINCT:
        return None
    return tuple(nums[:_MAX_SIGNATURE_LEN])


def module_key(page_id: str) -> str:
    """Derive the MODULE/WEEK key from a ``page_id`` (``week_01_content_02`` ->
    ``week_01``). Falls back to the whole ``page_id`` when no ``<word><number>``
    module prefix is present."""
    if not page_id:
        return "<no-page>"
    m = _MODULE_RE.match(page_id)
    return m.group(1) if m else page_id


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Resolve ``inputs['blocks']`` or hydrate from a blocks JSONL path.

    The gate-input router surfaces ``blocks`` (hydrated Block list); tests /
    standalone callers may pass ``blocks_outline_path`` / ``blocks_final_path``
    (a JSONL of block dicts) instead.
    """
    raw = inputs.get("blocks")
    if raw is not None:
        if not isinstance(raw, list):
            return [], GateIssue(
                severity="critical",
                code="INVALID_BLOCKS_INPUT",
                message=(
                    f"inputs['blocks'] must be a list; got {type(raw).__name__}."
                ),
            )
        return list(raw), None

    path_raw = (
        inputs.get("blocks_outline_path")
        or inputs.get("blocks_final_path")
        or inputs.get("blocks_path")
    )
    if not path_raw:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs requires 'blocks' (List[Block]) or a blocks JSONL path "
                "('blocks_outline_path' / 'blocks_final_path')."
            ),
        )
    try:
        path = Path(path_raw)
    except (TypeError, ValueError):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=f"blocks path is not a path: {path_raw!r}.",
        )
    if not path.exists():
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=f"blocks path does not exist: {path}.",
        )
    blocks: List[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return blocks, None


def _resolve_min_repeat(inputs: Dict[str, Any]) -> int:
    """Resolve the threshold: ``inputs['min_repeat']`` → default 3.

    Garbage / non-positive → default (a misconfigured threshold must never
    disable the gate or collapse to a 1-block "duplicate").
    """
    raw = inputs.get("min_repeat")
    if raw is None:
        return _DEFAULT_MIN_REPEAT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_REPEAT
    return val if val >= 2 else _DEFAULT_MIN_REPEAT


def _emit_decision(
    capture: Any,
    *,
    module: str,
    blocks_scanned: int,
    signature_blocks: int,
    distinct_signatures: int,
    max_repeat: int,
    min_repeat: int,
    flagged: bool,
) -> None:
    """Emit one ``near_dup_example_check`` decision per scored MODULE.

    Rationale interpolates dynamic per-call signals (>=20 chars) so the capture
    is replayable post-hoc. Swallows ``log_decision`` exceptions (capture must
    not break the gate).
    """
    if capture is None:
        return
    decision = f"flagged:{_CODE_NEAR_DUP}" if flagged else "passed"
    rationale = (
        f"Near-duplicate anchor-example scan on module {module!r}: "
        f"blocks_scanned={blocks_scanned}, signature_blocks={signature_blocks}, "
        f"distinct_signatures={distinct_signatures}, max_repeat={max_repeat}, "
        f"min_repeat={min_repeat}, flagged={flagged}. A worked example whose "
        f"normalized number-sequence signature recurs across >= min_repeat "
        f"blocks in one module is the same anchor reused (a monotony / "
        f"cognitive-load defect NLI / numeric-grounding / symbolic-math gates "
        f"cannot see — each recurrence is individually grounded and correct)."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on %s: %s", _DECISION_TYPE, exc
        )


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
class NearDupExampleValidator:
    """Within-module recurring anchor-example detector (monotony control).

    Groups blocks by module/week, fingerprints each block's worked problem as a
    normalized number-sequence signature, and flags a signature shared by
    ``>= min_repeat`` (default 3) blocks in the same module. Warning-severity
    day-1.

    Inputs:
        blocks: List[Block] | List[dict]
            Outline-tier blocks (hydrated Block instances from the router, or
            plain dict rows from a JSONL).
        blocks_outline_path / blocks_final_path: str | Path
            Alternative to ``blocks`` — a blocks JSONL.
        min_repeat: int
            Repeat threshold (default 3; merged from the gate config).
        decision_capture: Optional[DecisionCapture]
            When wired, one decision event per scored module.
    """

    name = "near_dup_example"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        min_repeat = _resolve_min_repeat(inputs)

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

        # Group signatures per module. ``module_sigs[module][signature]`` is the
        # ordered list of block ids carrying that signature.
        module_sigs: Dict[str, Dict[Tuple[str, ...], List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        module_scanned: Dict[str, int] = defaultdict(int)
        module_sig_blocks: Dict[str, int] = defaultdict(int)

        for block in blocks:
            content = _block_field(block, "content")
            if not isinstance(content, str) or not content.strip():
                continue
            page_id = _block_field(block, "page_id") or ""
            block_id = _block_field(block, "block_id") or "<unknown>"
            mkey = module_key(str(page_id))
            module_scanned[mkey] += 1
            sig = number_signature(content)
            if sig is None:
                continue
            module_sig_blocks[mkey] += 1
            module_sigs[mkey][sig].append(str(block_id))

        issues: List[GateIssue] = []
        flagged_modules = 0
        max_repeat_overall = 0

        # Deterministic module order for stable issue output.
        for mkey in sorted(module_sigs.keys()):
            sig_map = module_sigs[mkey]
            distinct = len(sig_map)
            module_max = max((len(ids) for ids in sig_map.values()), default=0)
            max_repeat_overall = max(max_repeat_overall, module_max)
            module_flagged = False
            # Deterministic signature order (by descending repeat, then signature).
            for sig, ids in sorted(
                sig_map.items(), key=lambda kv: (-len(kv[1]), kv[0])
            ):
                if len(ids) < min_repeat:
                    continue
                module_flagged = True
                if len(issues) >= _ISSUE_LIST_CAP:
                    continue
                sig_str = ", ".join(sig[:8]) + ("…" if len(sig) > 8 else "")
                sample = ", ".join(ids[:_BLOCK_ID_SAMPLE])
                more = "" if len(ids) <= _BLOCK_ID_SAMPLE else f" (+{len(ids) - _BLOCK_ID_SAMPLE} more)"
                issues.append(
                    GateIssue(
                        severity="warning",
                        code=_CODE_NEAR_DUP,
                        message=(
                            f"Module {mkey!r}: the same worked-example number "
                            f"signature [{sig_str}] recurs across {len(ids)} "
                            f"blocks (>= threshold {min_repeat}) — one anchor "
                            f"example reused. Blocks: {sample}{more}."
                        ),
                        location=mkey,
                        suggestion=(
                            "Diversify the anchor example across the module's "
                            "blocks. Root cause is co-located blocks fed the "
                            "identical top-K chunk head — enable "
                            "ED4ALL_CHUNK_ROLE_DIVERSIFY (chunk-universe remap) "
                            "or re-roll the offending blocks with "
                            "courseforge-rewrite --block-ids against varied "
                            "grounding, rather than re-rolling the same chunk "
                            "window."
                        ),
                    )
                )
            if module_flagged:
                flagged_modules += 1
            _emit_decision(
                capture,
                module=mkey,
                blocks_scanned=module_scanned.get(mkey, 0),
                signature_blocks=module_sig_blocks.get(mkey, 0),
                distinct_signatures=distinct,
                max_repeat=module_max,
                min_repeat=min_repeat,
                flagged=module_flagged,
            )

        # Warning-severity day-1: ``passed`` honors only critical issues (there
        # are none), so the gate never blocks — it COMPUTES + CAPTURES.
        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        modules_scanned = len(module_scanned)
        score = (
            1.0
            if modules_scanned == 0
            else round((modules_scanned - flagged_modules) / modules_scanned, 4)
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
            metadata={
                "modules_scanned": modules_scanned,
                "signature_module_count": len(module_sigs),
                "flagged_modules": flagged_modules,
                "max_repeat": max_repeat_overall,
                "min_repeat": min_repeat,
            },
        )


__all__ = [
    "NearDupExampleValidator",
    "number_signature",
    "module_key",
]
