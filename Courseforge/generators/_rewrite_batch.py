"""Multi-block BATCHED rewrite-tier envelope + packer + flag resolvers.

The Courseforge two-pass REWRITE tier authors ONE HTTP POST per block (per
candidate). Against a hosted endpoint (NVIDIA) that ~40-req/min rate cap is
exhausted by hundreds of per-block POSTs that then HTTP-429 on ~every call.
This module is the Courseforge-native batching primitive that packs many
blocks into ONE POST — mirroring the SHAPE of the SemantiK Stage-6 batch
(``SemantiK/semantik_structure/qwen_specialists/runner.py::_pack_batches`` +
``endpoint_runtime.py::parse_batch_envelope``) without sharing the SemantiK
code: the envelope keys differ (``CF_BLOCK`` not ``SEMANTIK_REGION``), the
per-block budget is the rewrite max_tokens (not a region ``max_new_tokens``),
and per-block-type validation differs.

Placed in its own module so BOTH the provider (``_rewrite_provider.py``) and
the router (``router.py``) import the envelope + packer without creating a
router→generator import cycle (the router already imports the provider lazily;
this module imports neither).

Envelope contract
-----------------

Each block travels in a delimited block carrying its OWN block_id::

    <<<CF_BLOCK id="page1#concept_intro_0">>>
    ...single rendered HTML5 fragment for that block...
    <<<CF_BLOCK_END id="page1#concept_intro_0">>>

The END id is backreference-pinned to the OPEN id (DOTALL), so a missing or
mismatched END tag yields no match for that block → ``None`` fail-soft
sentinel (identical contract to the per-block ``generate_rewrite`` path: a
None slot is re-batched, never silently dropped).

Flag resolvers (all parse-with-fallback)
-----------------------------------------

- ``COURSEFORGE_REWRITE_BATCH`` (default ON for the cloud lane) — gate.
- ``COURSEFORGE_REWRITE_BATCH_SIZE`` (default 3) — max blocks per POST.
- ``COURSEFORGE_REWRITE_BATCH_K`` (default → clamp to 1) — top-1 on cloud.
- ``COURSEFORGE_REWRITE_CONCURRENCY`` honoured as the explicit override;
  otherwise the batched lane uses a LOW default of 2.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover — typing only
    from Courseforge.scripts.blocks import Block


# ---------------------------------------------------------------------------
# Envelope constants + parser
# ---------------------------------------------------------------------------

# Open / close delimiters. ``{block_id}`` is substituted verbatim (the
# block_id may contain ``#`` / ``_`` / alphanumerics; it is matched back via a
# tolerant capture group below, never via this literal interpolation, so an
# arbitrary id round-trips).
CF_BLOCK_OPEN = '<<<CF_BLOCK id="{block_id}">>>'
CF_BLOCK_CLOSE = '<<<CF_BLOCK_END id="{block_id}">>>'

# Backreference-pinned DOTALL region regex (mirrors SemantiK's
# ``_BATCH_REGION_RE``). Group 1 = the block_id; group 2 = the fragment body.
# The block_id capture is permissive ([^"]+) so a block_id carrying ``#`` /
# ``_`` / spaces round-trips; the END id is ``\1`` (the SAME id), so a
# mismatched / missing END tag produces no match for that block.
_CF_BLOCK_RE = re.compile(
    r'<<<CF_BLOCK id="([^"]+)">>>(.*?)<<<CF_BLOCK_END id="\1">>>',
    re.DOTALL,
)

# Output-token ceiling for one batched POST. Matches the SemantiK runtime
# ceiling (16384) minus a safety margin so the summed per-block budget never
# asks one POST for more output than the model can emit.
_OUTPUT_TOKEN_CEILING = 16384
_OUTPUT_TOKEN_MARGIN = 512
DEFAULT_OUTPUT_TOKEN_CAP = _OUTPUT_TOKEN_CEILING - _OUTPUT_TOKEN_MARGIN

# Per-block fallback output budget when a block's resolved rewrite max_tokens
# cannot be determined (mirrors the rewrite spec default of 4096).
_DEFAULT_BLOCK_MAX_TOKENS = 4096

# Falsey strings that disable the (default-ON) cloud batched lane.
_BATCH_FALSEY = frozenset({"0", "false", "no", "off"})

# Default max blocks packed into ONE batched POST.
_DEFAULT_BATCH_SIZE = 3

# Default candidate-K cap on the batched cloud lane (top-1 — best-of-N
# multiplies POSTs by N, re-creating the rate-limit pressure batching removes).
_DEFAULT_BATCH_K = 1

# Default concurrency for the batched cloud lane (each POST already packs
# several blocks; a high fan-out would re-create the rate-limit pressure).
_DEFAULT_BATCHED_CONCURRENCY = 2


def _strip_code_fences(text: str) -> str:
    """Strip a single Markdown code fence (```lang … ``` / bare ```).

    A model that wraps the whole batched response in a fence must not defeat
    the delimiter parse. Returns the inner body when fenced; otherwise the
    input unchanged. Mirrors SemantiK's ``endpoint_runtime._strip_code_fences``
    (kept local to avoid a cross-subsystem import).
    """
    if not text:
        return ""
    s = text.strip()
    if "```" not in s:
        return s
    first = s.find("```")
    rest = s[first + 3:]
    nl = rest.find("\n")
    if nl != -1:
        head = rest[:nl].strip().lower()
        if head and head.isalpha():
            rest = rest[nl + 1:]
    close = rest.find("```")
    if close != -1:
        return rest[:close].strip()
    return rest.strip()


def parse_rewrite_batch_envelope(
    response: str, block_ids: Sequence[str]
) -> Dict[str, Optional[str]]:
    """Parse a batched response into ``{block_id: html | None}``.

    Tolerant DOTALL extraction of every
    ``<<<CF_BLOCK id="…">>> … <<<CF_BLOCK_END id="…">>>`` pair (the END id is
    backreference-pinned to the OPEN id, so a missing/mismatched END tag yields
    no match for that block). Code fences are stripped first. EVERY requested
    ``block_id`` is present in the result; any block whose pair is absent maps
    to ``None`` (the fail-soft sentinel, same contract as the per-block
    ``generate_rewrite`` path).
    """
    body = _strip_code_fences(response or "")
    found: Dict[str, str] = {}
    for match in _CF_BLOCK_RE.finditer(body):
        bid = match.group(1)
        # First occurrence wins (defensive against a duplicated block).
        found.setdefault(bid, match.group(2).strip())
    return {bid: found.get(bid) for bid in block_ids}


# ---------------------------------------------------------------------------
# Per-block output-budget resolution + packer
# ---------------------------------------------------------------------------


def _block_output_budget(block: "Block") -> int:
    """Resolve a block's per-POST output-token budget.

    Reads the block's own resolved rewrite max_tokens when the caller stamped
    one (the packer is fed blocks that already know their ``(provider, model,
    max_tokens)`` group). Falls back to the rewrite default 4096. A non-int /
    non-positive value falls back, mirroring the parse-with-fallback posture.
    """
    raw = getattr(block, "_rewrite_max_tokens", None)
    if raw is None:
        return _DEFAULT_BLOCK_MAX_TOKENS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BLOCK_MAX_TOKENS
    return val if val > 0 else _DEFAULT_BLOCK_MAX_TOKENS


def pack_rewrite_batches(
    blocks: Sequence["Block"],
    *,
    batch_size: int,
    output_token_cap: int = DEFAULT_OUTPUT_TOKEN_CAP,
) -> List[List["Block"]]:
    """Pack ``blocks`` (IN ORDER) into batches for the batched cloud lane.

    A new batch is started when EITHER the current batch reaches
    ``batch_size`` blocks OR adding the next block would push the batch's
    summed per-block output budget over ``output_token_cap`` — so a run of
    big-output blocks automatically gets smaller batches and never asks one
    POST for more output than the model can emit. A single block whose own
    budget already exceeds the cap still gets its own (one-block) batch rather
    than being dropped. Order within and across batches is preserved so the
    block_id → result mapping stays exact.

    Mirrors ``SemantiK ... runner._pack_batches``.
    """
    batches: List[List["Block"]] = []
    current: List["Block"] = []
    current_tokens = 0
    cap = max(1, int(batch_size))
    tok_cap = max(1, int(output_token_cap))
    for block in blocks:
        block_tokens = _block_output_budget(block)
        would_overflow_tokens = bool(current) and (
            current_tokens + block_tokens > tok_cap
        )
        would_overflow_count = len(current) >= cap
        if would_overflow_count or would_overflow_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------------
# Flag resolvers (parse-with-fallback)
# ---------------------------------------------------------------------------


def resolve_rewrite_batch_mode() -> bool:
    """True when the cloud rewrite lane should use the BATCHED path.

    Gate ``COURSEFORGE_REWRITE_BATCH``. **Default ON for the cloud lane**
    (the per-block path 429s against the hosted seat's ~40-req/min cap),
    mirroring SemantiK's ``SEMANTIK_SPECIALIST_BATCH``. Default-ON parse
    semantics: an explicit falsey value (``0``/``false``/``no``/``off``)
    selects the EXACT legacy per-block path (byte-stable); unset / truthy /
    garbage → on. Only consulted when the resolved rewrite lane is cloud
    (a loopback base_url always uses the per-block path).
    """
    raw = (os.environ.get("COURSEFORGE_REWRITE_BATCH") or "").strip().lower()
    return raw not in _BATCH_FALSEY


def resolve_rewrite_batch_size() -> int:
    """Resolve ``COURSEFORGE_REWRITE_BATCH_SIZE`` (default 3).

    The maximum number of blocks packed into ONE batched POST. A new batch is
    also started early when the running batch's summed per-block output budget
    would exceed the output-token cap (see :func:`pack_rewrite_batches`).
    Garbage / non-positive values fall back to the default (parse-with-
    fallback, mirroring SemantiK's ``resolve_specialist_batch_regions``).
    """
    raw = os.environ.get("COURSEFORGE_REWRITE_BATCH_SIZE")
    if not raw:
        return _DEFAULT_BATCH_SIZE
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BATCH_SIZE
    return val if val > 0 else _DEFAULT_BATCH_SIZE


def resolve_rewrite_batch_k(requested_k: int) -> int:
    """Resolve the candidate-K used on the BATCHED cloud lane (top-1 default).

    ``COURSEFORGE_REWRITE_BATCH_K`` (positive int) pins the cap explicitly;
    otherwise the caller's ``requested_k`` is clamped down to
    ``_DEFAULT_BATCH_K`` (1) so the batched cloud lane does not multiply POSTs
    by a best-of-N K. Never raises K above what the caller asked for (a caller
    passing k=1 keeps k=1). Parse-with-fallback (garbage / non-positive → the
    default-1 cap). Mirrors SemantiK's ``resolve_batched_endpoint_k`` but the
    cloud default is top-1 (the plan's explicit rate-limit-first choice).
    """
    raw = os.environ.get("COURSEFORGE_REWRITE_BATCH_K")
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = 0
        if val > 0:
            return min(requested_k, val)
    return min(requested_k, _DEFAULT_BATCH_K)


def resolve_rewrite_batched_concurrency() -> int:
    """Thread-pool size for dispatching BATCHES on the cloud lane.

    An explicitly-set ``COURSEFORGE_REWRITE_CONCURRENCY`` is honoured as an
    override; otherwise the batched lane uses a LOW default
    (``_DEFAULT_BATCHED_CONCURRENCY`` = 2) because each batch already packs
    several blocks — a high fan-out would re-create the rate-limit pressure
    batching exists to remove. Parse-with-fallback (garbage / non-positive on
    the explicit override → the low default). Mirrors SemantiK's
    ``resolve_batched_concurrency``.
    """
    raw = os.environ.get("COURSEFORGE_REWRITE_CONCURRENCY")
    if raw:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_BATCHED_CONCURRENCY
        if val >= 1:
            return val
    return _DEFAULT_BATCHED_CONCURRENCY


__all__ = [
    "CF_BLOCK_OPEN",
    "CF_BLOCK_CLOSE",
    "DEFAULT_OUTPUT_TOKEN_CAP",
    "pack_rewrite_batches",
    "parse_rewrite_batch_envelope",
    "resolve_rewrite_batch_k",
    "resolve_rewrite_batch_mode",
    "resolve_rewrite_batch_size",
    "resolve_rewrite_batched_concurrency",
]
