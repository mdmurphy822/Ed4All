"""Bounded OCR-confusable micro-repair pass (SEMANTIK_OCR_CONFUSABLE_REPAIR).

Plan ``plans/scan-conversion-improvements-2026-07.md`` Phase 5 channel 2 (+
the channel-3 exit signal it feeds). A gated, default-OFF pass that fixes the
handful of characters OCR mangles around inline math and decorated glyphs
(``√x`` scanned as ``Vx``, ``l``↔``1``, ``O``↔``0``, stray ``™`` / curly
quotes) — WITHOUT ever letting an LLM rewrite words.

Contract (the load-bearing invariant): the 7B is only a CANDIDATE GENERATOR.
It proposes ``{region_index, before, after}`` micro-edits on FLAGGED windows;
a DETERMINISTIC, auditable accept gate here owns the decision. An edit lands
only if ALL hold:

  (i)   ``before → after`` is reachable via :data:`CONFUSABLE_MAP` substitutions
        only (the map is the SINGLE substitution vocabulary — a digit↔letter
        change is therefore possible ONLY through an explicit map entry, so
        character-class conservation is enforced by construction);
  (ii)  Levenshtein(before, after) ≤ 2;
  (iii) the edit is WITHIN one token (no whitespace in before/after; token
        count unchanged);
  (iv)  the edited token is NOT inside a numeric / answer context (digit-dense);
  (v)   non-identity;
  (vi)  the region was actually flagged + in-window (enforced by the caller).

Per-block accepted-edit cap :data:`_MAX_EDITS_PER_BLOCK`; a per-doc runaway
tripwire reverts the WHOLE pass if edited-tokens/total-tokens exceeds
:data:`_RUNAWAY_TOKEN_FRACTION`. Every accepted edit is annotated
(``data-dart-repair`` downstream) — provenance amended honestly, never faked —
and :func:`assert_repair_conservation` is the strict replay check the
adapter/downstream seam runs before trusting a repaired block.

Pure/CPU: the detector, the gate, and the stats are deterministic and GPU-free;
the ONLY model touch is the proposer POST on the already-licensed specialist
seat (mocked in tests). Flag OFF → zero detector work, zero LLM calls, no
``ocr_repair`` key on the cascade result (byte-identical).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock

logger = logging.getLogger(__name__)

# Truthy tokens (mirror ``reviewer._TRUTHY`` / the project parse-with-fallback).
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_ocr_confusable_repair_mode() -> bool:
    """Return True when the OCR-confusable micro-repair pass is enabled.

    Reads ``SEMANTIK_OCR_CONFUSABLE_REPAIR``. Default OFF (unset / blank /
    falsey / garbage) -> byte-identical to today (mirrors
    ``reviewer.resolve_second_pass_mode``). A truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive) enables it.
    """
    raw = (os.environ.get("SEMANTIK_OCR_CONFUSABLE_REPAIR") or "").strip().lower()
    return raw in _TRUTHY


# ---------------------------------------------------------------------------
# The confusable map — the SINGLE source of truth for both the proposer's
# allowed corrections and the deterministic accept gate. The garble DETECTOR's
# patterns are derived from the ``src`` sides. ``crosses_class`` records whether
# the substitution crosses the digit↔letter boundary (metadata for logging; the
# reachability check already confines every substitution to this vocabulary).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusableRule:
    rule_id: str
    src: str
    dst: str
    crosses_class: bool  # digit↔letter crossing (audit metadata)


CONFUSABLE_MAP: tuple[ConfusableRule, ...] = (
    # Radical sign scanned as capital V (the math-dense loss the plan measured).
    ConfusableRule("v_to_radical", "V", "√", True),  # V -> √
    # l ↔ 1 (both directions — OCR confuses either way).
    ConfusableRule("l_to_1", "l", "1", True),
    ConfusableRule("1_to_l", "1", "l", True),
    # O ↔ 0.
    ConfusableRule("O_to_0", "O", "0", True),
    ConfusableRule("0_to_O", "0", "O", True),
    # I / | -> 1.
    ConfusableRule("I_to_1", "I", "1", True),
    # S ↔ 5.
    ConfusableRule("S_to_5", "S", "5", True),
    ConfusableRule("5_to_S", "5", "S", True),
    # B ↔ 8.
    ConfusableRule("B_to_8", "B", "8", True),
    ConfusableRule("8_to_B", "8", "B", True),
    # Ligature splits.
    ConfusableRule("rn_to_m", "rn", "m", False),
    ConfusableRule("vv_to_w", "vv", "w", False),
    # Degree / o confusion.
    ConfusableRule("deg_to_o", "°", "o", False),  # ° -> o
    ConfusableRule("o_to_deg", "o", "°", False),  # o -> °
    # Superscript flattening (digit read as its superscript form).
    ConfusableRule("2_to_sup2", "2", "²", False),  # 2 -> ²
    ConfusableRule("3_to_sup3", "3", "³", False),  # 3 -> ³
    # Decoration-strip rules (garbled decoration → removed).
    ConfusableRule("strip_tm", "™", "", False),  # ™
    ConfusableRule("strip_ldquo", "“", "", False),  # “
    ConfusableRule("strip_rdquo", "”", "", False),  # ”
    ConfusableRule("strip_pipe", "|", "", False),
    ConfusableRule("strip_cent", "¢", "", False),  # ¢
)

# Decoration characters whose mere presence flags a token (derived from the
# strip-rule ``src`` sides + the radical/pipe confusions).
_DECORATION_CHARS = frozenset(
    r.src for r in CONFUSABLE_MAP if r.dst == "" and len(r.src) == 1
)

# Confusable DIGITS that, embedded inside an alphabetic-dominant token, signal an
# OCR letter→digit confusion (the ``l/1`` / ``O/0`` / ``S/5`` / ``B/8`` garble).
_CONFUSABLE_DIGITS = frozenset("01358")

# Deterministic caps / thresholds (module constants — promoted to flags only on
# demonstrated need, per the plan's "keep this pass cheap" ordering note).
_MAX_EDITS_PER_BLOCK = 8
_RUNAWAY_TOKEN_FRACTION = 0.02
#: Absolute accepted-edit floor below which the (fraction-based) runaway tripwire
#: never fires — a single legitimate fix on a tiny region is not a runaway, and
#: the 2% fraction is only meaningful once a handful of edits have landed. A real
#: mass/off-target proposal clears BOTH the fraction and this floor.
_RUNAWAY_MIN_EDITS = 2
#: Ceiling on the repair-stats proxy score — deliberately < 1.0 so a residual-
#: garble-density proxy never masquerades as a real cross-encoder verdict.
_REPAIR_SCORE_CEILING = 0.95
#: Digit-density above which a line/token is a numeric / answer context.
_DIGIT_DENSE_FRACTION = 0.30
#: Generous output budget for the proposer's JSON-array response.
_OCR_REPAIR_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Lexicon-confusable merge (Wave #22 quick-wins).
#
# The SemantiK pedagogical lexicon (schemas/taxonomies/semantik_lexicon.json)
# now OWNS the label-level OCR confusables (e.g. ``tr[vy]it`` -> ``TRY IT`` the
# textbook scan corpus emits). Those are WHOLE-TOKEN label normalizations —
# a regex-matched garbled token maps to a multi-char (often multi-WORD)
# canonical form — so they do NOT fit the char-level ``CONFUSABLE_MAP`` gate
# (single-char substitutions, Levenshtein <= 2, no whitespace). We MERGE them
# additively: they extend the detector (a matching token flags its region) and
# add a SEPARATE deterministic accept arm in :func:`evaluate_edit` (the lexicon
# IS the substitution vocabulary, so the edit is still auditable + replay-
# conserved). The existing char-confusable behavior is unchanged.
#
# ``ocr_repair`` runs in the SemantiK venv, which may not carry Ed4All's
# ``lib/`` on ``sys.path`` — so we try the canonical ``lib.ontology.taxonomy``
# loader first (it honors ``SEMANTIK_LEXICON_PROFILE``) and fall back to reading
# the lexicon JSON directly by path, then to the built-in map (empty lexicon ->
# byte-identical to today).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LexiconConfusableRule:
    """One merged lexicon confusable: a whole-token pattern -> canonical form."""

    rule_id: str
    pattern: "re.Pattern[str]"  # anchored, case-insensitive, matches a token
    canonical: str


def _lexicon_confusable_entries() -> tuple[dict, ...]:
    """Return the raw ``{pattern, canonical}`` lexicon confusables (best-effort).

    Resolution: the canonical ``lib.ontology.taxonomy`` loader (profile-aware) >
    a direct read of ``schemas/taxonomies/semantik_lexicon.json`` by path (with a
    minimal ``SEMANTIK_LEXICON_PROFILE`` merge) > ``()`` (no lexicon -> the
    built-in char map only, byte-identical to today).
    """
    # 1. Canonical loader — works in-process or when lib/ is importable.
    try:
        from lib.ontology.taxonomy import get_lexicon_confusables  # type: ignore

        entries = get_lexicon_confusables()
        return tuple(e for e in entries if isinstance(e, dict))
    except Exception:  # noqa: BLE001 — lib/ not importable in the SemantiK venv
        pass
    # 2. Direct JSON read by path (SemantiK venv fallback).
    try:
        return _lexicon_confusables_from_json()
    except Exception as exc:  # noqa: BLE001 — any failure -> no lexicon confusables
        logger.debug("lexicon confusable JSON load failed (%s); built-in map only", exc)
        return ()


def _lexicon_confusables_from_json() -> tuple[dict, ...]:
    """Read lexicon confusables straight from the taxonomy JSON by path.

    Mirrors ``lib.ontology.taxonomy`` profile resolution minimally:
    ``SEMANTIK_LEXICON_PROFILE`` (default ``generic-academic+open-textbook``)
    is a ``+``-joined list of profile keys merged left-to-right, each segment
    folded through the legacy profile-key aliases. Unknown keys are skipped;
    an empty result falls back to every profile's confusables so a
    misconfigured profile never silently drops the OCR fixes.
    """
    # ocr_repair.py -> qwen_specialists -> dart_semantic -> SemantiK -> <repo>
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "schemas"
        / "taxonomies"
        / "semantik_lexicon.json"
    )
    if not schema_path.exists():
        return ()
    with open(schema_path, encoding="utf-8") as fh:
        data = json.load(fh)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return ()
    spec = (os.environ.get("SEMANTIK_LEXICON_PROFILE") or "").strip()
    if not spec:
        spec = "generic-academic+open-textbook"
    # legacy profile-key aliases (pre-rename compat) — cross-venv twin of
    # lib.ontology.taxonomy.LEGACY_PROFILE_ALIASES (lib/ is not importable
    # on this fallback path; keep the two maps in sync).
    aliases = {"openstax": "open-textbook", "openstax-shaped": "textbook-shaped"}
    keys = [aliases.get(k.strip(), k.strip()) for k in spec.split("+") if k.strip()]
    keys = [k for k in keys if k in profiles]
    if not keys:
        keys = list(profiles.keys())
    out: list[dict] = []
    for key in keys:
        for entry in profiles.get(key, {}).get("confusables", []) or []:
            if isinstance(entry, dict):
                out.append(entry)
    return tuple(out)


@lru_cache(maxsize=1)
def _lexicon_confusable_rules() -> tuple[LexiconConfusableRule, ...]:
    """Compile the merged lexicon confusables into whole-token match rules.

    Each entry's ``pattern`` (a regex BODY fragment) is anchored + made
    case-insensitive so it matches a WHOLE token; ``canonical`` is the
    replacement. Malformed entries (missing fields / bad regex) are skipped.
    Cached: the lexicon is static per process.
    """
    rules: list[LexiconConfusableRule] = []
    for i, entry in enumerate(_lexicon_confusable_entries()):
        pattern = str(entry.get("pattern") or "").strip()
        canonical = str(entry.get("canonical") or "")
        if not pattern or not canonical:
            continue
        try:
            compiled = re.compile(rf"^(?:{pattern})$", re.IGNORECASE)
        except re.error:
            continue
        rules.append(
            LexiconConfusableRule(
                rule_id=f"lexicon_confusable_{i}",
                pattern=compiled,
                canonical=canonical,
            )
        )
    return tuple(rules)


def _token_matches_lexicon_confusable(token: str) -> bool:
    """Whether a whitespace token wholly matches a merged lexicon confusable."""
    tok = (token or "").strip()
    if not tok:
        return False
    return any(rule.pattern.match(tok) for rule in _lexicon_confusable_rules())


def _match_lexicon_confusable(before: str, after: str) -> Optional[str]:
    """Return the rule_id when ``before -> after`` is a merged lexicon confusable.

    Requires ``before`` to WHOLLY match a lexicon confusable pattern (a single
    garbled token) AND ``after`` to equal that entry's canonical form exactly.
    ``None`` when no lexicon rule applies (defer to the char-level gates).
    """
    b = (before or "").strip()
    if not b:
        return None
    for rule in _lexicon_confusable_rules():
        if rule.pattern.match(b) and after == rule.canonical:
            return rule.rule_id
    return None


# ---------------------------------------------------------------------------
# Deterministic garble detector (CPU-only). Derived from the map src sides.
# ---------------------------------------------------------------------------


def _digit_density(text: str) -> float:
    """Digit chars / non-space chars (0.0 for empty)."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    digits = sum(1 for c in non_space if c.isdigit())
    return digits / len(non_space)


def _is_numeric_context(token: str) -> bool:
    """True when a token is a numeric / answer context repair must NOT touch.

    A digit-dense token, or an answer-list shape (``^\\s*\\d+[.)]``) — the plan's
    two exclusion rules applied at the token grain.
    """
    stripped = token.strip()
    if not stripped:
        return False
    if _digit_density(stripped) >= _DIGIT_DENSE_FRACTION:
        return True
    # Answer-list item shape: a leading integer followed by . or ).
    i = 0
    while i < len(stripped) and stripped[i].isdigit():
        i += 1
    return i > 0 and i < len(stripped) and stripped[i] in ".)"


def _token_is_garbled(token: str) -> bool:
    """Whether a single whitespace token trips a map-derived garble pattern.

    Fires on: a decoration char present; ``V`` glued to a following alnum
    (radical garble); a ``vv`` ligature; or an alphabetic-dominant token with an
    interior confusable digit (``1ike`` / ``hel1o`` / ``5imilar``). Numeric /
    answer-context tokens are NEVER flagged (excluded at detect time).
    """
    if not token or _is_numeric_context(token):
        return False
    if any(c in _DECORATION_CHARS for c in token):
        return True
    # Radical garble: capital V immediately followed by an alphanumeric.
    for i, c in enumerate(token[:-1]):
        if c == "V" and (token[i + 1].isalnum()):
            return True
    if "vv" in token:
        return True
    # Interior confusable digit inside an alphabetic-dominant token.
    letters = sum(1 for c in token if c.isalpha())
    conf_digits = sum(1 for c in token if c in _CONFUSABLE_DIGITS)
    if letters >= 2 and conf_digits >= 1:
        return True
    # Wave #22: a whole-token label confusable from the merged lexicon
    # (e.g. ``trvit`` -> ``TRY IT``). Additive garble signal; empty lexicon
    # (no schema / lib not importable) -> no-op, byte-identical to today.
    if _token_matches_lexicon_confusable(token):
        return True
    return False


def region_text_is_garbled(text: str) -> bool:
    """Whether a region's raw text should be FLAGGED for repair proposal.

    A region is flagged when ≥1 of its whitespace tokens trips
    :func:`_token_is_garbled`. Digit-dense lines are excluded at the token grain
    inside the detector, so an answer key / numbered exercise list never flags.
    """
    if not text:
        return False
    return any(_token_is_garbled(tok) for tok in text.split())


def _count_garble_tokens(text: str) -> int:
    """Count whitespace tokens tripping :func:`_token_is_garbled`."""
    return sum(1 for tok in (text or "").split() if _token_is_garbled(tok))


def _count_content_tokens(text: str) -> int:
    """Count whitespace tokens (the density denominator)."""
    return len((text or "").split())


# ---------------------------------------------------------------------------
# Deterministic accept gate.
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance (small strings; no external dep)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _map_reachable(before: str, after: str, *, max_steps: int = 2) -> str | None:
    """Return the FIRST rule_id on a substitution path ``before → after``.

    BFS applying single-occurrence CONFUSABLE_MAP substitutions (≤ ``max_steps``
    steps, since each rule changes ≥1 char and the caller also bounds Levenshtein
    ≤ 2). Returns the rule_id of the first substitution used on a successful
    path, or ``None`` when ``after`` is unreachable via the map. The map is the
    ONLY substitution vocabulary, so a reachable diff is character-class safe by
    construction.
    """
    if before == after:
        return None
    # State = (current_string, first_rule_id_used).
    frontier: list[tuple[str, str | None]] = [(before, None)]
    seen: set[str] = {before}
    for _step in range(max_steps):
        nxt: list[tuple[str, str | None]] = []
        for cur, first_rule in frontier:
            for rule in CONFUSABLE_MAP:
                if not rule.src or rule.src not in cur:
                    continue
                # Replace EACH occurrence position independently (short strings).
                start = 0
                while True:
                    pos = cur.find(rule.src, start)
                    if pos == -1:
                        break
                    candidate = (
                        cur[:pos] + rule.dst + cur[pos + len(rule.src) :]
                    )
                    fr = first_rule or rule.rule_id
                    if candidate == after:
                        return fr
                    if candidate not in seen:
                        seen.add(candidate)
                        nxt.append((candidate, fr))
                    start = pos + 1
        frontier = nxt
        if not frontier:
            break
    return None


@dataclass(frozen=True)
class EditDecision:
    accepted: bool
    reason: str  # rule_id on accept; a reason code on reject
    token_index: int | None = None


def _containing_token(text: str, before: str) -> tuple[int | None, str]:
    """Locate the whitespace token index (and the token) that contains ``before``.

    Returns ``(token_index, token)`` for the FIRST token containing ``before``,
    else ``(None, "")``.
    """
    pos = text.find(before)
    if pos == -1:
        return None, ""
    # Offset-aware scan (split() collapses runs; re-tokenize on whitespace).
    idx = 0
    tok_i = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        start = idx
        while idx < n and not text[idx].isspace():
            idx += 1
        token = text[start:idx]
        if start <= pos < idx:
            return tok_i, token
        tok_i += 1
    return None, ""


def evaluate_edit(region_text: str, before: str, after: str) -> EditDecision:
    """Apply the deterministic accept gate to ONE proposed edit.

    Returns an :class:`EditDecision` (``accepted`` + a rule_id on accept, or a
    reason code on reject). The gate is the auditable owner of the decision —
    the LLM only proposes.
    """
    # (v) non-identity.
    if before == after:
        return EditDecision(False, "identity")
    # Wave #22 — lexicon-confusable arm. A whole-token label normalization
    # (``trvit`` -> ``TRY IT``) from the merged SemantiK lexicon is a
    # deterministic, auditable substitution (the lexicon IS the vocabulary), so
    # it BYPASSES the char-level within-token / Levenshtein / map-reachability
    # gates below — those are for single-char confusables and would reject a
    # multi-word canonical. It still requires the ``before`` token to occur in
    # the region and to NOT be a numeric / answer context, and the accepted edit
    # is replay-conservation-checked downstream like every other edit.
    _lex_rule = _match_lexicon_confusable(before, after)
    if _lex_rule is not None:
        lex_token_index, lex_token = _containing_token(region_text, before)
        if lex_token_index is None:
            return EditDecision(False, "before_not_found")
        if _is_numeric_context(lex_token):
            return EditDecision(False, "numeric_context")
        return EditDecision(True, _lex_rule, token_index=lex_token_index)
    # (iii) within-token: no whitespace in either side.
    if any(c.isspace() for c in before) or any(c.isspace() for c in after):
        return EditDecision(False, "not_within_token")
    # (i) reachable via the confusable map only (also gives char-class safety).
    rule_id = _map_reachable(before, after)
    if rule_id is None:
        return EditDecision(False, "not_in_map")
    # (ii) Levenshtein ≤ 2.
    if _levenshtein(before, after) > 2:
        return EditDecision(False, "edit_distance_gt_2")
    # (vi-adjacent) the substring must actually occur in the region text.
    token_index, token = _containing_token(region_text, before)
    if token_index is None:
        return EditDecision(False, "before_not_found")
    # (iv) not inside a numeric / answer context.
    if _is_numeric_context(token):
        return EditDecision(False, "numeric_context")
    return EditDecision(True, rule_id, token_index=token_index)


# ---------------------------------------------------------------------------
# Replay-conservation check (the ONLY repair-aware downstream exemption).
# ---------------------------------------------------------------------------


class RepairConservationError(RuntimeError):
    """Raised when a repaired text does not replay EXACTLY the accepted edits."""


def apply_accepted_edits(
    original: str, accepted_edits: list[dict[str, Any]]
) -> str:
    """Deterministically apply accepted edits to ``original`` in recorded order.

    Each edit replaces the FIRST occurrence of ``before`` with ``after`` (count
    1). Sequential single-occurrence replacement is deterministic and replayable
    — :func:`assert_repair_conservation` re-runs exactly this to confirm a
    repaired string carries only the recorded edits.
    """
    text = original
    for edit in accepted_edits:
        before = str(edit.get("before", ""))
        after = str(edit.get("after", ""))
        if not before:
            continue
        text = text.replace(before, after, 1)
    return text


def assert_repair_conservation(
    original: str, repaired: str, accepted_edits: list[dict[str, Any]]
) -> None:
    """Fail-closed replay check: ``repaired`` must be EXACTLY ``original`` with
    the recorded ``accepted_edits`` applied in order.

    Raises :class:`RepairConservationError` on ANY other drift — the downstream
    seam catches it and reverts the block to verbatim ``original`` (the block's
    ``data-dart-repair`` exemption is honored ONLY when this check passes).
    """
    replayed = apply_accepted_edits(original, accepted_edits)
    if replayed != repaired:
        raise RepairConservationError(
            "repaired text does not replay the recorded accepted edits "
            f"({len(accepted_edits)} edit(s))"
        )


# ---------------------------------------------------------------------------
# Proposer dispatch (the ONLY model touch; mocked in tests).
# ---------------------------------------------------------------------------


def _dispatch_repair_once(runtime: Any, prompt: str, temperature: float) -> str | None:
    """ONE single-prompt ``generate_batch`` -> raw completion | None.

    Serialized (single-prompt batch never fans out). FAIL-SOFT: ANY endpoint
    error / None sentinel / non-str -> ``None`` (the caller contributes no edits
    for the window). Mirrors ``reviewer._dispatch_verify_once``.
    """
    try:
        try:
            completions = runtime.generate_batch(
                [prompt],
                max_tokens=_OCR_REPAIR_MAX_TOKENS,
                temperature=temperature,
                fail_soft=True,
            )
        except TypeError:
            completions = runtime.generate_batch(
                [prompt],
                max_tokens=_OCR_REPAIR_MAX_TOKENS,
                temperature=temperature,
            )
    except Exception as exc:  # noqa: BLE001 — fail-soft: ANY error -> no edits.
        logger.warning("ocr repair dispatch failed (%s) -> no edits", exc)
        return None
    completions = list(completions)
    raw = completions[0] if completions else None
    if not isinstance(raw, str):
        return None
    return raw


# ---------------------------------------------------------------------------
# The pass entry point.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairResult:
    """Return shape of :func:`run_ocr_confusable_repair`.

    ``edits_by_region`` maps a region_index to
    ``{"repaired_text": str, "ocr_repair": {"type", "n_edits", "edits": [...]}}``
    for every region that GAINED ≥1 accepted edit (empty when the pass made no
    change). ``stats`` is the channel-3 exit signal + audit payload.
    """

    edits_by_region: dict[int, dict[str, Any]]
    stats: dict[str, Any]


def _joined_region_text(region: Region, feature_blocks: list[FeatureBlock]) -> str:
    """Region raw text = its owned FeatureBlocks' raw text, index-joined.

    Mirrors ``reviewer._joined_source_text`` (the SOLE verbatim-source accessor)
    via a lazy import so this module stays cycle-free.

    Defect-6 investigation note (2026-07-03): on a VLM-fused run this returns
    the FB text the P1 fusion rewrote onto the tesseract bboxes — i.e. the CLEAN
    VLM markdown / LaTeX (``$\\sqrt{ab}$``), NOT the OCR garble (``Vab``). The
    gate-view is therefore wired correctly for fused text: it reads exactly the
    text that ships. The confusable detector (:func:`region_text_is_garbled` +
    the frozen ``CONFUSABLE_MAP`` of √→V / l↔1 / O↔0) finds nothing to repair
    because the VLM already resolved those confusions at the source, so the pass
    legitimately emits ZERO ``data-dart-repair`` annotations (``cascade.py``
    stamps only regions with >=1 gated edit). That is the expected outcome —
    "VLM text has no repairable confusables left" — not a mis-wired gate.
    """
    from .reviewer import _joined_source_text

    return _joined_source_text(region, feature_blocks)


def run_ocr_confusable_repair(
    regions: list[Region],
    feature_blocks: list[FeatureBlock],
    runtime: Any,
    *,
    temperature: float = 0.0,
    log: Any = None,
) -> RepairResult:
    """Run the bounded OCR-confusable micro-repair pass over ``regions``.

    1. Detect FLAGGED regions (``region_text_is_garbled``); digit-dense answer
       lines never flag.
    2. Window the flagged texts (reusing the second-pass section packer under
       ``resolve_second_pass_window_tokens``) and POST ONE proposer call per
       window on the injected ``runtime`` (fail-soft per window).
    3. Run the DETERMINISTIC accept gate on every proposal; cap accepted edits
       per block; apply them in-memory to produce ``repaired_text``.
    4. Runaway tripwire: revert the WHOLE pass if edited/total tokens exceed
       :data:`_RUNAWAY_TOKEN_FRACTION`.
    5. Re-measure residual garble density on the post-repair text → the
       ``repair_stats_score`` channel-3 signal.

    Returns a :class:`RepairResult`. NEVER raises (a broken proposer / window
    degrades to fewer edits). ``runtime`` is the already-licensed specialist
    seat (``make_runtime('endpoint', …)``); no model/GPU is loaded here.
    """
    _log = log or (lambda _m: None)

    # (1) Detect flagged regions + baseline token census across ALL regions.
    flagged_entries: list[dict[str, Any]] = []
    region_texts: dict[int, str] = {}
    total_content_tokens = 0
    for idx, region in enumerate(regions):
        text = _joined_region_text(region, feature_blocks)
        region_texts[idx] = text
        total_content_tokens += _count_content_tokens(text)
        if region_text_is_garbled(text):
            kind = getattr(region, "kind", "paragraph")
            flagged_entries.append(
                {"region_index": idx, "kind": kind, "text": text}
            )

    rejected_rows: list[dict[str, Any]] = []
    proposals = 0
    windows_dispatched = 0
    windows_failed = 0

    def _empty_stats() -> dict[str, Any]:
        residual = sum(
            _count_garble_tokens(t) for t in region_texts.values()
        )
        density = (
            residual / total_content_tokens if total_content_tokens else 0.0
        )
        score = max(0.0, min(_REPAIR_SCORE_CEILING, 1.0 - density))
        return {
            "flagged_blocks": len(flagged_entries),
            "windows_dispatched": windows_dispatched,
            "windows_failed": windows_failed,
            "proposals": proposals,
            "accepted": 0,
            "rejected": rejected_rows,
            "total_content_tokens": total_content_tokens,
            "residual_garble_tokens": residual,
            "unrepairable_defect_density": round(density, 6),
            "repair_stats_score": round(score, 6),
        }

    if not flagged_entries:
        return RepairResult({}, _empty_stats())

    # (2) Window the flagged texts (reuse the second-pass section packer).
    from ..assembler.verify_digest import window_verifier_digest
    from .ocr_repair_prompt import build_ocr_repair_request, parse_ocr_repair_response
    from .reviewer import resolve_second_pass_window_tokens

    digest = {"regions": flagged_entries, "heading_outline": []}
    windows = window_verifier_digest(
        digest, window_tokens=resolve_second_pass_window_tokens()
    )

    flagged_set = {e["region_index"] for e in flagged_entries}
    accepted_by_region: dict[int, list[dict[str, Any]]] = {}
    # RUNNING text per region: each accepted edit is validated against — and
    # then applied to — the region text as it stands AFTER the previously-
    # accepted edits (NOT the static original). This closes the k-th-duplicate
    # drift: two identical {before, after} proposals no longer both validate
    # against the FIRST occurrence in the original while the second silently
    # lands on the k-th (un-validated, possibly numeric-context) occurrence at
    # apply time. The recorded ``token_index`` is the ACTUAL landing token.
    running_by_region: dict[int, str] = {}

    for window in windows:
        prompt = build_ocr_repair_request(window)
        if not prompt:
            # Flag off (defensive — the caller gates on enabled) -> stop.
            continue
        windows_dispatched += 1
        allowed = {
            i
            for i in (window.get("allowed_region_indices") or [])
            if isinstance(i, int)
        }
        raw = _dispatch_repair_once(runtime, prompt, temperature)
        if raw is None:
            windows_failed += 1
            continue
        edits = parse_ocr_repair_response(raw)
        for edit in edits:
            proposals += 1
            ri = edit["region_index"]
            before = edit["before"]
            after = edit["after"]
            # (vi) region flagged + in this window.
            if ri not in flagged_set or (allowed and ri not in allowed):
                rejected_rows.append(
                    {"region_index": ri, "before": before, "after": after,
                     "reason": "region_not_in_window"}
                )
                continue
            already = accepted_by_region.get(ri, [])
            if len(already) >= _MAX_EDITS_PER_BLOCK:
                rejected_rows.append(
                    {"region_index": ri, "before": before, "after": after,
                     "reason": "per_block_cap"}
                )
                continue
            # Validate against the RUNNING text (post previously-accepted edits),
            # so the gate sees the exact occurrence this edit will land on.
            running = running_by_region.get(ri, region_texts[ri])
            decision = evaluate_edit(running, before, after)
            if not decision.accepted:
                rejected_rows.append(
                    {"region_index": ri, "before": before, "after": after,
                     "reason": decision.reason}
                )
                continue
            # Apply this single edit to the running text (first occurrence) so
            # the NEXT proposal for this region validates against the updated
            # text — the recorded token_index is the landing position here.
            running_by_region[ri] = running.replace(before, after, 1)
            already.append(
                {
                    "before": before,
                    "after": after,
                    "rule_id": decision.reason,
                    "token_index": decision.token_index,
                }
            )
            accepted_by_region[ri] = already

    accepted_total = sum(len(v) for v in accepted_by_region.values())

    # (4) Runaway tripwire — revert the WHOLE pass on a mass edit (fraction AND
    # a minimum absolute count, so a lone legit fix on a tiny region is safe).
    if (
        total_content_tokens
        and accepted_total >= _RUNAWAY_MIN_EDITS
        and accepted_total / total_content_tokens > _RUNAWAY_TOKEN_FRACTION
    ):
        _log(
            f"[ocr-repair] runaway tripwire: {accepted_total} edits / "
            f"{total_content_tokens} tokens > {_RUNAWAY_TOKEN_FRACTION:.2%} "
            f"-> reverting whole pass"
        )
        logger.warning(
            "ocr repair runaway tripwire fired (%d/%d) -> whole pass reverted",
            accepted_total,
            total_content_tokens,
        )
        stats = _empty_stats()
        stats["accepted"] = 0
        stats["runaway_reverted"] = True
        return RepairResult({}, stats)

    # (3b) Apply accepted edits -> repaired_text; assert conservation.
    edits_by_region: dict[int, dict[str, Any]] = {}
    post_repair_texts = dict(region_texts)
    for ri, edits in accepted_by_region.items():
        original = region_texts[ri]
        repaired = apply_accepted_edits(original, edits)
        try:
            assert_repair_conservation(original, repaired, edits)
        except RepairConservationError as exc:
            # Fail-closed for THIS region: drop its edits, keep verbatim.
            logger.warning("ocr repair conservation failed on region %d: %s", ri, exc)
            rejected_rows.append(
                {"region_index": ri, "reason": "conservation_replay_failed"}
            )
            continue
        if repaired == original:
            continue
        post_repair_texts[ri] = repaired
        edits_by_region[ri] = {
            "repaired_text": repaired,
            "ocr_repair": {
                "type": "ocr-confusable",
                "n_edits": len(edits),
                "edits": edits,
            },
        }

    # (5) Stats — residual garble density on the POST-repair text.
    residual = sum(_count_garble_tokens(t) for t in post_repair_texts.values())
    density = residual / total_content_tokens if total_content_tokens else 0.0
    score = max(0.0, min(_REPAIR_SCORE_CEILING, 1.0 - density))
    accepted_final = sum(
        v["ocr_repair"]["n_edits"] for v in edits_by_region.values()
    )
    stats = {
        "flagged_blocks": len(flagged_entries),
        "windows_dispatched": windows_dispatched,
        "windows_failed": windows_failed,
        "proposals": proposals,
        "accepted": accepted_final,
        "rejected": rejected_rows,
        "total_content_tokens": total_content_tokens,
        "residual_garble_tokens": residual,
        "unrepairable_defect_density": round(density, 6),
        "repair_stats_score": round(score, 6),
    }
    _log(
        f"[ocr-repair] flagged={len(flagged_entries)} windows={windows_dispatched} "
        f"proposals={proposals} accepted={accepted_final} "
        f"density={density:.4f} score={score:.4f}"
    )
    return RepairResult(edits_by_region, stats)


__all__ = [
    "resolve_ocr_confusable_repair_mode",
    "CONFUSABLE_MAP",
    "ConfusableRule",
    "LexiconConfusableRule",
    "region_text_is_garbled",
    "evaluate_edit",
    "EditDecision",
    "apply_accepted_edits",
    "assert_repair_conservation",
    "RepairConservationError",
    "run_ocr_confusable_repair",
    "RepairResult",
]
