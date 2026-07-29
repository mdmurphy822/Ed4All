"""Staged, source-grounded training-pair synthesis.

The legacy provider asks one response to plan, ground, instantiate a Bloom
skill, and realize every output field.  This opt-in wrapper separates those
jobs while retaining the existing provider's transport, model identity,
thinking-off payload, length bounds, and final factory/validator gates.

Stage A produces a small evidence plan.  Stage B realizes SFT.  Stages C1 and
C2 realize the DPO chosen and rejected answers independently.  A failed
deterministic stage contract is returned to the same stage as an exact,
bounded correction; it never triggers a blind whole-pair rewrite.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from Trainforge.generators._synthesis_common import SynthesisProviderError
from Trainforge.generators.synthesis_window_contract import (
    WINDOW_CONTRACT_VERSION,
    build_evidence_window,
    canonical_quote,
    objective_card,
    render_evidence_window,
    semantic_anchor_forms,
    semantic_anchor_hypothesis,
    semantic_anchors,
    semantic_anchor_sequence,
)


ENV_STAGED_SYNTHESIS_V4 = "TRAINFORGE_STAGED_SYNTHESIS_V4"
ENV_SERVED_CONTEXT_TOKENS = "TRAINFORGE_SYNTHESIS_SERVED_CONTEXT_TOKENS"
_TRUE = frozenset({"1", "true", "yes", "on"})
_BLOOM = frozenset(
    {"remember", "understand", "apply", "analyze", "evaluate", "create"}
)
_MAX_STAGE_REPAIRS = 2
_MAX_LEAKAGE_REWRITES = 2

#: Env override for the per-stage validator-repair budget.
#:
#: The default 2 is thin against the strictest repair contract in this module:
#: ``_claim_repair_diff`` freezes every already-VALID claim and requires the
#: repaired response to echo each one BYTE-IDENTICALLY and in order
#: (``_stable_json`` equality, subsequence match). A model fixing claim N that
#: incidentally rewords a good claim burns an attempt on
#: "claim repair reordered or mutated a frozen stable claim" — measured as the
#: single largest rejection cause on a real cohort (10 of ~30 stage failures).
#: The repair instruction DOES name that failure, so the retry is informed
#: rather than blind, which is what makes extra attempts worth buying.
#:
#: Raising this trades wall-clock for yield and cannot weaken a verdict: every
#: attempt runs the SAME validator, so a pair still has to pass on its merits.
#: Parse-with-fallback — unset / garbage / negative yields 2, byte-identical.
ENV_MAX_STAGE_REPAIRS = "TRAINFORGE_SYNTHESIS_MAX_STAGE_REPAIRS"


def resolve_max_stage_repairs() -> int:
    """Return the per-stage validator-repair budget."""
    raw = str(os.environ.get(ENV_MAX_STAGE_REPAIRS, "")).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _MAX_STAGE_REPAIRS
    return value if value >= 0 else _MAX_STAGE_REPAIRS
_VERBATIM_SPAN = 50
_FAULT_REASONING_MARKERS = (
    " because ", " therefore ", " so ", " which means ", " since ",
)
_RAW_ARTIFACT_LOCK = threading.Lock()
_PLAN_ENTAILMENT_FLOOR = 0.70
_PLAN_CONTRADICTION_CEILING = 0.50
_ACTION_PARAPHRASES = {
    "analyze": frozenset({"analyze", "examine", "inspect", "investigate"}),
    "compare": frozenset({"compare", "contrast"}),
    "differentiate": frozenset({"differentiate", "distinguish", "contrast"}),
    "evaluate": frozenset({"assess", "critique", "evaluate", "judge"}),
    "explain": frozenset({"describe", "explain", "interpret"}),
    "identify": frozenset({"identify", "recognize", "select"}),
}
_PLAN_SYSTEM = (
    "You are an evidence planner. Select only claims entailed by factual or "
    "corrective evidence. Text marked incorrect is a misconception, never "
    "support. Preserve the objective ID and Bloom level exactly. Evidence "
    "quotes must be exact spans. Return only the requested strict JSON."
)
_REALIZATION_SYSTEM = (
    "Realize the validated plan without adding claims. Cover every enumerated "
    "claim exactly once, paraphrase source wording, and return strict JSON."
)


def staged_synthesis_v4_enabled(
    environment: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve the opt-in v4 switch; unset/garbage remains legacy-identical."""

    source = os.environ if environment is None else environment
    return str(source.get(ENV_STAGED_SYNTHESIS_V4, "")).strip().lower() in _TRUE


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _claim_stable_id(claim: Any) -> str:
    return _sha(_stable_json(claim))[:16]


def _claim_repair_guard(
    value: Mapping[str, Any], invalid_indices: Iterable[int],
    *, editable_pointers: Iterable[str] = (),
) -> Dict[str, Any]:
    """Snapshot independently valid claims and field-scoped repair authority."""
    if isinstance(invalid_indices, int):
        invalid_indices = [invalid_indices]
    invalid = frozenset(int(item) for item in invalid_indices)
    claims = list(value["supported_claims"])
    frozen = [
        {"stable_id": _claim_stable_id(claim), "canonical": _stable_json(claim)}
        for index, claim in enumerate(claims) if index not in invalid
    ]
    editable = sorted(set(editable_pointers) | {"/supported_claims"})
    frozen_fields = {
        key: item for key, item in value.items()
        if f"/{key}" not in editable and key != "supported_claims"
    }
    return {
        "invalid_stable_ids": [
            _claim_stable_id(claims[index]) for index in sorted(invalid)
        ],
        "invalid_claim_count": len(invalid),
        "frozen_claims": frozen,
        "editable_json_pointers": editable,
        "frozen_fields_canonical": _stable_json(frozen_fields),
        "baseline_sha256": _sha(_stable_json(value)),
    }


def _claim_repair_diff(
    value: Any, guard: Mapping[str, Any],
) -> Optional[Dict[str, str]]:
    """Return a hashed reason when a claim repair mutates frozen plan state."""

    candidate_sha = _sha(_stable_json(value))
    if not isinstance(value, dict):
        return {
            "reason": "repaired response is not an object",
            "baseline_sha256": str(guard["baseline_sha256"]),
            "candidate_sha256": candidate_sha,
        }
    editable = set(guard.get("editable_json_pointers") or [])
    frozen_fields = {
        key: item for key, item in value.items()
        if f"/{key}" not in editable and key != "supported_claims"
    }
    if _stable_json(frozen_fields) != guard["frozen_fields_canonical"]:
        return {
            "reason": "claim repair mutated a noneditable field",
            "baseline_sha256": str(guard["baseline_sha256"]),
            "candidate_sha256": candidate_sha,
        }
    claims = value.get("supported_claims")
    reason: Optional[str] = None
    if not isinstance(claims, list):
        reason = "claim repair removed the supported_claims sequence"
    else:
        expected = [
            (str(item["stable_id"]), str(item["canonical"]))
            for item in guard["frozen_claims"]
        ]
        actual = [
            (_claim_stable_id(claim), _stable_json(claim)) for claim in claims
        ]
        if not (
            len(expected) <= len(actual)
            <= len(expected) + int(guard["invalid_claim_count"])
        ):
            reason = "claim repair inserted, merged, or deleted extra claims"
            return {
                "reason": reason,
                "baseline_sha256": str(guard["baseline_sha256"]),
                "candidate_sha256": candidate_sha,
            }
        cursor = 0
        for stable_id, canonical in actual:
            if cursor < len(expected) and (stable_id, canonical) == expected[cursor]:
                cursor += 1
        if cursor != len(expected):
            reason = "claim repair reordered or mutated a frozen stable claim"
        else:
            return None
    return {
        "reason": str(reason),
        "baseline_sha256": str(guard["baseline_sha256"]),
        "candidate_sha256": candidate_sha,
    }


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _operational_condition_error(task: Any, condition: Any) -> Optional[str]:
    """Reject vacuous conversion conditions while allowing paraphrases."""
    normalized_task = _clean(task).casefold()
    normalized_condition = _clean(condition).casefold()
    conversion = re.search(
        r"\b(?:convert|converting|rewrite|rewriting|transform|transforming|"
        r"rearrange|rearranging|express|expressing|put|putting|render|rendering)"
        r"\b",
        normalized_task,
    )
    target_match = re.search(
        r"\b(?:convert|converting|rewrite|rewriting|transform|transforming|"
        r"rearrange|rearranging|express|expressing|put|putting|render|rendering)"
        r"\b.*?\b(?:to|into|as)\s+(.+?)(?:\s+after\b|[,.;]|$)",
        normalized_condition,
    )
    if target_match is None:
        return None
    target = target_match.group(1).strip()
    if conversion is None:
        return (
            "learner task does not operationally perform the required "
            f"conversion to {target!r}"
        )
    # A requested conversion must transform the supplied input. For the
    # canonical y=m*x+b family, every supplied equation already beginning
    # with isolated y is already in slope-intercept representation.
    if "slope-intercept" in target:
        equality_count = len(re.findall(r"=", normalized_task))
        isolated_y_count = len(re.findall(r"\by\s*=", normalized_task))
        if equality_count >= 1 and isolated_y_count == equality_count:
            return (
                "learner task supplies every equation already in the required "
                "target representation, making the conversion vacuous"
            )
    return None


def _sentences(value: Any) -> List[str]:
    """Split realization text into stable semantic units without a tokenizer."""
    cleaned = _clean(value)
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|(?<=;)\s+", cleaned)
        if item.strip()
    ] or ([cleaned] if cleaned else [])


_NON_SUBSTANTIVE_DISCOURSE_MARKERS = frozenset({
    "additionally",
    "alternatively",
    "as a result",
    "consequently",
    "furthermore",
    "hence",
    "however",
    "in contrast",
    "meanwhile",
    "moreover",
    "nevertheless",
    "nonetheless",
    "on the other hand",
    "otherwise",
    "similarly",
    "therefore",
    "thus",
})


def _is_standalone_discourse_marker(value: Any) -> bool:
    """Recognize only a closed English marker with no claim-bearing tokens."""
    normalized = unicodedata.normalize("NFKC", _clean(value)).casefold()
    words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return " ".join(words) in _NON_SUBSTANTIVE_DISCOURSE_MARKERS


def _coverage_units(value: Any) -> List[str]:
    """Split semantic clauses so one grounded equation cannot bless a neighbor."""
    units: List[str] = []
    for sentence in _sentences(value):
        parts = [
            part.strip(" ,;")
            for part in re.split(
                r"\s*[,;]\s*"
                r"|\s+\b(?:although|and|but|or|plus|while|which)\b\s+",
                sentence,
                flags=re.IGNORECASE,
            )
            if part.strip(" ,;")
        ]
        parts = [
            (
                re.sub(
                    r"^(?:and|but|or|plus|therefore|thus)\s+", "", part,
                    flags=re.IGNORECASE,
                )
                if index else part
            )
            for index, part in enumerate(parts)
        ]
        # Sentence-initial transitions have no independent truth conditions.
        # A comma can isolate them as apparent clauses ("Therefore, X"). Drop
        # only a closed marker-only unit; the following substantive clause is
        # retained unchanged and still receives full adjudication.
        parts = [
            part for part in parts
            if not _is_standalone_discourse_marker(part)
        ]
        # An introductory dependent clause is a condition on its governing
        # clause, not an independently assertive proposition. Preserve the
        # pair as one NLI unit so e.g. "When X, Y" is never scored as bare X.
        if (
            len(parts) >= 2
            and re.match(
                r"^(?:after|although|because|before|given|if|once|since|unless|"
                r"when|whenever|where|whereas|while)\b",
                parts[0],
                flags=re.IGNORECASE,
            )
        ):
            parts[0:2] = [f"{parts[0]}, {parts[1]}"]
        units.extend(parts)
    return units


_PROPOSITION_ARTICLES = frozenset({"a", "an", "the"})
def _normalized_proposition(value: Any) -> Tuple[str, ...]:
    """Normalize only presentation-safe proposition differences.

    Deterministic provenance is intentionally narrower than semantic
    equivalence.  Unicode compatibility forms, case, whitespace, punctuation,
    and the three English articles are ignored; every other token must remain
    present and in the same order.
    """
    normalized = unicodedata.normalize("NFKC", _clean(value)).casefold()
    tokens = [
        token.replace("’", "'")
        for token in re.findall(r"[\w]+(?:['’][\w]+)?", normalized, re.UNICODE)
    ]
    return tuple(token for token in tokens if token not in _PROPOSITION_ARTICLES)


def _equal_canonical_proposition(left: Any, right: Any) -> bool:
    """Match one complete proposition; containment is never provenance."""
    normalized_left = _normalized_proposition(left)
    return bool(normalized_left) and normalized_left == _normalized_proposition(right)


def _relation_signature(value: Any) -> Optional[Dict[str, Any]]:
    """Build a conservative, direction-preserving proposition signature."""
    text = unicodedata.normalize("NFKC", _clean(value)).casefold()
    relations = semantic_anchor_sequence(text)
    if not relations:
        return None
    canonical = text
    for relation in relations:
        for form in sorted(semantic_anchor_forms(relation), key=len, reverse=True):
            canonical = re.sub(
                rf"\b{re.escape(form)}\b", f" relation_{relation} ", canonical,
            )
    tokens = _normalized_proposition(canonical)
    condition_markers = {"if", "when", "unless", "provided", "where"}
    return {
        "relations": tuple(relations),
        # Token order preserves subject/object direction and conditions.  It
        # also makes negation and additional claim material observable.
        "tokens": tokens,
        "polarity": tuple(
            token for token in tokens
            if token in {"not", "never", "no", "without"}
        ),
        "conditions": tuple(
            index for index, token in enumerate(tokens)
            if token in condition_markers
        ),
    }


def _relation_preserving_support(premise: Any, hypothesis: Any) -> bool:
    """Prove only structurally identical relations before invoking NLI."""
    left = _relation_signature(premise)
    right = _relation_signature(hypothesis)
    return bool(left is not None and left == right)


_RELATION_OPERATOR_RE = re.compile(r"(?<![<>=!])(?:≠|≤|≥|=|<|>)(?![=])")


def _relation_operator_mutation(
    premise: Any, hypothesis: Any,
) -> Optional[str]:
    """Reject introduced/conflicting mathematical relation operators."""
    normalize = {"≤": "<=", "≥": ">=", "≠": "!="}
    premise_ops = {
        normalize.get(item, item)
        for item in _RELATION_OPERATOR_RE.findall(_clean(premise))
    }
    hypothesis_ops = {
        normalize.get(item, item)
        for item in _RELATION_OPERATOR_RE.findall(_clean(hypothesis))
    }
    if hypothesis_ops and not hypothesis_ops <= premise_ops:
        return (
            "public claim mutates an authenticated mathematical relation "
            f"operator; source={sorted(premise_ops)}, "
            f"claim={sorted(hypothesis_ops)}"
        )
    return None


def _deterministic_entailment_basis(
    premise: Any, hypothesis: Any,
) -> Optional[str]:
    """Prove a complete authoritative clause without probabilistic NLI.

    A source proposition may append a contrast or qualification after a comma
    while the realization states only its positive governed relation.  Compare
    complete clause units as well as the complete premise, preserving the
    introductory governor produced by :func:`_coverage_units`.  Exact,
    relation-signature, and symbolic proofs are deliberately conservative:
    none permits containment, polarity reversal, operand reordering, or a
    lexical collision on an unrelated subject.
    """
    hypothesis_text = _clean(hypothesis)
    if not hypothesis_text:
        return None
    candidates = [_clean(premise)]
    candidates.extend(
        unit for unit in _coverage_units(premise)
        if not _equal_canonical_proposition(unit, premise)
    )
    for candidate in candidates:
        if _equal_canonical_proposition(candidate, hypothesis_text):
            return "exact_authoritative_clause"
        if _relation_preserving_support(candidate, hypothesis_text):
            return "relation_preserving_clause"
        if _symbolic_math_supports(candidate, hypothesis_text):
            return "symbolic_authoritative_clause"
        if formula_relation_proof(candidate, hypothesis_text) is not None:
            return "formula_relation_authoritative_clause"
    return None


def _explicit_relation_conflict(premise: Any, hypothesis: Any) -> bool:
    """Corroborate an NLI contradiction with an observable proposition conflict.

    A high contradiction score alone cannot distinguish an omitted relation
    from its negation.  We therefore require the same relation vocabulary plus
    either explicit polarity reversal or a direction-changing permutation of
    the same proposition tokens.  Novel or missing material remains
    unsupported and is still rejected by the unchanged entailment floor.
    """
    left = _relation_signature(premise)
    right = _relation_signature(hypothesis)
    if left is None or right is None:
        return False
    if left["relations"] != right["relations"]:
        return False
    left_tokens = tuple(left["tokens"])
    right_tokens = tuple(right["tokens"])
    polarity_words = {"do", "does", "did", "not", "never", "no", "without"}
    if bool(left["polarity"]) != bool(right["polarity"]):
        return (
            tuple(token for token in left_tokens if token not in polarity_words)
            == tuple(token for token in right_tokens if token not in polarity_words)
        )
    return (
        left_tokens != right_tokens
        and sorted(left_tokens) == sorted(right_tokens)
    )


def _first_verbatim_span(value: str, source: str) -> Optional[str]:
    candidate = _clean(value).lower()
    evidence = _clean(source).lower()
    if len(candidate) < _VERBATIM_SPAN or len(evidence) < _VERBATIM_SPAN:
        return None
    for index in range(len(candidate) - _VERBATIM_SPAN + 1):
        span = candidate[index:index + _VERBATIM_SPAN]
        if span in evidence:
            return span
    return None


def _first_distinctive_verbatim_span(
    value: str,
    source: str,
    *,
    mandatory_contract_text: Iterable[str] = (),
    canonical_contract_text: Iterable[str] = (),
    nondistinctive_contexts: Iterable[Mapping[str, Any]] = (),
) -> Optional[str]:
    """Return copied source text after narrowly subtracting mandated boilerplate.

    Contract text is exempt only when the same normalized text occurs in at
    least two independently identified contexts.  This makes the exemption a
    provenance/frequency result rather than a phrase whitelist.  Unmatched
    source narrative, including long narrative adjacent to an exempt phrase,
    remains visible to the ordinary 50-character detector.
    """
    candidate = _clean(value)
    evidence = _clean(source)
    if len(candidate) < _VERBATIM_SPAN or len(evidence) < _VERBATIM_SPAN:
        return None

    def structural_form(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", normalized)
        normalized = re.sub(r"\b[a-z]\b", "<symbol>", normalized)
        return " ".join(normalized.split())

    contexts: Dict[str, str] = {}
    for item in nondistinctive_contexts:
        if not isinstance(item, Mapping):
            continue
        context_id = _clean(item.get("context_id"))
        text = _clean(item.get("text")).casefold()
        if context_id and text:
            contexts[context_id] = text
    exempt = [False] * len(candidate)
    # Canonical objective/schema text is not provider-authored narrative.
    # Exempt only the exact canonical span itself; adjacent source prose is
    # never included in the exemption.
    for contract_text in canonical_contract_text:
        phrase = _clean(contract_text)
        if len(phrase) < _VERBATIM_SPAN:
            continue
        if phrase.casefold() not in evidence.casefold():
            continue
        for match in re.finditer(re.escape(phrase), candidate, re.IGNORECASE):
            exempt[match.start():match.end()] = [True] * len(phrase)
    for contract_text in mandatory_contract_text:
        phrase = _clean(contract_text)
        if len(phrase) < _VERBATIM_SPAN:
            continue
        normalized = phrase.casefold()
        structural = structural_form(phrase)
        frequency = sum(
            normalized in text or structural in structural_form(text)
            for text in contexts.values()
        )
        if frequency < 2:
            continue
        for match in re.finditer(re.escape(phrase), candidate, re.IGNORECASE):
            if normalized in evidence.casefold():
                exempt[match.start():match.end()] = [True] * len(phrase)

    lowered_source = evidence.casefold()
    lowered_candidate = candidate.casefold()
    for index in range(len(candidate) - _VERBATIM_SPAN + 1):
        if any(exempt[index:index + _VERBATIM_SPAN]):
            continue
        span = lowered_candidate[index:index + _VERBATIM_SPAN]
        if span in lowered_source:
            return span
    return None


def _canonical_evidence_quote(quote: str, source: str) -> Optional[str]:
    """Map punctuation/whitespace-normalized model text to an exact source span."""
    quote_clean = _clean(quote)
    source_clean = _clean(source)
    direct_index = source_clean.lower().find(quote_clean.lower())
    if direct_index >= 0:
        return source_clean[direct_index:direct_index + len(quote_clean)]
    quote_tokens = [
        match.group(0).lower()
        for match in re.finditer(r"[A-Za-z0-9]+", quote_clean)
    ]
    source_matches = list(re.finditer(r"[A-Za-z0-9]+", source_clean))
    source_tokens = [match.group(0).lower() for match in source_matches]
    if len(quote_tokens) < 4:
        return None
    width = len(quote_tokens)
    for index in range(len(source_tokens) - width + 1):
        if source_tokens[index:index + width] != quote_tokens:
            continue
        return source_clean[
            source_matches[index].start():source_matches[index + width - 1].end()
        ]
    return None


def _math_text(value: Any) -> str:
    return (
        _clean(value)
        .replace("²", "^2").replace("³", "^3")
        .replace("−", "-").replace("×", "*").replace("÷", "/")
    )


def _equation_signatures(value: Any) -> set[str]:
    """Canonical polynomial equalities, invariant to rearrangement/scaling."""
    import sympy
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        convert_xor,
        parse_expr,
        standard_transformations,
    )
    text = _math_text(value)
    number = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
    atom = (
        rf"(?:[A-Za-z]\w*|{number}|"
        rf"-?\d*\([^=<>]+\))"
    )
    side = rf"{atom}(?:\s*[\^*/+\-]\s*{atom})*"
    candidates = [
        f"{match.group('left')}={match.group('right')}"
        for match in re.finditer(
            rf"(?<![\w.])(?P<left>{side})\s*=\s*"
            rf"(?P<right>{side})(?!\w)",
            text,
        )
    ]
    transforms = standard_transformations + (
        convert_xor, implicit_multiplication_application,
    )
    signatures: set[str] = set()
    for candidate in candidates:
        left, right = candidate.split("=", 1)
        try:
            expression = sympy.expand(
                parse_expr(left.strip(), transformations=transforms)
                - parse_expr(right.strip(), transformations=transforms)
            )
            symbols = sorted(expression.free_symbols, key=str)
            if not symbols:
                signatures.add(sympy.srepr(expression))
                continue
            polynomial = sympy.Poly(expression, *symbols)
            if polynomial.is_zero:
                signatures.add("0")
                continue
            leading = polynomial.LC()
            canonical = sympy.expand(expression / leading)
            signatures.add(sympy.srepr(canonical))
        except (
            TypeError,
            ValueError,
            SyntaxError,
            sympy.SympifyError,
            sympy.PolynomialError,
        ):
            continue
    return signatures


def _symbolic_math_supports(premise: Any, hypothesis: Any) -> bool:
    premise_signatures = _equation_signatures(premise)
    hypothesis_signatures = _equation_signatures(hypothesis)
    if not hypothesis_signatures or not hypothesis_signatures <= premise_signatures:
        return False
    premise_text = _math_text(premise).lower()
    hypothesis_text = _math_text(hypothesis).lower()
    if (
        re.search(r"\b(?:not|never|incorrect|false)\b", hypothesis_text)
        and not re.search(r"\b(?:not|never|incorrect|false)\b", premise_text)
    ):
        return False
    stop = {
        "the", "a", "an", "to", "by", "when", "after", "before", "both",
        "sides", "equation", "solving", "yields", "gives", "result",
    }
    hypothesis_words = {
        token for token in re.findall(r"[a-z]{3,}", hypothesis_text)
        if token not in stop
    }
    premise_words = set(re.findall(r"[a-z]{3,}", premise_text))
    return (
        not hypothesis_words
        or len(hypothesis_words & premise_words) / len(hypothesis_words) >= 0.6
    )


def _formula_frame(value: Any) -> Optional[Tuple[str, str, str, str]]:
    """Extract one fully typed binary formula without inferring terminology."""
    text = _math_text(value).casefold()
    symbolic = re.search(
        r"\b(?P<left>[a-z])\s*=\s*(?P<first>[a-z])"
        r"(?:\s*\*\s*)?(?P<second>[a-z])\b",
        text,
    )
    if symbolic:
        return (
            symbolic.group("left"), "multiply",
            symbolic.group("first"), symbolic.group("second"),
        )
    verbal = re.search(
        r"\b(?P<left>[a-z][a-z_-]*)\s+equals\s+"
        r"(?P<first>[a-z][a-z_-]*)\s+"
        r"(?P<operator>times|plus|minus|divided\s+by)\s+"
        r"(?P<second>[a-z][a-z_-]*)\b",
        text,
    )
    if not verbal:
        return None
    operator = {
        "times": "multiply",
        "plus": "add",
        "minus": "subtract",
        "divided by": "divide",
    }[re.sub(r"\s+", " ", verbal.group("operator"))]
    return (
        verbal.group("left"), operator,
        verbal.group("first"), verbal.group("second"),
    )


def _alpha_formula_shape(
    frame: Tuple[str, str, str, str],
) -> Tuple[str, str, str, str]:
    """Rename operands by first occurrence while preserving role and reuse."""
    names: Dict[str, str] = {}

    def alpha(name: str) -> str:
        return names.setdefault(name, f"v{len(names)}")

    left, operator, first, second = frame
    return alpha(left), operator, alpha(first), alpha(second)


def formula_relation_proof(
    premise: Any, hypothesis: Any,
) -> Optional[Dict[str, Any]]:
    """Prove a relation plus alpha-equivalent formula before unchanged NLI.

    This deliberately returns ``None`` for unresolved language. Formula shape
    is necessary but insufficient: relation anchors, polarity, numeric values,
    unit constraints, and a minimum shared subject vocabulary must also agree.
    """
    premise_text = _math_text(premise).casefold()
    hypothesis_text = _math_text(hypothesis).casefold()
    premise_frame = _formula_frame(premise_text)
    hypothesis_frame = _formula_frame(hypothesis_text)
    if premise_frame is None or hypothesis_frame is None:
        return None
    premise_shape = _alpha_formula_shape(premise_frame)
    hypothesis_shape = _alpha_formula_shape(hypothesis_frame)
    if premise_shape != hypothesis_shape:
        return None
    premise_relations = tuple(semantic_anchor_sequence(premise_text))
    hypothesis_relations = tuple(semantic_anchor_sequence(hypothesis_text))
    if not hypothesis_relations or premise_relations != hypothesis_relations:
        return None
    polarity = re.compile(r"\b(?:not|never|no|without|incorrect|false)\b")
    if bool(polarity.search(premise_text)) != bool(polarity.search(hypothesis_text)):
        return None
    if {
        _number_key(item) for item in _NUMBER_RE.findall(premise_text)
    } != {
        _number_key(item) for item in _NUMBER_RE.findall(hypothesis_text)
    }:
        return None
    unit_constraint = re.compile(
        r"\b(?:consistent|inconsistent|same|different)\s+units?\b"
    )
    premise_units = tuple(unit_constraint.findall(premise_text))
    hypothesis_units = tuple(unit_constraint.findall(hypothesis_text))
    if premise_units != hypothesis_units:
        return None
    stop = {
        "when", "their", "them", "each", "using", "found", "calculated",
        "equals", "times", "plus", "minus", "divided", "with", "units",
        "consistent", "total", "individual", "the", "and", "by",
    }

    def vocabulary(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z]{3,}", text)
            if token not in stop
        }

    premise_words = vocabulary(premise_text)
    hypothesis_words = vocabulary(hypothesis_text)
    if (
        not hypothesis_words
        or len(premise_words & hypothesis_words) / len(hypothesis_words) < 0.4
    ):
        return None
    return {
        "proof": "formula_relation_proof.v1",
        "formula_shape": premise_shape,
        "relations": premise_relations,
        "numeric_literals": sorted(
            {_number_key(item) for item in _NUMBER_RE.findall(premise_text)}
        ),
        "unit_constraints": premise_units,
    }


def affine_two_line_relation_proof(
    source: Any, proposition: Any,
) -> Optional[Dict[str, Any]]:
    """Prove only the exact solution-count proposition of two affine lines."""
    text = _clean(source)
    claim = _clean(proposition).casefold()
    relation = (
        "same_line"
        if (
            ("same line" in claim or "coincident" in claim)
            and "infinitely many solution" in claim
        )
        else "parallel_distinct"
        if "no solution" in claim
        else "intersecting"
        if "one solution" in claim
        else None
    )
    if relation is None:
        return None
    try:
        import sympy
        from lib.validators.worked_example_math import (
            _collect_equations, _iter_fragments,
        )
        fragments = _iter_fragments(text)
        if not fragments:
            # Semantic synthesis blocks are plain text after HTML projection.
            # Recover only conservative two-variable affine equation shapes.
            coefficient = r"(?:\(?-?\d+(?:/\d+)?\)?\s*)?"
            variable_term = rf"{coefficient}[xy]"
            affine = re.compile(
                rf"(?P<eq>{variable_term}(?:\s*[+-]\s*{variable_term})?"
                rf"\s*=\s*(?:{variable_term}(?:\s*[+-]\s*"
                rf"(?:-?\d+(?:/\d+)?))?|-?\d+(?:/\d+)?))"
            )
            fragments = [
                (match.group("eq"), False) for match in affine.finditer(text)
            ]
        equations, _chains = _collect_equations(fragments)
    except Exception:  # noqa: BLE001
        return None
    x, y = sympy.symbols("x y")
    triples: list[tuple[Any, Any, Any]] = []
    for lhs, rhs, _raw, _ctx in equations:
        try:
            poly = sympy.Poly(sympy.expand(lhs - rhs), x, y)
        except Exception:  # noqa: BLE001
            continue
        if poly.total_degree() > 1:
            continue
        a = poly.coeff_monomial(x)
        b = poly.coeff_monomial(y)
        c = poly.coeff_monomial(1)
        if a == 0 and b == 0:
            continue
        triples.append((a, b, c))
    for left_index, left in enumerate(triples):
        for right in triples[left_index + 1:]:
            a1, b1, c1 = left
            a2, b2, c2 = right
            determinant = sympy.simplify(a1 * b2 - a2 * b1)
            ac = sympy.simplify(a1 * c2 - a2 * c1)
            bc = sympy.simplify(b1 * c2 - b2 * c1)
            observed = (
                "intersecting" if determinant != 0
                else "same_line" if ac == 0 and bc == 0
                else "parallel_distinct"
            )
            if observed != relation:
                continue
            payload = {
                "proof": "affine_two_line_relation_proof.v1",
                "relation": relation,
                "left_coefficients": [str(item) for item in left],
                "right_coefficients": [str(item) for item in right],
                "determinant": str(determinant),
                "proposition_sha256": _sha(claim),
            }
            payload["proof_sha256"] = _sha(_stable_json(payload))
            return payload
    return None


_NUMBER_RE = re.compile(r"(?<![\d.])-?(?:\d+(?:\.\d+)?|\.\d+)(?!\d)")
_SUPERSCRIPT_TRANSLATION = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁻",
    "0123456789-",
)
_SUBSCRIPT_TRANSLATION = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉₋",
    "0123456789-",
)
_SUPERSCRIPT_SEQUENCE_RE = re.compile(r"[⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_ARITHMETIC_EQUALITY_RE = re.compile(
    r"(?<![\w.])(-?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*[+\-*/]\s*-?(?:\d+(?:\.\d+)?|\.\d+))+)"
    r"\s*=\s*(-?(?:\d+(?:\.\d+)?|\.\d+))(?![\w.])"
)
_ARITHMETIC_RELATION_RE = re.compile(
    r"(?<![\w.])(?P<left>-?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*[+\-*/]\s*-?(?:\d+(?:\.\d+)?|\.\d+))+)"
    r"\s*(?P<operator><=|>=|=|<|>)\s*"
    r"(?P<right>-?(?:\d+(?:\.\d+)?|\.\d+))(?!\w)"
)


def _number_key(value: str) -> str:
    """Canonicalize a decimal literal without binary floating-point drift."""
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    normalized = number.normalize()
    return "0" if normalized == 0 else format(normalized, "f")


def _scenario_numeric_literals(value: Any) -> set[str]:
    """Return task values, excluding structural mathematical notation.

    Only exponent-position numerals are intrinsically structural. Coefficients,
    constants (including zero), equation right-hand sides, attached-unit
    quantities, and semantic subscripts remain closed-world values. They must
    occur in source/objective/accepted claims or in the typed
    ``generated_givens`` ledger. Any other schema/index notation needs
    independent proof before a caller may exempt it; spelling it with digits
    is not proof.
    """
    raw_text = _clean(value)
    # Preserve the syntactic fact that a Unicode superscript sequence is an
    # exponent before normalizing it to the ASCII form consumed below.
    text = _SUPERSCRIPT_SEQUENCE_RE.sub(
        lambda match: "^" + match.group(0).translate(_SUPERSCRIPT_TRANSLATION),
        raw_text,
    ).translate(_SUBSCRIPT_TRANSLATION)
    text = _math_text(text)
    literals: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        start, _ = match.span()
        before = text[:start]
        if before.rstrip().endswith("^"):
            continue
        literals.add(_number_key(match.group(0)))
    return literals


def _evaluate_arithmetic(expression: str) -> Optional[Decimal]:
    """Evaluate a literal-only arithmetic expression exactly."""
    import sympy
    if not re.fullmatch(
        r"\s*-?(?:\d+(?:\.\d+)?|\.\d+)"
        r"(?:\s*[+\-*/]\s*-?(?:\d+(?:\.\d+)?|\.\d+))+\s*",
        expression,
    ):
        return None
    try:
        result = sympy.sympify(expression, evaluate=True)
        if result.free_symbols or not result.is_finite:
            return None
        return Decimal(str(result.evalf(40)))
    except (TypeError, ValueError, sympy.SympifyError, InvalidOperation):
        return None


def _deterministic_derivation_ledger(
    *, prompt: str, output: str, claims: Iterable[str],
    generated_givens: Iterable[Mapping[str, Any]] = (),
    source_facts: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Prove every newly introduced number by a closed-world exact step.

    Prompt givens and grounded claims seed the ledger. A new numeric literal is
    admitted only as the exact result of a literal arithmetic equality whose
    operands were already grounded. Symbolic equalities are admitted only when
    algebraically equivalent to a seeded equality. Nothing is inferred from
    free-form prose.
    """
    given_facts = [
        " ".join((
            _clean(item.get("symbol")), _clean(item.get("value")),
            _clean(item.get("unit")),
        )).strip()
        for item in generated_givens
    ]
    authoritative_facts = (
        list(source_facts) if source_facts is not None else [prompt]
    )
    seed_text = " ".join([
        *(_clean(item) for item in authoritative_facts),
        *(_clean(item) for item in claims), *given_facts,
    ])
    grounded = {_number_key(item) for item in _NUMBER_RE.findall(seed_text)}
    seeded_equations: set[str] = set()
    for seed_unit in [
        *(
            unit
            for fact in authoritative_facts
            for unit in _coverage_units(fact)
        ),
        *(_coverage_units(item) for item in claims),
        *(_coverage_units(item) for item in given_facts),
    ]:
        if isinstance(seed_unit, list):
            for nested_unit in seed_unit:
                seeded_equations.update(_equation_signatures(nested_unit))
        else:
            seeded_equations.update(_equation_signatures(seed_unit))
    ledger: List[Dict[str, Any]] = []
    for unit_index, unit in enumerate(_coverage_units(output)):
        unit_numbers = [_number_key(item) for item in _NUMBER_RE.findall(unit)]
        pending = [item for item in unit_numbers if item not in grounded]
        numeric_spans: List[Tuple[int, int]] = []
        for relation in _ARITHMETIC_RELATION_RE.finditer(unit):
            expression = relation.group("left")
            operator = relation.group("operator")
            rendered_result = relation.group("right")
            numeric_spans.append(relation.span())
            operands = [_number_key(item) for item in _NUMBER_RE.findall(expression)]
            result_key = _number_key(rendered_result)
            if any(item not in grounded for item in operands):
                return ledger, (
                    f"output clause {unit_index} uses ungrounded arithmetic "
                    "operand(s)"
                )
            actual = _evaluate_arithmetic(expression)
            try:
                expected = Decimal(rendered_result)
            except InvalidOperation:
                actual = None
            truth = (
                actual == expected if operator == "=" else
                actual < expected if operator == "<" and actual is not None else
                actual > expected if operator == ">" and actual is not None else
                actual <= expected if operator == "<=" and actual is not None else
                actual >= expected if operator == ">=" and actual is not None else
                False
            )
            if actual is None or not truth:
                return ledger, (
                    f"output clause {unit_index} has false arithmetic relation"
                )
            step = {
                "kind": "numeric_arithmetic_relation",
                "unit_index": unit_index,
                "premise_sha256": _sha("|".join(operands)),
                "clause_span_sha256": _sha(relation.group(0)),
                "relation_signature_sha256": _sha(
                    f"{_number_key(str(actual))}{operator}{result_key}"
                ),
                "result_sha256": _sha(result_key),
                "grounds_entire_clause": not bool(
                    re.search(
                        r"[A-Za-z]",
                        unit[:relation.start()] + unit[relation.end():],
                    )
                ),
            }
            ledger.append(step)
            grounded.add(result_key)
            pending = [item for item in pending if item != result_key]
        if pending:
            return ledger, (
                f"output clause {unit_index} introduces ungrounded numeric "
                f"literal(s): {pending}"
            )
        symbolic_text = unit
        for start, end in reversed(numeric_spans):
            symbolic_text = symbolic_text[:start] + " " + symbolic_text[end:]
        signatures = _equation_signatures(symbolic_text)
        if signatures and not signatures <= seeded_equations:
            return ledger, (
                f"output clause {unit_index} introduces an ungrounded "
                "symbolic equation"
            )
        for signature in sorted(signatures & seeded_equations):
            ledger.append({
                "kind": "symbolic_equivalence",
                "unit_index": unit_index,
                "premise_sha256": _sha(seed_text),
                "clause_span_sha256": _sha(unit),
                "relation_signature_sha256": _sha(signature),
                "grounds_entire_clause": not bool(
                    re.search(r"[A-Za-z]", symbolic_text)
                ),
            })
    return ledger, None


@dataclass(frozen=True)
class _StageResult:
    value: Dict[str, Any]
    usage: Dict[str, int]
    requests: int


class StagedSynthesisProvider:
    """Expose the legacy paraphrase API through four constrained calls."""

    def __init__(self, base_provider: Any) -> None:
        self._base = base_provider
        self._capture = getattr(base_provider, "_capture", None)
        self._model = str(getattr(base_provider, "_model", "unknown"))
        self._provider_name = str(
            getattr(base_provider, "_provider_name", "unknown")
        )
        self._plan_nli = getattr(base_provider, "_plan_nli_scorer", None)
        self._validation_audit = threading.local()
        self._verified_served_context_tokens: Optional[int] = None

    def _decision_identity_context(self) -> Dict[str, Any]:
        """Return provider identity bound into every captured model call.

        Specialized staged contracts extend this mapping with their complete
        semantic identity.  Keeping the hook here ensures failures as well as
        successful calls carry the same identity evidence.
        """
        return {
            "provider": self._provider_name,
            "model": self._model,
        }

    def bind_verified_served_context(
        self, telemetry_contract: Mapping[str, Any],
    ) -> None:
        """Bind a fail-closed context window from benchmark preflight."""
        served = telemetry_contract.get("served_context_tokens")
        parser_facts = (
            telemetry_contract.get("parser_probe", {})
            .get("static_kv_startup_facts", {})
        )
        static_served = parser_facts.get("max_seq_len")
        if (
            telemetry_contract.get("status") != "accepted"
            or not isinstance(served, int)
            or served <= 0
            or served != static_served
        ):
            raise SynthesisProviderError(
                "verified telemetry context identities are incomplete or "
                "inconsistent",
                code="staged_context_window_unverified",
            )
        self._verified_served_context_tokens = served

    @property
    def api_url(self) -> str:
        return self._base.api_url

    @property
    def base_url(self) -> str:
        return self._base.base_url

    @property
    def client(self) -> Any:
        return self._base.client

    @staticmethod
    def _focus(chunk: Mapping[str, Any]) -> Dict[str, Any]:
        focus = chunk.get("synthesis_focus_objective")
        if not isinstance(focus, Mapping):
            raise SynthesisProviderError(
                "staged synthesis requires a canonical objective focus",
                code="staged_objective_focus_missing",
            )
        objective_id = _clean(focus.get("id")).lower()
        statement = _clean(focus.get("statement"))
        bloom = _clean(focus.get("bloom_level")).lower()
        if (
            not objective_id
            or not statement
            or bloom not in _BLOOM
            or list(chunk.get("learning_outcome_refs") or []) != [objective_id]
            or _clean(chunk.get("bloom_level")).lower() != bloom
        ):
            raise SynthesisProviderError(
                "focused chunk does not match its canonical objective/Bloom",
                code="staged_objective_focus_mismatch",
            )
        return objective_card(focus)

    def _nli(self) -> Any:
        if self._plan_nli is None:
            from lib.classifiers.nli_classifier import NliClassifier
            self._plan_nli = NliClassifier.get_or_load()
        if self._plan_nli is None:
            raise SynthesisProviderError(
                "staged planning requires the claim-support NLI classifier",
                code="staged_plan_nli_unavailable",
            )
        return self._plan_nli

    def _scores(
        self,
        pairs: List[Tuple[str, str]],
        *,
        decision_types: Any,
    ) -> List[Any]:
        scorer = self._nli()
        score_batch = getattr(scorer, "score_batch", None)
        if callable(score_batch):
            scores = list(score_batch(pairs=pairs))
        else:
            scores = [
            scorer.score_pair(premise=premise, hypothesis=hypothesis)
            for premise, hypothesis in pairs
            ]
        kinds = (
            [str(decision_types)] * len(pairs)
            if isinstance(decision_types, str)
            else list(decision_types)
        )
        if len(kinds) != len(pairs) or len(scores) != len(pairs):
            raise SynthesisProviderError(
                "NLI scorer result count differs from the submitted pair count",
                code="staged_nli_cardinality_mismatch",
            )
        for kind, (premise, hypothesis), score in zip(kinds, pairs, scores):
            self._record_nli(
                decision_type=kind,
                premise=premise,
                hypothesis=hypothesis,
                score=score,
            )
        return scores

    def _score_pair(
        self, *, decision_type: str, premise: str, hypothesis: str,
    ) -> Any:
        score = self._nli().score_pair(
            premise=premise, hypothesis=hypothesis,
        )
        self._record_nli(
            decision_type=decision_type,
            premise=premise,
            hypothesis=hypothesis,
            score=score,
        )
        return score

    def _record_nli(
        self, *, decision_type: str, premise: str, hypothesis: str, score: Any,
    ) -> None:
        rows = getattr(self._validation_audit, "nli", None)
        if rows is None:
            rows = []
            self._validation_audit.nli = rows
        rows.append({
            "decision_type": decision_type,
            "stage": str(getattr(self._validation_audit, "stage", "unknown")),
            "attempt": int(getattr(self._validation_audit, "attempt", 0)),
            "premise_sha256": _sha(premise),
            "hypothesis_sha256": _sha(hypothesis),
            "entailment": float(score.entailment),
            "contradiction": float(score.contradiction),
        })

    def _semantic_coverage_error(
        self, output: Any, plan: Mapping[str, Any], *, prompt: Any = "",
    ) -> Optional[str]:
        """Ground all claims and units with a deterministic bipartite assignment.

        Every claim needs a grounded realization unit, while an integrated
        unit may cover several atomic claims and additional units may
        reinforce a claim.  Every substantive unit must itself be grounded.
        Model-reported indices remain metadata and are checked separately.
        """
        units = _coverage_units(output)
        if not units:
            return "realization output is empty"
        claims = [
            _clean(item.get("claim"))
            for item in plan.get("supported_claims") or []
        ]
        quotes = [
            _clean(item.get("evidence_quote"))
            for item in plan.get("supported_claims") or []
        ]
        ledger, derivation_error = _deterministic_derivation_ledger(
            prompt=_clean(prompt), output=_clean(output), claims=claims,
            generated_givens=plan.get("generated_givens") or [],
            source_facts=[
                _clean(item.get("evidence_quote"))
                for item in plan.get("supported_claims") or []
            ],
        )
        self._validation_audit.derivations = ledger
        if derivation_error:
            return derivation_error
        prompt_facts = _coverage_units(prompt)
        premises = claims + prompt_facts
        pairs = [(unit, premise) for premise in premises for unit in units]
        deterministic_bases = [
            _deterministic_entailment_basis(premise, unit)
            for unit, premise in pairs
        ]
        unresolved_pairs = [
            pair for pair, basis in zip(pairs, deterministic_bases)
            if basis is None
        ]
        unresolved_scores = iter(self._scores(
            unresolved_pairs, decision_types="semantic_coverage",
        ))

        @dataclass(frozen=True)
        class _DeterministicEntailmentScore:
            entailment: float = 1.0
            contradiction: float = 0.0

        scores = [
            (
                _DeterministicEntailmentScore()
                if basis is not None else next(unresolved_scores)
            )
            for basis in deterministic_bases
        ]
        matrix: List[List[Any]] = []
        cursor = 0
        for _premise in premises:
            row = []
            for _unit in units:
                row.append(scores[cursor])
                cursor += 1
            matrix.append(row)

        grounded = [
            [
                (
                    float(score.entailment) >= _PLAN_ENTAILMENT_FLOOR
                    and float(score.contradiction)
                    < _PLAN_CONTRADICTION_CEILING
                )
                for score in row
            ]
            for row in matrix
        ]
        provenance_rows = []
        for claim_index, (claim, quote) in enumerate(zip(claims, quotes)):
            for unit_index, unit in enumerate(units):
                match = next((
                    (name, text) for name, text in (
                        ("accepted_plan_claim", claim),
                        ("exact_evidence_quote", quote),
                    )
                    if _equal_canonical_proposition(unit, text)
                ), None)
                if match is None:
                    continue
                basis, root_proposition = match
                grounded[claim_index][unit_index] = True
                provenance_rows.append({
                    "basis": basis,
                    "claim_index": claim_index,
                    "unit_index": unit_index,
                    "claim_sha256": _sha(claim),
                    "evidence_quote_sha256": _sha(quote),
                    "unit_sha256": _sha(unit),
                    "root_proposition_sha256": _sha(root_proposition),
                    "clause_span_sha256": _sha(unit),
                })

        # An exact full-output proposition may be split into several coverage
        # units (for example at a comma, conjunction, or relative clause).
        # Preserve its independently validated provenance across those units
        # only when the *entire* normalized output equals the authoritative
        # claim or quote.  This is inheritance from one root proposition, not
        # a relaxed per-unit substring match.
        full_proposition_matches: Dict[int, Tuple[str, str]] = {}
        for claim_index, (claim, quote) in enumerate(zip(claims, quotes)):
            match = next((
                (name, text) for name, text in (
                    ("accepted_plan_claim", claim),
                    ("exact_evidence_quote", quote),
                )
                if _equal_canonical_proposition(output, text)
            ), None)
            if match is None:
                continue
            full_proposition_matches[claim_index] = match
            basis, root_proposition = match
            root_sha = _sha(root_proposition)
            for unit_index, unit in enumerate(units):
                grounded[claim_index][unit_index] = True
                if not any(
                    row["claim_index"] == claim_index
                    and row["unit_index"] == unit_index
                    for row in provenance_rows
                ):
                    provenance_rows.append({
                        "basis": f"{basis}_full_output_inheritance",
                        "claim_index": claim_index,
                        "unit_index": unit_index,
                        "claim_sha256": _sha(claim),
                        "evidence_quote_sha256": _sha(quote),
                        "unit_sha256": _sha(unit),
                        "root_proposition_sha256": root_sha,
                        "clause_span_sha256": _sha(unit),
                    })
        # Record clause-level deterministic proofs only when a stronger exact
        # whole-proposition provenance row did not already establish the same
        # unit. This preserves the audit precedence contract.
        for pair_index, basis in enumerate(deterministic_bases):
            if basis is None:
                continue
            claim_index, unit_index = divmod(pair_index, len(units))
            if claim_index >= len(claims) or any(
                row["claim_index"] == claim_index
                and row["unit_index"] == unit_index
                for row in provenance_rows
            ):
                continue
            provenance_rows.append({
                "basis": basis,
                "claim_index": claim_index,
                "unit_index": unit_index,
                "claim_sha256": _sha(claims[claim_index]),
                "evidence_quote_sha256": _sha(quotes[claim_index]),
                "unit_sha256": _sha(units[unit_index]),
                "root_proposition_sha256": _sha(claims[claim_index]),
                "clause_span_sha256": _sha(units[unit_index]),
            })
        self._validation_audit.deterministic_provenance = provenance_rows

        # Claims must be supported by the full response and attributable to
        # monotonic, adjacent clause spans. This admits a faithful multi-clause
        # paraphrase without allowing partial or reordered coverage.
        full_scores = self._scores(
            [(_clean(output), claim) for claim in claims],
            decision_types="semantic_full_response_claim",
        )
        self._validation_audit.deterministic_full_claim_indices = []
        span_cursor = 0
        for claim_index, claim in enumerate(claims):
            full_score = full_scores[claim_index]
            deterministic_full = claim_index in full_proposition_matches
            if deterministic_full:
                self._validation_audit.deterministic_full_claim_indices.append(
                    claim_index
                )
            if (
                not deterministic_full
                and float(full_score.contradiction) >= _PLAN_CONTRADICTION_CEILING
                and _explicit_relation_conflict(claim, output)
            ):
                return f"full response contradicts atomic claim {claim_index}"
            span_candidates: List[Tuple[int, int, Any]] = []
            span_pairs: List[Tuple[str, str]] = []
            span_labels: List[Tuple[int, int]] = []
            for start in range(span_cursor, len(units)):
                for end in range(start + 1, len(units) + 1):
                    span_pairs.append((" ".join(units[start:end]), claim))
                    span_labels.append((start, end))
            span_scores = self._scores(
                span_pairs, decision_types="semantic_claim_adjacent_span",
            ) if span_pairs else []
            if deterministic_full:
                # The complete ordered unit sequence is the clause-span proof
                # for an exact authoritative root proposition.
                span_candidates.append((span_cursor, len(units), full_score))
            for (start, end), score in zip(span_labels, span_scores):
                span_text = " ".join(units[start:end])
                if any(
                    _equal_canonical_proposition(span_text, proposition)
                    for proposition in (claim, quotes[claim_index])
                ) or (
                    float(score.entailment) >= _PLAN_ENTAILMENT_FLOOR
                    and float(score.contradiction) < _PLAN_CONTRADICTION_CEILING
                ):
                    span_candidates.append((start, end, score))
            if (
                (
                    not deterministic_full
                    and float(full_score.entailment) < _PLAN_ENTAILMENT_FLOOR
                )
                or not span_candidates
            ):
                best_unit_index = max(
                    range(len(units)),
                    key=lambda unit_index: (
                        float(matrix[claim_index][unit_index].entailment),
                        -float(matrix[claim_index][unit_index].contradiction),
                        -unit_index,
                    ),
                )
                best = matrix[claim_index][best_unit_index]
                return (
                    f"missing atomic claim {claim_index}: {claim!r}; "
                    f"best_output_unit={best_unit_index}, "
                    f"entailment={float(best.entailment):.3f}, "
                    f"contradiction={float(best.contradiction):.3f}; "
                    "missing_obligation=add only this claim's grounded atomic "
                    "content without changing already-grounded units"
                )
            selected_start, selected_end, _ = min(
                span_candidates, key=lambda item: (item[1], item[0], item[1] - item[0]),
            )
            span_cursor = selected_end

        derived_units = {
            int(item["unit_index"]) for item in ledger
            if item["kind"] in {
                "numeric_arithmetic_relation", "symbolic_equivalence",
            }
            and bool(item.get("grounds_entire_clause"))
        }
        for unit_index, unit in enumerate(units):
            if any(row[unit_index] for row in grounded) or unit_index in derived_units:
                continue
            best_claim_index = max(
                range(len(premises)),
                key=lambda claim_index: (
                    float(matrix[claim_index][unit_index].entailment),
                    -float(matrix[claim_index][unit_index].contradiction),
                    -claim_index,
                ),
            )
            best = matrix[best_claim_index][unit_index]
            return (
                f"substantive output unit {unit_index} is ungrounded: {unit!r}; "
                f"best_premise={best_claim_index}, "
                f"entailment={float(best.entailment):.3f}, "
                f"contradiction={float(best.contradiction):.3f}; "
                "missing_obligation=remove or ground only this substantive unit"
            )
        return None

    def _plan_claim_overlap_error(
        self, claims: List[Mapping[str, Any]],
    ) -> Optional[str]:
        """Reject non-atomic or pairwise redundant claims before realization."""
        statements = [_clean(item.get("claim")) for item in claims]
        for index, statement in enumerate(statements):
            if len(_sentences(statement)) != 1:
                return (
                    f"supported_claims[{index}] is overbroad; express one "
                    "atomic proposition only"
                )
        pairs: List[Tuple[str, str]] = []
        labels: List[Tuple[int, int]] = []
        for left in range(len(statements)):
            for right in range(left + 1, len(statements)):
                left_terms = set(re.findall(r"[a-z0-9]+", statements[left].lower()))
                right_terms = set(re.findall(r"[a-z0-9]+", statements[right].lower()))
                lexical_overlap = (
                    len(left_terms & right_terms)
                    / max(1, min(len(left_terms), len(right_terms)))
                )
                left_quote = _clean(claims[left].get("evidence_quote")).lower()
                right_quote = _clean(claims[right].get("evidence_quote")).lower()
                quote_overlap = bool(
                    left_quote and right_quote
                    and (left_quote in right_quote or right_quote in left_quote)
                )
                # NLI is the semantic decision, while this cheap candidate
                # filter prevents unrelated claims from being compared under
                # classifiers whose neutral calibration is intentionally
                # permissive.
                if lexical_overlap < 0.30 and not quote_overlap:
                    continue
                pairs.extend([
                    (statements[left], statements[right]),
                    (statements[right], statements[left]),
                ])
                labels.extend([(left, right), (left, right)])
        if not pairs:
            return None
        scores = self._scores(
            pairs, decision_types="plan_claim_pairwise_overlap",
        )
        for offset in range(0, len(scores), 2):
            forward, reverse = scores[offset:offset + 2]
            if (
                max(float(forward.contradiction), float(reverse.contradiction))
                >= _PLAN_CONTRADICTION_CEILING
            ):
                continue
            if (
                float(forward.entailment) >= _PLAN_ENTAILMENT_FLOOR
                or float(reverse.entailment) >= _PLAN_ENTAILMENT_FLOOR
            ):
                left, right = labels[offset]
                return (
                    f"supported_claims[{right}] overlaps semantically with "
                    f"supported_claims[{left}]; delete or replace only the "
                    "later duplicate and retain the frozen claim's exact "
                    "evidence_quote"
                )
        return None

    def _objective_contract_error(
        self, task: Any, focus: Mapping[str, Any], *, answer_bearing: bool = True,
    ) -> Optional[str]:
        """Enforce objective delivery at the correct instructional surface.

        Learner prompts must elicit the canonical action over its subject and
        conditions, but must not be forced to disclose answer-bearing content
        obligations. Completions/chosen answers are validated separately as
        answer-bearing realizations.
        """
        normalized = _clean(task).lower()
        bloom_verb = _clean(focus.get("bloom_verb")).lower()
        if not answer_bearing:
            if bloom_verb and bloom_verb not in normalized:
                alternatives = _ACTION_PARAPHRASES.get(
                    bloom_verb, frozenset({bloom_verb}),
                )
                if not any(
                    re.search(rf"\b{re.escape(candidate)}\b", normalized)
                    for candidate in alternatives
                ):
                    return (
                        "learner task omits canonical bloom_verb obligation: "
                        f"{bloom_verb!r}"
                    )
            for condition in focus.get("conditions") or []:
                expected = _clean(condition).lower()
                operational_error = _operational_condition_error(
                    normalized, expected,
                )
                if operational_error:
                    return operational_error
                if not expected or expected in normalized:
                    continue
                score = self._score_pair(
                    decision_type="objective_prompt_condition_delivery",
                    premise=normalized,
                    hypothesis=expected,
                )
                if (
                    float(score.entailment) >= _PLAN_ENTAILMENT_FLOOR
                    and float(score.contradiction) < _PLAN_CONTRADICTION_CEILING
                ):
                    continue
                return f"learner task omits canonical condition: {expected!r}"
            action = _clean(focus.get("action_object")).lower()
            relation_words = {
                form
                for relation in semantic_anchors(action)
                for form in semantic_anchor_forms(relation)
            }
            subject_terms = {
                token for token in re.findall(r"[a-z][a-z0-9'-]{2,}", action)
                if token not in relation_words
                and token not in {
                    "and", "for", "from", "given", "the", "their", "to",
                    "under", "using", "with",
                }
            }
            delivered_terms = set(re.findall(r"[a-z][a-z0-9'-]{2,}", normalized))
            if subject_terms and not (subject_terms & delivered_terms):
                subject_hypothesis = " ".join(sorted(subject_terms))
                score = self._score_pair(
                    decision_type="objective_task_delivery",
                    premise=normalized,
                    hypothesis=subject_hypothesis,
                )
                if not (
                    float(score.entailment) >= _PLAN_ENTAILMENT_FLOOR
                    and float(score.contradiction) < _PLAN_CONTRADICTION_CEILING
                ):
                    return "learner task omits canonical action subject"
            return None
        anchor_records = focus.get("content_obligation_anchors") or []
        required_anchors = list(dict.fromkeys(
            str(anchor.get("relation"))
            for record in anchor_records
            if isinstance(record, Mapping)
            for anchor in (record.get("anchors") or [])
            if isinstance(anchor, Mapping) and str(anchor.get("relation"))
        ))
        delivered_anchors = set(semantic_anchors(normalized))
        for anchor in required_anchors:
            if (
                anchor == "selection"
                and _clean(focus.get("bloom_verb")).lower()
                in semantic_anchor_forms("selection")
            ):
                continue
            if anchor in delivered_anchors:
                continue
            hypothesis = semantic_anchor_hypothesis(anchor)
            score = self._score_pair(
                decision_type="objective_task_anchor_delivery",
                premise=normalized,
                hypothesis=hypothesis,
            )
            if (
                float(score.entailment) >= _PLAN_ENTAILMENT_FLOOR
                and float(score.contradiction) < _PLAN_CONTRADICTION_CEILING
            ):
                continue
            return (
                "learner task omits canonical content_obligation anchor: "
                f"{anchor!r}"
            )
        ordered_relations = [
            item for item in required_anchors if item != "selection"
        ]
        delivered_order = [
            item for item in semantic_anchor_sequence(normalized)
            if item in ordered_relations
        ]
        if (
            len(ordered_relations) > 1
            and delivered_order != ordered_relations
        ):
            return "learner task reorders canonical content_obligation relations"

        # When a telegraphic obligation binds several relations to distinct
        # objects, preserve those bindings. Articles and descriptive modifiers
        # are immaterial; swapping the relation targets is not.
        stop = {
            "a", "an", "and", "equals", "equal", "is", "of", "the", "to",
            "whose", "using", "use",
        }
        for record in anchor_records:
            if not isinstance(record, Mapping):
                continue
            obligation = _clean(record.get("obligation")).lower()
            clauses = [
                part for part in re.split(r"\s*;\s*|\s*,\s*", obligation)
                if semantic_anchors(part)
            ]
            for clause in clauses:
                relations = semantic_anchors(clause)
                if len(relations) != 1:
                    continue
                relation = relations[0]
                if relation == "selection":
                    continue
                relation_forms = semantic_anchor_forms(relation)
                required_terms = [
                    token for token in re.findall(r"[a-z0-9]+", clause)
                    if token not in stop and token not in relation_forms
                ]
                if not required_terms:
                    continue
                match = re.search(
                    r"\b(?:" + "|".join(map(re.escape, relation_forms)) + r")\b",
                    normalized,
                )
                if match is None:
                    continue
                next_positions = [
                    candidate.start() for other in required_anchors
                    if other != relation
                    for form in semantic_anchor_forms(other)
                    if (candidate := re.search(
                        r"\b" + re.escape(form) + r"\b",
                        normalized[match.end():],
                    ))
                ]
                end = match.end() + min(next_positions) if next_positions else len(normalized)
                relation_span = normalized[match.start():end]
                if not all(
                    re.search(r"\b" + re.escape(term) + r"\b", relation_span)
                    for term in required_terms
                ):
                    return (
                        "learner task changes canonical content_obligation "
                        f"relation target: {relation!r}"
                    )
        dimensions = [
            ("bloom_verb", bloom_verb),
            *(
                ("content_obligation", _clean(item))
                for item in focus.get("content_obligations") or []
            ),
            *(
                ("condition", _clean(item))
                for item in focus.get("conditions") or []
            ),
        ]
        for field, raw_expected in dimensions:
            expected = raw_expected.lower()
            if not expected or expected in normalized:
                continue
            if (
                field == "content_obligation"
                and semantic_anchors(expected)
                and set(semantic_anchors(expected)) <= set(required_anchors)
            ):
                # Ordered anchors and their relation targets were verified
                # above; do not let a paraphrase NLI result contradict that
                # stronger deterministic contract.
                continue
            # The Bloom verb controls the requested cognitive operation and
            # must remain explicit. Other dimensions may be validly
            # paraphrased, so use the same NLI contract as claim coverage.
            if field != "bloom_verb":
                score = self._score_pair(
                    decision_type="objective_task_delivery",
                    premise=normalized,
                    hypothesis=expected,
                )
                if (
                    float(score.entailment) >= _PLAN_ENTAILMENT_FLOOR
                    and float(score.contradiction)
                    < _PLAN_CONTRADICTION_CEILING
                ):
                    continue
            return f"learner task omits canonical {field}: {expected!r}"
        return None

    def _capture_call(
        self,
        *,
        stage: str,
        chunk_id: str,
        attempt: int,
        prompt: str,
        raw_response: str,
        usage: Mapping[str, int],
        validation_error: Optional[str],
        context_headroom_tokens: int,
        max_output_tokens: Optional[int] = None,
        validation_evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._capture is None:
            return
        resolved_max_output_tokens = (
            int(max_output_tokens)
            if max_output_tokens is not None
            else int(getattr(self._base, "_max_tokens", 0))
        )
        prompt_hash = _sha(prompt)
        response_hash = _sha(raw_response)
        fresh_start_id = getattr(self._capture, "fresh_start_id", None)
        marker_digest = getattr(
            self._capture, "fresh_start_marker_digest", None,
        )
        prompt_path = self._persist_raw_artifact(
            kind="prompt", digest=prompt_hash, content=prompt,
        )
        response_path = self._persist_raw_artifact(
            kind="response", digest=response_hash, content=raw_response,
        )
        task_digest = _sha(
            f"{fresh_start_id or ''}|{chunk_id}|{stage}|{attempt}"
        )[:8]
        ledger = getattr(self, "_pilot_attempt_ledger", None)
        intent_manifest = getattr(ledger, "_intent_manifest", None)
        intent = (
            intent_manifest.current()
            if intent_manifest is not None else None
        )
        decision_identity = self._decision_identity_context()
        identity_sha256 = _sha(_stable_json(decision_identity))
        decisions = getattr(self._capture, "decisions", None)
        prior_decision_count = (
            len(decisions) if isinstance(decisions, list) else None
        )
        capture_result = self._capture.log_decision(
            decision_type="synthesis_provider_call",
            decision=(
                f"Completed {stage} attempt {attempt} for chunk {chunk_id}; "
                f"validation={'pass' if validation_error is None else validation_error}"
            ),
            rationale=(
                f"model={self._model}, stage={stage}, chunk_id={chunk_id}, "
                f"attempt={attempt}, max_tokens={resolved_max_output_tokens}, "
                f"prompt_sha256={prompt_hash}, response_sha256={response_hash}, "
                f"prompt_tokens={int(usage.get('prompt_tokens', 0))}, "
                f"completion_tokens={int(usage.get('completion_tokens', 0))}; "
                f"context_headroom_tokens={context_headroom_tokens}; "
                f"semantic_identity_sha256={identity_sha256}; "
                f"validation_error={validation_error or 'none'}; "
                "the hashes bind the exact raw request text and response bytes."
            ),
            context=_stable_json({
                **decision_identity,
                "semantic_identity_sha256": identity_sha256,
                "stage": stage,
                "chunk_id": chunk_id,
                "attempt": attempt,
                "prompt_sha256": prompt_hash,
                "response_sha256": response_hash,
                "validation_error": validation_error,
                "context_headroom_tokens": context_headroom_tokens,
                "max_output_tokens": resolved_max_output_tokens,
                "validation_evidence": dict(validation_evidence or {}),
                **({
                    "intent_run_id": intent["run_id"],
                    "intent_cell_id": intent["cell_id"],
                    "intent_unit": intent["unit"],
                    "intent_kind": intent["kind"],
                    "intent_stage": intent["stage"],
                    "intent_logical_attempt": intent["logical_attempt"],
                    "intent_request_sha256": intent["request_sha256"],
                    "intent_model": intent["model"],
                    "intent_contract_sha256": intent["contract_sha256"],
                } if intent is not None else {}),
                **({
                    "fresh_start_id": fresh_start_id,
                    "fresh_start_marker_digest": marker_digest,
                } if fresh_start_id is not None else {}),
            }),
            inputs_ref=[{
                "source_type": "prompt_template",
                "path_or_id": str(prompt_path),
                "content_hash": prompt_hash,
                "hash_algorithm": "sha256",
                "size_bytes": len(prompt.encode("utf-8")),
            }],
            prompt_ref=str(prompt_path),
            outputs=[{
                "artifact_type": "llm_response",
                "path": str(response_path),
                "content_hash": response_hash,
                "hash_algorithm": "sha256",
                "size_bytes": len(raw_response.encode("utf-8")),
            }],
            task_id=f"T-{task_digest}",
        )
        event_id = ""
        if isinstance(capture_result, Mapping):
            event_id = str(capture_result.get("event_id") or "")
        decisions = getattr(self._capture, "decisions", None)
        if (
            not event_id
            and isinstance(decisions, list)
            and prior_decision_count is not None
            and len(decisions) == prior_decision_count + 1
            and isinstance(decisions[-1], Mapping)
        ):
            event_id = str(decisions[-1].get("event_id") or "")
        self._validation_audit.decision_capture_event_id = event_id

    def _persist_raw_artifact(
        self, *, kind: str, digest: str, content: str,
    ) -> Path:
        """Persist raw bytes beside captures with owner-only permissions."""
        root_value = getattr(self._capture, "output_dir", None)
        if root_value is None:
            # Lightweight test captures have no durable storage surface.
            return Path(f"capture://staged/{kind}/{digest}")
        root = Path(root_value) / "raw_staged_calls"
        fresh_start_id = getattr(self._capture, "fresh_start_id", None)
        if fresh_start_id is not None:
            safe_identity = re.sub(r"[^A-Za-z0-9_.-]", "_", fresh_start_id)
            root = root / f"fresh-{safe_identity}"
        path = root / f"{kind}-{digest}.txt"
        payload = content.encode("utf-8")
        with _RAW_ARTIFACT_LOCK:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(root, 0o700)
            if not path.exists():
                with path.open("xb") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.chmod(path, 0o600)
        return path

    def _structured_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        schema: Mapping[str, Any],
        meter_task: str = "staged_synthesis",
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int], int]:
        """Dispatch strict JSON Schema using the base provider's transport."""
        injected = getattr(
            self._base, "_chat_completion_raw_structured", None,
        )
        if callable(injected):
            if max_output_tokens is None:
                # Preserve the legacy injected-provider call surface exactly.
                return injected(messages, schema=dict(schema))
            return injected(
                messages,
                schema=dict(schema),
                max_tokens=int(max_output_tokens),
            )
        oa_client = getattr(self._base, "_oa_client", None)
        if oa_client is None:
            raise SynthesisProviderError(
                "staged synthesis requires structured JSON transport",
                code="staged_structured_transport_unavailable",
            )
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": float(getattr(self._base, "_temperature", 0.0)),
            "max_tokens": (
                int(max_output_tokens)
                if max_output_tokens is not None
                else int(getattr(self._base, "_max_tokens", 0))
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ed4all_staged_synthesis",
                    "strict": True,
                    "schema": dict(schema),
                },
            },
        }
        from Trainforge.generators._openai_compatible_client import (
            apply_reasoning_thinking_off_payload,
        )
        apply_reasoning_thinking_off_payload(payload, force_thinking_off=True)
        body, retries = oa_client.post_with_usage(payload, task=meter_task)
        message = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
        reasoning_content = str(
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        ).strip()
        if reasoning_content:
            raise SynthesisProviderError(
                "structured synthesis emitted reasoning despite thinking-off",
                code="staged_thinking_not_disabled",
                details={
                    "reasoning_sha256": _sha(reasoning_content),
                    "reasoning_bytes": len(reasoning_content.encode("utf-8")),
                },
            )
        finish_reason = str(
            ((body.get("choices") or [{}])[0] or {}).get("finish_reason") or ""
        ).strip().lower()
        if finish_reason in {"length", "max_tokens"}:
            raise SynthesisProviderError(
                "structured synthesis response reached its output cap",
                code="staged_output_truncated",
            )
        return (
            oa_client._extract_text(body),
            oa_client._extract_usage(body),
            retries,
        )

    def _call_stage(
        self,
        *,
        stage: str,
        chunk_id: str,
        system: str,
        user: str,
        required_keys: Iterable[str],
        validator: Callable[[Dict[str, Any]], Optional[str]],
        leakage_source: str = "",
        leakage_keys: Iterable[str] = (),
        leakage_contract_text: Iterable[str] = (),
        leakage_canonical_contract_text: Iterable[str] = (),
        leakage_nondistinctive_contexts: Iterable[Mapping[str, Any]] = (),
        response_schema: Optional[Mapping[str, Any]] = None,
        max_stage_repairs: Optional[int] = None,
        max_leakage_repairs: int = _MAX_LEAKAGE_REWRITES,
        max_output_tokens: Optional[int] = None,
    ) -> _StageResult:
        # None means "no caller opinion" -> take the env-resolved budget. An
        # explicit caller value still wins, and still has to pass the same
        # nonnegative-integer contract below.
        if max_stage_repairs is None:
            max_stage_repairs = resolve_max_stage_repairs()
        for budget_name, budget in (
            ("max_stage_repairs", max_stage_repairs),
            ("max_leakage_repairs", max_leakage_repairs),
        ):
            if (
                isinstance(budget, bool)
                or not isinstance(budget, int)
                or budget < 0
            ):
                raise SynthesisProviderError(
                    f"{budget_name} must be an explicit nonnegative integer",
                    code="staged_repair_budget_invalid",
                )
        required_keys = tuple(required_keys)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        repairs = 0
        leakage_repairs = 0
        aggregate = {"prompt_tokens": 0, "completion_tokens": 0}
        claim_repair_guard: Optional[Dict[str, Any]] = None
        response_progress: List[str] = []
        while True:
            rendered_prompt = _stable_json(messages)
            if self._verified_served_context_tokens is not None:
                served_window = self._verified_served_context_tokens
            else:
                raw_window = str(
                    os.environ.get(ENV_SERVED_CONTEXT_TOKENS, "")
                ).strip()
                try:
                    served_window = int(raw_window)
                except (TypeError, ValueError):
                    served_window = 0
            if served_window <= 0:
                raise SynthesisProviderError(
                    f"{ENV_SERVED_CONTEXT_TOKENS} must declare the measured "
                    "served model window for staged synthesis",
                    code="staged_context_window_unverified",
                )
            # UTF-8 byte count is a conservative tokenizer-independent upper
            # bound for ordinary text token count.  It may reject a prompt
            # early, but can never prove unsafe text safe.
            prompt_token_upper_bound = len(rendered_prompt.encode("utf-8"))
            resolved_max_output_tokens = (
                int(max_output_tokens)
                if max_output_tokens is not None
                else int(getattr(self._base, "_max_tokens", 0))
            )
            context_headroom = (
                served_window - prompt_token_upper_bound
                - resolved_max_output_tokens
            )
            if resolved_max_output_tokens <= 0 or context_headroom < 0:
                raise SynthesisProviderError(
                    "staged prompt plus maximum output exceeds the measured "
                    f"served window: prompt_upper_bound={prompt_token_upper_bound}, "
                    f"max_output={resolved_max_output_tokens}, window={served_window}",
                    code="staged_context_window_exceeded",
                )
            try:
                raw, usage, _http_retries = self._structured_completion(
                    messages,
                    schema=dict(response_schema or {}),
                    meter_task=f"staged_synthesis:{stage}",
                    max_output_tokens=resolved_max_output_tokens,
                )
            except Exception as exc:
                if (
                    self._capture is not None
                    and not getattr(exc, "_direct_capture_emitted", False)
                ):
                    error_code = str(
                        getattr(exc, "code", "") or type(exc).__name__
                    )
                    self._capture_call(
                        stage=stage,
                        chunk_id=chunk_id,
                        attempt=repairs + leakage_repairs + 1,
                        prompt=rendered_prompt,
                        raw_response="",
                        usage={},
                        validation_error=error_code,
                        context_headroom_tokens=context_headroom,
                        max_output_tokens=resolved_max_output_tokens,
                        validation_evidence={
                            "response_schema_sha256": _sha(
                                _stable_json(response_schema or {})
                            ),
                            "required_keys": list(required_keys),
                            "validator_stage": stage,
                            "validator_attempt": repairs + leakage_repairs + 1,
                            "validator_reason": error_code,
                            "nli_scores": [],
                        },
                    )
                raise
            for key in aggregate:
                aggregate[key] += int(usage.get(key, 0))
            parsed = self._base._oa_client._extract_json_lenient(raw)
            self._validation_audit.nli = []
            self._validation_audit.derivations = []
            self._validation_audit.invalid_claim_indices = []
            self._validation_audit.editable_json_pointers = []
            self._validation_audit.stage = stage
            self._validation_audit.attempt = repairs + leakage_repairs + 1
            error: Optional[str] = None
            repair_diff = (
                _claim_repair_diff(parsed, claim_repair_guard)
                if claim_repair_guard is not None else None
            )
            if repair_diff is not None:
                error = (
                    "claim repair immutability violation: "
                    f"{repair_diff['reason']}"
                )
            elif not isinstance(parsed, dict):
                error = "response was not a JSON object"
            else:
                missing = [key for key in required_keys if key not in parsed]
                if missing:
                    error = f"missing required keys: {','.join(missing)}"
                else:
                    error = validator(parsed)
                    if error is None and leakage_source:
                        for key in leakage_keys:
                            span = _first_distinctive_verbatim_span(
                                _clean(parsed.get(key)),
                                leakage_source,
                                mandatory_contract_text=leakage_contract_text,
                                canonical_contract_text=(
                                    leakage_canonical_contract_text
                                ),
                                nondistinctive_contexts=(
                                    leakage_nondistinctive_contexts
                                ),
                            )
                            if span:
                                error = (
                                    f"{key} copied a {_VERBATIM_SPAN}-character "
                                    f"source span sha256={_sha(span)}"
                                )
                                break
            response_sha = _sha(
                _stable_json(parsed) if isinstance(parsed, dict) else _clean(raw)
            )
            validator_error = error
            progress_cycle = (
                error is not None and (
                    response_sha in response_progress[-1:]
                    or (
                        len(response_progress) >= 2
                        and response_sha == response_progress[-2]
                    )
                )
            )
            response_progress.append(response_sha)
            if progress_cycle:
                error = (
                    "repair made no canonical progress: identical response or "
                    f"two-cycle detected; response_sha256={response_sha}"
                )
            evidence = {
                "response_schema_sha256": _sha(
                    _stable_json(response_schema or {})
                ),
                "required_keys": list(required_keys),
                "validator_stage": stage,
                "validator_attempt": repairs + leakage_repairs + 1,
                "validator_reason": error,
                "nli_scores": list(
                    getattr(self._validation_audit, "nli", []) or []
                ),
                "derivation_ledger": list(
                    getattr(self._validation_audit, "derivations", []) or []
                ),
                "present_keys": sorted(parsed) if isinstance(parsed, dict) else [],
                "repair_immutability": repair_diff,
                "invalid_claim_indices": list(
                    getattr(self._validation_audit, "invalid_claim_indices", [])
                    or []
                ),
                "invalid_claim_stable_ids": [
                    _claim_stable_id(parsed["supported_claims"][index])
                    for index in (
                        getattr(self._validation_audit, "invalid_claim_indices", [])
                        or []
                    )
                    if (
                        isinstance(parsed, dict)
                        and isinstance(parsed.get("supported_claims"), list)
                        and 0 <= index < len(parsed["supported_claims"])
                    )
                ],
                "editable_json_pointers": list(
                    getattr(self._validation_audit, "editable_json_pointers", [])
                    or []
                ),
                "canonical_response_sha256": response_sha,
            }
            self._capture_call(
                stage=stage,
                chunk_id=chunk_id,
                attempt=repairs + leakage_repairs + 1,
                prompt=rendered_prompt,
                raw_response=raw,
                usage=usage,
                validation_error=error,
                context_headroom_tokens=context_headroom,
                max_output_tokens=resolved_max_output_tokens,
                validation_evidence=evidence,
            )
            if error is None and isinstance(parsed, dict):
                return _StageResult(parsed, aggregate, repairs + leakage_repairs + 1)
            if progress_cycle:
                raise SynthesisProviderError(
                    f"{stage} exhausted validator-specific repairs early: "
                    f"{validator_error}; {error}",
                    code=f"staged_{stage}_invalid",
                    chunk_id=chunk_id,
                    details={
                        "terminal_content_rejection": True,
                        "stage": stage,
                        "validation_error": validator_error,
                        "prompt_ref": f"sha256:{_sha(rendered_prompt)}",
                        "response_ref": f"sha256:{response_sha}",
                    },
                )
            is_leakage = bool(error and "source span sha256=" in error)
            if is_leakage:
                if leakage_repairs >= max_leakage_repairs:
                    raise SynthesisProviderError(
                        f"{stage} exhausted two source-free leakage rewrites: {error}",
                        code="provider_output_verbatim_leakage",
                        chunk_id=chunk_id,
                        details={
                            "terminal_content_rejection": True,
                            "stage": stage,
                            "validation_error": error,
                            "prompt_ref": f"sha256:{_sha(rendered_prompt)}",
                            "response_ref": f"sha256:{_sha(raw)}",
                            **({"repair_immutability": repair_diff}
                               if repair_diff is not None else {}),
                        },
                    )
                leakage_repairs += 1
            else:
                if repairs >= max_stage_repairs:
                    raise SynthesisProviderError(
                        f"{stage} exhausted validator-specific repairs: {error}",
                        code=f"staged_{stage}_invalid",
                        chunk_id=chunk_id,
                        details={
                            "terminal_content_rejection": True,
                            "stage": stage,
                            "validation_error": error,
                            "prompt_ref": f"sha256:{_sha(rendered_prompt)}",
                            "response_ref": f"sha256:{_sha(raw)}",
                            **({"repair_immutability": repair_diff}
                               if repair_diff is not None else {}),
                        },
                    )
                repairs += 1
            repair_instruction = (
                f"VALIDATION FAILURE: {error}. Correct only this exact "
                "failure. Preserve all already-valid fields. Return the "
                "same strict JSON shape, without commentary."
            )
            pointer_matches = sorted(set(re.findall(
                r"(?<![A-Za-z0-9_])(/[A-Za-z0-9_./-]+)", str(error or ""),
            )))
            if is_leakage and not pointer_matches:
                legacy_leakage_field = re.match(
                    r"([A-Za-z_][A-Za-z0-9_]*) copied (?:a \d+-character )?"
                    r"source span sha256=",
                    str(error or ""),
                )
                if legacy_leakage_field:
                    pointer_matches = [f"/{legacy_leakage_field.group(1)}"]
            if pointer_matches:
                repair_instruction = (
                    f"VALIDATION FAILURE: {error}. Edit all and only these JSON "
                    f"pointers: {pointer_matches}. Preserve every other field "
                    "byte-for-byte in canonical JSON. Return the same strict "
                    "JSON shape without commentary."
                )
            invalid_indices = list(
                getattr(self._validation_audit, "invalid_claim_indices", []) or []
            )
            editable_pointers = list(
                getattr(self._validation_audit, "editable_json_pointers", []) or []
            )
            if not invalid_indices:
                invalid_indices = [
                    int(item) for item in re.findall(
                        r"supported_claims\[(\d+)\]", str(error or ""),
                    )
                ]
            if (
                (invalid_indices or editable_pointers)
                and isinstance(parsed, dict)
                and isinstance(parsed.get("supported_claims"), list)
            ):
                if all(
                    0 <= index < len(parsed["supported_claims"])
                    for index in invalid_indices
                ):
                    claim_repair_guard = _claim_repair_guard(
                        parsed, invalid_indices,
                        editable_pointers=editable_pointers,
                    )
            if invalid_indices and claim_repair_guard is not None:
                invalid_ids = claim_repair_guard["invalid_stable_ids"]
                frozen = [
                    claim for index, claim
                    in enumerate(parsed["supported_claims"])
                    if index not in invalid_indices
                ]
                repair_instruction = (
                    f"VALIDATION FAILURE: {error}. The only editable claim "
                    f"stable IDs are {invalid_ids}; the only editable plan "
                    f"element is supported_claims[{invalid_indices[0]}] when "
                    f"one ID is listed. Delete or replace all and only those "
                    "claims with atomic, directly evidenced claims. Repair "
                    "dependent fields only when their JSON pointers are also "
                    "listed. Do not edit, reorder, merge, or paraphrase any "
                    "other claim or non-permitted field. EDITABLE_JSON_POINTERS="
                    f"{claim_repair_guard['editable_json_pointers']}. "
                    "FROZEN_VALID_CLAIMS_SHA256="
                    f"{_sha(_stable_json(frozen))}. Return the same strict "
                    "JSON shape without commentary."
                )
            elif editable_pointers and claim_repair_guard is not None:
                repair_instruction = (
                    f"VALIDATION FAILURE: {error}. Edit all and only these JSON "
                    f"pointers: {editable_pointers}. Preserve every claim and "
                    "other field byte-for-byte in canonical JSON. The learner "
                    "task and typed generated givens may be repaired jointly. "
                    f"BASELINE_SHA256={claim_repair_guard['baseline_sha256']}. "
                    "Return the same strict JSON shape without commentary."
                )
            if repair_diff is not None:
                repair_instruction = (
                    f"VALIDATION FAILURE: {error}. Restore every frozen claim "
                    "and every noneditable field byte-for-byte in canonical "
                    "JSON. Only replace/delete the named invalid stable IDs. "
                    f"BASELINE_SHA256={repair_diff['baseline_sha256']} "
                    f"REJECTED_CANDIDATE_SHA256={repair_diff['candidate_sha256']}. "
                    "Return the same strict JSON shape without commentary."
                )
            if error and "missing atomic claim " in error:
                repair_instruction = (
                    f"VALIDATION FAILURE: {error}. Edit all and only "
                    f"{pointer_matches}. Make the named answer field explicitly "
                    "entail the exact missing immutable claim text/index/ID. "
                    "Preserve /prompt and all already-grounded answer content. "
                    "The corrected canonical JSON must differ from the rejected "
                    "candidate; never reissue identical canonical output. "
                    "Return the same strict JSON shape without commentary."
                )
            if error and "substantive output unit " in error:
                repair_instruction = (
                    f"VALIDATION FAILURE: {error}. Edit all and only "
                    f"{pointer_matches}. Repair or remove only the named "
                    "genuinely unsupported substantive clause. Preserve "
                    "/prompt, every deterministically proven immutable claim, "
                    "and all other grounded answer content byte-for-byte. "
                    "Never delete an immutable claim to repair an unrelated "
                    "clause. The corrected canonical JSON must differ from the "
                    "rejected candidate; never reissue identical canonical "
                    "output. Return the same strict JSON shape without "
                    "commentary."
                )
            if is_leakage:
                field_labels = [
                    pointer.rsplit("/", 1)[-1] for pointer in pointer_matches
                ]
                field_description = (
                    f"field {field_labels[0]!r} at "
                    if len(field_labels) == 1 else "fields at "
                )
                paraphrase_scope = (
                    f"only {field_labels[0]!r}"
                    if len(field_labels) == 1 else "all affected fields"
                )
                repair_instruction = (
                    "VALIDATION FAILURE: category=private_source_overlap in "
                    f"{field_description}JSON pointers {pointer_matches}. "
                    f"Completely paraphrase {paraphrase_scope}; "
                    "do not make a token-level "
                    "edit and do not reproduce the leaked span. Preserve every "
                    "other field byte-for-byte. The corrected canonical JSON "
                    "must differ from the rejected candidate; never reissue "
                    "identical canonical output. Return the same strict JSON "
                    "shape without commentary."
                )
            if error and "illegal numeric literals=" in error:
                if stage == "micro_A_task_design":
                    repair_instruction = (
                        f"NUMERIC FAILURE: {error}. EDITABLE=['/learner_task']. "
                        "Remove or rephrase each prohibited literal or "
                        "answer-bearing result in /learner_task. Generated-given "
                        "metadata is orchestrator-owned and must not be returned. "
                        "Preserve objective_id, bloom_level, objective meaning, "
                        "and every allowed numeric. Return only corrected compact "
                        "JSON; never reissue the rejected canonical JSON."
                    )
                else:
                    repair_instruction = (
                        f"VALIDATION FAILURE: {error}. Edit all and only the "
                        f"affected JSON pointers {pointer_matches}. Remove, "
                        "rephrase, or replace ONLY the listed illegal numeric "
                        "literals using the explicit allowed inventory. Preserve "
                        "the immutable claims and learner-task semantics and every "
                        "unaffected field. The corrected canonical JSON must differ "
                        "from the rejected candidate; never reissue the same "
                        "canonical output. Return the same strict JSON shape "
                        "without commentary."
                    )
            compact_prior = (
                _stable_json(parsed) if isinstance(parsed, dict) else raw
            )
            messages = list(messages) + [
                {"role": "assistant", "content": compact_prior},
                {"role": "user", "content": repair_instruction},
            ]

    def _plan(self, chunk: Mapping[str, Any], *, need_misconception: bool) -> Dict[str, Any]:
        focus = self._focus(chunk)
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        source = _clean(chunk.get("text"))
        try:
            window = build_evidence_window(chunk, focus)
        except ValueError as exc:
            raise SynthesisProviderError(
                str(exc),
                code="staged_evidence_window_unavailable",
                chunk_id=chunk_id,
                details={
                    "terminal_content_rejection": True,
                    "stage": "plan_dpo" if need_misconception else "plan_sft",
                },
            ) from exc
        rendered_window = render_evidence_window(window)
        misconception_rule = (
            "misconception_affordance must contain error_mechanism, faulty_step, "
            "and causal_rationale for one error enabled by incorrect evidence."
            if need_misconception
            else (
                "misconception_affordance must contain all three keys with "
                "empty-string values."
            )
        )
        user = (
            f"WINDOW_CONTRACT: {WINDOW_CONTRACT_VERSION}\n"
            f"OBJECTIVE_CARD: {_stable_json(focus)}\n"
            f"EVIDENCE_WINDOW:\n{rendered_window}\n\n"
            "Extract one to five independently supported claims. Each claim "
            "must include an exact evidence_quote copied from factual or "
            "corrective evidence. Define a "
            "learner_task that instantiates the canonical cognitive skill on "
            "only those claims; do not copy the objective literally. If the "
            "task needs invented numeric facts, declare each in generated_givens "
            "with symbol, decimal value, unit, role, synthetic=true, and "
            "provenance=generated; otherwise return an empty array. "
            f"{misconception_rule}\n"
            'Return only {"objective_id":"...","bloom_level":"...",'
            '"supported_claims":[{"claim":"...","evidence_quote":"..."}],'
            '"generated_givens":[],'
            '"learner_task":"...","misconception_affordance":'
            '{"error_mechanism":"","faulty_step":"","causal_rationale":""}}.'
        )

        def validate(value: Dict[str, Any]) -> Optional[str]:
            failures: List[str] = []
            if _clean(value.get("objective_id")).lower() != focus["id"]:
                failures.append("objective_id differs from canonical objective")
                self._validation_audit.editable_json_pointers.append(
                    "/objective_id"
                )
            if _clean(value.get("bloom_level")).lower() != focus["bloom_level"]:
                failures.append("bloom_level differs from canonical Bloom")
                self._validation_audit.editable_json_pointers.append(
                    "/bloom_level"
                )
            claims = value.get("supported_claims")
            # PACKAGE A: Raise supported_claims cap from 3 to 5 so single
            # unverifiable sentence doesn't become fatal (with only 3 claims,
            # one bad sentence = 0.33 unsupported_claim_rate, fatal at 0.20 ceiling)
            if not isinstance(claims, list) or not 1 <= len(claims) <= 5:
                return "supported_claims must contain one to five claims"
            invalid_claims: Dict[int, str] = {}
            for index, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    invalid_claims[index] = "is not an object"
                    continue
                statement = _clean(claim.get("claim"))
                quote = _clean(claim.get("evidence_quote"))
                if not statement or not quote:
                    invalid_claims[index] = "lacks claim/evidence_quote"
                    continue
                located = canonical_quote(quote, window)
                if located is None:
                    invalid_claims[index] = "evidence_quote is not in source"
                    continue
                exact_quote, block = located
                if block["polarity"] not in {"factual", "corrective"}:
                    invalid_claims[index] = (
                        f"evidence_quote uses {block['polarity']} evidence"
                    )
                    continue
                claim["evidence_quote"] = exact_quote
                deterministic_basis = None
                if _symbolic_math_supports(exact_quote, statement):
                    deterministic_basis = "plan_symbolic_equivalence"
                elif _relation_preserving_support(exact_quote, statement):
                    deterministic_basis = "plan_relation_signature"
                if deterministic_basis is not None:
                    class _SymbolicScore:
                        entailment = 1.0
                        contradiction = 0.0
                    score = _SymbolicScore()
                    self._record_nli(
                        decision_type=deterministic_basis,
                        premise=exact_quote,
                        hypothesis=statement,
                        score=score,
                    )
                else:
                    # The exact canonical quote is the authoritative premise.
                    # A containing local block is used only when the quote alone
                    # cannot resolve the relationship; never score the corpus.
                    score = self._score_pair(
                        decision_type="plan_quote_claim",
                        premise=exact_quote,
                        hypothesis=statement,
                    )
                    if (
                        float(score.entailment) < _PLAN_ENTAILMENT_FLOOR
                        and float(score.contradiction)
                        < _PLAN_CONTRADICTION_CEILING
                        and _clean(block["text"]) != exact_quote
                    ):
                        score = self._score_pair(
                            decision_type="plan_quote_claim_context",
                            premise=str(block["text"]),
                            hypothesis=statement,
                        )
                if (
                    float(score.contradiction) >= _PLAN_CONTRADICTION_CEILING
                    and _explicit_relation_conflict(exact_quote, statement)
                ):
                    invalid_claims[index] = (
                        "contradicts its evidence "
                        f"(contradiction={float(score.contradiction):.3f})"
                    )
                elif float(score.entailment) < _PLAN_ENTAILMENT_FLOOR:
                    invalid_claims[index] = (
                        "is not entailed by its evidence "
                        f"(entailment={float(score.entailment):.3f})"
                    )
            seen_claims: Dict[Tuple[str, ...], int] = {}
            for index, claim in enumerate(claims):
                if not isinstance(claim, Mapping):
                    continue
                identity = _normalized_proposition(claim.get("claim"))
                if identity in seen_claims:
                    invalid_claims[index] = (
                        f"duplicates supported_claims[{seen_claims[identity]}]"
                    )
                elif identity:
                    seen_claims[identity] = index
            if invalid_claims:
                self._validation_audit.invalid_claim_indices = sorted(invalid_claims)
                stable = {
                    _claim_stable_id(claims[index]): reason
                    for index, reason in invalid_claims.items()
                }
                failures.append(
                    "invalid supported claim stable IDs: "
                    f"{_stable_json(stable)}"
                )
            overlap_error = (
                self._plan_claim_overlap_error(claims)
                if not invalid_claims else None
            )
            if overlap_error:
                overlap_indices = [
                    int(item) for item in re.findall(
                        r"supported_claims\[(\d+)\]", overlap_error
                    )
                ]
                self._validation_audit.invalid_claim_indices = overlap_indices[:1]
                failures.append(overlap_error)
            givens = value.get("generated_givens", [])
            if not isinstance(givens, list) or len(givens) > 6:
                failures.append(
                    "generated_givens must be an array with at most six items"
                )
                self._validation_audit.editable_json_pointers.append(
                    "/generated_givens"
                )
                givens = []
            seen_symbols: set[str] = set()
            for index, given in enumerate(givens):
                if not isinstance(given, dict):
                    failures.append(
                        f"generated_givens[{index}] is not an object"
                    )
                    self._validation_audit.editable_json_pointers.append(
                        "/generated_givens"
                    )
                    continue
                symbol = _clean(given.get("symbol"))
                value_text = _clean(given.get("value"))
                unit = _clean(given.get("unit"))
                role = _clean(given.get("role"))
                if (
                    not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", symbol)
                    or symbol in seen_symbols
                    or not re.fullmatch(r"-?(?:\d+(?:\.\d+)?|\.\d+)", value_text)
                    or not unit
                    or role not in {
                        "rate", "distance", "time", "count", "constant", "other",
                    }
                    or given.get("synthetic") is not True
                    or given.get("provenance") != "generated"
                ):
                    failures.append(
                        f"generated_givens[{index}] violates the typed synthetic "
                        "provenance contract"
                    )
                    self._validation_audit.editable_json_pointers.append(
                        "/generated_givens"
                    )
                seen_symbols.add(symbol)
            if not _clean(value.get("learner_task")):
                failures.append("learner_task is empty")
                self._validation_audit.editable_json_pointers.append(
                    "/learner_task"
                )
            allowed_numbers = {
                item
                for claim in claims if isinstance(claim, Mapping)
                for item in _scenario_numeric_literals(
                    _clean(claim.get("evidence_quote"))
                )
            } | {
                _number_key(str(given.get("value")))
                for given in givens if isinstance(given, Mapping)
            } | {
                item
                for dimension in (
                    focus.get("statement"),
                    focus.get("action_object"),
                    focus.get("condition"),
                    *(focus.get("content_obligations") or []),
                    *(focus.get("conditions") or []),
                )
                for item in _scenario_numeric_literals(dimension)
            }
            task_numbers = _scenario_numeric_literals(value.get("learner_task"))
            untyped = sorted(task_numbers - allowed_numbers)
            if untyped:
                self._validation_audit.editable_json_pointers = [
                    *self._validation_audit.editable_json_pointers,
                    "/learner_task", "/generated_givens",
                ]
                failures.append(
                    "learner_task introduces numeric values absent from source "
                    f"facts and typed generated_givens: {untyped!r}"
                )
            objective_error = self._objective_contract_error(
                value.get("learner_task"), focus, answer_bearing=False,
            )
            if objective_error:
                failures.append(objective_error)
                self._validation_audit.editable_json_pointers.append(
                    "/learner_task"
                )
            affordance = value.get("misconception_affordance")
            if not isinstance(affordance, dict):
                failures.append("misconception_affordance must be an object")
                self._validation_audit.editable_json_pointers.append(
                    "/misconception_affordance"
                )
                affordance = {}
            fields = ("error_mechanism", "faulty_step", "causal_rationale")
            populated = [_clean(affordance.get(field)) for field in fields]
            if need_misconception and not all(populated):
                failures.append(
                    "misconception_affordance lacks structured causal fields"
                )
                self._validation_audit.editable_json_pointers.append(
                    "/misconception_affordance"
                )
            if not need_misconception and any(populated):
                failures.append("misconception_affordance must be empty for SFT")
                self._validation_audit.editable_json_pointers.append(
                    "/misconception_affordance"
                )
            self._validation_audit.editable_json_pointers = sorted(set(
                self._validation_audit.editable_json_pointers
            ))
            return (
                "validation failures: " + _stable_json(failures)
                if failures else None
            )

        result = self._call_stage(
            stage="plan_dpo" if need_misconception else "plan_sft",
            chunk_id=chunk_id,
            system=_PLAN_SYSTEM,
            user=user,
            required_keys=(
                "objective_id", "bloom_level", "supported_claims",
                "learner_task", "misconception_affordance",
            ),
            validator=validate,
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "objective_id", "bloom_level", "supported_claims",
                    "generated_givens", "learner_task", "misconception_affordance",
                ],
                "properties": {
                    "objective_id": {"type": "string"},
                    "bloom_level": {"type": "string", "enum": sorted(_BLOOM)},
                    # PACKAGE A: Raise maxItems from 3 to 5 to give downstream
                    # answer-sentence verification headroom (3 claims + 1 bad sentence
                    # = fatal 0.33 rate at 0.20 ceiling; 5 claims gives 0.20 exact)
                    "supported_claims": {
                        "type": "array", "minItems": 1, "maxItems": 5,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["claim", "evidence_quote"],
                            "properties": {
                                "claim": {"type": "string", "minLength": 1, "maxLength": 300},
                                "evidence_quote": {"type": "string", "minLength": 1, "maxLength": 500},
                            },
                        },
                    },
                    "generated_givens": {
                        "type": "array", "maxItems": 6,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": [
                                "symbol", "value", "unit", "role",
                                "synthetic", "provenance",
                            ],
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "pattern": "^[A-Za-z][A-Za-z0-9_]{0,31}$",
                                },
                                "value": {
                                    "type": "string",
                                    "pattern": "^-?(?:[0-9]+(?:\\\\.[0-9]+)?|\\\\.[0-9]+)$",
                                },
                                "unit": {"type": "string", "minLength": 1, "maxLength": 40},
                                "role": {
                                    "type": "string",
                                    "enum": [
                                        "rate", "distance", "time", "count",
                                        "constant", "other",
                                    ],
                                },
                                "synthetic": {"const": True},
                                "provenance": {"const": "generated"},
                            },
                        },
                    },
                    "learner_task": {"type": "string", "minLength": 1, "maxLength": 400},
                    "misconception_affordance": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["error_mechanism", "faulty_step", "causal_rationale"],
                        "properties": {
                            "error_mechanism": {"type": "string", "maxLength": 240},
                            "faulty_step": {"type": "string", "maxLength": 240},
                            "causal_rationale": {"type": "string", "maxLength": 320},
                        },
                    },
                },
            },
        ).value
        result["_objective_card"] = focus
        return result

    @staticmethod
    def _plan_context(plan: Mapping[str, Any], *, include_evidence: bool = False) -> str:
        claims = []
        for index, item in enumerate(plan["supported_claims"]):
            claim = {"claim_index": index, "claim": item["claim"]}
            if include_evidence:
                claim["evidence_quote"] = item["evidence_quote"]
            claims.append(claim)
        return _stable_json({
            "objective_id": plan["objective_id"],
            "bloom_level": plan["bloom_level"],
            "supported_claims": claims,
            "generated_givens": plan.get("generated_givens") or [],
            "learner_task": plan["learner_task"],
            "misconception_affordance": plan["misconception_affordance"],
            "objective_contract": plan.get("_objective_card") or {},
        })

    @staticmethod
    def _coverage_error(value: Mapping[str, Any], plan: Mapping[str, Any]) -> Optional[str]:
        expected = list(range(len(plan["supported_claims"])))
        actual = value.get("covered_claim_indices")
        if actual != expected:
            return (
                "covered_claim_indices must exactly cover every planned claim "
                f"in order: expected {expected}, got {actual!r}"
            )
        return None

    @staticmethod
    def _generated_givens_prompt_error(
        prompt: Any, plan: Mapping[str, Any],
    ) -> Optional[str]:
        normalized = _clean(prompt).lower()
        for index, given in enumerate(plan.get("generated_givens") or []):
            required = (
                _clean(given.get("symbol")).lower(),
                _clean(given.get("value")).lower(),
                _clean(given.get("unit")).lower(),
            )
            if any(item not in normalized for item in required):
                return (
                    f"prompt omits generated_givens[{index}] typed "
                    "symbol/value/unit"
                )
        return None

    def paraphrase_instruction(
        self, draft: Dict[str, Any], chunk: Dict[str, Any],
        *, preserve_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        plan = self._plan(chunk, need_misconception=False)
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        source = _clean(chunk.get("text"))
        expected = list(range(len(plan["supported_claims"])))
        user = (
            f"VALIDATED_PLAN:\n{self._plan_context(plan)}\n\n"
            "Realize the learner_task as one natural instructional prompt and "
            "answer it using only supported_claims. Paraphrase evidence; never "
            "copy a 50-character source span. Return only "
            '{"prompt":"...","completion":"...","covered_claim_indices":'
            f"{_stable_json(expected)}}}."
        )

        def validate(value: Dict[str, Any]) -> Optional[str]:
            prompt = _clean(value.get("prompt"))
            completion = _clean(value.get("completion"))
            failures: List[str] = []
            if not prompt or len(prompt) > 400:
                failures.append(f"prompt length {len(prompt)} is outside 1..400")
            # PACKAGE A: Raise completion cap from 600 to 1200 chars to accept
            # deeper worked answers that grounded generators properly realize
            if not completion or len(completion) > 1200:
                failures.append(
                    f"completion length {len(completion)} is outside 1..1200"
                )
            checks = (
                self._objective_contract_error(
                    prompt, plan.get("_objective_card") or {},
                    answer_bearing=False,
                ),
                self._generated_givens_prompt_error(prompt, plan),
                self._coverage_error(value, plan),
                self._semantic_coverage_error(completion, plan, prompt=prompt),
            )
            failures.extend(error for error in checks if error)
            return (
                "validation failures: " + _stable_json(failures)
                if failures else None
            )

        realized = self._call_stage(
            stage="sft_realization", chunk_id=chunk_id,
            system=_REALIZATION_SYSTEM,
            user=user,
            required_keys=("prompt", "completion", "covered_claim_indices"),
            validator=validate,
            leakage_source=source, leakage_keys=("prompt", "completion"),
            leakage_contract_text=(
                *((plan.get("_objective_card") or {}).get(
                    "content_obligations"
                ) or []),
            ),
            leakage_canonical_contract_text=(
                (plan.get("_objective_card") or {}).get("action_object") or "",
                *((plan.get("_objective_card") or {}).get("conditions") or []),
            ),
            leakage_nondistinctive_contexts=(
                (plan.get("_objective_card") or {}).get(
                    "nondistinctive_contexts"
                ) or []
            ),
            response_schema=self._realization_schema(
                ("prompt", "completion"), expected,
            ),
        ).value
        out = dict(draft)
        out.update({
            "prompt": _clean(realized["prompt"]),
            "completion": _clean(realized["completion"]),
            "observed_bloom": plan["bloom_level"],
            "objective_contract": dict(plan.get("_objective_card") or {}),
        })
        out["provider"] = getattr(
            self._base, "_provenance_provider", self._provider_name,
        )
        out["synthesis_plan_sha256"] = _sha(
            self._plan_context(plan, include_evidence=True)
        )
        self._stamp_claim_evidence(out, plan)
        return out

    @staticmethod
    def _stamp_claim_evidence(
        pair: Dict[str, Any], plan: Mapping[str, Any],
    ) -> None:
        """Export each claim's exact source quote onto the emitted pair.

        Every ``evidence_quote`` here has already been canonicalized against
        the source window during plan validation, so these are known-exact
        spans of the cited chunk rather than model-authored text. Downstream,
        ``lib/validators/pair/claim_support.py`` scores each answer sentence
        against these specific quotes instead of the whole parent chunk.

        Without this the producer holds verified evidence and then discards
        it, leaving the validator to rediscover support from several hundred
        words of surrounding text — which is how faithful answers were being
        recorded as unsupported.
        """
        rows: List[Dict[str, Any]] = []
        for item in plan.get("supported_claims") or []:
            if not isinstance(item, Mapping):
                continue
            claim = _clean(item.get("claim"))
            quote = _clean(item.get("evidence_quote"))
            if not claim or not quote:
                continue
            row: Dict[str, Any] = {"claim": claim, "evidence_quote": quote}
            block_id = item.get("source_block_id")
            if isinstance(block_id, str) and block_id:
                row["source_block_id"] = block_id
            rows.append(row)
        if not rows:
            return
        provenance = pair.get("provenance")
        provenance = dict(provenance) if isinstance(provenance, dict) else {}
        provenance["claim_evidence"] = rows
        pair["provenance"] = provenance

    @staticmethod
    def _realization_schema(
        fields: Tuple[str, ...], expected: List[int],
    ) -> Dict[str, Any]:
        bounds = {
            # JSON Schema prevents empty/truncated fields only. Semantic
            # coverage and objective validators decide adequacy; a character
            # floor would incorrectly reject concise answers such as ``x=2``.
            "prompt": (1, 400),
            # PACKAGE A: Raise completion/chosen caps from 600 to 1200 chars to
            # accept grounded but deep answers that actually execute their prompt
            # (no token starvation—Stage D budgets 1536 output tokens; accepted
            # answers used ~100 completion tokens and stopped normally). The
            # binding limits are CHARACTER and ITEM caps, not token budgets.
            "completion": (1, 1200),
            "chosen": (1, 1200),
        }
        properties: Dict[str, Any] = {}
        for field in fields:
            minimum, maximum = bounds[field]
            properties[field] = {
                "type": "string", "minLength": minimum, "maxLength": maximum,
            }
        properties["covered_claim_indices"] = {
            "type": "array",
            "const": expected,
            "items": {"type": "integer", "minimum": 0},
            "uniqueItems": True,
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [*fields, "covered_claim_indices"],
            "properties": properties,
        }

    def paraphrase_preference(
        self, draft: Dict[str, Any], chunk: Dict[str, Any],
        *, preserve_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        plan = self._plan(chunk, need_misconception=True)
        chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
        source = _clean(chunk.get("text"))
        context = self._plan_context(plan)
        expected = list(range(len(plan["supported_claims"])))

        def validate_chosen(value: Dict[str, Any]) -> Optional[str]:
            prompt = _clean(value.get("prompt"))
            chosen_text = _clean(value.get("chosen"))
            failures: List[str] = []
            if not prompt or len(prompt) > 400:
                failures.append(f"prompt length {len(prompt)} is outside 1..400")
            # PACKAGE A: Raise chosen cap from 600 to 1200 chars to accept deeper
            # DPO chosen answers (schema already updated to 1200)
            if not chosen_text or len(chosen_text) > 1200:
                failures.append(
                    f"chosen length {len(chosen_text)} is outside 1..1200"
                )
            checks = (
                self._objective_contract_error(
                    prompt, plan.get("_objective_card") or {},
                    answer_bearing=False,
                ),
                self._generated_givens_prompt_error(prompt, plan),
                self._coverage_error(value, plan),
                self._semantic_coverage_error(chosen_text, plan, prompt=prompt),
            )
            failures.extend(error for error in checks if error)
            return (
                "validation failures: " + _stable_json(failures)
                if failures else None
            )

        chosen = self._call_stage(
            stage="dpo_chosen_realization", chunk_id=chunk_id,
            system=_REALIZATION_SYSTEM,
            user=(
                f"VALIDATED_PLAN:\n{context}\n\nReturn only "
                '{"prompt":"...","chosen":"...","covered_claim_indices":'
                f"{_stable_json(expected)}}}. The prompt must instantiate "
                "learner_task. The chosen answer may use only supported_claims."
            ),
            required_keys=("prompt", "chosen"),
            validator=validate_chosen,
            leakage_source=source, leakage_keys=("prompt", "chosen"),
            leakage_contract_text=(
                *((plan.get("_objective_card") or {}).get(
                    "content_obligations"
                ) or []),
            ),
            leakage_canonical_contract_text=(
                (plan.get("_objective_card") or {}).get("action_object") or "",
                *((plan.get("_objective_card") or {}).get("conditions") or []),
            ),
            leakage_nondistinctive_contexts=(
                (plan.get("_objective_card") or {}).get(
                    "nondistinctive_contexts"
                ) or []
            ),
            response_schema=self._realization_schema(
                ("prompt", "chosen"), expected,
            ),
        ).value
        def validate_rejected(value: Dict[str, Any]) -> Optional[str]:
            rejected_text = _clean(value.get("rejected"))
            # PACKAGE A: Raise rejected cap from 600 to 1200 chars to accept deeper
            # DPO rejected answers (schema already updated to 1200)
            if not rejected_text or len(rejected_text) > 1200:
                return (
                    f"rejected length {len(rejected_text)} is outside 1..1200"
                )
            if rejected_text.lower() == _clean(chosen.get("chosen")).lower():
                return "rejected duplicates chosen"
            if not any(
                marker in f" {rejected_text.lower()} "
                for marker in _FAULT_REASONING_MARKERS
            ):
                return "rejected must explicitly express faulty causal reasoning"
            distorted = value.get("distorted_claim_index")
            if distorted not in expected:
                return "distorted_claim_index must identify one planned claim"
            affordance = plan["misconception_affordance"]
            for field in (
                "error_mechanism", "faulty_step", "causal_rationale",
            ):
                if (
                    _clean(value.get(field)).lower()
                    != _clean(affordance[field]).lower()
                ):
                    return f"{field} differs from the validated plan"
            semantic_scores = self._scores(
                [
                    (rejected_text, _clean(affordance["error_mechanism"])),
                    (rejected_text, _clean(affordance["faulty_step"])),
                    (rejected_text, _clean(affordance["causal_rationale"])),
                    (_clean(chosen["chosen"]), rejected_text),
                ],
                decision_types=[
                    "dpo_error_mechanism",
                    "dpo_faulty_step",
                    "dpo_causal_rationale",
                    "dpo_chosen_contrast",
                ],
            )
            for field, score in zip(
                ("error_mechanism", "faulty_step", "causal_rationale"),
                semantic_scores[:3],
            ):
                if float(score.entailment) < _PLAN_ENTAILMENT_FLOOR:
                    return f"rejected does not express declared {field}"
            contrast = semantic_scores[3]
            if (
                float(contrast.entailment) >= _PLAN_ENTAILMENT_FLOOR
                and float(contrast.contradiction)
                < _PLAN_CONTRADICTION_CEILING
            ):
                return "rejected does not locally contrast the chosen answer"
            evidence_premise = " ".join(
                item["evidence_quote"] for item in plan["supported_claims"]
            )
            evidence_score = self._score_pair(
                decision_type="dpo_evidence_support",
                premise=evidence_premise,
                hypothesis=rejected_text,
            )
            if (
                float(evidence_score.entailment) >= _PLAN_ENTAILMENT_FLOOR
                and float(evidence_score.contradiction)
                < _PLAN_CONTRADICTION_CEILING
            ):
                return "rejected is supported by factual evidence"
            _ledger, derivation_error = _deterministic_derivation_ledger(
                prompt=_clean(chosen["prompt"]),
                output=rejected_text,
                claims=[
                    _clean(item.get("claim"))
                    for item in plan.get("supported_claims") or []
                ],
                generated_givens=plan.get("generated_givens") or [],
                source_facts=[
                    _clean(item.get("evidence_quote"))
                    for item in plan.get("supported_claims") or []
                ],
            )
            if derivation_error:
                return f"rejected violates generated-givens contract: {derivation_error}"
            return None

        rejected = self._call_stage(
            stage="dpo_rejected_realization", chunk_id=chunk_id,
            system=(
                "You realize one deliberately incorrect preference branch. "
                "Do not alter or answer the prompt."
            ),
            user=(
                f"VALIDATED_PLAN:\n{context}\n\nFIXED_PROMPT:\n"
                f"{_clean(chosen['prompt'])}\n\nCORRECT_ANSWER:\n"
                f"{_clean(chosen['chosen'])}\n\nRealize only the planned "
                "misconception as a plausible but clearly inferior answer. "
                "The rejected answer must explicitly state its faulty reasoning "
                "with a causal explanation, not merely alter one token. Return "
                'only {"rejected":"...","distorted_claim_index":0,'
                '"error_mechanism":"...","faulty_step":"...",'
                '"causal_rationale":"..."}.'
            ),
            required_keys=(
                "rejected", "distorted_claim_index", "error_mechanism",
                "faulty_step", "causal_rationale",
            ),
            validator=validate_rejected,
            leakage_source=source, leakage_keys=("rejected",),
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "rejected", "distorted_claim_index", "error_mechanism",
                    "faulty_step", "causal_rationale",
                ],
                "properties": {
                    # PACKAGE A: Raise rejected cap from 600 to 1200 chars
                    "rejected": {
                        "type": "string", "minLength": 1, "maxLength": 1200,
                    },
                    "distorted_claim_index": {
                        "type": "integer", "enum": expected,
                    },
                    "error_mechanism": {
                        "type": "string", "minLength": 1, "maxLength": 240,
                    },
                    "faulty_step": {
                        "type": "string", "minLength": 1, "maxLength": 240,
                    },
                    "causal_rationale": {
                        "type": "string", "minLength": 1, "maxLength": 320,
                    },
                },
            },
        ).value
        out = dict(draft)
        out.update({
            "prompt": _clean(chosen["prompt"]),
            "chosen": _clean(chosen["chosen"]),
            "rejected": _clean(rejected["rejected"]),
            "observed_bloom": plan["bloom_level"],
            "objective_contract": dict(plan.get("_objective_card") or {}),
            "provider": getattr(
                self._base, "_provenance_provider", self._provider_name,
            ),
            "synthesis_plan_sha256": _sha(
                self._plan_context(plan, include_evidence=True)
            ),
        })
        self._stamp_claim_evidence(out, plan)
        return out


__all__ = [
    "ENV_STAGED_SYNTHESIS_V4",
    "ENV_SERVED_CONTEXT_TOKENS",
    "StagedSynthesisProvider",
    "staged_synthesis_v4_enabled",
]
