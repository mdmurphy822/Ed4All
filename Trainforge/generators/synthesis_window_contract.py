"""Compact, provenance-preserving evidence windows for staged synthesis."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional

from bs4 import BeautifulSoup


WINDOW_CONTRACT_VERSION = "ed4all.staged-synthesis-window.v1"
_OBJECTIVE_ATTRS = ("data-cf-objective-id", "data-cf-objective-ref")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


_PERFORMANCE_DEGREE_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*%|\baccuracy\b|\bcorrect(?:ly|ness)?\b|"
    r"\bwithout (?:error|errors)\b|\bwithin \d+\b)",
    re.IGNORECASE,
)
_CONTENT_DEGREE_RE = re.compile(
    r"^\s*to\s+(?P<content>.+?)(?:\s+with\s+(?P<performance>.+))?$",
    re.IGNORECASE,
)
_SEMANTIC_ANCHOR_TERMS = {
    "product": frozenset({"product", "multiply", "multiplied", "multiplication"}),
    "sum": frozenset({"sum", "add", "added", "addition", "total"}),
    "selection": frozenset({
        "choose", "chooses", "choosing", "identify", "identifies",
        "pick", "picks", "select", "selected", "selecting", "selection",
    }),
}
_SEMANTIC_ANCHOR_LEMMAS = {
    "product": "multiply",
    "sum": "add",
    "selection": "select",
}


def semantic_anchors(value: Any) -> list[str]:
    """Return normalized relation anchors from grammatical or telegraphic text."""
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{1,}", _clean(value))
    }
    return [
        anchor
        for anchor, forms in _SEMANTIC_ANCHOR_TERMS.items()
        if tokens & forms
    ]


def semantic_anchor_sequence(value: Any) -> list[str]:
    """Return distinct semantic relations in their textual order."""
    matches = []
    for anchor, forms in _SEMANTIC_ANCHOR_TERMS.items():
        pattern = "|".join(map(re.escape, sorted(forms, key=len, reverse=True)))
        for match in re.finditer(rf"\b(?:{pattern})\b", _clean(value), re.IGNORECASE):
            matches.append((match.start(), anchor))
    return list(dict.fromkeys(anchor for _, anchor in sorted(matches)))


def semantic_anchor_hypothesis(anchor: str) -> str:
    """Grammaticalize one normalized anchor for strict paraphrase NLI."""
    if anchor == "selection":
        return "The learner task requires selecting the correct candidate."
    return f"The learner task requires using the {anchor} relation."


def semantic_anchor_forms(anchor: str) -> frozenset[str]:
    """Return the normalized lexical forms owned by a semantic relation."""
    return _SEMANTIC_ANCHOR_TERMS.get(anchor, frozenset())


def semantic_anchor_records(value: Any) -> list[Dict[str, str]]:
    """Decompose telegraphic content into auditable lemma/relation anchors."""
    return [
        {
            "lemma": _SEMANTIC_ANCHOR_LEMMAS[relation],
            "relation": relation,
            "hypothesis": semantic_anchor_hypothesis(relation),
        }
        for relation in semantic_anchors(value)
    ]


def _typed_objective_dimensions(
    action_object: str, condition: str, degree: str,
) -> tuple[list[str], list[str], list[str]]:
    obligations = [action_object] if action_object else []
    conditions = [condition] if condition else []
    performance: list[str] = []
    degree_match = _CONTENT_DEGREE_RE.match(degree)
    if degree_match:
        content = _clean(degree_match.group("content"))
        if content:
            obligations.extend(
                re.sub(r"^and\s+", "", _clean(part), flags=re.IGNORECASE)
                for part in re.split(r"\s*,\s*|\s+and\s+", content)
                if _clean(part)
            )
        criterion = _clean(degree_match.group("performance"))
        if criterion:
            performance.append(criterion)
    elif degree and _PERFORMANCE_DEGREE_RE.search(degree):
        performance.append(degree)
    elif degree:
        conditions.append(degree)
    return obligations, conditions, performance


def objective_card(focus: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain canonical instructional dimensions without inventing values."""
    abcd = focus.get("abcd") if isinstance(focus.get("abcd"), Mapping) else {}
    behavior = (
        abcd.get("behavior")
        if isinstance(abcd.get("behavior"), Mapping)
        else {}
    )
    action_object = _clean(
        focus.get("action_object") or behavior.get("action_object")
    )
    condition = _clean(focus.get("condition") or abcd.get("condition"))
    degree = _clean(focus.get("degree") or abcd.get("degree"))
    obligations, conditions, performance = _typed_objective_dimensions(
        action_object, condition, degree,
    )
    obligation_anchors = [
        {
            "obligation": obligation,
            "anchors": semantic_anchor_records(obligation),
        }
        for obligation in obligations
    ]
    card = {
        "id": _clean(focus.get("id")).lower(),
        "statement": _clean(focus.get("statement")),
        "bloom_level": _clean(focus.get("bloom_level")).lower(),
        "bloom_verb": _clean(
            focus.get("bloom_verb") or behavior.get("verb")
        ).lower(),
        "action_object": action_object,
        "condition": condition,
        "degree": degree,
        "content_obligations": obligations,
        "content_obligation_anchors": obligation_anchors,
        "conditions": conditions,
        "performance_criteria": performance,
    }
    # Optional corpus-level evidence used only to prove that objective wording
    # is non-distinctive.  Omitting it preserves the canonical legacy card
    # byte-for-byte.
    contexts = [
            {
                "context_id": _clean(item.get("context_id")),
                "text": _clean(item.get("text")),
            }
            for item in (focus.get("nondistinctive_contexts") or [])
            if isinstance(item, Mapping)
            and _clean(item.get("context_id"))
            and _clean(item.get("text"))
    ]
    if contexts:
        card["nondistinctive_contexts"] = contexts
    return card


def _tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        values: Iterable[Any] = re.split(r"[\s,]+", value)
    elif isinstance(value, Iterable):
        values = value
    else:
        return []
    return [token for item in values if (token := _clean(item))]


def _content_obligations_evidenced(
    card: Mapping[str, Any], evidence: str,
) -> bool:
    def semantic_token(token: str) -> str:
        lowered = token.lower()
        for suffix in ("ically", "ality", "ity", "ly"):
            if lowered.endswith(suffix) and len(lowered) > len(suffix) + 3:
                return lowered[:-len(suffix)]
        return lowered

    source = {
        semantic_token(token)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", evidence)
    }
    for obligation in card.get("content_obligations") or []:
        required = {
            semantic_token(token)
            for token in re.findall(
                r"[A-Za-z][A-Za-z0-9'-]{2,}", str(obligation),
            )
            if token.lower() not in {
                "and", "for", "from", "the", "their", "with",
            }
        }
        if not required:
            continue
        overlap = len(required & source)
        if overlap < (1 if len(required) <= 2 else 2):
            return False
        if overlap / len(required) < 0.25:
            return False
    return True


def _matches_objective(tag: Any, objective_id: str) -> bool:
    wanted = objective_id.lower()
    for attr in _OBJECTIVE_ATTRS:
        if wanted in {item.lower() for item in _tokens(tag.get(attr))}:
            return True
    return False


def _role_and_polarity(tag: Any) -> tuple[str, str]:
    signal = " ".join([
        _clean(tag.get("data-cf-block-id")),
        _clean(tag.get("data-cf-content-type")),
        " ".join(tag.get("class") or []),
    ]).lower()
    if "misconception-claim" in signal or "incorrect" in signal:
        return "misconception_claim", "incorrect"
    if "misconception-correction" in signal or "correction" in signal:
        return "misconception_correction", "corrective"
    if "worked" in signal or "example" in signal:
        return "worked_example", "factual"
    if "question" in signal:
        return "question", "question"
    if "answer" in signal:
        return "answer", "factual"
    return "evidence", "factual"


def build_evidence_window(
    chunk: Mapping[str, Any], focus: Mapping[str, Any],
) -> Dict[str, Any]:
    """Select objective-aligned semantic blocks, preserving false/correct polarity."""
    card = objective_card(focus)
    chunk_id = _clean(chunk.get("id") or chunk.get("chunk_id"))
    html = str(chunk.get("html") or "")
    blocks: list[Dict[str, Any]] = []
    if html:
        soup = BeautifulSoup(html, "html.parser")
        candidates = [
            tag for tag in soup.find_all(True)
            if tag.get("data-cf-block-id")
            and _matches_objective(tag, card["id"])
        ]
        # Duplicate block IDs occur on nested wrappers. The deepest occurrence
        # is the narrowest semantic block and avoids accidentally sending a page.
        candidates = [
            tag for tag in candidates
            if not any(
                child.get("data-cf-block-id") == tag.get("data-cf-block-id")
                for child in tag.find_all(True)
            )
        ]
        seen: set[tuple[str, str]] = set()
        for tag in candidates:
            role, polarity = _role_and_polarity(tag)
            text = _clean(tag.get_text(" ", strip=True))
            block_id = _clean(tag.get("data-cf-block-id"))
            key = (polarity, text.lower())
            if not text or key in seen:
                continue
            seen.add(key)
            blocks.append({
                "block_id": block_id,
                "role": role,
                "polarity": polarity,
                "text": text,
            })
    if not blocks:
        source = _clean(chunk.get("text"))
        if not source:
            raise ValueError("staged synthesis evidence window has no source text")
        misconception_texts = [
            _clean(item.get("statement")).lower()
            for item in (chunk.get("misconceptions") or [])
            if isinstance(item, Mapping) and _clean(item.get("statement"))
        ]
        mixed_marker = re.search(
            r"\b(?:incorrect|wrong|misconception|myth|false|not true)\b",
            source,
            flags=re.IGNORECASE,
        )
        embeds_known_error = any(
            text in source.lower() for text in misconception_texts
        )
        if mixed_marker or embeds_known_error:
            raise ValueError(
                "flat synthesis source has unresolved misconception polarity"
            )
        blocks = [{
            "block_id": chunk_id,
            "role": "flat_source",
            "polarity": "factual",
            "text": source,
        }]
    evidence_text = " ".join(
        block["text"]
        for block in blocks
        if block["polarity"] in {"factual", "corrective"}
    )
    if not _content_obligations_evidenced(card, evidence_text):
        raise ValueError(
            "staged synthesis evidence does not cover every objective "
            "content obligation"
        )
    return {
        "contract_version": WINDOW_CONTRACT_VERSION,
        "parent_chunk_id": chunk_id,
        "objective": card,
        "blocks": blocks,
        "concept_tags": list(dict.fromkeys(_tokens(chunk.get("concept_tags")))),
        "curies": list(dict.fromkeys(_tokens(chunk.get("curies")))),
    }


def render_evidence_window(window: Mapping[str, Any]) -> str:
    lines = []
    for block in window["blocks"]:
        lines.append(
            f"[{block['block_id']}|{block['role']}|{block['polarity']}] "
            f"{block['text']}"
        )
    return "\n".join(lines)


def canonical_quote(
    quote: str, window: Mapping[str, Any],
) -> Optional[tuple[str, Mapping[str, Any]]]:
    """Locate a normalized quote in one semantic block and return exact text."""
    wanted = _clean(quote)
    for block in window["blocks"]:
        source = _clean(block["text"])
        index = source.lower().find(wanted.lower())
        if index >= 0:
            return source[index:index + len(wanted)], block
        wanted_tokens = re.findall(r"[A-Za-z0-9]+", wanted.lower())
        matches = list(re.finditer(r"[A-Za-z0-9]+", source))
        tokens = [item.group(0).lower() for item in matches]
        width = len(wanted_tokens)
        if width < 4:
            continue
        for start in range(len(tokens) - width + 1):
            if tokens[start:start + width] == wanted_tokens:
                return (
                    source[matches[start].start():matches[start + width - 1].end()],
                    block,
                )
    return None


__all__ = [
    "WINDOW_CONTRACT_VERSION",
    "build_evidence_window",
    "canonical_quote",
    "objective_card",
    "render_evidence_window",
    "semantic_anchor_hypothesis",
    "semantic_anchor_forms",
    "semantic_anchor_records",
    "semantic_anchor_sequence",
    "semantic_anchors",
]
