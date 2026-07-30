"""Compact, provenance-preserving evidence windows for staged synthesis."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, Mapping, Optional

from bs4 import BeautifulSoup

from lib.ontology.bloom import BLOOM_LEVELS


WINDOW_CONTRACT_VERSION = "ed4all.staged-synthesis-window.v1"
_OBJECTIVE_ATTRS = ("data-cf-objective-id", "data-cf-objective-ref")

#: Bloom-ladder initiative (WI-11) — gates per-CARD bloom-rung recovery in
#: ``_structured_misconception_pairs`` / ``resolve_chunk_misconceptions`` and
#: the ``target_rung`` partition/annotation in ``build_evidence_window``.
#: Default OFF; unset / falsey / garbage keeps every gated path byte-identical
#: to the pre-ladder contract. See ``resolve_bloom_windows_enabled``.
BLOOM_WINDOWS_ENV = "TRAINFORGE_BLOOM_WINDOWS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: The per-block Bloom-rung wrapper attribute a Courseforge block carries
#: alongside ``data-cf-block-id`` (see ``Courseforge/scripts/blocks.py``).
#: Reading it here is additive-only: the attribute already exists on authored
#: markup independent of this flag, this module just starts consulting it.
_BLOOM_LEVEL_ATTR = "data-cf-bloom-level"
#: Ordinal lookup over the canonical six-level order (remember=0 .. create=5)
#: so a rung ceiling comparison never hardcodes the taxonomy locally.
_BLOOM_ORDINAL = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}


def resolve_bloom_windows_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve ``TRAINFORGE_BLOOM_WINDOWS`` (parse-with-fallback, default OFF).

    Truthy (``1``/``true``/``yes``/``on``, case-insensitive) -> on;
    unset / empty / falsey / garbage -> off. Read at call time (never cached
    at import), so a test can drive it with an injected mapping and a
    long-lived process never observes a stale value.
    """
    source = os.environ if env is None else env
    raw = str(source.get(BLOOM_WINDOWS_ENV, "") or "").strip().lower()
    return raw in _TRUTHY


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


def _structured_misconception_pairs(tag: Any) -> list[Dict[str, str]]:
    """Recover a block's authored (claim, correction) split from its own markup.

    A misconception card is emitted as ONE block: only the outer element
    carries ``data-cf-block-id``, so :func:`build_evidence_window` collapses
    the whole card into a single ``text`` and the authored boundary between
    the false claim and its correction is lost.  Downstream, DPO admission
    needs exactly that boundary.

    The descendants ARE labelled — the same class tokens
    :func:`_role_and_polarity` already recognises — so the split is recovered
    by re-reading the card's own structure rather than re-deriving it from
    prose with a sentence heuristic (which corrupts every card whose heading
    merges into the first sentence).

    This is additive metadata only: block membership, ``role``, ``polarity``
    and ``text`` are untouched, so ``evidence_text``, the rendered window and
    every instruction-side consumer stay byte-identical.

    TRAINFORGE_BLOOM_WINDOWS (default off): when on, each returned pair also
    carries the CARD's own ``bloom_level``, recovered from ``tag``'s own
    ``data-cf-bloom-level`` attribute — the same wrapper this function is
    called on, since a misconception card is authored as one block. This is
    the per-CARD rung; ``resolve_chunk_misconceptions`` decides how it trades
    off against the chunk-wide fallback. Off, or the wrapper carries no rung
    of its own -> no ``bloom_level`` key, so the return shape is unchanged.
    """
    claims: list[str] = []
    corrections: list[str] = []
    for child in tag.find_all(True):
        if child.get("data-cf-block-id"):
            # A nested block owns its own window entry; never reach across.
            continue
        role, _polarity = _role_and_polarity(child)
        text = _clean(child.get_text(" ", strip=True))
        if not text:
            continue
        if role == "misconception_claim":
            claims.append(text)
        elif role == "misconception_correction":
            corrections.append(text)
    card_bloom_level = (
        _clean(tag.get(_BLOOM_LEVEL_ATTR)).lower()
        if resolve_bloom_windows_enabled()
        else ""
    )
    pairs: list[Dict[str, str]] = []
    for index, claim in enumerate(claims):
        if index >= len(corrections):
            break
        pair: Dict[str, str] = {"claim": claim, "correction": corrections[index]}
        if card_bloom_level:
            pair["bloom_level"] = card_bloom_level
        pairs.append(pair)
    return pairs


def resolve_chunk_misconceptions(
    chunk: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Return a chunk's structured misconceptions, recovering authored cards.

    ``chunk["misconceptions"]`` is the field EVERY downstream consumer reads:
    :func:`Trainforge.generators.staged_synthesis_micro.micro_preference_eligibility`
    Arm A (admission to synthesis), and — decisively —
    :func:`Trainforge.generators.preference_factory.synthesize_preference_pair`,
    which stamps ``source="misconception"`` + a real ``misconception_id`` ONLY
    from a non-empty value here.  Those two stamps are the whole of DPO
    admission: ``Trainforge.training.compute_backend.is_dpo_editorial_record``
    admits a pair on `misconception_id` or an editorial `source`, and nothing
    else.

    That is why recovering the authored split at the EVIDENCE-WINDOW layer
    alone is not enough.  ``build_evidence_window`` hands
    ``misconception_pairs`` to the eligibility predicate, so a corpus of
    authored cards reads as ELIGIBLE — while the emitter, reading this field
    and finding it empty, stamps every emitted pair ``rule_synthesized`` with
    no ``misconception_id``.  The filtered DPO count is then zero on a corpus
    the gate just called eligible, and (``dpo_fail_hard=true``) the trainer
    refuses the run.  Eligibility and emission have to read the SAME
    misconceptions, so this resolver is the one place that decides them.

    Authored value wins: a chunk carrying a non-empty ``misconceptions`` list
    is returned untouched.  Recovery only fills a field that is absent or
    empty, from the chunk's OWN markup — never from another chunk, never
    synthesized.  A corpus whose chunker already lands structured
    misconceptions is therefore byte-identical through this function.

    TRAINFORGE_BLOOM_WINDOWS (default off): a chunk-wide ``bloom_level``
    (below) stamped onto EVERY recovered misconception is a latent
    multi-rung collapse — a chunk holding cards at several Bloom rungs would
    have every one of them mislabeled with the same chunk-level value. When
    the flag is on, the per-CARD rung recovered by
    :func:`_structured_misconception_pairs` (from that card's own
    ``data-cf-bloom-level`` wrapper attribute) wins; the chunk-wide value is
    used only when the card carries no rung of its own. Off, or a card
    without its own rung, keeps the pre-existing chunk-wide-fallback
    behavior byte-identical.
    """
    authored = chunk.get("misconceptions")
    if isinstance(authored, list) and any(
        isinstance(item, Mapping) for item in authored
    ):
        return list(authored)
    html = str(chunk.get("html") or "")
    if not html:
        return []
    bloom_level = _clean(chunk.get("bloom_level"))
    soup = BeautifulSoup(html, "html.parser")
    recovered: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for tag in soup.find_all(True):
        block_id = _clean(tag.get("data-cf-block-id"))
        if not block_id:
            continue
        # Mirror build_evidence_window's deepest-occurrence rule so a nested
        # wrapper cannot claim its child's card.
        if any(
            child.get("data-cf-block-id") == tag.get("data-cf-block-id")
            for child in tag.find_all(True)
        ):
            continue
        for pair in _structured_misconception_pairs(tag):
            claim = _clean(pair.get("claim"))
            correction = _clean(pair.get("correction"))
            if not claim or not correction:
                continue
            key = (claim.lower(), correction.lower())
            if key in seen:
                continue
            seen.add(key)
            entry: Dict[str, Any] = {
                "misconception": claim,
                "correction": correction,
                # The authored correction IS the mechanism evidence, matching
                # micro_preference_eligibility's own
                # ``mechanism_evidence or error_mechanism or correction``
                # fallback chain. No `id` is stamped: Arm A recomputes the
                # canonical one and REJECTS a supplied id that disagrees.
                "mechanism_evidence": correction,
                "source_block_id": block_id,
            }
            # Per-CARD rung wins over the chunk-wide fallback (see the
            # TRAINFORGE_BLOOM_WINDOWS docstring paragraph above); pair only
            # carries "bloom_level" when the flag is on AND the card's own
            # wrapper declared a rung, so this is a no-op when off.
            card_bloom_level = _clean(pair.get("bloom_level"))
            if card_bloom_level:
                entry["bloom_level"] = card_bloom_level
            elif bloom_level:
                entry["bloom_level"] = bloom_level
            recovered.append(entry)
    return recovered


#: Class tokens marking a vocabulary/definition card in authored chunk HTML.
_VOCAB_CARD_CLASSES = ("vocab-card", "definition-box", "callout-definition")
#: Class token marking the TERM inside such a card.
_KEY_TERM_CLASS = "key-term"
#: Class token marking the provenance backlink, which is never definition text.
_KEY_TERM_SOURCE_CLASS = "key-term-source"


def resolve_chunk_key_terms(
    chunk: Mapping[str, Any],
) -> list[Dict[str, str]]:
    """Return a chunk's ``key_terms``, recovering authored vocabulary cards.

    ``key_terms`` is populated on **0%** of every archived course measured
    (openstax / alg / mmo), because the parser harvests it only from JSON-LD
    (``keyTerms`` / ``blocks[].key_terms``) and the rewrite tier emits
    vocabulary as HTML cards instead. The content is right there in the chunk:
    one measured course carries 300 ``vocab_card`` blocks, 248
    ``definition-box`` and 107 ``vocab-term`` spans while reporting zero key
    terms.

    That emptiness is load-bearing, not cosmetic.
    ``preference_factory._derive_topic`` needs ``concept_tags`` OR
    ``key_terms`` to name a topic; with both empty it falls back to splicing a
    database key into the topic slot ("...explain the parts of learning
    outcome co-117"), which is the documented source of content-free
    completions that NLI correctly scores ~0.03 entailment. Recovering the
    authored terms gives the generator a real topic and a real definition to
    ground against.

    Same contract as :func:`resolve_chunk_misconceptions`: an authored
    non-empty ``key_terms`` is returned untouched, recovery reads only the
    chunk's OWN markup, and nothing is synthesized.
    """
    authored = chunk.get("key_terms")
    if isinstance(authored, list) and any(
        isinstance(item, Mapping) for item in authored
    ):
        return list(authored)
    html = str(chunk.get("html") or "")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    recovered: list[Dict[str, str]] = []
    seen: set[str] = set()
    for card in soup.find_all(True):
        classes = card.get("class") or []
        if not any(token in classes for token in _VOCAB_CARD_CLASSES):
            continue
        term = ""
        term_node = card.find(
            lambda tag: _KEY_TERM_CLASS in (tag.get("class") or [])
            and _KEY_TERM_SOURCE_CLASS not in (tag.get("class") or [])
        )
        if term_node is not None:
            term = _clean(term_node.get_text(" ", strip=True))
        if not term:
            # `data-cf-term` carries the slug; only used when the card has no
            # rendered term, and never invented from the definition text.
            term = _clean(str(card.get("data-cf-term") or "")).replace("-", " ")
        if not term:
            continue
        definition = ""
        for node in card.find_all(["p", "dd", "div", "span"]):
            node_classes = node.get("class") or []
            if _KEY_TERM_SOURCE_CLASS in node_classes:
                continue
            text = _clean(node.get_text(" ", strip=True))
            if not text or text == term:
                continue
            # The rendered definition often repeats the term as a lead-in
            # ("absolute value The absolute value of ..."); keep the sentence,
            # just drop a duplicated leading term so the pair reads cleanly.
            lowered = text.lower()
            if lowered.startswith(term.lower()) and len(text) > len(term) + 1:
                text = text[len(term):].lstrip(" :-—")
            if len(text) < len(term) + 2:
                continue
            definition = text
            break
        if not definition:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        recovered.append({"term": term, "definition": definition})
    return recovered


def build_evidence_window(
    chunk: Mapping[str, Any],
    focus: Mapping[str, Any],
    target_rung: Optional[str] = None,
) -> Dict[str, Any]:
    """Select objective-aligned semantic blocks, preserving false/correct polarity.

    ``target_rung`` (optional; consulted only when ``TRAINFORGE_BLOOM_WINDOWS``
    is on) is a Bloom-ladder ceiling. A candidate block whose own
    ``data-cf-bloom-level`` ranks ABOVE ``target_rung`` on the canonical
    ``lib.ontology.bloom.BLOOM_LEVELS`` order is evidence authored for a
    later rung and is excluded from this window; a block with no rung
    annotation, or one at/below ``target_rung``, is kept and additionally
    gains a ``bloom_level`` field recording its own rung. An unrecognized
    ``target_rung`` (not one of the six canonical levels) is treated as "no
    ceiling" -- every candidate is annotated but none is dropped on a
    garbage value.

    Flag off, or flag on with ``target_rung`` omitted (the caller-side
    ladder wiring that supplies it is a separate work item), keeps every
    block exactly ``{block_id, role, polarity, text}`` as before -- this
    parameter is purely additive.
    """
    bloom_windows_on = resolve_bloom_windows_enabled()
    target_ordinal = (
        _BLOOM_ORDINAL.get(_clean(target_rung).lower())
        if bloom_windows_on and target_rung
        else None
    )
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
            block_bloom_level = ""
            if bloom_windows_on:
                block_bloom_level = _clean(tag.get(_BLOOM_LEVEL_ATTR)).lower()
                if block_bloom_level and target_ordinal is not None:
                    block_ordinal = _BLOOM_ORDINAL.get(block_bloom_level)
                    if block_ordinal is not None and block_ordinal > target_ordinal:
                        # Evidence authored for a later rung; not this window.
                        continue
            block: Dict[str, Any] = {
                "block_id": block_id,
                "role": role,
                "polarity": polarity,
                "text": text,
            }
            if block_bloom_level:
                block["bloom_level"] = block_bloom_level
            misconception_pairs = _structured_misconception_pairs(tag)
            if misconception_pairs:
                block["misconception_pairs"] = misconception_pairs
            blocks.append(block)
    if not blocks:
        source = _clean(chunk.get("text"))
        if not source:
            raise ValueError("staged synthesis evidence window has no source text")
        # DELIBERATELY reads ``statement`` only, NOT the ``misconception``
        # spelling chunk_v4 also permits. This asymmetry is load-bearing and
        # must not be "fixed" into symmetry: ``micro_preference_eligibility``
        # Arm A consumes ``chunk["misconceptions"]`` and REQUIRES the claim to
        # be source-backed (present in the chunk's own text), so a structured
        # DPO misconception is embedded in the source BY CONSTRUCTION. Making
        # this guard see the ``misconception`` spelling therefore refuses the
        # whole chunk for exactly the shape Arm A exists to consume, and Arm A
        # becomes unreachable on any flat (no-HTML) chunk.
        #
        # A ``statement``-keyed misconception on a flat chunk IS still refused
        # here, so that shape stays unreachable by Arm A. That is a real
        # residual tension between this polarity guard and Arm A's
        # source-backing requirement, not something to paper over by relaxing
        # either side; resolving it needs an owner decision about which
        # spelling means "structured DPO source" and which means "this prose
        # embeds a known error".
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
    """Render each block as ``[id|role|polarity] text``.

    A block carrying a ``bloom_level`` (only possible when
    ``TRAINFORGE_BLOOM_WINDOWS`` is on and ``build_evidence_window`` annotated
    it) renders a fourth tag segment: ``[id|role|polarity|bloom] text``. No
    block in the window carries the key when the flag is off, so the
    rendered string is byte-identical to before.
    """
    lines = []
    for block in window["blocks"]:
        rung = block.get("bloom_level")
        tag = (
            f"[{block['block_id']}|{block['role']}|{block['polarity']}|{rung}]"
            if rung
            else f"[{block['block_id']}|{block['role']}|{block['polarity']}]"
        )
        lines.append(f"{tag} {block['text']}")
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
    "BLOOM_WINDOWS_ENV",
    "build_evidence_window",
    "canonical_quote",
    "objective_card",
    "render_evidence_window",
    "resolve_bloom_windows_enabled",
    "semantic_anchor_hypothesis",
    "semantic_anchor_forms",
    "semantic_anchor_records",
    "semantic_anchor_sequence",
    "semantic_anchors",
]
