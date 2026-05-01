"""Schema-to-English translation SFT pair generator (Wave 124,
audit 2026-04-30 fix; Wave 125b expansion; Wave 133d loader pattern).

The cc07cc76 SLM adapter scored hallucination_rate=0.63 / faithfulness=0.37
in part because the corpus had no pairs that taught the model to map
formal SHACL/RDF/OWL surface forms (``sh:datatype``,
``rdfs:subClassOf``, ``owl:sameAs``, ...) to plain-English meanings.
The eval harness probes via property-aware questions; without a
schema-to-English bridge the adapter either parrots the surface form
back or hallucinates an unrelated paraphrase.

This generator is driven by ``lib/ontology/property_manifest.py``: it
loads the family manifest (e.g. ``property_manifest.rdf_shacl.yaml``),
walks every declared property, and emits a hand-curated catalog of
~250 pairs spread across SIX template families per surface form:

  1. ``schema_translation.definition`` (bloom: remember) — formal-spec
     and pedagogical renderings of "what the CURIE means".
  2. ``schema_translation.usage`` (bloom: understand) — concrete TTL
     fragments showing how the construct is applied.
  3. ``schema_translation.comparison`` (bloom: analyze) — pairwise
     contrasts with adjacent SHACL/RDF/OWL constructs.
  4. ``schema_translation.reasoning`` (bloom: apply) — scenario probes
     where the answer is the CURIE itself.
  5. ``schema_translation.pitfall`` (bloom: analyze) — error-mode
     scenarios + corrections (validation traps, forgetting parts of
     the contract, mis-applying the construct).
  6. ``schema_translation.combination`` (bloom: apply) — multi-construct
     scenarios where the surface form composes with another.

The catalog targets ~250 pairs total: 6 surface forms × 6 families ×
~7 pairs/family/form. The default ``max_pairs`` cap remains 50 for
backward compatibility; production rebuilds pass
``--schema-translation-max-pairs 200`` to land at the target ~6%
corpus balance.

Pair shape carries:

  * ``content_type="schema_translation"`` — downstream filters /
    diversity scorers can isolate the cohort without re-parsing prompts.
  * ``bloom_level`` — varies per family.
  * ``template_id="schema_translation.<family>"``.
  * ``concept_tags=[curie]`` — anchors the surface form so a future
    eval can re-verify the bridge. Comparison + combination pairs
    carry the primary CURIE as the first tag and the secondary CURIE
    as the second.
  * The literal primary CURIE appears in every completion so
    ``preserve_tokens`` plumbing in synthesize_training.py recognises
    the surface form (matches the manifest's preserve-token contract).

Decision capture: one ``schema_translation_generation`` event per
emitted pair, rationale interpolating curie + family so audit replay
can spot a drift between manifest surface forms and the hand-curated
table.

Implementation note: the manifest is the single source of truth for
the set of surface forms; this generator's hand-curated table is
indexed by CURIE and falls through silently for any CURIE the
manifest declares but this table doesn't define. That keeps the
manifest extensible — adding a new property doesn't break the
generator, it just doesn't generate translation pairs until the
table catches up. A logger.warning surfaces the gap.

Cap-with-balance behavior: when ``max_pairs`` is below the catalog
size, the emit loop walks family-by-family round-robin so a low cap
preserves balance across all 6 families instead of dumping every
definition pair before any usage pair.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.property_manifest import (  # noqa: E402
    PropertyManifest,
)

logger = logging.getLogger(__name__)


# Default total cap. Catalog now contains ~250 pairs across 6 families
# x 6 surface forms; default 50 keeps backward compatibility with
# existing call sites. Production rebuilds raise this via
# `--schema-translation-max-pairs 200`.
DEFAULT_MAX_PAIRS = 50


# Family identifiers — used by template_id and per-family balance
# counting. Order is stable (matches the round-robin emit order under
# capped runs).
_FAMILIES: Tuple[str, ...] = (
    "definition",
    "usage",
    "comparison",
    "reasoning",
    "pitfall",
    "combination",
)


_FAMILY_BLOOM: Dict[str, str] = {
    "definition": "remember",
    "usage": "understand",
    "comparison": "analyze",
    "reasoning": "apply",
    "pitfall": "analyze",
    "combination": "apply",
}


@dataclass
class SchemaTranslationStats:
    """Counts returned from :func:`generate_schema_translation_pairs`."""

    surface_forms_total: int = 0
    surface_forms_used: int = 0
    surface_forms_skipped_no_definition: int = 0
    pairs_emitted: int = 0
    capped_at_max_pairs: bool = False
    per_surface_form: Dict[str, int] = field(default_factory=dict)
    per_family: Dict[str, int] = field(default_factory=dict)


_VALID_ANCHORED_STATUS: Tuple[str, ...] = ("complete", "degraded_placeholder")
# Wave 135a placeholder strings (must be searchable via the literal
# "[degraded:" prefix so operators / Codex / Qwen can grep all 34 stub
# entries trivially).
_DEGRADED_DEFINITION_STUB = (
    "[degraded: anchored definition not yet authored — see Wave 135a contract]"
)
_DEGRADED_USAGE_PROMPT_STUB = (
    "[degraded: anchored usage prompt not yet authored]"
)
_DEGRADED_USAGE_COMPLETION_STUB = (
    "[degraded: anchored usage answer not yet authored]"
)


@dataclass(frozen=True)
class Provenance:
    """Wave 137c — operator/Codex/Qwen attribution + ToS audit trail.

    Required for anchored_status="complete" entries (Plan A's validator
    enforces). Optional for "degraded_placeholder" entries (no content
    yet to attribute).
    """

    provider: str          # qwen_local_14b_q4 / together_llama33_70b / operator_hand_curated
    generated_by: str      # draft_form_data_entry v1.0 / operator
    reviewed_by: str       # @mdmurphy822 / AUTOMATED / PENDING_REVIEW
    prompt_version: str    # wave-136c-v1.0 (only meaningful for Qwen-drafted)
    timestamp: str         # ISO-8601
    notes: Optional[str] = None


@dataclass
class SurfaceFormData:
    """Hand-curated data for one CURIE.

    Each list seeds one template family's emit — the family factories
    are thin templating around these strings. Authored from the
    canonical SHACL/RDF/RDFS/OWL specs; every entry must contain the
    primary CURIE literally (or the comparison/combination secondary
    CURIE in the off-axis fields) so ``preserve_tokens`` recognises
    the surface form.

    Wave 135a: ``anchored_status`` discriminator. ``"complete"`` entries
    carry real anchored content (their definitions / usage_examples
    are fit to be embedded in training pairs); ``"degraded_placeholder"``
    entries satisfy the structural contract via stub strings but MUST
    fall back to token-stuffing during force-injection (Wave 135b)
    with a WARN log + decision-capture event so operators see the
    degraded coverage. The 34 placeholder entries shipped in Wave 135a
    flip to ``"complete"`` over time as operator / Codex / Qwen backfills
    the anchored content per the project's "Claude does not generate
    training-data corpus content" operating principle.
    """

    curie: str
    short_name: str
    definitions: List[str] = field(default_factory=list)
    usage_examples: List[Tuple[str, str]] = field(default_factory=list)
    # (other_curie, contrast_explanation including both CURIEs)
    comparison_targets: List[Tuple[str, str]] = field(default_factory=list)
    # (scenario_prompt, completion_with_curie)
    reasoning_scenarios: List[Tuple[str, str]] = field(default_factory=list)
    # (scenario_prompt, completion_with_curie)
    pitfalls: List[Tuple[str, str]] = field(default_factory=list)
    # (other_curie, composition_explanation including both CURIEs)
    combinations: List[Tuple[str, str]] = field(default_factory=list)
    # Wave 135a: discriminator. Default "complete" preserves the
    # six pre-Wave-135a entries' shape without reauthoring.
    anchored_status: Literal["complete", "degraded_placeholder"] = "complete"
    # Wave 137c: optional operator/Codex/Qwen provenance block.
    # Populated only by the YAML loader (when overlay carries a
    # provenance block); the in-Python fallback dict's entries leave
    # this None, and Plan A's validator enforces presence on
    # anchored_status="complete" entries at backfill time.
    provenance: Optional[Provenance] = None


# -----------------------------------------------------------------------------
# Hand-curated catalog. ~7 entries per family per surface form.
# -----------------------------------------------------------------------------
#
# Authoring rules (Wave 125b):
#   * Every prompt probes a meaningfully different angle from its
#     siblings. No "what is X?" / "tell me about X" / "describe X"
#     thesaurus chains.
#   * Every completion contains the primary CURIE literally; comparison
#     and combination completions also contain the secondary CURIE.
#   * Length budget: prompt 40-400 chars, completion 50-600 chars.
#   * Authoritative answers cite SHACL/RDF spec terminology (focus
#     node, value node, target, IRI, literal, instance, ...).
#
# Total: 6 forms x 6 families x 7 entries = 252 base. Trim 2 in
# generate_schema_translation_pairs() to land at exactly 250.

_RDF_SHACL_FALLBACK_FORM_DATA: Dict[str, SurfaceFormData] = {
    "sh:datatype": SurfaceFormData(
        curie="sh:datatype",
        short_name="datatype",
        definitions=[
            # Formal-spec rendering.
            "sh:datatype is a SHACL property-shape constraint component that restricts each value node of the constrained property to RDF literals whose datatype IRI matches a given XSD or user-defined datatype IRI.",
            # Pedagogical rendering.
            "Think of sh:datatype as the SHACL way to say 'this property must be a typed literal.' It names a datatype IRI, and any value that isn't a literal of that datatype produces a sh:Violation at validation time.",
            # Context-anchored rendering ('where it lives').
            "sh:datatype lives on a sh:PropertyShape. It only constrains literal values — IRIs and blank nodes always fail sh:datatype because they aren't literals at all, regardless of any annotation.",
            # One-line spec quote.
            "sh:datatype: a single-IRI parameter that constrains value nodes to RDF literals carrying the named datatype IRI in their lexical-to-value-space mapping.",
            # Detailed authoritative rendering.
            "sh:datatype is the canonical SHACL constraint for literal-value typing. The validator checks each value node's datatype IRI; an empty datatype, an IRI value, or a literal of a different datatype fails the sh:datatype check and emits a violation result.",
            # Comparison-anchored definition.
            "sh:datatype constrains the lexical typing of literal values, in contrast to sh:nodeKind (which constrains node kind) and sh:class (which constrains class membership). Use sh:datatype when you need a typed literal specifically.",
            # Practitioner rendering.
            "When a SHACL shape needs 'this property must be xsd:integer' (or xsd:date, xsd:string, etc.), the constraint is sh:datatype. It is evaluated per value node and produces one violation per non-conforming value.",
        ],
        usage_examples=[
            (
                "Show how sh:datatype restricts a property to integer literals.",
                "On a sh:PropertyShape with sh:path ex:age, write `sh:datatype xsd:integer .` Every value of ex:age must then be a literal whose datatype IRI is xsd:integer; a string literal `\"42\"` or an IRI value fails the sh:datatype constraint.",
            ),
            (
                "Give a SHACL example using sh:datatype with xsd:date.",
                "`ex:EventShape a sh:NodeShape ; sh:property [ sh:path ex:occursOn ; sh:datatype xsd:date ] .` Now `\"2026-04-30\"^^xsd:date` validates, but `\"2026-04-30\"^^xsd:string` fails the sh:datatype check.",
            ),
            (
                "Demonstrate sh:datatype combined with sh:minCount.",
                "`[ sh:path ex:email ; sh:datatype xsd:string ; sh:minCount 1 ]` requires at least one value AND each value to be an xsd:string literal. The two constraints validate independently against the same focus node's value set under sh:datatype.",
            ),
            (
                "How does sh:datatype look applied to xsd:boolean?",
                "`[ sh:path ex:isActive ; sh:datatype xsd:boolean ]` admits `\"true\"^^xsd:boolean` / `\"false\"^^xsd:boolean` and rejects the strings `\"yes\"` / `\"no\"`. sh:datatype matches the datatype IRI exactly — no string-to-boolean coercion.",
            ),
            (
                "Apply sh:datatype with a custom user-defined datatype.",
                "Declaring `ex:HexColorCode` as a user datatype, you write `[ sh:path ex:colorCode ; sh:datatype ex:HexColorCode ]`. sh:datatype simply names the datatype IRI; SHACL doesn't care whether it's an XSD primitive or a custom one.",
            ),
            (
                "Write a SHACL shape using sh:datatype on xsd:decimal.",
                "`[ sh:path ex:price ; sh:datatype xsd:decimal ; sh:minInclusive 0 ]` — every value of ex:price must be an xsd:decimal literal AND ≥ 0. sh:datatype runs first; sh:minInclusive only fires once typing is satisfied.",
            ),
            (
                "Show sh:datatype with the rdf:langString datatype.",
                "`[ sh:path ex:label ; sh:datatype rdf:langString ]` requires every value to be a language-tagged literal (`\"hello\"@en`). Plain xsd:string literals fail because their datatype IRI differs from rdf:langString in sh:datatype.",
            ),
        ],
        comparison_targets=[
            (
                "sh:class",
                "sh:datatype constrains literal values to a named datatype IRI; sh:class constrains IRI- or blank-node values to instances of a class. A focus node value can pass exactly one of them — datatype rejects IRIs, sh:class rejects literals.",
            ),
            (
                "sh:nodeKind",
                "sh:nodeKind constrains the kind of RDF node (IRI / BlankNode / Literal); sh:datatype only fires once you've decided the value is a literal AND you need to pin its datatype IRI. sh:datatype is a refinement of `sh:nodeKind sh:Literal`.",
            ),
            (
                "sh:pattern",
                "sh:pattern matches a literal's lexical form against a regex; sh:datatype matches the literal's datatype IRI. Both can co-exist on the same property shape — sh:datatype filters by type, sh:pattern then constrains the lexical content.",
            ),
            (
                "sh:in",
                "sh:in pins the value to a closed enumeration of specific RDF terms; sh:datatype only restricts the datatype IRI of literals. sh:in is value-equality; sh:datatype is type-equality. A property can use both for typed enums.",
            ),
            (
                "sh:minCount",
                "sh:minCount counts how many values the property has; sh:datatype validates each individual value's literal type. They answer different questions — cardinality vs typing — and most property shapes use both together.",
            ),
            (
                "rdfs:range",
                "rdfs:range is an RDFS schema axiom that lets reasoners infer the type of values; sh:datatype is a SHACL constraint that REJECTS values failing the typing check. RDFS infers; SHACL validates. They serve opposing pipelines.",
            ),
            (
                "sh:PropertyShape",
                "sh:PropertyShape is the container; sh:datatype is one of the many constraint components that can ride inside one. A sh:PropertyShape without sh:datatype is fine — it just doesn't constrain the literal datatype of value nodes.",
            ),
        ],
        reasoning_scenarios=[
            (
                "You need to require that ex:height values are numeric (decimal) literals. Which SHACL constraint do you use?",
                "Use sh:datatype xsd:decimal on a sh:PropertyShape with sh:path ex:height. sh:datatype is the SHACL component that pins literal datatype IRIs; xsd:decimal is the right typing for arbitrary-precision numerics.",
            ),
            (
                "When does sh:datatype help over a free-form sh:nodeKind sh:Literal?",
                "sh:datatype helps when the application needs a SPECIFIC literal datatype (xsd:integer, xsd:date) rather than 'any literal'. sh:nodeKind admits any literal regardless of typing; sh:datatype rejects mistyped literals at validation time.",
            ),
            (
                "What's the trade-off between sh:datatype and sh:pattern when typing email addresses?",
                "sh:datatype xsd:string only checks that the value is an xsd:string literal — anything goes lexically. Add sh:pattern to constrain the lexical form to an email regex. sh:datatype handles type, sh:pattern handles syntax. Use both.",
            ),
            (
                "An IRI value mistakenly slipped into a property. Which constraint catches it earliest?",
                "sh:datatype catches it: any value that is not a literal fails sh:datatype regardless of the named datatype IRI. So `sh:datatype xsd:string` on the property shape is sufficient to reject IRI values at validation time.",
            ),
            (
                "You want to allow either xsd:integer or xsd:decimal — not just one. Can sh:datatype handle that alone?",
                "No — sh:datatype takes a single IRI. To admit a union of datatypes, wrap two property-shape branches in sh:or, each carrying its own sh:datatype. sh:datatype itself is single-valued by spec.",
            ),
            (
                "You need to enforce an xsd:date format ON TOP OF xsd:date typing. Which constraints?",
                "sh:datatype xsd:date pins the datatype IRI; xsd:date already enforces the YYYY-MM-DD lexical form via XSD value-space rules, so sh:pattern is usually redundant. The validator surfaces a typing violation if a malformed lexical form is presented.",
            ),
            (
                "Why might sh:datatype be preferred over sh:class for a property with literal values?",
                "sh:class always fails on literals — it tests rdf:type membership on IRI / blank-node values. sh:datatype is the spec-correct construct for literal-typed properties. Picking the wrong one produces 100% false-positive violations.",
            ),
        ],
        pitfalls=[
            (
                "Why does sh:datatype fail when applied to an IRI value?",
                "sh:datatype checks the datatype IRI of literal values; an IRI value is not a literal, so the check fails by construction. The validator emits a sourceConstraintComponent of sh:DatatypeConstraintComponent. Use sh:nodeKind or sh:class for IRI values.",
            ),
            (
                "What's the common mistake when combining sh:datatype with sh:in?",
                "Authors sometimes forget that sh:in matches RDF terms exactly — including their datatype. Listing `\"5\"` (a plain string) under sh:in while sh:datatype demands xsd:integer makes the enumeration unreachable. List `\"5\"^^xsd:integer` instead.",
            ),
            (
                "What goes wrong if sh:datatype is omitted from a literal-typed property shape?",
                "Without sh:datatype, the property accepts ANY literal (or even non-literal) value. Downstream consumers expecting xsd:integer crash on stray xsd:string values. The shape passes SHACL validation but doesn't enforce the typing contract.",
            ),
            (
                "Why doesn't sh:datatype xsd:string match a language-tagged literal?",
                "Language-tagged literals carry the rdf:langString datatype IRI, not xsd:string — they're a distinct datatype in RDF 1.1. `sh:datatype xsd:string` rejects `\"hello\"@en`. Use `sh:datatype rdf:langString` (or sh:or both) to admit language tags.",
            ),
            (
                "sh:datatype xsd:integer is reporting violations on values like 42. Why?",
                "Likely the values are lexically `\"42\"` with NO datatype tag — defaulting to xsd:string. Either add the `^^xsd:integer` annotation in the data, or relax sh:datatype to xsd:string. sh:datatype matches the datatype IRI exactly, no coercion.",
            ),
            (
                "What happens when sh:datatype is assigned a non-IRI value?",
                "sh:datatype expects a single IRI. A literal or blank node as the parameter produces an ill-formed shape — `pyshacl` raises a ShapeLoadError at parse time. The constraint never validates because the shape itself is invalid by sh:datatype's signature.",
            ),
            (
                "Why does adding sh:datatype xsd:string make sh:class ex:Person redundant?",
                "It doesn't make sh:class redundant — it makes the shape contradictory. sh:datatype demands literals; sh:class demands IRIs/blank nodes. Together they reject everything. Pick one: sh:datatype for literals, sh:class for class instances.",
            ),
        ],
        combinations=[
            (
                "sh:minCount",
                "sh:datatype + sh:minCount: typed cardinality. `sh:datatype xsd:integer ; sh:minCount 1` requires at least one value AND every value to be an integer literal. The two constraints validate independently against the same focus property.",
            ),
            (
                "sh:maxCount",
                "sh:datatype with sh:maxCount caps how many typed values a property has. `sh:datatype xsd:date ; sh:maxCount 1` is the SHACL idiom for 'optional single date.' Both fire per-focus-node.",
            ),
            (
                "sh:pattern",
                "sh:datatype + sh:pattern: type then lexical-content check. `sh:datatype xsd:string ; sh:pattern \"^[A-Z]{3}$\"` requires a 3-letter uppercase string literal. sh:datatype gates type; sh:pattern then constrains the lexical form.",
            ),
            (
                "sh:in",
                "sh:datatype + sh:in: typed enum. `sh:datatype xsd:string ; sh:in (\"red\" \"green\" \"blue\")` requires a string literal AND one of three exact terms. The enum's datatype must match sh:datatype or no value can pass.",
            ),
            (
                "sh:minInclusive",
                "sh:datatype + sh:minInclusive: typed numeric range. `sh:datatype xsd:decimal ; sh:minInclusive 0` requires a decimal literal ≥ 0. sh:minInclusive needs a comparable datatype, so sh:datatype is the gating constraint.",
            ),
            (
                "sh:or",
                "sh:datatype inside sh:or branches admits a union of datatypes. `sh:or ( [ sh:datatype xsd:integer ] [ sh:datatype xsd:decimal ] )` accepts either integer or decimal literals — needed because a single sh:datatype can name only one IRI.",
            ),
            (
                "sh:NodeShape",
                "sh:datatype lives inside a sh:PropertyShape, which is referenced from a sh:NodeShape via sh:property. So a NodeShape ex:Person → sh:property → [ sh:path ex:age ; sh:datatype xsd:integer ] is the canonical chain wiring sh:datatype onto a class.",
            ),
        ],
        anchored_status="complete",
    ),
    "sh:class": SurfaceFormData(
        curie="sh:class",
        short_name="class",
        definitions=[
            "sh:class is a SHACL property-shape constraint component that requires each value node of the constrained property to be a SHACL instance of a given class IRI, considering rdfs:subClassOf transitively.",
            "Use sh:class when a property must point at instances of a particular class. The validator checks that every value node has an rdf:type chain reaching the named class via rdfs:subClassOf — a literal value always fails sh:class because literals aren't class instances.",
            "sh:class is the SHACL way to require 'this value is an instance of class C.' It expects an IRI naming the required class. RDFS-aware reasoners propagate types through subclass hierarchies, so a value typed as a subclass also passes sh:class.",
            "sh:class: a single-IRI parameter naming a class; value nodes pass when they are SHACL instances (rdf:type ∪ transitive rdfs:subClassOf closure) of that class.",
            "Formally, sh:class C accepts a value node v iff there exists a path v rdf:type ?T (rdfs:subClassOf)* C in the data graph. Failure produces a sh:ClassConstraintComponent violation result.",
            "sh:class targets IRI and blank-node values; literals always fail. Pair sh:class with sh:nodeKind sh:IRI when you also need to forbid blank nodes — sh:class alone admits both IRI and blank-node instances.",
            "If you've ever written `?subj rdf:type ex:Person` in SPARQL, sh:class ex:Person is the SHACL counterpart: same membership check, run as part of validation rather than query.",
        ],
        usage_examples=[
            (
                "Show how sh:class restricts a property to ex:Person instances.",
                "On a sh:PropertyShape with sh:path ex:author, write `sh:class ex:Person .` Every value of ex:author must then have rdf:type ex:Person (or a subclass). A literal value or an instance of an unrelated class fails the sh:class check.",
            ),
            (
                "Give a SHACL example with sh:class chained through rdfs:subClassOf.",
                "If `ex:Student rdfs:subClassOf ex:Person`, then `[ sh:path ex:author ; sh:class ex:Person ]` admits values typed `ex:Student` because rdfs:subClassOf entailment puts them in the sh:class extension.",
            ),
            (
                "Demonstrate sh:class combined with sh:minCount.",
                "`[ sh:path ex:advisor ; sh:class ex:Faculty ; sh:minCount 1 ]` requires at least one advisor AND every advisor to be a SHACL instance of ex:Faculty. The two constraints validate independently in sh:class semantics.",
            ),
            (
                "How does sh:class interact with blank-node values?",
                "Blank nodes pass sh:class as long as they carry the right rdf:type triple. `_:b1 rdf:type ex:Address` makes _:b1 a valid value for `[ sh:path ex:home ; sh:class ex:Address ]`. Pair with sh:nodeKind sh:IRI to require an IRI.",
            ),
            (
                "Apply sh:class to require a value typed as ex:Organization.",
                "`ex:EmployeeShape sh:property [ sh:path ex:employer ; sh:class ex:Organization ; sh:minCount 1 ] .` Each employer value must be typed ex:Organization; an untyped IRI or a literal value fails the sh:class constraint.",
            ),
            (
                "Write a shape using sh:class on a hierarchy root.",
                "`[ sh:path ex:topic ; sh:class ex:Subject ]` where ex:MathSubject and ex:LangSubject both rdfs:subClassOf ex:Subject. Values of either subclass pass sh:class because rdfs:subClassOf transitivity is part of the sh:class semantics.",
            ),
            (
                "Show sh:class with multiple property shapes targeting the same class.",
                "`ex:CourseShape sh:targetClass ex:Course ; sh:property [ sh:path ex:teacher ; sh:class ex:Faculty ] ; sh:property [ sh:path ex:enrolls ; sh:class ex:Student ] .` Two sh:class constraints, each scoping a different relation.",
            ),
        ],
        comparison_targets=[
            (
                "sh:datatype",
                "sh:class constrains IRI/blank-node values to class membership; sh:datatype constrains literal values to a named datatype. Mutually exclusive in practice — a value passes one or the other, never both.",
            ),
            (
                "sh:nodeKind",
                "sh:nodeKind says 'the value is an IRI / blank node / literal' — about the kind of node. sh:class says 'the value is a member of class C' — about typing. sh:class implies sh:nodeKind sh:BlankNodeOrIRI, but adds a typing requirement.",
            ),
            (
                "rdf:type",
                "rdf:type is the data-graph predicate that asserts an instance's class. sh:class is a SHACL constraint that VALIDATES that assertion exists (transitively) on each value node. One declares; the other audits.",
            ),
            (
                "rdfs:subClassOf",
                "rdfs:subClassOf is the RDFS axiom for class hierarchy. sh:class CONSUMES rdfs:subClassOf transitively — `sh:class Person` admits Student instances when Student rdfs:subClassOf Person. The two cooperate at validation time.",
            ),
            (
                "sh:targetClass",
                "sh:targetClass picks WHICH focus nodes a shape applies to (instances of a class). sh:class constrains WHAT VALUES of a property must be (also instances of a class). One scopes the shape; the other constrains values inside it.",
            ),
            (
                "sh:in",
                "sh:in pins the value to specific RDF terms (closed enum); sh:class admits any instance of an open class. sh:class is open-world; sh:in is closed-world. Use sh:in when you need exact terms, sh:class for class-membership.",
            ),
            (
                "owl:Class",
                "owl:Class declares an OWL class. sh:class references that class by IRI to constrain values. SHACL doesn't require the class to be declared owl:Class — any IRI used as an rdf:type works under sh:class.",
            ),
        ],
        reasoning_scenarios=[
            (
                "You need a property's values to be instances of ex:Faculty. Which SHACL constraint do you use?",
                "Use sh:class ex:Faculty on the property shape. sh:class is the SHACL component for class-membership constraints; ex:Faculty must be an IRI naming the required class. RDFS subclass inference is honored by default.",
            ),
            (
                "When does sh:class help over a free sh:nodeKind sh:IRI?",
                "sh:class is needed when the values must be of a specific TYPE, not just any IRI. sh:nodeKind sh:IRI admits any IRI — a Course IRI passes the same as a Person IRI. sh:class ex:Person rejects the Course value.",
            ),
            (
                "What's the trade-off between sh:class and sh:in for restricting committee members?",
                "sh:class admits any instance of ex:CommitteeMember as new ones are added — open-world. sh:in pins to a specific list of IRIs and rejects new ones — closed-world. Use sh:class for growing populations, sh:in for fixed enums.",
            ),
            (
                "A SPARQL CONSTRUCT inserted untyped IRIs into ex:advisor. Which constraint catches them?",
                "sh:class on the ex:advisor property shape catches them. Without rdf:type triples reaching ex:Faculty (transitively via rdfs:subClassOf), the new IRIs fail sh:class and the validator emits sh:ClassConstraintComponent violations.",
            ),
            (
                "You want to allow ex:Student OR ex:Faculty values. How do you combine constraints?",
                "Either declare a common superclass (`ex:Person`) and use `sh:class ex:Person` so subclass entailment admits both, or branch with sh:or: `sh:or ( [ sh:class ex:Student ] [ sh:class ex:Faculty ] )`. sh:class itself names only one class.",
            ),
            (
                "Why might sh:class be preferred over sh:datatype for a property pointing at people?",
                "People are IRI-typed entities, not literal values. sh:datatype only checks literal datatypes — it would reject every Person IRI. sh:class ex:Person is the spec-correct constraint for IRI-valued person properties.",
            ),
            (
                "Validation says `sh:ClassConstraintComponent` violation. What does that mean?",
                "It means a value node failed sh:class — it's either a literal, or an IRI/blank node whose rdf:type closure doesn't reach the named class. Inspect the value's rdf:type triples and the rdfs:subClassOf hierarchy to debug.",
            ),
        ],
        pitfalls=[
            (
                "Why does sh:class fail when applied to a literal value?",
                "sh:class checks rdf:type membership, which is only defined for IRI and blank-node subjects. A literal value can't be an instance of a class in RDF, so sh:class always rejects literals. Use sh:datatype for literals.",
            ),
            (
                "What's the common mistake with sh:class and rdf:type?",
                "Authors forget that sh:class needs rdf:type to be ASSERTED on the value. An untyped IRI fails sh:class even when it 'obviously' represents a Person — SHACL doesn't fall back to OWL inference unless rdfs:subClassOf entailment is enabled.",
            ),
            (
                "What goes wrong if sh:class is omitted from a class-typed property shape?",
                "Without sh:class, the property admits any IRI, blank node, or literal. The shape passes validation but doesn't enforce the type contract — a Course IRI accidentally placed under ex:advisor goes undetected.",
            ),
            (
                "Why doesn't sh:class catch values typed only with rdfs:subClassOf?",
                "sh:class follows rdfs:subClassOf transitively from rdf:type — but it needs a starting rdf:type. If the value has only rdfs:subClassOf assertions (no rdf:type), there's no entry into the type chain and sh:class fails. Add rdf:type.",
            ),
            (
                "sh:class ex:Person reports violations on what looks like person IRIs. Why?",
                "Likely the IRIs lack the rdf:type ex:Person triple, OR ex:Person isn't the inferred class via rdfs:subClassOf. SHACL doesn't infer types from naming — `:Alice` won't pass sh:class ex:Person without an explicit type triple.",
            ),
            (
                "What happens when sh:class is assigned a non-IRI parameter?",
                "sh:class expects exactly one IRI. A literal or blank node breaks the shape — pyshacl raises a ShapeLoadError at parse. The constraint never reaches validation because sh:class's signature is unsatisfied.",
            ),
            (
                "Why is `sh:class ex:Person ; sh:nodeKind sh:Literal` always failing?",
                "It's contradictory: sh:class demands IRI/blank-node values; sh:nodeKind sh:Literal demands literal values. No value can pass both — the property shape rejects everything. Drop sh:nodeKind or switch to sh:datatype.",
            ),
        ],
        combinations=[
            (
                "sh:nodeKind",
                "sh:class + sh:nodeKind sh:IRI: 'value must be an IRI AND an instance of class C.' Forbids blank-node values that would otherwise pass sh:class. Common in published vocabularies where blank-node identity is undesirable.",
            ),
            (
                "sh:minCount",
                "sh:class + sh:minCount: typed cardinality on objects. `sh:class ex:Author ; sh:minCount 1` requires at least one author AND each author to be a SHACL instance of ex:Author.",
            ),
            (
                "sh:maxCount",
                "sh:class + sh:maxCount: caps typed values. `sh:class ex:Department ; sh:maxCount 1` says 'at most one department, and it must be typed as such.' Useful for many-to-one relations.",
            ),
            (
                "rdfs:subClassOf",
                "sh:class works WITH rdfs:subClassOf via SHACL's class-membership semantics: declaring `ex:Faculty rdfs:subClassOf ex:Person` lets a Faculty instance pass `sh:class ex:Person`. The two are designed to compose at validation time.",
            ),
            (
                "sh:targetClass",
                "sh:class on values + sh:targetClass on the shape: focus nodes are instances of one class, value nodes must be instances of another. Common pattern: `sh:targetClass ex:Course ; sh:property [ sh:class ex:Faculty ]` types course teachers.",
            ),
            (
                "sh:NodeShape",
                "sh:class names a class to constrain VALUES. sh:NodeShape is a SHAPE that can target a class. They aren't redundant — sh:class restricts who passes; sh:NodeShape decides what shape to apply. They sit at different layers.",
            ),
            (
                "sh:property",
                "sh:class is a constraint inside a sh:PropertyShape, which is referenced via sh:property from a sh:NodeShape. The full chain: NodeShape → sh:property → PropertyShape with sh:class → constrained values.",
            ),
        ],
        anchored_status="complete",
    ),
    "sh:NodeShape": SurfaceFormData(
        curie="sh:NodeShape",
        short_name="node shape",
        definitions=[
            "sh:NodeShape is the SHACL class of shapes that constrain RDF nodes themselves rather than the values of a single property. A node shape lists constraints (sh:property, sh:nodeKind, sh:targetClass, ...) that the focus node must satisfy.",
            "Think of sh:NodeShape as the SHACL counterpart of a class-level schema: it bundles all constraints that must hold for whole nodes — typically all instances of a target class — into one declarative shape.",
            "A sh:NodeShape is the entry point for SHACL validation against a focus node. Every constraint inside fires against that node directly; nested sh:property links extend the contract to the node's outgoing properties.",
            "sh:NodeShape: the RDF class whose instances are SHACL shapes that apply node-level constraints. Marked via `rdf:type sh:NodeShape` or implicitly by carrying sh:targetClass / sh:targetNode triples.",
            "Formally, a sh:NodeShape S is a SHACL shape that, when validated against focus node f, fires every constraint component listed on S directly against f (not against f's property values). Property-level checks are nested via sh:property.",
            "If sh:PropertyShape is for 'each value of property P,' then sh:NodeShape is for 'each whole node F.' Both are SHACL shapes; the difference is which nodes constraints fire against.",
            "In practice, an `ex:PersonShape a sh:NodeShape ; sh:targetClass ex:Person ; sh:property [...]` declares the shape that every Person instance must pass at SHACL validation.",
        ],
        usage_examples=[
            (
                "Show a minimal sh:NodeShape declaration.",
                "`ex:PersonShape a sh:NodeShape ; sh:targetClass ex:Person ; sh:property [ sh:path ex:name ; sh:datatype xsd:string ; sh:minCount 1 ] .` Every ex:Person instance is then validated by ex:PersonShape as a sh:NodeShape.",
            ),
            (
                "Demonstrate sh:NodeShape with sh:nodeKind on the focus node.",
                "`ex:IRIPersonShape a sh:NodeShape ; sh:targetClass ex:Person ; sh:nodeKind sh:IRI .` This requires every Person focus node to be an IRI (no blank-node Persons). sh:nodeKind here applies to the focus node itself, not values.",
            ),
            (
                "Apply sh:NodeShape with multiple sh:property bindings.",
                "`ex:CourseShape a sh:NodeShape ; sh:targetClass ex:Course ; sh:property [ sh:path ex:title ; sh:minCount 1 ] ; sh:property [ sh:path ex:credits ; sh:datatype xsd:integer ] .` Two property checks per Course focus node.",
            ),
            (
                "How does sh:NodeShape carry sh:closed in a tight-schema scenario?",
                "`ex:StrictShape a sh:NodeShape ; sh:targetClass ex:Strict ; sh:closed true ; sh:property [ sh:path ex:name ] .` Closed sh:NodeShape rejects any predicate on the focus node not declared via sh:property — useful for tight schemas.",
            ),
            (
                "Show a sh:NodeShape using sh:targetNode.",
                "`ex:RootShape a sh:NodeShape ; sh:targetNode ex:Alice ; sh:property [ sh:path ex:age ; sh:datatype xsd:integer ] .` Targets exactly ex:Alice as the focus node — useful when the shape applies to a single named individual.",
            ),
            (
                "Combine sh:NodeShape with sh:and to compose two reusable shapes.",
                "`ex:CombinedShape a sh:NodeShape ; sh:targetClass ex:User ; sh:and ( ex:HasNameShape ex:HasAgeShape ) .` The focus user passes only when both nested shapes hold — sh:and on a sh:NodeShape composes constraints declaratively.",
            ),
            (
                "Use sh:NodeShape with sh:targetSubjectsOf.",
                "`ex:AuthorShape a sh:NodeShape ; sh:targetSubjectsOf ex:wrote ; sh:property [ sh:path foaf:name ; sh:minCount 1 ] .` Every subject of an ex:wrote triple becomes a focus node for this sh:NodeShape.",
            ),
        ],
        comparison_targets=[
            (
                "sh:PropertyShape",
                "sh:NodeShape constrains whole nodes; sh:PropertyShape constrains values of a single property. PropertyShapes nest inside NodeShapes via sh:property — they're complementary layers of the SHACL shape model.",
            ),
            (
                "sh:targetClass",
                "sh:NodeShape is the shape class; sh:targetClass is a triggering predicate that picks WHICH instances become focus nodes for the shape. A sh:NodeShape without sh:targetClass / sh:targetNode applies to nothing implicitly.",
            ),
            (
                "owl:Class",
                "owl:Class declares a domain class; sh:NodeShape declares VALIDATION RULES for instances of that class. They serve different layers — schema vs validation — and a class can have many sh:NodeShape definitions.",
            ),
            (
                "sh:nodeKind",
                "sh:nodeKind is a constraint component (a value); sh:NodeShape is a class of shapes (a node). sh:nodeKind can ride INSIDE a sh:NodeShape to constrain the focus node's kind. Different roles in the ontology.",
            ),
            (
                "rdfs:Class",
                "rdfs:Class is RDFS schema-level — declares classes. sh:NodeShape is SHACL validation-level — declares shapes. A SHACL validator doesn't need rdfs:Class to validate; it needs the sh:NodeShape and target predicates.",
            ),
            (
                "sh:targetNode",
                "sh:targetNode picks one specific focus node; sh:NodeShape is the class of shapes that consumes target predicates. sh:targetNode is one of several ways to feed focus nodes into a sh:NodeShape.",
            ),
            (
                "sh:closed",
                "sh:closed is a constraint that, when set on a sh:NodeShape, rejects un-declared predicates on the focus node. sh:NodeShape provides the scope; sh:closed tightens it. The constraint is meaningless outside a node-shape context.",
            ),
        ],
        reasoning_scenarios=[
            (
                "You need to validate every ex:Person in the data graph. Which SHACL shape class do you use?",
                "Use sh:NodeShape with sh:targetClass ex:Person. sh:NodeShape is the SHACL shape class for whole-node validation; sh:targetClass picks instances of ex:Person as focus nodes for the shape.",
            ),
            (
                "When does using sh:NodeShape help over multiple stand-alone sh:PropertyShapes?",
                "sh:NodeShape gives you ONE declarative scope per node — sh:targetClass once, sh:closed once, and a list of sh:property links inside. Stand-alone sh:PropertyShapes fragment the contract across many shapes; sh:NodeShape consolidates it.",
            ),
            (
                "What's the trade-off between sh:NodeShape with sh:targetClass vs sh:NodeShape with sh:targetSubjectsOf?",
                "On a sh:NodeShape, sh:targetClass picks instances by rdf:type; sh:targetSubjectsOf picks subjects of a given predicate. Use targetClass for class-level shapes, targetSubjectsOf when membership is defined by a relation rather than a type.",
            ),
            (
                "You want to apply a SHACL contract only to ex:Alice. Which shape class fits?",
                "sh:NodeShape with sh:targetNode ex:Alice. sh:NodeShape applies to whole nodes; sh:targetNode names exactly one focus node. The contents of the shape (sh:property, sh:nodeKind, etc.) then validate ex:Alice specifically.",
            ),
            (
                "You need to forbid unknown predicates on a focus node. Which constraint goes on the sh:NodeShape?",
                "Set sh:closed true on the sh:NodeShape, optionally with sh:ignoredProperties to whitelist some. sh:closed is a node-level constraint; without sh:NodeShape providing scope, sh:closed has no focus to apply to.",
            ),
            (
                "Why might sh:NodeShape be preferred over sh:PropertyShape for class-level rules?",
                "Class-level rules belong on a sh:NodeShape: node-level checks like sh:closed, sh:nodeKind on the focus, sh:targetClass scoping. sh:PropertyShape can't carry sh:targetClass and only constrains property values, not the focus.",
            ),
            (
                "Validation says the focus node failed an sh:NodeShape. What does that mean?",
                "The focus node failed at least one constraint listed in the sh:NodeShape — could be sh:closed (unknown predicate), sh:nodeKind (wrong kind), or any nested sh:property check. Inspect the violation result's sh:sourceConstraintComponent.",
            ),
        ],
        pitfalls=[
            (
                "Why does a sh:NodeShape with NO sh:targetClass / sh:targetNode validate nothing?",
                "Without a target predicate, the sh:NodeShape has no implicit focus nodes — it's a defined-but-unused shape. The validator skips it. Add sh:targetClass, sh:targetNode, sh:targetSubjectsOf, or sh:targetObjectsOf to feed focus nodes.",
            ),
            (
                "What's the common mistake with sh:NodeShape and rdf:type?",
                "Authors forget the `a sh:NodeShape` triple — declaring sh:targetClass alone DOES make the shape a sh:NodeShape implicitly under SHACL semantics, but explicit typing makes the shape graph cleaner and easier to debug.",
            ),
            (
                "What goes wrong if a sh:NodeShape is omitted and only sh:PropertyShapes are declared?",
                "PropertyShapes alone can't carry sh:targetClass or sh:closed. Without a sh:NodeShape wrapper, you lose class-level scoping and unknown-predicate detection. Validation fragments into per-property checks with no whole-node coherence.",
            ),
            (
                "Why doesn't sh:NodeShape automatically validate property values?",
                "sh:NodeShape constrains the FOCUS NODE itself. To validate property values, the sh:NodeShape must reference a sh:PropertyShape via sh:property. Forgetting the link means the property layer goes unchecked.",
            ),
            (
                "An sh:NodeShape with sh:closed true is rejecting valid data. Why?",
                "On a closed sh:NodeShape, sh:closed forbids predicates not declared via sh:property. Adding new predicates (rdfs:label, dcterms:created) without listing them or adding them to sh:ignoredProperties trips the closure. Either list them or relax sh:closed.",
            ),
            (
                "What happens when two sh:NodeShape instances both target the same class?",
                "Both sh:NodeShape instances fire against every instance — SHACL semantics is conjunctive. The instance must pass both. This is intentional (lets you compose shapes); it can also surprise authors expecting the second shape to override the first.",
            ),
            (
                "Why is sh:NodeShape sometimes confused with rdfs:Class?",
                "Both are 'class-shaped' RDF resources, but rdfs:Class is the schema for domain classes and sh:NodeShape is the schema for VALIDATION shapes. A sh:NodeShape doesn't make its instances members of a domain — it makes them subjects of validation.",
            ),
        ],
        combinations=[
            (
                "sh:property",
                "sh:NodeShape + sh:property: the canonical wiring. The NodeShape's sh:property links pick PropertyShapes that fire against the focus node's outgoing properties. Together they form the two-layer SHACL contract.",
            ),
            (
                "sh:targetClass",
                "sh:NodeShape + sh:targetClass: scoped validation. The NodeShape applies to every instance of the named class via SHACL targeting semantics. The most common SHACL pattern in published shape graphs.",
            ),
            (
                "sh:closed",
                "sh:NodeShape + sh:closed: tight schema. Closed node shapes reject any predicate on the focus node not listed in sh:property. Combined with sh:ignoredProperties for an allow-list of metadata predicates.",
            ),
            (
                "sh:and",
                "sh:NodeShape + sh:and ( shape1 shape2 ): composed contract. The focus node passes when every nested shape passes. Useful for layering reusable shape fragments without duplicating constraints.",
            ),
            (
                "sh:or",
                "sh:NodeShape + sh:or ( shapeA shapeB ): disjunctive contract. The focus node passes when at least one branch passes — admits multiple valid 'flavors' of the same class without forking the shape graph.",
            ),
            (
                "sh:nodeKind",
                "sh:NodeShape + sh:nodeKind sh:IRI: identifier discipline. Forces every instance of the targeted class to be an IRI (no blank nodes). Critical for shapes whose instances must be linkable across datasets.",
            ),
            (
                "sh:targetSubjectsOf",
                "sh:NodeShape + sh:targetSubjectsOf ex:rel: relation-defined scope. Every subject of an ex:rel triple becomes a focus node — useful when membership is defined by participation in a relation rather than rdf:type.",
            ),
        ],
        anchored_status="complete",
    ),
    "sh:PropertyShape": SurfaceFormData(
        curie="sh:PropertyShape",
        short_name="property shape",
        definitions=[
            "sh:PropertyShape is the SHACL class of shapes that constrain the values of a single RDF property at a given focus node. A property shape names the constrained predicate via sh:path and lists value constraints (sh:datatype, sh:class, sh:minCount, ...).",
            "Think of sh:PropertyShape as the 'per-property block' of a SHACL contract. It fires constraints once per value found at the path predicate, producing one violation result per non-conforming value.",
            "A sh:PropertyShape requires sh:path; that path picks the predicate the shape constrains. Constraint components inside (sh:datatype, sh:class, sh:minCount, sh:pattern, ...) then validate the value set under that path.",
            "sh:PropertyShape: the RDF class whose instances are SHACL shapes scoped to one property. Carried via `rdf:type sh:PropertyShape` or implicit when an unnamed shape uses sh:path inside sh:property.",
            "Formally, a sh:PropertyShape S has a required sh:path P. Validation of S against focus node f fires every constraint component in S against the value set { v : (f, P, v) ∈ G } in the data graph.",
            "If sh:NodeShape is for whole-node validation, sh:PropertyShape is for per-edge validation. Both are SHACL shape classes; PropertyShapes nest inside NodeShapes via sh:property links.",
            "In practice, `ex:NameShape a sh:PropertyShape ; sh:path ex:name ; sh:datatype xsd:string ; sh:minCount 1 .` declares a per-property shape that can be referenced from any number of node shapes.",
        ],
        usage_examples=[
            (
                "Show a minimal sh:PropertyShape declaration.",
                "`ex:NameShape a sh:PropertyShape ; sh:path ex:name ; sh:datatype xsd:string ; sh:minCount 1 .` Validators apply this shape to whatever focus node references it via sh:property; each value of ex:name is checked.",
            ),
            (
                "Use sh:PropertyShape inline as a blank node.",
                "`ex:PersonShape sh:property [ sh:path ex:age ; sh:datatype xsd:integer ; sh:minInclusive 0 ] .` The blank-node sh:PropertyShape lives only inside the parent NodeShape — typical for one-off constraints.",
            ),
            (
                "Demonstrate sh:PropertyShape with sh:path as a sequence.",
                "`[ a sh:PropertyShape ; sh:path ( ex:friend ex:name ) ; sh:datatype xsd:string ]` follows ex:friend then ex:name on the focus node, validating each two-hop value as an xsd:string literal.",
            ),
            (
                "Apply sh:PropertyShape with sh:path inverse.",
                "`[ a sh:PropertyShape ; sh:path [ sh:inversePath ex:author ] ; sh:minCount 1 ]` requires the focus node to be the object of at least one ex:author triple — validates 'incoming' edges via inverse path.",
            ),
            (
                "Reuse a named sh:PropertyShape from multiple node shapes.",
                "`ex:EmailShape a sh:PropertyShape ; sh:path ex:email ; sh:datatype xsd:string ; sh:pattern \"@\" .` Then `ex:UserShape sh:property ex:EmailShape ; ex:CustomerShape sh:property ex:EmailShape .` Reuse via sh:property links.",
            ),
            (
                "Show sh:PropertyShape with sh:qualifiedValueShape.",
                "`[ a sh:PropertyShape ; sh:path ex:author ; sh:qualifiedValueShape ex:FacultyShape ; sh:qualifiedMinCount 1 ]` requires at least one ex:author value to satisfy ex:FacultyShape. Per-value qualified counts inside a sh:PropertyShape.",
            ),
            (
                "Combine sh:PropertyShape with sh:hasValue.",
                "`[ a sh:PropertyShape ; sh:path rdf:type ; sh:hasValue ex:ApprovedItem ]` requires the focus node to carry rdf:type ex:ApprovedItem among its values. sh:PropertyShape fits any path-based check, including rdf:type itself.",
            ),
        ],
        comparison_targets=[
            (
                "sh:NodeShape",
                "sh:PropertyShape constrains values of one property; sh:NodeShape constrains whole focus nodes. PropertyShapes nest inside NodeShapes via sh:property — two layers of the same SHACL shape model.",
            ),
            (
                "sh:path",
                "sh:path is a REQUIRED parameter on every sh:PropertyShape — it names which predicate the shape constrains. sh:PropertyShape without sh:path is malformed; the validator raises a shape-load error.",
            ),
            (
                "sh:property",
                "sh:property is the LINK from a sh:NodeShape to a sh:PropertyShape. The PropertyShape lives independently; sh:property is how the NodeShape pulls it into the validation chain for a focus node.",
            ),
            (
                "rdf:Property",
                "rdf:Property declares a domain-level predicate; sh:PropertyShape declares VALIDATION RULES for values of a predicate at a node. They sit in different layers — schema vs validation.",
            ),
            (
                "sh:targetClass",
                "sh:targetClass is a node-level target on sh:NodeShape — picks focus nodes by class. sh:PropertyShape can't carry sh:targetClass alone; PropertyShapes are scoped via sh:property from a NodeShape that owns the targeting.",
            ),
            (
                "sh:nodeKind",
                "sh:nodeKind is a constraint component that can ride inside a sh:PropertyShape (constraining value-node kind) OR a sh:NodeShape (constraining focus-node kind). Same component, different scope per host shape.",
            ),
            (
                "sh:datatype",
                "sh:datatype is a constraint component used INSIDE a sh:PropertyShape to constrain literal values. The PropertyShape provides the path scope; sh:datatype refines the per-value check.",
            ),
        ],
        reasoning_scenarios=[
            (
                "You need to constrain values of ex:age to non-negative integers. Which SHACL shape class do you declare?",
                "Declare a sh:PropertyShape with sh:path ex:age, sh:datatype xsd:integer, and sh:minInclusive 0. sh:PropertyShape is the SHACL shape class for per-property value constraints; the path scopes it to ex:age values only.",
            ),
            (
                "When does using sh:PropertyShape help over inlining everything in a sh:NodeShape?",
                "A named sh:PropertyShape can be REFERENCED from multiple node shapes — DRY across shape graphs. Inline blank-node property shapes work for one-off constraints but force duplication when the same property has the same rules elsewhere.",
            ),
            (
                "What's the trade-off between named sh:PropertyShape and inline blank-node property shapes?",
                "Named sh:PropertyShape resources are reusable + linkable; inline blank-node shapes are local to one NodeShape and read more cleanly for one-off uses. Pick named sh:PropertyShape for cross-shape reuse, inline for shape-specific constraints.",
            ),
            (
                "You want to validate paths longer than one predicate (e.g., friend's name). Which SHACL surface?",
                "A sh:PropertyShape with a sequence sh:path: `sh:path ( ex:friend ex:name )`. The PropertyShape's path can be any SHACL property path expression — sequence, alternative, inverse, zero-or-more.",
            ),
            (
                "You need per-value class checks beyond just sh:class. Which sh:PropertyShape construct?",
                "Use sh:qualifiedValueShape on a sh:PropertyShape. It applies a nested shape to each value individually with sh:qualifiedMinCount / sh:qualifiedMaxCount counts — finer-grained than a flat sh:class on the same property shape.",
            ),
            (
                "Why might sh:PropertyShape be preferred over sh:NodeShape for value-typing rules?",
                "Value-typing is per-property — sh:datatype, sh:class, sh:pattern, sh:minCount all apply to values at a path. sh:NodeShape can't carry sh:path, so it can't scope these constraints to one predicate alone. sh:PropertyShape is the natural host.",
            ),
            (
                "Validation reports a failure on a sh:PropertyShape. What's the next debug step?",
                "Inspect the violation's sh:focusNode (where the property check fired) + sh:resultPath (the predicate) + sh:value (the failing value). Then check which constraint component (sh:datatype, sh:minCount, ...) inside the sh:PropertyShape rejected it.",
            ),
        ],
        pitfalls=[
            (
                "Why does a sh:PropertyShape without sh:path fail to load?",
                "sh:path is a REQUIRED parameter on every sh:PropertyShape per the SHACL spec. Without it the validator raises a ShapeLoadError — there's no predicate to scope value checks against. Always declare sh:path first.",
            ),
            (
                "What's the common mistake nesting sh:PropertyShape inside sh:property?",
                "Authors sometimes write `sh:property ex:NameShape, ex:AgeShape` (a comma-separated list of sh:PropertyShape IRIs) when they mean two separate sh:property triples. The comma form is valid Turtle but assigns BOTH IRIs as values of one sh:property triple — semantics differ subtly.",
            ),
            (
                "What goes wrong if sh:PropertyShape is used standalone with no sh:property reference?",
                "The sh:PropertyShape exists in the shape graph but no NodeShape pulls it in — it never fires against any focus node. Always link it via `someNodeShape sh:property thePropertyShape` or via inline blank-node nesting.",
            ),
            (
                "Why doesn't sh:targetClass work on a bare sh:PropertyShape?",
                "Targets define focus-node selection at the NODE layer, not the property layer. A sh:PropertyShape is meant to fire against an existing focus node selected by an enclosing sh:NodeShape; putting sh:targetClass directly on the property shape is non-conformant.",
            ),
            (
                "An sh:PropertyShape is reporting violations on values that look correct. Why?",
                "Often a path mistake on the sh:PropertyShape — the sh:path picks the wrong predicate, or the values are typed differently than the constraint expects. Inspect the resolved value set under sh:path and compare to the constraint expectations.",
            ),
            (
                "What happens when two sh:PropertyShape instances target the same path on the same focus node?",
                "Both sh:PropertyShape instances fire — SHACL is conjunctive. The focus node must pass both. This composes by design; surprising results usually mean two contradictory shapes were defined on the same path (e.g., sh:datatype xsd:integer in one, xsd:string in another).",
            ),
            (
                "Why is `sh:PropertyShape sh:datatype xsd:string ; sh:class ex:Person` always failing?",
                "The sh:PropertyShape is contradictory: sh:datatype demands literals; sh:class demands IRIs/blank nodes. No value can pass both. Pick one based on whether the property carries literals (sh:datatype) or class instances (sh:class).",
            ),
        ],
        combinations=[
            (
                "sh:NodeShape",
                "sh:PropertyShape + sh:NodeShape: the canonical pairing. The NodeShape provides focus-node selection (sh:targetClass / sh:targetNode); sh:property links pull PropertyShapes into the validation chain. Two layers, one contract.",
            ),
            (
                "sh:property",
                "sh:property is the link predicate from sh:NodeShape to sh:PropertyShape. Without it, a PropertyShape is orphaned in the shape graph; with it, the NodeShape's focus nodes flow through to the PropertyShape's value checks.",
            ),
            (
                "sh:datatype",
                "sh:PropertyShape + sh:datatype: literal-typed property. The most common SHACL pattern for typed properties — PropertyShape provides the path scope; sh:datatype constrains literal values to a datatype IRI.",
            ),
            (
                "sh:class",
                "sh:PropertyShape + sh:class: object-typed property. PropertyShape scopes the path; sh:class requires every value to be an instance of the named class. The IRI-valued counterpart of sh:datatype on the same shape.",
            ),
            (
                "sh:minCount",
                "sh:PropertyShape + sh:minCount: required cardinality. PropertyShape scopes the predicate; sh:minCount counts values. `sh:minCount 1` is the SHACL idiom for 'this property is mandatory.'",
            ),
            (
                "sh:qualifiedValueShape",
                "sh:PropertyShape + sh:qualifiedValueShape: per-value subshape checks with cardinality. Useful when at LEAST k values must satisfy a richer nested shape (e.g., 'at least one author is faculty').",
            ),
            (
                "sh:path",
                "sh:PropertyShape requires sh:path — the two are inseparable. The path expression can be a single IRI, a sequence, an inverse, an alternative, or a zero-or-more — whatever SHACL property paths permit.",
            ),
        ],
        anchored_status="complete",
    ),
    "rdfs:subClassOf": SurfaceFormData(
        curie="rdfs:subClassOf",
        short_name="subclass-of",
        definitions=[
            "rdfs:subClassOf is an RDFS predicate stating that the subject class is a subset of the object class — every instance of the subject is also an instance of the object. RDFS entailment propagates rdf:type along rdfs:subClassOf transitively.",
            "Think of rdfs:subClassOf as the RDFS way to declare class hierarchy. `ex:Student rdfs:subClassOf ex:Person` means every Student IS a Person, and reasoners can infer Person typing from Student typing automatically.",
            "rdfs:subClassOf carries a transitive semantics in RDFS: if A rdfs:subClassOf B and B rdfs:subClassOf C, then A rdfs:subClassOf C is entailed. Reasoners walk the chain to materialize all inferred class memberships.",
            "rdfs:subClassOf: a binary RDFS predicate from rdfs:Class to rdfs:Class, asserting set inclusion of the extensions. Reflexive in spec (every class is a subclass of itself), transitive across chains.",
            "Formally, A rdfs:subClassOf B iff for all x, x rdf:type A entails x rdf:type B. The entailment regime depends on the active RDFS rules; pyshacl's rdfs entailment honors transitivity but not the reflexive axiom by default.",
            "If rdf:type says 'this individual is an instance of class C,' rdfs:subClassOf says 'every instance of class A is also an instance of class B.' One operates per-individual; the other operates per-class.",
            "Practically, rdfs:subClassOf wires up taxonomies: ex:Faculty rdfs:subClassOf ex:Employee rdfs:subClassOf ex:Person creates a chain along which RDFS entailment propagates types.",
        ],
        usage_examples=[
            (
                "Show a basic rdfs:subClassOf assertion linking two classes.",
                "`ex:Student rdfs:subClassOf ex:Person .` Now any individual typed `ex:Student` is also entailed to be `ex:Person` under RDFS reasoning. SHACL constraints targeting ex:Person also fire on ex:Student instances.",
            ),
            (
                "Demonstrate transitive rdfs:subClassOf entailment across three classes.",
                "`ex:Faculty rdfs:subClassOf ex:Employee . ex:Employee rdfs:subClassOf ex:Person .` Transitively, ex:Faculty rdfs:subClassOf ex:Person is entailed — RDFS-aware queries on ex:Person see Faculty instances.",
            ),
            (
                "Use rdfs:subClassOf to extend an external vocabulary.",
                "`ex:LocalAddress rdfs:subClassOf schema:PostalAddress .` Every ex:LocalAddress instance is a schema:PostalAddress under RDFS. SPARQL queries pulling schema:PostalAddress now match the local subclass too.",
            ),
            (
                "Apply rdfs:subClassOf with a SHACL shape.",
                "If `ex:Student rdfs:subClassOf ex:Person` and `ex:PersonShape sh:targetClass ex:Person`, the PersonShape applies to ex:Student instances too — sh:targetClass honors rdfs:subClassOf transitivity by default.",
            ),
            (
                "Express a multi-parent class with rdfs:subClassOf.",
                "`ex:GradStudent rdfs:subClassOf ex:Student , ex:Researcher .` Under RDFS, ex:GradStudent instances are simultaneously typed Student AND Researcher. RDFS allows multiple subclass-of parents.",
            ),
            (
                "Show how rdfs:subClassOf lets a SPARQL query find subclass instances.",
                "Query `SELECT ?p WHERE { ?p a/rdfs:subClassOf* ex:Person }` returns every individual whose rdf:type chains via rdfs:subClassOf to ex:Person. This pattern is the canonical way to consume rdfs:subClassOf in queries.",
            ),
            (
                "Use rdfs:subClassOf to subclass an OWL class declaration.",
                "`ex:HumanAuthor rdfs:subClassOf ex:Author .` Even though OWL adds richer class axioms, rdfs:subClassOf is the simplest way to assert subset semantics; OWL reasoners pick it up alongside owl:equivalentClass and friends.",
            ),
        ],
        comparison_targets=[
            (
                "rdf:type",
                "rdf:type is per-individual class membership; rdfs:subClassOf is per-CLASS subset assertion. RDFS entailment combines them: rdf:type + rdfs:subClassOf chains let reasoners infer all classes an individual belongs to.",
            ),
            (
                "owl:equivalentClass",
                "rdfs:subClassOf is one-directional set inclusion (A ⊆ B); owl:equivalentClass is mutual inclusion (A = B). Two rdfs:subClassOf assertions in opposite directions imply equivalence; owl:equivalentClass states it directly.",
            ),
            (
                "rdfs:subPropertyOf",
                "rdfs:subClassOf is for class hierarchy; rdfs:subPropertyOf is for property hierarchy. Both are RDFS-level subset assertions; one operates on rdfs:Class, the other on rdf:Property. Same shape, different domain.",
            ),
            (
                "sh:class",
                "sh:class is a SHACL VALIDATION constraint; rdfs:subClassOf is an RDFS SCHEMA assertion. sh:class CONSUMES rdfs:subClassOf (transitively) when checking value membership. One asserts; the other audits.",
            ),
            (
                "owl:disjointWith",
                "rdfs:subClassOf asserts A ⊆ B; owl:disjointWith asserts A ∩ B = ∅. Together they let reasoners detect contradictions: if A rdfs:subClassOf B AND A owl:disjointWith B, then A's extension is empty.",
            ),
            (
                "rdfs:Class",
                "rdfs:Class declares a class; rdfs:subClassOf places that class in a hierarchy. You can declare a class without rdfs:subClassOf (it implicitly subClassOf rdfs:Resource); subClassOf is how you wire it into the taxonomy.",
            ),
            (
                "owl:Class",
                "owl:Class is OWL's class declaration; rdfs:subClassOf works the same way for both rdfs:Class and owl:Class instances. RDFS subclass semantics are reused in OWL — same predicate, OWL just adds richer axioms on top.",
            ),
        ],
        reasoning_scenarios=[
            (
                "You want a SHACL shape on ex:Person to also validate ex:Student instances. Which RDFS construct?",
                "Assert `ex:Student rdfs:subClassOf ex:Person`. With RDFS entailment enabled, a SHACL validator targeting ex:Person via sh:targetClass also matches ex:Student instances through rdfs:subClassOf transitivity.",
            ),
            (
                "When does using rdfs:subClassOf help over duplicating sh:class constraints?",
                "Declaring rdfs:subClassOf once propagates membership across the entire hierarchy at validation time. Duplicating sh:class constraints per subclass is brittle: adding a new subclass requires touching every shape that constrains a parent.",
            ),
            (
                "What's the trade-off between rdfs:subClassOf and owl:equivalentClass for synonym mapping?",
                "rdfs:subClassOf is one-way set inclusion — useful when one class is genuinely narrower. owl:equivalentClass states mutual inclusion — useful when two IRIs name the same class. Pick subClassOf for taxonomy, equivalentClass for synonyms.",
            ),
            (
                "A SHACL shape targets ex:Animal but ex:Dog instances aren't being validated. Why?",
                "Most likely there's no `ex:Dog rdfs:subClassOf ex:Animal` triple, OR the validator was run without RDFS entailment. Add the rdfs:subClassOf assertion AND enable rdfs entailment in the validator (e.g., pyshacl's `inference='rdfs'`).",
            ),
            (
                "You want to model 'every Researcher is both an Employee and a Person.' Which RDFS pattern?",
                "Assert `ex:Researcher rdfs:subClassOf ex:Employee` and `ex:Researcher rdfs:subClassOf ex:Person`. Multi-parent rdfs:subClassOf is supported by RDFS — instances are entailed to be members of every superclass.",
            ),
            (
                "Why might rdfs:subClassOf be preferred over manual rdf:type duplication?",
                "RDFS entailment over rdfs:subClassOf auto-generates the transitive rdf:type triples for you. Manual duplication is brittle: every new instance must repeat the full class chain, and forgetting any rdf:type breaks queries that assume the hierarchy.",
            ),
            (
                "An entailment is generating unexpected rdf:type triples. Where to look?",
                "Inspect rdfs:subClassOf chains starting from the surprising class. Reasoners chase the predicate transitively, so a single `A rdfs:subClassOf rdfs:Resource` somewhere in the imported vocabulary can entail rdf:type rdfs:Resource everywhere.",
            ),
        ],
        pitfalls=[
            (
                "Why do SHACL shapes targeting a superclass miss subclass instances when no rdfs:subClassOf is asserted?",
                "Without an rdfs:subClassOf assertion, no entailment fires. The validator can't infer that subclass instances are also superclass instances — sh:targetClass matches only direct rdf:type, missing the implicit hierarchy.",
            ),
            (
                "What's the common mistake mixing rdfs:subClassOf and rdfs:subPropertyOf?",
                "Authors sometimes write `ex:hasPart rdfs:subClassOf ex:hasComponent` when they mean `rdfs:subPropertyOf`. The first treats predicates as classes — RDFS-loose tools may accept it, but reasoners won't propagate subproperty entailments. Use the right predicate.",
            ),
            (
                "What goes wrong if rdfs:subClassOf is omitted from an extending vocabulary?",
                "The new class is isolated — SHACL shapes, SPARQL queries, and consumers expecting the parent class won't match instances of the new class. Adding `rdfs:subClassOf <ParentClass>` wires it into the existing hierarchy.",
            ),
            (
                "Why doesn't rdfs:subClassOf produce the inferred triples without RDFS entailment?",
                "RDFS entailment is a separate validation/reasoning step. Most validators run in 'no entailment' by default — the rdfs:subClassOf triples sit in the graph as data, but no inference fires. Enable inference (e.g., `inference='rdfs'`) to materialize.",
            ),
            (
                "Validation reports duplicate rdf:type triples after enabling RDFS entailment. Why?",
                "RDFS entailment over rdfs:subClassOf materializes superclass rdf:type triples — that's by design. The 'duplicates' are the inferred entailment closure. If they're undesired, run validation without inference, or filter inferred triples in a downstream pipeline.",
            ),
            (
                "What happens if rdfs:subClassOf forms a cycle (A rdfs:subClassOf B rdfs:subClassOf A)?",
                "Under RDFS, an rdfs:subClassOf cycle entails A owl:equivalentClass B (mutual inclusion). Some reasoners flag cycles for review; most accept them and continue. Cycles are rare in well-modeled vocabularies — usually a sign of an authoring mistake.",
            ),
            (
                "Why does `ex:A rdfs:subClassOf ex:A` not produce a violation?",
                "rdfs:subClassOf is reflexive in RDFS — every class is a subclass of itself by spec. The triple is redundant but valid. Reasoners treat it as a no-op; SHACL validators don't reject it.",
            ),
        ],
        combinations=[
            (
                "rdf:type",
                "rdfs:subClassOf + rdf:type: the inference engine. With RDFS entailment, an rdf:type triple to a subclass automatically generates rdf:type triples to every transitive superclass via rdfs:subClassOf chains.",
            ),
            (
                "sh:targetClass",
                "rdfs:subClassOf + sh:targetClass: hierarchy-aware validation. SHACL shapes targeting a parent class also fire against subclass instances when RDFS entailment is enabled — the canonical way to share constraints across a hierarchy.",
            ),
            (
                "sh:class",
                "rdfs:subClassOf + sh:class: hierarchy-aware value-typing. `sh:class ex:Person` admits ex:Student values when ex:Student rdfs:subClassOf ex:Person — sh:class honors rdfs:subClassOf transitivity by SHACL spec.",
            ),
            (
                "owl:equivalentClass",
                "rdfs:subClassOf + owl:equivalentClass: hierarchy plus synonym. owl:equivalentClass is mutual rdfs:subClassOf in both directions; together they let you mix taxonomic and synonym links in one vocabulary.",
            ),
            (
                "rdfs:Class",
                "rdfs:subClassOf + rdfs:Class: declared classes participate in the subclass hierarchy. Most vocabularies declare each class with `a rdfs:Class` and then wire it via rdfs:subClassOf — the pair is idiomatic.",
            ),
            (
                "owl:Class",
                "rdfs:subClassOf + owl:Class: OWL class wiring. OWL classes use the SAME rdfs:subClassOf predicate as RDFS classes — OWL doesn't introduce a separate subclass predicate. Reasoners treat both the same.",
            ),
            (
                "rdfs:subPropertyOf",
                "rdfs:subClassOf + rdfs:subPropertyOf: class-and-property hierarchies. The two predicates compose — together they form the complete RDFS schema-extension surface for vocabularies that subclass both classes and properties.",
            ),
        ],
        anchored_status="complete",
    ),
    "owl:sameAs": SurfaceFormData(
        curie="owl:sameAs",
        short_name="sameAs",
        definitions=[
            "owl:sameAs is an OWL predicate asserting that two IRIs denote the same individual. An owl:sameAs link merges all facts about the two IRIs — predicates, types, and annotations — into a single conceptual entity from the reasoner's point of view.",
            "Think of owl:sameAs as the OWL way to declare 'these two URIs are the same person/thing.' Reasoners then treat any triple about either URI as a triple about the other, propagating facts both directions.",
            "owl:sameAs carries strong identity semantics: it's reflexive, symmetric, and transitive. Asserting `:a owl:sameAs :b` and `:b owl:sameAs :c` entails `:a owl:sameAs :c` and bidirectional fact merging across all three.",
            "owl:sameAs: a binary OWL predicate between two individuals (IRIs or blank nodes). Asserts identity in the OWL Direct Semantics — the two terms refer to the same entity in the model.",
            "Formally, `:a owl:sameAs :b` entails that for every triple (:a, p, o), the triple (:b, p, o) is also entailed (and vice versa for object position). Equality propagation in RDF/OWL reasoning.",
            "If rdfs:subClassOf is for class-level hierarchies and owl:equivalentClass for class synonyms, owl:sameAs is the individual-level analogue — synonym at the instance layer rather than the class layer.",
            "Practically, owl:sameAs is the linked-data backbone for cross-dataset identity: dbpedia, wikidata, and local datasets use owl:sameAs to merge identifiers for the same real-world entity.",
        ],
        usage_examples=[
            (
                "Show a basic owl:sameAs assertion bridging DBpedia and Wikidata.",
                "`dbr:Albert_Einstein owl:sameAs wd:Q937 .` Declares that the DBpedia URI and the Wikidata URI refer to the same person. Reasoners now propagate every fact about either to the other under owl:sameAs.",
            ),
            (
                "Use owl:sameAs across local + external IRIs.",
                "`ex:einstein owl:sameAs dbr:Albert_Einstein .` Now ex:einstein inherits all DBpedia facts about Einstein under OWL reasoning — birth date, profession, etc. — without copying triples manually.",
            ),
            (
                "Demonstrate owl:sameAs symmetry between two named individuals.",
                "`:a owl:sameAs :b .` Under OWL, the inverse `:b owl:sameAs :a` is entailed automatically (owl:sameAs is symmetric). Reasoners treat both directions as known facts in the entailment closure.",
            ),
            (
                "Apply owl:sameAs transitively across a three-IRI chain.",
                "`:a owl:sameAs :b . :b owl:sameAs :c .` Transitively entails `:a owl:sameAs :c`. All three IRIs share fact extensions — a fact about :a is a fact about :c via the chain.",
            ),
            (
                "Use owl:sameAs to merge legacy + new identifiers.",
                "`legacy:user_42 owl:sameAs new:alice .` After a system migration, declaring owl:sameAs lets old SPARQL queries against legacy:user_42 still find Alice's data — bridges the identifier change without re-writing triples.",
            ),
            (
                "Show owl:sameAs working with a SPARQL ASK across linked datasets.",
                "Under OWL entailment, `ASK { dbr:Albert_Einstein dbo:birthPlace ?p } ` matches even if the birth-place triple is asserted only on `wd:Q937`, provided `dbr:Albert_Einstein owl:sameAs wd:Q937` is in the graph.",
            ),
            (
                "Use owl:sameAs for blank-node coreference.",
                "`_:b1 owl:sameAs _:b2 .` Declares two blank nodes denote the same individual. Reasoners merge their property extensions; useful when two RDF documents independently mint blank-node IDs for the same entity.",
            ),
        ],
        comparison_targets=[
            (
                "owl:differentFrom",
                "owl:sameAs asserts identity; owl:differentFrom asserts NON-identity. Stating both produces a contradiction — reasoners flag it. They're the OWL identity yes/no surface; rdfs:subClassOf et al. operate at the class layer.",
            ),
            (
                "owl:equivalentClass",
                "owl:sameAs is identity at the INDIVIDUAL layer; owl:equivalentClass is at the CLASS layer. `:a owl:sameAs :b` says two individuals are the same; `ex:A owl:equivalentClass ex:B` says two class extensions are the same.",
            ),
            (
                "rdfs:subClassOf",
                "owl:sameAs links individuals; rdfs:subClassOf links classes (subset). Different layers. owl:sameAs propagates ALL facts; rdfs:subClassOf propagates only rdf:type into the superclass.",
            ),
            (
                "owl:equivalentProperty",
                "owl:sameAs aligns individuals; owl:equivalentProperty aligns properties. The OWL identity surface has three flavors — sameAs (individual), equivalentClass (class), equivalentProperty (property) — each at its own layer.",
            ),
            (
                "skos:exactMatch",
                "owl:sameAs is strong logical identity; skos:exactMatch is weaker — it asserts mapping equivalence in a SKOS scheme without forcing OWL fact propagation. Linked-data publishers often prefer skos:exactMatch when full OWL semantics is too strong.",
            ),
            (
                "owl:Thing",
                "owl:sameAs operates over instances of owl:Thing (every OWL individual is implicitly an owl:Thing). owl:Thing is the universal class; owl:sameAs is the identity predicate that lets reasoners merge two of its instances.",
            ),
            (
                "rdf:type",
                "owl:sameAs propagates rdf:type triples between the linked individuals. If `:a owl:sameAs :b` and `:a rdf:type ex:Person`, then `:b rdf:type ex:Person` is entailed. owl:sameAs subsumes type propagation.",
            ),
        ],
        reasoning_scenarios=[
            (
                "You want to bridge a DBpedia IRI and a Wikidata IRI for the same person. Which OWL construct?",
                "Use owl:sameAs: `dbr:Albert_Einstein owl:sameAs wd:Q937`. Reasoners then treat both IRIs as the same individual, propagating every fact about either across the link in both directions.",
            ),
            (
                "When does owl:sameAs help over copying triples between datasets?",
                "owl:sameAs lets reasoners infer the equivalence at query/validation time without duplicating triples. Copying breaks when source data updates; owl:sameAs stays current as long as both source graphs are loaded into the reasoner.",
            ),
            (
                "What's the trade-off between owl:sameAs and skos:exactMatch for linked-data alignment?",
                "owl:sameAs is full OWL identity — propagates everything. skos:exactMatch is mapping-level — asserts conceptual equivalence without forcing fact merging. Use owl:sameAs when datasets agree on semantics; skos:exactMatch for looser bridges.",
            ),
            (
                "A SPARQL query is missing facts that exist on a sameAs-linked IRI. Why?",
                "Likely OWL entailment is OFF in the query engine. owl:sameAs propagation requires a reasoner; without it the triples sit in the graph but facts don't merge. Enable OWL inference or add a `?x owl:sameAs* ?y` pattern manually.",
            ),
            (
                "You want to merge two blank-node IDs that independent documents minted for the same entity. Which construct?",
                "owl:sameAs between the two blank nodes. Reasoners then merge their property extensions. Common when consuming RDF documents from multiple sources that each mint their own blank-node IDs for shared entities.",
            ),
            (
                "Why might owl:sameAs be preferred over duplicating ex:identifier values?",
                "owl:sameAs merges IRIs at the LOGICAL layer — every fact about either IRI applies to both. ex:identifier or similar custom predicates duplicate string IDs without triggering reasoner entailment. Stronger semantics, fewer manual joins.",
            ),
            (
                "An OWL reasoner is producing too many sameAs entailments. Where to investigate?",
                "owl:sameAs is reflexive, symmetric, transitive — closures grow fast through chains. Inspect for unintended chains (a borderline-correct sameAs that connects two large clusters). Consider weakening to skos:exactMatch where full identity isn't needed.",
            ),
        ],
        pitfalls=[
            (
                "Why does owl:sameAs sometimes produce surprising fact propagation?",
                "owl:sameAs is full OWL identity — it propagates EVERY fact, including type assertions, annotation properties, and subjective attributes. Linking two near-but-not-quite-equivalent IRIs (e.g., a person and their authored work) produces nonsense entailments.",
            ),
            (
                "What's the common mistake using owl:sameAs at the class layer?",
                "Authors sometimes write `ex:A owl:sameAs ex:B` for two CLASSES they want equivalent. owl:sameAs is for individuals; classes need owl:equivalentClass. Reasoners treat the misuse as identity at the class layer (which usually still works in OWL Full but not OWL DL).",
            ),
            (
                "What goes wrong if owl:sameAs is omitted between equivalent IRIs?",
                "The two IRIs sit in disconnected fact extensions — queries against one miss data on the other. Each IRI's facts stay siloed. owl:sameAs is the explicit signal reasoners need to merge them.",
            ),
            (
                "Why doesn't owl:sameAs propagate annotation properties in OWL DL?",
                "OWL DL distinguishes object/data/annotation properties. owl:sameAs propagates object and data property facts but NOT annotation property facts in strict OWL DL — annotations are reasoner-opaque by spec. OWL Full propagates everything.",
            ),
            (
                "owl:sameAs is causing a SPARQL query to time out. Why?",
                "owl:sameAs closures grow combinatorially through transitive chains. A graph with many sameAs links can produce huge entailment closures. Either restrict the query, disable OWL entailment for that query, or simplify the sameAs graph.",
            ),
            (
                "What happens when owl:sameAs and owl:differentFrom contradict?",
                "Asserting both `:a owl:sameAs :b` and `:a owl:differentFrom :b` makes the ontology inconsistent. OWL DL reasoners (HermiT, Pellet, ELK) flag the contradiction; the entailment closure includes any triple, making the graph useless.",
            ),
            (
                "Why is owl:sameAs sometimes treated as a 'leak' in linked-data?",
                "Strong identity propagation can leak facts that the publisher didn't intend to merge — a typo, a borderline alignment, or a polysemous entity. Best practice: review owl:sameAs links carefully and prefer skos:exactMatch when looser mapping suffices.",
            ),
        ],
        combinations=[
            (
                "rdf:type",
                "owl:sameAs + rdf:type: type propagation. Linking two IRIs via owl:sameAs entails every rdf:type triple of one to the other. Common pattern when bridging local + external vocabularies that classify the same entity differently.",
            ),
            (
                "owl:equivalentClass",
                "owl:sameAs + owl:equivalentClass: identity at two layers. owl:sameAs handles individuals; owl:equivalentClass handles classes. Used together when bridging two ontologies — sameAs for shared instances, equivalentClass for shared classes.",
            ),
            (
                "owl:differentFrom",
                "owl:sameAs and owl:differentFrom together let an ontology assert both identity AND non-identity claims. Asserting both for the same pair contradicts; using them on different pairs creates a partition of named individuals.",
            ),
            (
                "owl:Thing",
                "owl:sameAs + owl:Thing: the universal identity surface. Every owl:sameAs assertion lives between owl:Thing instances — sameAs IS the identity predicate over the OWL universe of discourse.",
            ),
            (
                "skos:exactMatch",
                "owl:sameAs + skos:exactMatch in a single dataset: layered identity claims. owl:sameAs for hard logical identity (full propagation); skos:exactMatch for soft mapping equivalence. Lets publishers express both kinds of equivalence on the same pair.",
            ),
            (
                "rdfs:subClassOf",
                "owl:sameAs + rdfs:subClassOf: identity at instance layer + hierarchy at class layer. Together they let you bridge two datasets where individuals are equivalent (sameAs) AND classes form a hierarchy (subClassOf).",
            ),
            (
                "owl:NamedIndividual",
                "owl:sameAs + owl:NamedIndividual: explicit identity for declared individuals. OWL DL strict mode prefers `a owl:NamedIndividual` on every individual; owl:sameAs operates between such declarations to merge them in the reasoner.",
            ),
        ],
        anchored_status="complete",
    ),
    # ---------------------------------------------------------------------
    # Wave 135a: 34 degraded-placeholder entries covering the remainder of
    # the rdf-shacl property manifest. Each carries one stub definition +
    # one stub usage_example so it satisfies the structural contract
    # (>=1 def + >=1 usage_example) but is explicitly tagged
    # anchored_status="degraded_placeholder" so:
    #   1. validate_form_data_contract can count degraded vs complete.
    #   2. generate_schema_translation_pairs() skips it (no "[degraded:"
    #      strings ever land in instruction_pairs.jsonl).
    #   3. Wave 135b's force-injection path can dispatch on the status
    #      field and emit a WARN log + decision-capture event.
    #
    # Operator / Codex / Qwen backfills the anchored content over time
    # per the project's "Claude does not generate training-data corpus
    # content" operating principle. Each backfilled entry flips
    # anchored_status="degraded_placeholder" -> "complete" and silently
    # improves the adapter's anchored-injection coverage.
    #
    # Operators searching for degraded entries can grep for the literal
    # token "[degraded:" — it appears in every stub definition + stub
    # usage_example string.
    # ---------------------------------------------------------------------
    "sh:minCount": SurfaceFormData(
        curie="sh:minCount",
        short_name="sh:minCount",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:maxCount": SurfaceFormData(
        curie="sh:maxCount",
        short_name="sh:maxCount",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:path": SurfaceFormData(
        curie="sh:path",
        short_name="sh:path",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:nodeKind": SurfaceFormData(
        curie="sh:nodeKind",
        short_name="sh:nodeKind",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:property": SurfaceFormData(
        curie="sh:property",
        short_name="sh:property",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:targetClass": SurfaceFormData(
        curie="sh:targetClass",
        short_name="sh:targetClass",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:closed": SurfaceFormData(
        curie="sh:closed",
        short_name="sh:closed",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:in": SurfaceFormData(
        curie="sh:in",
        short_name="sh:in",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:pattern": SurfaceFormData(
        curie="sh:pattern",
        short_name="sh:pattern",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:focusNode": SurfaceFormData(
        curie="sh:focusNode",
        short_name="sh:focusNode",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:ValidationResult": SurfaceFormData(
        curie="sh:ValidationResult",
        short_name="sh:ValidationResult",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:ValidationReport": SurfaceFormData(
        curie="sh:ValidationReport",
        short_name="sh:ValidationReport",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:result": SurfaceFormData(
        curie="sh:result",
        short_name="sh:result",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "sh:Violation": SurfaceFormData(
        curie="sh:Violation",
        short_name="sh:Violation",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdf:type": SurfaceFormData(
        curie="rdf:type",
        short_name="rdf:type",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdf:Property": SurfaceFormData(
        curie="rdf:Property",
        short_name="rdf:Property",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdfs:domain": SurfaceFormData(
        curie="rdfs:domain",
        short_name="rdfs:domain",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdfs:range": SurfaceFormData(
        curie="rdfs:range",
        short_name="rdfs:range",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdfs:Class": SurfaceFormData(
        curie="rdfs:Class",
        short_name="rdfs:Class",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdfs:Resource": SurfaceFormData(
        curie="rdfs:Resource",
        short_name="rdfs:Resource",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdfs:comment": SurfaceFormData(
        curie="rdfs:comment",
        short_name="rdfs:comment",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdfs:label": SurfaceFormData(
        curie="rdfs:label",
        short_name="rdfs:label",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "rdfs:Literal": SurfaceFormData(
        curie="rdfs:Literal",
        short_name="rdfs:Literal",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "owl:disjointWith": SurfaceFormData(
        curie="owl:disjointWith",
        short_name="owl:disjointWith",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "owl:equivalentClass": SurfaceFormData(
        curie="owl:equivalentClass",
        short_name="owl:equivalentClass",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "owl:Class": SurfaceFormData(
        curie="owl:Class",
        short_name="owl:Class",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "xsd:string": SurfaceFormData(
        curie="xsd:string",
        short_name="xsd:string",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "xsd:integer": SurfaceFormData(
        curie="xsd:integer",
        short_name="xsd:integer",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "xsd:date": SurfaceFormData(
        curie="xsd:date",
        short_name="xsd:date",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "xsd:dateTime": SurfaceFormData(
        curie="xsd:dateTime",
        short_name="xsd:dateTime",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "foaf:Person": SurfaceFormData(
        curie="foaf:Person",
        short_name="foaf:Person",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "foaf:knows": SurfaceFormData(
        curie="foaf:knows",
        short_name="foaf:knows",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "foaf:Agent": SurfaceFormData(
        curie="foaf:Agent",
        short_name="foaf:Agent",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
    "dcterms:creator": SurfaceFormData(
        curie="dcterms:creator",
        short_name="dcterms:creator",
        definitions=[_DEGRADED_DEFINITION_STUB],
        usage_examples=[
            (_DEGRADED_USAGE_PROMPT_STUB, _DEGRADED_USAGE_COMPLETION_STUB),
        ],
        anchored_status="degraded_placeholder",
    ),
}


# -----------------------------------------------------------------------------
# Wave 135b: anchored-injection helpers consumed by the SFT / DPO
# factories. ``resolve_anchor_text_for_curie`` is the shared dispatcher:
# given a CURIE + chunk_id_hash, it either returns a real anchored
# definition sentence (``status="anchored"``) drawn from the FORM_DATA
# entry's ``definitions`` list, OR it returns ``None`` text + a
# ``status="degraded"`` marker the factory uses to fall back to legacy
# token-stuffing while emitting a decision-capture warning event.
# -----------------------------------------------------------------------------


def resolve_anchor_text_for_curie(
    curie: str,
    form_data: Dict[str, SurfaceFormData],
    chunk_id_hash: int,
) -> Tuple[Optional[str], Optional[str], str]:
    """Wave 135b — pick anchored prompt + completion text for ``curie``.

    Args:
        curie: The CURIE (e.g. ``"sh:datatype"``) the factory wants to
            anchor in the pair body.
        form_data: The FORM_DATA dict to dispatch on. Pass
            ``_RDF_SHACL_FALLBACK_FORM_DATA`` (or whichever family
            catalog ``_load_form_data`` returned).
        chunk_id_hash: Stable integer derived from chunk_id; used to
            rotate across multiple definitions for the same CURIE so
            different chunks don't all anchor on the same string.

    Returns:
        Tuple ``(prompt_anchor, completion_anchor, status)``. ``status``
        is one of:

        * ``"anchored"`` — both anchor strings are non-None and contain
          the CURIE literally. Caller embeds them in the pair body.
        * ``"degraded"`` — entry is absent, marked
          ``"degraded_placeholder"``, or has no usable definitions.
          Caller MUST fall back to legacy token-stuffing AND emit a
          ``form_data_degraded_placeholder_skipped`` decision-capture
          event so the operator sees the degraded coverage.

    Why two anchor strings, not one:
        * The prompt-side anchor is appended to the prompt as a brief
          recall hook (``"Recall how <short definition with curie>."``)
          — primes the model to surface the CURIE in its answer.
        * The completion-side anchor is the FULL definition sentence
          drawn from ``entry.definitions``. That's the actual canonical
          surface form the trained adapter learns to emit when asked
          about the CURIE.

    Determinism:
        Same (curie, form_data, chunk_id_hash) tuple always returns the
        same string. Different chunks rotate through the entry's
        definitions list via modular indexing on the hash.
    """
    entry = form_data.get(curie)
    if entry is None:
        return (None, None, "degraded")
    if entry.anchored_status != "complete":
        return (None, None, "degraded")
    definitions = list(entry.definitions or [])
    if not definitions:
        return (None, None, "degraded")
    # Rotate: which definition does this chunk_id-hash get?
    completion_anchor = definitions[chunk_id_hash % len(definitions)]
    # For the prompt side, prefer the SHORTEST definition that still
    # contains the literal CURIE — keeps the prompt budget under
    # PROMPT_MAX after the suffix is appended.
    prompt_candidates = [
        d for d in definitions if curie in d
    ]
    if not prompt_candidates:
        # Defensive: every authored 'complete' entry should literally
        # contain the CURIE per the Wave 135a contract, but if a
        # future YAML drops it we fall back to degraded rather than
        # emitting a definition without the canonical anchor.
        return (None, None, "degraded")
    prompt_anchor = min(prompt_candidates, key=len)
    return (prompt_anchor, completion_anchor, "anchored")


# -----------------------------------------------------------------------------
# Wave 135a: FORM_DATA coverage contract validator.
# Wave 136b: extended with content-quality rejection rules.
# -----------------------------------------------------------------------------


# Wave 136b: forbidden definition prefixes (the Wave 121 token-stuffing
# template patterns the FORM_DATA contract is designed to replace).
# Anchored at the start of the string via ``re.match``.
_OLD_SUFFIX_TEMPLATE_RE = re.compile(
    r"^(Canonical terms:|Required terms:|Reference:|Relevant terms:"
    r"|Key vocabulary:|The relevant terms are|This concerns)"
)

# Wave 136b: leak markers that must NEVER appear in a complete entry's
# content. ``"[degraded:"`` is the Wave 135a stub prefix; ``"not yet
# authored"`` is the Wave 135a stub-text suffix.
_PLACEHOLDER_LEAK_TOKENS: Tuple[str, ...] = (
    "[degraded:",
    "not yet authored",
)

# Wave 137a — content-quality rule constants

# Rule 1: per-entry pairwise definitions diversity (Jaccard).
# Calibrated against the 6 ground-truth complete entries: max
# pairwise observed 0.379 (sh:NodeShape); 0.45 floor sits ~0.07
# above worst, blocks thesaurus-cloned siblings (typically 0.55-0.85).
_DIVERSITY_JACCARD_MAX = 0.45

# Rule 3: anchor-verb capacity allowlists (calibrated against 6
# ground-truth entries' definitions and usage_examples; scoped
# explicitly to those two categories — comparison_targets and
# pitfalls in the ground truth use ;-separated parallel constructions
# and rhetorical Q-side framing that don't match the verb allowlist).
_DEF_ANCHOR_VERBS = frozenset({
    "defines", "describes", "restricts", "requires", "validates", "targets",
    "constrains", "specifies", "applies", "indicates", "compares", "differs",
    "is", "are", "says", "asserts", "admits", "rejects", "carries", "expects",
    "operates", "propagates", "works", "accepts", "means", "denotes", "wires",
    "enforces", "enables", "holds", "states", "declares", "marks", "lives", "fires",
})
_USAGE_ACTION_VERBS = frozenset({
    "applies", "uses", "enforces", "declares", "binds", "evaluates", "conforms",
    "fails", "passes", "requires", "restricts", "admits", "rejects", "adds",
    "combines", "validates", "catches", "demonstrates", "shows", "writes",
    "use", "show", "demonstrate", "apply", "combine", "give", "express", "reuse",
})


def _word_anchor_re(allowlist: frozenset) -> re.Pattern:
    return re.compile(
        r"\b(" + "|".join(re.escape(v) for v in sorted(allowlist)) + r")\b",
        re.IGNORECASE,
    )


_DEF_ANCHOR_RE = _word_anchor_re(_DEF_ANCHOR_VERBS)
_USAGE_ANCHOR_RE = _word_anchor_re(_USAGE_ACTION_VERBS)

# Wave 137a Rule 2: style consistency score thresholds + signal regexes.
#
# CALIBRATION FINDING: brief proposed `_STYLE_CONSISTENCY_MIN = 0.85`
# with a "ground truth all >=0.95" claim. Empirical probe of the 6
# pre-Wave-135a complete entries:
#     sh:datatype       1.00
#     sh:class          0.80   <-- below 0.85
#     sh:NodeShape      1.00
#     sh:PropertyShape  1.00
#     rdfs:subClassOf   1.00
#     owl:sameAs        1.00
# sh:class definition #6 contains "If you've ever written..." which
# trips the conversational-phrase signal (the brief's regex includes
# `you've`); its present-tense ratio also lands at 5/7 = 71% (below
# 80%). Threshold floored to 0.80 so the gold set passes; ~0.05 above
# worst observed gold-set score. Calibration sibling-pattern matches
# Rule 1's "worst+0.05" floor.
_STYLE_CONSISTENCY_MIN = 0.80
_HEDGE_TOKENS = frozenset({
    "may", "might", "often", "typically", "usually", "sometimes", "likely",
})
_PRESENT_TENSE_DECLARATIVE_RE = re.compile(
    r"\b(is|are|defines?|describes?|constrains?|requires?|admits|rejects|"
    r"carries|operates|propagates|works|targets|specifies|asserts|denotes|"
    r"wires|holds|states|declares|marks|lives|fires)\b",
    re.IGNORECASE,
)
# Calibrated against ground-truth: must NOT match bare "you need" (pedagogical,
# not chatty). Restrict to second-person imperatives + first-person collectives.
_CONVERSATIONAL_RE = re.compile(
    r"\b(you can|you should|you'll|you've|you're|"
    r"let's|we'll|we've|we're|we'll see|"
    r"i think|i'd say|"
    r"feel free|hope this helps)\b",
    re.IGNORECASE,
)


def _compute_style_score(entry: "SurfaceFormData") -> Tuple[float, List[str]]:
    """Wave 137a Rule 2: weighted entry-level style score in [0, 1].

    Returns ``(score, list_of_failing_signal_names)``.

    Aggregates 9 signals at fixed weights (sum to 1.0):
      * +0.15 — primary CURIE appears in every definition
      * +0.10 — every definition's length lies in [50, 400]
      * +0.15 — at least one definition contains an anchor verb
      * +0.15 — at least one usage answer contains an action verb
      * +0.10 — present-tense declarative dominant (>=80% of defs)
      * +0.10 — no conversational phrasing in definitions
      * +0.05 — no Wave 121 suffix-template prefix in definitions
      * +0.10 — no excessive hedging (hedge tokens < 2 in defs)
      * +0.10 — no repeated openings (first 4 words of each def unique)

    Calibrated against 6 ground-truth complete entries — sh:class lands
    at 0.80 because it carries "If you've ever written..." (conversational
    you've) and a present-tense ratio of 5/7. The other 5 entries score
    1.00. Threshold ``_STYLE_CONSISTENCY_MIN`` is calibrated to 0.80 so
    the gold set passes.
    """
    from collections import Counter

    failed: List[str] = []
    score = 0.0

    # +0.15 — primary CURIE in every definition
    if entry.definitions and all(entry.curie in d for d in entry.definitions):
        score += 0.15
    else:
        failed.append("curie_in_every_def")

    # +0.10 — each def length in [50, 400]
    if entry.definitions and all(
        50 <= len(d) <= 400 for d in entry.definitions
    ):
        score += 0.10
    else:
        failed.append("def_length_bounds")

    # +0.15 — entry has >=1 anchor verb in defs
    if entry.definitions and any(
        _DEF_ANCHOR_RE.search(d) for d in entry.definitions
    ):
        score += 0.15
    else:
        failed.append("def_anchor_verb")

    # +0.15 — entry has >=1 action verb in usage answers
    if entry.usage_examples and any(
        _USAGE_ANCHOR_RE.search(a) for _, a in entry.usage_examples
    ):
        score += 0.15
    else:
        failed.append("usage_action_verb")

    # +0.10 — present-tense declarative dominant (>=80% of defs)
    if entry.definitions:
        hits = sum(
            1 for d in entry.definitions
            if _PRESENT_TENSE_DECLARATIVE_RE.search(d)
        )
        if hits / len(entry.definitions) >= 0.80:
            score += 0.10
        else:
            failed.append("present_tense_dominant")
    else:
        failed.append("no_definitions")

    # +0.10 — no conversational phrasing in defs
    if not any(_CONVERSATIONAL_RE.search(d) for d in entry.definitions):
        score += 0.10
    else:
        failed.append("no_conversational")

    # +0.05 — no suffix-template leak (already critical in Wave 136b;
    # included as positive signal for the aggregate score).
    if not any(_OLD_SUFFIX_TEMPLATE_RE.match(d) for d in entry.definitions):
        score += 0.05
    else:
        failed.append("no_suffix_template")

    # +0.10 — no excessive hedging in defs
    hedge_count = sum(
        sum(
            1 for tok in d.lower().split()
            if tok.strip(".,;:!?") in _HEDGE_TOKENS
        )
        for d in entry.definitions
    )
    if hedge_count < 2:
        score += 0.10
    else:
        failed.append("excessive_hedging")

    # +0.10 — no repeated openings (first-4-words across defs unique)
    openings = Counter(
        tuple(d.lower().split()[:4]) for d in entry.definitions if d
    )
    if openings and all(c == 1 for c in openings.values()):
        score += 0.10
    else:
        failed.append("repeated_openings")

    return (score, failed)

# Wave 136b: extracts CURIE-shaped tokens (``prefix:LocalName``) from a
# definition string for the WRONG_CURIE_ONLY_MENTION rule.
_CURIE_TOKEN_RE = re.compile(r"\b[a-z]+:[A-Za-z][A-Za-z0-9_]*")


def validate_form_data_contract(
    form_data: Dict[str, SurfaceFormData],
    manifest_curies: Iterable[str],
    *,
    base_form_data: Optional[Dict[str, SurfaceFormData]] = None,
) -> Dict[str, Any]:
    """Wave 135a / Wave 136b — enforce the FORM_DATA coverage contract.

    Wave 135a established the structural contract (every manifest CURIE
    has >=1 def + >=1 usage_example, every entry's ``anchored_status``
    is in the canonical set). Wave 136b extends the validator with
    nine content-quality rejection rules that fire ONLY against entries
    with ``anchored_status="complete"`` — degraded entries skip every
    content rule because their stub strings are intentionally
    out-of-bounds.

    Args:
        form_data: The catalog dict (typically
            ``_RDF_SHACL_FALLBACK_FORM_DATA``) keyed by CURIE.
        manifest_curies: Iterable of CURIEs declared by the property
            manifest. Each one must have an entry in ``form_data`` with
            >=1 definition AND >=1 usage_example.
        base_form_data: Optional Wave 136a base form_data dict (the
            Python-fallback catalog before the YAML overlay merged).
            When passed, the validator emits an
            ``OVERLAY_LOAD_REGRESSION`` warning for any CURIE whose
            status flipped from ``"complete"`` (in base) to
            ``"degraded_placeholder"`` (in form_data) — visibility-only,
            non-blocking.

    Returns:
        Dict with keys:
          * ``passed``: bool — True iff structural checks AND no
            critical content_violations.
          * ``missing_curies``: list[str] — manifest CURIEs not in
            form_data at all (sorted).
          * ``incomplete_curies``: list[str] — entries failing >=1 def
            + >=1 usage_example (sorted).
          * ``degraded_count``: int — # of entries with
            ``anchored_status="degraded_placeholder"``.
          * ``complete_count``: int — # of entries with
            ``anchored_status="complete"``.
          * ``invalid_status_curies``: list[str] — entries whose
            ``anchored_status`` is not in the canonical set (sorted).
          * ``content_violations``: list[dict] — Wave 136b critical
            content-quality rule violations. Each entry has shape
            ``{curie, code, detail}``. Only entries with
            ``anchored_status="complete"`` are checked.
          * ``warnings``: list[dict] — Wave 136b non-blocking warning
            signals (currently ``OVERLAY_LOAD_REGRESSION``). Each entry
            has shape ``{curie, code, detail}``.
    """
    # Pre-import the length-bound constants from the canonical
    # synthesis-provider module so this validator stays consistent with
    # the runtime length checks the providers enforce. Imported inside
    # the function to keep the module-level import graph minimal — the
    # constants are referenced ONLY by Wave 136b's content rules.
    from Trainforge.generators._anthropic_provider import (
        COMPLETION_MAX,
        COMPLETION_MIN,
        PROMPT_MAX,
        PROMPT_MIN,
    )
    # Wave 137a: Jaccard helper for Rule 1 (diversity gate). Reuses
    # the canonical tokenizer + Jaccard implementation already in use
    # by the eval pipeline so this validator and key-term-precision
    # eval share one tokenization contract.
    from Trainforge.eval.key_term_precision import _jaccard, _tokenize

    manifest_set = list(manifest_curies)
    manifest_curie_set = set(manifest_set)
    # Plus every CURIE keyed in form_data — operators can author
    # content-only entries (e.g. comparison targets) for CURIEs the
    # manifest doesn't declare. Treat every keyed CURIE as a potential
    # "OTHER manifest CURIE" for the WRONG_CURIE_ONLY_MENTION rule.
    full_curie_set = manifest_curie_set | set(form_data.keys())

    missing: List[str] = []
    incomplete: List[str] = []
    invalid_status: List[str] = []

    for curie in manifest_set:
        entry = form_data.get(curie)
        if entry is None:
            missing.append(curie)
            continue
        if len(entry.definitions) < 1 or len(entry.usage_examples) < 1:
            incomplete.append(curie)

    degraded_count = 0
    complete_count = 0
    for curie, entry in form_data.items():
        status = entry.anchored_status
        if status == "complete":
            complete_count += 1
        elif status == "degraded_placeholder":
            degraded_count += 1
        else:
            invalid_status.append(curie)

    # Wave 136b: content-quality rules. Iterate ONLY over complete
    # entries — degraded entries skip every content check by design
    # (their stub strings violate length bounds and contain "[degraded:"
    # by construction).
    content_violations: List[Dict[str, str]] = []
    # Wave 137a Rule 2: style-consistency warnings collected per-entry
    # in the same loop as the critical rules; merged into warnings_list
    # at the validator's return.
    wave_137a_style_warnings: List[Dict[str, str]] = []

    for curie, entry in form_data.items():
        if entry.anchored_status != "complete":
            continue

        # Rule: CURIE_NOT_VERBATIM_DEFINITION (per definition string).
        for idx, definition in enumerate(entry.definitions):
            if curie not in definition:
                content_violations.append(
                    {
                        "curie": curie,
                        "code": "CURIE_NOT_VERBATIM_DEFINITION",
                        "detail": (
                            f"definitions[{idx}] does not contain the "
                            f"literal CURIE {curie!r}"
                        ),
                    }
                )

        # Rule: CURIE_NOT_VERBATIM_USAGE_ANSWER (per usage_examples
        # answer field).
        for idx, usage_tuple in enumerate(entry.usage_examples):
            # usage_examples is List[Tuple[str, str]] = (prompt, answer)
            if len(usage_tuple) < 2:
                continue
            answer = usage_tuple[1]
            if curie not in answer:
                content_violations.append(
                    {
                        "curie": curie,
                        "code": "CURIE_NOT_VERBATIM_USAGE_ANSWER",
                        "detail": (
                            f"usage_examples[{idx}] answer does not "
                            f"contain the literal CURIE {curie!r}"
                        ),
                    }
                )

        # Rule: OLD_SUFFIX_TEMPLATE_LEAK (per definition).
        for idx, definition in enumerate(entry.definitions):
            if _OLD_SUFFIX_TEMPLATE_RE.match(definition):
                content_violations.append(
                    {
                        "curie": curie,
                        "code": "OLD_SUFFIX_TEMPLATE_LEAK",
                        "detail": (
                            f"definitions[{idx}] starts with a "
                            f"forbidden Wave 121 token-stuffing "
                            f"template prefix"
                        ),
                    }
                )

        # Rule: PLACEHOLDER_LEAKAGE (per definition + per usage string).
        for idx, definition in enumerate(entry.definitions):
            for token in _PLACEHOLDER_LEAK_TOKENS:
                if token in definition:
                    content_violations.append(
                        {
                            "curie": curie,
                            "code": "PLACEHOLDER_LEAKAGE",
                            "detail": (
                                f"definitions[{idx}] contains "
                                f"placeholder marker {token!r}"
                            ),
                        }
                    )
                    break
        for idx, usage_tuple in enumerate(entry.usage_examples):
            if len(usage_tuple) < 2:
                continue
            for field_name, value in (
                ("prompt", usage_tuple[0]),
                ("answer", usage_tuple[1]),
            ):
                for token in _PLACEHOLDER_LEAK_TOKENS:
                    if token in value:
                        content_violations.append(
                            {
                                "curie": curie,
                                "code": "PLACEHOLDER_LEAKAGE",
                                "detail": (
                                    f"usage_examples[{idx}] "
                                    f"{field_name} contains "
                                    f"placeholder marker {token!r}"
                                ),
                            }
                        )
                        break

        # Rule: LENGTH_OUT_OF_BOUNDS_DEF (50 <= len <= 400).
        for idx, definition in enumerate(entry.definitions):
            if not (50 <= len(definition) <= 400):
                content_violations.append(
                    {
                        "curie": curie,
                        "code": "LENGTH_OUT_OF_BOUNDS_DEF",
                        "detail": (
                            f"definitions[{idx}] length {len(definition)} "
                            f"outside [50, 400]"
                        ),
                    }
                )

        # Rule: LENGTH_OUT_OF_BOUNDS_USAGE_PROMPT (PROMPT_MIN..PROMPT_MAX).
        # Rule: LENGTH_OUT_OF_BOUNDS_USAGE_ANSWER
        # (COMPLETION_MIN..COMPLETION_MAX).
        for idx, usage_tuple in enumerate(entry.usage_examples):
            if len(usage_tuple) < 2:
                continue
            prompt, answer = usage_tuple[0], usage_tuple[1]
            if not (PROMPT_MIN <= len(prompt) <= PROMPT_MAX):
                content_violations.append(
                    {
                        "curie": curie,
                        "code": "LENGTH_OUT_OF_BOUNDS_USAGE_PROMPT",
                        "detail": (
                            f"usage_examples[{idx}] prompt length "
                            f"{len(prompt)} outside "
                            f"[{PROMPT_MIN}, {PROMPT_MAX}]"
                        ),
                    }
                )
            if not (COMPLETION_MIN <= len(answer) <= COMPLETION_MAX):
                content_violations.append(
                    {
                        "curie": curie,
                        "code": "LENGTH_OUT_OF_BOUNDS_USAGE_ANSWER",
                        "detail": (
                            f"usage_examples[{idx}] answer length "
                            f"{len(answer)} outside "
                            f"[{COMPLETION_MIN}, {COMPLETION_MAX}]"
                        ),
                    }
                )

        # Rule: WRONG_CURIE_ONLY_MENTION (per definition).
        # Skip when the manifest set is empty / unknown — without other
        # CURIEs to compare against, the rule would false-positive any
        # entry that simply doesn't mention sibling vocabulary.
        if full_curie_set:
            for idx, definition in enumerate(entry.definitions):
                tokens = _CURIE_TOKEN_RE.findall(definition)
                if not tokens:
                    continue
                if curie in tokens:
                    continue  # entry's own CURIE present — fine.
                # Entry's own CURIE not in extracted tokens. Are any
                # OTHER manifest/keyed CURIEs in tokens? If so, the
                # definition mentions sibling vocabulary but not its
                # own — reject.
                other_curies_in_def = [
                    t
                    for t in tokens
                    if t in full_curie_set and t != curie
                ]
                if other_curies_in_def:
                    content_violations.append(
                        {
                            "curie": curie,
                            "code": "WRONG_CURIE_ONLY_MENTION",
                            "detail": (
                                f"definitions[{idx}] mentions sibling "
                                f"CURIEs {other_curies_in_def} but not "
                                f"its own {curie!r}"
                            ),
                        }
                    )

        # Rule: GENERIC_DEFINITIONS_NO_USAGE (entry-level, not per-item).
        if (
            len(entry.definitions) >= 1
            and len(entry.usage_examples) == 0
        ):
            content_violations.append(
                {
                    "curie": curie,
                    "code": "GENERIC_DEFINITIONS_NO_USAGE",
                    "detail": (
                        "complete entry has definitions but zero "
                        "usage_examples tuples"
                    ),
                }
            )

        # Wave 137a Rule 1: pairwise definitions diversity.
        if len(entry.definitions) >= 2:
            max_sim = 0.0
            max_pair = (0, 0)
            tokenized = [_tokenize(d) for d in entry.definitions]
            for i in range(len(entry.definitions)):
                for j in range(i + 1, len(entry.definitions)):
                    sim = _jaccard(tokenized[i], tokenized[j])
                    if sim > max_sim:
                        max_sim = sim
                        max_pair = (i, j)
            if max_sim > _DIVERSITY_JACCARD_MAX:
                content_violations.append({
                    "curie": curie,
                    "code": "LOW_DIVERSITY_DEFINITIONS",
                    "detail": (
                        f"definitions[{max_pair[0]}] vs definitions[{max_pair[1]}] "
                        f"Jaccard {max_sim:.3f} > {_DIVERSITY_JACCARD_MAX}"
                    ),
                })

        # Wave 137a Rule 3: anchor-verb capacity (entry-level, scoped).
        if entry.definitions and not any(
            _DEF_ANCHOR_RE.search(d) for d in entry.definitions
        ):
            sample = sorted(_DEF_ANCHOR_VERBS)[:8]
            content_violations.append({
                "curie": curie,
                "code": "MISSING_ANCHOR_VERB_DEFINITION",
                "detail": (
                    f"no definition contains a verb from anchor allowlist "
                    f"(need one of: {sample}, ...)"
                ),
            })
        if entry.usage_examples and not any(
            _USAGE_ANCHOR_RE.search(answer) for _, answer in entry.usage_examples
        ):
            sample = sorted(_USAGE_ACTION_VERBS)[:8]
            content_violations.append({
                "curie": curie,
                "code": "MISSING_ANCHOR_VERB_USAGE",
                "detail": (
                    f"no usage_example answer contains an action verb "
                    f"(need one of: {sample}, ...)"
                ),
            })

        # Wave 137a Rule 2: style consistency (warning-severity).
        # Aggregates 9 signals; warning fires below threshold. Distinct
        # from Rule 1 (diversity) + Rule 3 (anchor verbs) — those rules
        # are critical because they catch single hard failure modes;
        # Rule 2 is the soft holistic style sentinel.
        style_score, style_failed = _compute_style_score(entry)
        if style_score < _STYLE_CONSISTENCY_MIN:
            wave_137a_style_warnings.append({
                "curie": curie,
                "code": "STYLE_CONSISTENCY_BELOW_THRESHOLD",
                "detail": (
                    f"score={style_score:.2f} < {_STYLE_CONSISTENCY_MIN}; "
                    f"failing signals: {style_failed}"
                ),
            })

    # Wave 136b: warning rule — OVERLAY_LOAD_REGRESSION.
    # Surfaces the complete -> degraded_placeholder transition Wave
    # 136a's loader logger.warning's. Visibility-only (non-blocking).
    warnings_list: List[Dict[str, str]] = []
    if base_form_data is not None:
        for curie, base_entry in base_form_data.items():
            if base_entry.anchored_status != "complete":
                continue
            current = form_data.get(curie)
            if current is None:
                continue
            if current.anchored_status == "degraded_placeholder":
                warnings_list.append(
                    {
                        "curie": curie,
                        "code": "OVERLAY_LOAD_REGRESSION",
                        "detail": (
                            "base form_data marked this CURIE complete "
                            "but the merged form_data marks it "
                            "degraded_placeholder — overlay regression"
                        ),
                    }
                )

    # Wave 137a Rule 2: merge style-consistency warnings collected
    # in the per-entry loop above. Distinct from OVERLAY_LOAD_REGRESSION
    # — both are non-blocking, but Rule 2 covers entry style drift while
    # OVERLAY surfaces structural regressions.
    warnings_list.extend(wave_137a_style_warnings)

    passed = (
        not missing
        and not incomplete
        and not invalid_status
        and not content_violations
    )

    return {
        "passed": passed,
        "missing_curies": sorted(missing),
        "incomplete_curies": sorted(incomplete),
        "degraded_count": degraded_count,
        "complete_count": complete_count,
        "invalid_status_curies": sorted(invalid_status),
        "content_violations": content_violations,
        "warnings": warnings_list,
    }


def _last_event_id(capture: Any) -> str:
    """Return the event_id of the most recent decision logged via `capture`.

    Mirrors `synthesize_training._last_event_id` so the emitted pairs
    carry valid `decision_capture_id` strings (Wave 112 invariant).
    """
    decisions = getattr(capture, "decisions", None) or []
    if not decisions:
        raise RuntimeError(
            "schema_translation_generator: capture has no logged "
            "decisions; log a stage-start decision before generating "
            "pairs."
        )
    last = decisions[-1]
    return str(last.get("event_id", "")) if isinstance(last, dict) else ""


def _validate_pair(pair: Dict[str, Any]) -> None:
    """Validate a single pair against `instruction_pair.schema.json`.

    Mirrors `kg_metadata_generator`'s schema-validate-on-emit policy.
    """
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - dev-test dep
        return
    schema_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json"
    )
    if not schema_path.exists():  # pragma: no cover
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(pair, schema)


def _make_pair(
    *,
    prompt: str,
    completion: str,
    family: str,
    curie: str,
    extra_concept_tags: Optional[List[str]],
    decision_capture_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Render one pair envelope. Family-agnostic — bloom + template_id
    are looked up from ``family``."""
    bloom = _FAMILY_BLOOM[family]
    template_id = f"schema_translation.{family}"
    concept_tags = [curie]
    if extra_concept_tags:
        for tag in extra_concept_tags:
            if tag and tag not in concept_tags:
                concept_tags.append(tag)
    return {
        "prompt": prompt,
        "completion": completion,
        "chunk_id": "schema-translation",
        "lo_refs": ["schema-translation"],
        "bloom_level": bloom,
        "content_type": "schema_translation",
        "seed": seed,
        "decision_capture_id": decision_capture_id,
        "template_id": template_id,
        "provider": "mock",
        "schema_version": "v1",
        "requires_source_citation": False,
        "concept_tags": concept_tags,
    }


# -----------------------------------------------------------------------------
# Family factories. Each takes a SurfaceFormData and returns a list of
# (prompt, completion, family, extra_concept_tags) tuples. The main
# emitter wraps each into the full pair envelope (with capture id +
# seed).
# -----------------------------------------------------------------------------

# The "for an RDF/SHACL learner" suffix hits the schema's 40-char
# minLength on every CURIE without making prompts wordy. Definitions
# vary their angle through the prompt itself.
_DEFINITION_PROMPT_FRAMES: Tuple[str, ...] = (
    "What does {curie} mean in SHACL/RDF terms? Plain-English answer please.",
    "Define {curie} as it appears in the SHACL or RDF/RDFS/OWL specs.",
    "How would you explain {curie} to someone reading a shape graph for the first time?",
    "Give a one-paragraph definition of {curie} suitable for an RDF/SHACL learner.",
    "In plain English, what is {curie} and what does it constrain?",
    "Explain {curie} the way the SHACL spec would, in a few sentences.",
    "What is the role of {curie} in an RDF/SHACL knowledge graph?",
)


def _definition_pairs(
    form: SurfaceFormData,
) -> List[Tuple[str, str, str, Optional[List[str]]]]:
    out: List[Tuple[str, str, str, Optional[List[str]]]] = []
    for idx, definition in enumerate(form.definitions):
        frame = _DEFINITION_PROMPT_FRAMES[idx % len(_DEFINITION_PROMPT_FRAMES)]
        prompt = frame.format(curie=form.curie)
        out.append((prompt, definition, "definition", None))
    return out


def _usage_pairs(
    form: SurfaceFormData,
) -> List[Tuple[str, str, str, Optional[List[str]]]]:
    out: List[Tuple[str, str, str, Optional[List[str]]]] = []
    for prompt, completion in form.usage_examples:
        out.append((prompt, completion, "usage", None))
    return out


_COMPARISON_PROMPT_FRAMES: Tuple[str, ...] = (
    "What's the difference between {primary} and {other} in SHACL/RDF practice?",
    "Compare {primary} and {other}: when does each apply?",
    "Contrast {primary} with {other}. Different layers, different purposes?",
    "How does {primary} differ from {other} in a shape graph?",
    "{primary} vs {other} — which one constrains what?",
    "If I know {other}, what's new about {primary}?",
    "Where do {primary} and {other} fit in the SHACL/RDF ontology layer cake?",
)


def _comparison_pairs(
    form: SurfaceFormData,
) -> List[Tuple[str, str, str, Optional[List[str]]]]:
    out: List[Tuple[str, str, str, Optional[List[str]]]] = []
    for idx, (other, completion) in enumerate(form.comparison_targets):
        frame = _COMPARISON_PROMPT_FRAMES[idx % len(_COMPARISON_PROMPT_FRAMES)]
        prompt = frame.format(primary=form.curie, other=other)
        out.append((prompt, completion, "comparison", [other]))
    return out


def _reasoning_pairs(
    form: SurfaceFormData,
) -> List[Tuple[str, str, str, Optional[List[str]]]]:
    out: List[Tuple[str, str, str, Optional[List[str]]]] = []
    for prompt, completion in form.reasoning_scenarios:
        out.append((prompt, completion, "reasoning", None))
    return out


def _pitfall_pairs(
    form: SurfaceFormData,
) -> List[Tuple[str, str, str, Optional[List[str]]]]:
    out: List[Tuple[str, str, str, Optional[List[str]]]] = []
    for prompt, completion in form.pitfalls:
        out.append((prompt, completion, "pitfall", None))
    return out


_COMBINATION_PROMPT_FRAMES: Tuple[str, ...] = (
    "How do {primary} and {other} compose in a SHACL shape graph?",
    "Can {primary} and {other} be used together? What's the result?",
    "What happens when {primary} applies to a node also subject to {other}?",
    "Describe the canonical pattern using {primary} alongside {other}.",
    "What does combining {primary} with {other} give you that neither does alone?",
    "Show how {primary} and {other} interact in a typical SHACL contract.",
    "{primary} + {other}: what's the composed validation behavior?",
)


def _combination_pairs(
    form: SurfaceFormData,
) -> List[Tuple[str, str, str, Optional[List[str]]]]:
    out: List[Tuple[str, str, str, Optional[List[str]]]] = []
    for idx, (other, completion) in enumerate(form.combinations):
        frame = _COMBINATION_PROMPT_FRAMES[idx % len(_COMBINATION_PROMPT_FRAMES)]
        prompt = frame.format(primary=form.curie, other=other)
        out.append((prompt, completion, "combination", [other]))
    return out


_FAMILY_FACTORIES: Dict[
    str, Callable[[SurfaceFormData], List[Tuple[str, str, str, Optional[List[str]]]]]
] = {
    "definition": _definition_pairs,
    "usage": _usage_pairs,
    "comparison": _comparison_pairs,
    "reasoning": _reasoning_pairs,
    "pitfall": _pitfall_pairs,
    "combination": _combination_pairs,
}


def _python_fallback_for_family(family: str) -> Dict[str, SurfaceFormData]:
    """Wave 136a: returns the in-Python fallback dict for the family.

    Today only ``rdf_shacl`` ships with an in-Python fallback — Wave
    125b's hand-curated 40-entry catalog. Other families return an
    empty dict and rely entirely on their YAML overlay.
    """
    if family == "rdf_shacl":
        return _RDF_SHACL_FALLBACK_FORM_DATA
    return {}


def _coerce_provenance(raw: Any) -> Optional[Provenance]:
    """Wave 137c: project a YAML-loaded provenance dict into a
    :class:`Provenance` instance, or return ``None`` when absent /
    malformed.

    Required string keys: ``provider``, ``generated_by``,
    ``reviewed_by``, ``prompt_version``, ``timestamp`` — all must be
    non-empty strings after ``.strip()``. Missing-key or empty-value
    payloads emit ``logger.error`` and return ``None`` (preserves the
    base-fallback safety property: malformed provenance never raises
    here; Plan A's validator is the strict-enforcement surface).

    Optional ``notes`` is passed through verbatim when present.
    """
    if not isinstance(raw, dict):
        return None
    required = ("provider", "generated_by", "reviewed_by", "prompt_version", "timestamp")
    if not all(k in raw and isinstance(raw[k], str) and raw[k].strip() for k in required):
        logger.error("provenance block missing required keys or values empty: %s", raw)
        return None
    return Provenance(
        provider=raw["provider"],
        generated_by=raw["generated_by"],
        reviewed_by=raw["reviewed_by"],
        prompt_version=raw["prompt_version"],
        timestamp=raw["timestamp"],
        notes=raw.get("notes"),
    )


def _load_yaml_catalog(family: str) -> Dict[str, SurfaceFormData]:
    """Wave 136a: read the per-family YAML overlay and project into
    ``SurfaceFormData`` instances.

    Returns ``{}`` when the YAML file is absent — that's the steady
    state for any family whose catalog ships entirely in-Python (today,
    none) or for any family that has no per-family YAML at all (e.g.
    legacy non-rdf_shacl families).

    Critical safety: on YAML parse failure or a malformed payload (no
    top-level ``forms`` key), emit ``logger.error`` and return ``{}``
    — this is load-bearing ToS-mitigation. A silent return-empty here
    would let the per-CURIE merge fall back to the in-Python base
    unchanged, NEVER erasing complete entries. A crash would bubble up
    and block the synthesis run.
    """
    catalog_path = (
        PROJECT_ROOT
        / "schemas"
        / "training"
        / f"schema_translation_catalog.{family}.yaml"
    )
    try:
        import yaml  # local import; YAML isn't a hot-path dep elsewhere here.
    except ImportError:  # pragma: no cover — yaml is a hard dep elsewhere.
        logger.error(
            "PyYAML not importable; skipping YAML overlay for family=%s",
            family,
        )
        return {}

    try:
        raw_text = catalog_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        logger.error(
            "schema_translation catalog YAML for family=%s failed to "
            "parse (%s); skipping overlay merge to preserve in-Python "
            "fallback entries.",
            family,
            exc,
        )
        return {}

    if not isinstance(payload, dict) or "forms" not in payload:
        logger.error(
            "schema_translation catalog YAML for family=%s is malformed "
            "(missing top-level 'forms' key); skipping overlay merge to "
            "preserve in-Python fallback entries.",
            family,
        )
        return {}

    out: Dict[str, SurfaceFormData] = {}
    for curie, raw in (payload.get("forms") or {}).items():
        if not isinstance(raw, dict):
            continue
        anchored_status = raw.get("anchored_status", "complete")
        if anchored_status not in ("complete", "degraded_placeholder"):
            logger.error(
                "schema_translation YAML overlay entry %s carries "
                "anchored_status=%r outside the canonical enum; "
                "skipping this CURIE.",
                curie,
                anchored_status,
            )
            continue
        out[curie] = SurfaceFormData(
            curie=curie,
            short_name=str(raw.get("short_name") or curie.split(":")[-1]),
            anchored_status=anchored_status,
            definitions=list(raw.get("definitions") or []),
            usage_examples=[
                tuple(pair) for pair in (raw.get("usage_examples") or [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ],
            comparison_targets=[
                tuple(pair) for pair in (raw.get("comparison_targets") or [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ],
            reasoning_scenarios=[
                tuple(pair) for pair in (raw.get("reasoning_scenarios") or [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ],
            pitfalls=[
                tuple(pair) for pair in (raw.get("pitfalls") or [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ],
            combinations=[
                tuple(pair) for pair in (raw.get("combinations") or [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ],
            provenance=_coerce_provenance(raw.get("provenance")),
        )
    return out


def _deep_merge_by_curie(
    base: Dict[str, SurfaceFormData],
    overlay: Dict[str, SurfaceFormData],
) -> Dict[str, SurfaceFormData]:
    """Wave 136a: per-CURIE overlay merge.

    YAML wins per-CURIE: an overlay entry replaces the Python entry
    for the same CURIE; YAML CURIEs not in Python are added; Python
    CURIEs not in YAML are preserved. Critically, a partial YAML
    cannot erase the in-Python fallback's complete entries.

    Safety rails:
      * If an overlay entry regresses a complete base entry to
        ``degraded_placeholder``, emit ``logger.warning`` (mid-edit
        signal, not a hard block — operators sometimes need to mark a
        CURIE under-revision).
      * If an overlay entry claims ``anchored_status="complete"`` but
        carries no non-stub definitions, raise ``ValueError`` at
        load time — overlay-level safety distinct from Wave 136b's
        content-quality validator.

    Determinism: iterates the overlay in sorted-CURIE order so the
    "added by YAML" CURIEs land in a stable position regardless of
    YAML field ordering.
    """
    merged: Dict[str, SurfaceFormData] = dict(base)
    for curie, entry in sorted(overlay.items()):
        base_entry = base.get(curie)
        # Mid-edit signal: complete -> degraded regression in YAML.
        if (
            entry.anchored_status == "degraded_placeholder"
            and base_entry is not None
            and base_entry.anchored_status == "complete"
        ):
            logger.warning(
                "YAML overlay regressed CURIE=%s from complete to "
                "degraded_placeholder",
                curie,
            )
        # Hard reject: complete with empty definitions is a contract
        # violation. Wave 136b's content validator will widen this to
        # cover stub-string content as well; here we only enforce the
        # structural floor.
        if entry.anchored_status == "complete" and not entry.definitions:
            raise ValueError(
                f"YAML overlay entry {curie} claims complete but has "
                f"empty definitions"
            )
        merged[curie] = entry
    return merged


@functools.lru_cache(maxsize=8)
def _load_form_data(family: str) -> Dict[str, SurfaceFormData]:
    """Wave 136a — per-CURIE overlay merge.

    Loads the in-Python fallback dict for the family (today: only
    rdf_shacl has one), then overlays the YAML catalog at
    schemas/training/schema_translation_catalog.<family>.yaml.

    YAML wins per-CURIE: an overlay entry replaces the Python entry
    for the same CURIE; YAML CURIEs not in Python are added; Python
    CURIEs not in YAML are preserved. Critically, a partial YAML
    cannot erase the in-Python fallback's complete entries.

    Wave 133d's whole-family-swap behavior is replaced. Operators
    backfilling one CURIE at a time (Wave 136d's flow) no longer
    risk erasing the existing 6 complete entries.

    Cached via ``functools.lru_cache`` keyed on ``family``; Wave 136d
    will call ``_invalidate_form_data_cache()`` after appending to
    the YAML so the next read picks up the new content.
    """
    base = _python_fallback_for_family(family)
    overlay = _load_yaml_catalog(family)
    return _deep_merge_by_curie(base, overlay)


def _invalidate_form_data_cache() -> None:
    """Clear the ``_load_form_data`` lru_cache.

    Wave 136d's operator-paused backfill flow appends entries to the
    YAML overlay between paraphrase passes; calling this between
    appends forces the next ``_load_form_data`` call to re-read the
    YAML from disk so the freshly-authored CURIE is visible.
    """
    _load_form_data.cache_clear()


def _build_catalog_in_order(
    manifest: PropertyManifest,
    form_data: Dict[str, SurfaceFormData],
) -> List[Tuple[str, SurfaceFormData, str, str, str, Optional[List[str]]]]:
    """Build the full ordered catalog: list of
    (curie, form_data, family, prompt, completion, extra_tags).

    The traversal order is round-robin BY FAMILY across surface forms:
    family[0] for every form, then family[1] for every form, ... so a
    capped run sees a balanced sample across all 6 families before
    exhausting any single one.

    Wave 133d: ``form_data`` is now passed in by ``_load_form_data``
    (dispatched on ``manifest.family``) instead of read from a
    module-level constant, so non-rdf_shacl families can ship their
    own per-family YAML catalogs without editing this module.
    """
    # First, build per-form per-family lists in deterministic order.
    per_form_per_family: Dict[str, Dict[str, List[Tuple[str, str, str, Optional[List[str]]]]]] = {}
    for prop in manifest.properties:
        curie = prop.curie
        form = form_data.get(curie)
        if form is None:
            continue
        per_form_per_family[curie] = {}
        for family in _FAMILIES:
            per_form_per_family[curie][family] = _FAMILY_FACTORIES[family](form)

    # Round-robin over families, then forms (manifest order), then
    # entries within each (curie, family) bucket.
    ordered: List[Tuple[str, SurfaceFormData, str, str, str, Optional[List[str]]]] = []
    # Determine the max entries per (form, family) so we can iterate slot-by-slot.
    max_entries = 0
    for curie, fams in per_form_per_family.items():
        for family, entries in fams.items():
            if len(entries) > max_entries:
                max_entries = len(entries)

    # Outer: slot index. Middle: family. Inner: form (manifest order).
    # Yields balanced cap behavior — at slot 0 we visit (form0, fam0),
    # (form1, fam0), ... (form5, fam0), then (form0, fam1), etc.
    for slot in range(max_entries):
        for family in _FAMILIES:
            for prop in manifest.properties:
                curie = prop.curie
                fams = per_form_per_family.get(curie)
                if fams is None:
                    continue
                entries = fams[family]
                if slot >= len(entries):
                    continue
                prompt, completion, _fam, extra_tags = entries[slot]
                form = form_data[curie]
                ordered.append(
                    (curie, form, family, prompt, completion, extra_tags)
                )
    return ordered


def generate_schema_translation_pairs(
    manifest: PropertyManifest,
    *,
    capture: Any,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    seed: int = 17,
) -> Tuple[List[Dict[str, Any]], SchemaTranslationStats]:
    """Emit schema-to-English translation SFT pairs.

    Args:
        manifest: A loaded ``PropertyManifest``. Surface forms are
            taken from the manifest's ``properties`` list (one CURIE
            per ``PropertyEntry``), so the cohort scales with the
            family. The hand-curated table in this module determines
            which CURIEs actually emit pairs; surface forms not in
            the table are skipped with a warning.
        capture: A ``DecisionCapture``-shaped object exposing
            ``log_decision(...)`` and a ``decisions`` list. Every
            emitted pair anchors ``decision_capture_id`` to the most
            recent event, and the generator emits one
            ``schema_translation_generation`` event per pair.
        max_pairs: Hard cap on emitted pairs. Default 50 is a
            backward-compatible cap; production rebuilds raise this
            via ``--schema-translation-max-pairs 200``. The catalog
            holds ~250 base entries.
        seed: Base seed; mirrored into pairs' ``seed`` field for
            replay determinism.

    Returns:
        ``(pairs, stats)`` — the pair list (instruction_pair shape)
        and a ``SchemaTranslationStats`` with per-CURIE + per-family
        counts.
    """
    if not isinstance(manifest, PropertyManifest):
        raise TypeError(
            "schema_translation_generator requires a PropertyManifest"
        )
    if max_pairs <= 0:
        raise ValueError(f"max_pairs must be > 0, got {max_pairs}")
    if capture is None:
        raise ValueError(
            "schema_translation_generator requires a DecisionCapture "
            "(got None); every emitted pair anchors decision_capture_id "
            "to a per-pair schema_translation_generation event."
        )

    pairs: List[Dict[str, Any]] = []
    stats = SchemaTranslationStats(
        surface_forms_total=len(manifest.properties),
    )
    # Initialise per-family counter so all 6 keys exist even if a
    # capped run never reaches some families.
    for family in _FAMILIES:
        stats.per_family[family] = 0

    # Wave 133d: dispatch the catalog load on manifest.family. Falls
    # back to the in-Python rdf_shacl table when family == "rdf_shacl"
    # AND no per-family YAML is on disk; returns empty + warns for any
    # other family with no on-disk YAML so the run continues but emits
    # zero schema-translation pairs (instead of crashing or silently
    # using the rdf_shacl catalog for non-RDF families).
    form_data = _load_form_data(manifest.family)

    # Wave 135a: filter out degraded-placeholder entries BEFORE catalog
    # walk so no pair body ever contains the literal "[degraded:" stub
    # text. Degraded entries are structural-contract scaffolding only;
    # the schema_translation generator emits pairs for ``"complete"``
    # entries exclusively. Wave 135b's force-injection path hooks the
    # same status field to know which CURIEs need token-stuffing fallback.
    degraded_skipped: List[str] = [
        curie
        for curie, entry in form_data.items()
        if entry.anchored_status == "degraded_placeholder"
    ]
    if degraded_skipped:
        form_data = {
            curie: entry
            for curie, entry in form_data.items()
            if entry.anchored_status != "degraded_placeholder"
        }

    # Surface forms missing from the table.
    seen_curies = set()
    for prop in manifest.properties:
        if prop.curie not in form_data:
            stats.surface_forms_skipped_no_definition += 1
            logger.warning(
                "schema_translation_generator: manifest declares %r "
                "but no hand-curated definition is on file; skipping.",
                prop.curie,
            )
        else:
            seen_curies.add(prop.curie)

    catalog = _build_catalog_in_order(manifest, form_data)

    # Trim catalog to exactly 250 if it overshoots (we author 252 = 6*6*7).
    # The trim drops the LAST 2 entries — guaranteed to be from the
    # last family/last forms, so balance stays >= 35 per form and >=
    # 30 per family.
    if len(catalog) > 250:
        catalog = catalog[:250]

    for curie, form, family, prompt, completion, extra_tags in catalog:
        if stats.pairs_emitted >= max_pairs:
            stats.capped_at_max_pairs = True
            break

        # Per-emit decision capture. Rationale interpolates dynamic
        # signals so audit replay distinguishes a paraphrase drift on
        # one CURIE from a wholesale table-vs-manifest mismatch.
        capture.log_decision(
            decision_type="schema_translation_generation",
            decision=(
                f"Emitting schema-translation {family} pair for "
                f"curie={curie!r} (short_name={form.short_name!r})."
            ),
            rationale=(
                f"Bridges formal CURIE {curie!r} to plain-English "
                f"{family}-family pair; pair {stats.pairs_emitted + 1} "
                f"of max_pairs={max_pairs}. seed={seed}, "
                f"manifest_family={manifest.family!r}, "
                f"surface_forms_in_manifest="
                f"{stats.surface_forms_total}, "
                f"family={family!r}, "
                f"prompt_chars={len(prompt)}, "
                f"completion_chars={len(completion)}."
            ),
            alternatives_considered=[
                {
                    "option": "LLM-paraphrase the SHACL/RDF spec text",
                    "reason_rejected": (
                        "deterministic generators are required by "
                        "the project's no-Claude-training-data "
                        "operating principle; spec text is concise "
                        "enough that hand-curated renderings beat "
                        "paraphrase risk."
                    ),
                },
                {
                    "option": (
                        "emit only definition + usage variants per CURIE"
                    ),
                    "reason_rejected": (
                        "Wave 124's 12-pair catalog under-trained "
                        "the model on comparison / reasoning / "
                        "pitfall / combination probes that the eval "
                        "harness explicitly tests for. Six template "
                        "families x 7 entries balances volume against "
                        "thesaurus-padding risk at ~6% of corpus."
                    ),
                },
            ],
        )
        decision_id = _last_event_id(capture)

        pair = _make_pair(
            prompt=prompt,
            completion=completion,
            family=family,
            curie=curie,
            extra_concept_tags=extra_tags,
            decision_capture_id=decision_id,
            seed=seed,
        )
        _validate_pair(pair)
        pairs.append(pair)
        stats.pairs_emitted += 1
        stats.per_surface_form[curie] = (
            stats.per_surface_form.get(curie, 0) + 1
        )
        stats.per_family[family] = stats.per_family.get(family, 0) + 1

    stats.surface_forms_used = len(seen_curies)

    # Wave 135a: end-of-run summary warning when one or more
    # degraded-placeholder entries were skipped. Operator action
    # (manifest backfill) flips them to ``"complete"`` over time.
    if degraded_skipped:
        logger.warning(
            "schema_translation_generator: skipped %d "
            "degraded-placeholder entries (anchored_status="
            "'degraded_placeholder'); these contribute zero pairs "
            "until operator backfill flips them to "
            "anchored_status='complete'. CURIEs: %s",
            len(degraded_skipped),
            sorted(degraded_skipped),
        )

    return pairs, stats


__all__ = [
    "DEFAULT_MAX_PAIRS",
    "SchemaTranslationStats",
    "SurfaceFormData",
    # Wave 137c: provenance dataclass + coercion helper.
    "Provenance",
    "generate_schema_translation_pairs",
    # Wave 135a contract surface.
    "validate_form_data_contract",
    # Wave 135b anchored-injection helper for the SFT / DPO factories.
    "resolve_anchor_text_for_curie",
    # Wave 133d loader-pattern surface — exported so tests can verify
    # the rdf_shacl fallback and so future per-family YAML callers
    # can re-use the loader without touching internals.
    "_RDF_SHACL_FALLBACK_FORM_DATA",
    "_load_form_data",
    # Wave 136a overlay-merge surface — Wave 136d's backfill loop
    # invalidates the cache between YAML appends so each round-trip
    # picks up freshly-authored CURIEs.
    "_invalidate_form_data_cache",
]
