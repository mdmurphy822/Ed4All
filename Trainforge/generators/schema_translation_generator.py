"""Schema-to-English translation SFT pair generator (Wave 124,
audit 2026-04-30 fix).

The cc07cc76 SLM adapter scored hallucination_rate=0.63 / faithfulness=0.37
in part because the corpus had no pairs that taught the model to map
formal SHACL/RDF/OWL surface forms (``sh:datatype``,
``rdfs:subClassOf``, ``owl:sameAs``, ...) to plain-English meanings.
The eval harness probes via property-aware questions; without a
schema-to-English bridge the adapter either parrots the surface form
back or hallucinates an unrelated paraphrase.

This generator is driven by ``lib/ontology/property_manifest.py``: it
loads the family manifest (e.g. ``property_manifest.rdf_shacl.yaml``),
walks every declared property, and emits two deterministic pairs per
surface form — one definition pair and one usage pair — directly from
a hand-curated dictionary in this module. Definitions are authored
from the canonical SHACL / RDF / RDFS / OWL specs; the goal is a
faithful one-paragraph rendering of "what does this CURIE mean and
how is it used".

Pair shape carries:

  * ``content_type="schema_translation"`` — downstream filters /
    diversity scorers can isolate the cohort without re-parsing prompts.
  * ``bloom_level="remember"`` for definition pairs;
    ``"understand"`` for usage pairs.
  * ``template_id="schema_translation.definition"`` /
    ``"schema_translation.usage"``.
  * ``concept_tags=[curie]`` — anchors the surface form so a future
    eval can re-verify the bridge.
  * The literal CURIE appears in every completion so
    ``preserve_tokens`` plumbing in synthesize_training.py recognises
    the surface form (matches the manifest's preserve-token contract).

Decision capture: one ``schema_translation_generation`` event per
emitted pair, rationale interpolating curie + variant
(definition/usage) so audit replay can spot a drift between manifest
surface forms and the hand-curated definition table.

Implementation note: the manifest is the single source of truth for
the set of surface forms; this generator's hand-curated table is
indexed by CURIE and falls through silently for any CURIE the
manifest declares but this table doesn't define. That keeps the
manifest extensible — adding a new property doesn't break the
generator, it just doesn't generate translation pairs until the
table catches up. A logger.warning surfaces the gap.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.property_manifest import (  # noqa: E402
    PropertyManifest,
)

logger = logging.getLogger(__name__)


# Default total cap. 2 pairs * 6 surface forms = 12 base pairs; the
# 50 cap leaves room for future expansion (e.g. additional usage
# variants per surface form) without changing call sites.
DEFAULT_MAX_PAIRS = 50


@dataclass
class SchemaTranslationStats:
    """Counts returned from :func:`generate_schema_translation_pairs`."""

    surface_forms_total: int = 0
    surface_forms_used: int = 0
    surface_forms_skipped_no_definition: int = 0
    pairs_emitted: int = 0
    capped_at_max_pairs: bool = False
    per_surface_form: Dict[str, int] = field(default_factory=dict)


# Hand-curated definition + usage table. Authored from the canonical
# SHACL / RDF / RDFS / OWL specs. Keys are CURIEs (matching
# ``PropertyEntry.curie`` in the manifest). Values are
# ``(definition, usage)`` tuples; both are 50-1500 chars to satisfy
# the schema floor + cap on completion length, both contain the
# literal CURIE so preserve_tokens picks it up.
_TRANSLATION_TABLE: Dict[str, Tuple[str, str]] = {
    "sh:datatype": (
        # Definition.
        "sh:datatype is a SHACL property-shape constraint that "
        "restricts the values of a property to RDF literals of a "
        "specific datatype. It expects an IRI naming an XSD or "
        "user-defined datatype. A value passes the constraint only "
        "when it is a literal whose datatype IRI matches the one "
        "named by sh:datatype.",
        # Usage.
        "Use sh:datatype on a sh:PropertyShape to require literal-"
        "typed values. For example, sh:datatype xsd:integer requires "
        "every value of the constrained property to be an integer-"
        "typed RDF literal; a value typed as xsd:string or an IRI "
        "would fail the constraint and produce a sh:Violation.",
    ),
    "sh:class": (
        # Definition.
        "sh:class is a SHACL property-shape constraint that requires "
        "each value of a property to be a SHACL instance of a given "
        "class. It expects an IRI naming the required class. A value "
        "passes when its rdf:type (transitively, via rdfs:subClassOf) "
        "includes the named class.",
        # Usage.
        "Use sh:class on a sh:PropertyShape to require IRI- or blank-"
        "node-typed values. For example, sh:class ex:Person requires "
        "every value to be a SHACL instance of ex:Person; a literal "
        "value or an instance of an unrelated class would fail the "
        "sh:class constraint at validation time.",
    ),
    "sh:NodeShape": (
        # Definition.
        "sh:NodeShape is the SHACL class of shapes that constrain "
        "RDF nodes themselves rather than the values of a single "
        "property. A node shape lists constraints (sh:property, "
        "sh:nodeKind, sh:targetClass, ...) that the focus node must "
        "satisfy. It is the SHACL counterpart to a class-level "
        "schema.",
        # Usage.
        "Declare a shape with rdf:type sh:NodeShape (or via "
        "sh:targetClass) when you want to constrain whole nodes — "
        "typically every instance of a domain class. Property-level "
        "constraints are nested inside the node shape via sh:property "
        "links to sh:PropertyShape instances.",
    ),
    "sh:PropertyShape": (
        # Definition.
        "sh:PropertyShape is the SHACL class of shapes that constrain "
        "the values of a single RDF property at a given node. A "
        "property shape names the constrained predicate via sh:path "
        "and lists value constraints (sh:datatype, sh:class, "
        "sh:minCount, ...) that the values must satisfy.",
        # Usage.
        "Reference a sh:PropertyShape from a sh:NodeShape via "
        "sh:property. The property shape's sh:path picks the predicate, "
        "and its value-constraint properties (sh:datatype, sh:class, "
        "sh:minCount, ...) are evaluated for each value found at that "
        "predicate of the focus node.",
    ),
    "rdfs:subClassOf": (
        # Definition.
        "rdfs:subClassOf is an RDFS predicate stating that the "
        "subject class is a subset of the object class — every "
        "instance of the subject is also an instance of the object. "
        "RDFS entailment propagates rdf:type along rdfs:subClassOf "
        "transitively, so subclass relationships chain.",
        # Usage.
        "Assert ex:Student rdfs:subClassOf ex:Person to declare that "
        "every Student is also a Person. RDFS-aware reasoners and "
        "SHACL validators with rdfs:subClassOf entailment will then "
        "match an instance of ex:Student against shapes targeted at "
        "ex:Person.",
    ),
    "owl:sameAs": (
        # Definition.
        "owl:sameAs is an OWL predicate asserting that two IRIs "
        "denote the same individual. An owl:sameAs link merges all "
        "facts about the two IRIs — predicates, types, and "
        "annotations — into a single conceptual entity from the "
        "reasoner's point of view.",
        # Usage.
        "Use owl:sameAs to bridge identifiers across datasets — for "
        "example, dbr:Albert_Einstein owl:sameAs wd:Q937 declares "
        "that the DBpedia and Wikidata IRIs refer to the same person. "
        "Reasoners then propagate every fact stated about either IRI "
        "to both.",
    ),
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


def _build_definition_pair(
    *,
    curie: str,
    definition: str,
    decision_capture_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Render the definition variant pair for one CURIE.

    The prompt is padded with a "for an RDF/SHACL learner" suffix so
    it clears the schema's 40-char minLength floor on every CURIE
    (the bare "What does <curie> mean in SHACL?" is 36 chars on the
    shortest CURIE).
    """
    suffix = ", for an RDF/SHACL learner?"
    if curie.startswith("rdfs:"):
        prompt = f"What does {curie} mean in RDFS{suffix}"
    elif curie.startswith("owl:"):
        prompt = f"What does {curie} mean in OWL{suffix}"
    else:
        prompt = f"What does {curie} mean in SHACL{suffix}"
    return {
        "prompt": prompt,
        "completion": definition,
        # Surface forms aren't anchored to a single chunk; "schema-
        # translation" is a sentinel chunk_id that downstream filters
        # can recognise + treat as graph-wide rather than chunk-local.
        "chunk_id": "schema-translation",
        "lo_refs": ["schema-translation"],
        "bloom_level": "remember",
        "content_type": "schema_translation",
        "seed": seed,
        "decision_capture_id": decision_capture_id,
        "template_id": "schema_translation.definition",
        "provider": "mock",
        "schema_version": "v1",
        "requires_source_citation": False,
        "concept_tags": [curie],
    }


def _build_usage_pair(
    *,
    curie: str,
    usage: str,
    decision_capture_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Render the usage variant pair for one CURIE.

    Padded for the same 40-char schema floor as the definition prompt.
    """
    prompt = (
        f"How is {curie} used in an RDF/SHACL knowledge graph?"
    )
    return {
        "prompt": prompt,
        "completion": usage,
        "chunk_id": "schema-translation",
        "lo_refs": ["schema-translation"],
        "bloom_level": "understand",
        "content_type": "schema_translation",
        "seed": seed,
        "decision_capture_id": decision_capture_id,
        "template_id": "schema_translation.usage",
        "provider": "mock",
        "schema_version": "v1",
        "requires_source_citation": False,
        "concept_tags": [curie],
    }


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
        max_pairs: Hard cap on emitted pairs. Default 50 leaves head-
            room above the 6 * 2 = 12 base pairs for future variants.
        seed: Base seed; mirrored into pairs' ``seed`` field for
            replay determinism.

    Returns:
        ``(pairs, stats)`` — the pair list (instruction_pair shape)
        and a ``SchemaTranslationStats`` with per-CURIE counts.
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

    # Iterate properties in manifest declaration order so the emit
    # order is deterministic and tied to the manifest itself.
    for prop in manifest.properties:
        curie = prop.curie
        defn_usage = _TRANSLATION_TABLE.get(curie)
        if defn_usage is None:
            stats.surface_forms_skipped_no_definition += 1
            logger.warning(
                "schema_translation_generator: manifest declares "
                "%r but no hand-curated definition is on file; skipping.",
                curie,
            )
            continue

        definition, usage = defn_usage

        for variant in ("definition", "usage"):
            if stats.pairs_emitted >= max_pairs:
                stats.capped_at_max_pairs = True
                break

            # Per-emit decision. Rationale interpolates dynamic signals
            # so audit replay distinguishes a paraphrase drift on one
            # CURIE from a wholesale table-vs-manifest mismatch. Wave 22
            # alternatives_considered convention: list of {option,
            # reason_rejected} dicts.
            capture.log_decision(
                decision_type="schema_translation_generation",
                decision=(
                    f"Emitting schema-translation {variant} pair for "
                    f"curie={curie!r} (label={prop.label!r})."
                ),
                rationale=(
                    f"Bridges formal CURIE {curie!r} to plain-English "
                    f"{variant}; pair {stats.pairs_emitted + 1} of "
                    f"max_pairs={max_pairs}. seed={seed}, "
                    f"manifest_family={manifest.family!r}, "
                    f"surface_forms_in_manifest="
                    f"{stats.surface_forms_total}."
                ),
                alternatives_considered=[
                    {
                        "option": "LLM-paraphrase the SHACL/RDF spec text",
                        "reason_rejected": (
                            "deterministic generators are required by "
                            "the project's no-Claude-training-data "
                            "operating principle; spec text is "
                            "concise enough that hand-curated "
                            "renderings beat paraphrase risk."
                        ),
                    },
                    {
                        "option": "emit a single pair per CURIE",
                        "reason_rejected": (
                            "definition + usage are distinct cognitive "
                            "tasks (remember vs understand) and the "
                            "eval harness probes both surfaces."
                        ),
                    },
                ],
            )
            decision_id = _last_event_id(capture)

            if variant == "definition":
                pair = _build_definition_pair(
                    curie=curie,
                    definition=definition,
                    decision_capture_id=decision_id,
                    seed=seed,
                )
            else:
                pair = _build_usage_pair(
                    curie=curie,
                    usage=usage,
                    decision_capture_id=decision_id,
                    seed=seed,
                )

            _validate_pair(pair)
            pairs.append(pair)
            stats.pairs_emitted += 1
            stats.per_surface_form[curie] = (
                stats.per_surface_form.get(curie, 0) + 1
            )

        if stats.pairs_emitted == 0 or stats.per_surface_form.get(curie, 0) > 0:
            # We at least attempted this surface form (might have been
            # capped mid-variant; the per_surface_form count is the
            # truthful signal).
            stats.surface_forms_used += 1

        if stats.capped_at_max_pairs:
            break

    return pairs, stats


__all__ = [
    "DEFAULT_MAX_PAIRS",
    "SchemaTranslationStats",
    "generate_schema_translation_pairs",
]
