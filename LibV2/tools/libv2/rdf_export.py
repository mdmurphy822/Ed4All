"""Export a LibV2 course's JSON artifacts as RDF (Phase 1.5).

The Wave 1+2 work added JSON-LD ``@context`` files at
``schemas/context/*_v1.jsonld`` that bridge each per-course JSON
artifact (``course.json``, ``concept_graph_semantic.json``,
``pedagogy_graph.json``) to RDF without rewriting the data. This
module consumes those contexts to materialize Turtle files on disk so
downstream RDF tooling (Protégé, SPARQL stores, pyshacl pipelines) can
ingest the package without a JSON-LD-aware parser.

The bridge is consumer-side: Trainforge still emits JSON. This module
layers the matching ``@context`` on each artifact at export time and
runs ``pyld.to_rdf`` → ``rdflib.parse`` → ``rdflib.serialize`` to
produce ``.ttl`` (or ``.trig``) on disk.

Why this is in LibV2 rather than Trainforge: the export is a read-side
operation against an already-archived course, not part of the emit
pipeline. LibV2 owns the post-archive surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Per-artifact registry: (relative path under courses/<slug>/, @context filename, anchor IRI suffix)
# The anchor IRI prevents top-level metadata from landing on a blank
# node; mirrors the pattern in test_concept_graph_jsonld_roundtrip.py.
_ARTIFACT_REGISTRY: List[Tuple[str, str, str]] = [
    ("course.json", "course_v1.jsonld", "course"),
    ("graph/concept_graph_semantic.json", "concept_graph_semantic_v1.jsonld", "concept-graph"),
    ("graph/pedagogy_graph.json", "pedagogy_graph_v1.jsonld", "pedagogy-graph"),
]

# Formats that require a context-aware (named-graph) store — must be
# materialized as ``rdflib.Dataset`` rather than ``rdflib.Graph``,
# otherwise rdflib raises "NQuads serialization only makes sense for
# context-aware stores!" on serialize.
_CONTEXT_AWARE_FORMATS = {"nquads", "n-quads", "trig"}

_ANCHOR_BASE = "https://ed4all.io/"


@dataclass
class ExportResult:
    """One artifact's export outcome."""

    artifact_relpath: str
    context_path: str
    output_path: str
    triple_count: int

    def to_dict(self) -> dict:
        return {
            "artifact": self.artifact_relpath,
            "context": self.context_path,
            "output": self.output_path,
            "triples": self.triple_count,
        }


def _find_project_root(libv2_root: Path) -> Path:
    """LibV2 sits as a subdir of the project; ``schemas/context/``
    lives at the project root, not the LibV2 root. Walk up to find it."""
    candidate = libv2_root.resolve()
    for _ in range(4):
        if (candidate / "schemas" / "context").is_dir():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        f"Could not locate schemas/context/ above {libv2_root}; "
        "rdf_export requires the project schemas tree."
    )


def _materialize_rdf_graph(
    json_path: Path,
    context_path: Path,
    anchor_iri: str,
    output_format: str = "turtle",
):
    """Load JSON, inject @context, serialize to N-Quads, parse via rdflib.

    Returns an ``rdflib.Graph`` for triple formats (turtle, ntriples,
    xml) and an ``rdflib.Dataset`` for context-aware formats (nquads,
    trig). Imports rdflib + pyld lazily so the module is importable
    even when the dependencies aren't installed.
    """
    from pyld import jsonld
    import rdflib

    with json_path.open() as f:
        artifact = json.load(f)
    with context_path.open() as f:
        context_doc = json.load(f)

    if "@context" not in context_doc:
        raise ValueError(f"Context file missing @context block: {context_path}")

    doc = dict(artifact)
    doc["@context"] = context_doc["@context"]
    doc["@id"] = anchor_iri

    nquads = jsonld.to_rdf(doc, {"format": "application/n-quads"})
    if output_format.lower() in _CONTEXT_AWARE_FORMATS:
        store = rdflib.Dataset()
    else:
        store = rdflib.Graph()
    store.parse(data=nquads, format="nquads")
    return store


def export_course(
    repo_root: Path,
    course_slug: str,
    output_dir: Path,
    *,
    output_format: str = "turtle",
) -> List[ExportResult]:
    """Export every JSON artifact under ``courses/<slug>/`` as RDF.

    ``repo_root`` is the LibV2 root (the directory containing
    ``courses/`` and ``catalog/``). ``output_dir`` is created if
    missing; one ``.ttl`` (or ``.trig`` / ``.nq``) file per artifact
    lands inside, named after the source artifact.

    Skips silently when an artifact is missing on disk (some courses
    don't have every artifact). Raises if the @context file for an
    existing artifact is missing — that's a vocabulary regression.
    """
    course_dir = repo_root / "courses" / course_slug
    if not course_dir.is_dir():
        raise FileNotFoundError(f"Course not found: {course_dir}")

    project_root = _find_project_root(repo_root)
    context_dir = project_root / "schemas" / "context"

    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[ExportResult] = []
    for relpath, ctx_name, anchor_suffix in _ARTIFACT_REGISTRY:
        json_path = course_dir / relpath
        if not json_path.is_file():
            continue
        ctx_path = context_dir / ctx_name
        if not ctx_path.is_file():
            raise FileNotFoundError(
                f"Context file missing: {ctx_path} — required for artifact {json_path}"
            )

        anchor_iri = f"{_ANCHOR_BASE}{anchor_suffix}/{course_slug}"
        graph = _materialize_rdf_graph(json_path, ctx_path, anchor_iri, output_format)

        ext = _format_extension(output_format)
        out_path = output_dir / f"{json_path.stem}{ext}"
        graph.serialize(destination=str(out_path), format=output_format)

        results.append(
            ExportResult(
                artifact_relpath=str(relpath),
                context_path=str(ctx_path),
                output_path=str(out_path),
                triple_count=len(graph),
            )
        )
    return results


def _format_extension(output_format: str) -> str:
    return {
        "turtle": ".ttl",
        "ttl": ".ttl",
        "trig": ".trig",
        "nquads": ".nq",
        "n-quads": ".nq",
        "ntriples": ".nt",
        "n-triples": ".nt",
        "xml": ".rdf",
    }.get(output_format.lower(), ".ttl")


__all__ = ["ExportResult", "export_course"]
