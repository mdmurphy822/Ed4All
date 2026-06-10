"""Lexical domain-concept seed fallback — corpus-generalization fix.

Covers two surfaces of the ``TRAINFORGE_LEXICAL_CONCEPT_SEEDS`` fix:

1. The pure deriver ``lib.ontology.lexical_concept_seeds.
   derive_lexical_concept_seeds`` — harvests domain-concept slugs from chunk
   text using lexical / statistical signals only (no embeddings), and rejects
   function-word / boilerplate noise.

2. The end-to-end wiring in ``MCP/tools/pipeline_tools.py::
   _run_concept_extraction``:
   - flag ON + empty Stage-3 seeds → substantially MORE DomainConcept nodes
     than the empty-seed (flag OFF) path;
   - flag OFF (default) → the empty-seed path is byte-identical, i.e. the
     legacy degenerate behavior is preserved (RDF/SHACL corpus unaffected).

A small general-domain (cell-biology) chunk fixture is built inline so the
test does not depend on any course corpus on disk.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from lib.ontology.lexical_concept_seeds import (  # noqa: E402
    ALIAS_FOLD,
    derive_lexical_concept_seeds,
)
from lib.ontology.concept_node_merge import (  # noqa: E402
    merge_duplicate_concept_nodes,
)
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# General-domain (cell-biology) chunk fixture. No hand-set concept_tags — the
# point is that the lexical deriver recovers concepts from prose alone, which
# is exactly the empty-seed path that produces a degenerate graph today.
# ---------------------------------------------------------------------------

_BIO_SENTENCES = [
    "The cell membrane is a phospholipid bilayer that regulates transport.",
    "Active transport across the cell membrane requires ATP energy.",
    "Cellular respiration in the mitochondria produces ATP from glucose.",
    "The mitochondria are the site of cellular respiration and ATP synthesis.",
    "Protein synthesis occurs at the ribosome using messenger RNA.",
    "Messenger RNA carries the genetic code from DNA to the ribosome.",
    "DNA replication is a semiconservative process in the cell nucleus.",
    "The cell nucleus stores DNA and controls protein synthesis.",
    "Photosynthesis in the chloroplast converts light energy into glucose.",
    "The chloroplast contains chlorophyll that captures light energy.",
    "Enzyme catalysis lowers the activation energy of a reaction.",
    "An enzyme is a protein catalyst central to cellular metabolism.",
    "The phospholipid bilayer of the cell membrane is selectively permeable.",
    "Glucose metabolism feeds cellular respiration in the mitochondria.",
    "Genetic information flows from DNA to messenger RNA to protein.",
]


def _bio_chunkset() -> List[Dict[str, Any]]:
    """A general-domain chunkset with EMPTY concept_tags + rich prose.

    Each domain concept (cell membrane, mitochondria, ATP, DNA, ribosome,
    cellular respiration, …) recurs across multiple chunks so it clears both
    the lexical ``min_doc_freq`` floor and the downstream co-occurrence floor.
    """
    chunks: List[Dict[str, Any]] = []
    # 30 chunks: cycle the 15 sentences twice, pairing adjacent sentences so
    # concepts recur and co-occur.
    for i in range(30):
        a = _BIO_SENTENCES[i % len(_BIO_SENTENCES)]
        b = _BIO_SENTENCES[(i + 1) % len(_BIO_SENTENCES)]
        chunks.append({
            "id": f"bio_chunk_{i:05d}",
            "text": f"{a} {b}",
            "chunk_type": "content",
            "concept_tags": [],
            "learning_outcome_refs": [],
            "bloom_level": "understand",
            "difficulty": "intermediate",
            "source": {
                "module_id": f"week_{(i // 5) + 1:02d}",
                "item_path": f"week_{(i // 5) + 1:02d}/page_{i:03d}.html",
            },
        })
    return chunks


def _write_chunkset(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


def _run_extraction(
    tmp_path: Path,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    chunks_path = tmp_path / "dart_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, chunks)
    custom_libv2 = tmp_path / "libv2"
    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="BIO_LEXICAL",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            libv2_root=str(custom_libv2),
        )
    )
    payload = json.loads(result)
    graph = json.loads(Path(payload["concept_graph_path"]).read_text("utf-8"))
    payload["_graph"] = graph
    return payload


def _domain_concept_node_count(graph: Dict[str, Any]) -> int:
    classes = {"Concept", "DomainConcept", "concept", "domain_concept"}
    return sum(1 for n in (graph.get("nodes") or []) if n.get("class") in classes)


# ---------------------------------------------------------------------------
# (c) Deriver rejects function-word / boilerplate noise.
# ---------------------------------------------------------------------------


def test_deriver_recovers_domain_concepts() -> None:
    seeds = derive_lexical_concept_seeds(_bio_chunkset())
    # Multi-word + acronym domain concepts should be recovered.
    assert "cell-membrane" in seeds
    assert "cellular-respiration" in seeds
    # Acronyms (ATP, DNA, RNA) are high-signal and kept even at low freq.
    assert "atp" in seeds
    assert "dna" in seeds
    # A real corpus yields many seeds, not the ~0 of the empty-seed path.
    assert len(seeds) >= 5, f"too few seeds: {seeds}"


def test_deriver_rejects_function_word_and_boilerplate_noise() -> None:
    noisy = [
        {"id": "n0", "text": (
            "Accessibility is easier to use and must be one of the things "
            "that you should keep in mind. This content was converted by "
            "DART under the Open Government Licence. See the table of "
            "contents above for more."
        ) * 4},
        {"id": "n1", "text": (
            "It is also very important that the following example and the "
            "figure on this page are shown such as they are. "
            "Characters or less."
        ) * 4},
    ]
    seeds = derive_lexical_concept_seeds(noisy, min_doc_freq=1)
    lowered = set(seeds)
    # Syntactic fragments — any token a function word → rejected.
    for frag in (
        "accessibility-is", "easier-to", "must-be", "such-as",
        "is-also", "you-should",
    ):
        assert frag not in lowered, f"function-word fragment leaked: {frag}"
    # Pipeline / licence / boilerplate denylist.
    for junk in (
        "converted-by-dart", "open-government-licence",
        "table-of-contents", "characters-or-less",
    ):
        assert junk not in lowered, f"boilerplate leaked: {junk}"
    # Generic scaffolding nouns.
    for generic in ("the-following", "this-page", "the-figure"):
        assert generic not in lowered, f"generic term leaked: {generic}"


# ---------------------------------------------------------------------------
# ALIAS_FOLD domain-synonym additions — zero-shared-chunk folds (Pass A folds
# unconditionally). Assert key/value direction (variant -> canonical) holds.
# ---------------------------------------------------------------------------


def test_alias_fold_domain_synonym_entries_present() -> None:
    # Variant (key) folds onto canonical (value).
    assert ALIAS_FOLD["recursive-chunking"] == "chunking"
    assert ALIAS_FOLD["nvidia-nim-microservices"] == "nvidia-nim"
    assert ALIAS_FOLD["nvidia-ngc-service"] == "nvidia-ngc-catalog"
    assert ALIAS_FOLD["ragas-short-for-rag"] == "ragas"
    assert ALIAS_FOLD["openais-chatgpt"] == "chatgpt"


def _alias_concept(node_id: str, frequency: int) -> Dict[str, Any]:
    return {
        "id": node_id,
        "label": node_id.replace("-", " ").title(),
        "frequency": frequency,
        "class": "DomainConcept",
    }


def test_alias_fold_merges_to_canonical_no_shared_chunk() -> None:
    # No shared occurrences[] / edge: Pass A still folds (unconditional).
    graph = {
        "nodes": [
            _alias_concept("recursive-chunking", 2),
            _alias_concept("chunking", 5),
            _alias_concept("nvidia-nim-microservices", 1),
            _alias_concept("nvidia-nim", 4),
        ],
        "edges": [],
    }
    out = merge_duplicate_concept_nodes(graph)
    ids = sorted(n["id"] for n in out["nodes"])
    # Variants folded away; canonicals survive.
    assert ids == ["chunking", "nvidia-nim"]
    by_id = {n["id"]: n for n in out["nodes"]}
    assert by_id["chunking"]["frequency"] == 7  # 5 + 2
    assert by_id["nvidia-nim"]["frequency"] == 5  # 4 + 1
    assert out["concept_node_merges"] == 2


def test_deriver_empty_input_returns_empty() -> None:
    assert derive_lexical_concept_seeds([]) == []
    assert derive_lexical_concept_seeds([{"id": "x"}]) == []


def test_deriver_is_deterministic() -> None:
    a = derive_lexical_concept_seeds(_bio_chunkset())
    b = derive_lexical_concept_seeds(_bio_chunkset())
    assert a == b


# ---------------------------------------------------------------------------
# (a) Flag ON → substantially more DomainConcept nodes than empty-seed path.
# (b) Flag OFF (default) → empty-seed path unchanged.
# ---------------------------------------------------------------------------


def test_flag_off_yields_degenerate_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default (flag unset): the empty-seed path produces a degenerate
    DomainConcept node set — the legacy behavior we must preserve.
    """
    monkeypatch.delenv("TRAINFORGE_LEXICAL_CONCEPT_SEEDS", raising=False)
    monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)
    payload = _run_extraction(tmp_path, _bio_chunkset())
    assert payload.get("success") is True
    # No lexical keys when the flag is off.
    assert "lexical_concept_seed_count" not in payload
    off_nodes = _domain_concept_node_count(payload["_graph"])
    # The empty-seed path yields essentially no domain concepts.
    assert off_nodes <= 2, (
        f"empty-seed path unexpectedly rich ({off_nodes} concept nodes); "
        "legacy degenerate behavior changed."
    )


def test_flag_on_recovers_domain_concept_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag ON + empty Stage-3 seeds: substantially more DomainConcept
    nodes than the flag-OFF empty-seed path.
    """
    monkeypatch.delenv("TEXTBOOK_SYNTHESIS_PROVIDER", raising=False)

    # Baseline: flag OFF.
    monkeypatch.delenv("TRAINFORGE_LEXICAL_CONCEPT_SEEDS", raising=False)
    off = _run_extraction(tmp_path / "off", _bio_chunkset())
    off_nodes = _domain_concept_node_count(off["_graph"])

    # Treatment: flag ON.
    monkeypatch.setenv("TRAINFORGE_LEXICAL_CONCEPT_SEEDS", "true")
    on = _run_extraction(tmp_path / "on", _bio_chunkset())
    on_nodes = _domain_concept_node_count(on["_graph"])

    assert on["success"] is True
    assert on.get("lexical_concept_seed_count", 0) > 0, (
        f"lexical fallback did not fire; payload={on!r}"
    )
    assert on.get("lexical_chunks_retagged", 0) > 0
    assert on_nodes > off_nodes, (
        f"flag ON ({on_nodes} concept nodes) should beat flag OFF "
        f"({off_nodes}); the lexical fallback did not enrich the graph."
    )
    # Substantially more — clears the ConceptGraphValidator min_nodes floor.
    assert on_nodes >= 8, (
        f"flag ON should recover a substantial concept set; got {on_nodes}."
    )
