"""
Concept Graph Builder for Semantic Structure Extraction

Builds a concept co-occurrence graph and calculates centrality metrics
to identify the most important concepts for slide prioritization.

Implements:
- TF-IDF scoring
- PageRank centrality
- Betweenness centrality
- Composite importance scoring
"""

import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum


@dataclass
class ConceptNode:
    """A node in the concept graph."""
    term: str
    normalized_term: str
    frequency: int = 0
    document_frequency: int = 0  # Number of sections containing this concept
    sections_present: List[str] = field(default_factory=list)
    co_occurring_concepts: Dict[str, int] = field(default_factory=dict)
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    importance_rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "normalizedTerm": self.normalized_term,
            "frequency": self.frequency,
            "documentFrequency": self.document_frequency,
            "sectionsPresent": self.sections_present,
            "coOccurringConcepts": dict(list(self.co_occurring_concepts.items())[:10]),
            "centralityScores": {k: round(v, 4) for k, v in self.centrality_scores.items()},
            "compositeScore": round(self.composite_score, 4),
            "importanceRank": self.importance_rank
        }


@dataclass
class ConceptEdge:
    """An edge in the concept graph (co-occurrence relationship)."""
    source: str
    target: str
    weight: float = 1.0
    co_occurrence_count: int = 0
    sections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "coOccurrenceCount": self.co_occurrence_count
        }


@dataclass
class ConceptGraph:
    """The complete concept graph."""
    nodes: Dict[str, ConceptNode] = field(default_factory=dict)
    edges: List[ConceptEdge] = field(default_factory=list)
    total_sections: int = 0
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges[:100]],  # Limit edges
            "totalSections": self.total_sections,
            "statistics": self.statistics,
            "topConcepts": self.get_top_concepts_list(20)
        }

    def get_top_concepts_list(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get top N concepts as a simple list."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda x: x.composite_score,
            reverse=True
        )
        return [
            {
                "term": node.term,
                "score": round(node.composite_score, 4),
                "frequency": node.frequency,
                "rank": node.importance_rank
            }
            for node in sorted_nodes[:n]
        ]


class CentralityAlgorithm(Enum):
    """Available centrality algorithms."""
    FREQUENCY = "frequency"
    TFIDF = "tfidf"
    PAGERANK = "pagerank"
    BETWEENNESS = "betweenness"
    COMPOSITE = "composite"


class ConceptGraphBuilder:
    """
    Builds concept graphs and calculates centrality metrics.

    Used to identify key concepts that should be emphasized in presentations.
    """

    # Common stopwords to filter
    STOPWORDS: Set[str] = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'we', 'they', 'what', 'which', 'who', 'how', 'all', 'each', 'also'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the builder.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        config_path = Path(__file__).parent / "config" / "extractor_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def build_graph(self, structure: Dict[str, Any]) -> ConceptGraph:
        """
        Build concept graph from semantic structure.

        Args:
            structure: Semantic structure dictionary with chapters/sections

        Returns:
            ConceptGraph with nodes and edges
        """
        graph = ConceptGraph()

        # Extract concepts from all sections
        sections = self._extract_all_sections(structure)
        graph.total_sections = len(sections)

        # Build nodes from concepts in each section
        section_concepts: Dict[str, Set[str]] = {}

        for section in sections:
            section_id = section.get('id', '')
            concepts = self._extract_section_concepts(section)
            section_concepts[section_id] = concepts

            for concept in concepts:
                norm_concept = concept.lower()
                if norm_concept not in graph.nodes:
                    graph.nodes[norm_concept] = ConceptNode(
                        term=concept,
                        normalized_term=norm_concept
                    )
                graph.nodes[norm_concept].frequency += 1
                if section_id not in graph.nodes[norm_concept].sections_present:
                    graph.nodes[norm_concept].sections_present.append(section_id)
                    graph.nodes[norm_concept].document_frequency += 1

        # Build edges from co-occurrences
        edge_map: Dict[Tuple[str, str], ConceptEdge] = {}

        for section_id, concepts in section_concepts.items():
            concept_list = list(concepts)
            for i, c1 in enumerate(concept_list):
                for c2 in concept_list[i+1:]:
                    norm_c1, norm_c2 = c1.lower(), c2.lower()
                    # Ensure consistent edge key ordering
                    edge_key = tuple(sorted([norm_c1, norm_c2]))

                    if edge_key not in edge_map:
                        edge_map[edge_key] = ConceptEdge(
                            source=edge_key[0],
                            target=edge_key[1]
                        )

                    edge_map[edge_key].co_occurrence_count += 1
                    if section_id not in edge_map[edge_key].sections:
                        edge_map[edge_key].sections.append(section_id)

                    # Update node co-occurrence maps
                    if norm_c1 in graph.nodes and norm_c2 in graph.nodes:
                        graph.nodes[norm_c1].co_occurring_concepts[norm_c2] = (
                            graph.nodes[norm_c1].co_occurring_concepts.get(norm_c2, 0) + 1
                        )
                        graph.nodes[norm_c2].co_occurring_concepts[norm_c1] = (
                            graph.nodes[norm_c2].co_occurring_concepts.get(norm_c1, 0) + 1
                        )

        # Convert edge weights (normalize by max co-occurrence)
        max_cooccurrence = max((e.co_occurrence_count for e in edge_map.values()), default=1)
        for edge in edge_map.values():
            edge.weight = edge.co_occurrence_count / max_cooccurrence

        graph.edges = list(edge_map.values())

        # Calculate centrality scores
        self.calculate_centrality(graph)

        # Calculate statistics
        graph.statistics = self._calculate_statistics(graph)

        return graph

    def calculate_centrality(self, graph: ConceptGraph) -> None:
        """
        Calculate all centrality metrics for the graph.

        Args:
            graph: ConceptGraph to update with centrality scores
        """
        config = self.config.get('concept_extraction', {})
        algorithm = config.get('centrality_algorithm', 'composite')

        # Calculate individual centrality metrics
        self._calculate_frequency_centrality(graph)
        self._calculate_tfidf(graph)
        self._calculate_pagerank(graph)
        self._calculate_betweenness(graph)

        # Calculate composite score
        weights = config.get('centrality_weights', {
            'frequency': 0.2,
            'tfidf': 0.3,
            'pagerank': 0.3,
            'betweenness': 0.2
        })

        self._calculate_composite_score(graph, weights)

        # Assign importance ranks
        sorted_nodes = sorted(
            graph.nodes.values(),
            key=lambda x: x.composite_score,
            reverse=True
        )
        for rank, node in enumerate(sorted_nodes, 1):
            node.importance_rank = rank

    def _calculate_frequency_centrality(self, graph: ConceptGraph) -> None:
        """Calculate simple frequency-based centrality."""
        max_freq = max((n.frequency for n in graph.nodes.values()), default=1)
        for node in graph.nodes.values():
            node.centrality_scores['frequency'] = node.frequency / max_freq

    def _calculate_tfidf(self, graph: ConceptGraph) -> None:
        """Calculate TF-IDF scores."""
        total_docs = graph.total_sections or 1

        for node in graph.nodes.values():
            # Term frequency (normalized)
            tf = node.frequency
            # Inverse document frequency
            idf = math.log(total_docs / (node.document_frequency + 1)) + 1
            node.centrality_scores['tfidf'] = tf * idf

        # Normalize TF-IDF scores
        max_tfidf = max((n.centrality_scores.get('tfidf', 0) for n in graph.nodes.values()), default=1)
        if max_tfidf > 0:
            for node in graph.nodes.values():
                node.centrality_scores['tfidf'] /= max_tfidf

    def _calculate_pagerank(self, graph: ConceptGraph, damping: float = 0.85, iterations: int = 100) -> None:
        """
        Calculate PageRank centrality.

        Concepts that co-occur with important concepts are themselves important.
        """
        if not graph.nodes:
            return

        n = len(graph.nodes)
        node_ids = list(graph.nodes.keys())

        # Initialize PageRank scores
        pr = {node_id: 1.0 / n for node_id in node_ids}

        # Build adjacency list with weights
        adj: Dict[str, Dict[str, float]] = defaultdict(dict)
        for edge in graph.edges:
            adj[edge.source][edge.target] = edge.weight
            adj[edge.target][edge.source] = edge.weight

        # Iterate
        for _ in range(iterations):
            new_pr = {}
            for node_id in node_ids:
                # Sum of weighted contributions from neighbors
                rank_sum = 0.0
                for neighbor, weight in adj[node_id].items():
                    if neighbor in pr:
                        # Normalize by total outgoing weight
                        total_out = sum(adj[neighbor].values()) or 1
                        rank_sum += pr[neighbor] * weight / total_out

                new_pr[node_id] = (1 - damping) / n + damping * rank_sum

            pr = new_pr

        # Normalize and assign
        max_pr = max(pr.values()) or 1
        for node_id, score in pr.items():
            graph.nodes[node_id].centrality_scores['pagerank'] = score / max_pr

    def _calculate_betweenness(self, graph: ConceptGraph) -> None:
        """
        Calculate betweenness centrality.

        Concepts that bridge different topics are important.
        """
        if not graph.nodes or not graph.edges:
            for node in graph.nodes.values():
                node.centrality_scores['betweenness'] = 0.0
            return

        node_ids = list(graph.nodes.keys())
        betweenness = {node_id: 0.0 for node_id in node_ids}

        # Build adjacency list
        adj: Dict[str, Set[str]] = defaultdict(set)
        for edge in graph.edges:
            adj[edge.source].add(edge.target)
            adj[edge.target].add(edge.source)

        # Simplified betweenness: count paths through each node
        for source in node_ids:
            # BFS from source
            visited = {source}
            queue = [source]
            predecessors: Dict[str, List[str]] = defaultdict(list)
            distance: Dict[str, int] = {source: 0}

            while queue:
                current = queue.pop(0)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        distance[neighbor] = distance[current] + 1
                        predecessors[neighbor].append(current)
                    elif distance[neighbor] == distance[current] + 1:
                        predecessors[neighbor].append(current)

            # Count paths
            sigma = {node_id: 0 for node_id in node_ids}
            sigma[source] = 1

            for node in sorted(visited, key=lambda x: distance.get(x, 0)):
                for pred in predecessors[node]:
                    sigma[node] += sigma[pred]

            # Calculate dependencies
            delta = {node_id: 0.0 for node_id in node_ids}
            for node in sorted(visited, key=lambda x: -distance.get(x, 0)):
                for pred in predecessors[node]:
                    if sigma[node] > 0:
                        delta[pred] += (sigma[pred] / sigma[node]) * (1 + delta[node])
                if node != source:
                    betweenness[node] += delta[node]

        # Normalize
        max_betweenness = max(betweenness.values()) or 1
        for node_id, score in betweenness.items():
            graph.nodes[node_id].centrality_scores['betweenness'] = score / max_betweenness

    def _calculate_composite_score(self, graph: ConceptGraph, weights: Dict[str, float]) -> None:
        """Calculate weighted composite centrality score."""
        for node in graph.nodes.values():
            score = 0.0
            for metric, weight in weights.items():
                score += weight * node.centrality_scores.get(metric, 0.0)
            node.composite_score = score

    def get_top_concepts(self, graph: ConceptGraph, n: int = 20) -> List[ConceptNode]:
        """
        Get top N concepts by composite score.

        Args:
            graph: ConceptGraph
            n: Number of concepts to return

        Returns:
            List of top ConceptNode objects
        """
        sorted_nodes = sorted(
            graph.nodes.values(),
            key=lambda x: x.composite_score,
            reverse=True
        )
        return sorted_nodes[:n]

    def get_concepts_by_importance(self, graph: ConceptGraph) -> Dict[str, List[str]]:
        """
        Categorize concepts by importance level.

        Returns:
            Dictionary with 'high', 'medium', 'low' importance lists
        """
        sorted_nodes = sorted(
            graph.nodes.values(),
            key=lambda x: x.composite_score,
            reverse=True
        )

        n = len(sorted_nodes)
        high_cutoff = n // 5  # Top 20%
        medium_cutoff = n // 2  # Top 50%

        return {
            'high': [n.term for n in sorted_nodes[:high_cutoff]],
            'medium': [n.term for n in sorted_nodes[high_cutoff:medium_cutoff]],
            'low': [n.term for n in sorted_nodes[medium_cutoff:]]
        }

    def _extract_all_sections(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all sections from nested structure."""
        sections = []

        def extract_recursive(item: Dict[str, Any]) -> None:
            sections.append(item)
            for subsection in item.get('sections', []):
                extract_recursive(subsection)
            for subsection in item.get('subsections', []):
                extract_recursive(subsection)

        for chapter in structure.get('chapters', []):
            extract_recursive(chapter)

        return sections

    def _extract_section_concepts(self, section: Dict[str, Any]) -> Set[str]:
        """Extract concepts from a section."""
        concepts = set()

        # From heading
        heading = section.get('headingText', '')
        concepts.update(self._extract_terms(heading))

        # From content blocks
        for block in section.get('contentBlocks', []):
            text = ""
            if isinstance(block, dict):
                text = block.get('content', '') or block.get('text', '')
                if 'items' in block:
                    text += ' ' + ' '.join(block['items'])
            elif isinstance(block, str):
                text = block

            concepts.update(self._extract_terms(text))

        # From extracted concepts if available
        extracted = section.get('extractedConcepts', {})
        for def_item in extracted.get('definitions', []):
            if isinstance(def_item, dict):
                concepts.add(def_item.get('term', ''))

        for term_item in extracted.get('keyTerms', []):
            if isinstance(term_item, dict):
                concepts.add(term_item.get('term', ''))
            elif isinstance(term_item, str):
                concepts.add(term_item)

        # Filter empty and stopwords
        return {c for c in concepts if c and c.lower() not in self.STOPWORDS and len(c) > 2}

    def _extract_terms(self, text: str) -> Set[str]:
        """Extract meaningful terms from text."""
        import re

        # Split into words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z]+\b', text)

        # Filter stopwords and short words
        terms = set()
        for word in words:
            if word.lower() not in self.STOPWORDS and len(word) > 2:
                terms.add(word)

        # Also extract bigrams (important phrases)
        for i in range(len(words) - 1):
            w1, w2 = words[i].lower(), words[i+1].lower()
            if w1 not in self.STOPWORDS and w2 not in self.STOPWORDS:
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 5:
                    terms.add(bigram)

        return terms

    def _calculate_statistics(self, graph: ConceptGraph) -> Dict[str, Any]:
        """Calculate graph statistics."""
        return {
            "totalConcepts": len(graph.nodes),
            "totalEdges": len(graph.edges),
            "totalSections": graph.total_sections,
            "avgConceptsPerSection": len(graph.nodes) / (graph.total_sections or 1),
            "avgConnectionsPerConcept": sum(
                len(n.co_occurring_concepts) for n in graph.nodes.values()
            ) / (len(graph.nodes) or 1),
            "topConceptsCount": self.config.get('concept_extraction', {}).get('top_concepts_count', 20)
        }


# Convenience function
def build_concept_graph(structure: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build concept graph and return dictionary representation.

    Args:
        structure: Semantic structure dictionary
        config: Optional configuration

    Returns:
        Dictionary with graph data
    """
    builder = ConceptGraphBuilder(config)
    graph = builder.build_graph(structure)
    return graph.to_dict()
