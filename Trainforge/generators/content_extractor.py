"""
Content Extractor for Assessment Generation

Extracts question-worthy elements (key terms, factual statements,
relationships, procedures, examples) from retrieved RAG chunks.
Sits between retrieval and question generation to provide structured
content that each question type can consume.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KeyTerm:
    """A defined term extracted from content."""
    term: str
    definition: str
    source_chunk_id: str
    context_sentence: str  # Full sentence containing term + definition


@dataclass
class FactualStatement:
    """A declarative statement suitable for T/F or fill-in-blank."""
    statement: str
    source_chunk_id: str
    key_subject: str  # The main subject/noun that could be blanked or negated


@dataclass
class ConceptRelationship:
    """A relationship between two concepts."""
    concept_a: str
    concept_b: str
    relationship: str  # The nature of the relationship
    full_statement: str  # Complete sentence describing the relationship
    source_chunk_id: str


@dataclass
class Procedure:
    """A sequence of steps or process."""
    title: str
    steps: List[str]
    source_chunk_id: str


@dataclass
class Example:
    """An example, case study, or application from content."""
    description: str
    context: str  # What concept it illustrates
    source_chunk_id: str


def _strip_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


class ContentExtractor:
    """Extract question-worthy elements from retrieved RAG chunks.

    Usage:
        extractor = ContentExtractor()
        chunks = [{"id": "c1", "text": "...", "concept_tags": [...]}]
        terms = extractor.extract_key_terms(chunks)
        statements = extractor.extract_factual_statements(chunks)
    """

    # Patterns that indicate a definition
    DEFINITION_PATTERNS = [
        # "X is defined as Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+(?:is|are)\s+defined\s+as\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        # "X refers to Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+refers?\s+to\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        # "X is the Y" / "X is a Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+(?:is|are)\s+(?:the|a|an)\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        # "X, which is Y,"
        re.compile(
            r"([A-Z][^,]*?),\s+which\s+(?:is|are)\s+(.+?)(?:,|\.|$)",
            re.IGNORECASE,
        ),
        # "X means Y"
        re.compile(
            r"(?:^|(?<=\.\s))([A-Z][^.]*?)\s+means?\s+(.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating causal/comparative relationships
    RELATIONSHIP_PATTERNS = [
        # "X causes Y" / "X leads to Y"
        re.compile(
            r"([^,.]+?)\s+(?:causes?|leads?\s+to|results?\s+in|produces?)\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "X is related to Y"
        re.compile(
            r"([^,.]+?)\s+(?:is|are)\s+(?:related|connected|linked)\s+to\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "Unlike X, Y..."
        re.compile(
            r"Unlike\s+([^,]+?),\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "X differs from Y"
        re.compile(
            r"([^,.]+?)\s+differs?\s+from\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "While X..., Y..."
        re.compile(
            r"While\s+([^,]+?),\s+([^,.]+)",
            re.IGNORECASE,
        ),
        # "X because Y"
        re.compile(
            r"([^,.]+?)\s+because\s+([^,.]+)",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating procedural/step content
    STEP_INDICATORS = re.compile(
        r"(?:^|\n)\s*(?:"
        r"(?:step|stage|phase)\s+\d+"
        r"|(?:first|second|third|fourth|fifth|next|then|finally|lastly)"
        r"|\d+[.)]\s"
        r"|[a-z][.)]\s"
        r")",
        re.IGNORECASE,
    )

    # Patterns for examples
    EXAMPLE_PATTERNS = re.compile(
        r"(?:for\s+example|for\s+instance|such\s+as|e\.g\.|consider\s+the"
        r"|an?\s+example\s+(?:of|is)|to\s+illustrate|in\s+practice)",
        re.IGNORECASE,
    )

    def extract_from_metadata(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured content directly from chunk metadata.

        When chunks carry Courseforge metadata (key_terms, misconceptions,
        bloom_level, content_type_label), this method returns pre-structured
        data without regex parsing.

        Returns dict with keys: key_terms, misconceptions, bloom_levels.
        Empty lists for fields not present in metadata.
        """
        key_terms: List[KeyTerm] = []
        misconceptions: List[Dict[str, str]] = []
        bloom_levels: List[str] = []
        seen_terms: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))

            # Key terms from Courseforge metadata
            for kt in (chunk.get("key_terms") or []):
                if isinstance(kt, dict) and kt.get("term"):
                    term_key = kt["term"].lower()
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        key_terms.append(KeyTerm(
                            term=kt["term"],
                            definition=kt.get("definition", ""),
                            source_chunk_id=chunk_id,
                            context_sentence=f'{kt["term"]}: {kt.get("definition", "")}',
                        ))

            # Misconceptions
            for mc in (chunk.get("misconceptions") or []):
                if isinstance(mc, dict) and mc.get("misconception"):
                    misconceptions.append(mc)

            # Bloom's level
            bl = chunk.get("bloom_level")
            if bl and bl not in bloom_levels:
                bloom_levels.append(bl)

        return {
            "key_terms": key_terms,
            "misconceptions": misconceptions,
            "bloom_levels": bloom_levels,
        }

    def extract_key_terms(
        self, chunks: List[Dict[str, Any]]
    ) -> List[KeyTerm]:
        """Extract defined terms from chunk content.

        Prefers structured key_terms from chunk metadata when available,
        falls back to regex pattern matching.
        """
        # Check if any chunks have structured key_terms metadata
        metadata_result = self.extract_from_metadata(chunks)
        if metadata_result["key_terms"]:
            return metadata_result["key_terms"]

        terms: List[KeyTerm] = []
        seen_terms: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            raw_text = chunk.get("text", "")
            text = _strip_html(raw_text)
            concept_tags = chunk.get("concept_tags", [])

            # Strategy 1: Definition patterns
            for pattern in self.DEFINITION_PATTERNS:
                for match in pattern.finditer(text):
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    if len(term) < 3 or len(definition) < 10:
                        continue
                    term_key = term.lower()
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        terms.append(KeyTerm(
                            term=term,
                            definition=definition,
                            source_chunk_id=chunk_id,
                            context_sentence=match.group(0).strip(),
                        ))

            # Strategy 2: Bold/strong terms in HTML with surrounding context
            bold_matches = re.finditer(
                r"<(?:strong|b|em)>([^<]+)</(?:strong|b|em)>",
                raw_text,
                re.IGNORECASE,
            )
            for bold_match in bold_matches:
                term = bold_match.group(1).strip()
                if len(term) < 2 or term.lower() in seen_terms:
                    continue
                # Get surrounding sentence context
                pos = bold_match.start()
                text_around = _strip_html(raw_text[max(0, pos - 200): pos + 300])
                sentences = _split_sentences(text_around)
                context = ""
                definition = ""
                for sent in sentences:
                    if term.lower() in sent.lower():
                        context = sent
                        # Try to extract the part after the term as definition
                        parts = re.split(
                            re.escape(term), sent, flags=re.IGNORECASE, maxsplit=1
                        )
                        if len(parts) > 1:
                            definition = parts[1].strip().lstrip("—–-:,").strip()
                        break
                if context and len(definition) > 10:
                    seen_terms.add(term.lower())
                    terms.append(KeyTerm(
                        term=term,
                        definition=definition,
                        source_chunk_id=chunk_id,
                        context_sentence=context,
                    ))

            # Strategy 3: concept_tags matched in text
            for tag in concept_tags:
                tag_lower = tag.lower().replace("-", " ").replace("_", " ")
                if tag_lower in seen_terms:
                    continue
                # Find sentence containing the tag
                for sentence in _split_sentences(text):
                    if tag_lower in sentence.lower():
                        seen_terms.add(tag_lower)
                        terms.append(KeyTerm(
                            term=tag.replace("-", " ").replace("_", " ").title(),
                            definition=sentence,
                            source_chunk_id=chunk_id,
                            context_sentence=sentence,
                        ))
                        break

        return terms

    def extract_factual_statements(
        self, chunks: List[Dict[str, Any]]
    ) -> List[FactualStatement]:
        """Extract declarative factual statements from chunk content.

        Filters for sentences with clear subjects and predicates,
        suitable for true/false or fill-in-blank questions.
        """
        statements: List[FactualStatement] = []
        seen: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            text = _strip_html(chunk.get("text", ""))

            for sentence in _split_sentences(text):
                # Skip questions, fragments, and very long sentences
                if sentence.endswith("?") or len(sentence) < 20 or len(sentence) > 300:
                    continue

                # Must be declarative (contains a verb-like structure)
                if not re.search(r"\b(?:is|are|was|were|has|have|can|will|does|do|provides?|involves?|requires?|includes?|consists?|contains?|represents?)\b", sentence, re.IGNORECASE):
                    continue

                # Extract main subject (first noun phrase before first verb)
                subject_match = re.match(
                    r"^((?:The\s+|A\s+|An\s+)?[A-Z][^,;]*?)\s+(?:is|are|was|were|has|have|can|will|does|do|provides?|involves?|requires?)",
                    sentence,
                )
                subject = subject_match.group(1).strip() if subject_match else ""

                if not subject or len(subject) < 3:
                    continue

                norm = sentence.lower().strip()
                if norm not in seen:
                    seen.add(norm)
                    statements.append(FactualStatement(
                        statement=sentence,
                        source_chunk_id=chunk_id,
                        key_subject=subject,
                    ))

        return statements

    def extract_relationships(
        self, chunks: List[Dict[str, Any]]
    ) -> List[ConceptRelationship]:
        """Extract concept relationships (causal, comparative, associative)."""
        relationships: List[ConceptRelationship] = []
        seen: set = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            text = _strip_html(chunk.get("text", ""))

            for sentence in _split_sentences(text):
                for pattern in self.RELATIONSHIP_PATTERNS:
                    match = pattern.search(sentence)
                    if match:
                        a = match.group(1).strip()
                        b = match.group(2).strip()
                        if len(a) < 3 or len(b) < 3:
                            continue
                        # Determine relationship type from the matched pattern
                        rel = match.group(0).strip()
                        key = (a.lower(), b.lower())
                        if key not in seen:
                            seen.add(key)
                            relationships.append(ConceptRelationship(
                                concept_a=a,
                                concept_b=b,
                                relationship=rel,
                                full_statement=sentence,
                                source_chunk_id=chunk_id,
                            ))
                        break  # One relationship per sentence

        return relationships

    def extract_procedures(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Procedure]:
        """Extract step-by-step procedures from content."""
        procedures: List[Procedure] = []

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            raw_text = chunk.get("text", "")

            # Look for ordered lists in HTML
            ol_matches = re.finditer(
                r"<ol[^>]*>(.*?)</ol>", raw_text, re.DOTALL | re.IGNORECASE
            )
            for ol_match in ol_matches:
                items = re.findall(r"<li[^>]*>(.*?)</li>", ol_match.group(1), re.DOTALL)
                if len(items) >= 2:
                    steps = [_strip_html(item) for item in items]
                    # Try to find a heading before the list
                    pre_text = raw_text[: ol_match.start()]
                    heading_match = re.search(
                        r"<h[1-6][^>]*>([^<]+)</h[1-6]>(?:\s*$)",
                        pre_text[-300:],
                        re.IGNORECASE,
                    )
                    title = heading_match.group(1).strip() if heading_match else "Process"
                    procedures.append(Procedure(
                        title=title,
                        steps=steps,
                        source_chunk_id=chunk_id,
                    ))

            # Look for numbered/step text patterns
            text = _strip_html(raw_text)
            step_blocks = re.findall(
                r"(?:Step\s+\d+[:.]\s*)(.*?)(?=Step\s+\d+|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if len(step_blocks) >= 2:
                steps = [s.strip() for s in step_blocks if s.strip()]
                procedures.append(Procedure(
                    title="Process",
                    steps=steps,
                    source_chunk_id=chunk_id,
                ))

        return procedures

    def extract_examples(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Example]:
        """Extract examples, case studies, and applications from content."""
        examples: List[Example] = []

        for chunk in chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            text = _strip_html(chunk.get("text", ""))

            for sentence in _split_sentences(text):
                match = self.EXAMPLE_PATTERNS.search(sentence)
                if match:
                    # Get surrounding context (previous sentence if available)
                    all_sentences = _split_sentences(text)
                    idx = next(
                        (i for i, s in enumerate(all_sentences) if sentence in s),
                        -1,
                    )
                    context = all_sentences[idx - 1] if idx > 0 else ""

                    examples.append(Example(
                        description=sentence,
                        context=context,
                        source_chunk_id=chunk_id,
                    ))

        return examples

    def extract_all(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract all content types at once.

        Returns dict with keys: key_terms, factual_statements,
        relationships, procedures, examples.
        """
        return {
            "key_terms": self.extract_key_terms(chunks),
            "factual_statements": self.extract_factual_statements(chunks),
            "relationships": self.extract_relationships(chunks),
            "procedures": self.extract_procedures(chunks),
            "examples": self.extract_examples(chunks),
        }
