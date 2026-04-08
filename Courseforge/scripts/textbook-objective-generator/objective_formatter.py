"""
Objective Formatter Module

Formats learning objectives according to educational standards.
Creates properly structured objective statements with:
- Action verbs from Bloom's taxonomy
- Clear, measurable outcomes
- Consistent formatting

Equal Treatment: Generates objectives for ALL extracted content.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from bloom_taxonomy_mapper import (
    BloomLevel,
    BloomTaxonomyMapper,
    BloomVerb,
    BLOOM_VERBS
)


@dataclass
class LearningObjective:
    """A single learning objective."""
    objective_id: str
    statement: str
    bloom_level: BloomLevel
    bloom_verb: str
    key_concepts: List[str] = field(default_factory=list)
    source_reference: Optional[Dict[str, Any]] = None
    assessment_suggestions: List[str] = field(default_factory=list)
    prerequisite_objectives: List[str] = field(default_factory=list)
    extraction_source: str = "inferred"  # explicit, definition, concept, procedure, etc.
    hierarchy_level: str = "section"  # course, chapter, section, subsection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "objectiveId": self.objective_id,
            "statement": self.statement,
            "bloomLevel": self.bloom_level.value,
            "bloomVerb": self.bloom_verb,
            "keyConcepts": self.key_concepts,
            "sourceReference": self.source_reference,
            "assessmentSuggestions": self.assessment_suggestions,
            "prerequisiteObjectives": self.prerequisite_objectives,
            "extractionSource": self.extraction_source
        }

    def to_markdown(self) -> str:
        """Format as markdown string."""
        return f"- **{self.bloom_verb.capitalize()}** {self.statement[len(self.bloom_verb):].strip()} (Bloom's: {self.bloom_level.display_name})"


class ObjectiveFormatter:
    """
    Formats learning objectives from extracted content.

    Equal Treatment Principle:
    - Generates objectives for ALL definitions
    - Generates objectives for ALL key terms
    - Generates objectives for ALL procedures
    - Generates objectives for ALL sections
    - Does NOT filter based on perceived importance
    """

    # Assessment method suggestions by Bloom's level
    ASSESSMENT_SUGGESTIONS = {
        BloomLevel.REMEMBER: ["quiz", "exam", "matching"],
        BloomLevel.UNDERSTAND: ["discussion", "quiz", "assignment"],
        BloomLevel.APPLY: ["assignment", "demonstration", "case_study"],
        BloomLevel.ANALYZE: ["assignment", "case_study", "project"],
        BloomLevel.EVALUATE: ["discussion", "assignment", "portfolio"],
        BloomLevel.CREATE: ["project", "portfolio", "presentation"],
    }

    def __init__(self):
        self.mapper = BloomTaxonomyMapper()
        self._objective_counter = 0

    def _generate_id(self, prefix: str = "LO") -> str:
        """Generate a unique objective ID."""
        self._objective_counter += 1
        return f"{prefix}_{self._objective_counter}"

    def reset_counter(self) -> None:
        """Reset the objective counter."""
        self._objective_counter = 0

    def format_from_definition(
        self,
        term: str,
        definition: str,
        chapter_id: str,
        section_id: Optional[str] = None
    ) -> LearningObjective:
        """
        Create a learning objective from a term definition.

        Equal Treatment: Every definition gets an objective.

        Args:
            term: The term being defined
            definition: The definition text
            chapter_id: Parent chapter ID
            section_id: Parent section ID (optional)

        Returns:
            LearningObjective
        """
        # Clean the term
        term_clean = term.strip().rstrip('.:')

        # Get appropriate verb
        verb = self.mapper.get_verb(BloomLevel.REMEMBER, "terms and concepts")

        # Create the statement
        statement = f"{verb.verb.capitalize()} {term_clean} and explain its significance"

        return LearningObjective(
            objective_id=self._generate_id(f"{chapter_id}_{section_id or 'def'}"),
            statement=statement,
            bloom_level=BloomLevel.REMEMBER,
            bloom_verb=verb.verb,
            key_concepts=[term_clean],
            source_reference={
                "type": "definition",
                "term": term,
                "chapterId": chapter_id,
                "sectionId": section_id
            },
            assessment_suggestions=self.ASSESSMENT_SUGGESTIONS[BloomLevel.REMEMBER],
            extraction_source="definition",
            hierarchy_level="section" if section_id else "chapter"
        )

    def format_from_key_term(
        self,
        term: str,
        context: str,
        chapter_id: str,
        section_id: Optional[str] = None
    ) -> LearningObjective:
        """
        Create a learning objective from a key term.

        Equal Treatment: Every key term gets an objective.

        Args:
            term: The key term
            context: Surrounding context
            chapter_id: Parent chapter ID
            section_id: Parent section ID (optional)

        Returns:
            LearningObjective
        """
        term_clean = term.strip()

        # Analyze context to determine appropriate level
        level = self.mapper.analyze_text_complexity(context)

        # Get appropriate verb
        verb = self.mapper.get_verb(level, context[:100])

        # Create statement based on level
        if level == BloomLevel.REMEMBER:
            statement = f"{verb.verb.capitalize()} {term_clean}"
        elif level == BloomLevel.UNDERSTAND:
            statement = f"{verb.verb.capitalize()} {term_clean} and its role in the context"
        elif level == BloomLevel.APPLY:
            statement = f"{verb.verb.capitalize()} {term_clean} in practical scenarios"
        else:
            statement = f"{verb.verb.capitalize()} {term_clean} and its implications"

        return LearningObjective(
            objective_id=self._generate_id(f"{chapter_id}_{section_id or 'term'}"),
            statement=statement,
            bloom_level=level,
            bloom_verb=verb.verb,
            key_concepts=[term_clean],
            source_reference={
                "type": "key_term",
                "term": term,
                "context": context[:200],
                "chapterId": chapter_id,
                "sectionId": section_id
            },
            assessment_suggestions=self.ASSESSMENT_SUGGESTIONS[level],
            extraction_source="concept",
            hierarchy_level="section" if section_id else "chapter"
        )

    def format_from_procedure(
        self,
        procedure_name: str,
        steps: List[str],
        chapter_id: str,
        section_id: Optional[str] = None
    ) -> LearningObjective:
        """
        Create a learning objective from a procedure.

        Equal Treatment: Every procedure gets an objective.

        Args:
            procedure_name: Name of the procedure
            steps: List of procedure steps
            chapter_id: Parent chapter ID
            section_id: Parent section ID (optional)

        Returns:
            LearningObjective
        """
        # Procedures map to Apply level
        verb = self.mapper.get_verb(BloomLevel.APPLY, "procedures")

        # Create statement
        if procedure_name and procedure_name != "Procedure":
            statement = f"{verb.verb.capitalize()} the {procedure_name.lower()} procedure correctly"
        else:
            # Infer from first step
            first_step = steps[0] if steps else "process"
            statement = f"{verb.verb.capitalize()} the procedure to {first_step.lower()}"

        # Extract key concepts from steps
        key_concepts = []
        for step in steps[:5]:  # First 5 steps
            # Extract nouns/verbs
            words = re.findall(r'\b[A-Z][a-z]+|[a-z]{4,}\b', step)
            key_concepts.extend(words[:2])

        return LearningObjective(
            objective_id=self._generate_id(f"{chapter_id}_{section_id or 'proc'}"),
            statement=statement,
            bloom_level=BloomLevel.APPLY,
            bloom_verb=verb.verb,
            key_concepts=list(set(key_concepts))[:5],
            source_reference={
                "type": "procedure",
                "name": procedure_name,
                "stepCount": len(steps),
                "chapterId": chapter_id,
                "sectionId": section_id
            },
            assessment_suggestions=self.ASSESSMENT_SUGGESTIONS[BloomLevel.APPLY],
            extraction_source="procedure",
            hierarchy_level="section" if section_id else "chapter"
        )

    def format_from_section(
        self,
        section_title: str,
        content_summary: str,
        chapter_id: str,
        section_id: str
    ) -> LearningObjective:
        """
        Create a learning objective from a section.

        Equal Treatment: Every section gets at least one objective.

        Args:
            section_title: Title of the section
            content_summary: Summary or first paragraph of content
            chapter_id: Parent chapter ID
            section_id: Section ID

        Returns:
            LearningObjective
        """
        # Analyze content to determine level
        level = self.mapper.analyze_text_complexity(content_summary)

        # Get appropriate verb
        verb = self.mapper.get_verb(level, section_title)

        # Clean section title
        title_clean = re.sub(r'^\d+(\.\d+)*\.?\s*', '', section_title)  # Remove numbering

        # Create statement
        statement = f"{verb.verb.capitalize()} {title_clean.lower()}"

        # Extract key concepts from title and content
        key_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', section_title)

        return LearningObjective(
            objective_id=self._generate_id(f"{chapter_id}_{section_id}"),
            statement=statement,
            bloom_level=level,
            bloom_verb=verb.verb,
            key_concepts=key_concepts[:5],
            source_reference={
                "type": "section",
                "heading": section_title,
                "chapterId": chapter_id,
                "sectionId": section_id
            },
            assessment_suggestions=self.ASSESSMENT_SUGGESTIONS[level],
            extraction_source="inferred",
            hierarchy_level="section"
        )

    def format_from_explicit_objective(
        self,
        objective_text: str,
        chapter_id: str,
        section_id: Optional[str] = None
    ) -> LearningObjective:
        """
        Create a learning objective from an explicitly stated objective.

        Args:
            objective_text: The explicitly stated objective
            chapter_id: Parent chapter ID
            section_id: Parent section ID (optional)

        Returns:
            LearningObjective
        """
        # Clean the objective text
        text_clean = objective_text.strip()

        # Try to detect the verb
        words = text_clean.lower().split()
        detected_verb = None
        detected_level = None

        for word in words[:5]:
            clean_word = re.sub(r'[^\w]', '', word)
            for level, verbs in BLOOM_VERBS.items():
                for verb in verbs:
                    if verb.verb == clean_word:
                        detected_verb = verb.verb
                        detected_level = level
                        break
                if detected_verb:
                    break
            if detected_verb:
                break

        if not detected_level:
            detected_level = BloomLevel.UNDERSTAND
            detected_verb = "understand"

        # Extract key concepts
        key_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Za-z]+)*\b', text_clean)

        return LearningObjective(
            objective_id=self._generate_id(f"{chapter_id}_{section_id or 'exp'}"),
            statement=text_clean,
            bloom_level=detected_level,
            bloom_verb=detected_verb,
            key_concepts=key_concepts[:5],
            source_reference={
                "type": "explicit",
                "originalText": objective_text,
                "chapterId": chapter_id,
                "sectionId": section_id
            },
            assessment_suggestions=self.ASSESSMENT_SUGGESTIONS[detected_level],
            extraction_source="explicit",
            hierarchy_level="chapter" if not section_id else "section"
        )

    def format_chapter_objective(
        self,
        chapter_title: str,
        chapter_summary: str,
        chapter_id: str,
        key_topics: List[str]
    ) -> LearningObjective:
        """
        Create a chapter-level objective.

        Args:
            chapter_title: Title of the chapter
            chapter_summary: Summary of chapter content
            chapter_id: Chapter ID
            key_topics: Main topics covered in the chapter

        Returns:
            LearningObjective
        """
        # Chapter objectives are typically higher level
        level = BloomLevel.UNDERSTAND

        # If chapter covers practical skills, use Apply
        if any(word in chapter_title.lower() for word in ['how to', 'implementing', 'building', 'creating']):
            level = BloomLevel.APPLY

        # If chapter covers analysis, use Analyze
        if any(word in chapter_title.lower() for word in ['analyzing', 'comparing', 'examining']):
            level = BloomLevel.ANALYZE

        verb = self.mapper.get_verb(level)

        # Clean title
        title_clean = re.sub(r'^(chapter\s+\d+[:.]\s*)', '', chapter_title, flags=re.IGNORECASE)

        statement = f"{verb.verb.capitalize()} {title_clean.lower()}"

        return LearningObjective(
            objective_id=self._generate_id(f"{chapter_id}"),
            statement=statement,
            bloom_level=level,
            bloom_verb=verb.verb,
            key_concepts=key_topics[:5],
            source_reference={
                "type": "chapter",
                "heading": chapter_title,
                "chapterId": chapter_id
            },
            assessment_suggestions=self.ASSESSMENT_SUGGESTIONS[level],
            extraction_source="inferred",
            hierarchy_level="chapter"
        )

    def format_course_objective(
        self,
        topic: str,
        level: BloomLevel = BloomLevel.ANALYZE
    ) -> LearningObjective:
        """
        Create a course-level objective.

        Course objectives are typically at higher Bloom's levels.

        Args:
            topic: Main topic or skill
            level: Bloom's level (default Analyze)

        Returns:
            LearningObjective
        """
        verb = self.mapper.get_verb(level)

        statement = f"{verb.verb.capitalize()} {topic.lower()}"

        return LearningObjective(
            objective_id=self._generate_id("course"),
            statement=statement,
            bloom_level=level,
            bloom_verb=verb.verb,
            key_concepts=[topic],
            source_reference={"type": "course"},
            assessment_suggestions=self.ASSESSMENT_SUGGESTIONS[level],
            extraction_source="inferred",
            hierarchy_level="course"
        )

    def format_objectives_to_markdown(
        self,
        objectives: List[LearningObjective],
        source_title: str
    ) -> str:
        """
        Format a list of objectives as markdown.

        Args:
            objectives: List of LearningObjective objects
            source_title: Title for the document

        Returns:
            Markdown string
        """
        lines = [
            f"# Learning Objectives: {source_title}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        # Group by hierarchy level
        course_objs = [o for o in objectives if o.hierarchy_level == "course"]
        chapter_objs = [o for o in objectives if o.hierarchy_level == "chapter"]
        section_objs = [o for o in objectives if o.hierarchy_level == "section"]

        if course_objs:
            lines.extend([
                "## Course-Level Objectives",
                "",
            ])
            for i, obj in enumerate(course_objs, 1):
                lines.append(f"{i}. **{obj.bloom_verb.capitalize()}** {obj.statement[len(obj.bloom_verb):].strip()} (Bloom's: {obj.bloom_level.display_name})")
            lines.append("")

        if chapter_objs:
            lines.extend([
                "## Chapter Objectives",
                "",
            ])
            for obj in chapter_objs:
                lines.append(obj.to_markdown())
            lines.append("")

        if section_objs:
            lines.extend([
                "## Section Objectives",
                "",
            ])
            for obj in section_objs:
                lines.append(obj.to_markdown())
            lines.append("")

        # Summary statistics
        level_counts = {}
        for obj in objectives:
            level_name = obj.bloom_level.display_name
            level_counts[level_name] = level_counts.get(level_name, 0) + 1

        lines.extend([
            "---",
            "## Summary by Bloom's Level",
            "",
        ])
        for level in BloomLevel:
            count = level_counts.get(level.display_name, 0)
            lines.append(f"- **{level.display_name}**: {count}")

        lines.append("")
        lines.append(f"**Total Objectives**: {len(objectives)}")

        return "\n".join(lines)
