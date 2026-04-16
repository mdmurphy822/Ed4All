"""
Textbook Objective Generator

Main module that generates learning objectives from textbook structure.
Takes the output of semantic-structure-extractor and produces learning objectives
conforming to schemas/learning-objectives/learning_objectives_schema.json.

Equal Treatment Principle:
- Generates objectives for ALL extracted content
- Does NOT filter based on perceived importance
- Treats all definitions, concepts, and sections equally
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Add lib directory to path for semantic structure extractor
# (consolidated from Courseforge and Slideforge into shared lib)

# Add Ed4All lib to path for decision capture
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # scripts/textbook-objective-generator/... → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

from bloom_taxonomy_mapper import BloomLevel, BloomTaxonomyMapper  # noqa: E402
from objective_formatter import LearningObjective, ObjectiveFormatter  # noqa: E402


@dataclass
class ChapterObjectives:
    """Learning objectives for a chapter."""
    chapter_id: str
    chapter_title: str
    chapter_objectives: List[LearningObjective] = field(default_factory=list)
    sections: List['SectionObjectives'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chapterId": self.chapter_id,
            "chapterNumber": int(self.chapter_id.replace("ch", "")) if self.chapter_id.startswith("ch") else 0,
            "chapterTitle": self.chapter_title,
            "chapterObjectives": [o.to_dict() for o in self.chapter_objectives],
            "sections": [s.to_dict() for s in self.sections]
        }


@dataclass
class SectionObjectives:
    """Learning objectives for a section."""
    section_id: str
    section_title: str
    section_objectives: List[LearningObjective] = field(default_factory=list)
    subsections: List['SectionObjectives'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "sectionId": self.section_id,
            "sectionTitle": self.section_title,
            "sectionObjectives": [o.to_dict() for o in self.section_objectives]
        }
        if self.subsections:
            result["subsections"] = [s.to_dict() for s in self.subsections]
        return result


class TextbookObjectiveGenerator:
    """
    Generates learning objectives from textbook structure.

    Equal Treatment: ALL content is processed, nothing is filtered.
    Every definition, key term, procedure, and section gets objectives.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        capture: Optional["DecisionCapture"] = None,
    ):
        """
        Initialize the generator.

        Args:
            config: Optional configuration dictionary
            capture: Optional DecisionCapture for logging generation decisions
        """
        self.formatter = ObjectiveFormatter()
        self.mapper = BloomTaxonomyMapper()
        self.config = config or {}
        self.capture = capture

    def generate(self, textbook_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate learning objectives from textbook structure.

        Args:
            textbook_structure: Output from semantic-structure-extractor

        Returns:
            Dictionary conforming to learning_objectives_schema.json
        """
        self.formatter.reset_counter()

        # Extract document info
        doc_info = textbook_structure.get("documentInfo", {})

        # Generate course-level objectives
        course_objectives = self._generate_course_objectives(textbook_structure)

        # Generate chapter and section objectives
        chapters = self._generate_chapter_objectives(textbook_structure)

        # Compute summary statistics
        all_objectives = self._collect_all_objectives(course_objectives, chapters)
        summary = self._compute_summary(all_objectives, textbook_structure)

        # Log decision capture
        if self.capture:
            self.capture.log_decision(
                decision_type="learning_objective_mapping",
                decision=f"Generated {len(all_objectives)} learning objectives from {len(chapters)} chapters",
                rationale=(
                    f"Course-level: {len(course_objectives)}, "
                    f"Bloom distribution: {summary.get('bloomLevelDistribution', {})}, "
                    f"Equal treatment applied to all content"
                ),
            )

        return {
            "documentMetadata": {
                "sourceType": doc_info.get("sourceFormat", "textbook"),
                "sourcePath": doc_info.get("sourcePath", ""),
                "sourceTitle": doc_info.get("title", "Untitled"),
                "sourceAuthors": doc_info.get("metadata", {}).get("authors", []),
                "generationTimestamp": datetime.now().isoformat(),
                "toolVersion": "1.0.0",
                "extractionMethod": "semantic_structure"
            },
            "courseObjectives": [o.to_dict() for o in course_objectives],
            "chapters": [c.to_dict() for c in chapters],
            "objectivesSummary": summary
        }

    def generate_from_file(self, structure_file: str) -> Dict[str, Any]:
        """
        Generate objectives from a structure JSON file.

        Args:
            structure_file: Path to the textbook structure JSON file

        Returns:
            Learning objectives document
        """
        with open(structure_file, encoding='utf-8') as f:
            structure = json.load(f)
        return self.generate(structure)

    def _generate_course_objectives(
        self,
        structure: Dict[str, Any]
    ) -> List[LearningObjective]:
        """Generate course-level objectives from the overall structure."""
        objectives = []

        # Generate objectives from main topics (chapter titles)
        chapters = structure.get("chapters", [])

        # Determine appropriate levels for course objectives
        # Course objectives should span higher Bloom's levels
        levels = [
            BloomLevel.UNDERSTAND,
            BloomLevel.APPLY,
            BloomLevel.ANALYZE,
            BloomLevel.EVALUATE,
        ]

        for i, chapter in enumerate(chapters[:8]):  # Max 8 course objectives
            chapter_title = chapter.get("headingText", f"Chapter {i+1}")

            # Rotate through levels
            level = levels[i % len(levels)]

            obj = self.formatter.format_course_objective(
                topic=chapter_title,
                level=level
            )
            objectives.append(obj)

        return objectives

    def _generate_chapter_objectives(
        self,
        structure: Dict[str, Any]
    ) -> List[ChapterObjectives]:
        """Generate objectives for all chapters."""
        chapter_objs = []

        chapters = structure.get("chapters", [])
        extracted_concepts = structure.get("extractedConcepts", {})

        for chapter in chapters:
            chapter_id = chapter.get("id", "ch1")
            chapter_title = chapter.get("headingText", "Untitled Chapter")

            ch_obj = ChapterObjectives(
                chapter_id=chapter_id,
                chapter_title=chapter_title
            )

            # Generate from explicit objectives
            explicit = chapter.get("explicitObjectives", [])
            for exp_obj in explicit:
                obj = self.formatter.format_from_explicit_objective(
                    objective_text=exp_obj.get("text", ""),
                    chapter_id=chapter_id
                )
                ch_obj.chapter_objectives.append(obj)

            # Generate chapter-level objective
            content_blocks = chapter.get("contentBlocks", [])
            summary = self._get_content_summary(content_blocks)
            key_topics = self._extract_key_topics(chapter)

            chapter_obj = self.formatter.format_chapter_objective(
                chapter_title=chapter_title,
                chapter_summary=summary,
                chapter_id=chapter_id,
                key_topics=key_topics
            )
            ch_obj.chapter_objectives.append(chapter_obj)

            # Generate from definitions in this chapter
            definitions = [d for d in extracted_concepts.get("definitions", [])
                          if d.get("chapterId") == chapter_id and not d.get("sectionId")]
            for defn in definitions:
                obj = self.formatter.format_from_definition(
                    term=defn.get("term", ""),
                    definition=defn.get("definition", ""),
                    chapter_id=chapter_id
                )
                ch_obj.chapter_objectives.append(obj)

            # Generate from key terms in this chapter
            key_terms = [t for t in extracted_concepts.get("keyTerms", [])
                        if t.get("chapterId") == chapter_id and not t.get("sectionId")]
            for term in key_terms:
                obj = self.formatter.format_from_key_term(
                    term=term.get("term", ""),
                    context=term.get("context", ""),
                    chapter_id=chapter_id
                )
                ch_obj.chapter_objectives.append(obj)

            # Generate from procedures in this chapter
            procedures = [p for p in extracted_concepts.get("procedures", [])
                         if p.get("chapterId") == chapter_id and not p.get("sectionId")]
            for proc in procedures:
                obj = self.formatter.format_from_procedure(
                    procedure_name=proc.get("name", "Procedure"),
                    steps=proc.get("steps", []),
                    chapter_id=chapter_id
                )
                ch_obj.chapter_objectives.append(obj)

            # Process sections
            sections = chapter.get("sections", [])
            for section in sections:
                section_objs = self._generate_section_objectives(
                    section, chapter_id, extracted_concepts
                )
                ch_obj.sections.append(section_objs)

            chapter_objs.append(ch_obj)

        return chapter_objs

    def _generate_section_objectives(
        self,
        section: Dict[str, Any],
        chapter_id: str,
        extracted_concepts: Dict[str, Any]
    ) -> SectionObjectives:
        """Generate objectives for a section."""
        section_id = section.get("id", "s1")
        section_title = section.get("headingText", "Untitled Section")

        sec_obj = SectionObjectives(
            section_id=section_id,
            section_title=section_title
        )

        # Generate section-level objective
        content_blocks = section.get("contentBlocks", [])
        summary = self._get_content_summary(content_blocks)

        section_level_obj = self.formatter.format_from_section(
            section_title=section_title,
            content_summary=summary,
            chapter_id=chapter_id,
            section_id=section_id
        )
        sec_obj.section_objectives.append(section_level_obj)

        # Generate from definitions in this section
        definitions = [d for d in extracted_concepts.get("definitions", [])
                      if d.get("sectionId") == section_id]
        for defn in definitions:
            obj = self.formatter.format_from_definition(
                term=defn.get("term", ""),
                definition=defn.get("definition", ""),
                chapter_id=chapter_id,
                section_id=section_id
            )
            sec_obj.section_objectives.append(obj)

        # Generate from key terms in this section
        key_terms = [t for t in extracted_concepts.get("keyTerms", [])
                    if t.get("sectionId") == section_id]
        for term in key_terms:
            obj = self.formatter.format_from_key_term(
                term=term.get("term", ""),
                context=term.get("context", ""),
                chapter_id=chapter_id,
                section_id=section_id
            )
            sec_obj.section_objectives.append(obj)

        # Generate from procedures in this section
        procedures = [p for p in extracted_concepts.get("procedures", [])
                     if p.get("sectionId") == section_id]
        for proc in procedures:
            obj = self.formatter.format_from_procedure(
                procedure_name=proc.get("name", "Procedure"),
                steps=proc.get("steps", []),
                chapter_id=chapter_id,
                section_id=section_id
            )
            sec_obj.section_objectives.append(obj)

        # Process subsections recursively
        subsections = section.get("subsections", [])
        for subsection in subsections:
            sub_objs = self._generate_section_objectives(
                subsection, chapter_id, extracted_concepts
            )
            sec_obj.subsections.append(sub_objs)

        return sec_obj

    def _get_content_summary(self, content_blocks: List[Dict[str, Any]]) -> str:
        """Get a summary from content blocks."""
        for block in content_blocks:
            if block.get("blockType") in ["paragraph", "summary"]:
                content = block.get("content", "")
                if len(content) > 50:
                    return content[:500]
        return ""

    def _extract_key_topics(self, chapter: Dict[str, Any]) -> List[str]:
        """Extract key topics from a chapter."""
        topics = []

        # Get section titles
        for section in chapter.get("sections", []):
            title = section.get("headingText", "")
            if title:
                topics.append(title)

        return topics[:5]

    def _collect_all_objectives(
        self,
        course_objectives: List[LearningObjective],
        chapters: List[ChapterObjectives]
    ) -> List[LearningObjective]:
        """Collect all objectives into a flat list."""
        all_objs = list(course_objectives)

        for chapter in chapters:
            all_objs.extend(chapter.chapter_objectives)
            for section in chapter.sections:
                all_objs.extend(self._collect_section_objectives(section))

        return all_objs

    def _collect_section_objectives(
        self,
        section: SectionObjectives
    ) -> List[LearningObjective]:
        """Recursively collect objectives from a section."""
        objs = list(section.section_objectives)
        for subsection in section.subsections:
            objs.extend(self._collect_section_objectives(subsection))
        return objs

    def _compute_summary(
        self,
        objectives: List[LearningObjective],
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute summary statistics."""
        # Count by Bloom's level
        by_bloom = {}
        for level in BloomLevel:
            by_bloom[level.value] = sum(
                1 for o in objectives if o.bloom_level == level
            )

        # Count by hierarchy level
        by_hierarchy = {
            "course": sum(1 for o in objectives if o.hierarchy_level == "course"),
            "chapter": sum(1 for o in objectives if o.hierarchy_level == "chapter"),
            "section": sum(1 for o in objectives if o.hierarchy_level == "section"),
            "subsection": sum(1 for o in objectives if o.hierarchy_level == "subsection"),
        }

        # Collect all key concepts
        all_concepts = set()
        for obj in objectives:
            all_concepts.update(obj.key_concepts)

        return {
            "totalObjectives": len(objectives),
            "byBloomLevel": by_bloom,
            "byHierarchyLevel": by_hierarchy,
            "keyConcepts": list(all_concepts)[:100]  # Limit to 100
        }

    def generate_markdown(
        self,
        objectives_doc: Dict[str, Any]
    ) -> str:
        """
        Generate markdown representation of objectives.

        Args:
            objectives_doc: The objectives document

        Returns:
            Markdown string
        """
        lines = []
        metadata = objectives_doc.get("documentMetadata", {})

        lines.extend([
            f"# Learning Objectives: {metadata.get('sourceTitle', 'Untitled')}",
            "",
            f"**Source:** {metadata.get('sourcePath', 'Unknown')}",
            f"**Generated:** {metadata.get('generationTimestamp', 'Unknown')[:10]}",
            "",
        ])

        # Course objectives
        course_objs = objectives_doc.get("courseObjectives", [])
        if course_objs:
            lines.extend([
                "## Course-Level Objectives",
                "",
            ])
            for i, obj in enumerate(course_objs, 1):
                statement = obj.get("statement", "")
                level = obj.get("bloomLevel", "understand").capitalize()
                lines.append(f"{i}. **{obj.get('bloomVerb', '').capitalize()}** {statement[len(obj.get('bloomVerb', '')):].strip()} (Bloom's: {level})")
            lines.append("")

        # Chapter objectives
        chapters = objectives_doc.get("chapters", [])
        for chapter in chapters:
            lines.extend([
                f"## {chapter.get('chapterTitle', 'Chapter')}",
                "",
                "### Chapter Objectives",
            ])

            for obj in chapter.get("chapterObjectives", []):
                statement = obj.get("statement", "")
                verb = obj.get("bloomVerb", "")
                level = obj.get("bloomLevel", "").capitalize()
                lines.append(f"- **{verb.capitalize()}** {statement[len(verb):].strip()} (Bloom's: {level})")

            lines.append("")

            # Sections
            for section in chapter.get("sections", []):
                self._add_section_markdown(section, lines, 3)

        # Summary
        summary = objectives_doc.get("objectivesSummary", {})
        lines.extend([
            "---",
            "## Summary by Bloom's Level",
            "",
        ])

        by_bloom = summary.get("byBloomLevel", {})
        for level in ["remember", "understand", "apply", "analyze", "evaluate", "create"]:
            count = by_bloom.get(level, 0)
            lines.append(f"- **{level.capitalize()}**: {count}")

        lines.append("")
        lines.append(f"**Total Objectives**: {summary.get('totalObjectives', 0)}")

        return "\n".join(lines)

    def _add_section_markdown(
        self,
        section: Dict[str, Any],
        lines: List[str],
        heading_level: int
    ) -> None:
        """Add section markdown recursively."""
        heading = "#" * heading_level
        title = section.get("sectionTitle", "Section")

        lines.extend([
            f"{heading} {title}",
            "",
        ])

        for obj in section.get("sectionObjectives", []):
            statement = obj.get("statement", "")
            verb = obj.get("bloomVerb", "")
            level = obj.get("bloomLevel", "").capitalize()
            lines.append(f"- LO: **{verb.capitalize()}** {statement[len(verb):].strip()} (Bloom's: {level})")

        lines.append("")

        # Subsections
        for subsection in section.get("subsections", []):
            self._add_section_markdown(subsection, lines, min(heading_level + 1, 6))


def generate_objectives(structure_file: str, output_format: str = "json") -> str:
    """
    Convenience function to generate objectives from a structure file.

    Args:
        structure_file: Path to textbook structure JSON
        output_format: "json" or "markdown"

    Returns:
        Formatted output string
    """
    generator = TextbookObjectiveGenerator()
    result = generator.generate_from_file(structure_file)

    if output_format == "markdown":
        return generator.generate_markdown(result)
    else:
        return json.dumps(result, indent=2, ensure_ascii=False)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate learning objectives from textbook structure'
    )
    parser.add_argument(
        'structure_file',
        help='Path to textbook structure JSON file'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'markdown'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output'
    )

    args = parser.parse_args()

    generator = TextbookObjectiveGenerator()
    result = generator.generate_from_file(args.structure_file)

    if args.format == 'markdown':
        output = generator.generate_markdown(result)
    else:
        indent = 2 if args.pretty else None
        output = json.dumps(result, indent=indent, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
