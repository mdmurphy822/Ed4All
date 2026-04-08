"""
Bloom's Taxonomy Mapper Module

Maps content types and patterns to Bloom's taxonomy levels.
Provides action verbs and objective templates for each level.

Equal Treatment Principle: This module does NOT filter or rank importance.
All extracted content is treated equally and mapped to appropriate Bloom's levels.
"""

import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class BloomLevel(Enum):
    """Bloom's taxonomy cognitive levels (revised)."""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()

    @property
    def order(self) -> int:
        """Cognitive complexity order (1=lowest, 6=highest)."""
        order_map = {
            BloomLevel.REMEMBER: 1,
            BloomLevel.UNDERSTAND: 2,
            BloomLevel.APPLY: 3,
            BloomLevel.ANALYZE: 4,
            BloomLevel.EVALUATE: 5,
            BloomLevel.CREATE: 6,
        }
        return order_map[self]


@dataclass
class BloomVerb:
    """An action verb associated with a Bloom's level."""
    verb: str
    level: BloomLevel
    usage_context: str  # brief description of when to use
    example_template: str  # template for generating objectives


# Comprehensive verb mappings with usage contexts
BLOOM_VERBS: Dict[BloomLevel, List[BloomVerb]] = {
    BloomLevel.REMEMBER: [
        BloomVerb("define", BloomLevel.REMEMBER, "terms and concepts", "Define {concept}"),
        BloomVerb("list", BloomLevel.REMEMBER, "items, steps, or components", "List the {components} of {topic}"),
        BloomVerb("recall", BloomLevel.REMEMBER, "facts or information", "Recall {fact} about {topic}"),
        BloomVerb("identify", BloomLevel.REMEMBER, "elements or characteristics", "Identify {element} in {context}"),
        BloomVerb("name", BloomLevel.REMEMBER, "specific items", "Name the {items} associated with {topic}"),
        BloomVerb("state", BloomLevel.REMEMBER, "rules or principles", "State the {rule} for {topic}"),
        BloomVerb("label", BloomLevel.REMEMBER, "diagrams or parts", "Label the {parts} of {diagram}"),
        BloomVerb("match", BloomLevel.REMEMBER, "terms to definitions", "Match {terms} with their {definitions}"),
        BloomVerb("recognize", BloomLevel.REMEMBER, "patterns or examples", "Recognize {pattern} in {context}"),
        BloomVerb("select", BloomLevel.REMEMBER, "correct options", "Select the correct {option} for {question}"),
    ],
    BloomLevel.UNDERSTAND: [
        BloomVerb("explain", BloomLevel.UNDERSTAND, "concepts or processes", "Explain {concept} and its significance"),
        BloomVerb("describe", BloomLevel.UNDERSTAND, "characteristics or features", "Describe the {features} of {topic}"),
        BloomVerb("summarize", BloomLevel.UNDERSTAND, "main points", "Summarize the key points of {topic}"),
        BloomVerb("classify", BloomLevel.UNDERSTAND, "categories", "Classify {items} according to {criteria}"),
        BloomVerb("compare", BloomLevel.UNDERSTAND, "similarities and differences", "Compare {item1} and {item2}"),
        BloomVerb("interpret", BloomLevel.UNDERSTAND, "meaning or data", "Interpret the {data} from {source}"),
        BloomVerb("discuss", BloomLevel.UNDERSTAND, "topics in depth", "Discuss the implications of {topic}"),
        BloomVerb("paraphrase", BloomLevel.UNDERSTAND, "in own words", "Paraphrase {statement} in your own words"),
        BloomVerb("distinguish", BloomLevel.UNDERSTAND, "between concepts", "Distinguish between {concept1} and {concept2}"),
        BloomVerb("illustrate", BloomLevel.UNDERSTAND, "with examples", "Illustrate {concept} with examples"),
    ],
    BloomLevel.APPLY: [
        BloomVerb("apply", BloomLevel.APPLY, "knowledge to situations", "Apply {concept} to {situation}"),
        BloomVerb("demonstrate", BloomLevel.APPLY, "skills or techniques", "Demonstrate {skill} in {context}"),
        BloomVerb("implement", BloomLevel.APPLY, "procedures or solutions", "Implement {procedure} for {goal}"),
        BloomVerb("solve", BloomLevel.APPLY, "problems", "Solve {problem} using {method}"),
        BloomVerb("use", BloomLevel.APPLY, "tools or methods", "Use {tool} to accomplish {task}"),
        BloomVerb("execute", BloomLevel.APPLY, "procedures", "Execute {procedure} correctly"),
        BloomVerb("compute", BloomLevel.APPLY, "calculations", "Compute {value} given {inputs}"),
        BloomVerb("calculate", BloomLevel.APPLY, "numerical results", "Calculate {result} for {scenario}"),
        BloomVerb("practice", BloomLevel.APPLY, "skills", "Practice {skill} in {context}"),
        BloomVerb("perform", BloomLevel.APPLY, "tasks", "Perform {task} according to {standards}"),
    ],
    BloomLevel.ANALYZE: [
        BloomVerb("analyze", BloomLevel.ANALYZE, "components or relationships", "Analyze {topic} to identify {components}"),
        BloomVerb("differentiate", BloomLevel.ANALYZE, "elements", "Differentiate between {element1} and {element2}"),
        BloomVerb("examine", BloomLevel.ANALYZE, "in detail", "Examine {topic} to determine {aspect}"),
        BloomVerb("organize", BloomLevel.ANALYZE, "information", "Organize {information} by {criteria}"),
        BloomVerb("relate", BloomLevel.ANALYZE, "connections", "Relate {concept1} to {concept2}"),
        BloomVerb("categorize", BloomLevel.ANALYZE, "into groups", "Categorize {items} based on {features}"),
        BloomVerb("deconstruct", BloomLevel.ANALYZE, "into parts", "Deconstruct {system} into its components"),
        BloomVerb("investigate", BloomLevel.ANALYZE, "thoroughly", "Investigate {topic} to understand {aspect}"),
        BloomVerb("contrast", BloomLevel.ANALYZE, "differences", "Contrast {item1} with {item2}"),
        BloomVerb("attribute", BloomLevel.ANALYZE, "causes or sources", "Attribute {outcome} to {cause}"),
    ],
    BloomLevel.EVALUATE: [
        BloomVerb("evaluate", BloomLevel.EVALUATE, "based on criteria", "Evaluate {item} against {criteria}"),
        BloomVerb("assess", BloomLevel.EVALUATE, "quality or performance", "Assess the {quality} of {item}"),
        BloomVerb("critique", BloomLevel.EVALUATE, "strengths and weaknesses", "Critique {work} identifying strengths and weaknesses"),
        BloomVerb("justify", BloomLevel.EVALUATE, "decisions", "Justify {decision} based on {evidence}"),
        BloomVerb("judge", BloomLevel.EVALUATE, "merit", "Judge the {merit} of {approach}"),
        BloomVerb("argue", BloomLevel.EVALUATE, "positions", "Argue for or against {position}"),
        BloomVerb("defend", BloomLevel.EVALUATE, "choices", "Defend {choice} with supporting evidence"),
        BloomVerb("support", BloomLevel.EVALUATE, "claims", "Support {claim} with {evidence}"),
        BloomVerb("recommend", BloomLevel.EVALUATE, "best options", "Recommend {option} based on {analysis}"),
        BloomVerb("prioritize", BloomLevel.EVALUATE, "importance", "Prioritize {items} by {criteria}"),
    ],
    BloomLevel.CREATE: [
        BloomVerb("create", BloomLevel.CREATE, "new products", "Create {product} that demonstrates {concept}"),
        BloomVerb("design", BloomLevel.CREATE, "systems or solutions", "Design {solution} for {problem}"),
        BloomVerb("construct", BloomLevel.CREATE, "artifacts", "Construct {artifact} using {method}"),
        BloomVerb("develop", BloomLevel.CREATE, "plans or programs", "Develop {plan} for {goal}"),
        BloomVerb("formulate", BloomLevel.CREATE, "hypotheses or plans", "Formulate {hypothesis} about {topic}"),
        BloomVerb("compose", BloomLevel.CREATE, "written works", "Compose {work} addressing {topic}"),
        BloomVerb("plan", BloomLevel.CREATE, "strategies", "Plan {strategy} to achieve {objective}"),
        BloomVerb("invent", BloomLevel.CREATE, "new solutions", "Invent {solution} for {challenge}"),
        BloomVerb("produce", BloomLevel.CREATE, "outputs", "Produce {output} meeting {specifications}"),
        BloomVerb("generate", BloomLevel.CREATE, "ideas or content", "Generate {ideas} for {purpose}"),
    ],
}


class BloomTaxonomyMapper:
    """
    Maps content to Bloom's taxonomy levels.

    Equal Treatment: All content is mapped without filtering.
    The mapper determines appropriate cognitive levels but does not
    exclude any content based on perceived importance.
    """

    # Content type to default Bloom's level mapping
    CONTENT_TYPE_DEFAULTS: Dict[str, BloomLevel] = {
        # Definitions default to Remember
        "definition": BloomLevel.REMEMBER,
        "term": BloomLevel.REMEMBER,
        "glossary": BloomLevel.REMEMBER,

        # Explanations default to Understand
        "explanation": BloomLevel.UNDERSTAND,
        "description": BloomLevel.UNDERSTAND,
        "concept": BloomLevel.UNDERSTAND,
        "summary": BloomLevel.UNDERSTAND,

        # Procedures default to Apply
        "procedure": BloomLevel.APPLY,
        "steps": BloomLevel.APPLY,
        "how_to": BloomLevel.APPLY,
        "example": BloomLevel.APPLY,

        # Analysis content defaults to Analyze
        "comparison": BloomLevel.ANALYZE,
        "relationship": BloomLevel.ANALYZE,
        "structure": BloomLevel.ANALYZE,

        # Assessment content defaults to Evaluate
        "evaluation": BloomLevel.EVALUATE,
        "criteria": BloomLevel.EVALUATE,
        "judgment": BloomLevel.EVALUATE,

        # Creative content defaults to Create
        "design": BloomLevel.CREATE,
        "solution": BloomLevel.CREATE,
        "synthesis": BloomLevel.CREATE,
    }

    # Patterns that suggest higher-order thinking
    HIGHER_ORDER_PATTERNS = {
        BloomLevel.ANALYZE: [
            r'\b(relationship|structure|component|element|factor|cause|effect)\b',
            r'\b(how|why)\s+(?:does|do|is|are)\b',
            r'\b(compare|contrast|analyze)\b',
        ],
        BloomLevel.EVALUATE: [
            r'\b(best|worst|optimal|effective|efficient)\b',
            r'\b(advantage|disadvantage|pro|con|benefit|drawback)\b',
            r'\b(should|recommend|prefer)\b',
        ],
        BloomLevel.CREATE: [
            r'\b(design|develop|create|build|construct)\b',
            r'\b(plan|strategy|approach)\b',
            r'\b(new|novel|innovative)\b',
        ],
    }

    def __init__(self):
        # Build a flat list of all verbs for quick lookup
        self._verb_to_level: Dict[str, BloomLevel] = {}
        for level, verbs in BLOOM_VERBS.items():
            for verb in verbs:
                self._verb_to_level[verb.verb.lower()] = level

    def map_content_type(self, content_type: str) -> BloomLevel:
        """
        Map a content type to its default Bloom's level.

        Args:
            content_type: Type of content (e.g., "definition", "procedure")

        Returns:
            Appropriate BloomLevel
        """
        return self.CONTENT_TYPE_DEFAULTS.get(
            content_type.lower(),
            BloomLevel.UNDERSTAND  # Default
        )

    def analyze_text_complexity(self, text: str) -> BloomLevel:
        """
        Analyze text to determine suggested Bloom's level.

        Uses pattern matching to detect indicators of cognitive complexity.
        Does NOT filter content - only suggests appropriate level.

        Args:
            text: Text content to analyze

        Returns:
            Suggested BloomLevel
        """
        text_lower = text.lower()

        # Check for higher-order patterns first
        for level in [BloomLevel.CREATE, BloomLevel.EVALUATE, BloomLevel.ANALYZE]:
            patterns = self.HIGHER_ORDER_PATTERNS.get(level, [])
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level

        # Check for explicit verbs
        words = text_lower.split()
        for word in words[:10]:  # Check first 10 words
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self._verb_to_level:
                return self._verb_to_level[clean_word]

        # Default based on text characteristics
        if len(words) < 10:
            return BloomLevel.REMEMBER
        elif len(words) < 30:
            return BloomLevel.UNDERSTAND
        else:
            return BloomLevel.UNDERSTAND

    def get_verbs_for_level(self, level: BloomLevel) -> List[BloomVerb]:
        """Get all action verbs for a Bloom's level."""
        return BLOOM_VERBS.get(level, [])

    def get_verb(self, level: BloomLevel, context: Optional[str] = None) -> BloomVerb:
        """
        Get an appropriate verb for a Bloom's level.

        Args:
            level: The Bloom's taxonomy level
            context: Optional context hint to select best verb

        Returns:
            A BloomVerb object
        """
        verbs = self.get_verbs_for_level(level)

        if not verbs:
            # Fallback
            return BloomVerb("understand", BloomLevel.UNDERSTAND, "general", "Understand {concept}")

        if context:
            # Try to match context
            context_lower = context.lower()
            for verb in verbs:
                if verb.usage_context.lower() in context_lower or context_lower in verb.usage_context.lower():
                    return verb

        # Return a random verb for variety
        return random.choice(verbs)

    def suggest_level_for_definition(self) -> BloomLevel:
        """Suggest Bloom's level for a definition."""
        return BloomLevel.REMEMBER

    def suggest_level_for_concept(self, has_example: bool = False) -> BloomLevel:
        """
        Suggest Bloom's level for a concept.

        Args:
            has_example: Whether the concept includes an example

        Returns:
            BloomLevel
        """
        if has_example:
            return BloomLevel.UNDERSTAND
        return BloomLevel.UNDERSTAND

    def suggest_level_for_procedure(self, step_count: int) -> BloomLevel:
        """
        Suggest Bloom's level for a procedure.

        Args:
            step_count: Number of steps in the procedure

        Returns:
            BloomLevel
        """
        return BloomLevel.APPLY

    def suggest_level_for_review_question(self, question_text: str) -> BloomLevel:
        """
        Suggest Bloom's level for a review question.

        Analyzes the question text to determine cognitive level.

        Args:
            question_text: The review question text

        Returns:
            BloomLevel
        """
        return self.analyze_text_complexity(question_text)

    def get_level_distribution_recommendation(
        self,
        total_objectives: int
    ) -> Dict[BloomLevel, int]:
        """
        Get recommended distribution of objectives across Bloom's levels.

        Based on educational best practices:
        - Remember/Understand: ~30% (foundational)
        - Apply/Analyze: ~50% (core)
        - Evaluate/Create: ~20% (advanced)

        Args:
            total_objectives: Total number of objectives to distribute

        Returns:
            Dictionary mapping levels to recommended counts
        """
        distribution = {
            BloomLevel.REMEMBER: 0.10,
            BloomLevel.UNDERSTAND: 0.20,
            BloomLevel.APPLY: 0.30,
            BloomLevel.ANALYZE: 0.20,
            BloomLevel.EVALUATE: 0.12,
            BloomLevel.CREATE: 0.08,
        }

        result = {}
        remaining = total_objectives
        for level, ratio in distribution.items():
            count = int(total_objectives * ratio)
            result[level] = count
            remaining -= count

        # Distribute remaining to Apply
        result[BloomLevel.APPLY] += remaining

        return result


def get_bloom_verbs(level: str) -> List[str]:
    """
    Convenience function to get verb strings for a level.

    Args:
        level: Bloom's level name (e.g., "remember", "understand")

    Returns:
        List of verb strings
    """
    try:
        bloom_level = BloomLevel(level.lower())
        return [v.verb for v in BLOOM_VERBS.get(bloom_level, [])]
    except ValueError:
        return []


def suggest_bloom_level(content_type: str, text: str = "") -> str:
    """
    Convenience function to suggest a Bloom's level.

    Args:
        content_type: Type of content
        text: Optional text to analyze

    Returns:
        Bloom's level name string
    """
    mapper = BloomTaxonomyMapper()

    if text:
        level = mapper.analyze_text_complexity(text)
    else:
        level = mapper.map_content_type(content_type)

    return level.value
