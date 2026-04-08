"""
Learning Science MCP Tools

MCP tool registration for querying the Learning Science RAG corpus.
Provides research-backed pedagogical guidance during Courseforge content generation.

Tools:
- learning_science_query: General RAG queries for pedagogical context
- get_pedagogical_strategy: Strategy recommendations for specific objectives
- validate_with_research: Research-backed content validation
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Import advisor module
try:
    from Courseforge.lib.learning_science_advisor import (
        CONTEXT_KEYWORDS,
        LearningScienceAdvisor,  # noqa: F401
        PedagogicalContext,  # noqa: F401
        get_advisor,
    )
    HAS_ADVISOR = True
except ImportError as e:
    logger.warning(f"Could not import learning science advisor: {e}")
    HAS_ADVISOR = False


def _format_context_response(context) -> dict:
    """Format PedagogicalContext for MCP response."""
    if hasattr(context, 'to_dict'):
        result = context.to_dict()
        result["prompt_injection"] = context.to_prompt_injection()
        return result
    return {"error": "Invalid context object"}


def register_learning_science_tools(mcp):
    """Register learning science tools with the MCP server."""

    @mcp.tool()
    async def learning_science_query(
        query: str,
        context_type: str = "general",
        limit: int = 10,
    ) -> str:
        """
        Query the Learning Science corpus for pedagogical research context.

        Use this tool when generating educational content to get research-backed
        principles, strategies, and citations relevant to your topic.

        Args:
            query: The instructional topic or question (e.g., "teaching database normalization")
            context_type: Type of pedagogical context. Options:
                - general: Broad learning theory context
                - cognitive_load: Sweller, worked examples, split attention
                - multimedia: Mayer principles, dual coding, modality
                - retrieval_practice: Testing effect, spacing, interleaving
                - motivation: Self-determination, growth mindset
                - metacognition: Self-regulation, monitoring, planning
                - transfer: Near/far transfer, analogical reasoning
                - feedback: Formative assessment, corrective feedback
                - emotion: Affective learning, engagement, flow
                - social: Collaborative learning, Vygotsky, ZPD
                - expertise: Deliberate practice, chunking
                - schema: Prior knowledge, conceptual change
                - individual_differences: Learning preferences, aptitude
                - technology: E-learning, adaptive systems
            limit: Maximum results (1-25, default 10)

        Returns:
            JSON with principles, strategies, citations, and prompt_injection text
        """
        if not HAS_ADVISOR:
            return json.dumps({
                "error": "Learning Science advisor not available",
                "details": "Could not import Courseforge.lib.learning_science_advisor"
            })

        try:
            advisor = get_advisor()
            context = advisor.query(
                topic=query,
                context_type=context_type,
                limit=min(max(1, limit), 25),
            )
            return json.dumps(_format_context_response(context), indent=2)
        except Exception as e:
            logger.error(f"Learning science query failed: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_pedagogical_strategy(
        topic: str,
        objective: str,
        bloom_level: str = "apply",
    ) -> str:
        """
        Get specific pedagogical strategies for a learning objective.

        Use this when designing activities or assessments aligned to specific
        learning outcomes. The tool maps Bloom's taxonomy levels to appropriate
        research-backed strategies.

        Args:
            topic: The content topic (e.g., "SQL queries")
            objective: The learning objective (e.g., "Write basic SELECT statements")
            bloom_level: Bloom's taxonomy level:
                - remember: Focus on retrieval practice
                - understand: Focus on schema building
                - apply: Focus on cognitive load management
                - analyze: Focus on metacognition
                - evaluate: Focus on feedback
                - create: Focus on transfer

        Returns:
            JSON with strategies tailored to the Bloom's level and objective
        """
        if not HAS_ADVISOR:
            return json.dumps({
                "error": "Learning Science advisor not available"
            })

        try:
            advisor = get_advisor()
            context = advisor.get_pedagogical_strategy(
                topic=topic,
                objective=objective,
                bloom_level=bloom_level,
            )
            result = _format_context_response(context)
            result["bloom_level"] = bloom_level
            result["objective"] = objective
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Pedagogical strategy lookup failed: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def validate_with_research(
        content_summary: str,
        aspects: str = "cognitive_load,engagement",
    ) -> str:
        """
        Validate content design against learning science research.

        Use this after drafting content to check alignment with research-backed
        best practices across multiple pedagogical dimensions.

        Args:
            content_summary: Brief summary of the content being validated
                (e.g., "Interactive tutorial on loops with code examples and quiz")
            aspects: Comma-separated list of aspects to validate:
                - cognitive_load: Check for extraneous load, worked examples
                - multimedia: Check for Mayer's principles
                - retrieval_practice: Check for testing opportunities
                - motivation: Check for autonomy, competence, relatedness
                - metacognition: Check for self-regulation supports
                - feedback: Check for formative assessment
                - engagement: Check for interest, curiosity supports

        Returns:
            JSON with validation results per aspect, including relevant
            principles and recommendations
        """
        if not HAS_ADVISOR:
            return json.dumps({
                "error": "Learning Science advisor not available"
            })

        try:
            advisor = get_advisor()
            aspect_list = [a.strip() for a in aspects.split(",") if a.strip()]

            # Validate aspect names
            valid_aspects = list(CONTEXT_KEYWORDS.keys())
            for aspect in aspect_list:
                if aspect not in valid_aspects and aspect != "engagement":
                    return json.dumps({
                        "error": f"Invalid aspect: {aspect}",
                        "valid_aspects": valid_aspects
                    })

            # Map engagement to emotion (common alias)
            aspect_list = ["emotion" if a == "engagement" else a for a in aspect_list]

            validation = advisor.validate_with_research(content_summary, aspect_list)
            return json.dumps(validation, indent=2)
        except Exception as e:
            logger.error(f"Research validation failed: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def list_learning_science_domains() -> str:
        """
        List available learning science domains and their keywords.

        Use this to understand what pedagogical research areas are available
        for querying.

        Returns:
            JSON mapping domain names to their associated research keywords
        """
        if not HAS_ADVISOR:
            return json.dumps({
                "error": "Learning Science advisor not available"
            })

        return json.dumps({
            "domains": CONTEXT_KEYWORDS,
            "usage": "Use domain name as context_type in learning_science_query",
            "corpus_info": {
                "location": "LibV2/courses/learning-science-for-instructional-designers/",
                "chunks": 1144,
                "sections": 16,
            }
        }, indent=2)

    @mcp.tool()
    async def invalidate_learning_science_cache() -> str:
        """
        Invalidate the learning science query cache.

        Call this after updating the Learning Science corpus to ensure
        fresh results are retrieved.

        Returns:
            Number of cache entries removed
        """
        if not HAS_ADVISOR:
            return json.dumps({
                "error": "Learning Science advisor not available"
            })

        try:
            advisor = get_advisor()
            count = advisor.invalidate_cache()
            return json.dumps({
                "success": True,
                "entries_removed": count,
            })
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return json.dumps({"error": str(e)})

    logger.info("Learning Science tools registered: learning_science_query, get_pedagogical_strategy, validate_with_research, list_learning_science_domains, invalidate_learning_science_cache")
