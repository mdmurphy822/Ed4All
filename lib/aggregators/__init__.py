"""
Ed4All cross-phase aggregators.

Aggregator modules build operator-facing summary artifacts that span
multiple workflow phases. Unlike per-phase report writers (which live
inside individual phase handlers), aggregators run after the workflow's
phase loop and read previously-emitted reports / phase outputs without
modifying them.

Worker W5 (GPT-feedback follow-up):
    - :class:`courseforge_validation_report.CourseforgeValidationReport`
      walks every per-phase ``report.json`` plus in-memory gate results
      and writes a single top-level
      ``courseforge_validation_report.json`` at the project root.

Worker W2.B (GPT Feedback v2 Wave 2):
    - :class:`trainforge_assessment_quality_report.TrainforgeAssessmentQualityReport`
      aggregates synthesis-pair quality, KG quality, eval-gating,
      leakage, anchoring outputs into a single canonical
      ``<libv2_course>/quality/trainforge_assessment_quality_report.json``.

Worker W3.E (GPT Feedback v2 Wave 3):
    - :class:`coverage_map.CoverageMapAggregator` builds an objective-
      keyed coverage table linking objectives -> chunks -> questions ->
      training_pairs and surfaces orphan objectives / chunks / questions
      to operators via ``<libv2_course>/coverage_map.json``.

Worker W3.G (GPT Feedback v2 Wave 3 — governance G1):
    - :class:`promotion_chain_report.PromotionChainAggregator` is the
      master aggregator. Walks all 9 arrows of the DART -> eval-report
      chain, reads each per-stage report best-effort, and produces
      ``<libv2_course>/courseforge_promotion_chain_report.json`` with
      a top-level ``course_status`` enum + ``chain_hash``.
"""

from .courseforge_validation_report import (  # noqa: F401
    CourseforgeValidationReport,
)
from .coverage_map import (  # noqa: F401
    CoverageMapAggregator,
)
from .edge_consensus import (  # noqa: F401
    EdgeConsensusAggregator,
)
from .promotion_chain_report import (  # noqa: F401
    PromotionChainAggregator,
)
from .trainforge_assessment_quality_report import (  # noqa: F401
    TrainforgeAssessmentQualityReport,
)
