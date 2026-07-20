"""
Ed4All cross-phase aggregators.

Aggregator modules build operator-facing summary artifacts that span
multiple workflow phases. Unlike per-phase report writers (which live
inside individual phase handlers), aggregators run after the workflow's
phase loop and read previously-emitted reports / phase outputs without
modifying them.

- :class:`courseforge_validation_report.CourseforgeValidationReport`
  walks every per-phase ``report.json`` plus in-memory gate results and
  writes a single top-level ``courseforge_validation_report.json`` at the
  project root.

- :class:`trainforge_assessment_quality_report.TrainforgeAssessmentQualityReport`
  aggregates synthesis-pair quality, KG quality, eval-gating, leakage, and
  anchoring outputs into
  ``<libv2_course>/quality/trainforge_assessment_quality_report.json``.

- :class:`coverage_map.CoverageMapAggregator` builds an objective-keyed
  coverage table linking objectives -> chunks -> questions ->
  training_pairs and surfaces orphan objectives / chunks / questions via
  ``<libv2_course>/coverage_map.json``.

- :class:`promotion_chain_report.PromotionChainAggregator` is the master
  aggregator. Walks every arrow of the conversion -> eval-report chain,
  reads each per-stage report best-effort, and produces
  ``<libv2_course>/courseforge_promotion_chain_report.json`` with a
  top-level ``course_status`` enum + ``chain_hash``.

- :class:`concept_coverage.ConceptCoverageAggregator` builds a
  concept-keyed coverage table (the concept-graph analogue of
  coverage_map) surfacing which pedagogical surfaces touch each concept,
  emitted at ``<libv2_course>/concept_coverage.json``. Opt-in via
  ``ED4ALL_CONCEPT_COVERAGE`` (default OFF => no file).

- :class:`intelligence_level.IntelligenceLevelAggregator` scores a built
  course on a deterministic 0-5 capability rubric, emitted at
  ``<libv2_course>/intelligence_level_report.json``. Opt-in via
  ``ED4ALL_INTELLIGENCE_RUBRIC`` (default OFF => no file).
"""

from .concept_coverage import (  # noqa: F401
    ConceptCoverageAggregator,
    resolve_concept_coverage,
)
from .courseforge_validation_report import (  # noqa: F401
    CourseforgeValidationReport,
)
from .coverage_map import (  # noqa: F401
    CoverageMapAggregator,
)
from .intelligence_level import (  # noqa: F401
    IntelligenceLevelAggregator,
    resolve_intelligence_rubric,
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
