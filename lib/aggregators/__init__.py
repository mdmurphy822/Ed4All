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
"""

from .courseforge_validation_report import (  # noqa: F401
    CourseforgeValidationReport,
)
