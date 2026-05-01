"""Single source of truth for deterministic-generator template_id prefixes.

These prefixes identify pairs emitted by the four oracle-grounded
deterministic generators (Wave 124-127). Validators that need to
skip-or-treat-differently those pairs import this constant rather
than maintain their own copy.
"""
from __future__ import annotations

from typing import Tuple

DETERMINISTIC_TEMPLATE_PREFIXES: Tuple[str, ...] = (
    "kg_metadata.",
    "violation_detection.",
    "abstention.",
    "schema_translation.",
)

__all__ = ["DETERMINISTIC_TEMPLATE_PREFIXES"]
