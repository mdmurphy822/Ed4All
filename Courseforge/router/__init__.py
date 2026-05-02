"""Courseforge two-pass router (Phase 3 §3) — public surface."""

from Courseforge.router.policy import (
    BlockRoutingPolicy,
    load_block_routing_policy,
    match_block_id_glob,
)
from Courseforge.router.router import (
    BlockProviderSpec,
    CourseforgeRouter,
)

__all__ = [
    "BlockProviderSpec",
    "BlockRoutingPolicy",
    "CourseforgeRouter",
    "load_block_routing_policy",
    "match_block_id_glob",
]
