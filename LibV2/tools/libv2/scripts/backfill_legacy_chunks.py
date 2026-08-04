"""Compatibility facade for the relocated staged-chunkset backfill command.

New callers should use
``LibV2.tools.libv2.scripts.ops.backfill_legacy_chunks``.
"""

from .ops.backfill_legacy_chunks import *  # noqa: F403
from .ops.backfill_legacy_chunks import main

if __name__ == "__main__":
    raise SystemExit(main())
