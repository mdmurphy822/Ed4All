#!/usr/bin/env python3
"""Regenerable builder for a SYNTHETIC 2-corpus calibration fixture.

Everything emitted here is INVENTED — generic ``synthetic-corpus-alpha`` /
``synthetic-corpus-beta`` strings. NO real course slug, path, corpus prose, or
textbook byte is referenced or reproduced (the operator's local courses are
CC BY-NC-SA / internal-only). The fixture exists solely to give
``scripts/harness/calibration_harness.py``'s >=2-DISTINCT-corpora aggregation path a
deterministic, licensing-clean test target so the W8.3 across-corpora aggregate
can be exercised without discovering (or copying) any real corpus.

Shape: two Courseforge-export-shaped directories under ``<dest>``, each carrying

  <export>/04_rewrite/blocks_final.jsonl      (a block citing a DISTINCT dart doc)
  <export>/04_rewrite/02_validation_report/report.json  (per-block gate results)

The two exports cite two DISTINCT invented source documents, so the harness's
content-based corpus identity keys them to two distinct ``src:...`` corpora (the
>=2 precondition). The block-scoped ``udl_coverage`` gate is made to FIRE on a
different fraction in each corpus (alpha 1/N, beta 0/N) so the across-corpora
``max_fire_rate`` != ``min_fire_rate`` and the aggregate is observable.

Run standalone to regenerate into an arbitrary dir::

    python3 scripts/tests/fixtures/build_calibration_corpora.py /tmp/calib_fixture
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Invented (generic) source-document identities — NOT real textbook slugs.
ALPHA_SOURCE_DOC = "synthetic-corpus-alpha-s1to3_accessible"
BETA_SOURCE_DOC = "synthetic-corpus-beta-ch1_accessible"

# The stable corpus keys the harness derives from the docs above (range/marker
# suffixes stripped). Exposed so a test can assert against them without hardcoding
# the normalization rules.
ALPHA_CORPUS_KEY = "src:synthetic-corpus-alpha"
BETA_CORPUS_KEY = "src:synthetic-corpus-beta"

_N_BLOCKS = 10


def _write_export(
    dest: Path, name: str, source_doc: str, *, udl_fail_blocks: int
) -> Path:
    """Write one synthetic export dir and return its path.

    ``udl_fail_blocks`` = number of leading blocks on which the block-scoped
    ``udl_coverage`` gate carries an issue (issue_count>0 => that block fires).
    """
    export = dest / name
    rewrite = export / "04_rewrite"
    rewrite.mkdir(parents=True, exist_ok=True)

    # Blocks file — gives the export a resolvable content corpus identity.
    with (rewrite / "blocks_final.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(_N_BLOCKS):
            block = {
                "block_id": f"blk_{i:03d}",
                "source_ids": [f"dart:{source_doc}#aa{i:02d}bb{i:02d}"],
                "source_references": [],
            }
            fh.write(json.dumps(block) + "\n")

    # Per-block validation report — the harness's richest fire-rate source.
    per_block = []
    for i in range(_N_BLOCKS):
        fired = i < udl_fail_blocks
        per_block.append(
            {
                "block_id": f"blk_{i:03d}",
                "gate_results": [
                    {
                        "gate_id": "udl_coverage",
                        # PHASE-level flag is uniform; the harness fires on the
                        # per-block issue_count, not this smeared flag.
                        "passed": udl_fail_blocks == 0,
                        "issue_count": 1 if fired else 0,
                        "action": "regenerate" if fired else None,
                    }
                ],
            }
        )
    report = {"per_block": per_block, "phase_level_gate_results": []}
    vr = rewrite / "02_validation_report"
    vr.mkdir(parents=True, exist_ok=True)
    (vr / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return export


def build(dest: str | Path) -> Path:
    """Build the synthetic 2-corpus fixture under ``dest`` and return the dest path.

    ``dest`` is created if absent. Deterministic: re-running overwrites the same
    files byte-for-byte. The two exports live directly under ``dest`` so a caller
    can point ``discover_corpora(runs_dir=dest)`` (or ``--corpora dest``) at it.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # alpha: udl_coverage fires on 1/10 blocks (fire_rate 0.1).
    _write_export(dest, "synth-alpha-export", ALPHA_SOURCE_DOC, udl_fail_blocks=1)
    # beta: udl_coverage fires on 0/10 blocks (fire_rate 0.0) — a CLEAN corpus.
    _write_export(dest, "synth-beta-export", BETA_SOURCE_DOC, udl_fail_blocks=0)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dest", help="Directory to build the synthetic fixture into.")
    args = parser.parse_args(argv)
    out = build(args.dest)
    print(f"Wrote synthetic 2-corpus calibration fixture to {out}")
    print(f"  distinct corpus keys: {ALPHA_CORPUS_KEY}, {BETA_CORPUS_KEY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
