"""Operator-run trainer scripts for the per-Bloom-level DeBERTa heads.

Bloom-ladder initiative WI-13. Nothing here is wired into the pipeline —
no ``@mcp.tool()`` decorator, no ``AGENT_TOOL_MAPPING`` / phase-name
dispatch entry, no ``config/workflows.yaml`` row. The one module,
:mod:`lib.classifiers.training.train_bloom_deberta`, is a standalone
``argparse`` CLI an operator runs by hand (``python -m
lib.classifiers.training.train_bloom_deberta ...``) to produce the
checkpoints :class:`lib.classifiers.bloom_deberta_heads.BloomDebertaHeads`
loads at ``ED4ALL_BLOOM_HEADS_DIR``.
"""
