"""Qwen-specialist package (Stage 6 of the v2 pipeline).

The single v1 reasoner is replaced by 4 LoRA specialists that share one
Qwen base model and never co-tenant on the 8 GB GPU:

    prose      : long-form rewrite / paraphrase for accessibility tone
    table      : table cell normalization + caption synthesis
    math       : math notation -> MathML (or aria-described LaTeX)
    gap_fill   : per-GapSlot completion (Stage 9b consumer)

Every specialist invocation goes through `AdapterSwap`, which enforces
**serial** entry. Two simultaneous swaps poison the CUDA context on the
3060 — see feedback_qwen_build_serial.md.
"""
