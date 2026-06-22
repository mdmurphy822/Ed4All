"""Stage 3b: Qwen 3 as the single structural decision-maker.

Every block from stage 3a comes with a DistilBERT hint + confidence. Qwen
sees those hints as FEATURES alongside the typography and layout flags,
and emits the final structural label for every block.

Key architectural constraints (see project memory 2026-04-21):
  - Qwen runs on every block. No fast-path bypass.
  - DistilBERT's label is a hint, not a candidate to confirm/correct.
  - Chunking is by pages (3-5 pages + 1-page overlap), not block count.
  - Overlap pages appear in two chunks; the chunk-centered label wins;
    divergent overlap labels are logged for diagnostic review.

Model: Qwen/Qwen3-4B-Instruct-2507 (Apache 2.0). Loaded in 4-bit.
VRAM footprint: ~3.5 GB. Use sequentially with DistilBERT — release
DistilBERT's handle before loading Qwen to avoid co-tenancy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from .reason_chunking import (
    DEFAULT_OVERLAP,
    DEFAULT_PAGES_PER_CHUNK,
    PageChunk,
    build_page_chunks,
    reconcile_overlap,
)
from .reason_schema import (
    LABEL_TO_CODE,
    chunk_header,
    page_separator,
    parse_qwen_output,
    serialize_block,
)
from .types import ClassifiedBlock


_log = logging.getLogger("dart_semantic.reason")

QWEN_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
VALID_ROLES = set(LABEL_TO_CODE.keys())

# Canonical vocabulary block for the system prompt — formatted to match the
# short codes in reason_schema.LABEL_TO_CODE.
_ROLE_VOCAB = ", ".join(LABEL_TO_CODE.keys())

SYSTEM_PROMPT = f"""You are a document structure labeler. You see one chunk of a PDF at a time. Each block in the chunk has typography flags, layout signals, and a hint from a small pre-classifier. Your job is to emit the final structural label for EVERY block in the chunk.

Block line format:
  [flags] text

Flag tokens (omitted when default):
  s=<xl|lg|sm>   size bucket (md is default)
  b              bold
  c              centered horizontally
  t              top-of-page (top 15%)
  g=lg           large gap above (likely section break)
  cap=<a|t>      capitalization: all-caps / Title Case
  tbl            block is inside a detected table
  hdr            block is in a table header row (only with tbl)
  wid            block is inside a form widget
  w=<t|b|s|sg>   widget kind: text / button / select / signature
  p=<...>        extractor provenance when non-default
  h=<code>       hint from the pre-classifier (see below)
  hc=.XX         hint confidence — omitted when >= .90

Hint codes:
  ti=title, hd=heading, pa=paragraph, ls=list, li=list_item,
  tb=table, tr=table_row, thc=table_header_cell, tdc=table_data_cell,
  tbc=table_caption, fg=figure, fgc=figure_caption, bq=blockquote,
  cb=code_block, ff=form_field, fl=form_label, rf=reference, fn=footnote,
  df=definition, mt=metadata, ph=page_header, pf=page_footer

The hint is one signal among many. Agree with it when the other flags support it; override it when they don't.

Valid roles you may emit:
  {_ROLE_VOCAB}

Output format: a single JSON array, one object per block in chunk order:
  [{{"id": 0, "role": "heading", "depth": 2}}, {{"id": 1, "role": "paragraph"}}, ...]

- `id` is the zero-based block index within the chunk.
- `role` MUST be one of the valid roles above.
- `depth` is optional and only meaningful for heading (1-6) and list_item (0+).

Label every block. Output ONLY the JSON array — no prose, no markdown fences."""


class ReasoningError(Exception):
    pass


# ---------- model lifecycle ----------

def load_reasoner(base_model: str = QWEN_DEFAULT_MODEL,
                  *,
                  adapter_path: str | Path | None = None,
                  quantize_4bit: bool = True):
    """Load Qwen 3 for stage 3b. Returns {tokenizer, model}.

    `adapter_path` — when given, a LoRA adapter from train_reasoner.py is
    loaded on top of `base_model`. The adapter's saved config usually
    records its base-model id, but we still accept `base_model`
    explicitly so call sites can override (e.g., to test against a
    different base).

    `quantize_4bit` — keep on for the 3060 8GB. Turn off only for
    post-training adapter merges that need fp16/bf16 weights.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {"trust_remote_code": True, "device_map": {"": 0}}
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    # Use the adapter's tokenizer when available — it carries any special
    # tokens/chat-template tweaks saved at training time.
    tok_src = str(adapter_path) if adapter_path else base_model
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return {"tokenizer": tok, "model": model}


def unload_reasoner(reasoner) -> None:
    """Explicitly release Qwen's GPU memory so DistilBERT can load again."""
    import gc
    import torch
    if reasoner is None:
        return
    del reasoner["model"]
    del reasoner["tokenizer"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- core reasoning pass ----------

def reason_over_document(
    classified: list[ClassifiedBlock],
    reasoner: dict,
    *,
    pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
    overlap: int = DEFAULT_OVERLAP,
    max_blocks_per_chunk: int | None = None,
    max_new_tokens: int = 2048,
    doc_type: str | None = None,
    doc_conf: float | None = None,
) -> list[ClassifiedBlock]:
    """Run Qwen on every chunk. Every block gets a Qwen-assigned label.

    Blocks on overlap pages appear in two chunks; reconciliation picks the
    chunk where the block is more central. Divergent labels on overlap
    are logged at INFO level for diagnostic review.

    When Qwen's output for a chunk is unparseable or omits a block, that
    block falls back to its DistilBERT hint (the incoming
    `classified[i].role`) — a soft-fail that keeps the pipeline moving.
    """
    if not classified:
        return []

    pages = [cb.features.raw.page for cb in classified]
    chunk_kwargs = {"pages_per_chunk": pages_per_chunk, "overlap": overlap}
    if max_blocks_per_chunk is not None:
        chunk_kwargs["max_blocks_per_chunk"] = max_blocks_per_chunk
    chunks = build_page_chunks(pages, **chunk_kwargs)
    if not chunks:
        return list(classified)

    labels_by_chunk: list[dict[int, str]] = []
    for chunk in chunks:
        user_msg = _build_chunk_prompt(
            classified, chunk, chunk_count=len(chunks),
            doc_type=doc_type, doc_conf=doc_conf,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = _generate(reasoner, messages, max_new_tokens)
        chunk_labels = _parse_chunk_output(raw, chunk)
        labels_by_chunk.append(chunk_labels)

    final_labels, conflicts = reconcile_overlap(chunks, labels_by_chunk)
    if conflicts:
        _log.info("reason: %d overlap-label conflicts", len(conflicts))
        for c in conflicts[:5]:  # log up to five as samples
            _log.info("  conflict block=%d %s -> winner chunk=%d role=%s",
                      c["block_id"], c["chunk_labels"],
                      c["winner_chunk"], c["winner_role"])

    out: list[ClassifiedBlock] = []
    for i, cb in enumerate(classified):
        if i in final_labels:
            # Tag whether Qwen agreed or overrode DistilBERT — stage 8
            # escalation uses the override rate as a diagnostic signal
            # now that per-block confidence is always 1.0 post-Qwen.
            agreed = final_labels[i] == cb.role
            src = "qwen3:agree" if agreed else "qwen3:override"
            out.append(ClassifiedBlock(
                features=cb.features,
                role=final_labels[i],
                confidence=1.0,
                source=src,
            ))
        else:
            out.append(ClassifiedBlock(
                features=cb.features,
                role=cb.role,
                confidence=cb.confidence,
                source=f"{cb.source}:qwen_miss",
            ))
    return out


# ---------- prompt assembly ----------

def _build_chunk_prompt(
    classified: list[ClassifiedBlock],
    chunk: PageChunk,
    *,
    chunk_count: int,
    doc_type: str | None,
    doc_conf: float | None,
) -> str:
    """Compose the user-message body for one chunk: header, page separators,
    and one line per block in the chunk.
    """
    lines: list[str] = [chunk_header(
        chunk_index=chunk.index,
        chunk_count=chunk_count,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        doc_type=doc_type,
        doc_conf=doc_conf,
    )]

    current_page = -1
    for local_id, global_bid in enumerate(chunk.block_ids):
        cb = classified[global_bid]
        fb = cb.features
        if fb.raw.page != current_page:
            lines.append(page_separator(fb.raw.page))
            current_page = fb.raw.page
        sb = serialize_block(
            block_id=local_id,
            text=fb.raw.text,
            size_bucket=fb.size_bucket,
            is_bold=bool(fb.raw.is_bold),
            is_centered=fb.is_centered,
            is_top_of_page=fb.is_top_of_page,
            gap_above=fb.gap_above,
            caps=fb.caps,
            in_table=fb.in_table,
            in_header_row=fb.in_header_row,
            in_widget=fb.in_widget,
            widget_kind=fb.widget_kind,
            provenance=fb.provenance or fb.raw.source,
            hint_label=cb.role,
            hint_confidence=cb.confidence,
        )
        lines.append(sb.to_line())
    return "\n".join(lines)


# ---------- output parsing ----------

def _parse_chunk_output(raw: str, chunk: PageChunk) -> dict[int, str]:
    """Parse Qwen's JSON array, return {global_block_id: role}.

    Entries with unknown roles or out-of-range ids are dropped. Total
    parse failure returns {} so the caller falls back to DistilBERT hints.
    """
    try:
        entries = parse_qwen_output(raw)
    except (ValueError, Exception) as exc:  # json errors, shape errors
        _log.warning("reason: chunk %d parse failed: %s", chunk.index, exc)
        return {}

    out: dict[int, str] = {}
    max_id = len(chunk.block_ids) - 1
    for entry in entries:
        local_id = entry.get("id")
        role = entry.get("role")
        if not isinstance(local_id, int) or local_id < 0 or local_id > max_id:
            continue
        if role not in VALID_ROLES:
            continue
        global_bid = chunk.block_ids[local_id]
        out[global_bid] = role
    return out


# ---------- model call ----------

def _generate(reasoner: dict, messages: list[dict],
              max_new_tokens: int) -> str:
    import torch
    tok = reasoner["tokenizer"]
    model = reasoner["model"]

    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=8192).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    generated = out[0, inputs.input_ids.shape[1]:]
    return tok.decode(generated, skip_special_tokens=True)
