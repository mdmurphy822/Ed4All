"""Qwen chat-template wrapping for training + inference.

Per Phase 1.5 format decision (option B): the stored
``request_payload.payload.prompt`` field is the *raw* SYSTEM/USER prompt
the prompt-builders emit — it is NOT pre-wrapped in Qwen's chat
template. Both the trainer and the inference runtime call
``wrap_for_qwen`` to apply the template at the moment of use. This keeps
the on-disk dataset agnostic of tokenizer/version drift; only the
trained adapter and the runtime have to agree.

The wrap is identical in both phases:

* Training: ``wrap_for_qwen(tok, raw, target=...)`` produces the full
  user+assistant turn (with ``<|im_end|>``-terminated assistant text).
  The completion-only loss mask is then computed by tokenizing the same
  string with ``target=None, add_generation_prompt=True`` to get the
  prompt boundary.

* Inference: ``wrap_for_qwen(tok, raw, add_generation_prompt=True)``
  appends ``<|im_start|>assistant\\n`` so generation starts cleanly.

Why a single function: drift between the two paths is the canonical way
LoRA adapters silently degrade — the assistant prefix at training time
is one token off from inference time, the model never sees its own
opening token, and BLEU collapses without a clear cause.
"""
from __future__ import annotations

from typing import Any


def wrap_for_qwen(
    tok: Any,
    raw_prompt: str,
    target: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Apply Qwen3's chat template to a raw SYSTEM/USER prompt string.

    Parameters
    ----------
    tok
        A HuggingFace tokenizer with a ``chat_template`` attribute set
        (Qwen3-4B-Instruct-2507 ships one by default).
    raw_prompt
        The full ``"SYSTEM: ...\\nUSER: ..."`` string the prompt
        builders emit. We pass it as the user-turn content verbatim;
        the SYSTEM line stays inside the user turn so the trained
        adapter sees the same surface form the dataset was shaped on.
    target
        If provided, the assistant-turn content. Used at training time
        so the chat template emits ``<|im_end|>`` after the target.
    add_generation_prompt
        If True (and ``target is None``), the template appends
        ``<|im_start|>assistant\\n`` so the model can generate.

    Returns
    -------
    str
        The chat-template-wrapped string ready for tokenization.
    """
    messages: list[dict[str, str]] = [
        {"role": "user", "content": raw_prompt},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
        # add_generation_prompt is meaningless once an assistant turn is
        # present; the template already closes the turn with <|im_end|>.
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


__all__ = ["wrap_for_qwen"]
