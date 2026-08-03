"""Stage 3a: DistilBERT per-block structural role classifier.

Contract:
    classify_blocks(features: list[FeatureBlock], model=dict) -> list[ClassifiedBlock]

Every block runs through the model when one is provided — no fast-path
bypass. This is the architectural contract with stage 3b (Qwen): Qwen
sees DistilBERT's label and confidence for every block as features in
its prompt, so every block must actually have a DistilBERT pass.

Rules (below) are retained only as a fallback for the `model=None` case,
which is useful for end-to-end dev without loading DistilBERT. In
production, always pass a loaded classifier.

Do NOT split this into multiple models. The role classes are mutually
exclusive; a single multi-class classifier learns them more efficiently
than several binary classifiers (see project_8stage_architecture memory).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import ClassifiedBlock, FeatureBlock


class Role(str, Enum):
    """Structural roles — aligned with PDF/UA StructElem types
    (see docs/ontology.md §3: PDF/UA -> HTML5 role map)."""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    # Definition-list leaves (Axis-1 structural-shape expansion). A <dt> is a
    # term, a <dd> its definition; they group into a Stage-5 definition_list
    # Region. Distinct from the orthogonal Stage-5d definition_region
    # semantic_class (a boxed formal definition), which is a markup axis.
    DEFINITION_TERM = "definition_term"
    DEFINITION_DEF = "definition_def"
    # Table/figure CAPTION text shape (Axis-1). Previously collapsed onto
    # PARAGRAPH; now a first-class structural_role so the head can EMIT what
    # gold needs (a <caption>/<figcaption> is a caption, not body prose).
    CAPTION = "caption"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_HEADER_CELL = "table_header_cell"
    TABLE_DATA_CELL = "table_data_cell"
    TABLE_CAPTION = "table_caption"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    BLOCKQUOTE = "blockquote"
    CODE_BLOCK = "code_block"
    FORM_FIELD = "form_field"
    FORM_LABEL = "form_label"
    REFERENCE = "reference"
    FOOTNOTE = "footnote"
    METADATA = "metadata"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"


@dataclass
class RuleOutcome:
    role: Role
    rule_name: str


# ---------- rule implementations ----------

def _rule_page_header(fb: FeatureBlock) -> RuleOutcome | None:
    """Short line in top 5% of page; common for running heads and page numbers."""
    top_frac = fb.raw.bbox[1] / fb.raw.page_height
    if top_frac < 0.05 and len(fb.raw.text) < 100 and fb.size_bucket in ("sm", "md"):
        return RuleOutcome(Role.PAGE_HEADER, "page_header")
    return None


def _rule_page_footer(fb: FeatureBlock) -> RuleOutcome | None:
    """Short line in bottom 5% of page."""
    bottom_frac = fb.raw.bbox[3] / fb.raw.page_height
    if bottom_frac > 0.95 and len(fb.raw.text) < 100 and fb.size_bucket in ("sm", "md"):
        return RuleOutcome(Role.PAGE_FOOTER, "page_footer")
    return None


def _rule_title(fb: FeatureBlock) -> RuleOutcome | None:
    """Very large + top-of-page + on page 1 = document title."""
    if (fb.size_bucket == "xl"
            and fb.is_top_of_page
            and fb.raw.page == 1
            and len(fb.raw.text) < 250):
        return RuleOutcome(Role.TITLE, "title_xl_top_page1")
    return None


def _rule_heading(fb: FeatureBlock) -> RuleOutcome | None:
    """Non-body size OR caps=title/all after a gap, short-ish."""
    text = fb.raw.text.strip()
    if not text or len(text) > 200:
        return None
    # Sized up from body
    if fb.size_bucket in ("lg", "xl") and fb.gap_above == "lg":
        return RuleOutcome(Role.HEADING, "heading_size_gap")
    # Bold at body size with caps=title/all and gap
    if (fb.raw.is_bold
            and fb.gap_above == "lg"
            and fb.caps in ("title", "all")):
        return RuleOutcome(Role.HEADING, "heading_bold_caps_gap")
    # Numbered section pattern ("1 Introduction", "1.2 Method")
    if fb.gap_above == "lg" and _looks_like_numbered_heading(text):
        return RuleOutcome(Role.HEADING, "heading_numbered")
    return None


def _rule_list_item(fb: FeatureBlock) -> RuleOutcome | None:
    """Leading bullet / number / lettered marker followed by text."""
    text = fb.raw.text.lstrip()
    if not text:
        return None
    # Bullet-like first characters (Tesseract often misreads actual glyphs —
    # be permissive here).
    bullet_chars = "•○∘●◦·–—-*+"
    if text[0] in bullet_chars and len(text) > 2:
        return RuleOutcome(Role.LIST_ITEM, "list_bullet")
    # Numbered item: "1.", "1)", "(1)", "i.", "a)".
    import re
    if re.match(r"^\(?[\dA-Za-z]{1,3}[\.\)]\s+\S", text):
        return RuleOutcome(Role.LIST_ITEM, "list_numbered")
    return None


def _rule_plain_paragraph(fb: FeatureBlock) -> RuleOutcome | None:
    """The catch-everything-normal rule. Body size, no special formatting,
    reasonable length, ends in sentence punctuation."""
    t = fb.raw.text.rstrip()
    if not t:
        return None
    if fb.size_bucket != "md":
        return None
    if fb.caps:  # Title Case / ALL CAPS is usually a heading
        return None
    if t[-1] in ".!?":
        return RuleOutcome(Role.PARAGRAPH, "plain_paragraph_punct")
    # Long body line without terminal punctuation is still almost always a paragraph.
    if len(t) > 120:
        return RuleOutcome(Role.PARAGRAPH, "plain_paragraph_long")
    return None


def _rule_footnote(fb: FeatureBlock) -> RuleOutcome | None:
    """Small text in bottom third of page, often starting with a number."""
    top_frac = fb.raw.bbox[1] / fb.raw.page_height
    if fb.size_bucket == "sm" and top_frac > 0.75:
        return RuleOutcome(Role.FOOTNOTE, "footnote_small_bottom")
    return None


def _looks_like_numbered_heading(text: str) -> bool:
    import re
    return bool(re.match(r"^\d+(\.\d+)*\.?\s+\S", text))


# Ordered rule set. Rules are tried top-to-bottom; first match wins.
RULES = (
    _rule_page_header,
    _rule_page_footer,
    _rule_title,
    _rule_heading,
    _rule_list_item,
    _rule_footnote,
    _rule_plain_paragraph,
)


# ---------- dispatch ----------

def classify_blocks(features: list[FeatureBlock],
                    model=None,
                    *, batch_size: int = 64) -> list[ClassifiedBlock]:
    """Classify each feature block.

    When `model` is provided, every block is labeled by the model in
    batches of `batch_size` — this is the production path, and Qwen
    (stage 3b) depends on every block having a real DistilBERT hint.
    Rules are only consulted as a fallback when no model is loaded.

    Batching amortizes model overhead across large block collections and is
    required for practical document-scale inference.
    """
    if model is not None:
        return _model_classify_batched(model, features, batch_size=batch_size)

    # Model-less fallback: rules + paragraph default.
    out: list[ClassifiedBlock] = []
    for fb in features:
        match = _first_rule_match(fb)
        if match is not None:
            out.append(ClassifiedBlock(
                features=fb, role=match.role.value,
                confidence=1.0, source=f"rule:{match.rule_name}",
            ))
            continue
        out.append(ClassifiedBlock(
            features=fb, role=Role.PARAGRAPH.value,
            confidence=0.5, source="default"))
    return out


def _model_classify_batched(model, features: list[FeatureBlock],
                            *, batch_size: int) -> list[ClassifiedBlock]:
    """Run DistilBERT over all features in fixed-size batches.

    Output equivalence with the per-block path: same tokenizer settings
    (truncation, max_length=256), same softmax-then-argmax, same
    (role, confidence) semantics. The only behavioral difference is
    that padding within a batch changes attention masks — DistilBERT
    handles that correctly through the returned attention_mask, so
    results are numerically identical to single-block inference.
    """
    import torch
    tok = model["tokenizer"]
    mdl = model["model"]
    id2label = model["id2label"]
    n_classes = len(id2label)

    out: list[ClassifiedBlock] = []
    n = len(features)
    for i in range(0, n, batch_size):
        batch = features[i:i + batch_size]
        texts = [_feature_block_to_input_string(fb) for fb in batch]
        inputs = tok(texts, return_tensors="pt", truncation=True,
                     max_length=256, padding=True)
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = mdl(**inputs).logits
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
        preds_list = preds.tolist()
        # Sanity-check: on WSL2 we once saw CUDA return out-of-range indices
        # (huge negative ints, likely driver-level corruption). If that
        # recurs, raise so the caller can skip the pair — better than
        # propagating garbage labels into the training data.
        for pid in preds_list:
            if not (0 <= pid < n_classes):
                raise RuntimeError(
                    f"classifier returned out-of-range pred_id={pid} "
                    f"(expected 0..{n_classes - 1}); likely CUDA corruption")
        for fb, pred_id, prob_row in zip(batch, preds_list, probs):
            out.append(ClassifiedBlock(
                features=fb,
                role=id2label[pred_id],
                confidence=float(prob_row[pred_id].item()),
                source="model",
            ))
    return out


def _first_rule_match(fb: FeatureBlock) -> RuleOutcome | None:
    for rule in RULES:
        outcome = rule(fb)
        if outcome is not None:
            return outcome
    return None


def _model_classify(model, fb: FeatureBlock,
                    context: list[FeatureBlock]) -> tuple[str, float]:
    """Call the stage-3 classifier. `model` must be a dict with keys
    {'tokenizer', 'model', 'id2label'} — typically produced by
    `load_classifier(path)` below."""
    tok = model["tokenizer"]
    mdl = model["model"]
    id2label = model["id2label"]
    text = _feature_block_to_input_string(fb)

    import torch
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits[0]
        probs = logits.softmax(dim=-1)
        pred = int(probs.argmax())
    return id2label[pred], float(probs[pred])


def _feature_block_to_input_string(fb: FeatureBlock) -> str:
    """Serialize a FeatureBlock for the classifier. Delegates to the single
    source of truth in features.feature_block_to_classifier_input so that
    training-data format and inference-data format can never drift."""
    from .features import feature_block_to_classifier_input
    return feature_block_to_classifier_input(fb)


def load_classifier(adapter_path):
    """Convenience: load the trained classifier + tokenizer + label map
    into a dict that `classify_blocks(..., model=...)` accepts."""
    from pathlib import Path
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    p = Path(adapter_path)
    tok = AutoTokenizer.from_pretrained(str(p))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(p))
    mdl.eval()
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    # id2label is serialized into config.json by HuggingFace save_pretrained
    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    return {"tokenizer": tok, "model": mdl, "id2label": id2label}
