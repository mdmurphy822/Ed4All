"""Figure subtype router via SigLIP-2 (Apache-2.0, ungated).

Phase P1 of the figure-router plan. A frozen SigLIP-2 image/text encoder used
two ways:

  1. **Zero-shot subtype classification** — tag each figure with one of a small
     accessibility-relevant taxonomy (``chart``, ``diagram``,
     ``photo_micrograph``, ``map``, ``equation_or_table_image``, ``other``) by
     comparing its image embedding against averaged text-prompt embeddings per
     class. This is the routing signal the figure-alt path will use to pick a
     captioning strategy (e.g. a future DePlot chart-tier) and to gate
     decorative-vs-content.
  2. **Image/caption consistency primitive** — :func:`similarity` exposes the
     SigLIP sigmoid agreement score between a figure image and a candidate
     caption, the building block for a future no-hallucination consistency gate.

Design (mirrors ``figure_captioner.py``):
  * frozen encoder, **CPU fp32** at runtime (the 8GB GPU is reserved for the
    Qwen specialists / SmolVLM2 captioner);
  * heavy imports (torch / transformers / PIL) are LAZY — inside functions, so
    importing this module never pulls in torch (asserted by a test);
  * a single cached loader (``lru_cache(maxsize=1)``);
  * NO silent fallbacks (``feedback_no_silent_fallbacks``): a model-load or
    inference failure raises :class:`FigureRouterError`, never a benign-looking
    empty embedding or a guessed label.

Encoder: ``google/siglip2-base-patch16-224`` — Apache-2.0, ungated, the
fixed-resolution (non-NaFlex) base SigLIP-2 checkpoint, so the standard
``AutoProcessor`` handles both the 224px image resize and the text tokenizer.

SigLIP text cap: SigLIP/SigLIP-2 tokenize text to a FIXED 64-token sequence
(``max_length=64``, padding ``"max_length"``). Captions longer than 64 tokens
are truncated by the processor — :func:`embed_texts` documents and relies on
this; the short zero-shot prompts and class names are all well under the cap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np

# Apache-2.0, ungated, fixed-resolution base checkpoint.
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"

# SigLIP / SigLIP-2 text encoder is trained at a fixed 64-token sequence length.
SIGLIP_TEXT_MAX_TOKENS = 64


class FigureRouterError(RuntimeError):
    """SigLIP-2 figure router load or inference failed."""


class FigureRouterHeadMissing(FigureRouterError):
    """The trained 5-class head is not on disk.

    :func:`classify_subtype` is the TRAINED-head contract; falling back to
    the zero-shot path here would silently ship the 0.59-accuracy classifier
    the P1 spot-check rejected (``feedback_no_silent_fallbacks``). Callers
    that want zero-shot must call :func:`zero_shot_subtype` explicitly.
    """


# ---------------------------------------------------------------------------
# Taxonomy + zero-shot prompt templates (approved P1 taxonomy)
# ---------------------------------------------------------------------------
# Six accessibility-relevant subtypes. Order is canonical (used as the class
# axis of the averaged prompt-embedding matrix and for argmax label lookup).
SUBTYPES: tuple[str, ...] = (
    "chart",
    "diagram",
    "photo_micrograph",
    "map",
    "equation_or_table_image",
    "other",
)

# 2-3 natural-language prompt templates per subtype. Embedded and AVERAGED
# per class (then re-normalized) to make the class prototype robust to any one
# phrasing — the standard CLIP/SigLIP zero-shot prompt-ensembling trick.
SUBTYPE_PROMPTS: dict[str, tuple[str, ...]] = {
    "chart": (
        "a chart with plotted data",
        "a graph showing a line or bar plot",
        "a scientific figure plotting numeric data on axes",
    ),
    "diagram": (
        "a scientific diagram",
        "a schematic illustration with labeled parts",
        "a flowchart or block diagram",
    ),
    "photo_micrograph": (
        "a photograph",
        "a microscope image of cells or tissue",
        "a real-world photo or micrograph",
    ),
    "map": (
        "a map of a geographic region",
        "a map showing countries or terrain",
    ),
    "equation_or_table_image": (
        "an image of a mathematical equation",
        "a screenshot of a data table",
        "a rendered formula or tabular grid of numbers",
    ),
    "other": (
        "a miscellaneous figure",
        "an abstract or unclear image",
    ),
}


# ---------------------------------------------------------------------------
# Binary chart-vs-rest gate (P2 spot-check verdict: ship binary, defer 5-class)
# ---------------------------------------------------------------------------
# The P2 visual spot-check (data/eval_reports/figure_router_spotcheck_v1.json)
# found the raw zero-shot 'chart' label is precision-strong (~0.90) but
# recall-weak (~0.69): scatter/t-SNE embedding plots drift to 'map' and
# fan-charts / confidence-band plots drift to 'other'/'diagram'. The cheap,
# deterministic mitigation it recommends is folding those phrasings into the
# CHART side of the prompt ensemble — done here. :data:`SUBTYPE_PROMPTS` (the
# 5-class path) is intentionally left untouched for downstream consumers.
CHART_BINARY_EXTRA_PROMPTS: tuple[str, ...] = (
    "a scatter plot of data points",
    "a 2D embedding visualization of scattered colored points with cluster labels",
    "a t-SNE or PCA projection plot of data points",
    "a line chart with shaded confidence bands",
)

# Full chart-side prompt set for the binary gate: the approved 5-class chart
# templates plus the recall-leak variants above.
CHART_BINARY_PROMPTS: tuple[str, ...] = SUBTYPE_PROMPTS["chart"] + CHART_BINARY_EXTRA_PROMPTS

# The non-chart side of the binary gate competes with ALL five non-chart
# class prototypes (taxonomy order preserved).
NONCHART_SUBTYPES: tuple[str, ...] = tuple(s for s in SUBTYPES if s != "chart")

# Precision-first decision margin, in raw cosine units: the chart prototype's
# similarity must beat the BEST non-chart prototype by at least this much
# before the gate fires. Rationale: a false 'chart' positive sends a
# non-chart figure to the DePlot chart tier, producing a confidently WRONG
# data-reading alt-text — worse for WCAG output quality than the generic
# caption a false negative gets. So ties and narrow wins must resolve to
# is_chart=False, which a strictly positive margin guarantees. SigLIP cosine
# spreads are flat (the 5-class softmax uses temperature 100, i.e. ~0.01
# cosine ≈ 1 logit); 0.005 is half a logit at that scale. Validated offline
# against the spot-check's 60 visual labels (embeddings from the cached
# figure_embeddings npz, prototypes from the real text tower): at 0.005 the
# gate scores precision 1.000 / recall 0.769 vs the raw-argmax baseline's
# 0.90 / 0.69, and the hardest observed non-chart row (a physics
# vector diagram, raw margin -0.0038 — the baseline's one false positive)
# sits ~2x the margin below the threshold. Directional, not calibrated (n=60).
CHART_BINARY_MARGIN: float = 0.005


# ---------------------------------------------------------------------------
# Model loader (cached, lazy, CPU fp32, frozen)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_router() -> tuple[Any, Any]:
    """Load SigLIP-2 once per process. Returns ``(model, processor)``.

    CPU fp32, eval mode, grads disabled (frozen encoder). Lazy heavy imports.

    Raises
    ------
    FigureRouterError
        If the checkpoint cannot be loaded (download / config / weights).
    """
    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
        model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID, dtype=torch.float32)
        model = model.to("cpu")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    except Exception as exc:  # noqa: BLE001 — surface, don't swallow
        raise FigureRouterError(
            f"failed to load SigLIP-2 router ({SIGLIP2_MODEL_ID}): {type(exc).__name__}: {exc}"
        ) from exc
    return model, processor


# ---------------------------------------------------------------------------
# Pure helpers (no model) — unit-testable
# ---------------------------------------------------------------------------
def l2_normalize(arr: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize ``arr`` along ``axis`` (float32, safe against zero rows)."""
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(norm, eps)


def average_prompt_embeddings(
    per_template_embs: dict[str, np.ndarray],
) -> np.ndarray:
    """Average per-template text embeddings into one prototype per subtype.

    ``per_template_embs`` maps each :data:`SUBTYPES` entry to an
    ``(n_templates, dim)`` matrix of (already-normalized) template embeddings.
    Returns an ``(len(SUBTYPES), dim)`` matrix, one L2-normalized row per
    subtype in :data:`SUBTYPES` order.
    """
    rows = []
    for label in SUBTYPES:
        embs = np.asarray(per_template_embs[label], dtype=np.float32)
        if embs.ndim != 2 or embs.shape[0] == 0:
            raise FigureRouterError(
                f"prompt embeddings for {label!r} must be (n_templates, dim), "
                f"got shape {embs.shape}"
            )
        rows.append(embs.mean(axis=0))
    return l2_normalize(np.stack(rows, axis=0), axis=1)


# ---------------------------------------------------------------------------
# Embedding passes
# ---------------------------------------------------------------------------
def _to_pil_list(paths_or_pils: Sequence[Any]) -> list[Any]:
    """Open any path-like inputs as RGB PIL images; pass PIL images through."""
    from PIL import Image

    out = []
    for item in paths_or_pils:
        if isinstance(item, (str, bytes)) or hasattr(item, "__fspath__"):
            out.append(Image.open(item).convert("RGB"))
        else:
            out.append(item.convert("RGB") if item.mode != "RGB" else item)
    return out


def _pooled_numpy(features: Any) -> np.ndarray:
    """Extract the pooled embedding tensor as float32 numpy.

    transformers 5.x ``get_{image,text}_features`` returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` is the pooled
    embedding; older/other versions return the tensor directly. Handle both.
    """
    tensor = getattr(features, "pooler_output", features)
    return tensor.detach().cpu().numpy().astype(np.float32)


def embed_images(paths_or_pils: Sequence[Any], batch_size: int = 16) -> np.ndarray:
    """L2-normalized SigLIP-2 image embeddings for a list of images.

    ``paths_or_pils`` may mix filesystem paths (str / ``os.PathLike``) and PIL
    images. Returns an ``(N, dim)`` float32 array of unit-norm image features.

    Raises
    ------
    FigureRouterError
        On a model/inference failure (NOT for an individually-corrupt image —
        the caller batching over a dataset is responsible for pre-filtering /
        try-excepting per file so one bad image does not abort the pass).
    """
    if len(paths_or_pils) == 0:
        # Empty pass -> empty (0, 0) array. The embedding dim is only known
        # after a real forward pass, so callers concatenate real rows
        # themselves and check length rather than relying on the column count.
        return np.empty((0, 0), dtype=np.float32)
    try:
        import torch

        model, processor = load_router()
        feats: list[np.ndarray] = []
        pils = _to_pil_list(paths_or_pils)
        for start in range(0, len(pils), batch_size):
            chunk = pils[start : start + batch_size]
            inputs = processor(images=chunk, return_tensors="pt")
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            feats.append(_pooled_numpy(emb))
        stacked = np.concatenate(feats, axis=0)
    except FigureRouterError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface, don't swallow
        raise FigureRouterError(f"image embedding failed: {type(exc).__name__}: {exc}") from exc
    return l2_normalize(stacked, axis=1)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """L2-normalized SigLIP-2 text embeddings for a list of strings.

    SigLIP-2's text tower is fixed at :data:`SIGLIP_TEXT_MAX_TOKENS` (64)
    tokens with ``padding="max_length"``; longer strings are truncated by the
    processor. Returns an ``(N, dim)`` float32 array of unit-norm text features.

    Raises
    ------
    FigureRouterError
        On a model/inference failure.
    """
    if len(texts) == 0:
        return np.empty((0, 0), dtype=np.float32)
    try:
        import torch

        model, processor = load_router()
        inputs = processor(
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            max_length=SIGLIP_TEXT_MAX_TOKENS,
            truncation=True,
        )
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        out = _pooled_numpy(emb)
    except FigureRouterError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface, don't swallow
        raise FigureRouterError(f"text embedding failed: {type(exc).__name__}: {exc}") from exc
    return l2_normalize(out, axis=1)


@lru_cache(maxsize=1)
def subtype_prototype_embeddings() -> np.ndarray:
    """Cached ``(len(SUBTYPES), dim)`` matrix of averaged class prototypes.

    Embeds every template in :data:`SUBTYPE_PROMPTS`, averages per subtype, and
    re-normalizes. Cached because it is constant for a given model.
    """
    per_template: dict[str, np.ndarray] = {}
    for label in SUBTYPES:
        per_template[label] = embed_texts(SUBTYPE_PROMPTS[label])
    return average_prompt_embeddings(per_template)


def zero_shot_subtype(
    image_emb: np.ndarray, prompt_embs: np.ndarray | None = None
) -> tuple[str, float]:
    """Classify one figure's image embedding into the taxonomy.

    ``image_emb`` is a single ``(dim,)`` (or ``(1, dim)``) L2-normalized image
    embedding. ``prompt_embs`` is the ``(len(SUBTYPES), dim)`` class-prototype
    matrix (defaults to :func:`subtype_prototype_embeddings`). Returns
    ``(label, confidence)`` where ``confidence`` is the softmax probability of
    the winning class over the per-class cosine similarities — a 0..1 score
    that is comparable across images (raw cosine margins are not).
    """
    if prompt_embs is None:
        prompt_embs = subtype_prototype_embeddings()
    img = np.asarray(image_emb, dtype=np.float32).reshape(-1)
    sims = prompt_embs @ img  # cosine sims (both sides unit-norm)
    # Softmax over class sims for a calibrated-ish 0..1 confidence. A small
    # temperature sharpens the otherwise-flat SigLIP cosine spread.
    temp = 100.0
    z = sims * temp
    z = z - z.max()
    probs = np.exp(z)
    probs = probs / probs.sum()
    idx = int(np.argmax(probs))
    return SUBTYPES[idx], float(probs[idx])


@lru_cache(maxsize=1)
def binary_prototype_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Cached prototypes for the binary chart-vs-rest gate.

    Returns ``(chart_proto, nonchart_protos)``:

    * ``chart_proto`` — ``(dim,)`` L2-normalized average of every template in
      :data:`CHART_BINARY_PROMPTS` (the 5-class chart templates PLUS the
      scatter/embedding/fan-chart recall-leak variants);
    * ``nonchart_protos`` — ``(len(NONCHART_SUBTYPES), dim)`` matrix of the
      unchanged 5-class prototypes for the non-chart classes, one L2-normalized
      row per :data:`NONCHART_SUBTYPES` entry.

    Cached because both are constant for a given model.
    """
    chart_templates = embed_texts(CHART_BINARY_PROMPTS)
    chart_proto = l2_normalize(chart_templates.mean(axis=0), axis=-1)
    nonchart_rows = []
    for label in NONCHART_SUBTYPES:
        embs = embed_texts(SUBTYPE_PROMPTS[label])
        nonchart_rows.append(embs.mean(axis=0))
    nonchart_protos = l2_normalize(np.stack(nonchart_rows, axis=0), axis=1)
    return chart_proto, nonchart_protos


def route_chart_binary(
    image: Any,
    chart_proto: np.ndarray | None = None,
    nonchart_protos: np.ndarray | None = None,
    margin: float = CHART_BINARY_MARGIN,
) -> dict[str, Any]:
    """Binary chart-vs-rest routing decision — the shipped P2 figure gate.

    **Precision-first contract.** This gate decides whether a figure goes to
    the chart-description tier (future DePlot) or stays on the generic
    figure-caption tier. A FALSE POSITIVE here is the expensive error: a
    chart-tier description of a non-chart figure is a confidently *wrong*
    reading of the image, which is worse for the WCAG alt-text than the
    merely *generic* description a false negative receives. The decision is
    therefore margin-gated: ``is_chart`` is True only when the chart
    prototype's cosine similarity beats the BEST non-chart class prototype by
    at least ``margin`` (:data:`CHART_BINARY_MARGIN`); ties and narrow wins
    resolve to False. Spot-check baseline (raw zero-shot argmax): precision
    0.90 / recall 0.69; this gate adds scatter/embedding/fan-chart prompt
    variants on the chart side to claw back the known recall leaks without
    touching the 5-class taxonomy path.

    Parameters
    ----------
    image:
        Either a precomputed L2-normalized SigLIP-2 image embedding
        (``(dim,)`` or ``(1, dim)`` ``np.ndarray``) — no model needed — or a
        PIL image / filesystem path, which is embedded via
        :func:`embed_images` (loads the model).
    chart_proto, nonchart_protos:
        Optional prototype overrides (both default to
        :func:`binary_prototype_embeddings`; supplying both keeps the call
        model-free, which the tests rely on).
    margin:
        Required cosine lead of chart over the best non-chart class.

    Returns
    -------
    dict
        ``is_chart`` (bool), ``score`` (float 0..1 — two-way softmax at
        temperature 100 between the chart similarity and the best non-chart
        similarity; 0.5 at a tie, monotone in the raw margin), ``raw_margin``
        (float, cosine units), ``method`` (``"siglip2_zero_shot_margin"``).
    """
    if chart_proto is None or nonchart_protos is None:
        default_chart, default_nonchart = binary_prototype_embeddings()
        if chart_proto is None:
            chart_proto = default_chart
        if nonchart_protos is None:
            nonchart_protos = default_nonchart
    if isinstance(image, np.ndarray):
        img = np.asarray(image, dtype=np.float32).reshape(-1)
    else:
        img = embed_images([image])[0]
    chart_sim = float(np.asarray(chart_proto, dtype=np.float32).reshape(-1) @ img)
    non_sims = np.asarray(nonchart_protos, dtype=np.float32) @ img
    best_non = float(non_sims.max())
    raw_margin = chart_sim - best_non
    # Same temperature as the 5-class softmax: ~0.01 cosine ≈ 1 logit.
    temp = 100.0
    score = float(1.0 / (1.0 + np.exp(-temp * raw_margin)))
    return {
        "is_chart": bool(raw_margin >= margin),
        "score": score,
        "raw_margin": float(raw_margin),
        "method": "siglip2_zero_shot_margin",
    }


# ---------------------------------------------------------------------------
# Trained 5-class head (P2): calibrated logreg on the frozen embeddings
# ---------------------------------------------------------------------------
# Calibrated-probability floor below which the head ABSTAINS and routes the
# figure to "other" (the generic-caption tier). Chosen in the P2/P3 fine-plan
# (2026-06-09) alongside the head recipe; CalibratedClassifierCV at train
# time is what makes thresholding the probability meaningful.
SUBTYPE_ABSTAIN_THRESHOLD: float = 0.55

# Default head location (written by scripts/training/train_figure_router.py).
SUBTYPE_HEAD_PATH = "models/figure_router/v1/head.joblib"


@lru_cache(maxsize=2)
def load_subtype_head(path: str = SUBTYPE_HEAD_PATH) -> Any:
    """Load the trained 5-class head once per process.

    ``path`` is resolved relative to the repo root when not absolute.

    Raises
    ------
    FigureRouterHeadMissing
        If the head file does not exist (the train script has not been run).
    FigureRouterError
        If the file exists but cannot be unpickled/loaded.
    """
    from pathlib import Path

    from . import paths as _semantik_paths

    p = Path(path)
    if not p.is_absolute():
        # Resolve relative head paths through the shared SemantiK model-root
        # resolver (honors SEMANTIK_MODEL_DIR / SEMANTIK_HOME). The helper strips
        # the legacy leading ``models/`` segment so the package-relative default
        # stays byte-stable to ``<SemantiK>/models/figure_router/v1/head.joblib``.
        p = _semantik_paths.resolve_model_literal(p)
    if not p.exists():
        raise FigureRouterHeadMissing(
            f"trained figure-router head not found at {p} — run "
            "scripts/training/train_figure_router.py (requires the labeled candidate "
            "sets, see data/figure_labels/README.md)"
        )
    try:
        import joblib

        head = joblib.load(p)
    except Exception as exc:  # noqa: BLE001 — surface, don't swallow
        raise FigureRouterError(
            f"failed to load figure-router head {p}: {type(exc).__name__}: {exc}"
        ) from exc
    return head


def classify_subtype(
    image: Any,
    head: Any | None = None,
    abstain_threshold: float = SUBTYPE_ABSTAIN_THRESHOLD,
) -> dict[str, Any]:
    """Classify a figure into the 6-class taxonomy with the TRAINED head.

    The production subtype router (P2). Unlike :func:`zero_shot_subtype`
    this requires the trained calibrated head on disk and abstains rather
    than guess: when the top calibrated probability is below
    ``abstain_threshold`` the figure routes to ``"other"`` (the generic
    caption tier) with ``abstained=True`` — a low-confidence specific
    subtype must not send a figure to a specialized description tier.

    Parameters
    ----------
    image:
        A precomputed L2-normalized SigLIP-2 image embedding (``(dim,)`` or
        ``(1, dim)``) — no encoder load needed — or a PIL image / filesystem
        path, embedded via :func:`embed_images`.
    head:
        Optional head override (defaults to :func:`load_subtype_head`;
        passing it keeps the call model-free, which the tests rely on).
    abstain_threshold:
        Calibrated-probability floor for a non-abstained answer.

    Returns
    -------
    dict
        ``subtype`` (taxonomy member; ``"other"`` when abstained),
        ``probability`` (float — the top class's calibrated probability),
        ``abstained`` (bool), ``raw_subtype`` (the head's argmax class,
        regardless of abstention), ``method``
        (``"siglip2_logreg_head_calibrated"``).

    Raises
    ------
    FigureRouterHeadMissing
        If no trained head is on disk and none was passed.
    FigureRouterError
        On embedding/inference failure or a head emitting labels outside
        the taxonomy.
    """
    if head is None:
        head = load_subtype_head()
    if isinstance(image, np.ndarray):
        emb = np.asarray(image, dtype=np.float32).reshape(1, -1)
    else:
        emb = embed_images([image])[:1]
    try:
        proba = np.asarray(head.predict_proba(emb))[0]
        classes = list(head.classes_)
    except Exception as exc:  # noqa: BLE001 — surface, don't swallow
        raise FigureRouterError(
            f"subtype head inference failed: {type(exc).__name__}: {exc}"
        ) from exc
    idx = int(np.argmax(proba))
    raw_subtype = str(classes[idx])
    if raw_subtype not in SUBTYPES:
        raise FigureRouterError(f"head predicted {raw_subtype!r}, not in taxonomy {SUBTYPES}")
    top_p = float(proba[idx])
    abstained = top_p < abstain_threshold
    return {
        "subtype": "other" if abstained else raw_subtype,
        "probability": top_p,
        "abstained": abstained,
        "raw_subtype": raw_subtype,
        "method": "siglip2_logreg_head_calibrated",
    }


def similarity(image_emb: np.ndarray, text_emb: np.ndarray) -> float:
    """SigLIP image/text agreement score — the future consistency-gate primitive.

    Returns the **sigmoid** of the SigLIP-scaled logit
    (``logit = cosine * logit_scale + logit_bias``, then ``sigmoid``), i.e. the
    model's calibrated 0..1 probability that this image and text are a matched
    pair — the scale SigLIP was contrastively trained on (NOT a raw cosine).

    ``image_emb`` and ``text_emb`` are single L2-normalized embeddings. The
    SigLIP logit scale / bias are read from the loaded model.
    """
    try:
        import torch

        model, _ = load_router()
        img = np.asarray(image_emb, dtype=np.float32).reshape(-1)
        txt = np.asarray(text_emb, dtype=np.float32).reshape(-1)
        cosine = float(np.dot(img, txt))
        logit_scale = float(model.logit_scale.detach().cpu().exp())
        logit_bias = float(model.logit_bias.detach().cpu())
        logit = cosine * logit_scale + logit_bias
        return float(torch.sigmoid(torch.tensor(logit)))
    except FigureRouterError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface, don't swallow
        raise FigureRouterError(f"similarity failed: {type(exc).__name__}: {exc}") from exc
