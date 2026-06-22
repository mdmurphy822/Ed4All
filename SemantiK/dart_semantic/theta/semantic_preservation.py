"""Theta dimension #1 — semantic-preservation cross-encoder runtime.

Wraps the Phase 4b artifact at ``models/theta/semantic_preservation/v1/``
(DeBERTa-v3-small + LoRA + linear(768->1) head with optional isotonic
calibration). The runtime is import-light: nothing in this module's
top-level scope imports torch / transformers / peft / sklearn, so the
``stub_v1`` fallback path in :mod:`dart_semantic.theta.evaluator` does
not pull in any heavy GPU libraries.

Public surface:
    * :class:`SemanticPreservationModel` — frozen dataclass holding
      tokenizer + model + head + calibration + version info.
    * :func:`load(model_dir)` — returns the loaded model, or ``None``
      when the artifact directory is missing / incomplete (and
      ``DART_REQUIRE_THETA_MODEL`` is unset). Performs a one-shot
      health check on a sentinel pair after load.
    * :func:`score(doc, *, feature_blocks, model)` — scores an
      :class:`AssembledDoc` end-to-end and returns
      ``(doc_score, breakdown)``.

Aggregation rule (locked at v1):
    Per-region cross-encoder scores in [0, 1]. Skip regions where
    either side is < 20 words (assigned 1.0, weight 0). Length-weight
    regions by ``min(source_chars, candidate_chars)``. Final doc score:

        0.7 * length_weighted_mean + 0.3 * p20

    where p20 is the 20th-percentile of per-region scores (linear
    interpolation, numpy-compatible). ``p20`` defaults to 1.0 when
    no regions are scored.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - type-only
    from dart_semantic.assembler.types import AssembledDoc
    from dart_semantic.types import FeatureBlock


logger = logging.getLogger(__name__)


# --- Constants (locked at v1) -------------------------------------------------

#: Method tag stamped onto breakdown when the model is live.
METHOD_MODEL = "model_v1"
#: Method tag stamped onto breakdown when we're using the stub fallback.
METHOD_STUB = "stub_v1"

#: Minimum word count on EITHER side of a region pair before we'll score it.
#: Regions below this threshold are assigned a neutral 1.0 with weight 0.
MIN_WORDS = 20

#: Region kinds that carry running prose and are grouped into paragraph-sized
#: scoring units. The pipeline emits sentence/line-level regions (median ~12
#: words), so a lone region rarely clears MIN_WORDS; grouping consecutive prose
#: regions reconstructs paragraph-level units that match the cross-encoder's
#: 20-450 token training distribution. Non-prose kinds (heading, table, figure,
#: code_block, …) are unit boundaries and scored individually.
PROSE_KINDS = frozenset({"paragraph", "list", "blockquote"})

#: Flush a prose group once it reaches this many source words, to stay well
#: under the 512-token encoder limit (training MAX_REGION_TOKENS was 450).
MAX_GROUP_WORDS = 300

#: Cross-encoder forward batch size. Calibrated for the 8GB RTX 3060 with
#: max_length=512 — DeBERTa-v3-small + LoRA fits 64 in well under 4GB.
BATCH_SIZE = 64

#: max_length passed to the tokenizer (truncation="longest_first").
MAX_LENGTH = 512

#: Aggregation weights for the doc-level rollup.
W_WEIGHTED_MEAN = 0.7
W_P20 = 0.3

#: Cosmic-low fallback for the doc score when no regions could be scored
#: (e.g. tiny doc, all regions below MIN_WORDS). Per acceptance gate #6.
EMPTY_DOC_SCORE = 1.0

#: On-load health check — DISCRIMINATION, not an absolute threshold.
#: The old check ("identical strings score > 0.9") is wrong for this model
#: family: the trainer calibrates so even meaning-PRESERVING text lands near a
#: ~0.65 floor (matching ``floor_semantic_preservation``), so a correctly
#: trained model scores identical text ~0.77, not >0.9 — the 0.9 check rejected
#: the good model and (worse) would have *passed* nothing meaningful. A
#: mode-collapsed predictor (theta v1) emits a near-constant ~0.70 for ANY pair,
#: so the right invariant is: a meaning-preserving pair must score materially
#: ABOVE an unrelated pair. v8 separates these by ~0.38 (calibrated); a collapsed
#: model separates by ~0. We require a gap of at least ``_SENTINEL_MIN_GAP`` and
#: a preserved score above ``_SENTINEL_MIN_PRESERVED``. Texts are >=20 words so
#: they sit in the model's trained distribution (unlike the old "Hello world").
_SENTINEL_PRESERVED_SRC = (
    "The experiment measured the decoherence rate of a superconducting qubit "
    "as a function of temperature, finding an exponential increase above forty "
    "millikelvin consistent with a two-level-system noise model."
)
#: Identical candidate -> meaning fully preserved.
_SENTINEL_PRESERVED_CAND = _SENTINEL_PRESERVED_SRC
#: Unrelated candidate (different domain), similar length -> meaning destroyed.
_SENTINEL_VIOLATED_CAND = (
    "Quarterly retail sales declined two percent amid weak consumer sentiment, "
    "while mortgage applications rose as long-term interest rates eased for the "
    "third consecutive week according to the trade group."
)
#: Minimum calibrated gap (preserved - violated) for a non-collapsed model.
_SENTINEL_MIN_GAP = 0.10
#: Minimum calibrated score for the preserved pair (sanity floor).
_SENTINEL_MIN_PRESERVED = 0.55

#: Tag-stripping regex used to extract candidate innerText from doc.html.
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
#: Whitespace collapse for both source and candidate text.
_WS_RE = re.compile(r"\s+")

# Optional region kind labels — surfaced in breakdown for diagnostics.
# We don't introspect the AssembledDoc beyond what's documented; if a
# downstream caller adds a `region_kinds` field we'll pick it up via
# getattr.


# --- Data class ---------------------------------------------------------------


@dataclass(frozen=True)
class SemanticPreservationModel:
    """The loaded cross-encoder + calibration bundle.

    Fields are duck-typed because we don't want torch / transformers in
    this module's top-level imports. Treat ``model`` as a PeftModel,
    ``head`` as a torch.nn.Linear, ``isotonic`` as an
    ``sklearn.isotonic.IsotonicRegression`` or ``None``.
    """

    tokenizer: Any
    model: Any                # PeftModel wrapping AutoModel(deberta-v3-small)
    head: Any                 # torch.nn.Linear(768, 1) — emits sigmoid logits
    isotonic: Any | None      # sklearn IsotonicRegression or None
    offset: float
    version: str
    dataset_sha256: str
    device: str
    pooling: str = "cls"      # 'cls' or 'mean' — must match training (summary.json)


# --- Loader -------------------------------------------------------------------


def _required_artifact_paths(model_dir: Path) -> dict[str, Path]:
    """Map of artifact-name -> expected absolute path. Used by load().

    Only tokenizer + summary + head are universally required. The weights
    artifact is mode-specific (LoRA adapter vs full-FT model.safetensors) and
    validated in load() after reading summary.json's ``finetune`` field.
    calibration.json is OPTIONAL (identity map if absent).
    """
    return {
        "tokenizer_dir": model_dir / "tokenizer",
        "summary": model_dir / "summary.json",
    }


def _artifact_inventory(paths: dict[str, Path]) -> tuple[bool, list[str]]:
    missing = [name for name, p in paths.items() if not p.exists()]
    return (not missing), missing


def load(model_dir: Path) -> SemanticPreservationModel | None:
    """Load the cross-encoder bundle from ``model_dir``.

    Returns ``None`` when the artifact directory is missing or doesn't
    contain the expected files (and ``DART_REQUIRE_THETA_MODEL`` is
    unset — strict-mode escalation lives in
    :mod:`dart_semantic.theta._module_state`, not here, so this loader
    stays single-purpose).

    Raises ``RuntimeError`` if artifacts ARE present but loading fails
    (corrupt safetensors, version skew, etc.) — that's a deployment
    bug, not an "operating without a model" condition. The caller in
    ``_module_state`` will translate / log appropriately.

    Health-check: after load, runs one forward pass on
    ``("Hello world", "Hello world")`` and asserts the score is > 0.9.
    Raises if it isn't.
    """
    if not model_dir.exists():
        logger.info("Theta semantic-preservation: model_dir %s missing.", model_dir)
        return None

    paths = _required_artifact_paths(model_dir)
    ok, missing = _artifact_inventory(paths)
    if not ok:
        logger.info(
            "Theta semantic-preservation: artifacts %s missing under %s.",
            missing,
            model_dir,
        )
        return None

    # Heavy imports — only on the model-loaded path.
    import torch
    from peft import PeftModel
    from transformers import AutoModel, DebertaV2Tokenizer

    # Artifact: tokenizer (slow DebertaV2 with spm.model — locked by Phase 4b).
    tokenizer_dir = paths["tokenizer_dir"]
    tokenizer = DebertaV2Tokenizer.from_pretrained(str(tokenizer_dir))

    # Artifact: base model + LoRA adapter.
    base_name = "microsoft/deberta-v3-small"
    summary_path = paths["summary"]
    summary = json.loads(summary_path.read_text())
    base_name = summary.get("base_model", base_name)

    finetune = str(summary.get("finetune", "lora"))

    # Mirror the trainer: bf16 base, fp32 pooled cast before the head
    # (scripts/train_semantic_preservation.py). deberta-v3-small's config
    # defaults to torch_dtype=float16, which mismatches the fp32 head
    # ("mat1 and mat2 must have the same dtype, but got Half and Float");
    # the trainer used bf16 base + pooled.float().
    if finetune == "full":
        # Full fine-tune: the directory itself is a complete AutoModel
        # checkpoint (config.json + model.safetensors). No LoRA adapter.
        weights = model_dir / "model.safetensors"
        if not weights.exists():
            raise RuntimeError(
                f"Theta full-FT checkpoint missing model.safetensors under "
                f"{model_dir!s} (summary says finetune=full)."
            )
        model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    else:
        adapter = model_dir / "adapter_model.safetensors"
        if not adapter.exists():
            raise RuntimeError(
                f"Theta LoRA checkpoint missing adapter_model.safetensors under "
                f"{model_dir!s} (summary says finetune={finetune})."
            )
        base_model = AutoModel.from_pretrained(base_name, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()

    # Artifact: linear head — Phase 4b emits ``head.safetensors`` alongside
    # the adapter (linear 768->1 with bias). We accept either ``head.bin``
    # / ``head.safetensors`` / ``head.pt``.
    head = _load_linear_head(model_dir, hidden_size=_hidden_size(model))
    head.eval()

    # Artifact: calibration.json (isotonic params + offset). Optional.
    calibration_path = model_dir / "calibration.json"
    isotonic, offset = _load_calibration(calibration_path)

    # Move to device. We pick CPU by default (Phase 4c is integration —
    # callers can swap by setting DART_THETA_DEVICE=cuda).
    import os as _os
    device_pref = _os.environ.get("DART_THETA_DEVICE", "cpu")
    device = device_pref if device_pref == "cpu" or torch.cuda.is_available() else "cpu"
    model.to(device)
    head.to(device)

    bundle = SemanticPreservationModel(
        tokenizer=tokenizer,
        model=model,
        head=head,
        isotonic=isotonic,
        offset=offset,
        version=str(summary.get("phase", "4b")) + "/" + str(summary.get("base_model", base_name)),
        dataset_sha256=str(
            summary.get("preflight", {})
            .get("gate2_dataset_sha256", {})
            .get("sha256", {})
            .get("train.jsonl", "")
        ),
        device=device,
        pooling=str(summary.get("pooling", "cls")),
    )

    # On-load health check — discrimination, not an absolute threshold (see the
    # _SENTINEL_* constants). A meaning-preserving pair must score materially
    # above an unrelated pair; a mode-collapsed model (constant output) fails.
    raw = _raw_score_pairs(
        bundle,
        [
            (_SENTINEL_PRESERVED_SRC, _SENTINEL_PRESERVED_CAND),
            (_SENTINEL_PRESERVED_SRC, _SENTINEL_VIOLATED_CAND),
        ],
    )
    if len(raw) < 2:
        raise RuntimeError(
            "Theta semantic-preservation sentinel scoring returned <2 results."
        )
    preserved = _calibrate(float(raw[0]), isotonic, offset)
    violated = _calibrate(float(raw[1]), isotonic, offset)
    gap = preserved - violated
    if preserved < _SENTINEL_MIN_PRESERVED or gap < _SENTINEL_MIN_GAP:
        raise RuntimeError(
            "Theta semantic-preservation model failed sentinel health check "
            f"(preserved={preserved:.4f}, violated={violated:.4f}, gap={gap:.4f}; "
            f"need preserved>={_SENTINEL_MIN_PRESERVED} and "
            f"gap>={_SENTINEL_MIN_GAP}). Refusing to load — model is "
            "mode-collapsed or mis-wired (does not discriminate "
            "meaning-preserving from unrelated text)."
        )

    logger.info(
        "Theta semantic-preservation: loaded %s on %s "
        "(sentinel preserved=%.4f violated=%.4f gap=%.4f).",
        bundle.version,
        device,
        preserved,
        violated,
        gap,
    )
    return bundle


def _hidden_size(model: Any) -> int:
    """Pull hidden_size from the wrapped HF config."""
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "base_model"):
        cfg = getattr(model.base_model, "config", None)
    if cfg is None:
        return 768  # deberta-v3-small default
    return int(getattr(cfg, "hidden_size", 768))


def _load_linear_head(model_dir: Path, *, hidden_size: int) -> Any:
    """Load the linear(hidden_size -> 1) head from any of the standard names."""
    import torch

    # Phase 4b's training script emits ``heads.pt`` (a torch dict whose
    # ``head_score.state_dict`` key holds the linear head's state_dict).
    # Verified by reading scripts/train_semantic_preservation.py:save_adapter
    # — it does ``torch.save({"head_score.state_dict": ...}, "heads.pt")``.
    heads_pt = model_dir / "heads.pt"
    if heads_pt.exists():
        head = torch.nn.Linear(hidden_size, 1)
        bundle = torch.load(str(heads_pt), map_location="cpu")
        if isinstance(bundle, dict) and "head_score.state_dict" in bundle:
            head.load_state_dict(bundle["head_score.state_dict"])
        else:
            # Older / alternate layout: heads.pt is the bare state_dict.
            head.load_state_dict(bundle)
        return head

    # Backward-compat fallbacks for hand-rolled exports / future trainer revs.
    candidates = [
        model_dir / "head.safetensors",
        model_dir / "head.bin",
        model_dir / "head.pt",
        model_dir / "linear_head.bin",
    ]
    for path in candidates:
        if not path.exists():
            continue
        head = torch.nn.Linear(hidden_size, 1)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(str(path), map_location="cpu")
        head.load_state_dict(state)
        return head

    # Last-ditch fallback — Phase 4b might inline the head into a single
    # ``model.safetensors`` blob with key prefix ``head.``. We try to
    # peel it out before giving up.
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        blob = load_file(str(safetensors_path))
        prefix = "head."
        head_state = {
            k[len(prefix):]: v for k, v in blob.items() if k.startswith(prefix)
        }
        if head_state:
            head = torch.nn.Linear(hidden_size, 1)
            head.load_state_dict(head_state)
            return head

    raise RuntimeError(
        f"Theta semantic-preservation: no linear head artifact found in "
        f"{model_dir} (looked for head.safetensors / head.bin / head.pt)."
    )


def _load_calibration(calibration_path: Path) -> tuple[Any | None, float]:
    """Load isotonic calibration if present, else return identity (None, 0.0)."""
    if not calibration_path.exists():
        return None, 0.0
    try:
        payload = json.loads(calibration_path.read_text())
    except Exception as exc:  # noqa: BLE001 — best-effort; corrupt file -> identity
        logger.warning(
            "Theta semantic-preservation: calibration.json unreadable (%r) — "
            "using identity calibration.",
            exc,
        )
        return None, 0.0

    offset = float(payload.get("offset", 0.0))

    # Isotonic params: Phase 4b emits FLAT keys ``isotonic_X_thresholds`` and
    # ``isotonic_y_thresholds`` directly on the JSON root (verified by reading
    # scripts/train_semantic_preservation.py:fit_calibration). We accept both
    # the flat layout and a legacy nested ``isotonic: {x_thresholds, ...}``
    # layout; bounds default to [0, 1] since the trainer uses ``clip``.
    xs: list[float] | None = None
    ys: list[float] | None = None
    y_min, y_max = 0.0, 1.0
    if "isotonic_X_thresholds" in payload and "isotonic_y_thresholds" in payload:
        xs = payload["isotonic_X_thresholds"]
        ys = payload["isotonic_y_thresholds"]
    else:
        iso_params = payload.get("isotonic")
        if iso_params:
            xs = iso_params.get("x_thresholds")
            ys = iso_params.get("y_thresholds")
            y_min = iso_params.get("y_min", 0.0)
            y_max = iso_params.get("y_max", 1.0)
    if not xs or not ys:
        return None, offset

    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(
            out_of_bounds="clip",
            y_min=y_min,
            y_max=y_max,
        )
        iso.fit(xs, ys)
        return iso, offset
    except Exception as exc:  # noqa: BLE001 — corrupt -> identity
        logger.warning(
            "Theta semantic-preservation: failed to rebuild isotonic (%r); "
            "using identity calibration.",
            exc,
        )
        return None, offset


def _calibrate(raw: float, isotonic: Any | None, offset: float) -> float:
    """Apply isotonic + offset, clamp to [0, 1]."""
    x = float(raw)
    if isotonic is not None:
        try:
            x = float(isotonic.predict([x])[0])
        except Exception:  # noqa: BLE001
            # If the calibrator is broken at runtime, prefer raw scores
            # over a hard crash — log once and continue.
            pass
    x = x + float(offset)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# --- Scoring ------------------------------------------------------------------


def _norm_text(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _strip_html(html_fragment: str) -> str:
    return _norm_text(_TAG_STRIP_RE.sub(" ", html_fragment or ""))


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _split_emitted_blocks(html: str) -> list[str]:
    """Split ``doc.html`` into the sequence of "emitted blocks".

    The AssembledDoc spec says ``region_provenance[i]`` is the index back
    into the source-region list for the ``i``-th emitted block. The
    assembler emits one top-level child of ``<main>`` per block (or per
    table / figure / heading), so we approximate "emitted block" as
    "top-level element under ``<main>``". When the doc has no ``<main>``,
    we fall back to top-level body children.

    The regex is intentionally conservative — we don't try to perfectly
    re-tokenize HTML, we just need ordered chunks that line up with
    ``region_provenance``. If counts don't match, we truncate to the
    shorter list and stamp ``"length_mismatch"`` in the breakdown.
    """
    if not html:
        return []
    main_match = re.search(
        r"<main\b[^>]*>(.*?)</main>", html, re.IGNORECASE | re.DOTALL,
    )
    if main_match:
        inner = main_match.group(1)
    else:
        body_match = re.search(
            r"<body\b[^>]*>(.*?)</body>", html, re.IGNORECASE | re.DOTALL,
        )
        inner = body_match.group(1) if body_match else html

    # Top-level children: greedy match for each balanced-by-tagname block.
    # We accept any block-level tag; nested blocks stay nested.
    blocks: list[str] = []
    pos = 0
    tag_re = re.compile(
        r"<\s*(?P<tag>[a-zA-Z][a-zA-Z0-9]*)\b[^>]*>",
        re.DOTALL,
    )
    while pos < len(inner):
        m = tag_re.search(inner, pos)
        if not m:
            break
        tag = m.group("tag").lower()
        # Find the matching close tag (balanced for nested same-name tags).
        open_re = re.compile(rf"<\s*{tag}\b[^>]*>", re.IGNORECASE | re.DOTALL)
        close_re = re.compile(rf"</\s*{tag}\s*>", re.IGNORECASE | re.DOTALL)
        depth = 1
        cursor = m.end()
        while depth > 0 and cursor < len(inner):
            next_open = open_re.search(inner, cursor)
            next_close = close_re.search(inner, cursor)
            if next_close is None:
                cursor = len(inner)
                break
            if next_open is not None and next_open.start() < next_close.start():
                depth += 1
                cursor = next_open.end()
            else:
                depth -= 1
                cursor = next_close.end()
        blocks.append(inner[m.start():cursor])
        pos = cursor
    return blocks


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile in [0, 100]. Empty -> 1.0."""
    if not values:
        return 1.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _raw_score_pairs(
    model: SemanticPreservationModel,
    pairs: Sequence[tuple[str, str]],
) -> list[float]:
    """Run the cross-encoder over a list of (source, candidate) pairs.

    Returns RAW (uncalibrated) sigmoid scores. Caller applies isotonic
    + offset via :func:`_calibrate`.
    """
    if not pairs:
        return []
    import torch

    sources = [p[0] for p in pairs]
    candidates = [p[1] for p in pairs]

    enc = model.tokenizer(
        sources,
        candidates,
        truncation="longest_first",
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out_scores: list[float] = []
    with torch.inference_mode():
        n = enc["input_ids"].shape[0]
        for i in range(0, n, BATCH_SIZE):
            chunk = {k: v[i:i + BATCH_SIZE] for k, v in enc.items()}
            outputs = model.model(**chunk)
            # Mirror the trainer's forward (must match training-time pooling,
            # recorded in summary.json). cast to fp32 before the fp32 head.
            hidden = outputs.last_hidden_state  # [B, T, hidden]
            if getattr(model, "pooling", "cls") == "mean":
                m = chunk["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                pooled = pooled.float()
            else:
                pooled = hidden[:, 0, :].float()
            logits = model.head(pooled).squeeze(-1)        # [B]
            probs = torch.sigmoid(logits)
            out_scores.extend(probs.detach().cpu().tolist())
    return out_scores


def _region_kind_at(regions: Sequence[Any] | None, region_idx: int) -> str:
    """Kind of the region at ``region_idx`` (from the passed region list)."""
    if (
        regions is not None
        and isinstance(region_idx, int)
        and 0 <= region_idx < len(regions)
    ):
        return str(getattr(regions[region_idx], "kind", "unknown"))
    return "unknown"


def _group_prose_units(
    region_provenance: list,
    regions: Sequence[Any] | None,
    feature_blocks: Sequence["FeatureBlock"],
    cand_for_region,
) -> list[dict]:
    """Group consecutive prose regions into paragraph-sized scoring units.

    Walks regions in emitted (document) order. Consecutive ``PROSE_KINDS``
    regions accumulate into one unit (flushed at ``MAX_GROUP_WORDS`` or at any
    non-prose boundary); every non-prose region becomes its own singleton unit.
    Each unit carries the concatenated source text (aggregated member blocks)
    and the concatenated candidate text (per-region emitted HTML). This makes
    the scored units paragraph-level, matching the cross-encoder's training
    distribution, instead of the sentence-level raw regions.
    """
    units: list[dict] = []
    members: list[int] = []
    src_parts: list[str] = []
    cand_parts: list[str] = []
    words = 0

    def flush() -> None:
        nonlocal members, src_parts, cand_parts, words
        if members:
            units.append({
                "members": list(members),
                "kind": "prose_group" if len(members) > 1 else
                        _region_kind_at(regions, members[0]),
                "source": _norm_text(" ".join(src_parts)),
                "cand": _norm_text(" ".join(cand_parts)),
            })
        members, src_parts, cand_parts, words = [], [], [], 0

    for i in range(len(region_provenance)):
        region_idx = region_provenance[i]
        kind = _region_kind_at(regions, region_idx)
        src = _source_text_for_region(feature_blocks, region_idx, regions)
        cand = cand_for_region(region_idx)
        if kind in PROSE_KINDS:
            members.append(region_idx)
            src_parts.append(src)
            cand_parts.append(cand)
            words += _word_count(src)
            if words >= MAX_GROUP_WORDS:
                flush()
        else:
            # Boundary: close the running prose group, then emit this region
            # as its own unit (headings/short blocks will skip below MIN_WORDS).
            flush()
            units.append({
                "members": [region_idx],
                "kind": kind,
                "source": _norm_text(src),
                "cand": _norm_text(cand),
            })
    flush()
    return units


def score(
    doc: "AssembledDoc",
    *,
    feature_blocks: Sequence["FeatureBlock"],
    regions: Sequence[Any] | None = None,
    model: SemanticPreservationModel,
) -> tuple[float, dict]:
    """Score an AssembledDoc against its source feature_blocks.

    Returns ``(doc_score, breakdown)``. See module docstring for the
    aggregation rule. ``breakdown.method`` is always ``"model_v1"``
    when this function returns; the stub path lives in
    :mod:`dart_semantic.theta.evaluator`.
    """
    region_provenance = list(doc.region_provenance or [])
    if not region_provenance:
        # Empty doc — neutral score, weight 0. Acceptance gate #6.
        return EMPTY_DOC_SCORE, {
            "method": METHOD_MODEL,
            "model_version": model.version,
            "n_regions_scored": 0,
            "n_regions_skipped": 0,
            "weighted_mean": EMPTY_DOC_SCORE,
            "p20": EMPTY_DOC_SCORE,
            "per_region": [],
            "reason": "empty_region_provenance",
        }

    # Candidate text per region: prefer the assembler's exact per-region HTML
    # (``sub_task_log["region_html"]``, index-aligned with region_provenance)
    # over re-parsing ``doc.html``. ``_split_emitted_blocks`` misaligns whenever
    # the count of top-level <main> children != region count (length_mismatch),
    # mis-attributing candidate text to the wrong region.
    region_html = None
    _stl = getattr(doc, "sub_task_log", None) or {}
    if isinstance(_stl, dict):
        _rh = _stl.get("region_html")
        if isinstance(_rh, list):
            region_html = _rh

    if region_html is not None:
        length_mismatch = len(region_html) != len(region_provenance)

        def cand_for_region(ri):
            return _strip_html(region_html[ri]) if (
                isinstance(ri, int) and 0 <= ri < len(region_html)
            ) else ""
    else:
        # Legacy fallback — re-parse doc.html and map by region_provenance.
        emitted_blocks = _split_emitted_blocks(doc.html or "")
        n_emitted = min(len(region_provenance), len(emitted_blocks))
        length_mismatch = len(region_provenance) != len(emitted_blocks)
        _cand_map: dict[int, list[str]] = {}
        for i in range(n_emitted):
            _cand_map.setdefault(region_provenance[i], []).append(emitted_blocks[i])

        def cand_for_region(ri):
            return _strip_html(" ".join(_cand_map.get(ri, [])))

    # Group consecutive prose regions into paragraph-sized scoring units so the
    # comparison matches the cross-encoder's training distribution (the pipeline
    # emits sentence-level regions; a lone region rarely clears MIN_WORDS).
    units = _group_prose_units(
        region_provenance, regions, feature_blocks, cand_for_region,
    )

    # Build the scoring batch — one entry per unit clearing MIN_WORDS on BOTH
    # sides. ``label_idx`` is the unit's first member region index.
    pair_pack: list[tuple[int, list, str, str]] = []
    skipped: list[dict[str, Any]] = []
    for unit in units:
        members = unit["members"]
        label_idx = members[0] if members else -1
        source_text = unit["source"]
        cand_text = unit["cand"]
        src_words = _word_count(source_text)
        cand_words = _word_count(cand_text)
        if src_words < MIN_WORDS or cand_words < MIN_WORDS:
            skipped.append({
                "idx": label_idx,
                "kind": unit["kind"],
                "members": members,
                "source_words": src_words,
                "cand_words": cand_words,
                "reason": "below_min_words",
            })
            continue
        pair_pack.append((label_idx, members, source_text, cand_text))

    if not pair_pack:
        # Everything was skipped — neutral score. Acceptance gate #6 also
        # covers this: we report what we tried in the breakdown.
        return EMPTY_DOC_SCORE, {
            "method": METHOD_MODEL,
            "model_version": model.version,
            "n_regions_scored": 0,
            "n_regions_skipped": len(skipped),
            "weighted_mean": EMPTY_DOC_SCORE,
            "p20": EMPTY_DOC_SCORE,
            "per_region": [],
            "skipped": skipped,
            "length_mismatch": length_mismatch,
        }

    raw_scores = _raw_score_pairs(model, [(s, c) for _, _, s, c in pair_pack])

    per_region: list[dict[str, Any]] = []
    weighted_num = 0.0
    weighted_den = 0.0
    for (label_idx, members, source_text, cand_text), raw in zip(pair_pack, raw_scores):
        calibrated = _calibrate(raw, model.isotonic, model.offset)
        weight = float(min(len(source_text), len(cand_text)))
        weighted_num += calibrated * weight
        weighted_den += weight
        per_region.append({
            "idx": label_idx,
            "members": members,
            "kind": (_region_kind_at(regions, label_idx)
                     if len(members) == 1 else "prose_group"),
            "source_chars": len(source_text),
            "cand_chars": len(cand_text),
            "raw_score": float(raw),
            "calibrated_score": float(calibrated),
        })

    weighted_mean = (
        weighted_num / weighted_den if weighted_den > 0 else EMPTY_DOC_SCORE
    )
    p20 = _percentile([row["calibrated_score"] for row in per_region], 20.0)

    doc_score = W_WEIGHTED_MEAN * weighted_mean + W_P20 * p20
    if doc_score < 0.0:
        doc_score = 0.0
    elif doc_score > 1.0:
        doc_score = 1.0

    breakdown = {
        "method": METHOD_MODEL,
        "model_version": model.version,
        "n_regions_scored": len(per_region),
        "n_regions_skipped": len(skipped),
        "weighted_mean": float(weighted_mean),
        "p20": float(p20),
        "per_region": per_region,
    }
    if skipped:
        breakdown["skipped"] = skipped
    if length_mismatch:
        breakdown["length_mismatch"] = True
    return float(doc_score), breakdown


def _region_kind(doc: "AssembledDoc", region_idx: int) -> str:
    """Best-effort kind label for diagnostics. Falls back to ``"unknown"``."""
    # AssembledDoc doesn't carry per-region kind out of the box; if the
    # caller stuffed it onto sub_task_log, surface that. Otherwise return
    # "unknown" so the breakdown stays JSON-serializable.
    log = getattr(doc, "sub_task_log", None) or {}
    key = f"region_kind:{region_idx}"
    val = log.get(key) if isinstance(log, dict) else None
    return str(val) if val else "unknown"


def _source_text_for_region(
    feature_blocks: Sequence["FeatureBlock"],
    region_idx: int,
    regions: Sequence[Any] | None = None,
) -> str:
    """Source text for ``region_idx`` (an index into the Phase-8 region list).

    ``region_provenance[i]`` indexes into the Phase-8 top-1 regions, and each
    region carries ``member_block_indices`` pointing back into the flat
    ``feature_blocks`` list. The correct source text for a region is therefore
    the concatenation of ALL its member blocks.

    The v1 scorer instead indexed ``feature_blocks[region_idx]`` directly — but
    a region index is NOT a feature-block index (feature_blocks are fine PDF
    line-fragments; regions are the merged units). That pulled a single short
    line, so ~all regions fell below ``MIN_WORDS`` and were skipped (a real
    pipeline scored 1/365 regions). When ``regions`` is provided we aggregate
    member blocks; otherwise we fall back to the legacy 1:1 mapping (keeps
    older callers / unit fixtures that pass only feature_blocks working).
    """
    if region_idx is None or region_idx < 0:
        return ""
    if regions is not None and 0 <= region_idx < len(regions):
        # Stage-5 typed ``Region`` exposes ``feature_block_indices``; the
        # council ``RegionCandidate`` exposes ``member_block_indices``. Accept
        # either so the scorer works whichever region list the caller passes.
        reg = regions[region_idx]
        member_idx = (
            getattr(reg, "feature_block_indices", None)
            or getattr(reg, "member_block_indices", None)
            or []
        )
        parts: list[str] = []
        for i in member_idx:
            if 0 <= i < len(feature_blocks):
                t = getattr(getattr(feature_blocks[i], "raw", None), "text", "") or ""
                if t:
                    parts.append(t)
        if parts:
            return _norm_text(" ".join(parts))
        # Region had no usable member blocks — fall through to legacy mapping.
    if region_idx >= len(feature_blocks):
        return ""
    fb = feature_blocks[region_idx]
    text = getattr(getattr(fb, "raw", None), "text", "") or ""
    return _norm_text(text)


__all__ = [
    "METHOD_MODEL",
    "METHOD_STUB",
    "SemanticPreservationModel",
    "load",
    "score",
]
