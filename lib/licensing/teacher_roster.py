"""Multi-teacher license roster + fail-closed export/ingest guards.

Backlog SFT-C (S6/S7); owner memo ``runtime/scratchpad/licensing_memo.md`` §C-E.

This module is the **machine-readable source of truth** for which model
outputs may become training pairs in a commercially-shipped course-tutor
adapter, and which model *weights* may be ingested as a base checkpoint.
The prose posture lives in ``docs/LICENSING.md``; this module converts it
into a build invariant per the memo's maintenance-contract note.

Three guards, all fail-closed:

1. :func:`assert_export_licenses` — EXPORT-TIME filter over a corpus of
   instruction / preference pairs. Refuses export if **any** pair's teacher
   is ``barred`` / unregistered / claude-tagged, naming the offending pair
   + teacher in the raised error.
2. :func:`assert_checkpoint_license` — per-checkpoint LICENSE assertion at
   model ingest. ``role="base_model"`` bars only non-commercial licenses
   (Qwen-Research, Mistral-MRL); ``role="teacher"`` bars any ``barred``
   verdict.
3. :func:`assert_nemotron_pin` — the FAIL-BUILD-ON-RE-PIN guard. Asserts
   the roster's Nemotron record still pins the **NVIDIA Nemotron Open Model
   License (Dec 15 2025)** and has NOT drifted to the general *NVIDIA Open
   Model License* (which carries the Trustworthy-AI restriction, guardrail
   auto-termination, and a *revocable* grant). Wired into the training
   preflight.

**Registry defaults are byte-identical.** A pair that carries no teacher
signal at all (no ``provider`` / ``generating_seat`` / ``license`` field —
the legacy shape) PASSES the export filter: the filter fires only on a
*positively identified* barred / claude / unregistered teacher. Existing
license-clean ``local`` / ``together`` corpora are unaffected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "BARRED_PROVENANCE",
    "GENERAL_NVIDIA_OML_IDENTITY",
    "NEMOTRON_PINNED_LICENSE_IDENTITY",
    "NEMOTRON_ROSTER_KEY",
    "PROVENANCE_VERDICTS",
    "TEACHER_ROSTER",
    "LicenseGuardError",
    "LicenseRecord",
    "assert_checkpoint_license",
    "assert_export_licenses",
    "assert_nemotron_pin",
    "classify_pair_teacher",
    "license_for_model",
    "provenance_license_tag",
    "stamp_pair_license",
]


class LicenseGuardError(RuntimeError):
    """Raised (fail-closed) by every guard when a license posture is violated."""


LicenseVerdict = str  # one of {"safe", "conditional", "barred"}


@dataclass(frozen=True)
class LicenseRecord:
    """One roster row.

    Args:
        name: canonical roster key (e.g. ``"qwen2.5-32b"``).
        license_spdx: SPDX id or ``LicenseRef-*`` for non-SPDX licenses.
        license_url: authoritative LICENSE URL.
        verdict: TEACHER verdict — ``safe`` | ``conditional`` | ``barred``.
            Consumed by the export filter (``barred`` → refuse).
        obligations: shipping obligations (attribution strings, MAU flags).
        commercial_use: whether commercial use of the *weights* is permitted
            at all. ``False`` for non-commercial licenses (Qwen-Research,
            Mistral-MRL). Consumed by the base-model ingest assertion.
        license_identity: the human-readable license-identity string the
            license-pin guard compares against (defaults to ``license_spdx``).
    """

    name: str
    license_spdx: str
    license_url: str
    verdict: LicenseVerdict
    obligations: Tuple[str, ...] = field(default_factory=tuple)
    commercial_use: bool = True
    license_identity: str = ""

    def as_tag(self) -> str:
        """Compact ``name/spdx/verdict`` provenance tag stamped per pair."""
        return f"{self.name}/{self.license_spdx}/{self.verdict}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "license_spdx": self.license_spdx,
            "license_url": self.license_url,
            "verdict": self.verdict,
            "obligations": list(self.obligations),
            "commercial_use": self.commercial_use,
            "license_identity": self.identity,
        }

    @property
    def identity(self) -> str:
        return self.license_identity or self.license_spdx


# --------------------------------------------------------------------------- #
# Nemotron license pin (memo §A / §E)                                          #
# --------------------------------------------------------------------------- #
#
# The model actually running is governed by the NVIDIA *Nemotron* Open Model
# License (Dec 15 2025) — SAFE-Case-B for an output-trained adapter. This is
# NOT the general *NVIDIA Open Model License* (Oct 24 2025), which carries the
# Trustworthy-AI restriction, guardrail auto-termination, and a REVOCABLE
# grant. The pin guard fails the build if the roster ever drifts to the
# general license.
NEMOTRON_ROSTER_KEY = "nemotron"
NEMOTRON_PINNED_LICENSE_IDENTITY = "NVIDIA Nemotron Open Model License, Dec 15 2025"
GENERAL_NVIDIA_OML_IDENTITY = "NVIDIA Open Model License"


# --------------------------------------------------------------------------- #
# The roster (memo §C)                                                         #
# --------------------------------------------------------------------------- #

_APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
_MIT_URL = "https://opensource.org/license/mit"

TEACHER_ROSTER: Dict[str, LicenseRecord] = {
    # ---- Qwen tiers -------------------------------------------------------
    "qwen2.5-1.5b": LicenseRecord(
        "qwen2.5-1.5b", "Apache-2.0",
        "https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE",
        "safe",
    ),
    "qwen2.5-7b": LicenseRecord(
        "qwen2.5-7b", "Apache-2.0",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE",
        "safe",
    ),
    "qwen2.5-14b": LicenseRecord(
        "qwen2.5-14b", "Apache-2.0",
        "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/LICENSE",
        "safe",
    ),
    "qwen2.5-32b": LicenseRecord(
        "qwen2.5-32b", "Apache-2.0",
        "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/LICENSE",
        "safe",
    ),
    "qwen2.5-72b": LicenseRecord(
        "qwen2.5-72b", "LicenseRef-Qwen",
        "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE",
        "conditional",
        obligations=(
            "Emit 'Built with Qwen' / 'Improved using Qwen' in shipped docs",
            "Flag the 100M-MAU commercial trigger",
        ),
    ),
    "qwen2.5-3b": LicenseRecord(
        "qwen2.5-3b", "LicenseRef-Qwen-Research",
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE",
        "barred",
        obligations=("Non-commercial only (Qwen Research License)",),
        commercial_use=False,
    ),
    # ---- NVIDIA Nemotron (SAFE Case-B) ------------------------------------
    NEMOTRON_ROSTER_KEY: LicenseRecord(
        NEMOTRON_ROSTER_KEY, "LicenseRef-NVIDIA-Nemotron-OML-2025-12-15",
        "https://huggingface.co/nvidia/Nemotron-3-Super-120B",
        "safe",
        obligations=(
            "No NOTICE required on the output-trained adapter (Case B); "
            "carry voluntary attribution as provenance hygiene",
        ),
        license_identity=NEMOTRON_PINNED_LICENSE_IDENTITY,
    ),
    # ---- Mistral tiers ----------------------------------------------------
    "mistral-apache": LicenseRecord(
        "mistral-apache", "Apache-2.0",
        "https://huggingface.co/mistralai/Mistral-Small-Instruct-2409/blob/main/LICENSE",
        "safe",
    ),
    "mistral-mrl": LicenseRecord(
        "mistral-mrl", "LicenseRef-Mistral-MRL",
        "https://mistral.ai/licenses/MRL-0.1.md",
        "barred",
        obligations=("Non-commercial (Mistral Research License / MNPL)",),
        commercial_use=False,
    ),
    # ---- Phi --------------------------------------------------------------
    "phi": LicenseRecord(
        "phi", "MIT",
        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE.md",
        "safe",
        obligations=("Retain the MIT notice on redistributed weights only",),
    ),
    # ---- GLM --------------------------------------------------------------
    "glm": LicenseRecord(
        "glm", "MIT",
        "https://huggingface.co/zai-org/GLM-4.5",
        "safe",
        obligations=(
            "Retain the MIT notice; verify per-repo (legacy GLM-4-9B is custom)",
        ),
    ),
    # ---- OLMo -------------------------------------------------------------
    "olmo": LicenseRecord(
        "olmo", "Apache-2.0",
        "https://huggingface.co/allenai/OLMo-2-1124-7B",
        "safe",
        obligations=("Most-defensible provenance (data ODC-BY)",),
    ),
    # ---- DeepSeek ---------------------------------------------------------
    "deepseek": LicenseRecord(
        "deepseek", "MIT",
        "https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL",
        "safe",
        obligations=("Avoid the Llama-based R1-distill checkpoints",),
    ),
    # ---- SmolLM2 (student-base only) --------------------------------------
    "smollm2": LicenseRecord(
        "smollm2", "Apache-2.0",
        "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B",
        "safe",
    ),
    # ---- Llama (BARRED as a teacher) --------------------------------------
    "llama": LicenseRecord(
        "llama", "LicenseRef-Llama-Community",
        "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE",
        "barred",
        obligations=(
            "Forces a leading 'Llama-' name on the adapter",
            "AUP flow-down",
            ">700M-MAU special license",
        ),
        # Commercial use of the WEIGHTS is permitted (with attribution), so a
        # Llama *base checkpoint* is ingest-allowed; barred only as a TEACHER
        # (its outputs force the Llama naming flow-down onto the corpus).
        commercial_use=True,
    ),
    # ---- Gemma (conditional / investigate) --------------------------------
    "gemma": LicenseRecord(
        "gemma", "LicenseRef-Gemma",
        "https://ai.google.dev/gemma/terms",
        "conditional",
        obligations=(
            "Gemma Prohibited Use Policy flow-down",
            "Internal Output-vs-Model-Derivative interpretation contradiction",
        ),
    ),
    # ---- Anthropic / Claude (BARRED — ToS finding, memo §B) ---------------
    "anthropic": LicenseRecord(
        "anthropic", "LicenseRef-Anthropic-Proprietary",
        "https://www.anthropic.com/legal/commercial-terms",
        "barred",
        obligations=(
            "Commercial Terms §D.4 competing-model bar + Usage Policy "
            "distillation clause; PROHIBITED as training data",
        ),
        commercial_use=False,
    ),
}


# Coarse ``Touch.provider`` provenance value → default export verdict. These
# are the closed provenance values stamped on every synthesized pair's
# ``provider`` field (``schemas`` closed set:
# {anthropic, claude_session, deterministic, local, nvidia, together}) plus
# the ``mock`` template sentinel. Hosted ``nvidia`` API + ``anthropic`` /
# ``claude_session`` are license-restricted for training data (mirrors
# ``MCP/core/workflow_runner.py::_LICENSE_RESTRICTED_SYNTHESIS`` == {anthropic,
# nvidia}); the local self-hosted Nemotron seats carry ``local`` provenance,
# NOT ``nvidia``, so they read clean.
PROVENANCE_VERDICTS: Dict[str, LicenseVerdict] = {
    "local": "safe",
    "together": "conditional",
    "deterministic": "safe",
    "mock": "safe",
    "nvidia": "barred",          # hosted NVIDIA API seat — restricted
    "anthropic": "barred",       # claude-tagged
    "claude_session": "barred",  # claude-tagged
}

# Provenance values that are claude/anthropic-tagged (memo §B / D3). Export
# ALWAYS fail-closes on these regardless of any additive license field.
BARRED_PROVENANCE = frozenset({"anthropic", "claude_session"})


# --------------------------------------------------------------------------- #
# Model-id → roster resolution                                                 #
# --------------------------------------------------------------------------- #
#
# Ordered, most-specific-first. Each rule is (compiled-regex, roster-key). The
# id is lower-cased + ``:`` / ``_`` normalized to ``-`` before matching so
# Ollama tags (``qwen2.5:32b-instruct``), HF repos (``Qwen/Qwen2.5-32B``), and
# bare short-names all resolve.
_MODEL_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    # Anthropic / Claude first — claude-tagged is fail-closed everywhere.
    (re.compile(r"claude|anthropic|sonnet|opus|haiku"), "anthropic"),
    # Qwen size tiers — order matters (72b / 32b / 14b / 7b / 3b / 1.5b).
    (re.compile(r"qwen.*\b72b\b|qwen.*-72b"), "qwen2.5-72b"),
    (re.compile(r"qwen.*\b32b\b|qwen.*-32b"), "qwen2.5-32b"),
    (re.compile(r"qwen.*\b14b\b|qwen.*-14b"), "qwen2.5-14b"),
    (re.compile(r"qwen.*\b7b\b|qwen.*-7b"), "qwen2.5-7b"),
    (re.compile(r"qwen.*\b3b\b|qwen.*-3b"), "qwen2.5-3b"),
    (re.compile(r"qwen.*1\.5b|qwen.*-1-5b"), "qwen2.5-1.5b"),
    # Nemotron.
    (re.compile(r"nemotron"), NEMOTRON_ROSTER_KEY),
    # Mistral — flagships (MRL) before Apache tiers.
    (re.compile(r"mistral-(large|medium)|pixtral|ministral|codestral"), "mistral-mrl"),
    (re.compile(r"mistral|mixtral"), "mistral-apache"),
    # Phi / GLM / OLMo / DeepSeek / SmolLM2 / Gemma / Llama.
    (re.compile(r"\bphi\b|phi-?[0-9]"), "phi"),
    (re.compile(r"glm"), "glm"),
    (re.compile(r"olmo"), "olmo"),
    (re.compile(r"deepseek"), "deepseek"),
    (re.compile(r"smollm"), "smollm2"),
    (re.compile(r"gemma"), "gemma"),
    (re.compile(r"llama"), "llama"),
)


def _normalize_model_id(model_id: str) -> str:
    return re.sub(r"[:/_]", "-", str(model_id or "").strip().lower())


def license_for_model(model_id: Optional[str]) -> Optional[LicenseRecord]:
    """Resolve a model id / HF repo / Ollama tag to its :class:`LicenseRecord`.

    Returns ``None`` when the id carries no recognized model family (an
    *unregistered* teacher — the export filter fail-closes on that).
    """
    if not model_id:
        return None
    norm = _normalize_model_id(model_id)
    if not norm:
        return None
    for pattern, key in _MODEL_RULES:
        if pattern.search(norm):
            return TEACHER_ROSTER[key]
    return None


# --------------------------------------------------------------------------- #
# Per-pair provenance helper (S6)                                              #
# --------------------------------------------------------------------------- #


def provenance_license_tag(
    *,
    generating_seat: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Return a compact ``name/spdx/verdict`` license tag for a pair.

    Resolution: the fine-grained ``generating_seat`` model id wins (roster
    match); else the coarse ``provider`` provenance value maps to its verdict
    (``<provider>/provenance/<verdict>``). ``unregistered`` when a
    ``generating_seat`` is present but unrecognized; ``unknown`` when neither
    signal is present.
    """
    rec = license_for_model(generating_seat)
    if rec is not None:
        return rec.as_tag()
    if generating_seat:
        return "unregistered"
    if provider:
        verdict = PROVENANCE_VERDICTS.get(str(provider), "unregistered")
        return f"{provider}/provenance/{verdict}"
    return "unknown"


def stamp_pair_license(
    pair: Dict[str, Any],
    *,
    generating_seat: str,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Stamp ``generating_seat`` + ``license`` provenance onto a pair (in place).

    Additive only (both schemas set ``additionalProperties: true``), so the
    coarse closed-enum ``provider`` field is untouched. ``pair`` is returned
    for chaining.
    """
    pair["generating_seat"] = str(generating_seat)
    pair["license"] = provenance_license_tag(
        generating_seat=generating_seat, provider=provider
    )
    return pair


# --------------------------------------------------------------------------- #
# Export-time fail-closed filter (S6)                                          #
# --------------------------------------------------------------------------- #


def classify_pair_teacher(pair: Dict[str, Any]) -> Tuple[str, str, str]:
    """Classify one pair's teacher → ``(status, verdict, teacher_label)``.

    ``status`` is ``"ok"`` | ``"barred"`` | ``"unregistered"`` | ``"claude"``.
    A pair carrying NO teacher signal at all (legacy shape) is ``"ok"`` with
    verdict ``"legacy"`` — the filter never invents a violation from missing
    metadata (registry-defaults-byte-identical contract).
    """
    generating_seat = pair.get("generating_seat") or pair.get("model")
    provider = pair.get("provider")

    # 1. Fine-grained generating_seat model id (most authoritative).
    if generating_seat:
        rec = license_for_model(generating_seat)
        if rec is None:
            return ("unregistered", "unregistered", str(generating_seat))
        if rec.name == "anthropic":
            return ("claude", rec.verdict, rec.name)
        if rec.verdict == "barred":
            return ("barred", rec.verdict, rec.name)
        return ("ok", rec.verdict, rec.name)

    # 2. Coarse provider provenance value.
    if provider:
        prov = str(provider)
        if prov in BARRED_PROVENANCE:
            return ("claude", "barred", prov)
        verdict = PROVENANCE_VERDICTS.get(prov)
        if verdict is None:
            return ("unregistered", "unregistered", prov)
        if verdict == "barred":
            return ("barred", "barred", prov)
        return ("ok", verdict, prov)

    # 3. No teacher signal → legacy pair, pass.
    return ("ok", "legacy", "<none>")


def assert_export_licenses(
    pairs: Iterable[Dict[str, Any]],
    *,
    source_desc: str = "corpus",
) -> None:
    """Fail closed if any pair's teacher is barred / unregistered / claude.

    Raises :class:`LicenseGuardError` naming the FIRST offending pair
    (0-based index + a short identifier) and its teacher. A clean corpus
    (all pairs ``ok`` — including the legacy no-teacher shape and the
    license-clean ``local`` / ``together`` provenances) returns ``None``.
    """
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        status, verdict, teacher = classify_pair_teacher(pair)
        if status == "ok":
            continue
        ident = (
            pair.get("id")
            or pair.get("chunk_id")
            or pair.get("template_id")
            or (str(pair.get("prompt", ""))[:60] + "…")
        )
        if status == "claude":
            reason = (
                f"teacher {teacher!r} is Claude/Anthropic-tagged — PROHIBITED "
                f"as training data (docs/LICENSING.md § Anthropic)"
            )
        elif status == "barred":
            reason = (
                f"teacher {teacher!r} is BARRED (verdict={verdict!r}) — see "
                f"the teacher roster in docs/LICENSING.md"
            )
        else:  # unregistered
            reason = (
                f"teacher {teacher!r} is UNREGISTERED (no roster entry) — add "
                f"a roster row or fix the pair's provenance"
            )
        raise LicenseGuardError(
            f"refusing to export {source_desc}: pair #{idx} "
            f"(id={ident!r}) {reason}."
        )


# --------------------------------------------------------------------------- #
# Per-checkpoint LICENSE assertion at ingest (S6)                              #
# --------------------------------------------------------------------------- #


def assert_checkpoint_license(
    model_identity: Optional[str],
    *,
    role: str = "base_model",
) -> LicenseRecord:
    """Assert a checkpoint's LICENSE at model ingest; return its record.

    ``role="base_model"`` (default): resolve the base-weight license and bar
    only *non-commercial* licenses (Qwen-Research, Mistral-MRL, Anthropic) —
    a Llama community base is ingest-allowed with its naming obligations.
    ``role="teacher"``: bar any ``barred`` verdict.

    Raises :class:`LicenseGuardError` on an unregistered id or a
    role-disallowed license (verify the actual repo LICENSE, not family
    reputation — Qwen 3B/72B ≠ 7-32B; GLM-4-9B ≠ 4.5; DeepSeek distills
    inherit their base).
    """
    rec = license_for_model(model_identity)
    if rec is None:
        raise LicenseGuardError(
            f"checkpoint {model_identity!r} is UNREGISTERED in the teacher "
            f"roster — cannot assert its LICENSE at ingest; add a roster row."
        )
    if role == "teacher":
        if rec.verdict == "barred":
            raise LicenseGuardError(
                f"checkpoint {model_identity!r} ({rec.name}, {rec.license_spdx}) "
                f"is BARRED as a teacher — refusing ingest."
            )
    else:  # base_model
        if not rec.commercial_use:
            raise LicenseGuardError(
                f"base checkpoint {model_identity!r} ({rec.name}, "
                f"{rec.license_spdx}) is non-commercial — refusing ingest for "
                f"a commercially-shipped adapter."
            )
    return rec


# --------------------------------------------------------------------------- #
# Nemotron license-pin / FAIL-BUILD-ON-RE-PIN guard (S7)                       #
# --------------------------------------------------------------------------- #


def provider_verdict_roster() -> Dict[str, Dict[str, Any]]:
    """Return an endpoint-registry-shaped ``{provider: {...}}`` license map.

    S6→A1 contract: the assessment apply-arm license guard
    (``Trainforge/generators/assessment_generator.py::
    _apply_arm_provider_allowed``) consumes a roster dict keyed by the
    coarse provider/provenance name with a ``license_verdict`` (or
    ``verdict``) field, and refuses any seat whose verdict is ``barred``.
    This builds that map from :data:`PROVENANCE_VERDICTS` so the apply-arm
    (and any other coarse-provider consumer) shares one source of truth.
    """
    return {
        name: {"license_verdict": verdict, "verdict": verdict}
        for name, verdict in PROVENANCE_VERDICTS.items()
    }


def assert_nemotron_pin(registry_license_identity: Optional[str] = None) -> None:
    """Fail the build if the Nemotron license identity drifts off the pin.

    A small self-consistency check comparing the pinned license-identity
    string against the roster entry (the machine-readable registry of
    record). When ``registry_license_identity`` is supplied it is compared
    instead (e.g. a value resolved from ``config/endpoints.yaml`` at the call
    site); when ``None`` the roster's own Nemotron record is checked.

    Raises :class:`LicenseGuardError` when the identity does NOT equal the
    pinned :data:`NEMOTRON_PINNED_LICENSE_IDENTITY` — in particular a re-pin
    to the general :data:`GENERAL_NVIDIA_OML_IDENTITY` (which carries the
    Trustworthy-AI restriction, guardrail auto-termination, and a REVOCABLE
    grant) is caught here. Wired into the training preflight.
    """
    identity = registry_license_identity
    if identity is None:
        rec = TEACHER_ROSTER.get(NEMOTRON_ROSTER_KEY)
        identity = rec.identity if rec is not None else None

    if identity != NEMOTRON_PINNED_LICENSE_IDENTITY:
        raise LicenseGuardError(
            "Nemotron license pin violation: expected "
            f"{NEMOTRON_PINNED_LICENSE_IDENTITY!r} but got {identity!r}. "
            "A re-pin to the general "
            f"{GENERAL_NVIDIA_OML_IDENTITY!r} (Trustworthy-AI restriction, "
            "guardrail auto-termination, REVOCABLE grant) fails the build — "
            "re-verify the signed license PDF + procurement sign-off before "
            "updating the pin (docs/LICENSING.md § NVIDIA Nemotron)."
        )
