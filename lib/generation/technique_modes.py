"""C0..C5 generation-technique modes (W5 §2).

One provider codebase, six env states. A :class:`TechniqueMode` is a *named
config* that turns the 7B generation-quality curve into a measurable thing — each
mode adds one cumulative lever to the previous, from a naive whole-corpus
single-sample baseline (``C0``) up to chunk-scoped best-of-N with the NLI gate as
the selector (``C5``, the keystone).

The modes are resolved from ``ED4ALL_GENERATION_TECHNIQUE`` (default ``C5`` for
ship-quality runs; set ``C1`` for fast dev). :func:`apply_mode_to_env` writes the
router/env knobs (``COURSEFORGE_OUTLINE_N_CANDIDATES`` /
``COURSEFORGE_BEST_OF_N`` / the grammar-mode + self-verify + refine toggles) so
the SAME provider + router code runs every mode — the curve runner (§6) just
flips the env per arm and re-synthesizes (or re-selects).

Boundary: this module owns ONLY the mode resolution + the env projection. It does
NOT author the synthesis algorithm (W2), the manifest (W3), or the NLI floors
(W4); the router's best-of-N selector seam (``Courseforge/router/router.py``)
reads ``select_by`` to decide whether to flip to ``entailment_argmax``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional

#: Env var selecting the active technique mode. Default ``C5`` (ship); ``C1`` for
#: fast dev. Read by :func:`resolve_technique_mode` when ``name is None``.
ENV_GENERATION_TECHNIQUE = "ED4ALL_GENERATION_TECHNIQUE"

#: Default mode when the env var is unset/blank/unknown. ``C5`` is the
#: ship-quality target (best-of-N + NLI verifier).
DEFAULT_MODE = "C5"

#: The cumulative C-stack mode names, lowest→highest lever.
MODE_NAMES = ("C0", "C1", "C2", "C3", "C4", "C5")

#: Default best-of-N count for ``C5`` (the keystone). Lifted to 5 for the
#: selection arm (spec §3.2 / §2 table). Overridable per-run.
DEFAULT_BEST_OF_N = 5

#: ``select_by`` literals. ``gate_pass`` is the legacy first-passing-candidate
#: selector (C0..C4); ``entailment_argmax`` flips the router seam to NLI-rate
#: argmax among PASSING candidates (C5).
SELECT_GATE_PASS = "gate_pass"
SELECT_ENTAILMENT_ARGMAX = "entailment_argmax"

# Router/Courseforge env knobs written by :func:`apply_mode_to_env`. Names kept
# as literals here (not imported from the router) so this module has no import
# dependency on Courseforge — the curve runner can import it standalone.
_ENV_OUTLINE_N_CANDIDATES = "COURSEFORGE_OUTLINE_N_CANDIDATES"
_ENV_REWRITE_N_CANDIDATES = "COURSEFORGE_REWRITE_N_CANDIDATES"
_ENV_BEST_OF_N = "COURSEFORGE_BEST_OF_N"
_ENV_OUTLINE_GRAMMAR_MODE = "COURSEFORGE_OUTLINE_GRAMMAR_MODE"
_ENV_SELF_VERIFY = "COURSEFORGE_SELF_VERIFY"
_ENV_REFINE_ROUNDS = "COURSEFORGE_REFINE_ROUNDS"
_ENV_CHUNK_SCOPED = "COURSEFORGE_CHUNK_SCOPED"
_ENV_SELECT_BY = "COURSEFORGE_BEST_OF_N_SELECT_BY"


@dataclass(frozen=True)
class TechniqueMode:
    """A resolved generation-technique configuration (one of C0..C5).

    Cumulative: each higher mode keeps every lower mode's levers and adds one.
    The fields map directly onto router/env knobs via :func:`apply_mode_to_env`.
    """

    name: str  # "C0".."C5"
    chunk_scoped: bool  # C1+: one chunk-set/block in-context
    free_text_envelope: bool  # C2+: the json_mode-trap fix
    self_verify: bool  # C3+: model checks its own block vs chunk
    refine_rounds: int  # C4: 2 bounded critique rounds, else 0
    best_of_n: int  # C5: N (default 5), else 1
    select_by: str  # SELECT_GATE_PASS (C0..C4) | SELECT_ENTAILMENT_ARGMAX (C5)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "chunk_scoped": self.chunk_scoped,
            "free_text_envelope": self.free_text_envelope,
            "self_verify": self.self_verify,
            "refine_rounds": self.refine_rounds,
            "best_of_n": self.best_of_n,
            "select_by": self.select_by,
        }


def _build_mode(name: str, *, best_of_n: int) -> TechniqueMode:
    """Construct the cumulative mode definition for ``name``.

    Cumulative gating reads off the ordinal index in :data:`MODE_NAMES`:
    ``chunk_scoped`` at C1+, ``free_text_envelope`` at C2+, ``self_verify`` at
    C3+, ``refine_rounds=2`` at C4+, ``best_of_n`` + ``entailment_argmax`` at C5.
    """
    idx = MODE_NAMES.index(name)
    return TechniqueMode(
        name=name,
        chunk_scoped=idx >= 1,
        free_text_envelope=idx >= 2,
        self_verify=idx >= 3,
        refine_rounds=2 if idx >= 4 else 0,
        best_of_n=best_of_n if idx >= 5 else 1,
        select_by=SELECT_ENTAILMENT_ARGMAX if idx >= 5 else SELECT_GATE_PASS,
    )


def resolve_technique_mode(
    name: Optional[str] = None, *, best_of_n: Optional[int] = None
) -> TechniqueMode:
    """Resolve a :class:`TechniqueMode` by name (or from the env var).

    Resolution: explicit ``name`` arg > ``ED4ALL_GENERATION_TECHNIQUE`` env >
    :data:`DEFAULT_MODE`. An unknown/garbage value falls back to the default
    (never raises — a misconfigured env should ship-quality-degrade, not crash a
    run). ``best_of_n`` overrides the C5 candidate count (else
    :data:`DEFAULT_BEST_OF_N`).
    """
    raw = name if name is not None else os.environ.get(ENV_GENERATION_TECHNIQUE)
    candidate = str(raw or "").strip().upper()
    if candidate not in MODE_NAMES:
        candidate = DEFAULT_MODE
    n = best_of_n if (isinstance(best_of_n, int) and best_of_n > 0) else DEFAULT_BEST_OF_N
    return _build_mode(candidate, best_of_n=n)


def apply_mode_to_env(mode: TechniqueMode, env: MutableMapping[str, str]) -> None:
    """Project ``mode`` onto the router/Courseforge env knobs in ``env``.

    Writes the knobs so the same provider + router code runs every mode:

    * ``COURSEFORGE_BEST_OF_N`` — the umbrella N (``mode.best_of_n``); the router
      fills the per-tier outline/rewrite N from this unless a per-tier knob is
      already set (per-tier wins — we use ``setdefault`` for the rewrite tier).
    * ``COURSEFORGE_OUTLINE_N_CANDIDATES`` — set to ``best_of_n`` so the
      self-consistency loop samples N candidates (only meaningful at C5; C0..C4
      resolve to 1).
    * ``COURSEFORGE_BEST_OF_N_SELECT_BY`` — ``gate_pass`` (C0..C4) or
      ``entailment_argmax`` (C5). The router reads this to decide whether to flip
      the winner selector to NLI-rate argmax.
    * ``COURSEFORGE_OUTLINE_GRAMMAR_MODE`` — ``json_object`` at C0/C1 (the
      json-everything baseline); ``none`` at C2+ (the free-text/thin-JSON
      envelope fix — prose is no longer forced into JSON fields).
    * ``COURSEFORGE_SELF_VERIFY`` / ``COURSEFORGE_REFINE_ROUNDS`` /
      ``COURSEFORGE_CHUNK_SCOPED`` — the C3 / C4 / C1 toggles.

    Mutates ``env`` in place (pass ``os.environ`` for a live run, or a fresh dict
    for the curve runner's per-arm isolation).
    """
    env[_ENV_CHUNK_SCOPED] = "true" if mode.chunk_scoped else "false"
    env[_ENV_OUTLINE_GRAMMAR_MODE] = "none" if mode.free_text_envelope else "json_object"
    env[_ENV_SELF_VERIFY] = "true" if mode.self_verify else "false"
    env[_ENV_REFINE_ROUNDS] = str(mode.refine_rounds)
    env[_ENV_BEST_OF_N] = str(mode.best_of_n)
    env[_ENV_OUTLINE_N_CANDIDATES] = str(mode.best_of_n)
    env[_ENV_SELECT_BY] = mode.select_by
    # Per-tier rewrite N: only fill when not already operator-pinned (per-tier
    # wins over the umbrella per spec §3.2).
    env.setdefault(_ENV_REWRITE_N_CANDIDATES, str(mode.best_of_n))


def resolve_select_by(env: Optional[Mapping[str, str]] = None) -> str:
    """Read the active selector from the env (router-side helper).

    Returns :data:`SELECT_ENTAILMENT_ARGMAX` only when
    ``COURSEFORGE_BEST_OF_N_SELECT_BY`` is explicitly that value; any
    unset/other value yields :data:`SELECT_GATE_PASS` (the legacy
    first-passing-candidate selector). This is the seam the router's best-of-N
    loop consults — keeping the default ``gate_pass`` makes the flip strictly
    opt-in (additive: an un-projected env runs the legacy selector unchanged).
    """
    source = env if env is not None else os.environ
    value = str(source.get(_ENV_SELECT_BY, "") or "").strip()
    return (
        SELECT_ENTAILMENT_ARGMAX
        if value == SELECT_ENTAILMENT_ARGMAX
        else SELECT_GATE_PASS
    )


#: Truthy tokens shared by the C1/C3/C4 lever resolvers below. Mirrors the
#: case-insensitive match used elsewhere (``blocks._EMIT_BLOCKS_TRUTHY`` /
#: ``resolve_select_by``): only these explicit tokens enable a lever; anything
#: else (unset / blank / ``false`` / garbage) resolves OFF (parse-with-fallback).
_LEVER_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_chunk_scoped(env: Optional[Mapping[str, str]] = None) -> bool:
    """Read the C1 chunk-scoping lever from the env (router-side helper).

    Returns ``True`` only when ``COURSEFORGE_CHUNK_SCOPED`` is one of the
    canonical truthy tokens (projected by the C1+ technique modes via
    :func:`apply_mode_to_env`). Any unset / blank / ``false`` / garbage value
    yields ``False`` so the consumer runs its legacy whole-input path unchanged
    (additive: an un-projected env is byte-identical). parse-with-fallback.
    """
    source = env if env is not None else os.environ
    return str(source.get(_ENV_CHUNK_SCOPED, "") or "").strip().lower() in _LEVER_TRUTHY


def resolve_self_verify(env: Optional[Mapping[str, str]] = None) -> bool:
    """Read the C3 self-verify lever from the env (router-side helper).

    Returns ``True`` only when ``COURSEFORGE_SELF_VERIFY`` is a canonical truthy
    token (projected by the C3+ technique modes). Unset / blank / ``false`` /
    garbage → ``False`` (the consumer skips the self-verify micro-pass — legacy
    behaviour, byte-identical). parse-with-fallback.
    """
    source = env if env is not None else os.environ
    return str(source.get(_ENV_SELF_VERIFY, "") or "").strip().lower() in _LEVER_TRUTHY


def resolve_refine_rounds(env: Optional[Mapping[str, str]] = None) -> int:
    """Read the C4 bounded-refine-rounds lever from the env (router-side helper).

    Returns the parsed positive int from ``COURSEFORGE_REFINE_ROUNDS`` (projected
    as ``2`` by the C4+ technique modes) or ``0`` when unset / ``0`` / negative /
    non-int / garbage — so the consumer runs zero refine rounds by default (legacy
    behaviour, byte-identical). parse-with-fallback (mirrors ``resolve_best_of_n``).
    """
    source = env if env is not None else os.environ
    raw = source.get(_ENV_REFINE_ROUNDS)
    if raw is None:
        return 0
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def resolve_best_of_n(
    env: Optional[Mapping[str, str]] = None, *, tier: str = "outline"
) -> Optional[int]:
    """Resolve the best-of-N umbrella → per-tier N (router-side helper).

    Precedence (per spec §3.2): the per-tier knob
    (``COURSEFORGE_OUTLINE_N_CANDIDATES`` / ``COURSEFORGE_REWRITE_N_CANDIDATES``)
    wins; otherwise the umbrella ``COURSEFORGE_BEST_OF_N`` fills both tiers.
    Returns ``None`` when neither is set (the router then falls through to its own
    ``_resolve_n_candidates`` chain — this helper never overrides that chain when
    nothing best-of-N-specific is configured). A value of ``1`` means OFF.
    """
    source = env if env is not None else os.environ
    per_tier_key = (
        _ENV_REWRITE_N_CANDIDATES if tier == "rewrite" else _ENV_OUTLINE_N_CANDIDATES
    )
    for key in (per_tier_key, _ENV_BEST_OF_N):
        raw = source.get(key)
        if raw:
            try:
                n = int(str(raw).strip())
            except (TypeError, ValueError):
                continue
            if n > 0:
                return n
    return None


__all__ = [
    "TechniqueMode",
    "resolve_technique_mode",
    "apply_mode_to_env",
    "resolve_select_by",
    "resolve_best_of_n",
    "resolve_chunk_scoped",
    "resolve_self_verify",
    "resolve_refine_rounds",
    "ENV_GENERATION_TECHNIQUE",
    "DEFAULT_MODE",
    "MODE_NAMES",
    "DEFAULT_BEST_OF_N",
    "SELECT_GATE_PASS",
    "SELECT_ENTAILMENT_ARGMAX",
]
