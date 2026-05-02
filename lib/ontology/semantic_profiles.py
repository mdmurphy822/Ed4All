"""Wave 137 followup — semantic profile loader.

A semantic profile is a per-CURIE acceptance contract layered on top of
the Wave 136b / 137a structural validators. Structural rules catch
placeholders, suffix templates, anchor-verb capacity, diversity, and
provenance — but they do NOT catch semantic confusion (e.g. the auto-
redraft loop on rdf:type passed structural rules while drifting into
rdfs:domain / rdfs:range usage).

A profile names a target CURIE plus a contract:

  - bad_signals:  substrings that must NOT appear (zero tolerance)
  - good_signals: substrings of which at least ``min_good_signals``
                  must appear across defs + usage combined
  - required_in_definitions:    substrings that must appear in defs
  - required_in_usage_examples: substrings that must appear in usage
  - prompt_directive: text injected into the drafting CLI prompt

Profiles live at ``schemas/training/semantic_profiles.yaml`` and are
schema-validated against ``schemas/training/semantic_profiles.schema.json``
on load. Look up a profile by name (e.g. ``rdf_type_instanceof``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_PATH = (
    PROJECT_ROOT / "schemas" / "training" / "semantic_profiles.yaml"
)
DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "training" / "semantic_profiles.schema.json"
)


@dataclass(frozen=True)
class SemanticProfile:
    name: str
    target_curie: str
    bad_signals: Tuple[str, ...]
    good_signals: Tuple[str, ...]
    min_good_signals: int
    required_in_definitions: Tuple[str, ...] = ()
    required_in_usage_examples: Tuple[str, ...] = ()
    prompt_directive: str = ""
    description: str = ""


def _validate_against_schema(payload: dict, schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:  # pragma: no cover
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)


def load_semantic_profiles(
    path: Optional[Path] = None,
    *,
    schema_path: Optional[Path] = None,
) -> Dict[str, SemanticProfile]:
    """Load every profile from the YAML file. Returns a dict
    keyed by profile name (e.g. ``"rdf_type_instanceof"``)."""
    profiles_path = path or DEFAULT_PROFILES_PATH
    schema = schema_path or DEFAULT_SCHEMA_PATH

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"semantic_profiles.yaml not found at {profiles_path}"
        )

    payload = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    _validate_against_schema(payload, schema)

    out: Dict[str, SemanticProfile] = {}
    for name, body in (payload.get("profiles") or {}).items():
        out[name] = SemanticProfile(
            name=name,
            target_curie=body["target_curie"],
            bad_signals=tuple(body.get("bad_signals") or ()),
            good_signals=tuple(body.get("good_signals") or ()),
            min_good_signals=int(body["min_good_signals"]),
            required_in_definitions=tuple(
                body.get("required_in_definitions") or ()
            ),
            required_in_usage_examples=tuple(
                body.get("required_in_usage_examples") or ()
            ),
            prompt_directive=body.get("prompt_directive") or "",
            description=body.get("description") or "",
        )
    return out


def load_semantic_profile(
    name: str,
    *,
    path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> SemanticProfile:
    """Load a single profile by name. Raises ``KeyError`` if absent."""
    profiles = load_semantic_profiles(path=path, schema_path=schema_path)
    if name not in profiles:
        raise KeyError(
            f"semantic profile {name!r} not declared in "
            f"{path or DEFAULT_PROFILES_PATH}; known profiles: "
            f"{sorted(profiles.keys())}"
        )
    return profiles[name]


def evaluate_profile(
    profile: SemanticProfile,
    *,
    definitions: List[str],
    usage_examples: List[Tuple[str, str]],
) -> List[Dict[str, str]]:
    """Run profile rules against an entry. Returns a list of violation
    dicts; empty when the entry passes. Each violation has shape
    ``{"curie", "code", "detail"}`` mirroring the existing
    content-violation shape so the backfill loop's cumulative-feedback
    helper can dedupe / forward them uniformly."""
    violations: List[Dict[str, str]] = []
    target = profile.target_curie

    defs_blob = " ||| ".join(definitions or [])
    usage_answers_blob = " ||| ".join(
        a for _, a in (usage_examples or [])
    )
    full_blob = defs_blob + " ||| " + usage_answers_blob

    for bad in profile.bad_signals:
        if bad.lower() in full_blob.lower():
            violations.append({
                "curie": target,
                "code": "SEMANTIC_BAD_SIGNAL",
                "detail": (
                    f"profile={profile.name} bad_signal={bad!r} matched in "
                    f"defs/usage — the entry expressed a forbidden semantic"
                ),
            })

    matched_good = 0
    matched_signals: List[str] = []
    for good in profile.good_signals:
        if good.lower() in full_blob.lower():
            matched_good += 1
            matched_signals.append(good)
    if matched_good < profile.min_good_signals:
        violations.append({
            "curie": target,
            "code": "SEMANTIC_INSUFFICIENT_GOOD_SIGNALS",
            "detail": (
                f"profile={profile.name} matched {matched_good} of "
                f"required {profile.min_good_signals} good_signals "
                f"(matched={matched_signals!r}); the entry needs more "
                f"semantic-anchor language from "
                f"good_signals={list(profile.good_signals)!r}"
            ),
        })

    for req in profile.required_in_definitions:
        if req.lower() not in defs_blob.lower():
            violations.append({
                "curie": target,
                "code": "SEMANTIC_MISSING_REQUIRED_TERM_DEFINITIONS",
                "detail": (
                    f"profile={profile.name} required term {req!r} not "
                    f"found in any definition — at least one definition "
                    f"must contain this term"
                ),
            })

    for req in profile.required_in_usage_examples:
        if req.lower() not in usage_answers_blob.lower():
            violations.append({
                "curie": target,
                "code": "SEMANTIC_MISSING_REQUIRED_TERM_USAGE",
                "detail": (
                    f"profile={profile.name} required term {req!r} not "
                    f"found in any usage_examples answer-side — at "
                    f"least one usage example must contain this term"
                ),
            })

    return violations
