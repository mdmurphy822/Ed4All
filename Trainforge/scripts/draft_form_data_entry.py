"""Wave 136c — Qwen-driven drafting CLI for the FORM_DATA backfill flow.

Operator-facing CLI that drafts a single ``forms.<CURIE>:`` block for
the schema-translation YAML overlay
(``schemas/training/schema_translation_catalog.<family>.yaml``). The
operator reviews the rendered YAML, then appends it manually — there
is no auto-apply path here. Wave 136d adds the interactive backfill
loop that wraps this CLI.

ToS posture (load-bearing): the drafting prompt template is
structurally minimal — schema rules + the operator-authored manifest
metadata (``label``, ``surface_forms``) only. NO example sentences,
NO sample content, NO seed phrases. The provider produces the
content; this CLI never authors training-data text. The drafting
prompt template is asserted free of quoted ≥40-char strings via the
test_drafting_prompt_template_does_not_contain_example_content
regression test. Re-routes the standing "Claude does not generate
training-data corpus content" operating principle through the
drafting surface: Claude (or any dev tool) emits the CLI scaffolding
but never the per-CURIE definitions / usage examples.

Usage::

    python -m Trainforge.scripts.draft_form_data_entry \\
        --curie sh:minCount \\
        --course-code rdf-shacl-551-2 \\
        --provider local

Exit codes:
    0  success — YAML block printed (or "already populated" message
       when the entry is complete and ``--force-overwrite`` is unset).
    2  unknown CURIE: not declared in the property manifest.
    3  validator failure: ``validate_form_data_contract`` rejected
       the drafted entry.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.property_manifest import (  # noqa: E402
    PropertyEntry,
    load_property_manifest,
)
from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    Provenance,
    SurfaceFormData,
    _load_form_data,
    validate_form_data_contract,
)
from Trainforge.scripts._review_checklist import build_review_checklist  # noqa: E402

logger = logging.getLogger(__name__)


# Wave 137c: prompt-template version captured in every drafted entry's
# provenance block. Bump in lockstep with material edits to
# ``_DRAFTING_PROMPT_TEMPLATE``.
_PROMPT_VERSION = "wave-136c-v1.0"


def _resolve_provider_id(name: str, model: Optional[str]) -> str:
    """Wave 137c: canonical provider identifier for the audit trail.

    Stable identifiers across runs let the ToS audit table aggregate
    by source. The two canonical defaults — Qwen 14B Q4 (Ollama) and
    Llama 3.3 70B (Together) — get fixed strings; everything else
    falls back to a lowercased, slugified ``{name}_{model}`` form so
    a less-common server / model still produces a deterministic ID.
    """

    def _slugify(value: Optional[str]) -> str:
        if not value:
            return "unspecified"
        return value.lower().replace(":", "_").replace("/", "_")

    model_lower = (model or "").lower()
    if name == "local":
        if "qwen" in model_lower and "14b" in model_lower and "q4" in model_lower:
            return "qwen_local_14b_q4"
        return f"local_{_slugify(model)}"
    if name == "together":
        if "llama" in model_lower and "70b" in model_lower:
            return "together_llama33_70b"
        return f"together_{_slugify(model)}"
    return f"{name}_{_slugify(model)}"


# Drafting prompt template — ToS-load-bearing.
#
# The template MUST be exactly this. NO example outputs, NO sample
# sentences, NO seed phrases. Schema rules + the operator-authored
# manifest metadata (label, surface_forms) only. The
# test_drafting_prompt_template_does_not_contain_example_content
# regression test asserts ZERO quoted strings ≥40 chars are present
# in the rendered template — a structural sentinel against drift in
# this template that would re-route Claude-authored content into the
# drafting prompt.
_DRAFTING_PROMPT_TEMPLATE = """\
You are an editor producing structured catalog data for a knowledge-graph
schema-to-English translation training corpus. Output a single JSON object.
Do not include any prose outside the JSON object.

TARGET CURIE: {curie}
LABEL (operator-authored, from property manifest): {label}
SURFACE FORMS (operator-authored, from property manifest): {surface_forms_csv}

REQUIRED JSON SHAPE (one object with exactly these keys):
- short_name: string, 1-64 chars
- definitions: list of strings, exactly 7 entries
- usage_examples: list of [prompt, answer] pairs, exactly 7
- comparison_targets: list of [other_curie, explanation] pairs, 0-7
- reasoning_scenarios: list of [prompt, answer] pairs, 0-7
- pitfalls: list of [prompt, answer] pairs, 0-7
- combinations: list of [other_curie, explanation] pairs, 0-7

PER-FIELD RULES
- Every definitions entry MUST contain the exact substring {curie}.
- Every definitions entry MUST be 50-400 characters.
- Every usage_examples prompt MUST be 40-400 characters; every answer MUST
  contain the exact substring {curie} and be 50-600 characters.
- Every comparison_targets and combinations answer MUST contain BOTH
  {curie} and the paired secondary CURIE literally.
- Definitions MUST NOT begin with Canonical terms:, Required terms:,
  Reference:, Relevant terms:, or Key vocabulary: prefix banners.
- Definitions MUST NOT contain the substrings [degraded: or
  not yet authored.
- A definitions entry that mentions only a sibling CURIE (e.g. defines
  this CURIE by reference to a different one) without containing {curie}
  is invalid.
- Each definition probes a different cognitive angle (formal-spec /
  pedagogical / context-anchored / operational / contrastive / scenario /
  reasoning) — no thesaurus paraphrases of the same sentence.
- Output JSON only. No commentary, no markdown fences.
"""


def _build_drafting_prompt(entry: PropertyEntry) -> str:
    """Render the drafting prompt for ``entry``.

    Substitutes ``curie``, ``label``, and ``surface_forms_csv`` only.
    No example content is interpolated — that's the ToS contract this
    CLI exists to enforce.
    """
    surface_forms_csv = ", ".join(entry.surface_forms)
    # Note: the template uses doubled curly braces for the JSON shape
    # block (so .format escapes them to literal braces in the output).
    return _DRAFTING_PROMPT_TEMPLATE.format(
        curie=entry.curie,
        label=entry.label,
        surface_forms_csv=surface_forms_csv,
    )


def _draft_one_curie(provider: Any, prompt: str) -> Dict[str, Any]:
    """Issue one chat-completion call and return the parsed JSON dict.

    Routes through the provider's embedded :class:`OpenAICompatibleClient`
    (``provider._oa_client.chat_completion``) — the raw chat path
    without paraphrase plumbing. Wave 136d's loop wraps this for
    multi-CURIE backfill.

    Args:
        provider: A ``LocalSynthesisProvider`` or ``TogetherSynthesisProvider``
            instance with an ``_oa_client`` attribute.
        prompt: The fully-rendered drafting prompt.

    Returns:
        Parsed JSON dict from the provider's response.

    Raises:
        RuntimeError: when the response can't be parsed as JSON. We do
            not retry — operator inspects the raw output and re-runs.
    """
    messages = [{"role": "user", "content": prompt}]
    text = provider._oa_client.chat_completion(
        messages,
        max_tokens=4000,
        temperature=0.4,
        extra_payload={"response_format": {"type": "json_object"}},
        decision_metadata={
            "task": "draft_form_data_entry",
            "phase": "wave_136c_drafting",
        },
    )
    # Try lenient extraction for backends that wrap output in markdown
    # fences or surrounding prose (Wave 113 hardening on 7B-Q4 servers).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return provider._oa_client._extract_json_lenient(text)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "draft_form_data_entry: provider response could not be "
                f"parsed as JSON. First 500 chars of response: "
                f"{text[:500]!r}"
            ) from exc


def _coerce_to_surface_form_data(
    raw: Dict[str, Any],
    curie: str,
    provider_name: str,
    model: Optional[str],
) -> SurfaceFormData:
    """Coerce a provider-emitted JSON dict into a ``SurfaceFormData``.

    Marks the result ``anchored_status="complete"`` — that's the
    intended outcome of a successful drafting pass. The validator then
    confirms the structural contract (Wave 136a) and Wave 136b's
    content-quality rules (when present in the validator on the
    branch this CLI runs against).

    Wave 137c: every drafted entry carries an auto-stamped
    :class:`Provenance` block with ``reviewed_by="PENDING_REVIEW"``.
    Operators MUST replace ``PENDING_REVIEW`` with their handle (e.g.
    ``@mdmurphy822``) before committing; Plan A's validator rejects
    entries whose ``reviewed_by`` is ``PENDING_REVIEW`` or empty.
    """

    def _coerce_pair_list(field: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for pair in raw.get(field) or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                out.append((str(pair[0]), str(pair[1])))
        return out

    short_name = str(
        raw.get("short_name") or curie.split(":")[-1] or curie
    ).strip() or curie
    provenance = Provenance(
        provider=_resolve_provider_id(provider_name, model),
        generated_by="draft_form_data_entry v1.0",
        reviewed_by="PENDING_REVIEW",
        prompt_version=_PROMPT_VERSION,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=None,
    )
    return SurfaceFormData(
        curie=curie,
        short_name=short_name,
        definitions=[str(d) for d in (raw.get("definitions") or [])],
        usage_examples=_coerce_pair_list("usage_examples"),
        comparison_targets=_coerce_pair_list("comparison_targets"),
        reasoning_scenarios=_coerce_pair_list("reasoning_scenarios"),
        pitfalls=_coerce_pair_list("pitfalls"),
        combinations=_coerce_pair_list("combinations"),
        anchored_status="complete",
        provenance=provenance,
    )


def _surface_form_data_to_yaml_dict(entry: SurfaceFormData) -> Dict[str, Any]:
    """Project ``SurfaceFormData`` into a YAML-emit-friendly dict.

    Mirrors the on-disk shape used by
    ``schema_translation_catalog.<family>.yaml``: ``short_name`` +
    ``anchored_status`` + four list-of-strings or list-of-pairs fields.
    Pairs are emitted as 2-element lists so YAML round-trips through
    the existing ``_load_yaml_catalog`` reader unchanged.
    """
    out: Dict[str, Any] = {
        "short_name": entry.short_name,
        "anchored_status": entry.anchored_status,
        "definitions": list(entry.definitions),
        "usage_examples": [list(p) for p in entry.usage_examples],
    }
    if entry.comparison_targets:
        out["comparison_targets"] = [list(p) for p in entry.comparison_targets]
    if entry.reasoning_scenarios:
        out["reasoning_scenarios"] = [list(p) for p in entry.reasoning_scenarios]
    if entry.pitfalls:
        out["pitfalls"] = [list(p) for p in entry.pitfalls]
    if entry.combinations:
        out["combinations"] = [list(p) for p in entry.combinations]
    if entry.provenance is not None:
        out["provenance"] = {
            "provider": entry.provenance.provider,
            "generated_by": entry.provenance.generated_by,
            "reviewed_by": entry.provenance.reviewed_by,
            "prompt_version": entry.provenance.prompt_version,
            "timestamp": entry.provenance.timestamp,
        }
        if entry.provenance.notes is not None:
            out["provenance"]["notes"] = entry.provenance.notes
    return out


def _build_provider(
    provider_name: str,
    model: Optional[str],
    timeout: Optional[float] = None,
) -> Any:
    """Instantiate the configured provider.

    Supports ``local`` (default) and ``together``. Each provider's
    constructor falls back to its env-var defaults when the model
    kwarg is unset, so the CLI can be invoked with no ``--model``
    flag for the standard rdf-shacl Qwen 14B / Llama 3.3 70B layouts.

    ``timeout`` (Wave 137 follow-up): per-HTTP-request timeout in
    seconds. The drafting prompt asks for 35+ structured items so
    Qwen 14B-Q4 routinely exceeds the provider's 60s default; pass
    a higher value (e.g. 300) for high-coupling CURIEs.
    """
    if provider_name == "local":
        from Trainforge.generators._local_provider import (  # noqa: WPS433
            LocalSynthesisProvider,
        )
        kwargs: Dict[str, Any] = {}
        if model:
            kwargs["model"] = model
        if timeout is not None:
            kwargs["timeout"] = timeout
        return LocalSynthesisProvider(**kwargs)
    if provider_name == "together":
        from Trainforge.generators._together_provider import (  # noqa: WPS433
            TogetherSynthesisProvider,
        )
        kwargs2: Dict[str, Any] = {}
        if model:
            kwargs2["model"] = model
        if timeout is not None:
            kwargs2["timeout"] = timeout
        return TogetherSynthesisProvider(**kwargs2)
    raise ValueError(
        f"Unsupported provider: {provider_name!r}. "
        f"Use 'local' or 'together'."
    )


_NEXT_STEPS_TEMPLATE = """\
# NEXT STEPS
# 0. **REVIEW the drafted content + UPDATE provenance.reviewed_by from
#    PENDING_REVIEW to your operator handle (e.g. @mdmurphy822)** before
#    committing. Plan A's validator rejects entries with reviewed_by
#    set to PENDING_REVIEW or empty.
# 1. Append the `forms.{curie}:` block above to:
#    schemas/training/schema_translation_catalog.{family}.yaml
# 2. Run:
#    python -m pytest Trainforge/tests/test_schema_translation_generator.py
# 3. Commit.
"""


def _render_yaml_block(
    family: str,
    curie: str,
    entry: SurfaceFormData,
    *,
    validator_score_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the YAML block + review checklist + next-steps banner.

    The YAML uses ``yaml.safe_dump`` with sort_keys=False (preserve our
    field order), default_flow_style=False (block style — readable),
    allow_unicode=True (CURIE colons + UTF-8 spec quotes round-trip),
    and width=120.

    Wave 137d-1: emits the auto-printed review checklist between the
    YAML block and the operator next-steps banner. The checklist
    structures operator review (always-2 + sample-3) so reviewer
    fatigue stays bounded at backfill scale. The backfill loop's YAML
    slicer cuts at the first of either the checklist header or the
    next-steps header, so the YAML still round-trips cleanly through
    ``yaml.safe_load``.
    """
    import yaml

    payload = {
        "family": family,
        "forms": {curie: _surface_form_data_to_yaml_dict(entry)},
    }
    yaml_text = yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )
    checklist = build_review_checklist(
        curie,
        entry,
        validator_score_summary=validator_score_summary,
    )
    next_steps = _NEXT_STEPS_TEMPLATE.format(family=family, curie=curie)
    return f"{yaml_text}\n{checklist}\n\n{next_steps}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draft_form_data_entry",
        description=(
            "Draft a forms.<CURIE>: block for the schema-translation "
            "catalog using a Qwen / Together provider. Operator review "
            "+ manual append required."
        ),
    )
    parser.add_argument(
        "--curie",
        required=True,
        help="Target CURIE (e.g., 'sh:minCount').",
    )
    parser.add_argument(
        "--family",
        default="rdf_shacl",
        help="Catalog family. Default: rdf_shacl.",
    )
    parser.add_argument(
        "--course-code",
        required=True,
        help="Course slug for property-manifest resolution.",
    )
    parser.add_argument(
        "--provider",
        choices=("local", "together"),
        default="local",
        help="LLM provider. Default: local (Qwen via Ollama).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier override. Falls back to provider default.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output path. '-' (default) writes to stdout.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help=(
            "Bypass the already-complete skip and redraft the entry. "
            "Off by default — protects existing complete entries from "
            "accidental clobber."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Reserved for future use. Current default 0 — explicit no auto-retry.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help=(
            "Per-HTTP-request timeout in seconds. Default 300 (5 min) — "
            "the drafting prompt asks for 35+ structured items so 14B-Q4 "
            "routinely exceeds the provider's standard 60s budget."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Step 1: load + validate the property manifest.
    try:
        manifest = load_property_manifest(args.course_code)
    except FileNotFoundError as exc:
        print(f"ERROR: property manifest not found: {exc}", file=sys.stderr)
        return 2

    declared_curies = [p.curie for p in manifest.properties]
    if args.curie not in declared_curies:
        print(
            f"ERROR: CURIE {args.curie!r} is not declared in the "
            f"property manifest for course {args.course_code!r}. "
            f"Manifest declares {len(declared_curies)} CURIEs; first "
            f"5: {declared_curies[:5]}.",
            file=sys.stderr,
        )
        return 2

    # Step 2: pull the operator-authored entry.
    entry = next(p for p in manifest.properties if p.curie == args.curie)

    # Step 3: complete-skip guard. Fail-soft (exit 0) so operators
    # running this in a tight loop don't get a non-zero exit on
    # already-good entries.
    form_data = _load_form_data(args.family)
    existing = form_data.get(args.curie)
    if (
        existing is not None
        and existing.anchored_status == "complete"
        and not args.force_overwrite
    ):
        print(
            f"CURIE {args.curie!r} is already populated; pass "
            f"--force-overwrite to redraft."
        )
        return 0

    # Step 4: build the drafting prompt.
    prompt = _build_drafting_prompt(entry)

    # Step 5: instantiate the provider.
    try:
        provider = _build_provider(args.provider, args.model, timeout=args.timeout)
    except Exception as exc:
        print(
            f"ERROR: failed to instantiate provider "
            f"{args.provider!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    # Step 6: issue the single chat call.
    try:
        raw = _draft_one_curie(provider, prompt)
    except Exception as exc:
        print(
            f"ERROR: provider call failed: {exc}",
            file=sys.stderr,
        )
        return 1

    # Step 7: coerce into SurfaceFormData (anchored_status="complete").
    # Wave 137c: provenance is auto-stamped with reviewed_by="PENDING_REVIEW";
    # the operator-next-steps banner reminds the operator to replace it
    # with their handle before committing.
    drafted = _coerce_to_surface_form_data(
        raw, args.curie, args.provider, args.model
    )

    # Step 8: validate. Build a one-CURIE form_data dict and run the
    # canonical contract validator. Wave 136b widens this to content-
    # quality rules; this call works against either generation of the
    # validator because the structural floor is unchanged.
    synthetic = {args.curie: drafted}
    report = validate_form_data_contract(synthetic, [args.curie])
    if not report.get("passed"):
        print(
            f"ERROR: validate_form_data_contract rejected the drafted "
            f"entry for CURIE {args.curie!r}:",
            file=sys.stderr,
        )
        # Surface the per-violation table verbatim. Wave 136b's
        # validator extends this with a content_violations list; we
        # render whichever shape is present.
        for key in (
            "missing_curies",
            "incomplete_curies",
            "invalid_status_curies",
            "content_violations",
        ):
            value = report.get(key)
            if value:
                print(f"  {key}: {value}", file=sys.stderr)
        return 3

    # Step 9 + 10: render YAML + operator next-steps and emit.
    rendered = _render_yaml_block(args.family, args.curie, drafted)
    if args.output == "-":
        sys.stdout.write(rendered)
        sys.stdout.flush()
    else:
        out_path = Path(args.output)
        out_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote drafted YAML block to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
