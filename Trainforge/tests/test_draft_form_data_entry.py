"""Wave 136c: tests for the FORM_DATA drafting CLI.

Five tests pin the operator-facing contract:

1. ``test_cli_rejects_unknown_curie`` — the CLI exits 2 when the
   target CURIE isn't declared by the property manifest.
2. ``test_cli_skips_already_complete_without_force`` — the CLI exits
   0 with an "already populated" message when the existing entry is
   complete and ``--force-overwrite`` is unset; provider NOT called.
3. ``test_cli_invokes_validator_and_exits_nonzero_on_violation`` —
   when the provider returns a structurally short definition, the
   validator catches it and the CLI exits 3.
4. ``test_cli_renders_yaml_block_on_success`` — happy-path CLI prints
   a YAML block carrying ``family:`` and ``forms:`` keys plus the
   target CURIE.
5. ``test_drafting_prompt_template_does_not_contain_example_content``
   — ToS regression sentinel: ``r'"[^"]{40,}"'`` matches ZERO times
   in the rendered prompt template, ensuring the template never
   contains Claude-authored example sentences ≥ 40 chars enclosed in
   quotes.

Test fixtures use synthetic ``test:Foo`` CURIEs where structurally
possible. The success-path test uses the real ``rdf-shacl-551-2``
manifest and an existing ``sh:datatype`` entry only because the
manifest must be loadable end-to-end; we ``--force-overwrite`` past
the complete-skip and inject a synthetic provider response.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.scripts import draft_form_data_entry as cli  # noqa: E402
from Trainforge.generators.schema_translation_generator import (  # noqa: E402
    SurfaceFormData,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _FakeProvider:
    """Fake provider exposing the ``_oa_client.chat_completion`` shape.

    Composes a single attribute, ``_oa_client``, whose ``chat_completion``
    method returns whatever JSON string the test wired in. The real
    ``LocalSynthesisProvider`` / ``TogetherSynthesisProvider`` expose
    the same surface; this fake is the test seam.
    """

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: List[Dict[str, Any]] = []
        self._oa_client = self  # _draft_one_curie reads provider._oa_client

    # Mirror ``OpenAICompatibleClient.chat_completion``.
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.4,
        decision_metadata: Optional[Dict[str, Any]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "decision_metadata": dict(decision_metadata or {}),
                "extra_payload": dict(extra_payload or {}),
            }
        )
        return self._response_text


def _build_valid_form_data_payload(curie: str) -> Dict[str, Any]:
    """Build a structurally-valid 7-definition + 7-usage payload.

    Uses ``[TEST FIXTURE: ...]``-marked filler so even though each
    definition is >= 50 chars, the strings are obvious test artefacts
    that no one would mistake for real training-data content. Matches
    Wave 136a's structural floor (>= 1 def + >= 1 usage_example).
    """
    definitions = [
        f"[TEST FIXTURE: definition-{i} for {curie}; padded to clear "
        f"the structural minimum length floor of fifty chars.]"
        for i in range(7)
    ]
    usage_examples = [
        [
            f"[TEST FIXTURE: prompt-{i} for {curie}; padded over forty.]",
            f"[TEST FIXTURE: answer-{i} for {curie}; padded over fifty "
            f"so the structural floor is satisfied.]",
        ]
        for i in range(7)
    ]
    return {
        "short_name": curie.split(":")[-1] if ":" in curie else curie,
        "definitions": definitions,
        "usage_examples": usage_examples,
        "comparison_targets": [],
        "reasoning_scenarios": [],
        "pitfalls": [],
        "combinations": [],
    }


# ----------------------------------------------------------------------
# 1. Unknown CURIE → exit 2
# ----------------------------------------------------------------------


def test_cli_rejects_unknown_curie():
    """Synthetic ``unknown:not_in_manifest`` exits 2 with explicit error."""
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(
            [
                "--curie",
                "unknown:not_in_manifest",
                "--course-code",
                "rdf-shacl-551-2",
                "--provider",
                "local",
            ]
        )
    assert rc == 2, f"expected exit code 2 for unknown CURIE; got {rc}"
    err_text = err.getvalue()
    assert "not declared" in err_text or "manifest" in err_text


# ----------------------------------------------------------------------
# 2. Already-complete + no --force-overwrite → exit 0, no provider call
# ----------------------------------------------------------------------


def test_cli_skips_already_complete_without_force():
    """Mock ``_load_form_data`` to return a complete entry; CLI exits 0."""
    target_curie = "sh:datatype"  # known to exist in the rdf-shacl manifest
    fake_complete_entry = SurfaceFormData(
        curie=target_curie,
        short_name="datatype",
        definitions=[f"[TEST FIXTURE: stub def for {target_curie}.]"],
        usage_examples=[
            (
                f"[TEST FIXTURE: stub prompt for {target_curie}.]",
                f"[TEST FIXTURE: stub answer for {target_curie}.]",
            )
        ],
        anchored_status="complete",
    )
    fake_form_data = {target_curie: fake_complete_entry}

    out = io.StringIO()
    fake_build_provider_calls: List[Any] = []

    def _spy_build_provider(*args, **kwargs):
        fake_build_provider_calls.append((args, kwargs))
        raise AssertionError(
            "provider must NOT be instantiated when entry is already "
            "complete and --force-overwrite is unset"
        )

    with patch.object(cli, "_load_form_data", return_value=fake_form_data), \
        patch.object(cli, "_build_provider", side_effect=_spy_build_provider), \
        redirect_stdout(out):
        rc = cli.main(
            [
                "--curie",
                target_curie,
                "--course-code",
                "rdf-shacl-551-2",
                "--provider",
                "local",
            ]
        )

    assert rc == 0, f"expected exit 0 for already-complete; got {rc}"
    assert "already populated" in out.getvalue()
    assert fake_build_provider_calls == [], (
        "provider was instantiated despite already-complete entry"
    )


# ----------------------------------------------------------------------
# 3. Provider emits structurally-invalid output → validator → exit 3
# ----------------------------------------------------------------------


def test_cli_invokes_validator_and_exits_nonzero_on_violation():
    """Mock provider returning empty definitions list → validator catches it."""
    target_curie = "sh:datatype"
    # Empty definitions list violates the structural floor (>=1 def).
    bad_payload = {
        "short_name": "datatype",
        # Definitions list is EMPTY — Wave 136a contract requires >= 1.
        "definitions": [],
        "usage_examples": [
            [
                "[TEST FIXTURE: short usage prompt over forty chars.]",
                "[TEST FIXTURE: short usage answer over fifty chars padded.]",
            ]
        ],
        "comparison_targets": [],
        "reasoning_scenarios": [],
        "pitfalls": [],
        "combinations": [],
    }
    import json as _json
    fake_provider = _FakeProvider(_json.dumps(bad_payload))

    err = io.StringIO()
    with patch.object(
        cli, "_build_provider", return_value=fake_provider
    ), redirect_stderr(err):
        rc = cli.main(
            [
                "--curie",
                target_curie,
                "--course-code",
                "rdf-shacl-551-2",
                "--provider",
                "local",
                "--force-overwrite",
            ]
        )

    assert rc == 3, f"expected exit 3 for validator failure; got {rc}"
    assert len(fake_provider.calls) == 1, (
        "provider chat_completion should be invoked exactly once"
    )
    err_text = err.getvalue()
    assert "validate_form_data_contract" in err_text or "rejected" in err_text


# ----------------------------------------------------------------------
# 4. Happy path → YAML block emitted with family:, forms:, and the CURIE
# ----------------------------------------------------------------------


def test_cli_renders_yaml_block_on_success():
    """Mock provider with valid output → YAML block on stdout."""
    target_curie = "sh:datatype"
    valid_payload = _build_valid_form_data_payload(target_curie)
    import json as _json
    fake_provider = _FakeProvider(_json.dumps(valid_payload))

    out = io.StringIO()
    with patch.object(
        cli, "_build_provider", return_value=fake_provider
    ), redirect_stdout(out):
        rc = cli.main(
            [
                "--curie",
                target_curie,
                "--course-code",
                "rdf-shacl-551-2",
                "--provider",
                "local",
                "--force-overwrite",
            ]
        )

    assert rc == 0, f"expected exit 0 on happy path; got {rc}"
    rendered = out.getvalue()
    assert "family:" in rendered, "rendered YAML must carry 'family:' key"
    assert "forms:" in rendered, "rendered YAML must carry 'forms:' key"
    assert target_curie in rendered, (
        f"rendered YAML must contain target CURIE {target_curie!r}"
    )
    # NEXT STEPS comment block.
    assert "NEXT STEPS" in rendered
    assert "schema_translation_catalog.rdf_shacl.yaml" in rendered


# ----------------------------------------------------------------------
# 5. ToS regression — drafting prompt template has zero quoted-≥40-char spans
# ----------------------------------------------------------------------


def test_drafting_prompt_template_does_not_contain_example_content():
    """Regex sentinel: r'"[^"]{40,}"' matches ZERO times in the rendered template.

    Anchors the ToS contract that the drafting prompt is structurally
    minimal: schema rules + operator-authored manifest metadata only.
    Catches drift where a future maintainer adds a Claude-authored
    example sentence to the template by reflex.
    """
    import re

    rendered = cli._DRAFTING_PROMPT_TEMPLATE.format(
        curie="test:Foo",
        label="test fixture label",
        surface_forms_csv="test:Foo, fooBar",
    )
    matches = re.findall(r'"[^"]{40,}"', rendered)
    assert matches == [], (
        "Drafting prompt template MUST NOT contain quoted strings >=40 "
        f"chars (ToS regression sentinel). Found {len(matches)} matches:"
        f" {matches[:3]}"
    )
