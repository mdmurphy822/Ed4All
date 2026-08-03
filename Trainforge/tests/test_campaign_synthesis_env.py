"""Campaign-env synthesis routing + teacher-roster contract (2026-07-22).

Pins the exact provider/roster combination the operator campaign env
template (§ "Stage-B enablement") exports for in-build training synthesis:

    TRAINFORGE_SYNTHESIS_PROVIDER=local
    LOCAL_SYNTHESIS_BASE_URL=http://localhost:8001/v1   (the local model seat)
    LOCAL_SYNTHESIS_MODEL=nemotron-3-super              (--served-model-name)

Asserts, fully hermetically (no network, no LLM, no course slugs):

  1. the ``local`` endpoint-registry row resolves to the Super seat's base
     URL + served-model-name under those envs (registry resolution only);
  2. the teacher roster resolves the served Nemotron model to a SAFE verdict
     (the "nemotron roster rule resolves SAFE" runbook contract), the
     Nemotron license pin holds, and the coarse ``local`` provenance is SAFE;
  3. a pair stamped with this teacher passes the fail-closed export filter;
  4. the trap that motivated ``provider=local``: the pair factories hard-
     whitelist provider names, so the raw seat name ``local-seat`` (what the
     workflow runner's Gap-C fill would have setdefault'd from LLM_PROVIDER)
     raises ``NotImplementedError`` — the campaign env MUST export the
     literal ``local``;
  5. the env override is honored and Anthropic-family routing stays
     fail-closed even when the call site asks for ``local``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.licensing.teacher_roster import (  # noqa: E402
    PROVENANCE_VERDICTS,
    assert_export_licenses,
    assert_nemotron_pin,
    license_for_model,
    provenance_license_tag,
)

# The campaign-env contract under test (mirror of campaign-env.sh exports).
CAMPAIGN_ENV = {
    "TRAINFORGE_SYNTHESIS_PROVIDER": "local",
    "LOCAL_SYNTHESIS_BASE_URL": "http://localhost:8001/v1",
    "LOCAL_SYNTHESIS_MODEL": "nemotron-3-super",
}


@pytest.fixture()
def campaign_env(monkeypatch):
    for k, v in CAMPAIGN_ENV.items():
        monkeypatch.setenv(k, v)
    # Never allow the anthropic acknowledgment gate to leak in from the host.
    monkeypatch.delenv("TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS", raising=False)
    # NB: lib.llm.endpoints.resolve_endpoint reads *_env vars per call (it is
    # deliberately uncached), so monkeypatched env is honored directly.
    yield


def test_local_registry_row_resolves_to_super_seat(campaign_env):
    from lib.llm.endpoints import resolve_endpoint

    ep = resolve_endpoint("local")
    assert ep.base_url == "http://localhost:8001/v1"
    assert ep.model == "nemotron-3-super"
    # License-clean self-hosted provenance — never the hosted "nvidia" value.
    assert ep.provenance_provider == "local"
    # No API key requirement blocks the offline seat.
    assert ep.api_key_required is False


def test_nemotron_roster_rule_resolves_safe(campaign_env):
    rec = license_for_model(CAMPAIGN_ENV["LOCAL_SYNTHESIS_MODEL"])
    assert rec is not None, "served model must resolve a roster record"
    assert rec.name == "nemotron"
    assert rec.verdict == "safe"
    # The Case-B pin the training preflight asserts — must not drift to the
    # general NVIDIA Open Model License.
    assert_nemotron_pin()
    # Coarse provenance verdict for the stamped closed-enum provider value.
    assert PROVENANCE_VERDICTS["local"] == "safe"
    tag = provenance_license_tag(
        generating_seat=CAMPAIGN_ENV["LOCAL_SYNTHESIS_MODEL"], provider="local"
    )
    assert tag.endswith("/safe")
    assert tag.startswith("nemotron/")


def test_campaign_stamped_pairs_pass_export_filter(campaign_env):
    pairs = [
        # A paraphrase pair as the local provider stamps it (coarse provider).
        {"id": "p1", "prompt": "q?", "completion": "a.", "provider": "local"},
        # A deterministic SFT-program pair with the fine-grained seat stamp.
        {"id": "p2", "prompt": "q?", "completion": "a.",
         "provider": "deterministic",
         "generating_seat": CAMPAIGN_ENV["LOCAL_SYNTHESIS_MODEL"],
         "license": provenance_license_tag(
             generating_seat=CAMPAIGN_ENV["LOCAL_SYNTHESIS_MODEL"],
             provider="deterministic",
         )},
    ]
    # Must not raise (fail-closed filter passes the campaign teacher).
    assert_export_licenses(pairs, source_desc="campaign-env test corpus")


def test_raw_seat_name_is_rejected_by_pair_factories():
    """Why campaign-env exports the literal ``local``: the factories
    whitelist provider NAMES, so the raw registry seat name crashes."""
    from Trainforge.generators.instruction_factory import (
        synthesize_instruction_pair,
    )

    with pytest.raises(NotImplementedError):
        synthesize_instruction_pair({}, seed=0, provider="local-seat")


def test_env_override_keeps_anthropic_fail_closed(campaign_env, monkeypatch, tmp_path):
    """TRAINFORGE_SYNTHESIS_PROVIDER overrides the call-site kwarg inside
    run_synthesis, and the Anthropic SDK path fails closed unconditionally —
    even a caller asking for ``local`` cannot be re-routed to a barred
    teacher without the loud SynthesisLicensingError."""
    from Trainforge.synthesize_training import (
        SynthesisLicensingError,
        run_synthesis,
    )

    monkeypatch.setenv("TRAINFORGE_SYNTHESIS_PROVIDER", "anthropic")
    with pytest.raises(SynthesisLicensingError):
        run_synthesis(tmp_path, "CAMPAIGN_TEST", provider="local")
