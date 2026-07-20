"""Ed4All licensing guard package.

Machine-readable teacher-roster + fail-closed export/ingest license guards
(backlog SFT-C S6/S7; owner memo ``scratchpad/licensing_memo.md``).

Canonical prose reference: ``docs/LICENSING.md``. This package is the
*machine-readable* half of that posture — the export-time fail-closed
license filter + per-checkpoint LICENSE assertion + the Nemotron
license-pin / FAIL-BUILD-ON-RE-PIN guard the memo's maintenance-contract
note asks for ("register the endpoint-registry ``{verdict, obligations[]}``
fields as the machine-readable source of truth; the export-time tier
filter + fail-closed Claude-id gate convert this memo's posture into a
build invariant").
"""

from __future__ import annotations

from lib.licensing.teacher_roster import (
    BARRED_PROVENANCE,
    GENERAL_NVIDIA_OML_IDENTITY,
    NEMOTRON_PINNED_LICENSE_IDENTITY,
    NEMOTRON_ROSTER_KEY,
    PROVENANCE_VERDICTS,
    TEACHER_ROSTER,
    LicenseGuardError,
    LicenseRecord,
    assert_checkpoint_license,
    assert_export_licenses,
    assert_nemotron_pin,
    classify_pair_teacher,
    license_for_model,
    provenance_license_tag,
    provider_verdict_roster,
    stamp_pair_license,
)

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
    "provider_verdict_roster",
    "stamp_pair_license",
]
