"""Unit tests for ``lib/validators/cartridge_conformance.py``.

Track L (L2). Every fixture cartridge is SYNTHETIC and built in-test as a ZIP
(no course slugs, no device paths, no network / model). Resource-document
shapes mirror the real emitter output but are hand-authored strings so the
test is independent of the packager. XSD checks exercise the vendored resource
XSDs (QTI / imsdt / assignment) AND — since the 1EdTech CC 1.3 manifest-profile
XSDs were vendored 2026-07-19 (``ccv1p3_imscp_v1p2_v1p0.xsd`` + the lom* /
imscc* imports) — the manifest itself. A clean cartridge now carries ZERO
issues: the synthetic ``_manifest`` fixtures below are authored to be genuinely
CC-1.3-profile-conformant (``<metadata>`` with schema/schemaversion +
``lomimscc:lom``; a rooted-hierarchy organization whose ROOT item has NO title
and whose nested items DO). The intended-to-fail fixtures fail for their
specific reason (bad resource-type, dangling href, orphaned answer_key, …),
not incidental manifest non-conformance.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List

import pytest

from lib.validators.cartridge_conformance import CartridgeConformanceValidator

CC13_MANIFEST_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
# CC 1.2 manifest namespace — recognised (no CARTRIDGE_UNKNOWN_CC_VERSION) but
# NOT vendored, so a CC-1.2 manifest degrades to a CARTRIDGE_XSD_MISSING warning
# rather than being validated against the CC 1.3 imscp XSD. Used by the
# cross-version resource-type test so CC 1.1/1.2 resource-type tokens live in
# their own-version manifest instead of tripping the CC 1.3 XSD type enumeration.
CC12_MANIFEST_NS = "http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
LOM_MANIFEST_NS = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"
IMSDT_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
ASSIGNMENT_NS = "http://www.imsglobal.org/xsd/imscc_extensions/assignment"
QTI_NS = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"

RES_QTI = "imsqti_xmlv1p2/imscc_xmlv1p3/assessment"
RES_DT = "imsdt_xmlv1p3"
RES_ASGN = "associatedcontent/imscc_xmlv1p3/learning-application-resource"
RES_WEB = "webcontent"


# --------------------------------------------------------------------------
# Synthetic resource-document builders (valid against the vendored XSDs).
# --------------------------------------------------------------------------

def _valid_discussion() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<topic xmlns="{IMSDT_NS}">'
        "<title>Week 1 Discussion</title>"
        '<text texttype="text/html">Discuss the objective.</text>'
        "</topic>"
    )


def _xsd_invalid_discussion() -> str:
    # Missing the REQUIRED <title> (sequence violation) → XSD-invalid but
    # still well-formed XML, so it reaches the XSD check.
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<topic xmlns="{IMSDT_NS}">'
        '<text texttype="text/html">No title here.</text>'
        "</topic>"
    )


def _valid_assignment() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<assignment xmlns="{ASSIGNMENT_NS}" identifier="ASGN-CO-01">'
        "<title>Week 1 Assignment</title>"
        '<text texttype="text/html">Complete the task.</text>'
        '<instructor_text texttype="text/html">Grade it.</instructor_text>'
        '<gradable points_possible="10">true</gradable>'
        "</assignment>"
    )


def _valid_qti() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="{QTI_NS}">
  <assessment ident="quiz_w1" title="Week 1 Quiz">
    <section ident="root_section">
      <item ident="q1" title="MC Question">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_choice.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">2 + 2?</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A">
                <material><mattext texttype="text/html">4</mattext></material>
              </response_label>
              <response_label ident="B">
                <material><mattext texttype="text/html">5</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="5"/>
          </outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response1">A</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">5</setvar>
          </respcondition>
        </resprocessing>
      </item>
    </section>
  </assessment>
</questestinterop>"""


def _webcontent_html(body: str = "Hello") -> str:
    return f"<!DOCTYPE html><html><head><title>Page</title></head><body><p>{body}</p></body></html>"


# --------------------------------------------------------------------------
# Manifest builder + ZIP assembler.
# --------------------------------------------------------------------------

def _manifest(resources_xml: str, ns: str = CC13_MANIFEST_NS) -> str:
    # CC-1.3-profile-conformant (validates against ccv1p3_imscp_v1p2_v1p0.xsd):
    #  * <metadata> carries schema + schemaversion + the required
    #    lomimscc:lom (the imscp ManifestMetadata.Type sequence).
    #  * the rooted-hierarchy organization's ROOT item is ItemOrg.Type — it
    #    takes NO <title> (a <title> there is the exact legacy packager defect
    #    the vendored XSDs now catch); its nested Item.Type child REQUIRES a
    #    <title>.
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        f'<manifest xmlns="{ns}" xmlns:lomimscc="{LOM_MANIFEST_NS}" '
        'identifier="TEST_manifest">'
        "<metadata><schema>IMS Common Cartridge</schema>"
        "<schemaversion>1.3.0</schemaversion>"
        "<lomimscc:lom><lomimscc:general><lomimscc:title>"
        '<lomimscc:string language="en">Test Course</lomimscc:string>'
        "</lomimscc:title></lomimscc:general></lomimscc:lom>"
        "</metadata>"
        '<organizations><organization identifier="ORG_1" structure="rooted-hierarchy">'
        '<item identifier="ROOT">'
        '<item identifier="WEEK_1"><title>Week 1</title></item>'
        "</item>"
        "</organization></organizations>"
        f"<resources>{resources_xml}</resources>"
        "</manifest>"
    )


def _resource(res_id: str, res_type: str, href: str) -> str:
    return (
        f'<resource identifier="{res_id}" type="{res_type}" href="{href}">'
        f'<file href="{href}"/></resource>'
    )


def _build_zip(tmp_path: Path, members: Dict[str, str], name: str = "course.imscc") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member_name, content in members.items():
            zf.writestr(member_name, content)
    return path


def _default_members() -> Dict[str, str]:
    """A fully valid mini-cartridge: manifest + 4 resource docs."""
    resources = (
        _resource("RES_page", RES_WEB, "week_01/content.html")
        + _resource("RES_quiz", RES_QTI, "06_assessments/week_01_quiz.xml")
        + _resource("RES_disc", RES_DT, "06_assessments/week_01_discussion.xml")
        + _resource("RES_asgn", RES_ASGN, "06_assessments/week_01_assignment.xml")
    )
    return {
        "imsmanifest.xml": _manifest(resources),
        "week_01/content.html": _webcontent_html(),
        "06_assessments/week_01_quiz.xml": _valid_qti(),
        "06_assessments/week_01_discussion.xml": _valid_discussion(),
        "06_assessments/week_01_assignment.xml": _valid_assignment(),
    }


def _codes(result) -> List[str]:
    return [i.code for i in result.issues]


def _critical_codes(result) -> List[str]:
    return [i.code for i in result.issues if i.severity == "critical"]


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------

def test_valid_cartridge_passes_clean(tmp_path):
    path = _build_zip(tmp_path, _default_members())
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert result.passed, _codes(result)
    assert result.action is None
    assert _critical_codes(result) == []
    # The CC manifest-profile XSD is now vendored (2026-07-19) alongside the
    # resource XSDs, so every XML doc in a clean cartridge routes to a real
    # schema — a conformant fixture carries ZERO issues (no XSD_MISSING degrade).
    assert _codes(result) == []


def test_resource_doc_is_xsd_validated(tmp_path):
    """Proves resource docs ARE routed to + validated against their XSD."""
    members = _default_members()
    members["06_assessments/week_01_discussion.xml"] = _xsd_invalid_discussion()
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert not result.passed
    assert result.action == "block"
    assert "CARTRIDGE_XSD_INVALID" in _critical_codes(result)


def test_unresolved_href_trips_href_unresolved(tmp_path):
    members = _default_members()
    resources = (
        _resource("RES_page", RES_WEB, "week_01/content.html")
        + _resource("RES_ghost", RES_WEB, "week_99/missing.html")
    )
    members["imsmanifest.xml"] = _manifest(resources)
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert not result.passed
    assert "CARTRIDGE_HREF_UNRESOLVED" in _critical_codes(result)


def test_unknown_resource_type_trips_type_unknown(tmp_path):
    members = _default_members()
    resources = _resource("RES_page", "bogus_type_v9", "week_01/content.html")
    members["imsmanifest.xml"] = _manifest(resources)
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert not result.passed
    assert "CARTRIDGE_RESOURCE_TYPE_UNKNOWN" in _critical_codes(result)


def test_cc11_and_cc12_resource_types_are_conformant(tmp_path):
    """A CC 1.1 / 1.2 versioned type token must not be false-flagged by the
    programmatic resource-type enumeration.

    The CC 1.1/1.2 resource-type tokens are carried in a CC 1.2 manifest (their
    own version) — the CC 1.3 imscp XSD's ``type`` enumeration lists only the
    CC 1.3 tokens, so housing them in a CC 1.3 manifest would XSD-invalidate the
    fixture for a reason unrelated to what this test targets. The CC 1.2
    manifest namespace has no vendored XSD → clean degrade, and the programmatic
    ``_is_conformant_resource_type`` family patterns still accept the tokens.
    """
    members = _default_members()
    resources = (
        _resource("RES_page", RES_WEB, "week_01/content.html")
        + _resource("RES_disc11", "imsdt_xmlv1p1", "week_01/content.html")
        + _resource(
            "RES_quiz12",
            "imsqti_xmlv1p2/imscc_xmlv1p2/assessment",
            "week_01/content.html",
        )
    )
    members["imsmanifest.xml"] = _manifest(resources, ns=CC12_MANIFEST_NS)
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert "CARTRIDGE_RESOURCE_TYPE_UNKNOWN" not in _codes(result)
    # The versioned tokens are conformant, so nothing critical fires (the CC 1.2
    # manifest degrades to a warning, not an XSD-invalid critical).
    assert _critical_codes(result) == [], _critical_codes(result)


def test_missing_manifest_trips_manifest_missing(tmp_path):
    members = _default_members()
    del members["imsmanifest.xml"]
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert not result.passed
    assert "CARTRIDGE_MANIFEST_MISSING" in _critical_codes(result)


def test_malformed_manifest_trips_parse_error(tmp_path):
    members = _default_members()
    members["imsmanifest.xml"] = "<manifest><resources><oops></manifest>"
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert not result.passed
    assert "CARTRIDGE_MANIFEST_PARSE_ERROR" in _critical_codes(result)


def test_orphaned_answer_key_trips_unreferenced(tmp_path):
    members = _default_members()
    members["06_assessments/week_01_answer_key.html"] = _webcontent_html("Answers")
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert not result.passed
    assert "CARTRIDGE_ANSWER_KEY_UNREFERENCED" in _critical_codes(result)


def test_referenced_answer_key_is_clean(tmp_path):
    members = _default_members()
    ak = "06_assessments/week_01_answer_key.html"
    members[ak] = _webcontent_html("Answers")
    resources = (
        _resource("RES_page", RES_WEB, "week_01/content.html")
        + _resource("RES_quiz", RES_QTI, "06_assessments/week_01_quiz.xml")
        + _resource("RES_disc", RES_DT, "06_assessments/week_01_discussion.xml")
        + _resource("RES_asgn", RES_ASGN, "06_assessments/week_01_assignment.xml")
        + _resource("RES_ak", RES_WEB, ak)
    )
    members["imsmanifest.xml"] = _manifest(resources)
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert "CARTRIDGE_ANSWER_KEY_UNREFERENCED" not in _codes(result)
    assert result.passed, _critical_codes(result)


def test_unknown_cc_version_is_warning_only(tmp_path):
    members = _default_members()
    members["imsmanifest.xml"] = _manifest(
        _resource("RES_page", RES_WEB, "week_01/content.html"),
        ns="http://example.com/not/a/cc/namespace",
    )
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    codes = _codes(result)
    assert "CARTRIDGE_UNKNOWN_CC_VERSION" in codes
    # It is a warning — on its own it does not block.
    assert "CARTRIDGE_UNKNOWN_CC_VERSION" not in _critical_codes(result)


def test_external_href_is_skipped(tmp_path):
    members = _default_members()
    resources = (
        _resource("RES_page", RES_WEB, "week_01/content.html")
        + _resource("RES_link", "imswl_xmlv1p3", "https://example.org/reading")
    )
    members["imsmanifest.xml"] = _manifest(resources)
    path = _build_zip(tmp_path, members)
    result = CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    assert "CARTRIDGE_HREF_UNRESOLVED" not in _codes(result)


def test_no_input_trips_no_input():
    result = CartridgeConformanceValidator().validate({})
    assert not result.passed
    assert "CARTRIDGE_NO_INPUT" in _critical_codes(result)


def test_bad_zip_trips_no_input(tmp_path):
    bad = tmp_path / "not_a_zip.imscc"
    bad.write_text("this is not a zip")
    result = CartridgeConformanceValidator().validate({"imscc_path": str(bad)})
    assert not result.passed
    assert "CARTRIDGE_NO_INPUT" in _critical_codes(result)


def test_extracted_dir_form(tmp_path):
    members = _default_members()
    root = tmp_path / "extracted"
    for name, content in members.items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    result = CartridgeConformanceValidator().validate({"imscc_dir": str(root)})
    assert result.passed, _critical_codes(result)


def test_lxml_missing_degrades(tmp_path, monkeypatch):
    """When lxml is unavailable every XSD check degrades to one warning."""
    import lib.validators.cartridge_conformance as mod

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "lxml" or name.startswith("lxml."):
            raise ImportError("simulated: lxml not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    path = _build_zip(tmp_path, _default_members())
    result = mod.CartridgeConformanceValidator().validate({"imscc_path": str(path)})
    codes = _codes(result)
    assert "CARTRIDGE_LXML_MISSING" in codes
    # Programmatic checks still ran and the cartridge is otherwise clean.
    assert result.passed, _critical_codes(result)


# The exact XSD-error signature the CC 1.3 imscp profile emits for a <title>
# child on the ROOT (ItemOrg.Type) organization item — the ONE known legacy
# defect (see below). lxml's message reads e.g. "Element '{...}title': This
# element is not expected."
_ROOT_TITLE_XSD_SIGNATURE = "title': This element is not expected"


def test_real_built_cartridge_shape_if_present():
    """Smoke against a real built cartridge when one is on disk (never fails
    the suite when absent — keeps the test hermetic).

    The cartridge is located dynamically (no course slug in tracked code).

    2026-07-19: the archived cartridges on disk were all built BEFORE the
    packager's ROOT-item ``<title>`` fix (``package_multifile_imscc.py`` no
    longer emits a ``<title>`` on the ROOT ``ItemOrg.Type`` item, which the CC
    1.3 imscp profile forbids). So the newly-vendored manifest-profile XSDs
    now legitimately flag that ONE legacy defect as ``CARTRIDGE_XSD_INVALID``.
    We therefore assert:

      (a) the validator RAN with full XSD coverage — no ``CARTRIDGE_XSD_MISSING``
          for the imscp manifest namespace (proves the manifest XSDs vendored
          and routed), and
      (b) EVERY ``CARTRIDGE_XSD_INVALID`` issue matches the known root-title
          signature — i.e. the only conformance break on disk is the legacy
          ROOT-title one, nothing new / unexplained.

    Once the pipeline is re-run end-to-end with the fixed packager, the built
    cartridge will be fully conformant; at that point tighten this to
    ``assert result.passed`` (and drop the root-title carve-out below).
    """
    candidates = sorted(
        Path("LibV2/courses").glob("*/source/imscc/*.imscc")
    ) if Path("LibV2/courses").is_dir() else []
    if not candidates:
        pytest.skip("no built cartridge on disk")
    result = CartridgeConformanceValidator().validate(
        {"imscc_path": str(candidates[0])}
    )
    codes = _codes(result)
    # (a) Full XSD coverage: the imscp manifest namespace resolved to a real
    # vendored schema, so there is no degrade warning for it.
    assert "CARTRIDGE_XSD_MISSING" not in codes, codes
    # (b) Any XSD-invalid finding must be the single known legacy defect: a
    # <title> on the ROOT organization item (predates the packager fix).
    xsd_invalid = [i for i in result.issues if i.code == "CARTRIDGE_XSD_INVALID"]
    for issue in xsd_invalid:
        assert _ROOT_TITLE_XSD_SIGNATURE in issue.message, issue.message
    # And no OTHER critical conformance break beyond that root-title signature.
    other_critical = [
        i.code
        for i in result.issues
        if i.severity == "critical" and i.code != "CARTRIDGE_XSD_INVALID"
    ]
    assert other_critical == [], other_critical
