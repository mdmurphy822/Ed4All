"""Unit tests for ``lib/validators/qti_well_formed.py::QtiWellFormedValidator``.

W10 Phase 5. All QTI fixtures are HAND-AUTHORED in-test (no dependency on the
concurrent ``Courseforge/scripts/qti_emitter.py`` worker), following the
Pattern-15 shape documented at ``Courseforge/docs/troubleshooting.md:79-130``.
No course slugs or device paths — every fixture is an in-test string.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest

from lib.validators.qti_well_formed import QtiWellFormedValidator

QTI_NS = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"


# --------------------------------------------------------------------------
# Fixture builders (Pattern-15 shape — Courseforge/docs/troubleshooting.md).
# --------------------------------------------------------------------------

def _wrap(assessment_inner: str, assessment_attrs: str = 'ident="quiz_w1" title="Week 1 Quiz"') -> str:
    """Wrap a <section>…</section> body in the canonical questestinterop root."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="{QTI_NS}"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <assessment {assessment_attrs}>
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>cc_profile</fieldlabel>
        <fieldentry>cc.exam.v0p1</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="root_section">
{assessment_inner}
    </section>
  </assessment>
</questestinterop>"""


def _mc_item(
    ident: str = "question_1",
    cc_profile: str = "cc.multiple_choice.v0p1",
    with_answer_key: bool = True,
) -> str:
    """A multiple-choice <item> with two choices and an optional answer key."""
    resprocessing = ""
    if with_answer_key:
        resprocessing = """
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
        </resprocessing>"""
    return f"""      <item ident="{ident}" title="MC Question">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>{cc_profile}</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">What is 2 + 2?</mattext></material>
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
        </presentation>{resprocessing}
      </item>"""


def _well_formed_qti() -> str:
    """A fully valid, XSD-conformant, answer-keyed QTI 1.2 document."""
    return _wrap(_mc_item())


def _inputs(xml: str, label: str = "fixture.xml") -> Dict[str, Any]:
    """Wrap one XML string in the qti_strings input form."""
    return {"qti_strings": [{"id": label, "xml": xml}]}


def _codes(result) -> List[str]:
    return [i.code for i in result.issues]


# --------------------------------------------------------------------------
# Happy path.
# --------------------------------------------------------------------------

def test_well_formed_qti_passes_zero_critical():
    result = QtiWellFormedValidator().validate(_inputs(_well_formed_qti()))
    assert result.passed is True, _codes(result)
    assert result.action is None
    assert result.critical_count == 0
    assert result.validator_name == "qti_well_formed"
    assert result.validator_version == "0.1.0"


def test_well_formed_true_false_passes():
    tf_item = """      <item ident="q_tf" title="TF">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.true_false.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">The sky is blue.</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="T">
                <material><mattext texttype="text/html">True</mattext></material>
              </response_label>
              <response_label ident="F">
                <material><mattext texttype="text/html">False</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="1"/>
          </outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response1">T</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">1</setvar>
          </respcondition>
        </resprocessing>
      </item>"""
    result = QtiWellFormedValidator().validate(_inputs(_wrap(tf_item)))
    assert result.passed is True, _codes(result)


def test_essay_item_needs_no_answer_key():
    """An essay (manual-graded) item passes WITHOUT a resprocessing key."""
    essay_item = """      <item ident="q_essay" title="Essay">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.essay.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">Discuss the theorem.</mattext></material>
          <response_str ident="response1" rcardinality="Single">
            <render_fib rows="5" columns="60"/>
          </response_str>
        </presentation>
      </item>"""
    result = QtiWellFormedValidator().validate(_inputs(_wrap(essay_item)))
    assert result.passed is True, _codes(result)
    assert "QTI_NO_ANSWER_KEY" not in _codes(result)


# --------------------------------------------------------------------------
# Malformed fixtures → specific codes.
# --------------------------------------------------------------------------

def test_no_answer_key_trips_qti_no_answer_key():
    xml = _wrap(_mc_item(with_answer_key=False))
    result = QtiWellFormedValidator().validate(_inputs(xml))
    assert result.passed is False
    assert result.action == "block"
    assert "QTI_NO_ANSWER_KEY" in _codes(result)


def test_empty_varequal_trips_qti_no_answer_key():
    """A respcondition whose varequal has no text resolves no key."""
    item = """      <item ident="q_empty_key" title="MC">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_choice.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">Pick one.</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A">
                <material><mattext texttype="text/html">A</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <respcondition>
            <conditionvar>
              <varequal respident="response1"></varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">1</setvar>
          </respcondition>
        </resprocessing>
      </item>"""
    result = QtiWellFormedValidator().validate(_inputs(_wrap(item)))
    assert "QTI_NO_ANSWER_KEY" in _codes(result)


def test_bad_cc_profile_trips_qti_unsupported_profile():
    xml = _wrap(_mc_item(cc_profile="cc.matching.v0p1"))
    result = QtiWellFormedValidator().validate(_inputs(xml))
    assert result.passed is False
    assert "QTI_UNSUPPORTED_PROFILE" in _codes(result)


def test_missing_cc_profile_trips_qti_unsupported_profile():
    """An item with no cc_profile field at all is unclassifiable."""
    item = """      <item ident="q_noprofile" title="MC">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>5</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">Pick one.</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A">
                <material><mattext texttype="text/html">A</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
      </item>"""
    result = QtiWellFormedValidator().validate(_inputs(_wrap(item)))
    assert "QTI_UNSUPPORTED_PROFILE" in _codes(result)


def test_non_parseable_xml_trips_qti_parse_error():
    # Unclosed tag → ElementTree / QTIParser raises.
    broken = '<questestinterop><assessment ident="x"><section>'
    result = QtiWellFormedValidator().validate(_inputs(broken))
    assert result.passed is False
    assert result.action == "block"
    assert "QTI_PARSE_ERROR" in _codes(result)


def test_xsd_invalid_well_formed_trips_qti_xsd_invalid():
    """Well-formed XML that violates the XSD (assessment missing required
    ``ident`` attribute, which the XSD declares ``use="required"``)."""
    pytest.importorskip("lxml")
    # Well-formed (parses) but assessment has NO ident → XSD violation.
    xml = _wrap(_mc_item(), assessment_attrs='title="No Ident Quiz"')
    result = QtiWellFormedValidator().validate(_inputs(xml))
    assert result.passed is False
    assert "QTI_XSD_INVALID" in _codes(result)
    # The doc still parses, so no parse error.
    assert "QTI_PARSE_ERROR" not in _codes(result)


# --------------------------------------------------------------------------
# lxml-absent graceful degrade.
# --------------------------------------------------------------------------

def test_lxml_absent_emits_warning_and_still_runs_other_checks(monkeypatch):
    """When lxml import fails, the XSD dimension degrades to a warning
    (QTI_LXML_MISSING) but well-formedness / answer-key / profile checks
    still run, and a well-formed doc still passes."""
    real_import = importlib.import_module

    def _fake_import(name, *args, **kwargs):
        if name == "lxml" or name.startswith("lxml."):
            raise ImportError("simulated: lxml not installed")
        return real_import(name, *args, **kwargs)

    # The validator imports lxml lazily via ``from lxml import etree`` inside
    # _load_xsd_validator; patch the builtin import to fail for lxml.
    import builtins

    real_builtin_import = builtins.__import__

    def _fake_builtin_import(name, *args, **kwargs):
        if name == "lxml" or name.startswith("lxml."):
            raise ImportError("simulated: lxml not installed")
        return real_builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_builtin_import)

    result = QtiWellFormedValidator().validate(_inputs(_well_formed_qti()))
    codes = _codes(result)
    assert "QTI_LXML_MISSING" in codes
    # Warning-only degrade → the XSD dimension does not fail the gate.
    lxml_issue = next(i for i in result.issues if i.code == "QTI_LXML_MISSING")
    assert lxml_issue.severity == "warning"
    # Other checks still ran: a well-formed, answer-keyed doc passes.
    assert result.passed is True, codes
    assert result.critical_count == 0


def test_lxml_absent_still_catches_answer_key_miss(monkeypatch):
    """Graceful XSD degrade does not mask the answer-key check."""
    import builtins

    real_builtin_import = builtins.__import__

    def _fake_builtin_import(name, *args, **kwargs):
        if name == "lxml" or name.startswith("lxml."):
            raise ImportError("simulated: lxml not installed")
        return real_builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_builtin_import)

    xml = _wrap(_mc_item(with_answer_key=False))
    result = QtiWellFormedValidator().validate(_inputs(xml))
    codes = _codes(result)
    assert "QTI_LXML_MISSING" in codes
    assert "QTI_NO_ANSWER_KEY" in codes
    assert result.passed is False  # answer-key miss is still critical


# --------------------------------------------------------------------------
# Input-contract paths.
# --------------------------------------------------------------------------

def test_no_input_trips_qti_no_input():
    result = QtiWellFormedValidator().validate({})
    assert result.passed is False
    assert "QTI_NO_INPUT" in _codes(result)


def test_dir_input_globs_xml(tmp_path):
    (tmp_path / "quiz_a.xml").write_text(_well_formed_qti(), encoding="utf-8")
    (tmp_path / "quiz_b.xml").write_text(
        _wrap(_mc_item(with_answer_key=False)), encoding="utf-8"
    )
    result = QtiWellFormedValidator().validate({"qti_dir": str(tmp_path)})
    # quiz_b has no answer key → the gate blocks.
    assert result.passed is False
    assert "QTI_NO_ANSWER_KEY" in _codes(result)


def test_paths_input(tmp_path):
    p = tmp_path / "quiz.xml"
    p.write_text(_well_formed_qti(), encoding="utf-8")
    result = QtiWellFormedValidator().validate({"qti_paths": [str(p)]})
    assert result.passed is True, _codes(result)


def test_gate_id_override():
    result = QtiWellFormedValidator().validate(
        {**_inputs(_well_formed_qti()), "gate_id": "custom_gate"}
    )
    assert result.gate_id == "custom_gate"
