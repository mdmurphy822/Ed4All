"""W7.3 — tests for SynthesizedQuizDistractorValidator.

Builds QTI 1.2 MCQ documents (parsed by the in-tree QTIParser) and pins the
four distractor-degeneracy warnings (equals-key / duplicate / too-few /
placeholder), the clean happy-path pass, the no-input graceful no-op, and the
decision-capture emit. All issues are warning-severity day-1 (passed always
True).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from lib.validators.synthesized_quiz_distractor import (
    SynthesizedQuizDistractorValidator,
)

QTI_NS = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"


def _choice(ident: str, text: str) -> str:
    return (
        f'<response_label ident="{ident}">'
        f'<material><mattext texttype="text/html">{text}</mattext></material>'
        f"</response_label>"
    )


def _mc_item(
    choices: List[Tuple[str, str]],
    correct_ident: str,
    ident: str = "q1",
) -> str:
    labels = "".join(_choice(cid, ctext) for cid, ctext in choices)
    return f"""      <item ident="{ident}" title="MC">
        <itemmetadata><qtimetadata><qtimetadatafield>
          <fieldlabel>cc_profile</fieldlabel>
          <fieldentry>cc.multiple_choice.v0p1</fieldentry>
        </qtimetadatafield></qtimetadata></itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">Pick one.</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>{labels}</render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes><decvar varname="SCORE" vartype="Integer"/></outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response1">{correct_ident}</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">5</setvar>
          </respcondition>
        </resprocessing>
      </item>"""


def _wrap(inner: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="{QTI_NS}">
  <assessment ident="quiz_w1" title="Week 1 Quiz">
    <section ident="root">
{inner}
    </section>
  </assessment>
</questestinterop>"""


def _inputs(xml: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"qti_strings": [{"id": "w1.xml", "xml": xml}]}
    d.update(extra)
    return d


def _codes(result) -> List[str]:
    return [i.code for i in result.issues]


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


# --------------------------------------------------------------------- #

def test_clean_mcq_passes_zero_issues():
    xml = _wrap(
        _mc_item(
            [("A", "The Sun"), ("B", "The Moon"), ("C", "Mars"), ("D", "Venus")],
            correct_ident="A",
        )
    )
    result = SynthesizedQuizDistractorValidator().validate(_inputs(xml))
    assert result.passed is True
    # Only distractor-quality codes; a clean item emits none.
    assert not any(c.startswith("SYNTH_QUIZ_DISTRACTOR") for c in _codes(result))


def test_distractor_equals_key_flagged():
    xml = _wrap(
        _mc_item(
            [("A", "The Sun"), ("B", "the sun."), ("C", "Mars"), ("D", "Venus")],
            correct_ident="A",
        )
    )
    result = SynthesizedQuizDistractorValidator().validate(_inputs(xml))
    assert result.passed is True  # warning-day-1
    assert "SYNTH_QUIZ_DISTRACTOR_EQUALS_KEY" in _codes(result)


def test_duplicate_distractors_flagged():
    xml = _wrap(
        _mc_item(
            [("A", "The Sun"), ("B", "Mars"), ("C", "Mars"), ("D", "Venus")],
            correct_ident="A",
        )
    )
    result = SynthesizedQuizDistractorValidator().validate(_inputs(xml))
    assert "SYNTH_QUIZ_DISTRACTOR_DUPLICATE" in _codes(result)


def test_too_few_distractors_flagged():
    xml = _wrap(
        _mc_item([("A", "The Sun"), ("B", "Mars")], correct_ident="A")
    )
    result = SynthesizedQuizDistractorValidator().validate(_inputs(xml))
    assert "SYNTH_QUIZ_DISTRACTOR_TOO_FEW" in _codes(result)


def test_placeholder_distractor_flagged():
    xml = _wrap(
        _mc_item(
            [
                ("A", "The Sun"),
                ("B", "Not supported by the source (option 1)."),
                ("C", "None of the above"),
                ("D", "Venus"),
            ],
            correct_ident="A",
        )
    )
    result = SynthesizedQuizDistractorValidator().validate(_inputs(xml))
    assert "SYNTH_QUIZ_DISTRACTOR_PLACEHOLDER" in _codes(result)


def test_no_input_is_graceful_noop():
    result = SynthesizedQuizDistractorValidator().validate({})
    assert result.passed is True
    assert "SYNTH_QUIZ_NO_INPUT" in _codes(result)


def test_non_mcq_items_skipped():
    # An essay item (no response_lid) is not audited.
    essay = """      <item ident="e1" title="Essay">
        <itemmetadata><qtimetadata><qtimetadatafield>
          <fieldlabel>cc_profile</fieldlabel>
          <fieldentry>cc.essay.v0p1</fieldentry>
        </qtimetadatafield></qtimetadata></itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">Discuss.</mattext></material>
          <response_str ident="response1"><render_fib/></response_str>
        </presentation>
      </item>"""
    result = SynthesizedQuizDistractorValidator().validate(_inputs(_wrap(essay)))
    assert result.passed is True
    assert not any(c.startswith("SYNTH_QUIZ_DISTRACTOR") for c in _codes(result))


def test_decision_capture_fires():
    xml = _wrap(
        _mc_item(
            [("A", "The Sun"), ("B", "The Moon"), ("C", "Mars"), ("D", "Venus")],
            correct_ident="A",
        )
    )
    capture = _RecordingCapture()
    SynthesizedQuizDistractorValidator().validate(
        _inputs(xml, decision_capture=capture)
    )
    assert any(
        e.get("decision_type") == "validation_result" for e in capture.events
    )
