"""ProseStutterValidator unit tests (book-1 canary keystone fix).

Pins the deterministic phrase-repetition (stutter) detector and its gate
wiring. Positive fixtures are ANONYMIZED renditions of the real stutter
shapes found on the canary corpus (no course slugs / corpus text anywhere);
negative fixtures pin the documented false-positive guards: legitimate
lists, flip-card front/back term repetition (element boundaries are hard),
hidden ``data-cf-curie`` spans, code blocks, parallel/contrast
constructions, term-definition heading fusion, parenthetical restates, and
math-formula comparisons.
"""
from __future__ import annotations

from typing import Any, Dict, List

from lib.validators.prose_stutter import (
    ProseStutterValidator,
    StutterHit,
    find_stutters,
    strip_html_for_stutter,
)


def _rules(text: str) -> List[str]:
    return [h.rule for h in find_stutters(text)]


def _block(*, block_id: str = "week_01_content_01#concept_intro_0",
           content: Any = "") -> Dict[str, Any]:
    return {"block_id": block_id, "block_type": "concept",
            "page_id": "week_01_content_01", "content": content}


# --------------------------------------------------------------------- #
# Detector — positives (anonymized real stutter shapes)
# --------------------------------------------------------------------- #

def test_adjacent_list_duplication_detected():
    # Shape: an enumerated list duplicated verbatim inside one sentence.
    text = ("The pipeline consists of four stages: review, build, staging "
            "rollout, and final rollout review, build, staging rollout, and "
            "final rollout.")
    hits = find_stutters(text)
    assert any(h.rule == "adjacent_repeat" for h in hits)
    top = next(h for h in hits if h.rule == "adjacent_repeat")
    # The verbatim repeated span is reported.
    assert "review, build, staging rollout" in top.repeated_span


def test_mid_phrase_gap_zero_duplication_detected():
    # Shape: "X and Y are safe and Y are safe and idempotent".
    text = ("ALPHA and BETA are safe and BETA are safe and idempotent "
            "methods for retrieving resources without side effects.")
    assert "adjacent_repeat" in _rules(text)


def test_three_token_adjacent_duplication_detected():
    # Shape: "direct external requests to internal requests to internal paths".
    text = ("Gateways use routing maps to direct external requests to "
            "internal requests to internal paths so old paths keep working.")
    assert "adjacent_repeat" in _rules(text)


def test_near_adjacent_list_item_duplication_detected():
    # Shape: a list where one item repeats a few tokens later.
    text = ("Gateways centralize caching, rate limiting, authentication, "
            "authorization, and rate limiting to improve performance.")
    assert "near_adjacent_repeat" in _rules(text)


def test_windowed_same_sentence_duplication_detected():
    # Shape: an >=4-gram content phrase repeated within a 30-token window
    # inside ONE sentence.
    text = ("The resolver forwards the query to the authoritative name "
            "server responsible for it authoritative name server responsible "
            "for the zone in question, which then answers.")
    hits = find_stutters(text)
    assert any(h.rule in ("window_repeat", "adjacent_repeat") for h in hits)


def test_colon_label_duplication_detected():
    # Shape: "Key Idea: Idea:" duplicated label emission.
    text = "Key Idea: Idea: A server has little control over its queue."
    assert "label_dup" in _rules(text)


def test_verb_echo_detected():
    # Shape: "each retry at a lower level adds delay adds to the total time".
    text = ("This happens because each retry at a lower level adds delay "
            "adds to the overall execution time of the request.")
    assert "echo_word" in _rules(text)


def test_verbatim_span_reported():
    text = ("The scheduler retries until either a maximum number of retries "
            "until either a maximum number of attempts is reached.")
    hits = find_stutters(text)
    assert hits
    assert all(h.repeated_span for h in hits)
    assert all(h.segment_excerpt for h in hits)


# --------------------------------------------------------------------- #
# Detector — negatives (documented FP guards)
# --------------------------------------------------------------------- #

def test_legitimate_list_passes():
    text = ("Tests are classified as small, intermediate, or large based on "
            "computing resource requirements and execution environment; "
            "unit, integration, and end-to-end tests differ in scope.")
    assert _rules(text) == []


def test_flip_card_term_repetition_passes():
    # Flip-card grids legitimately repeat the term label front/back — the
    # element boundary must be a hard segment boundary.
    html = ("<div class='flip-card'><div class='flip-card-front'>Latency"
            "</div><div class='flip-card-back'>Latency is the time between "
            "request and response.</div></div>"
            "<div class='flip-card'><div class='flip-card-front'>Throughput"
            "</div><div class='flip-card-back'>Throughput is the rate of "
            "completed work.</div></div>")
    assert find_stutters(strip_html_for_stutter(html)) == []


def test_hidden_curie_span_stripped():
    # Postminted CURIE spans legitimately repeat token runs; they must be
    # stripped before scanning.
    html = ("<p>A clear single sentence about the topic at hand today.</p>"
            "<span hidden data-cf-curie='crs:load_balancer crs:load_balancer'>"
            "crs:load_balancer crs:load_balancer crs:load_balancer</span>")
    assert find_stutters(strip_html_for_stutter(html)) == []


def test_code_block_repetition_passes():
    html = ("<p>The loop body repeats the call as shown below.</p>"
            "<pre>do_work(); do_work(); do_work(); do_work();</pre>")
    assert find_stutters(strip_html_for_stutter(html)) == []


def test_contrast_parallel_construction_passes():
    # "...read locks, while read locks..." contrast parallelism is guarded.
    text = ("The upgrade path converts read locks, while read locks never "
            "convert back to their earlier shared representation.")
    assert _rules(text) == []


def test_determiner_gap_two_referent_construction_passes():
    # "merges the received clock with its local clock" — two referents.
    text = ("On receipt each process merges the received clock with its "
            "local clock by taking the element-wise maximum value.")
    assert _rules(text) == []


def test_term_definition_heading_fusion_passes():
    # A capitalised term restated at a definition start (heading fusion) is
    # furniture, not stutter.
    text = ("The pattern is called circuit breaking Circuit breaking stops "
            "calls to a failing dependency until it recovers fully.")
    assert _rules(text) == []


def test_parenthetical_restate_passes():
    text = ("Peak load reached nine hundred requests every second (nine "
            "hundred requests every second sustained for one minute).")
    assert _rules(text) == []


def test_math_formula_comparison_passes():
    text = ("Under resharding hash(key) mod 4 differs from hash(key) mod 5 "
            "for most keys, forcing large-scale data movement.")
    assert _rules(text) == []


def test_parallel_enumeration_marker_passes():
    text = ("One client only ever queries replica one, and another only "
            "ever queries replica two, so their reads may diverge.")
    assert _rules(text) == []


def test_clean_prose_passes():
    text = ("A load balancer distributes incoming requests across healthy "
            "instances. Health checks remove failing instances from the "
            "pool, and new instances join once they report ready.")
    assert _rules(text) == []


def test_empty_and_tiny_inputs_pass():
    assert find_stutters("") == []
    assert find_stutters("short text only") == []
    assert strip_html_for_stutter("") == ""


# --------------------------------------------------------------------- #
# Validator surface
# --------------------------------------------------------------------- #

_STUTTERED_HTML = ("<p>The pipeline consists of four stages: review, build, "
                   "staging rollout, and final rollout review, build, "
                   "staging rollout, and final rollout.</p>")
_CLEAN_HTML = ("<p>Tests are classified as small, intermediate, or large "
               "based on resource requirements alone.</p>")


def test_validator_flags_stuttered_block_with_regenerate_action():
    v = ProseStutterValidator()
    result = v.validate({"blocks": [_block(content=_STUTTERED_HTML)]})
    assert result.passed is False
    assert result.action == "regenerate"
    assert result.issues
    issue = result.issues[0]
    assert issue.code == "BLOCK_PROSE_STUTTER"
    assert issue.severity == "warning"
    assert issue.location == "week_01_content_01#concept_intro_0"
    # Verbatim repeated span surfaces in the message.
    assert "review, build, staging rollout" in issue.message
    assert result.metadata["blocks_stuttered"] == 1


def test_validator_passes_clean_blocks():
    v = ProseStutterValidator()
    result = v.validate({"blocks": [_block(content=_CLEAN_HTML)]})
    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.metadata["blocks_scanned"] == 1


def test_validator_skips_dict_content_outline_blocks():
    v = ProseStutterValidator()
    result = v.validate({"blocks": [_block(content={"key_claims": []})]})
    assert result.passed is True
    assert result.metadata["blocks_scanned"] == 0


def test_validator_single_block_chain_shape():
    # The router chain calls validate({"block": b, "blocks": [b]}).
    v = ProseStutterValidator()
    blk = _block(content=_STUTTERED_HTML)
    result = v.validate({"block": blk, "blocks": [blk]})
    assert result.passed is False
    assert result.action == "regenerate"


def test_validator_missing_inputs_fails_closed():
    v = ProseStutterValidator()
    result = v.validate({})
    assert result.passed is False
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"


def test_validator_hydrates_blocks_jsonl(tmp_path):
    import json

    path = tmp_path / "blocks_final.jsonl"
    rows = [_block(content=_STUTTERED_HTML),
            _block(block_id="week_01_content_01#concept_intro_1",
                   content=_CLEAN_HTML)]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8",
    )
    v = ProseStutterValidator()
    result = v.validate({"blocks_final_path": str(path)})
    assert result.passed is False
    assert result.metadata["blocks_scanned"] == 2
    assert result.metadata["blocks_stuttered"] == 1


# --------------------------------------------------------------------- #
# Gate registration
# --------------------------------------------------------------------- #

def test_gate_registered_in_router():
    from MCP.hardening.gate_input_routing import default_router

    r = default_router()
    assert (
        "lib.validators.prose_stutter.ProseStutterValidator" in r.builders
    )


def test_gate_configured_at_post_rewrite_validation():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load(
        (root / "config" / "workflows.yaml").read_text(encoding="utf-8")
    )
    workflows = cfg.get("workflows", cfg)
    found = {}
    for wf_name in ("textbook_to_course", "course_generation"):
        for phase in workflows[wf_name]["phases"]:
            for gate in phase.get("validation_gates") or []:
                if gate.get("gate_id") == "block_prose_stutter":
                    found[wf_name] = (phase["name"], gate)
    for wf_name in ("textbook_to_course", "course_generation"):
        assert wf_name in found, f"block_prose_stutter missing in {wf_name}"
        phase_name, gate = found[wf_name]
        assert phase_name == "post_rewrite_validation"
        assert gate["severity"] == "warning"
        assert gate["behavior"]["on_fail"] == "warn"
        assert (
            gate["validator"]
            == "lib.validators.prose_stutter.ProseStutterValidator"
        )
