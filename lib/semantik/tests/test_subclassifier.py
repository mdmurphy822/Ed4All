"""Build #23 Tier-3 — model-assisted composite-unit subclassifier.

All-mocked LLM (a plain callable). Covers: strict parse, fold-on-ingestion,
page-anomaly skip, the payload-only HTML invariant, decision-capture firing,
report shape + bucket-collapse detection. No GPU / network.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from lib.semantik import subclassifier as sc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _Capture:
    """Minimal DecisionCapture double recording log_decision calls."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _unit_html(
    unit_type: str = "worked_example",
    pages: str = "3-4",
    member_pages=("3", "4"),
    body: str = "Solve 2x plus 3 equals 7 for x by isolating the variable.",
) -> str:
    members = "\n".join(
        f'<section class="dart-section" data-semantik-block-id="s{i}" '
        f'data-semantik-pages="{p}"><p>{body}</p></section>'
        for i, p in enumerate(member_pages)
    )
    return (
        f'<section class="dart-unit dart-unit-{unit_type}" '
        f'data-semantik-unit="{unit_type}" role="group" aria-label="X" '
        f'data-semantik-pages="{pages}" data-semantik-page-kind="physical">\n'
        f"<h4>Example 1.1</h4>\n{members}\n</section>"
    )


def _fixed_client(label: str):
    def _c(prompt: str, *, max_tokens: int = 64) -> str:
        import json

        return json.dumps({"subclass": label, "confidence": 0.9})

    return _c


def _sequence_client(labels, confidence: float = 0.9):
    """Mock that returns a different label per successive call (round-robin).

    A ``None`` entry emits an unparseable prose response so the vote path can be
    exercised with dropped-out samples. Accepts (and ignores) any sampling
    kwargs the caller threads (``max_tokens`` / ``temperature`` / ``seed``).
    """
    import json

    seq = list(labels)
    state = {"i": 0}

    def _c(prompt: str, **kwargs) -> str:
        i = state["i"]
        state["i"] = i + 1
        lab = seq[i % len(seq)]
        if lab is None:
            return "this is prose, not a single kebab label at all"
        return json.dumps({"subclass": lab, "confidence": confidence})

    return _c


# ---------------------------------------------------------------------------
# Strict parse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("application-problem", "application-problem"),
        ("  DRILL  ", "drill"),
        ("word problems", "word-problems"),  # single space → hyphen typo-fix
        ("x" * 33, None),  # over ceiling
        ("word problems set", None),  # multi-space prose → reject
        ("has_underscore", None),  # not kebab-case
        ("Trailing.", None),  # punctuation
        ("", None),
        (None, None),
    ],
)
def test_normalize_label_strict(raw, expected):
    assert sc.normalize_label(raw) == expected


def test_parse_reject_leaves_unit_unsubclassed():
    html = _unit_html()
    out, report = sc.annotate_html_subclasses(
        html, client=_fixed_client("this is prose not a label at all"),
        capture=_Capture(),
    )
    assert "data-semantik-subclass=" not in out
    assert report["status_counts"].get("parse_reject") == 1


def test_extract_label_from_bare_token():
    label, conf = sc._extract_label_and_confidence("application-problem")
    assert label == "application-problem"
    assert conf is None


def test_extract_label_and_confidence_from_json():
    label, conf = sc._extract_label_and_confidence('{"subclass":"drill","confidence":0.8}')
    assert label == "drill"
    assert conf == 0.8


# ---------------------------------------------------------------------------
# Fold-on-ingestion
# ---------------------------------------------------------------------------
def test_fold_exact_hit_not_new():
    resolved, is_new, folded = sc.fold_label("drill", ("drill", "word-problems"))
    assert (resolved, is_new, folded) == ("drill", False, None)


def test_fold_near_duplicate_by_edit_distance():
    resolved, is_new, folded = sc.fold_label("drils", ("drill", "word-problems"))
    assert resolved == "drill" and is_new is False and folded == "drill"


def test_fold_same_stem():
    resolved, is_new, folded = sc.fold_label("drills", ("drill",))
    assert resolved == "drill" and folded == "drill"


def test_genuinely_new_label_accepted_and_logged(tmp_path):
    sidecar = tmp_path / "review.jsonl"
    html = _unit_html()
    out, report = sc.annotate_html_subclasses(
        html, client=_fixed_client("proof-sketch"), capture=_Capture(),
        review_sidecar_path=str(sidecar),
    )
    assert 'data-semantik-subclass="proof-sketch"' in out
    assert report["new_labels"] == [
        {"unit_id": "unit-0000", "unit_type": "worked_example", "label": "proof-sketch"}
    ]
    # New label logged to sidecar, NOT canonized into the taxonomy.
    assert sidecar.exists()
    assert "proof-sketch" in sidecar.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Page-anomaly guard
# ---------------------------------------------------------------------------
def test_page_anomaly_skip_non_monotonic_members():
    html = _unit_html(member_pages=("7", "3"))  # decreasing → anomaly
    out, report = sc.annotate_html_subclasses(
        html, client=_fixed_client("drill"), capture=_Capture(),
    )
    assert "data-semantik-subclass=" not in out
    assert report["status_counts"].get("skipped_anomaly") == 1


def test_monotonic_pages_not_skipped():
    html = _unit_html(member_pages=("3", "3", "4"))
    out, report = sc.annotate_html_subclasses(
        html, client=_fixed_client("symbolic-manipulation"), capture=_Capture(),
    )
    assert "data-semantik-subclass=" in out
    assert report["status_counts"].get("assigned") == 1


# ---------------------------------------------------------------------------
# Payload-only invariant
# ---------------------------------------------------------------------------
def test_payload_only_html_differs_by_two_attrs_only():
    html = _unit_html()
    out, _ = sc.annotate_html_subclasses(
        html, client=_fixed_client("symbolic-manipulation"), capture=_Capture(),
    )
    assert out != html
    # Stripping ONLY the two added tokens recovers the input byte-for-byte.
    recovered = out.replace(
        ' data-semantik-subclass="symbolic-manipulation"', ""
    ).replace(" dart-sub-symbolic-manipulation", "")
    assert recovered == html


def test_no_units_html_unchanged():
    html = "<section class='dart-section'><p>Plain content, no unit.</p></section>"
    out, report = sc.annotate_html_subclasses(
        html, client=_fixed_client("drill"), capture=_Capture(),
    )
    assert out == html
    assert report["total_units"] == 0


def test_idempotent_on_already_subclassed_unit():
    html = _unit_html().replace(
        'data-semantik-unit="worked_example"',
        'data-semantik-unit="worked_example" data-semantik-subclass="drill"',
    )
    out, report = sc.annotate_html_subclasses(
        html, client=_fixed_client("symbolic-manipulation"), capture=_Capture(),
    )
    assert out == html  # never re-annotate
    assert report["total_units"] == 0


# ---------------------------------------------------------------------------
# Decision capture fires (LLM call-site contract)
# ---------------------------------------------------------------------------
def test_decision_capture_fires_with_dynamic_rationale():
    cap = _Capture()
    sc.annotate_html_subclasses(
        _unit_html(), client=_fixed_client("symbolic-manipulation"), capture=cap,
    )
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["decision_type"] == "unit_subclass_assignment"
    # Rationale interpolates dynamic per-call signals.
    r = call["rationale"]
    assert "worked_example" in r and "symbolic-manipulation" in r
    assert "pages=3-4" in r and "0.90" in r
    assert len(r) >= 20


# ---------------------------------------------------------------------------
# Report shape + bucket-collapse
# ---------------------------------------------------------------------------
def test_report_shape():
    out, report = sc.annotate_html_subclasses(
        _unit_html(), client=_fixed_client("drill"), capture=_Capture(),
    )
    for key in (
        "total_units", "status_counts", "label_distribution",
        "fold_events", "new_labels", "bucket_collapse", "assignments",
    ):
        assert key in report
    assert report["label_distribution"]["worked_example"]["drill"] == 1


def test_bucket_collapse_detection():
    # 12 units of one type all get the same label → collapse.
    dist = {"worked_example": {"drill": 12}}
    flagged = sc.detect_bucket_collapse(dist)
    assert flagged and flagged[0]["unit_type"] == "worked_example"
    assert flagged[0]["dominant_label"] == "drill"


def test_bucket_collapse_healthy_spread_not_flagged():
    dist = {"worked_example": {"drill": 6, "word-problems": 5, "mixed-review": 4}}
    assert sc.detect_bucket_collapse(dist) == []


def test_bucket_collapse_small_n_not_flagged():
    # Below the n>=10 floor, a single label is not enough evidence.
    dist = {"worked_example": {"drill": 5}}
    assert sc.detect_bucket_collapse(dist) == []


# ---------------------------------------------------------------------------
# Flag resolver
# ---------------------------------------------------------------------------
def test_flag_default_off_and_truthy(monkeypatch):
    monkeypatch.delenv(sc.SEMANTIK_SUBCLASS_ENV, raising=False)
    assert sc.resolve_semantic_subclass_enabled() is False
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_ENV, "true")
    assert sc.resolve_semantic_subclass_enabled() is True
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_ENV, "0")
    assert sc.resolve_semantic_subclass_enabled() is False


def test_client_error_does_not_break_render():
    def _boom(prompt, *, max_tokens=64):
        raise RuntimeError("model down")

    out, report = sc.annotate_html_subclasses(
        _unit_html(), client=_boom, capture=_Capture(),
    )
    assert "data-semantik-subclass=" not in out
    assert report["status_counts"].get("error") == 1


# ---------------------------------------------------------------------------
# Self-consistency sampling (SEMANTIK_SUBCLASS_SAMPLES > 1)
# ---------------------------------------------------------------------------
def test_resolve_subclass_samples_default_and_fallback(monkeypatch):
    monkeypatch.delenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, raising=False)
    assert sc.resolve_subclass_samples() == 1
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, "5")
    assert sc.resolve_subclass_samples() == 5
    for garbage in ("0", "-3", "", "abc", "2.5"):
        monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, garbage)
        assert sc.resolve_subclass_samples() == 1


def test_majority_vote_wins_and_confidence_is_agreement(monkeypatch):
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, "5")
    # 3x application-problem, 1x symbolic-manipulation, 1x dropped parse.
    client = _sequence_client(
        ["application-problem", "symbolic-manipulation", "application-problem",
         None, "application-problem"]
    )
    out, report = sc.annotate_html_subclasses(
        _unit_html(), client=client, capture=_Capture(),
    )
    assert 'data-semantik-subclass="application-problem"' in out
    a = report["assignments"][0]
    assert a["label"] == "application-problem"
    assert a["status"] == "assigned"
    # dominant 3 of 4 valid samples (the None dropped out).
    assert a["confidence"] == pytest.approx(3 / 4)


def test_vote_tie_breaks_first_seen(monkeypatch):
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, "2")
    client = _sequence_client(["symbolic-manipulation", "application-problem"])
    out, report = sc.annotate_html_subclasses(
        _unit_html(), client=client, capture=_Capture(),
    )
    a = report["assignments"][0]
    # Both appear once → first-seen (symbolic-manipulation) wins, conf 1/2.
    assert a["label"] == "symbolic-manipulation"
    assert a["confidence"] == pytest.approx(1 / 2)


def test_all_samples_parse_fail_leaves_unit_unsubclassed(monkeypatch):
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, "3")
    client = _sequence_client([None, None, None])
    out, report = sc.annotate_html_subclasses(
        _unit_html(), client=client, capture=_Capture(),
    )
    assert "data-semantik-subclass=" not in out
    assert report["status_counts"].get("parse_reject") == 1


def test_all_samples_raise_reports_error(monkeypatch):
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, "3")

    def _boom(prompt, **kwargs):
        raise RuntimeError("model down")

    out, report = sc.annotate_html_subclasses(
        _unit_html(), client=_boom, capture=_Capture(),
    )
    assert "data-semantik-subclass=" not in out
    assert report["status_counts"].get("error") == 1


def test_decision_capture_rationale_interpolates_vote_distribution(monkeypatch):
    monkeypatch.setenv(sc.SEMANTIK_SUBCLASS_SAMPLES_ENV, "3")
    cap = _Capture()
    client = _sequence_client(
        ["application-problem", "application-problem", "symbolic-manipulation"]
    )
    sc.annotate_html_subclasses(_unit_html(), client=client, capture=cap)
    assert len(cap.calls) == 1
    r = cap.calls[0]["rationale"]
    # Dynamic vote distribution interpolated: labels + counts + valid/N.
    assert "vote [" in r
    assert "application-problem=2" in r and "symbolic-manipulation=1" in r
    assert "valid=3/3" in r
    assert len(r) >= 20


def test_prompt_renders_glosses_and_definition_instruction():
    p = sc._build_prompt(
        "worked_example",
        "Example 1.1",
        "Some text.",
        ("application-problem", "symbolic-manipulation"),
        glosses={
            "application-problem": "posed in a real-world context",
            "symbolic-manipulation": "pure expression rewriting",
        },
    )
    assert "- application-problem: posed in a real-world context" in p
    assert "- symbolic-manipulation: pure expression rewriting" in p
    assert "Match on the DEFINITION" in p
    assert "[application-problem, symbolic-manipulation]" not in p


def test_prompt_falls_back_to_bare_list_without_glosses():
    for glosses in (None, {}, {"application-problem": ""}):
        p = sc._build_prompt(
            "worked_example",
            "Example 1.1",
            "Some text.",
            ("application-problem", "symbolic-manipulation"),
            glosses=glosses,
        )
        assert "[application-problem, symbolic-manipulation]" in p
        assert "Match on the DEFINITION" not in p


def test_annotate_threads_lexicon_glosses_into_prompt():
    seen: List[str] = []

    def client(prompt: str, **kw: Any) -> str:
        seen.append(prompt)
        return '{"subclass": "application-problem", "confidence": 0.9}'

    html = (
        '<section class="dart-unit dart-unit-worked_example" '
        'data-semantik-unit="worked_example" role="group" '
        'aria-labelledby="example-1-1">'
        "<h4 id=\"example-1-1\">Example 1.1</h4><p>A ball costs 3 dollars.</p>"
        "</section>"
    )
    out, report = sc.annotate_html_subclasses(
        html, client=client, course_code="TEST", profile_spec="generic-academic"
    )
    assert seen and "Match on the DEFINITION" in seen[0]
    assert "real-world" in seen[0]  # the lexicon gloss reached the model
    assert 'data-semantik-subclass="application-problem"' in out
