"""Protect the public SemantiK behavior-flags documentation contract."""

from __future__ import annotations

import re
from pathlib import Path

from lib.assistant.tools import _flag_table_rows
from SemantiK.semantik_structure.glmocr import (
    resolve_glmocr_lane_mode,
    resolve_heading_judge_mode,
)
from SemantiK.semantik_structure.glmocr.heading_judge import (
    resolve_est_per_judgment_thinkoff,
    resolve_heading_judge_enable_thinking,
    resolve_heading_judge_frequency_penalty,
    resolve_max_tokens_ceiling_thinkoff,
    resolve_max_tokens_floor_thinkoff,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = PROJECT_ROOT / "docs" / "operations" / "behavior-flags-semantik.md"

OBSOLETE_BARE_FLAGS = {
    "SEMANTIK_ANCHOR_TRUNCATE",
    "SEMANTIK_API_KEY",
    "SEMANTIK_CONCURRENCY",
    "SEMANTIK_CONTEXT_TEXT_TRUNCATE",
    "SEMANTIK_HEADING_TEXT_TRUNCATE",
    "SEMANTIK_MAX_COVERAGE_RESPLIT_ROUNDS",
    "SEMANTIK_MAX_PENDING_PER_WINDOW",
    "SEMANTIK_MIN_PENDING_PER_SPLIT",
    "SEMANTIK_MIN_PENDING_WINDOW_CAP",
    "SEMANTIK_MODEL",
}

REQUIRED_FLAGS = {
    "SEMANTIK_GLMOCR_LANE",
    "SEMANTIK_HEADING_JUDGE",
    "SEMANTIK_HEADING_JUDGE_ENABLE_THINKING",
    "SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY",
    "SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF",
    "SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF",
    "SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT_THINKOFF",
}


def _guide() -> str:
    """Read the tracked public guide as UTF-8 text."""
    return GUIDE_PATH.read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    """Collapse Markdown whitespace for prose-level contract checks."""
    prose = re.sub(r"(?m)^>\s?", "", text)
    return " ".join(prose.lower().split())


def test_preferred_route_is_sdk_normalize_enrich_then_super_judge() -> None:
    """Keep the public route aligned with the preferred conversion pipeline."""
    preferred = _guide().split("## Flag reference", 1)[0]
    prose = _normalized(preferred)

    sdk = prose.index("glm-ocr sdk")
    normalize = prose.index("normaliz", sdk)
    enrich = prose.index("enrich", normalize)
    judge = prose.index("super heading judge", enrich)

    assert sdk < normalize < enrich < judge
    assert "preferred conversion path" in prose


def test_documented_master_defaults_match_live_resolvers(monkeypatch) -> None:
    """Distinguish the opt-in OCR lane from the default-on heading judge."""
    monkeypatch.delenv("SEMANTIK_GLMOCR_LANE", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE", raising=False)

    assert resolve_glmocr_lane_mode() is False
    assert resolve_heading_judge_mode() is True

    prose = _normalized(_guide())
    assert "glm-ocr lane is **opt-in**" in prose
    assert "heading judge is **default-on**" in prose
    assert all(token in prose for token in ("`0`", "`false`", "`no`", "`off`"))


def test_thinking_off_budget_defaults_match_live_resolvers(monkeypatch) -> None:
    """Pin the compact heading-classification request defaults to live code."""
    for name in REQUIRED_FLAGS:
        monkeypatch.delenv(name, raising=False)

    assert resolve_heading_judge_enable_thinking() is False
    assert resolve_heading_judge_frequency_penalty() == 0.3
    assert resolve_max_tokens_ceiling_thinkoff() == 4096
    assert resolve_max_tokens_floor_thinkoff() == 512
    assert resolve_est_per_judgment_thinkoff() == 64

    rows = "\n".join(_flag_table_rows(GUIDE_PATH))
    expected_defaults = {
        "SEMANTIK_HEADING_JUDGE_ENABLE_THINKING": "off",
        "SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY": "`0.3` thinking-off",
        "SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF": "`4096`",
        "SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF": "`512`",
        "SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT_THINKOFF": "`64`",
    }
    for flag, documented_default in expected_defaults.items():
        pattern = rf"^\| `{flag}` \| {re.escape(documented_default)}(?:;[^|]*)? \|"
        assert re.search(pattern, rows, flags=re.MULTILINE), flag


def test_private_data_posture_is_explicit_and_complete() -> None:
    """Require every sensitive operator-data family to remain private."""
    intro = _normalized(_guide().split("## Preferred conversion profile", 1)[0])

    for term in (
        "source documents",
        "converted html",
        "course and run identifiers",
        "caches",
        "model artifacts",
        "credentials",
        "endpoint values",
        "evaluation data",
        "logs",
    ):
        assert term in intro
    assert "always operator-private" in intro
    assert "ignored or external storage" in intro
    assert "never commit" in intro


def test_flag_tables_use_the_conventional_parseable_shape() -> None:
    """Keep flag rows consumable by the assistant's table-row helper."""
    rows = _flag_table_rows(GUIDE_PATH)
    assert rows

    parsed: dict[str, tuple[str, str]] = {}
    row_pattern = re.compile(
        r"^\| `(?P<flag>SEMANTIK_[A-Z0-9_]+)` "
        r"\| (?P<default>[^|]+) \| (?P<purpose>.+) \|$"
    )
    for row in rows:
        match = row_pattern.fullmatch(row)
        assert match, f"non-conventional behavior-flag row: {row}"
        flag = match.group("flag")
        assert flag not in parsed, f"duplicate behavior-flag row: {flag}"
        parsed[flag] = (
            match.group("default").strip(),
            match.group("purpose").strip(),
        )

    assert REQUIRED_FLAGS <= parsed.keys()


def test_local_markdown_links_resolve() -> None:
    """Reject broken relative links in the public guide."""
    missing: list[str] = []
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", _guide()):
        path_text = target.split("#", 1)[0]
        if not path_text or "://" in path_text:
            continue
        if not (GUIDE_PATH.parent / path_text).resolve().exists():
            missing.append(target)
    assert not missing, f"broken local Markdown links: {missing}"


def test_public_guide_rejects_private_and_historical_vocabulary() -> None:
    """Keep obsolete architecture and deployment history out of public docs."""
    text = _guide()

    forbidden_patterns = {
        "retired classifier architecture": r"\b(?:ModernBERT|council)\b",
        "machine/vendor profiles": r"\b(?:DGX|RTX|RunPod)\b",
        "literal loopback endpoint": r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0):\d+",
        "task/wave/campaign history": r"\b(?:task\s*#?\d+|wave[- ]?\d+|campaign)\b",
    }
    for label, pattern in forbidden_patterns.items():
        assert not re.search(pattern, text, flags=re.IGNORECASE), label

    for flag in OBSOLETE_BARE_FLAGS:
        assert not re.search(rf"(?<![A-Z0-9_]){re.escape(flag)}(?![A-Z0-9_])", text), flag
