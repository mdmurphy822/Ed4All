"""Validate links across every policy-approved public document."""

from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "docs" / "architecture" / "repository-layout.json"
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _approved_docs() -> list[Path]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    paths = [PROJECT_ROOT / entry for entry in policy["public_docs"]]
    assert len(paths) == 49
    return paths


def _heading_anchors(text: str) -> set[str]:
    """Return GitHub-style heading anchors, including duplicate suffixes."""
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in HEADING_RE.findall(text):
        plain = re.sub(r"<[^>]+>", "", heading)
        plain = re.sub(r"[`*_~]", "", plain).strip().casefold()
        base = re.sub(r"[^\w\- ]", "", plain, flags=re.UNICODE)
        base = re.sub(r"\s+", "-", base)
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return anchors


def _link_target(raw_target: str) -> tuple[str, str]:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    path, separator, fragment = target.partition("#")
    return urllib.parse.unquote(path), urllib.parse.unquote(fragment) if separator else ""


def test_policy_approved_public_document_links_and_anchors_resolve() -> None:
    failures: list[str] = []
    approved = _approved_docs()

    for source in approved:
        if not source.is_file():
            failures.append(f"missing approved document: {source.relative_to(PROJECT_ROOT)}")
            continue
        if source.suffix.casefold() != ".md":
            continue

        text = source.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            path_text, fragment = _link_target(raw_target)
            if path_text.startswith(("http://", "https://", "mailto:")):
                continue

            target = (source.parent / path_text).resolve() if path_text else source
            if not target.exists():
                failures.append(
                    f"{source.relative_to(PROJECT_ROOT)}: missing target {raw_target!r}"
                )
                continue
            if fragment and target.suffix.casefold() == ".md":
                anchors = _heading_anchors(target.read_text(encoding="utf-8"))
                if fragment.casefold() not in anchors:
                    failures.append(
                        f"{source.relative_to(PROJECT_ROOT)}: missing anchor "
                        f"{fragment!r} in {target.relative_to(PROJECT_ROOT)}"
                    )

    assert not failures, "broken public-document links:\n" + "\n".join(failures)
