#!/usr/bin/env python3
"""Update the public-safe Ed4All development-token summary.

Only aggregate token and session counts are written. Prompts, responses,
session identifiers, hostnames, usernames, and source paths never leave the
local machine.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


SCHEMA_VERSION = 2
README_START = "<!-- development-token-stats:start -->"
README_END = "<!-- development-token-stats:end -->"
DEFAULT_AGGREGATE = Path("docs/reference/development-token-stats.json")
DEFAULT_README = Path("README.md")
LOC_CATEGORIES = ("source", "tests", "docs", "tooling_config", "other")
LOC_EXCLUDED_PARTS = {"runtime", "vendor", "vendors", "node_modules", ".venv", "dist", "build", "courses"}


def _read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(item, dict):
                    yield item
    except OSError:
        return


def _inside_repo(value: object, repo: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        Path(value).expanduser().resolve().relative_to(repo.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _empty_source() -> dict[str, int]:
    return {
        "tokens": 0,
        "sessions": 0,
        "user_turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
        "duration_seconds": 0,
    }


def _is_claude_user_prompt(message: dict[str, Any]) -> bool:
    """Distinguish operator prompts from tool-result rows using no content."""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    # Claude records tool results with role=user, but they are continuations of
    # an assistant tool call rather than user-initiated prompts.
    return any(
        isinstance(block, dict) and block.get("type") in {"text", "image"}
        for block in content
    ) and not any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _loc_category(path: Path) -> str:
    parts = set(path.parts)
    name = path.name.lower()
    if "tests" in parts or name.startswith("test_") or name.endswith("_test.py"):
        return "tests"
    if "docs" in parts or path.suffix.lower() in {".md", ".rst"}:
        return "docs"
    if parts & {"scripts", "cli", "ci", "config", "schemas", ".github"} or path.suffix.lower() in {".yaml", ".yml", ".toml", ".ini"}:
        return "tooling_config"
    if path.suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".css", ".html", ".sql", ".rs", ".go"}:
        return "source"
    return "other"


def collect_loc(repo: Path) -> dict[str, int]:
    """Count newline-delimited lines in maintained Git-tracked text files."""
    result = {category: 0 for category in LOC_CATEGORIES}
    result["total"] = 0
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    for raw in listed.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if set(relative.parts) & LOC_EXCLUDED_PARTS:
            continue
        try:
            data = (repo / relative).read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        lines = len(data.splitlines())
        result[_loc_category(relative)] += lines
        result["total"] += lines
    return result


def collect_claude(paths: Iterable[Path], repo: Path) -> dict[str, int]:
    """Sum final Claude usage snapshots, keyed by session and message ID."""
    messages: dict[tuple[str, str], dict[str, int]] = {}
    sessions: set[str] = set()
    turns: set[tuple[str, str]] = set()
    ranges: dict[str, tuple[datetime, datetime]] = {}

    for path in paths:
        for index, item in enumerate(_read_json_lines(path)):
            if not _inside_repo(item.get("cwd"), repo):
                continue
            session_id = item.get("sessionId")
            if not isinstance(session_id, str) or not session_id:
                continue
            sessions.add(session_id)
            timestamp = item.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    prior = ranges.get(session_id, (moment, moment))
                    ranges[session_id] = (min(prior[0], moment), max(prior[1], moment))
                except ValueError:
                    pass
            message = item.get("message")
            if not isinstance(message, dict):
                continue
            if _is_claude_user_prompt(message):
                turn_id = item.get("uuid") or message.get("id") or f"record-{index}"
                turns.add((session_id, str(turn_id)))
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            message_id = message.get("id")
            if not isinstance(message_id, str) or not message_id:
                # The local UUID is only an in-memory deduplication key.
                message_id = str(item.get("uuid") or f"record-{index}")
            snapshot = {
                field: _nonnegative_int(usage.get(field))
                for field in (
                    "input_tokens", "cache_creation_input_tokens",
                    "cache_read_input_tokens", "output_tokens",
                )
            }
            total = sum(snapshot.values())
            key = (session_id, message_id)
            # Streaming snapshots repeat the same message with growing usage.
            if total >= sum(messages.get(key, {}).values()):
                messages[key] = snapshot

    result = _empty_source()
    for snapshot in messages.values():
        for field, value in snapshot.items():
            result[field] += value
    result["tokens"] = sum(
        result[field] for field in (
            "input_tokens", "cache_creation_input_tokens",
            "cache_read_input_tokens", "output_tokens",
        )
    )
    result["sessions"] = len(sessions)
    result["user_turns"] = len(turns)
    result["duration_seconds"] = sum(int((end - start).total_seconds()) for start, end in ranges.values())
    return result


def collect_codex(paths: Iterable[Path], repo: Path) -> dict[str, int]:
    """Use the final cumulative ``token_count`` value for each Codex session."""
    totals: dict[str, dict[str, int]] = {}
    turns: dict[str, set[str]] = {}
    durations: dict[str, int] = {}

    for path in paths:
        session_id: str | None = None
        in_scope = False
        file_totals: list[int] = []
        file_snapshots: list[dict[str, int]] = []
        file_turns: set[str] = set()
        file_range: tuple[datetime, datetime] | None = None
        for item in _read_json_lines(path):
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            timestamp = item.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    prior = file_range or (moment, moment)
                    file_range = (min(prior[0], moment), max(prior[1], moment))
                except ValueError:
                    pass
            if item.get("type") == "session_meta":
                candidate = payload.get("session_id")
                session_id = candidate if isinstance(candidate, str) and candidate else None
                in_scope = _inside_repo(payload.get("cwd"), repo)
                continue
            if item.get("type") == "event_msg" and payload.get("type") == "user_message":
                # The content is deliberately ignored; ordinal identity exists
                # only in memory to avoid counting repeated rows in one file.
                file_turns.add(str(payload.get("id") or len(file_turns)))
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            usage = info.get("total_token_usage")
            if not isinstance(usage, dict):
                continue
            total = _nonnegative_int(usage.get("total_tokens"))
            file_totals.append(total)
            file_snapshots.append({
                "tokens": total,
                "input_tokens": _nonnegative_int(usage.get("input_tokens")),
                "output_tokens": _nonnegative_int(usage.get("output_tokens")),
                "cached_input_tokens": _nonnegative_int(usage.get("cached_input_tokens")),
                "reasoning_output_tokens": _nonnegative_int(usage.get("reasoning_output_tokens")),
            })
        if in_scope and session_id and file_totals:
            # A rollout may exist in active and archived trees. Session ID
            # deduplication plus the highest final snapshot avoids double count.
            snapshot = file_snapshots[-1]
            if snapshot["tokens"] >= totals.get(session_id, {}).get("tokens", 0):
                totals[session_id] = snapshot
            turns.setdefault(session_id, set()).update(file_turns)
            if file_range is not None:
                duration = int((file_range[1] - file_range[0]).total_seconds())
                durations[session_id] = max(durations.get(session_id, 0), duration)

    result = _empty_source()
    for snapshot in totals.values():
        for field, value in snapshot.items():
            result[field] += value
    result["sessions"] = len(totals)
    result["user_turns"] = sum(len(value) for value in turns.values())
    result["duration_seconds"] = sum(durations.values())
    return result


def _merge_export(sources: dict[str, dict[str, int]], path: Path) -> None:
    """Merge an operator export while copying numeric aggregates only."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read aggregate export: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"aggregate export must use schema_version {SCHEMA_VERSION}")
    exported = payload.get("sources")
    if not isinstance(exported, dict):
        raise ValueError("aggregate export requires a sources object")
    for name in ("claude", "codex"):
        values = exported.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"aggregate export source {name!r} must be an object")
        for field in _empty_source():
            value = values.get(field, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"aggregate export {name}.{field} must be a non-negative integer")
            sources[name][field] += value


def collect(repo: Path, claude_root: Path, codex_root: Path, external: Path | None) -> dict[str, Any]:
    claude_paths = sorted((claude_root / "projects").glob("**/*.jsonl"))
    codex_paths = sorted((codex_root / "sessions").glob("**/*.jsonl"))
    codex_paths += sorted((codex_root / "archived_sessions").glob("*.jsonl"))
    sources = {
        "claude": collect_claude(claude_paths, repo),
        "codex": collect_codex(codex_paths, repo),
    }
    if external is not None:
        _merge_export(sources, external)
    total_tokens = sum(value["tokens"] for value in sources.values())
    total_sessions = sum(value["sessions"] for value in sources.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "Ed4All development sessions",
        "sources": sources,
        "total_tokens": total_tokens,
        "total_sessions": total_sessions,
        "average_tokens_per_session": total_tokens // total_sessions if total_sessions else 0,
        "total_user_turns": sum(value["user_turns"] for value in sources.values()),
        "repository_lines": collect_loc(repo),
    }


def render_readme(summary: dict[str, Any]) -> str:
    claude = summary["sources"]["claude"]
    codex = summary["sources"]["codex"]
    total = summary["total_tokens"]
    read = sum(source["input_tokens"] + source["cache_creation_input_tokens"] + source["cache_read_input_tokens"] for source in (claude, codex))
    written = sum(source["output_tokens"] for source in (claude, codex))
    date = str(summary["generated_at_utc"]).split("T", 1)[0]
    average_seconds = sum(source["duration_seconds"] for source in (claude, codex)) // summary["total_sessions"] if summary["total_sessions"] else 0
    average_duration = f"{average_seconds // 3600}h {(average_seconds % 3600) // 60}m"
    loc = summary["repository_lines"]
    return f"""{README_START}
## Built with a little help from our AI friends

![Observed tokens](https://img.shields.io/badge/observed_tokens-{total}-7C3AED?style=for-the-badge)
![Sessions](https://img.shields.io/badge/sessions-{summary['total_sessions']}-2563EB?style=for-the-badge)
![Recorded user turns](https://img.shields.io/badge/recorded_user_turns-{summary['total_user_turns']}-0F766E?style=for-the-badge)
![Average session span](https://img.shields.io/badge/avg_session_span-{average_seconds // 3600}h_{(average_seconds % 3600) // 60}m-EA580C?style=for-the-badge)
![Maintained lines](https://img.shields.io/badge/maintained_LOC-{loc['total']}-DB2777?style=for-the-badge)

Ed4All's local development logs record **{total:,} tokens** across
**{summary['total_sessions']:,} sessions**—a playful, approximate measure of the
Claude and Codex collaboration behind the project. Updated {date}; only numeric
aggregates are published. [How it is counted](docs/operations/development-token-stats.md).

| Collaborator | Tokens | Sessions | Avg tokens/session | Avg session span | Recorded user turns |
|---|---:|---:|---:|---:|---:|
| Claude | {claude['tokens']:,} | {claude['sessions']:,} | {claude['tokens'] // claude['sessions'] if claude['sessions'] else 0:,} | {claude['duration_seconds'] // claude['sessions'] // 3600 if claude['sessions'] else 0}h {(claude['duration_seconds'] // claude['sessions'] % 3600) // 60 if claude['sessions'] else 0}m | {claude['user_turns']:,} |
| Codex | {codex['tokens']:,} | {codex['sessions']:,} | {codex['tokens'] // codex['sessions'] if codex['sessions'] else 0:,} | {codex['duration_seconds'] // codex['sessions'] // 3600 if codex['sessions'] else 0}h {(codex['duration_seconds'] // codex['sessions'] % 3600) // 60 if codex['sessions'] else 0}m | {codex['user_turns']:,} |
| **Combined** | **{total:,}** | **{summary['total_sessions']:,}** | **{summary['average_tokens_per_session']:,}** | **{average_duration}** | **{summary['total_user_turns']:,}** |

### What those tokens did

| Token type | Claude | Codex | Combined |
|---|---:|---:|---:|
| Fresh input | {claude['input_tokens']:,} | {codex['input_tokens'] - codex['cached_input_tokens']:,} | {claude['input_tokens'] + codex['input_tokens'] - codex['cached_input_tokens']:,} |
| Cache creation | {claude['cache_creation_input_tokens']:,} | — | {claude['cache_creation_input_tokens']:,} |
| Cached input read | {claude['cache_read_input_tokens']:,} | {codex['cached_input_tokens']:,} | {claude['cache_read_input_tokens'] + codex['cached_input_tokens']:,} |
| Output written | {claude['output_tokens']:,} | {codex['output_tokens']:,} | {written:,} |
| ↳ reasoning output | Not separately reported | {codex['reasoning_output_tokens']:,} | {codex['reasoning_output_tokens']:,} |

**{read:,} input/read tokens** and **{written:,} output/write tokens** were
observed. Codex cached input is already included in its input total, while
Claude reports cache creation and reads as additive categories; reasoning is a
subset of Codex output, not an extra token charge.

### Maintained repository lines

| Source | Tests | Docs | Tooling/config | Other | Total |
|---:|---:|---:|---:|---:|---:|
| {loc['source']:,} | {loc['tests']:,} | {loc['docs']:,} | {loc['tooling_config']:,} | {loc['other']:,} | **{loc['total']:,}** |

```mermaid
%%{{init: {{"theme":"base","themeVariables":{{"xyChart":{{"plotColorPalette":"#7C3AED, #2563EB, #0F766E, #EA580C, #DB2777"}}}}}}}}%%
xychart-beta
    title "Maintained lines by role"
    x-axis [Source, Tests, Docs, Tooling, Other]
    y-axis "Lines" 0 --> {max(loc.values())}
    bar [{loc['source']}, {loc['tests']}, {loc['docs']}, {loc['tooling_config']}, {loc['other']}]
```

```mermaid
%%{{init: {{"theme":"base","themeVariables":{{"pie1":"#7C3AED","pie2":"#2563EB","pieStrokeColor":"#334155","pieOuterStrokeColor":"#334155","pieTitleTextColor":"#475569","pieSectionTextColor":"#ffffff","pieLegendTextColor":"#475569"}}}}}}%%
pie showData
    title Development tokens by collaborator
    "Claude" : {claude['tokens']}
    "Codex" : {codex['tokens']}
```
{README_END}"""


def _replace_marked(text: str, replacement: str) -> str:
    start = text.find(README_START)
    end = text.find(README_END)
    if start < 0 and end < 0:
        return text.rstrip() + "\n\n---\n\n" + replacement + "\n"
    if start < 0 or end < start:
        raise ValueError("README development-token markers are incomplete")
    end += len(README_END)
    return text[:start] + replacement + text[end:]


def _comparable(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "generated_at_utc"}


def _install_hook(repo: Path) -> int:
    hook = repo / ".git" / "hooks" / "pre-push"
    command = "python3 scripts/ops/update_development_tokens.py --check\n"
    content = "#!/bin/sh\nset -eu\n" + command
    if hook.exists() and hook.read_text(encoding="utf-8") != content:
        print(f"Refusing to overwrite existing hook: {hook}", file=sys.stderr)
        return 2
    hook.write_text(content, encoding="utf-8")
    hook.chmod(0o755)
    print("Installed token-stat freshness check in the local pre-push hook.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--claude-root", type=Path, default=Path.home() / ".claude")
    parser.add_argument("--codex-root", type=Path, default=Path.home() / ".codex")
    parser.add_argument("--external", type=Path, default=None)
    parser.add_argument(
        "--export-only",
        type=Path,
        default=None,
        metavar="PATH",
        help="write a numeric-only machine aggregate without changing tracked files",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--install-hook", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.install_hook:
        return _install_hook(repo)
    external = args.external
    if external is None and os.environ.get("ED4ALL_TOKEN_STATS_EXPORT"):
        external = Path(os.environ["ED4ALL_TOKEN_STATS_EXPORT"])

    try:
        summary = collect(repo, args.claude_root, args.codex_root, external)
        if args.export_only is not None:
            export = {
                "schema_version": SCHEMA_VERSION,
                "sources": summary["sources"],
            }
            args.export_only.parent.mkdir(parents=True, exist_ok=True)
            args.export_only.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
            print("Wrote numeric-only development-token export.")
            return 0
        aggregate_path = repo / DEFAULT_AGGREGATE
        readme_path = repo / DEFAULT_README
        existing = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else None
        if args.check:
            if not isinstance(existing, dict) or _comparable(existing) != _comparable(summary):
                print("Development-token statistics are stale; run the updater.", file=sys.stderr)
                return 1
            expected = _replace_marked(readme_path.read_text(encoding="utf-8"), render_readme(existing))
            if readme_path.read_text(encoding="utf-8") != expected:
                print("README development-token footer is stale; run the updater.", file=sys.stderr)
                return 1
            print("Development-token statistics are current.")
            return 0

        readme = readme_path.read_text(encoding="utf-8")
        # On first installation the generated section changes README's LOC.
        # Render once, recount, then render the stable public values.
        readme_path.write_text(_replace_marked(readme, render_readme(summary)), encoding="utf-8")
        summary["repository_lines"] = collect_loc(repo)
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        provisional = readme_path.read_text(encoding="utf-8")
        readme_path.write_text(_replace_marked(provisional, render_readme(summary)), encoding="utf-8")
        print(f"Updated development-token statistics ({summary['total_tokens']:,} tokens).")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"Development-token update failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
