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
import shlex
import subprocess
import sys
from typing import Any, Iterable


SCHEMA_VERSION = 2
README_START = "<!-- development-token-stats:start -->"
README_END = "<!-- development-token-stats:end -->"
DEFAULT_AGGREGATE = Path("docs/reference/development-token-stats.json")
DEFAULT_README = Path("README.md")
README_SECTION_ANCHOR = "## From source to course-grounded AI"
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
    average_seconds = sum(source["duration_seconds"] for source in (claude, codex)) // summary["total_sessions"] if summary["total_sessions"] else 0
    average_duration = f"{average_seconds // 3600}h {(average_seconds % 3600) // 60}m"
    loc = summary["repository_lines"]
    fresh = claude["input_tokens"] + codex["input_tokens"] - codex["cached_input_tokens"]
    cache_read = claude["cache_read_input_tokens"] + codex["cached_input_tokens"]
    cache_write = claude["cache_creation_input_tokens"]
    return f"""{README_START}
<div align="center">
<table>
<thead>
<tr bgcolor="#1F6FEB"><th align="center" colspan="4"><font color="#FFFFFF">Token Tracking</font></th></tr>
<tr>
<td align="center" width="25%" bgcolor="#EDE9FE"><font color="#111827"><strong>{total:,}</strong><br><sub>🧠 DEVELOPMENT TOKENS</sub></font></td>
<td align="center" width="25%" bgcolor="#DBEAFE"><font color="#111827"><strong>{summary['total_sessions']:,}</strong><br><sub>🧭 SESSIONS</sub></font></td>
<td align="center" width="25%" bgcolor="#D1FAE5"><font color="#111827"><strong>{summary['total_user_turns']:,}</strong><br><sub>💬 USER TURNS OBSERVED</sub></font></td>
<td align="center" width="25%" bgcolor="#FFEDD5"><font color="#111827"><strong>{loc['total']:,}</strong><br><sub>🧱 TRACKED TEXT LOC</sub></font></td>
</tr>
</thead>
<tbody>
<tr bgcolor="#334155"><th align="center"><font color="#FFFFFF">🤝 COLLABORATOR</font></th><th align="center"><font color="#FFFFFF">TOKENS</font></th><th align="center"><font color="#FFFFFF">SESSIONS</font></th><th align="center"><font color="#FFFFFF">USER TURNS</font></th></tr>
<tr><td align="center">Claude</td><td align="center">{claude['tokens']:,}</td><td align="center">{claude['sessions']:,}</td><td align="center">{claude['user_turns']:,}</td></tr>
<tr><td align="center">Codex</td><td align="center">{codex['tokens']:,}</td><td align="center">{codex['sessions']:,}</td><td align="center">{codex['user_turns']:,}</td></tr>
<tr bgcolor="#0E7490"><th align="center"><font color="#FFFFFF">↔️ TOKEN FLOW</font></th><th align="center"><font color="#FFFFFF">READ</font></th><th align="center"><font color="#FFFFFF">WRITTEN</font></th><th align="center"><font color="#FFFFFF">AVG / SESSION</font></th></tr>
<tr><td align="center">All sessions</td><td align="center">{read:,}</td><td align="center">{written:,}</td><td align="center">{summary['average_tokens_per_session']:,}</td></tr>
<tr bgcolor="#6D28D9"><th align="center"><font color="#FFFFFF">🔎 TOKEN DETAIL</font></th><th align="center"><font color="#FFFFFF">COUNT</font></th><th align="center"><font color="#FFFFFF">TOKEN DETAIL</font></th><th align="center"><font color="#FFFFFF">COUNT</font></th></tr>
<tr><td align="center">Fresh input</td><td align="center">{fresh:,}</td><td align="center">Cache writes</td><td align="center">{cache_write:,}</td></tr>
<tr><td align="center">Cache reads</td><td align="center">{cache_read:,}</td><td align="center">Model output</td><td align="center">{written:,}</td></tr>
<tr><td align="center">Reasoning output subset</td><td align="center">{codex['reasoning_output_tokens']:,}</td><td align="center">Counted again in total</td><td align="center">No</td></tr>
<tr bgcolor="#0369A1"><th align="center"><font color="#FFFFFF">⏱️ SESSION DURATION</font></th><th align="center"><font color="#FFFFFF">CLAUDE AVG</font></th><th align="center"><font color="#FFFFFF">CODEX AVG</font></th><th align="center"><font color="#FFFFFF">COMBINED AVG</font></th></tr>
<tr><td align="center">First-to-last observed event</td><td align="center">{claude['duration_seconds'] // claude['sessions'] // 3600 if claude['sessions'] else 0}h {(claude['duration_seconds'] // claude['sessions'] % 3600) // 60 if claude['sessions'] else 0}m</td><td align="center">{codex['duration_seconds'] // codex['sessions'] // 3600 if codex['sessions'] else 0}h {(codex['duration_seconds'] // codex['sessions'] % 3600) // 60 if codex['sessions'] else 0}m</td><td align="center">{average_duration}</td></tr>
<tr bgcolor="#C2410C"><th align="center"><font color="#FFFFFF">📚 TRACKED TEXT</font></th><th align="center"><font color="#FFFFFF">LINES</font></th><th align="center"><font color="#FFFFFF">TRACKED TEXT</font></th><th align="center"><font color="#FFFFFF">LINES</font></th></tr>
<tr><td align="center">Application source</td><td align="center">{loc['source']:,}</td><td align="center">Tests</td><td align="center">{loc['tests']:,}</td></tr>
<tr><td align="center">Documentation</td><td align="center">{loc['docs']:,}</td><td align="center">Tooling / configuration</td><td align="center">{loc['tooling_config']:,}</td></tr>
<tr><td align="center">Other text</td><td align="center">{loc['other']:,}</td><td align="center">Total physical lines</td><td align="center">{loc['total']:,}</td></tr>
</tbody>
</table>
<sub>[How these privacy-safe project metrics are counted →](docs/operations/development-token-stats.md)</sub>
</div>
{README_END}"""


def _replace_marked(text: str, replacement: str) -> str:
    start = text.find(README_START)
    end = text.find(README_END)
    if (start < 0) != (end < 0) or (start >= 0 and end < start):
        raise ValueError("README development-token markers are incomplete")
    if start >= 0:
        end += len(README_END)
        text = text[:start] + text[end:]
    anchor_start = text.find(README_SECTION_ANCHOR)
    if anchor_start < 0:
        raise ValueError("README source-to-course heading was not found")
    before = text[:anchor_start].rstrip()
    after = text[anchor_start:].lstrip("\n")
    return before + "\n\n" + replacement + "\n\n" + after


def _comparable(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "generated_at_utc"}


def _validate_summary(summary: object) -> dict[str, Any]:
    """Validate the public aggregate without consulting private session logs."""
    if not isinstance(summary, dict) or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"aggregate must use schema_version {SCHEMA_VERSION}")
    if not isinstance(summary.get("generated_at_utc"), str):
        raise ValueError("aggregate requires generated_at_utc")
    if summary.get("scope") != "Ed4All development sessions":
        raise ValueError("aggregate has an unexpected scope")

    sources = summary.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"claude", "codex"}:
        raise ValueError("aggregate sources must contain exactly claude and codex")
    for name in ("claude", "codex"):
        source = sources[name]
        if not isinstance(source, dict):
            raise ValueError(f"aggregate source {name!r} must be an object")
        for field in _empty_source():
            value = source.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"aggregate {name}.{field} must be a non-negative integer")

    total_tokens = sum(source["tokens"] for source in sources.values())
    total_sessions = sum(source["sessions"] for source in sources.values())
    total_user_turns = sum(source["user_turns"] for source in sources.values())
    expected_average = total_tokens // total_sessions if total_sessions else 0
    expected = {
        "total_tokens": total_tokens,
        "total_sessions": total_sessions,
        "average_tokens_per_session": expected_average,
        "total_user_turns": total_user_turns,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"aggregate {field} is inconsistent with its sources")

    lines = summary.get("repository_lines")
    if not isinstance(lines, dict):
        raise ValueError("aggregate repository_lines must be an object")
    for field in (*LOC_CATEGORIES, "total"):
        value = lines.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"aggregate repository_lines.{field} must be a non-negative integer")
    if lines["total"] != sum(lines[field] for field in LOC_CATEGORIES):
        raise ValueError("aggregate repository_lines.total is inconsistent")
    return summary


def _check_rendered(repo: Path) -> int:
    """Validate publishable tracker state without reading private logs."""
    aggregate_path = repo / DEFAULT_AGGREGATE
    readme_path = repo / DEFAULT_README
    try:
        summary = _validate_summary(json.loads(aggregate_path.read_text(encoding="utf-8")))
        if summary["repository_lines"] != collect_loc(repo):
            print("Development-token tracked LOC is stale; run the updater.", file=sys.stderr)
            return 1
        readme = readme_path.read_text(encoding="utf-8")
        if readme != _replace_marked(readme, render_readme(summary)):
            print("README Token Tracking section is stale; run the updater.", file=sys.stderr)
            return 1
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"Development-token rendered check failed: {exc}", file=sys.stderr)
        return 2
    print("Rendered development-token statistics are current.")
    return 0


def _install_hook(repo: Path, external: Path | None = None) -> int:
    hook = repo / ".git" / "hooks" / "pre-push"
    command = "python3 scripts/ops/update_development_tokens.py"
    if external is not None:
        command += f" --external {shlex.quote(str(external.expanduser().resolve()))}"
    command += " --check\n"
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
    parser.add_argument(
        "--check-rendered",
        action="store_true",
        help="validate aggregate schema, README rendering, and tracked LOC without session logs",
    )
    parser.add_argument("--install-hook", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.install_hook:
        return _install_hook(repo, args.external)
    if args.check_rendered:
        return _check_rendered(repo)
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
