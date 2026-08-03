from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[1] / "ops" / "update_development_tokens.py"
SPEC = importlib.util.spec_from_file_location("update_development_tokens", SCRIPT)
assert SPEC and SPEC.loader
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_claude_deduplicates_streaming_snapshots_and_scopes_cwd(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    log = tmp_path / "claude.jsonl"
    base = {"sessionId": "session-a", "cwd": str(repo)}
    _write_jsonl(
        log,
        [
            {**base, "type": "user", "uuid": "turn-a", "message": {"role": "user", "content": "Please inspect this."}},
            {**base, "type": "user", "uuid": "tool-a", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ignored"}]}},
            {**base, "message": {"id": "message-a", "usage": {"input_tokens": 10, "output_tokens": 2}}},
            {**base, "message": {"id": "message-a", "usage": {"input_tokens": 10, "output_tokens": 7}}},
            {**base, "message": {"id": "message-b", "usage": {"cache_read_input_tokens": 20, "output_tokens": 3}}},
            {"sessionId": "other", "cwd": str(tmp_path / "other"), "message": {"id": "x", "usage": {"output_tokens": 999}}},
        ],
    )
    assert stats.collect_claude([log], repo) == {
        "tokens": 40, "sessions": 1, "user_turns": 1,
        "input_tokens": 10, "output_tokens": 10,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 20,
        "cached_input_tokens": 0, "reasoning_output_tokens": 0, "duration_seconds": 0,
    }


def test_codex_uses_final_cumulative_value_and_deduplicates_session(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    records = [
        {"type": "session_meta", "payload": {"session_id": "session-a", "cwd": str(repo)}},
        {"type": "event_msg", "payload": {"type": "user_message", "id": "turn-a"}},
        {"type": "event_msg", "payload": {"info": {"total_token_usage": {"total_tokens": 10, "input_tokens": 8, "output_tokens": 2}}}},
        {"type": "event_msg", "payload": {"info": {"total_token_usage": {"total_tokens": 25, "input_tokens": 20, "output_tokens": 5, "cached_input_tokens": 12, "reasoning_output_tokens": 2}}}},
    ]
    first = tmp_path / "active.jsonl"
    second = tmp_path / "archived.jsonl"
    _write_jsonl(first, records)
    _write_jsonl(second, records)
    assert stats.collect_codex([first, second], repo) == {
        "tokens": 25, "sessions": 1, "user_turns": 1,
        "input_tokens": 20, "output_tokens": 5,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "cached_input_tokens": 12, "reasoning_output_tokens": 2, "duration_seconds": 0,
    }


def test_external_export_copies_only_numeric_aggregates(tmp_path: Path) -> None:
    export = tmp_path / "desktop.json"
    export.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sources": {
                    "claude": {**stats._empty_source(), "tokens": 10, "sessions": 1, "private_path": "/not/copied"},
                    "codex": {**stats._empty_source(), "tokens": 20, "sessions": 2},
                },
                "raw_prompt": "not copied",
            }
        ),
        encoding="utf-8",
    )
    sources = {
        "claude": {**stats._empty_source(), "tokens": 1, "sessions": 1},
        "codex": {**stats._empty_source(), "tokens": 2, "sessions": 1},
    }
    stats._merge_export(sources, export)
    assert sources == {
        "claude": {**stats._empty_source(), "tokens": 11, "sessions": 2},
        "codex": {**stats._empty_source(), "tokens": 22, "sessions": 3},
    }


def test_readme_marker_update_is_stable() -> None:
    summary = {
        "generated_at_utc": "2026-08-03T00:00:00+00:00",
        "sources": {
            "claude": {**stats._empty_source(), "tokens": 3, "sessions": 1},
            "codex": {**stats._empty_source(), "tokens": 7, "sessions": 2},
        },
        "total_tokens": 10,
        "total_sessions": 3,
        "average_tokens_per_session": 3,
        "total_user_turns": 0,
        "repository_lines": {"source": 1, "tests": 2, "docs": 3, "tooling_config": 4, "other": 5, "total": 15},
    }
    rendered = stats.render_readme(summary)
    first = stats._replace_marked("# Project\n", rendered)
    assert stats._replace_marked(first, rendered) == first
    assert '"Claude" : 3' in first
    assert '"Codex" : 7' in first


def test_update_then_check_with_synthetic_empty_logs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs" / "reference").mkdir(parents=True)
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    claude = tmp_path / "claude"
    codex = tmp_path / "codex"
    claude.mkdir()
    codex.mkdir()
    args = ["--repo", str(repo), "--claude-root", str(claude), "--codex-root", str(codex)]
    assert stats.main(args) == 0
    assert stats.main([*args, "--check"]) == 0


def test_export_only_contains_no_local_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    output = tmp_path / "aggregate.json"
    assert stats.main(
        [
            "--repo",
            str(repo),
            "--claude-root",
            str(tmp_path / "claude"),
            "--codex-root",
            str(tmp_path / "codex"),
            "--export-only",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "sources": {
            "claude": stats._empty_source(),
            "codex": stats._empty_source(),
        },
    }


def test_loc_counts_tracked_text_once_and_excludes_runtime_and_binary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "docs").mkdir()
    (repo / "runtime").mkdir()
    (repo / "src" / "app.py").write_text("one\ntwo\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text("one\n", encoding="utf-8")
    (repo / "docs" / "guide.md").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (repo / "runtime" / "generated.txt").write_text("ignored\n", encoding="utf-8")
    (repo / "asset.bin").write_bytes(b"binary\0data")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    assert stats.collect_loc(repo) == {
        "source": 2, "tests": 1, "docs": 3,
        "tooling_config": 0, "other": 0, "total": 6,
    }
