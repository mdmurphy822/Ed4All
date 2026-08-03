"""Phase 1.5 — atomically register a converted Qwen GGUF in config.yaml.

Updates ``semantik_structure/qwen_specialists/config.yaml`` so
``adapters.<name>.adapter_path`` points at a freshly-converted GGUF.
The previous config is backed up to
``models/qwen_specialists/_config_history/config.<UNIX-TS>.yaml.bak``
so a regression can be reverted in one ``cp`` command.

Atomicity:
    * Read existing YAML, edit in-memory.
    * Write to ``<config>.yaml.tmp``.
    * ``os.replace`` over the live config (POSIX-atomic).
    * Backup is taken BEFORE the rename (so a crash mid-rename still
      leaves the prior state recoverable).

CLI::

    python scripts/training/register_qwen_adapter.py \\
        --adapter math \\
        --gguf models/qwen_specialists/math/v1/math.q4_k_m.gguf \\
        [--config <override>] \\
        [--dry-run]

``--dry-run`` prints the planned update + diff without touching disk.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "semantik_structure" / "qwen_specialists" / "config.yaml"
DEFAULT_BACKUP_DIR = REPO_ROOT / "models" / "qwen_specialists" / "_config_history"

KNOWN_ADAPTERS = {"math", "table", "prose", "gap_fill"}


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _backup(config_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    bak = backup_dir / f"config.{ts}.yaml.bak"
    bak.write_bytes(config_path.read_bytes())
    return bak


def _update_config(cfg: dict, adapter: str, gguf_path: Path) -> dict:
    if adapter not in KNOWN_ADAPTERS:
        raise ValueError(
            f"unknown adapter {adapter!r}; expected one of "
            f"{sorted(KNOWN_ADAPTERS)}"
        )
    adapters = cfg.get("adapters") or {}
    if adapter not in adapters:
        raise KeyError(
            f"adapter {adapter!r} not present in config.adapters; "
            f"don't auto-create new entries — edit config.yaml by hand"
        )
    entry = dict(adapters[adapter])
    entry["adapter_path"] = str(gguf_path)
    adapters[adapter] = entry
    cfg["adapters"] = adapters
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, choices=sorted(KNOWN_ADAPTERS))
    ap.add_argument("--gguf", required=True, type=Path,
                    help="Path to the converted GGUF blob.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="Override config.yaml location (tests).")
    ap.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR,
                    help="Override backup directory.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned update; don't touch disk.")
    args = ap.parse_args()

    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    # Note: in dry-run we accept a missing GGUF (operator may be
    # planning the rename before the convert lands). In live mode we
    # require it.
    if not args.dry_run and not args.gguf.exists():
        print(f"gguf not found: {args.gguf}", file=sys.stderr)
        return 3

    raw = args.config.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw) or {}
    updated = _update_config(cfg, args.adapter, args.gguf)
    new_yaml = yaml.safe_dump(updated, sort_keys=False)

    if args.dry_run:
        print("--- dry-run: would write ---")
        print(new_yaml)
        return 0

    bak = _backup(args.config, args.backup_dir)
    print(f"[backup] {bak}")
    _atomic_write(args.config, new_yaml)
    print(f"[write] {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
