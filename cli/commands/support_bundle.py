"""``ed4all support-bundle`` — assemble a redacted diagnostics tarball (OP1).

A support bundle is the single artifact an operator hands to a maintainer when
a run misbehaves. It packages the *diagnostic* surface of a deployment — never
the confidential surface:

INCLUDED
    * ``doctor.json`` — the ``ed4all doctor`` diagnostics, run IN-PROCESS at
      bundle time (post-mortem group when ``--run-id`` is given, else the
      live-env gpu/window/environment groups).
    * ``run/<run_id>/`` — one run's ``state/runs/<id>/`` tree (checkpoints,
      ``vram_trajectory.jsonl``, ``llm_usage.jsonl``, decisions, audit). When
      ``--run-id`` is omitted the newest run dir is chosen.
    * ``gui-logs/`` — ``state/gui/logs/*.log`` (run consoles tailed by the GUI).
    * ``captures/`` — decision-capture ``*.jsonl`` — ONLY under
      ``--include-captures`` (off by default): capture rationales can quote
      verbatim source text, so they are excluded unless the operator opts in
      and accepts the warning.
    * ``manifest.json`` — every included file with its size + sha256.

NEVER INCLUDED / REDACTED
    * ``secrets.json`` / ``.env`` / ``.env.rendered`` / ``*.key`` / ``*.pem``
      — dropped outright (secret-only files).
    * Any included ``*.json`` is walked and every secret-shaped key
      (``*_API_KEY`` / ``*_KEY`` / ``*_TOKEN`` / ``*_SECRET`` / ``*_PASSWORD``
      and the bare ``authorization`` / ``password`` / ``secret`` / ``token``
      keys) has its value replaced with ``"***REDACTED***"`` — defense in depth
      so a stray credential in a config snapshot never rides along.
    * Course CONTENT (``LibV2/courses/``, ``Courseforge/exports/``) is never
      walked, so it is excluded by construction.

The bundle is best-effort: a missing run dir / gui-logs dir degrades to a
warning recorded in the manifest rather than an error.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import click

from lib.paths import STATE_PATH, get_training_captures_dir

# --------------------------------------------------------------------------- #
# Redaction policy
# --------------------------------------------------------------------------- #

#: Files whose ENTIRE contents are a credential — dropped, never bundled.
_SECRET_ONLY_BASENAMES = {
    "secrets.json",
    ".env",
    ".env.rendered",
    ".env.local",
    ".env.rendered.json",
}

#: Suffixes for key material — dropped, never bundled.
_SECRET_SUFFIXES = (".key", ".pem", ".pfx", ".p12", ".crt")

#: Placeholder written over any redacted secret value.
_REDACTED = "***REDACTED***"

#: Bare (whole-key) secret names, matched case-insensitively.
_BARE_SECRET_KEYS = {"authorization", "password", "secret", "token", "api_key"}

#: Secret-shaped key SUFFIXES (upper-cased key ``endswith`` one of these).
_SECRET_KEY_SUFFIXES = ("_API_KEY", "_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def _is_secret_key(key: str) -> bool:
    """True for env/config keys whose VALUE is a credential.

    Broadens ``gui.settings_store._is_secret_key`` (``*_API_KEY`` / ``*_KEY``)
    with the ``*_TOKEN`` / ``*_SECRET`` / ``*_PASSWORD`` families plus the bare
    ``authorization`` / ``password`` / ``secret`` / ``token`` / ``api_key``
    keys, so a redacted snapshot is conservative.
    """
    upper = str(key).upper()
    if upper in {k.upper() for k in _BARE_SECRET_KEYS}:
        return True
    return any(upper.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)


def _redact_json_obj(obj):
    """Recursively replace every secret-shaped value with the placeholder.

    Only string/scalar values under a secret-shaped KEY are masked; the
    structure is otherwise preserved so the snapshot stays legible.
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if _is_secret_key(key) and not isinstance(value, (dict, list)):
                # Preserve the "is it set?" signal without leaking the value.
                out[key] = None if value in (None, "") else _REDACTED
            else:
                out[key] = _redact_json_obj(value)
        return out
    if isinstance(obj, list):
        return [_redact_json_obj(v) for v in obj]
    return obj


def redact_file(path: Path, data: bytes) -> Optional[bytes]:
    """Return the bytes to bundle for ``path``, or ``None`` to EXCLUDE it.

    * Secret-only files (``secrets.json`` / ``.env*`` / ``*.key`` …) → ``None``.
    * ``*.json`` → parsed and secret-shaped keys masked (unparseable JSON is
      passed through verbatim — it is not a credential store we understand).
    * Everything else → returned unchanged.
    """
    name = path.name.lower()
    if name in _SECRET_ONLY_BASENAMES:
        return None
    if any(name.endswith(suffix) for suffix in _SECRET_SUFFIXES):
        return None
    if name.endswith(".json"):
        try:
            obj = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return data
        redacted = _redact_json_obj(obj)
        return json.dumps(redacted, indent=2, sort_keys=True).encode("utf-8")
    return data


# --------------------------------------------------------------------------- #
# Bundle assembly
# --------------------------------------------------------------------------- #


@dataclass
class BundleFile:
    """One member of the support bundle."""

    arcname: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class BundleResult:
    """Outcome of a build_support_bundle_files call."""

    files: List[BundleFile] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _newest_run_dir(runs_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified ``state/runs/<id>/`` dir, or ``None``."""
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name != ".gitkeep"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _collect_tree(root: Path, arc_prefix: str) -> Tuple[List[BundleFile], List[str]]:
    """Walk ``root`` into ``BundleFile``\\ s under ``arc_prefix`` with redaction.

    Excluded (secret-only) files are silently skipped; a note is added for each
    to the returned warnings so the manifest records what was withheld.
    """
    files: List[BundleFile] = []
    warnings: List[str] = []
    if not root.exists():
        return files, warnings
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            warnings.append(f"unreadable: {path} ({exc})")
            continue
        redacted = redact_file(path, data)
        rel = path.relative_to(root).as_posix()
        arcname = f"{arc_prefix}/{rel}"
        if redacted is None:
            warnings.append(f"excluded secret-shaped file: {arcname}")
            continue
        files.append(BundleFile(arcname=arcname, data=redacted))
    return files, warnings


def collect_doctor_json(run_id: Optional[str]) -> dict:
    """Run the doctor diagnostics IN-PROCESS and return the JSON payload.

    Post-mortem group for ``--run-id`` (reads the run's checkpoints + VRAM
    trajectory off disk); otherwise the live-env gpu/window/environment groups.
    NEVER raises — any failure degrades to ``{"error": ...}`` so the bundle
    still assembles.
    """
    try:
        from cli.commands import doctor as doctor_mod  # noqa: PLC0415
        from lib.diagnostics import (  # noqa: PLC0415
            CheckContext,
            resolve_exit_code,
            resolve_verdict,
            results_to_json,
            run_checks,
        )

        doctor_mod._bootstrap_checks()
        if run_id:
            ctx = CheckContext(run_id=run_id)
            groups = ("postmortem",)
        else:
            ctx = CheckContext()
            groups = ("gpu", "window", "environment")
        results = run_checks(ctx, groups=groups)
        return {
            "results": results_to_json(results),
            "exit_code": resolve_exit_code(results),
            "summary": resolve_verdict(results),
        }
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break the bundle
        return {"error": f"{type(exc).__name__}: {exc}"}


def build_support_bundle_files(
    *,
    run_id: Optional[str] = None,
    include_captures: bool = False,
    state_root: Optional[Path] = None,
    captures_root: Optional[Path] = None,
    doctor_payload: Optional[dict] = None,
    now: Optional[float] = None,
) -> BundleResult:
    """Assemble the (redacted) list of ``BundleFile``\\ s + warnings.

    Pure/deterministic given its inputs (``doctor_payload`` is injected so the
    live diagnostics can be stubbed in tests); the CLI wrapper computes the
    doctor payload via :func:`collect_doctor_json` and writes the tarball.
    """
    if now is None:
        now = time.time()
    root = Path(state_root) if state_root else STATE_PATH
    runs_dir = root / "runs"

    result = BundleResult()

    # 1) doctor.json ------------------------------------------------------
    payload = doctor_payload if doctor_payload is not None else collect_doctor_json(run_id)
    result.files.append(
        BundleFile(
            arcname="doctor.json",
            data=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        )
    )

    # 2) the run dir (named or newest) ------------------------------------
    resolved_run_id: Optional[str] = None
    if run_id:
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            result.warnings.append(f"run dir not found: {run_dir}")
            run_dir = None
        else:
            resolved_run_id = run_id
    else:
        run_dir = _newest_run_dir(runs_dir)
        if run_dir is None:
            result.warnings.append(f"no run dirs under {runs_dir}")
        else:
            resolved_run_id = run_dir.name

    if run_dir is not None:
        files, warnings = _collect_tree(run_dir, f"run/{resolved_run_id}")
        result.files.extend(files)
        result.warnings.extend(warnings)

    # 3) GUI logs ---------------------------------------------------------
    gui_logs = root / "gui" / "logs"
    if gui_logs.exists():
        files, warnings = _collect_tree(gui_logs, "gui-logs")
        result.files.extend(files)
        result.warnings.extend(warnings)
    else:
        result.warnings.append(f"no GUI logs under {gui_logs}")

    # 4) decision captures (opt-in only) ----------------------------------
    if include_captures:
        cap_root = Path(captures_root) if captures_root else get_training_captures_dir()
        result.warnings.append(
            "WARNING: --include-captures bundled decision-capture rationales, "
            "which can quote verbatim source text. Review before sharing."
        )
        files, warnings = _collect_tree(cap_root, "captures")
        result.files.extend(files)
        result.warnings.extend(warnings)

    # 5) manifest.json ----------------------------------------------------
    manifest = {
        "kind": "ed4all-support-bundle",
        "generated_at": now,
        "run_id": resolved_run_id,
        "include_captures": bool(include_captures),
        "warnings": list(result.warnings),
        "files": [
            {"arcname": f.arcname, "size": f.size, "sha256": f.sha256}
            for f in result.files
        ],
    }
    result.files.append(
        BundleFile(
            arcname="manifest.json",
            data=json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
    )
    return result


def write_bundle(files: List[BundleFile], output: Path, *, mtime: Optional[float] = None) -> int:
    """Write ``files`` into a gzip tarball at ``output``; return its byte size."""
    if mtime is None:
        mtime = time.time()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for bf in files:
            info = tarfile.TarInfo(name=bf.arcname)
            info.size = len(bf.data)
            info.mtime = int(mtime)
            tar.addfile(info, io.BytesIO(bf.data))
    return output.stat().st_size


def _format_bytes(n: int) -> str:
    """Human-readable byte count."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB"):
        if n < step or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= step
    return f"{n} B"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command("support-bundle")
@click.option(
    "--run-id",
    "run_id",
    default=None,
    help=(
        "Bundle a specific run's state/runs/<id>/ tree and run the doctor "
        "post-mortem group against it. Default: the newest run dir + live-env "
        "doctor groups."
    ),
)
@click.option(
    "--output",
    "-o",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for the .tar.gz. Default: ./ed4all-support-<ts>.tar.gz",
)
@click.option(
    "--include-captures",
    is_flag=True,
    help=(
        "ALSO bundle decision-capture *.jsonl. OFF by default: capture "
        "rationales can quote verbatim source text — review before sharing."
    ),
)
@click.option(
    "--state-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the state/ root (tests only). Defaults to the project state/.",
)
@click.option(
    "--captures-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the training-captures/ root (tests only).",
)
def support_bundle_command(
    run_id: Optional[str],
    output: Optional[Path],
    include_captures: bool,
    state_root: Optional[Path],
    captures_root: Optional[Path],
) -> None:
    """Assemble a redacted diagnostics tarball for a maintainer.

    Packages the doctor diagnostics + one run's state tree + GUI logs (and,
    with ``--include-captures``, decision captures), redacting every
    secret-shaped file. Prints the bundle path + size.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    if output is None:
        output = Path.cwd() / f"ed4all-support-{ts}.tar.gz"

    doctor_payload = collect_doctor_json(run_id)
    built = build_support_bundle_files(
        run_id=run_id,
        include_captures=include_captures,
        state_root=state_root,
        captures_root=captures_root,
        doctor_payload=doctor_payload,
    )

    size = write_bundle(built.files, output)

    click.echo(f"Support bundle: {output}")
    click.echo(f"  Size:  {_format_bytes(size)}")
    click.echo(f"  Files: {len(built.files)}")
    if include_captures:
        click.secho(
            "  NOTE: --include-captures bundled decision-capture rationales "
            "(may quote source text). Review before sharing.",
            fg="yellow",
        )
    for warning in built.warnings:
        if warning.startswith("WARNING"):
            continue  # already surfaced above
        click.echo(f"  - {warning}")


def register_support_bundle_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all support-bundle`` command to the top-level CLI group."""
    cli_group.add_command(support_bundle_command)
