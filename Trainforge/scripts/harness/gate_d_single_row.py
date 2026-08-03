"""Reviewed, one-use Gate-D authorization for one frozen micro-v1 row.

This module is deliberately transport-agnostic.  It validates the complete
eight-row authority before returning the sole authorized row and wraps the
existing micro provider's stage dispatcher with a fail-closed call budget.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import fcntl
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

GATE_D_SCHEMA = "ed4all.wp11-independent-gate-d-go.v1"
SIGNED_WRAPPER_SCHEMA = "ed4all.wp11-independent-gate-d-go-signature.v1"
CAPABILITY_SCHEMA = "ed4all.gate-d-strict-capability-proof.v1"
GATE_A_TRUST_SCHEMA = "ed4all.gate-a-wp11-trust.v1"
CONTROL_EVIDENCE_SCHEMA = "ed4all.gate-d-control-plane-evidence.v1"
SUBSET_SCHEMA = "ed4all.training-synthesis-gate-d-subset.v1"
FUNCTIONAL_PLAN_SCHEMA = "ed4all.training-synthesis-functional-v1.3.1"
FUNCTIONAL_PLAN_SEMANTIC_SHA256 = (
    "d26976363180c52f0748081db4e9cb6bcd983dfd24bcc34d2058259a363f0740"
)
FUNCTIONAL_PLAN_FILE_SHA256 = (
    "f317ec6af805f5c7d8b492dec95697f2769133f195db1a2f322424c459ef03e1"
)
ALLOWED_FAMILIES = ("A", "B", "D", "E", "F")
MAX_CALLS = 7
PINNED_GO_CANONICAL_SHA256 = (
    "cb9cc7cbe5f28b6222ab976306e87aeb5913765615ee325ba1f9ab652768944f"
)
PINNED_RELEASE_ROOT_SHA256 = (
    "61ac3959e914bf10c8481334669636218b233b206c4fb29d1f2bbda8b4a72720"
)
PINNED_AUTHORITY_SHA256 = (
    "ddda74d9b3acfaa27dd63e4b0c2653030d6767e70981bde6334e1b4406013221"
)
PINNED_TUPLE_SHA256 = (
    "7a9f3cdbc7d961777cf20395f95e95314df38596902ed140393242f4e4225f93"
)
PINNED_CONTRACT_SHA256 = (
    "5442ea5d1febd0fe4d1750f8757da65c0f18132efc2756082bf1020974d02315"
)
PINNED_ROW = {
    "chunk_id": "openstax_ea2e_scan_eval_chunk_00276",
    "pair_type": "preference",
    "chunk_sha256": "b34811be73afe740d9e38d1a93f1123180f1655642960d0f4ec9eacfef2c015e",
    "objective_id": "co-155",
    "cohort_index": 23,
    "order_index": 0,
    "stratum": ["analyze", "short", "ordinary"],
    "variant": "D_production_contract",
}
PREFERENCE_PAIR_SCHEMA_ID = (
    "https://ed4all.dev/ns/knowledge/v1/preference_pair.schema.json"
)
PAIR_AUDIT_FIELDS_SCHEMA_ID = (
    "https://ed4all.dev/ns/knowledge/v1/pair_audit_fields.schema.json"
)


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _stable(value).encode()
    return hashlib.sha256(payload).hexdigest()


def _offline_pair_validator(
    schema_path: Path,
    *,
    knowledge_dir: Path | None = None,
):
    """Build the Gate-D pair validator from local, exact-$id resources only."""
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schema_path = schema_path.resolve()
    knowledge_dir = (
        knowledge_dir.resolve()
        if knowledge_dir is not None
        else schema_path.parent
    )
    if not schema_path.is_file() or not knowledge_dir.is_dir():
        raise ValueError("Gate D local schema authority is missing")
    if schema_path != (knowledge_dir / "preference_pair.schema.json").resolve():
        raise ValueError("Gate D preference schema path is not canonical")

    resources: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for path in sorted(knowledge_dir.glob("*.schema.json")):
        try:
            raw = path.read_bytes()
            schema = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Gate D local schema is unreadable: {path.name}"
            ) from exc
        schema_id = schema.get("$id") if isinstance(schema, Mapping) else None
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError(f"Gate D local schema has no exact $id: {path.name}")
        if schema_id in resources:
            raise ValueError(f"Gate D duplicate local schema $id: {schema_id}")
        resources[schema_id] = (
            dict(schema), path.resolve(), hashlib.sha256(raw).hexdigest(),
        )

    required = {
        PREFERENCE_PAIR_SCHEMA_ID: schema_path,
        PAIR_AUDIT_FIELDS_SCHEMA_ID:
            (knowledge_dir / "pair_audit_fields.schema.json").resolve(),
    }
    for schema_id, expected_path in required.items():
        resolved = resources.get(schema_id)
        if resolved is None:
            raise ValueError(f"Gate D required local schema $id is missing: {schema_id}")
        if resolved[1] != expected_path:
            raise ValueError(
                f"Gate D local schema $id/path binding mismatched: {schema_id}"
            )
    root = resources[PREFERENCE_PAIR_SCHEMA_ID][0]
    for schema_id in required:
        schema = resources[schema_id][0]
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError(
                f"Gate D local schema dialect mismatched: {schema_id}"
            )
        Draft202012Validator.check_schema(schema)
    registry = Registry().with_resources([
        (
            schema_id,
            Resource.from_contents(
                schema, default_specification=DRAFT202012,
            ),
        )
        for schema_id, (schema, _path, _digest) in resources.items()
    ])
    evidence = {
        "schema": "ed4all.gate-d-local-schema-registry.v1",
        "root_id": PREFERENCE_PAIR_SCHEMA_ID,
        "resources": {
            schema_id: {
                "path": str(path.relative_to(knowledge_dir)),
                "sha256": digest,
                "dialect": schema.get("$schema"),
            }
            for schema_id, (schema, path, digest) in sorted(resources.items())
        },
    }
    evidence["sha256"] = _sha(evidence)
    return Draft202012Validator(root, registry=registry), evidence


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def row_identity(row: Mapping[str, Any], position: int) -> dict[str, Any]:
    return {
        "position": position,
        "chunk_id": row.get("chunk_id"),
        "chunk_sha256": row.get("chunk_sha256"),
        "kind": row.get("kind"),
        "variant": row.get("variant"),
        "repetition": row.get("repetition"),
        "focus_objective_sha256": _sha(row.get("focus_objective")),
    }


def secure_read(
    path: Path, *, require_owner: bool = True, require_safe_mode: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Read once through O_NOFOLLOW and bind bytes to descriptor provenance."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("authority path is not a regular file")
        if require_owner and before.st_uid != os.getuid():
            raise ValueError("authority file owner is not the current operator")
        if require_safe_mode and before.st_mode & 0o022:
            raise ValueError("authority file is group/world writable")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("authority file changed during descriptor read")
        return b"".join(chunks), {
            "device": before.st_dev, "inode": before.st_ino,
            "uid": before.st_uid, "mode": stat.S_IMODE(before.st_mode),
            "size": before.st_size,
        }
    finally:
        os.close(fd)


def collect_control_plane_evidence(
    *, endpoint: str, output_path: Path, served_model: str,
    model_revision: str, backend: str, backend_version: str,
    schema_sha256: str, max_context_tokens: int, max_output_tokens: int,
    thinking_enabled: bool, timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Observe control-plane state only; never issue a generation request."""
    if thinking_enabled is not False:
        raise ValueError("Gate D control-plane proof requires thinking=false")
    canonical = endpoint.rstrip("/")
    request = urllib.request.Request(
        canonical + "/models", headers={"Accept": "application/json"}
    )
    started = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
        status = response.status
    finished = time.monotonic_ns()
    parsed = json.loads(raw)
    served = {
        str(row.get("id")) for row in parsed.get("data", [])
        if isinstance(row, Mapping)
    }
    if status != 200 or served_model not in served:
        raise ValueError("control-plane /models identity mismatch")
    artifact = {
        "schema": CONTROL_EVIDENCE_SCHEMA,
        "endpoint": canonical,
        "models_url": canonical + "/models",
        "models_status": status,
        "models_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "models_json_sha256": _sha(parsed),
        "served_model": served_model,
        "model_revision": model_revision,
        "backend": backend,
        "backend_version": backend_version,
        "schema_sha256": schema_sha256,
        "max_context_tokens": int(max_context_tokens),
        "max_output_tokens": int(max_output_tokens),
        "thinking_enabled": False,
        "observed_monotonic_ns": finished,
        "request_started_monotonic_ns": started,
        "observed_utc_ns": time.time_ns(),
        "generation_requests": 0,
    }
    _atomic_json(output_path, artifact)
    return artifact


def collect_functional_preflight(
    *, endpoint: str, backend_config_path: Path, schema_path: Path,
    output_dir: Path, expected_model: str, timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Persist and derive the current non-inference functional service state."""
    output_dir.mkdir(parents=True, exist_ok=False)
    config_raw, config_provenance = secure_read(
        backend_config_path, require_safe_mode=False,
    )
    schema_raw, schema_provenance = secure_read(
        schema_path, require_safe_mode=False,
    )
    _validator, schema_registry = _offline_pair_validator(
        Path("schemas/knowledge/preference_pair.schema.json"),
    )
    if (
        schema_registry["resources"][PREFERENCE_PAIR_SCHEMA_ID]["sha256"]
        != hashlib.sha256(schema_raw).hexdigest()
    ):
        raise ValueError(
            "functional projection schema differs from local registry root"
        )
    config = json.loads(config_raw)
    canonical = endpoint.rstrip("/")

    def observe_models() -> tuple[bytes, Any, int]:
        request = urllib.request.Request(
            canonical + "/models", headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read(), response.status, time.monotonic_ns()

    first_raw, first_status, first_ns = observe_models()
    second_raw, second_status, second_ns = observe_models()
    first = json.loads(first_raw)
    second = json.loads(second_raw)
    if first_status != 200 or second_status != 200 or first != second:
        raise ValueError("functional preflight /models re-observation drifted")
    model_rows = [
        row for row in first.get("data", []) if isinstance(row, Mapping)
        and row.get("id") == expected_model
    ]
    if len(model_rows) != 1:
        raise ValueError("functional preflight served model is absent or ambiguous")
    required = {
        "served_model", "model_revision", "backend", "backend_version",
        "max_context_tokens", "max_output_tokens", "strict_dialect",
        "thinking_enabled", "health", "capacity_available", "active_clients",
        "workflow_paused", "stop_sentinel_clear", "tokenizer_identity",
    }
    if set(config) < required:
        raise ValueError("functional backend config evidence is incomplete")
    if (
        config["served_model"] != expected_model
        or config["thinking_enabled"] is not False
        or config["strict_dialect"] != "openai_json_schema_strict"
        or int(config["max_context_tokens"]) <= int(config["max_output_tokens"])
        or config["health"] != "ready"
        or config["capacity_available"] is not True
        or int(config["active_clients"]) != 0
        or config["workflow_paused"] is not True
        or config["stop_sentinel_clear"] is not True
    ):
        raise ValueError("functional observable service state is not canary-ready")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(mode=0o700)
    for name, raw in (
        ("models-first.bin", first_raw), ("models-second.bin", second_raw),
        ("backend-config.json", config_raw), ("projection-schema.json", schema_raw),
    ):
        path = raw_dir / name
        with path.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    artifact = {
        "schema": "ed4all.gate-d-functional-preflight.v1",
        "endpoint": canonical,
        **{key: config[key] for key in sorted(required)},
        "schema_sha256": hashlib.sha256(schema_raw).hexdigest(),
        "schema_registry_sha256": schema_registry["sha256"],
        "models_json_sha256": _sha(first),
        "raw_sources": {
            "models_first_sha256": hashlib.sha256(first_raw).hexdigest(),
            "models_second_sha256": hashlib.sha256(second_raw).hexdigest(),
            "backend_config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "projection_schema_sha256": hashlib.sha256(schema_raw).hexdigest(),
        },
        "provenance": {
            "backend_config": config_provenance,
            "projection_schema": schema_provenance,
        },
        "first_observed_monotonic_ns": first_ns,
        "second_observed_monotonic_ns": second_ns,
        "generation_requests": 0,
    }
    _atomic_json(output_dir / "preflight.json", artifact)
    return artifact


def collect_functional_postflight(
    *, endpoint: str, backend_config_path: Path,
    preflight: Mapping[str, Any], output_dir: Path,
    last_terminal_monotonic_seconds: float, timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Re-observe current control state after the final HTTP terminal."""
    output_dir.mkdir(parents=True, exist_ok=False)
    config_raw, _ = secure_read(backend_config_path, require_safe_mode=False)
    config = json.loads(config_raw)
    canonical = endpoint.rstrip("/")
    request = urllib.request.Request(
        canonical + "/models", headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        models_raw = response.read()
        status = response.status
    observed_monotonic_seconds = time.monotonic()
    models = json.loads(models_raw)
    model_rows = [
        row for row in models.get("data", []) if isinstance(row, Mapping)
        and row.get("id") == preflight.get("served_model")
    ]
    stable_fields = (
        "served_model", "model_revision", "backend", "backend_version",
        "max_context_tokens", "max_output_tokens", "strict_dialect",
        "thinking_enabled", "health", "capacity_available",
        "workflow_paused", "stop_sentinel_clear", "tokenizer_identity",
    )
    if (
        status != 200
        or len(model_rows) != 1
        or observed_monotonic_seconds <= float(last_terminal_monotonic_seconds)
        or any(config.get(key) != preflight.get(key) for key in stable_fields)
        or int(config.get("active_clients", -1)) != 0
    ):
        raise ValueError("functional postflight service identity/drain drifted")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(mode=0o700)
    for name, raw in (
        ("models.bin", models_raw), ("backend-config.json", config_raw),
    ):
        path = raw_dir / name
        with path.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    artifact = {
        "schema": "ed4all.gate-d-functional-postflight.v1",
        **{key: config[key] for key in (*stable_fields, "active_clients")},
        "endpoint": canonical,
        "models_json_sha256": _sha(models),
        "models_raw_sha256": hashlib.sha256(models_raw).hexdigest(),
        "backend_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "last_terminal_monotonic_seconds": float(
            last_terminal_monotonic_seconds
        ),
        "observed_monotonic_seconds": observed_monotonic_seconds,
        "observed_utc_ns": time.time_ns(),
        "generation_requests": 0,
    }
    _atomic_json(output_dir / "postflight.json", artifact)
    return artifact


class SecureOutputTree:
    """Retained dirfd chain for one private Gate-D output directory."""
    def __init__(self, *, trusted_root: Path, output_dir: Path) -> None:
        root = trusted_root.absolute()
        target = output_dir.absolute()
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise ValueError("Gate D output escapes trusted root") from exc
        self._fds: list[int] = []
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        self._fds.append(fd)
        for part in root.parts[1:]:
            fd = self._open_dir(fd, part, create=False)
            self._fds.append(fd)
        for index, part in enumerate(relative.parts):
            fd = self._open_dir(
                fd, part, create=True,
                exclusive=index == len(relative.parts) - 1,
            )
            self._fds.append(fd)
        self.fd = fd
        self.identity = self._identity(fd)
        self.target_path = target
        self.path = Path(f"/proc/self/fd/{fd}")

    @staticmethod
    def _identity(fd: int) -> tuple[int, int, int, int]:
        row = os.fstat(fd)
        if (
            not stat.S_ISDIR(row.st_mode)
            or row.st_uid not in {0, os.getuid()}
            or (
                row.st_mode & 0o022
                and not row.st_mode & stat.S_ISVTX
            )
        ):
            raise ValueError("Gate D output directory ownership/mode is unsafe")
        return row.st_dev, row.st_ino, row.st_uid, stat.S_IMODE(row.st_mode)

    def _open_dir(
        self, parent: int, name: str, *, create: bool,
        exclusive: bool = False,
    ) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        if exclusive:
            if not create:
                raise ValueError("exclusive Gate D directory must be creatable")
            os.mkdir(name, 0o700, dir_fd=parent)
            os.fsync(parent)
            child = os.open(name, flags, dir_fd=parent)
            self._identity(child)
            return child
        try:
            child = os.open(name, flags, dir_fd=parent)
        except FileNotFoundError:
            if not create:
                raise
            os.mkdir(name, 0o700, dir_fd=parent)
            os.fsync(parent)
            child = os.open(name, flags, dir_fd=parent)
        self._identity(child)
        return child

    def assert_identity(self) -> None:
        if self._identity(self.fd) != self.identity:
            raise RuntimeError("Gate D retained output directory identity drift")
        try:
            current = os.stat(self.target_path, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Gate D output directory was removed from trusted ancestry"
            ) from exc
        if (current.st_dev, current.st_ino) != self.identity[:2]:
            raise RuntimeError("Gate D output ancestry was substituted")

    def create(self, name: str, payload: bytes, *, mode: int = 0o600) -> None:
        self.assert_identity()
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError("Gate D output leaf is invalid")
        fd = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode, dir_fd=self.fd,
        )
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(self.fd)

    def reopen(self, name: str) -> tuple[bytes, dict[str, Any]]:
        self.assert_identity()
        fd = os.open(
            name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self.fd,
        )
        try:
            row = os.fstat(fd)
            if not stat.S_ISREG(row.st_mode) or row.st_uid != os.getuid():
                raise ValueError("Gate D artifact identity is unsafe")
            chunks = []
            while block := os.read(fd, 1024 * 1024):
                chunks.append(block)
            return b"".join(chunks), {
                "device": row.st_dev, "inode": row.st_ino,
                "size": row.st_size, "mode": stat.S_IMODE(row.st_mode),
            }
        finally:
            os.close(fd)

    def close(self) -> None:
        while self._fds:
            os.close(self._fds.pop())


def gate_a_trusted_output_root(path: Path) -> Path:
    raw, _ = secure_read(path)
    value = json.loads(raw)
    if value.get("schema") != GATE_A_TRUST_SCHEMA:
        raise ValueError("Gate A output-root authority is invalid")
    return Path(value["trusted_output_root"])


def _verify_external_signature(
    payload: Mapping[str, Any], signature_hex: str, key_path: Path,
    expected_fingerprint: str,
) -> dict[str, Any]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    key_bytes, provenance = secure_read(key_path)
    text_key = key_bytes.strip()
    if len(text_key) == 64 and all(
        chr(value).lower() in "0123456789abcdef" for value in text_key
    ):
        raw_key = bytes.fromhex(text_key.decode())
    else:
        raw_key = key_bytes
    fingerprint = hashlib.sha256(raw_key).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("WP11 trust-root fingerprint mismatch")
    Ed25519PublicKey.from_public_bytes(raw_key).verify(
        bytes.fromhex(signature_hex), _stable(payload).encode()
    )
    return {"fingerprint": fingerprint, "provenance": provenance}


def verify_wp11_authority(
    *, go_path: Path, wrapper_path: Path, capability_path: Path,
    trust_root_path: Path, gate_a_authority_path: Path,
) -> dict[str, Any]:
    """Verify the actual canonical GO through an operator-trusted wrapper."""
    go_bytes, go_prov = secure_read(go_path)
    wrapper_bytes, wrapper_prov = secure_read(wrapper_path)
    cap_bytes, cap_prov = secure_read(capability_path)
    go = json.loads(go_bytes)
    wrapper = json.loads(wrapper_bytes)
    capability = json.loads(cap_bytes)
    gate_a_bytes, gate_a_prov = secure_read(gate_a_authority_path)
    gate_a = json.loads(gate_a_bytes)
    if gate_a.get("schema") != GATE_A_TRUST_SCHEMA or set(gate_a) != {
        "schema", "tuple_sha256", "authority_sha256",
        "wp11_public_key_sha256", "trusted_output_root",
    }:
        raise ValueError("Gate A WP11 trust authority shape is invalid")
    if (
        gate_a["tuple_sha256"] != PINNED_TUPLE_SHA256
        or gate_a["authority_sha256"] != PINNED_AUTHORITY_SHA256
    ):
        raise ValueError("Gate A WP11 trust authority roots mismatch")
    expected_trust_fingerprint = gate_a["wp11_public_key_sha256"]
    trusted_output_root = Path(gate_a["trusted_output_root"]).resolve()
    if go.get("schema_version") != GATE_D_SCHEMA:
        raise ValueError("actual WP11 GO schema is invalid")
    unsigned = dict(go)
    declared = unsigned.pop("canonical_decision_sha256", None)
    canonical = _sha(unsigned)
    if canonical != declared or canonical != PINNED_GO_CANONICAL_SHA256:
        raise ValueError("actual WP11 GO canonical digest mismatch")
    if wrapper.get("schema") != SIGNED_WRAPPER_SCHEMA or set(wrapper) != {
        "schema", "payload", "signature_hex",
    }:
        raise ValueError("signed WP11 wrapper shape is invalid")
    payload = wrapper["payload"]
    required = {
        "go_canonical_sha256": PINNED_GO_CANONICAL_SHA256,
        "go_file_sha256": hashlib.sha256(go_bytes).hexdigest(),
        "release_root_sha256": PINNED_RELEASE_ROOT_SHA256,
        "authority_sha256": PINNED_AUTHORITY_SHA256,
        "tuple_sha256": PINNED_TUPLE_SHA256,
        "contract_sha256": PINNED_CONTRACT_SHA256,
        "capability_file_sha256": hashlib.sha256(cap_bytes).hexdigest(),
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"signed WP11 wrapper {key} mismatch")
    trust = _verify_external_signature(
        payload, wrapper["signature_hex"], trust_root_path,
        expected_trust_fingerprint,
    )
    if capability.get("schema") != CAPABILITY_SCHEMA or set(capability) != {
        "schema", "proof", "signature_hex",
    }:
        raise ValueError("strict capability proof shape is invalid")
    proof = capability["proof"]
    for key in (
        "served_model", "model_revision", "backend", "backend_version",
        "dialect", "schema_sha256", "max_context_tokens",
        "max_output_tokens", "thinking_enabled", "endpoint",
        "control_evidence_sha256", "observed_monotonic_ns",
        "fresh_for_ns",
    ):
        if key not in proof or proof[key] is None:
            raise ValueError(f"capability proof missing {key}")
    if (
        proof["dialect"] != "openai_json_schema_strict"
        or proof["thinking_enabled"] is not False
        or int(proof["max_context_tokens"]) <= int(proof["max_output_tokens"])
    ):
        raise ValueError("strict capability proof is not Gate-D capable")
    evidence_path = capability_path.with_name("control-plane-evidence.json")
    evidence_bytes, evidence_prov = secure_read(evidence_path)
    if hashlib.sha256(evidence_bytes).hexdigest() != proof[
        "control_evidence_sha256"
    ]:
        raise ValueError("control-plane evidence hash drift")
    evidence = json.loads(evidence_bytes)
    if (
        evidence.get("schema") != CONTROL_EVIDENCE_SCHEMA
        or evidence.get("endpoint") != proof["endpoint"]
        or evidence.get("served_model") != proof["served_model"]
        or evidence.get("model_revision") != proof["model_revision"]
        or evidence.get("backend") != proof["backend"]
        or evidence.get("thinking_enabled") is not False
    ):
        raise ValueError("control-plane evidence identity drift")
    now_ns = time.monotonic_ns()
    observed = int(proof["observed_monotonic_ns"])
    if (
        observed != int(evidence.get("observed_monotonic_ns", -1))
        or now_ns < observed
        or now_ns - observed > int(proof["fresh_for_ns"])
    ):
        raise ValueError("signed control-plane capability is stale")
    _verify_external_signature(
        proof, capability["signature_hex"], trust_root_path,
        expected_trust_fingerprint,
    )
    return {
        "go": go, "wrapper": wrapper, "capability": capability,
        "trust": trust,
        "provenance": {
            "go": go_prov, "wrapper": wrapper_prov,
            "capability": cap_prov, "gate_a": gate_a_prov,
            "control_evidence": evidence_prov,
        },
        "digests": {
            "go_file_sha256": hashlib.sha256(go_bytes).hexdigest(),
            "go_canonical_sha256": canonical,
            "wrapper_sha256": hashlib.sha256(wrapper_bytes).hexdigest(),
            "capability_sha256": hashlib.sha256(cap_bytes).hexdigest(),
        },
        "trusted_output_root": str(trusted_output_root),
    }


def authorize_single_row(
    *, rows: Sequence[Mapping[str, Any]], full_manifest_sha256: str,
    eligibility_sha256: str, ordered_identity_sha256: str,
    synthesis_seed: int, run_id: str, output_dir: Path, go_path: Path,
    wrapper_path: Path, capability_path: Path, trust_root_path: Path,
    gate_a_authority_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate full frozen-eight authority, then select exactly one row."""
    if len(rows) != 8:
        raise ValueError("Gate D requires the fully validated frozen8 cohort")
    if not all(isinstance(x, str) and len(x) == 64 for x in (
        full_manifest_sha256, eligibility_sha256, ordered_identity_sha256
    )):
        raise ValueError("Gate D authority hashes are invalid")
    authority = verify_wp11_authority(
        go_path=go_path, wrapper_path=wrapper_path,
        capability_path=capability_path, trust_root_path=trust_root_path,
        gate_a_authority_path=gate_a_authority_path,
    )
    try:
        output_dir.resolve().relative_to(
            Path(authority["trusted_output_root"])
        )
    except ValueError as exc:
        raise ValueError("Gate D output is outside Gate-A trusted root") from exc
    decision = authority["go"]
    canary = decision["canary"]
    frozen = canary["frozen_inputs"]
    reviewed = decision["reviewed_authority"]
    expected = {
        "full_manifest_sha256": (frozen["manifest_sha256"], full_manifest_sha256),
        "eligibility_sha256": (frozen["eligibility_sha256"], eligibility_sha256),
        "ordered_identity_sha256": (
            frozen["ordered_identity_sha256"], ordered_identity_sha256),
        "synthesis_seed": (canary["synthesis_seed"], synthesis_seed),
        "run_id": (canary["run_id"], run_id),
        "output_dir": (
            str((Path.cwd() / canary["output_root"]).resolve()),
            str(output_dir.resolve()),
        ),
        "release_root": (
            reviewed["release_evidence_manifest_sha256"], PINNED_RELEASE_ROOT_SHA256),
        "authority": (
            reviewed["standalone_authority_canonical_sha256"], PINNED_AUTHORITY_SHA256),
        "tuple": (reviewed["tuple_sha256"], PINNED_TUPLE_SHA256),
        "contract": (canary["contract"]["sha256"], PINNED_CONTRACT_SHA256),
    }
    for key, pair in expected.items():
        if pair[0] != pair[1]:
            raise ValueError(f"Gate D GO {key} mismatch")
    if canary["row"] != PINNED_ROW:
        raise ValueError("Gate D pinned row decision mismatch")
    positions = [
        i for i, row in enumerate(rows)
        if row.get("chunk_id") == PINNED_ROW["chunk_id"]
        and row.get("chunk_sha256") == PINNED_ROW["chunk_sha256"]
        and row.get("kind") == PINNED_ROW["pair_type"]
        and row.get("variant") == PINNED_ROW["variant"]
    ]
    if len(positions) != 1:
        raise ValueError("Gate D exact authorized row is absent or ambiguous")
    position = positions[0]
    identity = row_identity(rows[position], position)
    subset = {
        "schema": SUBSET_SCHEMA,
        "full_manifest_sha256": full_manifest_sha256,
        "eligibility_sha256": eligibility_sha256,
        "ordered_identity_sha256": ordered_identity_sha256,
        "go_artifact_sha256": authority["digests"]["go_file_sha256"],
        "go_decision_sha256": authority["digests"]["go_canonical_sha256"],
        "strict_dialect_capability_sha256": authority["digests"][
            "capability_sha256"
        ],
        "synthesis_seed": synthesis_seed,
        "run_id": run_id,
        "output_dir": str(output_dir.resolve()),
        "row_count": 1,
        "row": identity,
    }
    subset["subset_sha256"] = _sha(subset)
    return dict(rows[position]), subset


def authorize_functional_single_row(
    *, rows: Sequence[Mapping[str, Any]], full_manifest_sha256: str,
    eligibility_sha256: str, ordered_identity_sha256: str,
    synthesis_seed: int, run_id: str, output_dir: Path,
    expected_chunk_id: str, expected_chunk_sha256: str,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Standing-authorized functional selector; deliberately has no crypto gate."""
    plan_raw, _ = secure_read(plan_path, require_safe_mode=False)
    heading = b"## 1. Authority, supersession, and scope"
    offsets = [
        index for index in range(len(plan_raw))
        if plan_raw.startswith(heading, index)
        and (index == 0 or plan_raw[index - 1:index] == b"\n")
    ]
    if (
        hashlib.sha256(plan_raw).hexdigest() != FUNCTIONAL_PLAN_FILE_SHA256
        or len(offsets) != 1
        or hashlib.sha256(plan_raw[offsets[0]:]).hexdigest()
        != FUNCTIONAL_PLAN_SEMANTIC_SHA256
    ):
        raise ValueError("functional release plan bytes or semantic scope drifted")
    if len(rows) != 8:
        raise ValueError("functional Gate D requires the validated frozen8 cohort")
    hashes = (
        full_manifest_sha256, eligibility_sha256, ordered_identity_sha256,
    )
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
        for value in hashes
    ):
        raise ValueError("functional Gate D frozen input hashes are invalid")
    if synthesis_seed != 0:
        raise ValueError("functional Gate D requires explicit synthesis seed 0")
    if (
        not expected_chunk_id
        or len(expected_chunk_sha256) != 64
        or any(char not in "0123456789abcdef"
               for char in expected_chunk_sha256)
    ):
        raise ValueError("functional Gate D explicit row identity is invalid")
    positions = [
        index for index, row in enumerate(rows)
        if row.get("chunk_id") == expected_chunk_id
        and row.get("chunk_sha256") == expected_chunk_sha256
        and row.get("kind") == "preference"
        and row.get("variant") == "D_production_contract"
    ]
    if len(positions) != 1:
        raise ValueError("functional Gate D exact row is absent or ambiguous")
    position = positions[0]
    selected = rows[position]
    chunk = selected.get("_chunk")
    focus = selected.get("focus_objective")
    if not isinstance(chunk, Mapping) or not isinstance(focus, Mapping):
        raise ValueError(
            "functional Gate D row lacks rehydrated chunk/focus evidence"
        )
    from Trainforge.generators.staged_synthesis_micro import (
        micro_preference_eligibility,
    )
    eligibility = micro_preference_eligibility(chunk, focus=focus)
    if not eligibility["eligible"]:
        raise ValueError(
            "functional Gate D row is not DPO eligible: "
            f"{eligibility['reason']}"
        )
    identity = row_identity(rows[position], position)
    selector = {
        "schema": SUBSET_SCHEMA,
        "full_manifest_sha256": full_manifest_sha256,
        "eligibility_sha256": eligibility_sha256,
        "ordered_identity_sha256": ordered_identity_sha256,
        # Core consumes this established binding shape. These fields now bind
        # ordinary frozen evidence, not signatures or trust roots.
        "go_artifact_sha256": FUNCTIONAL_PLAN_FILE_SHA256,
        "go_decision_sha256": FUNCTIONAL_PLAN_SEMANTIC_SHA256,
        "strict_dialect_capability_sha256": _sha({
            "mode": "direct-observable-functional-preflight",
            "plan": FUNCTIONAL_PLAN_SEMANTIC_SHA256,
        }),
        "synthesis_seed": 0,
        "run_id": str(run_id),
        "output_dir": str(output_dir.resolve()),
        "row_count": 1,
        "row": identity,
    }
    selector["subset_sha256"] = _sha(selector)
    return dict(selected), selector


class GateDCallController:
    """Enforce one A-F traversal and atomically consume the signed GO."""
    def __init__(
        self, *, state_path: Path, binding: Mapping[str, Any],
        expected_stages: Sequence[str] | None = None,
        production_repairs: bool = False, max_calls: int = MAX_CALLS,
    ) -> None:
        self.state_path = state_path
        self.binding = dict(binding)
        self.calls: list[str] = []
        self.http_calls: list[str] = []
        self.http_call_evidence: list[dict[str, Any]] = []
        self.last_family_index = -1
        self._consumed_here = False
        self.expected_stages = list(expected_stages or [])
        self.production_repairs = bool(production_repairs)
        self.max_calls = int(max_calls)
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("state") != "unconsumed":
                raise RuntimeError("Gate D authorization has already been consumed")

    @staticmethod
    def _family(stage: str) -> str:
        if "dialect" in stage:
            raise RuntimeError("Gate D forbids an inference dialect probe")
        parts = stage.split("_")
        if len(parts) < 2 or parts[0] != "micro" or parts[1] not in ALLOWED_FAMILIES:
            raise RuntimeError(f"Gate D unknown/extra call stage: {stage}")
        return parts[1]

    def before_request(self, stage: str) -> None:
        if self.expected_stages and self.production_repairs:
            normalized = __import__("re").sub(
                r"_attempt_\d+(?=_|$)", "_attempt_1", stage,
            )
            if normalized not in self.expected_stages:
                raise RuntimeError(
                    f"Gate D production request is outside frozen schedule: {stage}"
                )
        if self.expected_stages and not self.production_repairs:
            offset = len(self.calls)
            if offset >= len(self.expected_stages) or stage != self.expected_stages[offset]:
                raise RuntimeError(
                    f"Gate D request differs from frozen schedule at {offset}"
                )
        family = self._family(stage)
        idx = ALLOWED_FAMILIES.index(family)
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(
                "Gate D production provider call ceiling is exhausted"
                if self.production_repairs
                else "Gate D eighth provider call is forbidden"
            )
        if idx < self.last_family_index:
            raise RuntimeError("Gate D stage traversal is not A-F ordered")
        if (
            not self.production_repairs
            and family != "B" and family in self.calls
        ):
            raise RuntimeError("Gate D repair/repeated stage is forbidden")
        if (
            not self.production_repairs
            and family == "B" and self.calls.count("B") >= 3
        ):
            raise RuntimeError("Gate D extra B slot is forbidden")
        self.calls.append(family)
        self.last_family_index = idx

    def http_attempt_started(self) -> None:
        """CAS unconsumed->started under a cross-process lock."""
        if self._consumed_here:
            return
        lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            state = json.loads(self.state_path.read_text())
            if state.get("state") != "unconsumed":
                raise RuntimeError("Gate D authorization replay is forbidden")
            _atomic_json(self.state_path, {
                "state": "started", "binding": self.binding,
                "calls_started": 1,
            })
            self._consumed_here = True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def install_http_attempt_hook(self, ledger: Any) -> None:
        original = ledger.record_started
        def hooked(**kwargs: Any):
            stage = str(ledger._stage.get() or "")
            if self.production_repairs:
                if len(self.http_calls) >= self.max_calls:
                    raise RuntimeError(
                        "Gate D production HTTP call ceiling is exhausted"
                    )
                normalized = __import__("re").sub(
                    r"_attempt_\d+(?=_|$)", "_attempt_1",
                    stage.removeprefix("staged_synthesis:"),
                )
                if self.expected_stages and normalized not in self.expected_stages:
                    raise RuntimeError(
                        "Gate D production HTTP stage is outside frozen schedule"
                    )
            result = original(**kwargs)
            if self.production_repairs:
                self.http_calls.append(stage)
                identity_var = getattr(ledger, "_last_identity", None)
                attempt_var = getattr(ledger, "_last_attempt", None)
                identity = (
                    identity_var.get() if identity_var is not None else None
                ) or {}
                self.http_call_evidence.append({
                    "stage": stage,
                    "attempt": (
                        attempt_var.get() if attempt_var is not None else None
                    ),
                    "request_sha256": identity.get("request_sha256"),
                    "model": identity.get("model"),
                    "endpoint": identity.get("endpoint"),
                })
            self.http_attempt_started()
            return result
        ledger.record_started = hooked

    def _functional_terminal_evidence(
        self, evidence_root: Path,
    ) -> dict[str, Any]:
        """Reconcile deterministic and provider evidence without fake calls."""
        root = Path(evidence_root)
        schedule = json.loads(
            (root / "gate-d-request-schedule.json").read_text(encoding="utf-8")
        )
        if schedule.get("binding_sha256") != self.binding.get("subset_sha256"):
            raise ValueError("functional terminal schedule binding mismatch")
        deterministic = [
            str(row["stage"]) for row in schedule.get("stages", [])
            if row.get("model_call") is False
        ]
        provider_schedule = [
            str(row["stage"]) for row in schedule.get("stages", [])
            if row.get("model_call") is True
        ]
        if deterministic != ["micro_A_task_design", "micro_C_assembly"]:
            raise ValueError("functional terminal deterministic schedule mismatch")
        if provider_schedule != self.expected_stages:
            raise ValueError("functional terminal provider schedule mismatch")

        def jsonl(path: Path) -> list[dict[str, Any]]:
            return [
                json.loads(line) for line in path.read_text(
                    encoding="utf-8"
                ).splitlines() if line.strip()
            ]

        intents = jsonl(root / "call-intents.jsonl")
        http = jsonl(root / "http_attempts.jsonl")
        starts = [
            row for row in http if row.get("event") == "http_attempt_started"
        ]
        terminals = [
            row for row in http if row.get("event") == "http_attempt_terminal"
        ]
        decision_paths = list(
            (root / "audit/decision-capture").glob("**/*.jsonl")
        )
        decisions = [
            row for path in decision_paths for row in jsonl(path)
            if row.get("decision_type") == "synthesis_provider_call"
        ]
        journal_paths = list((root / "micro-journals").glob("**/*.jsonl"))
        checkpoint_paths = list(root.glob("*.checkpoint.jsonl"))
        if len(journal_paths) != 1 or len(checkpoint_paths) != 1:
            raise ValueError("functional terminal journal/checkpoint cardinality")
        journal = jsonl(journal_paths[0])
        checkpoints = jsonl(checkpoint_paths[0])

        def normalized(value: Any) -> str:
            return __import__("re").sub(
                r"_attempt_\d+(?=_|$)", "_attempt_1",
                _functional_stage(value),
            )

        intent_stages = [normalized(row.get("stage")) for row in intents]
        start_stages = [normalized(row.get("stage")) for row in starts]
        terminal_stages = [normalized(row.get("stage")) for row in terminals]
        decision_stages = []
        for row in decisions:
            context = row.get("context")
            context = json.loads(context) if isinstance(context, str) else context
            decision_stages.append(normalized((context or {}).get("stage")))
        if not (
            intent_stages == start_stages == terminal_stages == decision_stages
            == [normalized(stage) for stage in self.http_calls]
            and len(intents) == len(starts) == len(terminals) == len(decisions)
            == len(self.http_calls)
            == len(self.http_call_evidence)
            and len(intents) <= self.max_calls
        ):
            raise ValueError(
                "functional terminal provider intent/HTTP/DC parity mismatch"
            )
        for intent, start, terminal, decision, hook in zip(
            intents, starts, terminals, decisions, self.http_call_evidence,
            strict=True,
        ):
            context = decision.get("context")
            context = json.loads(context) if isinstance(context, str) else context
            context = context or {}
            stages = {
                normalized(value) for value in (
                    intent.get("stage"), start.get("stage"),
                    terminal.get("stage"), context.get("stage"),
                    hook.get("stage"),
                )
            }
            attempts = {
                int(intent.get("logical_attempt", -1)),
                int(start.get("attempt", -2)),
                int(terminal.get("attempt", -3)),
                int(context.get("attempt", -4)),
                int(hook.get("attempt", -5)),
            }
            request_hashes = {
                str(value) for value in (
                    intent.get("request_sha256"),
                    start.get("request_sha256"),
                    terminal.get("request_sha256"),
                    context.get("intent_request_sha256"),
                    hook.get("request_sha256"),
                )
            }
            models = {
                str(value) for value in (
                    intent.get("model"), start.get("model"),
                    terminal.get("model"), context.get("model"),
                    hook.get("model"),
                )
            }
            endpoints = {
                str(value) for value in (
                    start.get("endpoint"), terminal.get("endpoint"),
                    hook.get("endpoint"),
                )
            }
            attempt = int(intent.get("logical_attempt", -1))
            expected_kind = "initial" if attempt == 1 else "repair"
            if (
                len(stages) != 1
                or len(attempts) != 1
                or len(request_hashes) != 1
                or "" in request_hashes
                or len(models) != 1
                or "" in models
                or len(endpoints) != 1
                or "" in endpoints
                or intent.get("kind") != expected_kind
                or terminal.get("http_status") != 200
                or terminal.get("finish_reason") != "stop"
                or terminal.get("exception_class") is not None
            ):
                raise ValueError(
                    "functional terminal positional provider tuple mismatch"
                )
        if any(
            stage.startswith(("micro_A_", "micro_C_"))
            for stage in intent_stages
        ):
            raise ValueError("functional terminal A/C transport/DC evidence")
        compressed = [
            stage for index, stage in enumerate(intent_stages)
            if index == 0 or stage != intent_stages[index - 1]
        ]
        if compressed != provider_schedule:
            raise ValueError(
                "functional terminal provider stage order/count mismatch"
            )
        physical_families = [self._family(stage) for stage in intent_stages]
        compact_physical_families = [
            family for index, family in enumerate(physical_families)
            if index == 0 or family != physical_families[index - 1]
        ]
        compact_controller_families = [
            family for index, family in enumerate(self.calls)
            if index == 0 or family != self.calls[index - 1]
        ]
        if (
            compact_physical_families != compact_controller_families
            or len(self.calls) > len(physical_families)
        ):
            raise ValueError("functional terminal physical call accounting mismatch")
        if any(
            terminal.get("finish_reason") != "stop"
            or terminal.get("exception_class") is not None
            for terminal in terminals
        ):
            raise ValueError("functional terminal provider call did not stop cleanly")

        previous = "0" * 64
        fingerprint = None
        from Trainforge.generators.staged_synthesis_micro import MicroResumeStore
        for sequence, row in enumerate(journal, 1):
            row_sha = row.get("row_sha256")
            if (
                row.get("sequence") != sequence
                or row.get("previous_sha256") != previous
                or row_sha != MicroResumeStore._row_hash(row)
            ):
                raise ValueError("functional terminal journal hash chain mismatch")
            current_fingerprint = row.get("contract_fingerprint")
            if fingerprint is None:
                fingerprint = current_fingerprint
            if (
                not current_fingerprint
                or current_fingerprint != fingerprint
                or (
                    row.get("store_identity", {}).get("execution_fingerprint")
                    or row.get("gate_d_binding", {}).get("subset_sha256")
                ) != self.binding.get("subset_sha256")
            ):
                raise ValueError("functional terminal journal binding mismatch")
            previous = str(row_sha)
        terminal_journal = [
            row for row in journal if row.get("state") == "terminal"
        ]
        expected_journal_stages = [
            "A",
            *(["B"] * sum(
                stage.startswith("micro_B_") for stage in provider_schedule
            )),
            "C", "D", "E", "F",
        ]
        if [
            str(row.get("stage")) for row in terminal_journal
        ] != expected_journal_stages:
            raise ValueError("functional terminal A-F journal order/cardinality")
        for stage, marker in (
            ("A", "_stage_a_deterministic"),
            ("C", "_stage_c_deterministic"),
        ):
            rows = [
                row for row in terminal_journal if row.get("stage") == stage
            ]
            artifact = rows[0].get("artifact") if len(rows) == 1 else None
            identity = artifact.get(marker) if isinstance(artifact, Mapping) else None
            telemetry = (
                identity.get("telemetry") if isinstance(identity, Mapping) else None
            )
            if (
                not isinstance(telemetry, Mapping)
                or identity.get("model_calls") != 0
                or identity.get("decision_capture_events") != 0
                or int(telemetry.get("deterministic_events", 0)) != 1
                or any(int(telemetry.get(key, -1)) != 0 for key in (
                    "model_calls", "prompt_tokens", "completion_tokens",
                    "total_tokens",
                ))
            ):
                raise ValueError(
                    f"functional terminal deterministic {stage} evidence mismatch"
                )
        if (
            len(checkpoints) != 1
            or checkpoints[0].get("_checkpoint_state") != "terminal"
            or checkpoints[0].get("accepted") is not True
            or checkpoints[0].get("stage_validity") is not True
            or checkpoints[0].get("gate_d_binding", {}).get("subset_sha256")
            != self.binding.get("subset_sha256")
        ):
            raise ValueError("functional terminal checkpoint mismatch")
        return {
            "deterministic_stages_completed": ["A", "C"],
            "journal_stages_completed": [
                str(row["stage"]) for row in terminal_journal
            ],
            "provider_stages_completed": intent_stages,
        }

    def terminal(
        self, *, outcome: str, evidence_root: Optional[Path] = None,
    ) -> str:
        if not self._consumed_here:
            return "unconsumed"
        reconciliation: dict[str, Any] = {}
        reconciliation_error: Optional[str] = None
        if (
            outcome == "completed"
            and self.production_repairs
            and evidence_root is not None
        ):
            try:
                reconciliation = self._functional_terminal_evidence(evidence_root)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                reconciliation_error = str(exc)
        complete_provider_traversal = (
            bool(self.calls)
            and (
                self.calls[0] == "A"
                if not self.production_repairs
                else self.calls[0] == "B"
            )
            and self.calls.count("B") >= 1
            and self.calls.count("D") >= 1
            and self.calls.count("E") >= 1
            and self.calls.count("F") >= 1
            and (
                set(self.calls) == set(ALLOWED_FAMILIES)
                if not self.production_repairs
                else set(self.calls) == {"B", "D", "E", "F"}
            )
            and len(self.calls) <= self.max_calls
            and (
                self.production_repairs
                or (
                not self.expected_stages
                or len(self.calls) == len(self.expected_stages)
                )
            )
        )
        complete_traversal = (
            complete_provider_traversal
            and (
                not self.production_repairs
                or evidence_root is not None
                and reconciliation_error is None
                and bool(reconciliation)
            )
        )
        if outcome == "completed" and not complete_traversal:
            outcome = "failed_incomplete_traversal"
        _atomic_json(self.state_path, {
            "state": "terminal", "outcome": outcome,
            "binding": self.binding,
            "calls_started": (
                len(self.http_calls) if self.production_repairs
                else len(self.calls)
            ),
            "families": self.calls,
            **reconciliation,
            **({
                "reconciliation_error": reconciliation_error,
            } if reconciliation_error else {}),
            **({
                "http_stages": self.http_calls,
                "maximum_model_calls": self.max_calls,
            } if self.production_repairs else {}),
        })
        return outcome

    def wrap(self, provider: Any) -> Any:
        original = provider._call_stage
        def controlled(*, stage: str, **kwargs: Any):
            if (
                not self.production_repairs
                and (
                    kwargs.get("max_stage_repairs", 0)
                    or kwargs.get("max_leakage_repairs", 0)
                )
            ):
                raise RuntimeError("Gate D semantic repairs are forbidden")
            self.before_request(stage)
            if not self.production_repairs:
                kwargs["max_stage_repairs"] = 0
                kwargs["max_leakage_repairs"] = 0
            return original(stage=stage, **kwargs)
        provider._call_stage = controlled
        return provider


def write_unconsumed(path: Path, binding: Mapping[str, Any]) -> None:
    """Record successful zero-traffic preflight without consuming the GO."""
    if path.exists():
        raise RuntimeError("Gate D state already exists")
    _atomic_json(path, {"state": "unconsumed", "binding": dict(binding)})


def verify_gate_d_precommit(
    candidate_path: Path, *, expected_binding_sha256: str,
    consumption_path: Path,
) -> dict[str, Any]:
    """Re-open and verify the immutable candidate before publication."""
    candidate_bytes, provenance = secure_read(candidate_path)
    candidate = json.loads(candidate_bytes)
    required = {
        "schema", "binding_sha256", "policy_sha256", "schedule_sha256",
        "results_sha256", "summary_sha256", "consumption_sha256",
        "schema_registry_sha256",
    }
    if set(candidate) != required or candidate["schema"] != (
        "ed4all.gate-d-precommit-candidate.v1"
    ):
        raise ValueError("Gate D precommit candidate shape is invalid")
    if candidate["binding_sha256"] != expected_binding_sha256:
        raise ValueError("Gate D precommit binding mismatch")
    _validator, schema_registry = _offline_pair_validator(
        Path("schemas/knowledge/preference_pair.schema.json"),
    )
    if candidate["schema_registry_sha256"] != schema_registry["sha256"]:
        raise ValueError("Gate D precommit schema registry mismatch")
    consumption_bytes, _ = secure_read(consumption_path)
    if hashlib.sha256(consumption_bytes).hexdigest() != candidate[
        "consumption_sha256"
    ]:
        raise ValueError("Gate D precommit consumption hash mismatch")
    state = json.loads(consumption_bytes)
    if state.get("state") != "terminal" or state.get("outcome") != "completed":
        raise ValueError("Gate D consumption is not terminal-completed")
    return {
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_provenance": provenance,
        "binding_sha256": expected_binding_sha256,
        "verified": True,
    }


def _functional_stage(value: Any) -> str:
    stage = str(value or "")
    if stage.startswith("staged_synthesis:"):
        stage = stage.split(":", 1)[1]
    if not stage.startswith("micro_"):
        raise ValueError(f"functional Gate D stage is not canonical: {value!r}")
    return stage


def functional_reasoning_bytes(raw_responses: Sequence[bytes]) -> int:
    """Measure explicit/visible reasoning from persisted HTTP response bytes."""
    total = 0
    marker = __import__("re").compile(rb"<think(?:\s[^>]*)?>.*?</think\s*>", __import__("re").I | __import__("re").S)
    def visit(value: Any) -> None:
        nonlocal total
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"reasoning_content", "reasoning"} and isinstance(
                    child, str,
                ):
                    total += len(child.encode("utf-8"))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for raw in raw_responses:
        try:
            visit(json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        total += sum(len(match.group(0)) for match in marker.finditer(raw))
    return total


def _verify_functional_reconciliation(
    *, schedule_doc: Mapping[str, Any], candidate: Mapping[str, Any],
    scheduled: Sequence[str], intents: Sequence[Mapping[str, Any]],
    http: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]],
    by_rel: Mapping[str, bytes], decision_rows: Sequence[Mapping[str, Any]],
    journal_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any], result_row: Mapping[str, Any],
    expected_binding_sha256: str, preflight: Mapping[str, Any],
) -> None:
    """Exact v1.3 functional transaction reconciliation."""
    policy = dict(schedule_doc.get("policy") or {})
    production_repairs = (
        schedule_doc.get("functional_policy_mode") == "production-repair-loops"
    )
    declared_policy_sha = policy.pop("sha256", None)
    recomputed_policy_sha = _sha(policy)
    if (
        declared_policy_sha != recomputed_policy_sha
        or candidate.get("policy_sha256") != recomputed_policy_sha
        or policy.get("version") != (
            "legacy-repair-loops.v1"
            if production_repairs else "gate-d-single-pass.v1"
        )
        or (
            not production_repairs
            and policy.get("trusted_binding_sha256")
            != expected_binding_sha256
        )
    ):
        raise ValueError("functional Gate D policy identity drifted")
    planned_schedule = [_functional_stage(value) for value in scheduled]
    canonical_intents = [_functional_stage(row.get("stage")) for row in intents]
    starts = [row for row in http if row.get("event") == "http_attempt_started"]
    terminals = [
        row for row in http if row.get("event") == "http_attempt_terminal"
    ]
    if [row.get("event") for row in http] != [
        event for _ in canonical_intents
        for event in ("http_attempt_started", "http_attempt_terminal")
    ]:
        raise ValueError("functional HTTP start/terminal ordering drifted")
    if not (
        canonical_intents
        == [_functional_stage(row.get("stage")) for row in starts]
        == [_functional_stage(row.get("stage")) for row in terminals]
    ):
        raise ValueError("functional schedule/intent/HTTP stage bijection drifted")
    raw_hashes = {
        item["sha256"]: item["path"] for item in inventory
        if "http-raw" in item["path"]
    }
    stage_budgets = {"A": 2048, "B": 1536, "D": 1536, "E": 1280, "F": 1024}
    models: set[str] = set()
    expected_model = str(preflight.get("served_model") or "")
    expected_revision = str(preflight.get("model_revision") or "")
    for index, (intent, start, terminal) in enumerate(
        zip(intents, starts, terminals, strict=True)
    ):
        stage = canonical_intents[index]
        match = __import__("re").search(r"micro_([ABDEF])(?:_|$)", stage)
        if not match:
            raise ValueError("functional schedule contains an unplanned model stage")
        if (
            int(intent.get("logical_attempt", -1)) < 1
            or int(start.get("attempt", -1))
            != int(intent.get("logical_attempt", -2))
            or int(terminal.get("attempt", -1))
            != int(intent.get("logical_attempt", -2))
            or intent.get("request_sha256") != start.get("request_sha256")
            or start.get("request_sha256")
            != (start.get("request_raw_ref") or {}).get("sha256")
            or start.get("request_sha256") not in raw_hashes
            or (terminal.get("response_raw_ref") or {}).get("sha256")
            not in raw_hashes
            or terminal.get("finish_reason") != "stop"
            or terminal.get("exception_class") is not None
            or int(intent.get("max_tokens", -1)) != stage_budgets[match.group(1)]
            or intent.get("model") != expected_model
            or start.get("model") != expected_model
            or terminal.get("model") != expected_model
        ):
            raise ValueError("functional intent/HTTP identity or outcome drifted")
        raw_request = by_rel[raw_hashes[start["request_sha256"]]]
        if hashlib.sha256(raw_request).hexdigest() != intent["request_sha256"]:
            raise ValueError("functional request hash differs from raw bytes")
        payload = json.loads(raw_request)
        if (
            payload.get("model") != intent.get("model")
            or int(payload.get("max_tokens", -1)) != int(intent["max_tokens"])
            or not isinstance(payload.get("response_format"), Mapping)
            or "json_schema" not in payload["response_format"]
            or payload.get("chat_template_kwargs", {}).get("enable_thinking")
            is not False
        ):
            raise ValueError("functional request lacks model/schema/budget/thinking")
        for row in (intent, start, terminal, payload):
            if (
                row.get("model_revision") is not None
                and str(row.get("model_revision")) != expected_revision
            ):
                raise ValueError("functional request model revision drifted")
        models.add(str(intent.get("model") or ""))
    decisions = [
        row for row in decision_rows
        if row.get("decision_type") == "synthesis_provider_call"
    ]
    if len(decisions) != len(intents):
        raise ValueError("functional DecisionCapture count differs from intents")
    artifact_hashes = {item["sha256"] for item in inventory}
    for index, decision in enumerate(decisions):
        context = decision.get("context")
        if isinstance(context, str):
            context = json.loads(context)
        context = dict(context or {})
        intent = intents[index]
        execution_policy = dict(context.get("execution_policy") or {})
        validation_evidence = dict(context.get("validation_evidence") or {})
        if (
            _functional_stage(context.get("stage")) != canonical_intents[index]
            or int(context.get("attempt", -1))
            != int(intent.get("logical_attempt", -2))
            or context.get("intent_request_sha256") != intent["request_sha256"]
            or context.get("intent_model") != intent.get("model")
            or context.get("model") != expected_model
            or context.get("intent_contract_sha256")
            != intent.get("contract_sha256")
            or (
                not production_repairs
                and (
                context.get("trusted_binding_sha256")
                or execution_policy.get("trusted_binding_sha256")
                )
                != expected_binding_sha256
            )
            or (
                context.get("response_schema_sha256")
                or validation_evidence.get("response_schema_sha256")
            )
            != intent.get("response_schema_sha256")
            or context.get("prompt_sha256") not in artifact_hashes
            or context.get("response_sha256") not in artifact_hashes
            or len(str(decision.get("rationale") or "")) < 20
        ):
            raise ValueError("functional DecisionCapture bijection/identity drifted")
        if (
            context.get("model_revision") is not None
            and str(context.get("model_revision")) != expected_revision
        ):
            raise ValueError("functional DecisionCapture model revision drifted")
    previous = "0" * 64
    terminals_by_unit: list[Mapping[str, Any]] = []
    for sequence, row in enumerate(journal_rows, 1):
        unsigned = dict(row)
        observed_hash = unsigned.pop("row_sha256", None)
        if (
            int(row.get("sequence", -1)) != sequence
            or row.get("previous_sha256") != previous
            or observed_hash != _sha(unsigned)
            or row.get("attempt") != 1
            or (
                not production_repairs
                and expected_binding_sha256
                not in json.dumps(row, sort_keys=True)
            )
        ):
            raise ValueError("functional micro journal hash chain drifted")
        previous = str(observed_hash)
        if row.get("state") == "terminal":
            terminals_by_unit.append(row)
    expected_journal = [
        __import__("re").search(
            r"micro_([ABCDEF])", str(row["stage"]),
        ).group(1)
        for row in schedule_doc["stages"]
    ]
    observed_journal = [str(row.get("stage")) for row in terminals_by_unit]
    if observed_journal != expected_journal:
        raise ValueError("functional journal stage/slot order differs from schedule")
    b_slots = [
        int(row.get("slot")) for row in terminals_by_unit if row.get("stage") == "B"
    ]
    expected_slots = [
        int(__import__("re").search(r"claim_(\d+)", stage).group(1))
        for stage in planned_schedule if "micro_B_" in stage
    ]
    if b_slots != expected_slots or len(set(b_slots)) != len(b_slots):
        raise ValueError("functional journal B-slot identity drifted")
    if len(checkpoint_rows) != 1:
        raise ValueError("functional transaction requires one checkpoint terminal row")
    checkpoint = checkpoint_rows[0]
    if (
        checkpoint.get("_checkpoint_state", checkpoint.get("state")) != "terminal"
        or checkpoint.get("gate_d_binding", {}).get("subset_sha256")
        != expected_binding_sha256
        or checkpoint.get("result") != result_row.get("result", result_row)
    ):
        raise ValueError("functional checkpoint/result projection differs")
    sampler = dict(telemetry.get("sampler_state") or {})
    stage_counts = telemetry.get("stage_request_counts") or {}
    if (
        int(telemetry.get("request_count", -1)) != len(intents)
        or stage_counts != {
            stage: canonical_intents.count(stage)
            for stage in set(canonical_intents)
        }
        or int(telemetry.get("active_clients_final", -1)) != 0
        or sampler.get("stop_requested") is not True
        or sampler.get("errors") not in (None, [])
        or int(telemetry.get("reasoning_bytes", -1)) != 0
        or telemetry.get("finish_reason_counts") != {"stop": len(intents)}
        or len(telemetry.get("token_observations") or []) != len(intents)
        or {
            row.get("stage") for row in telemetry.get("token_observations") or []
        } != set(canonical_intents)
        or any(
            not all(
                isinstance(row.get(key), int) and not isinstance(row.get(key), bool)
                and row[key] >= 0
                for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "max_output_tokens", "output_headroom_tokens",
                )
            )
            or row["total_tokens"]
            != row["prompt_tokens"] + row["completion_tokens"]
            for row in telemetry.get("token_observations") or []
        )
        or not telemetry.get("gpu_observations")
        or any(
            not all(key in row and isinstance(row[key], (int, float, str))
                    for key in (
                        "timestamp", "gpu_utilization_percent",
                        "memory_utilization_percent", "power_watts",
                        "temperature_c",
                    ))
            for row in telemetry.get("gpu_observations") or []
        )
        or not telemetry.get("kv_observations")
        or any(
            int(row.get("kv_blocks", 0)) <= 0
            or int(row.get("peak_scheduled_token_usage", 0)) <= 0
            or int(row.get("peak_scheduled_token_headroom", -1)) < 0
            for row in telemetry.get("kv_observations") or []
        )
        or telemetry.get("verifier_accepted") is not True
        or int(telemetry.get("abort_disconnect_count", -1)) != 0
    ):
        raise ValueError("functional telemetry is incomplete or unreconciled")


def verify_full_gate_d_transaction(
    root: Path, *, expected_stages: Sequence[str],
    expected_binding_sha256: str, candidate_path: Path,
    authority_paths: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Independently reconcile every Gate-D artifact before publication."""
    inventory = []
    by_rel: dict[str, bytes] = {}
    authority_reverification = None
    schema_registry_evidence = None
    if authority_paths is not None:
        authority_reverification = verify_wp11_authority(
            go_path=Path(authority_paths["go_path"]),
            wrapper_path=Path(authority_paths["wrapper_path"]),
            capability_path=Path(authority_paths["capability_path"]),
            trust_root_path=Path(authority_paths["trust_root_path"]),
            gate_a_authority_path=Path(
                authority_paths["gate_a_authority_path"]
            ),
        )
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError("Gate D artifact directory symlink is forbidden")
        for name in files:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError("Gate D artifact symlink is forbidden")
            raw, provenance = secure_read(path)
            rel = str(path.relative_to(root))
            by_rel[rel] = raw
            inventory.append({
                "path": rel, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "provenance": provenance,
            })
    def one(suffix: str) -> tuple[str, bytes]:
        if suffix in by_rel:
            return suffix, by_rel[suffix]
        matches = [(p, b) for p, b in by_rel.items() if p.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Gate D requires exactly one {suffix}")
        return matches[0]
    _, candidate_raw = one("gate-d-precommit-candidate.json")
    if candidate_path.read_bytes() != candidate_raw:
        raise ValueError("Gate D candidate descriptor/path identity drift")
    candidate = json.loads(candidate_raw)
    if candidate["binding_sha256"] != expected_binding_sha256:
        raise ValueError("Gate D candidate binding claim mismatch")
    _, results = one("results.jsonl")
    _, summary = one("summary.json")
    _, consumption = one("gate-d-consumption.json")
    _, schedule = one("gate-d-request-schedule.json")
    claims = {
        "results_sha256": hashlib.sha256(results).hexdigest(),
        "summary_sha256": hashlib.sha256(summary).hexdigest(),
        "consumption_sha256": hashlib.sha256(consumption).hexdigest(),
        "schedule_sha256": hashlib.sha256(schedule).hexdigest(),
    }
    if any(candidate.get(key) != value for key, value in claims.items()):
        raise ValueError("Gate D candidate claims differ from recomputed bytes")
    state = json.loads(consumption)
    if (
        state.get("state") != "terminal"
        or state.get("outcome") != "completed"
        or state.get("binding", {}).get("subset_sha256")
        != expected_binding_sha256
    ):
        raise ValueError("Gate D consumption/binding is invalid")
    schedule_doc = json.loads(schedule)
    production_repairs = (
        schedule_doc.get("functional_policy_mode") == "production-repair-loops"
    )
    scheduled = [
        row["stage"] for row in schedule_doc["stages"] if row["model_call"]
    ]
    deterministic_scheduled = [
        row["stage"] for row in schedule_doc["stages"]
        if not row["model_call"]
    ]
    functional_ac = schedule_doc.get("functional_version") == "1.3.1"
    if functional_ac and deterministic_scheduled != [
        "micro_A_task_design", "micro_C_assembly",
    ]:
        raise ValueError("Gate D deterministic A/C schedule is invalid")
    if scheduled != list(expected_stages):
        raise ValueError("Gate D persisted schedule differs from core schedule")
    _, intents_raw = one("call-intents.jsonl")
    _, http_raw = one("http_attempts.jsonl")
    intents = [
        json.loads(line) for line in intents_raw.splitlines() if line.strip()
    ]
    http = [json.loads(line) for line in http_raw.splitlines() if line.strip()]
    normalize_stage = (
        _functional_stage
        if schedule_doc.get("functional_version") == "1.3.1"
        else lambda value: str(value)
    )
    intent_stages = [normalize_stage(row.get("stage")) for row in intents]
    if functional_ac and any(
        stage.startswith(("micro_A_", "micro_C_"))
        for stage in intent_stages
    ):
        raise ValueError("Gate D deterministic A/C used model transport")
    starts = [row for row in http if row.get("event") == "http_attempt_started"]
    terminals = [
        row for row in http if row.get("event") == "http_attempt_terminal"
    ]
    start_stages = [normalize_stage(row.get("stage")) for row in starts]
    terminal_stages = [normalize_stage(row.get("stage")) for row in terminals]
    exact_transport = (
        intent_stages == start_stages == terminal_stages
        and len(starts) == len(terminals) == len(intents)
        and (
            all(
                int(intent.get("logical_attempt", -1))
                == int(start.get("attempt", -2))
                == int(terminal.get("attempt", -3))
                for intent, start, terminal in zip(
                    intents, starts, terminals, strict=True,
                )
            )
            if production_repairs
            else all(
                int(row.get("attempt", -1)) == 1
                for row in starts + terminals
            )
        )
        and all(
            row.get("finish_reason") not in {"length", "error"}
            for row in terminals
        )
        and not any(row.get("event") == "retry_backoff" for row in http)
    )
    if not (
        exact_transport
        and (
            production_repairs
            and len(intents) <= int(schedule_doc.get("maximum_model_calls", 0))
            or not production_repairs
            and intent_stages == scheduled
            and len(intents) == len(scheduled)
            and all(int(row.get("attempt", -1)) == 1
                    for row in starts + terminals)
        )
    ):
        raise ValueError("Gate D intent/HTTP order, count, attempt, or finish drift")
    for row in starts:
        ref = row.get("request_raw_ref") or {}
        if ref.get("sha256") != row.get("request_sha256"):
            raise ValueError("Gate D raw request hash binding is invalid")
        if ref.get("sha256") not in {
            item["sha256"] for item in inventory
            if "http-raw" in item["path"] and "request" in item["path"]
        }:
            raise ValueError("Gate D raw request bytes are missing or drifted")
    required_fragments = (
        "checkpoint", "micro-journals", "decision-capture",
        "telemetry/summary.json",
    )
    for fragment in required_fragments:
        if not any(fragment in path for path in by_rel):
            raise ValueError(f"Gate D evidence missing {fragment}")
    response_blobs = [
        raw for path, raw in by_rel.items()
        if "http-raw" in path and "response" in path
    ]
    if len(response_blobs) != len(intents):
        raise ValueError("Gate D raw response count is invalid")
    response_hashes = {
        hashlib.sha256(blob).hexdigest() for blob in response_blobs
    }
    for row in terminals:
        ref = row.get("response_raw_ref") or {}
        if ref.get("sha256") not in response_hashes:
            raise ValueError("Gate D raw response hash binding is invalid")
    if any(b"<think" in blob.lower() for blob in response_blobs):
        raise ValueError("Gate D reasoning leakage detected")
    decision_rows = []
    journal_rows = []
    checkpoint_rows = []
    for path, raw in by_rel.items():
        if not path.endswith(".jsonl"):
            continue
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if "decision-capture" in path:
            decision_rows.extend(rows)
        elif "micro-journals" in path:
            journal_rows.extend(rows)
        elif "checkpoint" in path:
            checkpoint_rows.extend(rows)
    provider_decisions = [
        row for row in decision_rows
        if row.get("decision_type") == "synthesis_provider_call"
    ]
    decision_parity = len(provider_decisions) == len(intents)
    if decision_parity and functional_ac:
        for intent, terminal, decision in zip(
            intents, terminals, provider_decisions, strict=True,
        ):
            context = decision.get("context")
            context = json.loads(context) if isinstance(context, str) else context
            context = context or {}
            if (
                normalize_stage(context.get("stage"))
                != normalize_stage(intent.get("stage"))
                or context.get("intent_request_sha256")
                != intent.get("request_sha256")
                or context.get("model") != intent.get("model")
                or int(context.get("attempt", -1))
                != int(intent.get("logical_attempt", -2))
                or terminal.get("request_sha256")
                != intent.get("request_sha256")
            ):
                decision_parity = False
                break
    if (
        not decision_parity
        or not functional_ac
        and any(
            expected_binding_sha256 not in json.dumps(row, sort_keys=True)
            for row in provider_decisions
        )
    ):
        raise ValueError("Gate D DecisionCapture count/binding is invalid")
    terminal_journal = [
        row for row in journal_rows if row.get("state") == "terminal"
    ]
    if (
        {str(row.get("stage")) for row in terminal_journal}
        != {"A", "B", "C", "D", "E", "F"}
        or any(expected_binding_sha256 not in json.dumps(row, sort_keys=True)
               for row in terminal_journal)
    ):
        raise ValueError("Gate D A-F journal coverage/binding is invalid")
    deterministic_artifacts = {
        str(row.get("stage")): row.get("artifact")
        for row in terminal_journal if row.get("stage") in {"A", "C"}
    }
    for stage, marker in (() if not functional_ac else (
        ("A", "_stage_a_deterministic"),
        ("C", "_stage_c_deterministic"),
    )):
        artifact = deterministic_artifacts.get(stage)
        identity = artifact.get(marker) if isinstance(artifact, Mapping) else None
        telemetry = identity.get("telemetry") if isinstance(identity, Mapping) else None
        if (
            not isinstance(telemetry, Mapping)
            or identity.get("model_calls") != 0
            or identity.get("decision_capture_events") != 0
            or any(int(telemetry.get(key, -1)) != 0 for key in (
                "model_calls", "prompt_tokens", "completion_tokens",
                "total_tokens",
            ))
            or int(telemetry.get("deterministic_events", 0)) != 1
        ):
            raise ValueError(
                f"Gate D deterministic Stage {stage} evidence is invalid"
            )
    if not checkpoint_rows or any(
        expected_binding_sha256 not in json.dumps(row, sort_keys=True)
        for row in checkpoint_rows
    ):
        raise ValueError("Gate D checkpoint binding is invalid")
    _, telemetry_raw = one("telemetry/summary.json")
    telemetry = json.loads(telemetry_raw)
    sampler = telemetry.get("sampler_state") or {}
    if (
        int(telemetry.get("abort_disconnect_count", 0)) != 0
        or sampler.get("errors") not in (None, [])
    ):
        raise ValueError("Gate D telemetry reports abort/error")
    result_rows = [
        json.loads(line) for line in results.splitlines() if line.strip()
    ]
    if len(result_rows) != 1:
        raise ValueError("Gate D must contain exactly one result row")
    text = json.dumps(result_rows[0], sort_keys=True)
    if not all(key in text for key in ("prompt", "chosen", "rejected")):
        raise ValueError("Gate D DPO projection is incomplete")
    result_row = result_rows[0]
    pair = result_row
    if isinstance(pair.get("result"), Mapping):
        pair = dict(pair["result"])
    for key in ("prompt", "chosen", "rejected"):
        if not isinstance(pair.get(key), str) or not pair[key].strip():
            raise ValueError("Gate D DPO projection field is empty")
    if pair["chosen"].strip() == pair["rejected"].strip():
        raise ValueError("Gate D DPO chosen/rejected are semantically identical")
    schema_path = Path("schemas/knowledge/preference_pair.schema.json")
    schema_raw, _ = secure_read(schema_path, require_safe_mode=False)
    if (
        authority_reverification is not None
        and hashlib.sha256(schema_raw).hexdigest()
        != authority_reverification["capability"]["proof"]["schema_sha256"]
    ):
        raise ValueError("Gate D projection schema differs from capability proof")
    pair_validator, schema_registry_evidence = _offline_pair_validator(
        schema_path,
    )
    if (
        candidate.get("schema_registry_sha256")
        != schema_registry_evidence["sha256"]
    ):
        raise ValueError("Gate D candidate schema registry binding drifted")
    pair_validator.validate(pair)
    from lib.validators.pair.objective_delivery import (
        recompute_complete_objective_bloom_authority,
    )
    bloom_alignment = pair.get("bloom_alignment")
    bloom_authority_valid = (
        bloom_alignment is True
        or (
            bloom_alignment is None
            and recompute_complete_objective_bloom_authority(pair) is not None
        )
    )
    if (
        float(pair.get("claim_support_rate", 0.0)) < 0.70
        or float(pair.get("claim_contradicted_rate", 1.0)) >= 0.50
        or not bloom_authority_valid
        or result_row.get("accepted") is not True
        or pair.get("promotion_status") not in {"validated", "trainable"}
        or (
            schedule_doc.get("functional_version") == "1.3.1"
            and (
                pair.get("projection_contract")
                != "ed4all-dpo-preference.v2"
                or pair.get("schema_version") != "v1"
            )
        )
    ):
        raise ValueError("Gate D DPO semantic validators did not pass")
    if schedule_doc.get("functional_version") == "1.3.1":
        _, preflight_raw = one("functional-preflight/preflight.json")
        preflight = json.loads(preflight_raw)
        raw_sources = dict(preflight.get("raw_sources") or {})
        required_preflight_raw = {
            "models_first_sha256": "functional-preflight/raw/models-first.bin",
            "models_second_sha256": "functional-preflight/raw/models-second.bin",
            "backend_config_sha256":
                "functional-preflight/raw/backend-config.json",
            "projection_schema_sha256":
                "functional-preflight/raw/projection-schema.json",
        }
        if (
            _sha(preflight)
            != state.get("binding", {}).get(
                "strict_dialect_capability_sha256"
            )
            or any(
                raw_sources.get(key)
                != hashlib.sha256(by_rel[path]).hexdigest()
                for key, path in required_preflight_raw.items()
            )
            or preflight.get("thinking_enabled") is not False
            or preflight.get("strict_dialect")
            != "openai_json_schema_strict"
            or int(preflight.get("active_clients", -1)) != 0
            or preflight.get("workflow_paused") is not True
            or preflight.get("stop_sentinel_clear") is not True
            or preflight.get("generation_requests") != 0
            or preflight.get("schema_sha256")
            != hashlib.sha256(schema_raw).hexdigest()
            or preflight.get("schema_registry_sha256")
            != schema_registry_evidence["sha256"]
        ):
            raise ValueError("functional preflight evidence or binding drifted")
        backend_config = json.loads(
            by_rel["functional-preflight/raw/backend-config.json"]
        )
        identity_fields = (
            "served_model", "model_revision", "backend", "backend_version",
            "max_context_tokens", "max_output_tokens", "strict_dialect",
            "thinking_enabled", "health", "capacity_available",
            "active_clients", "workflow_paused", "stop_sentinel_clear",
            "tokenizer_identity",
        )
        if any(
            backend_config.get(key) != preflight.get(key)
            for key in identity_fields
        ):
            raise ValueError("functional preflight differs from raw backend config")
        _, postflight_raw = one("functional-postflight/postflight.json")
        postflight = json.loads(postflight_raw)
        post_config_raw = by_rel[
            "functional-postflight/raw/backend-config.json"
        ]
        post_models_raw = by_rel["functional-postflight/raw/models.bin"]
        post_config = json.loads(post_config_raw)
        last_terminal = max(
            float(row.get("monotonic_seconds", -1)) for row in terminals
        )
        if (
            hashlib.sha256(post_config_raw).hexdigest()
            != postflight.get("backend_config_sha256")
            or hashlib.sha256(post_models_raw).hexdigest()
            != postflight.get("models_raw_sha256")
            or _sha(json.loads(post_models_raw))
            != postflight.get("models_json_sha256")
            or any(
                post_config.get(key) != postflight.get(key)
                for key in identity_fields
            )
            or any(
                postflight.get(key) != preflight.get(key)
                for key in identity_fields if key != "active_clients"
            )
            or int(postflight.get("active_clients", -1)) != 0
            or float(postflight.get("last_terminal_monotonic_seconds", -1))
            != last_terminal
            or float(postflight.get("observed_monotonic_seconds", -1))
            <= last_terminal
            or float(telemetry.get(
                "postflight_observed_monotonic_seconds", -1
            )) != float(postflight.get("observed_monotonic_seconds", -2))
        ):
            raise ValueError("functional postflight evidence/drain drifted")
        raw_sampler = dict(telemetry.get("raw_sampler_sources") or {})
        if (
            set(raw_sampler) != {"trtllm.log", "system.jsonl"}
            or any(
                raw_sampler[name] != hashlib.sha256(
                    by_rel[f"telemetry/{name}"]
                ).hexdigest()
                for name in raw_sampler
            )
            or int(telemetry.get("reasoning_bytes", -1))
            != functional_reasoning_bytes(response_blobs)
        ):
            raise ValueError("functional raw telemetry/reasoning drifted")
        _verify_functional_reconciliation(
            schedule_doc=schedule_doc,
            candidate=candidate,
            scheduled=scheduled,
            intents=intents,
            http=http,
            inventory=inventory,
            by_rel=by_rel,
            decision_rows=decision_rows,
            journal_rows=journal_rows,
            checkpoint_rows=checkpoint_rows,
            telemetry=telemetry,
            result_row=result_row,
            expected_binding_sha256=expected_binding_sha256,
            preflight=preflight,
        )
    return {
        "schema": "ed4all.gate-d-full-verification.v1",
        "verified": True, "binding_sha256": expected_binding_sha256,
        "stage_count": len(scheduled), "stages": scheduled,
        "artifact_count": len(inventory), "inventory": inventory,
        "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "authority_reverification": (
            authority_reverification["digests"]
            if authority_reverification is not None else None
        ),
        "schema_registry": schema_registry_evidence,
        "transaction_sha256": _sha({
            "binding": expected_binding_sha256,
            "stages": scheduled,
            "inventory": inventory,
            "schema_registry_sha256": schema_registry_evidence["sha256"],
        }),
    }
