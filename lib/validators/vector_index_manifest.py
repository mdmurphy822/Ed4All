"""VectorIndexManifestValidator — gate for the on-device vector index.

Gates the provenance + integrity manifest emitted alongside
``embeddings.npy`` + ``id_map.json`` at
``LibV2/courses/<slug>/vector_index/manifest.json``.

Template: ``lib/validators/chunkset_manifest.py::ChunksetManifestValidator``
(the closest sibling — file-on-disk + SHA-256 verification + schema gate).
This validator follows the same severity-honest contract: it emits
**critical-severity** GateIssues for every integrity / schema failure so a
tampered or partially-written index is visible to downstream consumers; the
gate config (``config/workflows.yaml``) decides whether the workflow blocks.
Per the WS2 plan the indexing-time gate is wired **warning**-severity, while
query-time staleness is fail-closed in ``load_vector_index`` regardless —
so this validator stays unwired in ``config/workflows.yaml`` for now
(constructed/dispatched directly), exactly like the WS1 foundation
validator.

Per-manifest contract enforced here:

1. **Manifest path is provided** — required input
   ``vector_index_manifest_path``. Missing → ``VECTOR_INDEX_MANIFEST_MISSING_INPUT``.
2. **Manifest exists on disk** — ``VECTOR_INDEX_MANIFEST_NOT_FOUND``.
3. **Manifest JSON parses** — ``VECTOR_INDEX_MANIFEST_INVALID_JSON``.
4. **Manifest validates against the canonical schema** — when
   ``jsonschema`` is installed validates against
   ``schemas/library/vector_index_manifest.schema.json`` and emits
   ``VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION`` per violation; otherwise a
   structural fallback covers the ``required`` fields + the SHA / enum
   regexes.
5. **Sibling ``embeddings.npy`` + ``id_map.json`` exist** —
   ``VECTOR_INDEX_ARTIFACT_NOT_FOUND``.
6. **Recomputed SHA-256 of each artifact matches the manifest** —
   ``VECTOR_INDEX_HASH_MISMATCH`` (the load-bearing tamper-detection check).
7. **``chunks_count`` matches ``id_map.json``'s ``chunk_ids`` length** —
   ``VECTOR_INDEX_COUNT_MISMATCH``.
8. **``source_chunks_sha256`` matches the resolved live chunkset** (when a
   ``course_dir`` is supplied so the validator can resolve it) — staleness
   surfaces as a **warning** here (query-time enforcement is fail-closed):
   ``VECTOR_INDEX_STALE``.
9. **``chunker_version`` matches the current chunker** (when resolvable) —
   ``VECTOR_INDEX_CHUNKER_VERSION_MISMATCH`` (warning).

Inputs contract:

* ``inputs["vector_index_manifest_path"]`` — path to manifest.json. Required.
* ``inputs["course_dir"]`` — optional; when present the validator resolves
  the live chunkset and performs the staleness cross-check (8).
* ``inputs["gate_id"]`` — optional GateResult ID override.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.utils.hashing import sha256_file

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNKER_VERSION_RE = re.compile(
    r"^(?:v\d+|\d+\.\d+\.\d+(?:[+-][A-Za-z0-9.+-]+)?)$"
)

_REQUIRED_FIELDS = (
    "schema_version",
    "embedding_provider",
    "embedding_kind",
    "embedding_model_id",
    "embedding_dim",
    "normalized",
    "index_type",
    "chunker_version",
    "chunkset_kind",
    "source_chunks_sha256",
    "embeddings_sha256",
    "id_map_sha256",
    "chunks_count",
    "text_field_policy",
    "batch_size",
    "device",
)

_KNOWN_FIELDS = frozenset(_REQUIRED_FIELDS) | {
    "embedding_model_revision",
    "document_prefix",
    "query_prefix",
    "generated_at",
}

_ALLOWED_KINDS = {"st", "openai-embeddings", "fake"}
_ALLOWED_CHUNKSET_KINDS = {"imscc", "dart", "corpus-legacy"}
_ALLOWED_INDEX_TYPES = {"exact-numpy"}
_ALLOWED_TEXT_FIELD_POLICIES = {"text+heading", "text", "retrieval_text"}

_EMBEDDINGS_FILENAME = "embeddings.npy"
_ID_MAP_FILENAME = "id_map.json"


def _resolve_schema_path() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = (
            parent
            / "schemas"
            / "library"
            / "vector_index_manifest.schema.json"
        )
        if candidate.exists():
            return candidate
    return None


class VectorIndexManifestValidator:
    """Vector-index manifest integrity + provenance gate."""

    name = "vector_index_manifest"
    version = "0.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        issues: List[GateIssue] = []

        path_raw = inputs.get("vector_index_manifest_path")
        if not path_raw:
            return self._fail(
                gate_id,
                "VECTOR_INDEX_MANIFEST_MISSING_INPUT",
                "VectorIndexManifestValidator requires "
                "inputs['vector_index_manifest_path'].",
            )

        manifest_path = Path(path_raw)
        if not manifest_path.exists() or not manifest_path.is_file():
            return self._fail(
                gate_id,
                "VECTOR_INDEX_MANIFEST_NOT_FOUND",
                f"vector-index manifest not found at {manifest_path}",
                location=str(manifest_path),
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._fail(
                gate_id,
                "VECTOR_INDEX_MANIFEST_INVALID_JSON",
                f"vector-index manifest failed to parse: "
                f"{exc.__class__.__name__}: {exc}",
                location=str(manifest_path),
            )

        if not isinstance(manifest, dict):
            return self._fail(
                gate_id,
                "VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION",
                f"vector-index manifest root is not a JSON object "
                f"(got {type(manifest).__name__}).",
                location=str(manifest_path),
            )

        # ---- schema validation
        issues.extend(self._validate_against_schema(manifest, manifest_path))

        # If load-bearing fields are missing/invalid, downstream sha checks
        # are noise — short-circuit on a schema-critical manifest.
        critical_so_far = [i for i in issues if i.severity == "critical"]
        if critical_so_far and (
            not isinstance(manifest.get("embeddings_sha256"), str)
            or not isinstance(manifest.get("id_map_sha256"), str)
        ):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=issues,
                action="block",
            )

        index_dir = manifest_path.parent
        embeddings_path = index_dir / _EMBEDDINGS_FILENAME
        id_map_path = index_dir / _ID_MAP_FILENAME

        # ---- sibling artifacts exist
        for artifact in (embeddings_path, id_map_path):
            if not artifact.exists() or not artifact.is_file():
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="VECTOR_INDEX_ARTIFACT_NOT_FOUND",
                        message=(
                            f"sibling {artifact.name} not found at "
                            f"{artifact}; the vector-index manifest is "
                            f"orphaned."
                        ),
                        location=str(artifact),
                        suggestion=(
                            "Rebuild the index (`libv2 vector-index build "
                            "--force`) or remove the orphaned manifest."
                        ),
                    )
                )

        # ---- sha re-verification
        self._verify_sha(
            issues,
            embeddings_path,
            manifest.get("embeddings_sha256"),
            "embeddings.npy",
        )
        self._verify_sha(
            issues,
            id_map_path,
            manifest.get("id_map_sha256"),
            "id_map.json",
        )

        # ---- chunks_count cross-check against id_map
        self._verify_count(issues, id_map_path, manifest.get("chunks_count"))

        # ---- staleness + chunker-version cross-check (when course_dir given)
        self._verify_chunkset(issues, inputs, manifest)

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        action: Optional[str] = "block" if not passed else None
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    # -------------------------------------------------------------- helpers

    def _fail(
        self,
        gate_id: str,
        code: str,
        message: str,
        *,
        location: Optional[str] = None,
    ) -> GateResult:
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=False,
            issues=[
                GateIssue(
                    severity="critical",
                    code=code,
                    message=message,
                    location=location,
                )
            ],
            action="block",
        )

    @staticmethod
    def _verify_sha(
        issues: List[GateIssue],
        artifact_path: Path,
        manifest_sha: Any,
        label: str,
    ) -> None:
        if not (isinstance(manifest_sha, str) and _SHA256_RE.match(manifest_sha)):
            return
        if not artifact_path.exists():
            return
        try:
            actual = sha256_file(artifact_path)
        except OSError as exc:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="VECTOR_INDEX_ARTIFACT_READ_ERROR",
                    message=f"failed to recompute SHA-256 for {label}: {exc}",
                    location=str(artifact_path),
                )
            )
            return
        if actual != manifest_sha:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="VECTOR_INDEX_HASH_MISMATCH",
                    message=(
                        f"{label} sha in manifest ({manifest_sha[:16]}...) "
                        f"does not match on-disk file ({actual[:16]}...)."
                    ),
                    location=str(artifact_path),
                    suggestion="Rebuild the index or update the manifest.",
                )
            )

    @staticmethod
    def _verify_count(
        issues: List[GateIssue],
        id_map_path: Path,
        manifest_count: Any,
    ) -> None:
        if not isinstance(manifest_count, int) or isinstance(manifest_count, bool):
            return
        if not id_map_path.exists():
            return
        try:
            doc = json.loads(id_map_path.read_text(encoding="utf-8"))
            actual = len(doc.get("chunk_ids", []))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="VECTOR_INDEX_ID_MAP_READ_ERROR",
                    message=f"failed to read id_map.json: {exc}",
                    location=str(id_map_path),
                )
            )
            return
        if actual != manifest_count:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="VECTOR_INDEX_COUNT_MISMATCH",
                    message=(
                        f"manifest chunks_count ({manifest_count}) does not "
                        f"match id_map chunk_ids length ({actual})."
                    ),
                    location=str(id_map_path),
                    suggestion="Rebuild the index.",
                )
            )

    @staticmethod
    def _verify_chunkset(
        issues: List[GateIssue],
        inputs: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        course_dir_raw = inputs.get("course_dir")

        # chunker_version drift (resolvable without a course_dir).
        try:
            from Trainforge.chunker import CHUNKER_SCHEMA_VERSION as current_ver
        except Exception:  # noqa: BLE001
            current_ver = None
        manifest_ver = manifest.get("chunker_version")
        if (
            current_ver is not None
            and isinstance(manifest_ver, str)
            and manifest_ver != current_ver
        ):
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VECTOR_INDEX_CHUNKER_VERSION_MISMATCH",
                    message=(
                        f"index chunker_version ({manifest_ver}) != current "
                        f"chunker ({current_ver}); a re-chunk may have "
                        f"happened. Rebuild to be safe."
                    ),
                )
            )

        if not course_dir_raw:
            return
        try:
            from LibV2.tools.libv2.vector_index import resolve_chunks_for_index
        except Exception:  # noqa: BLE001
            return

        try:
            chunks_path, _kind = resolve_chunks_for_index(
                Path(course_dir_raw), manifest.get("chunkset_kind")
            )
        except Exception:  # noqa: BLE001
            return
        if not chunks_path.exists():
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VECTOR_INDEX_STALE",
                    message=(
                        f"chunkset {chunks_path} embedded by the index no "
                        f"longer exists; the index is stale."
                    ),
                    location=str(chunks_path),
                )
            )
            return
        source_sha = manifest.get("source_chunks_sha256")
        if isinstance(source_sha, str):
            try:
                live = sha256_file(chunks_path)
            except OSError:
                return
            if live != source_sha:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="VECTOR_INDEX_STALE",
                        message=(
                            f"index source_chunks_sha256 "
                            f"({source_sha[:16]}...) != live chunks.jsonl "
                            f"({live[:16]}...); rebuild the index."
                        ),
                        location=str(chunks_path),
                    )
                )

    @staticmethod
    def _validate_against_schema(
        manifest: Dict[str, Any], manifest_path: Path
    ) -> List[GateIssue]:
        issues: List[GateIssue] = []
        try:
            import jsonschema  # type: ignore
        except ImportError:
            return VectorIndexManifestValidator._structural_fallback(
                manifest, manifest_path
            )

        schema_path = _resolve_schema_path()
        if not schema_path or not schema_path.exists():
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VECTOR_INDEX_MANIFEST_SCHEMA_UNAVAILABLE",
                    message=(
                        "vector_index_manifest.schema.json not found; "
                        "falling back to structural check."
                    ),
                )
            )
            issues.extend(
                VectorIndexManifestValidator._structural_fallback(
                    manifest, manifest_path
                )
            )
            return issues

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VECTOR_INDEX_MANIFEST_SCHEMA_LOAD_ERROR",
                    message=f"failed to load vector-index schema: {exc}",
                )
            )
            issues.extend(
                VectorIndexManifestValidator._structural_fallback(
                    manifest, manifest_path
                )
            )
            return issues

        validator = jsonschema.Draft7Validator(schema)
        for err in validator.iter_errors(manifest):
            location = ".".join(str(p) for p in err.absolute_path) or "<root>"
            issues.append(
                GateIssue(
                    severity="critical",
                    code="VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION",
                    message=f"schema check: {err.message}",
                    location=location,
                    suggestion=(
                        "See schemas/library/"
                        "vector_index_manifest.schema.json."
                    ),
                )
            )
        return issues

    @staticmethod
    def _structural_fallback(
        manifest: Dict[str, Any], manifest_path: Path
    ) -> List[GateIssue]:
        issues: List[GateIssue] = []

        def violation(msg: str) -> None:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION",
                    message=msg,
                    location=str(manifest_path),
                )
            )

        for required in _REQUIRED_FIELDS:
            if required not in manifest:
                violation(f"missing required field: {required!r}")

        for key in manifest:
            if key not in _KNOWN_FIELDS:
                violation(
                    f"unknown top-level field: {key!r} "
                    f"(additionalProperties: false)"
                )

        if manifest.get("schema_version") not in (None, "1.0"):
            violation(
                f"schema_version must be '1.0'; got "
                f"{manifest.get('schema_version')!r}"
            )

        for sha_field in (
            "source_chunks_sha256",
            "embeddings_sha256",
            "id_map_sha256",
        ):
            val = manifest.get(sha_field)
            if val is not None and not (
                isinstance(val, str) and _SHA256_RE.match(val)
            ):
                violation(
                    f"{sha_field} must be 64-char lowercase hex; got {val!r}"
                )

        ver = manifest.get("chunker_version")
        if ver is not None and not (
            isinstance(ver, str) and _CHUNKER_VERSION_RE.match(ver)
        ):
            violation(f"chunker_version pattern invalid: {ver!r}")

        kind = manifest.get("embedding_kind")
        if kind is not None and kind not in _ALLOWED_KINDS:
            violation(
                f"embedding_kind must be one of {sorted(_ALLOWED_KINDS)}; "
                f"got {kind!r}"
            )

        cskind = manifest.get("chunkset_kind")
        if cskind is not None and cskind not in _ALLOWED_CHUNKSET_KINDS:
            violation(
                f"chunkset_kind must be one of "
                f"{sorted(_ALLOWED_CHUNKSET_KINDS)}; got {cskind!r}"
            )

        itype = manifest.get("index_type")
        if itype is not None and itype not in _ALLOWED_INDEX_TYPES:
            violation(
                f"index_type must be one of {sorted(_ALLOWED_INDEX_TYPES)}; "
                f"got {itype!r}"
            )

        tfp = manifest.get("text_field_policy")
        if tfp is not None and tfp not in _ALLOWED_TEXT_FIELD_POLICIES:
            violation(
                f"text_field_policy must be one of "
                f"{sorted(_ALLOWED_TEXT_FIELD_POLICIES)}; got {tfp!r}"
            )

        if manifest.get("normalized") not in (None, True):
            violation("normalized must be const true")

        for int_field in ("embedding_dim", "chunks_count", "batch_size"):
            val = manifest.get(int_field)
            if val is not None and (
                not isinstance(val, int) or isinstance(val, bool) or val < 0
            ):
                violation(f"{int_field} must be a non-negative integer; got {val!r}")

        return issues


__all__ = ["VectorIndexManifestValidator"]
