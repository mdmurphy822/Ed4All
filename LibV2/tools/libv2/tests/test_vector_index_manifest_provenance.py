"""Manifest replay provenance: device / batch_size / dtype are MEASURED.

The vector-index manifest exists so a build is reproducible and a
mixed-provenance comparison is refusable. It used to fall through to the
literals ``device="cpu"`` and ``batch_size=1`` whenever the embedding client
did not volunteer them — which was always, so every index on disk carried a
provenance block that was a fabrication rather than a measurement. A
cuda-built index was indistinguishable from a cpu-built one, and no
batch-size or precision experiment was verifiable after the fact.

With the CUDA product default in force this stops being cosmetic: byte-identity
of ``embeddings.npy`` holds only WITHIN a fixed (device, dtype, batch_size)
triple, because GPU reductions are not associative. A manifest that guesses any
of the three cannot support the comparison it exists to support.

These tests pin the replacement contract:

* the three values come from the client's own reported provenance, through
  every surface the duck-typed protocol allows;
* a client that reports a value NOWHERE is a loud failure, not a default;
* ``dtype`` is optional (absent == unknown, never "assume fp32") so manifests
  built before the precision seam stay loadable and byte-identical;
* the query path refuses a client whose dtype disagrees with the index, the
  same way it already refuses a foreign model.

Hermetic + CPU-only: local duck-typed clients honoring the frozen
``EmbeddingClient`` protocol. No weights, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from LibV2.tools.libv2.vector_index import (
    VectorIndexManifest,
    build_vector_index,
    load_vector_index,
)

from LibV2.tools.libv2.tests.test_vector_index import (
    _FakeEmbeddingClient,
    _write_course,
)


# --------------------------------------------------------------------------
# Clients that report provenance through each allowed surface.
# --------------------------------------------------------------------------


class _ResolvedStub:
    """Stands in for ``ResolvedEmbeddingProvider`` on a duck-typed client."""

    def __init__(self, **fields) -> None:
        self.document_prefix = ""
        self.query_prefix = ""
        for key, value in fields.items():
            setattr(self, key, value)


class _ProvenanceClient(_FakeEmbeddingClient):
    """Fake client with configurable provenance reporting.

    ``fingerprint_extra`` controls what ``model_fingerprint()`` volunteers;
    ``attrs`` controls the duck-typed attributes; ``resolved`` controls the
    resolved-provider record. Any of the three may be omitted entirely, which
    is how the precedence chain gets exercised.
    """

    def __init__(
        self,
        *,
        fingerprint_extra=None,
        attrs=None,
        resolved=None,
        kind: str = "fake",
    ) -> None:
        super().__init__()
        # Drop the base class's own device/batch_size so each case declares
        # exactly one source of truth.
        del self.batch_size
        del self.device
        self._fingerprint_extra = dict(fingerprint_extra or {})
        self._kind = kind
        for key, value in (attrs or {}).items():
            setattr(self, key, value)
        if resolved is not None:
            self.resolved = _ResolvedStub(**resolved)

    def model_fingerprint(self) -> dict:
        return {
            "model_id": "fake-deterministic-v1",
            "revision": None,
            "provider_name": "fake",
            "kind": self._kind,
            "dim": self.dim,
            **self._fingerprint_extra,
        }


# --------------------------------------------------------------------------
# Precedence: fingerprint -> client attribute -> resolved provider
# --------------------------------------------------------------------------


def test_fingerprint_is_the_canonical_source(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cuda:1", "batch_size": 64, "dtype": "bf16"},
        attrs={"device": "cpu", "batch_size": 1},
        resolved={"device": "cpu", "batch_size": 8, "dtype": "fp32"},
    )
    m = build_vector_index(course_dir, client=client)
    assert (m.device, m.batch_size, m.dtype) == ("cuda:1", 64, "bf16")


def test_client_attributes_are_used_when_the_fingerprint_is_silent(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        attrs={"device": "cuda", "batch_size": 32, "dtype": "fp16"},
    )
    m = build_vector_index(course_dir, client=client)
    assert (m.device, m.batch_size, m.dtype) == ("cuda", 32, "fp16")


def test_resolved_provider_is_the_last_real_source(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        resolved={"device": "cuda:0", "batch_size": 16, "dtype": "fp32"},
    )
    m = build_vector_index(course_dir, client=client)
    assert (m.device, m.batch_size, m.dtype) == ("cuda:0", 16, "fp32")


def test_cuda_build_is_recorded_as_cuda_not_cpu(tmp_path):
    """The regression this whole item exists for: a cuda build must not be
    recorded as a cpu build."""
    course_dir = _write_course(tmp_path, n=4)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cuda", "batch_size": 16, "dtype": "fp32"},
    )
    build_vector_index(course_dir, client=client)
    raw = json.loads(
        (course_dir / "vector_index" / "manifest.json").read_text("utf-8")
    )
    assert raw["device"] == "cuda"
    assert raw["batch_size"] == 16
    assert raw["dtype"] == "fp32"


# --------------------------------------------------------------------------
# Loud failure instead of a fabricated default
# --------------------------------------------------------------------------


def test_unreportable_device_raises_rather_than_defaulting_to_cpu(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"batch_size": 16},
        kind="unregistered-kind",
    )
    with pytest.raises(ValueError, match="reports no build device"):
        build_vector_index(course_dir, client=client)


def test_unreportable_batch_size_raises_rather_than_defaulting_to_one(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(fingerprint_extra={"device": "cpu"})
    with pytest.raises(ValueError, match="reports no encode batch_size"):
        build_vector_index(course_dir, client=client)


@pytest.mark.parametrize("bad", [0, -4])
def test_non_positive_batch_size_raises(tmp_path, bad):
    """The schema requires >= 1; a client reporting 0 is a broken client, and
    silently rewriting it to 1 is the fabrication this contract forbids."""
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cpu", "batch_size": bad}
    )
    with pytest.raises(ValueError, match="non-positive batch_size"):
        build_vector_index(course_dir, client=client)


def test_non_integer_batch_size_raises(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cpu", "batch_size": "many"}
    )
    with pytest.raises(ValueError, match="non-integer batch_size"):
        build_vector_index(course_dir, client=client)


def test_deviceless_kinds_record_where_they_actually_compute(tmp_path):
    """A kind with no local device concept still computes SOMEWHERE, and the
    manifest says where. A kind that declares neither raises."""
    for kind, expected in (("fake", "cpu"), ("openai-embeddings", "server")):
        course_dir = _write_course(tmp_path / kind, n=2)
        client = _ProvenanceClient(
            fingerprint_extra={"batch_size": 16}, kind=kind
        )
        m = build_vector_index(course_dir, client=client)
        assert m.device == expected, kind


# --------------------------------------------------------------------------
# dtype is optional — absent means unknown, not "assume fp32"
# --------------------------------------------------------------------------


def test_dtype_omitted_when_the_client_has_no_precision_seam(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cpu", "batch_size": 16}
    )
    m = build_vector_index(course_dir, client=client)
    assert m.dtype is None
    assert "dtype" not in m.to_dict()
    assert "dtype" not in m.content_dict()
    raw = (course_dir / "vector_index" / "manifest.json").read_text("utf-8")
    assert "dtype" not in raw


def test_blank_dtype_is_treated_as_absent(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cpu", "batch_size": 16, "dtype": "  "}
    )
    assert build_vector_index(course_dir, client=client).dtype is None


def test_dtype_round_trips_through_the_on_disk_manifest(tmp_path):
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cuda", "batch_size": 16, "dtype": "bf16"}
    )
    built = build_vector_index(course_dir, client=client)
    loaded = VectorIndexManifest.from_file(
        course_dir / "vector_index" / "manifest.json"
    )
    assert loaded.dtype == built.dtype == "bf16"
    assert "dtype" in built.content_dict()


def test_pre_seam_manifest_still_loads(tmp_path):
    """A manifest written before the precision seam carries no dtype key at
    all; it must keep loading, with dtype reading as unknown."""
    course_dir = _write_course(tmp_path, n=3)
    build_vector_index(course_dir, client=_FakeEmbeddingClient())
    manifest_path = course_dir / "vector_index" / "manifest.json"
    doc = json.loads(manifest_path.read_text("utf-8"))
    doc.pop("dtype", None)
    manifest_path.write_text(
        json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    index = load_vector_index(course_dir, allow_fake=True)
    assert index.manifest.dtype is None


# --------------------------------------------------------------------------
# Schema + validator agreement
# --------------------------------------------------------------------------


def test_provenance_manifest_passes_the_validator(tmp_path):
    from lib.validators.vector_index_manifest import VectorIndexManifestValidator

    course_dir = _write_course(tmp_path, n=4)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cuda", "batch_size": 64, "dtype": "bf16"}
    )
    build_vector_index(course_dir, client=client)
    result = VectorIndexManifestValidator().validate(
        {
            "vector_index_manifest_path": str(
                course_dir / "vector_index" / "manifest.json"
            )
        }
    )
    schema_violations = [
        i
        for i in result.issues
        if i.code == "VECTOR_INDEX_MANIFEST_SCHEMA_VIOLATION"
    ]
    assert schema_violations == [], schema_violations
    assert result.passed, [i.code for i in result.issues]


# --------------------------------------------------------------------------
# Query-side dtype guard (rides with the manifest change)
# --------------------------------------------------------------------------


def _built_index(tmp_path: Path, dtype: str):
    repo_root = tmp_path
    course_dir = _write_course(tmp_path, n=3)
    client = _ProvenanceClient(
        fingerprint_extra={"device": "cpu", "batch_size": 16, "dtype": dtype}
    )
    build_vector_index(course_dir, client=client)
    return repo_root, course_dir.name


def test_query_client_with_a_different_dtype_is_refused(tmp_path):
    from LibV2.tools.libv2.semantic_retriever import semantic_retrieve_chunks
    from LibV2.tools.libv2.vector_index import SemanticModelMismatch

    repo_root, slug = _built_index(tmp_path, "fp32")
    mismatched = _ProvenanceClient(
        fingerprint_extra={"device": "cpu", "batch_size": 16, "dtype": "bf16"}
    )
    with pytest.raises(SemanticModelMismatch, match="dtype"):
        semantic_retrieve_chunks(
            repo_root,
            "topic 1",
            course_slug=slug,
            client=mismatched,
            allow_fake=True,
        )


def test_query_client_with_the_same_dtype_is_accepted(tmp_path):
    from LibV2.tools.libv2.semantic_retriever import semantic_retrieve_chunks

    repo_root, slug = _built_index(tmp_path, "fp32")
    matching = _ProvenanceClient(
        fingerprint_extra={"device": "cuda", "batch_size": 64, "dtype": "fp32"}
    )
    # Device and batch size deliberately DIFFER: they are build provenance,
    # not vector-space identity, so they must not gate a query.
    semantic_retrieve_chunks(
        repo_root,
        "topic 1",
        course_slug=slug,
        client=matching,
        allow_fake=True,
    )


def test_unknown_dtype_on_either_side_does_not_refuse(tmp_path):
    """Absent == unknown. Refusing on an unknown would break every index
    built before the precision seam."""
    from LibV2.tools.libv2.semantic_retriever import semantic_retrieve_chunks

    repo_root, slug = _built_index(tmp_path, "fp32")
    silent = _ProvenanceClient(
        fingerprint_extra={"device": "cpu", "batch_size": 16}
    )
    semantic_retrieve_chunks(
        repo_root,
        "topic 1",
        course_slug=slug,
        client=silent,
        allow_fake=True,
    )
