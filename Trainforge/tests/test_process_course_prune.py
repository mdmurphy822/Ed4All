"""Tests for the Wave 74 ``--prune-after-import`` flag on process_course.

The prune logic lives in ``Trainforge.process_course.prune_output_after_import``
(unit-tested directly) and is wired into ``main()`` after the LibV2 import.
The ``main()`` integration tests stub out ``CourseProcessor.process`` and the
LibV2 importer so the heavy IMSCC parsing pipeline isn't exercised — the
prune branch is what we care about here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge import process_course as pc  # noqa: E402


# ---------------------------------------------------------------------------
# Direct unit tests on prune_output_after_import
# ---------------------------------------------------------------------------


def _seed_output_dir(output_dir: Path, *, with_chunks: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "concept_graph.json").write_text("{}\n", encoding="utf-8")
    sub = output_dir / "corpus"
    sub.mkdir()
    (sub / "metadata.json").write_text("{}\n", encoding="utf-8")
    if with_chunks:
        chunks_path = output_dir / "chunks.jsonl"
        chunks_path.write_text(
            '{"id": "c1"}\n{"id": "c2"}\n{"id": "c3"}\n',
            encoding="utf-8",
        )


def test_prune_helper_drops_contents_and_writes_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "trainforge_v2_output"
    _seed_output_dir(output_dir)
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir()
    libv2_target = libv2_root / "courses" / "test-course"
    libv2_target.mkdir(parents=True)

    receipt_path = pc.prune_output_after_import(
        output_dir=output_dir,
        course_code="TEST_101",
        libv2_slug="test-course",
        libv2_target_path=libv2_target,
        libv2_root=libv2_root,
    )

    assert receipt_path is not None
    # Output dir survives, but only IMPORT_RECEIPT.json remains.
    assert output_dir.exists()
    assert {p.name for p in output_dir.iterdir()} == {"IMPORT_RECEIPT.json"}

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt.keys()) == {
        "run_at",
        "course_code",
        "libv2_slug",
        "libv2_path",
        "chunks_imported",
    }
    assert receipt["course_code"] == "TEST_101"
    assert receipt["libv2_slug"] == "test-course"
    assert receipt["chunks_imported"] == 3
    assert receipt["libv2_path"].endswith("test-course")


def test_prune_helper_refuses_when_output_inside_libv2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    libv2_root = tmp_path / "LibV2"
    output_dir = libv2_root / "courses" / "danger"
    _seed_output_dir(output_dir)

    receipt_path = pc.prune_output_after_import(
        output_dir=output_dir,
        course_code="X",
        libv2_slug="danger",
        libv2_target_path=output_dir,
        libv2_root=libv2_root,
    )

    captured = capsys.readouterr()
    assert receipt_path is None
    assert "Refusing to prune" in captured.out
    # Nothing was deleted.
    assert (output_dir / "concept_graph.json").exists()
    assert (output_dir / "corpus" / "metadata.json").exists()


def test_import_receipt_shape(tmp_path: Path) -> None:
    """The receipt MUST carry the documented keys with the documented types."""
    output_dir = tmp_path / "out"
    _seed_output_dir(output_dir)
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir()

    receipt_path = pc.prune_output_after_import(
        output_dir=output_dir,
        course_code="ASTRO_101",
        libv2_slug="astro-101-fall26",
        libv2_target_path=libv2_root / "courses" / "astro-101-fall26",
        libv2_root=libv2_root,
    )

    assert receipt_path is not None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(receipt["run_at"], str)
    # ISO 8601 with timezone (UTC suffix).
    assert receipt["run_at"].endswith("+00:00") or receipt["run_at"].endswith("Z")
    assert isinstance(receipt["course_code"], str)
    assert isinstance(receipt["libv2_slug"], str)
    assert isinstance(receipt["libv2_path"], str)
    assert isinstance(receipt["chunks_imported"], int)
    assert receipt["chunks_imported"] == 3


# ---------------------------------------------------------------------------
# Integration tests via main() with mocked CourseProcessor + LibV2 importer
# ---------------------------------------------------------------------------


class _StubProcessor:
    """Minimal stand-in for ``CourseProcessor`` used in main()."""

    def __init__(self, *, output_dir: str, course_code: str, **kwargs: Any) -> None:
        self.output_dir = Path(output_dir)
        self.course_code = course_code
        self.division = kwargs.get("division") or "STEM"
        self.domain = kwargs.get("domain") or "physics"
        self.subdomains = kwargs.get("subdomains") or []
        self.secondary_domains = kwargs.get("secondary_domains") or []
        self.topics = kwargs.get("topics") or []

    def process(self) -> Dict[str, Any]:
        # Seed the output dir with the same fingerprint a real run leaves.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "chunks.jsonl").write_text(
            '{"id": "c1"}\n{"id": "c2"}\n', encoding="utf-8"
        )
        (self.output_dir / "concept_graph.json").write_text("{}\n", encoding="utf-8")
        sub = self.output_dir / "corpus"
        sub.mkdir(exist_ok=True)
        (sub / "metadata.json").write_text("{}\n", encoding="utf-8")
        return {
            "title": "Stub Course",
            "output_dir": str(self.output_dir),
            "stats": {
                "total_chunks": 2,
                "total_words": 100,
                "total_tokens_estimate": 200,
                "chunk_types": {"text": 2},
                "difficulty_distribution": {"intermediate": 2},
            },
        }


def _patch_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    libv2_import: Any,
    libv2_root: Path,
) -> None:
    """Wire stubs in for ``CourseProcessor`` + ``do_import`` + LibV2 root."""
    monkeypatch.setattr(pc, "CourseProcessor", _StubProcessor)
    monkeypatch.setattr(pc, "PROJECT_ROOT", libv2_root.parent)
    # Ensure LibV2 dir exists so PROJECT_ROOT/"LibV2" resolves.
    libv2_root.mkdir(parents=True, exist_ok=True)
    # The importer is imported lazily inside main(); inject a stub module.
    import types

    stub_module = types.ModuleType("LibV2.tools.libv2.importer")
    stub_module.import_course = libv2_import  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "LibV2.tools.libv2.importer", stub_module)


def _make_imscc_stub(tmp_path: Path) -> Path:
    """A throwaway IMSCC path. Stub processor doesn't read it."""
    imscc = tmp_path / "stub.imscc"
    imscc.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip EOCD
    # Sit a course_metadata.json next to it so --domain isn't required.
    (tmp_path / "course_metadata.json").write_text(
        '{"division": "STEM", "domain": "physics"}\n', encoding="utf-8"
    )
    return imscc


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["process_course"] + argv)
    pc.main()


def test_prune_after_import_drops_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imscc = _make_imscc_stub(tmp_path)
    output_dir = tmp_path / "out"
    libv2_root = tmp_path / "LibV2"

    def good_import(**kwargs: Any) -> str:
        target = libv2_root / "courses" / "stub-physics"
        target.mkdir(parents=True, exist_ok=True)
        return "stub-physics"

    _patch_main_dependencies(
        monkeypatch, libv2_import=good_import, libv2_root=libv2_root
    )

    _run_main(
        monkeypatch,
        [
            "--imscc",
            str(imscc),
            "--course-code",
            "PHYS_101",
            "--output",
            str(output_dir),
            "--import-to-libv2",
            "--prune-after-import",
        ],
    )

    assert output_dir.exists()
    assert {p.name for p in output_dir.iterdir()} == {"IMPORT_RECEIPT.json"}
    receipt = json.loads((output_dir / "IMPORT_RECEIPT.json").read_text())
    assert receipt["course_code"] == "PHYS_101"
    assert receipt["libv2_slug"] == "stub-physics"
    assert receipt["chunks_imported"] == 2


def test_prune_without_import_warns_and_no_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    imscc = _make_imscc_stub(tmp_path)
    output_dir = tmp_path / "out"
    libv2_root = tmp_path / "LibV2"

    def unused_import(**kwargs: Any) -> str:  # pragma: no cover - must not run
        raise AssertionError("LibV2 importer must not be invoked")

    _patch_main_dependencies(
        monkeypatch, libv2_import=unused_import, libv2_root=libv2_root
    )

    _run_main(
        monkeypatch,
        [
            "--imscc",
            str(imscc),
            "--course-code",
            "PHYS_101",
            "--output",
            str(output_dir),
            "--prune-after-import",
        ],
    )

    captured = capsys.readouterr()
    assert "no effect without --import-to-libv2" in captured.out
    # Output dir is intact: stub processor's seed files are still present.
    names = {p.name for p in output_dir.iterdir()}
    assert "chunks.jsonl" in names
    assert "concept_graph.json" in names
    assert "IMPORT_RECEIPT.json" not in names


def test_prune_skipped_when_import_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    imscc = _make_imscc_stub(tmp_path)
    output_dir = tmp_path / "out"
    libv2_root = tmp_path / "LibV2"

    def failing_import(**kwargs: Any) -> str:
        raise RuntimeError("Simulated LibV2 import failure")

    _patch_main_dependencies(
        monkeypatch, libv2_import=failing_import, libv2_root=libv2_root
    )

    _run_main(
        monkeypatch,
        [
            "--imscc",
            str(imscc),
            "--course-code",
            "PHYS_101",
            "--output",
            str(output_dir),
            "--import-to-libv2",
            "--prune-after-import",
        ],
    )

    captured = capsys.readouterr()
    assert "Import failed" in captured.out
    assert "preserving --output dir verbatim" in captured.out
    names = {p.name for p in output_dir.iterdir()}
    assert "chunks.jsonl" in names
    assert "concept_graph.json" in names
    assert "IMPORT_RECEIPT.json" not in names


def test_import_to_libv2_without_prune_keeps_full_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: legacy behavior (full output dir kept) when flag is OFF."""
    imscc = _make_imscc_stub(tmp_path)
    output_dir = tmp_path / "out"
    libv2_root = tmp_path / "LibV2"

    def good_import(**kwargs: Any) -> str:
        target = libv2_root / "courses" / "stub-physics"
        target.mkdir(parents=True, exist_ok=True)
        return "stub-physics"

    _patch_main_dependencies(
        monkeypatch, libv2_import=good_import, libv2_root=libv2_root
    )

    _run_main(
        monkeypatch,
        [
            "--imscc",
            str(imscc),
            "--course-code",
            "PHYS_101",
            "--output",
            str(output_dir),
            "--import-to-libv2",
        ],
    )

    names = {p.name for p in output_dir.iterdir()}
    # All seed artifacts survive.
    assert "chunks.jsonl" in names
    assert "concept_graph.json" in names
    assert "corpus" in names
    # Receipt is NOT written when prune is off.
    assert "IMPORT_RECEIPT.json" not in names
