"""P0 VLM extraction-source unit tests.

Covers the seat resolvers (in ``extract_shared``), the P0 image-chat client +
per-page disk cache + lifecycle unload (in ``vlm_extract``), and the additive
fusion-free ``_extract_page`` arm. The VLM endpoint + the page render are
MOCKED throughout — no live model call, no pypdfium2 render, no real PDF IO.

Conventions mirror ``test_ocr_render_scale.py``: resolvers are parse-with-
fallback and read at call time (monkeypatch env), and the render / requests
boundary is injected.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from PIL import Image

from dart_semantic import extract_shared
from dart_semantic import vlm_extract


# ---- fakes ----------------------------------------------------------------


class _FakeResp:
    def __init__(self, *, status_code=200, payload=None, text="", raise_exc=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self._raise_exc = raise_exc

    def json(self):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._payload


class _FakeRequests:
    """Records every POST; returns queued responses (or a fixed one)."""

    def __init__(self, response=None, raise_exc=None):
        self.calls = []
        self._response = response
        self._raise_exc = raise_exc

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def _ok_response(markdown="# Title\n\nBody prose $x^2$."):
    return _FakeResp(
        payload={"choices": [{"message": {"content": markdown}}]}
    )


def _render_fn(w=1600, h=2000):
    return lambda pdf_path, page_num, scale: Image.new("RGB", (w, h), (255, 255, 255))


def _local_seat():
    return extract_shared.resolve_vlm_seat()


# ---- resolvers: parse-with-fallback (read at call time) -------------------


def test_extract_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_VLM_EXTRACT", raising=False)
    assert extract_shared.resolve_vlm_extract_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "On"])
def test_extract_mode_truthy(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", val)
    assert extract_shared.resolve_vlm_extract_mode() is True


@pytest.mark.parametrize("val", ["", "0", "off", "garbage", "  "])
def test_extract_mode_falsey_and_garbage(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", val)
    assert extract_shared.resolve_vlm_extract_mode() is False


def test_seat_defaults(monkeypatch):
    for k in (
        "SEMANTIK_VLM_PROVIDER",
        "SEMANTIK_VLM_BASE_URL",
        "SEMANTIK_VLM_API_KEY",
        "SEMANTIK_VLM_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)
    seat = extract_shared.resolve_vlm_seat()
    assert seat.provider == "local"
    assert seat.base_url == "http://localhost:11434"
    assert seat.api_key is None
    assert seat.model == "qwen2.5vl:7b"
    assert seat.is_local is True


def test_seat_env_precedence(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", "spark")
    monkeypatch.setenv("SEMANTIK_VLM_BASE_URL", "https://spark.example/v1")
    monkeypatch.setenv("SEMANTIK_VLM_API_KEY", "sk-abc")
    monkeypatch.setenv("SEMANTIK_VLM_MODEL", "qwen2.5-vl:32b")
    seat = extract_shared.resolve_vlm_seat()
    assert seat.provider == "spark"
    assert seat.base_url == "https://spark.example/v1"
    assert seat.api_key == "sk-abc"
    assert seat.model == "qwen2.5-vl:32b"
    assert seat.is_local is False


def test_no_nvidia_leakage_into_vlm_seat(monkeypatch):
    # The VLM seat is local-first — it must NOT inherit the NVIDIA specialist
    # fallback chain.
    for k in (
        "SEMANTIK_VLM_PROVIDER",
        "SEMANTIK_VLM_BASE_URL",
        "SEMANTIK_VLM_API_KEY",
        "SEMANTIK_VLM_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nvidia.example/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    monkeypatch.setenv("NVIDIA_LARGE_MODEL", "meta/llama-3.3-70b-instruct")
    seat = extract_shared.resolve_vlm_seat()
    assert seat.base_url == "http://localhost:11434"
    assert seat.api_key is None
    assert seat.model == "qwen2.5vl:7b"


@pytest.mark.parametrize("bad", ["garbage", "", "0", "-1", "nan", "inf"])
def test_timeout_fallback(monkeypatch, bad):
    monkeypatch.setenv("SEMANTIK_VLM_TIMEOUT_SECONDS", bad)
    assert extract_shared.resolve_vlm_timeout() == 180.0


def test_timeout_override(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_TIMEOUT_SECONDS", "240.5")
    assert extract_shared.resolve_vlm_timeout() == 240.5


# ---- fail-loud: non-local seat with no key raises before any POST ----------


def test_nonlocal_seat_no_key_raises_before_post(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", "spark")
    monkeypatch.setenv("SEMANTIK_VLM_BASE_URL", "https://remote.example")
    monkeypatch.delenv("SEMANTIK_VLM_API_KEY", raising=False)
    seat = extract_shared.resolve_vlm_seat()
    fake = _FakeRequests(response=_ok_response())
    with pytest.raises(extract_shared.VLMSeatError):
        vlm_extract.extract_page_markdown(
            Path("x.pdf"),
            1,
            seat=seat,
            render_scale=2.0,
            timeout=180.0,
            cache_dir=tmp_path,
            render_fn=_render_fn(),
            requests_module=fake,
            pdf_sha_override="sha",
        )
    assert fake.calls == []  # zero network


# ---- request shape --------------------------------------------------------


def test_request_shape(monkeypatch, tmp_path):
    fake = _FakeRequests(response=_ok_response("# Hi"))
    out = vlm_extract.extract_page_markdown(
        Path("x.pdf"),
        3,
        seat=_local_seat(),
        render_scale=4.0,
        timeout=180.0,
        cache_dir=tmp_path,
        render_fn=_render_fn(),
        requests_module=fake,
        pdf_sha_override="sha3",
    )
    assert out["markdown"] == "# Hi"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"].endswith("/chat/completions")
    body = call["json"]
    assert body["model"] == "qwen2.5vl:7b"
    assert body["temperature"] == 0
    user = body["messages"][-1]
    parts = user["content"]
    assert any(p.get("type") == "text" for p in parts)
    img = next(p for p in parts if p.get("type") == "image_url")
    assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # Local seat → no Authorization header.
    assert "Authorization" not in call["headers"]


# ---- per-page cache -------------------------------------------------------


def test_cache_hit_makes_no_post(monkeypatch, tmp_path):
    fake1 = _FakeRequests(response=_ok_response("# Cached page"))
    first = vlm_extract.extract_page_markdown(
        Path("x.pdf"), 7, seat=_local_seat(), render_scale=2.0, timeout=180.0,
        cache_dir=tmp_path, render_fn=_render_fn(), requests_module=fake1,
        pdf_sha_override="shaC",
    )
    assert len(fake1.calls) == 1
    assert first["markdown"] == "# Cached page"

    fake2 = _FakeRequests(raise_exc=AssertionError("must not POST on cache hit"))
    second = vlm_extract.extract_page_markdown(
        Path("x.pdf"), 7, seat=_local_seat(), render_scale=2.0, timeout=180.0,
        cache_dir=tmp_path, render_fn=_render_fn(), requests_module=fake2,
        pdf_sha_override="shaC",
    )
    assert fake2.calls == []
    assert second["markdown"] == "# Cached page"


def test_cache_miss_on_model_change(monkeypatch, tmp_path):
    fake1 = _FakeRequests(response=_ok_response("A"))
    vlm_extract.extract_page_markdown(
        Path("x.pdf"), 1, seat=_local_seat(), render_scale=2.0, timeout=180.0,
        cache_dir=tmp_path, render_fn=_render_fn(), requests_module=fake1,
        pdf_sha_override="shaM",
    )
    # Different model id → different key → cache miss → a second POST.
    monkeypatch.setenv("SEMANTIK_VLM_MODEL", "qwen2.5-vl:32b")
    fake2 = _FakeRequests(response=_ok_response("B"))
    out = vlm_extract.extract_page_markdown(
        Path("x.pdf"), 1, seat=extract_shared.resolve_vlm_seat(), render_scale=2.0,
        timeout=180.0, cache_dir=tmp_path, render_fn=_render_fn(),
        requests_module=fake2, pdf_sha_override="shaM",
    )
    assert len(fake2.calls) == 1
    assert out["markdown"] == "B"


# ---- vlm source shape -----------------------------------------------------


def test_vlm_source_shape(monkeypatch, tmp_path):
    md = "# Heading\nLine two\n\nLine four"
    fake = _FakeRequests(response=_ok_response(md))
    out = vlm_extract.extract_page_markdown(
        Path("x.pdf"), 1, seat=_local_seat(), render_scale=2.0, timeout=180.0,
        cache_dir=tmp_path, render_fn=_render_fn(1600, 2000), requests_module=fake,
        pdf_sha_override="shaS",
    )
    assert out["markdown"] == md  # verbatim
    assert out["model"] == "qwen2.5vl:7b"
    assert out["provider"] == "local"
    assert out["prompt_version"] == vlm_extract.PROMPT_VERSION
    assert out["render_px"] == [1200, 1500]  # downscaled to 1200 width
    tbs = out["text_blocks"]
    assert [b["text"] for b in tbs] == ["# Heading", "Line two", "Line four"]
    assert all(b["bbox"] is None and b["confidence"] is None for b in tbs)


# ---- transient vs permanent HTTP failure ----------------------------------


def test_transient_http_failure_raises_transient(monkeypatch, tmp_path):
    monkeypatch.setattr(vlm_extract, "_VLM_RETRY_BACKOFF_BASE_SECONDS", 0.0)
    fake = _FakeRequests(raise_exc=RuntimeError("connection reset"))
    with pytest.raises(vlm_extract.VlmExtractError) as ei:
        vlm_extract.extract_page_markdown(
            Path("x.pdf"), 1, seat=_local_seat(), render_scale=2.0, timeout=1.0,
            cache_dir=tmp_path, render_fn=_render_fn(), requests_module=fake,
            pdf_sha_override="shaT",
        )
    assert ei.value.transient is True
    # Retried: 1 initial + _VLM_MAX_RETRIES.
    assert len(fake.calls) == vlm_extract._VLM_MAX_RETRIES + 1


def test_permanent_401_raises_permanent(monkeypatch, tmp_path):
    fake = _FakeRequests(response=_FakeResp(status_code=401, text="unauthorized"))
    with pytest.raises(vlm_extract.VlmExtractError) as ei:
        vlm_extract.extract_page_markdown(
            Path("x.pdf"), 1, seat=_local_seat(), render_scale=2.0, timeout=1.0,
            cache_dir=tmp_path, render_fn=_render_fn(), requests_module=fake,
            pdf_sha_override="shaP",
        )
    assert ei.value.transient is False
    assert len(fake.calls) == 1  # permanent → no retry


# ---- _extract_page arm: flag-off byte-identity + additive shape ------------


def _prep_scanned_page(monkeypatch):
    """Force _extract_page down the scanned (not is_text_ok) branch with mocked
    pypdfium2 / pdfplumber / tesseract / pikepdf so no real PDF is opened."""
    monkeypatch.setattr(
        extract_shared, "_pypdfium2_page_blocks",
        lambda p, n: ({"text_blocks": []}, 612.0, 792.0),
    )
    monkeypatch.setattr(extract_shared, "_text_layer_quality_ok", lambda t: False)
    monkeypatch.setattr(
        extract_shared, "_tesseract_page_blocks",
        lambda p, n: {"text_blocks": [{"bbox": [0, 0, 1, 1], "text": "ocr",
                                       "font_size": 10.0, "font_name": None,
                                       "is_bold": None, "is_italic": None,
                                       "confidence": 0.9}],
                      "page_width": 1224.0, "page_height": 1584.0},
    )
    monkeypatch.setattr(extract_shared, "_pikepdf_page_widgets", lambda p, n: [])


def test_extract_page_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_VLM_EXTRACT", raising=False)
    _prep_scanned_page(monkeypatch)
    called = {"n": 0}
    monkeypatch.setattr(
        vlm_extract, "extract_page_markdown",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    page = extract_shared._extract_page(Path("x.pdf"), 1, {"is_tagged": False})
    assert "vlm" not in page
    assert "vlm" not in page["sources_used"]
    assert called["n"] == 0
    # Merged provenance never contains vlm (fusion-free invariant holds trivially).
    for b in page["merged"]["text_blocks"]:
        assert "vlm" not in b.get("provenance", [])


def test_extract_page_flag_on_additive(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    _prep_scanned_page(monkeypatch)
    fake = _FakeRequests(response=_ok_response("# VLM page\nprose"))
    monkeypatch.setattr(
        vlm_extract, "_default_render_fn",
        lambda p, n, s: Image.new("RGB", (1600, 2000)),
    )

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Route the client through the injected requests + a temp cache by
        # patching the module default cache root.
        monkeypatch.setattr(vlm_extract, "_lazy_requests", lambda: fake)
        monkeypatch.setattr(vlm_extract, "_document_sha", lambda p: "shaON")
        monkeypatch.setattr(
            vlm_extract._semantik_paths, "resolve_cache", lambda name: Path(td)
        )
        page = extract_shared._extract_page(Path("x.pdf"), 1, {"is_tagged": False})

    assert "vlm" in page
    assert "vlm" in page["sources_used"]
    assert page["vlm"]["markdown"] == "# VLM page\nprose"
    # Tesseract source intact (additive contract).
    assert page["tesseract"]["text_blocks"][0]["text"] == "ocr"
    # Fusion-free invariant: merged provenance carries NO vlm.
    for b in page["merged"]["text_blocks"]:
        assert "vlm" not in b.get("provenance", [])


def test_extract_page_transient_failsoft(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    _prep_scanned_page(monkeypatch)
    monkeypatch.setattr(vlm_extract, "_VLM_RETRY_BACKOFF_BASE_SECONDS", 0.0)

    def _boom(*a, **k):
        raise vlm_extract.VlmExtractError("timeout", transient=True)

    monkeypatch.setattr(vlm_extract, "extract_page_markdown", _boom)
    page = extract_shared._extract_page(Path("x.pdf"), 1, {"is_tagged": False})
    assert page["vlm"]["markdown"] == ""
    assert "error" in page["vlm"]
    assert page["tesseract"]["text_blocks"][0]["text"] == "ocr"  # intact


def test_extract_page_permanent_raises(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    _prep_scanned_page(monkeypatch)

    def _boom(*a, **k):
        raise vlm_extract.VlmExtractError("bad request", transient=False)

    monkeypatch.setattr(vlm_extract, "extract_page_markdown", _boom)
    with pytest.raises(vlm_extract.VlmExtractError):
        extract_shared._extract_page(Path("x.pdf"), 1, {"is_tagged": False})


# ---- whole-doc extract-cache salt -----------------------------------------


def test_extract_cache_salt_asymmetric(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake bytes")

    monkeypatch.delenv("SEMANTIK_VLM_EXTRACT", raising=False)
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    off_key = extract_shared._compute_extract_cache_key(pdf)

    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    on_key = extract_shared._compute_extract_cache_key(pdf)
    assert on_key != off_key  # the vlm source changes the extracted shape


# ---- lifecycle unload -----------------------------------------------------


def test_unload_strips_v1_and_posts_keepalive():
    fake = _FakeRequests(response=_FakeResp(status_code=200))
    ok = vlm_extract.unload_vlm_model(
        "http://localhost:11434/v1", "qwen2.5vl:7b", requests_module=fake
    )
    assert ok is True
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"] == "http://localhost:11434/api/generate"
    assert call["json"] == {"model": "qwen2.5vl:7b", "keep_alive": 0}


def test_unload_never_raises_on_failure():
    fake = _FakeRequests(raise_exc=RuntimeError("ollama down"))
    assert vlm_extract.unload_vlm_model("http://localhost:11434", "m",
                                        requests_module=fake) is False


# ---- prompt hardening: no fabricated image markdown -----------------------


def test_system_directive_forbids_image_markdown():
    directive = vlm_extract._SYSTEM_DIRECTIVE.lower()
    # The transcriber is instructed NOT to emit Markdown image syntax or invent
    # image URLs — the scan-lane fabricated-image defect fix.
    assert "![" in vlm_extract._SYSTEM_DIRECTIVE
    assert "image url" in directive or "invent" in directive
    # An untranscribable figure gets a plain-text 'Figure:' line, not a link.
    assert "figure:" in directive


def test_prompt_version_bumped():
    # Bumped 1 -> 2 to invalidate the per-page markdown disk cache keyed on
    # prompt_version (the old prompt could have cached fabricated-image pages).
    assert vlm_extract.PROMPT_VERSION >= 2


def _run_extract_shared(monkeypatch, td, *, requests_obj, provider_local=True):
    """Drive extract_shared over a 2-page scanned doc with mocked extractors."""
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    monkeypatch.setattr(
        extract_shared, "_pikepdf_inspect",
        lambda p: {"is_encrypted": False, "is_tagged": False, "page_count": 2},
    )
    _prep_scanned_page(monkeypatch)
    monkeypatch.setattr(
        vlm_extract, "_default_render_fn",
        lambda p, n, s: Image.new("RGB", (1600, 2000)),
    )
    monkeypatch.setattr(vlm_extract, "_lazy_requests", lambda: requests_obj)
    monkeypatch.setattr(vlm_extract, "_document_sha", lambda p: "shaDOC")
    monkeypatch.setattr(
        vlm_extract._semantik_paths, "resolve_cache", lambda name: Path(td)
    )
    return extract_shared.extract_shared(Path("x.pdf"))


def test_lifecycle_one_unload_after_live_posts(monkeypatch, tmp_path):
    # A combined fake: chat POSTs return markdown; the unload POST (to
    # /api/generate) returns 200. Distinguish by URL.
    class _Combined:
        def __init__(self):
            self.chat = 0
            self.unload = 0

        def post(self, url, json=None, headers=None, timeout=None):
            if url.endswith("/api/generate"):
                self.unload += 1
                return _FakeResp(status_code=200)
            self.chat += 1
            return _ok_response("# page")

    combined = _Combined()
    td = tmp_path / "cache"
    td.mkdir()
    _run_extract_shared(monkeypatch, td, requests_obj=combined)
    assert combined.chat == 2  # 2 scanned pages, both live POSTs
    assert combined.unload == 1  # EXACTLY one keep_alive:0 unload


def test_lifecycle_no_unload_on_all_cache_hit(monkeypatch, tmp_path):
    class _Combined:
        def __init__(self):
            self.chat = 0
            self.unload = 0

        def post(self, url, json=None, headers=None, timeout=None):
            if url.endswith("/api/generate"):
                self.unload += 1
                return _FakeResp(status_code=200)
            self.chat += 1
            return _ok_response("# page")

    td = tmp_path / "cache"
    td.mkdir()
    # First run populates the cache (2 live posts + 1 unload).
    first = _Combined()
    _run_extract_shared(monkeypatch, td, requests_obj=first)
    assert first.unload == 1
    # Second run is all-cache-hit → zero chat POSTs → NO unload.
    second = _Combined()
    _run_extract_shared(monkeypatch, td, requests_obj=second)
    assert second.chat == 0
    assert second.unload == 0


def test_lifecycle_no_unload_for_nonlocal_provider(monkeypatch, tmp_path):
    class _Combined:
        def __init__(self):
            self.chat = 0
            self.unload = 0

        def post(self, url, json=None, headers=None, timeout=None):
            if url.endswith("/api/generate"):
                self.unload += 1
                return _FakeResp(status_code=200)
            self.chat += 1
            return _ok_response("# page")

    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", "spark")
    monkeypatch.setenv("SEMANTIK_VLM_BASE_URL", "https://remote.example")
    monkeypatch.setenv("SEMANTIK_VLM_API_KEY", "sk-x")
    combined = _Combined()
    td = tmp_path / "cache"
    td.mkdir()
    _run_extract_shared(monkeypatch, td, requests_obj=combined)
    assert combined.chat == 2  # regions still transcribed on the hosted seat
    assert combined.unload == 0  # hosted seat has no ollama /api root
