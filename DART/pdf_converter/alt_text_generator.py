"""
Alt Text Generation Module

Generates accessible alt text for images using Claude's vision API
with OCR fallback.
"""

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

# Phase 6 Subtask 22 (Phase 3c env-vars): canonical resolver lives in
# ``DART/pdf_converter/claude_processor.py``; importing here avoids
# duplicating the helper and keeps a single env var pin
# (``DART_CLAUDE_MODEL``) in charge of every DART call site.
from .claude_processor import _resolve_dart_claude_model

if TYPE_CHECKING:
    from MCP.orchestrator.llm_backend import LLMBackend

    from .image_extractor import ExtractedImage

logger = logging.getLogger(__name__)


@dataclass
class AltTextResult:
    """Result of alt text generation."""
    alt_text: str               # Short alt text (< 150 chars)
    long_description: str       # Detailed description for expandable section
    source: str                 # 'claude', 'ocr', 'caption', 'generic'
    success: bool = True


class AltTextGenerator:
    """Generate accessible alt text for images."""

    # Maximum retries for API calls
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_ai: bool = True,
        use_ocr_fallback: bool = True,
        llm: Optional["LLMBackend"] = None,
        capture: Optional[object] = None,
    ):
        """
        Initialize alt text generator.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env). Ignored
                when ``llm`` is provided.
            model: Claude model to use for vision. When ``None``, resolves
                via env-var-first chain: ``DART_CLAUDE_MODEL`` env var, then
                the legacy default ``claude-sonnet-4-20250514`` (Phase 6
                Subtask 22 / Phase 3c env-vars).
            use_ai: Whether to use Claude API for alt text
            use_ocr_fallback: Whether to fall back to OCR
            llm: Optional pre-built LLM backend (e.g., an
                :class:`MCP.orchestrator.LLMBackend`). When provided, vision
                completions route through it instead of constructing an
                Anthropic client directly. Backward compatible: existing
                callers that pass only ``api_key`` continue to work.
            capture: Optional
                :class:`lib.decision_capture.DARTDecisionCapture` (or
                compatible ``DecisionCapture`` with
                ``log_alt_text_decision``). Wave 22 DC1: when supplied,
                every generated alt-text fires one
                ``alt_text_generation`` decision with dynamic rationale
                (page, bbox, image hash, detected type, caption
                presence). When ``None`` (the default), logging is
                silently skipped so existing tests keep passing.
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        # Phase 6 Subtask 22: resolve effective model via env-var-first chain.
        self.model = _resolve_dart_claude_model(model)
        self.use_ocr_fallback = use_ocr_fallback
        self._llm = llm
        self.capture = capture

        # Either an injected backend or a resolvable API key enables AI.
        self.use_ai = use_ai and (llm is not None or self.api_key is not None)

        self._client = None
        if self.use_ai and llm is None:
            try:
                from MCP.orchestrator.llm_backend import AnthropicBackend

                self._client = AnthropicBackend(
                    api_key=self.api_key,
                    default_model=self.model,
                )
                logger.debug(f"LLM backend initialized with model {self.model}")
            except ImportError:
                logger.warning("anthropic package not installed. AI alt text unavailable.")
                self.use_ai = False
            except Exception as e:
                logger.warning(f"Failed to initialize LLM backend: {e}")
                self.use_ai = False

        # Check for OCR availability
        self._pytesseract = None
        self._pil = None
        if self.use_ocr_fallback:
            try:
                import pytesseract
                from PIL import Image
                self._pytesseract = pytesseract
                self._pil = Image
            except ImportError:
                logger.debug("OCR packages not available for fallback")

    def generate(
        self,
        image: "ExtractedImage",
        context: str = ""
    ) -> AltTextResult:
        """
        Generate alt text for an image.

        Args:
            image: ExtractedImage object
            context: Document context (nearby text, title, etc.)

        Returns:
            AltTextResult with generated text
        """
        # Try Claude API first (via injected backend or legacy SDK path)
        if self.use_ai and (self._llm is not None or self._client is not None):
            result = self._try_claude_with_retry(image, context)
            if result.success:
                self._log_alt_text_decision(image, result, context)
                return result

        # Try OCR fallback
        if self.use_ocr_fallback and self._pytesseract:
            result = self._try_ocr_fallback(image)
            if result.success and result.alt_text:
                self._log_alt_text_decision(image, result, context)
                return result

        # Try caption-based fallback
        if image.nearby_caption:
            result = self._use_caption_fallback(image)
            self._log_alt_text_decision(image, result, context)
            return result

        # Final generic fallback
        result = self._generic_fallback(image)
        self._log_alt_text_decision(image, result, context)
        return result

    def _log_alt_text_decision(
        self,
        image: "ExtractedImage",
        result: AltTextResult,
        context: str,
    ) -> None:
        """Wave 22 DC1: dynamic per-figure decision capture.

        Uses the ``DARTDecisionCapture.log_alt_text_decision`` helper
        when the injected capture is one; otherwise falls back to a
        generic ``log_decision`` call. Every rationale interpolates
        page, bbox, image hash, source strategy (claude / ocr /
        caption / generic), and caption presence — no two captures
        are byte-identical, meeting the audit's explicit "no static
        boilerplate rationales" constraint.
        """
        capture = getattr(self, "capture", None)
        if capture is None:
            return

        try:
            import hashlib

            # Image hash: first 12 chars of sha256(bytes). We only use
            # the hash for correlation, never for content recovery, so
            # 12 is plenty and keeps the rationale short.
            img_bytes = getattr(image, "data", b"") or b""
            img_hash = (
                hashlib.sha256(img_bytes).hexdigest()[:12]
                if img_bytes
                else "none"
            )

            page = getattr(image, "page", 0) or 0
            bbox = getattr(image, "bbox", None)
            bbox_str = (
                f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
                if bbox and len(bbox) == 4
                else "unknown"
            )
            width = getattr(image, "width", None) or "?"
            height = getattr(image, "height", None) or "?"
            nearby_caption = getattr(image, "nearby_caption", "") or ""
            caption_state = (
                "caption-present"
                if nearby_caption
                else "caption-absent"
            )

            image_id = f"p{page:04d}-{img_hash}"
            method = result.source  # 'claude' | 'ocr' | 'caption' | 'generic'

            # Dynamic rationale interpolating every auditable signal.
            rationale = (
                f"Image {image_id} at page {page} bbox={bbox_str} "
                f"size={width}x{height}; chose source={method} "
                f"({caption_state}); "
                f"alt len={len(result.alt_text)} chars, "
                f"long_desc len={len(result.long_description)} chars; "
                f"ctx len={len(context)} chars"
            )

            # Prefer the specialised helper when the capture carries it
            # (Wave 22 DC1 re-uses the zero-caller helper at
            # ``lib/decision_capture.py::DARTDecisionCapture.log_alt_text_decision``).
            helper = getattr(capture, "log_alt_text_decision", None)
            if callable(helper):
                # The helper only accepts (image_id, alt_text, method),
                # so after it logs the summary we also emit a richer
                # ``log_decision`` carrying the dynamic rationale so
                # the 20-char minimum and interpolated-signal audit
                # requirements are satisfied together.
                helper(image_id, result.alt_text, method=method)
                log_fn = getattr(capture, "log_decision", None)
                if callable(log_fn):
                    log_fn(
                        decision_type="alt_text_generation",
                        decision=(
                            f"Generated alt text for {image_id} via {method}"
                        ),
                        rationale=rationale,
                        context=(
                            f"alt={result.alt_text[:120]!r}"
                            if result.alt_text
                            else "alt=<empty>"
                        ),
                    )
            else:
                log_fn = getattr(capture, "log_decision", None)
                if callable(log_fn):
                    log_fn(
                        decision_type="alt_text_generation",
                        decision=(
                            f"Generated alt text for {image_id} via {method}"
                        ),
                        rationale=rationale,
                        context=(
                            f"alt={result.alt_text[:120]!r}"
                            if result.alt_text
                            else "alt=<empty>"
                        ),
                    )
        except Exception as exc:  # noqa: BLE001 — capture is best-effort
            logger.debug(
                "Alt-text capture emit failed (%s); continuing", exc
            )

    def _try_claude_with_retry(
        self,
        image: "ExtractedImage",
        context: str
    ) -> AltTextResult:
        """
        Try Claude API with exponential backoff retry.

        Args:
            image: ExtractedImage object
            context: Document context

        Returns:
            AltTextResult
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._call_claude_vision(image, context)

            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit
                if 'rate' in error_str or '429' in error_str:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    continue

                # Check for timeout
                if 'timeout' in error_str:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Timeout, retrying in {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    continue

                # Other errors - don't retry
                logger.warning(f"Claude API error: {e}")
                break

        return AltTextResult(
            alt_text="",
            long_description="",
            source="claude",
            success=False
        )

    def _call_claude_vision(
        self,
        image: "ExtractedImage",
        context: str
    ) -> AltTextResult:
        """
        Call Claude vision API for alt text generation.

        Args:
            image: ExtractedImage object
            context: Document context

        Returns:
            AltTextResult
        """
        # Get base64 image data
        if image.data_uri:
            # Extract base64 from data URI
            base64_data = image.data_uri.split(',')[1] if ',' in image.data_uri else image.data_uri
        else:
            base64_data = base64.b64encode(image.data).decode()

        # Determine media type
        media_type = f"image/{image.format}"
        if image.format in ('jpg', 'jpeg'):
            media_type = "image/jpeg"

        # Build prompt
        prompt = self._build_prompt(image, context)

        # Route through the LLM backend abstraction (injected or lazy-built).
        backend = self._llm if self._llm is not None else self._client
        response_text = backend.complete_sync(
            system="",
            user=prompt,
            model=self.model,
            max_tokens=500,
            images=[
                {"media_type": media_type, "data": base64_data},
            ],
        ).strip()

        # Extract alt text and long description
        alt_text, long_desc = self._parse_response(response_text)

        return AltTextResult(
            alt_text=alt_text,
            long_description=long_desc,
            source="claude",
            success=True
        )

    def _build_prompt(self, image: "ExtractedImage", context: str) -> str:
        """Build the prompt for Claude."""
        prompt = """Analyze this image and provide accessibility text.

REQUIREMENTS:
1. ALT TEXT: A concise description (max 150 characters) suitable for screen readers.
   - Focus on the essential content/purpose
   - Don't start with "Image of" or "Picture of"
   - Be specific but brief

2. LONG DESCRIPTION: A detailed description (2-4 sentences) for users who want more detail.
   - Include important visual details
   - Describe data, charts, or diagrams precisely
   - Mention colors/layout if relevant to meaning

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
ALT: [your alt text here]
LONG: [your long description here]"""

        if context:
            prompt += f"\n\nDOCUMENT CONTEXT:\n{context[:500]}"

        if image.nearby_caption:
            prompt += f"\n\nCAPTION FOUND NEAR IMAGE:\n{image.nearby_caption}"

        return prompt

    def _parse_response(self, response: str) -> tuple:
        """Parse Claude's response into alt text and long description."""
        alt_text = ""
        long_desc = ""

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('ALT:'):
                alt_text = line[4:].strip()
            elif line.upper().startswith('LONG:'):
                long_desc = line[5:].strip()

        # If parsing failed, use the whole response
        if not alt_text and not long_desc:
            # Try to split by sentence
            if len(response) <= 150:
                alt_text = response
            else:
                sentences = response.split('.')
                alt_text = sentences[0].strip() + '.' if sentences else response[:150]
                long_desc = response

        # Ensure alt text isn't too long
        if len(alt_text) > 150:
            alt_text = alt_text[:147] + "..."

        return alt_text, long_desc

    def _try_ocr_fallback(self, image: "ExtractedImage") -> AltTextResult:
        """
        Try OCR on the image to extract text content.

        Args:
            image: ExtractedImage object

        Returns:
            AltTextResult
        """
        try:
            import io
            img = self._pil.open(io.BytesIO(image.data))

            # Run OCR
            text = self._pytesseract.image_to_string(img, config='--psm 6').strip()

            if text and len(text) > 10:
                # Clean up OCR text
                text = ' '.join(text.split())

                # Create alt text
                if len(text) <= 140:
                    alt_text = f"Image containing text: {text}"
                else:
                    alt_text = f"Image containing text: {text[:120]}..."

                return AltTextResult(
                    alt_text=alt_text[:150],
                    long_description=f"Text extracted from image: {text}",
                    source="ocr",
                    success=True
                )

        except Exception as e:
            logger.debug(f"OCR fallback failed: {e}")

        return AltTextResult(
            alt_text="",
            long_description="",
            source="ocr",
            success=False
        )

    def _use_caption_fallback(self, image: "ExtractedImage") -> AltTextResult:
        """
        Use nearby caption as alt text.

        Args:
            image: ExtractedImage object

        Returns:
            AltTextResult
        """
        caption = image.nearby_caption.strip()

        # Clean up caption
        if len(caption) <= 150:
            alt_text = caption
        else:
            alt_text = caption[:147] + "..."

        return AltTextResult(
            alt_text=alt_text,
            long_description=caption,
            source="caption",
            success=True
        )

    def _generic_fallback(self, image: "ExtractedImage") -> AltTextResult:
        """
        Generate generic alt text when all else fails.

        Args:
            image: ExtractedImage object

        Returns:
            AltTextResult
        """
        # Generate basic description
        alt_text = f"Figure on page {image.page}"

        # Add dimensions if known
        if image.width and image.height:
            long_desc = f"Image ({image.width}x{image.height} pixels) on page {image.page}. No description available."
        else:
            long_desc = f"Image on page {image.page}. No description available."

        return AltTextResult(
            alt_text=alt_text,
            long_description=long_desc,
            source="generic",
            success=True
        )

    def generate_batch(
        self,
        images: list,
        context: str = "",
        batch_delay: float = 0.5
    ) -> list:
        """
        Generate alt text for multiple images.

        Args:
            images: List of ExtractedImage objects
            context: Document context
            batch_delay: Delay between API calls to avoid rate limiting

        Returns:
            List of AltTextResult objects
        """
        results = []

        for i, image in enumerate(images):
            result = self.generate(image, context)
            results.append(result)

            # Apply delay between API calls (if using AI)
            if self.use_ai and i < len(images) - 1:
                time.sleep(batch_delay)

        return results
