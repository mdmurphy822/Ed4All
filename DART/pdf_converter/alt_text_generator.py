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
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
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
        model: str = "claude-sonnet-4-20250514",
        use_ai: bool = True,
        use_ocr_fallback: bool = True
    ):
        """
        Initialize alt text generator.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env)
            model: Claude model to use for vision
            use_ai: Whether to use Claude API for alt text
            use_ocr_fallback: Whether to fall back to OCR
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model
        self.use_ai = use_ai and self.api_key is not None
        self.use_ocr_fallback = use_ocr_fallback

        self._client = None
        if self.use_ai:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.debug(f"Claude API initialized with model {model}")
            except ImportError:
                logger.warning("anthropic package not installed. AI alt text unavailable.")
                self.use_ai = False
            except Exception as e:
                logger.warning(f"Failed to initialize Claude API: {e}")
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
        # Try Claude API first
        if self.use_ai and self._client:
            result = self._try_claude_with_retry(image, context)
            if result.success:
                return result

        # Try OCR fallback
        if self.use_ocr_fallback and self._pytesseract:
            result = self._try_ocr_fallback(image)
            if result.success and result.alt_text:
                return result

        # Try caption-based fallback
        if image.nearby_caption:
            return self._use_caption_fallback(image)

        # Final generic fallback
        return self._generic_fallback(image)

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
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return self._call_claude_vision(image, context)

            except Exception as e:
                error_str = str(e).lower()
                last_error = e

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

        # Call Claude API
        response = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        # Parse response
        response_text = response.content[0].text.strip()

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
