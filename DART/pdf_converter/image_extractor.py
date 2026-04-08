"""
PDF Image Extraction Module

Extracts images from PDF documents using PyMuPDF and processes them
for embedding in accessible HTML.
"""

import base64
import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents an image extracted from a PDF."""
    data: bytes                    # Raw image bytes
    format: str                    # 'png', 'jpeg', etc.
    page: int                      # Page number (1-indexed)
    width: int                     # Image width in pixels
    height: int                    # Image height in pixels
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # Bounding box
    nearby_caption: str = ""       # Caption text found near image
    data_uri: str = ""             # Base64 data URI for embedding
    image_hash: str = ""           # Hash for deduplication
    alt_text: str = ""             # Generated alt text
    long_description: str = ""     # Extended description
    is_vector_render: bool = False # True if rendered from vector graphics


class VectorRegionExtractor:
    """
    Detect and extract vector graphics regions from PDF pages.

    Vector graphics (lines, curves, shapes) are rendered as PNG images
    for accessibility processing.
    """

    def __init__(
        self,
        min_drawings: int = 5,
        cluster_distance: float = 50.0,
        render_dpi: int = 150,
        min_region_size: Tuple[int, int] = (50, 50),
        max_region_ratio: float = 0.9,
    ):
        """
        Initialize vector region extractor.

        Args:
            min_drawings: Minimum drawing operations to consider a region
            cluster_distance: Distance in points to cluster nearby drawings
            render_dpi: DPI for rendering regions as images
            min_region_size: Minimum (width, height) for a valid region
            max_region_ratio: Maximum ratio of page size (filters full-page borders)
        """
        self.min_drawings = min_drawings
        self.cluster_distance = cluster_distance
        self.render_dpi = render_dpi
        self.min_region_size = min_region_size
        self.max_region_ratio = max_region_ratio

    def extract_regions(self, page, page_num: int) -> List[ExtractedImage]:
        """
        Extract vector regions from a page.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            List of ExtractedImage objects from rendered vector regions
        """
        try:
            drawings = page.get_drawings()
        except Exception as e:
            logger.debug(f"Could not get drawings from page {page_num + 1}: {e}")
            return []

        if len(drawings) < self.min_drawings:
            return []

        # Cluster drawings into regions
        regions = self._cluster_drawings(drawings, page.rect)

        if not regions:
            return []

        logger.debug(f"Page {page_num + 1}: Found {len(regions)} vector regions from {len(drawings)} drawings")

        # Render each region as an image
        images = []
        for i, region_bbox in enumerate(regions):
            try:
                img = self._render_region(page, region_bbox, page_num, i)
                if img:
                    images.append(img)
            except Exception as e:
                logger.warning(f"Failed to render vector region {i} on page {page_num + 1}: {e}")

        return images

    def _cluster_drawings(self, drawings: List, page_rect) -> List:
        """
        Cluster nearby drawings into diagram regions.

        Args:
            drawings: List of drawing dicts from page.get_drawings()
            page_rect: Page bounding rectangle

        Returns:
            List of fitz.Rect bounding boxes for each cluster
        """
        if not drawings:
            return []

        try:
            import fitz
        except ImportError:
            return []

        # Get bounding boxes and expand by cluster distance
        boxes = []
        for d in drawings:
            rect = d.get("rect")
            if rect:
                # Expand box for clustering
                expanded = fitz.Rect(
                    rect.x0 - self.cluster_distance,
                    rect.y0 - self.cluster_distance,
                    rect.x1 + self.cluster_distance,
                    rect.y1 + self.cluster_distance,
                )
                boxes.append(expanded)

        if not boxes:
            return []

        # Merge overlapping boxes
        merged = self._merge_overlapping_boxes(boxes, fitz)

        # Filter regions
        valid_regions = []
        page_width = page_rect.width
        page_height = page_rect.height

        for box in merged:
            # Shrink back by cluster distance to get actual region
            actual = fitz.Rect(
                box.x0 + self.cluster_distance,
                box.y0 + self.cluster_distance,
                box.x1 - self.cluster_distance,
                box.y1 - self.cluster_distance,
            )

            # Clip to page bounds
            actual = actual & page_rect

            if actual.is_empty:
                continue

            width = actual.width
            height = actual.height

            # Filter by size
            if width < self.min_region_size[0] or height < self.min_region_size[1]:
                continue

            # Filter out page-spanning regions (likely borders/frames)
            if width > page_width * self.max_region_ratio and height > page_height * self.max_region_ratio:
                continue

            valid_regions.append(actual)

        return valid_regions

    def _merge_overlapping_boxes(self, boxes: List, fitz) -> List:
        """
        Merge overlapping bounding boxes.

        Args:
            boxes: List of fitz.Rect objects
            fitz: PyMuPDF module

        Returns:
            List of merged fitz.Rect objects
        """
        if not boxes:
            return []

        # Sort boxes by x0 coordinate
        boxes = sorted(boxes, key=lambda b: (b.y0, b.x0))

        merged = []
        current = boxes[0]

        for box in boxes[1:]:
            # Check if boxes overlap or are close
            if current.intersects(box):
                # Merge into current
                current = current | box  # Union
            else:
                merged.append(current)
                current = box

        merged.append(current)

        # Multiple passes to catch transitive overlaps
        changed = True
        while changed:
            changed = False
            new_merged = []
            used = set()

            for i, box1 in enumerate(merged):
                if i in used:
                    continue

                current = box1
                for j, box2 in enumerate(merged):
                    if j <= i or j in used:
                        continue

                    if current.intersects(box2):
                        current = current | box2
                        used.add(j)
                        changed = True

                new_merged.append(current)
                used.add(i)

            merged = new_merged

        return merged

    def _render_region(
        self,
        page,
        bbox,
        page_num: int,
        region_index: int
    ) -> Optional[ExtractedImage]:
        """
        Render a vector region as a PNG image.

        Args:
            page: PyMuPDF page object
            bbox: Region bounding box
            page_num: Page number (0-indexed)
            region_index: Index of region on this page

        Returns:
            ExtractedImage or None
        """
        try:
            # Add padding to capture text labels beyond drawing boundaries
            import fitz
            padded = fitz.Rect(
                bbox.x0 - 40,
                bbox.y0 - 40,
                bbox.x1 + 40,
                bbox.y1 + 40,
            )
            padded = padded & page.rect  # Clip to page

            # Render to pixmap
            mat = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)
            pixmap = page.get_pixmap(matrix=mat, clip=padded, alpha=False)

            # Convert to PNG bytes
            png_data = pixmap.tobytes("png")

            # Calculate hash for deduplication
            image_hash = hashlib.md5(png_data).hexdigest()

            # Create data URI
            data_uri = f"data:image/png;base64,{base64.b64encode(png_data).decode()}"

            return ExtractedImage(
                data=png_data,
                format='png',
                page=page_num + 1,
                width=pixmap.width,
                height=pixmap.height,
                bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                data_uri=data_uri,
                image_hash=image_hash,
                is_vector_render=True,
                nearby_caption="",  # Will be filled by caption finder
            )

        except Exception as e:
            logger.warning(f"Failed to render vector region: {e}")
            return None


class PDFImageExtractor:
    """Extract images from PDF documents using PyMuPDF."""

    # Minimum dimensions to consider (skip tiny icons/spacers)
    MIN_WIDTH = 50
    MIN_HEIGHT = 50

    # Maximum image size in bytes (5MB)
    MAX_SIZE_BYTES = 5 * 1024 * 1024

    # Caption detection patterns
    CAPTION_PATTERNS = [
        re.compile(r'^(Figure|Fig\.?)\s*\d+[.:]?\s*(.*)$', re.IGNORECASE),
        re.compile(r'^(Image|Diagram|Chart|Graph|Photo)\s*\d*[.:]?\s*(.*)$', re.IGNORECASE),
    ]

    def __init__(
        self,
        pdf_path: str,
        extract_vector_graphics: bool = True,
        vector_min_drawings: int = 5,
        vector_cluster_distance: float = 50.0,
        vector_render_dpi: int = 150,
    ):
        """
        Initialize extractor with PDF path.

        Args:
            pdf_path: Path to PDF file
            extract_vector_graphics: Whether to extract vector graphics as images
            vector_min_drawings: Minimum drawings to consider a region
            vector_cluster_distance: Distance to cluster nearby drawings
            vector_render_dpi: DPI for rendering vector regions
        """
        self.pdf_path = pdf_path
        self.extract_vector_graphics = extract_vector_graphics
        self._doc = None
        self._fitz = None

        # Initialize vector extractor if enabled
        self._vector_extractor = None
        if extract_vector_graphics:
            self._vector_extractor = VectorRegionExtractor(
                min_drawings=vector_min_drawings,
                cluster_distance=vector_cluster_distance,
                render_dpi=vector_render_dpi,
            )

        try:
            import fitz  # PyMuPDF
            self._fitz = fitz
            self._doc = fitz.open(pdf_path)
            logger.debug(f"Opened PDF with {len(self._doc)} pages")
        except ImportError:
            logger.warning("PyMuPDF not installed. Image extraction unavailable.")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")

    def extract_all(self) -> List[ExtractedImage]:
        """
        Extract all images from the PDF.

        Returns:
            List of ExtractedImage objects
        """
        if not self._doc:
            return []

        images = []
        seen_hashes = set()

        for page_num in range(len(self._doc)):
            page_images = self.extract_from_page(page_num)

            # Deduplicate within document
            for img in page_images:
                if img.image_hash not in seen_hashes:
                    seen_hashes.add(img.image_hash)
                    images.append(img)

        logger.info(f"Extracted {len(images)} unique images from PDF")
        return images

    def extract_from_page(self, page_num: int) -> List[ExtractedImage]:
        """
        Extract images from a specific page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            List of ExtractedImage objects from this page
        """
        if not self._doc or page_num >= len(self._doc):
            return []

        images = []
        page = self._doc[page_num]

        # Extract raster images
        try:
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                try:
                    extracted = self._extract_image(page, img_info, page_num)
                    if extracted:
                        # Try to find nearby caption
                        extracted.nearby_caption = self._find_nearby_caption(
                            page, extracted.bbox
                        )
                        images.append(extracted)
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")

        except Exception as e:
            logger.error(f"Failed to get images from page {page_num + 1}: {e}")

        # Extract vector graphics regions
        if self._vector_extractor:
            try:
                vector_images = self._vector_extractor.extract_regions(page, page_num)
                for vimg in vector_images:
                    # Try to find nearby caption for vector regions
                    vimg.nearby_caption = self._find_nearby_caption(page, vimg.bbox)
                    images.append(vimg)
            except Exception as e:
                logger.warning(f"Failed to extract vector regions from page {page_num + 1}: {e}")

        return images

    def _extract_image(
        self,
        page,
        img_info: tuple,
        page_num: int
    ) -> Optional[ExtractedImage]:
        """
        Extract a single image from page.

        Args:
            page: PyMuPDF page object
            img_info: Image info tuple from get_images()
            page_num: Page number (0-indexed)

        Returns:
            ExtractedImage or None if extraction fails
        """
        xref = img_info[0]

        try:
            base_image = self._doc.extract_image(xref)
        except Exception as e:
            logger.debug(f"Could not extract image xref {xref}: {e}")
            return None

        if not base_image:
            return None

        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)

        # Skip tiny images
        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            logger.debug(f"Skipping small image: {width}x{height}")
            return None

        # Skip oversized images
        if len(image_bytes) > self.MAX_SIZE_BYTES:
            logger.warning(f"Skipping oversized image: {len(image_bytes) / 1024 / 1024:.1f}MB")
            return None

        # Calculate hash for deduplication
        image_hash = hashlib.md5(image_bytes).hexdigest()

        # Find bounding box on page
        bbox = self._get_image_bbox(page, xref)

        # Generate data URI
        mime_type = self._get_mime_type(image_ext)
        data_uri = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

        return ExtractedImage(
            data=image_bytes,
            format=image_ext,
            page=page_num + 1,  # 1-indexed for display
            width=width,
            height=height,
            bbox=bbox,
            data_uri=data_uri,
            image_hash=image_hash,
        )

    def _get_image_bbox(self, page, xref: int) -> Tuple[float, float, float, float]:
        """
        Get bounding box for image on page.

        Args:
            page: PyMuPDF page object
            xref: Image xref

        Returns:
            Bounding box (x0, y0, x1, y1)
        """
        try:
            for img in page.get_images(full=True):
                if img[0] == xref:
                    # Get image rectangle
                    img_rects = page.get_image_rects(img)
                    if img_rects:
                        rect = img_rects[0]
                        return (rect.x0, rect.y0, rect.x1, rect.y1)
        except Exception as e:
            logger.debug(f"Could not get bbox for image: {e}")

        return (0, 0, 0, 0)

    def _find_nearby_caption(
        self,
        page,
        bbox: Tuple[float, float, float, float]
    ) -> str:
        """
        Find caption text near an image.

        Args:
            page: PyMuPDF page object
            bbox: Image bounding box

        Returns:
            Caption text if found, empty string otherwise
        """
        if bbox == (0, 0, 0, 0):
            return ""

        try:
            x0, y0, x1, y1 = bbox

            # Search in area below the image (most common caption location)
            search_rect = self._fitz.Rect(
                x0 - 10,           # Slightly wider
                y1,                 # Start at bottom of image
                x1 + 10,
                y1 + 100            # Look 100 points below
            )

            text_below = page.get_text("text", clip=search_rect).strip()

            # Check if it matches caption pattern
            for pattern in self.CAPTION_PATTERNS:
                match = pattern.match(text_below.split('\n')[0] if text_below else "")
                if match:
                    # Return full caption (may span multiple lines)
                    lines = text_below.split('\n')
                    caption_lines = [lines[0]]
                    # Include continuation lines (no new caption start)
                    for line in lines[1:4]:  # Max 4 lines
                        if not any(p.match(line) for p in self.CAPTION_PATTERNS):
                            if line.strip():
                                caption_lines.append(line.strip())
                        else:
                            break
                    return ' '.join(caption_lines)

            # Also check above the image
            search_rect = self._fitz.Rect(
                x0 - 10,
                y0 - 60,            # Look 60 points above
                x1 + 10,
                y0
            )

            text_above = page.get_text("text", clip=search_rect).strip()
            for pattern in self.CAPTION_PATTERNS:
                match = pattern.match(text_above.split('\n')[-1] if text_above else "")
                if match:
                    return text_above.split('\n')[-1]

        except Exception as e:
            logger.debug(f"Caption search failed: {e}")

        return ""

    def _get_mime_type(self, ext: str) -> str:
        """Get MIME type from file extension."""
        mime_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'tif': 'image/tiff',
        }
        return mime_map.get(ext.lower(), 'image/png')

    def close(self):
        """Close the PDF document."""
        if self._doc:
            self._doc.close()
            self._doc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ImageProcessor:
    """Process and optimize images for HTML embedding."""

    def __init__(self, max_width: int = 800, quality: int = 85):
        """
        Initialize processor.

        Args:
            max_width: Maximum image width in pixels
            quality: JPEG compression quality (1-100)
        """
        self.max_width = max_width
        self.quality = quality
        self._pil = None

        try:
            from PIL import Image
            self._pil = Image
            logger.debug("PIL available for image processing")
        except ImportError:
            logger.warning("PIL not installed. Image processing unavailable.")

    def process(self, image: ExtractedImage) -> ExtractedImage:
        """
        Process image: resize if needed and compress.

        Args:
            image: ExtractedImage to process

        Returns:
            Processed ExtractedImage with updated data_uri
        """
        if not self._pil:
            return image

        try:
            img = self._pil.open(io.BytesIO(image.data))

            # Resize if wider than max
            if img.width > self.max_width:
                ratio = self.max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((self.max_width, new_height), self._pil.Resampling.LANCZOS)
                image.width = self.max_width
                image.height = new_height

            # Convert to RGB if needed (for JPEG)
            if img.mode in ('RGBA', 'P', 'LA'):
                background = self._pil.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
                img = background

            # Compress to JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=self.quality, optimize=True)
            image.data = output.getvalue()
            image.format = 'jpeg'

            # Update data URI
            image.data_uri = f"data:image/jpeg;base64,{base64.b64encode(image.data).decode()}"

        except Exception as e:
            logger.warning(f"Image processing failed: {e}")

        return image

    def process_all(self, images: List[ExtractedImage]) -> List[ExtractedImage]:
        """
        Process all images.

        Args:
            images: List of ExtractedImage objects

        Returns:
            List of processed ExtractedImage objects
        """
        return [self.process(img) for img in images]
