# Test Fixtures Package
# Contains sample HTML and IMSCC files for testing

"""
Fixture directories:
- sample_html/: HTML files for accessibility and component testing
- sample_imscc/: Minimal IMSCC manifests and packages for extraction testing
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
SAMPLE_HTML_DIR = FIXTURES_DIR / 'sample_html'
SAMPLE_IMSCC_DIR = FIXTURES_DIR / 'sample_imscc'
