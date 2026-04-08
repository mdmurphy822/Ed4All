# IMSCC Extractor Module
# Universal IMSCC package import and parsing

"""
This module provides universal IMSCC package extraction and parsing
supporting imports from Brightspace, Canvas, Blackboard, Moodle, Sakai,
and generic IMS Common Cartridge packages.
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ['IMSCCExtractor', 'ExtractedCourse', 'LMSType', 'ResourceType']
