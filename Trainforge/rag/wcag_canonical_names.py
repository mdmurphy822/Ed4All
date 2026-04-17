"""WCAG 2.2 success-criterion canonical names and variant canonicalisation.

Chunks carry SC references in prose, in key-term metadata, and in concept
tags. A single SC appears under multiple labels across a real corpus (see
the v0.1.0 defect report in VERSIONING.md §7). This module ships:

- `WCAG_22_SC`: canonical name for every WCAG 2.2 success criterion.
- `canonicalize_sc_references(text)`: rewrite variants inside free text.
- `canonicalize_sc_tag(tag)`: rewrite tag-form variants (hyphenated) to
  the single form downstream consumers can rely on for retrieval and
  graph construction.

Canonical names follow the W3C WCAG 2.2 Recommendation wording.
"""
from __future__ import annotations

import re
from typing import Dict

# Canonical (SC number → canonical name) per W3C WCAG 2.2 Recommendation.
WCAG_22_SC: Dict[str, str] = {
    "1.1.1": "Non-text Content",
    "1.2.1": "Audio-only and Video-only (Prerecorded)",
    "1.2.2": "Captions (Prerecorded)",
    "1.2.3": "Audio Description or Media Alternative (Prerecorded)",
    "1.2.4": "Captions (Live)",
    "1.2.5": "Audio Description (Prerecorded)",
    "1.2.6": "Sign Language (Prerecorded)",
    "1.2.7": "Extended Audio Description (Prerecorded)",
    "1.2.8": "Media Alternative (Prerecorded)",
    "1.2.9": "Audio-only (Live)",
    "1.3.1": "Info and Relationships",
    "1.3.2": "Meaningful Sequence",
    "1.3.3": "Sensory Characteristics",
    "1.3.4": "Orientation",
    "1.3.5": "Identify Input Purpose",
    "1.3.6": "Identify Purpose",
    "1.4.1": "Use of Color",
    "1.4.2": "Audio Control",
    "1.4.3": "Contrast (Minimum)",
    "1.4.4": "Resize Text",
    "1.4.5": "Images of Text",
    "1.4.6": "Contrast (Enhanced)",
    "1.4.7": "Low or No Background Audio",
    "1.4.8": "Visual Presentation",
    "1.4.9": "Images of Text (No Exception)",
    "1.4.10": "Reflow",
    "1.4.11": "Non-text Contrast",
    "1.4.12": "Text Spacing",
    "1.4.13": "Content on Hover or Focus",
    "2.1.1": "Keyboard",
    "2.1.2": "No Keyboard Trap",
    "2.1.3": "Keyboard (No Exception)",
    "2.1.4": "Character Key Shortcuts",
    "2.2.1": "Timing Adjustable",
    "2.2.2": "Pause, Stop, Hide",
    "2.2.3": "No Timing",
    "2.2.4": "Interruptions",
    "2.2.5": "Re-authenticating",
    "2.2.6": "Timeouts",
    "2.3.1": "Three Flashes or Below Threshold",
    "2.3.2": "Three Flashes",
    "2.3.3": "Animation from Interactions",
    "2.4.1": "Bypass Blocks",
    "2.4.2": "Page Titled",
    "2.4.3": "Focus Order",
    "2.4.4": "Link Purpose (In Context)",
    "2.4.5": "Multiple Ways",
    "2.4.6": "Headings and Labels",
    "2.4.7": "Focus Visible",
    "2.4.8": "Location",
    "2.4.9": "Link Purpose (Link Only)",
    "2.4.10": "Section Headings",
    "2.4.11": "Focus Not Obscured (Minimum)",
    "2.4.12": "Focus Not Obscured (Enhanced)",
    "2.4.13": "Focus Appearance",
    "2.5.1": "Pointer Gestures",
    "2.5.2": "Pointer Cancellation",
    "2.5.3": "Label in Name",
    "2.5.4": "Motion Actuation",
    "2.5.5": "Target Size (Enhanced)",
    "2.5.6": "Concurrent Input Mechanisms",
    "2.5.7": "Dragging Movements",
    "2.5.8": "Target Size (Minimum)",
    "3.1.1": "Language of Page",
    "3.1.2": "Language of Parts",
    "3.1.3": "Unusual Words",
    "3.1.4": "Abbreviations",
    "3.1.5": "Reading Level",
    "3.1.6": "Pronunciation",
    "3.2.1": "On Focus",
    "3.2.2": "On Input",
    "3.2.3": "Consistent Navigation",
    "3.2.4": "Consistent Identification",
    "3.2.5": "Change on Request",
    "3.2.6": "Consistent Help",
    "3.3.1": "Error Identification",
    "3.3.2": "Labels or Instructions",
    "3.3.3": "Error Suggestion",
    "3.3.4": "Error Prevention (Legal, Financial, Data)",
    "3.3.5": "Help",
    "3.3.6": "Error Prevention (All)",
    "3.3.7": "Redundant Entry",
    "3.3.8": "Accessible Authentication (Minimum)",
    "3.3.9": "Accessible Authentication (Enhanced)",
    "4.1.2": "Name, Role, Value",
    "4.1.3": "Status Messages",
}


# Explicit drift-variant → canonical fragment.
# Keys are substrings (lowercased) that appear in real v0.1.0 corpora;
# values are the canonical name fragment to splice in. The full rewrite
# replaces the matched substring and strips any trailing Level-tag suffix.
_SC_NAME_VARIANTS: Dict[str, str] = {
    "contrast minimum, level aa": "Contrast (Minimum)",
    "contrast minimum, 4.5:1 for normal text": "Contrast (Minimum)",
    "contrast minimum": "Contrast (Minimum)",
    "contrast enhanced, level aaa": "Contrast (Enhanced)",
    "contrast enhanced": "Contrast (Enhanced)",
    "no keyboard trap, level a": "No Keyboard Trap",
    "no keyboard trap , level a": "No Keyboard Trap",
    "no keyboard trap": "No Keyboard Trap",
    "non-text content, level a": "Non-text Content",
    "non text content": "Non-text Content",
}


# Tag-form drift. Each alias key maps to the canonical hyphen-tag.
SC_TAG_ALIASES: Dict[str, str] = {
    "contrast-minimum-level-aa": "contrast-minimum",
    "contrast-minimum-4-5-1-for-normal-text": "contrast-minimum",
    "contrast-minimum-level": "contrast-minimum",
    "contrast-enhanced-level-aaa": "contrast-enhanced",
    "no-keyboard-trap-level-a": "no-keyboard-trap",
    "non-text-content-level-a": "non-text-content",
    "non-text-content": "non-text-content",
    "non-text": "non-text-content",
}


# Build a single ordered regex for text canonicalization: longest variant first
# so "contrast minimum, level aa" wins over "contrast minimum".
_VARIANT_PATTERN = re.compile(
    "|".join(re.escape(v) for v in sorted(_SC_NAME_VARIANTS, key=len, reverse=True)),
    re.IGNORECASE,
)


def canonicalize_sc_references(text: str) -> str:
    """Rewrite SC-name variants in free text to their canonical form.

    Only replaces patterns explicitly listed in `_SC_NAME_VARIANTS`. This is
    deliberately conservative — it doesn't try to rewrite arbitrary SC names
    in the WCAG table because real prose contains many SC mentions that are
    already correct, and aggressive rewriting would mangle them.
    """
    if not text:
        return text

    def _replace(m: re.Match) -> str:
        key = m.group(0).lower()
        canonical = _SC_NAME_VARIANTS.get(key)
        if canonical:
            return canonical
        # Fall back: try whitespace-normalised lookup.
        norm = re.sub(r"\s+", " ", key).strip()
        return _SC_NAME_VARIANTS.get(norm, m.group(0))

    return _VARIANT_PATTERN.sub(_replace, text)


def canonicalize_sc_tag(tag: str) -> str:
    """Rewrite tag-form SC variants to the canonical hyphen-tag."""
    if not tag:
        return tag
    return SC_TAG_ALIASES.get(tag.lower(), tag)
