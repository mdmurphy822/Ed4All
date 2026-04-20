#!/usr/bin/env python3
"""
DART Multi-Source Interpreter

Synthesizes combined JSON from multiple extraction sources (pdftotext, pdfplumber, OCR)
to generate accessible HTML. Uses multi-source synthesis where Claude determines
the best data from each source.

Architecture:
- pdftotext: Text accuracy (99%+), URLs, phone/email
- pdfplumber: Table structure (headers, rows, cols)
- OCR: Layout/position validation

Each section is synthesized by combining the best of all sources.
"""

import hashlib
import html
import json
import logging
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Add lib to path for decision capture
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
try:
    from decision_capture import DecisionCapture  # noqa: F401
    CAPTURE_AVAILABLE = True
except ImportError:
    CAPTURE_AVAILABLE = False

# Add Courseforge accessibility validator to path
_VALIDATOR_DIR = (
    Path(__file__).parent.parent
    / "Courseforge" / "scripts" / "accessibility-validator"
)
if str(_VALIDATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATOR_DIR))
try:
    from accessibility_validator import AccessibilityValidator, IssueSeverity
    WCAG_VALIDATOR_AVAILABLE = True
except ImportError:
    WCAG_VALIDATOR_AVAILABLE = False

# Import semantic structure extractor for content profiling
try:
    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        SemanticStructureExtractor,
    )
    SEMANTIC_EXTRACTOR_AVAILABLE = True
except ImportError:
    SEMANTIC_EXTRACTOR_AVAILABLE = False

# =============================================================================
# CAMPUS NAME MAPPING
# =============================================================================

CAMPUS_NAMES = {
    "ADI": "Adirondack Community College",
    "BRK": "SUNY Downstate Medical Center",
    "BRM": "SUNY Broome",
    "BUC": "Buffalo State University",
    "CAN": "Canisius University",
    "CAY": "Cayuga Community College",
    "CGC": "Columbia-Greene Community College",
    "CNG": "Corning Community College",
    "DEL": "SUNY Delhi",
    "DUT": "Dutchess Community College",
    "FAR": "Farmingdale State College",
    "FIT": "Fashion Institute of Technology",
    "FLC": "Finger Lakes Community College",
    "FMC": "Fulton-Montgomery Community College",
    "FRE": "SUNY Fredonia",
    "GNC": "Genesee Community College",
    "HER": "Herkimer College",
    "HVC": "Hudson Valley Community College",
    "JAM": "Jamestown Community College",
    "JEF": "Jefferson Community College",
    "MAR": "SUNY Maritime College",
    "MON": "Monroe Community College",
    "MOR": "Morrisville State College",
    "NAS": "Nassau Community College",
    "NIA": "Niagara County Community College",
    "NOR": "North Country Community College",
    "OLD": "SUNY Old Westbury",
    "ONE": "SUNY Oneonta",
    "ONO": "Onondaga Community College",
    "ORA": "Orange County Community College",
    "OSW": "SUNY Oswego",
    "PLA": "SUNY Plattsburgh",
    "POT": "SUNY Potsdam",
    "PUR": "SUNY Purchase",
    "ROC": "Rochester Institute of Technology",
    "SCH": "Schenectady County Community College",
    "STB": "Stony Brook University",
    "SUF": "Suffolk County Community College",
    "SYR": "Syracuse University",
    "TC3": "Tompkins Cortland Community College",
    "ULS": "Ulster County Community College",
    "UTI": "Utica University",
    "WES": "Westchester Community College",
}

# =============================================================================
# PROVENANCE CONSTANTS (Wave 8)
# =============================================================================

# Canonical confidence scale (P1 decision). Also documented in DART/CLAUDE.md.
# Every matcher branch selects one of these values; downstream consumers
# (Courseforge source-router, Trainforge inference rules) read them as-is.
CONFIDENCE_DIRECT_TABLE = 1.0       # pdfplumber structured row/cell
CONFIDENCE_NAME_PATTERN = 0.8       # name-pattern match (e.g. jdoe@)
CONFIDENCE_PROXIMITY = 0.6          # near-by-text proximity match
CONFIDENCE_DERIVATION = 0.4         # synthesized from email local-part etc.
CONFIDENCE_OCR_FALLBACK = 0.2       # pure OCR, no other source corroborates

# Typed extractor enum values (matches SourceReference schema).
SOURCE_PDFTOTEXT = "pdftotext"
SOURCE_PDFPLUMBER = "pdfplumber"
SOURCE_OCR = "ocr"
SOURCE_SYNTHESIZED = "synthesized"
SOURCE_CLAUDE = "claude"


def _document_slug(code: str) -> str:
    """Normalize a document identifier (campus code / stem) to the slug form
    used in ``dart:{slug}#{block}`` source IDs."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(code)).strip("_").lower()
    return slug or "document"


def _content_hash_block_id(*parts: str) -> str:
    """Return a 16-hex content-hash block ID.

    Used when ``TRAINFORGE_CONTENT_HASH_IDS`` is truthy to produce re-chunk-
    stable block identifiers. The fallback positional form (``s3_c0``) is
    emitted otherwise so legacy corpora keep their existing IDs.
    """
    joined = "\x1f".join(p for p in parts if p is not None)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _use_content_hash_ids() -> bool:
    return os.environ.get("TRAINFORGE_CONTENT_HASH_IDS", "").lower() in (
        "1", "true", "yes", "on",
    )


def _make_block_id(section_id: str, positional: str, *content_parts: str) -> str:
    """Build a block ID using either the content-hash or positional scheme.

    Positional IDs look like ``s3_c0`` and match the section record's
    ``section_id``. Content-hash IDs collapse to a 16-hex string that is
    stable across re-runs. Both schemes validate against the canonical
    ``source_reference.schema.json`` pattern.
    """
    if _use_content_hash_ids() and content_parts:
        return _content_hash_block_id(section_id, *content_parts)
    return positional


def _envelope(
    value: Any,
    source: str,
    *,
    pages: Optional[List[int]] = None,
    confidence: float = 1.0,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a per-block provenance envelope.

    Shape:
        {"value": ..., "source": "pdfplumber", "pages": [3],
         "confidence": 0.87, "method": "name_pattern"}

    ``value`` is the leaf payload (string / list / dict). ``source`` is a
    typed extractor enum value. ``pages`` is emitted as an empty list when
    unknown (per design: real per-block page tracking is deferred — we
    model the field but do not guess).
    """
    env: Dict[str, Any] = {
        "value": value,
        "source": source,
        "pages": list(pages or []),
        "confidence": float(confidence),
    }
    if method:
        env["method"] = method
    return env


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """Clean pdftotext output."""
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('\f', '\n\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fix_broken_urls(text: str) -> str:
    """Rejoin URLs split across lines."""
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line:
                break
            if re.search(r'(https?://[^\s]*[=/])$', line.rstrip()):
                line = line.rstrip() + next_line
                i += 1
                continue
            if re.search(r'\?[A-Za-z]+=\d*$', line) and re.match(r'^\d+$', next_line):
                line = line.rstrip() + next_line
                i += 1
                continue
            if line.rstrip() and line.rstrip()[-1].islower() and re.match(r'^[a-z]{1,3}$', next_line):
                line = line.rstrip() + next_line
                i += 1
                continue
            break
        result.append(line)
        i += 1
    return '\n'.join(result)


def linkify(text: str) -> str:
    """Convert URLs to anchor tags."""
    url_pattern = r'(https?://[^\s<>"\']+)'
    def repl(m):
        url = m.group(1).rstrip('.,;:)')
        display = url if len(url) <= 60 else url[:57] + '...'
        return f'<a href="{html.escape(url)}">{html.escape(display)}</a>'
    return re.sub(url_pattern, repl, text)


def extract_kv_pairs(lines: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Extract key: value pairs from lines."""
    pairs = []
    remaining = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if re.match(r'^https?://', line):
            remaining.append(line)
            i += 1
            continue

        match = re.match(r'^([A-Za-z][A-Za-z\s/&()\-\']{1,35}):\s*(.*)$', line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()

            if key.lower() in ('http', 'https', 'ftp'):
                remaining.append(line)
                i += 1
                continue

            while i + 1 < len(lines):
                next_line = lines[i + 1]
                if not next_line:
                    break
                if re.match(r'^[A-Za-z][A-Za-z\s/&()\-\']{1,35}:\s', next_line):
                    break
                value = value + ' ' + next_line
                i += 1

            pairs.append((key, value.strip()))
        else:
            remaining.append(line)
        i += 1

    return pairs, remaining


# =============================================================================
# SECTION PARSING
# =============================================================================

def parse_sections_from_text(text: str) -> List[Dict]:
    """Parse sections from clean pdftotext."""
    text = clean_text(text)
    text = fix_broken_urls(text)

    sections = []
    current = {'title': 'Overview', 'type': 'overview', 'content': []}

    section_patterns = [
        (r'campus\s*information', 'Campus Information', 'campus-info'),
        (r'username.*password', 'Username/Password/Accounts', 'credentials'),
        (r'no\s*account', 'No Account', 'no-account'),
        (r'guest\s*/?\s*observer', 'Guest / Observer', 'guest'),
        (r'campus\s*contacts?', 'Campus Contacts', 'contacts'),
        (r'course\s*/?\s*roster', 'Course / Roster', 'roster'),
        (r'campus\s*systems?\s*(and)?\s*resources?', 'Campus Systems and Resources', 'systems'),
    ]

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        line_lower = stripped.lower()
        found = False
        for pattern, title, stype in section_patterns:
            if re.search(pattern, line_lower):
                if current['content']:
                    sections.append(current)
                current = {'title': title, 'type': stype, 'content': []}
                found = True
                break

        if not found:
            current['content'].append(stripped)

    if current['content']:
        sections.append(current)

    # Merge consecutive sections of same type
    merged = []
    for section in sections:
        if merged and merged[-1]['type'] == section['type']:
            merged[-1]['content'].extend(section['content'])
        else:
            merged.append(section)

    return merged


# =============================================================================
# MULTI-SOURCE CONTEXT BUILDING
# =============================================================================

def estimate_page_range(section: Dict, total_pages: int = 6) -> Tuple[int, int]:
    """Estimate which pages a section spans based on section type."""
    stype = section['type']
    page_ranges = {
        'overview': (1, 1),
        'campus-info': (1, 2),
        'credentials': (2, 3),
        'no-account': (2, 3),
        'guest': (2, 3),
        'contacts': (3, 4),
        'roster': (4, 5),
        'systems': (5, total_pages),
    }
    return page_ranges.get(stype, (1, total_pages))


def build_section_context(section: Dict, tables: List[Dict],
                          ocr_pages: List[Dict], total_pages: int = 6) -> Dict:
    """
    Combine ALL extraction sources for a section.
    This context is used for multi-source synthesis.
    """
    page_range = estimate_page_range(section, total_pages)

    # Get tables from relevant pages
    relevant_tables = []
    for t in tables:
        page = t.get('page', 0)
        if page_range[0] <= page <= page_range[1]:
            table_info = {
                'page': page,
                'index': t.get('index', 0),
                'headers': t.get('headers', []),
                'rows': t.get('rows', []),
                'num_cols': t.get('num_cols', 0),
                'num_rows': t.get('num_rows', 0),
            }
            if table_info['rows']:
                first_row = table_info['rows'][0]
                empty_cells = sum(1 for c in first_row if not c or not str(c).strip())
                table_info['quality'] = 'headers_only' if empty_cells > len(first_row) * 0.5 else 'complete'
            else:
                table_info['quality'] = 'empty'
            relevant_tables.append(table_info)

    # Get OCR from relevant pages
    relevant_ocr = []
    for o in ocr_pages:
        page = o.get('page', 0)
        if page_range[0] <= page <= page_range[1]:
            relevant_ocr.append({
                'page': page,
                'text': o.get('ocr_text', '')[:1000]
            })

    # Pre-extract entities from pdftotext
    text = '\n'.join(section['content'])
    phones = re.findall(r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})', text)
    emails = re.findall(r'([\w.-]+@[\w.-]+\.\w+)', text)
    urls = re.findall(r'(https?://[^\s]+)', text)
    names = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z\-]+)', text)

    return {
        'section_type': section['type'],
        'section_title': section['title'],
        'page_range': page_range,
        'sources': {
            'pdftotext': section['content'],
            'tables': relevant_tables,
            'ocr': relevant_ocr
        },
        'entities': {
            'phones': list(set(phones)),
            'emails': list(set(emails)),
            'urls': list(set(urls)),
            'names': list(set(names))
        }
    }


def export_section_contexts(combined: Dict) -> List[Dict]:
    """
    Parse sections and build multi-source contexts for each.
    """
    sections = parse_sections_from_text(combined['sources']['pdftotext'])
    tables = combined['sources'].get('tables', [])
    ocr_pages = combined['sources'].get('ocr_pages', [])

    total_pages = max(
        max((o.get('page', 0) for o in ocr_pages), default=6),
        max((t.get('page', 0) for t in tables), default=6)
    )

    contexts = []
    for section in sections:
        ctx = build_section_context(section, tables, ocr_pages, total_pages)
        contexts.append(ctx)

    return contexts


# =============================================================================
# MULTI-SOURCE SYNTHESIS FUNCTIONS
# =============================================================================

def _is_likely_contact_name(name: str) -> bool:
    """Check if a string looks like a contact name or entity."""
    if not name or len(name) < 3:
        return False

    patterns = [
        r'^[A-Z][a-z]+\s+[A-Z][a-z\-]+$',
        r'^Help\s*Desk',
        r'^Service\s*Desk',
        r'^(?:Academic|Student|Tech)',
        r'^(?:IT|LMS)\s+',
        r'^Registrar',
        r'^(?:Primary|Secondary)\s+',
        r'^Authentication',
    ]

    for pattern in patterns:
        if re.match(pattern, name, re.IGNORECASE):
            return True
    return False


def match_email_by_name(
    name: str, emails: List[str], text: str
) -> Tuple[str, str, float]:
    """Match an email to a contact name by pattern or proximity.

    Returns a 3-tuple ``(email, method, confidence)``. ``method`` is one of
    ``name_pattern`` / ``proximity`` / ``special_case`` (helpdesk/registrar)
    and seeds the per-block provenance envelope; an empty email is returned
    with method ``""`` and confidence ``0.0`` when no match is found.
    """
    name_lower = name.lower()
    name_parts = name_lower.split()

    # Special cases
    if 'help desk' in name_lower or 'service desk' in name_lower:
        for email in emails:
            if 'help@' in email.lower() or 'helpdesk@' in email.lower():
                return email, "special_case", CONFIDENCE_NAME_PATTERN

    if 'registrar' in name_lower:
        for email in emails:
            if 'registrar' in email.lower():
                return email, "special_case", CONFIDENCE_NAME_PATTERN

    # Name pattern matching
    if len(name_parts) >= 2:
        first = name_parts[0]
        last = name_parts[-1]

        patterns = [
            first + last,
            first[0] + last,
            last + first[0],
            first + '.' + last,
            first + '_' + last,
            first[:1] + '.' + last,
            last[:4],
        ]

        for email in emails:
            email_local = email.split('@')[0].lower().replace('.', '').replace('_', '')
            for pattern in patterns:
                pattern_clean = pattern.replace('.', '').replace('_', '')
                if pattern_clean in email_local or email_local.startswith(pattern_clean[:4]):
                    return email, "name_pattern", CONFIDENCE_NAME_PATTERN

    # Proximity matching
    name_pos = text.lower().find(name_lower)
    if name_pos >= 0:
        for email in emails:
            email_pos = text.lower().find(email.lower())
            if email_pos >= 0 and abs(name_pos - email_pos) < 300:
                return email, "proximity", CONFIDENCE_PROXIMITY

    return "", "", 0.0


def match_phone_by_proximity(
    name: str, email: str, phones: List[str], text: str
) -> Tuple[str, str, float]:
    """Match a phone number by proximity to name or email in text.

    Returns ``(phone, method, confidence)``. Method is ``proximity`` (both
    email-anchored and name-anchored) with the proximity confidence value.
    """
    if email:
        email_pos = text.lower().find(email.lower())
        if email_pos >= 0:
            for phone in phones:
                phone_pos = text.find(phone)
                if phone_pos >= 0 and abs(phone_pos - email_pos) < 200:
                    return (
                        phone.replace('.', '-').replace(' ', '-'),
                        "proximity",
                        CONFIDENCE_PROXIMITY,
                    )

    name_pos = text.lower().find(name.lower())
    if name_pos >= 0:
        for phone in phones:
            phone_pos = text.find(phone)
            if phone_pos >= 0 and abs(phone_pos - name_pos) < 200:
                return (
                    phone.replace('.', '-').replace(' ', '-'),
                    "proximity",
                    CONFIDENCE_PROXIMITY,
                )

    return "", "", 0.0


def extract_title_near_name(name: str, text: str) -> str:
    """Extract job title near a person's name in text."""
    title_patterns = [
        r'((?:Coordinator|Director|Manager|Specialist|Administrator|Supervisor)[^,\n]{0,30})',
        r'((?:Associate|Assistant|Senior|Chief)\s+(?:Dean|Director|Manager)[^,\n]{0,30})',
        r'((?:VP|Vice President|AVP)[^,\n]{0,30})',
        r'((?:IT|LMS|Academic|Student|Technology)\s+(?:Manager|Director|Coordinator|Specialist|Administrator)[^,\n]{0,30})',
    ]

    name_pos = text.lower().find(name.lower())
    if name_pos < 0:
        return ''

    context = text[max(0, name_pos - 50):name_pos + len(name) + 150]

    for pattern in title_patterns:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            if len(title) > 5:
                return title

    return ''


def synthesize_contacts(
    tables: List[Dict],
    entities: Dict,
    pdftotext: List[str],
    *,
    section_id: str = "s",
    section_pages: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Synthesize contacts using multi-source approach:
    1. Get headers from pdfplumber tables (contact names)
    2. Extract phones/emails from pdftotext
    3. Match by: email prefix, name proximity, pattern recognition

    Each returned contact carries per-field provenance envelopes:
        {
          "block_id": "s3_c0",
          "name":  {"value": ..., "source": ..., "pages": [...], "confidence": ...},
          "email": {...},
          "phone": {...},
          "title": {...},
          ...
        }

    Plus top-level plain-string fields (``name``, ``email``, ``phone``,
    ``title``, ``notes``) for renderer / legacy consumer backward compat.
    """
    text = '\n'.join(pdftotext)
    phones = entities.get('phones', [])
    emails = entities.get('emails', [])
    pages = list(section_pages or [])

    # Get contact names from table headers
    contact_names: List[Tuple[str, str]] = []  # (name, origin_source)
    seen_names = set()

    for table in tables:
        for h in table.get('headers', []):
            name = h.replace('\n', ' ').strip()
            if name and _is_likely_contact_name(name) and name not in seen_names:
                contact_names.append((name, SOURCE_PDFPLUMBER))
                seen_names.add(name)

    # If no table headers, look for names in entities (pdftotext-derived)
    if not contact_names and entities.get('names'):
        for name in entities['names']:
            if name not in seen_names:
                contact_names.append((name, SOURCE_PDFTOTEXT))
                seen_names.add(name)

    # Build contacts with entity matching
    used_emails = set()
    used_phones = set()
    contacts: List[Dict[str, Any]] = []

    for idx, (name, name_source) in enumerate(contact_names):
        positional = f"{section_id}_c{idx}"
        block_id = _make_block_id(section_id, positional, "contact", name)

        name_conf = (
            CONFIDENCE_DIRECT_TABLE if name_source == SOURCE_PDFPLUMBER
            else CONFIDENCE_NAME_PATTERN
        )
        contact: Dict[str, Any] = {
            "block_id": block_id,
            "name": name,  # back-compat plain string
            "phone": "",
            "email": "",
            "title": "",
            "notes": "",
            "name_provenance": _envelope(
                name, name_source, pages=pages, confidence=name_conf,
                method="table_header" if name_source == SOURCE_PDFPLUMBER else "entity_extraction",
            ),
        }

        email, email_method, email_conf = match_email_by_name(
            name, [e for e in emails if e not in used_emails], text
        )
        if email:
            contact['email'] = email
            contact['email_provenance'] = _envelope(
                email, SOURCE_PDFTOTEXT, pages=pages,
                confidence=email_conf, method=email_method,
            )
            used_emails.add(email)

        phone, phone_method, phone_conf = match_phone_by_proximity(
            name, contact['email'],
            [p for p in phones if p not in used_phones], text,
        )
        if phone:
            contact['phone'] = phone
            contact['phone_provenance'] = _envelope(
                phone, SOURCE_PDFTOTEXT, pages=pages,
                confidence=phone_conf, method=phone_method,
            )
            used_phones.add(phone.replace('-', '').replace('.', '').replace(' ', ''))

        title = extract_title_near_name(name, text)
        if title:
            contact['title'] = title
            contact['title_provenance'] = _envelope(
                title, SOURCE_PDFTOTEXT, pages=pages,
                confidence=CONFIDENCE_PROXIMITY, method="proximity",
            )

        contacts.append(contact)

    # If we have emails but no contacts, create contacts from emails
    # (synthesized / derivation path — confidence drops to CONFIDENCE_DERIVATION)
    if not contacts and emails:
        for idx, email in enumerate(emails):
            local = email.split('@')[0]
            if '_' in local or '.' in local:
                parts = re.split(r'[._]', local)
                name = ' '.join(p.capitalize() for p in parts if len(p) > 1)
            else:
                name = local.capitalize()

            positional = f"{section_id}_c{idx}"
            block_id = _make_block_id(section_id, positional, "contact", email)

            phone, phone_method, phone_conf = match_phone_by_proximity(
                name, email, phones, text
            )
            contact: Dict[str, Any] = {
                "block_id": block_id,
                "name": name,
                "email": email,
                "phone": phone,
                "title": "",
                "notes": "",
                "name_provenance": _envelope(
                    name, SOURCE_SYNTHESIZED, pages=pages,
                    confidence=CONFIDENCE_DERIVATION,
                    method="email_local_part",
                ),
                "email_provenance": _envelope(
                    email, SOURCE_PDFTOTEXT, pages=pages,
                    confidence=CONFIDENCE_NAME_PATTERN, method="entity_extraction",
                ),
            }
            if phone:
                contact['phone_provenance'] = _envelope(
                    phone, SOURCE_PDFTOTEXT, pages=pages,
                    confidence=phone_conf, method=phone_method,
                )
            contacts.append(contact)

    return contacts


def synthesize_systems_table(
    tables: List[Dict],
    pdftotext: List[str],
    *,
    section_id: str = "s",
    section_pages: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Synthesize systems table from pdfplumber structure + pdftotext content.

    Each returned row carries a ``block_id`` and a ``provenance`` envelope
    indicating which source produced the row (``pdfplumber`` for structured
    extraction, ``pdftotext`` for the label-search fallback).
    """
    text = '\n'.join(pdftotext)
    rows: List[Dict[str, Any]] = []
    section_pages = list(section_pages or [])

    system_labels = [
        'Campus Email', 'LTIs', 'Media Server', 'Virtual Classroom',
        'Campus Software', 'Campus D2L Resources', 'Campus D2L resources + training'
    ]

    # Try to find 3-column table in pdfplumber data
    for table in tables:
        headers = table.get('headers', [])
        table_rows = table.get('rows', [])
        table_page = table.get('page')
        table_pages = [table_page] if isinstance(table_page, int) and table_page > 0 else section_pages

        if len(headers) >= 3:
            header_text = ' '.join(str(h).lower() for h in headers)
            if 'student' in header_text and 'faculty' in header_text:
                for ridx, row in enumerate(table_rows):
                    if len(row) >= 3:
                        label = str(row[0]).replace('\n', ' ').strip() if row[0] else ''
                        student = str(row[1]).replace('\n', ' ').strip() if row[1] else ''
                        faculty = str(row[2]).replace('\n', ' ').strip() if row[2] else ''
                        if label:
                            positional = f"{section_id}_r{len(rows)}"
                            block_id = _make_block_id(
                                section_id, positional, "systems", label, student, faculty,
                            )
                            rows.append({
                                'block_id': block_id,
                                'label': label,
                                'student': student,
                                'faculty': faculty,
                                'provenance': _envelope(
                                    {"label": label, "student": student, "faculty": faculty},
                                    SOURCE_PDFPLUMBER,
                                    pages=table_pages,
                                    confidence=CONFIDENCE_DIRECT_TABLE,
                                    method="structured_table",
                                ),
                            })

    # Fallback: extract from text
    if not rows:
        for label in system_labels:
            label_pos = text.lower().find(label.lower())
            if label_pos >= 0:
                context = text[label_pos + len(label):label_pos + len(label) + 500]
                url_match = re.search(r'(https?://[^\s]+)', context)
                content = url_match.group(1) if url_match else context[:100].split('\n')[0].strip()
                positional = f"{section_id}_r{len(rows)}"
                block_id = _make_block_id(
                    section_id, positional, "systems", label, content,
                )
                rows.append({
                    'block_id': block_id,
                    'label': label,
                    'student': content,
                    'faculty': content,
                    'provenance': _envelope(
                        {"label": label, "student": content, "faculty": content},
                        SOURCE_PDFTOTEXT,
                        pages=section_pages,
                        confidence=CONFIDENCE_PROXIMITY,
                        method="label_search",
                    ),
                })

    return rows


def synthesize_roster(
    tables: List[Dict],
    pdftotext: List[str],
    *,
    section_id: str = "s",
    section_pages: Optional[List[int]] = None,
) -> List[Tuple[str, str]]:
    """
    Synthesize roster/course info from pdfplumber row labels + pdftotext content.

    Returns ``List[Tuple[label, value]]`` for back-compat with the
    ``render_kv_table`` consumer. Per-pair provenance is also accumulated
    on the module-level ``_LAST_ROSTER_PROVENANCE`` cache, which
    ``auto_synthesize_section`` reads so downstream artifacts see the
    envelope shape. Kept as a side-channel to preserve the existing
    tuple-based render path.
    """
    text = '\n'.join(pdftotext)
    pairs: List[Tuple[str, str]] = []
    envelopes: List[Dict[str, Any]] = []
    section_pages = list(section_pages or [])

    # Get labels from tables (with source tracking)
    table_labels: List[str] = []
    for table in tables:
        for h in table.get('headers', []):
            label = str(h).replace('\n', ' ').strip()
            if label and len(label) > 3:
                table_labels.append(label)
        for row in table.get('rows', []):
            if row and row[0]:
                label = str(row[0]).replace('\n', ' ').strip()
                if label and len(label) > 3:
                    table_labels.append(label)

    # Extract key-value pairs from pdftotext (primary source for value)
    kv_pairs, _ = extract_kv_pairs(pdftotext)

    # Merge
    seen_labels = set()
    for label, value in kv_pairs:
        pairs.append((label, value))
        seen_labels.add(label.lower())
        positional = f"{section_id}_p{len(envelopes)}"
        block_id = _make_block_id(section_id, positional, "roster", label, value)
        # Label from pdftotext parsing; value also pdftotext -> high confidence.
        envelopes.append({
            "block_id": block_id,
            "label": label,
            "value": value,
            "provenance": _envelope(
                {"label": label, "value": value},
                SOURCE_PDFTOTEXT,
                pages=section_pages,
                confidence=CONFIDENCE_NAME_PATTERN,
                method="kv_parse",
            ),
        })

    # Add table labels not in kv_pairs — these are pdfplumber-sourced labels
    # with pdftotext-sourced values, so we call it multi-source (synthesized).
    for label in table_labels:
        if label.lower() not in seen_labels:
            label_pos = text.lower().find(label.lower())
            if label_pos >= 0:
                context = text[label_pos + len(label):label_pos + len(label) + 300]
                value = context.split('\n')[0].strip()
                value = re.sub(r'^[:\s]+', '', value)
                if value and len(value) > 3:
                    pairs.append((label, value))
                    positional = f"{section_id}_p{len(envelopes)}"
                    block_id = _make_block_id(
                        section_id, positional, "roster", label, value,
                    )
                    envelopes.append({
                        "block_id": block_id,
                        "label": label,
                        "value": value,
                        "provenance": _envelope(
                            {"label": label, "value": value},
                            SOURCE_SYNTHESIZED,
                            pages=section_pages,
                            confidence=CONFIDENCE_PROXIMITY,
                            method="label_search",
                        ),
                    })

    # Publish per-pair envelopes via the module-level cache so
    # auto_synthesize_section can attach them to the section record.
    _LAST_ROSTER_PROVENANCE[section_id] = envelopes
    return pairs


# Side-channel cache: section_id -> per-pair envelope list. Populated by
# synthesize_roster, consumed by auto_synthesize_section. Keyed so concurrent
# sections don't stomp each other within a single synthesize run.
_LAST_ROSTER_PROVENANCE: Dict[str, List[Dict[str, Any]]] = {}


def _page_range_list(page_range: Any) -> List[int]:
    """Coerce an ``estimate_page_range`` tuple into a list of page ints.

    Returns the integer span ``[start, start+1, ..., end]``. Always emits a
    list (never a bare tuple) so the JSON output matches the
    ``SourceReference.pages`` shape.
    """
    if not page_range:
        return []
    try:
        start, end = int(page_range[0]), int(page_range[1])
    except (TypeError, ValueError, IndexError):
        return []
    if start <= 0 or end < start:
        return []
    return list(range(start, end + 1))


def _aggregate_section_confidence(envelopes: List[Dict[str, Any]]) -> float:
    """Average per-block envelope confidences; return 1.0 when empty."""
    confidences = [
        float(e.get("confidence", 0.0))
        for e in envelopes
        if isinstance(e, dict) and "confidence" in e
    ]
    if not confidences:
        return 1.0
    return round(sum(confidences) / len(confidences), 3)


def _envelopes_from_contacts(contacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-contact field provenance envelopes into a single list."""
    out: List[Dict[str, Any]] = []
    for c in contacts:
        for key in ("name_provenance", "email_provenance",
                    "phone_provenance", "title_provenance"):
            env = c.get(key)
            if isinstance(env, dict):
                out.append(env)
    return out


def auto_synthesize_section(ctx: Dict, section_index: int = 0) -> Dict:
    """
    Auto-synthesize a section using multi-source patterns.

    Emits the Wave 8 per-section record shape::

        {
          "section_id": "s3",
          "section_type": "...",
          "section_title": "...",
          "page_range": [3, 4],
          "provenance": {
            "sources": ["pdftotext", "pdfplumber"],
            "strategy": "pdfplumber_headers+pdftotext_entities",
            "confidence": 0.87
          },
          "data": { ... },
          "sources_used": { ... }   # legacy back-compat field
        }

    ``section_index`` is the zero-based index within the document; it seeds
    the canonical ``section_id`` (``"s{index}"``) that every block ID
    derives from. ``auto_synthesize_section`` is additive: it preserves the
    legacy ``sources_used`` dict so older consumers keep working.
    """
    stype = ctx['section_type']
    pdftotext = ctx['sources']['pdftotext']
    tables = ctx['sources']['tables']
    entities = ctx['entities']
    page_range = ctx.get("page_range")
    pages_list = _page_range_list(page_range)
    section_id = f"s{section_index}"

    sources_present: List[str] = []
    if pdftotext:
        sources_present.append(SOURCE_PDFTOTEXT)
    if tables:
        sources_present.append(SOURCE_PDFPLUMBER)
    if ctx['sources'].get('ocr'):
        sources_present.append(SOURCE_OCR)

    def _record(
        data: Dict[str, Any],
        *,
        strategy: str,
        sources: List[str],
        confidence: float,
        legacy_sources_used: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "section_id": section_id,
            "section_type": stype,
            "section_title": ctx["section_title"],
            "page_range": list(page_range) if isinstance(page_range, (list, tuple)) else [],
            "provenance": {
                "sources": sources,
                "strategy": strategy,
                "confidence": confidence,
            },
            "data": data,
            # Back-compat: the old sources_used field is populated from
            # provenance.strategy so legacy consumers keep working.
            "sources_used": legacy_sources_used,
        }

    if stype == 'campus-info':
        pairs, _ = extract_kv_pairs(pdftotext)
        pair_envelopes: List[Dict[str, Any]] = []
        for pidx, (label, value) in enumerate(pairs):
            positional = f"{section_id}_p{pidx}"
            block_id = _make_block_id(section_id, positional, "campus-info", label, value)
            pair_envelopes.append({
                "block_id": block_id,
                "label": label,
                "value": value,
                "provenance": _envelope(
                    {"label": label, "value": value},
                    SOURCE_PDFTOTEXT,
                    pages=pages_list,
                    confidence=CONFIDENCE_NAME_PATTERN,
                    method="kv_parse",
                ),
            })
        data = {"pairs": pairs, "pair_provenance": pair_envelopes}
        return _record(
            data,
            strategy="pdftotext_kv_parsing",
            sources=[SOURCE_PDFTOTEXT],
            confidence=_aggregate_section_confidence(
                [e["provenance"] for e in pair_envelopes]
            ),
            legacy_sources_used={
                "structure": "pdftotext key-value parsing",
                "urls": "pdftotext",
            },
        )

    elif stype == 'credentials':
        pairs, _ = extract_kv_pairs(pdftotext)
        pair_envelopes = []
        for pidx, (label, value) in enumerate(pairs):
            positional = f"{section_id}_p{pidx}"
            block_id = _make_block_id(section_id, positional, "credentials", label, value)
            pair_envelopes.append({
                "block_id": block_id,
                "label": label,
                "value": value,
                "provenance": _envelope(
                    {"label": label, "value": value},
                    SOURCE_PDFTOTEXT,
                    pages=pages_list,
                    confidence=CONFIDENCE_NAME_PATTERN,
                    method="kv_parse",
                ),
            })
        data = {"pairs": pairs, "pair_provenance": pair_envelopes}
        return _record(
            data,
            strategy="pdftotext_kv_parsing",
            sources=[SOURCE_PDFTOTEXT],
            confidence=_aggregate_section_confidence(
                [e["provenance"] for e in pair_envelopes]
            ),
            legacy_sources_used={"structure": "pdftotext key-value parsing"},
        )

    elif stype == 'contacts':
        contacts = synthesize_contacts(
            tables, entities, pdftotext,
            section_id=section_id, section_pages=pages_list,
        )
        contact_sources = set()
        for c in contacts:
            for key in ("name_provenance", "email_provenance",
                        "phone_provenance", "title_provenance"):
                env = c.get(key)
                if isinstance(env, dict):
                    contact_sources.add(env.get("source", SOURCE_PDFTOTEXT))
        sources = sorted(contact_sources) or [SOURCE_PDFTOTEXT]
        return _record(
            {"contacts": contacts},
            strategy="pdfplumber_headers+pdftotext_entities",
            sources=sources,
            confidence=_aggregate_section_confidence(_envelopes_from_contacts(contacts)),
            legacy_sources_used={
                "structure": "pdfplumber table headers",
                "content": "pdftotext entity matching",
            },
        )

    elif stype == 'systems':
        rows = synthesize_systems_table(
            tables, pdftotext,
            section_id=section_id, section_pages=pages_list,
        )
        row_envelopes = [r.get("provenance") for r in rows if isinstance(r.get("provenance"), dict)]
        row_sources = sorted({e.get("source") for e in row_envelopes if e.get("source")})
        return _record(
            {"headers": ['', 'Students', 'Faculty'], "rows": rows},
            strategy="pdfplumber_3col_table+pdftotext_fills",
            sources=row_sources or [SOURCE_PDFPLUMBER],
            confidence=_aggregate_section_confidence(row_envelopes),
            legacy_sources_used={
                "structure": "pdfplumber 3-column table",
                "content": "pdftotext fills gaps",
            },
        )

    elif stype == 'roster':
        pairs = synthesize_roster(
            tables, pdftotext,
            section_id=section_id, section_pages=pages_list,
        )
        envelopes = _LAST_ROSTER_PROVENANCE.pop(section_id, [])
        env_sources = sorted({
            e["provenance"].get("source")
            for e in envelopes
            if isinstance(e.get("provenance"), dict)
        }) or [SOURCE_PDFTOTEXT]
        return _record(
            {"pairs": pairs, "pair_provenance": envelopes},
            strategy="pdfplumber_row_labels+pdftotext_descriptions",
            sources=env_sources,
            confidence=_aggregate_section_confidence(
                [e["provenance"] for e in envelopes if isinstance(e.get("provenance"), dict)]
            ),
            legacy_sources_used={
                "structure": "pdfplumber row labels",
                "content": "pdftotext descriptions",
            },
        )

    else:
        # Prose sections (no-account, guest, overview)
        paragraph_envelopes: List[Dict[str, Any]] = []
        for pidx, para in enumerate(pdftotext):
            if not para:
                continue
            positional = f"{section_id}_p{pidx}"
            block_id = _make_block_id(section_id, positional, "prose", para)
            paragraph_envelopes.append({
                "block_id": block_id,
                "text": para,
                "provenance": _envelope(
                    para,
                    SOURCE_PDFTOTEXT,
                    pages=pages_list,
                    confidence=CONFIDENCE_NAME_PATTERN,
                    method="prose_extract",
                ),
            })
        data = {"paragraphs": pdftotext, "paragraph_provenance": paragraph_envelopes}
        return _record(
            data,
            strategy="pdftotext_prose",
            sources=[SOURCE_PDFTOTEXT],
            confidence=_aggregate_section_confidence(
                [e["provenance"] for e in paragraph_envelopes]
            ),
            legacy_sources_used={"content": "pdftotext prose"},
        )


# =============================================================================
# HTML RENDERING
# =============================================================================

def _format_pages_attr(pages: Any) -> str:
    """Format a list of page ints as a ``data-dart-pages`` attribute value.

    Returns a single page as ``"3"``, a contiguous range as ``"3-5"``, and
    a non-contiguous list as ``"3,5,7"``. Returns the empty string when
    ``pages`` is empty or contains no positive ints — callers should omit
    the attribute entirely in that case (avoid emitting lies).
    """
    if not pages:
        return ""
    try:
        ints = sorted({int(p) for p in pages if int(p) > 0})
    except (TypeError, ValueError):
        return ""
    if not ints:
        return ""
    if len(ints) == 1:
        return str(ints[0])
    # Check for contiguous range
    if ints == list(range(ints[0], ints[-1] + 1)):
        return f"{ints[0]}-{ints[-1]}"
    return ",".join(str(i) for i in ints)


def _build_dart_attrs(
    block_id: Optional[str] = None,
    source: Optional[str] = None,
    sources: Optional[List[str]] = None,
    pages: Optional[List[int]] = None,
    confidence: Optional[float] = None,
    strategy: Optional[str] = None,
) -> str:
    """Build a string of ``data-dart-*`` attributes (with leading space).

    Empty / None values are skipped — in particular, ``data-dart-pages`` is
    omitted when the page list is empty (we do not emit lies), and
    ``data-dart-confidence`` is omitted when ``1.0`` (the implicit default
    for directly-extracted blocks).
    """
    attrs: List[str] = []
    if block_id:
        attrs.append(f'data-dart-block-id="{html.escape(block_id, quote=True)}"')
    if source:
        attrs.append(f'data-dart-source="{html.escape(source, quote=True)}"')
    if sources:
        joined = ",".join(sources)
        attrs.append(f'data-dart-sources="{html.escape(joined, quote=True)}"')
    pages_attr = _format_pages_attr(pages) if pages else ""
    if pages_attr:
        attrs.append(f'data-dart-pages="{pages_attr}"')
    if confidence is not None and confidence < 1.0 and confidence >= 0.0:
        attrs.append(f'data-dart-confidence="{confidence:.2f}"')
    if strategy:
        attrs.append(f'data-dart-strategy="{html.escape(strategy, quote=True)}"')
    if not attrs:
        return ""
    return " " + " ".join(attrs)


def render_contact_cards(contacts: List[Dict]) -> str:
    """Render contacts as semantic contact cards.

    When contacts carry Wave 8 provenance envelopes (``block_id``,
    ``*_provenance``), the contact-card wrapper emits ``data-dart-*``
    attributes so downstream Trainforge / Courseforge can attribute
    content back to its source. Plain-dict contacts without provenance
    continue to render cleanly (attributes are omitted).
    """
    if not contacts:
        return ""

    cards = []
    for contact in contacts:
        name = contact.get('name', '')
        if not name:
            continue

        # Collect provenance from the primary name envelope (the "identity"
        # of the contact) and union sources/pages from other fields.
        name_env = contact.get("name_provenance") or {}
        block_id = contact.get("block_id") or ""
        source = name_env.get("source")
        pages = list(name_env.get("pages", []) or [])
        confidence = name_env.get("confidence")

        # Union sources across all field envelopes.
        all_sources: List[str] = []
        for key in ("name_provenance", "email_provenance",
                    "phone_provenance", "title_provenance"):
            env = contact.get(key)
            if isinstance(env, dict):
                src = env.get("source")
                if src and src not in all_sources:
                    all_sources.append(src)
                for p in env.get("pages", []) or []:
                    if p not in pages:
                        pages.append(p)
        sources_attr = all_sources if len(all_sources) > 1 else None

        attrs = _build_dart_attrs(
            block_id=block_id or None,
            source=source,
            sources=sources_attr,
            pages=pages,
            confidence=confidence,
        )

        parts = [f'<div class="contact-card dart-contact-card"{attrs}>']
        parts.append(f'  <h3>{html.escape(name)}</h3>')

        title = contact.get('title', '')
        if title:
            parts.append(f'  <p class="title">{html.escape(title)}</p>')

        dl_items = []
        phone = contact.get('phone', '')
        if phone:
            dl_items.append(f'    <dt>Phone</dt>\n    <dd><a href="tel:{phone}">{html.escape(phone)}</a></dd>')

        email = contact.get('email', '')
        if email:
            dl_items.append(f'    <dt>Email</dt>\n    <dd><a href="mailto:{email}">{html.escape(email)}</a></dd>')

        if dl_items:
            parts.append('  <dl>')
            parts.extend(dl_items)
            parts.append('  </dl>')

        notes = contact.get('notes', '')
        if notes:
            parts.append(f'  <p class="note">{html.escape(notes)}</p>')

        parts.append('</div>')
        cards.append('\n'.join(parts))

    return '\n\n'.join(cards)


def render_kv_table(pairs: List[Tuple[str, str]], caption: str = "") -> str:
    """Render key-value pairs as accessible table."""
    if not pairs:
        return ""

    rows = []
    for key, val in pairs:
        if not key:
            continue
        val_html = linkify(html.escape(val)) if val else '-'
        rows.append(f'    <tr>\n      <th scope="row">{html.escape(key)}</th>\n      <td>{val_html}</td>\n    </tr>')

    cap = f'    <caption>{html.escape(caption)}</caption>\n' if caption else ''
    return f'<table class="info-table dart-table dart-table--info">\n{cap}  <tbody>\n' + '\n'.join(rows) + '\n  </tbody>\n</table>'


def _render_systems_table(rows: List[Dict], headers: List[str]) -> str:
    """Render 3-column systems table."""
    parts = ['<table class="info-table dart-table dart-table--systems">', '  <caption>Campus Systems</caption>']

    parts.append('  <thead><tr>')
    for h in headers:
        parts.append(f'    <th scope="col">{html.escape(h)}</th>')
    parts.append('  </tr></thead>')

    parts.append('  <tbody>')
    for row in rows:
        parts.append('    <tr>')
        label = row.get('label', '')
        student = row.get('student', '')
        faculty = row.get('faculty', '')
        parts.append(f'      <th scope="row">{linkify(html.escape(label))}</th>')
        parts.append(f'      <td>{linkify(html.escape(student))}</td>')
        parts.append(f'      <td>{linkify(html.escape(faculty))}</td>')
        parts.append('    </tr>')
    parts.append('  </tbody></table>')

    return '\n'.join(parts)


def render_from_synthesized(synthesized: Dict) -> str:
    """Render WCAG HTML from synthesized data."""
    stype = synthesized['section_type']
    data = synthesized.get('data', {})

    if stype == 'contacts':
        contacts = data.get('contacts', [])
        return render_contact_cards(contacts)

    elif stype == 'systems':
        rows = data.get('rows', [])
        headers = data.get('headers', ['', 'Students', 'Faculty'])
        return _render_systems_table(rows, headers)

    elif stype in ('campus-info', 'credentials', 'roster'):
        pairs = data.get('pairs', [])
        captions = {
            'campus-info': 'Campus Resources',
            'credentials': 'Login Credentials',
            'roster': 'Course & Roster Information'
        }
        return render_kv_table(pairs, captions.get(stype, ''))

    else:
        paragraphs = data.get('paragraphs', [])
        return '\n'.join(f'<p>{linkify(html.escape(p))}</p>' for p in paragraphs if p)


def generate_html_from_synthesized(synthesized: Dict) -> str:
    """Generate complete accessible HTML from synthesized data."""
    name = synthesized['campus_name']
    sections = synthesized['sections']

    # Build TOC
    toc_items = []
    for i, section in enumerate(sections):
        toc_items.append(f'    <li><a href="#s{i}">{html.escape(section["section_title"])}</a></li>')

    toc_html = ''
    if toc_items:
        toc_html = f'''<nav class="toc" aria-labelledby="toc-h">
  <h2 id="toc-h">Contents</h2>
  <ul>
{chr(10).join(toc_items)}
  </ul>
</nav>'''

    # Map section types to CSS modifier classes for semantic subclassing
    _SECTION_TYPE_CSS = {
        'campus-info': 'dart-section--campus-info',
        'credentials': 'dart-section--credentials',
        'contacts': 'dart-section--contacts',
        'systems': 'dart-section--systems',
        'roster': 'dart-section--roster',
        'no-account': 'dart-section--prose',
        'guest': 'dart-section--prose',
        'overview': 'dart-section--prose',
    }

    # Render sections (Wave 8 emits data-dart-* provenance on each <section>)
    sections_html = []
    for i, section in enumerate(sections):
        content = render_from_synthesized(section)
        stype = section.get('section_type', '')
        type_class = _SECTION_TYPE_CSS.get(stype, 'dart-section--prose')
        section_id = section.get('section_id', f"s{i}")

        prov = section.get('provenance', {}) if isinstance(section, dict) else {}
        prov_sources = prov.get('sources', []) if isinstance(prov, dict) else []
        prov_strategy = prov.get('strategy') if isinstance(prov, dict) else None
        prov_confidence = prov.get('confidence') if isinstance(prov, dict) else None
        section_pages = _page_range_list(section.get('page_range'))

        primary_source = prov_sources[0] if prov_sources else None
        sources_attr = prov_sources if len(prov_sources) > 1 else None

        section_attrs = _build_dart_attrs(
            block_id=section_id,
            source=primary_source,
            sources=sources_attr,
            pages=section_pages,
            confidence=prov_confidence,
            strategy=prov_strategy,
        )

        sections_html.append(f'''<section id="s{i}" class="dart-section {type_class}" aria-labelledby="s{i}-h"{section_attrs}>
  <h2 id="s{i}-h">{html.escape(section["section_title"])}</h2>
{content}
</section>''')

    css = '''
:root {
  --text: #1a1a1a; --muted: #666; --bg: #fff; --bg2: #f5f5f5;
  --accent: #0055aa; --accent2: #003366; --border: #ddd; --focus: #0066cc;
}
@media (prefers-color-scheme: dark) {
  :root { --text: #eee; --muted: #aaa; --bg: #1a1a1a; --bg2: #252525;
    --accent: #6db3f2; --accent2: #4a9de8; --border: #444; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.65; color: var(--text); background: var(--bg);
  max-width: 900px; margin: 0 auto; padding: 2rem;
}
h1 {
  font-size: 1.75rem; text-align: center;
  border-bottom: 2px solid var(--border);
  padding-bottom: 0.75rem; margin: 0 0 0.5rem;
}
h2 {
  font-size: 1.3rem; color: var(--accent2);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem; margin: 2rem 0 1rem;
  scroll-margin-top: 80px;
}
h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem; }
p { margin: 0.75rem 0; }
a { color: var(--accent); word-break: break-word; }
a:hover { text-decoration: none; }
:focus { outline: 3px solid var(--focus); outline-offset: 2px; }
:focus:not(:focus-visible) { outline: none; }
.skip {
  position: absolute; top: -50px; left: 0;
  background: var(--accent); color: #fff;
  padding: 0.5rem 1rem; z-index: 100; text-decoration: none;
}
.skip:focus { top: 0; }
nav.toc {
  background: var(--bg2); padding: 1rem 1.25rem;
  border-radius: 6px; margin: 1.5rem 0;
}
nav.toc h2 { margin: 0 0 0.5rem; border: none; font-size: 1rem; }
nav.toc ul { margin: 0; padding: 0 0 0 1.25rem; }
nav.toc li { margin: 0.3rem 0; }
nav.toc a { text-decoration: none; }
table.info-table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
table.info-table caption {
  font-weight: 600; text-align: left;
  padding: 0.5rem 0; color: var(--accent2);
}
table.info-table th, table.info-table td {
  border: 1px solid var(--border);
  padding: 0.6rem 0.75rem;
  text-align: left; vertical-align: top;
}
table.info-table th { background: var(--bg2); font-weight: 600; width: 28%; }
.contact-card {
  background: var(--bg2); padding: 1rem 1.25rem;
  border-radius: 6px; margin: 1rem 0;
  border-left: 4px solid var(--accent);
}
.contact-card h3 { margin: 0 0 0.25rem; }
.contact-card .title {
  font-style: italic; color: var(--muted);
  margin: 0 0 0.75rem; font-size: 0.95rem;
}
.contact-card dl {
  display: grid; grid-template-columns: auto 1fr;
  gap: 0.25rem 1rem; margin: 0.5rem 0;
}
.contact-card dt { font-weight: 600; color: var(--muted); }
.contact-card dd { margin: 0; }
.contact-card .note {
  font-size: 0.9rem; color: var(--muted);
  margin: 0.75rem 0 0; font-style: italic;
}
ul { margin: 0.75rem 0; padding-left: 1.5rem; }
li { margin: 0.3rem 0; }
footer {
  margin-top: 2rem; padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.85rem; color: var(--muted);
}
@media (max-width: 600px) {
  body { padding: 1rem; }
  table.info-table th { width: 35%; }
}
@media print { .skip, nav.toc { display: none; } }

/* DART Semantic Subclassing */
.dart-document { position: relative; }
.dart-title h1 { font-variant: small-caps; letter-spacing: 0.02em; }
.dart-section { margin: 1.5rem 0; }
.dart-section--campus-info { border-left: 3px solid var(--accent); padding-left: 1rem; }
.dart-section--credentials { background: var(--bg2); padding: 1rem 1.25rem; border-radius: 6px; }
.dart-section--contacts .dart-contact-card + .dart-contact-card { margin-top: 0.75rem; }
.dart-section--systems .dart-table--systems { font-size: 0.95rem; }
.dart-section--roster { border-left: 3px solid var(--accent2); padding-left: 1rem; }
.dart-section--prose > p:first-of-type { font-size: 1.05rem; }
.dart-table { border-collapse: collapse; width: 100%; }
.dart-table caption { font-weight: 600; text-align: left; color: var(--accent2); }
.dart-contact-card { position: relative; }
'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(name)} - Campus Information</title>
  <meta name="description" content="Campus information for {html.escape(name)}">
  <style>{css}</style>
</head>
<body>
<a href="#main" class="skip">Skip to content</a>
<main id="main" role="main">
  <article class="dart-document">
    <header class="dart-title">
      <h1>{html.escape(name)}</h1>
    </header>
{toc_html}
{chr(10).join(sections_html)}
    <footer role="contentinfo">
      <p>Accessible HTML - WCAG 2.2 AA compliant</p>
    </footer>
  </article>
</main>
</body>
</html>'''


# =============================================================================
# QUALITY REPORT
# =============================================================================

def build_quality_report(
    contexts: List[Dict], synthesized_sections: List[Dict],
    wcag_result: Optional[Dict[str, Any]] = None,
    content_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a DARTQualityReport from synthesis data and WCAG validation.

    This report travels with the HTML output so downstream consumers
    (Courseforge, Trainforge) can assess source reliability.

    Returns dict with:
        extraction_sources: list of sources that contributed
        table_quality: dict of section_type -> quality rating
        entity_counts: dict of entity type -> count
        wcag_validation: WCAG validation results
        confidence_score: 0.0-1.0 overall quality score
        section_count: number of sections synthesized
    """
    # Collect which extraction sources contributed
    sources_used = set()
    table_quality = {}
    total_entities = {"phones": 0, "emails": 0, "urls": 0, "names": 0}

    for ctx in contexts:
        src = ctx.get("sources", {})
        if src.get("pdftotext"):
            sources_used.add("pdftotext")
        if src.get("tables"):
            sources_used.add("pdfplumber")
            for t in src["tables"]:
                quality = t.get("quality", "unknown")
                key = f"{ctx.get('section_type', 'unknown')}_p{t.get('page', 0)}"
                table_quality[key] = quality
        if src.get("ocr"):
            sources_used.add("ocr")

        entities = ctx.get("entities", {})
        for etype in total_entities:
            total_entities[etype] += len(entities.get(etype, []))

    # Compute confidence score based on:
    # - Multi-source coverage (more sources = higher confidence)
    # - Table quality (complete > headers_only > empty)
    # - Entity extraction success
    # - WCAG compliance
    source_score = len(sources_used) / 3.0  # max 1.0 with all 3 sources

    table_scores = {"complete": 1.0, "headers_only": 0.5, "empty": 0.0}
    if table_quality:
        avg_table = sum(
            table_scores.get(q, 0.5) for q in table_quality.values()
        ) / len(table_quality)
    else:
        avg_table = 0.5  # no tables, neutral

    entity_score = min(1.0, sum(total_entities.values()) / 10.0)

    wcag_score = wcag_result.get("quality_score", 1.0) if wcag_result else 1.0

    confidence = (
        source_score * 0.3
        + avg_table * 0.2
        + entity_score * 0.2
        + wcag_score * 0.3
    )

    return {
        "extraction_sources": sorted(sources_used),
        "table_quality": table_quality,
        "entity_counts": total_entities,
        "wcag_validation": wcag_result,
        "content_profile": content_profile,
        "confidence_score": round(confidence, 3),
        "section_count": len(synthesized_sections),
    }


# =============================================================================
# SEMANTIC STRUCTURE EXTRACTION
# =============================================================================

def extract_content_profile(html_content: str, label: str = "") -> Optional[Dict[str, Any]]:
    """
    Run semantic structure extraction with content profiling.

    Returns a summary dict with concept_count, topic structure, and
    pedagogical pattern, or None if extractor isn't available.
    """
    if not SEMANTIC_EXTRACTOR_AVAILABLE:
        logger.debug("Semantic extractor not available, skipping profiling")
        return None

    try:
        extractor = SemanticStructureExtractor()
        result = extractor.extract_with_profiling(html_content, source_path=label)

        # Extract key signals for downstream consumers
        profiles = result.get("contentProfiles", {})
        concept_graph = result.get("conceptGraph", {})

        return {
            "pedagogical_pattern": result.get("pedagogicalPattern", "unknown"),
            "concept_count": len(concept_graph.get("concepts", [])),
            "concept_names": [
                c.get("name", "") for c in concept_graph.get("concepts", [])[:20]
            ],
            "chapter_count": len(result.get("chapters", [])),
            "content_profiles": profiles,
        }
    except Exception as e:
        logger.warning("Semantic extraction failed for %s: %s", label, e)
        return None


# =============================================================================
# WCAG VALIDATION
# =============================================================================

def validate_wcag(html_content: str, label: str = "") -> Dict[str, Any]:
    """
    Run WCAG 2.2 AA validation on generated HTML.

    Returns dict with:
        compliant: bool - whether the HTML passes WCAG AA
        critical_count: int - number of critical issues
        high_count: int - number of high severity issues
        total_issues: int - total issue count
        issues: list - detailed issue dicts
        quality_score: float - 0.0-1.0 quality score
    """
    if not WCAG_VALIDATOR_AVAILABLE:
        logger.debug("WCAG validator not available, skipping validation")
        return {
            "compliant": True,
            "critical_count": 0,
            "high_count": 0,
            "total_issues": 0,
            "issues": [],
            "quality_score": 1.0,
        }

    validator = AccessibilityValidator(strict_mode=False)

    # Write HTML to temp file for validation
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.html', encoding='utf-8', delete=False
    ) as tmp:
        tmp.write(html_content)
        tmp_path = Path(tmp.name)

    try:
        report = validator.validate_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Convert issues to serializable dicts
    issues = []
    for issue in report.issues:
        severity = (
            issue.severity.value
            if isinstance(issue.severity, IssueSeverity)
            else issue.severity
        )
        issues.append({
            "criterion": issue.criterion,
            "severity": severity,
            "element": issue.element,
            "message": issue.message,
            "suggestion": issue.suggestion,
        })

    # Compute quality score: 1.0 if no issues, penalized by severity
    # Critical: -0.15 each, High: -0.08, Medium: -0.03, Low: -0.01
    penalty = (
        report.critical_count * 0.15
        + report.high_count * 0.08
        + report.medium_count * 0.03
        + report.low_count * 0.01
    )
    quality_score = max(0.0, 1.0 - penalty)

    prefix = f"[{label}] " if label else ""
    if report.critical_count > 0:
        logger.warning(
            "%sWCAG validation: %d critical, %d high issues",
            prefix, report.critical_count, report.high_count,
        )
    elif report.total_issues > 0:
        logger.info(
            "%sWCAG validation: %d issues (no critical)",
            prefix, report.total_issues,
        )

    return {
        "compliant": report.wcag_aa_compliant,
        "critical_count": report.critical_count,
        "high_count": report.high_count,
        "total_issues": report.total_issues,
        "issues": issues,
        "quality_score": quality_score,
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_synthesize_all(combined_dir: str = None, output_dir: str = None) -> List[Path]:
    """
    Batch synthesize all campuses and generate HTML.
    Returns list of generated HTML file paths.
    """
    if combined_dir is None:
        combined_dir = Path(__file__).parent / "batch_output" / "combined"
    else:
        combined_dir = Path(combined_dir)

    if output_dir is None:
        output_dir = Path(__file__).parent / "batch_output"
    else:
        output_dir = Path(output_dir)

    synthesized_dir = output_dir / "synthesized"
    html_dir = output_dir / "html"

    synthesized_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(combined_dir.glob("*_combined.json"))
    html_files = []

    print(f"Batch synthesizing {len(files)} campuses...")

    for i, f in enumerate(files, 1):
        code = f.stem.replace('_combined', '')
        print(f"[{i}/{len(files)}] {code}...", end=" ")

        try:
            combined = json.loads(f.read_text(encoding='utf-8'))
            contexts = export_section_contexts(combined)

            synthesized = {
                'campus_code': code,
                'campus_name': CAMPUS_NAMES.get(code, code),
                'sections': [
                    auto_synthesize_section(ctx, section_index=i)
                    for i, ctx in enumerate(contexts)
                ]
            }

            synth_path = synthesized_dir / f"{code}_synthesized.json"
            synth_path.write_text(json.dumps(synthesized, indent=2), encoding='utf-8')

            html_out = generate_html_from_synthesized(synthesized)

            # Run WCAG validation
            wcag_result = validate_wcag(html_out, label=code)

            html_path = html_dir / f"{code}_synthesized.html"
            html_path.write_text(html_out, encoding='utf-8')
            html_files.append(html_path)

            status = "OK"
            if wcag_result['critical_count'] > 0:
                status = f"OK (WCAG: {wcag_result['critical_count']} critical)"
            print(status)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    return html_files


def create_zip(html_files: List[Path], output: str):
    """Zip all HTML files to output location."""
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in html_files:
            zf.write(f, f.name)

    print(f"Created {output} with {len(html_files)} files")


def convert_single_pdf(combined_json_path: str, output_path: str = None) -> Dict:
    """
    Convert a single PDF using multi-source synthesis.
    Expects a combined JSON file with pdftotext, tables, and OCR data.
    """
    combined_path = Path(combined_json_path)
    combined = json.loads(combined_path.read_text(encoding='utf-8'))

    code = combined.get('campus_code', combined_path.stem.replace('_combined', ''))
    name = CAMPUS_NAMES.get(code, code)

    contexts = export_section_contexts(combined)
    sections = [
        auto_synthesize_section(ctx, section_index=i)
        for i, ctx in enumerate(contexts)
    ]
    synthesized = {
        'campus_code': code,
        'campus_name': name,
        'sections': sections,
    }

    html_out = generate_html_from_synthesized(synthesized)

    # Run WCAG validation on generated HTML
    wcag_result = validate_wcag(html_out, label=code)

    # Run semantic structure extraction for content profiling
    content_profile = extract_content_profile(html_out, label=code)

    # Build quality report for downstream consumers
    quality_report = build_quality_report(
        contexts, sections, wcag_result, content_profile
    )

    if output_path:
        Path(output_path).write_text(html_out, encoding='utf-8')
        # Write quality report alongside HTML
        report_path = Path(output_path).with_suffix('.quality.json')
        report_path.write_text(
            json.dumps(quality_report, indent=2), encoding='utf-8'
        )

    return {
        'success': True,
        'campus_code': code,
        'campus_name': name,
        'html': html_out,
        'synthesized': synthesized,
        'quality_report': quality_report,
        'quality_score': quality_report['confidence_score'],
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DART Multi-Source Synthesizer')
    parser.add_argument('--batch', action='store_true', help='Batch process all campuses')
    parser.add_argument('--zip', type=str, help='Output zip path for batch mode')
    parser.add_argument('--input', type=str, help='Input combined JSON for single conversion')
    parser.add_argument('--output', type=str, help='Output HTML path for single conversion')

    args = parser.parse_args()

    if args.batch:
        html_files = batch_synthesize_all()
        if args.zip:
            create_zip(html_files, args.zip)
        print(f"Processed {len(html_files)} campuses")
    elif args.input:
        result = convert_single_pdf(args.input, args.output)
        print(f"Converted {result['campus_name']}")
        if not args.output:
            print(result['html'])
    else:
        parser.print_help()
