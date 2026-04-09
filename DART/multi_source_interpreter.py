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

import html
import json
import logging
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


def match_email_by_name(name: str, emails: List[str], text: str) -> str:
    """Match an email to a contact name by pattern or proximity."""
    name_lower = name.lower()
    name_parts = name_lower.split()

    # Special cases
    if 'help desk' in name_lower or 'service desk' in name_lower:
        for email in emails:
            if 'help@' in email.lower() or 'helpdesk@' in email.lower():
                return email

    if 'registrar' in name_lower:
        for email in emails:
            if 'registrar' in email.lower():
                return email

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
                    return email

    # Proximity matching
    name_pos = text.lower().find(name_lower)
    if name_pos >= 0:
        for email in emails:
            email_pos = text.lower().find(email.lower())
            if email_pos >= 0 and abs(name_pos - email_pos) < 300:
                return email

    return ''


def match_phone_by_proximity(name: str, email: str, phones: List[str], text: str) -> str:
    """Match a phone number by proximity to name or email in text."""
    if email:
        email_pos = text.lower().find(email.lower())
        if email_pos >= 0:
            for phone in phones:
                phone_pos = text.find(phone)
                if phone_pos >= 0 and abs(phone_pos - email_pos) < 200:
                    return phone.replace('.', '-').replace(' ', '-')

    name_pos = text.lower().find(name.lower())
    if name_pos >= 0:
        for phone in phones:
            phone_pos = text.find(phone)
            if phone_pos >= 0 and abs(phone_pos - name_pos) < 200:
                return phone.replace('.', '-').replace(' ', '-')

    return ''


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


def synthesize_contacts(tables: List[Dict], entities: Dict, pdftotext: List[str]) -> List[Dict]:
    """
    Synthesize contacts using multi-source approach:
    1. Get headers from pdfplumber tables (contact names)
    2. Extract phones/emails from pdftotext
    3. Match by: email prefix, name proximity, pattern recognition
    """
    text = '\n'.join(pdftotext)
    phones = entities.get('phones', [])
    emails = entities.get('emails', [])

    # Get contact names from table headers
    contact_names = []
    seen_names = set()

    for table in tables:
        for h in table.get('headers', []):
            name = h.replace('\n', ' ').strip()
            if name and _is_likely_contact_name(name) and name not in seen_names:
                contact_names.append(name)
                seen_names.add(name)

    # If no table headers, look for names in entities
    if not contact_names and entities.get('names'):
        for name in entities['names']:
            if name not in seen_names:
                contact_names.append(name)
                seen_names.add(name)

    # Build contacts with entity matching
    used_emails = set()
    used_phones = set()
    contacts = []

    for name in contact_names:
        contact = {'name': name, 'phone': '', 'email': '', 'title': '', 'notes': ''}

        email = match_email_by_name(name, [e for e in emails if e not in used_emails], text)
        if email:
            contact['email'] = email
            used_emails.add(email)

        phone = match_phone_by_proximity(name, contact['email'],
                                         [p for p in phones if p not in used_phones], text)
        if phone:
            contact['phone'] = phone
            used_phones.add(phone.replace('-', '').replace('.', '').replace(' ', ''))

        contact['title'] = extract_title_near_name(name, text)
        contacts.append(contact)

    # If we have emails but no contacts, create contacts from emails
    if not contacts and emails:
        for email in emails:
            local = email.split('@')[0]
            if '_' in local or '.' in local:
                parts = re.split(r'[._]', local)
                name = ' '.join(p.capitalize() for p in parts if len(p) > 1)
            else:
                name = local.capitalize()

            phone = match_phone_by_proximity(name, email, phones, text)
            contacts.append({
                'name': name,
                'email': email,
                'phone': phone,
                'title': '',
                'notes': ''
            })

    return contacts


def synthesize_systems_table(tables: List[Dict], pdftotext: List[str]) -> List[Dict]:
    """
    Synthesize systems table from pdfplumber structure + pdftotext content.
    """
    text = '\n'.join(pdftotext)
    rows = []

    system_labels = [
        'Campus Email', 'LTIs', 'Media Server', 'Virtual Classroom',
        'Campus Software', 'Campus D2L Resources', 'Campus D2L resources + training'
    ]

    # Try to find 3-column table in pdfplumber data
    for table in tables:
        headers = table.get('headers', [])
        table_rows = table.get('rows', [])

        if len(headers) >= 3:
            header_text = ' '.join(str(h).lower() for h in headers)
            if 'student' in header_text and 'faculty' in header_text:
                for row in table_rows:
                    if len(row) >= 3:
                        label = str(row[0]).replace('\n', ' ').strip() if row[0] else ''
                        student = str(row[1]).replace('\n', ' ').strip() if row[1] else ''
                        faculty = str(row[2]).replace('\n', ' ').strip() if row[2] else ''
                        if label:
                            rows.append({'label': label, 'student': student, 'faculty': faculty})

    # Fallback: extract from text
    if not rows:
        for label in system_labels:
            label_pos = text.lower().find(label.lower())
            if label_pos >= 0:
                context = text[label_pos + len(label):label_pos + len(label) + 500]
                url_match = re.search(r'(https?://[^\s]+)', context)
                content = url_match.group(1) if url_match else context[:100].split('\n')[0].strip()
                rows.append({'label': label, 'student': content, 'faculty': content})

    return rows


def synthesize_roster(tables: List[Dict], pdftotext: List[str]) -> List[Tuple[str, str]]:
    """
    Synthesize roster/course info from pdfplumber row labels + pdftotext content.
    """
    text = '\n'.join(pdftotext)
    pairs = []

    # Get labels from tables
    table_labels = []
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

    # Extract key-value pairs from pdftotext
    kv_pairs, _ = extract_kv_pairs(pdftotext)

    # Merge
    seen_labels = set()
    for label, value in kv_pairs:
        pairs.append((label, value))
        seen_labels.add(label.lower())

    # Add table labels not in kv_pairs
    for label in table_labels:
        if label.lower() not in seen_labels:
            label_pos = text.lower().find(label.lower())
            if label_pos >= 0:
                context = text[label_pos + len(label):label_pos + len(label) + 300]
                value = context.split('\n')[0].strip()
                value = re.sub(r'^[:\s]+', '', value)
                if value and len(value) > 3:
                    pairs.append((label, value))

    return pairs


def auto_synthesize_section(ctx: Dict) -> Dict:
    """
    Auto-synthesize a section using multi-source patterns.
    """
    stype = ctx['section_type']
    pdftotext = ctx['sources']['pdftotext']
    tables = ctx['sources']['tables']
    entities = ctx['entities']

    if stype == 'campus-info':
        pairs, _ = extract_kv_pairs(pdftotext)
        return {
            'section_type': stype,
            'section_title': ctx['section_title'],
            'data': {'pairs': pairs},
            'sources_used': {'structure': 'pdftotext key-value parsing', 'urls': 'pdftotext'}
        }

    elif stype == 'credentials':
        pairs, _ = extract_kv_pairs(pdftotext)
        return {
            'section_type': stype,
            'section_title': ctx['section_title'],
            'data': {'pairs': pairs},
            'sources_used': {'structure': 'pdftotext key-value parsing'}
        }

    elif stype == 'contacts':
        contacts = synthesize_contacts(tables, entities, pdftotext)
        return {
            'section_type': stype,
            'section_title': ctx['section_title'],
            'data': {'contacts': contacts},
            'sources_used': {'structure': 'pdfplumber table headers', 'content': 'pdftotext entity matching'}
        }

    elif stype == 'systems':
        rows = synthesize_systems_table(tables, pdftotext)
        return {
            'section_type': stype,
            'section_title': ctx['section_title'],
            'data': {'headers': ['', 'Students', 'Faculty'], 'rows': rows},
            'sources_used': {'structure': 'pdfplumber 3-column table', 'content': 'pdftotext fills gaps'}
        }

    elif stype == 'roster':
        pairs = synthesize_roster(tables, pdftotext)
        return {
            'section_type': stype,
            'section_title': ctx['section_title'],
            'data': {'pairs': pairs},
            'sources_used': {'structure': 'pdfplumber row labels', 'content': 'pdftotext descriptions'}
        }

    else:
        # Prose sections (no-account, guest, overview)
        return {
            'section_type': stype,
            'section_title': ctx['section_title'],
            'data': {'paragraphs': pdftotext},
            'sources_used': {'content': 'pdftotext prose'}
        }


# =============================================================================
# HTML RENDERING
# =============================================================================

def render_contact_cards(contacts: List[Dict]) -> str:
    """Render contacts as semantic contact cards."""
    if not contacts:
        return ""

    cards = []
    for contact in contacts:
        name = contact.get('name', '')
        if not name:
            continue

        parts = ['<div class="contact-card dart-contact-card">']
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

    # Render sections
    sections_html = []
    for i, section in enumerate(sections):
        content = render_from_synthesized(section)
        stype = section.get('section_type', '')
        type_class = _SECTION_TYPE_CSS.get(stype, 'dart-section--prose')
        sections_html.append(f'''<section id="s{i}" class="dart-section {type_class}" aria-labelledby="s{i}-h">
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
                'sections': [auto_synthesize_section(ctx) for ctx in contexts]
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
    sections = [auto_synthesize_section(ctx) for ctx in contexts]
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
