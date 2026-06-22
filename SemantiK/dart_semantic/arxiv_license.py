"""arXiv license filter — gate training data by commercial-training suitability.

Every arXiv paper carries a license URL in its metadata. This module classifies
each URL as commercially usable (may include in training set) or not, and
provides a bulk lookup path via the arXiv API.

Usage:
    from dart_semantic.arxiv_license import is_commercial_ok, fetch_license

    ok, license_url = fetch_license("2005.00129")
    if ok:
        include_in_training(...)
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"

# License URLs that permit commercial redistribution and derivative works.
# Checked as case-insensitive prefixes to tolerate trailing slashes and versions.
COMMERCIAL_OK_PREFIXES = (
    "http://creativecommons.org/licenses/by/",          # CC-BY (any version)
    "https://creativecommons.org/licenses/by/",
    "http://creativecommons.org/licenses/by-sa/",       # CC-BY-SA (share-alike)
    "https://creativecommons.org/licenses/by-sa/",
    "http://creativecommons.org/publicdomain/zero/",    # CC0
    "https://creativecommons.org/publicdomain/zero/",
    "http://creativecommons.org/publicdomain/mark/",    # Public domain mark
    "https://creativecommons.org/publicdomain/mark/",
    "https://opendatacommons.org/licenses/by/",         # ODC-By (olmOCR-mix licence)
)

# Explicitly rejected — these ARE recognized but are NOT usable for us.
REJECT_PREFIXES = (
    "http://creativecommons.org/licenses/by-nc",        # Non-commercial
    "https://creativecommons.org/licenses/by-nc",
    "http://creativecommons.org/licenses/by-nd",        # No derivatives
    "https://creativecommons.org/licenses/by-nd",
    "http://arxiv.org/licenses/nonexclusive-distrib",   # arXiv default
    "https://arxiv.org/licenses/nonexclusive-distrib",
)


@dataclass(frozen=True)
class LicenseCheck:
    arxiv_id: str
    license_url: str | None
    commercial_ok: bool
    reason: str


def is_commercial_ok(license_url: str | None) -> tuple[bool, str]:
    """Classify a license URL. Returns (ok, human-readable reason)."""
    if not license_url:
        return False, "no license declared (author copyright retained)"
    url = license_url.strip().lower()
    for prefix in COMMERCIAL_OK_PREFIXES:
        if url.startswith(prefix.lower()):
            return True, f"allowed: {prefix.rstrip('/')}"
    for prefix in REJECT_PREFIXES:
        if url.startswith(prefix.lower()):
            return False, f"rejected: {prefix.rstrip('/')}"
    return False, f"unrecognized license, treating as not-OK: {license_url}"


def fetch_license(arxiv_id: str, *, timeout: int = 15) -> LicenseCheck:
    """Look up one paper's license via the arXiv Atom API.

    Network call — prefer batched use via fetch_licenses for >1 paper.
    """
    results = fetch_licenses([arxiv_id], timeout=timeout)
    return results[0] if results else LicenseCheck(
        arxiv_id=arxiv_id, license_url=None, commercial_ok=False,
        reason="arXiv API returned no entry"
    )


def fetch_licenses(arxiv_ids: list[str], *, timeout: int = 30,
                   batch_size: int = 100) -> list[LicenseCheck]:
    """Bulk license lookup. arXiv API accepts up to 100 id_list entries per query."""
    out: list[LicenseCheck] = []
    for i in range(0, len(arxiv_ids), batch_size):
        chunk = arxiv_ids[i:i + batch_size]
        qs = urlencode({
            "id_list": ",".join(chunk),
            "max_results": len(chunk),
        })
        url = f"{ARXIV_API}?{qs}"
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"arXiv API failed for batch: {exc}")
            for aid in chunk:
                out.append(LicenseCheck(aid, None, False, f"api error: {exc}"))
            continue
        out.extend(_parse_atom(r.text, chunk))
    return out


# Atom namespaces used by arXiv
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _parse_atom(xml_text: str, expected_ids: list[str]) -> list[LicenseCheck]:
    """Parse arXiv Atom response into one LicenseCheck per expected ID."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return [LicenseCheck(aid, None, False, f"parse error: {exc}")
                for aid in expected_ids]

    # Map returned entries by bare arxiv id.
    entries: dict[str, str | None] = {}
    for entry in root.findall("atom:entry", _NS):
        id_el = entry.find("atom:id", _NS)
        if id_el is None or not id_el.text:
            continue
        # Atom id looks like "http://arxiv.org/abs/2005.00129v2"
        bare = id_el.text.rsplit("/", 1)[-1].split("v")[0]
        license_el = entry.find("arxiv:license", _NS)
        entries[bare] = license_el.text.strip() if (license_el is not None and license_el.text) else None

    results: list[LicenseCheck] = []
    for aid in expected_ids:
        bare = aid.split("v")[0]
        if bare not in entries:
            results.append(LicenseCheck(
                aid, None, False, "arXiv API returned no entry"
            ))
            continue
        license_url = entries[bare]
        ok, reason = is_commercial_ok(license_url)
        results.append(LicenseCheck(aid, license_url, ok, reason))
    return results


# ---------- arxiv-id extraction from local filenames ----------

import re

_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v\d+)?")
_OLD_ID_RE = re.compile(r"(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?")


def extract_arxiv_id(filename: str) -> str | None:
    """Pull an arXiv ID out of a filename like '2005.00129v2_Title.pdf'."""
    m = _ID_RE.search(filename)
    if m:
        return m.group("id")
    m = _OLD_ID_RE.search(filename)
    if m:
        return m.group("id")
    return None
