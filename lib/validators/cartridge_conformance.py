"""Track-L (L2) — CartridgeConformanceValidator.

A FULL-CARTRIDGE strict-conformance gate that validates a *built* ``.imscc``
package end-to-end against the vendored IMS Common Cartridge / QTI schema pack,
so a cartridge that would silently fail (or import wrong) into an LMS is caught
at the packaging boundary. This is the LMS-agnostic portability guarantee that
replaces per-LMS testing: SPEC conformance is the contract, not per-vendor
targets (a passing vendor import proves nothing about the next vendor).

Unlike ``lib/validators/qti_well_formed.py`` — which validates only the loose
``06_assessments/*.xml`` documents BEFORE packaging — this validator opens the
assembled ``.imscc`` ZIP and audits the whole thing:

(A) **Manifest well-formedness + CC-version detection.** ``imsmanifest.xml``
    must be present and parse. Its CC profile version is detected from the
    root-element namespace (``imsccv1p1`` / ``imsccv1p2`` / ``imsccv1p3`` →
    the ``imscp_v1p1`` profile namespace). An unrecognised manifest namespace
    emits ``CARTRIDGE_UNKNOWN_CC_VERSION`` (warning) — version-detection is
    best-effort and never blocks on its own.

(B) **Namespace-routed XSD conformance (the historical wrong-XSD failure
    class).** Every XML document in the cartridge — the manifest AND every
    resource document — is validated against the vendored XSD whose
    ``targetNamespace`` EQUALS the document's root-element namespace. The
    ns→XSD map is built by scanning the vendored schema dir
    (``Courseforge/schemas/imscc/`` + any nested vendor subdir) and reading
    each ``.xsd``'s ``@targetNamespace`` — so routing is by root namespace,
    never by filename, and a NEW vendored XSD (e.g. the CC manifest-profile
    schema the fetch-window is meant to add) is picked up automatically with
    no code change here. A schema violation emits
    ``CARTRIDGE_XSD_INVALID`` (critical). When NO vendored XSD carries the
    document's root namespace the check degrades to a de-duplicated
    ``CARTRIDGE_XSD_MISSING`` (warning) and the programmatic checks below
    still run. When ``lxml`` is absent every XSD check degrades to a single
    ``CARTRIDGE_LXML_MISSING`` (warning), mirroring the qti_well_formed
    graceful-degrade contract.

(C) **Resource-type enumeration.** Every manifest ``<resource type="…">``
    string is checked against the CC resource-type enumeration (the literal
    set the packager emits plus the versioned CC family patterns). An unknown
    type emits ``CARTRIDGE_RESOURCE_TYPE_UNKNOWN`` (critical) — the LMS would
    not know how to import it.

(D) **In-zip href integrity.** Every manifest ``<resource href="…">`` and
    child ``<file href="…">`` must resolve to a member of the ZIP. A dangling
    href emits ``CARTRIDGE_HREF_UNRESOLVED`` (critical). External hrefs
    (``http(s)://`` / ``mailto:``) are skipped — those legitimately point
    outside the cartridge.

(E) **answer_key webcontent presence.** When an instructor-facing
    ``answer_key`` HTML sidecar is present in the ZIP (the rendered page
    ``Courseforge/scripts/answer_key_emitter.py`` emits), the manifest MUST
    carry a resource that references it (so it survives import as a real
    content page). An orphaned answer_key file emits
    ``CARTRIDGE_ANSWER_KEY_UNREFERENCED`` (critical). No answer_key in the
    ZIP → no issue (it is optional; "present when emitted").

Severity contract (mirrors ``QtiWellFormedValidator``): the validator emits
critical-severity issues for every genuine conformance break and
warning-severity issues for the graceful-degrade codes
(``CARTRIDGE_LXML_MISSING`` / ``CARTRIDGE_XSD_MISSING`` /
``CARTRIDGE_UNKNOWN_CC_VERSION``). ``passed`` / ``action="block"`` derive from
the critical count. The GATE that wires this after ``packaging`` is
**warning day-1** (``severity: warning`` / ``on_fail: warn``) per the
calibration precedent (``synthesized_quiz_distractor`` etc.) — the validator's
own issue severities stay honest so a later critical-flip is a config change.

Input contract (``validate(inputs) -> GateResult``):

* ``inputs["imscc_path"]`` — path to the built ``.imscc`` (or any ZIP). This
  is the form the wiring builder surfaces (reuses ``_build_imscc``, which
  resolves ``packaging.package_path`` / ``imscc_path``).
* ``inputs["imscc_dir"]`` — an already-EXTRACTED cartridge directory (for
  in-memory / test use, and for a packager that leaves an unzipped tree).

Exactly one must resolve to a readable cartridge; otherwise
``CARTRIDGE_NO_INPUT`` (critical). ``inputs["gate_id"]`` overrides the
GateResult id (defaults to the validator name).

Cross-references:

* ``lib/validators/qti_well_formed.py`` — the loose-QTI validator this one
  complements (that runs pre-packaging on ``06_assessments/``; this runs
  post-packaging on the assembled ZIP). Shares the graceful-degrade shape.
* ``Courseforge/scripts/package_multifile_imscc.py`` — the packager whose
  manifest shape / resource-type strings this validator asserts
  (``_ASSESSMENT_RES_TYPE`` + ``webcontent``).
* ``Courseforge/schemas/imscc/`` — the vendored XSD dir scanned for the
  ns→XSD routing map.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# ── CC manifest (imscp) profile namespaces, by CC version ────────────────────
# The manifest root element carries one of these; the ``imsccv1pN`` token is
# the CC profile version. (CC 1.3's real value is confirmed against a built
# cartridge: ``http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1``.)
_CC_MANIFEST_NS = {
    "http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1": "1.1",
    "http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1": "1.2",
    "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1": "1.3",
    # A pre-profile / plain IMS CP manifest — accepted, version "cp".
    "http://www.imsglobal.org/xsd/imscp_v1p1": "cp",
}

# ── CC resource-type enumeration ─────────────────────────────────────────────
# The literal strings the packager emits (``_ASSESSMENT_RES_TYPE`` +
# ``webcontent``) plus the other canonical CC resource-type literals. Anything
# not in this set is matched against the versioned family patterns below before
# being flagged unknown, so a CC 1.1/1.2 cartridge (different version token) is
# not false-flagged.
_CC_RESOURCE_TYPES_LITERAL = frozenset({
    "webcontent",
    "imsbasiclti_xmlv1p0",
    "imsapip_xmlv1p0",
    # discussion topic
    "imsdt_xmlv1p1", "imsdt_xmlv1p2", "imsdt_xmlv1p3",
    # web link
    "imswl_xmlv1p1", "imswl_xmlv1p2", "imswl_xmlv1p3",
})

# Versioned CC resource-type families (the ``v1pN`` token varies by CC
# version); a type matching any of these is conformant.
_CC_RESOURCE_TYPE_PATTERNS = (
    # QTI assessment / question-bank, e.g.
    # imsqti_xmlv1p2/imscc_xmlv1p3/assessment
    re.compile(r"^imsqti_xmlv1p2/imscc_xmlv1p[123]/(assessment|question-bank)$"),
    # learning-application-resource (assignment / basic-lti), e.g.
    # associatedcontent/imscc_xmlv1p3/learning-application-resource
    re.compile(r"^associatedcontent/imscc_xmlv1p[123]/learning-application-resource$"),
    # discussion / web link families (version-generic)
    re.compile(r"^imsdt_xmlv1p[123]$"),
    re.compile(r"^imswl_xmlv1p[123]$"),
)


def _is_conformant_resource_type(res_type: str) -> bool:
    """True when ``res_type`` is a recognised CC resource-type string."""
    if res_type in _CC_RESOURCE_TYPES_LITERAL:
        return True
    return any(p.match(res_type) for p in _CC_RESOURCE_TYPE_PATTERNS)


def _localname(tag: str) -> str:
    """Strip an XML ``{uri}local`` prefix → ``local`` (lowercased)."""
    if "}" in tag:
        tag = tag.rsplit("}", 1)[1]
    return tag.lower()


def _root_namespace(root: ET.Element) -> str:
    """Return the root element's namespace URI (``""`` when none)."""
    tag = root.tag
    if tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def _iter_local(root: ET.Element, local: str):
    """Yield every descendant (and self) whose local tag == ``local``."""
    for elem in root.iter():
        if _localname(elem.tag) == local:
            yield elem


def _is_external_href(href: str) -> bool:
    """True for an href pointing OUTSIDE the cartridge (has a URL scheme)."""
    try:
        scheme = urlparse(href).scheme
    except (ValueError, TypeError):
        return False
    return scheme in ("http", "https", "mailto", "ftp", "data")


def _normalize_zip_href(href: str) -> str:
    """Normalise a manifest href to compare against ZIP member names.

    URL-unquotes, strips a leading ``./``, collapses backslashes, and strips a
    leading ``/``. Does NOT resolve ``..`` (a traversal href is a real defect —
    it simply won't match any member and is reported unresolved).
    """
    h = unquote(href.strip()).replace("\\", "/")
    while h.startswith("./"):
        h = h[2:]
    return h.lstrip("/")


def _resolve_vendored_schema_dir() -> Optional[Path]:
    """Locate the vendored ``Courseforge/schemas/imscc/`` dir (walk upward)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "Courseforge" / "schemas" / "imscc"
        if candidate.is_dir():
            return candidate
    return None


class CartridgeConformanceValidator:
    """Track-L full-cartridge strict-conformance gate (post-packaging)."""

    name = "cartridge_conformance"
    version = "0.1.0"  # Track L (L2)

    # Answer-key sidecar detection: a rendered HTML page whose basename carries
    # the ``answer_key`` marker (matches Courseforge/scripts/answer_key_emitter).
    _ANSWER_KEY_RE = re.compile(r"answer[_-]?key", re.IGNORECASE)

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        issues: List[GateIssue] = []

        # ---- 1. Resolve the cartridge (ZIP path or extracted dir) → members.
        members, member_reader, resolve_issue = self._open_cartridge(inputs)
        if resolve_issue is not None:
            issues.append(resolve_issue)
            return self._result(gate_id, issues)

        # ---- 2. Manifest must be present + well-formed.
        if "imsmanifest.xml" not in members:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CARTRIDGE_MANIFEST_MISSING",
                    message=(
                        "cartridge has no imsmanifest.xml at its root; no LMS "
                        "can import it."
                    ),
                )
            )
            return self._result(gate_id, issues)

        manifest_bytes = member_reader("imsmanifest.xml")
        try:
            manifest_root = ET.fromstring(manifest_bytes)
        except ET.ParseError as exc:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CARTRIDGE_MANIFEST_PARSE_ERROR",
                    message=f"imsmanifest.xml is not well-formed XML: {exc}",
                    location="imsmanifest.xml",
                )
            )
            return self._result(gate_id, issues)

        # ---- 3. CC-version detection (best-effort; warning-only).
        manifest_ns = _root_namespace(manifest_root)
        if manifest_ns not in _CC_MANIFEST_NS:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CARTRIDGE_UNKNOWN_CC_VERSION",
                    message=(
                        f"imsmanifest.xml root namespace {manifest_ns!r} is not "
                        "a recognised IMS Common Cartridge imscp profile "
                        "namespace; XSD routing may fall back to degrade."
                    ),
                    location="imsmanifest.xml",
                )
            )

        # ---- 4. Namespace-routed XSD conformance (manifest + every XML doc).
        xsd_router = _XsdRouter()
        seen_degrade: set = set()

        def _emit_degrade(issue: Optional[GateIssue]) -> None:
            if issue is not None and issue.message not in seen_degrade:
                seen_degrade.add(issue.message)
                issues.append(issue)

        # Manifest.
        issues.extend(
            self._xsd_check(
                "imsmanifest.xml", manifest_bytes, manifest_ns,
                xsd_router, _emit_degrade,
            )
        )
        # Every other *.xml member (resource documents: QTI / imsdt /
        # assignment / weblink). HTML webcontent is skipped (no XSD).
        for member in sorted(members):
            if member == "imsmanifest.xml" or not member.lower().endswith(".xml"):
                continue
            data = member_reader(member)
            try:
                root = ET.fromstring(data)
            except ET.ParseError as exc:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CARTRIDGE_RESOURCE_PARSE_ERROR",
                        message=(
                            f"resource document {member} is not well-formed "
                            f"XML: {exc}"
                        ),
                        location=member,
                    )
                )
                continue
            issues.extend(
                self._xsd_check(
                    member, data, _root_namespace(root),
                    xsd_router, _emit_degrade,
                )
            )

        # ---- 5. Resource-type enumeration + (6) in-zip href integrity.
        referenced_hrefs: set = set()
        for resource in _iter_local(manifest_root, "resource"):
            res_id = resource.get("identifier", "<no-id>")
            res_type = (resource.get("type") or "").strip()
            if res_type and not _is_conformant_resource_type(res_type):
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="CARTRIDGE_RESOURCE_TYPE_UNKNOWN",
                        message=(
                            f"resource {res_id!r} declares type {res_type!r}, "
                            "which is not a recognised IMS Common Cartridge "
                            "resource-type string; the LMS cannot classify it."
                        ),
                        location=f"imsmanifest.xml#{res_id}",
                    )
                )

            # resource-level href + every child <file href>.
            hrefs: List[str] = []
            r_href = resource.get("href")
            if r_href:
                hrefs.append(r_href)
            for file_el in _iter_local(resource, "file"):
                f_href = file_el.get("href")
                if f_href:
                    hrefs.append(f_href)

            for href in hrefs:
                if _is_external_href(href):
                    continue
                norm = _normalize_zip_href(href)
                if not norm:
                    continue
                referenced_hrefs.add(norm)
                if norm not in members:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="CARTRIDGE_HREF_UNRESOLVED",
                            message=(
                                f"resource {res_id!r} references href "
                                f"{href!r} which is not present in the "
                                "cartridge ZIP; the import would 404 the file."
                            ),
                            location=f"imsmanifest.xml#{res_id}",
                        )
                    )

        # ---- 7. answer_key webcontent presence ("present when emitted").
        for member in members:
            if member.lower().endswith((".html", ".htm")) and self._ANSWER_KEY_RE.search(
                Path(member).name
            ):
                norm = _normalize_zip_href(member)
                if norm not in referenced_hrefs:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="CARTRIDGE_ANSWER_KEY_UNREFERENCED",
                            message=(
                                f"answer_key sidecar {member} is present in the "
                                "cartridge but no manifest resource references "
                                "it; it would not survive import as a content "
                                "page."
                            ),
                            location=member,
                        )
                    )

        return self._result(gate_id, issues)

    # --------------------------------------------------------------- helpers

    def _open_cartridge(
        self, inputs: Dict[str, Any],
    ) -> Tuple[set, Any, Optional[GateIssue]]:
        """Resolve the cartridge to ``(member_set, reader_fn, issue_or_None)``.

        ``reader_fn(name) -> bytes`` reads one member. On failure returns
        ``(set(), None, <critical GateIssue>)``.
        """
        raw_path = inputs.get("imscc_path")
        raw_dir = inputs.get("imscc_dir")

        # (a) explicit extracted directory.
        if raw_dir:
            root = Path(raw_dir)
            if not root.is_dir():
                return set(), None, GateIssue(
                    severity="critical",
                    code="CARTRIDGE_NO_INPUT",
                    message=f"inputs['imscc_dir'] is not a directory: {root}",
                    location=str(root),
                )
            members = {
                str(p.relative_to(root)).replace("\\", "/")
                for p in root.rglob("*") if p.is_file()
            }

            def _dir_reader(name: str, _root=root) -> bytes:
                return (_root / name).read_bytes()

            return members, _dir_reader, None

        # (b) ZIP path.
        if raw_path:
            zpath = Path(raw_path)
            if not zpath.exists():
                return set(), None, GateIssue(
                    severity="critical",
                    code="CARTRIDGE_NO_INPUT",
                    message=f"inputs['imscc_path'] does not exist: {zpath}",
                    location=str(zpath),
                )
            try:
                zf = zipfile.ZipFile(zpath)
            except (zipfile.BadZipFile, OSError) as exc:
                return set(), None, GateIssue(
                    severity="critical",
                    code="CARTRIDGE_NO_INPUT",
                    message=(
                        f"inputs['imscc_path'] is not a readable ZIP: "
                        f"{exc.__class__.__name__}: {exc}"
                    ),
                    location=str(zpath),
                )
            members = {
                n for n in zf.namelist() if not n.endswith("/")
            }

            def _zip_reader(name: str, _zf=zf) -> bytes:
                return _zf.read(name)

            return members, _zip_reader, None

        return set(), None, GateIssue(
            severity="critical",
            code="CARTRIDGE_NO_INPUT",
            message=(
                "CartridgeConformanceValidator requires inputs['imscc_path'] "
                "or inputs['imscc_dir'] to resolve to a built cartridge."
            ),
        )

    def _xsd_check(
        self,
        label: str,
        data: bytes,
        ns: str,
        xsd_router: "_XsdRouter",
        emit_degrade: Any,
    ) -> List[GateIssue]:
        """Validate one XML doc against the XSD matching its root namespace."""
        schema, degrade = xsd_router.schema_for(ns)
        if schema is None:
            emit_degrade(degrade)
            return []

        from lxml import etree  # type: ignore

        try:
            doc = etree.fromstring(data)
        except etree.XMLSyntaxError as exc:
            return [
                GateIssue(
                    severity="critical",
                    code="CARTRIDGE_RESOURCE_PARSE_ERROR",
                    message=f"{label} failed lxml parse: {exc}",
                    location=label,
                )
            ]

        if schema.validate(doc):
            return []

        issues: List[GateIssue] = []
        for err in list(schema.error_log)[:10]:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CARTRIDGE_XSD_INVALID",
                    message=(
                        f"{label} violates the XSD for namespace {ns!r} "
                        f"({xsd_router.basename_for(ns)}): {err.message}"
                    ),
                    location=f"{label}:{getattr(err, 'line', '?')}",
                )
            )
        if not issues:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="CARTRIDGE_XSD_INVALID",
                    message=(
                        f"{label} failed XSD validation for namespace {ns!r} "
                        "(no detailed error available)."
                    ),
                    location=label,
                )
            )
        return issues

    def _result(self, gate_id: str, issues: List[GateIssue]) -> GateResult:
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        action: Optional[str] = "block" if not passed else None
        score = max(0.0, 1.0 - len(issues) * 0.1) if issues else 1.0
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )


class _XsdRouter:
    """Lazy ns→``lxml.XMLSchema`` router built from the vendored schema dir.

    Scans ``Courseforge/schemas/imscc/`` (recursively — nested vendor subdirs
    included) once, reading each ``.xsd``'s ``@targetNamespace`` to build a
    ``{namespace: xsd_path}`` map. ``schema_for(ns)`` lazily compiles (and
    caches) the matching XMLSchema. Every failure mode (lxml absent, no XSD
    for the namespace, XSD load error) returns ``(None, <warning GateIssue>)``
    so the caller degrades gracefully — this is what makes the validator work
    TODAY (only the 3 resource XSDs vendored) and auto-upgrade when the CC
    manifest-profile XSDs are vendored.
    """

    def __init__(self) -> None:
        self._ns_to_path: Optional[Dict[str, Path]] = None
        self._compiled: Dict[str, Any] = {}
        self._lxml_ok: Optional[bool] = None

    def _ensure_lxml(self) -> bool:
        if self._lxml_ok is None:
            try:
                import lxml.etree  # noqa: F401
                self._lxml_ok = True
            except ImportError:
                self._lxml_ok = False
        return self._lxml_ok

    def _ensure_map(self) -> Dict[str, Path]:
        if self._ns_to_path is not None:
            return self._ns_to_path
        mapping: Dict[str, Path] = {}
        schema_dir = _resolve_vendored_schema_dir()
        if schema_dir is not None:
            for xsd in sorted(schema_dir.rglob("*.xsd")):
                try:
                    root = ET.parse(str(xsd)).getroot()
                except (ET.ParseError, OSError):
                    continue
                tns = root.get("targetNamespace")
                if tns and tns not in mapping:
                    mapping[tns] = xsd
        self._ns_to_path = mapping
        return mapping

    def basename_for(self, ns: str) -> str:
        path = self._ensure_map().get(ns)
        return path.name if path is not None else "<none>"

    def schema_for(self, ns: str) -> Tuple[Any, Optional[GateIssue]]:
        if not self._ensure_lxml():
            return None, GateIssue(
                severity="warning",
                code="CARTRIDGE_LXML_MISSING",
                message=(
                    "lxml is not installed; skipping cartridge XSD conformance "
                    "checks (manifest presence, resource-type, href integrity, "
                    "and answer_key checks still run). Install lxml for full "
                    "XSD validation."
                ),
                suggestion="pip install lxml",
            )

        if ns in self._compiled:
            return self._compiled[ns], None

        xsd_path = self._ensure_map().get(ns)
        if xsd_path is None:
            return None, GateIssue(
                severity="warning",
                code="CARTRIDGE_XSD_MISSING",
                message=(
                    f"no vendored XSD carries targetNamespace {ns!r}; skipping "
                    "the XSD conformance check for documents with that root "
                    "namespace (vendor the matching CC/QTI schema to enable "
                    "it). Programmatic checks still run."
                ),
            )

        from lxml import etree  # type: ignore

        try:
            schema = etree.XMLSchema(etree.parse(str(xsd_path)))
        except Exception as exc:  # noqa: BLE001 — any lxml load failure degrades
            self._compiled[ns] = None
            return None, GateIssue(
                severity="warning",
                code="CARTRIDGE_XSD_MISSING",
                message=(
                    f"failed to load vendored XSD {xsd_path.name} for "
                    f"namespace {ns!r}: {exc.__class__.__name__}: {exc}; "
                    "skipping XSD check for that namespace."
                ),
                location=str(xsd_path),
            )
        self._compiled[ns] = schema
        return schema, None


__all__ = [
    "CartridgeConformanceValidator",
]
