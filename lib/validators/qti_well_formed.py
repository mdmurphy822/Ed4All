"""W10 Phase 5 — QtiWellFormedValidator.

A structural validator that catches malformed QTI 1.2 assessment XML that
would silently fail to import into an LMS (Brightspace / Canvas). It runs at
the new ``assessment_synthesis`` phase boundary (the GATE WIRING is a later
W10 wave; this module ships only the validator class + its unit tests).

Per the W10 spec §7 the validator performs four checks against each emitted
QTI ``<questestinterop>`` document:

(a) **Parse / round-trip** — every QTI XML must parse and round-trip the
    in-tree QTI 1.2 oracle ``Trainforge/parsers/qti_parser.py::QTIParser.
    parse_string``. A non-parseable string emits ``QTI_PARSE_ERROR``
    (critical, ``action="block"``).

(b) **XSD conformance** — when ``lxml`` is installed, validate the document
    against the vendored XSD MATCHING ITS ROOT NAMESPACE (assessment-tail
    fix 2026-07-19): QTI ``questestinterop`` → ``ccv1p3_qtiasiv1p2p1.xsd``;
    imsdt discussion ``topic`` → ``ccv1p3_imsdt_v1p3.xsd``; ``assignment``
    → ``cc_extresource_assignmentv1p0.xsd`` (all under
    ``Courseforge/schemas/imscc/``). Non-QTI docs skip the QTI-only checks
    (a)/(c)/(d) — they carry no items. Unknown roots fall through to the QTI
    XSD so they fail loudly.
    Schema violations emit ``QTI_XSD_INVALID`` (critical). When ``lxml`` is
    absent the validator GRACEFULLY DEGRADES to "well-formed only" — it emits
    a single warning-severity ``QTI_LXML_MISSING`` GateIssue (``passed=True``
    for the XSD dimension), mirroring the ``[embedding]``-extras degrade
    pattern other validators use (see
    ``lib/embedding/sentence_embedder.py::is_strict_mode``).

(c) **Answer-key resolution** — every AUTO-GRADABLE item (``cc_profile`` in
    the multiple_choice / multiple_response / true_false / fib set) must
    carry a resolvable ``resprocessing`` → ``respcondition`` → ``varequal``
    answer key. A missing key emits ``QTI_NO_ANSWER_KEY`` (critical).

(d) **Supported profile** — every ``item``'s ``cc_profile`` must be one of
    the §2.2 supported set. An unknown profile emits ``QTI_UNSUPPORTED_PROFILE``
    (critical).

Severity contract:

* The validator emits **critical-severity** GateIssues for every malformed
  case so a cartridge that would silently fail to import is visible to
  downstream consumers. ``QTI_LXML_MISSING`` is the sole warning-severity
  code (graceful degrade).
* On any critical issue, ``action="block"`` and ``passed=False``; otherwise
  ``action`` is ``None`` and ``passed=True``.

Input contract (``validate(inputs: Dict[str, Any]) -> GateResult``):

The validator accepts the QTI documents in EITHER of two mutually-exclusive
forms (mirroring the chunkset validator's path-or-data flexibility), so the
later wiring wave can point it at the ``06_assessments/`` sibling without a
code change here:

* ``inputs["qti_dir"]`` — a directory (or ``Path``/str) under which the
  validator globs ``*.xml`` files (recursively). Each file is read and
  validated. This is the form the wiring wave uses (the new phase writes QTI
  XML into a ``06_assessments/`` sibling per spec §4).
* ``inputs["qti_paths"]`` — an explicit list of XML file paths.
* ``inputs["qti_strings"]`` — a list of ``{"id": <label>, "xml": <str>}``
  dicts (or bare XML strings), for in-memory / test use.

At least one of the three must be provided and resolve to ≥1 document;
otherwise ``QTI_NO_INPUT`` (critical) is emitted. ``inputs["gate_id"]`` is an
optional GateResult-id override (defaults to the validator name).

Cross-references:

* ``lib/validators/chunkset_manifest.py::ChunksetManifestValidator`` — the
  structural-validator template this class mirrors (constructor, ``validate``
  signature, GateIssue/GateResult shape, schema-on-disk + round-trip check,
  short-circuit-on-missing-input contract).
* ``Trainforge/parsers/qti_parser.py::QTIParser.parse_string`` — the parse
  oracle (imported + reused; never modified).
* ``Courseforge/schemas/imscc/ccv1p3_qtiasiv1p2p1.xsd`` — the XSD oracle.
* ``plans/finegrain/w10-assessments-qti-discussions-2026-06.md`` §7 / §2.2 /
  §11 Phase 5 — the spec.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# Supported cc_profile values per W10 spec §2.2. The four AUTO-GRADABLE
# objective types require a resolvable answer key; essay / short_answer are
# emitted but not client-side graded (manual). Mirror of the XSD header
# comment in ccv1p3_qtiasiv1p2p1.xsd.
_AUTO_GRADABLE_PROFILES = frozenset({
    "cc.multiple_choice.v0p1",
    "cc.multiple_response.v0p1",
    "cc.true_false.v0p1",
    "cc.fib.v0p1",
    # cc.pattern_match.v0p1 is the sixth CC QTI 1.2 profile (optional): a
    # response_str/render_fib fill-in whose resprocessing ORs one varequal per
    # accepted answer string. Auto-gradable — it must carry a resolvable
    # answer key like the other four. (assessment-quality overhaul — the
    # emitter emits it for a fill_in_blank declaring multiple correct_answers.)
    "cc.pattern_match.v0p1",
})
_MANUAL_PROFILES = frozenset({
    "cc.essay.v0p1",
})
# The full supported set for the per-item cc_profile gate (check d).
_SUPPORTED_ITEM_PROFILES = _AUTO_GRADABLE_PROFILES | _MANUAL_PROFILES


# Per-document-type XSD routing (assessment-tail fix 2026-07-19).
#
# The ``assessment_synthesis`` phase emits THREE canonical IMS CC resource
# document types into ``06_assessments/`` — QTI 1.2 assessments
# (``questestinterop``), discussion topics (``imsdt`` ``topic``), and
# assignment learning-application resources (``assignment``). Each has its
# OWN vendored XSD under ``Courseforge/schemas/imscc/``. The pre-fix gate
# validated every ``*.xml`` against the QTI XSD only, so every (schema-valid)
# discussion/assignment doc failed with "No matching global declaration
# available for the validation root". The gate now routes each document to
# its own XSD by root-element namespace; the QTI-specific checks (parse
# oracle, answer-key, cc_profile) still run only on QTI documents.
_QTI_ROOT_NS = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
_IMSDT_ROOT_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
_ASSIGNMENT_ROOT_NS = "http://www.imsglobal.org/xsd/imscc_extensions/assignment"

# doc_kind -> vendored XSD basename. ``qti`` doubles as the fallback for
# unknown roots so a mystery document still fails LOUDLY against the QTI
# schema rather than silently passing.
_XSD_BY_DOC_KIND = {
    "qti": "ccv1p3_qtiasiv1p2p1.xsd",
    "imsdt": "ccv1p3_imsdt_v1p3.xsd",
    "assignment": "cc_extresource_assignmentv1p0.xsd",
}


def _doc_kind_of_root(root: ET.Element) -> str:
    """Classify a parsed document by its root element namespace + name.

    Returns ``"qti"`` / ``"imsdt"`` / ``"assignment"``; unknown roots
    classify as ``"qti"`` (fail-loud fallback — see ``_XSD_BY_DOC_KIND``).
    """
    tag = root.tag
    ns = tag.rsplit("}", 1)[0].lstrip("{") if "}" in tag else ""
    local = _localname(tag)
    if local == "topic" and ns == _IMSDT_ROOT_NS:
        return "imsdt"
    if local == "assignment" and ns == _ASSIGNMENT_ROOT_NS:
        return "assignment"
    return "qti"


def _resolve_xsd_path(basename: str = "ccv1p3_qtiasiv1p2p1.xsd") -> Optional[Path]:
    """Locate a vendored XSD under ``Courseforge/schemas/imscc/``.

    Walks up from this file until it finds the vendored XSD. Returns ``None``
    when not found (validator degrades to well-formed-only with a warning).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "Courseforge" / "schemas" / "imscc" / basename
        if candidate.exists():
            return candidate
    return None


def _localname(tag: str) -> str:
    """Strip an XML namespace ``{uri}local`` prefix → ``local`` (lowercased).

    QTI 1.2 documents declare a default namespace, so ElementTree tags arrive
    as ``{http://www.imsglobal.org/xsd/ims_qtiasiv1p2}item``. The parser
    oracle matches tags case-insensitively via ``.lower()``; mirror that.
    """
    if "}" in tag:
        tag = tag.rsplit("}", 1)[1]
    return tag.lower()


def _iter_local(root: ET.Element, local: str):
    """Yield every descendant (and self) whose local tag == ``local``."""
    for elem in root.iter():
        if _localname(elem.tag) == local:
            yield elem


def _extract_item_cc_profile(item: ET.Element) -> Optional[str]:
    """Read an item's ``cc_profile`` from its ``itemmetadata`` qtimetadata.

    Walks the item's ``qtimetadatafield`` pairs looking for the
    ``fieldlabel == cc_profile`` → sibling ``fieldentry`` value (Pattern-15
    shape). Returns ``None`` when no such field is present.
    """
    for field in _iter_local(item, "qtimetadatafield"):
        label = None
        entry = None
        for child in field:
            ln = _localname(child.tag)
            if ln == "fieldlabel":
                label = (child.text or "").strip()
            elif ln == "fieldentry":
                entry = (child.text or "").strip()
        if label == "cc_profile":
            return entry or ""
    return None


def _item_has_answer_key(item: ET.Element) -> bool:
    """True when the item carries a resolvable ``respcondition``/``varequal``.

    The auto-grade answer key (W10 spec §7c) is a ``resprocessing`` →
    ``respcondition`` → ``conditionvar`` → ``varequal`` chain carrying a
    non-empty correct-response token. A FIB key may live in a ``varequal`` or
    a ``varsubstring``; accept either. The key must have non-empty text (an
    empty ``<varequal/>`` resolves nothing).
    """
    resprocessing = next(_iter_local(item, "resprocessing"), None)
    if resprocessing is None:
        return False
    for cond in _iter_local(resprocessing, "respcondition"):
        for tok_local in ("varequal", "varsubstring"):
            for tok in _iter_local(cond, tok_local):
                if tok.text is not None and tok.text.strip():
                    return True
    return False


class QtiWellFormedValidator:
    """W10 Phase 5 QTI well-formedness gate.

    Validator-protocol-compatible class wired as the ``qti_well_formed`` gate
    on the new ``assessment_synthesis`` phase (the workflow YAML wiring is a
    later W10 wave). This validator emits critical-severity issues for every
    malformed case so the gate can run ``severity: critical`` /
    ``on_fail: block`` per the spec without code changes.
    """

    name = "qti_well_formed"
    version = "0.1.0"  # W10 Phase 5

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        issues: List[GateIssue] = []

        # ---- 1. Resolve the QTI documents from the (path | data) inputs.
        documents, input_issues = self._collect_documents(inputs)
        issues.extend(input_issues)
        if not documents:
            # No documents at all — either a missing-input or an
            # unreadable-path issue is already recorded above; if neither,
            # record QTI_NO_INPUT.
            if not input_issues:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="QTI_NO_INPUT",
                        message=(
                            "QtiWellFormedValidator requires one of "
                            "inputs['qti_dir'] / ['qti_paths'] / "
                            "['qti_strings'] to resolve to >=1 QTI document."
                        ),
                    )
                )
            return self._result(gate_id, issues)

        # ---- 2. Lazy per-doc-type XSD oracle cache (lazy lxml import).
        # Each of the three emitted resource types (QTI assessment / imsdt
        # discussion topic / assignment) validates against ITS OWN vendored
        # XSD; the cache loads each at most once per validate() call and the
        # degrade warnings are de-duplicated by message.
        xsd_cache: Dict[str, Any] = {}
        seen_degrade_messages: set = set()

        def _xsd_for(doc_kind: str) -> Any:
            if doc_kind in xsd_cache:
                return xsd_cache[doc_kind]
            basename = _XSD_BY_DOC_KIND.get(
                doc_kind, _XSD_BY_DOC_KIND["qti"]
            )
            schema, degrade_issue = self._load_xsd_validator(basename)
            xsd_cache[doc_kind] = schema
            if degrade_issue is not None:
                if degrade_issue.message not in seen_degrade_messages:
                    seen_degrade_messages.add(degrade_issue.message)
                    issues.append(degrade_issue)
            return schema

        # ---- 3. Validate each document.
        for label, xml in documents:
            issues.extend(
                self._validate_one(label, xml, _xsd_for)
            )

        return self._result(gate_id, issues)

    # ------------------------------------------------------------- helpers

    def _result(
        self, gate_id: str, issues: List[GateIssue]
    ) -> GateResult:
        """Assemble the GateResult from accumulated issues."""
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        action: Optional[str] = "block" if not passed else None
        # Score: 1.0 when clean; degrades 0.1 per issue with a 0.0 floor.
        # Mirrors ChunksetManifestValidator's convention.
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

    @staticmethod
    def _collect_documents(
        inputs: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, str]], List[GateIssue]]:
        """Normalise the three input forms into ``[(label, xml_str), ...]``.

        Returns the document list plus any input-resolution GateIssues
        (unreadable file, etc.).
        """
        documents: List[Tuple[str, str]] = []
        issues: List[GateIssue] = []

        # (a) explicit in-memory strings.
        raw_strings = inputs.get("qti_strings")
        if raw_strings:
            for idx, entry in enumerate(raw_strings):
                if isinstance(entry, dict):
                    label = str(entry.get("id", f"qti_strings[{idx}]"))
                    xml = entry.get("xml", "")
                else:
                    label = f"qti_strings[{idx}]"
                    xml = entry
                documents.append((label, xml if isinstance(xml, str) else ""))

        # (b) explicit file paths.
        raw_paths = inputs.get("qti_paths")
        path_candidates: List[Path] = []
        if raw_paths:
            path_candidates.extend(Path(p) for p in raw_paths)

        # (c) a directory to glob.
        raw_dir = inputs.get("qti_dir")
        if raw_dir:
            qti_dir = Path(raw_dir)
            if not qti_dir.exists() or not qti_dir.is_dir():
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="QTI_NO_INPUT",
                        message=(
                            f"inputs['qti_dir'] does not resolve to a "
                            f"directory: {qti_dir}"
                        ),
                        location=str(qti_dir),
                    )
                )
            else:
                path_candidates.extend(sorted(qti_dir.rglob("*.xml")))

        for path in path_candidates:
            try:
                xml = path.read_text(encoding="utf-8")
            except OSError as exc:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="QTI_PARSE_ERROR",
                        message=(
                            f"failed to read QTI file {path}: "
                            f"{exc.__class__.__name__}: {exc}"
                        ),
                        location=str(path),
                    )
                )
                continue
            documents.append((str(path), xml))

        return documents, issues

    @staticmethod
    def _load_xsd_validator(
        basename: str = "ccv1p3_qtiasiv1p2p1.xsd",
    ) -> Tuple[Any, Optional[GateIssue]]:
        """Lazily build the lxml ``XMLSchema`` validator for one vendored XSD.

        Returns ``(schema_or_None, lxml_or_schema_issue_or_None)``. When
        ``lxml`` is absent, returns ``(None, QTI_LXML_MISSING warning)`` —
        the graceful-degrade path. When the XSD file can't be located /
        loaded, returns ``(None, warning)`` so the XSD dimension passes.
        """
        try:
            from lxml import etree  # type: ignore
        except ImportError:
            return None, GateIssue(
                severity="warning",
                code="QTI_LXML_MISSING",
                message=(
                    "lxml is not installed; skipping QTI XSD conformance "
                    "check (well-formedness + round-trip + answer-key + "
                    "profile checks still run). Install the validation "
                    "extras for full XSD validation."
                ),
                suggestion="pip install lxml",
            )

        xsd_path = _resolve_xsd_path(basename)
        if xsd_path is None:
            return None, GateIssue(
                severity="warning",
                code="QTI_LXML_MISSING",
                message=(
                    f"{basename} not found (searched upward "
                    f"from {Path(__file__).resolve()}); skipping XSD "
                    "conformance check for this document type."
                ),
            )

        try:
            schema_doc = etree.parse(str(xsd_path))
            return etree.XMLSchema(schema_doc), None
        except Exception as exc:  # noqa: BLE001 — any lxml load failure degrades
            return None, GateIssue(
                severity="warning",
                code="QTI_LXML_MISSING",
                message=(
                    f"failed to load QTI XSD at {xsd_path}: "
                    f"{exc.__class__.__name__}: {exc}; skipping XSD check."
                ),
                location=str(xsd_path),
            )

    def _validate_one(
        self, label: str, xml: str, xsd_for: Any,
    ) -> List[GateIssue]:
        """Validate a single ``06_assessments`` document string.

        ``xsd_for`` is a ``doc_kind -> XMLSchema-or-None`` resolver (the
        per-call cache built in :meth:`validate`). The document is first
        classified by root-element namespace: QTI documents run the full
        (a)-(d) check set against the QTI XSD; imsdt discussion-topic and
        assignment documents run well-formedness + THEIR OWN vendored XSD
        (they carry no QTI items, so the parse-oracle / answer-key /
        cc_profile checks do not apply). Unknown roots fall through to the
        QTI XSD so they still fail loudly.
        """
        issues: List[GateIssue] = []

        # ---- Well-formedness + doc-type classification (ElementTree).
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="QTI_PARSE_ERROR",
                    message=(
                        f"QTI document {label} failed ElementTree parse: "
                        f"{exc}"
                    ),
                    location=label,
                )
            )
            return issues

        doc_kind = _doc_kind_of_root(root)

        if doc_kind != "qti":
            # imsdt topic / assignment resource — validate against its own
            # vendored XSD; no QTI item structure to check.
            xsd_validator = xsd_for(doc_kind)
            if xsd_validator is not None:
                issues.extend(self._validate_xsd(label, xml, xsd_validator))
            return issues

        # ---- (a) Parse / round-trip the in-tree QTIParser oracle.
        try:
            from Trainforge.parsers.qti_parser import QTIParser
            QTIParser().parse_string(xml)
        except ValueError as exc:
            # QTIParser raises ValueError("Invalid QTI XML: ...") on a
            # non-parseable string — this is the parse failure.
            issues.append(
                GateIssue(
                    severity="critical",
                    code="QTI_PARSE_ERROR",
                    message=(
                        f"QTI document {label} failed to parse / round-trip "
                        f"QTIParser.parse_string: {exc}"
                    ),
                    location=label,
                )
            )
            # Non-parseable → no point running the structural checks.
            return issues
        except Exception as exc:  # noqa: BLE001 — defensive: any oracle raise = parse error
            issues.append(
                GateIssue(
                    severity="critical",
                    code="QTI_PARSE_ERROR",
                    message=(
                        f"QTI document {label} raised in the parse oracle: "
                        f"{exc.__class__.__name__}: {exc}"
                    ),
                    location=label,
                )
            )
            return issues

        # ---- (b) XSD conformance (when lxml present).
        xsd_validator = xsd_for("qti")
        if xsd_validator is not None:
            issues.extend(self._validate_xsd(label, xml, xsd_validator))

        # ---- (c) + (d) per-item answer-key + profile checks.
        items = list(_iter_local(root, "item"))
        for item in items:
            item_id = item.get("ident", "<no-ident>")
            profile = _extract_item_cc_profile(item)

            # (d) supported profile.
            if profile is None or profile == "":
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="QTI_UNSUPPORTED_PROFILE",
                        message=(
                            f"item {item_id!r} in {label} carries no "
                            f"cc_profile item-metadata field; the LMS cannot "
                            f"classify the question type."
                        ),
                        location=f"{label}#{item_id}",
                        suggestion=(
                            "Emit an itemmetadata/qtimetadata "
                            "cc_profile field (one of "
                            f"{sorted(_SUPPORTED_ITEM_PROFILES)})."
                        ),
                    )
                )
            elif profile not in _SUPPORTED_ITEM_PROFILES:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="QTI_UNSUPPORTED_PROFILE",
                        message=(
                            f"item {item_id!r} in {label} carries "
                            f"unsupported cc_profile {profile!r}; supported "
                            f"set is {sorted(_SUPPORTED_ITEM_PROFILES)}."
                        ),
                        location=f"{label}#{item_id}",
                    )
                )

            # (c) answer key — only required for AUTO-GRADABLE items.
            if profile in _AUTO_GRADABLE_PROFILES:
                if not _item_has_answer_key(item):
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="QTI_NO_ANSWER_KEY",
                            message=(
                                f"auto-gradable item {item_id!r} "
                                f"({profile}) in {label} has no resolvable "
                                f"resprocessing/respcondition/varequal answer "
                                f"key; the LMS would import it ungradable."
                            ),
                            location=f"{label}#{item_id}",
                            suggestion=(
                                "Emit a resprocessing/respcondition with a "
                                "non-empty varequal (or varsubstring for "
                                "fib) carrying the correct response token."
                            ),
                        )
                    )

        return issues

    @staticmethod
    def _validate_xsd(
        label: str, xml: str, xsd_validator: Any,
    ) -> List[GateIssue]:
        """Validate one document against the QTI XSD via lxml.

        ``xsd_validator`` is a loaded ``lxml.etree.XMLSchema``. Emits one
        ``QTI_XSD_INVALID`` per schema violation (first few, capped) plus a
        ``QTI_PARSE_ERROR`` if lxml itself can't parse the bytes (should not
        happen after the ElementTree round-trip, but fail-closed).
        """
        from lxml import etree  # type: ignore

        try:
            doc = etree.fromstring(xml.encode("utf-8"))
        except etree.XMLSyntaxError as exc:
            return [
                GateIssue(
                    severity="critical",
                    code="QTI_PARSE_ERROR",
                    message=(
                        f"QTI document {label} failed lxml parse: {exc}"
                    ),
                    location=label,
                )
            ]

        if xsd_validator.validate(doc):
            return []

        issues: List[GateIssue] = []
        # Cap the number of emitted violations so a deeply-broken doc
        # doesn't flood the report; the first handful pinpoint the shape.
        for err in list(xsd_validator.error_log)[:10]:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="QTI_XSD_INVALID",
                    message=(
                        f"QTI document {label} violates "
                        f"ccv1p3_qtiasiv1p2p1.xsd: {err.message}"
                    ),
                    location=f"{label}:{getattr(err, 'line', '?')}",
                    suggestion=(
                        "See Courseforge/schemas/imscc/"
                        "ccv1p3_qtiasiv1p2p1.xsd for the canonical QTI 1.2 "
                        "Common-Cartridge shape."
                    ),
                )
            )
        if not issues:
            # validate() returned False but the error log was empty —
            # surface a generic XSD-invalid so we never silently pass.
            issues.append(
                GateIssue(
                    severity="critical",
                    code="QTI_XSD_INVALID",
                    message=(
                        f"QTI document {label} failed XSD validation "
                        f"(no detailed error available)."
                    ),
                    location=label,
                )
            )
        return issues


__all__ = [
    "QtiWellFormedValidator",
]
