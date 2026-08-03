# IMS Common Cartridge schema dependency

Ed4All validates Common Cartridge packages offline, but it does not publish the
third-party IMS Global/1EdTech and W3C schema payloads in this repository.
Operators must install the upstream files in this directory before running QTI
or cartridge-conformance gates. Missing or unreadable schemas are blocking
validation errors; Ed4All does not silently fall back to partial validation.

## Required files

Install these nine files directly beside this README:

- `cc_extresource_assignmentv1p0.xsd`
- `ccv1p3_imsccauth_v1p3.xsd`
- `ccv1p3_imscp_v1p2_v1p0.xsd`
- `ccv1p3_imscsmd_v1p0.xsd`
- `ccv1p3_imsdt_v1p3.xsd`
- `ccv1p3_lommanifest_v1p0.xsd`
- `ccv1p3_lomresource_v1p0.xsd`
- `ccv1p3_qtiasiv1p2p1.xsd`
- `xml.xsd`

The exact installation path is `Courseforge/schemas/imscc/<filename>`,
relative to the repository root.

## Provenance and acquisition

Acquire the Common Cartridge 1.3 schemas from the official IMS Global/1EdTech
Common Cartridge schema locations under
`http://www.imsglobal.org/profile/cc/ccv1p3/`. The assignment schema is
published under
`http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd`.
The QTI and discussion schemas identify these official source files in their
headers:

- `ccv1p3_qtiasiv1p2p1_v1p0.xsd`
- `ccv1p3_imsdt_v1p3.xsd`

The manifest profile and its IMS authorization, curriculum-metadata, and LOM
imports identify the IMS Common Cartridge and Curriculum Standards Metadata
specifications in their own headers. Acquire `xml.xsd` from the official W3C
XML namespace schema location at `http://www.w3.org/2001/xml.xsd`.

Preserve every upstream copyright, IPR, license, and distribution notice in
full. Do not copy these payloads into a commit. The repository's `.gitignore`
intentionally excludes every `*.xsd` in this directory.

The manifest schema must resolve its five imports locally. In
`ccv1p3_imscp_v1p2_v1p0.xsd`, set the `schemaLocation` values to these sibling
filenames while leaving the namespace declarations unchanged:

- `xml.xsd`
- `ccv1p3_imsccauth_v1p3.xsd`
- `ccv1p3_lommanifest_v1p0.xsd`
- `ccv1p3_lomresource_v1p0.xsd`
- `ccv1p3_imscsmd_v1p0.xsd`

## Verify the installation

From the repository root, run:

```bash
python - <<'PY'
from pathlib import Path
from lxml import etree

root = Path("Courseforge/schemas/imscc")
required = {
    "cc_extresource_assignmentv1p0.xsd",
    "ccv1p3_imsccauth_v1p3.xsd",
    "ccv1p3_imscp_v1p2_v1p0.xsd",
    "ccv1p3_imscsmd_v1p0.xsd",
    "ccv1p3_imsdt_v1p3.xsd",
    "ccv1p3_lommanifest_v1p0.xsd",
    "ccv1p3_lomresource_v1p0.xsd",
    "ccv1p3_qtiasiv1p2p1.xsd",
    "xml.xsd",
}
missing = sorted(name for name in required if not (root / name).is_file())
if missing:
    raise SystemExit("Missing IMSCC schemas: " + ", ".join(missing))
for name in sorted(required):
    etree.parse(str(root / name))
etree.XMLSchema(etree.parse(str(root / "ccv1p3_imscp_v1p2_v1p0.xsd")))
print("IMSCC schema dependency is complete and loadable.")
PY
```

Then run the schema-dependent validator tests:

```bash
pytest -q lib/validators/tests/test_qti_well_formed.py \
  lib/validators/tests/test_cartridge_conformance.py
```

See [the installation guide](../../../docs/operations/installation.md) for the
full environment and dependency setup.
