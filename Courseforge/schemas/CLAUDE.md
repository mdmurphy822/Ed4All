# Courseforge Schemas

Courseforge-local schemas for UI components, layouts, template integration, and
the IMSCC XSD set used by cartridge validation. Cross-project schemas live in the
unified project-root `/schemas/` tree, **not** here.

`README.md` in this directory is the authoritative index (per-schema key features
plus the migration table for schemas that moved to `/schemas/`). This file is the
short orientation map.

## Directory Structure

```
Courseforge/schemas/
├── README.md                # Authoritative per-schema index
├── CLAUDE.md                # This file
├── content-display/         # Content presentation standards
│   ├── content-display-schema.json
│   ├── enhanced-content-display-schema.json
│   ├── accordion-schema.json
│   └── page-title-standards.json
├── template-integration/
│   └── educational_template_schema.json
├── framework-migration/
│   └── bootstrap5_migration_schema.json
├── layouts/
│   └── course_card_schema.json
├── assessment/              # Placeholder (empty — .gitkeep only)
└── imscc/                   # IMS CC XSD set + README
```

### `imscc/`

Nine XSD files consumed by `lib/validators/cartridge_conformance.py`, which
auto-discovers every schema in this directory by `targetNamespace`:

- CC 1.3 manifest profile — `ccv1p3_imscp_v1p2_v1p0.xsd` plus its five imports
  (`ccv1p3_lommanifest_v1p0.xsd`, `ccv1p3_lomresource_v1p0.xsd`,
  `ccv1p3_imsccauth_v1p3.xsd`, `ccv1p3_imscsmd_v1p0.xsd`, `xml.xsd`).
- Resource-type schemas — `ccv1p3_qtiasiv1p2p1.xsd` (QTI 1.2 assessments),
  `ccv1p3_imsdt_v1p3.xsd` (discussion topics),
  `cc_extresource_assignmentv1p0.xsd` (assignment extension).

Remote `schemaLocation` URLs inside the manifest-profile XSD were rewritten to
local relative filenames so validation resolves offline. See `imscc/README.md`
for the full file → namespace → purpose table.

### `assessment/`

Reserved directory, currently empty (`.gitkeep` only). Assessment emission is
driven by code, not by a schema in this directory:
`Courseforge/scripts/qti_emitter.py` writes the QTI 1.2 / discussion-topic /
assignment XML and `Courseforge/scripts/package_multifile_imscc.py` assigns the
manifest resource types.

## Accessibility (migrated)

WCAG 2.2 AA compliance requirements now live at
`/schemas/compliance/wcag22_compliance.schema.json` in the unified project-root
schema tree. That file is an authoring / audit reference matrix — runtime
accessibility validation is performed by `lib/validators/wcag.py::WCAGValidator`
(gate-wired in `config/workflows.yaml` as the `wcag_compliance` gate), which
encodes the same success-criterion matrix in code rather than `$ref`-ing the
schema.

## Usage Guidelines

1. **Schema validation**: generated content validates against the schema that
   owns its surface; cartridge conformance validates against `imscc/`.
2. **Version control**: keep schema versions backward-compatible; a breaking
   shape change needs a version bump plus a consumer audit.
3. **Placement**: a schema used by exactly one Courseforge surface belongs here;
   anything consumed cross-project belongs in the project-root `/schemas/` tree.
