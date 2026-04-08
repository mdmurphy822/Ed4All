# IMSCC XML Schemas

This directory contains XML Schema Definition (XSD) files for validating IMSCC package components.

## Schema Files

| File | Namespace | Purpose |
|------|-----------|---------|
| `cc_extresource_assignmentv1p0.xsd` | `http://www.imsglobal.org/xsd/imscc_extensions/assignment` | Assignment XML validation |
| `ccv1p3_imsdt_v1p3.xsd` | `http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3` | Discussion topic XML validation |
| `ccv1p3_qtiasiv1p2p1.xsd` | `http://www.imsglobal.org/xsd/ims_qtiasiv1p2` | QTI 1.2 assessment validation |
| `ccv1p3_imscp_v1p2.xsd` | `http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1` | Manifest validation |

## Usage

### Python (lxml)
```python
from lxml import etree

# Load schema
with open('cc_extresource_assignmentv1p0.xsd', 'rb') as f:
    schema_doc = etree.parse(f)
    schema = etree.XMLSchema(schema_doc)

# Validate XML
xml_doc = etree.parse('assignment.xml')
is_valid = schema.validate(xml_doc)
if not is_valid:
    print(schema.error_log)
```

## Official Sources

These schemas are based on official IMS Global specifications:
- Assignment: http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd
- Discussion: http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsdt_v1p3.xsd
- QTI: http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd
- Manifest: http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd

## Resource Types in Manifest

| Content Type | Resource Type Attribute |
|--------------|------------------------|
| Assignment | `assignment_xmlv1p0` |
| Discussion | `imsdt_xmlv1p3` |
| Quiz/Assessment | `imsqti_xmlv1p2/imscc_xmlv1p3/assessment` |
| Web Content | `webcontent` |
