# Package Creators - IMSCC Generation Tools

## Purpose
Collection of tools for creating IMS Common Cartridge (.imscc) packages from course content for Brightspace LMS import.

## Available Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `production_imscc_generator.py` | Production-ready IMSCC generator with Pattern 7 prevention | **Primary** |
| `imscc-master-generator.py` | Master orchestration script for modular IMSCC generation | Production |
| `build_imscc_package.py` | Advanced package builder with configuration options | Production |
| `simple_imscc_generator.py` | Lightweight generator without external dependencies | Utility |
| `simple_imscc_creator.py` | Streamlined creator for basic course structures | Utility |

## Primary Usage

For most use cases, use `production_imscc_generator.py`:

```bash
cd /scripts/package-creators/
python3 production_imscc_generator.py --input /path/to/course --output /path/to/package.imscc
```

## Note

The main production IMSCC packaging is handled by the `brightspace-packager` agent which uses `/scripts/brightspace-packager/brightspace_packager.py` (82KB comprehensive solution). The scripts in this directory are standalone utilities and development tools.

## Configuration

Configuration files located in `config/` subdirectory:
- `package_config.json` - Default packaging settings
- `imscc_standards.json` - IMS Common Cartridge compliance settings

## Input Requirements

- Course content directory with proper structure
- `imsmanifest.xml` file (IMS Common Cartridge 1.2.0 compliant)
- HTML content files with Bootstrap 4.3.1 framework
- Assessment XML files (QTI 1.2, D2L format)

## Output Format

- Single `.imscc` file ready for Brightspace import
- IMS Common Cartridge 1.2.0 compliant structure
- Native Brightspace assessment tool integration
- WCAG 2.2 AA accessibility compliance

## Dependencies

- Python 3.7+
- zipfile (built-in)
- pathlib (built-in)
- xml.etree.ElementTree (built-in)
