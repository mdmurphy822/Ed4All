# Utilities - Helper Scripts and Tools

## Purpose
Collection of utility scripts for file management, copying, and support operations for IMSCC package development.

## Scripts Overview

### File Management
- **`copy_existing_package.py`** - Copy existing IMSCC packages between directories
- **`manual_zip_creator.py`** - Manual ZIP file creation with custom options

## Dependencies
- Python 3.7+
- pathlib (built-in)
- shutil (built-in)
- zipfile (built-in)

## Usage
```bash
cd /scripts/utilities/
python3 copy_existing_package.py [source] [destination]
```

## Configuration
Utility-specific configuration files in `config/` subdirectory.

## Error Handling
All utilities include comprehensive error handling and validation.

## Changelog
- v1.0.0 (2025-08-05): Initial organization from scattered scripts