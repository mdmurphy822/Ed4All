# Test Utilities

**Purpose**: Collection of test scripts and validation utilities for IMSCC generation and Pattern 7 prevention testing.

## Scripts

### Pattern 7 Testing
- `test_bulletproof.py` - Bulletproof generator validation tests
- `manual_bulletproof_test.py` - Manual testing procedures
- `simple_imscc_test.py` - Basic IMSCC creation tests

### Execution Testing
- `test_execution.py` - Script execution validation
- `execute_bulletproof_inline.py` - Inline bulletproof execution
- `execute_imscc_generation.py` - IMSCC generation testing

### Simple Tests
- `simple_test.py` - Basic functionality tests
- `run_corrected_generator.py` - Corrected generator execution

### Compression and Component Testing
- `test_compression.py` - Tests ZIP compression functionality and capabilities
- `test_master_generator.py` - Tests master generator orchestration functionality

## Usage

These utilities are used for validating script functionality and ensuring Pattern 7 prevention protocols work correctly.

All test scripts should be run from the main project directory and will reference the appropriate production scripts in their respective directories.

---

**Version**: 1.0.0  
**Created**: 2025-08-05  
**Purpose**: Testing and validation support